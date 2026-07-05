//! Metal working context: device + command queue + the FP32 compute pipelines
//! (M2-01-T05/T06/T08 for GEMM; T09-T13 for the Phase-4 kernels). Apple targets
//! only.
//!
//! This is the **directly callable** compute surface, mirroring
//! `vokra-backend-cpu`'s `kernels::*`: [`MetalContext::gemm_f32`] runs a
//! row-major single-precision GEMM on the GPU (what the parity tests call,
//! M2-01-T17/T18), and the Phase-4 additions [`MetalContext::gemv_f32`],
//! [`MetalContext::softmax_f32`], [`MetalContext::layer_norm_f32`],
//! [`MetalContext::gelu_f32`] and [`MetalContext::conv1d_f32`] cover the rest of
//! the Whisper hot-op set, each matching the CPU kernel's shape contract and
//! numerics within the FP32 bound (NFR-QL-01, `atol = 0.01`). Together they let
//! the imperative `Compute::Metal` seam run a full Whisper forward on the GPU.
//! [`crate::MetalBackend`] wraps a context for the `Backend` trait but, exactly
//! like `CpuBackend`, keeps graph-level `execute` an honest stub until the
//! data-carrying graph engine lands (a later WP).
//!
//! # Precision (FP32, red line)
//!
//! The kernel is authored in explicit `float` (FP32) — Vokra does **not** run
//! this parity path through MPS/MPSGraph, so there is no implicit FP16 fast
//! path to fall into (M2-01 scope note; the FP16 / quantised tiers are M2-08).
//!
//! # Shader build (`newLibraryWithSource:`, no CPU JIT)
//!
//! The MSL is compiled at runtime with
//! `-[MTLDevice newLibraryWithSource:options:error:]`. This is **not** CPU-side
//! W^X code generation (NFR-RL-05): the host emits no executable code; the Metal
//! framework / GPU driver compiles GPU shader code. iOS ships a W^X constraint
//! on *CPU* pages, and Apple's guidance there is to precompile to a `.metallib`
//! at build time — that iOS precompile path is a followup for M2-02 (this slice
//! is macOS, where `newLibraryWithSource:` is the pragmatic route).

use core::ffi::c_void;

use vokra_core::{Result, VokraError};

use crate::sys::{self, Id, MtlSize};

/// The GEMM shader, compiled once per [`MetalContext`]. Row-major, FP32:
/// `C[r, c] = (has_bias ? bias[c] : 0) + Σ_k A[r, k] · B[k, c]` — identical
/// semantics to `vokra_backend_cpu::kernels::gemm_f32`.
const GEMM_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct GemmDims {
    uint M;
    uint N;
    uint K;
    uint has_bias;
};

kernel void vokra_gemm_f32(
    device const float*   A    [[buffer(0)]],
    device const float*   B    [[buffer(1)]],
    device const float*   bias [[buffer(2)]],
    device float*         C    [[buffer(3)]],
    constant GemmDims&    dims [[buffer(4)]],
    uint2                 gid  [[thread_position_in_grid]])
{
    const uint row = gid.y;
    const uint col = gid.x;
    if (row >= dims.M || col >= dims.N) {
        return;
    }
    float acc = 0.0f;
    const uint arow = row * dims.K;
    for (uint k = 0; k < dims.K; ++k) {
        acc += A[arow + k] * B[k * dims.N + col];
    }
    if (dims.has_bias != 0u) {
        acc += bias[col];
    }
    C[row * dims.N + col] = acc;
}
"#;

/// The five Phase-4 kernels (M2-01 T09-T13), compiled once into one library.
/// Each mirrors the semantics — and, within the FP32 bound, the numerics — of
/// the matching `vokra_backend_cpu::kernels` function. All FP32 (explicit
/// `float`), no MPS/MPSGraph, so there is no implicit FP16 fast path.
///
/// One thread per output row (gemv / softmax / layer_norm) or element (gelu),
/// or per `(out_channel, out_pos)` pair (conv1d); the launch guards the ragged
/// tail against the grid bound, exactly like the GEMM kernel above.
const KERNELS_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

// ---- gemv: out[i] = (has_bias ? bias[i] : 0) + Σ_l A[i*K + l] · x[l] --------
// Bias-first accumulation matches vokra_backend_cpu::kernels' scalar `gemv`.
struct GemvDims {
    uint M;
    uint K;
    uint has_bias;
};

kernel void vokra_gemv_f32(
    device const float* A    [[buffer(0)]],
    device const float* x    [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float*       out  [[buffer(3)]],
    constant GemvDims&  d    [[buffer(4)]],
    uint                gid  [[thread_position_in_grid]])
{
    const uint i = gid;
    if (i >= d.M) {
        return;
    }
    float acc = (d.has_bias != 0u) ? bias[i] : 0.0f;
    const uint arow = i * d.K;
    for (uint l = 0; l < d.K; ++l) {
        acc += A[arow + l] * x[l];
    }
    out[i] = acc;
}

// ---- softmax: row-wise, max-subtracted (numerically stabilised) -------------
struct SoftmaxDims {
    uint rows;
    uint cols;
};

kernel void vokra_softmax_f32(
    device const float*   inp [[buffer(0)]],
    device float*         out [[buffer(1)]],
    constant SoftmaxDims& d   [[buffer(2)]],
    uint                  gid [[thread_position_in_grid]])
{
    const uint r = gid;
    if (r >= d.rows) {
        return;
    }
    const uint base = r * d.cols;
    // Row max over every column (seeded with column 0). A causal-mask -INF entry
    // is never the max and becomes exp(-INF) = 0 below — as on the CPU.
    float m = inp[base];
    for (uint j = 1; j < d.cols; ++j) {
        m = fmax(m, inp[base + j]);
    }
    float sum = 0.0f;
    for (uint j = 0; j < d.cols; ++j) {
        float e = exp(inp[base + j] - m);
        out[base + j] = e;
        sum += e;
    }
    const float inv = 1.0f / sum;
    for (uint j = 0; j < d.cols; ++j) {
        out[base + j] *= inv;
    }
}

// ---- layer_norm: affine, biased (population) variance -----------------------
struct LayerNormDims {
    uint  rows;
    uint  cols;
    float eps;
};

kernel void vokra_layer_norm_f32(
    device const float*     inp   [[buffer(0)]],
    device const float*     gamma [[buffer(1)]],
    device const float*     beta  [[buffer(2)]],
    device float*           out   [[buffer(3)]],
    constant LayerNormDims& d     [[buffer(4)]],
    uint                    gid   [[thread_position_in_grid]])
{
    const uint r = gid;
    if (r >= d.rows) {
        return;
    }
    const uint base = r * d.cols;
    const float inv_cols = 1.0f / (float)d.cols;
    float mean = 0.0f;
    for (uint c = 0; c < d.cols; ++c) {
        mean += inp[base + c];
    }
    mean *= inv_cols;
    float var = 0.0f;
    for (uint c = 0; c < d.cols; ++c) {
        const float dv = inp[base + c] - mean;
        var += dv * dv;
    }
    var *= inv_cols;
    const float inv_std = 1.0f / sqrt(var + d.eps);
    for (uint c = 0; c < d.cols; ++c) {
        out[base + c] = (inp[base + c] - mean) * inv_std * gamma[c] + beta[c];
    }
}

// ---- gelu: exact (erf) form, out = 0.5·x·(1 + erf(x/√2)) ---------------------
// MSL has no builtin `erf`, so we inline the *identical* Abramowitz & Stegun
// 7.1.26 approximation (and constants, and Horner order) that
// vokra_backend_cpu's scalar `gelu` uses. The only CPU⇔GPU numeric difference in
// gelu is then the vendor `exp()` (a few ULP) — far inside the FP32 bound.
struct GeluDims {
    uint n;
};

// erf(x) — A&S 7.1.26 (max abs error ≤ 1.5e-7), matching the CPU constants.
inline float vokra_erf(float x) {
    const float sign = (x < 0.0f) ? -1.0f : 1.0f;
    const float ax = fabs(x);
    const float t = 1.0f / (1.0f + 0.3275911f * ax);
    const float poly =
        ((((1.061405429f * t - 1.453152027f) * t + 1.421413741f) * t - 0.284496736f) * t
            + 0.254829592f) * t;
    const float y = 1.0f - poly * exp(-ax * ax);
    return sign * y;
}

kernel void vokra_gelu_f32(
    device const float* x   [[buffer(0)]],
    device float*       out [[buffer(1)]],
    constant GeluDims&  d   [[buffer(2)]],
    uint                gid [[thread_position_in_grid]])
{
    const uint i = gid;
    if (i >= d.n) {
        return;
    }
    const float v = x[i];
    out[i] = 0.5f * v * (1.0f + vokra_erf(v * 0.70710678118654752440f));
}

// ---- conv1d: direct convolution (im2col + GEMM equivalent) -------------------
// `kernel` is an MSL reserved word, so the tap count is `kernel_size`. The (c
// outer, kk inner) accumulation order equals the im2col+GEMM reduction the CPU
// runs, so the two agree within the FP32 bound; bias is added after, as on CPU.
struct Conv1dDims {
    uint in_ch;
    uint in_len;
    uint out_ch;
    uint kernel_size;
    uint out_len;
    uint stride;
    uint padding;
    uint has_bias;
};

kernel void vokra_conv1d_f32(
    device const float*  inp    [[buffer(0)]],
    device const float*  weight [[buffer(1)]],
    device const float*  bias   [[buffer(2)]],
    device float*        out    [[buffer(3)]],
    constant Conv1dDims& d      [[buffer(4)]],
    uint2                gid    [[thread_position_in_grid]])
{
    const uint t  = gid.x; // output position
    const uint oc = gid.y; // output channel
    if (t >= d.out_len || oc >= d.out_ch) {
        return;
    }
    const uint k     = d.in_ch * d.kernel_size;
    const uint wbase = oc * k;
    float acc = 0.0f;
    for (uint c = 0; c < d.in_ch; ++c) {
        const uint wc    = wbase + c * d.kernel_size;
        const uint ibase = c * d.in_len;
        for (uint kk = 0; kk < d.kernel_size; ++kk) {
            const uint pos = t * d.stride + kk;
            if (pos >= d.padding && pos < d.padding + d.in_len) {
                acc += weight[wc + kk] * inp[ibase + (pos - d.padding)];
            }
        }
    }
    if (d.has_bias != 0u) {
        acc += bias[oc];
    }
    out[oc * d.out_len + t] = acc;
}
"#;

/// GEMM dimension block handed to the kernel via `setBytes:` (buffer index 4).
/// Field order and `u32` widths mirror the MSL `struct GemmDims`.
#[repr(C)]
#[derive(Clone, Copy)]
struct GemmDims {
    m: u32,
    n: u32,
    k: u32,
    has_bias: u32,
}

/// GEMV dims (`setBytes:` index 4). Field order / `u32` widths mirror the MSL
/// `struct GemvDims`.
#[repr(C)]
#[derive(Clone, Copy)]
struct GemvDims {
    m: u32,
    k: u32,
    has_bias: u32,
}

/// Softmax dims (`setBytes:` index 2). Mirrors the MSL `struct SoftmaxDims`.
#[repr(C)]
#[derive(Clone, Copy)]
struct SoftmaxDims {
    rows: u32,
    cols: u32,
}

/// Layer-norm dims (`setBytes:` index 4). The trailing `f32 eps` matches the MSL
/// `struct LayerNormDims` (all fields 4-byte, so `#[repr(C)]` needs no padding).
#[repr(C)]
#[derive(Clone, Copy)]
struct LayerNormDims {
    rows: u32,
    cols: u32,
    eps: f32,
}

/// GELU dims (`setBytes:` index 2). Mirrors the MSL `struct GeluDims`.
#[repr(C)]
#[derive(Clone, Copy)]
struct GeluDims {
    n: u32,
}

/// Conv1d dims (`setBytes:` index 4). Field order / `u32` widths mirror the MSL
/// `struct Conv1dDims`; `kernel_size` (not `kernel`, an MSL reserved word) is the
/// tap count.
#[repr(C)]
#[derive(Clone, Copy)]
struct Conv1dDims {
    in_ch: u32,
    in_len: u32,
    out_ch: u32,
    kernel_size: u32,
    out_len: u32,
    stride: u32,
    padding: u32,
    has_bias: u32,
}

/// RAII wrapper for a `+1`-owned Objective-C object, released once on drop unless
/// defused with [`Owned::into_raw`]. Used for the transient device objects during
/// [`MetalContext::build`] so an early `?`-return releases everything already
/// created; the survivors are defused into the [`MetalContext`] (whose `Drop`
/// then owns them).
struct Owned(Id);

impl Owned {
    /// Takes the raw `id`, cancelling the drop-release: ownership moves to the
    /// caller, which must release it (here, the [`MetalContext`] `Drop`).
    fn into_raw(self) -> Id {
        let id = self.0;
        core::mem::forget(self);
        id
    }
}

impl Drop for Owned {
    fn drop(&mut self) {
        // SAFETY: `self.0` is a valid `+1`-owned object (or null) not yet defused.
        unsafe { release(self.0) };
    }
}

/// RAII wrapper for a `+1`-owned `MTLBuffer`, released exactly once on drop.
///
/// Using drop (rather than a manual release ladder) means an early `?`-return
/// mid-setup still releases every buffer already allocated.
struct OwnedBuf(Id);

impl Drop for OwnedBuf {
    fn drop(&mut self) {
        // SAFETY: `self.0` is a valid `+1`-owned MTLBuffer (or null) obtained
        // from a `newBuffer…` call; `release` is sent once.
        unsafe { release(self.0) };
    }
}

/// A Metal device + command queue + compiled GEMM pipeline.
///
/// Holds three `+1`-owned Objective-C objects (device, queue, pipeline),
/// released in [`Drop`]. Not `Send`/`Sync`: the raw `id` handles must be used
/// from the thread that created them (sufficient for the parity harness; a
/// thread-affine or `Send` wrapper is a later concern).
pub struct MetalContext {
    device: Id,
    queue: Id,
    gemm_pipeline: Id,
    gemv_pipeline: Id,
    softmax_pipeline: Id,
    layer_norm_pipeline: Id,
    gelu_pipeline: Id,
    conv1d_pipeline: Id,
}

impl MetalContext {
    /// Creates the system default device, a command queue, and compiles the
    /// FP32 GEMM pipeline.
    ///
    /// # Errors
    ///
    /// [`VokraError::BackendUnavailable`] if there is no Metal device, the
    /// command queue cannot be created, or the shader fails to compile /
    /// pipeline creation fails (the Metal error description is included).
    pub fn new() -> Result<MetalContext> {
        // SAFETY: `MTLCreateSystemDefaultDevice` takes no arguments and returns
        // an owned `id` (or null), checked below.
        let device = unsafe { sys::MTLCreateSystemDefaultDevice() };
        if device.is_null() {
            return Err(VokraError::BackendUnavailable(
                "no system default Metal device".to_owned(),
            ));
        }

        // SAFETY: `objc_autoreleasePoolPush` returns a token consumed by the one
        // matching pop below; `build` sends only documented selectors to the
        // just-created device.
        let pool = unsafe { sys::objc_autoreleasePoolPush() };
        // SAFETY: `device` is a valid, non-null MTLDevice owned by us.
        let result = unsafe { Self::build(device) };
        // SAFETY: `pool` is the token from the push above.
        unsafe { sys::objc_autoreleasePoolPop(pool) };

        if result.is_err() {
            // SAFETY: release the device we owned before the failure.
            unsafe { release(device) };
        }
        result
    }

    /// Builds queue + every compute pipeline for an already-owned `device`. Runs
    /// inside the caller's autorelease pool.
    ///
    /// Every transient (`queue`, the two libraries, the six pipelines) is held in
    /// an [`Owned`] guard, so an early `?`-return releases exactly what was
    /// created; on success the survivors are defused into the [`MetalContext`].
    /// `device` itself is **not** released here — the caller ([`Self::new`])
    /// releases it on our error, and the returned context owns it on success.
    ///
    /// # Safety
    /// `device` must be a valid, non-null `MTLDevice` owned by the caller.
    unsafe fn build(device: Id) -> Result<MetalContext> {
        // Command queue (thread-affine; released with the context).
        // SAFETY: `device` is a valid MTLDevice per the caller contract.
        let queue = Owned(unsafe { sys::send_id(device, sys::sel(b"newCommandQueue\0")) });
        if queue.0.is_null() {
            return Err(VokraError::BackendUnavailable(
                "MTLDevice newCommandQueue returned nil".to_owned(),
            ));
        }

        // GEMM pipeline from its own library (the proven M2-01 slice); the
        // library is released as soon as the pipeline is built.
        // SAFETY: `device` is a valid MTLDevice.
        let gemm_lib = unsafe { compile_library(device, GEMM_MSL, "GEMM") }?;
        // SAFETY: `device` valid; `gemm_lib` owns the `vokra_gemm_f32` function.
        let gemm_pipeline = unsafe { make_pipeline(device, gemm_lib.0, c"vokra_gemm_f32") }?;
        drop(gemm_lib);

        // The five Phase-4 kernels share one library (compiled once); each named
        // function becomes its own pipeline.
        // SAFETY: `device` is a valid MTLDevice.
        let klib = unsafe { compile_library(device, KERNELS_MSL, "kernels") }?;
        // SAFETY: `device` valid; `klib` owns each named function below.
        let gemv_pipeline = unsafe { make_pipeline(device, klib.0, c"vokra_gemv_f32") }?;
        // SAFETY: as above.
        let softmax_pipeline = unsafe { make_pipeline(device, klib.0, c"vokra_softmax_f32") }?;
        // SAFETY: as above.
        let layer_norm_pipeline =
            unsafe { make_pipeline(device, klib.0, c"vokra_layer_norm_f32") }?;
        // SAFETY: as above.
        let gelu_pipeline = unsafe { make_pipeline(device, klib.0, c"vokra_gelu_f32") }?;
        // SAFETY: as above.
        let conv1d_pipeline = unsafe { make_pipeline(device, klib.0, c"vokra_conv1d_f32") }?;
        drop(klib);

        Ok(MetalContext {
            device,
            queue: queue.into_raw(),
            gemm_pipeline: gemm_pipeline.into_raw(),
            gemv_pipeline: gemv_pipeline.into_raw(),
            softmax_pipeline: softmax_pipeline.into_raw(),
            layer_norm_pipeline: layer_norm_pipeline.into_raw(),
            gelu_pipeline: gelu_pipeline.into_raw(),
            conv1d_pipeline: conv1d_pipeline.into_raw(),
        })
    }

    /// Row-major FP32 GEMM on the GPU with optional per-column bias:
    /// `out[i, j] = bias[j] + Σ_l a[i, l] · b[l, j]`.
    ///
    /// `a` is `m×k`, `b` is `k×n`, `out` is `m×n`, and `bias` (when `Some`) has
    /// length `n` — the exact contract of
    /// `vokra_backend_cpu::kernels::gemm_f32`, so the two are differentially
    /// comparable (M2-01-T18).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on any shape mismatch or a zero
    /// dimension; [`VokraError::BackendUnavailable`] if a Metal buffer /
    /// command object cannot be created or the command buffer reports an error.
    #[allow(clippy::too_many_arguments)] // intrinsic GEMM parameter set (matches CPU gemm_f32)
    pub fn gemm_f32(
        &self,
        m: usize,
        n: usize,
        k: usize,
        a: &[f32],
        b: &[f32],
        bias: Option<&[f32]>,
        out: &mut [f32],
    ) -> Result<()> {
        validate_gemm(m, n, k, a, b, bias, out)?;

        // Bracket the GPU work in an autorelease pool so the autoreleased
        // command buffer / encoder / any NSError drain here rather than leaking
        // until some outer pool (there is none on a plain worker thread).
        // SAFETY: `objc_autoreleasePoolPush` returns a token consumed by the one
        // matching pop below.
        let pool = unsafe { sys::objc_autoreleasePoolPush() };
        let r = self.run_gemm(m, n, k, a, b, bias, out);
        // SAFETY: `pool` is the token from the push above.
        unsafe { sys::objc_autoreleasePoolPop(pool) };
        r
    }

    /// GEMM body: allocate shared buffers, encode + run, read back. Runs inside
    /// `gemm_f32`'s autorelease pool. Shapes are already validated.
    #[allow(clippy::too_many_arguments)] // intrinsic GEMM parameter set
    fn run_gemm(
        &self,
        m: usize,
        n: usize,
        k: usize,
        a: &[f32],
        b: &[f32],
        bias: Option<&[f32]>,
        out: &mut [f32],
    ) -> Result<()> {
        // Input buffers copy host data into shared storage (Apple silicon: one
        // physical pool, so the later `contents` readback is copy-free). A
        // failed alloc `?`-returns; already-built `OwnedBuf`s release on drop.
        let a_buf = self.new_buffer_from_slice(a)?;
        let b_buf = self.new_buffer_from_slice(b)?;

        // Bias buffer: the real bias when present, else a 1-float dummy the
        // kernel never reads (has_bias = 0). Always bound so buffer(2) is set.
        let dummy = [0.0f32];
        let bias_slice = bias.unwrap_or(&dummy);
        let bias_buf = self.new_buffer_from_slice(bias_slice)?;

        // Output buffer (uninitialised shared storage of m*n floats).
        let c_buf = self.new_buffer_output(out.len())?;

        let dims = GemmDims {
            m: m as u32,
            n: n as u32,
            k: k as u32,
            has_bias: u32::from(bias.is_some()),
        };

        self.encode_and_run(&a_buf, &b_buf, &bias_buf, &c_buf, &dims, m, n)?;

        // SAFETY: `c_buf` is a valid shared buffer of `m*n` floats; after
        // `waitUntilCompleted` its `contents` pointer is host-readable.
        let contents = unsafe { sys::send_ptr(c_buf.0, sys::sel(b"contents\0")) } as *const f32;
        if contents.is_null() {
            return Err(VokraError::BackendUnavailable(
                "output MTLBuffer contents pointer is null".to_owned(),
            ));
        }
        // SAFETY: `contents` is the base of `out.len()` valid, non-overlapping
        // f32s in shared memory; copy them into the caller's slice.
        unsafe { core::ptr::copy_nonoverlapping(contents, out.as_mut_ptr(), out.len()) };
        Ok(())
    }

    /// Encodes and submits the GEMM, waiting for completion. Returns an error if
    /// the command buffer reports one.
    #[allow(clippy::too_many_arguments)] // encoder + four buffers + dims + m/n
    fn encode_and_run(
        &self,
        a_buf: &OwnedBuf,
        b_buf: &OwnedBuf,
        bias_buf: &OwnedBuf,
        c_buf: &OwnedBuf,
        dims: &GemmDims,
        m: usize,
        n: usize,
    ) -> Result<()> {
        // SAFETY: `queue` and `gemm_pipeline` are valid for the context's
        // lifetime; `commandBuffer` / `computeCommandEncoder` return
        // autoreleased objects (drained by the caller's pool). Each setter uses
        // the argument contract documented in `sys`; the four buffers are valid
        // and `dims` matches them. The two `MtlSize`s are passed per AAPCS64.
        unsafe {
            let cmd = sys::send_id(self.queue, sys::sel(b"commandBuffer\0"));
            if cmd.is_null() {
                return Err(VokraError::BackendUnavailable(
                    "MTLCommandQueue commandBuffer returned nil".to_owned(),
                ));
            }
            let enc = sys::send_id(cmd, sys::sel(b"computeCommandEncoder\0"));
            if enc.is_null() {
                return Err(VokraError::BackendUnavailable(
                    "MTLCommandBuffer computeCommandEncoder returned nil".to_owned(),
                ));
            }

            sys::send_void_id(
                enc,
                sys::sel(b"setComputePipelineState:\0"),
                self.gemm_pipeline,
            );
            let set_buffer = sys::sel(b"setBuffer:offset:atIndex:\0");
            sys::send_set_buffer(enc, set_buffer, a_buf.0, 0, 0);
            sys::send_set_buffer(enc, set_buffer, b_buf.0, 0, 1);
            sys::send_set_buffer(enc, set_buffer, bias_buf.0, 0, 2);
            sys::send_set_buffer(enc, set_buffer, c_buf.0, 0, 3);
            sys::send_set_bytes(
                enc,
                sys::sel(b"setBytes:length:atIndex:\0"),
                (dims as *const GemmDims).cast::<c_void>(),
                size_of::<GemmDims>(),
                4,
            );

            // Grid: x = columns (N), y = rows (M). 16x16 threadgroups; the
            // kernel guards row/col against M/N for ragged edges.
            const TG: usize = 16;
            let grid = MtlSize {
                width: n.div_ceil(TG),
                height: m.div_ceil(TG),
                depth: 1,
            };
            let tg = MtlSize {
                width: TG,
                height: TG,
                depth: 1,
            };
            sys::send_dispatch(
                enc,
                sys::sel(b"dispatchThreadgroups:threadsPerThreadgroup:\0"),
                grid,
                tg,
            );

            sys::send_void(enc, sys::sel(b"endEncoding\0"));
            sys::send_void(cmd, sys::sel(b"commit\0"));
            sys::send_void(cmd, sys::sel(b"waitUntilCompleted\0"));

            // Surface a GPU-side execution error explicitly (no silent success).
            let cmd_err = sys::send_id(cmd, sys::sel(b"error\0"));
            if !cmd_err.is_null() {
                let detail = error_description(cmd_err);
                return Err(VokraError::BackendUnavailable(format!(
                    "GEMM command buffer failed: {detail}"
                )));
            }
            Ok(())
        }
    }

    /// Allocates a shared-storage `MTLBuffer` initialised from `data`.
    ///
    /// A safe wrapper: `data` is a valid slice, so its pointer is valid for
    /// `size_of_val(data)` bytes, which is what `newBufferWithBytes:` copies.
    fn new_buffer_from_slice(&self, data: &[f32]) -> Result<OwnedBuf> {
        let bytes = size_of_val(data).max(size_of::<f32>());
        // SAFETY: `device` is valid; `data.as_ptr()` is valid for
        // `size_of_val(data)` bytes; shared storage mode (0). +1-owned buffer.
        let buf = unsafe {
            sys::send_new_buffer_bytes(
                self.device,
                sys::sel(b"newBufferWithBytes:length:options:\0"),
                data.as_ptr().cast::<c_void>(),
                bytes,
                sys::STORAGE_MODE_SHARED,
            )
        };
        if buf.is_null() {
            return Err(VokraError::BackendUnavailable(
                "MTLDevice newBufferWithBytes returned nil".to_owned(),
            ));
        }
        Ok(OwnedBuf(buf))
    }

    /// Allocates an uninitialised shared-storage `MTLBuffer` of `len` f32s.
    fn new_buffer_output(&self, len: usize) -> Result<OwnedBuf> {
        let bytes = (len * size_of::<f32>()).max(size_of::<f32>());
        // SAFETY: `device` is valid; shared storage mode (0). +1-owned buffer.
        let buf = unsafe {
            sys::send_new_buffer_len(
                self.device,
                sys::sel(b"newBufferWithLength:options:\0"),
                bytes,
                sys::STORAGE_MODE_SHARED,
            )
        };
        if buf.is_null() {
            return Err(VokraError::BackendUnavailable(
                "MTLDevice newBufferWithLength returned nil".to_owned(),
            ));
        }
        Ok(OwnedBuf(buf))
    }

    // ---- Phase-4 kernels (M2-01 T09-T13): gemv / softmax / layer_norm / gelu /
    // conv1d. Each mirrors the `vokra_backend_cpu::kernels` contract and numerics
    // (FP32, `atol = 0.01`), brackets the GPU work in an autorelease pool, and
    // reads back copy-free from shared storage — exactly like `gemm_f32`.

    /// Row-major FP32 matrix-vector product with optional per-row bias:
    /// `out[i] = bias[i] + Σ_l a[i, l] · x[l]`. `a` is `m×k`, `x` length `k`,
    /// `out` length `m`, `bias` (when `Some`) length `m` — the exact contract of
    /// `vokra_backend_cpu::kernels::gemv_f32`.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on a shape mismatch;
    /// [`VokraError::BackendUnavailable`] on a Metal allocation / command failure.
    pub fn gemv_f32(
        &self,
        m: usize,
        k: usize,
        a: &[f32],
        x: &[f32],
        bias: Option<&[f32]>,
        out: &mut [f32],
    ) -> Result<()> {
        validate_gemv(m, k, a, x, bias, out)?;
        if out.is_empty() {
            return Ok(());
        }
        // SAFETY: `objc_autoreleasePoolPush` returns a token consumed by the one
        // matching pop below.
        let pool = unsafe { sys::objc_autoreleasePoolPush() };
        let r = self.run_gemv(m, k, a, x, bias, out);
        // SAFETY: `pool` is the token from the push above.
        unsafe { sys::objc_autoreleasePoolPop(pool) };
        r
    }

    fn run_gemv(
        &self,
        m: usize,
        k: usize,
        a: &[f32],
        x: &[f32],
        bias: Option<&[f32]>,
        out: &mut [f32],
    ) -> Result<()> {
        let a_buf = self.new_buffer_from_slice(a)?;
        let x_buf = self.new_buffer_from_slice(x)?;
        let dummy = [0.0f32];
        let bias_buf = self.new_buffer_from_slice(bias.unwrap_or(&dummy))?;
        let out_buf = self.new_buffer_output(out.len())?;
        let dims = GemvDims {
            m: m as u32,
            k: k as u32,
            has_bias: u32::from(bias.is_some()),
        };
        let (grid, tg) = grid_1d(m);
        self.dispatch_compute(
            self.gemv_pipeline,
            &[&a_buf, &x_buf, &bias_buf, &out_buf],
            (&dims as *const GemvDims).cast::<c_void>(),
            size_of::<GemvDims>(),
            grid,
            tg,
            "gemv",
        )?;
        read_back(&out_buf, out)
    }

    /// Row-wise softmax over the innermost axis of a `rows × cols` buffer,
    /// max-subtracted — the exact contract of
    /// `vokra_backend_cpu::kernels::softmax_f32` (a causal-mask `-inf` score maps
    /// to a 0 weight, as on the CPU).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on a shape mismatch;
    /// [`VokraError::BackendUnavailable`] on a Metal failure.
    pub fn softmax_f32(
        &self,
        input: &[f32],
        out: &mut [f32],
        rows: usize,
        cols: usize,
    ) -> Result<()> {
        validate_rows_cols(input, out, rows, cols)?;
        if out.is_empty() {
            return Ok(());
        }
        // SAFETY: token consumed by the matching pop below.
        let pool = unsafe { sys::objc_autoreleasePoolPush() };
        let r = self.run_softmax(input, out, rows, cols);
        // SAFETY: `pool` is the token from the push above.
        unsafe { sys::objc_autoreleasePoolPop(pool) };
        r
    }

    fn run_softmax(&self, input: &[f32], out: &mut [f32], rows: usize, cols: usize) -> Result<()> {
        let in_buf = self.new_buffer_from_slice(input)?;
        let out_buf = self.new_buffer_output(out.len())?;
        let dims = SoftmaxDims {
            rows: rows as u32,
            cols: cols as u32,
        };
        let (grid, tg) = grid_1d(rows);
        self.dispatch_compute(
            self.softmax_pipeline,
            &[&in_buf, &out_buf],
            (&dims as *const SoftmaxDims).cast::<c_void>(),
            size_of::<SoftmaxDims>(),
            grid,
            tg,
            "softmax",
        )?;
        read_back(&out_buf, out)
    }

    /// Affine layer normalisation over the innermost axis of a `rows × cols`
    /// buffer, biased (population) variance — the exact contract of
    /// `vokra_backend_cpu::kernels::layer_norm_f32` (`gamma` / `beta` length
    /// `cols`).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on a shape mismatch;
    /// [`VokraError::BackendUnavailable`] on a Metal failure.
    #[allow(clippy::too_many_arguments)] // intrinsic layer-norm parameter set (matches CPU layer_norm_f32)
    pub fn layer_norm_f32(
        &self,
        input: &[f32],
        out: &mut [f32],
        rows: usize,
        cols: usize,
        gamma: &[f32],
        beta: &[f32],
        eps: f32,
    ) -> Result<()> {
        validate_layer_norm(input, out, rows, cols, gamma, beta)?;
        if out.is_empty() {
            return Ok(());
        }
        // SAFETY: token consumed by the matching pop below.
        let pool = unsafe { sys::objc_autoreleasePoolPush() };
        let r = self.run_layer_norm(input, out, rows, cols, gamma, beta, eps);
        // SAFETY: `pool` is the token from the push above.
        unsafe { sys::objc_autoreleasePoolPop(pool) };
        r
    }

    #[allow(clippy::too_many_arguments)] // intrinsic layer-norm parameter set
    fn run_layer_norm(
        &self,
        input: &[f32],
        out: &mut [f32],
        rows: usize,
        cols: usize,
        gamma: &[f32],
        beta: &[f32],
        eps: f32,
    ) -> Result<()> {
        let in_buf = self.new_buffer_from_slice(input)?;
        let gamma_buf = self.new_buffer_from_slice(gamma)?;
        let beta_buf = self.new_buffer_from_slice(beta)?;
        let out_buf = self.new_buffer_output(out.len())?;
        let dims = LayerNormDims {
            rows: rows as u32,
            cols: cols as u32,
            eps,
        };
        let (grid, tg) = grid_1d(rows);
        self.dispatch_compute(
            self.layer_norm_pipeline,
            &[&in_buf, &gamma_buf, &beta_buf, &out_buf],
            (&dims as *const LayerNormDims).cast::<c_void>(),
            size_of::<LayerNormDims>(),
            grid,
            tg,
            "layer_norm",
        )?;
        read_back(&out_buf, out)
    }

    /// Element-wise exact (erf) GELU (`x` and `out` equal length) — the contract
    /// of `vokra_backend_cpu::kernels::gelu_f32`. Uses MSL's precise `erf`; the
    /// CPU uses the A&S 7.1.26 approximation, so the two agree far inside the FP32
    /// bound.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on a length mismatch;
    /// [`VokraError::BackendUnavailable`] on a Metal failure.
    pub fn gelu_f32(&self, x: &[f32], out: &mut [f32]) -> Result<()> {
        validate_unary(x, out)?;
        if out.is_empty() {
            return Ok(());
        }
        // SAFETY: token consumed by the matching pop below.
        let pool = unsafe { sys::objc_autoreleasePoolPush() };
        let r = self.run_gelu(x, out);
        // SAFETY: `pool` is the token from the push above.
        unsafe { sys::objc_autoreleasePoolPop(pool) };
        r
    }

    fn run_gelu(&self, x: &[f32], out: &mut [f32]) -> Result<()> {
        let x_buf = self.new_buffer_from_slice(x)?;
        let out_buf = self.new_buffer_output(out.len())?;
        let dims = GeluDims {
            n: out.len() as u32,
        };
        let (grid, tg) = grid_1d(out.len());
        self.dispatch_compute(
            self.gelu_pipeline,
            &[&x_buf, &out_buf],
            (&dims as *const GeluDims).cast::<c_void>(),
            size_of::<GeluDims>(),
            grid,
            tg,
            "gelu",
        )?;
        read_back(&out_buf, out)
    }

    /// 1-D convolution (`input` is `in_ch × in_len`, `weight` is
    /// `out_ch × in_ch × kernel`, `out` is `out_ch × out_len`) — the exact
    /// contract of `vokra_backend_cpu::kernels::conv1d_f32`. The direct GPU
    /// convolution reduces in the same `(in_ch, tap)` order as the CPU's
    /// im2col + GEMM, so the two agree within the FP32 bound.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on a zero `stride`/`kernel`, a padded
    /// length below `kernel`, or a shape mismatch;
    /// [`VokraError::BackendUnavailable`] on a Metal failure.
    #[allow(clippy::too_many_arguments)] // convolution's intrinsic parameter set (matches CPU conv1d_f32)
    pub fn conv1d_f32(
        &self,
        input: &[f32],
        in_ch: usize,
        in_len: usize,
        weight: &[f32],
        out_ch: usize,
        kernel: usize,
        bias: Option<&[f32]>,
        stride: usize,
        padding: usize,
        out: &mut [f32],
    ) -> Result<()> {
        let out_len = validate_conv1d(
            input, in_ch, in_len, weight, out_ch, kernel, bias, stride, padding, out,
        )?;
        if out.is_empty() {
            return Ok(());
        }
        // SAFETY: token consumed by the matching pop below.
        let pool = unsafe { sys::objc_autoreleasePoolPush() };
        let r = self.run_conv1d(
            input, in_ch, in_len, weight, out_ch, kernel, bias, stride, padding, out_len, out,
        );
        // SAFETY: `pool` is the token from the push above.
        unsafe { sys::objc_autoreleasePoolPop(pool) };
        r
    }

    #[allow(clippy::too_many_arguments)] // convolution's intrinsic parameter set
    fn run_conv1d(
        &self,
        input: &[f32],
        in_ch: usize,
        in_len: usize,
        weight: &[f32],
        out_ch: usize,
        kernel: usize,
        bias: Option<&[f32]>,
        stride: usize,
        padding: usize,
        out_len: usize,
        out: &mut [f32],
    ) -> Result<()> {
        let in_buf = self.new_buffer_from_slice(input)?;
        let w_buf = self.new_buffer_from_slice(weight)?;
        let dummy = [0.0f32];
        let bias_buf = self.new_buffer_from_slice(bias.unwrap_or(&dummy))?;
        let out_buf = self.new_buffer_output(out.len())?;
        let dims = Conv1dDims {
            in_ch: in_ch as u32,
            in_len: in_len as u32,
            out_ch: out_ch as u32,
            kernel_size: kernel as u32,
            out_len: out_len as u32,
            stride: stride as u32,
            padding: padding as u32,
            has_bias: u32::from(bias.is_some()),
        };
        let (grid, tg) = grid_2d(out_len, out_ch);
        self.dispatch_compute(
            self.conv1d_pipeline,
            &[&in_buf, &w_buf, &bias_buf, &out_buf],
            (&dims as *const Conv1dDims).cast::<c_void>(),
            size_of::<Conv1dDims>(),
            grid,
            tg,
            "conv1d",
        )?;
        read_back(&out_buf, out)
    }

    // ---- Phase-5 fusion: device-resident MLP (readback elimination) ----------

    /// Fused MLP `fc2(gelu(fc1(x)))` on the GPU with the two `[t, ffn]`
    /// intermediates **resident on the device** — the Phase-5 readback-
    /// elimination slice.
    ///
    /// `x` is `[t, d]`; `fc1` maps `d → ffn` (`fc1_w` is `[d, ffn]`, optional
    /// bias `[ffn]`); `fc2` maps `ffn → d` (`fc2_w` is `[ffn, d]`, optional bias
    /// `[d]`); `out` is `[t, d]`. It runs the very same three kernels
    /// (`vokra_gemm_f32` → `vokra_gelu_f32` → `vokra_gemm_f32`) the per-op
    /// [`Self::gemm_f32`] / [`Self::gelu_f32`] path runs, in the same order and
    /// with the same launch geometry, so the result is **bit-identical** to three
    /// separate calls — but the `[t, ffn]` intermediates `h` and `a` are never
    /// copied back to the host, and the whole chain is ONE command buffer with
    /// ONE `waitUntilCompleted` and ONE readback (of `out`) instead of three of
    /// each.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on any shape mismatch or a zero dimension;
    /// [`VokraError::BackendUnavailable`] on a Metal buffer / command failure.
    #[allow(clippy::too_many_arguments)] // fused-MLP operand set (two Linears + dims)
    pub fn mlp_f32(
        &self,
        t: usize,
        d: usize,
        ffn: usize,
        x: &[f32],
        fc1_w: &[f32],
        fc1_bias: Option<&[f32]>,
        fc2_w: &[f32],
        fc2_bias: Option<&[f32]>,
        out: &mut [f32],
    ) -> Result<()> {
        validate_mlp(t, d, ffn, x, fc1_w, fc1_bias, fc2_w, fc2_bias, out)?;
        // Bracket the GPU work in an autorelease pool (as the per-op methods do).
        // SAFETY: `objc_autoreleasePoolPush` returns a token consumed by the one
        // matching pop below.
        let pool = unsafe { sys::objc_autoreleasePoolPush() };
        let r = self.run_mlp(t, d, ffn, x, fc1_w, fc1_bias, fc2_w, fc2_bias, out);
        // SAFETY: `pool` is the token from the push above.
        unsafe { sys::objc_autoreleasePoolPop(pool) };
        r
    }

    /// Fused-MLP body: copy the five inputs H2D, allocate the two `[t, ffn]`
    /// intermediates **device-resident** (never read back) plus the `[t, d]`
    /// output, encode the three passes (fc1 GEMM → GELU → fc2 GEMM) into ONE
    /// command buffer, commit + wait ONCE, and read back only `out`. Runs inside
    /// `mlp_f32`'s autorelease pool; shapes are already validated.
    #[allow(clippy::too_many_arguments)] // fused-MLP operand set (two Linears + dims)
    fn run_mlp(
        &self,
        t: usize,
        d: usize,
        ffn: usize,
        x: &[f32],
        fc1_w: &[f32],
        fc1_bias: Option<&[f32]>,
        fc2_w: &[f32],
        fc2_bias: Option<&[f32]>,
        out: &mut [f32],
    ) -> Result<()> {
        // Inputs copied H2D into shared storage (a failed alloc `?`-returns;
        // already-built `OwnedBuf`s release on drop).
        let x_buf = self.new_buffer_from_slice(x)?;
        let fc1_w_buf = self.new_buffer_from_slice(fc1_w)?;
        let dummy = [0.0f32];
        let fc1_bias_buf = self.new_buffer_from_slice(fc1_bias.unwrap_or(&dummy))?;
        let fc2_w_buf = self.new_buffer_from_slice(fc2_w)?;
        let fc2_bias_buf = self.new_buffer_from_slice(fc2_bias.unwrap_or(&dummy))?;

        // The two `[t, ffn]` intermediates live only on the GPU: uninitialised
        // shared buffers the kernels write and read but that are NEVER copied
        // back to the host (the readback this slice exists to eliminate). `out`
        // is the single buffer read back.
        let inter = checked_mul(t, ffn, "mlp t*ffn")?;
        let h_buf = self.new_buffer_output(inter)?; // fc1 output [t, ffn]
        let a_buf = self.new_buffer_output(inter)?; // gelu output [t, ffn]
        let out_buf = self.new_buffer_output(out.len())?; // [t, d]

        let fc1_dims = GemmDims {
            m: t as u32,
            n: ffn as u32,
            k: d as u32,
            has_bias: u32::from(fc1_bias.is_some()),
        };
        let gelu_dims = GeluDims { n: inter as u32 };
        let fc2_dims = GemmDims {
            m: t as u32,
            n: d as u32,
            k: ffn as u32,
            has_bias: u32::from(fc2_bias.is_some()),
        };

        // One command buffer for the whole chain.
        // SAFETY: `queue` is valid for the context's lifetime; `commandBuffer`
        // returns an autoreleased command buffer drained by `mlp_f32`'s pool.
        let cmd = unsafe { sys::send_id(self.queue, sys::sel(b"commandBuffer\0")) };
        if cmd.is_null() {
            return Err(VokraError::BackendUnavailable(
                "mlp: MTLCommandQueue commandBuffer returned nil".to_owned(),
            ));
        }

        // Pass 1: h = x[t,d] · fc1_w[d,ffn] (+bias) — GEMM (grid = N×M, 16×16).
        let (fc1_grid, fc1_tg) = grid_2d(ffn, t);
        self.encode_pass(
            cmd,
            self.gemm_pipeline,
            &[&x_buf, &fc1_w_buf, &fc1_bias_buf, &h_buf],
            (&fc1_dims as *const GemmDims).cast::<c_void>(),
            size_of::<GemmDims>(),
            fc1_grid,
            fc1_tg,
            "mlp fc1",
        )?;
        // Pass 2: a = gelu(h) — element-wise (1-D grid over t*ffn).
        let (g_grid, g_tg) = grid_1d(inter);
        self.encode_pass(
            cmd,
            self.gelu_pipeline,
            &[&h_buf, &a_buf],
            (&gelu_dims as *const GeluDims).cast::<c_void>(),
            size_of::<GeluDims>(),
            g_grid,
            g_tg,
            "mlp gelu",
        )?;
        // Pass 3: out = a[t,ffn] · fc2_w[ffn,d] (+bias) — GEMM (grid = N×M).
        let (fc2_grid, fc2_tg) = grid_2d(d, t);
        self.encode_pass(
            cmd,
            self.gemm_pipeline,
            &[&a_buf, &fc2_w_buf, &fc2_bias_buf, &out_buf],
            (&fc2_dims as *const GemmDims).cast::<c_void>(),
            size_of::<GemmDims>(),
            fc2_grid,
            fc2_tg,
            "mlp fc2",
        )?;

        // Commit the whole chain and wait ONCE, then surface any GPU-side error.
        // SAFETY: `cmd` is the valid command buffer encoded above; `commit` then
        // `waitUntilCompleted` submit and block; `error` is read after completion
        // (no silent success).
        unsafe {
            sys::send_void(cmd, sys::sel(b"commit\0"));
            sys::send_void(cmd, sys::sel(b"waitUntilCompleted\0"));
            let cmd_err = sys::send_id(cmd, sys::sel(b"error\0"));
            if !cmd_err.is_null() {
                let detail = error_description(cmd_err);
                return Err(VokraError::BackendUnavailable(format!(
                    "mlp command buffer failed: {detail}"
                )));
            }
        }

        // Single readback of the final output; `h`/`a` stay resident and drop.
        read_back(&out_buf, out)
    }

    /// Encodes ONE compute pass into `cmd` **without** committing or waiting: a
    /// fresh compute encoder binds `buffers` at indices `0..buffers.len()`, sets
    /// `dims` (a `constant` struct) at `buffers.len()` via `setBytes:`,
    /// dispatches `grid` threadgroups of `tg`, and ends. The fused MLP
    /// ([`Self::mlp_f32`]) chains three of these into one command buffer, then
    /// commits + waits once. Each pass is its own encoder over hazard-tracked
    /// shared buffers, so Metal orders a later pass's reads after an earlier
    /// pass's writes (fc1 → gelu → fc2 see each other's outputs) with no host
    /// round-trip. Distinct from [`Self::dispatch_compute`], which owns the whole
    /// command-buffer lifecycle for a single per-op kernel (left untouched).
    #[allow(clippy::too_many_arguments)] // cmd + pipeline + buffers + dims + grid/tg + label
    fn encode_pass(
        &self,
        cmd: Id,
        pipeline: Id,
        buffers: &[&OwnedBuf],
        dims: *const c_void,
        dims_len: usize,
        grid: MtlSize,
        tg: MtlSize,
        what: &str,
    ) -> Result<()> {
        // SAFETY: `cmd` is a valid command buffer from this context's queue;
        // `computeCommandEncoder` returns an autoreleased encoder (drained by the
        // caller's pool); `pipeline` is one of the context's compiled pipelines;
        // each `buffers[i]` is a valid MTLBuffer bound at index `i`; `dims` points
        // to `dims_len` readable bytes matching the kernel's `constant` struct at
        // index `buffers.len()`; the two `MtlSize`s are passed per AAPCS64.
        unsafe {
            let enc = sys::send_id(cmd, sys::sel(b"computeCommandEncoder\0"));
            if enc.is_null() {
                return Err(VokraError::BackendUnavailable(format!(
                    "{what}: MTLCommandBuffer computeCommandEncoder returned nil"
                )));
            }
            sys::send_void_id(enc, sys::sel(b"setComputePipelineState:\0"), pipeline);
            let set_buffer = sys::sel(b"setBuffer:offset:atIndex:\0");
            for (i, buf) in buffers.iter().enumerate() {
                sys::send_set_buffer(enc, set_buffer, buf.0, 0, i);
            }
            sys::send_set_bytes(
                enc,
                sys::sel(b"setBytes:length:atIndex:\0"),
                dims,
                dims_len,
                buffers.len(),
            );
            sys::send_dispatch(
                enc,
                sys::sel(b"dispatchThreadgroups:threadsPerThreadgroup:\0"),
                grid,
                tg,
            );
            sys::send_void(enc, sys::sel(b"endEncoding\0"));
            Ok(())
        }
    }

    /// Encodes a compute pass: binds `buffers` at indices `0..buffers.len()`, sets
    /// `dims` (a `constant` struct) at index `buffers.len()` via `setBytes:`,
    /// dispatches `grid` threadgroups of `tg` threads, waits, and surfaces a
    /// command-buffer error explicitly. Shared by the five Phase-4 kernels
    /// (the GEMM keeps its own bespoke `encode_and_run`).
    #[allow(clippy::too_many_arguments)] // encoder + buffers + dims + grid/tg + label
    fn dispatch_compute(
        &self,
        pipeline: Id,
        buffers: &[&OwnedBuf],
        dims: *const c_void,
        dims_len: usize,
        grid: MtlSize,
        tg: MtlSize,
        what: &str,
    ) -> Result<()> {
        // SAFETY: `queue` and `pipeline` are valid for the context's lifetime;
        // `commandBuffer` / `computeCommandEncoder` return autoreleased objects
        // (drained by the caller's pool). Each `buffers[i]` is a valid MTLBuffer
        // bound at index `i`; `dims` points to `dims_len` readable bytes matching
        // the kernel's `constant` struct at index `buffers.len()`. The two
        // `MtlSize`s are passed per AAPCS64.
        unsafe {
            let cmd = sys::send_id(self.queue, sys::sel(b"commandBuffer\0"));
            if cmd.is_null() {
                return Err(VokraError::BackendUnavailable(format!(
                    "{what}: MTLCommandQueue commandBuffer returned nil"
                )));
            }
            let enc = sys::send_id(cmd, sys::sel(b"computeCommandEncoder\0"));
            if enc.is_null() {
                return Err(VokraError::BackendUnavailable(format!(
                    "{what}: MTLCommandBuffer computeCommandEncoder returned nil"
                )));
            }

            sys::send_void_id(enc, sys::sel(b"setComputePipelineState:\0"), pipeline);
            let set_buffer = sys::sel(b"setBuffer:offset:atIndex:\0");
            for (i, buf) in buffers.iter().enumerate() {
                sys::send_set_buffer(enc, set_buffer, buf.0, 0, i);
            }
            sys::send_set_bytes(
                enc,
                sys::sel(b"setBytes:length:atIndex:\0"),
                dims,
                dims_len,
                buffers.len(),
            );
            sys::send_dispatch(
                enc,
                sys::sel(b"dispatchThreadgroups:threadsPerThreadgroup:\0"),
                grid,
                tg,
            );

            sys::send_void(enc, sys::sel(b"endEncoding\0"));
            sys::send_void(cmd, sys::sel(b"commit\0"));
            sys::send_void(cmd, sys::sel(b"waitUntilCompleted\0"));

            let cmd_err = sys::send_id(cmd, sys::sel(b"error\0"));
            if !cmd_err.is_null() {
                let detail = error_description(cmd_err);
                return Err(VokraError::BackendUnavailable(format!(
                    "{what} command buffer failed: {detail}"
                )));
            }
            Ok(())
        }
    }
}

impl Drop for MetalContext {
    fn drop(&mut self) {
        // SAFETY: every handle is a valid `+1`-owned object created in
        // `new` / `build`; release each exactly once.
        unsafe {
            release(self.conv1d_pipeline);
            release(self.gelu_pipeline);
            release(self.layer_norm_pipeline);
            release(self.softmax_pipeline);
            release(self.gemv_pipeline);
            release(self.gemm_pipeline);
            release(self.queue);
            release(self.device);
        }
    }
}

/// 1-D launch: `count` threads in `TG`-wide threadgroups (grid measured in
/// threadgroups, like the GEMM launch); the kernel guards the ragged tail.
fn grid_1d(count: usize) -> (MtlSize, MtlSize) {
    const TG: usize = 256;
    (
        MtlSize {
            width: count.div_ceil(TG),
            height: 1,
            depth: 1,
        },
        MtlSize {
            width: TG,
            height: 1,
            depth: 1,
        },
    )
}

/// 2-D launch: `nx × ny` threads in `16×16` threadgroups (grid in threadgroups);
/// the kernel guards the ragged edges.
fn grid_2d(nx: usize, ny: usize) -> (MtlSize, MtlSize) {
    const TG: usize = 16;
    (
        MtlSize {
            width: nx.div_ceil(TG),
            height: ny.div_ceil(TG),
            depth: 1,
        },
        MtlSize {
            width: TG,
            height: TG,
            depth: 1,
        },
    )
}

/// Copies `out.len()` f32s from a shared output buffer's `contents` into `out`.
/// On Apple silicon `contents` is the same physical memory the GPU wrote, so
/// this is copy-free after `waitUntilCompleted`.
fn read_back(buf: &OwnedBuf, out: &mut [f32]) -> Result<()> {
    // SAFETY: `buf` is a valid shared MTLBuffer of at least `out.len()` f32s;
    // after the dispatch's `waitUntilCompleted` its `contents` is host-readable.
    let contents = unsafe { sys::send_ptr(buf.0, sys::sel(b"contents\0")) } as *const f32;
    if contents.is_null() {
        return Err(VokraError::BackendUnavailable(
            "output MTLBuffer contents pointer is null".to_owned(),
        ));
    }
    // SAFETY: `contents` is the base of `out.len()` valid, non-overlapping f32s in
    // shared memory; copy them into the caller's slice.
    unsafe { core::ptr::copy_nonoverlapping(contents, out.as_mut_ptr(), out.len()) };
    Ok(())
}

/// Compiles MSL `source` into an `MTLLibrary` on `device` (returned owned).
/// `what` names the shader in any compile-error message.
///
/// # Safety
/// `device` must be a valid, non-null `MTLDevice`.
unsafe fn compile_library(device: Id, source: &str, what: &str) -> Result<Owned> {
    let csource = std::ffi::CString::new(source).map_err(|_| {
        VokraError::InvalidArgument(format!("{what} MSL source contains an interior NUL"))
    })?;
    // SAFETY: NSString class is loaded (Foundation linked); `csource` is a valid
    // NUL-terminated C string. The returned NSString is autoreleased.
    let ns_source = unsafe {
        sys::send_id_cstr(
            sys::class(b"NSString\0"),
            sys::sel(b"stringWithUTF8String:\0"),
            csource.as_ptr(),
        )
    };
    let mut err: Id = core::ptr::null_mut();
    // SAFETY: `newLibraryWithSource:options:error:` on a valid device; nil options
    // selects defaults; `&mut err` receives an autoreleased NSError on failure.
    let library = unsafe {
        sys::send_new_library(
            device,
            sys::sel(b"newLibraryWithSource:options:error:\0"),
            ns_source,
            core::ptr::null_mut(),
            &mut err,
        )
    };
    if library.is_null() {
        // SAFETY: `err` is null or a valid autoreleased NSError.
        let detail = unsafe { error_description(err) };
        return Err(VokraError::BackendUnavailable(format!(
            "MSL {what} shader failed to compile: {detail}"
        )));
    }
    Ok(Owned(library))
}

/// Builds a compute pipeline for the function named `fname` in `library`
/// (returned owned). The transient `MTLFunction` is released on every path.
///
/// # Safety
/// `device` must be a valid `MTLDevice`; `library` a valid `MTLLibrary`.
unsafe fn make_pipeline(device: Id, library: Id, fname: &core::ffi::CStr) -> Result<Owned> {
    // SAFETY: NSString built from a valid C string; `newFunctionWithName:` returns
    // a `+1`-owned function (or null).
    let function = unsafe {
        let ns = sys::send_id_cstr(
            sys::class(b"NSString\0"),
            sys::sel(b"stringWithUTF8String:\0"),
            fname.as_ptr(),
        );
        sys::send_id_id(library, sys::sel(b"newFunctionWithName:\0"), ns)
    };
    if function.is_null() {
        return Err(VokraError::BackendUnavailable(format!(
            "MTLLibrary has no function named {fname:?}"
        )));
    }
    // Owned so it is released whether pipeline creation succeeds or fails.
    let function = Owned(function);
    let mut perr: Id = core::ptr::null_mut();
    // SAFETY: `newComputePipelineStateWithFunction:error:` on a valid device with
    // a valid function; `&mut perr` receives an autoreleased NSError on failure.
    let pipeline = unsafe {
        sys::send_new_pipeline(
            device,
            sys::sel(b"newComputePipelineStateWithFunction:error:\0"),
            function.0,
            &mut perr,
        )
    };
    if pipeline.is_null() {
        // SAFETY: `perr` is null or a valid autoreleased NSError.
        let detail = unsafe { error_description(perr) };
        return Err(VokraError::BackendUnavailable(format!(
            "compute pipeline creation failed for {fname:?}: {detail}"
        )));
    }
    // `function` drops here → released (the pipeline retains what it needs).
    Ok(Owned(pipeline))
}

impl core::fmt::Debug for MetalContext {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("MetalContext").finish_non_exhaustive()
    }
}

/// Sends `-release` to a non-null owned object.
///
/// # Safety
/// `obj` must be a valid `+1`-owned Objective-C object (or null).
#[inline]
unsafe fn release(obj: Id) {
    if !obj.is_null() {
        // SAFETY: `obj` is a valid owned object per the caller contract.
        unsafe { sys::send_void(obj, sys::sel(b"release\0")) };
    }
}

/// Extracts `-[NSError localizedDescription]` as a String (best effort).
///
/// # Safety
/// `err` must be null or a valid `NSError`.
unsafe fn error_description(err: Id) -> String {
    if err.is_null() {
        return "(no error object)".to_owned();
    }
    // SAFETY: `localizedDescription` is a valid `-(NSString*)` selector on
    // NSError; the result is autoreleased and read within the caller's pool.
    let desc = unsafe { sys::send_id(err, sys::sel(b"localizedDescription\0")) };
    // SAFETY: `desc` is null or a valid NSString.
    unsafe { sys::nsstring_to_string(desc) }.unwrap_or_else(|| "(no description)".to_owned())
}

// ---- shape validation (mirrors vokra-backend-cpu's gemm validator) ----

fn checked_mul(a: usize, b: usize, what: &str) -> Result<usize> {
    a.checked_mul(b).ok_or_else(|| {
        VokraError::InvalidArgument(format!("{what}: dimension product overflows usize"))
    })
}

fn expect_len(name: &str, got: usize, want: usize) -> Result<()> {
    if got == want {
        Ok(())
    } else {
        Err(VokraError::InvalidArgument(format!(
            "{name} length {got} does not match expected {want}"
        )))
    }
}

fn validate_gemm(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    bias: Option<&[f32]>,
    out: &[f32],
) -> Result<()> {
    if m == 0 || n == 0 || k == 0 {
        return Err(VokraError::InvalidArgument(
            "gemm dimensions m, n, k must all be >= 1".to_owned(),
        ));
    }
    expect_len("gemm a", a.len(), checked_mul(m, k, "gemm m*k")?)?;
    expect_len("gemm b", b.len(), checked_mul(k, n, "gemm k*n")?)?;
    expect_len("gemm out", out.len(), checked_mul(m, n, "gemm m*n")?)?;
    if let Some(bias) = bias {
        expect_len("gemm bias", bias.len(), n)?;
    }
    Ok(())
}

fn validate_gemv(
    m: usize,
    k: usize,
    a: &[f32],
    x: &[f32],
    bias: Option<&[f32]>,
    out: &[f32],
) -> Result<()> {
    expect_len("gemv a", a.len(), checked_mul(m, k, "gemv m*k")?)?;
    expect_len("gemv x", x.len(), k)?;
    expect_len("gemv out", out.len(), m)?;
    if let Some(bias) = bias {
        expect_len("gemv bias", bias.len(), m)?;
    }
    Ok(())
}

fn validate_rows_cols(input: &[f32], out: &[f32], rows: usize, cols: usize) -> Result<()> {
    let total = checked_mul(rows, cols, "rows*cols")?;
    expect_len("input", input.len(), total)?;
    expect_len("out", out.len(), total)
}

fn validate_layer_norm(
    input: &[f32],
    out: &[f32],
    rows: usize,
    cols: usize,
    gamma: &[f32],
    beta: &[f32],
) -> Result<()> {
    validate_rows_cols(input, out, rows, cols)?;
    expect_len("layer_norm gamma", gamma.len(), cols)?;
    expect_len("layer_norm beta", beta.len(), cols)
}

fn validate_unary(x: &[f32], out: &[f32]) -> Result<()> {
    expect_len("unary out", out.len(), x.len())
}

/// Validates the conv1d shapes (mirroring the CPU `conv1d` guard) and returns the
/// derived `out_len = (in_len + 2·padding − kernel) / stride + 1`.
#[allow(clippy::too_many_arguments)] // convolution's intrinsic parameter set
fn validate_conv1d(
    input: &[f32],
    in_ch: usize,
    in_len: usize,
    weight: &[f32],
    out_ch: usize,
    kernel: usize,
    bias: Option<&[f32]>,
    stride: usize,
    padding: usize,
    out: &[f32],
) -> Result<usize> {
    if stride == 0 {
        return Err(VokraError::InvalidArgument(
            "conv1d stride must be >= 1".to_owned(),
        ));
    }
    if kernel == 0 {
        return Err(VokraError::InvalidArgument(
            "conv1d kernel must be >= 1".to_owned(),
        ));
    }
    let padded = in_len
        .checked_add(checked_mul(2, padding, "conv1d 2*padding")?)
        .ok_or_else(|| VokraError::InvalidArgument("conv1d padded length overflow".to_owned()))?;
    if padded < kernel {
        return Err(VokraError::InvalidArgument(format!(
            "conv1d padded length {padded} is smaller than kernel {kernel}"
        )));
    }
    let out_len = (padded - kernel) / stride + 1;
    expect_len(
        "conv1d input",
        input.len(),
        checked_mul(in_ch, in_len, "conv1d in_ch*in_len")?,
    )?;
    let k = checked_mul(in_ch, kernel, "conv1d in_ch*kernel")?;
    expect_len(
        "conv1d weight",
        weight.len(),
        checked_mul(out_ch, k, "conv1d out_ch*k")?,
    )?;
    expect_len(
        "conv1d out",
        out.len(),
        checked_mul(out_ch, out_len, "conv1d out_ch*out_len")?,
    )?;
    if let Some(bias) = bias {
        expect_len("conv1d bias", bias.len(), out_ch)?;
    }
    Ok(out_len)
}

/// Validates the fused-MLP shapes: `x` is `[t, d]`, `fc1_w` is `[d, ffn]` (bias
/// `[ffn]`), `fc2_w` is `[ffn, d]` (bias `[d]`), `out` is `[t, d]` — the
/// composition of the two GEMM validators the fused path chains, so a mis-shaped
/// call is an explicit `InvalidArgument` rather than a GPU fault.
#[allow(clippy::too_many_arguments)] // fused-MLP operand set (two Linears + dims)
fn validate_mlp(
    t: usize,
    d: usize,
    ffn: usize,
    x: &[f32],
    fc1_w: &[f32],
    fc1_bias: Option<&[f32]>,
    fc2_w: &[f32],
    fc2_bias: Option<&[f32]>,
    out: &[f32],
) -> Result<()> {
    if t == 0 || d == 0 || ffn == 0 {
        return Err(VokraError::InvalidArgument(
            "mlp dimensions t, d, ffn must all be >= 1".to_owned(),
        ));
    }
    expect_len("mlp x", x.len(), checked_mul(t, d, "mlp t*d")?)?;
    expect_len("mlp fc1_w", fc1_w.len(), checked_mul(d, ffn, "mlp d*ffn")?)?;
    if let Some(bias) = fc1_bias {
        expect_len("mlp fc1_bias", bias.len(), ffn)?;
    }
    expect_len("mlp fc2_w", fc2_w.len(), checked_mul(ffn, d, "mlp ffn*d")?)?;
    if let Some(bias) = fc2_bias {
        expect_len("mlp fc2_bias", bias.len(), d)?;
    }
    expect_len("mlp out", out.len(), checked_mul(t, d, "mlp out t*d")?)?;
    Ok(())
}
