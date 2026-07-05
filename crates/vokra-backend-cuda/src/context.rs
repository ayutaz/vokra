//! CUDA working context: driver + device + context + stream + the FP32 GEMM
//! kernel (NVRTC-compiled PTX). Unix / Windows only.
//!
//! This is the **directly callable** compute surface, mirroring
//! `vokra-backend-metal`'s `MetalContext` and `vokra-backend-cpu`'s `kernels::*`:
//! [`CudaContext::gemm_f32`] runs a row-major single-precision GEMM on the GPU
//! with the **exact** shape/semantics contract of
//! `vokra_backend_cpu::kernels::gemm_f32` (row-major, per-column bias,
//! `out = A·B + bias`), so the two are differentially comparable
//! (M2-03-T18/T19; NFR-QL-01, FP32 `atol = 0.01`). Phase 4 (M2-03 T10-T14) adds
//! [`CudaContext::gemv_f32`], [`CudaContext::softmax_f32`],
//! [`CudaContext::layer_norm_f32`], [`CudaContext::gelu_f32`] and
//! [`CudaContext::conv1d_f32`] — each the CUDA-C port of the matching Metal
//! kernel and the same CPU contract — so the CUDA backend now covers the whole
//! Whisper hot-op set and a full Whisper forward runs on the GPU through the
//! imperative `Compute::Cuda` seam. FlashAttention-v2 (FR-BE-03; FA v3 is pushed
//! to v1.5+ and must not be implemented) remains a later ticket.
//!
//! # Precision (FP32, red line)
//!
//! The kernel is authored in explicit `float` (FP32). It uses no cuBLAS Tensor
//! Core / TF32 fast path, so there is no implicit precision reduction (FP16 /
//! quantised tiers are M2-08).
//!
//! # Kernel build (NVRTC → PTX, device-side JIT — not CPU codegen)
//!
//! The CUDA C GEMM is compiled to PTX at runtime with NVRTC and loaded via
//! `cuModuleLoadData`. This is **GPU** just-in-time compilation performed by the
//! NVIDIA toolchain; the host emits no executable CPU pages (NFR-RL-05). The
//! `cuBLAS`-based GEMM path (FR-BE-03) is a follow-on ticket; this slice proves
//! the end-to-end driver path (module load → alloc → H2D → launch → D2H → free)
//! with a self-contained kernel and no extra NVIDIA runtime dependency.
//!
//! # No bundling (NVIDIA EULA, FR-BE-08)
//!
//! Every NVIDIA entry point is resolved at runtime via dlopen ([`crate::sys`]);
//! nothing is linked or shipped. On a host with no driver (e.g. an Apple Mac)
//! [`CudaContext::new`] returns an explicit [`VokraError::BackendUnavailable`]
//! (never a silent CPU fall back — NFR-RL-06).

use core::cell::Cell;
use core::ffi::{c_char, c_uint, c_void};
use core::marker::PhantomData;

use vokra_core::{PrenormLayer, Result, VokraError};

use crate::sys::{self, CUcontext, CUdeviceptr, CUfunction, CUmodule, CUstream, CudaDriver, Nvrtc};

/// The GEMM kernel, compiled once per [`CudaContext`]. Row-major, FP32:
/// `C[r, c] = (has_bias ? bias[c] : 0) + Σ_k A[r, k] · B[k, c]` — identical
/// semantics to `vokra_backend_cpu::kernels::gemm_f32`. `extern "C"` suppresses
/// C++ name mangling so `cuModuleGetFunction("vokra_gemm_f32")` resolves it.
const GEMM_CUDA: &str = r#"
extern "C" __global__ void vokra_gemm_f32(
    const float* A,
    const float* B,
    const float* bias,
    float* C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int has_bias)
{
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= M || col >= N) {
        return;
    }
    float acc = 0.0f;
    unsigned int arow = row * K;
    for (unsigned int k = 0; k < K; ++k) {
        acc += A[arow + k] * B[k * N + col];
    }
    if (has_bias != 0u) {
        acc += bias[col];
    }
    C[row * N + col] = acc;
}
"#;

/// The five Phase-4 kernels (M2-03 T10-T14), NVRTC-compiled once into one
/// module. Each mirrors the semantics — and, within the FP32 bound, the numerics
/// — of the matching `vokra_backend_cpu::kernels` function and the
/// `vokra-backend-metal` `KERNELS_MSL` port. All FP32 (explicit `float`, the
/// `*f` single-precision math intrinsics `fmaxf`/`expf`/`sqrtf`/`fabsf`, no
/// double promotion), no cuBLAS/cuDNN, so there is no implicit TF32/FP16 fast
/// path. `extern "C"` on each kernel suppresses C++ mangling so
/// `cuModuleGetFunction` resolves the names; the `vokra_erf` device helper stays
/// internal (inlined). One thread per output row (gemv / softmax / layer_norm)
/// or element (gelu), or per `(out_pos, out_channel)` pair (conv1d); the launch
/// guards the ragged tail against the grid bound, exactly like the GEMM kernel.
const KERNELS_CUDA: &str = r#"
// ---- gemv: out[i] = (has_bias ? bias[i] : 0) + Σ_l A[i*K + l] · x[l] --------
// Bias-first accumulation matches vokra_backend_cpu::kernels' scalar `gemv`.
extern "C" __global__ void vokra_gemv_f32(
    const float* A,
    const float* x,
    const float* bias,
    float* out,
    unsigned int M,
    unsigned int K,
    unsigned int has_bias)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M) {
        return;
    }
    float acc = (has_bias != 0u) ? bias[i] : 0.0f;
    unsigned int arow = i * K;
    for (unsigned int l = 0; l < K; ++l) {
        acc += A[arow + l] * x[l];
    }
    out[i] = acc;
}

// ---- softmax: row-wise, max-subtracted (numerically stabilised) -------------
extern "C" __global__ void vokra_softmax_f32(
    const float* inp,
    float* out,
    unsigned int rows,
    unsigned int cols)
{
    unsigned int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) {
        return;
    }
    unsigned int base = r * cols;
    // Row max over every column (seeded with column 0). A causal-mask -INF entry
    // is never the max and becomes exp(-INF) = 0 below — as on the CPU/Metal.
    float m = inp[base];
    for (unsigned int j = 1; j < cols; ++j) {
        m = fmaxf(m, inp[base + j]);
    }
    float sum = 0.0f;
    for (unsigned int j = 0; j < cols; ++j) {
        float e = expf(inp[base + j] - m);
        out[base + j] = e;
        sum += e;
    }
    float inv = 1.0f / sum;
    for (unsigned int j = 0; j < cols; ++j) {
        out[base + j] *= inv;
    }
}

// ---- layer_norm: affine, biased (population) variance -----------------------
extern "C" __global__ void vokra_layer_norm_f32(
    const float* inp,
    const float* gamma,
    const float* beta,
    float* out,
    unsigned int rows,
    unsigned int cols,
    float eps)
{
    unsigned int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) {
        return;
    }
    unsigned int base = r * cols;
    float inv_cols = 1.0f / (float)cols;
    float mean = 0.0f;
    for (unsigned int c = 0; c < cols; ++c) {
        mean += inp[base + c];
    }
    mean *= inv_cols;
    float var = 0.0f;
    for (unsigned int c = 0; c < cols; ++c) {
        float dv = inp[base + c] - mean;
        var += dv * dv;
    }
    var *= inv_cols;
    float inv_std = 1.0f / sqrtf(var + eps);
    for (unsigned int c = 0; c < cols; ++c) {
        out[base + c] = (inp[base + c] - mean) * inv_std * gamma[c] + beta[c];
    }
}

// ---- gelu: exact (erf) form, out = 0.5·x·(1 + erf(x/√2)) ---------------------
// We do NOT use CUDA's builtin erff(): to stay bit-comparable with the CPU and
// Metal paths we inline the *identical* Abramowitz & Stegun 7.1.26 approximation
// (same constants, same Horner order). The only CPU⇔GPU numeric difference in
// gelu is then the vendor expf() (a few ULP) — far inside the FP32 bound.
__device__ float vokra_erf(float x) {
    float sign = (x < 0.0f) ? -1.0f : 1.0f;
    float ax = fabsf(x);
    float t = 1.0f / (1.0f + 0.3275911f * ax);
    float poly =
        ((((1.061405429f * t - 1.453152027f) * t + 1.421413741f) * t - 0.284496736f) * t
            + 0.254829592f) * t;
    float y = 1.0f - poly * expf(-ax * ax);
    return sign * y;
}

extern "C" __global__ void vokra_gelu_f32(
    const float* x,
    float* out,
    unsigned int n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    float v = x[i];
    out[i] = 0.5f * v * (1.0f + vokra_erf(v * 0.70710678118654752440f));
}

// ---- conv1d: direct convolution (im2col + GEMM equivalent) -------------------
// The (c outer, kk inner) accumulation order equals the im2col+GEMM reduction
// the CPU runs, so the two agree within the FP32 bound; bias is added after, as
// on CPU. Whisper's encoder stem (80→512 k3 s1 p1, then 512→512 k3 s2 p1) is the
// motivating shape set.
extern "C" __global__ void vokra_conv1d_f32(
    const float* inp,
    const float* weight,
    const float* bias,
    float* out,
    unsigned int in_ch,
    unsigned int in_len,
    unsigned int out_ch,
    unsigned int kernel_size,
    unsigned int out_len,
    unsigned int stride,
    unsigned int padding,
    unsigned int has_bias)
{
    unsigned int t  = blockIdx.x * blockDim.x + threadIdx.x; // output position
    unsigned int oc = blockIdx.y * blockDim.y + threadIdx.y; // output channel
    if (t >= out_len || oc >= out_ch) {
        return;
    }
    unsigned int k     = in_ch * kernel_size;
    unsigned int wbase = oc * k;
    float acc = 0.0f;
    for (unsigned int c = 0; c < in_ch; ++c) {
        unsigned int wc    = wbase + c * kernel_size;
        unsigned int ibase = c * in_len;
        for (unsigned int kk = 0; kk < kernel_size; ++kk) {
            unsigned int pos = t * stride + kk;
            if (pos >= padding && pos < padding + in_len) {
                acc += weight[wc + kk] * inp[ibase + (pos - padding)];
            }
        }
    }
    if (has_bias != 0u) {
        acc += bias[oc];
    }
    out[oc * out_len + t] = acc;
}

// ---- Phase-5 attention fusion: three pure-copy column movers -----------------
// CUDA-C ports of the Metal `col_gather` / `col_gather_t` / `col_scatter`
// kernels. Each replaces the host `copy_from_slice` / transpose / `*= scale` the
// per-op `whisper::nn::attention_from_kv_into` runs between GPU ops: a pure data
// move (+ one FP32 multiply in the gather), one thread per destination (gather /
// gather_t) or source (scatter) element, ragged-tail guarded like every kernel
// above — so the fused path stays bit-for-bit equal to the per-op path.

// col_gather: dst[i*hd + c] = src[i*width + c0 + c] * scale (folds the query
// scale; qh: scale = head_dim^-0.5, vh: scale = 1).
extern "C" __global__ void vokra_col_gather_f32(
    const float* src,
    float* dst,
    unsigned int rows,
    unsigned int hd,
    unsigned int width,
    unsigned int c0,
    float scale)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int n = rows * hd;
    if (gid >= n) {
        return;
    }
    unsigned int i = gid / hd;
    unsigned int c = gid % hd;
    dst[gid] = src[i * width + c0 + c] * scale;
}

// col_gather_t: dst[c*t_kv + j] = src[j*width + c0 + c] (gather one head's key
// block AND transpose it to [hd, t_kv], the scores GEMM's right operand).
extern "C" __global__ void vokra_col_gather_t_f32(
    const float* src,
    float* dst,
    unsigned int t_kv,
    unsigned int hd,
    unsigned int width,
    unsigned int c0)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int n = hd * t_kv;
    if (gid >= n) {
        return;
    }
    unsigned int c = gid / t_kv;
    unsigned int j = gid % t_kv;
    dst[gid] = src[j * width + c0 + c];
}

// col_scatter: dst[i*width + c0 + c] = src[i*hd + c] (scatter this head's
// [rows, hd] context back into its column block of [rows, width]). Because
// n_head*hd == width every column is written by exactly one head, so the target
// needs no zeroing (fully overwritten, as on the CPU).
extern "C" __global__ void vokra_col_scatter_f32(
    const float* src,
    float* dst,
    unsigned int rows,
    unsigned int hd,
    unsigned int width,
    unsigned int c0)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int n = rows * hd;
    if (gid >= n) {
        return;
    }
    unsigned int i = gid / hd;
    unsigned int c = gid % hd;
    dst[i * width + c0 + c] = src[gid];
}

// ---- Phase-5 follow-on: in-place residual add (dst[i] += src[i]) -------------
// CUDA-C port of the Metal `add_assign` kernel: the device kernel for the encoder
// block's `h += sub_block` residual, replacing the host `whisper::nn::add_assign`
// loop so `h` stays resident across a whole device-resident encoder. One thread
// per element, ragged-tail guarded — a single FP32 add of the same two operands
// the host loop adds, so it is bit-identical to `add_assign`.
extern "C" __global__ void vokra_add_assign_f32(
    float* dst,
    const float* src,
    unsigned int n)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) {
        return;
    }
    dst[gid] = dst[gid] + src[gid];
}
"#;

/// 16×16 thread block (matches the Metal GEMM launch); the kernel guards the
/// ragged tail against `M`/`N`. Also the 2-D conv1d block dim.
const BLOCK: u32 = 16;

/// 1-D thread block for the row/element kernels (gemv / softmax / layer_norm /
/// gelu), matching the Metal `grid_1d` threadgroup width (256).
const BLOCK_1D: u32 = 256;

/// A `+`-owned device allocation, freed exactly once on drop via the borrowed
/// driver. Borrowing (rather than storing a fn pointer) keeps `sys`' private
/// signature aliases encapsulated, and an early `?`-return mid-`run_gemm`
/// still frees every buffer already allocated.
struct DeviceBuf<'a> {
    driver: &'a CudaDriver,
    ptr: CUdeviceptr,
}

impl Drop for DeviceBuf<'_> {
    fn drop(&mut self) {
        if self.ptr == 0 {
            return;
        }
        // SAFETY: `ptr` is a live device allocation from `cuMemAlloc`, freed once.
        unsafe { (self.driver.cu_mem_free)(self.ptr) };
    }
}

/// A public, cross-call handle to a device-resident `[f32]` buffer — the
/// Phase-5-follow-on surface mirroring `vokra-backend-metal`'s `MetalDeviceTensor`
/// (produced by [`CudaContext::upload`] / [`alloc_dev`], read back by
/// [`download`], consumed by the `*_dev` ops).
///
/// - Owns its device allocation through the existing [`DeviceBuf`] RAII (freed
///   once on drop via the borrowed driver), so it adds no new `unsafe`.
/// - `len` is the f32 element count (buffer sizing / readback validation).
/// - `DeviceBuf<'ctx>` already borrows `&'ctx CudaDriver` (which lives inside the
///   context), so holding a tensor past the context's `Drop` is a **compile
///   error**. `CUdeviceptr` (a `u64`) and `&CudaDriver` (fn pointers) are
///   `Send + Sync`, so the `PhantomData<*const CudaContext>` forces the handle
///   `!Send`/`!Sync` — thread-affine like the context (`cuMemFree` needs the
///   creating context current), symmetric with Metal.
///
/// [`alloc_dev`]: CudaContext::alloc_dev
/// [`download`]: CudaContext::download
pub struct CudaDeviceTensor<'ctx> {
    buf: DeviceBuf<'ctx>,
    len: usize,
    _aff: PhantomData<*const CudaContext>,
}

impl CudaDeviceTensor<'_> {
    /// The number of `f32` elements this device buffer holds.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the device buffer is empty (holds zero elements).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// Scalar shape of one fused-MLP launch chain, shared by [`CudaContext::run_mlp`],
/// [`CudaContext::mlp_dev`] and [`CudaContext::encode_prenorm_stack`].
struct MlpChainDims {
    t: usize,
    d: usize,
    ffn: usize,
    has_fc1_bias: bool,
    has_fc2_bias: bool,
}

/// Device pointers (`CUdeviceptr`, copied by value into the launch params) for one
/// fused-MLP launch chain (`x` `[t,d]`, `fc1_w` `[d,ffn]`, `fc2_w` `[ffn,d]`,
/// biases — a dummy when absent, `h`/`a` `[t,ffn]` intermediates, `out` `[t,d]`).
struct MlpChainPtrs {
    x: CUdeviceptr,
    fc1_w: CUdeviceptr,
    fc1_bias: CUdeviceptr,
    fc2_w: CUdeviceptr,
    fc2_bias: CUdeviceptr,
    h: CUdeviceptr,
    a: CUdeviceptr,
    out: CUdeviceptr,
}

/// Scalar shape of one fused-attention launch chain, shared by
/// [`CudaContext::run_attn`], [`CudaContext::attn_dev`] and
/// [`CudaContext::encode_prenorm_stack`]. `scale = head_dim^-0.5` is folded into
/// the qh gather.
struct AttnChainDims {
    t_q: usize,
    t_kv: usize,
    d: usize,
    n_head: usize,
    scale: f32,
    has_q_bias: bool,
    has_out_bias: bool,
}

/// Device pointers for one fused-attention launch chain: inputs (`xq`, `q_w`,
/// `q_bias`, `k`, `v`, `out_w`, `out_bias`), device-resident scratch (`q`,
/// `context`, `qh`, `vh`, `kh_t`, `scores`, `probs`, `ctx_h`) and `out`.
struct AttnChainPtrs {
    xq: CUdeviceptr,
    q_w: CUdeviceptr,
    q_bias: CUdeviceptr,
    k: CUdeviceptr,
    v: CUdeviceptr,
    out_w: CUdeviceptr,
    out_bias: CUdeviceptr,
    q: CUdeviceptr,
    context: CUdeviceptr,
    qh: CUdeviceptr,
    vh: CUdeviceptr,
    kh_t: CUdeviceptr,
    scores: CUdeviceptr,
    probs: CUdeviceptr,
    ctx_h: CUdeviceptr,
    out: CUdeviceptr,
}

/// One pre-norm block's weights uploaded to the device (the on-GPU mirror of
/// [`vokra_core::PrenormLayer`]), held for the life of an
/// [`CudaContext::encode_prenorm_stack`] call. Absent biases (Whisper's `k`) stay
/// `None` and bind the shared dummy at launch time.
struct DevLayer<'c> {
    attn_ln_g: CudaDeviceTensor<'c>,
    attn_ln_b: CudaDeviceTensor<'c>,
    q_w: CudaDeviceTensor<'c>,
    q_bias: Option<CudaDeviceTensor<'c>>,
    k_w: CudaDeviceTensor<'c>,
    k_bias: Option<CudaDeviceTensor<'c>>,
    v_w: CudaDeviceTensor<'c>,
    v_bias: Option<CudaDeviceTensor<'c>>,
    out_w: CudaDeviceTensor<'c>,
    out_bias: Option<CudaDeviceTensor<'c>>,
    mlp_ln_g: CudaDeviceTensor<'c>,
    mlp_ln_b: CudaDeviceTensor<'c>,
    fc1_w: CudaDeviceTensor<'c>,
    fc1_bias: Option<CudaDeviceTensor<'c>>,
    fc2_w: CudaDeviceTensor<'c>,
    fc2_bias: Option<CudaDeviceTensor<'c>>,
}

/// A CUDA driver + device context + stream + compiled GEMM kernel.
///
/// Holds the owned driver context, stream and module, released in [`Drop`] in
/// reverse creation order. Not `Send`/`Sync`: the driver handles are used from
/// the thread that created them (sufficient for the parity harness; a
/// thread-affine / `Send` wrapper is a later concern, mirroring `MetalContext`).
pub struct CudaContext {
    driver: CudaDriver,
    context: CUcontext,
    stream: CUstream,
    /// Module holding the FP32 GEMM kernel (the proven M2-03 slice).
    gemm_module: CUmodule,
    /// Module holding the five Phase-4 kernels (gemv / softmax / layer_norm /
    /// gelu / conv1d).
    kernels_module: CUmodule,
    gemm: CUfunction,
    gemv: CUfunction,
    softmax: CUfunction,
    layer_norm: CUfunction,
    gelu: CUfunction,
    conv1d: CUfunction,
    col_gather: CUfunction,
    col_gather_t: CUfunction,
    col_scatter: CUfunction,
    add_assign: CUfunction,
    /// Count of stream synchronisations issued through this context — the
    /// env-independent readback/sync metric the Phase-5-follow-on encoder-residency
    /// slice proves against (the whole encoder in ONE synchronise vs the per-op
    /// path's `6·N + 1`). `Cell` because every op takes `&self` and the context is
    /// already thread-affine (`!Send`/`!Sync`).
    submissions: Cell<u64>,
}

impl CudaContext {
    /// Loads the driver, creates a context + stream on device 0, and
    /// NVRTC-compiles + loads the FP32 GEMM kernel.
    ///
    /// # Errors
    ///
    /// [`VokraError::BackendUnavailable`] if there is no NVIDIA driver/GPU (e.g.
    /// on an Apple Mac — dlopen finds no `libcuda`), if a driver call fails, or
    /// if NVRTC is absent / the kernel fails to compile (the NVRTC log is
    /// included). Never a silent CPU fall back (NFR-RL-06).
    pub fn new() -> Result<CudaContext> {
        let driver = CudaDriver::load()?;

        // SAFETY: `cuInit(0)` must precede any other driver call; flag 0 is the
        // only defined value.
        let r = unsafe { (driver.cu_init)(0) };
        sys::check(&driver, r, "cuInit")?;

        let mut count = 0;
        // SAFETY: writes the device count into `count`.
        let r = unsafe { (driver.cu_device_get_count)(&mut count) };
        sys::check(&driver, r, "cuDeviceGetCount")?;
        if count <= 0 {
            return Err(VokraError::BackendUnavailable(
                "CUDA driver present but no CUDA-capable GPU (device count 0)".to_owned(),
            ));
        }

        let mut dev: sys::CUdevice = 0;
        // SAFETY: writes the ordinal-0 device handle into `dev`.
        let r = unsafe { (driver.cu_device_get)(&mut dev, 0) };
        sys::check(&driver, r, "cuDeviceGet")?;

        let mut context: CUcontext = core::ptr::null_mut();
        // SAFETY: creates a context (flags 0) on the valid device `dev`, writing
        // the owned handle into `context`.
        let r = unsafe { (driver.cu_ctx_create)(&mut context, 0, dev) };
        sys::check(&driver, r, "cuCtxCreate")?;

        // From here a failure must destroy `context`; `build_pipeline` cleans up
        // its own stream/modules on partial failure.
        match build_pipeline(&driver) {
            Ok((stream, m)) => Ok(CudaContext {
                driver,
                context,
                stream,
                gemm_module: m.gemm_module,
                kernels_module: m.kernels_module,
                gemm: m.gemm,
                gemv: m.gemv,
                softmax: m.softmax,
                layer_norm: m.layer_norm,
                gelu: m.gelu,
                conv1d: m.conv1d,
                col_gather: m.col_gather,
                col_gather_t: m.col_gather_t,
                col_scatter: m.col_scatter,
                add_assign: m.add_assign,
                submissions: Cell::new(0),
            }),
            Err(e) => {
                // SAFETY: `context` is the just-created owned context; destroy it
                // before propagating the error (no leak).
                unsafe { (driver.cu_ctx_destroy)(context) };
                Err(e)
            }
        }
    }

    /// Row-major FP32 GEMM on the GPU with optional per-column bias:
    /// `out[i, j] = bias[j] + Σ_l a[i, l] · b[l, j]`.
    ///
    /// `a` is `m×k`, `b` is `k×n`, `out` is `m×n`, and `bias` (when `Some`) has
    /// length `n` — the exact contract of
    /// `vokra_backend_cpu::kernels::gemm_f32`, so the two are differentially
    /// comparable (M2-03-T19).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on any shape mismatch or a zero
    /// dimension; [`VokraError::BackendUnavailable`] if a device allocation /
    /// copy / launch fails.
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
        self.run_gemm(m, n, k, a, b, bias, out)
    }

    /// GEMM body: allocate device buffers, H2D, launch, synchronise, D2H, free.
    /// Shapes are already validated.
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
        let d = &self.driver;

        // Device inputs (copied H2D). A failed alloc `?`-returns; already-built
        // DeviceBufs free on drop.
        let a_buf = self.alloc(size_of_val(a))?;
        self.htod(&a_buf, a)?;
        let b_buf = self.alloc(size_of_val(b))?;
        self.htod(&b_buf, b)?;

        // Bias: the real bias when present, else a 1-float dummy the kernel never
        // reads (has_bias = 0). Always allocated so the kernel arg is bound.
        let dummy = [0.0f32];
        let bias_slice = bias.unwrap_or(&dummy);
        let bias_buf = self.alloc(size_of_val(bias_slice))?;
        self.htod(&bias_buf, bias_slice)?;

        // Output (uninitialised device storage of m*n floats).
        let c_buf = self.alloc(size_of_val(out))?;

        // Scalar kernel args (must outlive the launch call, which captures them).
        let m_u = m as c_uint;
        let n_u = n as c_uint;
        let k_u = k as c_uint;
        let has_bias: c_uint = u32::from(bias.is_some());

        // `kernelParams`: one pointer per argument, in the kernel's declared
        // order (A, B, bias, C, M, N, K, has_bias). Each points to the value:
        // for the pointer args that is the CUdeviceptr; for the scalars the u32.
        let mut params: [*mut c_void; 8] = [
            (&a_buf.ptr as *const CUdeviceptr)
                .cast::<c_void>()
                .cast_mut(),
            (&b_buf.ptr as *const CUdeviceptr)
                .cast::<c_void>()
                .cast_mut(),
            (&bias_buf.ptr as *const CUdeviceptr)
                .cast::<c_void>()
                .cast_mut(),
            (&c_buf.ptr as *const CUdeviceptr)
                .cast::<c_void>()
                .cast_mut(),
            (&m_u as *const c_uint).cast::<c_void>().cast_mut(),
            (&n_u as *const c_uint).cast::<c_void>().cast_mut(),
            (&k_u as *const c_uint).cast::<c_void>().cast_mut(),
            (&has_bias as *const c_uint).cast::<c_void>().cast_mut(),
        ];

        // Grid: x = columns (N), y = rows (M), measured in blocks; the kernel
        // guards row/col against M/N for the ragged edges.
        let grid_x = n.div_ceil(BLOCK as usize) as c_uint;
        let grid_y = m.div_ceil(BLOCK as usize) as c_uint;

        // SAFETY: `self.gemm` is the loaded `vokra_gemm_f32` function; the launch
        // dims are non-zero (validated m,n,k >= 1); `self.stream` is the owned
        // stream; `params` holds one valid pointer per kernel argument, matching
        // the kernel's signature and alive across this synchronous launch; no
        // dynamic shared memory (0) and no `extra` (null).
        let launch = unsafe {
            (d.cu_launch_kernel)(
                self.gemm,
                grid_x,
                grid_y,
                1,
                BLOCK,
                BLOCK,
                1,
                0,
                self.stream,
                params.as_mut_ptr(),
                core::ptr::null_mut(),
            )
        };
        sys::check(d, launch, "cuLaunchKernel(vokra_gemm_f32)")?;

        self.sync_stream("cuStreamSynchronize")?;
        self.dtoh(&c_buf, out)
    }

    // ---- Phase-4 kernels (M2-03 T10-T14): gemv / softmax / layer_norm / gelu /
    // conv1d. Each mirrors the `vokra_backend_cpu::kernels` contract and numerics
    // (FP32, `atol = 0.01`) and the Metal port: validate → H2D → launch (one
    // pointer per kernel arg, in declared order) → synchronise → D2H, freeing
    // every device buffer on drop. An empty output is a no-op (a zero-dim launch
    // would be a driver error), matching the Metal backend.

    /// Row-major FP32 matrix-vector product with optional per-row bias:
    /// `out[i] = bias[i] + Σ_l a[i, l] · x[l]`. `a` is `m×k`, `x` length `k`,
    /// `out` length `m`, `bias` (when `Some`) length `m` — the exact contract of
    /// `vokra_backend_cpu::kernels::gemv_f32`.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on a shape mismatch;
    /// [`VokraError::BackendUnavailable`] on a device allocation / launch failure.
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
        self.run_gemv(m, k, a, x, bias, out)
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
        let a_buf = self.alloc(size_of_val(a))?;
        self.htod(&a_buf, a)?;
        let x_buf = self.alloc(size_of_val(x))?;
        self.htod(&x_buf, x)?;
        let dummy = [0.0f32];
        let bias_slice = bias.unwrap_or(&dummy);
        let bias_buf = self.alloc(size_of_val(bias_slice))?;
        self.htod(&bias_buf, bias_slice)?;
        let out_buf = self.alloc(size_of_val(out))?;

        // Scalars outlive the launch (their addresses go into `params`).
        let m_u = m as c_uint;
        let k_u = k as c_uint;
        let has_bias: c_uint = u32::from(bias.is_some());
        let mut params: [*mut c_void; 7] = [
            ptr_arg(&a_buf.ptr),
            ptr_arg(&x_buf.ptr),
            ptr_arg(&bias_buf.ptr),
            ptr_arg(&out_buf.ptr),
            uint_arg(&m_u),
            uint_arg(&k_u),
            uint_arg(&has_bias),
        ];
        let grid_x = m.div_ceil(BLOCK_1D as usize) as c_uint;
        self.launch(
            self.gemv,
            (grid_x, 1, 1),
            (BLOCK_1D, 1, 1),
            &mut params,
            "cuLaunchKernel(vokra_gemv_f32)",
        )?;
        self.dtoh(&out_buf, out)
    }

    /// Row-wise softmax over the innermost axis of a `rows × cols` buffer,
    /// max-subtracted — the exact contract of
    /// `vokra_backend_cpu::kernels::softmax_f32` (a causal-mask `-inf` score maps
    /// to a 0 weight, as on the CPU).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on a shape mismatch;
    /// [`VokraError::BackendUnavailable`] on a device failure.
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
        self.run_softmax(input, out, rows, cols)
    }

    fn run_softmax(&self, input: &[f32], out: &mut [f32], rows: usize, cols: usize) -> Result<()> {
        let in_buf = self.alloc(size_of_val(input))?;
        self.htod(&in_buf, input)?;
        let out_buf = self.alloc(size_of_val(out))?;

        let rows_u = rows as c_uint;
        let cols_u = cols as c_uint;
        let mut params: [*mut c_void; 4] = [
            ptr_arg(&in_buf.ptr),
            ptr_arg(&out_buf.ptr),
            uint_arg(&rows_u),
            uint_arg(&cols_u),
        ];
        let grid_x = rows.div_ceil(BLOCK_1D as usize) as c_uint;
        self.launch(
            self.softmax,
            (grid_x, 1, 1),
            (BLOCK_1D, 1, 1),
            &mut params,
            "cuLaunchKernel(vokra_softmax_f32)",
        )?;
        self.dtoh(&out_buf, out)
    }

    /// Affine layer normalisation over the innermost axis of a `rows × cols`
    /// buffer, biased (population) variance — the exact contract of
    /// `vokra_backend_cpu::kernels::layer_norm_f32` (`gamma` / `beta` length
    /// `cols`).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on a shape mismatch;
    /// [`VokraError::BackendUnavailable`] on a device failure.
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
        self.run_layer_norm(input, out, rows, cols, gamma, beta, eps)
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
        let in_buf = self.alloc(size_of_val(input))?;
        self.htod(&in_buf, input)?;
        let gamma_buf = self.alloc(size_of_val(gamma))?;
        self.htod(&gamma_buf, gamma)?;
        let beta_buf = self.alloc(size_of_val(beta))?;
        self.htod(&beta_buf, beta)?;
        let out_buf = self.alloc(size_of_val(out))?;

        let rows_u = rows as c_uint;
        let cols_u = cols as c_uint;
        let eps_v = eps;
        let mut params: [*mut c_void; 7] = [
            ptr_arg(&in_buf.ptr),
            ptr_arg(&gamma_buf.ptr),
            ptr_arg(&beta_buf.ptr),
            ptr_arg(&out_buf.ptr),
            uint_arg(&rows_u),
            uint_arg(&cols_u),
            f32_arg(&eps_v),
        ];
        let grid_x = rows.div_ceil(BLOCK_1D as usize) as c_uint;
        self.launch(
            self.layer_norm,
            (grid_x, 1, 1),
            (BLOCK_1D, 1, 1),
            &mut params,
            "cuLaunchKernel(vokra_layer_norm_f32)",
        )?;
        self.dtoh(&out_buf, out)
    }

    /// Element-wise exact (erf) GELU (`x` and `out` equal length) — the contract
    /// of `vokra_backend_cpu::kernels::gelu_f32`. Uses the *same* inlined A&S
    /// 7.1.26 erf approximation (not CUDA's `erff`), so it agrees with the CPU
    /// far inside the FP32 bound.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on a length mismatch;
    /// [`VokraError::BackendUnavailable`] on a device failure.
    pub fn gelu_f32(&self, x: &[f32], out: &mut [f32]) -> Result<()> {
        validate_unary(x, out)?;
        if out.is_empty() {
            return Ok(());
        }
        self.run_gelu(x, out)
    }

    fn run_gelu(&self, x: &[f32], out: &mut [f32]) -> Result<()> {
        let x_buf = self.alloc(size_of_val(x))?;
        self.htod(&x_buf, x)?;
        let out_buf = self.alloc(size_of_val(out))?;

        let n_u = out.len() as c_uint;
        let mut params: [*mut c_void; 3] =
            [ptr_arg(&x_buf.ptr), ptr_arg(&out_buf.ptr), uint_arg(&n_u)];
        let grid_x = out.len().div_ceil(BLOCK_1D as usize) as c_uint;
        self.launch(
            self.gelu,
            (grid_x, 1, 1),
            (BLOCK_1D, 1, 1),
            &mut params,
            "cuLaunchKernel(vokra_gelu_f32)",
        )?;
        self.dtoh(&out_buf, out)
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
    /// [`VokraError::BackendUnavailable`] on a device failure.
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
        self.run_conv1d(
            input, in_ch, in_len, weight, out_ch, kernel, bias, stride, padding, out_len, out,
        )
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
        let in_buf = self.alloc(size_of_val(input))?;
        self.htod(&in_buf, input)?;
        let w_buf = self.alloc(size_of_val(weight))?;
        self.htod(&w_buf, weight)?;
        let dummy = [0.0f32];
        let bias_slice = bias.unwrap_or(&dummy);
        let bias_buf = self.alloc(size_of_val(bias_slice))?;
        self.htod(&bias_buf, bias_slice)?;
        let out_buf = self.alloc(size_of_val(out))?;

        let in_ch_u = in_ch as c_uint;
        let in_len_u = in_len as c_uint;
        let out_ch_u = out_ch as c_uint;
        let kernel_u = kernel as c_uint;
        let out_len_u = out_len as c_uint;
        let stride_u = stride as c_uint;
        let padding_u = padding as c_uint;
        let has_bias: c_uint = u32::from(bias.is_some());
        let mut params: [*mut c_void; 12] = [
            ptr_arg(&in_buf.ptr),
            ptr_arg(&w_buf.ptr),
            ptr_arg(&bias_buf.ptr),
            ptr_arg(&out_buf.ptr),
            uint_arg(&in_ch_u),
            uint_arg(&in_len_u),
            uint_arg(&out_ch_u),
            uint_arg(&kernel_u),
            uint_arg(&out_len_u),
            uint_arg(&stride_u),
            uint_arg(&padding_u),
            uint_arg(&has_bias),
        ];
        // Grid: x = output positions, y = output channels, in blocks (matches the
        // Metal `grid_2d` launch); the kernel guards the ragged edges.
        let grid_x = out_len.div_ceil(BLOCK as usize) as c_uint;
        let grid_y = out_ch.div_ceil(BLOCK as usize) as c_uint;
        self.launch(
            self.conv1d,
            (grid_x, grid_y, 1),
            (BLOCK, BLOCK, 1),
            &mut params,
            "cuLaunchKernel(vokra_conv1d_f32)",
        )?;
        self.dtoh(&out_buf, out)
    }

    // ---- Phase-5 fusion: device-resident MLP (readback elimination) ----------

    /// Fused MLP `fc2(gelu(fc1(x)))` on the GPU with the two `[t, ffn]`
    /// intermediates **resident on the device** — the Phase-5 readback-
    /// elimination slice, mirroring [`vokra_backend_metal`]'s `mlp_f32`.
    ///
    /// `x` is `[t, d]`; `fc1` maps `d → ffn` (`fc1_w` is `[d, ffn]`, optional
    /// bias `[ffn]`); `fc2` maps `ffn → d` (`fc2_w` is `[ffn, d]`, optional bias
    /// `[d]`); `out` is `[t, d]`. It runs the very same three kernels
    /// (`vokra_gemm_f32` → `vokra_gelu_f32` → `vokra_gemm_f32`) the per-op
    /// [`Self::gemm_f32`] / [`Self::gelu_f32`] path runs, in the same order and
    /// launch geometry, so the result is **bit-identical** to three separate
    /// calls — but the `[t, ffn]` intermediates are never copied D2H, the three
    /// launches share one stream, and there is ONE `cuStreamSynchronize` and ONE
    /// D2H (of `out`) instead of three of each.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on any shape mismatch or a zero dimension;
    /// [`VokraError::BackendUnavailable`] on a device allocation / launch failure.
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
        self.run_mlp(t, d, ffn, x, fc1_w, fc1_bias, fc2_w, fc2_bias, out)
    }

    /// Fused-MLP body: H2D the five inputs, allocate the two `[t, ffn]`
    /// intermediates **device-resident** (never D2H'd) plus the `[t, d]` output,
    /// launch the three kernels back to back on the one stream, synchronise ONCE,
    /// and D2H only `out`. Shapes are already validated.
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
        // Inputs H2D (a failed alloc `?`-returns; already-built DeviceBufs free
        // on drop).
        let x_buf = self.alloc(size_of_val(x))?;
        self.htod(&x_buf, x)?;
        let fc1_w_buf = self.alloc(size_of_val(fc1_w))?;
        self.htod(&fc1_w_buf, fc1_w)?;
        let dummy = [0.0f32];
        let fc1_bias_slice = fc1_bias.unwrap_or(&dummy);
        let fc1_bias_buf = self.alloc(size_of_val(fc1_bias_slice))?;
        self.htod(&fc1_bias_buf, fc1_bias_slice)?;
        let fc2_w_buf = self.alloc(size_of_val(fc2_w))?;
        self.htod(&fc2_w_buf, fc2_w)?;
        let fc2_bias_slice = fc2_bias.unwrap_or(&dummy);
        let fc2_bias_buf = self.alloc(size_of_val(fc2_bias_slice))?;
        self.htod(&fc2_bias_buf, fc2_bias_slice)?;

        // The two `[t, ffn]` intermediates live only on the GPU: allocated device
        // storage the kernels write and read but that is NEVER copied D2H (the
        // readback this slice eliminates). `out` is the single buffer copied back.
        let inter = checked_mul(t, ffn, "mlp t*ffn")?;
        let inter_bytes = checked_mul(inter, size_of::<f32>(), "mlp t*ffn bytes")?;
        let h_buf = self.alloc(inter_bytes)?; // fc1 output [t, ffn]
        let a_buf = self.alloc(inter_bytes)?; // gelu output [t, ffn]
        let out_buf = self.alloc(size_of_val(out))?; // [t, d]

        // Three launches on the one stream (shared with `mlp_dev` /
        // `encode_prenorm_stack` so the numerics are single-sourced), no
        // intermediate synchronise, then ONE synchronise + D2H of the output.
        self.launch_mlp_chain(
            &MlpChainDims {
                t,
                d,
                ffn,
                has_fc1_bias: fc1_bias.is_some(),
                has_fc2_bias: fc2_bias.is_some(),
            },
            &MlpChainPtrs {
                x: x_buf.ptr,
                fc1_w: fc1_w_buf.ptr,
                fc1_bias: fc1_bias_buf.ptr,
                fc2_w: fc2_w_buf.ptr,
                fc2_bias: fc2_bias_buf.ptr,
                h: h_buf.ptr,
                a: a_buf.ptr,
                out: out_buf.ptr,
            },
        )?;
        self.sync_stream("cuStreamSynchronize")?;
        self.dtoh(&out_buf, out)
    }

    /// Issues the three fused-MLP launches (`fc1` GEMM → GELU → `fc2` GEMM) on the
    /// one stream, **without** synchronising / allocating / copying. Factored out
    /// of [`Self::run_mlp`] so the host-in/out [`Self::mlp_f32`], the device-in/out
    /// [`Self::mlp_dev`] and the whole-encoder [`Self::encode_prenorm_stack`] issue
    /// byte-for-byte identical launches. Every device pointer / scalar is passed by
    /// value or captured by address in a local `params` array read during the
    /// launch; the caller keeps the device buffers alive and synchronises once.
    fn launch_mlp_chain(&self, dims: &MlpChainDims, ptrs: &MlpChainPtrs) -> Result<()> {
        let (t, d, ffn) = (dims.t, dims.d, dims.ffn);
        // `t*ffn` fits: the caller allocated the `[t, ffn]` buffers.
        let inter = t * ffn;
        let t_u = t as c_uint;
        let ffn_u = ffn as c_uint;
        let d_u = d as c_uint;
        let inter_u = inter as c_uint;
        let has_bias1: c_uint = u32::from(dims.has_fc1_bias);
        let has_bias2: c_uint = u32::from(dims.has_fc2_bias);

        // GEMM arg order: (A, B, bias, C, M, N, K, has_bias).
        // fc1: h = x[t,d] · fc1_w[d,ffn] (+bias) — M=t, N=ffn, K=d.
        let mut p_fc1: [*mut c_void; 8] = [
            ptr_arg(&ptrs.x),
            ptr_arg(&ptrs.fc1_w),
            ptr_arg(&ptrs.fc1_bias),
            ptr_arg(&ptrs.h),
            uint_arg(&t_u),
            uint_arg(&ffn_u),
            uint_arg(&d_u),
            uint_arg(&has_bias1),
        ];
        // gelu: a = gelu(h) — n = t*ffn.
        let mut p_gelu: [*mut c_void; 3] = [ptr_arg(&ptrs.h), ptr_arg(&ptrs.a), uint_arg(&inter_u)];
        // fc2: out = a[t,ffn] · fc2_w[ffn,d] (+bias) — M=t, N=d, K=ffn.
        let mut p_fc2: [*mut c_void; 8] = [
            ptr_arg(&ptrs.a),
            ptr_arg(&ptrs.fc2_w),
            ptr_arg(&ptrs.fc2_bias),
            ptr_arg(&ptrs.out),
            uint_arg(&t_u),
            uint_arg(&d_u),
            uint_arg(&ffn_u),
            uint_arg(&has_bias2),
        ];

        // Launch geometries identical to the per-op path (GEMM 16×16 grid = N×M;
        // gelu 1-D grid over t*ffn).
        let fc1_grid = (
            ffn.div_ceil(BLOCK as usize) as c_uint,
            t.div_ceil(BLOCK as usize) as c_uint,
            1,
        );
        let gelu_grid = (inter.div_ceil(BLOCK_1D as usize) as c_uint, 1, 1);
        let fc2_grid = (
            d.div_ceil(BLOCK as usize) as c_uint,
            t.div_ceil(BLOCK as usize) as c_uint,
            1,
        );

        self.launch_async(
            self.gemm,
            fc1_grid,
            (BLOCK, BLOCK, 1),
            &mut p_fc1,
            "cuLaunchKernel(vokra_gemm_f32 mlp fc1)",
        )?;
        self.launch_async(
            self.gelu,
            gelu_grid,
            (BLOCK_1D, 1, 1),
            &mut p_gelu,
            "cuLaunchKernel(vokra_gelu_f32 mlp gelu)",
        )?;
        self.launch_async(
            self.gemm,
            fc2_grid,
            (BLOCK, BLOCK, 1),
            &mut p_fc2,
            "cuLaunchKernel(vokra_gemm_f32 mlp fc2)",
        )
    }

    // ---- Phase-5 fusion: device-resident non-causal attention ----------------

    /// Fused **non-causal** multi-head attention on the GPU with every
    /// intermediate **resident on the device** — the Phase-5 attention
    /// readback-elimination slice, mirroring [`vokra_backend_metal`]'s
    /// `attn_f32` (the sibling of [`Self::mlp_f32`]).
    ///
    /// Computes `out = out_proj( concat_h softmax(scale · qₕ·kₕᵀ) · vₕ )` for
    /// `xq` `[t_q, d]`, pre-projected `k` / `v` `[t_kv, d]`, `q_w` / `out_w`
    /// `[d, d]` (both projections `d → d`), optional biases `[d]`, and
    /// `scale = head_dim^-0.5` (the caller folds the query scale in). `out` is
    /// `[t_q, d]`.
    ///
    /// It runs the **same** `vokra_gemm_f32` (q-proj, per-head scores, per-head
    /// context, out-proj) and `vokra_softmax_f32` kernels the per-op
    /// `whisper::nn::attention_from_kv_into` runs, in the same order and launch
    /// geometry, with the head gather / transpose / scatter (formerly host
    /// `copy_from_slice`) done by the three pure-copy `col_*` kernels — so the
    /// result is **bit-identical** to the per-op path. The per-head scratch
    /// (`qh` / `vh` / `kh_t` / `scores` / `probs` / `ctx_h`) and the `q` /
    /// `context` intermediates never leave the device: all `2 + 7·n_head`
    /// launches share ONE stream with ONE `cuStreamSynchronize` and ONE D2H (of
    /// `out`) instead of the per-op path's per-op H2D/D2H round-trips. Stream
    /// ordering serialises the reused per-head scratch (head h+1's gather into
    /// `qh` after head h's scores GEMM read of `qh`), the same guarantee
    /// [`Self::mlp_f32`] relies on.
    ///
    /// **Non-causal only** (encoder self-attention and decoder cross-attention).
    /// Causal decoder self-attention stays on the per-op path (it needs the mask
    /// write between the scores GEMM and the softmax).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on any shape mismatch, a zero dimension, or
    /// `d % n_head != 0`; [`VokraError::BackendUnavailable`] on a device
    /// allocation / launch failure.
    #[allow(clippy::too_many_arguments)] // fused-attention operand set (two Linears + K/V + dims)
    pub fn attn_f32(
        &self,
        t_q: usize,
        t_kv: usize,
        d: usize,
        n_head: usize,
        xq: &[f32],
        q_w: &[f32],
        q_bias: Option<&[f32]>,
        k: &[f32],
        v: &[f32],
        out_w: &[f32],
        out_bias: Option<&[f32]>,
        scale: f32,
        out: &mut [f32],
    ) -> Result<()> {
        validate_attn(
            t_q, t_kv, d, n_head, xq, q_w, q_bias, k, v, out_w, out_bias, out,
        )?;
        self.run_attn(
            t_q, t_kv, d, n_head, xq, q_w, q_bias, k, v, out_w, out_bias, scale, out,
        )
    }

    /// Fused-attention body: H2D the inputs, allocate every intermediate
    /// **device-resident** (never D2H'd) plus the `[t_q, d]` output, launch the
    /// `2 + 7·n_head` kernels back to back on the one stream (q-proj GEMM → per
    /// head {gather qh, gather vh, gather-transpose kh_t, scores GEMM, softmax,
    /// context GEMM, scatter} → out-proj GEMM), synchronise ONCE, and D2H only
    /// `out`. Shapes are already validated (so `hd = d / n_head` is exact). Every
    /// scalar kernel arg is captured by address in a `params` array read by the
    /// driver during each synchronous `cuLaunchKernel`, so the per-head locals
    /// (e.g. `c0`) need not outlive the loop; the device buffers stay alive until
    /// the single synchronise below.
    #[allow(clippy::too_many_arguments)] // fused-attention operand set (two Linears + K/V + dims)
    fn run_attn(
        &self,
        t_q: usize,
        t_kv: usize,
        d: usize,
        n_head: usize,
        xq: &[f32],
        q_w: &[f32],
        q_bias: Option<&[f32]>,
        k: &[f32],
        v: &[f32],
        out_w: &[f32],
        out_bias: Option<&[f32]>,
        scale: f32,
        out: &mut [f32],
    ) -> Result<()> {
        let hd = d / n_head;

        // Inputs H2D (a failed alloc `?`-returns; already-built DeviceBufs free
        // on drop).
        let xq_buf = self.alloc(size_of_val(xq))?;
        self.htod(&xq_buf, xq)?;
        let q_w_buf = self.alloc(size_of_val(q_w))?;
        self.htod(&q_w_buf, q_w)?;
        let dummy = [0.0f32];
        let q_bias_slice = q_bias.unwrap_or(&dummy);
        let q_bias_buf = self.alloc(size_of_val(q_bias_slice))?;
        self.htod(&q_bias_buf, q_bias_slice)?;
        let k_buf = self.alloc(size_of_val(k))?;
        self.htod(&k_buf, k)?;
        let v_buf = self.alloc(size_of_val(v))?;
        self.htod(&v_buf, v)?;
        let out_w_buf = self.alloc(size_of_val(out_w))?;
        self.htod(&out_w_buf, out_w)?;
        let out_bias_slice = out_bias.unwrap_or(&dummy);
        let out_bias_buf = self.alloc(size_of_val(out_bias_slice))?;
        self.htod(&out_bias_buf, out_bias_slice)?;

        // Device-resident intermediates: `q` / `context` `[t_q, d]` and the reused
        // per-head scratch. None is ever D2H'd — that is the readback this slice
        // eliminates. `out` `[t_q, d]` is the single buffer copied back.
        let f = size_of::<f32>();
        let tqd = checked_mul(checked_mul(t_q, d, "attn t_q*d")?, f, "attn t_q*d bytes")?;
        let tq_hd_n = checked_mul(t_q, hd, "attn t_q*hd")?;
        let tkv_hd_n = checked_mul(t_kv, hd, "attn t_kv*hd")?;
        let hd_tkv_n = checked_mul(hd, t_kv, "attn hd*t_kv")?;
        let tq_tkv_n = checked_mul(t_q, t_kv, "attn t_q*t_kv")?;
        let q_buf = self.alloc(tqd)?; // q-proj [t_q, d]
        let context_buf = self.alloc(tqd)?; // per-head scatter target [t_q, d]
        let qh_buf = self.alloc(checked_mul(tq_hd_n, f, "attn qh bytes")?)?; // [t_q, hd]
        let vh_buf = self.alloc(checked_mul(tkv_hd_n, f, "attn vh bytes")?)?; // [t_kv, hd]
        let kh_t_buf = self.alloc(checked_mul(hd_tkv_n, f, "attn kh_t bytes")?)?; // [hd, t_kv]
        let scores_buf = self.alloc(checked_mul(tq_tkv_n, f, "attn scores bytes")?)?; // [t_q, t_kv]
        let probs_buf = self.alloc(checked_mul(tq_tkv_n, f, "attn probs bytes")?)?; // [t_q, t_kv]
        let ctx_h_buf = self.alloc(checked_mul(tq_hd_n, f, "attn ctx_h bytes")?)?; // [t_q, hd]
        let out_buf = self.alloc(size_of_val(out))?; // [t_q, d]

        // Issue the `2 + 7·n_head` launches on the one stream (shared with
        // `attn_dev` / `encode_prenorm_stack`), then ONE synchronise + D2H.
        self.launch_attn_chain(
            &AttnChainDims {
                t_q,
                t_kv,
                d,
                n_head,
                scale,
                has_q_bias: q_bias.is_some(),
                has_out_bias: out_bias.is_some(),
            },
            &AttnChainPtrs {
                xq: xq_buf.ptr,
                q_w: q_w_buf.ptr,
                q_bias: q_bias_buf.ptr,
                k: k_buf.ptr,
                v: v_buf.ptr,
                out_w: out_w_buf.ptr,
                out_bias: out_bias_buf.ptr,
                q: q_buf.ptr,
                context: context_buf.ptr,
                qh: qh_buf.ptr,
                vh: vh_buf.ptr,
                kh_t: kh_t_buf.ptr,
                scores: scores_buf.ptr,
                probs: probs_buf.ptr,
                ctx_h: ctx_h_buf.ptr,
                out: out_buf.ptr,
            },
        )?;
        self.sync_stream("cuStreamSynchronize")?;
        self.dtoh(&out_buf, out)
    }

    /// Issues the `2 + 7·n_head` fused-attention launches (q-proj GEMM → per head
    /// {gather qh/vh, gather-transpose kh_t, scores GEMM, softmax, context GEMM,
    /// scatter} → out-proj GEMM) on the one stream, **without** synchronising /
    /// allocating / copying. Factored out of [`Self::run_attn`] so the host-in/out
    /// [`Self::attn_f32`], the device-in/out [`Self::attn_dev`] and the
    /// whole-encoder [`Self::encode_prenorm_stack`] issue byte-for-byte identical
    /// launches. Stream ordering serialises the reused per-head scratch (head h+1's
    /// gather into `qh` after head h's scores GEMM read of it). `hd = d / n_head`
    /// is exact (the caller validated it). Bias-less GEMMs bind `ptrs.q_bias` as
    /// the never-read dummy (`has_bias = 0`).
    fn launch_attn_chain(&self, dims: &AttnChainDims, ptrs: &AttnChainPtrs) -> Result<()> {
        let (t_q, t_kv, d, n_head) = (dims.t_q, dims.t_kv, dims.d, dims.n_head);
        let hd = d / n_head;
        // These products all fit: the caller allocated buffers of these sizes.
        let tq_hd_n = t_q * hd;
        let tkv_hd_n = t_kv * hd;
        let hd_tkv_n = hd * t_kv;

        let t_q_u = t_q as c_uint;
        let t_kv_u = t_kv as c_uint;
        let d_u = d as c_uint;
        let hd_u = hd as c_uint;
        let zero_u: c_uint = 0; // has_bias / bias-less GEMMs
        let has_bias_q: c_uint = u32::from(dims.has_q_bias);
        let has_bias_out: c_uint = u32::from(dims.has_out_bias);
        let scale_v = dims.scale;
        let one_v = 1.0f32;

        let gemm_block = (BLOCK, BLOCK, 1);
        let gemm_grid = |n: usize, m: usize| {
            (
                n.div_ceil(BLOCK as usize) as c_uint,
                m.div_ceil(BLOCK as usize) as c_uint,
                1,
            )
        };
        let lin_block = (BLOCK_1D, 1, 1);
        let lin_grid = |elems: usize| (elems.div_ceil(BLOCK_1D as usize) as c_uint, 1, 1);

        // q = xq[t_q,d] · q_w[d,d] (+q_bias) — GEMM (M=t_q, N=d, K=d). The query
        // scale is NOT applied here; it is folded into the qh gather below (the
        // same single FP32 multiply the CPU does after this GEMM).
        let mut p_q: [*mut c_void; 8] = [
            ptr_arg(&ptrs.xq),
            ptr_arg(&ptrs.q_w),
            ptr_arg(&ptrs.q_bias),
            ptr_arg(&ptrs.q),
            uint_arg(&t_q_u),
            uint_arg(&d_u),
            uint_arg(&d_u),
            uint_arg(&has_bias_q),
        ];
        self.launch_async(
            self.gemm,
            gemm_grid(d, t_q),
            gemm_block,
            &mut p_q,
            "cuLaunchKernel(vokra_gemm_f32 attn q-proj)",
        )?;

        for h in 0..n_head {
            let c0_u = (h * hd) as c_uint;
            // qh[i,c] = q[i, c0+c] * scale.
            let mut p_qh: [*mut c_void; 7] = [
                ptr_arg(&ptrs.q),
                ptr_arg(&ptrs.qh),
                uint_arg(&t_q_u),
                uint_arg(&hd_u),
                uint_arg(&d_u),
                uint_arg(&c0_u),
                f32_arg(&scale_v),
            ];
            self.launch_async(
                self.col_gather,
                lin_grid(tq_hd_n),
                lin_block,
                &mut p_qh,
                "cuLaunchKernel(vokra_col_gather_f32 attn qh)",
            )?;
            // vh[j,c] = v[j, c0+c] (scale = 1).
            let mut p_vh: [*mut c_void; 7] = [
                ptr_arg(&ptrs.v),
                ptr_arg(&ptrs.vh),
                uint_arg(&t_kv_u),
                uint_arg(&hd_u),
                uint_arg(&d_u),
                uint_arg(&c0_u),
                f32_arg(&one_v),
            ];
            self.launch_async(
                self.col_gather,
                lin_grid(tkv_hd_n),
                lin_block,
                &mut p_vh,
                "cuLaunchKernel(vokra_col_gather_f32 attn vh)",
            )?;
            // kh_t[c,j] = k[j, c0+c] (gather + transpose to [hd, t_kv]).
            let mut p_kh: [*mut c_void; 6] = [
                ptr_arg(&ptrs.k),
                ptr_arg(&ptrs.kh_t),
                uint_arg(&t_kv_u),
                uint_arg(&hd_u),
                uint_arg(&d_u),
                uint_arg(&c0_u),
            ];
            self.launch_async(
                self.col_gather_t,
                lin_grid(hd_tkv_n),
                lin_block,
                &mut p_kh,
                "cuLaunchKernel(vokra_col_gather_t_f32 attn kh_t)",
            )?;
            // scores[t_q,t_kv] = qh[t_q,hd] · kh_t[hd,t_kv] (bias-less GEMM).
            let mut p_scores: [*mut c_void; 8] = [
                ptr_arg(&ptrs.qh),
                ptr_arg(&ptrs.kh_t),
                ptr_arg(&ptrs.q_bias),
                ptr_arg(&ptrs.scores),
                uint_arg(&t_q_u),
                uint_arg(&t_kv_u),
                uint_arg(&hd_u),
                uint_arg(&zero_u),
            ];
            self.launch_async(
                self.gemm,
                gemm_grid(t_kv, t_q),
                gemm_block,
                &mut p_scores,
                "cuLaunchKernel(vokra_gemm_f32 attn scores)",
            )?;
            // probs = softmax_rows(scores) (no mask — non-causal).
            let mut p_soft: [*mut c_void; 4] = [
                ptr_arg(&ptrs.scores),
                ptr_arg(&ptrs.probs),
                uint_arg(&t_q_u),
                uint_arg(&t_kv_u),
            ];
            self.launch_async(
                self.softmax,
                lin_grid(t_q),
                lin_block,
                &mut p_soft,
                "cuLaunchKernel(vokra_softmax_f32 attn)",
            )?;
            // ctx_h[t_q,hd] = probs[t_q,t_kv] · vh[t_kv,hd] (bias-less GEMM).
            let mut p_ctx: [*mut c_void; 8] = [
                ptr_arg(&ptrs.probs),
                ptr_arg(&ptrs.vh),
                ptr_arg(&ptrs.q_bias),
                ptr_arg(&ptrs.ctx_h),
                uint_arg(&t_q_u),
                uint_arg(&hd_u),
                uint_arg(&t_kv_u),
                uint_arg(&zero_u),
            ];
            self.launch_async(
                self.gemm,
                gemm_grid(hd, t_q),
                gemm_block,
                &mut p_ctx,
                "cuLaunchKernel(vokra_gemm_f32 attn context)",
            )?;
            // context[i, c0+c] = ctx_h[i,c].
            let mut p_scatter: [*mut c_void; 6] = [
                ptr_arg(&ptrs.ctx_h),
                ptr_arg(&ptrs.context),
                uint_arg(&t_q_u),
                uint_arg(&hd_u),
                uint_arg(&d_u),
                uint_arg(&c0_u),
            ];
            self.launch_async(
                self.col_scatter,
                lin_grid(tq_hd_n),
                lin_block,
                &mut p_scatter,
                "cuLaunchKernel(vokra_col_scatter_f32 attn)",
            )?;
        }

        // out = context[t_q,d] · out_w[d,d] (+out_bias) — GEMM (M=t_q, N=d, K=d).
        let mut p_out: [*mut c_void; 8] = [
            ptr_arg(&ptrs.context),
            ptr_arg(&ptrs.out_w),
            ptr_arg(&ptrs.out_bias),
            ptr_arg(&ptrs.out),
            uint_arg(&t_q_u),
            uint_arg(&d_u),
            uint_arg(&d_u),
            uint_arg(&has_bias_out),
        ];
        self.launch_async(
            self.gemm,
            gemm_grid(d, t_q),
            gemm_block,
            &mut p_out,
            "cuLaunchKernel(vokra_gemm_f32 attn out-proj)",
        )
    }

    // ---- Phase-5 follow-on: public device-resident handle + ops --------------

    /// The number of stream synchronisations issued through this context so far.
    /// The env-independent readback/sync metric: the whole-encoder
    /// [`Self::encode_prenorm_stack`] issues ONE, versus the per-op path's
    /// `6·N + 1` for an `N`-block encoder.
    #[must_use]
    pub fn submission_count(&self) -> u64 {
        self.submissions.get()
    }

    /// Uploads `data` into a fresh device-resident buffer (H2D once). The returned
    /// [`CudaDeviceTensor`] borrows the context, so it cannot outlive it.
    ///
    /// # Errors
    ///
    /// [`VokraError::BackendUnavailable`] on an allocation / copy failure.
    pub fn upload(&self, data: &[f32]) -> Result<CudaDeviceTensor<'_>> {
        let buf = self.alloc(size_of_val(data))?;
        self.htod(&buf, data)?;
        Ok(CudaDeviceTensor {
            buf,
            len: data.len(),
            _aff: PhantomData,
        })
    }

    /// Allocates an uninitialised device-resident buffer of `len` f32s (the
    /// residency slice's intermediates; never copied D2H until an explicit
    /// [`Self::download`]).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on an overflowing byte count;
    /// [`VokraError::BackendUnavailable`] on an allocation failure.
    pub fn alloc_dev(&self, len: usize) -> Result<CudaDeviceTensor<'_>> {
        let bytes = checked_mul(len, size_of::<f32>(), "alloc_dev bytes")?;
        let buf = self.alloc(bytes)?;
        Ok(CudaDeviceTensor {
            buf,
            len,
            _aff: PhantomData,
        })
    }

    /// Reads a device-resident buffer back into `out` (D2H). Call after the owning
    /// submission has completed (the `*_dev` ops and [`Self::encode_prenorm_stack`]
    /// synchronise before returning).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] if `out.len()` differs from the tensor's
    /// element count; [`VokraError::BackendUnavailable`] on a copy failure.
    pub fn download(&self, t: &CudaDeviceTensor<'_>, out: &mut [f32]) -> Result<()> {
        expect_len("download out", out.len(), t.len)?;
        self.dtoh(&t.buf, out)
    }

    /// Uploads an optional weight slice (a `None` bias stays `None`, bound as the
    /// shared dummy at launch time).
    fn upload_opt(&self, data: Option<&[f32]>) -> Result<Option<CudaDeviceTensor<'_>>> {
        data.map(|d| self.upload(d)).transpose()
    }

    /// Device-in/out affine layer normalisation (one self-contained submission).
    /// Bit-identical to the host-in/out [`Self::layer_norm_f32`]; `out` must be a
    /// distinct buffer from `x`.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on a shape mismatch;
    /// [`VokraError::BackendUnavailable`] on a device failure.
    #[allow(clippy::too_many_arguments)] // intrinsic layer-norm parameter set
    pub fn layer_norm_dev(
        &self,
        out: &mut CudaDeviceTensor<'_>,
        x: &CudaDeviceTensor<'_>,
        gamma: &CudaDeviceTensor<'_>,
        beta: &CudaDeviceTensor<'_>,
        rows: usize,
        cols: usize,
        eps: f32,
    ) -> Result<()> {
        let total = checked_mul(rows, cols, "layer_norm_dev rows*cols")?;
        expect_len("layer_norm_dev x", x.len, total)?;
        expect_len("layer_norm_dev out", out.len, total)?;
        expect_len("layer_norm_dev gamma", gamma.len, cols)?;
        expect_len("layer_norm_dev beta", beta.len, cols)?;
        if total == 0 {
            return Ok(());
        }
        self.launch_layer_norm_async(
            x.buf.ptr,
            gamma.buf.ptr,
            beta.buf.ptr,
            out.buf.ptr,
            rows,
            cols,
            eps,
        )?;
        self.sync_stream("cuStreamSynchronize")
    }

    /// Device-in/out in-place residual add `dst[i] += src[i]` (one self-contained
    /// submission). Bit-identical to the host `whisper::nn::add_assign`.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] if the lengths differ;
    /// [`VokraError::BackendUnavailable`] on a device failure.
    pub fn residual_add_dev(
        &self,
        dst: &mut CudaDeviceTensor<'_>,
        src: &CudaDeviceTensor<'_>,
    ) -> Result<()> {
        expect_len("residual_add_dev src", src.len, dst.len)?;
        if dst.len == 0 {
            return Ok(());
        }
        self.launch_residual_add_async(dst.buf.ptr, src.buf.ptr, dst.len)?;
        self.sync_stream("cuStreamSynchronize")
    }

    /// Device-in/out fused MLP `fc2(gelu(fc1(x)))` (one self-contained submission,
    /// the two `[t, ffn]` intermediates allocated internally and never copied D2H).
    /// Bit-identical to the host-in/out [`Self::mlp_f32`].
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on a shape mismatch;
    /// [`VokraError::BackendUnavailable`] on a device failure.
    #[allow(clippy::too_many_arguments)] // fused-MLP operand set (two Linears + dims)
    pub fn mlp_dev(
        &self,
        t: usize,
        d: usize,
        ffn: usize,
        x: &CudaDeviceTensor<'_>,
        fc1_w: &CudaDeviceTensor<'_>,
        fc1_bias: Option<&CudaDeviceTensor<'_>>,
        fc2_w: &CudaDeviceTensor<'_>,
        fc2_bias: Option<&CudaDeviceTensor<'_>>,
        out: &mut CudaDeviceTensor<'_>,
    ) -> Result<()> {
        if t == 0 || d == 0 || ffn == 0 {
            return Err(VokraError::InvalidArgument(
                "mlp_dev dimensions t, d, ffn must all be >= 1".to_owned(),
            ));
        }
        expect_len("mlp_dev x", x.len, checked_mul(t, d, "mlp_dev t*d")?)?;
        expect_len(
            "mlp_dev fc1_w",
            fc1_w.len,
            checked_mul(d, ffn, "mlp_dev d*ffn")?,
        )?;
        expect_len(
            "mlp_dev fc2_w",
            fc2_w.len,
            checked_mul(ffn, d, "mlp_dev ffn*d")?,
        )?;
        expect_len(
            "mlp_dev out",
            out.len,
            checked_mul(t, d, "mlp_dev out t*d")?,
        )?;
        if let Some(b) = fc1_bias {
            expect_len("mlp_dev fc1_bias", b.len, ffn)?;
        }
        if let Some(b) = fc2_bias {
            expect_len("mlp_dev fc2_bias", b.len, d)?;
        }
        let inter_bytes = checked_mul(
            checked_mul(t, ffn, "mlp_dev t*ffn")?,
            size_of::<f32>(),
            "mlp_dev t*ffn bytes",
        )?;
        let dummy = self.alloc(size_of::<f32>())?;
        let h_buf = self.alloc(inter_bytes)?;
        let a_buf = self.alloc(inter_bytes)?;
        self.launch_mlp_chain(
            &MlpChainDims {
                t,
                d,
                ffn,
                has_fc1_bias: fc1_bias.is_some(),
                has_fc2_bias: fc2_bias.is_some(),
            },
            &MlpChainPtrs {
                x: x.buf.ptr,
                fc1_w: fc1_w.buf.ptr,
                fc1_bias: bias_ptr(fc1_bias, dummy.ptr),
                fc2_w: fc2_w.buf.ptr,
                fc2_bias: bias_ptr(fc2_bias, dummy.ptr),
                h: h_buf.ptr,
                a: a_buf.ptr,
                out: out.buf.ptr,
            },
        )?;
        self.sync_stream("cuStreamSynchronize")
    }

    /// Device-in/out fused **non-causal** attention (one self-contained
    /// submission, every intermediate allocated internally and never copied D2H).
    /// Bit-identical to the host-in/out [`Self::attn_f32`].
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on a shape mismatch or `d % n_head != 0`;
    /// [`VokraError::BackendUnavailable`] on a device failure.
    #[allow(clippy::too_many_arguments)] // fused-attention operand set (two Linears + K/V + dims)
    pub fn attn_dev(
        &self,
        t_q: usize,
        t_kv: usize,
        d: usize,
        n_head: usize,
        xq: &CudaDeviceTensor<'_>,
        q_w: &CudaDeviceTensor<'_>,
        q_bias: Option<&CudaDeviceTensor<'_>>,
        k: &CudaDeviceTensor<'_>,
        v: &CudaDeviceTensor<'_>,
        out_w: &CudaDeviceTensor<'_>,
        out_bias: Option<&CudaDeviceTensor<'_>>,
        scale: f32,
        out: &mut CudaDeviceTensor<'_>,
    ) -> Result<()> {
        if t_q == 0 || t_kv == 0 || d == 0 || n_head == 0 {
            return Err(VokraError::InvalidArgument(
                "attn_dev dimensions t_q, t_kv, d, n_head must all be >= 1".to_owned(),
            ));
        }
        if d % n_head != 0 {
            return Err(VokraError::InvalidArgument(format!(
                "attn_dev d ({d}) must be divisible by n_head ({n_head})"
            )));
        }
        let dd = checked_mul(d, d, "attn_dev d*d")?;
        let tkvd = checked_mul(t_kv, d, "attn_dev t_kv*d")?;
        expect_len(
            "attn_dev xq",
            xq.len,
            checked_mul(t_q, d, "attn_dev t_q*d")?,
        )?;
        expect_len("attn_dev q_w", q_w.len, dd)?;
        expect_len("attn_dev k", k.len, tkvd)?;
        expect_len("attn_dev v", v.len, tkvd)?;
        expect_len("attn_dev out_w", out_w.len, dd)?;
        expect_len(
            "attn_dev out",
            out.len,
            checked_mul(t_q, d, "attn_dev out")?,
        )?;
        if let Some(b) = q_bias {
            expect_len("attn_dev q_bias", b.len, d)?;
        }
        if let Some(b) = out_bias {
            expect_len("attn_dev out_bias", b.len, d)?;
        }
        let hd = d / n_head;
        let f = size_of::<f32>();
        let tqd = checked_mul(
            checked_mul(t_q, d, "attn_dev t_q*d")?,
            f,
            "attn_dev t_q*d bytes",
        )?;
        let tq_hd_b = checked_mul(
            checked_mul(t_q, hd, "attn_dev t_q*hd")?,
            f,
            "attn_dev qh bytes",
        )?;
        let tkv_hd_b = checked_mul(
            checked_mul(t_kv, hd, "attn_dev t_kv*hd")?,
            f,
            "attn_dev vh bytes",
        )?;
        let hd_tkv_b = checked_mul(
            checked_mul(hd, t_kv, "attn_dev hd*t_kv")?,
            f,
            "attn_dev kh_t bytes",
        )?;
        let tq_tkv_b = checked_mul(
            checked_mul(t_q, t_kv, "attn_dev t_q*t_kv")?,
            f,
            "attn_dev scores bytes",
        )?;
        let dummy = self.alloc(f)?;
        let q_buf = self.alloc(tqd)?;
        let context_buf = self.alloc(tqd)?;
        let qh_buf = self.alloc(tq_hd_b)?;
        let vh_buf = self.alloc(tkv_hd_b)?;
        let kh_t_buf = self.alloc(hd_tkv_b)?;
        let scores_buf = self.alloc(tq_tkv_b)?;
        let probs_buf = self.alloc(tq_tkv_b)?;
        let ctx_h_buf = self.alloc(tq_hd_b)?;
        self.launch_attn_chain(
            &AttnChainDims {
                t_q,
                t_kv,
                d,
                n_head,
                scale,
                has_q_bias: q_bias.is_some(),
                has_out_bias: out_bias.is_some(),
            },
            &AttnChainPtrs {
                xq: xq.buf.ptr,
                q_w: q_w.buf.ptr,
                q_bias: bias_ptr(q_bias, dummy.ptr),
                k: k.buf.ptr,
                v: v.buf.ptr,
                out_w: out_w.buf.ptr,
                out_bias: bias_ptr(out_bias, dummy.ptr),
                q: q_buf.ptr,
                context: context_buf.ptr,
                qh: qh_buf.ptr,
                vh: vh_buf.ptr,
                kh_t: kh_t_buf.ptr,
                scores: scores_buf.ptr,
                probs: probs_buf.ptr,
                ctx_h: ctx_h_buf.ptr,
                out: out.buf.ptr,
            },
        )?;
        self.sync_stream("cuStreamSynchronize")
    }

    // ---- Phase-5 follow-on: device-resident whole-encoder stack --------------

    /// Runs the whole Whisper pre-norm **encoder** device-resident in ONE
    /// synchronise: `n × [ln → attn → residual → ln → mlp → residual]` + final ln,
    /// with the hidden state `h` and every intermediate kept on the GPU across all
    /// blocks. Mirrors `vokra-backend-metal`'s `encode_prenorm_stack`: `hidden` is
    /// the `[t, d]` post-conv-stem input (H2D once), `out` the `[t, d]` normed
    /// output (D2H once); per-block weights come as [`PrenormLayer`] slices
    /// (H2D'd once up front, before any launch, so the synchronous `cuMemcpyHtoD`
    /// never stalls on pending launches). `n_head` splits `d`,
    /// `scale = (d / n_head)^-0.5`.
    ///
    /// Bit-identical to running the blocks per-op on the GPU (same kernels, order,
    /// launch geometry) and matches the CPU within the FP32 bound; the difference
    /// is ONE `cuStreamSynchronize` for the whole encoder instead of the per-op
    /// path's `6·N + 1`. Stream ordering serialises the reused scratch across
    /// blocks and the two residual adds' read-modify-write of `h`.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on a shape mismatch or `d % n_head != 0`;
    /// [`VokraError::BackendUnavailable`] on a device allocation / launch failure.
    #[allow(clippy::too_many_arguments)] // whole-encoder operand set (dims + weights + I/O)
    pub fn encode_prenorm_stack(
        &self,
        t: usize,
        d: usize,
        ff: usize,
        n_head: usize,
        eps: f32,
        hidden: &[f32],
        layers: &[PrenormLayer<'_>],
        final_ln_gamma: &[f32],
        final_ln_beta: &[f32],
        out: &mut [f32],
    ) -> Result<()> {
        validate_prenorm_stack(
            t,
            d,
            ff,
            n_head,
            hidden,
            layers,
            final_ln_gamma,
            final_ln_beta,
            out,
        )?;
        self.run_prenorm_stack(
            t,
            d,
            ff,
            n_head,
            eps,
            hidden,
            layers,
            final_ln_gamma,
            final_ln_beta,
            out,
        )
    }

    /// Body of [`Self::encode_prenorm_stack`]: H2D `h` + all weights, allocate the
    /// device-resident scratch once, issue every block's launches on the one
    /// stream, synchronise ONCE, and D2H the final normed output. Shapes validated.
    #[allow(clippy::too_many_arguments)] // whole-encoder operand set (dims + weights + I/O)
    fn run_prenorm_stack(
        &self,
        t: usize,
        d: usize,
        ff: usize,
        n_head: usize,
        eps: f32,
        hidden: &[f32],
        layers: &[PrenormLayer<'_>],
        final_ln_gamma: &[f32],
        final_ln_beta: &[f32],
        out: &mut [f32],
    ) -> Result<()> {
        let hd = d / n_head;
        let scale = (hd as f32).powf(-0.5);

        // H2D `h` + every weight up front (before any launch).
        let h = self.upload(hidden)?;
        let dummy = self.upload(&[0.0f32])?;
        let mut dev_layers: Vec<DevLayer<'_>> = Vec::with_capacity(layers.len());
        for l in layers {
            dev_layers.push(DevLayer {
                attn_ln_g: self.upload(l.attn_ln_gamma)?,
                attn_ln_b: self.upload(l.attn_ln_beta)?,
                q_w: self.upload(l.q_w)?,
                q_bias: self.upload_opt(l.q_bias)?,
                k_w: self.upload(l.k_w)?,
                k_bias: self.upload_opt(l.k_bias)?,
                v_w: self.upload(l.v_w)?,
                v_bias: self.upload_opt(l.v_bias)?,
                out_w: self.upload(l.out_w)?,
                out_bias: self.upload_opt(l.out_bias)?,
                mlp_ln_g: self.upload(l.mlp_ln_gamma)?,
                mlp_ln_b: self.upload(l.mlp_ln_beta)?,
                fc1_w: self.upload(l.fc1_w)?,
                fc1_bias: self.upload_opt(l.fc1_bias)?,
                fc2_w: self.upload(l.fc2_w)?,
                fc2_bias: self.upload_opt(l.fc2_bias)?,
            });
        }
        let ln_post_g = self.upload(final_ln_gamma)?;
        let ln_post_b = self.upload(final_ln_beta)?;

        // Persistent device scratch (`t_q == t_kv == t`).
        let td = checked_mul(t, d, "prenorm t*d")?;
        let thd = checked_mul(t, hd, "prenorm t*hd")?;
        let tt = checked_mul(t, t, "prenorm t*t")?;
        let tff = checked_mul(t, ff, "prenorm t*ff")?;
        let ln = self.alloc_dev(td)?;
        let k = self.alloc_dev(td)?;
        let v = self.alloc_dev(td)?;
        let block_out = self.alloc_dev(td)?;
        let normed = self.alloc_dev(td)?;
        let q = self.alloc_dev(td)?;
        let context = self.alloc_dev(td)?;
        let qh = self.alloc_dev(thd)?;
        let vh = self.alloc_dev(thd)?;
        let kh_t = self.alloc_dev(thd)?;
        let scores = self.alloc_dev(tt)?;
        let probs = self.alloc_dev(tt)?;
        let ctx_h = self.alloc_dev(thd)?;
        let mlp_h = self.alloc_dev(tff)?;
        let mlp_a = self.alloc_dev(tff)?;

        for layer in &dev_layers {
            // 1. ln = layer_norm(h, attn_ln)
            self.launch_layer_norm_async(
                h.buf.ptr,
                layer.attn_ln_g.buf.ptr,
                layer.attn_ln_b.buf.ptr,
                ln.buf.ptr,
                t,
                d,
                eps,
            )?;
            // 2. k = ln · k_w (Whisper k has no bias)
            self.launch_gemm_async(
                ln.buf.ptr,
                layer.k_w.buf.ptr,
                bias_ptr(layer.k_bias.as_ref(), dummy.buf.ptr),
                k.buf.ptr,
                t,
                d,
                d,
                layer.k_bias.is_some(),
            )?;
            // 3. v = ln · v_w (+v_bias)
            self.launch_gemm_async(
                ln.buf.ptr,
                layer.v_w.buf.ptr,
                bias_ptr(layer.v_bias.as_ref(), dummy.buf.ptr),
                v.buf.ptr,
                t,
                d,
                d,
                layer.v_bias.is_some(),
            )?;
            // 4. attn → block_out
            self.launch_attn_chain(
                &AttnChainDims {
                    t_q: t,
                    t_kv: t,
                    d,
                    n_head,
                    scale,
                    has_q_bias: layer.q_bias.is_some(),
                    has_out_bias: layer.out_bias.is_some(),
                },
                &AttnChainPtrs {
                    xq: ln.buf.ptr,
                    q_w: layer.q_w.buf.ptr,
                    q_bias: bias_ptr(layer.q_bias.as_ref(), dummy.buf.ptr),
                    k: k.buf.ptr,
                    v: v.buf.ptr,
                    out_w: layer.out_w.buf.ptr,
                    out_bias: bias_ptr(layer.out_bias.as_ref(), dummy.buf.ptr),
                    q: q.buf.ptr,
                    context: context.buf.ptr,
                    qh: qh.buf.ptr,
                    vh: vh.buf.ptr,
                    kh_t: kh_t.buf.ptr,
                    scores: scores.buf.ptr,
                    probs: probs.buf.ptr,
                    ctx_h: ctx_h.buf.ptr,
                    out: block_out.buf.ptr,
                },
            )?;
            // 5. h += block_out
            self.launch_residual_add_async(h.buf.ptr, block_out.buf.ptr, td)?;
            // 6. ln = layer_norm(h, mlp_ln)
            self.launch_layer_norm_async(
                h.buf.ptr,
                layer.mlp_ln_g.buf.ptr,
                layer.mlp_ln_b.buf.ptr,
                ln.buf.ptr,
                t,
                d,
                eps,
            )?;
            // 7. mlp → block_out
            self.launch_mlp_chain(
                &MlpChainDims {
                    t,
                    d,
                    ffn: ff,
                    has_fc1_bias: layer.fc1_bias.is_some(),
                    has_fc2_bias: layer.fc2_bias.is_some(),
                },
                &MlpChainPtrs {
                    x: ln.buf.ptr,
                    fc1_w: layer.fc1_w.buf.ptr,
                    fc1_bias: bias_ptr(layer.fc1_bias.as_ref(), dummy.buf.ptr),
                    fc2_w: layer.fc2_w.buf.ptr,
                    fc2_bias: bias_ptr(layer.fc2_bias.as_ref(), dummy.buf.ptr),
                    h: mlp_h.buf.ptr,
                    a: mlp_a.buf.ptr,
                    out: block_out.buf.ptr,
                },
            )?;
            // 8. h += block_out
            self.launch_residual_add_async(h.buf.ptr, block_out.buf.ptr, td)?;
        }
        // Final LayerNorm into `normed`.
        self.launch_layer_norm_async(
            h.buf.ptr,
            ln_post_g.buf.ptr,
            ln_post_b.buf.ptr,
            normed.buf.ptr,
            t,
            d,
            eps,
        )?;

        self.sync_stream("cuStreamSynchronize")?;
        self.dtoh(&normed.buf, out)
    }

    /// Issues a single affine layer-norm launch on the stream (no synchronise).
    #[allow(clippy::too_many_arguments)] // intrinsic layer-norm parameter set
    fn launch_layer_norm_async(
        &self,
        inp: CUdeviceptr,
        gamma: CUdeviceptr,
        beta: CUdeviceptr,
        out: CUdeviceptr,
        rows: usize,
        cols: usize,
        eps: f32,
    ) -> Result<()> {
        let rows_u = rows as c_uint;
        let cols_u = cols as c_uint;
        let eps_v = eps;
        let mut params: [*mut c_void; 7] = [
            ptr_arg(&inp),
            ptr_arg(&gamma),
            ptr_arg(&beta),
            ptr_arg(&out),
            uint_arg(&rows_u),
            uint_arg(&cols_u),
            f32_arg(&eps_v),
        ];
        let grid = (rows.div_ceil(BLOCK_1D as usize) as c_uint, 1, 1);
        self.launch_async(
            self.layer_norm,
            grid,
            (BLOCK_1D, 1, 1),
            &mut params,
            "cuLaunchKernel(vokra_layer_norm_f32 prenorm)",
        )
    }

    /// Issues a single GEMM launch on the stream (no synchronise):
    /// `out[m,n] = bias?[n] + a[m,k]·b[k,n]`.
    #[allow(clippy::too_many_arguments)] // intrinsic GEMM parameter set
    fn launch_gemm_async(
        &self,
        a: CUdeviceptr,
        b: CUdeviceptr,
        bias: CUdeviceptr,
        out: CUdeviceptr,
        m: usize,
        n: usize,
        k: usize,
        has_bias: bool,
    ) -> Result<()> {
        let m_u = m as c_uint;
        let n_u = n as c_uint;
        let k_u = k as c_uint;
        let hb: c_uint = u32::from(has_bias);
        let mut params: [*mut c_void; 8] = [
            ptr_arg(&a),
            ptr_arg(&b),
            ptr_arg(&bias),
            ptr_arg(&out),
            uint_arg(&m_u),
            uint_arg(&n_u),
            uint_arg(&k_u),
            uint_arg(&hb),
        ];
        let grid = (
            n.div_ceil(BLOCK as usize) as c_uint,
            m.div_ceil(BLOCK as usize) as c_uint,
            1,
        );
        self.launch_async(
            self.gemm,
            grid,
            (BLOCK, BLOCK, 1),
            &mut params,
            "cuLaunchKernel(vokra_gemm_f32 prenorm)",
        )
    }

    /// Issues a single in-place residual-add launch on the stream (no synchronise):
    /// `dst[i] += src[i]`.
    fn launch_residual_add_async(
        &self,
        dst: CUdeviceptr,
        src: CUdeviceptr,
        n: usize,
    ) -> Result<()> {
        let n_u = n as c_uint;
        let mut params: [*mut c_void; 3] = [ptr_arg(&dst), ptr_arg(&src), uint_arg(&n_u)];
        let grid = (n.div_ceil(BLOCK_1D as usize) as c_uint, 1, 1);
        self.launch_async(
            self.add_assign,
            grid,
            (BLOCK_1D, 1, 1),
            &mut params,
            "cuLaunchKernel(vokra_add_assign_f32 prenorm)",
        )
    }

    /// Launches `func` on the context stream **without** synchronising — the
    /// fused MLP path issues three launches back to back and synchronises the
    /// stream once at the end. CUDA stream ordering guarantees each launch sees
    /// the previous launch's device writes (fc1 → gelu → fc2), so the `[t, ffn]`
    /// intermediates stay device-resident and are never copied D2H. Same launch
    /// contract as [`Self::launch`], minus the trailing `cuStreamSynchronize`.
    fn launch_async(
        &self,
        func: CUfunction,
        grid: (c_uint, c_uint, c_uint),
        block: (c_uint, c_uint, c_uint),
        params: &mut [*mut c_void],
        what: &str,
    ) -> Result<()> {
        let d = &self.driver;
        // SAFETY: `func` is a loaded kernel function; the grid/block dims are all
        // non-zero (validated t,d,ffn >= 1); `self.stream` is the owned stream;
        // `params` holds one valid pointer per kernel argument, matching the
        // kernel's signature and read by the driver during this launch; no dynamic
        // shared memory (0) and no `extra` (null). No synchronise — the caller
        // synchronises the stream once after chaining the launches.
        let r = unsafe {
            (d.cu_launch_kernel)(
                func,
                grid.0,
                grid.1,
                grid.2,
                block.0,
                block.1,
                block.2,
                0,
                self.stream,
                params.as_mut_ptr(),
                core::ptr::null_mut(),
            )
        };
        sys::check(d, r, what)
    }

    /// Launches `func` with the given `grid`/`block` dims (all `>= 1`) and
    /// `params` (one pointer per kernel argument, in declared order, each alive
    /// across this synchronous call), then synchronises the stream and surfaces a
    /// launch / sync error explicitly. Shared by the five Phase-4 kernels; the
    /// GEMM keeps its own bespoke launch in `run_gemm`.
    fn launch(
        &self,
        func: CUfunction,
        grid: (c_uint, c_uint, c_uint),
        block: (c_uint, c_uint, c_uint),
        params: &mut [*mut c_void],
        what: &str,
    ) -> Result<()> {
        let d = &self.driver;
        // SAFETY: `func` is a loaded kernel function from `kernels_module`; the
        // grid/block dims are all non-zero (the empty-output early return in each
        // caller guarantees it); `self.stream` is the owned stream; `params`
        // holds one valid pointer per kernel argument, matching the kernel's
        // signature and alive across this synchronous launch; no dynamic shared
        // memory (0) and no `extra` (null).
        let r = unsafe {
            (d.cu_launch_kernel)(
                func,
                grid.0,
                grid.1,
                grid.2,
                block.0,
                block.1,
                block.2,
                0,
                self.stream,
                params.as_mut_ptr(),
                core::ptr::null_mut(),
            )
        };
        sys::check(d, r, what)?;
        self.sync_stream("cuStreamSynchronize")
    }

    /// Synchronises the context stream ONCE, counting the submission (the
    /// env-independent readback/sync metric [`Self::submission_count`] reports),
    /// and surfaces a sync error. Every op that waits on the stream routes through
    /// here so the count is exact.
    fn sync_stream(&self, what: &str) -> Result<()> {
        self.submissions.set(self.submissions.get() + 1);
        // SAFETY: waits for every launch on the owned stream to complete before D2H.
        let sync = unsafe { (self.driver.cu_stream_synchronize)(self.stream) };
        sys::check(&self.driver, sync, what)
    }

    /// Allocates `bytes` (min one float) of device memory, tied to `&self`.
    fn alloc(&self, bytes: usize) -> Result<DeviceBuf<'_>> {
        let bytes = bytes.max(size_of::<f32>());
        let mut ptr: CUdeviceptr = 0;
        // SAFETY: `cuMemAlloc` writes a device pointer to `bytes` into `ptr`.
        let r = unsafe { (self.driver.cu_mem_alloc)(&mut ptr, bytes) };
        sys::check(&self.driver, r, "cuMemAlloc")?;
        Ok(DeviceBuf {
            driver: &self.driver,
            ptr,
        })
    }

    /// Copies `host` into device buffer `buf` (host-to-device).
    fn htod(&self, buf: &DeviceBuf<'_>, host: &[f32]) -> Result<()> {
        // SAFETY: `buf.ptr` is a device allocation of at least `size_of_val(host)`
        // bytes; `host.as_ptr()` is valid for that many bytes.
        let r = unsafe {
            (self.driver.cu_memcpy_htod)(buf.ptr, host.as_ptr().cast::<c_void>(), size_of_val(host))
        };
        sys::check(&self.driver, r, "cuMemcpyHtoD")
    }

    /// Copies device buffer `buf` into `host` (device-to-host).
    fn dtoh(&self, buf: &DeviceBuf<'_>, host: &mut [f32]) -> Result<()> {
        // SAFETY: `host.as_mut_ptr()` is valid for `size_of_val(host)` bytes;
        // `buf.ptr` is a device allocation of at least that many bytes.
        let r = unsafe {
            (self.driver.cu_memcpy_dtoh)(
                host.as_mut_ptr().cast::<c_void>(),
                buf.ptr,
                size_of_val(host),
            )
        };
        sys::check(&self.driver, r, "cuMemcpyDtoH")
    }
}

impl Drop for CudaContext {
    fn drop(&mut self) {
        // SAFETY: each handle is a valid owned object created in `new` /
        // `build_pipeline`; released once, in reverse creation order. The
        // `driver` (which owns the libcuda handle whose fn pointers these are)
        // is dropped only after this method returns, so the pointers are live.
        unsafe {
            if !self.stream.is_null() {
                (self.driver.cu_stream_synchronize)(self.stream);
                (self.driver.cu_stream_destroy)(self.stream);
            }
            if !self.kernels_module.is_null() {
                (self.driver.cu_module_unload)(self.kernels_module);
            }
            if !self.gemm_module.is_null() {
                (self.driver.cu_module_unload)(self.gemm_module);
            }
            if !self.context.is_null() {
                (self.driver.cu_ctx_destroy)(self.context);
            }
        }
    }
}

impl core::fmt::Debug for CudaContext {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("CudaContext").finish_non_exhaustive()
    }
}

/// The compiled modules + resolved kernel functions built by [`load_modules`]
/// (everything the [`CudaContext`] owns except the driver / context / stream).
struct Modules {
    gemm_module: CUmodule,
    kernels_module: CUmodule,
    gemm: CUfunction,
    gemv: CUfunction,
    softmax: CUfunction,
    layer_norm: CUfunction,
    gelu: CUfunction,
    conv1d: CUfunction,
    col_gather: CUfunction,
    col_gather_t: CUfunction,
    col_scatter: CUfunction,
    add_assign: CUfunction,
}

/// Owns a loaded CUDA module, unloading it once on drop unless defused with
/// [`ModuleGuard::into_raw`]. Mirrors the Metal backend's `Owned` guard: an early
/// `?`-return mid-build unloads every module already loaded, and the survivors
/// are defused into the [`CudaContext`] (whose `Drop` then owns them).
struct ModuleGuard<'a> {
    driver: &'a CudaDriver,
    module: CUmodule,
}

impl ModuleGuard<'_> {
    /// Takes the raw module handle, cancelling the drop-unload: ownership moves
    /// to the caller (the [`CudaContext`], whose `Drop` unloads it).
    fn into_raw(self) -> CUmodule {
        let module = self.module;
        core::mem::forget(self);
        module
    }
}

impl Drop for ModuleGuard<'_> {
    fn drop(&mut self) {
        if self.module.is_null() {
            return;
        }
        // SAFETY: `module` is a live module from `cuModuleLoadData`, unloaded once.
        unsafe { (self.driver.cu_module_unload)(self.module) };
    }
}

/// Creates the stream, then NVRTC-compiles + loads both kernel modules. On a
/// partial failure (module build fails after the stream exists) the stream is
/// destroyed before the error propagates; [`load_modules`] cleans up any module
/// it already loaded.
fn build_pipeline(driver: &CudaDriver) -> Result<(CUstream, Modules)> {
    let mut stream: CUstream = core::ptr::null_mut();
    // SAFETY: creates a stream (flags 0 = default) on the current context.
    let r = unsafe { (driver.cu_stream_create)(&mut stream, 0) };
    sys::check(driver, r, "cuStreamCreate")?;
    match load_modules(driver) {
        Ok(modules) => Ok((stream, modules)),
        Err(e) => {
            // SAFETY: `stream` is the just-created owned stream; destroy it.
            unsafe { (driver.cu_stream_destroy)(stream) };
            Err(e)
        }
    }
}

/// NVRTC-compiles both CUDA sources to PTX, loads each as a module, and resolves
/// every kernel function. Every module is held in a [`ModuleGuard`] until the
/// final success, so any `?`-return partway through unloads what was already
/// loaded (no leak); the survivors are defused into the returned [`Modules`].
fn load_modules(driver: &CudaDriver) -> Result<Modules> {
    let nvrtc = Nvrtc::load()?;
    let gemm_ptx = compile_ptx(&nvrtc, GEMM_CUDA, c"vokra_gemm.cu", "GEMM")?;
    let kernels_ptx = compile_ptx(&nvrtc, KERNELS_CUDA, c"vokra_kernels.cu", "kernels")?;
    drop(nvrtc); // libnvrtc no longer needed once the PTX text is in hand

    // GEMM module (guarded: a later `?` unloads it).
    let gemm_module = ModuleGuard {
        driver,
        module: load_module(driver, &gemm_ptx)?,
    };
    let gemm = get_function(driver, gemm_module.module, c"vokra_gemm_f32")?;

    // The five Phase-4 kernels share one module.
    let kernels_module = ModuleGuard {
        driver,
        module: load_module(driver, &kernels_ptx)?,
    };
    let gemv = get_function(driver, kernels_module.module, c"vokra_gemv_f32")?;
    let softmax = get_function(driver, kernels_module.module, c"vokra_softmax_f32")?;
    let layer_norm = get_function(driver, kernels_module.module, c"vokra_layer_norm_f32")?;
    let gelu = get_function(driver, kernels_module.module, c"vokra_gelu_f32")?;
    let conv1d = get_function(driver, kernels_module.module, c"vokra_conv1d_f32")?;
    // The three Phase-5 attention column-mover kernels share the same module.
    let col_gather = get_function(driver, kernels_module.module, c"vokra_col_gather_f32")?;
    let col_gather_t = get_function(driver, kernels_module.module, c"vokra_col_gather_t_f32")?;
    let col_scatter = get_function(driver, kernels_module.module, c"vokra_col_scatter_f32")?;
    // The Phase-5-follow-on residual-add kernel shares the same module.
    let add_assign = get_function(driver, kernels_module.module, c"vokra_add_assign_f32")?;

    // All resolved: defuse the guards into the owned handle set.
    Ok(Modules {
        gemm_module: gemm_module.into_raw(),
        kernels_module: kernels_module.into_raw(),
        gemm,
        gemv,
        softmax,
        layer_norm,
        gelu,
        conv1d,
        col_gather,
        col_gather_t,
        col_scatter,
        add_assign,
    })
}

/// Loads a NUL-terminated PTX image as a module (owned handle).
fn load_module(driver: &CudaDriver, ptx: &[u8]) -> Result<CUmodule> {
    let mut module: CUmodule = core::ptr::null_mut();
    // SAFETY: `ptx` is a NUL-terminated PTX image produced by NVRTC; the driver
    // parses it and writes the owned module handle into `module`.
    let r = unsafe { (driver.cu_module_load_data)(&mut module, ptx.as_ptr().cast::<c_void>()) };
    sys::check(driver, r, "cuModuleLoadData")?;
    Ok(module)
}

/// Resolves the `extern "C"` kernel named `name` in `module`.
fn get_function(
    driver: &CudaDriver,
    module: CUmodule,
    name: &core::ffi::CStr,
) -> Result<CUfunction> {
    let mut func: CUfunction = core::ptr::null_mut();
    // SAFETY: `module` is valid; `name` is a valid C string naming an `extern "C"`
    // kernel; the resolved handle is written into `func`.
    let r = unsafe { (driver.cu_module_get_function)(&mut func, module, name.as_ptr()) };
    sys::check(
        driver,
        r,
        &format!("cuModuleGetFunction({})", name.to_string_lossy()),
    )?;
    Ok(func)
}

/// A kernel-argument pointer for `cuLaunchKernel`: the address of `p` (a device
/// pointer), which the launch reads to get the `CUdeviceptr` value. `p` must
/// outlive the launch (it does — each caller keeps the `DeviceBuf` alive).
fn ptr_arg(p: &CUdeviceptr) -> *mut c_void {
    (p as *const CUdeviceptr).cast::<c_void>().cast_mut()
}

/// A kernel-argument pointer to a `u32` scalar (must outlive the launch).
fn uint_arg(p: &c_uint) -> *mut c_void {
    (p as *const c_uint).cast::<c_void>().cast_mut()
}

/// A kernel-argument pointer to an `f32` scalar (must outlive the launch).
fn f32_arg(p: &f32) -> *mut c_void {
    (p as *const f32).cast::<c_void>().cast_mut()
}

/// NVRTC-compiles a CUDA C `source` to a PTX byte buffer (NUL-terminated),
/// naming the translation unit `unit` and using `what` in any error. The program
/// handle is always destroyed before returning.
fn compile_ptx(nvrtc: &Nvrtc, source: &str, unit: &core::ffi::CStr, what: &str) -> Result<Vec<u8>> {
    let src = std::ffi::CString::new(source).map_err(|_| {
        VokraError::InvalidArgument(format!("{what} CUDA source contains an interior NUL"))
    })?;

    let mut prog: sys::NvrtcProgram = core::ptr::null_mut();
    // SAFETY: `src` is a valid NUL-terminated C string; `unit` is a C string;
    // 0 headers with null header/include arrays. Writes the program handle into
    // `prog`.
    let r = unsafe {
        (nvrtc.create_program)(
            &mut prog,
            src.as_ptr(),
            unit.as_ptr(),
            0,
            core::ptr::null(),
            core::ptr::null(),
        )
    };
    sys::check_nvrtc(nvrtc, r, "nvrtcCreateProgram")?;

    // `prog` now exists; ensure it is destroyed on every exit path.
    let result = compile_and_extract_ptx(nvrtc, prog, what);
    // SAFETY: `prog` is a valid program handle; destroyed exactly once here.
    unsafe { (nvrtc.destroy_program)(&mut prog) };
    result
}

/// Compiles `prog` (no options → NVRTC's default target arch) and extracts its
/// PTX. On a compile failure the NVRTC log is surfaced in the error (labelled
/// `what`).
///
/// vast.ai TODO: pin `--gpu-architecture=compute_89` (Ada / RTX 4090) once the
/// runner's toolkit version is confirmed, rather than relying on the default.
fn compile_and_extract_ptx(nvrtc: &Nvrtc, prog: sys::NvrtcProgram, what: &str) -> Result<Vec<u8>> {
    // SAFETY: `prog` is a valid program; 0 options with a null options array.
    let compile_res = unsafe { (nvrtc.compile_program)(prog, 0, core::ptr::null()) };
    if compile_res != sys::NVRTC_SUCCESS {
        return Err(VokraError::BackendUnavailable(format!(
            "NVRTC {what} compile failed: {}",
            fetch_nvrtc_log(nvrtc, prog)
        )));
    }

    let mut size = 0usize;
    // SAFETY: writes the PTX byte length (incl. trailing NUL) into `size`.
    let r = unsafe { (nvrtc.get_ptx_size)(prog, &mut size) };
    sys::check_nvrtc(nvrtc, r, "nvrtcGetPTXSize")?;

    let mut ptx = vec![0u8; size.max(1)];
    // SAFETY: `ptx` has `size` writable bytes; NVRTC writes the NUL-terminated
    // PTX text into it.
    let r = unsafe { (nvrtc.get_ptx)(prog, ptx.as_mut_ptr().cast::<c_char>()) };
    sys::check_nvrtc(nvrtc, r, "nvrtcGetPTX")?;
    Ok(ptx)
}

/// Best-effort NVRTC compile log for a program (`"(no log)"` if empty).
fn fetch_nvrtc_log(nvrtc: &Nvrtc, prog: sys::NvrtcProgram) -> String {
    let mut size = 0usize;
    // SAFETY: writes the log byte length into `size`.
    let ok = unsafe { (nvrtc.get_program_log_size)(prog, &mut size) };
    if ok != sys::NVRTC_SUCCESS || size <= 1 {
        return "(no log)".to_owned();
    }
    let mut buf = vec![0u8; size];
    // SAFETY: `buf` has `size` writable bytes; NVRTC writes the NUL-terminated
    // log into it.
    let ok = unsafe { (nvrtc.get_program_log)(prog, buf.as_mut_ptr().cast::<c_char>()) };
    if ok != sys::NVRTC_SUCCESS {
        return "(log unavailable)".to_owned();
    }
    sys::name_from_buf(&buf)
}

// ---- shape validation (mirrors the CPU / Metal gemm validator) ----

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

/// Validates the conv1d shapes (mirroring the CPU / Metal `conv1d` guard) and
/// returns the derived `out_len = (in_len + 2·padding − kernel) / stride + 1`.
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
/// composition of the two GEMM validators the fused path chains (mirrors the
/// Metal backend's `validate_mlp`), so a mis-shaped call is an explicit
/// `InvalidArgument` rather than a device fault.
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

/// Validates the fused non-causal attention shapes: `xq` is `[t_q, d]`, `k` / `v`
/// are `[t_kv, d]`, `q_w` / `out_w` are `[d, d]` (both projections `d → d`),
/// biases `[d]`, `out` is `[t_q, d]`, and `d` splits evenly into `n_head` heads —
/// so a mis-shaped call is an explicit `InvalidArgument` rather than a device
/// fault (mirrors the Metal backend's `validate_attn`).
#[allow(clippy::too_many_arguments)] // fused-attention operand set (two Linears + K/V + dims)
fn validate_attn(
    t_q: usize,
    t_kv: usize,
    d: usize,
    n_head: usize,
    xq: &[f32],
    q_w: &[f32],
    q_bias: Option<&[f32]>,
    k: &[f32],
    v: &[f32],
    out_w: &[f32],
    out_bias: Option<&[f32]>,
    out: &[f32],
) -> Result<()> {
    if t_q == 0 || t_kv == 0 || d == 0 || n_head == 0 {
        return Err(VokraError::InvalidArgument(
            "attn dimensions t_q, t_kv, d, n_head must all be >= 1".to_owned(),
        ));
    }
    if d % n_head != 0 {
        return Err(VokraError::InvalidArgument(format!(
            "attn d ({d}) must be divisible by n_head ({n_head})"
        )));
    }
    let dd = checked_mul(d, d, "attn d*d")?;
    let tkvd = checked_mul(t_kv, d, "attn t_kv*d")?;
    expect_len("attn xq", xq.len(), checked_mul(t_q, d, "attn t_q*d")?)?;
    expect_len("attn q_w", q_w.len(), dd)?;
    if let Some(bias) = q_bias {
        expect_len("attn q_bias", bias.len(), d)?;
    }
    expect_len("attn k", k.len(), tkvd)?;
    expect_len("attn v", v.len(), tkvd)?;
    expect_len("attn out_w", out_w.len(), dd)?;
    if let Some(bias) = out_bias {
        expect_len("attn out_bias", bias.len(), d)?;
    }
    expect_len(
        "attn out",
        out.len(),
        checked_mul(t_q, d, "attn out t_q*d")?,
    )?;
    Ok(())
}

/// The device bias pointer for a projection: the real bias when present, else the
/// shared `dummy` the kernel never reads (`has_bias = 0`).
fn bias_ptr(bias: Option<&CudaDeviceTensor<'_>>, dummy: CUdeviceptr) -> CUdeviceptr {
    bias.map_or(dummy, |t| t.buf.ptr)
}

/// Validates the whole-encoder pre-norm stack shapes (mirrors the Metal backend's
/// `validate_prenorm_stack`): `hidden` / `out` are `[t, d]`, `d` splits evenly
/// into `n_head`, the final LayerNorm `γ`/`β` are `[d]`, and every
/// [`PrenormLayer`]'s LayerNorms are `[d]`, projections `[d, d]` (biases `[d]`),
/// MLP linears `[d, ff]` / `[ff, d]` (biases `[ff]` / `[d]`).
#[allow(clippy::too_many_arguments)] // whole-encoder operand set (dims + weights + I/O)
fn validate_prenorm_stack(
    t: usize,
    d: usize,
    ff: usize,
    n_head: usize,
    hidden: &[f32],
    layers: &[PrenormLayer<'_>],
    final_ln_gamma: &[f32],
    final_ln_beta: &[f32],
    out: &[f32],
) -> Result<()> {
    if t == 0 || d == 0 || ff == 0 || n_head == 0 {
        return Err(VokraError::InvalidArgument(
            "prenorm stack dimensions t, d, ff, n_head must all be >= 1".to_owned(),
        ));
    }
    if d % n_head != 0 {
        return Err(VokraError::InvalidArgument(format!(
            "prenorm stack d ({d}) must be divisible by n_head ({n_head})"
        )));
    }
    let td = checked_mul(t, d, "prenorm t*d")?;
    let dd = checked_mul(d, d, "prenorm d*d")?;
    let dff = checked_mul(d, ff, "prenorm d*ff")?;
    let ffd = checked_mul(ff, d, "prenorm ff*d")?;
    expect_len("prenorm hidden", hidden.len(), td)?;
    expect_len("prenorm out", out.len(), td)?;
    expect_len("prenorm final_ln_gamma", final_ln_gamma.len(), d)?;
    expect_len("prenorm final_ln_beta", final_ln_beta.len(), d)?;
    for (i, l) in layers.iter().enumerate() {
        let opt = |name: &str, b: Option<&[f32]>, want: usize| -> Result<()> {
            match b {
                Some(s) => expect_len(&format!("prenorm layer {i} {name}"), s.len(), want),
                None => Ok(()),
            }
        };
        expect_len(
            &format!("prenorm layer {i} attn_ln_gamma"),
            l.attn_ln_gamma.len(),
            d,
        )?;
        expect_len(
            &format!("prenorm layer {i} attn_ln_beta"),
            l.attn_ln_beta.len(),
            d,
        )?;
        expect_len(&format!("prenorm layer {i} q_w"), l.q_w.len(), dd)?;
        expect_len(&format!("prenorm layer {i} k_w"), l.k_w.len(), dd)?;
        expect_len(&format!("prenorm layer {i} v_w"), l.v_w.len(), dd)?;
        expect_len(&format!("prenorm layer {i} out_w"), l.out_w.len(), dd)?;
        opt("q_bias", l.q_bias, d)?;
        opt("k_bias", l.k_bias, d)?;
        opt("v_bias", l.v_bias, d)?;
        opt("out_bias", l.out_bias, d)?;
        expect_len(
            &format!("prenorm layer {i} mlp_ln_gamma"),
            l.mlp_ln_gamma.len(),
            d,
        )?;
        expect_len(
            &format!("prenorm layer {i} mlp_ln_beta"),
            l.mlp_ln_beta.len(),
            d,
        )?;
        expect_len(&format!("prenorm layer {i} fc1_w"), l.fc1_w.len(), dff)?;
        expect_len(&format!("prenorm layer {i} fc2_w"), l.fc2_w.len(), ffd)?;
        opt("fc1_bias", l.fc1_bias, ff)?;
        opt("fc2_bias", l.fc2_bias, d)?;
    }
    Ok(())
}
