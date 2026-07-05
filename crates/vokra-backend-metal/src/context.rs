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

use core::cell::Cell;
use core::ffi::c_void;
use core::marker::PhantomData;

use vokra_core::{PrenormLayer, Result, VokraError};

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

// ---- Phase-5 attention fusion: three pure-copy column movers -----------------
// These replace the host `copy_from_slice` / transpose / `*= scale` the per-op
// `whisper::nn::attention_from_kv_into` runs between GPU ops. Each is a pure data
// move (+ one FP32 multiply in the gather) — one thread per destination (gather /
// gather_t) or source (scatter) element, ragged-tail guarded like every kernel
// above — so the bits they move are trivially identical to the host code they
// replace, keeping the fused path bit-for-bit equal to the per-op path.

// col_gather: dst[i*hd + c] = src[i*width + c0 + c] * scale. Gathers one head's
// `hd`-wide column block out of a `[rows, width]` row-major matrix, folding the
// query scale (qh: scale = head_dim^-0.5; vh: scale = 1).
struct ColGatherDims {
    uint rows;
    uint hd;
    uint width;
    uint c0;
    float scale;
};

kernel void vokra_col_gather_f32(
    device const float*     src [[buffer(0)]],
    device float*           dst [[buffer(1)]],
    constant ColGatherDims& d   [[buffer(2)]],
    uint                    gid [[thread_position_in_grid]])
{
    const uint n = d.rows * d.hd;
    if (gid >= n) {
        return;
    }
    const uint i = gid / d.hd;
    const uint c = gid % d.hd;
    dst[gid] = src[i * d.width + d.c0 + c] * d.scale;
}

// col_gather_t: dst[c*t_kv + j] = src[j*width + c0 + c]. Gathers one head's key
// column block AND transposes it to `[hd, t_kv]` (what the scores GEMM needs as
// its right operand), replacing the host `kh_t[c*t_kv + j] = k[j*d + c0 + c]`.
struct ColGatherTDims {
    uint t_kv;
    uint hd;
    uint width;
    uint c0;
};

kernel void vokra_col_gather_t_f32(
    device const float*      src [[buffer(0)]],
    device float*            dst [[buffer(1)]],
    constant ColGatherTDims& d   [[buffer(2)]],
    uint                     gid [[thread_position_in_grid]])
{
    const uint n = d.hd * d.t_kv;
    if (gid >= n) {
        return;
    }
    const uint c = gid / d.t_kv;
    const uint j = gid % d.t_kv;
    dst[gid] = src[j * d.width + d.c0 + c];
}

// col_scatter: dst[i*width + c0 + c] = src[i*hd + c]. Scatters this head's
// `[rows, hd]` context back into its `hd`-wide column block of `[rows, width]`,
// replacing the host `context[i*d + c0 + c] = ctx_h[i*hd + c]`. Because
// n_head*hd == width every column is written by exactly one head, so `context`
// needs no zeroing (it is fully overwritten, as on the CPU).
struct ColScatterDims {
    uint rows;
    uint hd;
    uint width;
    uint c0;
};

kernel void vokra_col_scatter_f32(
    device const float*      src [[buffer(0)]],
    device float*            dst [[buffer(1)]],
    constant ColScatterDims& d   [[buffer(2)]],
    uint                     gid [[thread_position_in_grid]])
{
    const uint n = d.rows * d.hd;
    if (gid >= n) {
        return;
    }
    const uint i = gid / d.hd;
    const uint c = gid % d.hd;
    dst[i * d.width + d.c0 + c] = src[gid];
}

// ---- Phase-5 follow-on: in-place residual add (dst[i] += src[i]) -------------
// The device kernel for the encoder block's `h += sub_block` residual, replacing
// the host `whisper::nn::add_assign` loop so `h` stays resident across a whole
// device-resident encoder. `dst` is bound read-write at index 0. One thread per
// element, ragged-tail guarded — a single FP32 add of the same two operands the
// host loop adds, so it is bit-identical to `add_assign`.
struct AddAssignDims {
    uint n;
};

kernel void vokra_add_assign_f32(
    device float*           dst [[buffer(0)]],
    device const float*     src [[buffer(1)]],
    constant AddAssignDims& d   [[buffer(2)]],
    uint                    gid [[thread_position_in_grid]])
{
    if (gid >= d.n) {
        return;
    }
    dst[gid] = dst[gid] + src[gid];
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

/// `col_gather` dims (`setBytes:` index 2). Field order / widths mirror the MSL
/// `struct ColGatherDims`; the trailing `f32 scale` is folded into the copy (all
/// fields 4-byte, so `#[repr(C)]` needs no padding).
#[repr(C)]
#[derive(Clone, Copy)]
struct ColGatherDims {
    rows: u32,
    hd: u32,
    width: u32,
    c0: u32,
    scale: f32,
}

/// `col_gather_t` dims (`setBytes:` index 2). Mirrors the MSL `struct
/// ColGatherTDims`.
#[repr(C)]
#[derive(Clone, Copy)]
struct ColGatherTDims {
    t_kv: u32,
    hd: u32,
    width: u32,
    c0: u32,
}

/// `col_scatter` dims (`setBytes:` index 2). Mirrors the MSL `struct
/// ColScatterDims`.
#[repr(C)]
#[derive(Clone, Copy)]
struct ColScatterDims {
    rows: u32,
    hd: u32,
    width: u32,
    c0: u32,
}

/// `add_assign` dims (`setBytes:` index 2). Mirrors the MSL `struct
/// AddAssignDims`.
#[repr(C)]
#[derive(Clone, Copy)]
struct AddAssignDims {
    n: u32,
}

/// Scalar shape of one fused-MLP pass chain, shared by the host-in/out
/// [`MetalContext::run_mlp`], the device-in/out [`MetalContext::mlp_dev`] and the
/// whole-encoder [`MetalContext::encode_prenorm_stack`] so all three encode the
/// same three passes.
struct MlpPassDims {
    t: usize,
    d: usize,
    ffn: usize,
    has_fc1_bias: bool,
    has_fc2_bias: bool,
}

/// The already-allocated device buffers for one fused-MLP pass chain (`x` `[t,d]`,
/// `fc1_w` `[d,ffn]`, `fc2_w` `[ffn,d]`, biases `[ffn]`/`[d]` — a 1-float dummy
/// when absent, `h`/`a` `[t,ffn]` device-resident intermediates, `out` `[t,d]`).
struct MlpPassBufs<'b> {
    x: &'b OwnedBuf,
    fc1_w: &'b OwnedBuf,
    fc1_bias: &'b OwnedBuf,
    fc2_w: &'b OwnedBuf,
    fc2_bias: &'b OwnedBuf,
    h: &'b OwnedBuf,
    a: &'b OwnedBuf,
    out: &'b OwnedBuf,
}

/// Scalar shape of one fused non-causal attention pass chain, shared by
/// [`MetalContext::run_attn`], [`MetalContext::attn_dev`] and
/// [`MetalContext::encode_prenorm_stack`]. `scale = head_dim^-0.5` is folded into
/// the qh gather.
struct AttnPassDims {
    t_q: usize,
    t_kv: usize,
    d: usize,
    n_head: usize,
    scale: f32,
    has_q_bias: bool,
    has_out_bias: bool,
}

/// The already-allocated device buffers for one fused-attention pass chain: the
/// inputs (`xq` `[t_q,d]`, `q_w`/`out_w` `[d,d]`, biases `[d]`, pre-projected
/// `k`/`v` `[t_kv,d]`), the device-resident scratch (`q`/`context` `[t_q,d]`,
/// per-head `qh`/`ctx_h` `[t_q,hd]`, `vh` `[t_kv,hd]`, `kh_t` `[hd,t_kv]`,
/// `scores`/`probs` `[t_q,t_kv]`), and `out` `[t_q,d]`.
struct AttnPassBufs<'b> {
    xq: &'b OwnedBuf,
    q_w: &'b OwnedBuf,
    q_bias: &'b OwnedBuf,
    k: &'b OwnedBuf,
    v: &'b OwnedBuf,
    out_w: &'b OwnedBuf,
    out_bias: &'b OwnedBuf,
    q: &'b OwnedBuf,
    context: &'b OwnedBuf,
    qh: &'b OwnedBuf,
    vh: &'b OwnedBuf,
    kh_t: &'b OwnedBuf,
    scores: &'b OwnedBuf,
    probs: &'b OwnedBuf,
    ctx_h: &'b OwnedBuf,
    out: &'b OwnedBuf,
}

/// One pre-norm block's weights uploaded to the device (the on-GPU mirror of
/// [`vokra_core::PrenormLayer`]), held for the life of an
/// [`MetalContext::encode_prenorm_stack`] call. Absent biases (Whisper's `k`)
/// stay `None` and bind the shared dummy at encode time.
struct DevLayer<'c> {
    attn_ln_g: MetalDeviceTensor<'c>,
    attn_ln_b: MetalDeviceTensor<'c>,
    q_w: MetalDeviceTensor<'c>,
    q_bias: Option<MetalDeviceTensor<'c>>,
    k_w: MetalDeviceTensor<'c>,
    k_bias: Option<MetalDeviceTensor<'c>>,
    v_w: MetalDeviceTensor<'c>,
    v_bias: Option<MetalDeviceTensor<'c>>,
    out_w: MetalDeviceTensor<'c>,
    out_bias: Option<MetalDeviceTensor<'c>>,
    mlp_ln_g: MetalDeviceTensor<'c>,
    mlp_ln_b: MetalDeviceTensor<'c>,
    fc1_w: MetalDeviceTensor<'c>,
    fc1_bias: Option<MetalDeviceTensor<'c>>,
    fc2_w: MetalDeviceTensor<'c>,
    fc2_bias: Option<MetalDeviceTensor<'c>>,
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

/// A public, cross-call handle to a device-resident `[f32]` buffer — the
/// Phase-5-follow-on surface that lets a caller keep intermediates on the GPU
/// between op calls (produced by [`MetalContext::upload`] / [`alloc_dev`], read
/// back by [`download`], consumed by the `*_dev` ops).
///
/// - Owns its `MTLBuffer` through the existing [`OwnedBuf`] RAII (released once on
///   drop), so it adds no new `unsafe`.
/// - `len` is the f32 element count (buffer sizing / readback validation).
/// - The `PhantomData<&'ctx MetalContext>` ties the handle's lifetime to the
///   context it was allocated from: because every producer is an `&'ctx self`
///   method returning `MetalDeviceTensor<'ctx>`, holding a tensor past the
///   context's `Drop` is a **compile error**. It also inherits `OwnedBuf`'s
///   `!Send`/`!Sync` (the raw `Id` is a `*mut c_void`), matching the context's
///   thread affinity with no manual marker.
///
/// [`alloc_dev`]: MetalContext::alloc_dev
/// [`download`]: MetalContext::download
pub struct MetalDeviceTensor<'ctx> {
    buf: OwnedBuf,
    len: usize,
    _ctx: PhantomData<&'ctx MetalContext>,
}

impl MetalDeviceTensor<'_> {
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
    col_gather_pipeline: Id,
    col_gather_t_pipeline: Id,
    col_scatter_pipeline: Id,
    add_assign_pipeline: Id,
    /// Count of command-buffer submissions (`commit` + `waitUntilCompleted`)
    /// issued through this context — the env-independent readback/sync metric the
    /// Phase-5-follow-on encoder-residency slice proves against (the whole encoder
    /// in ONE submission vs the per-op path's `6·N + 1`). `Cell` because every op
    /// takes `&self` and the context is already thread-affine (`!Send`/`!Sync`).
    submissions: Cell<u64>,
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
        // The three Phase-5 attention column-mover kernels share the same library.
        // SAFETY: as above.
        let col_gather_pipeline =
            unsafe { make_pipeline(device, klib.0, c"vokra_col_gather_f32") }?;
        // SAFETY: as above.
        let col_gather_t_pipeline =
            unsafe { make_pipeline(device, klib.0, c"vokra_col_gather_t_f32") }?;
        // SAFETY: as above.
        let col_scatter_pipeline =
            unsafe { make_pipeline(device, klib.0, c"vokra_col_scatter_f32") }?;
        // The Phase-5-follow-on residual-add kernel shares the same library.
        // SAFETY: as above.
        let add_assign_pipeline =
            unsafe { make_pipeline(device, klib.0, c"vokra_add_assign_f32") }?;
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
            col_gather_pipeline: col_gather_pipeline.into_raw(),
            col_gather_t_pipeline: col_gather_t_pipeline.into_raw(),
            col_scatter_pipeline: col_scatter_pipeline.into_raw(),
            add_assign_pipeline: add_assign_pipeline.into_raw(),
            submissions: Cell::new(0),
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
            self.submissions.set(self.submissions.get() + 1);
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

        // One command buffer for the whole chain: encode the three passes (shared
        // with `mlp_dev` / `encode_prenorm_stack` so the numerics are single-
        // sourced), then commit + wait ONCE.
        let cmd = self.new_command_buffer("mlp")?;
        self.encode_mlp_passes(
            cmd,
            &MlpPassDims {
                t,
                d,
                ffn,
                has_fc1_bias: fc1_bias.is_some(),
                has_fc2_bias: fc2_bias.is_some(),
            },
            &MlpPassBufs {
                x: &x_buf,
                fc1_w: &fc1_w_buf,
                fc1_bias: &fc1_bias_buf,
                fc2_w: &fc2_w_buf,
                fc2_bias: &fc2_bias_buf,
                h: &h_buf,
                a: &a_buf,
                out: &out_buf,
            },
        )?;
        self.commit_and_wait(cmd, "mlp")?;

        // Single readback of the final output; `h`/`a` stay resident and drop.
        read_back(&out_buf, out)
    }

    /// Encodes the three fused-MLP passes (`fc1` GEMM → GELU → `fc2` GEMM) into
    /// the already-open `cmd`, operating on already-allocated device buffers,
    /// **without** committing / allocating / reading back. Factored out of
    /// [`Self::run_mlp`] so the host-in/out [`Self::mlp_f32`], the device-in/out
    /// [`Self::mlp_dev`] and the whole-encoder [`Self::encode_prenorm_stack`] run
    /// byte-for-byte identical passes (same kernels, order, launch geometry). The
    /// caller sized every buffer (`h` / `a` are `[t, ffn]`, `out` is `[t, d]`) and
    /// commits + waits once afterwards.
    fn encode_mlp_passes(&self, cmd: Id, dims: &MlpPassDims, bufs: &MlpPassBufs<'_>) -> Result<()> {
        let (t, d, ffn) = (dims.t, dims.d, dims.ffn);
        // `t*ffn` cannot overflow here: the caller allocated the `[t, ffn]`
        // buffers, which required the same product to fit.
        let inter = t * ffn;
        let fc1_dims = GemmDims {
            m: t as u32,
            n: ffn as u32,
            k: d as u32,
            has_bias: u32::from(dims.has_fc1_bias),
        };
        let gelu_dims = GeluDims { n: inter as u32 };
        let fc2_dims = GemmDims {
            m: t as u32,
            n: d as u32,
            k: ffn as u32,
            has_bias: u32::from(dims.has_fc2_bias),
        };

        // Pass 1: h = x[t,d] · fc1_w[d,ffn] (+bias) — GEMM (grid = N×M, 16×16).
        let (fc1_grid, fc1_tg) = grid_2d(ffn, t);
        self.encode_pass(
            cmd,
            self.gemm_pipeline,
            &[bufs.x, bufs.fc1_w, bufs.fc1_bias, bufs.h],
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
            &[bufs.h, bufs.a],
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
            &[bufs.a, bufs.fc2_w, bufs.fc2_bias, bufs.out],
            (&fc2_dims as *const GemmDims).cast::<c_void>(),
            size_of::<GemmDims>(),
            fc2_grid,
            fc2_tg,
            "mlp fc2",
        )?;
        Ok(())
    }

    // ---- Phase-5 fusion: device-resident non-causal attention ----------------

    /// Fused **non-causal** multi-head attention on the GPU with every
    /// intermediate **resident on the device** — the Phase-5 attention
    /// readback-elimination slice (the sibling of [`Self::mlp_f32`]).
    ///
    /// Computes `out = out_proj( concat_h softmax(scale · qₕ·kₕᵀ) · vₕ )` for
    /// `xq` `[t_q, d]`, pre-projected `k` / `v` `[t_kv, d]`, `q_w` / `out_w`
    /// `[d, d]` (both projections are `d → d`), optional biases `[d]`, and
    /// `scale = head_dim^-0.5` (the caller folds the query scale in). `out` is
    /// `[t_q, d]`.
    ///
    /// It runs the **same** `vokra_gemm_f32` (q-proj, per-head scores, per-head
    /// context, out-proj) and `vokra_softmax_f32` kernels the per-op
    /// `whisper::nn::attention_from_kv_into` runs, in the same order and launch
    /// geometry, with the head gather / transpose / scatter (formerly host
    /// `copy_from_slice`) done by the three pure-copy `col_*` kernels — so the
    /// result is **bit-identical** to the per-op path. The difference is that the
    /// per-head scratch (`qh` / `vh` / `kh_t` / `scores` / `probs` / `ctx_h`) and
    /// the `q` / `context` intermediates never leave the device: the whole chain
    /// is ONE command buffer with ONE `waitUntilCompleted` and ONE readback (of
    /// `out`) instead of the per-op path's per-op H2D/D2H round-trips.
    ///
    /// **Non-causal only** (encoder self-attention and decoder cross-attention).
    /// Causal decoder self-attention stays on the per-op path (it needs the mask
    /// write between the scores GEMM and the softmax).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on any shape mismatch, a zero dimension, or
    /// `d % n_head != 0`; [`VokraError::BackendUnavailable`] on a Metal buffer /
    /// command failure.
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
        // Bracket the GPU work in an autorelease pool (as the per-op methods do).
        // SAFETY: `objc_autoreleasePoolPush` returns a token consumed by the one
        // matching pop below.
        let pool = unsafe { sys::objc_autoreleasePoolPush() };
        let r = self.run_attn(
            t_q, t_kv, d, n_head, xq, q_w, q_bias, k, v, out_w, out_bias, scale, out,
        );
        // SAFETY: `pool` is the token from the push above.
        unsafe { sys::objc_autoreleasePoolPop(pool) };
        r
    }

    /// Fused-attention body: copy the inputs H2D, allocate every intermediate
    /// **device-resident** (never read back) plus the `[t_q, d]` output, encode
    /// the `2 + 7·n_head` passes (q-proj GEMM → per head {gather qh, gather vh,
    /// gather-transpose kh_t, scores GEMM, softmax, context GEMM, scatter} →
    /// out-proj GEMM) into ONE command buffer, commit + wait ONCE, and read back
    /// only `out`. Runs inside `attn_f32`'s autorelease pool; shapes are already
    /// validated (so `hd = d / n_head` is exact).
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

        // Inputs copied H2D into shared storage (a failed alloc `?`-returns;
        // already-built `OwnedBuf`s release on drop).
        let xq_buf = self.new_buffer_from_slice(xq)?;
        let q_w_buf = self.new_buffer_from_slice(q_w)?;
        let dummy = [0.0f32];
        let q_bias_buf = self.new_buffer_from_slice(q_bias.unwrap_or(&dummy))?;
        let k_buf = self.new_buffer_from_slice(k)?;
        let v_buf = self.new_buffer_from_slice(v)?;
        let out_w_buf = self.new_buffer_from_slice(out_w)?;
        let out_bias_buf = self.new_buffer_from_slice(out_bias.unwrap_or(&dummy))?;

        // Device-resident intermediates: `q` / `context` `[t_q, d]` and the reused
        // per-head scratch. None is ever read back — that is the readback this
        // slice eliminates. `out` `[t_q, d]` is the single buffer read back.
        let tqd = checked_mul(t_q, d, "attn t_q*d")?;
        let tq_hd = checked_mul(t_q, hd, "attn t_q*hd")?;
        let tkv_hd = checked_mul(t_kv, hd, "attn t_kv*hd")?;
        let hd_tkv = checked_mul(hd, t_kv, "attn hd*t_kv")?;
        let tq_tkv = checked_mul(t_q, t_kv, "attn t_q*t_kv")?;
        let q_buf = self.new_buffer_output(tqd)?; // q-proj [t_q, d]
        let context_buf = self.new_buffer_output(tqd)?; // per-head scatter target [t_q, d]
        let qh_buf = self.new_buffer_output(tq_hd)?; // this head's q [t_q, hd]
        let vh_buf = self.new_buffer_output(tkv_hd)?; // this head's v [t_kv, hd]
        let kh_t_buf = self.new_buffer_output(hd_tkv)?; // this head's kᵀ [hd, t_kv]
        let scores_buf = self.new_buffer_output(tq_tkv)?; // scores [t_q, t_kv]
        let probs_buf = self.new_buffer_output(tq_tkv)?; // softmax [t_q, t_kv]
        let ctx_h_buf = self.new_buffer_output(tq_hd)?; // this head's ctx [t_q, hd]
        let out_buf = self.new_buffer_output(out.len())?; // [t_q, d]

        // One command buffer for the whole chain: encode every pass (shared with
        // `attn_dev` / `encode_prenorm_stack` so the numerics are single-sourced),
        // then commit + wait ONCE.
        let cmd = self.new_command_buffer("attn")?;
        self.encode_attn_passes(
            cmd,
            &AttnPassDims {
                t_q,
                t_kv,
                d,
                n_head,
                scale,
                has_q_bias: q_bias.is_some(),
                has_out_bias: out_bias.is_some(),
            },
            &AttnPassBufs {
                xq: &xq_buf,
                q_w: &q_w_buf,
                q_bias: &q_bias_buf,
                k: &k_buf,
                v: &v_buf,
                out_w: &out_w_buf,
                out_bias: &out_bias_buf,
                q: &q_buf,
                context: &context_buf,
                qh: &qh_buf,
                vh: &vh_buf,
                kh_t: &kh_t_buf,
                scores: &scores_buf,
                probs: &probs_buf,
                ctx_h: &ctx_h_buf,
                out: &out_buf,
            },
        )?;
        self.commit_and_wait(cmd, "attn")?;

        // Single readback of the final output; every intermediate stays resident
        // and drops.
        read_back(&out_buf, out)
    }

    /// Encodes the fused non-causal attention passes (q-proj GEMM → per head
    /// {gather qh/vh, gather-transpose kh_t, scores GEMM, softmax, context GEMM,
    /// scatter} → out-proj GEMM) into the already-open `cmd`, operating on
    /// already-allocated device buffers, **without** committing / allocating /
    /// reading back. Factored out of [`Self::run_attn`] so the host-in/out
    /// [`Self::attn_f32`], the device-in/out [`Self::attn_dev`] and the
    /// whole-encoder [`Self::encode_prenorm_stack`] run byte-for-byte identical
    /// passes. The per-head scratch (`qh` / `vh` / `kh_t` / `scores` / `probs` /
    /// `ctx_h`) is reused across heads; Metal hazard-tracks the shared buffers so
    /// head h+1's gather into `qh` is ordered after head h's scores GEMM read of
    /// it. `dims.scale` is folded into the qh gather (the query scale). Bias-less
    /// GEMMs bind `bufs.q_bias` as the never-read dummy (`has_bias = 0`).
    /// `hd = d / n_head` is exact (the caller validated it).
    fn encode_attn_passes(
        &self,
        cmd: Id,
        dims: &AttnPassDims,
        bufs: &AttnPassBufs<'_>,
    ) -> Result<()> {
        let (t_q, t_kv, d, n_head) = (dims.t_q, dims.t_kv, dims.d, dims.n_head);
        let hd = d / n_head;
        // These products all fit: the caller allocated buffers of these sizes.
        let tq_hd = t_q * hd;
        let tkv_hd = t_kv * hd;
        let hd_tkv = hd * t_kv;

        // Pass 1: q = xq[t_q,d] · q_w[d,d] (+q_bias) — GEMM (grid = N×M, 16×16).
        // The query scale is NOT applied here; it is folded into the qh gather
        // below (the same single FP32 multiply the CPU does after this GEMM).
        let q_dims = GemmDims {
            m: t_q as u32,
            n: d as u32,
            k: d as u32,
            has_bias: u32::from(dims.has_q_bias),
        };
        let (q_grid, q_tg) = grid_2d(d, t_q);
        self.encode_pass(
            cmd,
            self.gemm_pipeline,
            &[bufs.xq, bufs.q_w, bufs.q_bias, bufs.q],
            (&q_dims as *const GemmDims).cast::<c_void>(),
            size_of::<GemmDims>(),
            q_grid,
            q_tg,
            "attn q-proj",
        )?;

        // Per head: gather qh (scaled) / vh / kh_tᵀ, scores GEMM, softmax, context
        // GEMM, scatter. `setBytes:` copies the dims eagerly, so the per-head dims
        // locals need not outlive the loop.
        for h in 0..n_head {
            let c0 = (h * hd) as u32;
            // qh[i,c] = q[i, c0+c] * scale.
            let qh_dims = ColGatherDims {
                rows: t_q as u32,
                hd: hd as u32,
                width: d as u32,
                c0,
                scale: dims.scale,
            };
            let (gq_grid, gq_tg) = grid_1d(tq_hd);
            self.encode_pass(
                cmd,
                self.col_gather_pipeline,
                &[bufs.q, bufs.qh],
                (&qh_dims as *const ColGatherDims).cast::<c_void>(),
                size_of::<ColGatherDims>(),
                gq_grid,
                gq_tg,
                "attn gather qh",
            )?;
            // vh[j,c] = v[j, c0+c] (scale = 1).
            let vh_dims = ColGatherDims {
                rows: t_kv as u32,
                hd: hd as u32,
                width: d as u32,
                c0,
                scale: 1.0,
            };
            let (gv_grid, gv_tg) = grid_1d(tkv_hd);
            self.encode_pass(
                cmd,
                self.col_gather_pipeline,
                &[bufs.v, bufs.vh],
                (&vh_dims as *const ColGatherDims).cast::<c_void>(),
                size_of::<ColGatherDims>(),
                gv_grid,
                gv_tg,
                "attn gather vh",
            )?;
            // kh_t[c,j] = k[j, c0+c] (gather + transpose to [hd, t_kv]).
            let kh_dims = ColGatherTDims {
                t_kv: t_kv as u32,
                hd: hd as u32,
                width: d as u32,
                c0,
            };
            let (gk_grid, gk_tg) = grid_1d(hd_tkv);
            self.encode_pass(
                cmd,
                self.col_gather_t_pipeline,
                &[bufs.k, bufs.kh_t],
                (&kh_dims as *const ColGatherTDims).cast::<c_void>(),
                size_of::<ColGatherTDims>(),
                gk_grid,
                gk_tg,
                "attn gather kh_t",
            )?;
            // scores[t_q,t_kv] = qh[t_q,hd] · kh_t[hd,t_kv].
            let scores_dims = GemmDims {
                m: t_q as u32,
                n: t_kv as u32,
                k: hd as u32,
                has_bias: 0,
            };
            let (s_grid, s_tg) = grid_2d(t_kv, t_q);
            self.encode_pass(
                cmd,
                self.gemm_pipeline,
                &[bufs.qh, bufs.kh_t, bufs.q_bias, bufs.scores],
                (&scores_dims as *const GemmDims).cast::<c_void>(),
                size_of::<GemmDims>(),
                s_grid,
                s_tg,
                "attn scores",
            )?;
            // probs = softmax_rows(scores) (no mask — non-causal).
            let sm_dims = SoftmaxDims {
                rows: t_q as u32,
                cols: t_kv as u32,
            };
            let (sm_grid, sm_tg) = grid_1d(t_q);
            self.encode_pass(
                cmd,
                self.softmax_pipeline,
                &[bufs.scores, bufs.probs],
                (&sm_dims as *const SoftmaxDims).cast::<c_void>(),
                size_of::<SoftmaxDims>(),
                sm_grid,
                sm_tg,
                "attn softmax",
            )?;
            // ctx_h[t_q,hd] = probs[t_q,t_kv] · vh[t_kv,hd].
            let ctx_dims = GemmDims {
                m: t_q as u32,
                n: hd as u32,
                k: t_kv as u32,
                has_bias: 0,
            };
            let (c_grid, c_tg) = grid_2d(hd, t_q);
            self.encode_pass(
                cmd,
                self.gemm_pipeline,
                &[bufs.probs, bufs.vh, bufs.q_bias, bufs.ctx_h],
                (&ctx_dims as *const GemmDims).cast::<c_void>(),
                size_of::<GemmDims>(),
                c_grid,
                c_tg,
                "attn context",
            )?;
            // context[i, c0+c] = ctx_h[i,c].
            let scatter_dims = ColScatterDims {
                rows: t_q as u32,
                hd: hd as u32,
                width: d as u32,
                c0,
            };
            let (sc_grid, sc_tg) = grid_1d(tq_hd);
            self.encode_pass(
                cmd,
                self.col_scatter_pipeline,
                &[bufs.ctx_h, bufs.context],
                (&scatter_dims as *const ColScatterDims).cast::<c_void>(),
                size_of::<ColScatterDims>(),
                sc_grid,
                sc_tg,
                "attn scatter",
            )?;
        }

        // Pass last: out = context[t_q,d] · out_w[d,d] (+out_bias) — GEMM.
        let out_dims = GemmDims {
            m: t_q as u32,
            n: d as u32,
            k: d as u32,
            has_bias: u32::from(dims.has_out_bias),
        };
        let (o_grid, o_tg) = grid_2d(d, t_q);
        self.encode_pass(
            cmd,
            self.gemm_pipeline,
            &[bufs.context, bufs.out_w, bufs.out_bias, bufs.out],
            (&out_dims as *const GemmDims).cast::<c_void>(),
            size_of::<GemmDims>(),
            o_grid,
            o_tg,
            "attn out-proj",
        )?;
        Ok(())
    }

    // ---- Phase-5 follow-on: public device-resident handle + ops --------------

    /// The number of command-buffer submissions (`commit` + `waitUntilCompleted`)
    /// issued through this context so far. The env-independent readback/sync
    /// metric: the whole-encoder [`Self::encode_prenorm_stack`] issues ONE, versus
    /// the per-op path's `6·N + 1` for an `N`-block encoder.
    #[must_use]
    pub fn submission_count(&self) -> u64 {
        self.submissions.get()
    }

    /// Uploads `data` into a fresh device-resident buffer (H2D once). The returned
    /// [`MetalDeviceTensor`] borrows the context, so it cannot outlive it.
    ///
    /// # Errors
    ///
    /// [`VokraError::BackendUnavailable`] if the Metal buffer cannot be created.
    pub fn upload(&self, data: &[f32]) -> Result<MetalDeviceTensor<'_>> {
        let buf = self.new_buffer_from_slice(data)?;
        Ok(MetalDeviceTensor {
            buf,
            len: data.len(),
            _ctx: PhantomData,
        })
    }

    /// Allocates an uninitialised device-resident buffer of `len` f32s (the
    /// residency slice's intermediates; never round-tripped to the host until an
    /// explicit [`Self::download`]).
    ///
    /// # Errors
    ///
    /// [`VokraError::BackendUnavailable`] if the Metal buffer cannot be created.
    pub fn alloc_dev(&self, len: usize) -> Result<MetalDeviceTensor<'_>> {
        let buf = self.new_buffer_output(len)?;
        Ok(MetalDeviceTensor {
            buf,
            len,
            _ctx: PhantomData,
        })
    }

    /// Reads a device-resident buffer back into `out` (D2H). Call after the owning
    /// submission has completed (the `*_dev` ops and [`Self::encode_prenorm_stack`]
    /// wait before returning, so a tensor they produced is readable immediately).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] if `out.len()` differs from the tensor's
    /// element count; [`VokraError::BackendUnavailable`] on a null contents
    /// pointer.
    pub fn download(&self, t: &MetalDeviceTensor<'_>, out: &mut [f32]) -> Result<()> {
        expect_len("download out", out.len(), t.len)?;
        read_back(&t.buf, out)
    }

    /// Device-in/out affine layer normalisation (one self-contained submission):
    /// `out = layer_norm(x)·γ + β` over the innermost axis of a `rows × cols`
    /// buffer. Bit-identical to the host-in/out [`Self::layer_norm_f32`] (same
    /// kernel); `out` must be a distinct buffer from `x`.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on a shape mismatch;
    /// [`VokraError::BackendUnavailable`] on a Metal failure.
    #[allow(clippy::too_many_arguments)] // intrinsic layer-norm parameter set
    pub fn layer_norm_dev(
        &self,
        out: &mut MetalDeviceTensor<'_>,
        x: &MetalDeviceTensor<'_>,
        gamma: &MetalDeviceTensor<'_>,
        beta: &MetalDeviceTensor<'_>,
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
        self.pooled(|| {
            let cmd = self.new_command_buffer("layer_norm_dev")?;
            self.encode_layer_norm(
                cmd, &x.buf, &gamma.buf, &beta.buf, &out.buf, rows, cols, eps,
            )?;
            self.commit_and_wait(cmd, "layer_norm_dev")
        })
    }

    /// Device-in/out in-place residual add (one self-contained submission):
    /// `dst[i] += src[i]`. Bit-identical to the host `whisper::nn::add_assign`
    /// loop (the same single FP32 add).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] if the lengths differ;
    /// [`VokraError::BackendUnavailable`] on a Metal failure.
    pub fn residual_add_dev(
        &self,
        dst: &mut MetalDeviceTensor<'_>,
        src: &MetalDeviceTensor<'_>,
    ) -> Result<()> {
        expect_len("residual_add_dev src", src.len, dst.len)?;
        if dst.len == 0 {
            return Ok(());
        }
        let n = dst.len;
        self.pooled(|| {
            let cmd = self.new_command_buffer("residual_add_dev")?;
            self.encode_residual_add(cmd, &dst.buf, &src.buf, n)?;
            self.commit_and_wait(cmd, "residual_add_dev")
        })
    }

    /// Device-in/out fused MLP `fc2(gelu(fc1(x)))` (one self-contained submission,
    /// the two `[t, ffn]` intermediates allocated internally and never read back).
    /// Bit-identical to the host-in/out [`Self::mlp_f32`] (same passes).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on a shape mismatch;
    /// [`VokraError::BackendUnavailable`] on a Metal failure.
    #[allow(clippy::too_many_arguments)] // fused-MLP operand set (two Linears + dims)
    pub fn mlp_dev(
        &self,
        t: usize,
        d: usize,
        ffn: usize,
        x: &MetalDeviceTensor<'_>,
        fc1_w: &MetalDeviceTensor<'_>,
        fc1_bias: Option<&MetalDeviceTensor<'_>>,
        fc2_w: &MetalDeviceTensor<'_>,
        fc2_bias: Option<&MetalDeviceTensor<'_>>,
        out: &mut MetalDeviceTensor<'_>,
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
        let inter = checked_mul(t, ffn, "mlp_dev t*ffn")?;
        self.pooled(|| {
            let dummy = self.new_buffer_from_slice(&[0.0f32])?;
            let h_buf = self.new_buffer_output(inter)?;
            let a_buf = self.new_buffer_output(inter)?;
            let cmd = self.new_command_buffer("mlp_dev")?;
            self.encode_mlp_passes(
                cmd,
                &MlpPassDims {
                    t,
                    d,
                    ffn,
                    has_fc1_bias: fc1_bias.is_some(),
                    has_fc2_bias: fc2_bias.is_some(),
                },
                &MlpPassBufs {
                    x: &x.buf,
                    fc1_w: &fc1_w.buf,
                    fc1_bias: bias_or_dummy(fc1_bias, &dummy),
                    fc2_w: &fc2_w.buf,
                    fc2_bias: bias_or_dummy(fc2_bias, &dummy),
                    h: &h_buf,
                    a: &a_buf,
                    out: &out.buf,
                },
            )?;
            self.commit_and_wait(cmd, "mlp_dev")
        })
    }

    /// Device-in/out fused **non-causal** attention (one self-contained
    /// submission, every intermediate allocated internally and never read back).
    /// `xq` `[t_q,d]`; pre-projected `k`/`v` `[t_kv,d]`; `q_w`/`out_w` `[d,d]`;
    /// `scale = head_dim^-0.5`; `out` `[t_q,d]`. Bit-identical to the host-in/out
    /// [`Self::attn_f32`] (same passes).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on a shape mismatch or `d % n_head != 0`;
    /// [`VokraError::BackendUnavailable`] on a Metal failure.
    #[allow(clippy::too_many_arguments)] // fused-attention operand set (two Linears + K/V + dims)
    pub fn attn_dev(
        &self,
        t_q: usize,
        t_kv: usize,
        d: usize,
        n_head: usize,
        xq: &MetalDeviceTensor<'_>,
        q_w: &MetalDeviceTensor<'_>,
        q_bias: Option<&MetalDeviceTensor<'_>>,
        k: &MetalDeviceTensor<'_>,
        v: &MetalDeviceTensor<'_>,
        out_w: &MetalDeviceTensor<'_>,
        out_bias: Option<&MetalDeviceTensor<'_>>,
        scale: f32,
        out: &mut MetalDeviceTensor<'_>,
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
        let tqd = checked_mul(t_q, d, "attn_dev t_q*d")?;
        let tq_hd = checked_mul(t_q, hd, "attn_dev t_q*hd")?;
        let tkv_hd = checked_mul(t_kv, hd, "attn_dev t_kv*hd")?;
        let hd_tkv = checked_mul(hd, t_kv, "attn_dev hd*t_kv")?;
        let tq_tkv = checked_mul(t_q, t_kv, "attn_dev t_q*t_kv")?;
        self.pooled(|| {
            let dummy = self.new_buffer_from_slice(&[0.0f32])?;
            let q_buf = self.new_buffer_output(tqd)?;
            let context_buf = self.new_buffer_output(tqd)?;
            let qh_buf = self.new_buffer_output(tq_hd)?;
            let vh_buf = self.new_buffer_output(tkv_hd)?;
            let kh_t_buf = self.new_buffer_output(hd_tkv)?;
            let scores_buf = self.new_buffer_output(tq_tkv)?;
            let probs_buf = self.new_buffer_output(tq_tkv)?;
            let ctx_h_buf = self.new_buffer_output(tq_hd)?;
            let cmd = self.new_command_buffer("attn_dev")?;
            self.encode_attn_passes(
                cmd,
                &AttnPassDims {
                    t_q,
                    t_kv,
                    d,
                    n_head,
                    scale,
                    has_q_bias: q_bias.is_some(),
                    has_out_bias: out_bias.is_some(),
                },
                &AttnPassBufs {
                    xq: &xq.buf,
                    q_w: &q_w.buf,
                    q_bias: bias_or_dummy(q_bias, &dummy),
                    k: &k.buf,
                    v: &v.buf,
                    out_w: &out_w.buf,
                    out_bias: bias_or_dummy(out_bias, &dummy),
                    q: &q_buf,
                    context: &context_buf,
                    qh: &qh_buf,
                    vh: &vh_buf,
                    kh_t: &kh_t_buf,
                    scores: &scores_buf,
                    probs: &probs_buf,
                    ctx_h: &ctx_h_buf,
                    out: &out.buf,
                },
            )?;
            self.commit_and_wait(cmd, "attn_dev")
        })
    }

    // ---- Phase-5 follow-on: device-resident whole-encoder stack --------------

    /// Runs the whole Whisper pre-norm **encoder** device-resident in ONE
    /// submission: `n × [ln → attn → residual → ln → mlp → residual]` + final ln,
    /// with the hidden state `h` and every intermediate kept on the GPU across all
    /// blocks. `hidden` is the `[t, d]` post-conv-stem input (H2D once), `out` the
    /// `[t, d]` final-LayerNorm output (D2H once); the per-block weights come as
    /// [`PrenormLayer`] slices (uploaded once up front). `n_head` splits `d`,
    /// `scale = (d / n_head)^-0.5`.
    ///
    /// It encodes **exactly** the per-op path's op sequence — the same
    /// `layer_norm` / GEMM / [`encode_attn_passes`](Self::encode_attn_passes) /
    /// [`encode_mlp_passes`](Self::encode_mlp_passes) / residual-add kernels, in
    /// the same order and launch geometry — so it is **bit-identical** to running
    /// the blocks per-op on the GPU, and matches the CPU within the FP32 bound. The
    /// difference is the readback: ONE `commit` + `waitUntilCompleted` for the
    /// whole encoder instead of the per-op path's `6·N + 1`. Intra-command-buffer
    /// hazard tracking serialises the reused `ln`/`k`/`v`/`block_out`/per-head
    /// scratch across blocks and the two residual adds' read-modify-write of `h`.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on a shape mismatch or `d % n_head != 0`;
    /// [`VokraError::BackendUnavailable`] on a Metal buffer / command failure.
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
        self.pooled(|| {
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
        })
    }

    /// Body of [`Self::encode_prenorm_stack`]: uploads `h` + all weights, allocates
    /// the device-resident scratch once, encodes every block's passes into ONE
    /// command buffer, commits + waits ONCE, and reads back the final normed
    /// output. Runs inside the caller's autorelease pool; shapes are validated.
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

        // Up front (before any pass), H2D `h` + every layer's weights + the final
        // LayerNorm + a 1-float dummy for absent biases (Whisper's `k`).
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

        // Persistent device scratch (mirrors `EncoderScratch`; `t_q == t_kv == t`,
        // so nothing grows between blocks). All reused across blocks/heads.
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

        // One command buffer for the whole encoder.
        let cmd = self.new_command_buffer("prenorm stack")?;
        for layer in &dev_layers {
            // h += attn(ln(h)):
            // 1. ln = layer_norm(h, attn_ln)
            self.encode_layer_norm(
                cmd,
                &h.buf,
                &layer.attn_ln_g.buf,
                &layer.attn_ln_b.buf,
                &ln.buf,
                t,
                d,
                eps,
            )?;
            // 2. k = ln · k_w (Whisper k has no bias)
            self.encode_gemm(
                cmd,
                &ln.buf,
                &layer.k_w.buf,
                bias_or_dummy(layer.k_bias.as_ref(), &dummy.buf),
                &k.buf,
                t,
                d,
                d,
                layer.k_bias.is_some(),
            )?;
            // 3. v = ln · v_w (+v_bias)
            self.encode_gemm(
                cmd,
                &ln.buf,
                &layer.v_w.buf,
                bias_or_dummy(layer.v_bias.as_ref(), &dummy.buf),
                &v.buf,
                t,
                d,
                d,
                layer.v_bias.is_some(),
            )?;
            // 4. attn: block_out = out_proj(concat_h softmax(scale·qₕ·kₕᵀ)·vₕ)
            self.encode_attn_passes(
                cmd,
                &AttnPassDims {
                    t_q: t,
                    t_kv: t,
                    d,
                    n_head,
                    scale,
                    has_q_bias: layer.q_bias.is_some(),
                    has_out_bias: layer.out_bias.is_some(),
                },
                &AttnPassBufs {
                    xq: &ln.buf,
                    q_w: &layer.q_w.buf,
                    q_bias: bias_or_dummy(layer.q_bias.as_ref(), &dummy.buf),
                    k: &k.buf,
                    v: &v.buf,
                    out_w: &layer.out_w.buf,
                    out_bias: bias_or_dummy(layer.out_bias.as_ref(), &dummy.buf),
                    q: &q.buf,
                    context: &context.buf,
                    qh: &qh.buf,
                    vh: &vh.buf,
                    kh_t: &kh_t.buf,
                    scores: &scores.buf,
                    probs: &probs.buf,
                    ctx_h: &ctx_h.buf,
                    out: &block_out.buf,
                },
            )?;
            // 5. h += block_out
            self.encode_residual_add(cmd, &h.buf, &block_out.buf, td)?;

            // h += mlp(ln(h)):
            // 6. ln = layer_norm(h, mlp_ln)
            self.encode_layer_norm(
                cmd,
                &h.buf,
                &layer.mlp_ln_g.buf,
                &layer.mlp_ln_b.buf,
                &ln.buf,
                t,
                d,
                eps,
            )?;
            // 7. mlp: block_out = fc2(gelu(fc1(ln)))
            self.encode_mlp_passes(
                cmd,
                &MlpPassDims {
                    t,
                    d,
                    ffn: ff,
                    has_fc1_bias: layer.fc1_bias.is_some(),
                    has_fc2_bias: layer.fc2_bias.is_some(),
                },
                &MlpPassBufs {
                    x: &ln.buf,
                    fc1_w: &layer.fc1_w.buf,
                    fc1_bias: bias_or_dummy(layer.fc1_bias.as_ref(), &dummy.buf),
                    fc2_w: &layer.fc2_w.buf,
                    fc2_bias: bias_or_dummy(layer.fc2_bias.as_ref(), &dummy.buf),
                    h: &mlp_h.buf,
                    a: &mlp_a.buf,
                    out: &block_out.buf,
                },
            )?;
            // 8. h += block_out
            self.encode_residual_add(cmd, &h.buf, &block_out.buf, td)?;
        }
        // Final LayerNorm into `normed`.
        self.encode_layer_norm(
            cmd,
            &h.buf,
            &ln_post_g.buf,
            &ln_post_b.buf,
            &normed.buf,
            t,
            d,
            eps,
        )?;

        self.commit_and_wait(cmd, "prenorm stack")?;
        self.download(&normed, out)
    }

    /// Uploads an optional weight slice (a `None` bias stays `None`, bound as the
    /// shared dummy at encode time).
    fn upload_opt(&self, data: Option<&[f32]>) -> Result<Option<MetalDeviceTensor<'_>>> {
        data.map(|d| self.upload(d)).transpose()
    }

    /// Opens a fresh command buffer on the context queue.
    fn new_command_buffer(&self, what: &str) -> Result<Id> {
        // SAFETY: `queue` is valid for the context's lifetime; `commandBuffer`
        // returns an autoreleased command buffer drained by the caller's pool.
        let cmd = unsafe { sys::send_id(self.queue, sys::sel(b"commandBuffer\0")) };
        if cmd.is_null() {
            return Err(VokraError::BackendUnavailable(format!(
                "{what}: MTLCommandQueue commandBuffer returned nil"
            )));
        }
        Ok(cmd)
    }

    /// Commits + waits on `cmd` ONCE (counting the submission) and surfaces a
    /// GPU-side execution error explicitly. Shared by every device-resident op.
    fn commit_and_wait(&self, cmd: Id, what: &str) -> Result<()> {
        self.submissions.set(self.submissions.get() + 1);
        // SAFETY: `cmd` is the valid command buffer with passes encoded above;
        // `commit` then `waitUntilCompleted` submit and block; `error` is read
        // after completion (no silent success).
        unsafe {
            sys::send_void(cmd, sys::sel(b"commit\0"));
            sys::send_void(cmd, sys::sel(b"waitUntilCompleted\0"));
            let cmd_err = sys::send_id(cmd, sys::sel(b"error\0"));
            if !cmd_err.is_null() {
                let detail = error_description(cmd_err);
                return Err(VokraError::BackendUnavailable(format!(
                    "{what} command buffer failed: {detail}"
                )));
            }
        }
        Ok(())
    }

    /// Brackets `f` in an autorelease pool so the command buffer / encoders it
    /// creates drain here rather than leaking on a plain worker thread.
    fn pooled<T>(&self, f: impl FnOnce() -> Result<T>) -> Result<T> {
        // SAFETY: `objc_autoreleasePoolPush` returns a token consumed by the one
        // matching pop below.
        let pool = unsafe { sys::objc_autoreleasePoolPush() };
        let r = f();
        // SAFETY: `pool` is the token from the push above.
        unsafe { sys::objc_autoreleasePoolPop(pool) };
        r
    }

    /// Encodes an affine layer-norm pass into `cmd` (one row per thread).
    #[allow(clippy::too_many_arguments)] // intrinsic layer-norm parameter set
    fn encode_layer_norm(
        &self,
        cmd: Id,
        inp: &OwnedBuf,
        gamma: &OwnedBuf,
        beta: &OwnedBuf,
        out: &OwnedBuf,
        rows: usize,
        cols: usize,
        eps: f32,
    ) -> Result<()> {
        let dims = LayerNormDims {
            rows: rows as u32,
            cols: cols as u32,
            eps,
        };
        let (grid, tg) = grid_1d(rows);
        self.encode_pass(
            cmd,
            self.layer_norm_pipeline,
            &[inp, gamma, beta, out],
            (&dims as *const LayerNormDims).cast::<c_void>(),
            size_of::<LayerNormDims>(),
            grid,
            tg,
            "prenorm layer_norm",
        )
    }

    /// Encodes a GEMM pass into `cmd` (`out[m,n] = bias?[n] + a[m,k]·b[k,n]`).
    #[allow(clippy::too_many_arguments)] // intrinsic GEMM parameter set
    fn encode_gemm(
        &self,
        cmd: Id,
        a: &OwnedBuf,
        b: &OwnedBuf,
        bias: &OwnedBuf,
        out: &OwnedBuf,
        m: usize,
        n: usize,
        k: usize,
        has_bias: bool,
    ) -> Result<()> {
        let dims = GemmDims {
            m: m as u32,
            n: n as u32,
            k: k as u32,
            has_bias: u32::from(has_bias),
        };
        let (grid, tg) = grid_2d(n, m);
        self.encode_pass(
            cmd,
            self.gemm_pipeline,
            &[a, b, bias, out],
            (&dims as *const GemmDims).cast::<c_void>(),
            size_of::<GemmDims>(),
            grid,
            tg,
            "prenorm gemm",
        )
    }

    /// Encodes an in-place residual-add pass into `cmd` (`dst[i] += src[i]`).
    fn encode_residual_add(&self, cmd: Id, dst: &OwnedBuf, src: &OwnedBuf, n: usize) -> Result<()> {
        let dims = AddAssignDims { n: n as u32 };
        let (grid, tg) = grid_1d(n);
        self.encode_pass(
            cmd,
            self.add_assign_pipeline,
            &[dst, src],
            (&dims as *const AddAssignDims).cast::<c_void>(),
            size_of::<AddAssignDims>(),
            grid,
            tg,
            "prenorm residual add",
        )
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
            self.submissions.set(self.submissions.get() + 1);
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
            release(self.add_assign_pipeline);
            release(self.col_scatter_pipeline);
            release(self.col_gather_t_pipeline);
            release(self.col_gather_pipeline);
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

/// Validates the fused non-causal attention shapes: `xq` is `[t_q, d]`, `k` / `v`
/// are `[t_kv, d]`, `q_w` / `out_w` are `[d, d]` (both projections `d → d`),
/// biases `[d]`, `out` is `[t_q, d]`, and `d` splits evenly into `n_head` heads —
/// so a mis-shaped call is an explicit `InvalidArgument` rather than a GPU fault
/// (mirrors [`validate_mlp`]).
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

/// The device bias buffer for a projection: the real bias when present, else the
/// shared 1-float `dummy` the kernel never reads (`has_bias = 0`). The returned
/// borrow lives as long as the shorter of the two inputs.
fn bias_or_dummy<'a>(bias: Option<&'a MetalDeviceTensor<'_>>, dummy: &'a OwnedBuf) -> &'a OwnedBuf {
    match bias {
        Some(t) => &t.buf,
        None => dummy,
    }
}

/// Validates the whole-encoder pre-norm stack shapes: `hidden` / `out` are
/// `[t, d]`, `d` splits evenly into `n_head`, the final LayerNorm `γ`/`β` are
/// `[d]`, and every [`PrenormLayer`]'s LayerNorms are `[d]`, projections `[d, d]`
/// (biases `[d]`), and MLP linears `[d, ff]` / `[ff, d]` (biases `[ff]` / `[d]`) —
/// so a mis-shaped call is an explicit `InvalidArgument` rather than a GPU fault.
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
