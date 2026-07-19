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

use vokra_core::{DecoderLayerView, PrenormLayer, Result, VokraError};

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

// ---- softmax_causal: row-wise softmax over the causally-visible key prefix ---
// The decoder self-attention mask, fused into the softmax so the causal decode
// step needs no separate mask write. Row `r` (query at absolute position
// `q_offset + r`) attends keys `[0, q_offset + r]`; keys beyond that are the
// "future" the causal mask hides. This is BIT-IDENTICAL to writing -INF into
// scores[r, j>last] and running the plain softmax above:
//   * max: column 0 is always visible (0 <= q_offset+r), the same seed; masked
//     columns j>last would be -INF and never the max — so max over [0,last] is
//     the same value;
//   * sum: the masked columns contribute exp(-INF - m) = 0.0f, and `acc + 0.0f`
//     is exactly `acc` (IEEE-754), so summing only [0,last] gives the identical
//     partial sums in the identical ascending order;
//   * out: masked columns get exactly 0.0f (as `0 * inv`), visible columns get
//     `exp * inv` — identical.
// For a single new token (t_q = 1) `last = q_offset = t_kv - 1`, so ALL keys are
// visible and this is the plain softmax bit-for-bit; the mask only bites on the
// multi-token prefix step (t_q > 1).
struct SoftmaxCausalDims {
    uint rows;
    uint cols;
    uint q_offset; // absolute position of query row 0
};

kernel void vokra_softmax_causal_f32(
    device const float*         inp [[buffer(0)]],
    device float*               out [[buffer(1)]],
    constant SoftmaxCausalDims& d   [[buffer(2)]],
    uint                        gid [[thread_position_in_grid]])
{
    const uint r = gid;
    if (r >= d.rows) {
        return;
    }
    const uint base = r * d.cols;
    // Last visible key column for this row (clamped; the caller guarantees
    // last < cols, so the clamp is defensive only).
    uint last = d.q_offset + r;
    if (last >= d.cols) {
        last = d.cols - 1u;
    }
    float m = inp[base]; // column 0 is always visible (0 <= q_offset + r)
    for (uint j = 1u; j <= last; ++j) {
        m = fmax(m, inp[base + j]);
    }
    float sum = 0.0f;
    for (uint j = 0u; j <= last; ++j) {
        float e = exp(inp[base + j] - m);
        out[base + j] = e;
        sum += e;
    }
    const float inv = 1.0f / sum;
    for (uint j = 0u; j <= last; ++j) {
        out[base + j] *= inv;
    }
    for (uint j = last + 1u; j < d.cols; ++j) {
        out[base + j] = 0.0f; // future keys -> 0 (exactly as the host mask does)
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

// ---- cc-27: element-wise multiply + copy (graph-executor `Mul` / `Copy`) -----
// The two kernels that bring the Metal graph arm level with the CUDA / Vulkan /
// WebGPU arms. Both reuse `AddAssignDims` (a single `uint n`) — the operand
// layout is identical to `vokra_add_assign_f32`, only the combining operation
// differs — so no new dims struct is needed on either side of the FFI.
//
// `vokra_mul_f32` is in-place (`dst` read-write at index 0) exactly like the
// residual add, so `eval_mul` mirrors `eval_add` operand-for-operand. One FP32
// multiply per element — the same single rounding the CPU `kernels::mul_f32`
// performs, with no reduction order to disagree about. Measured bit-identical
// against the CPU backend over normal-range operands on M1
// (`graph_metal.rs::mul_matches_cpu_backend`, max |Δ| = 0). MSL is compiled
// with fast-math defaults, which permit denormal flush-to-zero, so the
// bit-identity claim is scoped to normal-range operands; the parity test pins
// that scope explicitly rather than asserting it universally.
kernel void vokra_mul_f32(
    device float*           dst [[buffer(0)]],
    device const float*     src [[buffer(1)]],
    constant AddAssignDims& d   [[buffer(2)]],
    uint                    gid [[thread_position_in_grid]])
{
    if (gid >= d.n) {
        return;
    }
    dst[gid] = dst[gid] * src[gid];
}

// `vokra_copy_f32` is the identity element-wise move `dst[i] = src[i]` into a
// SEPARATE destination buffer (mirrors the Vulkan hand-crafted `copy_f32`).
// Distinct from `MetalContext::download`: this is a real compute dispatch, so
// `OpKind::Copy` genuinely executes on the GPU rather than being emulated by a
// host memcpy through the upload / read-back pair.
kernel void vokra_copy_f32(
    device float*           dst [[buffer(0)]],
    device const float*     src [[buffer(1)]],
    constant AddAssignDims& d   [[buffer(2)]],
    uint                    gid [[thread_position_in_grid]])
{
    if (gid >= d.n) {
        return;
    }
    dst[gid] = src[gid];
}

// ---- M3-04 fused KV-cache dequant + GEMV kernels ----------------------------
//
// One thread per output row. Each block of 32 quantised values is dequantised
// into a per-thread scalar inside the GEMV reduction — no shared / threadgroup
// scratch. Byte layout mirrors `vokra_core::kv_quant::dequantize_bytes` exactly
// (Q4_0 = 18 B, Q5_0 = 22 B, Q8_0 = 34 B), so the same on-wire block payload
// feeds the CPU differential oracle (`dequant_gemv_scalar`) and this GPU
// kernel.
//
// MSL has no builtin `f16 → f32` helper for a raw `u16` bit pattern, so we
// duplicate the CPU `vokra_core::kv_quant::half::f16_bits_to_f32` semantics in
// device code here. Kept in the same file as the kernels so a future update
// touches one place.
inline float vokra_kv_f16_to_f32(uint h) {
    uint sign = (h >> 15u) & 1u;
    uint exp  = (h >> 10u) & 0x1Fu;
    uint mant = h & 0x3FFu;
    float sign_f = (sign == 1u) ? -1.0f : 1.0f;
    if (exp == 0u) {
        // Subnormal / zero (matches CPU: sign_f * mant * 2^-24).
        return sign_f * (float)mant * ldexp(1.0f, -24);
    }
    if (exp == 0x1Fu) {
        if (mant == 0u) {
            return sign_f * INFINITY;
        }
        return 0.0f / 0.0f; // NaN, matching CPU `f32::NAN`.
    }
    return sign_f * (1.0f + (float)mant / 1024.0f) * ldexp(1.0f, (int)exp - 15);
}

// Dims common to the three Q_0 fused-dequant GEMV kernels. `n_rows` sizes the
// output; `n_blocks_per_row` * 32 sizes `x` and the per-row byte length via
// the format-specific `block_bytes` (18 / 22 / 34).
struct DequantGemvDims {
    uint n_rows;
    uint n_blocks_per_row;
};

// Q4_0: 32 elems / block, 18 B (2 B FP16 scale + 16 B nibbles biased +8).
kernel void vokra_dequant_gemv_q4_0_f32(
    device const uchar*      blocks [[buffer(0)]],
    device const float*      x      [[buffer(1)]],
    device float*            y      [[buffer(2)]],
    constant DequantGemvDims& d     [[buffer(3)]],
    uint                     gid    [[thread_position_in_grid]])
{
    const uint row = gid;
    if (row >= d.n_rows) {
        return;
    }
    const uint block_bytes = 18u;
    const uint per_row_bytes = d.n_blocks_per_row * block_bytes;
    const uint row_start = row * per_row_bytes;

    float acc = 0.0f;
    for (uint b = 0; b < d.n_blocks_per_row; ++b) {
        const uint block_off = row_start + b * block_bytes;
        const uint d_bits = (uint)blocks[block_off]
                          | ((uint)blocks[block_off + 1u] << 8u);
        const float dq = vokra_kv_f16_to_f32(d_bits);
        const uint x_base = b * 32u;
        for (uint i = 0; i < 16u; ++i) {
            const uchar byte = blocks[block_off + 2u + i];
            const int lo = (int)(byte & 0x0Fu) - 8;
            const int hi = (int)((byte >> 4) & 0x0Fu) - 8;
            acc += (float)lo * dq * x[x_base + 2u * i];
            acc += (float)hi * dq * x[x_base + 2u * i + 1u];
        }
    }
    y[row] = acc;
}

// Q5_0: 32 elems / block, 22 B (2 B FP16 scale + 4 B qh + 16 B qs low 4 bits).
kernel void vokra_dequant_gemv_q5_0_f32(
    device const uchar*      blocks [[buffer(0)]],
    device const float*      x      [[buffer(1)]],
    device float*            y      [[buffer(2)]],
    constant DequantGemvDims& d     [[buffer(3)]],
    uint                     gid    [[thread_position_in_grid]])
{
    const uint row = gid;
    if (row >= d.n_rows) {
        return;
    }
    const uint block_bytes = 22u;
    const uint per_row_bytes = d.n_blocks_per_row * block_bytes;
    const uint row_start = row * per_row_bytes;

    float acc = 0.0f;
    for (uint b = 0; b < d.n_blocks_per_row; ++b) {
        const uint block_off = row_start + b * block_bytes;
        const uint d_bits = (uint)blocks[block_off]
                          | ((uint)blocks[block_off + 1u] << 8u);
        const float dq = vokra_kv_f16_to_f32(d_bits);
        const uint qh_base = block_off + 2u;
        const uint qs_base = block_off + 6u;
        const uint x_base = b * 32u;
        for (uint i = 0; i < 32u; ++i) {
            const uchar lo4_byte = blocks[qs_base + (i >> 1u)];
            const uint lo4 = ((i & 1u) != 0u)
                                ? ((uint)(lo4_byte >> 4) & 0x0Fu)
                                : ((uint)lo4_byte & 0x0Fu);
            const uchar hi1_byte = blocks[qh_base + (i >> 3u)];
            const uint hi1 = ((uint)hi1_byte >> (i & 7u)) & 0x01u;
            const uint biased = (hi1 << 4u) | lo4;
            const int signed_v = (int)biased - 16;
            acc += (float)signed_v * dq * x[x_base + i];
        }
    }
    y[row] = acc;
}

// Q8_0: 32 elems / block, 34 B (2 B FP16 scale + 32 B i8 qs).
kernel void vokra_dequant_gemv_q8_0_f32(
    device const uchar*      blocks [[buffer(0)]],
    device const float*      x      [[buffer(1)]],
    device float*            y      [[buffer(2)]],
    constant DequantGemvDims& d     [[buffer(3)]],
    uint                     gid    [[thread_position_in_grid]])
{
    const uint row = gid;
    if (row >= d.n_rows) {
        return;
    }
    const uint block_bytes = 34u;
    const uint per_row_bytes = d.n_blocks_per_row * block_bytes;
    const uint row_start = row * per_row_bytes;

    float acc = 0.0f;
    for (uint b = 0; b < d.n_blocks_per_row; ++b) {
        const uint block_off = row_start + b * block_bytes;
        const uint d_bits = (uint)blocks[block_off]
                          | ((uint)blocks[block_off + 1u] << 8u);
        const float dq = vokra_kv_f16_to_f32(d_bits);
        const uint x_base = b * 32u;
        for (uint i = 0; i < 32u; ++i) {
            // uchar `bytes[off]` reinterpreted as signed i8. MSL does not
            // expose an `int8_t` type on buffers; the explicit `>= 128 ? -256`
            // fold is the portable sign-extension pattern for a byte -> int
            // conversion (equivalent to `(int)(int8_t)byte`, no
            // implementation-defined signed shift).
            uint raw = (uint)blocks[block_off + 2u + i];
            int q_ext = (int)raw;
            if (raw >= 128u) {
                q_ext -= 256;
            }
            acc += (float)q_ext * dq * x[x_base + i];
        }
    }
    y[row] = acc;
}

// ---- M4-05/06 Llama-family decode primitives (rms_norm / rope / silu / swiglu)
//
// The device MSL mirrors — and, within the FP32 bound, the numerics of — the
// CPU oracles the CSM / Moshi backbones already run on the Compute seam:
//   * gamma-only RMSNorm  — `vokra_models::voxtral::text_decoder::rms_norm`;
//   * adjacent-pair RoPE  — `vokra_models::csm::rope::rope_apply_adjacent`
//     (torchtune `reshape(..., -1, 2)` convention; Moshi's `interleave=True`
//     is the same pairing);
//   * SiLU                — `vokra_models::voxtral::text_decoder::silu_inplace`;
//   * SwiGLU              — the fused `silu_inplace(gate); hadamard_inplace(gate, up)`.
// The reduction / arithmetic order equals the CPU code, so the only CPU⇔GPU
// difference is the vendor `sqrt` / `sin` / `cos` / `exp` (a few ULP) — far
// inside the NFR-QL-01 FP32 `atol = 0.01`. One thread per row (rms_norm) or
// per element (silu / swiglu), or per `(pair, row)` (rope); the launch guards
// the ragged tail against the grid bound, like every kernel above.

// ---- rms_norm: gamma-only RMSNorm, out[i,c] = x[i,c] * gamma[c] / sqrt(mean(x^2)+eps)
struct RmsNormDims {
    uint  rows;
    uint  cols;
    float eps;
};

kernel void vokra_rms_norm_f32(
    device const float*   inp   [[buffer(0)]],
    device const float*   gamma [[buffer(1)]],
    device float*         out   [[buffer(2)]],
    constant RmsNormDims& d     [[buffer(3)]],
    uint                  gid   [[thread_position_in_grid]])
{
    const uint r = gid;
    if (r >= d.rows) {
        return;
    }
    const uint base = r * d.cols;
    // sum of squares, then 1/sqrt(mean + eps) — the CPU `rms_norm` order.
    float ss = 0.0f;
    for (uint c = 0; c < d.cols; ++c) {
        const float v = inp[base + c];
        ss += v * v;
    }
    const float inv = 1.0f / sqrt(ss / (float)d.cols + d.eps);
    for (uint c = 0; c < d.cols; ++c) {
        out[base + c] = inp[base + c] * inv * gamma[c];
    }
}

// ---- rope: adjacent-pair rotation over [seq_len, head_dim] row-major ----------
// Row `i` rotates pair `j` = (x[2j], x[2j+1]) by angle (pos_offset + i)·inv_freqs[j].
// One thread per (pair, row); `inv_freqs` has head_dim/2 entries (precomputed by
// `llama3_inv_freqs`, so the wavelength-band rescale is already folded in — the
// kernel is scale-agnostic). Out-of-place (out = rotated(inp)); the caller can
// alias out == inp only via distinct buffers (this path uses distinct buffers).
struct RopeDims {
    uint seq_len;
    uint head_dim;
    uint pos_offset;
};

kernel void vokra_rope_adjacent_f32(
    device const float* inp       [[buffer(0)]],
    device const float* inv_freqs [[buffer(1)]],
    device float*       out       [[buffer(2)]],
    constant RopeDims&  d         [[buffer(3)]],
    uint2               gid       [[thread_position_in_grid]])
{
    const uint j = gid.x; // pair index
    const uint i = gid.y; // sequence row
    // `half` is an MSL reserved type name, so the pair count is `n_pairs`.
    const uint n_pairs = d.head_dim / 2u;
    if (i >= d.seq_len || j >= n_pairs) {
        return;
    }
    const uint base = i * d.head_dim;
    const float m = (float)(d.pos_offset + i);
    const float angle = m * inv_freqs[j];
    const float s = sin(angle);
    const float c = cos(angle);
    const float a = inp[base + 2u * j];
    const float b = inp[base + 2u * j + 1u];
    out[base + 2u * j]      = a * c - b * s;
    out[base + 2u * j + 1u] = a * s + b * c;
}

// ---- silu: elementwise x * sigmoid(x) ---------------------------------------
struct SiluDims {
    uint n;
};

kernel void vokra_silu_f32(
    device const float* x   [[buffer(0)]],
    device float*       out [[buffer(1)]],
    constant SiluDims&  d   [[buffer(2)]],
    uint                gid [[thread_position_in_grid]])
{
    const uint i = gid;
    if (i >= d.n) {
        return;
    }
    const float v = x[i];
    const float sig = 1.0f / (1.0f + exp(-v));
    out[i] = v * sig;
}

// ---- swiglu: fused SiLU(gate) * up (the SwiGLU FFN activation) ---------------
// out[i] = (gate[i] * sigmoid(gate[i])) * up[i] — the CPU does silu then the
// Hadamard, so the same (silu-first) product order is reproduced here.
struct SwigluDims {
    uint n;
};

kernel void vokra_swiglu_f32(
    device const float*  gate [[buffer(0)]],
    device const float*  up   [[buffer(1)]],
    device float*        out  [[buffer(2)]],
    constant SwigluDims& d    [[buffer(3)]],
    uint                 gid  [[thread_position_in_grid]])
{
    const uint i = gid;
    if (i >= d.n) {
        return;
    }
    const float g = gate[i];
    const float sig = 1.0f / (1.0f + exp(-g));
    out[i] = (g * sig) * up[i];
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

/// Causal-softmax dims (`setBytes:` index 2). Mirrors the MSL `struct
/// SoftmaxCausalDims`; `q_offset` is the absolute position of query row 0.
#[repr(C)]
#[derive(Clone, Copy)]
struct SoftmaxCausalDims {
    rows: u32,
    cols: u32,
    q_offset: u32,
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

/// M3-04 fused dequant + GEMV dims (`setBytes:` index 3). Mirrors the MSL
/// `struct DequantGemvDims`; `n_blocks_per_row * 32` sizes the FP32 `x`
/// vector and the format-specific block byte count (18 / 22 / 34) sizes each
/// row of the packed byte payload.
#[repr(C)]
#[derive(Clone, Copy)]
struct DequantGemvDims {
    n_rows: u32,
    n_blocks_per_row: u32,
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
/// AddAssignDims`. Shared verbatim by the cc-27 `vokra_mul_f32` /
/// `vokra_copy_f32` kernels, whose operand layout is identical.
#[repr(C)]
#[derive(Clone, Copy)]
struct AddAssignDims {
    n: u32,
}

/// Gamma-only RMSNorm dims (`setBytes:` index 3). The trailing `f32 eps` matches
/// the MSL `struct RmsNormDims` (all fields 4-byte, so `#[repr(C)]` needs no
/// padding).
#[repr(C)]
#[derive(Clone, Copy)]
struct RmsNormDims {
    rows: u32,
    cols: u32,
    eps: f32,
}

/// Adjacent-pair RoPE dims (`setBytes:` index 3). Mirrors the MSL `struct
/// RopeDims`; `pos_offset` is the absolute position of sequence row 0.
#[repr(C)]
#[derive(Clone, Copy)]
struct RopeDims {
    seq_len: u32,
    head_dim: u32,
    pos_offset: u32,
}

/// SiLU dims (`setBytes:` index 2). Mirrors the MSL `struct SiluDims`.
#[repr(C)]
#[derive(Clone, Copy)]
struct SiluDims {
    n: u32,
}

/// SwiGLU dims (`setBytes:` index 3). Mirrors the MSL `struct SwigluDims`.
#[repr(C)]
#[derive(Clone, Copy)]
struct SwigluDims {
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
    /// Whether the softmax over each query row masks the causal future
    /// (`vokra_softmax_causal_f32`); `false` = the plain softmax (encoder
    /// self-attention and decoder cross-attention). Decoder self-attention sets
    /// this `true`.
    causal: bool,
    /// Absolute position of query row 0 (only read when `causal`): row `i`
    /// attends keys `[0, q_offset + i]`. For a steady-state single-token step
    /// this is `t_kv - 1` (all keys visible); for the prefix step it is 0.
    q_offset: usize,
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

/// A device-resident autoregressive self-attention key/value cache — the
/// decoder-step Phase 2 primitive (create with [`MetalContext::new_kv_cache`],
/// grow with [`MetalContext::kv_append`], read with
/// [`MetalContext::kv_download`]).
///
/// Two `[cap_rows, width]` row-major buffers are reserved **once** to the hard
/// `cap_rows` bound (the decoder's `n_text_ctx`); each decode step appends its
/// new `[t, width]` rows by having the k/v-projection GEMM write in place at row
/// `len`, so the cache never reallocates or copies mid-decode — the device
/// analogue of the host [`vokra_core::KvCache`] (same append semantics, same
/// bytes, only the destination is a device buffer at a row offset).
///
/// It owns raw [`OwnedBuf`]s (no `MetalDeviceTensor<'ctx>` borrow), so — like the
/// [`MetalDecodeSession`]'s inline self-KV — it can outlive any single op and be
/// carried across decode steps. `cap`/`len`/`width` are plain `usize`.
pub struct MetalKvCache {
    /// Key rows `[cap_rows, width]`, filled `[0, len)` from row 0 up.
    k: OwnedBuf,
    /// Value rows `[cap_rows, width]`, filled in lockstep with `k`.
    v: OwnedBuf,
    /// Reserved row capacity — the hard bound `kv_append` never exceeds.
    cap_rows: usize,
    /// Width (hidden size) of one cached row.
    width: usize,
    /// Committed rows (positions) currently in the cache.
    len: usize,
}

impl MetalKvCache {
    /// Committed rows (positions) currently in the cache.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether no rows have been appended yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The reserved row capacity (the hard `n_text_ctx` bound, never exceeded).
    #[must_use]
    pub fn capacity_rows(&self) -> usize {
        self.cap_rows
    }

    /// The width (hidden size) of one cached key / value row.
    #[must_use]
    pub fn width(&self) -> usize {
        self.width
    }

    /// Rewinds to empty, keeping the reserved buffers so a fresh decode of the
    /// same audio overwrites from row 0. Mirrors [`vokra_core::KvCache::reset`].
    pub fn reset(&mut self) {
        self.len = 0;
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
    softmax_causal_pipeline: Id,
    layer_norm_pipeline: Id,
    gelu_pipeline: Id,
    conv1d_pipeline: Id,
    col_gather_pipeline: Id,
    col_gather_t_pipeline: Id,
    col_scatter_pipeline: Id,
    add_assign_pipeline: Id,
    /// cc-27 graph-executor element-wise multiply (`dst[i] *= src[i]`).
    mul_pipeline: Id,
    /// cc-27 graph-executor element-wise copy (`dst[i] = src[i]`).
    copy_pipeline: Id,
    /// M3-04 fused KV-cache dequant + GEMV pipelines, one per Q_0 format
    /// (`vokra_dequant_gemv_q4_0_f32` / `_q5_0_f32` / `_q8_0_f32`). Symmetric
    /// with the CUDA `dequant_gemv_q*_0` kernels; each is the GPU
    /// implementation of the [`vokra_core::KvQuantDequantGemvOps`] trait,
    /// whose CPU differential oracle is
    /// [`vokra_core::kv_quant::dequant_gemm::dequant_gemv_scalar`].
    dequant_gemv_q4_0_pipeline: Id,
    dequant_gemv_q5_0_pipeline: Id,
    dequant_gemv_q8_0_pipeline: Id,
    /// M4-05/06 Llama-family decode primitives: gamma-only RMSNorm,
    /// adjacent-pair RoPE, elementwise SiLU, and the fused SwiGLU FFN
    /// activation. Each is the GPU implementation of the matching CSM / Moshi
    /// CPU op (module docs on `KERNELS_MSL`); share the Phase-4/5 library.
    rms_norm_pipeline: Id,
    rope_adjacent_pipeline: Id,
    silu_pipeline: Id,
    swiglu_pipeline: Id,
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
        let softmax_causal_pipeline =
            unsafe { make_pipeline(device, klib.0, c"vokra_softmax_causal_f32") }?;
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
        // cc-27 graph-executor element-wise multiply / copy; same library.
        // SAFETY: as above.
        let mul_pipeline = unsafe { make_pipeline(device, klib.0, c"vokra_mul_f32") }?;
        // SAFETY: as above.
        let copy_pipeline = unsafe { make_pipeline(device, klib.0, c"vokra_copy_f32") }?;
        // M3-04 fused KV-cache dequant + GEMV pipelines, one per Q_0 format;
        // share the same library as every other Phase-4/5 kernel.
        // SAFETY: as above.
        let dequant_gemv_q4_0_pipeline =
            unsafe { make_pipeline(device, klib.0, c"vokra_dequant_gemv_q4_0_f32") }?;
        // SAFETY: as above.
        let dequant_gemv_q5_0_pipeline =
            unsafe { make_pipeline(device, klib.0, c"vokra_dequant_gemv_q5_0_f32") }?;
        // SAFETY: as above.
        let dequant_gemv_q8_0_pipeline =
            unsafe { make_pipeline(device, klib.0, c"vokra_dequant_gemv_q8_0_f32") }?;
        // M4-05/06 Llama-family decode primitives; share the same library.
        // SAFETY: as above.
        let rms_norm_pipeline = unsafe { make_pipeline(device, klib.0, c"vokra_rms_norm_f32") }?;
        // SAFETY: as above.
        let rope_adjacent_pipeline =
            unsafe { make_pipeline(device, klib.0, c"vokra_rope_adjacent_f32") }?;
        // SAFETY: as above.
        let silu_pipeline = unsafe { make_pipeline(device, klib.0, c"vokra_silu_f32") }?;
        // SAFETY: as above.
        let swiglu_pipeline = unsafe { make_pipeline(device, klib.0, c"vokra_swiglu_f32") }?;
        drop(klib);

        Ok(MetalContext {
            device,
            queue: queue.into_raw(),
            gemm_pipeline: gemm_pipeline.into_raw(),
            gemv_pipeline: gemv_pipeline.into_raw(),
            softmax_pipeline: softmax_pipeline.into_raw(),
            softmax_causal_pipeline: softmax_causal_pipeline.into_raw(),
            layer_norm_pipeline: layer_norm_pipeline.into_raw(),
            gelu_pipeline: gelu_pipeline.into_raw(),
            conv1d_pipeline: conv1d_pipeline.into_raw(),
            col_gather_pipeline: col_gather_pipeline.into_raw(),
            col_gather_t_pipeline: col_gather_t_pipeline.into_raw(),
            col_scatter_pipeline: col_scatter_pipeline.into_raw(),
            add_assign_pipeline: add_assign_pipeline.into_raw(),
            mul_pipeline: mul_pipeline.into_raw(),
            copy_pipeline: copy_pipeline.into_raw(),
            dequant_gemv_q4_0_pipeline: dequant_gemv_q4_0_pipeline.into_raw(),
            dequant_gemv_q5_0_pipeline: dequant_gemv_q5_0_pipeline.into_raw(),
            dequant_gemv_q8_0_pipeline: dequant_gemv_q8_0_pipeline.into_raw(),
            rms_norm_pipeline: rms_norm_pipeline.into_raw(),
            rope_adjacent_pipeline: rope_adjacent_pipeline.into_raw(),
            silu_pipeline: silu_pipeline.into_raw(),
            swiglu_pipeline: swiglu_pipeline.into_raw(),
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

    /// Byte-oriented sibling of [`Self::new_buffer_from_slice`] used by the
    /// M3-04 fused dequant GEMV path — the packed KV block payload is a
    /// `&[u8]`, not `&[f32]`, and a mistyped call site here would silently
    /// upload the wrong element count. Kept as its own method for that
    /// reason.
    fn new_buffer_from_bytes(&self, data: &[u8]) -> Result<OwnedBuf> {
        let bytes = data.len().max(size_of::<f32>());
        // SAFETY: `device` is valid; `data.as_ptr()` is valid for `data.len()`
        // bytes (the buffer copies at most `bytes >= data.len()`; the tail
        // padding is unread by the kernel); shared storage mode (0).
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
                "MTLDevice newBufferWithBytes (u8) returned nil".to_owned(),
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

    // ---- M3-04 fused KV-cache dequant + GEMV ------------------------------

    /// GPU-side fused dequantisation + row-wise GEMV over a quantised KV block
    /// matrix — the Metal implementation of the
    /// [`KvQuantDequantGemvOps`](vokra_core::KvQuantDequantGemvOps) seam
    /// (M3-04-T10).
    ///
    /// The GPU kernel dequantises one 32-elem block at a time *inside* the
    /// per-row GEMV loop, so the intermediate FP32 row is never materialised
    /// (unlike the two-stage `dequantize_bytes → dense_gemv_f32` reference).
    /// Byte layout is identical to the CPU differential oracle
    /// [`vokra_core::kv_quant::dequant_gemm::dequant_gemv_scalar`], so both
    /// paths consume the same on-wire payload.
    ///
    /// # Precision
    ///
    /// Output matches the CPU oracle within the FP32 GEMV rounding bound. The
    /// backend parity test pins this to `atol = 1e-4`.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on shape mismatch or `mode ==
    /// KvQuant::Fp32`; [`VokraError::BackendUnavailable`] on a Metal
    /// allocation / command-buffer failure.
    pub fn dequant_gemv_f32(
        &self,
        mode: vokra_core::KvQuant,
        blocks_bytes: &[u8],
        n_rows: usize,
        n_blocks_per_row: usize,
        x: &[f32],
    ) -> Result<Vec<f32>> {
        vokra_core::validate_dequant_gemv(mode, blocks_bytes, n_rows, n_blocks_per_row, x)?;
        if n_rows == 0 {
            return Ok(Vec::new());
        }
        // SAFETY: token consumed by the matching pop below.
        let pool = unsafe { sys::objc_autoreleasePoolPush() };
        let r = self.run_dequant_gemv(mode, blocks_bytes, n_rows, n_blocks_per_row, x);
        // SAFETY: `pool` is the token from the push above.
        unsafe { sys::objc_autoreleasePoolPop(pool) };
        r
    }

    fn run_dequant_gemv(
        &self,
        mode: vokra_core::KvQuant,
        blocks_bytes: &[u8],
        n_rows: usize,
        n_blocks_per_row: usize,
        x: &[f32],
    ) -> Result<Vec<f32>> {
        let (pipeline, label) = match mode {
            vokra_core::KvQuant::Q4_0 => (self.dequant_gemv_q4_0_pipeline, "dequant_gemv_q4_0"),
            vokra_core::KvQuant::Q5_0 => (self.dequant_gemv_q5_0_pipeline, "dequant_gemv_q5_0"),
            vokra_core::KvQuant::Q8_0 => (self.dequant_gemv_q8_0_pipeline, "dequant_gemv_q8_0"),
            vokra_core::KvQuant::Fp32 => {
                // Guarded by `validate_dequant_gemv`; keep as an explicit error
                // (never a silent fallback, FR-EX-08).
                return Err(VokraError::InvalidArgument(
                    "dequant_gemv_f32: mode=Fp32 rejected".to_owned(),
                ));
            }
        };

        let blocks_buf = self.new_buffer_from_bytes(blocks_bytes)?;
        let x_buf = self.new_buffer_from_slice(x)?;
        let out_buf = self.new_buffer_output(n_rows)?;
        let dims = DequantGemvDims {
            n_rows: n_rows as u32,
            n_blocks_per_row: n_blocks_per_row as u32,
        };
        let (grid, tg) = grid_1d(n_rows);
        self.dispatch_compute(
            pipeline,
            &[&blocks_buf, &x_buf, &out_buf],
            (&dims as *const DequantGemvDims).cast::<c_void>(),
            size_of::<DequantGemvDims>(),
            grid,
            tg,
            label,
        )?;
        let mut out = vec![0.0f32; n_rows];
        read_back(&out_buf, &mut out)?;
        Ok(out)
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

    /// Row-wise **causal** softmax over a `rows × cols` buffer: row `r` (query at
    /// absolute position `q_offset + r`) normalises over the visible key prefix
    /// `[0, q_offset + r]` and writes `0.0` for future columns — bit-identical to
    /// writing `-inf` into those columns and running [`Self::softmax_f32`] (see
    /// the `vokra_softmax_causal_f32` kernel proof). The decode-step primitive;
    /// exposed so the causal fused attention is unit-testable against the
    /// host-mask + plain-softmax reference.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on a shape mismatch;
    /// [`VokraError::BackendUnavailable`] on a Metal failure.
    pub fn softmax_causal_f32(
        &self,
        input: &[f32],
        out: &mut [f32],
        rows: usize,
        cols: usize,
        q_offset: usize,
    ) -> Result<()> {
        validate_rows_cols(input, out, rows, cols)?;
        if out.is_empty() {
            return Ok(());
        }
        // SAFETY: token consumed by the matching pop below.
        let pool = unsafe { sys::objc_autoreleasePoolPush() };
        let r = self.run_softmax_causal(input, out, rows, cols, q_offset);
        // SAFETY: `pool` is the token from the push above.
        unsafe { sys::objc_autoreleasePoolPop(pool) };
        r
    }

    fn run_softmax_causal(
        &self,
        input: &[f32],
        out: &mut [f32],
        rows: usize,
        cols: usize,
        q_offset: usize,
    ) -> Result<()> {
        let in_buf = self.new_buffer_from_slice(input)?;
        let out_buf = self.new_buffer_output(out.len())?;
        let dims = SoftmaxCausalDims {
            rows: rows as u32,
            cols: cols as u32,
            q_offset: q_offset as u32,
        };
        let (grid, tg) = grid_1d(rows);
        self.dispatch_compute(
            self.softmax_causal_pipeline,
            &[&in_buf, &out_buf],
            (&dims as *const SoftmaxCausalDims).cast::<c_void>(),
            size_of::<SoftmaxCausalDims>(),
            grid,
            tg,
            "softmax_causal",
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

    // ---- M4-05/06 Llama-family decode primitives (rms_norm / rope / silu /
    // swiglu). Each mirrors the CSM / Moshi CPU op contract and numerics (FP32,
    // `atol = 0.01`), brackets the GPU work in an autorelease pool, and reads
    // back copy-free from shared storage — exactly like the Phase-4 kernels.

    /// Gamma-only RMSNorm applied row-wise:
    /// `out[i, c] = x[i, c] · gamma[c] / sqrt(mean_c(x[i, c]²) + eps)`. Distinct
    /// from the affine, mean-subtracting [`Self::layer_norm_f32`]: this is the
    /// CSM / Moshi `rms_norm` (gamma only, no bias, no mean subtraction).
    ///
    /// `input` / `out` are `rows × cols`; `gamma` has length `cols`.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on a shape mismatch;
    /// [`VokraError::BackendUnavailable`] on a Metal failure.
    pub fn rms_norm_f32(
        &self,
        input: &[f32],
        out: &mut [f32],
        rows: usize,
        cols: usize,
        gamma: &[f32],
        eps: f32,
    ) -> Result<()> {
        validate_rms_norm(input, out, rows, cols, gamma)?;
        if out.is_empty() {
            return Ok(());
        }
        // SAFETY: token consumed by the matching pop below.
        let pool = unsafe { sys::objc_autoreleasePoolPush() };
        let r = self.run_rms_norm(input, out, rows, cols, gamma, eps);
        // SAFETY: `pool` is the token from the push above.
        unsafe { sys::objc_autoreleasePoolPop(pool) };
        r
    }

    fn run_rms_norm(
        &self,
        input: &[f32],
        out: &mut [f32],
        rows: usize,
        cols: usize,
        gamma: &[f32],
        eps: f32,
    ) -> Result<()> {
        let in_buf = self.new_buffer_from_slice(input)?;
        let gamma_buf = self.new_buffer_from_slice(gamma)?;
        let out_buf = self.new_buffer_output(out.len())?;
        let dims = RmsNormDims {
            rows: rows as u32,
            cols: cols as u32,
            eps,
        };
        let (grid, tg) = grid_1d(rows);
        self.dispatch_compute(
            self.rms_norm_pipeline,
            &[&in_buf, &gamma_buf, &out_buf],
            (&dims as *const RmsNormDims).cast::<c_void>(),
            size_of::<RmsNormDims>(),
            grid,
            tg,
            "rms_norm",
        )?;
        read_back(&out_buf, out)
    }

    /// Adjacent-pair RoPE over `input = [seq_len, head_dim]` row-major, writing
    /// the rotated tensor to `out` (same shape). Row `i` rotates each pair
    /// `(x[2j], x[2j+1])` by angle `(pos_offset + i) · inv_freqs[j]`; `inv_freqs`
    /// has `head_dim / 2` entries (precomputed by `llama3_inv_freqs`, so the
    /// Llama-3 wavelength-band rescale is already folded in). The exact contract
    /// of `vokra_models::csm::rope::rope_apply_adjacent` (out-of-place form).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on an odd `head_dim` or a shape mismatch;
    /// [`VokraError::BackendUnavailable`] on a Metal failure.
    pub fn rope_adjacent_f32(
        &self,
        input: &[f32],
        out: &mut [f32],
        seq_len: usize,
        head_dim: usize,
        inv_freqs: &[f32],
        pos_offset: usize,
    ) -> Result<()> {
        validate_rope(input, out, seq_len, head_dim, inv_freqs)?;
        if out.is_empty() {
            return Ok(());
        }
        // SAFETY: token consumed by the matching pop below.
        let pool = unsafe { sys::objc_autoreleasePoolPush() };
        let r = self.run_rope_adjacent(input, out, seq_len, head_dim, inv_freqs, pos_offset);
        // SAFETY: `pool` is the token from the push above.
        unsafe { sys::objc_autoreleasePoolPop(pool) };
        r
    }

    #[allow(clippy::too_many_arguments)] // intrinsic RoPE parameter set (matches CPU rope_apply_adjacent)
    fn run_rope_adjacent(
        &self,
        input: &[f32],
        out: &mut [f32],
        seq_len: usize,
        head_dim: usize,
        inv_freqs: &[f32],
        pos_offset: usize,
    ) -> Result<()> {
        let in_buf = self.new_buffer_from_slice(input)?;
        let freq_buf = self.new_buffer_from_slice(inv_freqs)?;
        let out_buf = self.new_buffer_output(out.len())?;
        let dims = RopeDims {
            seq_len: seq_len as u32,
            head_dim: head_dim as u32,
            pos_offset: pos_offset as u32,
        };
        // One thread per (pair, row): grid.x = head_dim/2 pairs, grid.y = rows.
        let (grid, tg) = grid_2d(head_dim / 2, seq_len);
        self.dispatch_compute(
            self.rope_adjacent_pipeline,
            &[&in_buf, &freq_buf, &out_buf],
            (&dims as *const RopeDims).cast::<c_void>(),
            size_of::<RopeDims>(),
            grid,
            tg,
            "rope_adjacent",
        )?;
        read_back(&out_buf, out)
    }

    /// Element-wise SiLU (`x` and `out` equal length): `out = x · sigmoid(x)` —
    /// the contract of `vokra_models::voxtral::text_decoder::silu_inplace`.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on a length mismatch;
    /// [`VokraError::BackendUnavailable`] on a Metal failure.
    pub fn silu_f32(&self, x: &[f32], out: &mut [f32]) -> Result<()> {
        validate_unary(x, out)?;
        if out.is_empty() {
            return Ok(());
        }
        // SAFETY: token consumed by the matching pop below.
        let pool = unsafe { sys::objc_autoreleasePoolPush() };
        let r = self.run_silu(x, out);
        // SAFETY: `pool` is the token from the push above.
        unsafe { sys::objc_autoreleasePoolPop(pool) };
        r
    }

    fn run_silu(&self, x: &[f32], out: &mut [f32]) -> Result<()> {
        let x_buf = self.new_buffer_from_slice(x)?;
        let out_buf = self.new_buffer_output(out.len())?;
        let dims = SiluDims {
            n: out.len() as u32,
        };
        let (grid, tg) = grid_1d(out.len());
        self.dispatch_compute(
            self.silu_pipeline,
            &[&x_buf, &out_buf],
            (&dims as *const SiluDims).cast::<c_void>(),
            size_of::<SiluDims>(),
            grid,
            tg,
            "silu",
        )?;
        read_back(&out_buf, out)
    }

    /// Fused SwiGLU FFN activation: `out[i] = (gate[i] · sigmoid(gate[i])) ·
    /// up[i]` — the fused `silu_inplace(gate); hadamard_inplace(gate, up)` the
    /// CSM / Moshi FFN runs. `gate`, `up`, `out` share one length.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on a length mismatch;
    /// [`VokraError::BackendUnavailable`] on a Metal failure.
    pub fn swiglu_f32(&self, gate: &[f32], up: &[f32], out: &mut [f32]) -> Result<()> {
        validate_swiglu(gate, up, out)?;
        if out.is_empty() {
            return Ok(());
        }
        // SAFETY: token consumed by the matching pop below.
        let pool = unsafe { sys::objc_autoreleasePoolPush() };
        let r = self.run_swiglu(gate, up, out);
        // SAFETY: `pool` is the token from the push above.
        unsafe { sys::objc_autoreleasePoolPop(pool) };
        r
    }

    fn run_swiglu(&self, gate: &[f32], up: &[f32], out: &mut [f32]) -> Result<()> {
        let gate_buf = self.new_buffer_from_slice(gate)?;
        let up_buf = self.new_buffer_from_slice(up)?;
        let out_buf = self.new_buffer_output(out.len())?;
        let dims = SwigluDims {
            n: out.len() as u32,
        };
        let (grid, tg) = grid_1d(out.len());
        self.dispatch_compute(
            self.swiglu_pipeline,
            &[&gate_buf, &up_buf, &out_buf],
            (&dims as *const SwigluDims).cast::<c_void>(),
            size_of::<SwigluDims>(),
            grid,
            tg,
            "swiglu",
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
            t_q, t_kv, d, n_head, xq, q_w, q_bias, k, v, out_w, out_bias, scale, false, 0, out,
        );
        // SAFETY: `pool` is the token from the push above.
        unsafe { sys::objc_autoreleasePoolPop(pool) };
        r
    }

    /// Fused **causal** multi-head attention (host-in/out) — the decoder
    /// self-attention sibling of [`Self::attn_f32`]. Query row `i` (absolute
    /// position `q_offset + i`) attends keys `[0, q_offset + i]`; the causal mask
    /// is fused into the softmax (`vokra_softmax_causal_f32`), so this is
    /// bit-identical to writing `-inf` into the future scores and running the
    /// plain fused attention. Every other pass is shared with [`Self::attn_f32`],
    /// so the two chains are single-sourced. Used by the decode-step parity tests
    /// and (via [`Self::encode_attn_passes`]) by [`MetalDecodeSession`].
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on any shape mismatch or `d % n_head != 0`;
    /// [`VokraError::BackendUnavailable`] on a Metal failure.
    #[allow(clippy::too_many_arguments)] // fused-attention operand set (two Linears + K/V + dims)
    pub fn attn_causal_f32(
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
        q_offset: usize,
        out: &mut [f32],
    ) -> Result<()> {
        validate_attn(
            t_q, t_kv, d, n_head, xq, q_w, q_bias, k, v, out_w, out_bias, out,
        )?;
        // SAFETY: `objc_autoreleasePoolPush` returns a token consumed by the one
        // matching pop below.
        let pool = unsafe { sys::objc_autoreleasePoolPush() };
        let r = self.run_attn(
            t_q, t_kv, d, n_head, xq, q_w, q_bias, k, v, out_w, out_bias, scale, true, q_offset,
            out,
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
        causal: bool,
        q_offset: usize,
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
                causal,
                q_offset,
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
            // probs = softmax_rows(scores). Causal decoder self-attention masks
            // the future in the fused `vokra_softmax_causal_f32` (the ONLY pass
            // that differs from the non-causal chain); everything else — gather,
            // transpose, both GEMMs, scatter — is byte-for-byte identical, so the
            // numerics stay single-sourced. The dims locals are copied eagerly by
            // `setBytes:`, so they need not outlive this pass.
            let (sm_grid, sm_tg) = grid_1d(t_q);
            if dims.causal {
                let smc_dims = SoftmaxCausalDims {
                    rows: t_q as u32,
                    cols: t_kv as u32,
                    q_offset: dims.q_offset as u32,
                };
                self.encode_pass(
                    cmd,
                    self.softmax_causal_pipeline,
                    &[bufs.scores, bufs.probs],
                    (&smc_dims as *const SoftmaxCausalDims).cast::<c_void>(),
                    size_of::<SoftmaxCausalDims>(),
                    sm_grid,
                    sm_tg,
                    "attn softmax causal",
                )?;
            } else {
                let sm_dims = SoftmaxDims {
                    rows: t_q as u32,
                    cols: t_kv as u32,
                };
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
            }
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

    /// Device-in/out in-place element-wise multiply (one self-contained
    /// submission): `dst[i] *= src[i]` (cc-27). The GPU half of the
    /// graph-executor's [`OpKind::Mul`](vokra_core::OpKind::Mul), shaped
    /// exactly like [`Self::residual_add_dev`] so the two `eval_op` arms are
    /// operand-for-operand mirrors.
    ///
    /// One FP32 multiply per element, so the result carries the same single
    /// rounding as the CPU `kernels::mul_f32` (measured bit-identical over
    /// normal-range operands — see the kernel comment for the denormal caveat).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] if the lengths differ;
    /// [`VokraError::BackendUnavailable`] on a Metal failure.
    pub fn mul_dev(
        &self,
        dst: &mut MetalDeviceTensor<'_>,
        src: &MetalDeviceTensor<'_>,
    ) -> Result<()> {
        expect_len("mul_dev src", src.len, dst.len)?;
        if dst.len == 0 {
            return Ok(());
        }
        let n = dst.len;
        self.pooled(|| {
            let cmd = self.new_command_buffer("mul_dev")?;
            self.encode_elementwise(cmd, self.mul_pipeline, &dst.buf, &src.buf, n, "mul_dev")?;
            self.commit_and_wait(cmd, "mul_dev")
        })
    }

    /// Device-in/out element-wise copy (one self-contained submission):
    /// `dst[i] = src[i]` (cc-27). The GPU half of the graph-executor's
    /// [`OpKind::Copy`](vokra_core::OpKind::Copy).
    ///
    /// A real compute dispatch, not a host memcpy: `Copy` on the Metal graph
    /// arm executes on the device exactly as it does on Vulkan / WebGPU.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] if the lengths differ;
    /// [`VokraError::BackendUnavailable`] on a Metal failure.
    pub fn copy_dev(
        &self,
        dst: &mut MetalDeviceTensor<'_>,
        src: &MetalDeviceTensor<'_>,
    ) -> Result<()> {
        expect_len("copy_dev src", src.len, dst.len)?;
        if dst.len == 0 {
            return Ok(());
        }
        let n = dst.len;
        self.pooled(|| {
            let cmd = self.new_command_buffer("copy_dev")?;
            self.encode_elementwise(cmd, self.copy_pipeline, &dst.buf, &src.buf, n, "copy_dev")?;
            self.commit_and_wait(cmd, "copy_dev")
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
                    causal: false,
                    q_offset: 0,
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

    /// Device-in/out row-major GEMM writing its `[m, n]` output at **row**
    /// `out_row_offset` of `out` (one self-contained submission):
    /// `out[out_row_offset + i, j] = bias?[j] + Σ_l a[i,l]·b[l,j]`. The
    /// device-resident KV-cache append primitive — the k/v-proj GEMM writes the
    /// step's new `[t, d]` rows directly into the resident `[n_text_ctx, d]` cache
    /// at row `start`, so no separate copy is needed. Bit-identical to a plain
    /// GEMM into a fresh `[m, n]` buffer (same kernel, same order, only the
    /// destination is a byte offset into a bigger buffer).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on a shape mismatch or if the offset region
    /// `[out_row_offset, out_row_offset + m)` exceeds `out`'s rows;
    /// [`VokraError::BackendUnavailable`] on a Metal failure.
    #[allow(clippy::too_many_arguments)] // intrinsic GEMM parameter set + the output row offset
    pub fn gemm_dev(
        &self,
        out: &mut MetalDeviceTensor<'_>,
        out_row_offset: usize,
        a: &MetalDeviceTensor<'_>,
        b: &MetalDeviceTensor<'_>,
        bias: Option<&MetalDeviceTensor<'_>>,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        if m == 0 || n == 0 || k == 0 {
            return Err(VokraError::InvalidArgument(
                "gemm_dev dimensions m, n, k must all be >= 1".to_owned(),
            ));
        }
        expect_len("gemm_dev a", a.len, checked_mul(m, k, "gemm_dev m*k")?)?;
        expect_len("gemm_dev b", b.len, checked_mul(k, n, "gemm_dev k*n")?)?;
        // The written region ends at row (out_row_offset + m); it must fit `out`.
        let end_rows = out_row_offset.checked_add(m).ok_or_else(|| {
            VokraError::InvalidArgument("gemm_dev row offset overflow".to_owned())
        })?;
        let need = checked_mul(end_rows, n, "gemm_dev (offset+m)*n")?;
        if out.len < need {
            return Err(VokraError::InvalidArgument(format!(
                "gemm_dev out holds {} f32 but the offset write needs {need}",
                out.len
            )));
        }
        if let Some(bs) = bias {
            expect_len("gemm_dev bias", bs.len, n)?;
        }
        self.pooled(|| {
            let dummy = self.new_buffer_from_slice(&[0.0f32])?;
            let cmd = self.new_command_buffer("gemm_dev")?;
            self.encode_gemm_off(
                cmd,
                &a.buf,
                &b.buf,
                bias_or_dummy(bias, &dummy),
                &out.buf,
                out_row_offset * n,
                m,
                n,
                k,
                bias.is_some(),
            )?;
            self.commit_and_wait(cmd, "gemm_dev")
        })
    }

    // ---- Decoder-step Phase 2: device-resident self-attention K/V cache ------

    /// Reserves a device-resident autoregressive self-attention K/V cache
    /// ([`MetalKvCache`]): two `[cap_rows, width]` buffers allocated **once** to
    /// the hard `cap_rows` bound (the decoder's `n_text_ctx`), starting empty.
    ///
    /// This is the decode-step Phase 2 primitive: a growable-by-append device KV
    /// cache whose rows are written in place by the k/v-projection GEMM (see
    /// [`Self::kv_append`]), matching the host [`vokra_core::KvCache`] semantics
    /// on the GPU without any per-step reallocation or copy. The
    /// **cross**-attention encoder K/V, being fixed, is uploaded once with
    /// [`Self::upload`] instead — it needs no reserve/append.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] if `cap_rows` or `width` is zero;
    /// [`VokraError::BackendUnavailable`] if a buffer cannot be created.
    pub fn new_kv_cache(&self, cap_rows: usize, width: usize) -> Result<MetalKvCache> {
        if cap_rows == 0 || width == 0 {
            return Err(VokraError::InvalidArgument(
                "kv cache cap_rows and width must both be >= 1".to_owned(),
            ));
        }
        let cap = checked_mul(cap_rows, width, "kv cache cap_rows*width")?;
        Ok(MetalKvCache {
            k: self.new_buffer_output(cap)?,
            v: self.new_buffer_output(cap)?,
            cap_rows,
            width,
            len: 0,
        })
    }

    /// Appends one decode step's `t` new rows to `cache`, projected from the
    /// device-resident `x` `[t, d]` by the key / value weight matrices
    /// `k_w` / `v_w` `[d, width]` (+ optional `[width]` bias): the two projection
    /// GEMMs write their `[t, width]` outputs **in place at row `cache.len`** of
    /// the resident K / V buffers within **one** command buffer, then the
    /// committed length advances by `t`.
    ///
    /// This is **bit-identical** to a host `project_kv` + [`vokra_core::KvCache`]
    /// `append`: the very same GEMM kernel and operands, the only difference being
    /// that the destination is a resident device buffer at a row byte-offset
    /// (`cache.len * width * 4`) rather than a fresh host buffer — exactly the
    /// offset write [`Self::gemm_dev`] proves. Reserve is a hard bound: appending
    /// past `cache.capacity_rows()` is an explicit error, never a realloc.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on a zero `t`/`d`, an operand-shape
    /// mismatch, or an append that would exceed the reserved capacity;
    /// [`VokraError::BackendUnavailable`] on a Metal failure.
    #[allow(clippy::too_many_arguments)] // k/v projection operand set (x + two weights + biases)
    pub fn kv_append(
        &self,
        cache: &mut MetalKvCache,
        t: usize,
        d: usize,
        x: &MetalDeviceTensor<'_>,
        k_w: &MetalDeviceTensor<'_>,
        k_bias: Option<&MetalDeviceTensor<'_>>,
        v_w: &MetalDeviceTensor<'_>,
        v_bias: Option<&MetalDeviceTensor<'_>>,
    ) -> Result<()> {
        if t == 0 || d == 0 {
            return Err(VokraError::InvalidArgument(
                "kv_append t and d must both be >= 1".to_owned(),
            ));
        }
        let width = cache.width;
        expect_len("kv_append x", x.len, checked_mul(t, d, "kv_append t*d")?)?;
        let dw = checked_mul(d, width, "kv_append d*width")?;
        expect_len("kv_append k_w", k_w.len, dw)?;
        expect_len("kv_append v_w", v_w.len, dw)?;
        if let Some(b) = k_bias {
            expect_len("kv_append k_bias", b.len, width)?;
        }
        if let Some(b) = v_bias {
            expect_len("kv_append v_bias", b.len, width)?;
        }
        // The new rows [len, len + t) must fit the reserved capacity (a hard
        // bound: a device cache cannot grow mid-command-buffer).
        let end = cache
            .len
            .checked_add(t)
            .ok_or_else(|| VokraError::InvalidArgument("kv_append position overflow".to_owned()))?;
        if end > cache.cap_rows {
            return Err(VokraError::InvalidArgument(format!(
                "kv_append: appending {t} rows at row {} exceeds the reserved capacity of {} rows",
                cache.len, cache.cap_rows
            )));
        }
        let off = checked_mul(cache.len, width, "kv_append len*width")?;
        self.pooled(|| {
            let dummy = self.new_buffer_from_slice(&[0.0f32])?;
            let cmd = self.new_command_buffer("kv_append")?;
            // K = x[t,d] @ k_w[d,width] (+k_bias) written at row `len`.
            self.encode_gemm_off(
                cmd,
                &x.buf,
                &k_w.buf,
                bias_or_dummy(k_bias, &dummy),
                &cache.k,
                off,
                t,
                width,
                d,
                k_bias.is_some(),
            )?;
            // V = x[t,d] @ v_w[d,width] (+v_bias) written at the same row `len`.
            self.encode_gemm_off(
                cmd,
                &x.buf,
                &v_w.buf,
                bias_or_dummy(v_bias, &dummy),
                &cache.v,
                off,
                t,
                width,
                d,
                v_bias.is_some(),
            )?;
            self.commit_and_wait(cmd, "kv_append")
        })?;
        cache.len = end;
        Ok(())
    }

    /// Reads the committed `[len, width]` key and value rows back into host
    /// buffers (`k_out` / `v_out`, each `len * width` f32). Appended rows occupy
    /// the front of the reserved buffers (growth is from row 0), so this is a
    /// prefix copy; call after the last [`Self::kv_append`] (which waits, so the
    /// rows are readable immediately).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] if either output length differs from
    /// `cache.len() * cache.width()`; [`VokraError::BackendUnavailable`] on a null
    /// contents pointer.
    pub fn kv_download(
        &self,
        cache: &MetalKvCache,
        k_out: &mut [f32],
        v_out: &mut [f32],
    ) -> Result<()> {
        let committed = checked_mul(cache.len, cache.width, "kv_download len*width")?;
        expect_len("kv_download k_out", k_out.len(), committed)?;
        expect_len("kv_download v_out", v_out.len(), committed)?;
        if committed == 0 {
            return Ok(());
        }
        read_back(&cache.k, k_out)?;
        read_back(&cache.v, v_out)
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
                    causal: false,
                    q_offset: 0,
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

    /// Encodes a GEMM pass whose `[m, n]` output is written at element offset
    /// `out_off` in `out` (the destination buffer bound at byte offset
    /// `out_off·4`). Used by the decode-step KV-cache append: the k/v-proj GEMM
    /// writes the step's new rows directly at cache row `start` (`out_off =
    /// start·d`). `a`/`b`/`bias` are bound at offset 0. Same kernel / geometry as
    /// [`Self::encode_gemm`]; the only difference is the output offset.
    #[allow(clippy::too_many_arguments)] // intrinsic GEMM parameter set + the output offset
    fn encode_gemm_off(
        &self,
        cmd: Id,
        a: &OwnedBuf,
        b: &OwnedBuf,
        bias: &OwnedBuf,
        out: &OwnedBuf,
        out_off: usize,
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
        self.encode_pass_off(
            cmd,
            self.gemm_pipeline,
            &[a, b, bias, out],
            Some(&[0, 0, 0, out_off * size_of::<f32>()]),
            (&dims as *const GemmDims).cast::<c_void>(),
            size_of::<GemmDims>(),
            grid,
            tg,
            "decode gemm@offset",
        )
    }

    /// Encodes a matrix-vector pass whose input vector `x` starts at element
    /// offset `x_off` in its buffer and whose `[m]` output is written at
    /// element offset `out_off` in `out`: `out[out_off + i] = Σ_l a[i·k + l]·
    /// x[x_off + l]` (bias-less). Used by the decode-step tied-logits head:
    /// the driver invokes this once per decoded row (`x_off = i·d`,
    /// `out_off = i·n_vocab`), so ALL `[t, n_vocab]` rows are produced in ONE
    /// command buffer while each row remains a plain per-row reduction (the
    /// same math the CPU [`project_logits_into`]'s `t == 1` fast path runs on
    /// its single row).
    ///
    /// [`project_logits_into`]: crate (whisper decoder)
    #[allow(clippy::too_many_arguments)] // gemv operand set + I/O offsets (Phase-3 decode head)
    fn encode_gemv_off(
        &self,
        cmd: Id,
        a: &OwnedBuf,
        x: &OwnedBuf,
        x_off: usize,
        out: &OwnedBuf,
        out_off: usize,
        m: usize,
        k: usize,
    ) -> Result<()> {
        let dims = GemvDims {
            m: m as u32,
            k: k as u32,
            has_bias: 0,
        };
        let (grid, tg) = grid_1d(m);
        self.encode_pass_off(
            cmd,
            self.gemv_pipeline,
            &[a, x, a, out], // bias buffer is unused (has_bias = 0); bind `a` as a valid dummy
            Some(&[0, x_off * size_of::<f32>(), 0, out_off * size_of::<f32>()]),
            (&dims as *const GemvDims).cast::<c_void>(),
            size_of::<GemvDims>(),
            grid,
            tg,
            "decode logits gemv@offset",
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

    /// Encodes one two-operand element-wise pass (`dst`, `src`, `{n}`) for the
    /// cc-27 `Mul` / `Copy` kernels. Both share `AddAssignDims` and the
    /// `residual_add` binding layout, so the only per-op difference is which
    /// pipeline is bound — hence one encoder parameterised by `pipeline`.
    fn encode_elementwise(
        &self,
        cmd: Id,
        pipeline: Id,
        dst: &OwnedBuf,
        src: &OwnedBuf,
        n: usize,
        label: &str,
    ) -> Result<()> {
        let dims = AddAssignDims { n: n as u32 };
        let (grid, tg) = grid_1d(n);
        self.encode_pass(
            cmd,
            pipeline,
            &[dst, src],
            (&dims as *const AddAssignDims).cast::<c_void>(),
            size_of::<AddAssignDims>(),
            grid,
            tg,
            label,
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
        self.encode_pass_off(cmd, pipeline, buffers, None, dims, dims_len, grid, tg, what)
    }

    /// Like [`Self::encode_pass`] but binds each buffer at an explicit **byte**
    /// offset (`offsets[i]`, or `0` for every buffer when `offsets` is `None`).
    /// The device-resident KV-cache append binds the k/v-proj GEMM output at the
    /// cache row `start` (`offset = start·d·4`), and the tied-logits gemv binds
    /// its input at the last decoded row — both a plain `setBuffer:offset:` on a
    /// buffer the caller sized to hold the offset region. `offsets`, when `Some`,
    /// must be exactly `buffers.len()` long.
    #[allow(clippy::too_many_arguments)] // cmd + pipeline + buffers + offsets + dims + grid/tg + label
    fn encode_pass_off(
        &self,
        cmd: Id,
        pipeline: Id,
        buffers: &[&OwnedBuf],
        offsets: Option<&[usize]>,
        dims: *const c_void,
        dims_len: usize,
        grid: MtlSize,
        tg: MtlSize,
        what: &str,
    ) -> Result<()> {
        debug_assert!(
            offsets.is_none_or(|o| o.len() == buffers.len()),
            "encode_pass_off: offsets length must match buffers length"
        );
        // SAFETY: `cmd` is a valid command buffer from this context's queue;
        // `computeCommandEncoder` returns an autoreleased encoder (drained by the
        // caller's pool); `pipeline` is one of the context's compiled pipelines;
        // each `buffers[i]` is a valid MTLBuffer bound at index `i` with byte
        // offset `offsets[i]` (0 when `None`), which the caller guarantees lies
        // within that buffer's length; `dims` points to `dims_len` readable bytes
        // matching the kernel's `constant` struct at index `buffers.len()`; the
        // two `MtlSize`s are passed per AAPCS64.
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
                let off = offsets.map_or(0, |o| o[i]);
                sys::send_set_buffer(enc, set_buffer, buf.0, off, i);
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
            release(self.swiglu_pipeline);
            release(self.silu_pipeline);
            release(self.rope_adjacent_pipeline);
            release(self.rms_norm_pipeline);
            release(self.dequant_gemv_q8_0_pipeline);
            release(self.dequant_gemv_q5_0_pipeline);
            release(self.dequant_gemv_q4_0_pipeline);
            release(self.copy_pipeline);
            release(self.mul_pipeline);
            release(self.add_assign_pipeline);
            release(self.col_scatter_pipeline);
            release(self.col_gather_t_pipeline);
            release(self.col_gather_pipeline);
            release(self.conv1d_pipeline);
            release(self.gelu_pipeline);
            release(self.layer_norm_pipeline);
            release(self.softmax_causal_pipeline);
            release(self.softmax_pipeline);
            release(self.gemv_pipeline);
            release(self.gemm_pipeline);
            release(self.queue);
            release(self.device);
        }
    }
}

// ---- Phase-5 decoder-step: device-resident autoregressive decode session -----

/// One decoder layer's device-resident weights + KV cache for
/// [`MetalDecodeSession`]. All buffers are `OwnedBuf` (no lifetime), uploaded /
/// reserved once in [`MetalDecodeSession::new`] and reused for every decode step.
/// Absent biases (Whisper's `k_proj`) stay `None` and bind the session's shared
/// dummy at encode time.
struct DevDecoderLayer {
    self_ln_g: OwnedBuf,
    self_ln_b: OwnedBuf,
    self_q_w: OwnedBuf,
    self_q_bias: Option<OwnedBuf>,
    self_k_w: OwnedBuf,
    self_k_bias: Option<OwnedBuf>,
    self_v_w: OwnedBuf,
    self_v_bias: Option<OwnedBuf>,
    self_out_w: OwnedBuf,
    self_out_bias: Option<OwnedBuf>,
    cross_ln_g: OwnedBuf,
    cross_ln_b: OwnedBuf,
    cross_q_w: OwnedBuf,
    cross_q_bias: Option<OwnedBuf>,
    cross_out_w: OwnedBuf,
    cross_out_bias: Option<OwnedBuf>,
    /// Pre-projected cross-attention keys `[n_ctx, d]`, resident (uploaded once).
    cross_k: OwnedBuf,
    /// Pre-projected cross-attention values `[n_ctx, d]`, resident.
    cross_v: OwnedBuf,
    mlp_ln_g: OwnedBuf,
    mlp_ln_b: OwnedBuf,
    fc1_w: OwnedBuf,
    fc1_bias: Option<OwnedBuf>,
    fc2_w: OwnedBuf,
    fc2_bias: Option<OwnedBuf>,
    /// Resident self-attention **key** cache `[n_text_ctx, d]`; each step's k-proj
    /// GEMM writes the new `[t, d]` rows at row `start` (`encode_gemm_off`).
    self_k: OwnedBuf,
    /// Resident self-attention **value** cache `[n_text_ctx, d]`.
    self_v: OwnedBuf,
}

/// A device-resident autoregressive Whisper **decode session** (Phase-5
/// decoder-step residency). Weights are uploaded **once**, the self-attention
/// key/value cache is kept **on the GPU** and appended each step, the
/// cross-attention keys/values are uploaded **once** from the (already projected)
/// encoder output, and each decode step is collapsed to **one command-buffer
/// submission + one logits readback** — versus the per-op path's `~20·N`
/// submissions *and* a full-weight H2D on every op, every token.
///
/// It runs **exactly** the per-op decoder's op sequence (the same layer-norm /
/// GEMM / fused attention / fused MLP / residual-add kernels, in the same order
/// and launch geometry, with the causal self-attention using the fused
/// masked-softmax proven bit-identical to the host `-inf` mask), so it is
/// bit-identical to running the decoder step per-op on the GPU, and matches the
/// CPU decoder within the FP32 bound — and the greedy argmax sequence is
/// therefore identical.
///
/// # `Send`, thread-affine at use
///
/// The session **owns** its [`MetalContext`] and holds only raw [`OwnedBuf`]
/// device buffers (no `MetalDeviceTensor<'ctx>`, so no self-referential
/// lifetime). Even though the raw `Id` handles in [`MetalContext`] / [`OwnedBuf`]
/// are `!Send` at the Rust type level (`*mut c_void`), the objects they refer
/// to — `MTLDevice`, `MTLCommandQueue`, `MTLBuffer` and compute-pipeline
/// objects — are documented by Apple as thread-safe, and the one non-thread-
/// safe class (`MTLCommandBuffer` / `MTLCommandEncoder`) is created, encoded,
/// committed and released **within a single [`Self::step`] call** (inside one
/// autorelease pool), never held across calls. So moving the session from the
/// thread that built it to another thread is safe: the next step creates its
/// command buffer / encoder on the new thread. `Send` is asserted here (in the
/// backend crate, whose `#![allow(unsafe_code)]` opt-out permits it) so the
/// model layer can hold `Option<MetalDecodeSession>` inside a `Send` host
/// `DecoderState` — the compile-time `assert_send::<DecoderState>()` bound and
/// the cross-thread decode test both stay green — **without** either
/// reuploading every weight per step or forcing the CPU / GPU decode paths to
/// diverge in shape. `Sync` is deliberately **not** asserted: an
/// autoregressive step depends on the previous step's KV cache write, and the
/// session sits behind a `&mut` on `DecoderState`, so Rust's ownership rules
/// already enforce single-thread-at-a-time access — a shared-borrow `Sync`
/// bound would add no correctness value and (unlike `Send`) is not what any
/// caller needs.
///
/// The device buffers are declared **before** `ctx` so Rust drops them first
/// (every `MTLBuffer` released before the device the context owns is released).
pub struct MetalDecodeSession {
    layers: Vec<DevDecoderLayer>,
    /// Tied logits head `[n_vocab, d]`, resident (also the token embedding table,
    /// but the token gather is a host op, so only the logits projection needs it
    /// on the device).
    token_emb: OwnedBuf,
    ln_post_g: OwnedBuf,
    ln_post_b: OwnedBuf,
    /// A 1-float never-read buffer bound where a bias is absent (`has_bias = 0`).
    dummy: OwnedBuf,
    /// Residual hidden stream `[max_t_q, d]` (each step's `[t, d]` embedding is
    /// written here, then the residual adds mutate it in place).
    h: OwnedBuf,
    ln: OwnedBuf,
    block_out: OwnedBuf,
    normed: OwnedBuf,
    q: OwnedBuf,
    context: OwnedBuf,
    qh: OwnedBuf,
    ctx_h: OwnedBuf,
    vh: OwnedBuf,
    kh_t: OwnedBuf,
    scores: OwnedBuf,
    probs: OwnedBuf,
    mlp_h: OwnedBuf,
    mlp_a: OwnedBuf,
    /// Resident `[max_t_q, n_vocab]` logits (contiguous per-row, one per decoded
    /// row of the last step). The step readback pulls only the `[t, n_vocab]`
    /// prefix that step 実際に wrote; the tail past `t` is left untouched between
    /// steps.
    logits: OwnedBuf,
    /// Host copy of the last step's `[max_t_q, n_vocab]` logits scratch — the
    /// tied-head produces every decoded row (`[t, n_vocab]`) so the model layer
    /// can compare against the CPU decoder's full-row output. [`Self::last_logits`]
    /// returns the last row; [`Self::all_logits`] returns the `[last_t, n_vocab]`
    /// prefix `step` wrote.
    logits_host: Vec<f32>,
    d: usize,
    n_head: usize,
    ff: usize,
    n_text_ctx: usize,
    n_vocab: usize,
    n_ctx: usize,
    max_t_q: usize,
    eps: f32,
    scale: f32,
    /// Committed token positions (the causal query offset for the next step).
    pos: usize,
    /// Row count the last [`Self::step`] wrote (`0` before the first step);
    /// [`Self::all_logits`] returns `logits_host[..last_t * n_vocab]` and
    /// [`Self::last_logits`] returns the last row of that prefix.
    last_t: usize,
    /// Owned last so it drops **after** every device buffer above.
    ctx: MetalContext,
}

// SAFETY: The session owns a [`MetalContext`] and a set of [`OwnedBuf`]
// (`MTLDevice`, `MTLCommandQueue`, `MTLBuffer` handles + compiled compute
// pipelines). Apple's Metal "Thread-Safety Summary" documents `MTLDevice`,
// `MTLCommandQueue`, `MTLBuffer` and pipeline-state objects as thread-safe:
// their reference counts and use through the documented Objective-C APIs are
// safe from any thread. The one non-thread-safe class family —
// `MTLCommandBuffer` / `MTLCommandEncoder` — is created, encoded, committed
// and released **inside a single [`Self::step`] call** (bracketed by one
// autorelease pool); no command buffer or encoder is stored on the session
// between calls. So moving the whole session across threads is safe: the next
// `step` allocates its command buffer / encoder from the queue on the new
// thread. This `Send` impl was deferred to keep the earlier per-op path
// defensively thread-affine; asserting it here now lets the model-layer
// `DecoderState` (the Whisper decoder session) stay `Send` — required by its
// existing compile-time `assert_send::<DecoderState>()` bound + the
// cross-thread decode test — while embedding this device-resident driver.
// `Sync` is deliberately NOT asserted: every step depends on the previous
// step's KV write, and the caller borrows the session `&mut`, so shared-borrow
// concurrency has no meaning here.
unsafe impl Send for MetalDecodeSession {}

impl MetalDecodeSession {
    /// Builds a decode session: creates its own [`MetalContext`], uploads every
    /// decoder weight + the pre-projected cross-attention K/V (from `layers`) and
    /// the tied logits head, and reserves the self-attention KV cache to the hard
    /// `n_text_ctx` bound and the per-step scratch to `max_t_q` × the key window —
    /// all **once**. `max_t_q` is the widest single step (the forced-prefix
    /// width; steady-state steps decode one token).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on a zero / mismatched dimension or a
    /// weight-slice shape mismatch; [`VokraError::BackendUnavailable`] if there is
    /// no Metal device or a buffer cannot be created.
    #[allow(clippy::too_many_arguments)] // whole-decoder operand set (dims + weights + I/O)
    pub fn new(
        d: usize,
        n_head: usize,
        ff: usize,
        n_text_ctx: usize,
        n_vocab: usize,
        n_ctx: usize,
        max_t_q: usize,
        eps: f32,
        layers: &[DecoderLayerView<'_>],
        token_emb: &[f32],
        ln_post_gamma: &[f32],
        ln_post_beta: &[f32],
    ) -> Result<MetalDecodeSession> {
        if d == 0 || n_head == 0 || ff == 0 || n_vocab == 0 || n_ctx == 0 {
            return Err(VokraError::InvalidArgument(
                "decode session dims d, n_head, ff, n_vocab, n_ctx must all be >= 1".to_owned(),
            ));
        }
        if d % n_head != 0 {
            return Err(VokraError::InvalidArgument(format!(
                "decode session d ({d}) must be divisible by n_head ({n_head})"
            )));
        }
        if n_text_ctx == 0 || max_t_q == 0 || max_t_q > n_text_ctx {
            return Err(VokraError::InvalidArgument(format!(
                "decode session needs 1 <= max_t_q ({max_t_q}) <= n_text_ctx ({n_text_ctx})"
            )));
        }
        let dd = checked_mul(d, d, "decode d*d")?;
        let dff = checked_mul(d, ff, "decode d*ff")?;
        let nctx_d = checked_mul(n_ctx, d, "decode n_ctx*d")?;
        expect_len(
            "decode token_emb",
            token_emb.len(),
            checked_mul(n_vocab, d, "decode n_vocab*d")?,
        )?;
        expect_len("decode ln_post_gamma", ln_post_gamma.len(), d)?;
        expect_len("decode ln_post_beta", ln_post_beta.len(), d)?;
        // Validate each layer's weight shapes before touching the GPU.
        for (li, l) in layers.iter().enumerate() {
            let w = |name: &str, got: usize, want: usize| {
                expect_len(&format!("decode layer {li} {name}"), got, want)
            };
            w("self_ln_gamma", l.self_ln_gamma.len(), d)?;
            w("self_ln_beta", l.self_ln_beta.len(), d)?;
            w("self_q_w", l.self_q_w.len(), dd)?;
            w("self_k_w", l.self_k_w.len(), dd)?;
            w("self_v_w", l.self_v_w.len(), dd)?;
            w("self_out_w", l.self_out_w.len(), dd)?;
            w("cross_ln_gamma", l.cross_ln_gamma.len(), d)?;
            w("cross_ln_beta", l.cross_ln_beta.len(), d)?;
            w("cross_q_w", l.cross_q_w.len(), dd)?;
            w("cross_out_w", l.cross_out_w.len(), dd)?;
            w("cross_k", l.cross_k.len(), nctx_d)?;
            w("cross_v", l.cross_v.len(), nctx_d)?;
            w("mlp_ln_gamma", l.mlp_ln_gamma.len(), d)?;
            w("mlp_ln_beta", l.mlp_ln_beta.len(), d)?;
            w("fc1_w", l.fc1_w.len(), dff)?;
            w("fc2_w", l.fc2_w.len(), dff)?;
        }

        let ctx = MetalContext::new()?;
        // Upload is bracketed by one autorelease pool (the buffer creations send
        // Objective-C messages).
        // SAFETY: token consumed by the matching pop below.
        let pool = unsafe { sys::objc_autoreleasePoolPush() };
        let built = Self::build(
            &ctx,
            d,
            n_head,
            ff,
            n_text_ctx,
            n_vocab,
            n_ctx,
            max_t_q,
            layers,
            token_emb,
            ln_post_gamma,
            ln_post_beta,
        );
        // SAFETY: `pool` is the token from the push above.
        unsafe { sys::objc_autoreleasePoolPop(pool) };
        let (mut buffers, dummy) = built?;

        Ok(MetalDecodeSession {
            layers: buffers.layers,
            token_emb: buffers.token_emb.take().expect("token_emb built"),
            ln_post_g: buffers.ln_post_g.take().expect("ln_post_g built"),
            ln_post_b: buffers.ln_post_b.take().expect("ln_post_b built"),
            dummy,
            h: buffers.h.take().expect("h built"),
            ln: buffers.ln.take().expect("ln built"),
            block_out: buffers.block_out.take().expect("block_out built"),
            normed: buffers.normed.take().expect("normed built"),
            q: buffers.q.take().expect("q built"),
            context: buffers.context.take().expect("context built"),
            qh: buffers.qh.take().expect("qh built"),
            ctx_h: buffers.ctx_h.take().expect("ctx_h built"),
            vh: buffers.vh.take().expect("vh built"),
            kh_t: buffers.kh_t.take().expect("kh_t built"),
            scores: buffers.scores.take().expect("scores built"),
            probs: buffers.probs.take().expect("probs built"),
            mlp_h: buffers.mlp_h.take().expect("mlp_h built"),
            mlp_a: buffers.mlp_a.take().expect("mlp_a built"),
            logits: buffers.logits.take().expect("logits built"),
            logits_host: vec![0.0f32; checked_mul(max_t_q, n_vocab, "decode max_t_q*n_vocab")?],
            d,
            n_head,
            ff,
            n_text_ctx,
            n_vocab,
            n_ctx,
            max_t_q,
            eps,
            scale: ((d / n_head) as f32).powf(-0.5),
            pos: 0,
            last_t: 0,
            ctx,
        })
    }

    /// Uploads all weights + the pre-projected cross-KV, reserves the self-KV
    /// cache and the per-step scratch. Factored out of [`Self::new`] so the whole
    /// H2D / allocation burst runs inside one autorelease pool. Returns the
    /// buffers (in a builder holder) plus the shared bias dummy.
    #[allow(clippy::too_many_arguments)]
    fn build(
        ctx: &MetalContext,
        d: usize,
        n_head: usize,
        ff: usize,
        n_text_ctx: usize,
        _n_vocab: usize,
        n_ctx: usize,
        max_t_q: usize,
        layers: &[DecoderLayerView<'_>],
        token_emb: &[f32],
        ln_post_gamma: &[f32],
        ln_post_beta: &[f32],
    ) -> Result<(SessionBuffers, OwnedBuf)> {
        let up = |s: &[f32]| ctx.new_buffer_from_slice(s);
        let up_opt = |s: Option<&[f32]>| -> Result<Option<OwnedBuf>> {
            s.map(|d| ctx.new_buffer_from_slice(d)).transpose()
        };
        let hd = d / n_head;
        let max_tkv = n_text_ctx.max(n_ctx);
        // Reserve amounts (all fit — validated in `new`).
        let ntc_d = checked_mul(n_text_ctx, d, "decode n_text_ctx*d")?;
        let td = checked_mul(max_t_q, d, "decode max_t_q*d")?;
        let thd = checked_mul(max_t_q, hd, "decode max_t_q*hd")?;
        let tkvhd = checked_mul(max_tkv, hd, "decode max_tkv*hd")?;
        let ttkv = checked_mul(max_t_q, max_tkv, "decode max_t_q*max_tkv")?;
        let tff = checked_mul(max_t_q, ff, "decode max_t_q*ff")?;
        // `[max_t_q, n_vocab]` — the tied head produces every decoded row, so the
        // model-layer path can compare against the CPU decoder's `[t, n_vocab]`
        // output (not just the greedy last-row read). `t == 1` uses only the first
        // `n_vocab` entries; `t == max_t_q` (the forced prefix step) uses all.
        let tv = checked_mul(max_t_q, _n_vocab, "decode max_t_q*n_vocab")?;

        let mut dev_layers = Vec::with_capacity(layers.len());
        for l in layers {
            dev_layers.push(DevDecoderLayer {
                self_ln_g: up(l.self_ln_gamma)?,
                self_ln_b: up(l.self_ln_beta)?,
                self_q_w: up(l.self_q_w)?,
                self_q_bias: up_opt(l.self_q_bias)?,
                self_k_w: up(l.self_k_w)?,
                self_k_bias: up_opt(l.self_k_bias)?,
                self_v_w: up(l.self_v_w)?,
                self_v_bias: up_opt(l.self_v_bias)?,
                self_out_w: up(l.self_out_w)?,
                self_out_bias: up_opt(l.self_out_bias)?,
                cross_ln_g: up(l.cross_ln_gamma)?,
                cross_ln_b: up(l.cross_ln_beta)?,
                cross_q_w: up(l.cross_q_w)?,
                cross_q_bias: up_opt(l.cross_q_bias)?,
                cross_out_w: up(l.cross_out_w)?,
                cross_out_bias: up_opt(l.cross_out_bias)?,
                cross_k: up(l.cross_k)?,
                cross_v: up(l.cross_v)?,
                mlp_ln_g: up(l.mlp_ln_gamma)?,
                mlp_ln_b: up(l.mlp_ln_beta)?,
                fc1_w: up(l.fc1_w)?,
                fc1_bias: up_opt(l.fc1_bias)?,
                fc2_w: up(l.fc2_w)?,
                fc2_bias: up_opt(l.fc2_bias)?,
                self_k: ctx.new_buffer_output(ntc_d)?,
                self_v: ctx.new_buffer_output(ntc_d)?,
            });
        }
        let dummy = ctx.new_buffer_from_slice(&[0.0f32])?;
        let buffers = SessionBuffers {
            layers: dev_layers,
            token_emb: Some(up(token_emb)?),
            ln_post_g: Some(up(ln_post_gamma)?),
            ln_post_b: Some(up(ln_post_beta)?),
            h: Some(ctx.new_buffer_output(td)?),
            ln: Some(ctx.new_buffer_output(td)?),
            block_out: Some(ctx.new_buffer_output(td)?),
            normed: Some(ctx.new_buffer_output(td)?),
            q: Some(ctx.new_buffer_output(td)?),
            context: Some(ctx.new_buffer_output(td)?),
            qh: Some(ctx.new_buffer_output(thd)?),
            ctx_h: Some(ctx.new_buffer_output(thd)?),
            vh: Some(ctx.new_buffer_output(tkvhd)?),
            kh_t: Some(ctx.new_buffer_output(tkvhd)?),
            scores: Some(ctx.new_buffer_output(ttkv)?),
            probs: Some(ctx.new_buffer_output(ttkv)?),
            mlp_h: Some(ctx.new_buffer_output(tff)?),
            mlp_a: Some(ctx.new_buffer_output(tff)?),
            logits: Some(ctx.new_buffer_output(tv)?),
        };
        Ok((buffers, dummy))
    }

    /// Advances the decode by the `t` tokens whose `[t, d]` token+positional
    /// embedding is `embedded` (the host gather; `t <= max_t_q`), starting at
    /// committed position `start`. Runs the whole step device-resident in ONE
    /// command buffer and leaves the full `[t, n_vocab]` logits (one row per
    /// decoded token, row-major) in the host buffer [`Self::all_logits`]
    /// returns; [`Self::last_logits`] reads the last of those rows for the greedy
    /// / argmax path.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on a bad `t` / `start` / `embedded` length;
    /// [`VokraError::BackendUnavailable`] on a Metal failure.
    pub fn step(&mut self, embedded: &[f32], t: usize, start: usize) -> Result<()> {
        let d = self.d;
        if t == 0 {
            return Err(VokraError::InvalidArgument(
                "decode step: t must be >= 1".to_owned(),
            ));
        }
        if t > self.max_t_q {
            return Err(VokraError::InvalidArgument(format!(
                "decode step: t ({t}) exceeds the session's max_t_q ({})",
                self.max_t_q
            )));
        }
        expect_len(
            "decode step embedded",
            embedded.len(),
            checked_mul(t, d, "decode step t*d")?,
        )?;
        let t_kv = start.checked_add(t).ok_or_else(|| {
            VokraError::InvalidArgument("decode step position overflow".to_owned())
        })?;
        if t_kv > self.n_text_ctx {
            return Err(VokraError::InvalidArgument(format!(
                "decode step: position {t_kv} exceeds n_text_ctx {}",
                self.n_text_ctx
            )));
        }
        // Write this step's embedding into the resident `h` buffer (host copy on
        // unified memory; no new device allocation).
        write_buf(&self.h, embedded)?;

        // SAFETY: token consumed by the matching pop below.
        let pool = unsafe { sys::objc_autoreleasePoolPush() };
        let r = self.run_decode_step(t, start, t_kv);
        // SAFETY: `pool` is the token from the push above.
        unsafe { sys::objc_autoreleasePoolPop(pool) };
        r?;

        // Single per-step readback of ALL `[t, n_vocab]` rows the tied head wrote
        // (only the `t·n_vocab` prefix — the `max_t_q` tail past `t` is left
        // untouched and never observed).
        let take = checked_mul(t, self.n_vocab, "decode step t*n_vocab")?;
        read_back(&self.logits, &mut self.logits_host[..take])?;
        self.pos = t_kv;
        self.last_t = t;
        Ok(())
    }

    /// Encodes the whole decode step (`n_text_layer` blocks + final LayerNorm +
    /// tied-logits gemv) into ONE command buffer and commits it once. `&self`: it
    /// only reads the resident buffers and encodes passes (the host `pos` is
    /// advanced by the caller after the readback). `t_kv = start + t`.
    fn run_decode_step(&self, t: usize, start: usize, t_kv: usize) -> Result<()> {
        let d = self.d;
        let n_head = self.n_head;
        let scale = self.scale;
        let eps = self.eps;
        let td = t * d;
        let cmd = self.ctx.new_command_buffer("decode step")?;
        for layer in &self.layers {
            // --- causal self-attention over the growing KV cache ---
            // ln = layer_norm(h, self_ln)
            self.ctx.encode_layer_norm(
                cmd,
                &self.h,
                &layer.self_ln_g,
                &layer.self_ln_b,
                &self.ln,
                t,
                d,
                eps,
            )?;
            // Append this step's k/v rows AT cache row `start` (GEMM-writes-at-offset).
            self.ctx.encode_gemm_off(
                cmd,
                &self.ln,
                &layer.self_k_w,
                opt_buf_or(layer.self_k_bias.as_ref(), &self.dummy),
                &layer.self_k,
                start * d,
                t,
                d,
                d,
                layer.self_k_bias.is_some(),
            )?;
            self.ctx.encode_gemm_off(
                cmd,
                &self.ln,
                &layer.self_v_w,
                opt_buf_or(layer.self_v_bias.as_ref(), &self.dummy),
                &layer.self_v,
                start * d,
                t,
                d,
                d,
                layer.self_v_bias.is_some(),
            )?;
            // Causal fused attention over the whole cache `[0, t_kv)`.
            self.ctx.encode_attn_passes(
                cmd,
                &AttnPassDims {
                    t_q: t,
                    t_kv,
                    d,
                    n_head,
                    scale,
                    has_q_bias: layer.self_q_bias.is_some(),
                    has_out_bias: layer.self_out_bias.is_some(),
                    causal: true,
                    q_offset: start,
                },
                &AttnPassBufs {
                    xq: &self.ln,
                    q_w: &layer.self_q_w,
                    q_bias: opt_buf_or(layer.self_q_bias.as_ref(), &self.dummy),
                    k: &layer.self_k,
                    v: &layer.self_v,
                    out_w: &layer.self_out_w,
                    out_bias: opt_buf_or(layer.self_out_bias.as_ref(), &self.dummy),
                    q: &self.q,
                    context: &self.context,
                    qh: &self.qh,
                    vh: &self.vh,
                    kh_t: &self.kh_t,
                    scores: &self.scores,
                    probs: &self.probs,
                    ctx_h: &self.ctx_h,
                    out: &self.block_out,
                },
            )?;
            self.ctx
                .encode_residual_add(cmd, &self.h, &self.block_out, td)?;

            // --- cross-attention over the (fixed) encoder output ---
            self.ctx.encode_layer_norm(
                cmd,
                &self.h,
                &layer.cross_ln_g,
                &layer.cross_ln_b,
                &self.ln,
                t,
                d,
                eps,
            )?;
            self.ctx.encode_attn_passes(
                cmd,
                &AttnPassDims {
                    t_q: t,
                    t_kv: self.n_ctx,
                    d,
                    n_head,
                    scale,
                    has_q_bias: layer.cross_q_bias.is_some(),
                    has_out_bias: layer.cross_out_bias.is_some(),
                    causal: false,
                    q_offset: 0,
                },
                &AttnPassBufs {
                    xq: &self.ln,
                    q_w: &layer.cross_q_w,
                    q_bias: opt_buf_or(layer.cross_q_bias.as_ref(), &self.dummy),
                    k: &layer.cross_k,
                    v: &layer.cross_v,
                    out_w: &layer.cross_out_w,
                    out_bias: opt_buf_or(layer.cross_out_bias.as_ref(), &self.dummy),
                    q: &self.q,
                    context: &self.context,
                    qh: &self.qh,
                    vh: &self.vh,
                    kh_t: &self.kh_t,
                    scores: &self.scores,
                    probs: &self.probs,
                    ctx_h: &self.ctx_h,
                    out: &self.block_out,
                },
            )?;
            self.ctx
                .encode_residual_add(cmd, &self.h, &self.block_out, td)?;

            // --- MLP ---
            self.ctx.encode_layer_norm(
                cmd,
                &self.h,
                &layer.mlp_ln_g,
                &layer.mlp_ln_b,
                &self.ln,
                t,
                d,
                eps,
            )?;
            self.ctx.encode_mlp_passes(
                cmd,
                &MlpPassDims {
                    t,
                    d,
                    ffn: self.ff,
                    has_fc1_bias: layer.fc1_bias.is_some(),
                    has_fc2_bias: layer.fc2_bias.is_some(),
                },
                &MlpPassBufs {
                    x: &self.ln,
                    fc1_w: &layer.fc1_w,
                    fc1_bias: opt_buf_or(layer.fc1_bias.as_ref(), &self.dummy),
                    fc2_w: &layer.fc2_w,
                    fc2_bias: opt_buf_or(layer.fc2_bias.as_ref(), &self.dummy),
                    h: &self.mlp_h,
                    a: &self.mlp_a,
                    out: &self.block_out,
                },
            )?;
            self.ctx
                .encode_residual_add(cmd, &self.h, &self.block_out, td)?;
        }

        // Final LayerNorm into `normed`, then the tied-logits head on EVERY
        // decoded row (`t` gemvs into `logits[i·n_vocab .. (i+1)·n_vocab]`,
        // reading `normed[i·d .. (i+1)·d]`). One gemv per row keeps each
        // reduction identical to the CPU decoder's `t == 1` fast path — the
        // same math, just repeated `t` times inside the SAME command buffer, so
        // the whole step still commits + waits exactly once (unchanged
        // submission accounting). All `t` rows land in `logits_host` so the
        // model-layer path can compare against the CPU decoder's full `[t,
        // n_vocab]` output, not only the greedy last row.
        self.ctx.encode_layer_norm(
            cmd,
            &self.h,
            &self.ln_post_g,
            &self.ln_post_b,
            &self.normed,
            t,
            d,
            eps,
        )?;
        for i in 0..t {
            self.ctx.encode_gemv_off(
                cmd,
                &self.token_emb,
                &self.normed,
                i * d,
                &self.logits,
                i * self.n_vocab,
                self.n_vocab,
                d,
            )?;
        }
        self.ctx.commit_and_wait(cmd, "decode step")
    }

    /// The last decoded row of the last [`Self::step`] — `[n_vocab]` logits, the
    /// greedy / argmax read. Empty before any step (`last_t == 0`).
    #[must_use]
    pub fn last_logits(&self) -> &[f32] {
        if self.last_t == 0 {
            return &[];
        }
        let v = self.n_vocab;
        let start = (self.last_t - 1) * v;
        &self.logits_host[start..start + v]
    }

    /// All `[t, n_vocab]` rows the last [`Self::step`] wrote, row-major (row `i`
    /// at offset `i·n_vocab`). This is the full-row output the model-layer path
    /// compares against the CPU decoder's [`t, n_vocab]` logits (not just the
    /// last row). Empty before any step.
    #[must_use]
    pub fn all_logits(&self) -> &[f32] {
        &self.logits_host[..self.last_t * self.n_vocab]
    }

    /// Committed token positions in the self-attention cache (the causal query
    /// offset for the next [`Self::step`]).
    #[must_use]
    pub fn positions(&self) -> usize {
        self.pos
    }

    /// Rewinds the position clock to 0 for a fresh decode of the same audio
    /// (the resident weights + cross-KV stay valid; the self-KV rows are simply
    /// overwritten from row 0 again). Mirrors [`vokra_core::KvCache::reset`].
    pub fn reset(&mut self) {
        self.pos = 0;
        // `last_t = 0` invalidates the stale `all_logits` / `last_logits` views
        // so a caller reading them before the next `step` sees an empty slice
        // (the CPU decoder's post-reset semantics — its logits scratch is not
        // observable until the next step writes it either).
        self.last_t = 0;
    }

    /// Command-buffer submissions issued through the owned context — one per
    /// [`Self::step`] (plus the session's construction issues none).
    #[must_use]
    pub fn submission_count(&self) -> u64 {
        self.ctx.submission_count()
    }
}

/// Owned-buffer holder used only while [`MetalDecodeSession::new`] assembles the
/// session: every scratch/weight buffer starts here (as `Option`, `take`n into
/// the final struct) so the whole allocation burst can happen inside one
/// autorelease pool before the `MetalDecodeSession` is formed.
struct SessionBuffers {
    layers: Vec<DevDecoderLayer>,
    token_emb: Option<OwnedBuf>,
    ln_post_g: Option<OwnedBuf>,
    ln_post_b: Option<OwnedBuf>,
    h: Option<OwnedBuf>,
    ln: Option<OwnedBuf>,
    block_out: Option<OwnedBuf>,
    normed: Option<OwnedBuf>,
    q: Option<OwnedBuf>,
    context: Option<OwnedBuf>,
    qh: Option<OwnedBuf>,
    ctx_h: Option<OwnedBuf>,
    vh: Option<OwnedBuf>,
    kh_t: Option<OwnedBuf>,
    scores: Option<OwnedBuf>,
    probs: Option<OwnedBuf>,
    mlp_h: Option<OwnedBuf>,
    mlp_a: Option<OwnedBuf>,
    logits: Option<OwnedBuf>,
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

/// Copies `data` into the first `data.len()` f32s of a shared buffer's `contents`
/// (H2D on Apple unified memory). The decode session writes each step's
/// `[t, d]` token embedding into its resident `h` buffer this way — one small
/// host copy, no new device allocation. The write is host-ordered before the
/// step's command buffer is committed, and shared storage is coherent, so the
/// GPU sees it. `buf` must hold at least `data.len()` f32s.
fn write_buf(buf: &OwnedBuf, data: &[f32]) -> Result<()> {
    // SAFETY: `buf` is a valid shared MTLBuffer of at least `data.len()` f32s; its
    // `contents` is host-writable (shared storage) before the buffer is used by a
    // committed command buffer.
    let contents = unsafe { sys::send_ptr(buf.0, sys::sel(b"contents\0")) } as *mut f32;
    if contents.is_null() {
        return Err(VokraError::BackendUnavailable(
            "input MTLBuffer contents pointer is null".to_owned(),
        ));
    }
    // SAFETY: `data` is valid for `data.len()` f32s; `contents` is the base of at
    // least that many valid, non-overlapping f32s in shared memory.
    unsafe { core::ptr::copy_nonoverlapping(data.as_ptr(), contents, data.len()) };
    Ok(())
}

/// Picks the real `bias` buffer or the shared 1-float `dummy` (for an absent
/// bias, bound but never read because `has_bias = 0`) — the `OwnedBuf` sibling of
/// [`bias_or_dummy`].
fn opt_buf_or<'a>(bias: Option<&'a OwnedBuf>, dummy: &'a OwnedBuf) -> &'a OwnedBuf {
    bias.unwrap_or(dummy)
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

fn validate_rms_norm(
    input: &[f32],
    out: &[f32],
    rows: usize,
    cols: usize,
    gamma: &[f32],
) -> Result<()> {
    validate_rows_cols(input, out, rows, cols)?;
    expect_len("rms_norm gamma", gamma.len(), cols)
}

/// Validates the adjacent-pair RoPE shapes: `input`/`out` are `seq_len ×
/// head_dim`, `head_dim` is even, and `inv_freqs` has `head_dim / 2` entries
/// (mirroring the CPU `rope_apply_adjacent` guard).
fn validate_rope(
    input: &[f32],
    out: &[f32],
    seq_len: usize,
    head_dim: usize,
    inv_freqs: &[f32],
) -> Result<()> {
    if head_dim % 2 != 0 {
        return Err(VokraError::InvalidArgument(format!(
            "rope head_dim ({head_dim}) must be even"
        )));
    }
    let total = checked_mul(seq_len, head_dim, "rope seq_len*head_dim")?;
    expect_len("rope input", input.len(), total)?;
    expect_len("rope out", out.len(), total)?;
    expect_len("rope inv_freqs", inv_freqs.len(), head_dim / 2)
}

/// Validates the SwiGLU shapes: `gate`, `up` and `out` are the same length
/// (mirroring the CPU `silu_inplace` + `hadamard_inplace` guard).
fn validate_swiglu(gate: &[f32], up: &[f32], out: &[f32]) -> Result<()> {
    expect_len("swiglu up", up.len(), gate.len())?;
    expect_len("swiglu out", out.len(), gate.len())
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

// =====================================================================
// M3-04 fused KV-cache dequant + GEMV trait impl (Metal backend arm)
// =====================================================================
//
// The concrete GPU implementation of the
// [`vokra_core::KvQuantDequantGemvOps`] trait: dispatches into
// [`MetalContext::dequant_gemv_f32`] (defined above). Kept at the bottom of
// the file so it sits alongside the other trait impls / helpers rather than
// inside the impl block that owns the launcher — keeps grep-locality with the
// CUDA analogue.
impl vokra_core::KvQuantDequantGemvOps for MetalContext {
    fn fused_dequant_gemv(
        &self,
        mode: vokra_core::KvQuant,
        blocks_bytes: &[u8],
        n_rows: usize,
        n_blocks_per_row: usize,
        x: &[f32],
    ) -> Result<Vec<f32>> {
        self.dequant_gemv_f32(mode, blocks_bytes, n_rows, n_blocks_per_row, x)
    }
}
