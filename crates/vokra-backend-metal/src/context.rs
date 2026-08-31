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

#pragma clang fp contract(off)
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
    float acc = (dims.has_bias != 0u) ? bias[col] : 0.0f;
    const uint arow = row * dims.K;
    for (uint k = 0; k < dims.K; ++k) {
        acc += A[arow + k] * B[k * dims.N + col];
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

// ---- mixed FP32 activation × raw BF16 weight GEMM -------------------------
// The weight buffer is deliberately `ushort` storage: these are the numeric
// BF16 bit patterns decoded from little-endian model bytes, not an eagerly
// widened FP32 mirror.  Widening is exact (the BF16 exponent/significand are
// the high 16 bits of an IEEE-754 binary32 value) and every product is added
// to an FP32 accumulator.
struct GemmF32Bf16BitsDims {
    uint M;
    uint N;
    uint K;
};

inline float vokra_bf16_bits_to_f32(ushort bits) {
    return as_type<float>(uint(bits) << 16);
}

kernel void vokra_gemm_f32_bf16_bits(
    device const float*  A    [[buffer(0)]],
    device const ushort* B    [[buffer(1)]],
    device float*        C    [[buffer(2)]],
    constant GemmF32Bf16BitsDims& d [[buffer(3)]],
    uint2                 gid  [[thread_position_in_grid]])
{
    const uint row = gid.y;
    const uint col = gid.x;
    if (row >= d.M || col >= d.N) return;
    float acc = 0.0f;
    const uint arow = row * d.K;
    for (uint k = 0u; k < d.K; ++k) {
        acc += A[arow + k] * vokra_bf16_bits_to_f32(B[k * d.N + col]);
    }
    C[row * d.N + col] = acc;
}

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

// ---- one-group GroupNorm: channel-major [channels, positions] ---------------
// SepFormer's mask network reduces 130k–384k values per group. A single long
// FP32 left fold loses enough low bits for the dual-path stack to amplify, so
// this kernel uses 256 strided partials and a fixed pairwise reduction tree.
struct GroupNormDims {
    uint channels;
    uint positions;
    float eps;
};

kernel void vokra_group_norm_f32(
    device const float*     inp   [[buffer(0)]],
    device const float*     gamma [[buffer(1)]],
    device const float*     beta  [[buffer(2)]],
    device float*           out   [[buffer(3)]],
    constant GroupNormDims& d     [[buffer(4)]],
    uint                    gid   [[thread_position_in_grid]])
{
    if (gid != 0u) {
        return;
    }
    constexpr uint partial_count = 256u;
    const uint total = d.channels * d.positions;
    thread float partial[partial_count];
    for (uint lane = 0u; lane < partial_count; ++lane) {
        float sum = 0.0f;
        for (uint index = lane; index < total; index += partial_count) {
            sum += inp[index];
        }
        partial[lane] = sum;
    }
    for (uint width = partial_count / 2u; width > 0u; width /= 2u) {
        for (uint index = 0u; index < width; ++index) {
            partial[index] += partial[index + width];
        }
    }
    const float mean = partial[0] / (float)total;

    for (uint lane = 0u; lane < partial_count; ++lane) {
        float sum = 0.0f;
        for (uint index = lane; index < total; index += partial_count) {
            const float delta = inp[index] - mean;
            sum += delta * delta;
        }
        partial[lane] = sum;
    }
    for (uint width = partial_count / 2u; width > 0u; width /= 2u) {
        for (uint index = 0u; index < width; ++index) {
            partial[index] += partial[index + width];
        }
    }
    const float inv_std = 1.0f / sqrt(partial[0] / (float)total + d.eps);
    for (uint channel = 0u; channel < d.channels; ++channel) {
        for (uint position = 0u; position < d.positions; ++position) {
            const uint index = channel * d.positions + position;
            out[index] = (inp[index] - mean) * inv_std * gamma[channel] + beta[channel];
        }
    }
}

// ---- multi-group GroupNorm: channel-major [channels, positions] -----------
// NCSN++ uses GroupNorm with an explicit group count. One thread owns one
// complete group: it reduces that group once, then writes every channel and
// position in it. This preserves the CPU sibling's channel-major membership
// and left-fold order without repeating the reduction for every output.
struct GroupNormGroupsDims {
    uint channels;
    uint positions;
    uint groups;
    float eps;
};

kernel void vokra_group_norm_groups_f32(
    device const float* inp   [[buffer(0)]],
    device const float* gamma [[buffer(1)]],
    device const float* beta  [[buffer(2)]],
    device float*       out   [[buffer(3)]],
    constant GroupNormGroupsDims& d [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= d.groups) {
        return;
    }
    const uint channels_per_group = d.channels / d.groups;
    const uint first_channel = gid * channels_per_group;
    const uint count = channels_per_group * d.positions;
    float sum = 0.0f;
    for (uint c = first_channel; c < first_channel + channels_per_group; ++c) {
        const uint base = c * d.positions;
        for (uint p = 0u; p < d.positions; ++p) {
            sum += inp[base + p];
        }
    }
    const float mean = sum / (float)count;
    float variance_sum = 0.0f;
    for (uint c = first_channel; c < first_channel + channels_per_group; ++c) {
        const uint base = c * d.positions;
        for (uint p = 0u; p < d.positions; ++p) {
            const float delta = inp[base + p] - mean;
            variance_sum += delta * delta;
        }
    }
    const float inv_std = 1.0f / sqrt(variance_sum / (float)count + d.eps);
    for (uint channel = first_channel;
         channel < first_channel + channels_per_group;
         ++channel) {
        const uint base = channel * d.positions;
        for (uint position = 0u; position < d.positions; ++position) {
            const uint index = base + position;
            const float normalized = (inp[index] - mean) * inv_std;
            out[index] = normalized * gamma[channel] + beta[channel];
        }
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

// ---- gelu_new: GPT-2 / Transformers tanh approximation --------------------
// Kept distinct from the exact/erf kernel above. MOSS-TTS Nano's released
// custom GPT-2 explicitly selects activation_function="gelu_new".
kernel void vokra_gelu_new_f32(
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
    const float inner = 0.7978845608028654f * (v + 0.044715f * v * v * v);
    out[i] = 0.5f * v * (1.0f + tanh(inner));
}

// ---- relu: exact element-wise max(x, 0) ------------------------------------
struct ReluDims {
    uint n;
};

kernel void vokra_relu_f32(
    device const float* x   [[buffer(0)]],
    device float*       out [[buffer(1)]],
    constant ReluDims&  d   [[buffer(2)]],
    uint                gid [[thread_position_in_grid]])
{
    const uint i = gid;
    if (i >= d.n) {
        return;
    }
    out[i] = max(x[i], 0.0f);
}

// ---- elu: Bark embedded EnCodec activation (alpha = 1) --------------------
struct EluDims {
    uint n;
};

kernel void vokra_elu_f32(
    device const float* x   [[buffer(0)]],
    device float*       out [[buffer(1)]],
    constant EluDims&   d   [[buffer(2)]],
    uint                gid [[thread_position_in_grid]])
{
    const uint i = gid;
    if (i >= d.n) {
        return;
    }
    const float v = x[i];
    out[i] = v > 0.0f ? v : exp(v) - 1.0f;
}

// ---- linear-to-one + abs: HiFTNet F0 predictor head -----------------------
// Input is channel-major [channels, time], matching the preceding Conv1d
// stack. The scalar reduction is deliberately kept in channel order so the
// device route has the same contract as `abs(linear(x)).squeeze(-1)`.
struct LinearAbsDims {
    uint channels;
    uint time;
};

kernel void vokra_linear_abs_f32(
    device const float* x      [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* bias   [[buffer(2)]],
    device float*       out    [[buffer(3)]],
    constant LinearAbsDims& d  [[buffer(4)]],
    uint                 gid   [[thread_position_in_grid]])
{
    const uint t = gid;
    if (t >= d.time) return;
    float acc = bias[0];
    for (uint c = 0u; c < d.channels; ++c) {
        acc += x[c * d.time + t] * weight[c];
    }
    out[t] = fabs(acc);
}

kernel void vokra_linear_tanh_f32(
    device const float* x      [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* bias   [[buffer(2)]],
    device float*       out    [[buffer(3)]],
    constant LinearAbsDims& d  [[buffer(4)]],
    uint                 gid   [[thread_position_in_grid]])
{
    const uint t = gid;
    if (t >= d.time) return;
    float acc = bias[0];
    for (uint c = 0u; c < d.channels; ++c) {
        acc += x[c * d.time + t] * weight[c];
    }
    out[t] = tanh(acc);
}

// ---- nearest-neighbour temporal upsample ----------------------------------
struct NearestUpsampleDims {
    uint channels;
    uint time_in;
    uint time_out;
    uint factor;
};

kernel void vokra_nearest_upsample_f32(
    device const float* x       [[buffer(0)]],
    device float*       out     [[buffer(1)]],
    constant NearestUpsampleDims& d [[buffer(2)]],
    uint2               gid     [[thread_position_in_grid]])
{
    const uint t = gid.x;
    const uint c = gid.y;
    if (c >= d.channels || t >= d.time_out) return;
    out[c * d.time_out + t] = x[c * d.time_in + t / d.factor];
}

// ---- HiFTNet centered Hann STFT --------------------------------------------
// Output layout is channel-major [Re F, frames; Im F, frames]. This is the
// exact layout consumed by HiFTNet's source_downs convolutions. The kernel is
// intentionally a direct DFT: it keeps the centered reflect convention and
// avoids an intermediate host-side frame/FFT representation.
struct HiftStftDims {
    uint time_in;
    uint n_fft;
    uint hop;
    uint frames;
    uint bins;
};

inline int vokra_reflect_index(int i, uint n) {
    if (n <= 1u) return 0;
    const int period = 2 * (int(n) - 1);
    int m = i % period;
    if (m < 0) m += period;
    return m >= int(n) ? period - m : m;
}

kernel void vokra_hift_stft_f32(
    device const float* x       [[buffer(0)]],
    device float*       out     [[buffer(1)]],
    constant HiftStftDims& d    [[buffer(2)]],
    uint2               gid     [[thread_position_in_grid]])
{
    const uint frame = gid.x;
    const uint bin = gid.y;
    if (frame >= d.frames || bin >= d.bins) return;
    const float pi2 = 6.28318530717958647692f;
    float re = 0.0f;
    float im = 0.0f;
    const int center = int(d.n_fft / 2u);
    for (uint k = 0u; k < d.n_fft; ++k) {
        const int logical = int(frame * d.hop + k) - center;
        const int src = vokra_reflect_index(logical, d.time_in);
        const float sample = d.time_in == 0u ? 0.0f : x[uint(src)];
        const float window = d.n_fft == 1u ? 1.0f :
            0.5f - 0.5f * cos(pi2 * float(k) / float(d.n_fft));
        const float value = sample * window;
        const float angle = pi2 * float(bin * k) / float(d.n_fft);
        re += value * cos(angle);
        im -= value * sin(angle);
    }
    out[bin * d.frames + frame] = re;
    out[d.bins * d.frames + bin * d.frames + frame] = im;
}

// ---- HiFTNet logits -> complex spectrum ------------------------------------
// Input logits are [2F, frames] with magnitude logits followed by phase
// logits. Output is [Re F, frames; Im F, frames]. Keeping this as its own
// pass mirrors HiFTResidentOps::complex_from_logits and lets callers choose
// their configured audio-limit clamp separately.
struct HiftComplexDims {
    uint frames;
    uint bins;
};

kernel void vokra_hift_complex_from_logits_f32(
    device const float* logits [[buffer(0)]],
    device float*       out    [[buffer(1)]],
    constant HiftComplexDims& d [[buffer(2)]],
    uint2               gid    [[thread_position_in_grid]])
{
    const uint frame = gid.x;
    const uint channel = gid.y;
    if (frame >= d.frames || channel >= 2u * d.bins) return;
    const uint bin = channel < d.bins ? channel : channel - d.bins;
    const uint base = bin * d.frames + frame;
    if (channel < d.bins) {
        const float log_magnitude = logits[base];
        // Rust's f32::min follows IEEE behavior for NaN (the other operand
        // wins), while f32::sin(inf) is NaN. Spell those cases out so MSL's
        // optimizer cannot choose a different fast-math result.
        const float magnitude = isnan(log_magnitude) ? 100.0f :
            min(exp(log_magnitude), 100.0f);
        const float log_phase = logits[d.bins * d.frames + base];
        const float phase = (isnan(log_phase) || isinf(log_phase)) ?
            (0.0f / 0.0f) : sin(log_phase);
        out[base] = magnitude * cos(phase);
    } else {
        const float log_magnitude = logits[base];
        const float magnitude = isnan(log_magnitude) ? 100.0f :
            min(exp(log_magnitude), 100.0f);
        const float log_phase = logits[d.bins * d.frames + base];
        const float phase = (isnan(log_phase) || isinf(log_phase)) ?
            (0.0f / 0.0f) : sin(log_phase);
        out[d.bins * d.frames + base] = magnitude * sin(phase);
    }
}

// ---- HiFTNet complex spectrum -> centered Hann iSTFT ----------------------
// Input is [Re F, frames; Im F, frames]. This pass intentionally does not
// clamp: HiFT's configured audio limit is a separate elementwise operation.
struct HiftIstftDims {
    uint n_fft;
    uint hop;
    uint frames;
    uint bins;
    uint out_len;
};

kernel void vokra_hift_istft_f32(
    device const float* spectrum [[buffer(0)]],
    device float*       out    [[buffer(1)]],
    constant HiftIstftDims& d  [[buffer(2)]],
    uint                 gid   [[thread_position_in_grid]])
{
    if (gid >= d.out_len) return;
    const float pi2 = 6.28318530717958647692f;
    const uint raw = gid + d.n_fft / 2u;
    float acc = 0.0f;
    float wss = 0.0f;
    for (uint frame = 0u; frame < d.frames; ++frame) {
        const uint start = frame * d.hop;
        if (raw < start || raw >= start + d.n_fft) continue;
        const uint k = raw - start;
        const float window = d.n_fft == 1u ? 1.0f :
            0.5f - 0.5f * cos(pi2 * float(k) / float(d.n_fft));
        float frame_value = 0.0f;
        for (uint freq = 0u; freq < d.n_fft; ++freq) {
            const uint bin = freq < d.bins ? freq : d.n_fft - freq;
            const uint base = bin * d.frames + frame;
            const float re = spectrum[base];
            const float im0 = spectrum[d.bins * d.frames + base];
            const float im = freq < d.bins ? im0 : -im0;
            const float angle = pi2 * float(freq * k) / float(d.n_fft);
            frame_value += re * cos(angle) - im * sin(angle);
        }
        frame_value /= float(d.n_fft);
        acc += frame_value * window;
        wss += window * window;
    }
    if (wss > 1.0e-8f) acc /= wss;
    // Match Rust's f32::clamp NaN behavior for non-finite spectral inputs.
    out[gid] = acc;
}

// ---- tanh: SpeechT5 postnet activation ------------------------------------
struct TanhDims {
    uint n;
};

kernel void vokra_tanh_f32(
    device const float* x   [[buffer(0)]],
    device float*       out [[buffer(1)]],
    constant TanhDims&  d   [[buffer(2)]],
    uint                gid [[thread_position_in_grid]])
{
    const uint i = gid;
    if (i >= d.n) {
        return;
    }
    out[i] = tanh(x[i]);
}

struct LeakyReluDims {
    uint n;
    float slope;
};

kernel void vokra_leaky_relu_f32(
    device const float* x [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant LeakyReluDims& d [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= d.n) return;
    const float v = x[gid];
    out[gid] = v >= 0.0f ? v : d.slope * v;
}

// ---- conv1d: direct convolution (im2col + GEMM equivalent) -------------------
// `kernel` is an MSL reserved word, so the tap count is `kernel_size`. The (c
// outer, kk inner) accumulation order equals the scalar CPU helper: bias is
// the accumulator seed, followed by c/kk products in row-major weight order.
struct Conv1dDims {
    uint in_per_group;
    uint in_len;
    uint out_ch;
    uint kernel_size;
    uint out_len;
    uint stride;
    uint padding;
    uint has_bias;
    uint out_per_group;
    uint dilation;
};

struct ConvTranspose1dDims {
    uint in_ch;
    uint out_ch;
    uint kernel_size;
    uint t_in;
    uint t_out;
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
    const uint group = oc / d.out_per_group;
    const uint k     = d.in_per_group * d.kernel_size;
    const uint wbase = oc * k;
    float acc = d.has_bias != 0u ? bias[oc] : 0.0f;
    for (uint c = 0; c < d.in_per_group; ++c) {
        const uint wc    = wbase + c * d.kernel_size;
        const uint ibase = (group * d.in_per_group + c) * d.in_len;
        for (uint kk = 0; kk < d.kernel_size; ++kk) {
            const uint pos = t * d.stride + kk * d.dilation;
            if (pos >= d.padding && pos < d.padding + d.in_len) {
                acc += weight[wc + kk] * inp[ibase + (pos - d.padding)];
            }
        }
    }
    out[oc * d.out_len + t] = acc;
}

// Direct PyTorch-layout ConvTranspose1d. Keeping the scatter on the device
// avoids the host expansion/readback dance used by the legacy slice.
kernel void vokra_conv_transpose1d_f32(
    device const float* inp    [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* bias   [[buffer(2)]],
    device float*       out    [[buffer(3)]],
    constant ConvTranspose1dDims& d [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint t = gid.x;
    const uint oc = gid.y;
    if (t >= d.t_out || oc >= d.out_ch) return;
    float acc = d.has_bias != 0u ? bias[oc] : 0.0f;
    for (uint ic = 0u; ic < d.in_ch; ++ic) {
        for (uint ti = 0u; ti < d.t_in; ++ti) {
            const int tap = int(t) - int(ti * d.stride) + int(d.padding);
            if (tap >= 0 && uint(tap) < d.kernel_size) {
                const uint wi = (ic * d.out_ch + oc) * d.kernel_size + uint(tap);
                acc += inp[ic * d.t_in + ti] * weight[wi];
            }
        }
    }
    out[oc * d.t_out + t] = acc;
}

// Direct dense/grouped PyTorch-layout Conv2d. The host flattens the output
// channel and height axes into gid.y; one thread computes one output element.
struct Conv2dDims {
    uint in_ch;
    uint in_h;
    uint in_w;
    uint out_ch;
    uint out_h;
    uint out_w;
    uint kernel_h;
    uint kernel_w;
    uint stride_h;
    uint stride_w;
    uint padding_h;
    uint padding_w;
    uint dilation_h;
    uint dilation_w;
    uint in_per_group;
    uint out_per_group;
    uint has_bias;
};

kernel void vokra_conv2d_f32(
    device const float* inp    [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* bias   [[buffer(2)]],
    device float*       out    [[buffer(3)]],
    constant Conv2dDims& d     [[buffer(4)]],
    uint2 gid                  [[thread_position_in_grid]])
{
    const uint ow = gid.x;
    const uint row = gid.y;
    if (ow >= d.out_w || row >= d.out_ch * d.out_h) return;
    const uint oc = row / d.out_h;
    const uint oh = row % d.out_h;
    const uint group = oc / d.out_per_group;
    const uint kernel_plane = d.kernel_h * d.kernel_w;
    const uint input_plane = d.in_h * d.in_w;
    const uint output_plane = d.out_h * d.out_w;
    const uint wbase = oc * d.in_per_group * kernel_plane;
    float acc = d.has_bias != 0u ? bias[oc] : 0.0f;
    for (uint ic_local = 0u; ic_local < d.in_per_group; ++ic_local) {
        const uint ic = group * d.in_per_group + ic_local;
        for (uint kh = 0u; kh < d.kernel_h; ++kh) {
            const uint input_h = oh * d.stride_h + kh * d.dilation_h;
            if (input_h < d.padding_h || input_h >= d.padding_h + d.in_h) continue;
            for (uint kw = 0u; kw < d.kernel_w; ++kw) {
                const uint input_w = ow * d.stride_w + kw * d.dilation_w;
                if (input_w < d.padding_w || input_w >= d.padding_w + d.in_w) continue;
                const uint input_index = ic * input_plane + (input_h - d.padding_h) * d.in_w + (input_w - d.padding_w);
                const uint weight_index = wbase + ic_local * kernel_plane + kh * d.kernel_w + kw;
                acc += inp[input_index] * weight[weight_index];
            }
        }
    }
    out[oc * output_plane + oh * d.out_w + ow] = acc;
}

// Direct dense/grouped PyTorch-layout ConvTranspose2d. The weight layout is
// [in_ch, out_ch/groups, kernel_h, kernel_w]. Output padding is represented in
// the host-derived output dimensions; positions without a source contribution
// retain only the optional bias.
struct ConvTranspose2dDims {
    uint in_ch;
    uint in_h;
    uint in_w;
    uint out_ch;
    uint out_h;
    uint out_w;
    uint kernel_h;
    uint kernel_w;
    uint stride_h;
    uint stride_w;
    uint padding_h;
    uint padding_w;
    uint dilation_h;
    uint dilation_w;
    uint in_per_group;
    uint out_per_group;
    uint has_bias;
};

kernel void vokra_conv_transpose2d_f32(
    device const float* inp    [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* bias   [[buffer(2)]],
    device float*       out    [[buffer(3)]],
    constant ConvTranspose2dDims& d [[buffer(4)]],
    uint2 gid                  [[thread_position_in_grid]])
{
    const uint ow = gid.x;
    const uint row = gid.y;
    if (ow >= d.out_w || row >= d.out_ch * d.out_h) return;
    const uint oc = row / d.out_h;
    const uint oh = row % d.out_h;
    const uint group = oc / d.out_per_group;
    const uint oc_local = oc % d.out_per_group;
    const uint kernel_plane = d.kernel_h * d.kernel_w;
    const uint input_plane = d.in_h * d.in_w;
    const uint output_plane = d.out_h * d.out_w;
    float acc = d.has_bias != 0u ? bias[oc] : 0.0f;
    for (uint ic_local = 0u; ic_local < d.in_per_group; ++ic_local) {
        const uint ic = group * d.in_per_group + ic_local;
        for (uint kh = 0u; kh < d.kernel_h; ++kh) {
            const uint numerator_h = oh + d.padding_h;
            const uint tap_h = kh * d.dilation_h;
            if (numerator_h < tap_h) continue;
            const uint source_h = numerator_h - tap_h;
            if (source_h % d.stride_h != 0u || source_h / d.stride_h >= d.in_h) continue;
            for (uint kw = 0u; kw < d.kernel_w; ++kw) {
                const uint numerator_w = ow + d.padding_w;
                const uint tap_w = kw * d.dilation_w;
                if (numerator_w < tap_w) continue;
                const uint source_w = numerator_w - tap_w;
                if (source_w % d.stride_w != 0u || source_w / d.stride_w >= d.in_w) continue;
                const uint input_index = ic * input_plane + (source_h / d.stride_h) * d.in_w + source_w / d.stride_w;
                const uint weight_index = (ic * d.out_per_group + oc_local) * kernel_plane + kh * d.kernel_w + kw;
                acc += inp[input_index] * weight[weight_index];
            }
        }
    }
    out[oc * output_plane + oh * d.out_w + ow] = acc;
}

struct Pad1dDims {
    uint channels;
    uint time_in;
    uint time_out;
    uint left;
    uint right;
    uint mode; // 0 = reflect, 1 = replicate
};

kernel void vokra_pad1d_f32(
    device const float* inp [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant Pad1dDims& d [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint t = gid.x;
    const uint c = gid.y;
    if (c >= d.channels || t >= d.time_out) return;
    int logical = int(t) - int(d.left);
    int src;
    if (d.mode == 0u) {
        src = logical < 0 ? -logical : (logical >= int(d.time_in) ? 2 * int(d.time_in) - 2 - logical : logical);
    } else {
        src = logical < 0 ? 0 : (logical >= int(d.time_in) ? int(d.time_in) - 1 : logical);
    }
    out[c * d.time_out + t] = inp[c * d.time_in + uint(src)];
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

// ---- scale_norm: released MossFormer2 ScaleNorm -----------------------------
// out[r,c] = inp[r,c] / max(sqrt(sum(inp[r,:]^2))*scale, eps) * gain.
// `scale = cols^-0.5` is computed once on the Rust host and passed as f32 so
// the CPU and Metal paths consume the identical rounded constant.
struct ScaleNormDims {
    uint  rows;
    uint  cols;
    float scale;
    float eps;
    float gain;
};

kernel void vokra_scale_norm_f32(
    device const float*     inp [[buffer(0)]],
    device float*           out [[buffer(1)]],
    constant ScaleNormDims& d   [[buffer(2)]],
    uint                    gid [[thread_position_in_grid]])
{
    const uint r = gid;
    if (r >= d.rows) {
        return;
    }
    const uint base = r * d.cols;
    float ss = 0.0f;
    for (uint c = 0; c < d.cols; ++c) {
        const float v = inp[base + c];
        ss += v * v;
    }
    const float denominator = max(sqrt(ss) * d.scale, d.eps);
    for (uint c = 0; c < d.cols; ++c) {
        out[base + c] = inp[base + c] / denominator * d.gain;
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

// ---- OUVE-SDE score-model sampler primitives -------------------------------
// These element-wise kernels mirror `vokra_ops::ouve_sde` and the pinned
// `sp-uhh/sgmse` source.  The state, conditioning signal, score, noise, and
// both outputs stay in device-resident buffers; callers can chain predictor
// and corrector steps without a host readback between iterations.
struct OuvEReverseDims {
    uint n;
    float theta;
    float sigma_min;
    float sigma_max;
    float t;
    float step;
    uint probability_flow;
};

kernel void vokra_ouve_reverse_diffusion_f32(
    device const float* x        [[buffer(0)]],
    device const float* y        [[buffer(1)]],
    device const float* score    [[buffer(2)]],
    device const float* noise    [[buffer(3)]],
    device float*       out      [[buffer(4)]],
    device float*       out_mean [[buffer(5)]],
    constant OuvEReverseDims& d  [[buffer(6)]],
    uint gid                     [[thread_position_in_grid]])
{
    if (gid >= d.n) return;
    const float log_ratio = log(d.sigma_max / d.sigma_min);
    const float diffusion = d.sigma_min * exp(log_ratio * d.t) * sqrt(2.0f * log_ratio);
    const float score_scale = diffusion * diffusion * (d.probability_flow != 0u ? 0.5f : 1.0f);
    const float forward_drift = d.theta * (y[gid] - x[gid]);
    const float reverse_increment = (forward_drift - score_scale * score[gid]) * d.step;
    const float mean = x[gid] - reverse_increment;
    const float noise_scale = d.probability_flow != 0u ? 0.0f : diffusion * sqrt(d.step);
    out_mean[gid] = mean;
    out[gid] = mean + noise_scale * noise[gid];
}

struct OuvEAnnealedDims {
    uint n;
    float theta;
    float sigma_min;
    float sigma_max;
    float t;
    float snr;
};

kernel void vokra_ouve_annealed_langevin_f32(
    device const float* x        [[buffer(0)]],
    device const float* score    [[buffer(1)]],
    device const float* noise    [[buffer(2)]],
    device float*       out      [[buffer(3)]],
    device float*       out_mean [[buffer(4)]],
    constant OuvEAnnealedDims& d [[buffer(5)]],
    uint gid                     [[thread_position_in_grid]])
{
    if (gid >= d.n) return;
    const float log_ratio = log(d.sigma_max / d.sigma_min);
    const float numerator = d.sigma_min * d.sigma_min * exp(-2.0f * d.theta * d.t)
        * (exp(2.0f * (d.theta + log_ratio) * d.t) - 1.0f) * log_ratio;
    const float variance = numerator / (d.theta + log_ratio);
    const float step_size = 2.0f * (d.snr * sqrt(variance)) * (d.snr * sqrt(variance));
    const float mean = x[gid] + step_size * score[gid];
    out_mean[gid] = mean;
    out[gid] = mean + sqrt(2.0f * step_size) * noise[gid];
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

// ---- M3-06 T14 mimi_rvq gather + FP32 fold ----------------------------------
//
// Semantics identical to `vokra_ops::mimi_rvq::rvq_fold_core` (the shape-
// generic core behind `mimi_rvq_decode`):
//
//     out[t, d] = Σ_cb tables[cb].row(codes[t, cb])[d]
//
// * `codes`       — [time, n_codebooks] row-major u32 codebook indices.
// * `tables`      — [n_codebooks, codebook_size, d_model] row-major FP32,
//                   codebook `cb` starts at `cb * codebook_size * d_model`.
// * `out`         — [time, d_model] row-major FP32.
//
// Naive layout per the module docs on `vokra_ops::mimi_rvq`: one thread per
// output element `(t, d)`, each iterating over `n_codebooks` gather-and-add
// steps. FP32 accumulator throughout — the "BF16 mantissa loss is the real
// problem" note in CLAUDE.md applies to every codec-side fold. Index bound
// checks are done on the host before dispatch (FR-EX-08 — the kernel itself
// has no per-element bound check, so silent OOB reads would be the failure
// mode without host-side validation).
//
// Grid: `(d_model, time)` — same 2D launch as the GEMM kernel; the shader
// guards the ragged tail against both bounds. Threadgroup 16x16 (the
// `grid_2d` default), which is sized for the canonical Mimi
// [d_model=512, time≈750] envelope and small enough to not waste threads on
// the tiny [d_model=5, time=3] parity fixtures.
struct MimiRvqDims {
    uint n_codebooks;
    uint codebook_size;
    uint d_model;
    uint time;
};

kernel void vokra_mimi_rvq_gather_fold_f32(
    device const uint*      codes  [[buffer(0)]],
    device const float*     tables [[buffer(1)]],
    device float*           out    [[buffer(2)]],
    constant MimiRvqDims&   d      [[buffer(3)]],
    uint2                   gid    [[thread_position_in_grid]])
{
    const uint t     = gid.y;
    const uint delem = gid.x;
    if (t >= d.time || delem >= d.d_model) {
        return;
    }
    const uint code_base = t * d.n_codebooks;
    const uint cb_stride = d.codebook_size * d.d_model;
    // FP32 accumulator — matches `rvq_fold_core`'s `out[..]` FP32 fold order
    // (bit-identical when the same operands are added in the same sequence;
    // MSL fast-math may re-associate, so the parity bound is FP32 GEMV-scale
    // rather than a bit-for-bit assertion).
    float acc = 0.0f;
    for (uint cb = 0; cb < d.n_codebooks; ++cb) {
        const uint idx       = codes[code_base + cb];
        const uint table_off = cb * cb_stride + idx * d.d_model + delem;
        acc += tables[table_off];
    }
    out[t * d.d_model + delem] = acc;
}

// ---- M4-04 dac_rvq gather + factorized projection + FP32 fold ---------------
//
// Semantics identical to `vokra_ops::dac_rvq::dac_rvq_decode` (the DAC
// factorized RVQ decode: each quantizer owns a low-dim codebook plus a per-
// quantizer 1x1 projection with bias applied *before* the residual sum):
//
//     out[t, d] = Σ_cb (
//         proj_biases[cb, d]
//       + Σ_c proj_weights[cb, d, c] * low_tables[cb, codes[t, cb], c]
//     )
//
// * `codes`         — [time, n_codebooks] row-major u32 codebook indices.
// * `low_tables`    — [n_codebooks, codebook_size, codebook_dim] row-major FP32;
//                     codebook `cb` starts at `cb * codebook_size *
//                     codebook_dim`.
// * `proj_weights`  — [n_codebooks, d_model, codebook_dim] row-major FP32;
//                     quantizer `cb`'s W row `d` starts at
//                     `cb * d_model * codebook_dim + d * codebook_dim`.
// * `proj_biases`   — [n_codebooks, d_model] row-major FP32; quantizer `cb`'s
//                     bias for output `d` at `cb * d_model + d`.
// * `out`           — [time, d_model] row-major FP32.
//
// Same naive one-thread-per-output-element layout as `mimi_rvq`, extended with
// the per-quantizer GEMV + bias fold. Host-side range checks on `codes[..]`
// (FR-EX-08 — the kernel does no per-element bound check). FP32 accumulator
// throughout — the "BF16 mantissa loss is the real problem" audio-dialect rule
// applies (CLAUDE.md).
//
// Grid: `(d_model, time)` — same 2D launch as mimi_rvq. Threadgroup 16x16 (the
// `grid_2d` default). The ragged tail is guarded against both bounds.
struct DacRvqDims {
    uint n_codebooks;
    uint codebook_size;
    uint codebook_dim;
    uint d_model;
    uint time;
};

kernel void vokra_dac_rvq_gather_project_fold_f32(
    device const uint*      codes        [[buffer(0)]],
    device const float*     low_tables   [[buffer(1)]],
    device const float*     proj_weights [[buffer(2)]],
    device const float*     proj_biases  [[buffer(3)]],
    device float*           out          [[buffer(4)]],
    constant DacRvqDims&    d            [[buffer(5)]],
    uint2                   gid          [[thread_position_in_grid]])
{
    const uint t     = gid.y;
    const uint delem = gid.x;
    if (t >= d.time || delem >= d.d_model) {
        return;
    }
    const uint code_base   = t * d.n_codebooks;
    const uint low_stride  = d.codebook_size * d.codebook_dim;
    const uint w_stride    = d.d_model * d.codebook_dim;
    // FP32 accumulator — matches `vokra_ops::dac_rvq::dac_rvq_decode`'s
    // `out[..]` FP32 fold order over quantizers. MSL fast-math may re-associate
    // the inner (W · low) dot product, so the parity bound is FP32 GEMV-scale
    // rather than bit-for-bit.
    float acc = 0.0f;
    for (uint cb = 0; cb < d.n_codebooks; ++cb) {
        const uint idx     = codes[code_base + cb];
        const uint low_off = cb * low_stride + idx * d.codebook_dim;
        const uint w_off   = cb * w_stride + delem * d.codebook_dim;
        float y = proj_biases[cb * d.d_model + delem];
        for (uint c = 0; c < d.codebook_dim; ++c) {
            y += proj_weights[w_off + c] * low_tables[low_off + c];
        }
        acc += y;
    }
    out[t * d.d_model + delem] = acc;
}

// ---- M4-16 wavtokenizer_vq single-codebook gather --------------------------
//
// Semantics identical to `vokra_ops::fsq_codec::wavtokenizer_vq_decode` — the
// FSQ family's single-stage large-vocab lookup (FR-OP-31, *separate subgraph
// from the RVQ family* — module docs in `vokra_ops::fsq_codec`):
//
//     out[t, d] = codebook_table[codes[t]].row[d]
//
// * `codes` — [time] u32 codebook indices (one code per timestep — the RVQ
//             family's [time, n_codebooks] layout does NOT apply here; the
//             signature-level distinction that the CPU op takes a *singular*
//             `&CodebookTable` mirrors here as a single flat table buffer).
// * `table` — [vocab_size, d_model] row-major FP32 codebook.
// * `out`   — [time, d_model] row-major FP32.
//
// Pure gather — no residual sum, no per-dim decompose, no arithmetic (upstream
// decodes via a raw `F.embedding` lookup — module docs). CPU vs GPU is
// bit-identical (no fold to re-associate), so the parity test is tight; the
// atol budget is kept in sync with the sibling mimi/dac kernels for
// consistency (both would trivially pass at atol=0).
//
// Host-side per-index bound check on `codes[..]` (FR-EX-08 — the kernel does
// no per-element bound check; the caller validates upstream of the dispatch).
//
// Grid: `(d_model, time)` — same 2D launch pattern as mimi_rvq / dac_rvq for a
// consistent launch geometry across the codec family; threadgroup 16x16 (the
// `grid_2d` default), which is small enough not to waste threads on the tiny
// parity fixtures and big enough to keep the canonical WavTokenizer shape
// (vocab_size=4096, d_model=512) memory-bound-optimal.
struct WavTokenizerVqDims {
    uint vocab_size;
    uint d_model;
    uint time;
};

kernel void vokra_wavtokenizer_vq_gather_f32(
    device const uint*              codes  [[buffer(0)]],
    device const float*             table  [[buffer(1)]],
    device float*                   out    [[buffer(2)]],
    constant WavTokenizerVqDims&    d      [[buffer(3)]],
    uint2                           gid    [[thread_position_in_grid]])
{
    const uint t     = gid.y;
    const uint delem = gid.x;
    if (t >= d.time || delem >= d.d_model) {
        return;
    }
    const uint idx = codes[t];
    out[t * d.d_model + delem] = table[idx * d.d_model + delem];
}

// ---- M4-16 xcodec2_fsq grid-decompose + optional GEMV ----------------------
//
// Semantics identical to `vokra_ops::fsq_codec::xcodec2_fsq_decode` — the FSQ
// family's grid-based decode (FR-OP-31 single-stage GEMV bound; no codebook
// tensor, implicit per-dimension grid + one out-projection GEMV per timestep):
//
//   For each timestep t:
//     rem = codes[t]
//     for k in 0..n_dims:
//       level_index[k] = rem % levels[k]
//       rem            = rem / levels[k]
//       half_width[k]  = levels[k] / 2                     (integer division)
//       grid[k]        = (level_index[k] − half_width[k]) / half_width[k]
//     out[t, o] = has_projection
//                 ? proj_bias[o] + Σ_k proj_weight[o, k] · grid[k]
//                 : grid[o]                                (Identity requires
//                                                          d_model == n_dims)
//
// * `codes`       — [time] u32 FSQ indices, each < Π levels.
// * `levels`      — [n_dims] u32 mixed-radix bases (each ≥ 2 — validated on
//                   the host; the kernel assumes this so `half_width ≥ 1`,
//                   preventing a divide-by-zero in the grid formula).
// * `proj_weight` — [d_model, n_dims] row-major FP32, or a dummy [0.0] buffer
//                   when `has_projection == 0`.
// * `proj_bias`   — [d_model] FP32, or a dummy [0.0] buffer when
//                   `has_projection == 0`.
// * `out`         — [time, d_model] row-major FP32.
//
// FP32 accumulator throughout (the "BF16 mantissa loss is the real problem"
// audio-dialect rule; CLAUDE.md). MSL fast-math may re-associate the inner
// `Σ_k proj_weight[o, k] · grid[k]` GEMV, so the parity bound is FP32
// GEMV-scale rather than bit-for-bit — the same 5e-4 budget the sibling
// mimi_rvq / dac_rvq kernels use.
//
// Grid: `(d_model, time)` — same 2D launch pattern as mimi_rvq / dac_rvq for a
// consistent codec-family launch geometry. Each thread recomputes the whole
// `grid[0..n_dims]` (cheap: n_dims = 8 on the released X-Codec 2) rather than
// staging it through threadgroup memory — the FSQ decode is single-stage
// GEMV bound (module docs), the n_dims scan is O(n_dims) and cache-friendly
// on the register file. Identity path (`has_projection == 0`) walks the same
// scan and takes the value at `k == delem` (mixed-radix decompose can't be
// short-circuited before dim `delem` — the `rem` state carries forward).
struct Xcodec2FsqDims {
    uint d_model;         // output width (= FsqOutProj::d_model, or = n_dims for Identity)
    uint n_dims;          // len(levels) (= X-Codec 2's 8)
    uint time;
    uint has_projection;  // 0 = Identity (d_model == n_dims), 1 = GEMV
};

kernel void vokra_xcodec2_fsq_decode_f32(
    device const uint*          codes        [[buffer(0)]],
    device const uint*          levels       [[buffer(1)]],
    device const float*         proj_weight  [[buffer(2)]],
    device const float*         proj_bias    [[buffer(3)]],
    device float*               out          [[buffer(4)]],
    constant Xcodec2FsqDims&    d            [[buffer(5)]],
    uint2                       gid          [[thread_position_in_grid]])
{
    const uint t     = gid.y;
    const uint delem = gid.x;
    if (t >= d.time || delem >= d.d_model) {
        return;
    }

    if (d.has_projection != 0u) {
        // GEMV path: decompose `codes[t]` onto the grid and do a single
        // Linear(n_dims → d_model)+bias dot product per output column.
        uint rem = codes[t];
        float acc = proj_bias[delem];
        const uint w_base = delem * d.n_dims;
        for (uint k = 0; k < d.n_dims; ++k) {
            const uint level = levels[k];
            const uint level_index = rem % level;
            rem /= level;
            const uint half_width = level / 2u;  // >= 1: host validates level >= 2
            const float grid_val =
                (float)((int)level_index - (int)half_width) / (float)half_width;
            acc += proj_weight[w_base + k] * grid_val;
        }
        out[t * d.d_model + delem] = acc;
    } else {
        // Identity path (d_model == n_dims): each thread walks the mixed-
        // radix decompose from dim 0 up to `delem` (the `rem` state carries
        // forward — dim `delem`'s level_index depends on rem after dividing
        // by every earlier level). Cost is O(delem+1) integer ops per thread;
        // for the released X-Codec 2 Identity case is unreachable
        // (`requires_projection = true`; d_model=2048 != n_dims=8), so this
        // path exists for the small parity fixtures and callers whose codebook
        // dim equals the FSQ n_dims.
        uint rem = codes[t];
        for (uint k = 0; k <= delem; ++k) {
            const uint level = levels[k];
            const uint level_index = rem % level;
            if (k == delem) {
                const uint half_width = level / 2u;
                out[t * d.d_model + delem] =
                    (float)((int)level_index - (int)half_width) / (float)half_width;
                return;
            }
            rem /= level;
        }
    }
}

// ---- Vocoder Metal wave WF5 snac_decode (2026-08-13) ------------------------
//
// Semantics identical to `vokra_ops::snac_decode::SnacDecoder::decode` (the
// upstream `ResidualVectorQuantize.from_codes` algorithm, `hubertsiuzdak/snac
// /blob/main/snac/vq.py` L61-71):
//
//     For each active stage s:
//       z_p_s = codebooks[s].row(codes[s][t_stage])                # embed lookup
//       z_q_s = W_s @ z_p_s + b_s                                   # WNConv1d(codebook_dim → d_model)
//       z_q_s = repeat_interleave(z_q_s, stride=strides[s], dim=-1) # temporal upsample
//       z_q  += z_q_s                                                # residual sum
//
// SNAC is a **hierarchical / multi-scale** residual VQ: unlike Mimi / DAC where
// every quantizer shares the same time axis, SNAC's `k`th stage runs at frame
// rate `base / vq_strides[k]`. The 24 kHz variant used by Orpheus and Maya1
// has `vq_strides = [4, 2, 1]` giving per-stage rates ~12 / 23 / 47 Hz.
//
// * `codes`         — u32 codebook indices, concatenated across the 3 stages:
//                     `codes[stage_offsets[s] + t_stage]` for stage `s`,
//                     stage frame `t_stage in 0..codes[s].len()`. Every stage
//                     must satisfy `codes[s].len() * strides[s] == t_expanded`
//                     (host-validated, FR-EX-08).
// * `codebooks`     — [3, codebook_size, codebook_dim] row-major FP32; stage
//                     `s` starts at `s * codebook_size * codebook_dim`. The
//                     three stages share `codebook_size` and `codebook_dim`
//                     (validated on the host).
// * `proj_weights`  — [3, d_model, codebook_dim] row-major FP32; stage `s`'s
//                     W row `o` starts at `s * d_model * codebook_dim + o *
//                     codebook_dim`.
// * `proj_biases`   — [3, d_model] row-major FP32; stage `s`'s bias for
//                     output `o` at `s * d_model + o`.
// * `out`           — [t_expanded, d_model] row-major FP32.
//
// For each output element `(t_out, d_out)` and each stage `s`, look up the
// stage frame `t_stage = t_out / strides[s]`, then compute `W_s @ low_s + b_s`
// for that stage and accumulate into the FP32 output. The temporal upsample
// (`repeat_interleave(stride)`) is baked into `t_stage = t_out / stride` so
// contiguous output timesteps within one stage frame share the same projected
// row — bit-identical to the CPU fold's `t_start..t_start + stride` inner
// loop.
//
// FP32 accumulator throughout (the "BF16 mantissa loss is the real problem"
// audio-dialect rule; CLAUDE.md). MSL fast-math may re-associate the inner
// `Σ_c W[o, c] · low[c]` dot product, so the parity bound is FP32 GEMV-scale
// rather than bit-for-bit — the same 5e-4 budget the sibling mimi_rvq /
// dac_rvq / fsq_codec / snake_activation kernels use.
//
// Host-side per-index bound check on `codes[..]` (FR-EX-08 — the kernel does
// no per-element bound check; the caller validates upstream of the dispatch).
//
// Grid: `(d_model, t_expanded)` — same 2D launch pattern as the sibling
// mimi_rvq / dac_rvq kernels for a consistent codec-family launch geometry.
// Threadgroup 16x16 (the `grid_2d` default); the ragged tail is guarded
// against both bounds.
//
// The published family has three stages at 24 kHz and four at 44.1 kHz. The
// ABI reserves four slots and `n_stages` selects the active prefix.
struct SnacDecodeDims {
    uint d_model;
    uint codebook_dim;
    uint codebook_size;
    uint t_expanded;
    uint n_stages;
    // Per-stage temporal strides (SNAC 24 kHz canonical = [4, 2, 1]).
    uint strides[4];
    // Start of each stage in the flat `codes` buffer. `codes[stage_offsets[s]
    // + t_stage]` for stage `s`. stage_offsets[0] is always 0; stage_offsets[1]
    // = len(codes[0]); stage_offsets[2] = len(codes[0]) + len(codes[1]).
    uint stage_offsets[4];
};

kernel void vokra_snac_decode_f32(
    device const uint*         codes         [[buffer(0)]],
    device const float*        codebooks     [[buffer(1)]],
    device const float*        proj_weights  [[buffer(2)]],
    device const float*        proj_biases   [[buffer(3)]],
    device float*              out           [[buffer(4)]],
    constant SnacDecodeDims&   d             [[buffer(5)]],
    uint2                      gid           [[thread_position_in_grid]])
{
    const uint t_out = gid.y;
    const uint d_out = gid.x;
    if (t_out >= d.t_expanded || d_out >= d.d_model) {
        return;
    }
    const uint cb_stride = d.codebook_size * d.codebook_dim;
    const uint w_stride  = d.d_model * d.codebook_dim;
    // FP32 accumulator — matches `SnacDecoder::decode`'s outer stage fold in
    // left-to-right stage order. MSL fast-math may re-associate the inner
    // GEMV over `codebook_dim`, so the parity bound is FP32 GEMV-scale
    // rather than bit-for-bit.
    float acc = 0.0f;
    for (uint s = 0; s < d.n_stages; ++s) {
        const uint stride_s = d.strides[s];
        const uint t_stage  = t_out / stride_s;
        const uint idx      = codes[d.stage_offsets[s] + t_stage];
        const uint low_off  = s * cb_stride + idx * d.codebook_dim;
        const uint w_off    = s * w_stride + d_out * d.codebook_dim;
        float y = proj_biases[s * d.d_model + d_out];
        for (uint c = 0; c < d.codebook_dim; ++c) {
            y += proj_weights[w_off + c] * codebooks[low_off + c];
        }
        acc += y;
    }
    out[t_out * d.d_model + d_out] = acc;
}

// ---- Vocoder Metal wave WF2 snake activation (2026-08-13) -------------------
//
// Semantics identical to `vokra_ops::snake::snake_activation_f32` (the
// stateless out-of-place free function that mirrors
// `vokra_ops::hiftnet::Snake::forward_in_place` under `alpha_logscale = false`
// and the private `kokoro::nn::snake_activation` helper in vokra-models):
//
//     out[c, t] = x[c, t] + (1 / (alpha[c] + eps)) * sin(alpha[c] * x[c, t])^2
//
// with `eps = 1.0e-9f` (upstream `no_div_by_zero` — `activations.py:97` and
// every downstream port: HiFTNet, BigVGAN, Kokoro-82M).
//
// * `x`     — [channels, time] row-major FP32 (channel-outer).
// * `alpha` — [channels] FP32 per-channel scale (already the "effective" α;
//             the `alpha_logscale = true` case is expected to have the
//             `exp(alpha_raw)` applied on the host by the converter, matching
//             the way the CPU free function is called — no in-kernel branch).
// * `out`   — [channels, time] row-major FP32.
//
// Trivially element-wise (no reduction, no gather) — CPU and GPU agree
// bit-for-bit when the MSL fast-math `sin` matches the host `f32::sin` for
// the tested inputs. The parity bound is `atol ≤ 5e-4` to match the sibling
// codec-family bound (`mimi_rvq`, `dac_rvq`, `fsq_codec`); a transcendental
// re-implementation gap between MSL's `sin` and Rust's `f32::sin` would
// still fall well inside 5e-4 for finite inputs.
//
// Precision: FP32 throughout (BF16 mantissa loss is the real problem —
// CLAUDE.md audio-dialect activation attribute rule of thumb). The kernel
// takes an FP32 `alpha` and an FP32 `x`; the intermediate `sin(a*v)` and the
// `inv_a * s * s` term are computed in FP32.
//
// Grid: `(time, channels)` 2-D dispatch — same 16x16 threadgroup shape as
// the codec kernels. `time` on the fast axis (grid.x) matches the row-major
// stride and keeps adjacent threads reading adjacent floats. The ragged
// tail is guarded against both bounds.
struct SnakeActivationDims {
    uint channels;
    uint time;
};

kernel void vokra_snake_activation_f32(
    device const float*                x     [[buffer(0)]],
    device const float*                alpha [[buffer(1)]],
    device float*                      out   [[buffer(2)]],
    constant SnakeActivationDims&      d     [[buffer(3)]],
    uint2                              gid   [[thread_position_in_grid]])
{
    const uint t = gid.x;
    const uint c = gid.y;
    if (c >= d.channels || t >= d.time) {
        return;
    }
    const float a     = alpha[c];
    const float inv_a = 1.0f / (a + 1.0e-9f);
    const uint  idx   = c * d.time + t;
    const float v     = x[idx];
    const float s     = sin(a * v);
    out[idx] = v + inv_a * s * s;
}

// ---- Vocoder Metal wave common vocoder primitive: snake_beta ---------------
//
// Semantics identical to `vokra_ops::snake_beta_f32` — the two-vector
// SnakeBeta closed form consumed by the BigVGAN family (upstream
// `activations.py:62-114`, MIT / NVIDIA):
//
//     out[c, t] = x[c, t] + (1 / (beta[c] + eps)) * sin(alpha[c] * x[c, t])^2
//
// with `eps = 1.0e-9f` (upstream `no_div_by_zero`, matching the sibling
// snake_activation kernel).
//
// Distinct from `vokra_snake_activation_f32` because SnakeBeta separates
// frequency (`alpha`) from magnitude (`beta`) — the plain-Snake kernel
// couples both to a single per-channel alpha and would silently squash the
// beta axis. Kept as its own MSL kernel so the coverage seam can dispatch
// SnakeBeta directly (BigVGAN's terminal activation + every AMP block that
// asks for SnakeBeta).
//
// * `x`     — [channels, time] row-major FP32 (channel-outer).
// * `alpha` — [channels] FP32 per-channel frequency scale (already effective;
//             the `alpha_logscale = true` case pre-exponentiates on the host,
//             mirroring the CPU free function's contract).
// * `beta`  — [channels] FP32 per-channel magnitude scale (same convention).
// * `out`   — [channels, time] row-major FP32.
//
// Trivially element-wise (no reduction, no gather) — CPU and GPU agree
// within the FP32 `sin` transcendental gap. The parity bound is
// `atol ≤ 5e-4` (same sibling codec-family / snake_activation envelope).
//
// Grid: `(time, channels)` 2-D dispatch — same 16×16 threadgroup shape as
// the sibling snake_activation kernel. `time` on the fast axis (grid.x)
// matches the row-major stride and keeps adjacent threads reading adjacent
// floats. The ragged tail is guarded against both bounds.
struct SnakeBetaDims {
    uint channels;
    uint time;
};

kernel void vokra_snake_beta_f32(
    device const float*        x     [[buffer(0)]],
    device const float*        alpha [[buffer(1)]],
    device const float*        beta  [[buffer(2)]],
    device float*              out   [[buffer(3)]],
    constant SnakeBetaDims&    d     [[buffer(4)]],
    uint2                      gid   [[thread_position_in_grid]])
{
    const uint t = gid.x;
    const uint c = gid.y;
    if (c >= d.channels || t >= d.time) {
        return;
    }
    const float a     = alpha[c];
    const float b     = beta[c];
    const float inv_b = 1.0f / (b + 1.0e-9f);
    const uint  idx   = c * d.time + t;
    const float v     = x[idx];
    const float s     = sin(a * v);
    out[idx] = v + inv_b * s * s;
}

// ---- Vocoder Metal wave common vocoder primitive: sinegen_deterministic ----
//
// Semantics identical to `vokra_ops::sinegen_deterministic_f32` — the
// deterministic-only path of `vokra_ops::nsf::SineGen::forward` (upstream
// CosyVoice `cosyvoice/hifigan/generator.py:200-214`, `NsfEntropy::
// Deterministic`). Under that mode the per-harmonic phase and Gaussian
// noise both collapse to zero, and the CPU forward reduces to (for
// `i ∈ [0, harmonic_num]`, `j ∈ [0, T)`):
//
//     cs_i(j) = cs_i(j-1) + f0[j] * (i+1) / samp_rate       (per-harmonic cumsum)
//     theta   = 2π * (cs_i(j) - floor(cs_i(j)))              (cs mod 1)
//     sine    = sine_amp * sin(theta)
//     uv      = f0[j] > voiced_threshold ? 1.0 : 0.0
//     out[j * (H+1) + i] = sine * uv                         (transposed layout)
//
// # Per-harmonic sequential cumsum → one thread per harmonic
//
// The cumsum is a sequential dependency across time (`cs_i(j)` depends on
// `cs_i(j-1)`), so we can only parallelise across harmonics. The kernel
// launches ONE thread per harmonic (grid.x = harmonic_num + 1, grid.y = 1);
// each thread walks the full time axis in-kernel, accumulating its own cs
// in a private register. Every write is to the transposed layout directly
// (`out[j * h1 + i] = sine * uv`), so no scratch buffer is needed and no
// follow-up transpose pass is dispatched.
//
// The `cs += f0[j] * harmonic_gain` accumulation matches the CPU forward's
// per-harmonic reduction order exactly, so under MSL fast-math the only
// bit-level source of divergence is `sin(theta)` — well inside the
// `atol ≤ 5e-4` codec-family bound for finite inputs.
//
// * `f0`               — [T] FP32 fundamental frequency per sample.
// * `out`              — [T * (H+1)] FP32 row-major (time-outer / harmonic-
//                        inner, upstream `sine_wavs.transpose(1, 2)`).
// * `samp_rate`        — audio sample rate (Hz).
// * `harmonic_num`     — number of harmonics beyond the fundamental.
// * `sine_amp`         — sinusoid amplitude scale.
// * `voiced_threshold` — F0 threshold above which a frame is voiced.
// * `t`                — number of input timesteps (== f0.len()).
//
// Grid: `(H+1, 1)` 1-D dispatch. Threadgroup 256; the kernel's `i >= h1`
// guard covers the ragged tail (H+1 is typically ~1-10, far below 256, so
// most threads in the block early-return — that is acceptable for a
// once-per-utterance vocoder-front op).
struct SinegenDeterministicDims {
    uint t;
    uint h1;                 // harmonic_num + 1
    float samp_rate_f;
    float sine_amp;
    float voiced_threshold;
};

kernel void vokra_sinegen_deterministic_f32(
    device const float*                  f0    [[buffer(0)]],
    device float*                        out   [[buffer(1)]],
    constant SinegenDeterministicDims&   d     [[buffer(2)]],
    uint                                 gid   [[thread_position_in_grid]])
{
    const uint i = gid;
    if (i >= d.h1) {
        return;
    }
    const float harmonic_gain = (float(i) + 1.0f) / d.samp_rate_f;
    const float two_pi = 6.28318530717958647692f;
    float cs = 0.0f;
    for (uint j = 0; j < d.t; ++j) {
        const float f0_j = f0[j];
        cs += f0_j * harmonic_gain;
        const float modded = cs - floor(cs);
        const float theta  = two_pi * modded;
        const float sine   = d.sine_amp * sin(theta);
        const float uv     = (f0_j > d.voiced_threshold) ? 1.0f : 0.0f;
        // Transposed layout: (j, i) → out[j * h1 + i].
        out[j * d.h1 + i] = sine * uv;
    }
}

// Same deterministic SineGen arithmetic as above, with the channel-major
// `[H+1, T]` layout required by HiFTResidentOps. The legacy kernel remains
// time-major for its existing host-facing API; this separate entry point
// prevents callers from reinterpreting bytes under a different layout.
kernel void vokra_sinegen_deterministic_channel_major_f32(
    device const float*                  f0    [[buffer(0)]],
    device float*                        out   [[buffer(1)]],
    constant SinegenDeterministicDims&   d     [[buffer(2)]],
    uint                                 gid   [[thread_position_in_grid]])
{
    const uint i = gid;
    if (i >= d.h1) return;
    const float harmonic_gain = (float(i) + 1.0f) / d.samp_rate_f;
    const float two_pi = 6.28318530717958647692f;
    float cs = 0.0f;
    for (uint j = 0u; j < d.t; ++j) {
        const float f0_j = f0[j];
        cs += f0_j * harmonic_gain;
        const float modded = cs - floor(cs);
        const float theta = two_pi * modded;
        const float sine = d.sine_amp * sin(theta);
        const float uv = f0_j > d.voiced_threshold ? 1.0f : 0.0f;
        out[i * d.t + j] = sine * uv;
    }
}

// ---- Vocoder Metal wave common vocoder primitive: anti_aliased_upsample ---
//
// Semantics identical to `vokra_ops::anti_aliased_upsample_f32` — polyphase
// decomposition of "upsample by `ratio` then convolve with a causal FIR
// low-pass". The Kaiser-window filter design lives on the host (see the
// vokra-ops module docstring); this kernel consumes the already-designed
// `kernel` taps and does the per-timestep multiply-add.
//
//     out[c, t_out] = Σ_j kernel[j * ratio + r] * x[c, t - j]
//
// with `t = t_out / ratio`, `r = t_out % ratio` (polyphase branch), and the
// sum running over every `j >= 0` where `t - j >= 0` and
// `j * ratio + r < taps` (the `k_idx >= taps { break; }` guard on the CPU
// side, transliterated into a bounded C-style `for` loop here).
//
// # No reduction across taps → FMA divergence is bounded
//
// Under MSL fast-math the compiler may fuse `acc += x * k` to `fma`. The
// CPU op runs the same reduction in strict left-fold FP32 order (no FMA in
// rustc for a plain `+= x * k` pattern), so there is a fused-vs-unfused
// gap of ~1 ULP per tap. For a typical Kaiser kernel of length 12-64 taps
// the accumulated divergence stays well inside the parity bound
// `atol ≤ 1e-4`.
//
// * `x`      — [channels, time_in] row-major FP32.
// * `kernel` — [taps] causal FIR taps (kernel[0] multiplies x[c, t],
//              kernel[1] multiplies x[c, t-1], …).
// * `out`    — [channels, time_in * ratio] row-major FP32.
//
// Grid: `(time_in * ratio, channels)` 2-D dispatch — same 16×16
// threadgroup shape as the sibling snake / codec kernels. The ragged tail
// is guarded against both bounds.
struct AntiAliasedUpsampleDims {
    uint channels;
    uint time_in;
    uint time_out;   // == time_in * ratio
    uint ratio;
    uint taps;       // == kernel.len()
};

kernel void vokra_anti_aliased_upsample_f32(
    device const float*                  x      [[buffer(0)]],
    device const float*                  kernel_ [[buffer(1)]],
    device float*                        out    [[buffer(2)]],
    constant AntiAliasedUpsampleDims&    d      [[buffer(3)]],
    uint2                                gid    [[thread_position_in_grid]])
{
    const uint t_out = gid.x;
    const uint c     = gid.y;
    if (c >= d.channels || t_out >= d.time_out) {
        return;
    }
    const uint t = t_out / d.ratio;
    const uint r = t_out % d.ratio;
    const uint x_row_off   = c * d.time_in;
    const uint out_row_off = c * d.time_out;
    float acc = 0.0f;
    // Bounded per-branch tap walk: `k_idx = j * ratio + r`; break as soon
    // as either `k_idx >= taps` or `j > t` (which would step past the
    // input's causal end). `j` cannot overflow because both `t` and
    // `taps / ratio` are bounded by the CPU host-side length check.
    for (uint j = 0u; ; ++j) {
        const uint k_idx = j * d.ratio + r;
        if (k_idx >= d.taps) {
            break;
        }
        if (j > t) {
            break;
        }
        const uint src = t - j;
        acc += x[x_row_off + src] * kernel_[k_idx];
    }
    out[out_row_off + t_out] = acc;
}

// Alias-free DownSample1d. The asymmetric edge replication and FIR/stride
// indexing intentionally mirror vokra-ops' scalar reference exactly.
struct AntiAliasedDownsampleDims {
    uint channels;
    uint time_in;
    uint time_out;
    uint ratio;
    uint taps;
    uint pad_left;
};

kernel void vokra_anti_aliased_downsample_f32(
    device const float* x [[buffer(0)]],
    device const float* kernel_ [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant AntiAliasedDownsampleDims& d [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint t = gid.x;
    const uint c = gid.y;
    if (c >= d.channels || t >= d.time_out) return;
    const uint in_off = c * d.time_in;
    float sum = 0.0f;
    for (uint tap = 0u; tap < d.taps; ++tap) {
        const uint padded = t * d.ratio + tap;
        int source = int(padded) - int(d.pad_left);
        if (source < 0) source = 0;
        if (source >= int(d.time_in)) source = int(d.time_in) - 1;
        sum += x[in_off + uint(source)] * kernel_[tap];
    }
    out[c * d.time_out + t] = sum;
}

struct ScaleDims {
    uint n;
    float scale;
};

kernel void vokra_scale_f32(
    device const float* x [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant ScaleDims& d [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid < d.n) out[gid] = x[gid] * d.scale;
}

struct ClampDims {
    uint n;
    float lower;
    float upper;
};

kernel void vokra_clamp_f32(
    device const float* x [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant ClampDims& d [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= d.n) return;
    const float v = x[gid];
    // Rust's f32::clamp preserves NaN; do not let a backend min/max choose a
    // bound for a non-finite intermediate from the logits path.
    out[gid] = isnan(v) ? v : min(max(v, d.lower), d.upper);
}

// ---- Vocoder Metal wave WF5 denoise spectral-gate primitive (2026-08-13) ----
//
// Semantics identical to `vokra_ops::denoise::denoise_apply_mask_f32` — the
// element-wise complex × real-gain multiply extracted from the
// [`DenoiseModel::enhance_inner`] output-stage loop (denoise.rs L1852-1870).
// GTCRN / RNNoise / any per-freq-per-time mask trainer that emits per-position
// real gains can dispatch through this kernel; DFN3 pre-expands its ERB mask
// through `erb_inv_fb` to per-position gains upstream of the call.
//
//     out_re[t, f] = spec_re[t, f] * gain[t, f]
//     out_im[t, f] = spec_im[t, f] * gain[t, f]
//
// * `spec_re` / `spec_im` — [n_frames, n_bins] row-major FP32 (bins on the
//                           inner stride — the layout `Spectrogram { re, im }`
//                           uses in vokra_ops::denoise).
// * `gain`               — [n_frames, n_bins] row-major FP32 per-position
//                           real gain (already expanded from whatever mask
//                           the upstream front-end produced).
// * `out_re` / `out_im`  — [n_frames, n_bins] row-major FP32 outputs.
//
// Phase preservation is exact (`atan2(im · g, re · g) = atan2(im, re)` for
// every finite g ≥ 0, and a global π flip for g < 0 — never a per-bin phase
// distortion). Multiplication is IEEE-754 correctly-rounded on every finite
// input; there is no reduction, no transcendental, no FMA opportunity, so
// CPU and GPU produce **bit-for-bit identical** FP32 outputs. The parity
// harness allows the sibling `atol ≤ 5e-4` codec-family bound to keep a
// discriminating negative control while logging the actual max |Δ| (0 in
// practice — any future drift immediately visible).
//
// Precision: FP32 throughout (BF16 mantissa loss is the real problem —
// CLAUDE.md audio-dialect rule; denoise is FP32 by construction upstream too).
//
// Grid: `(n_bins, n_frames)` 2-D dispatch — same 16×16 threadgroup shape as
// the sibling snake_activation / codec kernels. `n_bins` on the fast axis
// (grid.x) matches the row-major stride and keeps adjacent threads reading
// adjacent floats. The ragged tail is guarded against both bounds.
struct DenoiseApplyMaskDims {
    uint n_bins;
    uint n_frames;
};

kernel void vokra_denoise_apply_mask_f32(
    device const float*                spec_re [[buffer(0)]],
    device const float*                spec_im [[buffer(1)]],
    device const float*                gain    [[buffer(2)]],
    device float*                      out_re  [[buffer(3)]],
    device float*                      out_im  [[buffer(4)]],
    constant DenoiseApplyMaskDims&     d       [[buffer(5)]],
    uint2                              gid     [[thread_position_in_grid]])
{
    const uint f = gid.x;
    const uint t = gid.y;
    if (f >= d.n_bins || t >= d.n_frames) {
        return;
    }
    const uint  idx = t * d.n_bins + f;
    const float g   = gain[idx];
    out_re[idx] = spec_re[idx] * g;
    out_im[idx] = spec_im[idx] * g;
}

// ---- Vocoder Metal wave WF5 qwen3_tts_codec RVQ decode ---------------------
//
// Semantics identical to `vokra_ops::qwen3_tts_codec::qwen3_tts_codec_decode`
// (the Qwen3-TTS-12Hz codec's per-quantizer summed feature decode step
// consumed by every released Qwen3-TTS-12Hz voice —
// `Qwen/Qwen3-TTS-12Hz-{0.6B,1.7B}-{Base,CustomVoice,VoiceDesign}`,
// Apache-2.0). Given `[time, num_quantizers]` row-major `u32` codes and the
// per-quantizer codebook tables, the primitive gathers the corresponding
// codebook rows and FP32-sums them into `out[t, :]`:
//
//     out[t, d] = Σ_q tables[q].row(codes[t, q])[d]
//
// # Semantic vs acoustic vocab split (why two table buffers)
//
// Qwen3-TTS-Codec is a hybrid semantic + acoustic RVQ: the first
// `num_semantic_quantizers` quantizers use a **larger** `semantic_codebook_size`
// vocab (canonical 4096); the remaining acoustic quantizers use
// `codebook_size` (canonical 2048). Every codebook still emits the same
// `codebook_dim`-wide row (canonical 512), so the FP32 fold is
// well-defined — but the per-quantizer stride differs between the two
// families. Rather than fake a shared vocab (which would either waste memory
// or silently clamp the semantic index; both violate FR-EX-08 / the module
// docs' "no silent clamp" rule), the kernel takes TWO flat table buffers:
//
//   * `semantic_tables` — `[num_semantic_quantizers, semantic_codebook_size,
//     codebook_dim]` row-major FP32; quantizer `q < num_semantic_quantizers`
//     starts at `q * semantic_codebook_size * codebook_dim`.
//   * `acoustic_tables` — `[num_acoustic_quantizers, codebook_size,
//     codebook_dim]` row-major FP32; quantizer `q >= num_semantic_quantizers`
//     is `q - num_semantic_quantizers` within this buffer.
//
// The host guards `codes[t, q] < per_quantizer_vocab(q)` before dispatch
// (FR-EX-08 — the kernel does no per-element bound check, so silent OOB reads
// would be the failure mode). An empty semantic (or acoustic) side is legal —
// the dispatch allocates a zeroed 4-byte placeholder via
// `newBufferWithLength:` in that case (Metal requires a non-null buffer
// binding at every declared `[[buffer(N)]]` slot even when the kernel never
// reads through it).
//
// # FP32 accumulator (audio-dialect rule)
//
// The residual sum is FP32-accumulated even if a future variant stores
// codebook tables in FP16 / BF16 — same rule as mimi_rvq / dac_rvq (audio-
// dialect: "BF16 mantissa loss is the real problem", CLAUDE.md).
//
// Grid: `(codebook_dim, time)` — same 2D launch as mimi_rvq / dac_rvq; the
// shader guards the ragged tail against both bounds. Threadgroup 16x16 (the
// `grid_2d` default), sized for the canonical Qwen3-TTS
// [codebook_dim=512, time≈37 (3 s at 12.5 Hz)] envelope and small enough to
// not waste threads on the tiny parity fixtures.
struct Qwen3TtsCodecDims {
    uint num_quantizers;
    uint num_semantic_quantizers;
    uint semantic_codebook_size;
    uint codebook_size;
    uint codebook_dim;
    uint time;
};

kernel void vokra_qwen3_tts_codec_decode_f32(
    device const uint*                 codes            [[buffer(0)]],
    device const float*                semantic_tables  [[buffer(1)]],
    device const float*                acoustic_tables  [[buffer(2)]],
    device float*                      out              [[buffer(3)]],
    constant Qwen3TtsCodecDims&        d                [[buffer(4)]],
    uint2                              gid              [[thread_position_in_grid]])
{
    const uint t     = gid.y;
    const uint delem = gid.x;
    if (t >= d.time || delem >= d.codebook_dim) {
        return;
    }
    const uint code_base     = t * d.num_quantizers;
    const uint sem_cb_stride = d.semantic_codebook_size * d.codebook_dim;
    const uint ac_cb_stride  = d.codebook_size          * d.codebook_dim;
    // FP32 accumulator — matches `qwen3_tts_codec_decode`'s `out[..]` FP32
    // fold order over quantizers (bit-identical when the same operands are
    // added in the same sequence; MSL fast-math may re-associate, so the
    // parity bound is FP32 GEMV-scale rather than a bit-for-bit assertion).
    float acc = 0.0f;
    for (uint q = 0; q < d.num_quantizers; ++q) {
        const uint idx = codes[code_base + q];
        if (q < d.num_semantic_quantizers) {
            const uint off = q * sem_cb_stride + idx * d.codebook_dim + delem;
            acc += semantic_tables[off];
        } else {
            const uint ac_q = q - d.num_semantic_quantizers;
            const uint off  = ac_q * ac_cb_stride + idx * d.codebook_dim + delem;
            acc += acoustic_tables[off];
        }
    }
    out[t * d.codebook_dim + delem] = acc;
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

/// Dense/grouped Conv2d dimensions (`setBytes:` index 4). Mirrors the MSL
/// `Conv2dDims` field order exactly.
#[repr(C)]
#[derive(Clone, Copy)]
struct Conv2dDims {
    in_ch: u32,
    in_h: u32,
    in_w: u32,
    out_ch: u32,
    out_h: u32,
    out_w: u32,
    kernel_h: u32,
    kernel_w: u32,
    stride_h: u32,
    stride_w: u32,
    padding_h: u32,
    padding_w: u32,
    dilation_h: u32,
    dilation_w: u32,
    in_per_group: u32,
    out_per_group: u32,
    has_bias: u32,
}

/// PyTorch-layout ConvTranspose2d dimensions (`setBytes:` index 4). Mirrors
/// the MSL `ConvTranspose2dDims` field order exactly.
#[repr(C)]
#[derive(Clone, Copy)]
struct ConvTranspose2dDims {
    in_ch: u32,
    in_h: u32,
    in_w: u32,
    out_ch: u32,
    out_h: u32,
    out_w: u32,
    kernel_h: u32,
    kernel_w: u32,
    stride_h: u32,
    stride_w: u32,
    padding_h: u32,
    padding_w: u32,
    dilation_h: u32,
    dilation_w: u32,
    in_per_group: u32,
    out_per_group: u32,
    has_bias: u32,
}

/// Mixed FP32-activation/raw-BF16-weight GEMM dimensions. Mirrors
/// `GemmF32Bf16BitsDims` in `KERNELS_MSL`.
#[repr(C)]
#[derive(Clone, Copy)]
struct GemmF32Bf16BitsDims {
    m: u32,
    n: u32,
    k: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct HiftComplexDims {
    frames: u32,
    bins: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Pad1dDims {
    channels: u32,
    time_in: u32,
    time_out: u32,
    left: u32,
    right: u32,
    mode: u32,
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

/// One-group GroupNorm dims (`setBytes:` index 4). Mirrors the MSL
/// `struct GroupNormDims`.
#[repr(C)]
#[derive(Clone, Copy)]
struct GroupNormDims {
    channels: u32,
    positions: u32,
    eps: f32,
}

/// Multi-group GroupNorm dims (`setBytes:` index 4). Mirrors the MSL
/// `struct GroupNormGroupsDims`.
#[repr(C)]
#[derive(Clone, Copy)]
struct GroupNormGroupsDims {
    channels: u32,
    positions: u32,
    groups: u32,
    eps: f32,
}

/// GELU dims (`setBytes:` index 2). Mirrors the MSL `struct GeluDims`.
#[repr(C)]
#[derive(Clone, Copy)]
struct GeluDims {
    n: u32,
}

/// ReLU dims (`setBytes:` index 2). Mirrors the MSL `struct ReluDims`.
#[repr(C)]
#[derive(Clone, Copy)]
struct ReluDims {
    n: u32,
}

/// ELU dims (`setBytes:` index 2). Mirrors the MSL `struct EluDims`.
#[repr(C)]
#[derive(Clone, Copy)]
struct EluDims {
    n: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct LinearAbsDims {
    channels: u32,
    time: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct NearestUpsampleDims {
    channels: u32,
    time_in: u32,
    time_out: u32,
    factor: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct HiftStftDims {
    time_in: u32,
    n_fft: u32,
    hop: u32,
    frames: u32,
    bins: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct HiftIstftDims {
    n_fft: u32,
    hop: u32,
    frames: u32,
    bins: u32,
    out_len: u32,
}

/// Tanh dims (`setBytes:` index 2). Mirrors the MSL `struct TanhDims`.
#[repr(C)]
#[derive(Clone, Copy)]
struct TanhDims {
    n: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct LeakyReluDims {
    n: u32,
    slope: f32,
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
    in_per_group: u32,
    in_len: u32,
    out_ch: u32,
    kernel_size: u32,
    out_len: u32,
    stride: u32,
    padding: u32,
    has_bias: u32,
    out_per_group: u32,
    dilation: u32,
}

/// Device-resident ConvTranspose1d dimensions. Weight layout is PyTorch's
/// `[in_ch, out_ch, kernel]`; input/output are channel-major.
#[repr(C)]
#[derive(Clone, Copy)]
struct ConvTranspose1dDims {
    in_ch: u32,
    out_ch: u32,
    kernel_size: u32,
    t_in: u32,
    t_out: u32,
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

/// MossFormer2 ScaleNorm dims (`setBytes:` index 2). Field order and widths
/// mirror the MSL `ScaleNormDims`; all members are four-byte scalars.
#[repr(C)]
#[derive(Clone, Copy)]
struct ScaleNormDims {
    rows: u32,
    cols: u32,
    scale: f32,
    eps: f32,
    gain: f32,
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

/// OUVE predictor dims (`setBytes:` index 6). Field order mirrors the MSL
/// `OuvEReverseDims` and is intentionally scalar: model-specific layout stays
/// in the caller while every element uses the same source equation.
#[repr(C)]
#[derive(Clone, Copy)]
struct OuvEReverseDims {
    n: u32,
    theta: f32,
    sigma_min: f32,
    sigma_max: f32,
    t: f32,
    step: f32,
    probability_flow: u32,
}

/// OUVE annealed-Langevin corrector dims (`setBytes:` index 5).
#[repr(C)]
#[derive(Clone, Copy)]
struct OuvEAnnealedDims {
    n: u32,
    theta: f32,
    sigma_min: f32,
    sigma_max: f32,
    t: f32,
    snr: f32,
}

/// SwiGLU dims (`setBytes:` index 3). Mirrors the MSL `struct SwigluDims`.
#[repr(C)]
#[derive(Clone, Copy)]
struct SwigluDims {
    n: u32,
}

/// M3-06 T14 mimi_rvq gather + FP32 fold dims (`setBytes:` index 3). Field
/// order / `u32` widths mirror the MSL `struct MimiRvqDims`.
///
/// - `n_codebooks` — number of codebooks (Mimi canonical = 8).
/// - `codebook_size` — entries per codebook (Mimi canonical = 2048).
/// - `d_model` — feature dim per codebook entry (Mimi canonical = 512).
/// - `time` — number of timesteps in this decode chunk.
#[repr(C)]
#[derive(Clone, Copy)]
struct MimiRvqDims {
    n_codebooks: u32,
    codebook_size: u32,
    d_model: u32,
    time: u32,
}

/// M4-04 dac_rvq gather + factorized projection + FP32 fold dims (`setBytes:`
/// index 5). Field order / `u32` widths mirror the MSL `struct DacRvqDims`.
///
/// The extra `codebook_dim` axis vs [`MimiRvqDims`] is the DAC factorized
/// design (module docs `vokra_ops::dac_rvq`): codebook rows live in the
/// low-dim space and are projected up to `d_model` per quantizer.
///
/// - `n_codebooks` — number of quantizers (DAC 24 kHz canonical = 32).
/// - `codebook_size` — entries per codebook (DAC canonical = 1024).
/// - `codebook_dim` — factorized codebook row width (DAC canonical = 8).
/// - `d_model` — output feature dim per timestep (DAC 24 kHz canonical = 1024).
/// - `time` — number of timesteps in this decode chunk.
#[repr(C)]
#[derive(Clone, Copy)]
struct DacRvqDims {
    n_codebooks: u32,
    codebook_size: u32,
    codebook_dim: u32,
    d_model: u32,
    time: u32,
}

/// M4-16 wavtokenizer_vq single-codebook gather dims (`setBytes:` index 3).
/// Field order / `u32` widths mirror the MSL `struct WavTokenizerVqDims`.
///
/// - `vocab_size` — number of codebook entries (released WavTokenizer = 4096;
///   FR-OP-31 "65k+ vocab embedding" scale pinned by the vokra-ops synthetic
///   test in `fsq_codec::tests::wavtokenizer_65k_plus_vocab_path_is_exact`).
/// - `d_model` — embedding width per entry (released WavTokenizer = 512).
/// - `time` — number of timesteps in this decode chunk.
#[repr(C)]
#[derive(Clone, Copy)]
struct WavTokenizerVqDims {
    vocab_size: u32,
    d_model: u32,
    time: u32,
}

/// M4-16 xcodec2_fsq grid-decompose + optional GEMV dims (`setBytes:` index 5).
/// Field order / `u32` widths mirror the MSL `struct Xcodec2FsqDims`.
///
/// - `d_model` — output width (canonical released X-Codec 2 = 2048; for the
///   Identity path this must equal `n_dims`).
/// - `n_dims` — `len(levels)` (canonical = 8; the mixed-radix decompose
///   walks `levels[0..n_dims]` per timestep).
/// - `time` — number of timesteps in this decode chunk.
/// - `has_projection` — 0 = Identity (`d_model == n_dims`, copy grid to out),
///   1 = GEMV (Linear `n_dims → d_model` + bias per timestep). Canonical
///   released X-Codec 2 is `has_projection = 1`.
#[repr(C)]
#[derive(Clone, Copy)]
struct Xcodec2FsqDims {
    d_model: u32,
    n_dims: u32,
    time: u32,
    has_projection: u32,
}

/// Vocoder Metal wave WF2 snake activation dims (`setBytes:` index 3). Field
/// order / `u32` widths mirror the MSL `struct SnakeActivationDims`.
///
/// - `channels` — number of channels (== `alpha.len()`; Kokoro-82M decoder =
///   512 for the terminal decoder Snake, BigVGAN AMP blocks vary
///   32〜1024).
/// - `time` — number of frames per channel in this call.
#[repr(C)]
#[derive(Clone, Copy)]
struct SnakeActivationDims {
    channels: u32,
    time: u32,
}

/// Vocoder Metal wave common vocoder primitive: SnakeBeta dims
/// (`setBytes:` index 4). Field order / `u32` widths mirror the MSL
/// `struct SnakeBetaDims`. Semantically identical to
/// [`SnakeActivationDims`] but a distinct type so the wrong-op dispatch
/// (a callsite that binds a snake_beta buffer set to the snake_activation
/// pipeline) is a Rust-side type error.
///
/// - `channels` — number of channels (== `alpha.len() == beta.len()`).
/// - `time`     — number of frames per channel in this call.
#[repr(C)]
#[derive(Clone, Copy)]
struct SnakeBetaDims {
    channels: u32,
    time: u32,
}

/// Vocoder Metal wave common vocoder primitive: SineGen deterministic dims
/// (`setBytes:` index 2). Field order / widths mirror the MSL
/// `struct SinegenDeterministicDims`.
///
/// - `t`                — number of input timesteps (== `f0.len()`).
/// - `h1`               — harmonics + fundamental (== `harmonic_num + 1`).
/// - `samp_rate_f`      — audio sample rate as FP32 (`samp_rate as f32`).
/// - `sine_amp`         — sinusoid amplitude scale.
/// - `voiced_threshold` — F0 threshold above which a frame is voiced.
///
/// FP32 fields keep the arithmetic in the same precision the CPU op runs
/// (BF16 mantissa loss is the audio-dialect rule of thumb — CLAUDE.md).
#[repr(C)]
#[derive(Clone, Copy)]
struct SinegenDeterministicDims {
    t: u32,
    h1: u32,
    samp_rate_f: f32,
    sine_amp: f32,
    voiced_threshold: f32,
}

/// Vocoder Metal wave common vocoder primitive: anti_aliased_upsample dims
/// (`setBytes:` index 3). Field order / `u32` widths mirror the MSL
/// `struct AntiAliasedUpsampleDims`.
///
/// - `channels` — number of channels in `x` / `out` (channel-outer layout).
/// - `time_in`  — number of input timesteps.
/// - `time_out` — number of output timesteps (== `time_in * ratio`, kept
///   explicitly to avoid recomputing per thread).
/// - `ratio`    — integer upsample factor.
/// - `taps`     — number of filter taps (== `kernel.len()`).
#[repr(C)]
#[derive(Clone, Copy)]
struct AntiAliasedUpsampleDims {
    channels: u32,
    time_in: u32,
    time_out: u32,
    ratio: u32,
    taps: u32,
}

/// Device-resident alias-free downsample dimensions. `pad_left` is the
/// asymmetric replicate-padding amount used by BigVGAN.
#[repr(C)]
#[derive(Clone, Copy)]
struct AntiAliasedDownsampleDims {
    channels: u32,
    time_in: u32,
    time_out: u32,
    ratio: u32,
    taps: u32,
    pad_left: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ScaleDims {
    n: u32,
    scale: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ClampDims {
    n: u32,
    lower: f32,
    upper: f32,
}

/// Vocoder Metal wave WF5 denoise spectral-gate dims (`setBytes:` index 5).
/// Field order / `u32` widths mirror the MSL `struct DenoiseApplyMaskDims`.
///
/// - `n_bins`   — number of frequency bins per frame (STFT `n_fft/2 + 1`).
///   For DFN3 24 kHz canonical this is 481 (`n_fft = 960`), for GTCRN 16 kHz
///   canonical 257 (`n_fft = 512`).
/// - `n_frames` — number of time frames in this decode chunk (the
///   [`vokra_ops::denoise::Spectrogram::frames`] axis).
#[repr(C)]
#[derive(Clone, Copy)]
struct DenoiseApplyMaskDims {
    n_bins: u32,
    n_frames: u32,
}

/// Vocoder Metal wave WF5 snac_decode dims (`setBytes:` index 5). Field
/// order / `u32` widths mirror the MSL `struct SnacDecodeDims`.
///
/// - `d_model` — output feature width per timestep (SNAC 24 kHz canonical =
///   768 — the shared decoder input dim for Orpheus / Maya1's model config).
/// - `codebook_dim` — factorized codebook row width (SNAC 24 kHz canonical =
///   8, mirror of DAC's factorization).
/// - `codebook_size` — entries per codebook (SNAC 24 kHz canonical = 4096;
///   equal across the three stages by construction).
/// - `t_expanded` — number of output timesteps (`codes[s].len() *
///   strides[s]`, same for every stage — the "co-aligned base frames"
///   invariant `SnacDecoder::check_and_measure` enforces).
/// - `strides` — per-stage temporal strides `[u32; 3]` (SNAC 24 kHz canonical
///   = `[4, 2, 1]` giving per-stage rates ~12 / 23 / 47 Hz).
/// - `stage_offsets` — start of each stage in the flat `codes` buffer, so
///   stage `s`'s frame `t_stage` is `codes[stage_offsets[s] + t_stage]`. By
///   construction `stage_offsets[0] = 0`; `stage_offsets[1] = len(codes[0])`;
///   `stage_offsets[2] = len(codes[0]) + len(codes[1])`.
#[repr(C)]
#[derive(Clone, Copy)]
struct SnacDecodeDims {
    d_model: u32,
    codebook_dim: u32,
    codebook_size: u32,
    t_expanded: u32,
    n_stages: u32,
    strides: [u32; 4],
    stage_offsets: [u32; 4],
}

/// Vocoder Metal wave WF5 qwen3_tts_codec dims (`setBytes:` index 4). Field
/// order / `u32` widths mirror the MSL `struct Qwen3TtsCodecDims`.
///
/// - `num_quantizers` — total quantizers (Qwen3-TTS-12Hz canonical = 16).
/// - `num_semantic_quantizers` — number of semantic-vocab quantizers at the
///   head of the RVQ stack (Qwen3-TTS-12Hz canonical = 1). Quantizers
///   `[0, num_semantic_quantizers)` read from `semantic_tables` with vocab
///   `semantic_codebook_size`; the rest read from `acoustic_tables` with
///   vocab `codebook_size`.
/// - `semantic_codebook_size` — entries per semantic codebook (Qwen3-TTS-12Hz
///   canonical = 4096).
/// - `codebook_size` — entries per acoustic codebook (Qwen3-TTS-12Hz canonical
///   = 2048).
/// - `codebook_dim` — feature width per codebook entry (= the codec latent
///   width, Qwen3-TTS-12Hz canonical = 512).
/// - `time` — number of timesteps in this decode chunk.
#[repr(C)]
#[derive(Clone, Copy)]
struct Qwen3TtsCodecDims {
    num_quantizers: u32,
    num_semantic_quantizers: u32,
    semantic_codebook_size: u32,
    codebook_size: u32,
    codebook_dim: u32,
    time: u32,
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

/// A device-resident matrix of raw BF16 numeric bit patterns.
///
/// The logical elements are `u16` values obtained from little-endian BF16
/// storage (for example with `u16::from_le_bytes`). The Metal buffer retains
/// those two-byte values as `ushort`; it is never widened to an FP32 host or
/// device tensor before a mixed GEMM consumes it.
pub struct MetalBf16DeviceTensor<'ctx> {
    buf: OwnedBuf,
    len: usize,
    owner: *const MetalContext,
    _ctx: PhantomData<&'ctx MetalContext>,
}

impl MetalBf16DeviceTensor<'_> {
    /// The number of raw BF16 (`u16`) elements in this device buffer.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the device buffer contains no logical BF16 elements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
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
/// - `owner` records the allocating context identity. The lifetime marker
///   prevents use-after-drop; the owner check prevents cross-context/device
///   mixing before any Metal command is encoded.
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
    /// Identity of the allocating context. This is compared before every
    /// public device operation; it is never dereferenced.
    owner: *const MetalContext,
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
/// It owns raw `OwnedBuf`s (no `MetalDeviceTensor<'ctx>` borrow), so — like the
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
    /// Mixed FP32-activation × raw-BF16-weight GEMM pipeline. The weight
    /// operand remains a `ushort` buffer and is widened in the shader.
    gemm_f32_bf16_bits_pipeline: Id,
    gemv_pipeline: Id,
    softmax_pipeline: Id,
    softmax_causal_pipeline: Id,
    layer_norm_pipeline: Id,
    group_norm_pipeline: Id,
    group_norm_groups_pipeline: Id,
    gelu_pipeline: Id,
    gelu_new_pipeline: Id,
    relu_pipeline: Id,
    elu_pipeline: Id,
    /// HiFTNet F0-predictor linear-to-one followed by absolute value.
    linear_abs_pipeline: Id,
    /// HiFTNet NSF source linear-to-one followed by tanh.
    linear_tanh_pipeline: Id,
    tanh_pipeline: Id,
    conv1d_pipeline: Id,
    conv2d_pipeline: Id,
    conv_transpose2d_pipeline: Id,
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
    /// M4-05/06 Llama-family decode primitives plus MossFormer2 ScaleNorm.
    /// Each is the GPU implementation of its matching CPU equation (module
    /// docs on `KERNELS_MSL`) and shares the Phase-4/5 library.
    rms_norm_pipeline: Id,
    scale_norm_pipeline: Id,
    rope_adjacent_pipeline: Id,
    silu_pipeline: Id,
    swiglu_pipeline: Id,
    /// Device-resident OUVE-SDE predictor and annealed-Langevin sampler steps.
    ouve_reverse_diffusion_pipeline: Id,
    ouve_annealed_langevin_pipeline: Id,
    /// M3-06 T14 mimi_rvq gather + FP32 fold (`vokra_mimi_rvq_gather_fold_f32`),
    /// the GPU implementation of `vokra_ops::mimi_rvq::rvq_fold_core`. Also the
    /// current M4-04 GPU seam target for DAC / EnCodec siblings after their
    /// respective factorized-projection / plain-fold arms are wired (each will
    /// land its own kernel or reuse this one — coverage flags stay per-op).
    mimi_rvq_gather_fold_pipeline: Id,
    /// M4-04 dac_rvq gather + factorized projection + FP32 fold
    /// (`vokra_dac_rvq_gather_project_fold_f32`), the GPU implementation of
    /// `vokra_ops::dac_rvq::dac_rvq_decode`. Distinct from
    /// [`Self::mimi_rvq_gather_fold_pipeline`] because DAC's per-quantizer
    /// factorized projection (W · low + b) folds into the same kernel as the
    /// gather — a plain gather-only reuse of the Mimi pipeline would need a
    /// second GEMV pass per timestep. See `vokra_ops::dac_rvq` module docs for
    /// the factorization rationale.
    dac_rvq_gather_project_fold_pipeline: Id,
    /// M4-16 wavtokenizer_vq single-codebook gather
    /// (`vokra_wavtokenizer_vq_gather_f32`), the GPU implementation of
    /// `vokra_ops::wavtokenizer_vq_decode`. **Distinct from every RVQ
    /// pipeline**: the FSQ family is deliberately a *separate subgraph* from
    /// the RVQ family (FR-OP-31 — module docs on `vokra_ops::fsq_codec`); the
    /// signature-level distinction that the CPU op takes a *singular*
    /// `&CodebookTable` (not a slice) is mirrored on the GPU by a dedicated
    /// pipeline that expects a single flat table buffer. Pure gather — no
    /// residual fold, no per-dim decompose.
    wavtokenizer_vq_gather_pipeline: Id,
    /// M4-16 xcodec2_fsq grid-decompose + optional GEMV
    /// (`vokra_xcodec2_fsq_decode_f32`), the GPU implementation of
    /// `vokra_ops::xcodec2_fsq_decode`. FSQ-family sibling of
    /// [`Self::wavtokenizer_vq_gather_pipeline`]. Handles both the
    /// `requires_projection = true` (canonical released X-Codec 2: Linear
    /// n_dims → d_model + bias) and the `requires_projection = false`
    /// (Identity, d_model == n_dims) paths through a `has_projection` flag
    /// in the dims struct.
    xcodec2_fsq_decode_pipeline: Id,
    /// Vocoder Metal wave WF2 snake activation
    /// (`vokra_snake_activation_f32`), the GPU implementation of
    /// [`vokra_ops::snake_activation_f32`]. Per-channel closed-form periodic
    /// activation `y = x + (1/(α+ε))·sin(α·x)²` shared by the BigVGAN /
    /// HiFTNet / Kokoro-82M vocoder lineage. Trivially element-wise (no
    /// gather, no fold), so CPU vs GPU is bit-identical for finite inputs
    /// within the FP32 transcendental gap (atol ≤ 5e-4 — the same
    /// codec-family bound). SnakeBeta ([`vokra_ops::bigvgan_generator::SnakeBeta`])
    /// is a **distinct** two-vector closed form and would land its own
    /// pipeline separately.
    snake_activation_pipeline: Id,
    /// Vocoder Metal wave common vocoder primitive: SnakeBeta
    /// (`vokra_snake_beta_f32`), the GPU implementation of
    /// [`vokra_ops::snake_beta_f32`]. Per-channel two-vector closed-form
    /// periodic activation `y = x + (1/(β+ε))·sin(α·x)²` consumed by the
    /// BigVGAN family (upstream `activations.py:62-114`, MIT). Distinct
    /// pipeline from [`Self::snake_activation_pipeline`] because SnakeBeta
    /// separates frequency (α) from magnitude (β) — a shared pipeline
    /// would silently squash one of the two axes.
    snake_beta_pipeline: Id,
    /// Vocoder Metal wave common vocoder primitive: SineGen deterministic
    /// (`vokra_sinegen_deterministic_f32`), the GPU implementation of
    /// [`vokra_ops::sinegen_deterministic_f32`]. Deterministic-only path
    /// (zero phase, zero noise) of `SineGen::forward` from upstream
    /// CosyVoice `generator.py:200-214`. Consumed by every HiFTNet-family
    /// vocoder (CosyVoice2/3, Chatterbox family). One thread per harmonic
    /// walking the full time axis sequentially; grid launches
    /// `(H+1, 1)` threadgroups.
    sinegen_deterministic_pipeline: Id,
    /// Channel-major `[H+1, T]` deterministic SineGen for HiFTResidentOps.
    sinegen_deterministic_channel_major_pipeline: Id,
    /// Vocoder Metal wave common vocoder primitive: polyphase anti-aliased
    /// upsample (`vokra_anti_aliased_upsample_f32`), the GPU implementation
    /// of [`vokra_ops::anti_aliased_upsample_f32`]. Multiply-add core of
    /// BigVGAN's `UpSample1d` (upstream `alias_free_activation.torch.act`,
    /// MIT). Consumes a caller-supplied Kaiser-window filter kernel; the
    /// Kaiser design lives on the host (once per model load), keeping the
    /// runtime op signature narrow. Ordinary FIR reduction — the FMA-vs-
    /// non-FMA gap between MSL fast-math and the CPU strict-left-fold is
    /// well inside the parity bound `atol ≤ 1e-4`.
    anti_aliased_upsample_pipeline: Id,
    /// Device-resident BigVGAN alias-free downsample (replicate + FIR).
    anti_aliased_downsample_pipeline: Id,
    /// Device-resident scalar used for MRF branch averaging.
    scale_pipeline: Id,
    /// Device-resident terminal/general clamp.
    clamp_pipeline: Id,
    /// Device-resident PyTorch-layout ConvTranspose1d used by HiFTNet and
    /// BigVGAN upsample stages. Unlike the legacy host-facing Conv1d seam,
    /// this pipeline never reads an intermediate back to the host.
    conv_transpose1d_pipeline: Id,
    /// Device-resident reflect/replicate channel-major padding for vocoder
    /// seams. Padding is kept as a device copy so no host Vec is materialized.
    pad1d_pipeline: Id,
    /// Device-resident LeakyReLU used between HiFTNet upsample stages.
    leaky_relu_pipeline: Id,
    /// Device-resident nearest-neighbour temporal upsample for HiFTNet F0.
    nearest_upsample_pipeline: Id,
    /// Device-resident centered periodic-Hann reflect-padded HiFT STFT.
    hift_stft_pipeline: Id,
    /// Device-resident HiFT magnitude/phase logits to complex spectrum.
    hift_complex_pipeline: Id,
    /// Device-resident HiFT magnitude/phase postprocess and centered iSTFT.
    hift_istft_pipeline: Id,
    /// Vocoder Metal wave WF5 SNAC 3-stage hierarchical RVQ decode
    /// (`vokra_snac_decode_f32`), the GPU implementation of
    /// `vokra_ops::snac_decode::SnacDecoder::decode` (upstream
    /// `hubertsiuzdak/snac`, MIT / Apache-2.0). **Distinct from every RVQ
    /// pipeline**: SNAC's multi-scale structure (each stage runs at
    /// `base / vq_strides[s]`) is baked into the kernel via a
    /// `t_stage = t_out / strides[s]` lookup per stage, which no other RVQ /
    /// FSQ codec kernel does. Reuses the [`DacRvqDims`] factorized shape
    /// (per-stage `WNConv1d(codebook_dim → d_model)` + bias) but folds
    /// `repeat_interleave(stride)` into the temporal indexing instead of the
    /// per-output copy loop the CPU fold does.
    snac_decode_pipeline: Id,
    /// Vocoder Metal wave WF5 denoise spectral-gate primitive
    /// (`vokra_denoise_apply_mask_f32`), the GPU implementation of
    /// [`vokra_ops::denoise_apply_mask_f32`]. Element-wise complex × real
    /// gain multiply extracted from the [`DenoiseModel::enhance_inner`]
    /// output-stage loop; the primitive shared with any per-freq-per-time
    /// mask trainer (GTCRN / RNNoise). Trivially per-element (no reduction,
    /// no transcendental, no FMA opportunity), so CPU vs GPU is
    /// bit-for-bit identical on every finite input — the parity harness
    /// still enforces the sibling `atol ≤ 5e-4` codec-family bound to keep
    /// a discriminating negative control, but logs the measured max |Δ|
    /// (0 in practice) so any future drift is immediately visible.
    denoise_apply_mask_pipeline: Id,
    /// Vocoder Metal wave WF5 Qwen3-TTS-Codec RVQ decode
    /// (`vokra_qwen3_tts_codec_decode_f32`), the GPU implementation of
    /// `vokra_ops::qwen3_tts_codec::qwen3_tts_codec_decode`. **Distinct from
    /// every other RVQ pipeline**: Qwen3-TTS-Codec is a **hybrid semantic +
    /// acoustic RVQ** where the first `num_semantic_quantizers` quantizers use
    /// a larger `semantic_codebook_size` vocab (canonical 4096) than the
    /// remaining acoustic quantizers use `codebook_size` (canonical 2048).
    /// The kernel takes TWO flat table buffers (semantic + acoustic) so
    /// per-quantizer strides differ correctly — a silent shared-vocab clamp
    /// would violate FR-EX-08 and the CPU op's "no silent clamp" rule. FP32
    /// accumulator throughout (audio-dialect rule); host-side per-index bound
    /// check upstream of the dispatch.
    qwen3_tts_codec_decode_pipeline: Id,
    /// Count of command-buffer submissions (`commit` + `waitUntilCompleted`)
    /// issued through this context — the env-independent readback/sync metric the
    /// Phase-5-follow-on encoder-residency slice proves against (the whole encoder
    /// in ONE submission vs the per-op path's `6·N + 1`). `Cell` because every op
    /// takes `&self` and the context is already thread-affine (`!Send`/`!Sync`).
    submissions: Cell<u64>,
    /// Explicit host readbacks requested through the resident API. Device
    /// primitives never increment this counter; it is a cheap assertion hook
    /// for vocoder callers that only the final PCM crosses D2H.
    readbacks: Cell<u64>,
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
        let gemm_f32_bf16_bits_pipeline =
            unsafe { make_pipeline(device, klib.0, c"vokra_gemm_f32_bf16_bits") }?;
        // SAFETY: `device` is valid and `klib` owns the named function.
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
        let group_norm_pipeline =
            unsafe { make_pipeline(device, klib.0, c"vokra_group_norm_f32") }?;
        // Multi-group NCSN++ normalization; explicit rather than reusing the
        // one-group SepFormer reduction.
        let group_norm_groups_pipeline =
            unsafe { make_pipeline(device, klib.0, c"vokra_group_norm_groups_f32") }?;
        // SAFETY: as above.
        let gelu_pipeline = unsafe { make_pipeline(device, klib.0, c"vokra_gelu_f32") }?;
        // SAFETY: as above.
        let gelu_new_pipeline = unsafe { make_pipeline(device, klib.0, c"vokra_gelu_new_f32") }?;
        // SAFETY: as above.
        let relu_pipeline = unsafe { make_pipeline(device, klib.0, c"vokra_relu_f32") }?;
        // SAFETY: as above.
        let elu_pipeline = unsafe { make_pipeline(device, klib.0, c"vokra_elu_f32") }?;
        // HiFTNet F0-predictor linear head: abs(linear(x)).
        // SAFETY: as above.
        let linear_abs_pipeline =
            unsafe { make_pipeline(device, klib.0, c"vokra_linear_abs_f32") }?;
        // SAFETY: as above.
        let linear_tanh_pipeline =
            unsafe { make_pipeline(device, klib.0, c"vokra_linear_tanh_f32") }?;
        // SAFETY: as above.
        let tanh_pipeline = unsafe { make_pipeline(device, klib.0, c"vokra_tanh_f32") }?;
        // HiFTNet's explicit negative-slope activation.
        // SAFETY: as above.
        let leaky_relu_pipeline =
            unsafe { make_pipeline(device, klib.0, c"vokra_leaky_relu_f32") }?;
        // SAFETY: as above.
        let conv1d_pipeline = unsafe { make_pipeline(device, klib.0, c"vokra_conv1d_f32") }?;
        // Dense/grouped 2-D convolution seams for HTDemucs.
        // SAFETY: as above.
        let conv2d_pipeline = unsafe { make_pipeline(device, klib.0, c"vokra_conv2d_f32") }?;
        // SAFETY: as above.
        let conv_transpose2d_pipeline =
            unsafe { make_pipeline(device, klib.0, c"vokra_conv_transpose2d_f32") }?;
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
        // MossFormer2 FLASH ScaleNorm; exact clamp-after-norm equation.
        // SAFETY: as above.
        let scale_norm_pipeline =
            unsafe { make_pipeline(device, klib.0, c"vokra_scale_norm_f32") }?;
        // SAFETY: as above.
        let rope_adjacent_pipeline =
            unsafe { make_pipeline(device, klib.0, c"vokra_rope_adjacent_f32") }?;
        // SAFETY: as above.
        let silu_pipeline = unsafe { make_pipeline(device, klib.0, c"vokra_silu_f32") }?;
        // OUVE-SDE predictor / corrector.  Both are device-resident
        // element-wise kernels; no host fallback is permitted.
        // SAFETY: as above.
        let ouve_reverse_diffusion_pipeline =
            unsafe { make_pipeline(device, klib.0, c"vokra_ouve_reverse_diffusion_f32") }?;
        // SAFETY: as above.
        let ouve_annealed_langevin_pipeline =
            unsafe { make_pipeline(device, klib.0, c"vokra_ouve_annealed_langevin_f32") }?;
        // SAFETY: as above.
        let swiglu_pipeline = unsafe { make_pipeline(device, klib.0, c"vokra_swiglu_f32") }?;
        // M3-06 T14 mimi_rvq gather + FP32 fold; shares the same library.
        // SAFETY: as above.
        let mimi_rvq_gather_fold_pipeline =
            unsafe { make_pipeline(device, klib.0, c"vokra_mimi_rvq_gather_fold_f32") }?;
        // M4-04 dac_rvq gather + factorized projection + FP32 fold; shares the
        // same library. Distinct kernel because DAC folds W · low + b per
        // quantizer into the gather (see `KERNELS_MSL` module docs).
        // SAFETY: as above.
        let dac_rvq_gather_project_fold_pipeline =
            unsafe { make_pipeline(device, klib.0, c"vokra_dac_rvq_gather_project_fold_f32") }?;
        // M4-16 FSQ family: single-codebook gather + grid-decompose GEMV; both
        // share the same library. Separate pipelines because the FSQ family
        // is a deliberately-separate subgraph from the RVQ family (FR-OP-31 —
        // no cross-family kernel reuse, matching the CPU op's singular
        // `&CodebookTable` signature).
        // SAFETY: as above.
        let wavtokenizer_vq_gather_pipeline =
            unsafe { make_pipeline(device, klib.0, c"vokra_wavtokenizer_vq_gather_f32") }?;
        // SAFETY: as above.
        let xcodec2_fsq_decode_pipeline =
            unsafe { make_pipeline(device, klib.0, c"vokra_xcodec2_fsq_decode_f32") }?;
        // Vocoder Metal wave WF2 snake activation; shares the same library.
        // SAFETY: as above.
        let snake_activation_pipeline =
            unsafe { make_pipeline(device, klib.0, c"vokra_snake_activation_f32") }?;
        // Vocoder Metal wave common vocoder primitives (2026-08-14):
        // SnakeBeta / SineGen deterministic / anti-aliased upsample. All
        // three share the same library as the sibling snake_activation /
        // codec kernels.
        // SAFETY: as above.
        let snake_beta_pipeline =
            unsafe { make_pipeline(device, klib.0, c"vokra_snake_beta_f32") }?;
        // SAFETY: as above.
        let sinegen_deterministic_pipeline =
            unsafe { make_pipeline(device, klib.0, c"vokra_sinegen_deterministic_f32") }?;
        // SAFETY: as above.
        let sinegen_deterministic_channel_major_pipeline = unsafe {
            make_pipeline(
                device,
                klib.0,
                c"vokra_sinegen_deterministic_channel_major_f32",
            )
        }?;
        // SAFETY: as above.
        let anti_aliased_upsample_pipeline =
            unsafe { make_pipeline(device, klib.0, c"vokra_anti_aliased_upsample_f32") }?;
        // BigVGAN alias-free downsample, branch scaling, and terminal clamp.
        // SAFETY: as above.
        let anti_aliased_downsample_pipeline =
            unsafe { make_pipeline(device, klib.0, c"vokra_anti_aliased_downsample_f32") }?;
        // SAFETY: as above.
        let scale_pipeline = unsafe { make_pipeline(device, klib.0, c"vokra_scale_f32") }?;
        // SAFETY: as above.
        let clamp_pipeline = unsafe { make_pipeline(device, klib.0, c"vokra_clamp_f32") }?;
        // Device-resident vocoder upsample primitive.
        // SAFETY: as above.
        let conv_transpose1d_pipeline =
            unsafe { make_pipeline(device, klib.0, c"vokra_conv_transpose1d_f32") }?;
        // Device-resident vocoder padding primitive.
        // SAFETY: as above.
        let pad1d_pipeline = unsafe { make_pipeline(device, klib.0, c"vokra_pad1d_f32") }?;
        // HiFTNet resident source upsample and spectral seams.
        // SAFETY: as above.
        let nearest_upsample_pipeline =
            unsafe { make_pipeline(device, klib.0, c"vokra_nearest_upsample_f32") }?;
        // SAFETY: as above.
        let hift_stft_pipeline = unsafe { make_pipeline(device, klib.0, c"vokra_hift_stft_f32") }?;
        // SAFETY: as above.
        let hift_complex_pipeline =
            unsafe { make_pipeline(device, klib.0, c"vokra_hift_complex_from_logits_f32") }?;
        // SAFETY: as above.
        let hift_istft_pipeline =
            unsafe { make_pipeline(device, klib.0, c"vokra_hift_istft_f32") }?;
        // Vocoder Metal wave WF5 SNAC 3-stage hierarchical RVQ decode; shares
        // the same library. Distinct kernel from the RVQ / FSQ family because
        // SNAC's multi-scale structure requires a per-stage
        // `t_stage = t_out / strides[s]` lookup that no other codec kernel
        // needs.
        // SAFETY: as above.
        let snac_decode_pipeline =
            unsafe { make_pipeline(device, klib.0, c"vokra_snac_decode_f32") }?;
        // Vocoder Metal wave WF5 denoise spectral-gate primitive; shares the
        // same library. Element-wise complex × real gain multiply — the
        // primitive extracted from `DenoiseModel::enhance_inner`'s output
        // stage, shared with any per-freq-per-time mask denoiser.
        // SAFETY: as above.
        let denoise_apply_mask_pipeline =
            unsafe { make_pipeline(device, klib.0, c"vokra_denoise_apply_mask_f32") }?;
        // Vocoder Metal wave WF5 Qwen3-TTS-Codec RVQ decode; shares the same
        // library. Distinct kernel from the RVQ / FSQ family because Qwen3-
        // TTS-Codec is a **hybrid semantic + acoustic RVQ** (two flat table
        // buffers with different per-quantizer strides) — a shared-vocab
        // clamp would violate FR-EX-08 / the CPU op's "no silent clamp" rule.
        // SAFETY: as above.
        let qwen3_tts_codec_decode_pipeline =
            unsafe { make_pipeline(device, klib.0, c"vokra_qwen3_tts_codec_decode_f32") }?;
        drop(klib);

        Ok(MetalContext {
            device,
            queue: queue.into_raw(),
            gemm_pipeline: gemm_pipeline.into_raw(),
            gemm_f32_bf16_bits_pipeline: gemm_f32_bf16_bits_pipeline.into_raw(),
            gemv_pipeline: gemv_pipeline.into_raw(),
            softmax_pipeline: softmax_pipeline.into_raw(),
            softmax_causal_pipeline: softmax_causal_pipeline.into_raw(),
            layer_norm_pipeline: layer_norm_pipeline.into_raw(),
            group_norm_pipeline: group_norm_pipeline.into_raw(),
            group_norm_groups_pipeline: group_norm_groups_pipeline.into_raw(),
            gelu_pipeline: gelu_pipeline.into_raw(),
            gelu_new_pipeline: gelu_new_pipeline.into_raw(),
            relu_pipeline: relu_pipeline.into_raw(),
            elu_pipeline: elu_pipeline.into_raw(),
            linear_abs_pipeline: linear_abs_pipeline.into_raw(),
            linear_tanh_pipeline: linear_tanh_pipeline.into_raw(),
            tanh_pipeline: tanh_pipeline.into_raw(),
            leaky_relu_pipeline: leaky_relu_pipeline.into_raw(),
            conv1d_pipeline: conv1d_pipeline.into_raw(),
            conv2d_pipeline: conv2d_pipeline.into_raw(),
            conv_transpose2d_pipeline: conv_transpose2d_pipeline.into_raw(),
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
            scale_norm_pipeline: scale_norm_pipeline.into_raw(),
            rope_adjacent_pipeline: rope_adjacent_pipeline.into_raw(),
            silu_pipeline: silu_pipeline.into_raw(),
            ouve_reverse_diffusion_pipeline: ouve_reverse_diffusion_pipeline.into_raw(),
            ouve_annealed_langevin_pipeline: ouve_annealed_langevin_pipeline.into_raw(),
            swiglu_pipeline: swiglu_pipeline.into_raw(),
            mimi_rvq_gather_fold_pipeline: mimi_rvq_gather_fold_pipeline.into_raw(),
            dac_rvq_gather_project_fold_pipeline: dac_rvq_gather_project_fold_pipeline.into_raw(),
            wavtokenizer_vq_gather_pipeline: wavtokenizer_vq_gather_pipeline.into_raw(),
            xcodec2_fsq_decode_pipeline: xcodec2_fsq_decode_pipeline.into_raw(),
            snake_activation_pipeline: snake_activation_pipeline.into_raw(),
            snake_beta_pipeline: snake_beta_pipeline.into_raw(),
            sinegen_deterministic_pipeline: sinegen_deterministic_pipeline.into_raw(),
            sinegen_deterministic_channel_major_pipeline:
                sinegen_deterministic_channel_major_pipeline.into_raw(),
            anti_aliased_upsample_pipeline: anti_aliased_upsample_pipeline.into_raw(),
            anti_aliased_downsample_pipeline: anti_aliased_downsample_pipeline.into_raw(),
            scale_pipeline: scale_pipeline.into_raw(),
            clamp_pipeline: clamp_pipeline.into_raw(),
            conv_transpose1d_pipeline: conv_transpose1d_pipeline.into_raw(),
            pad1d_pipeline: pad1d_pipeline.into_raw(),
            nearest_upsample_pipeline: nearest_upsample_pipeline.into_raw(),
            hift_stft_pipeline: hift_stft_pipeline.into_raw(),
            hift_complex_pipeline: hift_complex_pipeline.into_raw(),
            hift_istft_pipeline: hift_istft_pipeline.into_raw(),
            snac_decode_pipeline: snac_decode_pipeline.into_raw(),
            denoise_apply_mask_pipeline: denoise_apply_mask_pipeline.into_raw(),
            qwen3_tts_codec_decode_pipeline: qwen3_tts_codec_decode_pipeline.into_raw(),
            submissions: Cell::new(0),
            readbacks: Cell::new(0),
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
    /// A safe wrapper: non-empty `data` is a valid slice, so its pointer is
    /// valid for `size_of_val(data)` bytes, which is what
    /// `newBufferWithBytes:` copies. Metal does not reliably accept a
    /// zero-length `newBufferWithBytes:` allocation (and copying four bytes
    /// from an empty slice would be invalid), so an empty upload gets a
    /// one-f32 placeholder buffer instead. The logical tensor length remains
    /// zero and kernels must not dereference it.
    fn new_buffer_from_slice(&self, data: &[f32]) -> Result<OwnedBuf> {
        if data.is_empty() {
            return self.new_buffer_output(0);
        }
        let bytes = size_of_val(data);
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
        if data.is_empty() {
            return self.new_buffer_output(0);
        }
        let bytes = data.len();
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

    /// Byte-preserving upload for raw BF16 `u16` bit patterns. Apple targets
    /// supported by this crate are little-endian, so the in-memory bytes of a
    /// decoded `u16` are exactly the model's little-endian BF16 payload.
    fn new_buffer_from_u16(&self, data: &[u16]) -> Result<OwnedBuf> {
        if data.is_empty() {
            return self.new_buffer_output(0);
        }
        let bytes = data.len().checked_mul(size_of::<u16>()).ok_or_else(|| {
            VokraError::InvalidArgument("Metal BF16 buffer byte size overflow".to_owned())
        })?;
        // SAFETY: `device` is valid; a non-empty `u16` slice is valid for its
        // complete byte length. Apple Metal targets are little-endian, so the
        // raw memory bytes preserve each decoded BF16 bit pattern exactly.
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
                "MTLDevice newBufferWithBytes (BF16) returned nil".to_owned(),
            ));
        }
        Ok(OwnedBuf(buf))
    }

    /// Allocates an uninitialised shared-storage `MTLBuffer` of `len` f32s.
    fn new_buffer_output(&self, len: usize) -> Result<OwnedBuf> {
        let bytes = len
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| VokraError::InvalidArgument("Metal buffer size overflow".to_owned()))?
            .max(size_of::<f32>());
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

    /// Affine GroupNorm with one group over channel-major
    /// `[channels, positions]`, using the same 256-partial pairwise reduction
    /// as the CPU kernel.
    #[allow(clippy::too_many_arguments)]
    pub fn group_norm_f32(
        &self,
        input: &[f32],
        out: &mut [f32],
        channels: usize,
        positions: usize,
        gamma: &[f32],
        beta: &[f32],
        eps: f32,
    ) -> Result<()> {
        validate_group_norm(input, out, channels, positions, gamma, beta)?;
        // SAFETY: token consumed by the matching pop below.
        let pool = unsafe { sys::objc_autoreleasePoolPush() };
        let result = self.run_group_norm(input, out, channels, positions, gamma, beta, eps);
        // SAFETY: `pool` is the token from the push above.
        unsafe { sys::objc_autoreleasePoolPop(pool) };
        result
    }

    #[allow(clippy::too_many_arguments)]
    fn run_group_norm(
        &self,
        input: &[f32],
        out: &mut [f32],
        channels: usize,
        positions: usize,
        gamma: &[f32],
        beta: &[f32],
        eps: f32,
    ) -> Result<()> {
        let in_buf = self.new_buffer_from_slice(input)?;
        let gamma_buf = self.new_buffer_from_slice(gamma)?;
        let beta_buf = self.new_buffer_from_slice(beta)?;
        let out_buf = self.new_buffer_output(out.len())?;
        let dims = GroupNormDims {
            channels: channels as u32,
            positions: positions as u32,
            eps,
        };
        let (grid, tg) = grid_1d(1);
        self.dispatch_compute(
            self.group_norm_pipeline,
            &[&in_buf, &gamma_buf, &beta_buf, &out_buf],
            (&dims as *const GroupNormDims).cast::<c_void>(),
            size_of::<GroupNormDims>(),
            grid,
            tg,
            "group_norm",
        )?;
        read_back(&out_buf, out)
    }

    /// Affine multi-group GroupNorm over channel-major `[channels, positions]`.
    /// Each group reduces `channels / groups × positions`; gamma and beta are
    /// per-channel. This is the source NCSN++ path and never falls back to CPU.
    #[allow(clippy::too_many_arguments)]
    pub fn group_norm_groups_f32(
        &self,
        input: &[f32],
        out: &mut [f32],
        channels: usize,
        positions: usize,
        groups: usize,
        gamma: &[f32],
        beta: &[f32],
        eps: f32,
    ) -> Result<()> {
        validate_group_norm_groups(input, out, channels, positions, groups, gamma, beta, eps)?;
        if out.is_empty() {
            return Ok(());
        }
        // SAFETY: token consumed by the matching pop below.
        let pool = unsafe { sys::objc_autoreleasePoolPush() };
        let result =
            self.run_group_norm_groups(input, out, channels, positions, groups, gamma, beta, eps);
        // SAFETY: `pool` is the token from the push above.
        unsafe { sys::objc_autoreleasePoolPop(pool) };
        result?;
        if out.iter().any(|value| !value.is_finite()) {
            return Err(VokraError::InvalidArgument(
                "group_norm_groups output overflowed".to_owned(),
            ));
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn run_group_norm_groups(
        &self,
        input: &[f32],
        out: &mut [f32],
        channels: usize,
        positions: usize,
        groups: usize,
        gamma: &[f32],
        beta: &[f32],
        eps: f32,
    ) -> Result<()> {
        let in_buf = self.new_buffer_from_slice(input)?;
        let gamma_buf = self.new_buffer_from_slice(gamma)?;
        let beta_buf = self.new_buffer_from_slice(beta)?;
        let out_buf = self.new_buffer_output(out.len())?;
        let dims = GroupNormGroupsDims {
            channels: checked_u32(channels, "group_norm_groups channels")?,
            positions: checked_u32(positions, "group_norm_groups positions")?,
            groups: checked_u32(groups, "group_norm_groups groups")?,
            eps,
        };
        // The kernel owns one complete reduction/output loop per group.
        let (grid, tg) = grid_1d(groups);
        self.dispatch_compute(
            self.group_norm_groups_pipeline,
            &[&in_buf, &gamma_buf, &beta_buf, &out_buf],
            (&dims as *const GroupNormGroupsDims).cast::<c_void>(),
            size_of::<GroupNormGroupsDims>(),
            grid,
            tg,
            "group_norm_groups",
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

    /// Element-wise GPT-2 / Transformers `gelu_new` tanh approximation.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on a length mismatch;
    /// [`VokraError::BackendUnavailable`] on a Metal failure.
    pub fn gelu_new_f32(&self, x: &[f32], out: &mut [f32]) -> Result<()> {
        validate_unary(x, out)?;
        if out.is_empty() {
            return Ok(());
        }
        // SAFETY: token consumed by the matching pop below.
        let pool = unsafe { sys::objc_autoreleasePoolPush() };
        let r = self.run_gelu_new(x, out);
        // SAFETY: `pool` is the token from the push above.
        unsafe { sys::objc_autoreleasePoolPop(pool) };
        r
    }

    fn run_gelu_new(&self, x: &[f32], out: &mut [f32]) -> Result<()> {
        let x_buf = self.new_buffer_from_slice(x)?;
        let out_buf = self.new_buffer_output(out.len())?;
        let dims = GeluDims {
            n: out.len() as u32,
        };
        let (grid, tg) = grid_1d(out.len());
        self.dispatch_compute(
            self.gelu_new_pipeline,
            &[&x_buf, &out_buf],
            (&dims as *const GeluDims).cast::<c_void>(),
            size_of::<GeluDims>(),
            grid,
            tg,
            "gelu_new",
        )?;
        read_back(&out_buf, out)
    }

    /// Element-wise ReLU (`out = max(x, 0)`). This is the Metal half of the
    /// T5-base feed-forward activation used by MusicGen-family text encoders.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on a length mismatch;
    /// [`VokraError::BackendUnavailable`] on a Metal failure.
    pub fn relu_f32(&self, x: &[f32], out: &mut [f32]) -> Result<()> {
        validate_unary(x, out)?;
        if out.is_empty() {
            return Ok(());
        }
        // SAFETY: token consumed by the matching pop below.
        let pool = unsafe { sys::objc_autoreleasePoolPush() };
        let result = self.run_relu(x, out);
        // SAFETY: `pool` is the token from the push above.
        unsafe { sys::objc_autoreleasePoolPop(pool) };
        result
    }

    fn run_relu(&self, x: &[f32], out: &mut [f32]) -> Result<()> {
        let x_buf = self.new_buffer_from_slice(x)?;
        let out_buf = self.new_buffer_output(out.len())?;
        let dims = ReluDims {
            n: out.len() as u32,
        };
        let (grid, tg) = grid_1d(out.len());
        self.dispatch_compute(
            self.relu_pipeline,
            &[&x_buf, &out_buf],
            (&dims as *const ReluDims).cast::<c_void>(),
            size_of::<ReluDims>(),
            grid,
            tg,
            "relu",
        )?;
        read_back(&out_buf, out)
    }

    /// Element-wise ELU with the EnCodec/Bark default `alpha = 1`.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on a length mismatch;
    /// [`VokraError::BackendUnavailable`] on a Metal failure.
    pub fn elu_f32(&self, x: &[f32], out: &mut [f32]) -> Result<()> {
        validate_unary(x, out)?;
        if out.is_empty() {
            return Ok(());
        }
        // SAFETY: token consumed by the matching pop below.
        let pool = unsafe { sys::objc_autoreleasePoolPush() };
        let result = self.run_elu(x, out);
        // SAFETY: `pool` is the token from the push above.
        unsafe { sys::objc_autoreleasePoolPop(pool) };
        result
    }

    fn run_elu(&self, x: &[f32], out: &mut [f32]) -> Result<()> {
        let x_buf = self.new_buffer_from_slice(x)?;
        let out_buf = self.new_buffer_output(out.len())?;
        let dims = EluDims {
            n: out.len() as u32,
        };
        let (grid, tg) = grid_1d(out.len());
        self.dispatch_compute(
            self.elu_pipeline,
            &[&x_buf, &out_buf],
            (&dims as *const EluDims).cast::<c_void>(),
            size_of::<EluDims>(),
            grid,
            tg,
            "elu",
        )?;
        read_back(&out_buf, out)
    }

    /// Element-wise hyperbolic tangent. This is the Metal half of
    /// SpeechT5's activated postnet convolution blocks.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on a length mismatch;
    /// [`VokraError::BackendUnavailable`] on a Metal failure.
    pub fn tanh_f32(&self, x: &[f32], out: &mut [f32]) -> Result<()> {
        validate_unary(x, out)?;
        if out.is_empty() {
            return Ok(());
        }
        // SAFETY: token consumed by the matching pop below.
        let pool = unsafe { sys::objc_autoreleasePoolPush() };
        let result = self.run_tanh(x, out);
        // SAFETY: `pool` is the token from the push above.
        unsafe { sys::objc_autoreleasePoolPop(pool) };
        result
    }

    fn run_tanh(&self, x: &[f32], out: &mut [f32]) -> Result<()> {
        let x_buf = self.new_buffer_from_slice(x)?;
        let out_buf = self.new_buffer_output(out.len())?;
        let dims = TanhDims {
            n: out.len() as u32,
        };
        let (grid, tg) = grid_1d(out.len());
        self.dispatch_compute(
            self.tanh_pipeline,
            &[&x_buf, &out_buf],
            (&dims as *const TanhDims).cast::<c_void>(),
            size_of::<TanhDims>(),
            grid,
            tg,
            "tanh",
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

    /// Released MossFormer2 ScaleNorm applied row-wise:
    /// `out[r,c] = input[r,c] / max(||row||₂ · cols⁻¹ᐟ², eps) · gain`.
    ///
    /// ScaleNorm is not an RMSNorm alias: its epsilon clamps the completed
    /// norm instead of being added inside the square root. This dedicated
    /// kernel keeps the FLASH projection normalization on Metal and prevents
    /// a silent host reduction.
    pub fn scale_norm_f32(
        &self,
        input: &[f32],
        out: &mut [f32],
        rows: usize,
        cols: usize,
        gain: f32,
        eps: f32,
    ) -> Result<()> {
        validate_scale_norm(input, out, rows, cols, gain, eps)?;
        // SAFETY: token consumed by the matching pop below.
        let pool = unsafe { sys::objc_autoreleasePoolPush() };
        let result = self.run_scale_norm(input, out, rows, cols, gain, eps);
        // SAFETY: `pool` is the token from the push above.
        unsafe { sys::objc_autoreleasePoolPop(pool) };
        result
    }

    fn run_scale_norm(
        &self,
        input: &[f32],
        out: &mut [f32],
        rows: usize,
        cols: usize,
        gain: f32,
        eps: f32,
    ) -> Result<()> {
        let input_buffer = self.new_buffer_from_slice(input)?;
        let output_buffer = self.new_buffer_output(out.len())?;
        let dims = ScaleNormDims {
            rows: rows as u32,
            cols: cols as u32,
            scale: (cols as f64).sqrt().recip() as f32,
            eps,
            gain,
        };
        let (grid, threads) = grid_1d(rows);
        self.dispatch_compute(
            self.scale_norm_pipeline,
            &[&input_buffer, &output_buffer],
            (&dims as *const ScaleNormDims).cast::<c_void>(),
            size_of::<ScaleNormDims>(),
            grid,
            threads,
            "scale_norm",
        )?;
        read_back(&output_buffer, out)
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

    /// Host-facing OUVE predictor wrapper backed by the resident Metal kernel.
    /// This convenience API uploads its host slices and downloads the two
    /// requested outputs. Cross-step state residency is provided only by
    /// [`Self::ouve_reverse_diffusion_dev`], whose caller-owned device tensors
    /// remain on Metal until an explicit [`Self::download`].
    #[allow(clippy::too_many_arguments)]
    pub fn ouve_reverse_diffusion_f32(
        &self,
        theta: f32,
        sigma_min: f32,
        sigma_max: f32,
        x: &[f32],
        y: &[f32],
        score: &[f32],
        t: f32,
        step: f32,
        noise: &[f32],
        probability_flow: bool,
        out: &mut [f32],
        out_mean: &mut [f32],
    ) -> Result<()> {
        validate_ouve_host_buffers(x, y, score, noise, out, out_mean)?;
        validate_ouve_params(theta, sigma_min, sigma_max, t, step, None)?;
        let x_dev = self.upload(x)?;
        let y_dev = self.upload(y)?;
        let score_dev = self.upload(score)?;
        let noise_dev = self.upload(noise)?;
        let mut out_dev = self.alloc_dev(out.len())?;
        let mut mean_dev = self.alloc_dev(out_mean.len())?;
        self.ouve_reverse_diffusion_dev(
            &mut out_dev,
            &mut mean_dev,
            &x_dev,
            &y_dev,
            &score_dev,
            &noise_dev,
            theta,
            sigma_min,
            sigma_max,
            t,
            step,
            probability_flow,
        )?;
        self.download(&out_dev, out)?;
        self.download(&mean_dev, out_mean)
    }

    /// Host-facing OUVE annealed-Langevin wrapper backed by the resident Metal
    /// kernel. It uploads host slices and downloads the requested outputs; no
    /// CPU fallback or per-element host computation is performed. Use
    /// [`Self::ouve_annealed_langevin_dev`] to keep cross-step state resident.
    #[allow(clippy::too_many_arguments)]
    pub fn ouve_annealed_langevin_f32(
        &self,
        theta: f32,
        sigma_min: f32,
        sigma_max: f32,
        x: &[f32],
        score: &[f32],
        t: f32,
        snr: f32,
        noise: &[f32],
        out: &mut [f32],
        out_mean: &mut [f32],
    ) -> Result<()> {
        validate_ouve_host_buffers_unary(x, score, noise, out, out_mean)?;
        validate_ouve_params(theta, sigma_min, sigma_max, t, 1.0, Some(snr))?;
        let x_dev = self.upload(x)?;
        let score_dev = self.upload(score)?;
        let noise_dev = self.upload(noise)?;
        let mut out_dev = self.alloc_dev(out.len())?;
        let mut mean_dev = self.alloc_dev(out_mean.len())?;
        self.ouve_annealed_langevin_dev(
            &mut out_dev,
            &mut mean_dev,
            &x_dev,
            &score_dev,
            &noise_dev,
            theta,
            sigma_min,
            sigma_max,
            t,
            snr,
        )?;
        self.download(&out_dev, out)?;
        self.download(&mean_dev, out_mean)
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

    /// Vocoder Metal wave WF2: Snake activation
    /// (`vokra_snake_activation_f32`) — per-channel closed-form periodic
    /// activation on the GPU.
    ///
    /// Applies `out[c, t] = x[c, t] + (1 / (alpha[c] + ε)) · sin(alpha[c] ·
    /// x[c, t])²` for a `[channels, time]` row-major FP32 tensor (channel-
    /// outer). `alpha` is length-`channels`; `x` and `out` are both length
    /// `channels · time`. The exact contract of
    /// [`vokra_ops::snake_activation_f32`], which itself is bit-identical to
    /// [`vokra_ops::hiftnet::Snake::forward_in_place`] under
    /// `alpha_logscale = false` and the private `kokoro::nn::snake_activation`
    /// helper in vokra-models (same eps, same primitives, same reduction
    /// order — trivial per-element, no reduction).
    ///
    /// # Numerics
    ///
    /// FP32 accumulator on both sides; MSL is compiled with fast-math
    /// defaults, and the intrinsic `sin` may differ from Rust's `f32::sin`
    /// in the low bits. The CPU vs GPU bound is `atol ≤ 5e-4` (the same
    /// codec-family bound used by [`Self::mimi_rvq_gather_fold_f32`] /
    /// [`Self::dac_rvq_gather_project_fold_f32`] / the FSQ family) — not a
    /// bit-for-bit equality, because that would over-constrain the
    /// transcendental. In practice max |Δ| stays well inside 5e-4 for
    /// finite inputs (see `tests/snake_activation_metal_bit_identical.rs`).
    ///
    /// # Errors
    ///
    /// - [`VokraError::InvalidArgument`] on `alpha.len() != channels`,
    ///   `x.len() != channels * time`, `out.len() != channels * time`, or a
    ///   `channels * time` overflow. Mirrors the CPU
    ///   [`vokra_ops::snake_activation_f32`] guards.
    /// - [`VokraError::BackendUnavailable`] on a Metal buffer / command-
    ///   buffer / pipeline dispatch failure.
    pub fn snake_activation_f32(
        &self,
        x: &[f32],
        alpha: &[f32],
        channels: usize,
        time: usize,
        out: &mut [f32],
    ) -> Result<()> {
        // Mirror the CPU op's shape guards on the host side (the MSL kernel
        // guards `c >= channels` and `t >= time` but assumes the buffers
        // have the expected element counts, so a wrong-shape upload would be
        // an OOB read — FR-EX-08 forbids silent OOB).
        let expected = checked_mul(channels, time, "snake_activation channels*time")?;
        expect_len("snake_activation alpha", alpha.len(), channels)?;
        expect_len("snake_activation x", x.len(), expected)?;
        expect_len("snake_activation out", out.len(), expected)?;
        if channels == 0 || time == 0 {
            // Nothing to write — mirrors the CPU no-op path; both buffers
            // are already empty per the length checks above.
            return Ok(());
        }
        // SAFETY: token consumed by the matching pop below.
        let pool = unsafe { sys::objc_autoreleasePoolPush() };
        let r = self.run_snake_activation(x, alpha, channels, time, out);
        // SAFETY: `pool` is the token from the push above.
        unsafe { sys::objc_autoreleasePoolPop(pool) };
        r
    }

    fn run_snake_activation(
        &self,
        x: &[f32],
        alpha: &[f32],
        channels: usize,
        time: usize,
        out: &mut [f32],
    ) -> Result<()> {
        let x_buf = self.new_buffer_from_slice(x)?;
        let alpha_buf = self.new_buffer_from_slice(alpha)?;
        let out_buf = self.new_buffer_output(out.len())?;
        let dims = SnakeActivationDims {
            channels: channels as u32,
            time: time as u32,
        };
        // One thread per (t, c). `time` on the fast axis (grid.x) matches the
        // row-major stride and keeps adjacent threads reading adjacent
        // floats. The kernel guards the ragged tail against both bounds so
        // `grid_2d`'s round-up-to-16 is safe.
        let (grid, tg) = grid_2d(time, channels);
        self.dispatch_compute(
            self.snake_activation_pipeline,
            &[&x_buf, &alpha_buf, &out_buf],
            (&dims as *const SnakeActivationDims).cast::<c_void>(),
            size_of::<SnakeActivationDims>(),
            grid,
            tg,
            "snake_activation",
        )?;
        read_back(&out_buf, out)
    }

    /// Vocoder Metal wave common vocoder primitive: SnakeBeta activation
    /// (`vokra_snake_beta_f32`) — the per-channel two-vector closed-form
    /// periodic activation on the GPU.
    ///
    /// Applies `out[c, t] = x[c, t] + (1 / (beta[c] + ε)) · sin(alpha[c] ·
    /// x[c, t])²` for a `[channels, time]` row-major FP32 tensor (channel-
    /// outer). `alpha` and `beta` are length-`channels`; `x` and `out` are
    /// both length `channels · time`. The exact contract of
    /// [`vokra_ops::snake_beta_f32`], which itself matches
    /// [`vokra_ops::bigvgan_generator::SnakeBeta::forward_in_place`] under
    /// `alpha_logscale = false` (same eps, same primitives, trivial per-
    /// element).
    ///
    /// # Numerics
    ///
    /// FP32 accumulator on both sides; MSL is compiled with fast-math
    /// defaults, and the intrinsic `sin` may differ from Rust's `f32::sin`
    /// in the low bits. The CPU vs GPU bound is `atol ≤ 5e-4` (same
    /// codec-family bound as the sibling snake_activation kernel).
    ///
    /// # Errors
    ///
    /// - [`VokraError::InvalidArgument`] on `alpha.len() != channels`,
    ///   `beta.len() != channels`, `x.len() != channels * time`,
    ///   `out.len() != channels * time`, or a `channels * time` overflow.
    /// - [`VokraError::BackendUnavailable`] on a Metal buffer / command-
    ///   buffer / pipeline dispatch failure.
    pub fn snake_beta_f32(
        &self,
        x: &[f32],
        alpha: &[f32],
        beta: &[f32],
        channels: usize,
        time: usize,
        out: &mut [f32],
    ) -> Result<()> {
        let expected = checked_mul(channels, time, "snake_beta channels*time")?;
        expect_len("snake_beta alpha", alpha.len(), channels)?;
        expect_len("snake_beta beta", beta.len(), channels)?;
        expect_len("snake_beta x", x.len(), expected)?;
        expect_len("snake_beta out", out.len(), expected)?;
        if channels == 0 || time == 0 {
            return Ok(());
        }
        // SAFETY: token consumed by the matching pop below.
        let pool = unsafe { sys::objc_autoreleasePoolPush() };
        let r = self.run_snake_beta(x, alpha, beta, channels, time, out);
        // SAFETY: `pool` is the token from the push above.
        unsafe { sys::objc_autoreleasePoolPop(pool) };
        r
    }

    #[allow(clippy::too_many_arguments)] // intrinsic to the two-vector SnakeBeta shape
    fn run_snake_beta(
        &self,
        x: &[f32],
        alpha: &[f32],
        beta: &[f32],
        channels: usize,
        time: usize,
        out: &mut [f32],
    ) -> Result<()> {
        let x_buf = self.new_buffer_from_slice(x)?;
        let alpha_buf = self.new_buffer_from_slice(alpha)?;
        let beta_buf = self.new_buffer_from_slice(beta)?;
        let out_buf = self.new_buffer_output(out.len())?;
        let dims = SnakeBetaDims {
            channels: channels as u32,
            time: time as u32,
        };
        let (grid, tg) = grid_2d(time, channels);
        self.dispatch_compute(
            self.snake_beta_pipeline,
            &[&x_buf, &alpha_buf, &beta_buf, &out_buf],
            (&dims as *const SnakeBetaDims).cast::<c_void>(),
            size_of::<SnakeBetaDims>(),
            grid,
            tg,
            "snake_beta",
        )?;
        read_back(&out_buf, out)
    }

    /// Vocoder Metal wave common vocoder primitive: SineGen deterministic
    /// forward (`vokra_sinegen_deterministic_f32`) — the F0-driven multi-
    /// harmonic sinusoid source of HiFTNet-family vocoders, on the GPU.
    ///
    /// Writes `t * (harmonic_num + 1)` FP32 samples to `out`, matching the
    /// deterministic path of [`vokra_ops::nsf::SineGen::forward`]
    /// bit-for-bit modulo the MSL `sin` transcendental gap
    /// (`atol ≤ 5e-4`). Output layout is `[T, H+1]` row-major
    /// (time-outer / harmonic-inner — upstream
    /// `sine_wavs.transpose(1, 2)`).
    ///
    /// # Parameters
    ///
    /// - `f0`               — length-`t` FP32 fundamental frequency per sample.
    /// - `samp_rate`        — audio sample rate (Hz); `> 0`.
    /// - `harmonic_num`     — number of harmonics beyond the fundamental.
    /// - `sine_amp`         — sinusoid amplitude scale.
    /// - `voiced_threshold` — F0 threshold above which a frame is voiced.
    /// - `out`              — output buffer of length `t * (harmonic_num + 1)`.
    ///
    /// # Errors
    ///
    /// - [`VokraError::InvalidArgument`] on empty `f0`, `samp_rate == 0`, or
    ///   `out.len() != t * (harmonic_num + 1)`.
    /// - [`VokraError::BackendUnavailable`] on a Metal buffer / command-
    ///   buffer / pipeline dispatch failure.
    pub fn sinegen_deterministic_f32(
        &self,
        f0: &[f32],
        samp_rate: u32,
        harmonic_num: u32,
        sine_amp: f32,
        voiced_threshold: f32,
        out: &mut [f32],
    ) -> Result<()> {
        let t = f0.len();
        if t == 0 {
            return Err(VokraError::InvalidArgument(
                "sinegen_deterministic_f32: empty f0 sequence".to_owned(),
            ));
        }
        if samp_rate == 0 {
            return Err(VokraError::InvalidArgument(
                "sinegen_deterministic_f32: samp_rate must be > 0".to_owned(),
            ));
        }
        let h1 = harmonic_num as usize + 1;
        let expected = checked_mul(t, h1, "sinegen_deterministic t*(H+1)")?;
        expect_len("sinegen_deterministic out", out.len(), expected)?;
        // SAFETY: token consumed by the matching pop below.
        let pool = unsafe { sys::objc_autoreleasePoolPush() };
        let r = self.run_sinegen_deterministic(
            f0,
            samp_rate,
            harmonic_num,
            sine_amp,
            voiced_threshold,
            out,
        );
        // SAFETY: `pool` is the token from the push above.
        unsafe { sys::objc_autoreleasePoolPop(pool) };
        r
    }

    fn run_sinegen_deterministic(
        &self,
        f0: &[f32],
        samp_rate: u32,
        harmonic_num: u32,
        sine_amp: f32,
        voiced_threshold: f32,
        out: &mut [f32],
    ) -> Result<()> {
        let f0_buf = self.new_buffer_from_slice(f0)?;
        let out_buf = self.new_buffer_output(out.len())?;
        let h1 = harmonic_num + 1;
        let dims = SinegenDeterministicDims {
            t: f0.len() as u32,
            h1,
            samp_rate_f: samp_rate as f32,
            sine_amp,
            voiced_threshold,
        };
        // 1-D launch: one thread per harmonic. `grid_1d` uses TG=256, so a
        // typical H+1 = 1..10 vocoder produces a single threadgroup with a
        // very short active prefix (the kernel guards `i >= h1` for the tail).
        let (grid, tg) = grid_1d(h1 as usize);
        self.dispatch_compute(
            self.sinegen_deterministic_pipeline,
            &[&f0_buf, &out_buf],
            (&dims as *const SinegenDeterministicDims).cast::<c_void>(),
            size_of::<SinegenDeterministicDims>(),
            grid,
            tg,
            "sinegen_deterministic",
        )?;
        read_back(&out_buf, out)
    }

    /// Vocoder Metal wave common vocoder primitive: polyphase anti-aliased
    /// upsample (`vokra_anti_aliased_upsample_f32`) — the multiply-add core
    /// of BigVGAN's `UpSample1d` and every HiFTNet-family alias-free
    /// activation chain, on the GPU.
    ///
    /// Writes `channels * time_in * ratio` FP32 samples to `out`, matching
    /// [`vokra_ops::anti_aliased_upsample_f32`] within `atol ≤ 1e-4` (the
    /// FMA-vs-non-FMA gap between MSL fast-math and the CPU strict-left-fold
    /// FIR accumulator; see the module docstring).
    ///
    /// # Parameters
    ///
    /// - `x`         — `[channels, time_in]` row-major FP32 input.
    /// - `kernel`    — `[taps]` FP32 causal low-pass filter taps.
    /// - `ratio`     — integer upsample factor (`>= 1`).
    /// - `channels`  — number of channels in `x` / `out`.
    /// - `time_in`   — number of input timesteps.
    /// - `out`       — `[channels, time_in * ratio]` row-major FP32 output.
    ///
    /// # Errors
    ///
    /// - [`VokraError::InvalidArgument`] on `ratio == 0`, empty `kernel`,
    ///   `x.len() != channels * time_in`, `out.len() != channels * time_in
    ///   * ratio`, or a dimension overflow.
    /// - [`VokraError::BackendUnavailable`] on a Metal buffer / command-
    ///   buffer / pipeline dispatch failure.
    #[allow(clippy::too_many_arguments)] // intrinsic to the polyphase upsample shape
    pub fn anti_aliased_upsample_f32(
        &self,
        x: &[f32],
        kernel: &[f32],
        ratio: usize,
        channels: usize,
        time_in: usize,
        out: &mut [f32],
    ) -> Result<()> {
        if ratio == 0 {
            return Err(VokraError::InvalidArgument(
                "anti_aliased_upsample_f32: ratio must be >= 1".to_owned(),
            ));
        }
        if kernel.is_empty() {
            return Err(VokraError::InvalidArgument(
                "anti_aliased_upsample_f32: kernel must not be empty".to_owned(),
            ));
        }
        let expected_x = checked_mul(channels, time_in, "anti_aliased_upsample channels*time_in")?;
        let time_out = checked_mul(time_in, ratio, "anti_aliased_upsample time_in*ratio")?;
        let expected_out = checked_mul(
            channels,
            time_out,
            "anti_aliased_upsample channels*(time_in*ratio)",
        )?;
        expect_len("anti_aliased_upsample x", x.len(), expected_x)?;
        expect_len("anti_aliased_upsample out", out.len(), expected_out)?;
        if channels == 0 || time_in == 0 {
            return Ok(());
        }
        // SAFETY: token consumed by the matching pop below.
        let pool = unsafe { sys::objc_autoreleasePoolPush() };
        let r = self.run_anti_aliased_upsample(x, kernel, ratio, channels, time_in, time_out, out);
        // SAFETY: `pool` is the token from the push above.
        unsafe { sys::objc_autoreleasePoolPop(pool) };
        r
    }

    #[allow(clippy::too_many_arguments)] // intrinsic to the polyphase upsample shape
    fn run_anti_aliased_upsample(
        &self,
        x: &[f32],
        kernel: &[f32],
        ratio: usize,
        channels: usize,
        time_in: usize,
        time_out: usize,
        out: &mut [f32],
    ) -> Result<()> {
        let x_buf = self.new_buffer_from_slice(x)?;
        let kernel_buf = self.new_buffer_from_slice(kernel)?;
        let out_buf = self.new_buffer_output(out.len())?;
        let dims = AntiAliasedUpsampleDims {
            channels: channels as u32,
            time_in: time_in as u32,
            time_out: time_out as u32,
            ratio: ratio as u32,
            taps: kernel.len() as u32,
        };
        // 2-D launch: one thread per (t_out, c). `time_out` on the fast axis
        // (grid.x) matches the row-major stride and keeps adjacent threads
        // reading adjacent output floats. The kernel guards the ragged tail
        // against both bounds.
        let (grid, tg) = grid_2d(time_out, channels);
        self.dispatch_compute(
            self.anti_aliased_upsample_pipeline,
            &[&x_buf, &kernel_buf, &out_buf],
            (&dims as *const AntiAliasedUpsampleDims).cast::<c_void>(),
            size_of::<AntiAliasedUpsampleDims>(),
            grid,
            tg,
            "anti_aliased_upsample",
        )?;
        read_back(&out_buf, out)
    }

    /// Vocoder Metal wave WF5: SNAC 3-stage hierarchical RVQ decode —
    /// per-stage gather + factorized projection + temporal upsample + FP32
    /// residual sum on the GPU.
    ///
    /// Returns a fresh `[t_expanded × d_model]` row-major `Vec<f32>` where
    /// `out[t, :] = Σ_s (W_s @ codebooks[s].row(codes[s][t / strides[s]]) +
    /// b_s)` — the exact contract of
    /// `vokra_ops::snac_decode::SnacDecoder::decode` (upstream
    /// `ResidualVectorQuantize.from_codes` at `hubertsiuzdak/snac/blob/main
    /// /snac/vq.py` L61-71). Heap-returning (not `out: &mut [f32]`) for the
    /// same chunk-granularity reason as
    /// [`Self::mimi_rvq_gather_fold_f32`] / the DAC / FSQ family — this is a
    /// codec-side chunk op, not a per-token hot path.
    ///
    /// # Parameters
    ///
    /// - `codes_flat` — `[Σ codes[s].len()]` row-major `u32` codebook indices
    ///   concatenated across the 3 stages. Stage `s`'s frame `t_stage`
    ///   is `codes_flat[stage_offsets[s] + t_stage]`. Every entry must
    ///   satisfy `idx < codebook_size` — the caller validates on the host
    ///   before dispatch (FR-EX-08 — the MSL kernel has no per-element
    ///   bound check, so silent OOB reads are the failure mode we prevent
    ///   by delegating to explicit host-side validation).
    /// - `stage_offsets` — `[3]` start of each stage in `codes_flat`. By
    ///   construction `stage_offsets[0] = 0`; `stage_offsets[1] =
    ///   len(codes[0])`; `stage_offsets[2] = len(codes[0]) + len(codes[1])`.
    /// - `strides` — `[3]` per-stage temporal strides. Every entry must be
    ///   `> 0` (upstream `SnacDecoder::new` rejects `stride = 0` because it
    ///   would divide the base frame rate by zero — FR-EX-08).
    /// - `codebooks_flat` — `[3 × codebook_size × codebook_dim]` row-major
    ///   FP32; stage `s` starts at `s * codebook_size * codebook_dim`.
    /// - `proj_weights_flat` — `[3 × d_model × codebook_dim]` row-major
    ///   FP32; stage `s`'s W row `o` starts at `s * d_model * codebook_dim
    ///   + o * codebook_dim`.
    /// - `proj_biases_flat` — `[3 × d_model]` row-major FP32; stage `s`'s
    ///   bias for output `o` at `s * d_model + o`.
    /// - `codebook_size` / `codebook_dim` / `d_model` — decode shape,
    ///   shared across the three stages by construction. All must be `> 0`
    ///   (an empty `t_expanded` short-circuits at the caller with an empty
    ///   `Vec<f32>`; every other zero axis is an explicit
    ///   `InvalidArgument`).
    /// - `t_expanded` — number of output timesteps (== `codes[s].len() *
    ///   strides[s]` for every stage — the "co-aligned base frames"
    ///   invariant `SnacDecoder::check_and_measure` enforces).
    ///
    /// # Numerics
    ///
    /// FP32 accumulator on both sides; MSL is compiled with fast-math
    /// defaults which permit re-association of the inner
    /// `Σ_c W[o, c] · low[c]` GEMV, so the CPU vs GPU bound is
    /// `atol ≤ 5e-4` (the same FP32 GEMV-scale bound used by the sibling
    /// [`Self::mimi_rvq_gather_fold_f32`] /
    /// [`Self::dac_rvq_gather_project_fold_f32`] / the FSQ family) — not
    /// bit-for-bit equality. The temporal-upsample step
    /// (`t_stage = t_out / stride`) is exact integer division and adds no
    /// numeric slack.
    ///
    /// # Errors
    ///
    /// - [`VokraError::InvalidArgument`] on a shape mismatch, a zero axis in
    ///   `codebook_size` / `codebook_dim` / `d_model`, a zero stride, or any
    ///   overflow in the size / offset math.
    /// - [`VokraError::BackendUnavailable`] on a Metal buffer / command-
    ///   buffer / pipeline dispatch failure.
    #[allow(clippy::too_many_arguments)] // intrinsic to SNAC's 3-stage factorized shape
    pub fn snac_decode_f32(
        &self,
        codes_flat: &[u32],
        stage_offsets: [u32; 4],
        strides: [u32; 4],
        n_stages: usize,
        codebooks_flat: &[f32],
        proj_weights_flat: &[f32],
        proj_biases_flat: &[f32],
        codebook_size: usize,
        codebook_dim: usize,
        d_model: usize,
        t_expanded: usize,
    ) -> Result<Vec<f32>> {
        // Explicit shape validation. The MSL kernel guards `t >= t_expanded`
        // and `d >= d_model` but assumes every buffer has the expected
        // element count, so a wrong-shape upload would be a silent OOB read
        // (FR-EX-08 forbids). Mirror the vokra_ops::snac_decode shape checks.
        if codebook_size == 0 || codebook_dim == 0 || d_model == 0 {
            return Err(VokraError::InvalidArgument(format!(
                "snac_decode_f32: codebook_size / codebook_dim / d_model must all be > 0, got \
                 codebook_size={codebook_size} codebook_dim={codebook_dim} d_model={d_model}"
            )));
        }
        if !(1..=4).contains(&n_stages) {
            return Err(VokraError::InvalidArgument(format!(
                "snac_decode_f32: n_stages {n_stages} is outside 1..=4"
            )));
        }
        for (s, &stride) in strides[..n_stages].iter().enumerate() {
            if stride == 0 {
                return Err(VokraError::InvalidArgument(format!(
                    "snac_decode_f32: strides[{s}] = 0 (would divide base frame rate by zero; \
                     FR-EX-08 catches it upstream)"
                )));
            }
        }
        // Codebook / projection buffer size checks (per-stage sizes multiplied
        // by the active stage count).
        let per_stage_cb = codebook_size.checked_mul(codebook_dim).ok_or_else(|| {
            VokraError::InvalidArgument(format!(
                "snac_decode_f32: codebook_size * codebook_dim overflows usize \
                         (codebook_size={codebook_size} codebook_dim={codebook_dim})"
            ))
        })?;
        let expected_cb = per_stage_cb.checked_mul(n_stages).ok_or_else(|| {
            VokraError::InvalidArgument(
                "snac_decode_f32: n_stages * codebook_size * codebook_dim overflows usize"
                    .to_owned(),
            )
        })?;
        if codebooks_flat.len() != expected_cb {
            return Err(VokraError::InvalidArgument(format!(
                "snac_decode_f32: codebooks_flat.len() {} != n_stages ({n_stages}) * \
                 codebook_size * codebook_dim = {expected_cb}",
                codebooks_flat.len(),
            )));
        }
        let per_stage_w = d_model.checked_mul(codebook_dim).ok_or_else(|| {
            VokraError::InvalidArgument(format!(
                "snac_decode_f32: d_model * codebook_dim overflows usize \
                 (d_model={d_model} codebook_dim={codebook_dim})"
            ))
        })?;
        let expected_w = per_stage_w.checked_mul(n_stages).ok_or_else(|| {
            VokraError::InvalidArgument(
                "snac_decode_f32: n_stages * d_model * codebook_dim overflows usize".to_owned(),
            )
        })?;
        if proj_weights_flat.len() != expected_w {
            return Err(VokraError::InvalidArgument(format!(
                "snac_decode_f32: proj_weights_flat.len() {} != n_stages ({n_stages}) * \
                 d_model * codebook_dim = {expected_w}",
                proj_weights_flat.len(),
            )));
        }
        let expected_b = d_model.checked_mul(n_stages).ok_or_else(|| {
            VokraError::InvalidArgument(
                "snac_decode_f32: n_stages * d_model overflows usize".to_owned(),
            )
        })?;
        if proj_biases_flat.len() != expected_b {
            return Err(VokraError::InvalidArgument(format!(
                "snac_decode_f32: proj_biases_flat.len() {} != n_stages ({n_stages}) * \
                 d_model = {expected_b}",
                proj_biases_flat.len(),
            )));
        }
        // Per-stage code count check: `codes[s].len() * strides[s]` must
        // equal `t_expanded` for every stage (SnacDecoder::check_and_measure
        // invariant). We reconstruct `codes[s].len()` from
        // `stage_offsets` + `codes_flat.len()`.
        if stage_offsets[0] != 0 {
            return Err(VokraError::InvalidArgument(format!(
                "snac_decode_f32: stage_offsets[0] must be 0, got {}",
                stage_offsets[0]
            )));
        }
        let mut stage_lens = [0usize; 4];
        for s in 0..n_stages {
            let start = stage_offsets[s] as usize;
            let end = if s + 1 < n_stages {
                stage_offsets[s + 1] as usize
            } else {
                codes_flat.len()
            };
            stage_lens[s] = end.checked_sub(start).ok_or_else(|| {
                VokraError::InvalidArgument(format!(
                    "snac_decode_f32: stage end {end} < stage_offsets[{s}] {start}"
                ))
            })?;
            if end > codes_flat.len() {
                return Err(VokraError::InvalidArgument(format!(
                    "snac_decode_f32: stage {s} end {end} exceeds codes_flat.len() {}",
                    codes_flat.len()
                )));
            }
        }
        for (s, &len) in stage_lens[..n_stages].iter().enumerate() {
            let expanded = len.checked_mul(strides[s] as usize).ok_or_else(|| {
                VokraError::InvalidArgument(format!(
                    "snac_decode_f32: stage {s} codes.len() ({len}) * strides[{s}] ({}) \
                     overflows usize",
                    strides[s]
                ))
            })?;
            if expanded != t_expanded {
                return Err(VokraError::InvalidArgument(format!(
                    "snac_decode_f32: stage {s} expands to T={expanded}, but t_expanded={t_expanded} \
                     (SNAC's co-aligned base frames invariant — every stage must expand to the \
                     same T)"
                )));
            }
        }
        // Per-index bound check — the MSL kernel does NOT range-check
        // `codes[..]`. Cheap: O(Σ codes[s].len()) unpredictable branches,
        // dwarfed by the GPU dispatch.
        for (i, &idx) in codes_flat.iter().enumerate() {
            if (idx as usize) >= codebook_size {
                return Err(VokraError::InvalidArgument(format!(
                    "snac_decode_f32: codes_flat[{i}] = {idx} >= codebook_size {codebook_size} \
                     (no silent clamp — FR-EX-08)"
                )));
            }
        }
        // Empty output → return an empty Vec, mirroring
        // `SnacDecoder::decode` (zero-`t_expanded` decode is well-defined
        // and returns an empty `Vec`; every other empty axis is a shape
        // error above).
        if t_expanded == 0 {
            return Ok(Vec::new());
        }
        // SAFETY: token consumed by the matching pop below.
        let pool = unsafe { sys::objc_autoreleasePoolPush() };
        let r = self.run_snac_decode(
            codes_flat,
            stage_offsets,
            strides,
            n_stages,
            codebooks_flat,
            proj_weights_flat,
            proj_biases_flat,
            codebook_size,
            codebook_dim,
            d_model,
            t_expanded,
        );
        // SAFETY: `pool` is the token from the push above.
        unsafe { sys::objc_autoreleasePoolPop(pool) };
        r
    }

    #[allow(clippy::too_many_arguments)] // intrinsic to SNAC's 3-stage factorized shape
    fn run_snac_decode(
        &self,
        codes_flat: &[u32],
        stage_offsets: [u32; 4],
        strides: [u32; 4],
        n_stages: usize,
        codebooks_flat: &[f32],
        proj_weights_flat: &[f32],
        proj_biases_flat: &[f32],
        codebook_size: usize,
        codebook_dim: usize,
        d_model: usize,
        t_expanded: usize,
    ) -> Result<Vec<f32>> {
        // SAFETY: `codes_flat` is a valid read-only slice of `u32`; reinterpret
        // its backing storage as a `u8` slice of the same byte length for the
        // byte-oriented shared MTLBuffer upload. `u32` alignment is stricter
        // than `u8`, so the pointer cast is well-defined, and the borrow scope
        // of `codes_bytes` is limited to this function.
        let codes_bytes: &[u8] = unsafe {
            core::slice::from_raw_parts(
                codes_flat.as_ptr().cast::<u8>(),
                core::mem::size_of_val(codes_flat),
            )
        };
        let codes_buf = self.new_buffer_from_bytes(codes_bytes)?;
        let cb_buf = self.new_buffer_from_slice(codebooks_flat)?;
        let w_buf = self.new_buffer_from_slice(proj_weights_flat)?;
        let b_buf = self.new_buffer_from_slice(proj_biases_flat)?;
        let out_len = t_expanded * d_model;
        let out_buf = self.new_buffer_output(out_len)?;
        let dims = SnacDecodeDims {
            d_model: d_model as u32,
            codebook_dim: codebook_dim as u32,
            codebook_size: codebook_size as u32,
            t_expanded: t_expanded as u32,
            n_stages: n_stages as u32,
            strides,
            stage_offsets,
        };
        // One thread per (d_model column, t_expanded row) — same launch
        // geometry as the sibling mimi_rvq / dac_rvq kernels for a
        // consistent codec-family shape. The kernel guards the ragged tail
        // against both bounds so `grid_2d`'s round-up-to-16 is safe.
        let (grid, tg) = grid_2d(d_model, t_expanded);
        self.dispatch_compute(
            self.snac_decode_pipeline,
            &[&codes_buf, &cb_buf, &w_buf, &b_buf, &out_buf],
            (&dims as *const SnacDecodeDims).cast::<c_void>(),
            size_of::<SnacDecodeDims>(),
            grid,
            tg,
            "snac_decode",
        )?;
        let mut out = vec![0.0_f32; out_len];
        read_back(&out_buf, &mut out)?;
        Ok(out)
    }

    /// Vocoder Metal wave WF5: denoise spectral-gate primitive
    /// (`vokra_denoise_apply_mask_f32`) — element-wise complex × real gain
    /// multiply on the GPU (phase-preserving spectral gate).
    ///
    /// Applies `out_re[t, f] = spec_re[t, f] · gain[t, f]` and
    /// `out_im[t, f] = spec_im[t, f] · gain[t, f]` for a `[n_frames, n_bins]`
    /// row-major FP32 complex spectrogram (bins on the inner stride — the
    /// layout `Spectrogram { re, im }` uses in `vokra_ops::denoise`). The
    /// exact contract of [`vokra_ops::denoise_apply_mask_f32`], which itself
    /// reproduces the [`vokra_ops::denoise::DenoiseModel::enhance_inner`]
    /// output-stage loop (denoise.rs L1852-1870) when the caller
    /// pre-expands the ERB mask through `erb_inv_fb` to per-position gains.
    ///
    /// # Numerics
    ///
    /// FP32 throughout on both sides. Multiplication is IEEE-754
    /// correctly-rounded on every finite input; there is no reduction, no
    /// transcendental, and no FMA opportunity (a single `re * g` cannot be
    /// fused with anything), so CPU and GPU produce **bit-for-bit identical**
    /// outputs. The sibling `atol ≤ 5e-4` codec-family bound in the parity
    /// harness keeps a discriminating negative control while the measured
    /// max |Δ| is logged (0 in practice — any future drift becomes visible).
    ///
    /// # Errors
    ///
    /// - [`VokraError::InvalidArgument`] on any of `spec_re.len()`,
    ///   `spec_im.len()`, `gain.len()`, `out_re.len()`, `out_im.len()` !=
    ///   `n_frames * n_bins`, or an overflow in `n_frames * n_bins`. Mirrors
    ///   the CPU [`vokra_ops::denoise_apply_mask_f32`] guards; the MSL
    ///   kernel guards the ragged grid tail but assumes the buffers have
    ///   the expected element counts, so a wrong-shape upload would be a
    ///   silent OOB (FR-EX-08 forbids).
    /// - [`VokraError::BackendUnavailable`] on a Metal buffer / command-
    ///   buffer / pipeline dispatch failure.
    #[allow(clippy::too_many_arguments)] // intrinsic to the two-output complex-multiply shape
    pub fn denoise_apply_mask_f32(
        &self,
        spec_re: &[f32],
        spec_im: &[f32],
        gain: &[f32],
        n_frames: usize,
        n_bins: usize,
        out_re: &mut [f32],
        out_im: &mut [f32],
    ) -> Result<()> {
        // Mirror the CPU op's shape guards on the host side. The MSL kernel
        // guards `f >= n_bins` and `t >= n_frames` but assumes the buffers
        // have the expected element counts, so a wrong-shape upload would be
        // an OOB read — FR-EX-08 forbids silent OOB.
        let expected = checked_mul(n_frames, n_bins, "denoise_apply_mask n_frames*n_bins")?;
        expect_len("denoise_apply_mask spec_re", spec_re.len(), expected)?;
        expect_len("denoise_apply_mask spec_im", spec_im.len(), expected)?;
        expect_len("denoise_apply_mask gain", gain.len(), expected)?;
        expect_len("denoise_apply_mask out_re", out_re.len(), expected)?;
        expect_len("denoise_apply_mask out_im", out_im.len(), expected)?;
        if n_frames == 0 || n_bins == 0 {
            // Nothing to write — mirrors the CPU no-op path; every buffer
            // is already empty per the length checks above.
            return Ok(());
        }
        // SAFETY: token consumed by the matching pop below.
        let pool = unsafe { sys::objc_autoreleasePoolPush() };
        let r =
            self.run_denoise_apply_mask(spec_re, spec_im, gain, n_frames, n_bins, out_re, out_im);
        // SAFETY: `pool` is the token from the push above.
        unsafe { sys::objc_autoreleasePoolPop(pool) };
        r
    }

    #[allow(clippy::too_many_arguments)] // intrinsic to the two-output complex-multiply shape
    fn run_denoise_apply_mask(
        &self,
        spec_re: &[f32],
        spec_im: &[f32],
        gain: &[f32],
        n_frames: usize,
        n_bins: usize,
        out_re: &mut [f32],
        out_im: &mut [f32],
    ) -> Result<()> {
        let spec_re_buf = self.new_buffer_from_slice(spec_re)?;
        let spec_im_buf = self.new_buffer_from_slice(spec_im)?;
        let gain_buf = self.new_buffer_from_slice(gain)?;
        let out_re_buf = self.new_buffer_output(out_re.len())?;
        let out_im_buf = self.new_buffer_output(out_im.len())?;
        let dims = DenoiseApplyMaskDims {
            n_bins: n_bins as u32,
            n_frames: n_frames as u32,
        };
        // One thread per (bin, frame). `n_bins` on the fast axis (grid.x)
        // matches the row-major stride and keeps adjacent threads reading
        // adjacent floats. The kernel guards the ragged tail against both
        // bounds so `grid_2d`'s round-up-to-16 is safe.
        let (grid, tg) = grid_2d(n_bins, n_frames);
        self.dispatch_compute(
            self.denoise_apply_mask_pipeline,
            &[
                &spec_re_buf,
                &spec_im_buf,
                &gain_buf,
                &out_re_buf,
                &out_im_buf,
            ],
            (&dims as *const DenoiseApplyMaskDims).cast::<c_void>(),
            size_of::<DenoiseApplyMaskDims>(),
            grid,
            tg,
            "denoise_apply_mask",
        )?;
        read_back(&out_re_buf, out_re)?;
        read_back(&out_im_buf, out_im)
    }

    /// Vocoder Metal wave WF5: Qwen3-TTS-Codec RVQ decode — gather (semantic +
    /// acoustic) + FP32 fold on the GPU.
    ///
    /// Returns a fresh `[time × codebook_dim]` row-major `Vec<f32>` where
    /// `out[t, d] = Σ_q tables[q].row(codes[t, q])[d]` — the exact contract of
    /// `vokra_ops::qwen3_tts_codec::qwen3_tts_codec_decode`. Heap-returning
    /// for the same chunk-granularity reason as `mimi_rvq_gather_fold_f32` /
    /// `dac_rvq_gather_project_fold_f32` (this is a codec-side chunk op, not
    /// a per-token hot path).
    ///
    /// # Semantic vs acoustic vocab split
    ///
    /// Qwen3-TTS-Codec is a hybrid semantic + acoustic RVQ: the first
    /// `num_semantic_quantizers` quantizers read from `semantic_tables_flat`
    /// with vocab `semantic_codebook_size` (canonical 4096); the remaining
    /// acoustic quantizers read from `acoustic_tables_flat` with vocab
    /// `codebook_size` (canonical 2048). Every codebook still emits the same
    /// `codebook_dim`-wide row (canonical 512), so the FP32 fold is
    /// well-defined. Rather than fake a shared vocab (which would either
    /// waste memory or silently clamp the semantic index; both violate
    /// FR-EX-08 / the CPU op's "no silent clamp" rule) the kernel takes TWO
    /// flat table buffers.
    ///
    /// # Parameters
    ///
    /// - `codes` — `[time × num_quantizers]` row-major `u32` codebook indices.
    ///   Every entry must satisfy `idx < per_quantizer_vocab(q)` — the caller
    ///   validates on the host before dispatch (FR-EX-08 — the MSL kernel has
    ///   no per-element bound check, so silent OOB reads are the failure mode
    ///   we prevent by delegating to explicit host-side validation, mirror of
    ///   the mimi_rvq contract).
    /// - `semantic_tables_flat` — `[num_semantic_quantizers × semantic_codebook_size × codebook_dim]`
    ///   row-major FP32; quantizer `q < num_semantic_quantizers` starts at
    ///   `q * semantic_codebook_size * codebook_dim` (the caller concatenates
    ///   the per-codebook `CodebookTable::data` slices verbatim, matching the
    ///   MSL kernel's stride math). If `num_semantic_quantizers == 0` an
    ///   empty slice is legal — the dispatch allocates a zeroed 4-byte
    ///   placeholder via `newBufferWithLength:` (never a
    ///   `newBufferWithBytes:` on a dangling empty-slice pointer, which would
    ///   SIGSEGV) and the kernel never reads it because the loop skips the
    ///   semantic branch entirely.
    /// - `acoustic_tables_flat` — `[(num_quantizers - num_semantic_quantizers) × codebook_size × codebook_dim]`
    ///   row-major FP32; the acoustic quantizer `q_ac = q - num_semantic_quantizers`
    ///   starts at `q_ac * codebook_size * codebook_dim`. Symmetric empty-side
    ///   treatment when `num_semantic_quantizers == num_quantizers`.
    /// - `num_quantizers` / `num_semantic_quantizers` / `semantic_codebook_size`
    ///   / `codebook_size` / `codebook_dim` / `time` — decode shape. All must
    ///   be non-zero except `num_semantic_quantizers` (which may be 0 or up
    ///   to `num_quantizers`) and `time` (which short-circuits at the caller
    ///   with an empty `Vec<f32>`).
    ///
    /// # Numerics
    ///
    /// FP32 accumulator on both sides; MSL is compiled with fast-math
    /// defaults which permit re-association, so the CPU vs GPU bound is
    /// `atol ≤ 5e-4` (the same FP32 GEMV-scale bound used by
    /// [`Self::mimi_rvq_gather_fold_f32`] — see
    /// `tests/qwen3_tts_codec_metal_bit_identical.rs`), not bit-for-bit
    /// equality.
    ///
    /// # Errors
    ///
    /// - [`VokraError::InvalidArgument`] on a shape mismatch, a zero axis in
    ///   `num_quantizers` / `semantic_codebook_size` / `codebook_size` /
    ///   `codebook_dim`, `num_semantic_quantizers > num_quantizers`, or any of
    ///   the `num_quantizers * ...` / `time * num_quantizers` overflows.
    /// - [`VokraError::BackendUnavailable`] on a Metal buffer / command-
    ///   buffer / pipeline dispatch failure.
    #[allow(clippy::too_many_arguments)] // intrinsic to Qwen3-TTS-Codec's hybrid-vocab shape
    pub fn qwen3_tts_codec_decode_f32(
        &self,
        codes: &[u32],
        semantic_tables_flat: &[f32],
        acoustic_tables_flat: &[f32],
        num_quantizers: usize,
        num_semantic_quantizers: usize,
        semantic_codebook_size: usize,
        codebook_size: usize,
        codebook_dim: usize,
        time: usize,
    ) -> Result<Vec<f32>> {
        // Explicit shape validation. The MSL kernel guards `t >= time` and
        // `delem >= codebook_dim` but assumes every buffer has the expected
        // element count, so a wrong-shape upload would be a silent OOB read
        // (FR-EX-08 forbids). Mirror the vokra_ops::qwen3_tts_codec shape
        // checks.
        if num_quantizers == 0
            || semantic_codebook_size == 0
            || codebook_size == 0
            || codebook_dim == 0
        {
            return Err(VokraError::InvalidArgument(format!(
                "qwen3_tts_codec_decode_f32: num_quantizers / semantic_codebook_size / \
                 codebook_size / codebook_dim must all be > 0, got num_quantizers={num_quantizers} \
                 semantic_codebook_size={semantic_codebook_size} codebook_size={codebook_size} \
                 codebook_dim={codebook_dim}"
            )));
        }
        if num_semantic_quantizers > num_quantizers {
            return Err(VokraError::InvalidArgument(format!(
                "qwen3_tts_codec_decode_f32: num_semantic_quantizers {num_semantic_quantizers} > \
                 num_quantizers {num_quantizers}"
            )));
        }
        let num_acoustic_quantizers = num_quantizers - num_semantic_quantizers;
        let expected_semantic = num_semantic_quantizers
            .checked_mul(semantic_codebook_size)
            .and_then(|v| v.checked_mul(codebook_dim))
            .ok_or_else(|| {
                VokraError::InvalidArgument(format!(
                    "qwen3_tts_codec_decode_f32: num_semantic_quantizers * \
                     semantic_codebook_size * codebook_dim overflows usize \
                     (num_semantic_quantizers={num_semantic_quantizers} \
                     semantic_codebook_size={semantic_codebook_size} \
                     codebook_dim={codebook_dim})"
                ))
            })?;
        if semantic_tables_flat.len() != expected_semantic {
            return Err(VokraError::InvalidArgument(format!(
                "qwen3_tts_codec_decode_f32: semantic_tables_flat.len() {} != \
                 num_semantic_quantizers * semantic_codebook_size * codebook_dim \
                 {expected_semantic}",
                semantic_tables_flat.len()
            )));
        }
        let expected_acoustic = num_acoustic_quantizers
            .checked_mul(codebook_size)
            .and_then(|v| v.checked_mul(codebook_dim))
            .ok_or_else(|| {
                VokraError::InvalidArgument(format!(
                    "qwen3_tts_codec_decode_f32: num_acoustic_quantizers * codebook_size * \
                     codebook_dim overflows usize \
                     (num_acoustic_quantizers={num_acoustic_quantizers} \
                     codebook_size={codebook_size} codebook_dim={codebook_dim})"
                ))
            })?;
        if acoustic_tables_flat.len() != expected_acoustic {
            return Err(VokraError::InvalidArgument(format!(
                "qwen3_tts_codec_decode_f32: acoustic_tables_flat.len() {} != \
                 num_acoustic_quantizers * codebook_size * codebook_dim {expected_acoustic}",
                acoustic_tables_flat.len()
            )));
        }
        let expected_codes = time.checked_mul(num_quantizers).ok_or_else(|| {
            VokraError::InvalidArgument(format!(
                "qwen3_tts_codec_decode_f32: time ({time}) * num_quantizers ({num_quantizers}) \
                 overflows usize"
            ))
        })?;
        if codes.len() != expected_codes {
            return Err(VokraError::InvalidArgument(format!(
                "qwen3_tts_codec_decode_f32: codes.len() {} != time * num_quantizers \
                 {expected_codes}",
                codes.len()
            )));
        }
        // Empty output → return an empty Vec, mirroring `qwen3_tts_codec_decode`
        // (a zero-`time` decode is well-defined; every other empty axis is a
        // shape error above).
        if time == 0 {
            return Ok(Vec::new());
        }
        // SAFETY: token consumed by the matching pop below.
        let pool = unsafe { sys::objc_autoreleasePoolPush() };
        let r = self.run_qwen3_tts_codec_decode(
            codes,
            semantic_tables_flat,
            acoustic_tables_flat,
            num_quantizers,
            num_semantic_quantizers,
            semantic_codebook_size,
            codebook_size,
            codebook_dim,
            time,
        );
        // SAFETY: `pool` is the token from the push above.
        unsafe { sys::objc_autoreleasePoolPop(pool) };
        r
    }

    #[allow(clippy::too_many_arguments)] // intrinsic to Qwen3-TTS-Codec's hybrid-vocab shape
    fn run_qwen3_tts_codec_decode(
        &self,
        codes: &[u32],
        semantic_tables_flat: &[f32],
        acoustic_tables_flat: &[f32],
        num_quantizers: usize,
        num_semantic_quantizers: usize,
        semantic_codebook_size: usize,
        codebook_size: usize,
        codebook_dim: usize,
        time: usize,
    ) -> Result<Vec<f32>> {
        // SAFETY: `codes` is a valid read-only slice of `u32`; reinterpret its
        // backing storage as a `u8` slice of the same byte length for the
        // byte-oriented shared MTLBuffer upload. `u32` alignment is stricter
        // than `u8`, so the pointer cast is well-defined, and the borrow
        // scope of `codes_bytes` is limited to this function.
        let codes_bytes: &[u8] = unsafe {
            core::slice::from_raw_parts(codes.as_ptr().cast::<u8>(), core::mem::size_of_val(codes))
        };
        let codes_buf = self.new_buffer_from_bytes(codes_bytes)?;
        // Empty semantic / acoustic side: `new_buffer_from_slice` would ask
        // Metal to copy 4 bytes from an empty slice's dangling pointer
        // (SIGSEGV on macOS), so we allocate a zeroed 4-byte placeholder via
        // `new_buffer_output(1)` instead. The kernel never reads the
        // corresponding buffer because the loop's `q <
        // num_semantic_quantizers` branch is dead when
        // `num_semantic_quantizers == 0` (and symmetric for the acoustic
        // side when `num_acoustic_quantizers == 0`) — the placeholder is
        // there only to satisfy Metal's non-null buffer binding requirement.
        let semantic_buf = if semantic_tables_flat.is_empty() {
            self.new_buffer_output(1)?
        } else {
            self.new_buffer_from_slice(semantic_tables_flat)?
        };
        let acoustic_buf = if acoustic_tables_flat.is_empty() {
            self.new_buffer_output(1)?
        } else {
            self.new_buffer_from_slice(acoustic_tables_flat)?
        };
        let out_len = time * codebook_dim;
        let out_buf = self.new_buffer_output(out_len)?;
        let dims = Qwen3TtsCodecDims {
            num_quantizers: num_quantizers as u32,
            num_semantic_quantizers: num_semantic_quantizers as u32,
            semantic_codebook_size: semantic_codebook_size as u32,
            codebook_size: codebook_size as u32,
            codebook_dim: codebook_dim as u32,
            time: time as u32,
        };
        // One thread per (codebook_dim column, timestep row) — same
        // `(codebook_dim, time)` 2D dispatch as mimi_rvq / dac_rvq. The
        // kernel guards the ragged tail against both bounds so `grid_2d`'s
        // round-up-to-16 is safe.
        let (grid, tg) = grid_2d(codebook_dim, time);
        self.dispatch_compute(
            self.qwen3_tts_codec_decode_pipeline,
            &[&codes_buf, &semantic_buf, &acoustic_buf, &out_buf],
            (&dims as *const Qwen3TtsCodecDims).cast::<c_void>(),
            size_of::<Qwen3TtsCodecDims>(),
            grid,
            tg,
            "qwen3_tts_codec_decode",
        )?;
        let mut out = vec![0.0_f32; out_len];
        read_back(&out_buf, &mut out)?;
        Ok(out)
    }

    /// M3-06 T14: Mimi RVQ codec decode — gather + FP32 fold on the GPU.
    ///
    /// Returns a fresh `[time × d_model]` row-major `Vec<f32>` where
    /// `out[t, d] = Σ_cb tables[cb].row(codes[t, cb])[d]` — the exact
    /// contract of `vokra_ops::mimi_rvq::rvq_fold_core` (the shape-generic
    /// core behind `mimi_rvq_decode`). Heap-returning (not `out: &mut [f32]`)
    /// for the same reason `Compute::mimi_rvq_f32` is heap-returning — this
    /// is a chunk-granularity op, not a per-token hot path.
    ///
    /// # Parameters
    ///
    /// - `codes` — `[time × n_codebooks]` row-major `u32` codebook indices.
    ///   Every entry must satisfy `idx < codebook_size` — the caller
    ///   validates on the host before dispatch (FR-EX-08 — the MSL kernel has
    ///   no per-element bound check, so silent OOB reads are the failure
    ///   mode we prevent by delegating to explicit host-side validation).
    /// - `tables_flat` — `[n_codebooks × codebook_size × d_model]` row-major
    ///   FP32; codebook `cb` starts at `cb * codebook_size * d_model` (the
    ///   caller concatenates the per-codebook `CodebookTable::data` slices
    ///   verbatim, no re-layout — matches the MSL kernel's stride math).
    /// - `n_codebooks` / `codebook_size` / `d_model` / `time` — decode
    ///   shape. All must be non-zero (an empty `time` short-circuits at the
    ///   caller with an empty `Vec<f32>`; we still refuse
    ///   `n_codebooks = 0` / `codebook_size = 0` / `d_model = 0` here as an
    ///   explicit `InvalidArgument`).
    ///
    /// # Numerics
    ///
    /// FP32 accumulator on both sides; MSL is compiled with fast-math
    /// defaults which permit re-association, so the CPU vs GPU bound is
    /// `atol ≤ 5e-4` (the M4-05 CSM / Moshi Metal parity band, mirrored
    /// here — see `tests/mimi_rvq_metal_bit_identical.rs`), not a
    /// bit-for-bit equality. The gather-only ordering with unit weights keeps
    /// max |Δ| well inside that bound on the canonical Mimi shape
    /// (n_codebooks=8, codebook_size=2048, d_model=512).
    ///
    /// # Errors
    ///
    /// - [`VokraError::InvalidArgument`] on a shape mismatch, a zero axis in
    ///   `n_codebooks` / `codebook_size` / `d_model`, or a
    ///   `time * n_codebooks` / `n_codebooks * codebook_size * d_model`
    ///   overflow.
    /// - [`VokraError::BackendUnavailable`] on a Metal buffer / command-
    ///   buffer / pipeline dispatch failure.
    pub fn mimi_rvq_gather_fold_f32(
        &self,
        codes: &[u32],
        tables_flat: &[f32],
        n_codebooks: usize,
        codebook_size: usize,
        d_model: usize,
        time: usize,
    ) -> Result<Vec<f32>> {
        // Explicit shape validation. The MSL kernel guards `t >= time` and
        // `d >= d_model` but assumes the buffers have the expected element
        // counts, so a wrong-shape upload would be an OOB read (silent —
        // FR-EX-08 forbids). Mirror the vokra_ops::mimi_rvq shape checks.
        if n_codebooks == 0 || codebook_size == 0 || d_model == 0 {
            return Err(VokraError::InvalidArgument(format!(
                "mimi_rvq_gather_fold_f32: n_codebooks / codebook_size / d_model \
                 must all be > 0, got n_codebooks={n_codebooks} \
                 codebook_size={codebook_size} d_model={d_model}"
            )));
        }
        let expected_tables = n_codebooks
            .checked_mul(codebook_size)
            .and_then(|v| v.checked_mul(d_model))
            .ok_or_else(|| {
                VokraError::InvalidArgument(format!(
                    "mimi_rvq_gather_fold_f32: n_codebooks * codebook_size * d_model overflows \
                     usize (n_codebooks={n_codebooks} codebook_size={codebook_size} d_model={d_model})"
                ))
            })?;
        if tables_flat.len() != expected_tables {
            return Err(VokraError::InvalidArgument(format!(
                "mimi_rvq_gather_fold_f32: tables_flat.len() {} != n_codebooks * codebook_size * \
                 d_model {expected_tables}",
                tables_flat.len()
            )));
        }
        let expected_codes = time.checked_mul(n_codebooks).ok_or_else(|| {
            VokraError::InvalidArgument(format!(
                "mimi_rvq_gather_fold_f32: time ({time}) * n_codebooks ({n_codebooks}) \
                 overflows usize"
            ))
        })?;
        if codes.len() != expected_codes {
            return Err(VokraError::InvalidArgument(format!(
                "mimi_rvq_gather_fold_f32: codes.len() {} != time * n_codebooks {expected_codes}",
                codes.len()
            )));
        }
        // Empty output → return an empty Vec, mirroring `mimi_rvq_decode`
        // (a zero-`time` decode is well-defined; every other empty axis is a
        // shape error above).
        if time == 0 {
            return Ok(Vec::new());
        }
        // SAFETY: token consumed by the matching pop below.
        let pool = unsafe { sys::objc_autoreleasePoolPush() };
        let r = self.run_mimi_rvq_gather_fold(
            codes,
            tables_flat,
            n_codebooks,
            codebook_size,
            d_model,
            time,
        );
        // SAFETY: `pool` is the token from the push above.
        unsafe { sys::objc_autoreleasePoolPop(pool) };
        r
    }

    fn run_mimi_rvq_gather_fold(
        &self,
        codes: &[u32],
        tables_flat: &[f32],
        n_codebooks: usize,
        codebook_size: usize,
        d_model: usize,
        time: usize,
    ) -> Result<Vec<f32>> {
        // SAFETY: `codes` is a valid read-only slice of `u32`; reinterpret its
        // backing storage as a `u8` slice of the same byte length for the
        // byte-oriented shared MTLBuffer upload. `u32` alignment is stricter
        // than `u8`, so the pointer cast is well-defined, and the borrow
        // scope of `codes_bytes` is limited to this function.
        let codes_bytes: &[u8] = unsafe {
            core::slice::from_raw_parts(codes.as_ptr().cast::<u8>(), core::mem::size_of_val(codes))
        };
        let codes_buf = self.new_buffer_from_bytes(codes_bytes)?;
        let tables_buf = self.new_buffer_from_slice(tables_flat)?;
        let out_len = time * d_model;
        let out_buf = self.new_buffer_output(out_len)?;
        let dims = MimiRvqDims {
            n_codebooks: n_codebooks as u32,
            codebook_size: codebook_size as u32,
            d_model: d_model as u32,
            time: time as u32,
        };
        // One thread per (d_model column, timestep row). The kernel guards the
        // ragged tail against both bounds so `grid_2d`'s round-up-to-16 is safe.
        let (grid, tg) = grid_2d(d_model, time);
        self.dispatch_compute(
            self.mimi_rvq_gather_fold_pipeline,
            &[&codes_buf, &tables_buf, &out_buf],
            (&dims as *const MimiRvqDims).cast::<c_void>(),
            size_of::<MimiRvqDims>(),
            grid,
            tg,
            "mimi_rvq_gather_fold",
        )?;
        let mut out = vec![0.0_f32; out_len];
        read_back(&out_buf, &mut out)?;
        Ok(out)
    }

    /// M4-04: DAC (Descript) factorized RVQ decode — gather + per-quantizer
    /// projection + FP32 fold on the GPU.
    ///
    /// Returns a fresh `[time × d_model]` row-major `Vec<f32>` where
    /// `out[t, :] = Σ_cb (W_cb @ codebook_cb[codes[t, cb]] + b_cb)` — the exact
    /// contract of `vokra_ops::dac_rvq::dac_rvq_decode`. Heap-returning for
    /// the same chunk-granularity reason as
    /// [`Self::mimi_rvq_gather_fold_f32`] (this is a codec-side chunk op, not
    /// a per-token hot path).
    ///
    /// # Parameters
    ///
    /// - `codes` — `[time × n_codebooks]` row-major `u32` codebook indices.
    ///   Every entry must satisfy `idx < codebook_size` — the caller validates
    ///   on the host before dispatch (FR-EX-08 — the MSL kernel has no per-
    ///   element bound check, so silent OOB reads are the failure mode we
    ///   prevent by delegating to explicit host-side validation, mirror of
    ///   the mimi_rvq contract).
    /// - `low_tables_flat` — `[n_codebooks × codebook_size × codebook_dim]`
    ///   row-major FP32; codebook `cb` starts at
    ///   `cb * codebook_size * codebook_dim` (the caller concatenates the
    ///   per-codebook `CodebookTable::data` slices verbatim, matching the MSL
    ///   kernel's stride math).
    /// - `proj_weights_flat` — `[n_codebooks × d_model × codebook_dim]`
    ///   row-major FP32; quantizer `cb`'s weight `W_cb[d, :]` at
    ///   `cb * d_model * codebook_dim + d * codebook_dim` (concat of the
    ///   per-quantizer `DacOutProj::weight` slices verbatim).
    /// - `proj_biases_flat` — `[n_codebooks × d_model]` row-major FP32;
    ///   quantizer `cb`'s bias `b_cb[d]` at `cb * d_model + d` (concat of the
    ///   per-quantizer `DacOutProj::bias` slices verbatim).
    /// - `n_codebooks` / `codebook_size` / `codebook_dim` / `d_model` /
    ///   `time` — decode shape. All must be non-zero (an empty `time` short-
    ///   circuits at the caller with an empty `Vec<f32>`; we still refuse
    ///   any other zero axis here as an explicit `InvalidArgument`).
    ///
    /// # Numerics
    ///
    /// FP32 accumulator on both sides; MSL is compiled with fast-math defaults
    /// which permit re-association, so the CPU vs GPU bound is `atol ≤ 5e-4`
    /// (the same FP32 GEMV-scale bound used by
    /// [`Self::mimi_rvq_gather_fold_f32`] — see
    /// `tests/dac_rvq_metal_bit_identical.rs`), not bit-for-bit equality.
    ///
    /// # Errors
    ///
    /// - [`VokraError::InvalidArgument`] on a shape mismatch, a zero axis in
    ///   `n_codebooks` / `codebook_size` / `codebook_dim` / `d_model`, or a
    ///   `time * n_codebooks` /
    ///   `n_codebooks * codebook_size * codebook_dim` /
    ///   `n_codebooks * d_model * codebook_dim` /
    ///   `n_codebooks * d_model` overflow.
    /// - [`VokraError::BackendUnavailable`] on a Metal buffer / command-
    ///   buffer / pipeline dispatch failure.
    #[allow(clippy::too_many_arguments)] // intrinsic to DAC's factorized-RVQ shape
    pub fn dac_rvq_gather_project_fold_f32(
        &self,
        codes: &[u32],
        low_tables_flat: &[f32],
        proj_weights_flat: &[f32],
        proj_biases_flat: &[f32],
        n_codebooks: usize,
        codebook_size: usize,
        codebook_dim: usize,
        d_model: usize,
        time: usize,
    ) -> Result<Vec<f32>> {
        // Explicit shape validation. The MSL kernel guards `t >= time` and
        // `d >= d_model` but assumes every buffer has the expected element
        // count, so a wrong-shape upload would be a silent OOB read
        // (FR-EX-08 forbids). Mirror the vokra_ops::dac_rvq shape checks.
        if n_codebooks == 0 || codebook_size == 0 || codebook_dim == 0 || d_model == 0 {
            return Err(VokraError::InvalidArgument(format!(
                "dac_rvq_gather_project_fold_f32: n_codebooks / codebook_size / codebook_dim / \
                 d_model must all be > 0, got n_codebooks={n_codebooks} \
                 codebook_size={codebook_size} codebook_dim={codebook_dim} d_model={d_model}"
            )));
        }
        let expected_low = n_codebooks
            .checked_mul(codebook_size)
            .and_then(|v| v.checked_mul(codebook_dim))
            .ok_or_else(|| {
                VokraError::InvalidArgument(format!(
                    "dac_rvq_gather_project_fold_f32: n_codebooks * codebook_size * codebook_dim \
                     overflows usize (n_codebooks={n_codebooks} codebook_size={codebook_size} \
                     codebook_dim={codebook_dim})"
                ))
            })?;
        if low_tables_flat.len() != expected_low {
            return Err(VokraError::InvalidArgument(format!(
                "dac_rvq_gather_project_fold_f32: low_tables_flat.len() {} != n_codebooks * \
                 codebook_size * codebook_dim {expected_low}",
                low_tables_flat.len()
            )));
        }
        let expected_w = n_codebooks
            .checked_mul(d_model)
            .and_then(|v| v.checked_mul(codebook_dim))
            .ok_or_else(|| {
                VokraError::InvalidArgument(format!(
                    "dac_rvq_gather_project_fold_f32: n_codebooks * d_model * codebook_dim \
                     overflows usize (n_codebooks={n_codebooks} d_model={d_model} \
                     codebook_dim={codebook_dim})"
                ))
            })?;
        if proj_weights_flat.len() != expected_w {
            return Err(VokraError::InvalidArgument(format!(
                "dac_rvq_gather_project_fold_f32: proj_weights_flat.len() {} != n_codebooks * \
                 d_model * codebook_dim {expected_w}",
                proj_weights_flat.len()
            )));
        }
        let expected_b = n_codebooks.checked_mul(d_model).ok_or_else(|| {
            VokraError::InvalidArgument(format!(
                "dac_rvq_gather_project_fold_f32: n_codebooks * d_model overflows usize \
                 (n_codebooks={n_codebooks} d_model={d_model})"
            ))
        })?;
        if proj_biases_flat.len() != expected_b {
            return Err(VokraError::InvalidArgument(format!(
                "dac_rvq_gather_project_fold_f32: proj_biases_flat.len() {} != n_codebooks * \
                 d_model {expected_b}",
                proj_biases_flat.len()
            )));
        }
        let expected_codes = time.checked_mul(n_codebooks).ok_or_else(|| {
            VokraError::InvalidArgument(format!(
                "dac_rvq_gather_project_fold_f32: time ({time}) * n_codebooks ({n_codebooks}) \
                 overflows usize"
            ))
        })?;
        if codes.len() != expected_codes {
            return Err(VokraError::InvalidArgument(format!(
                "dac_rvq_gather_project_fold_f32: codes.len() {} != time * n_codebooks \
                 {expected_codes}",
                codes.len()
            )));
        }
        // Empty output → return an empty Vec, mirroring `dac_rvq_decode`
        // (a zero-`time` decode is well-defined; every other empty axis is a
        // shape error above).
        if time == 0 {
            return Ok(Vec::new());
        }
        // SAFETY: token consumed by the matching pop below.
        let pool = unsafe { sys::objc_autoreleasePoolPush() };
        let r = self.run_dac_rvq_gather_project_fold(
            codes,
            low_tables_flat,
            proj_weights_flat,
            proj_biases_flat,
            n_codebooks,
            codebook_size,
            codebook_dim,
            d_model,
            time,
        );
        // SAFETY: `pool` is the token from the push above.
        unsafe { sys::objc_autoreleasePoolPop(pool) };
        r
    }

    #[allow(clippy::too_many_arguments)] // intrinsic to DAC's factorized-RVQ shape
    fn run_dac_rvq_gather_project_fold(
        &self,
        codes: &[u32],
        low_tables_flat: &[f32],
        proj_weights_flat: &[f32],
        proj_biases_flat: &[f32],
        n_codebooks: usize,
        codebook_size: usize,
        codebook_dim: usize,
        d_model: usize,
        time: usize,
    ) -> Result<Vec<f32>> {
        // SAFETY: `codes` is a valid read-only slice of `u32`; reinterpret its
        // backing storage as a `u8` slice of the same byte length for the
        // byte-oriented shared MTLBuffer upload. `u32` alignment is stricter
        // than `u8`, so the pointer cast is well-defined, and the borrow
        // scope of `codes_bytes` is limited to this function.
        let codes_bytes: &[u8] = unsafe {
            core::slice::from_raw_parts(codes.as_ptr().cast::<u8>(), core::mem::size_of_val(codes))
        };
        let codes_buf = self.new_buffer_from_bytes(codes_bytes)?;
        let low_buf = self.new_buffer_from_slice(low_tables_flat)?;
        let w_buf = self.new_buffer_from_slice(proj_weights_flat)?;
        let b_buf = self.new_buffer_from_slice(proj_biases_flat)?;
        let out_len = time * d_model;
        let out_buf = self.new_buffer_output(out_len)?;
        let dims = DacRvqDims {
            n_codebooks: n_codebooks as u32,
            codebook_size: codebook_size as u32,
            codebook_dim: codebook_dim as u32,
            d_model: d_model as u32,
            time: time as u32,
        };
        // One thread per (d_model column, timestep row). The kernel guards the
        // ragged tail against both bounds so `grid_2d`'s round-up-to-16 is safe.
        let (grid, tg) = grid_2d(d_model, time);
        self.dispatch_compute(
            self.dac_rvq_gather_project_fold_pipeline,
            &[&codes_buf, &low_buf, &w_buf, &b_buf, &out_buf],
            (&dims as *const DacRvqDims).cast::<c_void>(),
            size_of::<DacRvqDims>(),
            grid,
            tg,
            "dac_rvq_gather_project_fold",
        )?;
        let mut out = vec![0.0_f32; out_len];
        read_back(&out_buf, &mut out)?;
        Ok(out)
    }

    /// M4-16: WavTokenizer single-codebook VQ decode — pure gather on the GPU.
    ///
    /// Returns a fresh `[time × d_model]` row-major `Vec<f32>` where
    /// `out[t, :] = table.row(codes[t])` — the exact contract of
    /// `vokra_ops::wavtokenizer_vq_decode`. Heap-returning (not
    /// `out: &mut [f32]`) for the same chunk-granularity reason as
    /// [`Self::mimi_rvq_gather_fold_f32`] — this is a codec-side chunk op, not
    /// a per-token hot path.
    ///
    /// # Parameters
    ///
    /// - `codes` — `[time]` row-major `u32` codebook indices (single-stage —
    ///   one code per timestep; the RVQ family's `[time, n_codebooks]`
    ///   layout does NOT apply here). Every entry must satisfy
    ///   `idx < vocab_size` — the caller validates on the host before
    ///   dispatch (FR-EX-08 — the MSL kernel has no per-element bound check,
    ///   so silent OOB reads are the failure mode we prevent by delegating to
    ///   explicit host-side validation).
    /// - `table_flat` — `[vocab_size × d_model]` row-major FP32 codebook (the
    ///   caller passes `CodebookTable::data` verbatim — the flat layout
    ///   matches the MSL kernel's `idx * d_model + delem` stride math).
    /// - `vocab_size` — number of codebook entries (released WavTokenizer =
    ///   4096; the op is shape-generic and handles the FR-OP-31 "65k+ vocab"
    ///   scale pinned by the vokra-ops synthetic test).
    /// - `d_model` — embedding width per entry (released WavTokenizer = 512).
    /// - `time` — number of timesteps in this decode chunk.
    ///
    /// # Numerics
    ///
    /// Pure gather — no arithmetic, no fold. CPU and GPU are bit-identical
    /// (no re-association possible in a raw row copy), so the parity bound is
    /// trivially tight. The parity test asserts the same 5e-4 budget as the
    /// sibling mimi_rvq / dac_rvq kernels for a consistent codec-family
    /// bound.
    ///
    /// # Errors
    ///
    /// - [`VokraError::InvalidArgument`] on a shape mismatch, a zero axis in
    ///   `vocab_size` / `d_model`, or a `vocab_size * d_model` overflow.
    /// - [`VokraError::BackendUnavailable`] on a Metal buffer / command-
    ///   buffer / pipeline dispatch failure.
    pub fn wavtokenizer_vq_gather_f32(
        &self,
        codes: &[u32],
        table_flat: &[f32],
        vocab_size: usize,
        d_model: usize,
        time: usize,
    ) -> Result<Vec<f32>> {
        // Explicit shape validation. The MSL kernel guards `t >= time` and
        // `d >= d_model` but assumes the buffers have the expected element
        // counts, so a wrong-shape upload would be a silent OOB read
        // (FR-EX-08 forbids). Mirror the
        // vokra_ops::fsq_codec::wavtokenizer_vq_decode shape checks
        // (`wavtokenizer_vq` is the op name, not a module path).
        if vocab_size == 0 || d_model == 0 {
            return Err(VokraError::InvalidArgument(format!(
                "wavtokenizer_vq_gather_f32: vocab_size / d_model must both be > 0, got \
                 vocab_size={vocab_size} d_model={d_model}"
            )));
        }
        let expected_table = vocab_size.checked_mul(d_model).ok_or_else(|| {
            VokraError::InvalidArgument(format!(
                "wavtokenizer_vq_gather_f32: vocab_size * d_model overflows usize \
                 (vocab_size={vocab_size} d_model={d_model})"
            ))
        })?;
        if table_flat.len() != expected_table {
            return Err(VokraError::InvalidArgument(format!(
                "wavtokenizer_vq_gather_f32: table_flat.len() {} != vocab_size * d_model \
                 {expected_table}",
                table_flat.len()
            )));
        }
        if codes.len() != time {
            return Err(VokraError::InvalidArgument(format!(
                "wavtokenizer_vq_gather_f32: codes.len() {} != time {time} (single codebook — \
                 one code per timestep; the [time, n_codebooks] layout is the RVQ family's)",
                codes.len()
            )));
        }
        // Empty output → return an empty Vec, mirroring `wavtokenizer_vq_decode`.
        if time == 0 {
            return Ok(Vec::new());
        }
        // SAFETY: token consumed by the matching pop below.
        let pool = unsafe { sys::objc_autoreleasePoolPush() };
        let r = self.run_wavtokenizer_vq_gather(codes, table_flat, vocab_size, d_model, time);
        // SAFETY: `pool` is the token from the push above.
        unsafe { sys::objc_autoreleasePoolPop(pool) };
        r
    }

    fn run_wavtokenizer_vq_gather(
        &self,
        codes: &[u32],
        table_flat: &[f32],
        vocab_size: usize,
        d_model: usize,
        time: usize,
    ) -> Result<Vec<f32>> {
        // SAFETY: `codes` is a valid read-only slice of `u32`; reinterpret its
        // backing storage as a `u8` slice of the same byte length for the
        // byte-oriented shared MTLBuffer upload. `u32` alignment is stricter
        // than `u8`, so the pointer cast is well-defined, and the borrow
        // scope of `codes_bytes` is limited to this function.
        let codes_bytes: &[u8] = unsafe {
            core::slice::from_raw_parts(codes.as_ptr().cast::<u8>(), core::mem::size_of_val(codes))
        };
        let codes_buf = self.new_buffer_from_bytes(codes_bytes)?;
        let table_buf = self.new_buffer_from_slice(table_flat)?;
        let out_len = time * d_model;
        let out_buf = self.new_buffer_output(out_len)?;
        let dims = WavTokenizerVqDims {
            vocab_size: vocab_size as u32,
            d_model: d_model as u32,
            time: time as u32,
        };
        // One thread per (d_model column, timestep row). The kernel guards the
        // ragged tail against both bounds so `grid_2d`'s round-up-to-16 is
        // safe.
        let (grid, tg) = grid_2d(d_model, time);
        self.dispatch_compute(
            self.wavtokenizer_vq_gather_pipeline,
            &[&codes_buf, &table_buf, &out_buf],
            (&dims as *const WavTokenizerVqDims).cast::<c_void>(),
            size_of::<WavTokenizerVqDims>(),
            grid,
            tg,
            "wavtokenizer_vq_gather",
        )?;
        let mut out = vec![0.0_f32; out_len];
        read_back(&out_buf, &mut out)?;
        Ok(out)
    }

    /// M4-16: X-Codec 2 FSQ decode — grid decompose + optional GEMV on the
    /// GPU.
    ///
    /// Returns a fresh `[time × d_model]` row-major `Vec<f32>` where per
    /// timestep the index is decomposed onto the implicit per-dimension grid
    /// and (when `proj_weight.is_some()`) projected by one GEMV `out[t, :] =
    /// W @ grid + b` — the exact contract of `vokra_ops::xcodec2_fsq_decode`.
    /// Heap-returning for the same chunk-granularity reason as
    /// [`Self::mimi_rvq_gather_fold_f32`].
    ///
    /// # Parameters
    ///
    /// - `codes` — `[time]` row-major `u32` FSQ indices. Every entry must
    ///   satisfy `idx < Π levels` — the caller validates on the host before
    ///   dispatch (FR-EX-08 — the MSL kernel has no per-element bound check).
    /// - `levels` — `[n_dims]` `u32` mixed-radix bases (each ≥ 2 — the
    ///   caller validates this so the kernel's `half_width = levels[k] / 2`
    ///   is `≥ 1`, preventing a divide-by-zero in the grid formula).
    /// - `proj_weight` — `Some([d_model × n_dims])` row-major FP32 out-
    ///   projection weight, or `None` for the Identity path (which requires
    ///   `d_model == n_dims`).
    /// - `proj_bias` — `Some([d_model])` FP32 bias (must be `Some` iff
    ///   `proj_weight.is_some()`), or `None` for Identity.
    /// - `d_model` — output width per timestep (canonical released X-Codec 2
    ///   = 2048; must equal `n_dims` when `proj_weight.is_none()`).
    /// - `n_dims` — `len(levels)` (canonical = 8).
    /// - `time` — number of timesteps in this decode chunk.
    ///
    /// # Numerics
    ///
    /// FP32 accumulator throughout (audio-dialect rule — no FP16/BF16 fold);
    /// MSL is compiled with fast-math defaults which permit re-association of
    /// the inner `Σ_k W[o, k] · grid[k]` GEMV, so the CPU vs GPU bound is
    /// `atol ≤ 5e-4` — the same FP32 GEMV-scale bound used by
    /// [`Self::mimi_rvq_gather_fold_f32`] / [`Self::dac_rvq_gather_project_fold_f32`].
    /// The Identity path is bit-identical (pure grid decompose, no fold).
    ///
    /// # Errors
    ///
    /// - [`VokraError::InvalidArgument`] on a shape mismatch, a zero axis in
    ///   `d_model` / `n_dims`, `proj_weight.is_some() != proj_bias.is_some()`,
    ///   the Identity path with `d_model != n_dims`, or a
    ///   `d_model * n_dims` / `time * d_model` overflow.
    /// - [`VokraError::BackendUnavailable`] on a Metal buffer / command-
    ///   buffer / pipeline dispatch failure.
    #[allow(clippy::too_many_arguments)] // intrinsic to FSQ's optional-projection shape
    pub fn xcodec2_fsq_decode_f32(
        &self,
        codes: &[u32],
        levels: &[u32],
        proj_weight: Option<&[f32]>,
        proj_bias: Option<&[f32]>,
        d_model: usize,
        n_dims: usize,
        time: usize,
    ) -> Result<Vec<f32>> {
        // Explicit shape validation. Mirrors the
        // vokra_ops::fsq_codec::xcodec2_fsq_decode shape checks
        // (`xcodec2_fsq` is the op name, not a module path)
        // (FR-EX-08 — no silent OOB or CPU fall back).
        if d_model == 0 || n_dims == 0 {
            return Err(VokraError::InvalidArgument(format!(
                "xcodec2_fsq_decode_f32: d_model / n_dims must both be > 0, got \
                 d_model={d_model} n_dims={n_dims}"
            )));
        }
        if levels.len() != n_dims {
            return Err(VokraError::InvalidArgument(format!(
                "xcodec2_fsq_decode_f32: levels.len() {} != n_dims {n_dims}",
                levels.len()
            )));
        }
        // Match host-side validation to CPU: `proj_weight.is_some() ==
        // proj_bias.is_some()` and shape agrees with attrs.
        match (proj_weight, proj_bias) {
            (Some(w), Some(b)) => {
                let expected_w = d_model.checked_mul(n_dims).ok_or_else(|| {
                    VokraError::InvalidArgument(format!(
                        "xcodec2_fsq_decode_f32: d_model * n_dims overflows usize \
                         (d_model={d_model} n_dims={n_dims})"
                    ))
                })?;
                if w.len() != expected_w {
                    return Err(VokraError::InvalidArgument(format!(
                        "xcodec2_fsq_decode_f32: proj_weight.len() {} != d_model * n_dims \
                         {expected_w}",
                        w.len()
                    )));
                }
                if b.len() != d_model {
                    return Err(VokraError::InvalidArgument(format!(
                        "xcodec2_fsq_decode_f32: proj_bias.len() {} != d_model {d_model}",
                        b.len()
                    )));
                }
            }
            (None, None) => {
                // Identity path: d_model must equal n_dims (upstream
                // `requires_projection = false` invariant).
                if d_model != n_dims {
                    return Err(VokraError::InvalidArgument(format!(
                        "xcodec2_fsq_decode_f32: Identity path (proj_weight = None) requires \
                         d_model == n_dims, got d_model={d_model} n_dims={n_dims}"
                    )));
                }
            }
            _ => {
                return Err(VokraError::InvalidArgument(
                    "xcodec2_fsq_decode_f32: proj_weight and proj_bias must both be Some or \
                     both None (partial projection is not a valid X-Codec 2 shape)"
                        .to_owned(),
                ));
            }
        }
        // Per-level validation: each level ≥ 2 (host contract for the MSL
        // kernel's `half_width = levels[k] / 2` ≥ 1; else divide-by-zero).
        for (k, &level) in levels.iter().enumerate() {
            if level < 2 {
                return Err(VokraError::InvalidArgument(format!(
                    "xcodec2_fsq_decode_f32: levels[{k}] = {level} < 2 (half_width would be 0 \
                     — divide-by-zero in the grid formula; FR-EX-08 catches it upstream)"
                )));
            }
        }
        if codes.len() != time {
            return Err(VokraError::InvalidArgument(format!(
                "xcodec2_fsq_decode_f32: codes.len() {} != time {time} (single-stage — one code \
                 per timestep; the [time, n_codebooks] layout is the RVQ family's)",
                codes.len()
            )));
        }
        // Effective vocab = Π levels — per-code bound check. Overflow of the
        // product is an explicit error (FR-EX-08 — no silent wrap).
        let mut vocab: usize = 1;
        for (k, &level) in levels.iter().enumerate() {
            vocab = vocab.checked_mul(level as usize).ok_or_else(|| {
                VokraError::InvalidArgument(format!(
                    "xcodec2_fsq_decode_f32: Π levels overflows usize at levels[{k}] = {level}"
                ))
            })?;
        }
        for (t, &idx) in codes.iter().enumerate() {
            if (idx as usize) >= vocab {
                return Err(VokraError::InvalidArgument(format!(
                    "xcodec2_fsq_decode_f32: codes[{t}] = {idx} >= Π levels {vocab} (no silent \
                     clamp — FR-EX-08)"
                )));
            }
        }
        if time == 0 {
            return Ok(Vec::new());
        }
        // SAFETY: token consumed by the matching pop below.
        let pool = unsafe { sys::objc_autoreleasePoolPush() };
        let r = self.run_xcodec2_fsq_decode(
            codes,
            levels,
            proj_weight,
            proj_bias,
            d_model,
            n_dims,
            time,
        );
        // SAFETY: `pool` is the token from the push above.
        unsafe { sys::objc_autoreleasePoolPop(pool) };
        r
    }

    #[allow(clippy::too_many_arguments)] // intrinsic to FSQ's optional-projection shape
    fn run_xcodec2_fsq_decode(
        &self,
        codes: &[u32],
        levels: &[u32],
        proj_weight: Option<&[f32]>,
        proj_bias: Option<&[f32]>,
        d_model: usize,
        n_dims: usize,
        time: usize,
    ) -> Result<Vec<f32>> {
        // SAFETY: `codes` is a valid read-only slice of `u32`; reinterpret its
        // backing storage as a `u8` slice for the byte-oriented shared
        // MTLBuffer upload. `u32` alignment is stricter than `u8`, so the
        // pointer cast is well-defined, and the borrow scope of `codes_bytes`
        // is limited to this function.
        let codes_bytes: &[u8] = unsafe {
            core::slice::from_raw_parts(codes.as_ptr().cast::<u8>(), core::mem::size_of_val(codes))
        };
        let codes_buf = self.new_buffer_from_bytes(codes_bytes)?;
        // Same trick for `levels: &[u32]`.
        // SAFETY: same rationale as `codes_bytes` above.
        let levels_bytes: &[u8] = unsafe {
            core::slice::from_raw_parts(
                levels.as_ptr().cast::<u8>(),
                core::mem::size_of_val(levels),
            )
        };
        let levels_buf = self.new_buffer_from_bytes(levels_bytes)?;
        // Bind a `[0.0]` dummy buffer when the projection is absent (same
        // convention as `conv1d_f32`'s optional bias — MSL requires all
        // bindings to be non-nil, so a dummy stand-in is uploaded but never
        // read because the kernel's `has_projection == 0` arm doesn't touch
        // `proj_weight` / `proj_bias`).
        let dummy = [0.0f32];
        let w_buf = self.new_buffer_from_slice(proj_weight.unwrap_or(&dummy))?;
        let b_buf = self.new_buffer_from_slice(proj_bias.unwrap_or(&dummy))?;
        let out_len = time.checked_mul(d_model).ok_or_else(|| {
            VokraError::InvalidArgument(format!(
                "xcodec2_fsq_decode_f32: time ({time}) * d_model ({d_model}) overflows usize"
            ))
        })?;
        let out_buf = self.new_buffer_output(out_len)?;
        let dims = Xcodec2FsqDims {
            d_model: d_model as u32,
            n_dims: n_dims as u32,
            time: time as u32,
            has_projection: u32::from(proj_weight.is_some()),
        };
        // One thread per (d_model column, timestep row) — same launch geometry
        // as the sibling mimi_rvq / dac_rvq kernels for a consistent codec-
        // family shape. The kernel guards the ragged tail against both bounds
        // so `grid_2d`'s round-up-to-16 is safe.
        let (grid, tg) = grid_2d(d_model, time);
        self.dispatch_compute(
            self.xcodec2_fsq_decode_pipeline,
            &[&codes_buf, &levels_buf, &w_buf, &b_buf, &out_buf],
            (&dims as *const Xcodec2FsqDims).cast::<c_void>(),
            size_of::<Xcodec2FsqDims>(),
            grid,
            tg,
            "xcodec2_fsq_decode",
        )?;
        let mut out = vec![0.0_f32; out_len];
        read_back(&out_buf, &mut out)?;
        Ok(out)
    }

    /// Dense/grouped PyTorch-layout Conv2d on channel-major host buffers.
    /// Inputs and weights are uploaded once, one Metal dispatch computes the
    /// caller-sized output, and only that output is read back.
    #[allow(clippy::too_many_arguments)]
    pub fn conv2d_f32(
        &self,
        input: &[f32],
        in_ch: usize,
        in_h: usize,
        in_w: usize,
        weight: &[f32],
        out_ch: usize,
        kernel_h: usize,
        kernel_w: usize,
        bias: Option<&[f32]>,
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
        groups: usize,
        out: &mut [f32],
    ) -> Result<()> {
        let (out_h, out_w) = validate_conv2d(
            input, in_ch, in_h, in_w, weight, out_ch, kernel_h, kernel_w, bias, stride, padding,
            dilation, groups, out,
        )?;
        if out.is_empty() {
            return Ok(());
        }
        // SAFETY: token consumed by the matching pop below.
        let pool = unsafe { sys::objc_autoreleasePoolPush() };
        let result = self.run_conv2d(
            input, in_h, in_w, weight, out_ch, out_h, out_w, kernel_h, kernel_w, bias, stride,
            padding, dilation, groups, in_ch, out,
        );
        // SAFETY: `pool` is the token from the push above.
        unsafe { sys::objc_autoreleasePoolPop(pool) };
        result
    }

    /// Dense/grouped PyTorch-layout ConvTranspose2d on channel-major host
    /// buffers. Weight layout is `[in_ch, out_ch / groups, kernel_h, kernel_w]`.
    /// The host validates the exact output formula before one GPU submission;
    /// there is no CPU fallback or hidden host convolution.
    #[allow(clippy::too_many_arguments)]
    pub fn conv_transpose2d_f32(
        &self,
        input: &[f32],
        in_ch: usize,
        in_h: usize,
        in_w: usize,
        weight: &[f32],
        out_ch: usize,
        kernel_h: usize,
        kernel_w: usize,
        bias: Option<&[f32]>,
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
        output_padding: (usize, usize),
        groups: usize,
        out: &mut [f32],
    ) -> Result<()> {
        let (out_h, out_w) = validate_conv_transpose2d(
            input,
            in_ch,
            in_h,
            in_w,
            weight,
            out_ch,
            kernel_h,
            kernel_w,
            bias,
            stride,
            padding,
            dilation,
            output_padding,
            groups,
            out,
        )?;
        if out.is_empty() {
            return Ok(());
        }
        // SAFETY: token consumed by the matching pop below.
        let pool = unsafe { sys::objc_autoreleasePoolPush() };
        let result = self.run_conv_transpose2d(
            input, in_h, in_w, weight, out_ch, out_h, out_w, kernel_h, kernel_w, bias, stride,
            padding, dilation, groups, in_ch, out,
        );
        // SAFETY: `pool` is the token from the push above.
        unsafe { sys::objc_autoreleasePoolPop(pool) };
        result
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
            input, in_len, weight, out_ch, kernel, bias, stride, padding, out_len, in_ch, out_ch,
            out,
        );
        // SAFETY: `pool` is the token from the push above.
        unsafe { sys::objc_autoreleasePoolPop(pool) };
        r
    }

    /// Host-facing stride/dilation-aware Conv1d.  This is the host wrapper for
    /// [`Self::conv1d_dev`], so the same Metal kernel and validation contract
    /// are used by both the resident and imperative Compute seams.  Inputs,
    /// weights, and the final output cross the host/device boundary once each;
    /// there is no CPU fallback when `dilation > 1`.
    #[allow(clippy::too_many_arguments)]
    pub fn conv1d_f32_dilated(
        &self,
        input: &[f32],
        in_ch: usize,
        in_len: usize,
        weight: &[f32],
        out_ch: usize,
        kernel: usize,
        bias: Option<&[f32]>,
        stride: usize,
        dilation: usize,
        padding: usize,
        out: &mut [f32],
    ) -> Result<()> {
        let out_len = validate_conv1d_dilated(
            input, in_ch, in_len, weight, out_ch, kernel, bias, stride, dilation, padding, out,
        )?;
        if out_len == 0 {
            return Ok(());
        }
        if dilation == 1 {
            return self.conv1d_f32(
                input, in_ch, in_len, weight, out_ch, kernel, bias, stride, padding, out,
            );
        }
        // SAFETY: token consumed by the matching pop below.
        let pool = unsafe { sys::objc_autoreleasePoolPush() };
        let result = (|| {
            let input_dev = self.upload(input)?;
            let weight_dev = self.upload(weight)?;
            let bias_dev = bias.map(|values| self.upload(values)).transpose()?;
            let mut output_dev = self.alloc_dev(out.len())?;
            self.conv1d_dev(
                &mut output_dev,
                &input_dev,
                &weight_dev,
                bias_dev.as_ref(),
                in_ch,
                in_len,
                out_ch,
                kernel,
                stride,
                dilation,
                padding,
            )?;
            read_back(&output_dev.buf, out)
        })();
        // SAFETY: `pool` is the token from the push above.
        unsafe { sys::objc_autoreleasePoolPop(pool) };
        result
    }

    /// Host-facing PyTorch-layout ConvTranspose1d.  The explicit
    /// `output_padding` participates in the output-length formula and is
    /// passed unchanged to the device kernel; it is never inferred or
    /// silently cropped.
    #[allow(clippy::too_many_arguments)]
    pub fn conv_transpose1d_f32(
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
        output_padding: usize,
        out: &mut [f32],
    ) -> Result<()> {
        let out_len = validate_conv_transpose1d(
            input,
            in_ch,
            in_len,
            weight,
            out_ch,
            kernel,
            bias,
            stride,
            padding,
            output_padding,
            out,
        )?;
        if out_len == 0 {
            return Ok(());
        }
        // SAFETY: token consumed by the matching pop below.
        let pool = unsafe { sys::objc_autoreleasePoolPush() };
        let result = (|| {
            let input_dev = self.upload(input)?;
            let weight_dev = self.upload(weight)?;
            let bias_dev = bias.map(|values| self.upload(values)).transpose()?;
            let mut output_dev = self.alloc_dev(out.len())?;
            self.conv_transpose1d_dev(
                &mut output_dev,
                &input_dev,
                &weight_dev,
                bias_dev.as_ref(),
                in_ch,
                in_len,
                out_ch,
                kernel,
                stride,
                padding,
                output_padding,
            )?;
            read_back(&output_dev.buf, out)
        })();
        // SAFETY: `pool` is the token from the push above.
        unsafe { sys::objc_autoreleasePoolPop(pool) };
        result
    }

    /// Grouped 1-D convolution with PyTorch-compatible weight layout
    /// `[out_ch, in_ch / groups, kernel]`.
    ///
    /// `groups == in_ch == out_ch` is the depthwise form used by Vocos'
    /// ConvNeXt blocks. The same Metal pipeline as [`Self::conv1d_f32`] is
    /// used; group-local input indexing is performed by the shader, so no
    /// dense diagonal weight expansion or CPU convolution is hidden here.
    #[allow(clippy::too_many_arguments)]
    pub fn grouped_conv1d_f32(
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
        groups: usize,
        out: &mut [f32],
    ) -> Result<()> {
        let out_len = validate_grouped_conv1d(
            input, in_ch, in_len, weight, out_ch, kernel, bias, stride, padding, groups, out,
        )?;
        if out.is_empty() {
            return Ok(());
        }
        // SAFETY: token consumed by the matching pop below.
        let pool = unsafe { sys::objc_autoreleasePoolPush() };
        let r = self.run_conv1d(
            input,
            in_len,
            weight,
            out_ch,
            kernel,
            bias,
            stride,
            padding,
            out_len,
            in_ch / groups,
            out_ch / groups,
            out,
        );
        // SAFETY: `pool` is the token from the push above.
        unsafe { sys::objc_autoreleasePoolPop(pool) };
        r
    }

    #[allow(clippy::too_many_arguments)]
    fn run_conv2d(
        &self,
        input: &[f32],
        in_h: usize,
        in_w: usize,
        weight: &[f32],
        out_ch: usize,
        out_h: usize,
        out_w: usize,
        kernel_h: usize,
        kernel_w: usize,
        bias: Option<&[f32]>,
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
        groups: usize,
        in_ch: usize,
        out: &mut [f32],
    ) -> Result<()> {
        let in_buf = self.new_buffer_from_slice(input)?;
        let w_buf = self.new_buffer_from_slice(weight)?;
        let dummy = [0.0f32];
        let bias_buf = self.new_buffer_from_slice(bias.unwrap_or(&dummy))?;
        let out_buf = self.new_buffer_output(out.len())?;
        let dims = Conv2dDims {
            in_ch: checked_u32(in_ch, "conv2d in_ch")?,
            in_h: checked_u32(in_h, "conv2d in_h")?,
            in_w: checked_u32(in_w, "conv2d in_w")?,
            out_ch: checked_u32(out_ch, "conv2d out_ch")?,
            out_h: checked_u32(out_h, "conv2d out_h")?,
            out_w: checked_u32(out_w, "conv2d out_w")?,
            kernel_h: checked_u32(kernel_h, "conv2d kernel_h")?,
            kernel_w: checked_u32(kernel_w, "conv2d kernel_w")?,
            stride_h: checked_u32(stride.0, "conv2d stride_h")?,
            stride_w: checked_u32(stride.1, "conv2d stride_w")?,
            padding_h: checked_u32(padding.0, "conv2d padding_h")?,
            padding_w: checked_u32(padding.1, "conv2d padding_w")?,
            dilation_h: checked_u32(dilation.0, "conv2d dilation_h")?,
            dilation_w: checked_u32(dilation.1, "conv2d dilation_w")?,
            in_per_group: checked_u32(in_ch / groups, "conv2d in_per_group")?,
            out_per_group: checked_u32(out_ch / groups, "conv2d out_per_group")?,
            has_bias: u32::from(bias.is_some()),
        };
        let out_rows = checked_mul(out_ch, out_h, "conv2d dispatch rows")?;
        let (grid, tg) = grid_2d(out_w, out_rows);
        self.dispatch_compute(
            self.conv2d_pipeline,
            &[&in_buf, &w_buf, &bias_buf, &out_buf],
            (&dims as *const Conv2dDims).cast::<c_void>(),
            size_of::<Conv2dDims>(),
            grid,
            tg,
            "conv2d",
        )?;
        read_back(&out_buf, out)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_conv_transpose2d(
        &self,
        input: &[f32],
        in_h: usize,
        in_w: usize,
        weight: &[f32],
        out_ch: usize,
        out_h: usize,
        out_w: usize,
        kernel_h: usize,
        kernel_w: usize,
        bias: Option<&[f32]>,
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
        groups: usize,
        in_ch: usize,
        out: &mut [f32],
    ) -> Result<()> {
        let in_buf = self.new_buffer_from_slice(input)?;
        let w_buf = self.new_buffer_from_slice(weight)?;
        let dummy = [0.0f32];
        let bias_buf = self.new_buffer_from_slice(bias.unwrap_or(&dummy))?;
        let out_buf = self.new_buffer_output(out.len())?;
        let dims = ConvTranspose2dDims {
            in_ch: checked_u32(in_ch, "conv_transpose2d in_ch")?,
            in_h: checked_u32(in_h, "conv_transpose2d in_h")?,
            in_w: checked_u32(in_w, "conv_transpose2d in_w")?,
            out_ch: checked_u32(out_ch, "conv_transpose2d out_ch")?,
            out_h: checked_u32(out_h, "conv_transpose2d out_h")?,
            out_w: checked_u32(out_w, "conv_transpose2d out_w")?,
            kernel_h: checked_u32(kernel_h, "conv_transpose2d kernel_h")?,
            kernel_w: checked_u32(kernel_w, "conv_transpose2d kernel_w")?,
            stride_h: checked_u32(stride.0, "conv_transpose2d stride_h")?,
            stride_w: checked_u32(stride.1, "conv_transpose2d stride_w")?,
            padding_h: checked_u32(padding.0, "conv_transpose2d padding_h")?,
            padding_w: checked_u32(padding.1, "conv_transpose2d padding_w")?,
            dilation_h: checked_u32(dilation.0, "conv_transpose2d dilation_h")?,
            dilation_w: checked_u32(dilation.1, "conv_transpose2d dilation_w")?,
            in_per_group: checked_u32(in_ch / groups, "conv_transpose2d in_per_group")?,
            out_per_group: checked_u32(out_ch / groups, "conv_transpose2d out_per_group")?,
            has_bias: u32::from(bias.is_some()),
        };
        let out_rows = checked_mul(out_ch, out_h, "conv_transpose2d dispatch rows")?;
        let (grid, tg) = grid_2d(out_w, out_rows);
        self.dispatch_compute(
            self.conv_transpose2d_pipeline,
            &[&in_buf, &w_buf, &bias_buf, &out_buf],
            (&dims as *const ConvTranspose2dDims).cast::<c_void>(),
            size_of::<ConvTranspose2dDims>(),
            grid,
            tg,
            "conv_transpose2d",
        )?;
        read_back(&out_buf, out)
    }

    #[allow(clippy::too_many_arguments)] // convolution's intrinsic parameter set
    fn run_conv1d(
        &self,
        input: &[f32],
        in_len: usize,
        weight: &[f32],
        out_ch: usize,
        kernel: usize,
        bias: Option<&[f32]>,
        stride: usize,
        padding: usize,
        out_len: usize,
        in_per_group: usize,
        out_per_group: usize,
        out: &mut [f32],
    ) -> Result<()> {
        let in_buf = self.new_buffer_from_slice(input)?;
        let w_buf = self.new_buffer_from_slice(weight)?;
        let dummy = [0.0f32];
        let bias_buf = self.new_buffer_from_slice(bias.unwrap_or(&dummy))?;
        let out_buf = self.new_buffer_output(out.len())?;
        let dims = Conv1dDims {
            in_per_group: in_per_group as u32,
            in_len: in_len as u32,
            out_ch: out_ch as u32,
            kernel_size: kernel as u32,
            out_len: out_len as u32,
            stride: stride as u32,
            padding: padding as u32,
            has_bias: u32::from(bias.is_some()),
            out_per_group: out_per_group as u32,
            dilation: 1,
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
    /// and (via `Self::encode_attn_passes`) by [`MetalDecodeSession`].
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

    /// Number of explicit D2H copies made through [`Self::download`].
    ///
    /// This intentionally excludes the legacy slice APIs, whose contract is
    /// host-in/host-out by design. Resident vocoder tests use this counter to
    /// assert that all intermediate device tensors remain on Metal.
    #[must_use]
    pub fn readback_count(&self) -> u64 {
        self.readbacks.get()
    }

    /// Uploads `data` into a fresh device-resident buffer (H2D once). The returned
    /// `MetalDeviceTensor` borrows the context, so it cannot outlive it.
    ///
    /// # Errors
    ///
    /// [`VokraError::BackendUnavailable`] if the Metal buffer cannot be created.
    pub fn upload(&self, data: &[f32]) -> Result<MetalDeviceTensor<'_>> {
        let buf = self.new_buffer_from_slice(data)?;
        Ok(MetalDeviceTensor {
            buf,
            len: data.len(),
            owner: self as *const MetalContext,
            _ctx: PhantomData,
        })
    }

    /// Uploads raw BF16 numeric bit patterns into a device-resident `ushort`
    /// buffer. Each `u16` is the value obtained after decoding the model's
    /// little-endian two-byte BF16 payload (for example,
    /// `u16::from_le_bytes([lo, hi])`). The bytes are retained as BF16 storage;
    /// this method does not widen the weight matrix to an FP32 tensor.
    ///
    /// The returned handle is consumed by
    /// [`Self::gemm_f32_bf16_bits_dev`]. It borrows this context and therefore
    /// cannot outlive it.
    ///
    /// # Errors
    ///
    /// [`VokraError::BackendUnavailable`] if the Metal buffer cannot be
    /// created; [`VokraError::InvalidArgument`] if the raw buffer byte size
    /// overflows.
    pub fn upload_bf16_bits(&self, data: &[u16]) -> Result<MetalBf16DeviceTensor<'_>> {
        let buf = self.new_buffer_from_u16(data)?;
        Ok(MetalBf16DeviceTensor {
            buf,
            len: data.len(),
            owner: self as *const MetalContext,
            _ctx: PhantomData,
        })
    }

    /// Host-in/host-out mixed GEMM using the raw-BF16 device path.
    ///
    /// This is the bounded wrapper for callers whose surrounding model graph
    /// is host-resident: `a` is FP32 `[m,k]`, `b` is raw little-endian-decoded
    /// BF16 `[k,n]`, and `out` is FP32 `[m,n]`. The complete BF16 matrix is
    /// uploaded as `ushort` storage and widened only inside the Metal shader;
    /// no FP32 weight mirror is created. The wrapper performs one final output
    /// download and never falls back to CPU execution.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] is returned for shape, overflow, or
    /// MSL-dimension conversion errors. [`VokraError::BackendUnavailable`]
    /// covers Metal allocation and command failures.
    pub fn gemm_f32_bf16_bits(
        &self,
        m: usize,
        n: usize,
        k: usize,
        a: &[f32],
        b: &[u16],
        out: &mut [f32],
    ) -> Result<()> {
        validate_mixed_bf16_host(m, n, k, a, b, out)?;
        if m == 0 || n == 0 {
            return Ok(());
        }
        // Bracket the GPU work in an autorelease pool, matching `gemm_f32`.
        // `dispatch_compute` creates autoreleased command objects and callers
        // must drain them even when the operation returns an error.
        // SAFETY: `objc_autoreleasePoolPush` returns a token consumed by the
        // one matching pop below.
        let pool = unsafe { sys::objc_autoreleasePoolPush() };
        let result = self.run_gemm_f32_bf16_bits(m, n, k, a, b, out);
        // SAFETY: `pool` is the token from the push above.
        unsafe { sys::objc_autoreleasePoolPop(pool) };
        result
    }

    /// Mixed-BF16 host GEMM body. Shapes are already validated and the caller
    /// has established the autorelease-pool boundary.
    #[allow(clippy::too_many_arguments)]
    fn run_gemm_f32_bf16_bits(
        &self,
        m: usize,
        n: usize,
        k: usize,
        a: &[f32],
        b: &[u16],
        out: &mut [f32],
    ) -> Result<()> {
        let activation = self.upload(a)?;
        let weight = self.upload_bf16_bits(b)?;
        let mut output = self.alloc_dev(out.len())?;
        self.gemm_f32_bf16_bits_dev(&mut output, &activation, &weight, m, n, k)?;
        // This is the legacy host-in/host-out surface, matching `gemm_f32`:
        // copy the completed shared buffer directly rather than composing the
        // resident `download` API. The resident readback counter therefore
        // remains reserved for explicit device-tensor downloads.
        read_back(&output.buf, out)
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
            owner: self as *const MetalContext,
            _ctx: PhantomData,
        })
    }

    /// Rejects buffers allocated by a different live context. The lifetime
    /// marker prevents use-after-drop, while this identity check prevents
    /// mixing queues/devices from independent Metal contexts.
    fn expect_owner(&self, tensor: &MetalDeviceTensor<'_>, name: &str) -> Result<()> {
        if !std::ptr::eq(tensor.owner, self as *const MetalContext) {
            return Err(VokraError::InvalidArgument(format!(
                "{name} belongs to a different MetalContext"
            )));
        }
        Ok(())
    }

    /// Sibling owner check for the raw-BF16 device handle. Keeping this
    /// separate from [`Self::expect_owner`] prevents accidentally accepting a
    /// typed FP32 tensor where a `ushort` storage buffer is required.
    fn expect_owner_bf16(&self, tensor: &MetalBf16DeviceTensor<'_>, name: &str) -> Result<()> {
        if !std::ptr::eq(tensor.owner, self as *const MetalContext) {
            return Err(VokraError::InvalidArgument(format!(
                "{name} belongs to a different MetalContext"
            )));
        }
        Ok(())
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
        self.expect_owner(t, "download tensor")?;
        expect_len("download out", out.len(), t.len)?;
        let result = read_back(&t.buf, out);
        if result.is_ok() {
            self.readbacks.set(self.readbacks.get() + 1);
        }
        result
    }

    /// Device-resident mixed GEMM with FP32 activations and raw BF16 weights.
    ///
    /// Computes `out[r,c] = Σ_l activation[r,l] * bf16_to_f32(weight[l,c])`
    /// for `activation` shaped `[m,k]`, `weight` shaped `[k,n]`, and `out`
    /// shaped `[m,n]`. Activations and output are device-resident FP32
    /// [`MetalDeviceTensor`]s. `weight` is a distinct raw-BF16
    /// [`MetalBf16DeviceTensor`], so the complete weight matrix is never
    /// expanded to an FP32 host or device tensor. The shader reconstructs each
    /// BF16 value exactly from its sign/exponent/mantissa bits and accumulates
    /// products in FP32. This operation issues one Metal submission and no
    /// host readback; callers explicitly use [`Self::download`] for the final
    /// result.
    ///
    /// Zero-size semantics match the CPU raw-BF16 GEMM contract: if `m == 0`
    /// or `n == 0`, a correctly shaped output is left untouched and no
    /// submission is issued. If `k == 0` with non-empty output, one GPU pass
    /// writes exact zeros. The empty input buffers are bound to Metal-safe
    /// placeholders and are never dereferenced by that pass.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] is returned before dispatch for a
    /// cross-context tensor, shape mismatch, zero-sized output inconsistency,
    /// dimension product overflow, or a dimension that cannot be represented
    /// by the MSL `uint` shape block. [`VokraError::BackendUnavailable`] is
    /// returned for allocation or command-buffer failures. There is no CPU
    /// fallback.
    pub fn gemm_f32_bf16_bits_dev(
        &self,
        out: &mut MetalDeviceTensor<'_>,
        activation: &MetalDeviceTensor<'_>,
        weight: &MetalBf16DeviceTensor<'_>,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        self.expect_owner(out, "gemm_f32_bf16_bits_dev output")?;
        self.expect_owner(activation, "gemm_f32_bf16_bits_dev activation")?;
        self.expect_owner_bf16(weight, "gemm_f32_bf16_bits_dev weight")?;

        let (activation_len, weight_len, output_len) =
            validate_mixed_bf16_dims(m, n, k, "gemm_f32_bf16_bits_dev")?;
        expect_len(
            "gemm_f32_bf16_bits_dev activation",
            activation.len,
            activation_len,
        )?;
        expect_len("gemm_f32_bf16_bits_dev weight", weight.len, weight_len)?;
        expect_len("gemm_f32_bf16_bits_dev output", out.len, output_len)?;
        if m == 0 || n == 0 {
            return Ok(());
        }

        let dims = GemmF32Bf16BitsDims {
            m: m as u32,
            n: n as u32,
            k: k as u32,
        };
        let (grid, tg) = grid_2d(n, m);
        self.dispatch_compute(
            self.gemm_f32_bf16_bits_pipeline,
            &[&activation.buf, &weight.buf, &out.buf],
            (&dims as *const GemmF32Bf16BitsDims).cast::<c_void>(),
            size_of::<GemmF32Bf16BitsDims>(),
            grid,
            tg,
            "gemm_f32_bf16_bits",
        )
    }

    /// Device-resident stride/dilation-aware Conv1d. The input, weights,
    /// optional bias, and output all remain in Metal buffers; this method
    /// submits one pass and performs no host readback. The original
    /// `[out,in,kernel]` weight layout is retained and `dilation` is applied
    /// directly by the Metal kernel, so callers must not pre-expand weights.
    /// Like every current `*_dev` primitive, this is one synchronous
    /// commit/wait; fusing a complete vocoder graph is a separate layer.
    #[allow(clippy::too_many_arguments)]
    pub fn conv1d_dev(
        &self,
        out: &mut MetalDeviceTensor<'_>,
        input: &MetalDeviceTensor<'_>,
        weight: &MetalDeviceTensor<'_>,
        bias: Option<&MetalDeviceTensor<'_>>,
        in_ch: usize,
        in_len: usize,
        out_ch: usize,
        kernel: usize,
        stride: usize,
        dilation: usize,
        padding: usize,
    ) -> Result<()> {
        self.expect_owner(out, "conv1d_dev output")?;
        self.expect_owner(input, "conv1d_dev input")?;
        self.expect_owner(weight, "conv1d_dev weight")?;
        if let Some(b) = bias {
            self.expect_owner(b, "conv1d_dev bias")?;
        }
        if in_ch == 0 || in_len == 0 || out_ch == 0 || kernel == 0 || stride == 0 || dilation == 0 {
            return Err(VokraError::InvalidArgument(
                "conv1d_dev dimensions and stride/dilation must be > 0".to_owned(),
            ));
        }
        let effective = checked_mul(kernel - 1, dilation, "conv1d_dev effective kernel")?
            .checked_add(1)
            .ok_or_else(|| {
                VokraError::InvalidArgument("conv1d_dev effective kernel overflow".to_owned())
            })?;
        let padded = checked_mul(2, padding, "conv1d_dev padding")?
            .checked_add(in_len)
            .ok_or_else(|| {
                VokraError::InvalidArgument("conv1d_dev padded length overflow".to_owned())
            })?;
        if padded < effective {
            return Err(VokraError::InvalidArgument(
                "conv1d_dev padded input is smaller than the effective kernel".to_owned(),
            ));
        }
        let out_len = (padded - effective) / stride + 1;
        let last_t = checked_mul(out_len - 1, stride, "conv1d_dev dispatch index")?
            .checked_add(checked_mul(
                kernel - 1,
                dilation,
                "conv1d_dev dispatch tap",
            )?)
            .ok_or_else(|| {
                VokraError::InvalidArgument("conv1d_dev dispatch index overflow".to_owned())
            })?;
        checked_u32(last_t, "conv1d_dev dispatch index")?;
        expect_len(
            "conv1d_dev input",
            input.len,
            checked_mul(in_ch, in_len, "conv1d_dev input")?,
        )?;
        expect_len(
            "conv1d_dev weight",
            weight.len,
            checked_mul(
                checked_mul(out_ch, in_ch, "conv1d_dev weight")?,
                kernel,
                "conv1d_dev weight",
            )?,
        )?;
        expect_len(
            "conv1d_dev output",
            out.len,
            checked_mul(out_ch, out_len, "conv1d_dev output")?,
        )?;
        if let Some(b) = bias {
            expect_len("conv1d_dev bias", b.len, out_ch)?;
        }
        let dummy = self.new_buffer_from_slice(&[0.0f32])?;
        let bias_buf = bias.map_or(&dummy, |b| &b.buf);
        let dims = Conv1dDims {
            in_per_group: checked_u32(in_ch, "conv1d_dev in_ch")?,
            in_len: checked_u32(in_len, "conv1d_dev in_len")?,
            out_ch: checked_u32(out_ch, "conv1d_dev out_ch")?,
            kernel_size: checked_u32(kernel, "conv1d_dev kernel")?,
            out_len: checked_u32(out_len, "conv1d_dev out_len")?,
            stride: checked_u32(stride, "conv1d_dev stride")?,
            padding: checked_u32(padding, "conv1d_dev padding")?,
            has_bias: u32::from(bias.is_some()),
            out_per_group: checked_u32(out_ch, "conv1d_dev out_ch")?,
            dilation: checked_u32(dilation, "conv1d_dev dilation")?,
        };
        self.pooled(|| {
            let cmd = self.new_command_buffer("conv1d_dev")?;
            let (grid, tg) = grid_2d(out_len, out_ch);
            self.encode_pass(
                cmd,
                self.conv1d_pipeline,
                &[&input.buf, &weight.buf, bias_buf, &out.buf],
                (&dims as *const Conv1dDims).cast::<c_void>(),
                size_of::<Conv1dDims>(),
                grid,
                tg,
                "conv1d_dev",
            )?;
            self.commit_and_wait(cmd, "conv1d_dev")
        })
    }

    /// Device-resident PyTorch-layout ConvTranspose1d. This is the native
    /// upsample primitive for HiFTNet and BigVGAN; only the final caller-owned
    /// tensor should be passed to [`Self::download`]. The operation performs
    /// one synchronous commit/wait and does not itself read back intermediates.
    #[allow(clippy::too_many_arguments)]
    pub fn conv_transpose1d_dev(
        &self,
        out: &mut MetalDeviceTensor<'_>,
        input: &MetalDeviceTensor<'_>,
        weight: &MetalDeviceTensor<'_>,
        bias: Option<&MetalDeviceTensor<'_>>,
        in_ch: usize,
        in_len: usize,
        out_ch: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
        output_padding: usize,
    ) -> Result<()> {
        self.expect_owner(out, "conv_transpose1d_dev output")?;
        self.expect_owner(input, "conv_transpose1d_dev input")?;
        self.expect_owner(weight, "conv_transpose1d_dev weight")?;
        if let Some(b) = bias {
            self.expect_owner(b, "conv_transpose1d_dev bias")?;
        }
        if in_ch == 0 || in_len == 0 || out_ch == 0 || kernel == 0 || stride == 0 {
            return Err(VokraError::InvalidArgument(
                "conv_transpose1d_dev dimensions and stride/kernel must be > 0".to_owned(),
            ));
        }
        if output_padding >= stride {
            return Err(VokraError::InvalidArgument(
                "conv_transpose1d_dev output_padding must be < stride".to_owned(),
            ));
        }
        let base = checked_mul(in_len - 1, stride, "conv_transpose1d_dev output")?
            .checked_add(kernel)
            .and_then(|v| v.checked_add(output_padding))
            .ok_or_else(|| {
                VokraError::InvalidArgument("conv_transpose1d_dev output overflow".to_owned())
            })?;
        let trim = checked_mul(2, padding, "conv_transpose1d_dev padding")?;
        if trim > base {
            return Err(VokraError::InvalidArgument(
                "conv_transpose1d_dev padding exceeds output extent".to_owned(),
            ));
        }
        let out_len = base - trim;
        if out_len == 0 {
            return Err(VokraError::InvalidArgument(
                "conv_transpose1d_dev output extent must be > 0".to_owned(),
            ));
        }
        let last_input_step = checked_mul(in_len - 1, stride, "conv_transpose1d_dev input step")?;
        checked_i32(last_input_step, "conv_transpose1d_dev input step")?;
        checked_i32(out_len - 1, "conv_transpose1d_dev output index")?;
        let signed_sum = (out_len - 1).checked_add(padding).ok_or_else(|| {
            VokraError::InvalidArgument("conv_transpose1d_dev index overflow".to_owned())
        })?;
        checked_i32(signed_sum, "conv_transpose1d_dev signed index")?;
        expect_len(
            "conv_transpose1d_dev input",
            input.len,
            checked_mul(in_ch, in_len, "conv_transpose1d_dev input")?,
        )?;
        expect_len(
            "conv_transpose1d_dev weight",
            weight.len,
            checked_mul(
                checked_mul(in_ch, out_ch, "conv_transpose1d_dev weight")?,
                kernel,
                "conv_transpose1d_dev weight",
            )?,
        )?;
        expect_len(
            "conv_transpose1d_dev output",
            out.len,
            checked_mul(out_ch, out_len, "conv_transpose1d_dev output")?,
        )?;
        if let Some(b) = bias {
            expect_len("conv_transpose1d_dev bias", b.len, out_ch)?;
        }
        let dummy = self.new_buffer_from_slice(&[0.0f32])?;
        let bias_buf = bias.map_or(&dummy, |b| &b.buf);
        let dims = ConvTranspose1dDims {
            in_ch: checked_u32(in_ch, "conv_transpose1d_dev in_ch")?,
            out_ch: checked_u32(out_ch, "conv_transpose1d_dev out_ch")?,
            kernel_size: checked_u32(kernel, "conv_transpose1d_dev kernel")?,
            t_in: checked_u32(in_len, "conv_transpose1d_dev in_len")?,
            t_out: checked_u32(out_len, "conv_transpose1d_dev out_len")?,
            stride: checked_u32(stride, "conv_transpose1d_dev stride")?,
            padding: checked_u32(padding, "conv_transpose1d_dev padding")?,
            has_bias: u32::from(bias.is_some()),
        };
        self.pooled(|| {
            let cmd = self.new_command_buffer("conv_transpose1d_dev")?;
            let (grid, tg) = grid_2d(out_len, out_ch);
            self.encode_pass(
                cmd,
                self.conv_transpose1d_pipeline,
                &[&input.buf, &weight.buf, bias_buf, &out.buf],
                (&dims as *const ConvTranspose1dDims).cast::<c_void>(),
                size_of::<ConvTranspose1dDims>(),
                grid,
                tg,
                "conv_transpose1d_dev",
            )?;
            self.commit_and_wait(cmd, "conv_transpose1d_dev")
        })
    }

    /// Device-resident channel-major 1-D padding. `reflect = true` implements
    /// PyTorch ReflectionPad1d (requiring each side to be smaller than the
    /// input length); `false` implements ReplicationPad1d. The operation is a
    /// pure device copy and never performs an intermediate D2H transfer. It
    /// is submitted synchronously as one commit/wait.
    #[allow(clippy::too_many_arguments)] // intrinsic padding shape + device tensors
    pub fn pad1d_dev(
        &self,
        out: &mut MetalDeviceTensor<'_>,
        input: &MetalDeviceTensor<'_>,
        channels: usize,
        time_in: usize,
        left: usize,
        right: usize,
        reflect: bool,
    ) -> Result<()> {
        self.expect_owner(out, "pad1d_dev output")?;
        self.expect_owner(input, "pad1d_dev input")?;
        if channels == 0 || time_in == 0 {
            return Err(VokraError::InvalidArgument(
                "pad1d_dev channels and time_in must be > 0".to_owned(),
            ));
        }
        if reflect && (left >= time_in || right >= time_in) {
            return Err(VokraError::InvalidArgument(
                "pad1d_dev reflect padding must be smaller than input length".to_owned(),
            ));
        }
        let time_out = time_in
            .checked_add(left)
            .and_then(|v| v.checked_add(right))
            .ok_or_else(|| {
                VokraError::InvalidArgument("pad1d_dev output length overflow".to_owned())
            })?;
        checked_i32(time_out, "pad1d_dev time_out")?;
        if reflect {
            let reflect_bound = checked_mul(time_in, 3, "pad1d_dev reflect index")?;
            checked_i32(reflect_bound, "pad1d_dev reflect index")?;
        } else {
            checked_i32(time_in, "pad1d_dev time_in")?;
        }
        expect_len(
            "pad1d_dev input",
            input.len,
            checked_mul(channels, time_in, "pad1d_dev input")?,
        )?;
        expect_len(
            "pad1d_dev output",
            out.len,
            checked_mul(channels, time_out, "pad1d_dev output")?,
        )?;
        let dims = Pad1dDims {
            channels: checked_u32(channels, "pad1d_dev channels")?,
            time_in: checked_u32(time_in, "pad1d_dev time_in")?,
            time_out: checked_u32(time_out, "pad1d_dev time_out")?,
            left: checked_u32(left, "pad1d_dev left")?,
            right: checked_u32(right, "pad1d_dev right")?,
            mode: u32::from(!reflect),
        };
        self.pooled(|| {
            let cmd = self.new_command_buffer("pad1d_dev")?;
            let (grid, tg) = grid_2d(time_out, channels);
            self.encode_pass(
                cmd,
                self.pad1d_pipeline,
                &[&input.buf, &out.buf],
                (&dims as *const Pad1dDims).cast::<c_void>(),
                size_of::<Pad1dDims>(),
                grid,
                tg,
                "pad1d_dev",
            )?;
            self.commit_and_wait(cmd, "pad1d_dev")
        })
    }

    /// Device-resident LeakyReLU. This is the activation seam used around
    /// HiFTNet upsample stages; `slope` is an explicit model attribute. One
    /// synchronous commit/wait is issued and no intermediate is read back.
    pub fn leaky_relu_dev(
        &self,
        out: &mut MetalDeviceTensor<'_>,
        input: &MetalDeviceTensor<'_>,
        slope: f32,
    ) -> Result<()> {
        self.expect_owner(out, "leaky_relu_dev output")?;
        self.expect_owner(input, "leaky_relu_dev input")?;
        expect_len("leaky_relu_dev output", out.len, input.len)?;
        if input.is_empty() {
            return Ok(());
        }
        let dims = LeakyReluDims {
            n: u32::try_from(input.len).map_err(|_| {
                VokraError::InvalidArgument("leaky_relu_dev length exceeds u32".to_owned())
            })?,
            slope,
        };
        self.pooled(|| {
            let cmd = self.new_command_buffer("leaky_relu_dev")?;
            let (grid, tg) = grid_1d(input.len);
            self.encode_pass(
                cmd,
                self.leaky_relu_pipeline,
                &[&input.buf, &out.buf],
                (&dims as *const LeakyReluDims).cast::<c_void>(),
                size_of::<LeakyReluDims>(),
                grid,
                tg,
                "leaky_relu_dev",
            )?;
            self.commit_and_wait(cmd, "leaky_relu_dev")
        })
    }

    /// Device-resident terminal tanh. Used by BigVGAN's final output seam.
    /// This is one synchronous commit/wait with no intermediate readback.
    pub fn tanh_dev(
        &self,
        out: &mut MetalDeviceTensor<'_>,
        input: &MetalDeviceTensor<'_>,
    ) -> Result<()> {
        self.expect_owner(out, "tanh_dev output")?;
        self.expect_owner(input, "tanh_dev input")?;
        expect_len("tanh_dev output", out.len, input.len)?;
        if input.is_empty() {
            return Ok(());
        }
        let dims = TanhDims {
            n: u32::try_from(input.len).map_err(|_| {
                VokraError::InvalidArgument("tanh_dev length exceeds u32".to_owned())
            })?,
        };
        self.pooled(|| {
            let cmd = self.new_command_buffer("tanh_dev")?;
            let (grid, tg) = grid_1d(input.len);
            self.encode_pass(
                cmd,
                self.tanh_pipeline,
                &[&input.buf, &out.buf],
                (&dims as *const TanhDims).cast::<c_void>(),
                size_of::<TanhDims>(),
                grid,
                tg,
                "tanh_dev",
            )?;
            self.commit_and_wait(cmd, "tanh_dev")
        })
    }

    /// Device-resident OUVE reverse-diffusion predictor. All state and noise
    /// buffers remain on Metal; the caller can chain this with the corrector
    /// without an intermediate readback. The scalar coefficients mirror the
    /// pinned `sp-uhh/sgmse` OUVE implementation.
    #[allow(clippy::too_many_arguments)]
    pub fn ouve_reverse_diffusion_dev(
        &self,
        out: &mut MetalDeviceTensor<'_>,
        out_mean: &mut MetalDeviceTensor<'_>,
        x: &MetalDeviceTensor<'_>,
        y: &MetalDeviceTensor<'_>,
        score: &MetalDeviceTensor<'_>,
        noise: &MetalDeviceTensor<'_>,
        theta: f32,
        sigma_min: f32,
        sigma_max: f32,
        t: f32,
        step: f32,
        probability_flow: bool,
    ) -> Result<()> {
        self.expect_owner(out, "ouve_reverse_diffusion_dev output")?;
        self.expect_owner(out_mean, "ouve_reverse_diffusion_dev mean")?;
        self.expect_owner(x, "ouve_reverse_diffusion_dev x")?;
        self.expect_owner(y, "ouve_reverse_diffusion_dev y")?;
        self.expect_owner(score, "ouve_reverse_diffusion_dev score")?;
        self.expect_owner(noise, "ouve_reverse_diffusion_dev noise")?;
        validate_ouve_params(theta, sigma_min, sigma_max, t, step, None)?;
        let n = x.len();
        for (name, len) in [
            ("y", y.len()),
            ("score", score.len()),
            ("noise", noise.len()),
            ("out", out.len()),
            ("out_mean", out_mean.len()),
        ] {
            expect_len("ouve_reverse_diffusion_dev buffer", len, n).map_err(|_| {
                VokraError::InvalidArgument(format!(
                    "ouve_reverse_diffusion_dev {name} length {len} != x length {n}"
                ))
            })?;
        }
        if n == 0 {
            return Ok(());
        }
        let dims = OuvEReverseDims {
            n: checked_u32(n, "ouve_reverse_diffusion_dev length")?,
            theta,
            sigma_min,
            sigma_max,
            t,
            step,
            probability_flow: u32::from(probability_flow),
        };
        self.pooled(|| {
            let cmd = self.new_command_buffer("ouve_reverse_diffusion_dev")?;
            let (grid, tg) = grid_1d(n);
            self.encode_pass(
                cmd,
                self.ouve_reverse_diffusion_pipeline,
                &[
                    &x.buf,
                    &y.buf,
                    &score.buf,
                    &noise.buf,
                    &out.buf,
                    &out_mean.buf,
                ],
                (&dims as *const OuvEReverseDims).cast::<c_void>(),
                size_of::<OuvEReverseDims>(),
                grid,
                tg,
                "ouve_reverse_diffusion_dev",
            )?;
            self.commit_and_wait(cmd, "ouve_reverse_diffusion_dev")
        })
    }

    /// Device-resident OUVE annealed-Langevin corrector. This is the exact
    /// source `2 * (snr * std(t))²` update and never falls back to the CPU.
    #[allow(clippy::too_many_arguments)]
    pub fn ouve_annealed_langevin_dev(
        &self,
        out: &mut MetalDeviceTensor<'_>,
        out_mean: &mut MetalDeviceTensor<'_>,
        x: &MetalDeviceTensor<'_>,
        score: &MetalDeviceTensor<'_>,
        noise: &MetalDeviceTensor<'_>,
        theta: f32,
        sigma_min: f32,
        sigma_max: f32,
        t: f32,
        snr: f32,
    ) -> Result<()> {
        self.expect_owner(out, "ouve_annealed_langevin_dev output")?;
        self.expect_owner(out_mean, "ouve_annealed_langevin_dev mean")?;
        self.expect_owner(x, "ouve_annealed_langevin_dev x")?;
        self.expect_owner(score, "ouve_annealed_langevin_dev score")?;
        self.expect_owner(noise, "ouve_annealed_langevin_dev noise")?;
        validate_ouve_params(theta, sigma_min, sigma_max, t, 1.0, Some(snr))?;
        let n = x.len();
        for (name, len) in [
            ("score", score.len()),
            ("noise", noise.len()),
            ("out", out.len()),
            ("out_mean", out_mean.len()),
        ] {
            expect_len("ouve_annealed_langevin_dev buffer", len, n).map_err(|_| {
                VokraError::InvalidArgument(format!(
                    "ouve_annealed_langevin_dev {name} length {len} != x length {n}"
                ))
            })?;
        }
        if n == 0 {
            return Ok(());
        }
        let dims = OuvEAnnealedDims {
            n: checked_u32(n, "ouve_annealed_langevin_dev length")?,
            theta,
            sigma_min,
            sigma_max,
            t,
            snr,
        };
        self.pooled(|| {
            let cmd = self.new_command_buffer("ouve_annealed_langevin_dev")?;
            let (grid, tg) = grid_1d(n);
            self.encode_pass(
                cmd,
                self.ouve_annealed_langevin_pipeline,
                &[&x.buf, &score.buf, &noise.buf, &out.buf, &out_mean.buf],
                (&dims as *const OuvEAnnealedDims).cast::<c_void>(),
                size_of::<OuvEAnnealedDims>(),
                grid,
                tg,
                "ouve_annealed_langevin_dev",
            )?;
            self.commit_and_wait(cmd, "ouve_annealed_langevin_dev")
        })
    }

    /// Device-resident Snake activation. Parameters are effective FP32 alpha
    /// values; exponentiation for log-scale checkpoints belongs at load time.
    pub fn snake_activation_dev(
        &self,
        out: &mut MetalDeviceTensor<'_>,
        input: &MetalDeviceTensor<'_>,
        alpha: &MetalDeviceTensor<'_>,
        channels: usize,
        time: usize,
    ) -> Result<()> {
        self.expect_owner(out, "snake_activation_dev output")?;
        self.expect_owner(input, "snake_activation_dev input")?;
        self.expect_owner(alpha, "snake_activation_dev alpha")?;
        let n = checked_mul(channels, time, "snake_activation_dev channels*time")?;
        expect_len("snake_activation_dev input", input.len, n)?;
        expect_len("snake_activation_dev output", out.len, n)?;
        expect_len("snake_activation_dev alpha", alpha.len, channels)?;
        if n == 0 {
            return Ok(());
        }
        let dims = SnakeActivationDims {
            channels: checked_u32(channels, "snake_activation_dev channels")?,
            time: checked_u32(time, "snake_activation_dev time")?,
        };
        self.pooled(|| {
            let cmd = self.new_command_buffer("snake_activation_dev")?;
            let (grid, tg) = grid_2d(time, channels);
            self.encode_pass(
                cmd,
                self.snake_activation_pipeline,
                &[&input.buf, &alpha.buf, &out.buf],
                (&dims as *const SnakeActivationDims).cast::<c_void>(),
                size_of::<SnakeActivationDims>(),
                grid,
                tg,
                "snake_activation_dev",
            )?;
            self.commit_and_wait(cmd, "snake_activation_dev")
        })
    }

    /// Device-resident SnakeBeta activation with effective alpha/beta vectors.
    pub fn snake_beta_dev(
        &self,
        out: &mut MetalDeviceTensor<'_>,
        input: &MetalDeviceTensor<'_>,
        alpha: &MetalDeviceTensor<'_>,
        beta: &MetalDeviceTensor<'_>,
        channels: usize,
        time: usize,
    ) -> Result<()> {
        self.expect_owner(out, "snake_beta_dev output")?;
        self.expect_owner(input, "snake_beta_dev input")?;
        self.expect_owner(alpha, "snake_beta_dev alpha")?;
        self.expect_owner(beta, "snake_beta_dev beta")?;
        let n = checked_mul(channels, time, "snake_beta_dev channels*time")?;
        expect_len("snake_beta_dev input", input.len, n)?;
        expect_len("snake_beta_dev output", out.len, n)?;
        expect_len("snake_beta_dev alpha", alpha.len, channels)?;
        expect_len("snake_beta_dev beta", beta.len, channels)?;
        if n == 0 {
            return Ok(());
        }
        let dims = SnakeBetaDims {
            channels: checked_u32(channels, "snake_beta_dev channels")?,
            time: checked_u32(time, "snake_beta_dev time")?,
        };
        self.pooled(|| {
            let cmd = self.new_command_buffer("snake_beta_dev")?;
            let (grid, tg) = grid_2d(time, channels);
            self.encode_pass(
                cmd,
                self.snake_beta_pipeline,
                &[&input.buf, &alpha.buf, &beta.buf, &out.buf],
                (&dims as *const SnakeBetaDims).cast::<c_void>(),
                size_of::<SnakeBetaDims>(),
                grid,
                tg,
                "snake_beta_dev",
            )?;
            self.commit_and_wait(cmd, "snake_beta_dev")
        })
    }

    /// Device-resident anti-aliased upsample. FIR taps are uploaded once by
    /// the caller and can be shared by every invocation. Each invocation is a
    /// single synchronous commit/wait and performs no intermediate readback.
    #[allow(clippy::too_many_arguments)] // intrinsic FIR upsample shape + device tensors
    pub fn anti_aliased_upsample_dev(
        &self,
        out: &mut MetalDeviceTensor<'_>,
        input: &MetalDeviceTensor<'_>,
        kernel: &MetalDeviceTensor<'_>,
        ratio: usize,
        channels: usize,
        time_in: usize,
        taps: usize,
    ) -> Result<()> {
        self.expect_owner(out, "anti_aliased_upsample_dev output")?;
        self.expect_owner(input, "anti_aliased_upsample_dev input")?;
        self.expect_owner(kernel, "anti_aliased_upsample_dev kernel")?;
        if ratio == 0 || taps == 0 {
            return Err(VokraError::InvalidArgument(
                "anti_aliased_upsample_dev ratio/taps must be > 0".to_owned(),
            ));
        }
        let time_out = checked_mul(time_in, ratio, "anti_aliased_upsample_dev time_out")?;
        expect_len(
            "anti_aliased_upsample_dev input",
            input.len,
            checked_mul(channels, time_in, "anti_aliased_upsample_dev input")?,
        )?;
        expect_len("anti_aliased_upsample_dev kernel", kernel.len, taps)?;
        expect_len(
            "anti_aliased_upsample_dev output",
            out.len,
            checked_mul(channels, time_out, "anti_aliased_upsample_dev output")?,
        )?;
        if channels == 0 || time_in == 0 {
            return Ok(());
        }
        let dims = AntiAliasedUpsampleDims {
            channels: checked_u32(channels, "anti_aliased_upsample_dev channels")?,
            time_in: checked_u32(time_in, "anti_aliased_upsample_dev time_in")?,
            time_out: checked_u32(time_out, "anti_aliased_upsample_dev time_out")?,
            ratio: checked_u32(ratio, "anti_aliased_upsample_dev ratio")?,
            taps: checked_u32(taps, "anti_aliased_upsample_dev taps")?,
        };
        self.pooled(|| {
            let cmd = self.new_command_buffer("anti_aliased_upsample_dev")?;
            let (grid, tg) = grid_2d(time_out, channels);
            self.encode_pass(
                cmd,
                self.anti_aliased_upsample_pipeline,
                &[&input.buf, &kernel.buf, &out.buf],
                (&dims as *const AntiAliasedUpsampleDims).cast::<c_void>(),
                size_of::<AntiAliasedUpsampleDims>(),
                grid,
                tg,
                "anti_aliased_upsample_dev",
            )?;
            self.commit_and_wait(cmd, "anti_aliased_upsample_dev")
        })
    }

    /// Device-resident BigVGAN `DownSample1d`: asymmetric replicate padding,
    /// FIR filtering, and strided decimation. The filter and all tensors are
    /// channel-major; only an explicit final `download` crosses D2H. This
    /// primitive is one synchronous commit/wait, not an async/fused graph.
    #[allow(clippy::too_many_arguments)] // intrinsic FIR downsample shape + device tensors
    pub fn anti_aliased_downsample_dev(
        &self,
        out: &mut MetalDeviceTensor<'_>,
        input: &MetalDeviceTensor<'_>,
        kernel: &MetalDeviceTensor<'_>,
        ratio: usize,
        channels: usize,
        time_in: usize,
        taps: usize,
    ) -> Result<()> {
        self.expect_owner(out, "anti_aliased_downsample_dev output")?;
        self.expect_owner(input, "anti_aliased_downsample_dev input")?;
        self.expect_owner(kernel, "anti_aliased_downsample_dev kernel")?;
        if ratio == 0 || taps == 0 || time_in == 0 {
            return Err(VokraError::InvalidArgument(
                "anti_aliased_downsample_dev ratio/taps/time_in must be > 0".to_owned(),
            ));
        }
        let even = usize::from(taps % 2 == 0);
        let pad_left = taps / 2 - even;
        let pad_right = taps / 2;
        let padded_time = time_in
            .checked_add(pad_left)
            .and_then(|v| v.checked_add(pad_right))
            .ok_or_else(|| {
                VokraError::InvalidArgument(
                    "anti_aliased_downsample_dev padded length overflow".to_owned(),
                )
            })?;
        if padded_time < taps {
            return Err(VokraError::InvalidArgument(
                "anti_aliased_downsample_dev filter exceeds padded input".to_owned(),
            ));
        }
        let time_out = (padded_time - taps) / ratio + 1;
        checked_u32(padded_time, "anti_aliased_downsample_dev padded length")?;
        checked_i32(time_in, "anti_aliased_downsample_dev time_in")?;
        checked_i32(padded_time, "anti_aliased_downsample_dev padded length")?;
        checked_i32(pad_left, "anti_aliased_downsample_dev pad_left")?;
        expect_len(
            "anti_aliased_downsample_dev input",
            input.len,
            checked_mul(channels, time_in, "anti_aliased_downsample_dev input")?,
        )?;
        expect_len("anti_aliased_downsample_dev kernel", kernel.len, taps)?;
        expect_len(
            "anti_aliased_downsample_dev output",
            out.len,
            checked_mul(channels, time_out, "anti_aliased_downsample_dev output")?,
        )?;
        let dims = AntiAliasedDownsampleDims {
            channels: checked_u32(channels, "anti_aliased_downsample_dev channels")?,
            time_in: checked_u32(time_in, "anti_aliased_downsample_dev time_in")?,
            time_out: checked_u32(time_out, "anti_aliased_downsample_dev time_out")?,
            ratio: checked_u32(ratio, "anti_aliased_downsample_dev ratio")?,
            taps: checked_u32(taps, "anti_aliased_downsample_dev taps")?,
            pad_left: checked_u32(pad_left, "anti_aliased_downsample_dev pad_left")?,
        };
        if channels == 0 {
            return Ok(());
        }
        self.pooled(|| {
            let cmd = self.new_command_buffer("anti_aliased_downsample_dev")?;
            let (grid, tg) = grid_2d(time_out, channels);
            self.encode_pass(
                cmd,
                self.anti_aliased_downsample_pipeline,
                &[&input.buf, &kernel.buf, &out.buf],
                (&dims as *const AntiAliasedDownsampleDims).cast::<c_void>(),
                size_of::<AntiAliasedDownsampleDims>(),
                grid,
                tg,
                "anti_aliased_downsample_dev",
            )?;
            self.commit_and_wait(cmd, "anti_aliased_downsample_dev")
        })
    }

    /// Device-resident scalar multiply, intended for branch averaging after
    /// an explicit residual sum. One synchronous commit/wait; no D2H copy.
    pub fn scale_dev(
        &self,
        out: &mut MetalDeviceTensor<'_>,
        input: &MetalDeviceTensor<'_>,
        scale: f32,
    ) -> Result<()> {
        self.expect_owner(out, "scale_dev output")?;
        self.expect_owner(input, "scale_dev input")?;
        expect_len("scale_dev output", out.len, input.len)?;
        if input.is_empty() {
            return Ok(());
        }
        let dims = ScaleDims {
            n: checked_u32(input.len, "scale_dev length")?,
            scale,
        };
        self.pooled(|| {
            let cmd = self.new_command_buffer("scale_dev")?;
            let (grid, tg) = grid_1d(input.len);
            self.encode_pass(
                cmd,
                self.scale_pipeline,
                &[&input.buf, &out.buf],
                (&dims as *const ScaleDims).cast::<c_void>(),
                size_of::<ScaleDims>(),
                grid,
                tg,
                "scale_dev",
            )?;
            self.commit_and_wait(cmd, "scale_dev")
        })
    }

    /// Device-resident clamp. The lower/upper bounds are explicit model
    /// attributes; BigVGAN's terminal seam uses `[-1, 1]`.
    pub fn clamp_dev(
        &self,
        out: &mut MetalDeviceTensor<'_>,
        input: &MetalDeviceTensor<'_>,
        lower: f32,
        upper: f32,
    ) -> Result<()> {
        self.expect_owner(out, "clamp_dev output")?;
        self.expect_owner(input, "clamp_dev input")?;
        if lower > upper {
            return Err(VokraError::InvalidArgument(
                "clamp_dev lower bound must not exceed upper bound".to_owned(),
            ));
        }
        expect_len("clamp_dev output", out.len, input.len)?;
        if input.is_empty() {
            return Ok(());
        }
        let dims = ClampDims {
            n: checked_u32(input.len, "clamp_dev length")?,
            lower,
            upper,
        };
        self.pooled(|| {
            let cmd = self.new_command_buffer("clamp_dev")?;
            let (grid, tg) = grid_1d(input.len);
            self.encode_pass(
                cmd,
                self.clamp_pipeline,
                &[&input.buf, &out.buf],
                (&dims as *const ClampDims).cast::<c_void>(),
                size_of::<ClampDims>(),
                grid,
                tg,
                "clamp_dev",
            )?;
            self.commit_and_wait(cmd, "clamp_dev")
        })
    }

    /// Device-resident ELU with `alpha = 1`, matching HiFTNet's F0 predictor
    /// activation. One synchronous commit/wait is issued; no D2H transfer is
    /// performed.
    pub fn elu_dev(
        &self,
        out: &mut MetalDeviceTensor<'_>,
        input: &MetalDeviceTensor<'_>,
    ) -> Result<()> {
        self.expect_owner(out, "elu_dev output")?;
        self.expect_owner(input, "elu_dev input")?;
        expect_len("elu_dev output", out.len, input.len)?;
        if input.is_empty() {
            return Ok(());
        }
        let dims = EluDims {
            n: checked_u32(input.len, "elu_dev length")?,
        };
        self.pooled(|| {
            let cmd = self.new_command_buffer("elu_dev")?;
            let (grid, tg) = grid_1d(input.len);
            self.encode_pass(
                cmd,
                self.elu_pipeline,
                &[&input.buf, &out.buf],
                (&dims as *const EluDims).cast::<c_void>(),
                size_of::<EluDims>(),
                grid,
                tg,
                "elu_dev",
            )?;
            self.commit_and_wait(cmd, "elu_dev")
        })
    }

    /// Device-resident HiFTNet F0 linear head: `abs(bias + W·x)`. `input` is
    /// channel-major `[channels, time]`, `weight` is `[channels]`, `bias` is a
    /// one-element tensor, and `out` is `[time]`. The operation is one
    /// synchronous commit/wait and keeps all tensors on Metal.
    pub fn linear_abs_dev(
        &self,
        out: &mut MetalDeviceTensor<'_>,
        input: &MetalDeviceTensor<'_>,
        weight: &MetalDeviceTensor<'_>,
        bias: &MetalDeviceTensor<'_>,
        channels: usize,
        time: usize,
    ) -> Result<()> {
        self.expect_owner(out, "linear_abs_dev output")?;
        self.expect_owner(input, "linear_abs_dev input")?;
        self.expect_owner(weight, "linear_abs_dev weight")?;
        self.expect_owner(bias, "linear_abs_dev bias")?;
        if channels == 0 || time == 0 {
            return Err(VokraError::InvalidArgument(
                "linear_abs_dev channels and time must be > 0".to_owned(),
            ));
        }
        let input_len = checked_mul(channels, time, "linear_abs_dev channels*time")?;
        expect_len("linear_abs_dev input", input.len, input_len)?;
        expect_len("linear_abs_dev weight", weight.len, channels)?;
        expect_len("linear_abs_dev bias", bias.len, 1)?;
        expect_len("linear_abs_dev output", out.len, time)?;
        let dims = LinearAbsDims {
            channels: checked_u32(channels, "linear_abs_dev channels")?,
            time: checked_u32(time, "linear_abs_dev time")?,
        };
        self.pooled(|| {
            let cmd = self.new_command_buffer("linear_abs_dev")?;
            let (grid, tg) = grid_1d(time);
            self.encode_pass(
                cmd,
                self.linear_abs_pipeline,
                &[&input.buf, &weight.buf, &bias.buf, &out.buf],
                (&dims as *const LinearAbsDims).cast::<c_void>(),
                size_of::<LinearAbsDims>(),
                grid,
                tg,
                "linear_abs_dev",
            )?;
            self.commit_and_wait(cmd, "linear_abs_dev")
        })
    }

    /// Device-resident HiFTNet NSF source mixer: `tanh(bias + W·x)`.
    /// `input` is channel-major `[channels, time]`, `weight` is `[channels]`,
    /// `bias` is a one-element tensor, and `out` is `[time]`. This affine
    /// reduction is separate from [`Self::linear_abs_dev`] so the source
    /// branch can consume channel-major SineGen bytes without reinterpretation.
    pub fn linear_tanh_dev(
        &self,
        out: &mut MetalDeviceTensor<'_>,
        input: &MetalDeviceTensor<'_>,
        weight: &MetalDeviceTensor<'_>,
        bias: &MetalDeviceTensor<'_>,
        channels: usize,
        time: usize,
    ) -> Result<()> {
        self.expect_owner(out, "linear_tanh_dev output")?;
        self.expect_owner(input, "linear_tanh_dev input")?;
        self.expect_owner(weight, "linear_tanh_dev weight")?;
        self.expect_owner(bias, "linear_tanh_dev bias")?;
        if channels == 0 || time == 0 {
            return Err(VokraError::InvalidArgument(
                "linear_tanh_dev channels and time must be > 0".to_owned(),
            ));
        }
        let input_len = checked_mul(channels, time, "linear_tanh_dev channels*time")?;
        expect_len("linear_tanh_dev input", input.len, input_len)?;
        expect_len("linear_tanh_dev weight", weight.len, channels)?;
        expect_len("linear_tanh_dev bias", bias.len, 1)?;
        expect_len("linear_tanh_dev output", out.len, time)?;
        let dims = LinearAbsDims {
            channels: checked_u32(channels, "linear_tanh_dev channels")?,
            time: checked_u32(time, "linear_tanh_dev time")?,
        };
        self.pooled(|| {
            let cmd = self.new_command_buffer("linear_tanh_dev")?;
            let (grid, tg) = grid_1d(time);
            self.encode_pass(
                cmd,
                self.linear_tanh_pipeline,
                &[&input.buf, &weight.buf, &bias.buf, &out.buf],
                (&dims as *const LinearAbsDims).cast::<c_void>(),
                size_of::<LinearAbsDims>(),
                grid,
                tg,
                "linear_tanh_dev",
            )?;
            self.commit_and_wait(cmd, "linear_tanh_dev")
        })
    }

    /// Device-resident nearest-neighbour temporal upsample. Both tensors use
    /// channel-major `[channels, time]` layout and `out` must have shape
    /// `[channels, time_in * factor]`. One synchronous commit/wait is issued.
    pub fn nearest_upsample_dev(
        &self,
        out: &mut MetalDeviceTensor<'_>,
        input: &MetalDeviceTensor<'_>,
        channels: usize,
        time_in: usize,
        factor: usize,
    ) -> Result<()> {
        self.expect_owner(out, "nearest_upsample_dev output")?;
        self.expect_owner(input, "nearest_upsample_dev input")?;
        if channels == 0 || time_in == 0 || factor == 0 {
            return Err(VokraError::InvalidArgument(
                "nearest_upsample_dev channels, time_in, and factor must be > 0".to_owned(),
            ));
        }
        let time_out = checked_mul(time_in, factor, "nearest_upsample_dev time_out")?;
        expect_len(
            "nearest_upsample_dev input",
            input.len,
            checked_mul(channels, time_in, "nearest_upsample_dev input")?,
        )?;
        expect_len(
            "nearest_upsample_dev output",
            out.len,
            checked_mul(channels, time_out, "nearest_upsample_dev output")?,
        )?;
        let dims = NearestUpsampleDims {
            channels: checked_u32(channels, "nearest_upsample_dev channels")?,
            time_in: checked_u32(time_in, "nearest_upsample_dev time_in")?,
            time_out: checked_u32(time_out, "nearest_upsample_dev time_out")?,
            factor: checked_u32(factor, "nearest_upsample_dev factor")?,
        };
        self.pooled(|| {
            let cmd = self.new_command_buffer("nearest_upsample_dev")?;
            let (grid, tg) = grid_2d(time_out, channels);
            self.encode_pass(
                cmd,
                self.nearest_upsample_pipeline,
                &[&input.buf, &out.buf],
                (&dims as *const NearestUpsampleDims).cast::<c_void>(),
                size_of::<NearestUpsampleDims>(),
                grid,
                tg,
                "nearest_upsample_dev",
            )?;
            self.commit_and_wait(cmd, "nearest_upsample_dev")
        })
    }

    /// Device-resident deterministic SineGen route. `f0` is `[T]` and `out`
    /// is `[T, harmonic_num + 1]` (time-major). This matches
    /// `sinegen_deterministic_f32` and the deterministic `NsfEntropy` path;
    /// phase and stochastic noise are intentionally not part of this API.
    pub fn sinegen_deterministic_dev(
        &self,
        out: &mut MetalDeviceTensor<'_>,
        f0: &MetalDeviceTensor<'_>,
        samp_rate: u32,
        harmonic_num: u32,
        sine_amp: f32,
        voiced_threshold: f32,
    ) -> Result<()> {
        self.expect_owner(out, "sinegen_deterministic_dev output")?;
        self.expect_owner(f0, "sinegen_deterministic_dev f0")?;
        if f0.is_empty() {
            return Err(VokraError::InvalidArgument(
                "sinegen_deterministic_dev f0 must be non-empty".to_owned(),
            ));
        }
        if samp_rate == 0 {
            return Err(VokraError::InvalidArgument(
                "sinegen_deterministic_dev samp_rate must be > 0".to_owned(),
            ));
        }
        let h1 = (harmonic_num as usize).checked_add(1).ok_or_else(|| {
            VokraError::InvalidArgument(
                "sinegen_deterministic_dev harmonic count overflow".to_owned(),
            )
        })?;
        let expected = checked_mul(f0.len, h1, "sinegen_deterministic_dev output")?;
        expect_len("sinegen_deterministic_dev output", out.len, expected)?;
        let dims = SinegenDeterministicDims {
            t: checked_u32(f0.len, "sinegen_deterministic_dev time")?,
            h1: checked_u32(h1, "sinegen_deterministic_dev harmonics")?,
            samp_rate_f: samp_rate as f32,
            sine_amp,
            voiced_threshold,
        };
        self.pooled(|| {
            let cmd = self.new_command_buffer("sinegen_deterministic_dev")?;
            let (grid, tg) = grid_1d(h1);
            self.encode_pass(
                cmd,
                self.sinegen_deterministic_pipeline,
                &[&f0.buf, &out.buf],
                (&dims as *const SinegenDeterministicDims).cast::<c_void>(),
                size_of::<SinegenDeterministicDims>(),
                grid,
                tg,
                "sinegen_deterministic_dev",
            )?;
            self.commit_and_wait(cmd, "sinegen_deterministic_dev")
        })
    }

    /// Device-resident deterministic SineGen with the HiFT graph's required
    /// channel-major `[harmonic_num + 1, time]` output layout. This is kept
    /// separate from [`Self::sinegen_deterministic_dev`], whose legacy
    /// host-facing contract is time-major `[time, harmonic_num + 1]`.
    pub fn sinegen_deterministic_channel_major_dev(
        &self,
        out: &mut MetalDeviceTensor<'_>,
        f0: &MetalDeviceTensor<'_>,
        samp_rate: u32,
        harmonic_num: u32,
        sine_amp: f32,
        voiced_threshold: f32,
    ) -> Result<()> {
        self.expect_owner(out, "sinegen_deterministic_channel_major_dev output")?;
        self.expect_owner(f0, "sinegen_deterministic_channel_major_dev f0")?;
        if f0.is_empty() {
            return Err(VokraError::InvalidArgument(
                "sinegen_deterministic_channel_major_dev f0 must be non-empty".to_owned(),
            ));
        }
        if samp_rate == 0 {
            return Err(VokraError::InvalidArgument(
                "sinegen_deterministic_channel_major_dev samp_rate must be > 0".to_owned(),
            ));
        }
        let h1 = (harmonic_num as usize).checked_add(1).ok_or_else(|| {
            VokraError::InvalidArgument(
                "sinegen_deterministic_channel_major_dev harmonic count overflow".to_owned(),
            )
        })?;
        let expected = checked_mul(f0.len, h1, "sinegen_deterministic_channel_major_dev output")?;
        expect_len(
            "sinegen_deterministic_channel_major_dev output",
            out.len,
            expected,
        )?;
        let dims = SinegenDeterministicDims {
            t: checked_u32(f0.len, "sinegen_deterministic_channel_major_dev time")?,
            h1: checked_u32(h1, "sinegen_deterministic_channel_major_dev harmonics")?,
            samp_rate_f: samp_rate as f32,
            sine_amp,
            voiced_threshold,
        };
        self.pooled(|| {
            let cmd = self.new_command_buffer("sinegen_deterministic_channel_major_dev")?;
            let (grid, tg) = grid_1d(h1);
            self.encode_pass(
                cmd,
                self.sinegen_deterministic_channel_major_pipeline,
                &[&f0.buf, &out.buf],
                (&dims as *const SinegenDeterministicDims).cast::<c_void>(),
                size_of::<SinegenDeterministicDims>(),
                grid,
                tg,
                "sinegen_deterministic_channel_major_dev",
            )?;
            self.commit_and_wait(cmd, "sinegen_deterministic_channel_major_dev")
        })
    }

    /// Device-resident HiFT source STFT. The input is `[T]`; the output is
    /// channel-major `[Re F, frames; Im F, frames]`, using periodic Hann,
    /// centered `n_fft/2` reflect padding, backward (unscaled) RFFT, and the
    /// CPU frame formula from `StftAttrs::new`: pad to `T + 2*(n_fft/2)` and
    /// derive frames from the padded extent. This direct-DTF pass is
    /// synchronous and performs no host readback.
    pub fn hift_stft_dev(
        &self,
        out: &mut MetalDeviceTensor<'_>,
        input: &MetalDeviceTensor<'_>,
        n_fft: usize,
        hop: usize,
    ) -> Result<()> {
        self.expect_owner(out, "hift_stft_dev output")?;
        self.expect_owner(input, "hift_stft_dev input")?;
        if n_fft == 0 || hop == 0 {
            return Err(VokraError::InvalidArgument(
                "hift_stft_dev n_fft and hop must be non-zero".to_owned(),
            ));
        }
        let bins = n_fft / 2 + 1;
        let center = n_fft / 2;
        let padded_len = checked_mul(center, 2, "hift_stft_dev center padding")?
            .checked_add(input.len)
            .ok_or_else(|| {
                VokraError::InvalidArgument("hift_stft_dev padded length overflow".to_owned())
            })?;
        let frames = if padded_len >= n_fft {
            (padded_len - n_fft)
                .checked_div(hop)
                .and_then(|v| v.checked_add(1))
                .ok_or_else(|| {
                    VokraError::InvalidArgument("hift_stft_dev frame count overflow".to_owned())
                })?
        } else {
            0
        };
        let padded_total = if frames == 0 {
            0
        } else {
            checked_mul(frames - 1, hop, "hift_stft_dev frame extent")?
                .checked_add(n_fft)
                .ok_or_else(|| {
                    VokraError::InvalidArgument("hift_stft_dev extent overflow".to_owned())
                })?
        };
        checked_i32(input.len, "hift_stft_dev input length")?;
        if !input.is_empty() {
            checked_i32(
                checked_mul(input.len, 2, "hift_stft_dev reflect period")?,
                "hift_stft_dev reflect period",
            )?;
        }
        checked_i32(padded_total, "hift_stft_dev frame extent")?;
        checked_u32(
            checked_mul(n_fft, n_fft, "hift_stft_dev frequency product")?,
            "hift_stft_dev frequency product",
        )?;
        checked_u32(n_fft, "hift_stft_dev n_fft")?;
        checked_u32(hop, "hift_stft_dev hop")?;
        let expected = checked_mul(
            2,
            checked_mul(bins, frames, "hift_stft_dev frame buffer")?,
            "hift_stft_dev output",
        )?;
        expect_len("hift_stft_dev output", out.len, expected)?;
        if frames == 0 {
            return Ok(());
        }
        let dims = HiftStftDims {
            time_in: checked_u32(input.len, "hift_stft_dev time_in")?,
            n_fft: checked_u32(n_fft, "hift_stft_dev n_fft")?,
            hop: checked_u32(hop, "hift_stft_dev hop")?,
            frames: checked_u32(frames, "hift_stft_dev frames")?,
            bins: checked_u32(bins, "hift_stft_dev bins")?,
        };
        self.pooled(|| {
            let cmd = self.new_command_buffer("hift_stft_dev")?;
            let (grid, tg) = grid_2d(frames, bins);
            self.encode_pass(
                cmd,
                self.hift_stft_pipeline,
                &[&input.buf, &out.buf],
                (&dims as *const HiftStftDims).cast::<c_void>(),
                size_of::<HiftStftDims>(),
                grid,
                tg,
                "hift_stft_dev",
            )?;
            self.commit_and_wait(cmd, "hift_stft_dev")
        })
    }

    /// Device-resident HiFT logits postprocess. `logits` and `out` use the
    /// channel-major `[magnitude F, frames; phase F, frames]` layout; `out`
    /// receives `[Re F, frames; Im F, frames]`. Magnitudes use the exact
    /// `min(exp(logit), 100)` order and phases use `sin(logit)`. Non-finite
    /// phase logits deliberately produce NaN, matching Rust `f32::sin`.
    pub fn complex_from_logits_dev(
        &self,
        out: &mut MetalDeviceTensor<'_>,
        logits: &MetalDeviceTensor<'_>,
        n_fft: usize,
        frames: usize,
    ) -> Result<()> {
        self.expect_owner(out, "complex_from_logits_dev output")?;
        self.expect_owner(logits, "complex_from_logits_dev logits")?;
        if n_fft == 0 || frames == 0 {
            return Err(VokraError::InvalidArgument(
                "complex_from_logits_dev n_fft and frames must be non-zero".to_owned(),
            ));
        }
        let bins = n_fft / 2 + 1;
        checked_u32(
            checked_mul(n_fft, n_fft, "complex_from_logits_dev frequency product")?,
            "complex_from_logits_dev frequency product",
        )?;
        let spectral_len = checked_mul(
            2,
            checked_mul(bins, frames, "complex_from_logits_dev frame buffer")?,
            "complex_from_logits_dev spectral buffer",
        )?;
        expect_len("complex_from_logits_dev logits", logits.len, spectral_len)?;
        expect_len("complex_from_logits_dev output", out.len, spectral_len)?;
        let dims = HiftComplexDims {
            frames: checked_u32(frames, "complex_from_logits_dev frames")?,
            bins: checked_u32(bins, "complex_from_logits_dev bins")?,
        };
        self.pooled(|| {
            let cmd = self.new_command_buffer("complex_from_logits_dev")?;
            let (grid, tg) = grid_2d(frames, 2 * bins);
            self.encode_pass(
                cmd,
                self.hift_complex_pipeline,
                &[&logits.buf, &out.buf],
                (&dims as *const HiftComplexDims).cast::<c_void>(),
                size_of::<HiftComplexDims>(),
                grid,
                tg,
                "complex_from_logits_dev",
            )?;
            self.commit_and_wait(cmd, "complex_from_logits_dev")
        })
    }

    /// Device-resident centered periodic-Hann iSTFT from a complex spectrum.
    /// `spectrum` and `out` are channel-major `[Re F, frames; Im F, frames]`
    /// and `[samples]`; this pass performs WOLA normalization and center trim,
    /// but intentionally does not apply an audio-limit clamp. Use
    /// [`Self::clamp_dev`] with the model's configured limit afterward.
    pub fn istft_dev(
        &self,
        out: &mut MetalDeviceTensor<'_>,
        spectrum: &MetalDeviceTensor<'_>,
        n_fft: usize,
        hop: usize,
        frames: usize,
    ) -> Result<()> {
        self.expect_owner(out, "istft_dev output")?;
        self.expect_owner(spectrum, "istft_dev spectrum")?;
        if n_fft == 0 || hop == 0 || frames == 0 {
            return Err(VokraError::InvalidArgument(
                "istft_dev n_fft, hop, and frames must be non-zero".to_owned(),
            ));
        }
        let bins = n_fft / 2 + 1;
        let total = checked_mul(frames - 1, hop, "istft_dev frame extent")?
            .checked_add(n_fft)
            .ok_or_else(|| VokraError::InvalidArgument("istft_dev extent overflow".to_owned()))?;
        let center_trim = checked_mul(2, n_fft / 2, "istft_dev center trim")?;
        let out_len = total.checked_sub(center_trim).ok_or_else(|| {
            VokraError::InvalidArgument("istft_dev center trim underflow".to_owned())
        })?;
        checked_u32(
            checked_mul(n_fft, n_fft, "istft_dev frequency product")?,
            "istft_dev frequency product",
        )?;
        checked_i32(total, "istft_dev frame extent")?;
        let spectral_len = checked_mul(
            2,
            checked_mul(bins, frames, "istft_dev frame buffer")?,
            "istft_dev spectrum",
        )?;
        expect_len("istft_dev spectrum", spectrum.len, spectral_len)?;
        expect_len("istft_dev output", out.len, out_len)?;
        let dims = HiftIstftDims {
            n_fft: checked_u32(n_fft, "istft_dev n_fft")?,
            hop: checked_u32(hop, "istft_dev hop")?,
            frames: checked_u32(frames, "istft_dev frames")?,
            bins: checked_u32(bins, "istft_dev bins")?,
            out_len: checked_u32(out_len, "istft_dev output length")?,
        };
        if out_len == 0 {
            return Ok(());
        }
        self.pooled(|| {
            let cmd = self.new_command_buffer("istft_dev")?;
            let (grid, tg) = grid_1d(out_len);
            self.encode_pass(
                cmd,
                self.hift_istft_pipeline,
                &[&spectrum.buf, &out.buf],
                (&dims as *const HiftIstftDims).cast::<c_void>(),
                size_of::<HiftIstftDims>(),
                grid,
                tg,
                "istft_dev",
            )?;
            self.commit_and_wait(cmd, "istft_dev")
        })
    }

    /// Device-resident convenience route for HiFTNet's logits → complex →
    /// iSTFT → configured clamp sequence. `audio_limit` is explicit (rather
    /// than an unconditional ±1), so model configurations with a larger or
    /// smaller limit retain their CPU semantics. For trait-level orchestration
    /// use [`Self::complex_from_logits_dev`], [`Self::istft_dev`], and
    /// [`Self::clamp_dev`] separately.
    pub fn hift_istft_dev(
        &self,
        out: &mut MetalDeviceTensor<'_>,
        logits: &MetalDeviceTensor<'_>,
        n_fft: usize,
        hop: usize,
        frames: usize,
        audio_limit: f32,
    ) -> Result<()> {
        self.expect_owner(out, "hift_istft_dev output")?;
        self.expect_owner(logits, "hift_istft_dev logits")?;
        if !audio_limit.is_finite() || audio_limit < 0.0 {
            return Err(VokraError::InvalidArgument(
                "hift_istft_dev audio_limit must be finite and non-negative".to_owned(),
            ));
        }
        if n_fft == 0 || hop == 0 || frames == 0 {
            return Err(VokraError::InvalidArgument(
                "hift_istft_dev n_fft, hop, and frames must be non-zero".to_owned(),
            ));
        }
        let bins = n_fft / 2 + 1;
        let spectral_len = checked_mul(
            2,
            checked_mul(bins, frames, "hift_istft_dev spectral buffer")?,
            "hift_istft_dev logits",
        )?;
        expect_len("hift_istft_dev logits", logits.len, spectral_len)?;
        let total = checked_mul(frames - 1, hop, "hift_istft_dev frame extent")?
            .checked_add(n_fft)
            .ok_or_else(|| {
                VokraError::InvalidArgument("hift_istft_dev extent overflow".to_owned())
            })?;
        let center_trim = checked_mul(2, n_fft / 2, "hift_istft_dev center trim")?;
        let out_len = total.checked_sub(center_trim).ok_or_else(|| {
            VokraError::InvalidArgument("hift_istft_dev center trim underflow".to_owned())
        })?;
        expect_len("hift_istft_dev output", out.len, out_len)?;
        let mut complex = self.alloc_dev(spectral_len)?;
        self.complex_from_logits_dev(&mut complex, logits, n_fft, frames)?;
        let mut raw = self.alloc_dev(out_len)?;
        self.istft_dev(&mut raw, &complex, n_fft, hop, frames)?;
        self.clamp_dev(out, &raw, -audio_limit, audio_limit)
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
        self.expect_owner(out, "layer_norm_dev output")?;
        self.expect_owner(x, "layer_norm_dev input")?;
        self.expect_owner(gamma, "layer_norm_dev gamma")?;
        self.expect_owner(beta, "layer_norm_dev beta")?;
        let total = checked_mul(rows, cols, "layer_norm_dev rows*cols")?;
        checked_u32(rows, "layer_norm_dev rows")?;
        checked_u32(cols, "layer_norm_dev cols")?;
        checked_u32(total, "layer_norm_dev rows*cols")?;
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
        self.expect_owner(dst, "residual_add_dev destination")?;
        self.expect_owner(src, "residual_add_dev source")?;
        expect_len("residual_add_dev src", src.len, dst.len)?;
        checked_u32(dst.len, "residual_add_dev length")?;
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
        self.expect_owner(dst, "mul_dev destination")?;
        self.expect_owner(src, "mul_dev source")?;
        expect_len("mul_dev src", src.len, dst.len)?;
        checked_u32(dst.len, "mul_dev length")?;
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
        self.expect_owner(dst, "copy_dev destination")?;
        self.expect_owner(src, "copy_dev source")?;
        expect_len("copy_dev src", src.len, dst.len)?;
        checked_u32(dst.len, "copy_dev length")?;
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
        self.expect_owner(out, "mlp_dev output")?;
        self.expect_owner(x, "mlp_dev input")?;
        self.expect_owner(fc1_w, "mlp_dev fc1 weight")?;
        self.expect_owner(fc2_w, "mlp_dev fc2 weight")?;
        if let Some(b) = fc1_bias {
            self.expect_owner(b, "mlp_dev fc1 bias")?;
        }
        if let Some(b) = fc2_bias {
            self.expect_owner(b, "mlp_dev fc2 bias")?;
        }
        if t == 0 || d == 0 || ffn == 0 {
            return Err(VokraError::InvalidArgument(
                "mlp_dev dimensions t, d, ffn must all be >= 1".to_owned(),
            ));
        }
        checked_u32(t, "mlp_dev t")?;
        checked_u32(d, "mlp_dev d")?;
        checked_u32(ffn, "mlp_dev ffn")?;
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
        checked_u32(inter, "mlp_dev t*ffn")?;
        checked_u32(checked_mul(t, d, "mlp_dev t*d")?, "mlp_dev t*d")?;
        checked_u32(checked_mul(d, ffn, "mlp_dev d*ffn")?, "mlp_dev d*ffn")?;
        checked_u32(checked_mul(ffn, d, "mlp_dev ffn*d")?, "mlp_dev ffn*d")?;
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
        self.expect_owner(out, "attn_dev output")?;
        self.expect_owner(xq, "attn_dev query")?;
        self.expect_owner(q_w, "attn_dev query weight")?;
        self.expect_owner(k, "attn_dev key")?;
        self.expect_owner(v, "attn_dev value")?;
        self.expect_owner(out_w, "attn_dev output weight")?;
        if let Some(b) = q_bias {
            self.expect_owner(b, "attn_dev query bias")?;
        }
        if let Some(b) = out_bias {
            self.expect_owner(b, "attn_dev output bias")?;
        }
        if t_q == 0 || t_kv == 0 || d == 0 || n_head == 0 {
            return Err(VokraError::InvalidArgument(
                "attn_dev dimensions t_q, t_kv, d, n_head must all be >= 1".to_owned(),
            ));
        }
        checked_u32(t_q, "attn_dev t_q")?;
        checked_u32(t_kv, "attn_dev t_kv")?;
        checked_u32(d, "attn_dev d")?;
        checked_u32(n_head, "attn_dev n_head")?;
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
        for (name, value) in [
            ("attn_dev d*d", dd),
            ("attn_dev t_kv*d", tkvd),
            ("attn_dev t_q*d", tqd),
            ("attn_dev t_q*hd", tq_hd),
            ("attn_dev t_kv*hd", tkv_hd),
            ("attn_dev hd*t_kv", hd_tkv),
            ("attn_dev t_q*t_kv", tq_tkv),
        ] {
            checked_u32(value, name)?;
        }
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
        self.expect_owner(out, "gemm_dev output")?;
        self.expect_owner(a, "gemm_dev lhs")?;
        self.expect_owner(b, "gemm_dev rhs")?;
        if let Some(bs) = bias {
            self.expect_owner(bs, "gemm_dev bias")?;
        }
        if m == 0 || n == 0 || k == 0 {
            return Err(VokraError::InvalidArgument(
                "gemm_dev dimensions m, n, k must all be >= 1".to_owned(),
            ));
        }
        checked_u32(m, "gemm_dev m")?;
        checked_u32(n, "gemm_dev n")?;
        checked_u32(k, "gemm_dev k")?;
        expect_len("gemm_dev a", a.len, checked_mul(m, k, "gemm_dev m*k")?)?;
        expect_len("gemm_dev b", b.len, checked_mul(k, n, "gemm_dev k*n")?)?;
        // The written region ends at row (out_row_offset + m); it must fit `out`.
        let end_rows = out_row_offset.checked_add(m).ok_or_else(|| {
            VokraError::InvalidArgument("gemm_dev row offset overflow".to_owned())
        })?;
        let need = checked_mul(end_rows, n, "gemm_dev (offset+m)*n")?;
        checked_u32(need, "gemm_dev output region")?;
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
    /// `layer_norm` / GEMM / `encode_attn_passes` /
    /// `encode_mlp_passes` / residual-add kernels, in
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
            release(self.qwen3_tts_codec_decode_pipeline);
            release(self.denoise_apply_mask_pipeline);
            release(self.snac_decode_pipeline);
            release(self.hift_istft_pipeline);
            release(self.hift_complex_pipeline);
            release(self.hift_stft_pipeline);
            release(self.nearest_upsample_pipeline);
            // Vocoder Metal wave common vocoder primitives — released after
            // the WF5 codec siblings and before the WF2 snake_activation so
            // the LIFO order matches the construction order in `build`.
            release(self.pad1d_pipeline);
            release(self.conv_transpose1d_pipeline);
            release(self.clamp_pipeline);
            release(self.scale_pipeline);
            release(self.anti_aliased_downsample_pipeline);
            release(self.anti_aliased_upsample_pipeline);
            release(self.sinegen_deterministic_channel_major_pipeline);
            release(self.sinegen_deterministic_pipeline);
            release(self.snake_beta_pipeline);
            release(self.snake_activation_pipeline);
            release(self.xcodec2_fsq_decode_pipeline);
            release(self.wavtokenizer_vq_gather_pipeline);
            release(self.dac_rvq_gather_project_fold_pipeline);
            release(self.mimi_rvq_gather_fold_pipeline);
            release(self.swiglu_pipeline);
            release(self.ouve_annealed_langevin_pipeline);
            release(self.ouve_reverse_diffusion_pipeline);
            release(self.silu_pipeline);
            release(self.rope_adjacent_pipeline);
            release(self.scale_norm_pipeline);
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
            release(self.conv_transpose2d_pipeline);
            release(self.conv2d_pipeline);
            release(self.conv1d_pipeline);
            release(self.leaky_relu_pipeline);
            release(self.tanh_pipeline);
            release(self.linear_tanh_pipeline);
            release(self.linear_abs_pipeline);
            release(self.elu_pipeline);
            release(self.relu_pipeline);
            release(self.gelu_new_pipeline);
            release(self.gelu_pipeline);
            release(self.group_norm_groups_pipeline);
            release(self.group_norm_pipeline);
            release(self.layer_norm_pipeline);
            release(self.softmax_causal_pipeline);
            release(self.softmax_pipeline);
            release(self.gemv_pipeline);
            release(self.gemm_f32_bf16_bits_pipeline);
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
/// The session **owns** its [`MetalContext`] and holds only raw `OwnedBuf`
/// device buffers (no `MetalDeviceTensor<'ctx>`, so no self-referential
/// lifetime). Even though the raw `Id` handles in [`MetalContext`] / `OwnedBuf`
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

fn checked_u32(value: usize, what: &str) -> Result<u32> {
    u32::try_from(value)
        .map_err(|_| VokraError::InvalidArgument(format!("{what}: value exceeds u32")))
}

fn checked_i32(value: usize, what: &str) -> Result<i32> {
    i32::try_from(value)
        .map_err(|_| VokraError::InvalidArgument(format!("{what}: value exceeds i32")))
}

/// Validates every element-count product used by mixed BF16 GEMM.
///
/// The MSL kernel computes row-major indexes as uint; a product that fits
/// usize but exceeds u32::MAX is therefore rejected before allocation or
/// dispatch.
fn validate_mixed_bf16_dims(
    m: usize,
    n: usize,
    k: usize,
    what: &str,
) -> Result<(usize, usize, usize)> {
    checked_u32(m, &format!("{what} m"))?;
    checked_u32(n, &format!("{what} n"))?;
    checked_u32(k, &format!("{what} k"))?;
    let mk = checked_mul(m, k, &format!("{what} m*k"))?;
    let kn = checked_mul(k, n, &format!("{what} k*n"))?;
    let mn = checked_mul(m, n, &format!("{what} m*n"))?;
    checked_u32(mk, &format!("{what} m*k"))?;
    checked_u32(kn, &format!("{what} k*n"))?;
    checked_u32(mn, &format!("{what} m*n"))?;
    Ok((mk, kn, mn))
}

#[cfg(test)]
mod tests {
    use super::{
        checked_i32, checked_u32, validate_conv_transpose2d, validate_conv2d,
        validate_group_norm_groups, validate_mixed_bf16_dims, validate_ouve_host_buffers,
        validate_ouve_params,
    };

    #[test]
    fn device_dimension_conversions_fail_closed() {
        assert!(checked_u32(usize::MAX, "test").is_err());
        assert!(checked_i32(usize::MAX, "test").is_err());
    }

    #[test]
    fn ouve_bounds_reject_invalid_parameters_and_shapes() {
        assert!(validate_ouve_params(1.0, 0.05, 0.5, 0.5, 0.01, None).is_ok());
        assert!(validate_ouve_params(1.0, 0.5, 0.05, 0.5, 0.01, None).is_err());
        assert!(validate_ouve_params(1.0, 0.05, 0.5, 1.1, 0.01, None).is_err());
        assert!(validate_ouve_params(1.0, 0.05, 0.5, 0.5, 0.0, None).is_err());
        assert!(validate_ouve_params(1.0, 0.05, 0.5, 0.5, 1.0, Some(-1.0)).is_err());
        assert!(
            validate_ouve_host_buffers(
                &[0.0, 1.0],
                &[0.0],
                &[0.0, 1.0],
                &[0.0, 1.0],
                &[0.0, 1.0],
                &[0.0, 1.0],
            )
            .is_err()
        );
    }

    #[test]
    fn mixed_bf16_gemm_rejects_msl_u32_product_overflow() {
        let max = u32::MAX as usize;
        assert!(validate_mixed_bf16_dims(1, max, 1, "test").is_ok());
        assert!(validate_mixed_bf16_dims(max, 1, 1, "test").is_ok());
        assert!(validate_mixed_bf16_dims(2, max, 1, "test").is_err());
        assert!(validate_mixed_bf16_dims(max, 1, 2, "test").is_err());
        assert!(validate_mixed_bf16_dims(1, max, 2, "test").is_err());
    }

    #[test]
    fn mixed_bf16_gemm_preserves_zero_k_dimension_semantics() {
        assert_eq!(
            validate_mixed_bf16_dims(2, 3, 0, "test").unwrap(),
            (0, 0, 6)
        );
        assert!(validate_mixed_bf16_dims(u32::MAX as usize, 2, 0, "test").is_err());
    }

    #[test]
    fn group_norm_groups_rejects_msl_u32_index_product_overflow() {
        let max = u32::MAX as usize;
        // Every individual dimension fits uint, but channels*positions does
        // not. Empty buffers are intentional: product validation must happen
        // before host slice-length checks and reject the launch fail-closed.
        assert!(validate_group_norm_groups(&[], &[], 2, max, 2, &[], &[], 1e-6).is_err());
        // The group product is checked independently as well; this case keeps
        // the same boundary visible for the per-group MSL loop.
        assert!(validate_group_norm_groups(&[], &[], 4, max, 2, &[], &[], 1e-6).is_err());
    }

    #[test]
    fn conv2d_validates_grouped_dilated_asymmetric_shape() {
        let input = [0.0; 12];
        let weight = [0.0; 8];
        let out = [0.0; 6];
        assert_eq!(
            validate_conv2d(
                &input,
                2,
                2,
                3,
                &weight,
                2,
                2,
                2,
                Some(&[0.0; 2]),
                (1, 1),
                (0, 1),
                (1, 2),
                2,
                &out,
            )
            .unwrap(),
            (1, 3)
        );
    }

    #[test]
    fn conv_transpose2d_validates_output_padding_and_rejects_bad_shapes() {
        let input = [0.0; 4];
        let weight = [0.0; 8];
        // Output shape follows
        // `(in - 1) * stride + dilation * (kernel - 1) + 1
        //  + output_padding - 2 * padding` per spatial axis:
        // height = 3, width = 6, with two output channels => 36 values.
        let out = [0.0; 36];
        assert_eq!(
            validate_conv_transpose2d(
                &input,
                2,
                1,
                2,
                &weight,
                2,
                2,
                2,
                Some(&[0.0; 2]),
                (2, 2),
                (0, 0),
                (1, 2),
                (1, 1),
                2,
                &out,
            )
            .unwrap(),
            (3, 6)
        );
        // ATen permits output_padding == stride when it remains smaller than
        // dilation on that axis.
        assert_eq!(
            validate_conv_transpose2d(
                &[0.0],
                1,
                1,
                1,
                &[0.0],
                1,
                1,
                1,
                None,
                (1, 1),
                (0, 0),
                (2, 1),
                (1, 0),
                1,
                &[0.0; 2],
            )
            .unwrap(),
            (2, 1)
        );
        assert!(
            validate_conv_transpose2d(
                &input,
                2,
                1,
                2,
                &weight,
                2,
                2,
                2,
                None,
                (2, 2),
                (0, 0),
                (1, 2),
                (2, 0),
                2,
                &out,
            )
            .is_err()
        );
        // Once output_padding reaches both bounds on an axis, reject it.
        assert!(
            validate_conv_transpose2d(
                &[0.0],
                1,
                1,
                1,
                &[0.0],
                1,
                1,
                1,
                None,
                (1, 1),
                (0, 0),
                (2, 1),
                (2, 0),
                1,
                &[0.0; 3],
            )
            .is_err()
        );
        assert!(
            validate_conv2d(
                &input,
                2,
                1,
                2,
                &weight,
                2,
                2,
                2,
                None,
                (1, 1),
                (0, 0),
                (1, 1),
                3,
                &out,
            )
            .is_err()
        );
    }
}

fn validate_ouve_params(
    theta: f32,
    sigma_min: f32,
    sigma_max: f32,
    t: f32,
    step: f32,
    snr: Option<f32>,
) -> Result<()> {
    if !theta.is_finite()
        || !sigma_min.is_finite()
        || !sigma_max.is_finite()
        || theta < 0.0
        || sigma_min <= 0.0
        || sigma_max <= sigma_min
    {
        return Err(VokraError::InvalidArgument(
            "OUVE config requires finite theta >= 0 and 0 < sigma_min < sigma_max".to_owned(),
        ));
    }
    if !t.is_finite() || !(0.0..=1.0).contains(&t) {
        return Err(VokraError::InvalidArgument(
            "OUVE time must be finite and in [0, 1]".to_owned(),
        ));
    }
    if !step.is_finite() || step <= 0.0 {
        return Err(VokraError::InvalidArgument(
            "OUVE reverse step must be finite and positive".to_owned(),
        ));
    }
    if let Some(snr) = snr {
        if !snr.is_finite() || snr < 0.0 {
            return Err(VokraError::InvalidArgument(
                "OUVE annealed Langevin SNR must be finite and non-negative".to_owned(),
            ));
        }
    }
    let ratio = sigma_max / sigma_min;
    let log_ratio = ratio.ln();
    let diffusion_exponent = log_ratio * t;
    let diffusion = sigma_min * diffusion_exponent.exp() * (2.0 * log_ratio).sqrt();
    let score_scale = diffusion * diffusion;
    let noise_variance = score_scale * step;
    if !ratio.is_finite()
        || ratio <= 1.0
        || !log_ratio.is_finite()
        || log_ratio <= 0.0
        || !diffusion_exponent.is_finite()
        || !diffusion.is_finite()
        || !score_scale.is_finite()
        || !noise_variance.is_finite()
    {
        return Err(VokraError::InvalidArgument(
            "OUVE ratio/diffusion/score/noise coefficient is not finite".to_owned(),
        ));
    }
    if let Some(snr) = snr {
        let denominator = theta + log_ratio;
        let decay = -2.0 * theta * t;
        let growth = 2.0 * denominator * t;
        let variance =
            sigma_min * sigma_min * decay.exp() * (growth.exp() - 1.0) * log_ratio / denominator;
        let std = variance.sqrt();
        let scaled_std = snr * std;
        let step_size = 2.0 * scaled_std.powi(2);
        let langevin_noise_variance = 2.0 * step_size;
        if !denominator.is_finite()
            || denominator <= 0.0
            || !decay.is_finite()
            || !growth.is_finite()
            || !variance.is_finite()
            || variance < 0.0
            || !std.is_finite()
            || !scaled_std.is_finite()
            || !step_size.is_finite()
            || !langevin_noise_variance.is_finite()
        {
            return Err(VokraError::InvalidArgument(
                "OUVE variance/score/noise coefficient is not finite".to_owned(),
            ));
        }
    }
    Ok(())
}

fn validate_ouve_host_buffers(
    x: &[f32],
    y: &[f32],
    score: &[f32],
    noise: &[f32],
    out: &[f32],
    out_mean: &[f32],
) -> Result<()> {
    let n = x.len();
    if [y.len(), score.len(), noise.len(), out.len(), out_mean.len()]
        .into_iter()
        .any(|len| len != n)
    {
        return Err(VokraError::InvalidArgument(
            "OUVE predictor buffers must have equal lengths".to_owned(),
        ));
    }
    if x.iter()
        .chain(y)
        .chain(score)
        .chain(noise)
        .any(|value| !value.is_finite())
    {
        return Err(VokraError::InvalidArgument(
            "OUVE predictor buffers must contain finite values".to_owned(),
        ));
    }
    Ok(())
}

fn validate_ouve_host_buffers_unary(
    x: &[f32],
    score: &[f32],
    noise: &[f32],
    out: &[f32],
    out_mean: &[f32],
) -> Result<()> {
    let n = x.len();
    if [score.len(), noise.len(), out.len(), out_mean.len()]
        .into_iter()
        .any(|len| len != n)
    {
        return Err(VokraError::InvalidArgument(
            "OUVE corrector buffers must have equal lengths".to_owned(),
        ));
    }
    if x.iter()
        .chain(score)
        .chain(noise)
        .any(|value| !value.is_finite())
    {
        return Err(VokraError::InvalidArgument(
            "OUVE corrector buffers must contain finite values".to_owned(),
        ));
    }
    Ok(())
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

fn validate_mixed_bf16_host(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[u16],
    out: &[f32],
) -> Result<()> {
    let (mk, kn, mn) = validate_mixed_bf16_dims(m, n, k, "gemm_f32_bf16_bits")?;
    expect_len("gemm_f32_bf16_bits activation", a.len(), mk)?;
    expect_len("gemm_f32_bf16_bits weight", b.len(), kn)?;
    expect_len("gemm_f32_bf16_bits output", out.len(), mn)
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

fn validate_group_norm(
    input: &[f32],
    out: &[f32],
    channels: usize,
    positions: usize,
    gamma: &[f32],
    beta: &[f32],
) -> Result<()> {
    if channels == 0 || positions == 0 {
        return Err(VokraError::InvalidArgument(format!(
            "group_norm channels and positions must be non-zero, got {channels}x{positions}"
        )));
    }
    let total = checked_mul(channels, positions, "group_norm channels*positions")?;
    expect_len("group_norm input", input.len(), total)?;
    expect_len("group_norm out", out.len(), total)?;
    expect_len("group_norm gamma", gamma.len(), channels)?;
    expect_len("group_norm beta", beta.len(), channels)
}

fn validate_group_norm_groups(
    input: &[f32],
    out: &[f32],
    channels: usize,
    positions: usize,
    groups: usize,
    gamma: &[f32],
    beta: &[f32],
    eps: f32,
) -> Result<()> {
    if channels == 0 || positions == 0 || groups == 0 || channels % groups != 0 {
        return Err(VokraError::InvalidArgument(format!(
            "group_norm_groups requires non-zero channels/positions, positive groups dividing channels; got {channels}x{positions}, groups={groups}"
        )));
    }
    if !eps.is_finite() || eps <= 0.0 {
        return Err(VokraError::InvalidArgument(
            "group_norm_groups eps must be finite and positive".to_owned(),
        ));
    }
    checked_u32(channels, "group_norm_groups channels")?;
    checked_u32(positions, "group_norm_groups positions")?;
    checked_u32(groups, "group_norm_groups groups")?;
    let total = checked_mul(channels, positions, "group_norm_groups channels*positions")?;
    // MSL computes both products with uint indexing. Checking dimensions
    // independently is insufficient: a pair of valid u32 dimensions can
    // still overflow the product used by the kernel.
    let channels_per_group = channels / groups;
    let group_total = checked_mul(
        channels_per_group,
        positions,
        "group_norm_groups channels_per_group*positions",
    )?;
    checked_u32(
        group_total,
        "group_norm_groups channels_per_group*positions indexing",
    )?;
    checked_u32(total, "group_norm_groups channels*positions indexing")?;
    expect_len("group_norm_groups input", input.len(), total)?;
    expect_len("group_norm_groups out", out.len(), total)?;
    expect_len("group_norm_groups gamma", gamma.len(), channels)?;
    expect_len("group_norm_groups beta", beta.len(), channels)?;
    if input
        .iter()
        .chain(gamma)
        .chain(beta)
        .any(|value| !value.is_finite())
    {
        return Err(VokraError::InvalidArgument(
            "group_norm_groups input and affine parameters must be finite".to_owned(),
        ));
    }
    Ok(())
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

fn validate_scale_norm(
    input: &[f32],
    out: &[f32],
    rows: usize,
    cols: usize,
    gain: f32,
    eps: f32,
) -> Result<()> {
    if rows == 0 || cols == 0 {
        return Err(VokraError::InvalidArgument(format!(
            "scale_norm rows and cols must be non-zero, got {rows}x{cols}"
        )));
    }
    if !gain.is_finite() || !eps.is_finite() || eps <= 0.0 {
        return Err(VokraError::InvalidArgument(format!(
            "scale_norm gain must be finite and eps positive, got gain={gain}, eps={eps}"
        )));
    }
    validate_rows_cols(input, out, rows, cols)
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

/// Validates the dense/grouped Conv2d host contract and returns `(out_h, out_w)`.
#[allow(clippy::too_many_arguments)]
fn validate_conv2d(
    input: &[f32],
    in_ch: usize,
    in_h: usize,
    in_w: usize,
    weight: &[f32],
    out_ch: usize,
    kernel_h: usize,
    kernel_w: usize,
    bias: Option<&[f32]>,
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
    groups: usize,
    out: &[f32],
) -> Result<(usize, usize)> {
    let (stride_h, stride_w) = stride;
    let (padding_h, padding_w) = padding;
    let (dilation_h, dilation_w) = dilation;
    if in_ch == 0 || in_h == 0 || in_w == 0 || out_ch == 0 || kernel_h == 0 || kernel_w == 0 {
        return Err(VokraError::InvalidArgument(
            "conv2d channels, spatial dimensions, and kernel dimensions must be > 0".into(),
        ));
    }
    if groups == 0 || in_ch % groups != 0 || out_ch % groups != 0 {
        return Err(VokraError::InvalidArgument(
            "conv2d groups must be > 0 and divide both channel counts".into(),
        ));
    }
    if stride_h == 0 || stride_w == 0 || dilation_h == 0 || dilation_w == 0 {
        return Err(VokraError::InvalidArgument(
            "conv2d stride and dilation dimensions must be > 0".into(),
        ));
    }
    let effective_h = checked_mul(kernel_h - 1, dilation_h, "conv2d effective kernel height")?
        .checked_add(1)
        .ok_or_else(|| {
            VokraError::InvalidArgument("conv2d effective kernel height overflow".into())
        })?;
    let effective_w = checked_mul(kernel_w - 1, dilation_w, "conv2d effective kernel width")?
        .checked_add(1)
        .ok_or_else(|| {
            VokraError::InvalidArgument("conv2d effective kernel width overflow".into())
        })?;
    let padded_h = checked_mul(2, padding_h, "conv2d 2*padding_h")?
        .checked_add(in_h)
        .ok_or_else(|| VokraError::InvalidArgument("conv2d padded height overflow".into()))?;
    let padded_w = checked_mul(2, padding_w, "conv2d 2*padding_w")?
        .checked_add(in_w)
        .ok_or_else(|| VokraError::InvalidArgument("conv2d padded width overflow".into()))?;
    if padded_h < effective_h || padded_w < effective_w {
        return Err(VokraError::InvalidArgument(
            "conv2d padded input is smaller than the effective kernel".into(),
        ));
    }
    // The MSL shader performs the padded extent and effective-kernel
    // additions/multiplications in `uint`; reject any valid-`usize` shape
    // whose intermediate arithmetic would wrap on the device.
    checked_u32(effective_h, "conv2d effective kernel height")?;
    checked_u32(effective_w, "conv2d effective kernel width")?;
    checked_u32(padded_h, "conv2d padded height")?;
    checked_u32(padded_w, "conv2d padded width")?;
    let out_h = (padded_h - effective_h) / stride_h + 1;
    let out_w = (padded_w - effective_w) / stride_w + 1;
    checked_u32(in_ch, "conv2d in_ch")?;
    checked_u32(in_h, "conv2d in_h")?;
    checked_u32(in_w, "conv2d in_w")?;
    checked_u32(out_ch, "conv2d out_ch")?;
    checked_u32(out_h, "conv2d out_h")?;
    checked_u32(out_w, "conv2d out_w")?;
    checked_u32(kernel_h, "conv2d kernel_h")?;
    checked_u32(kernel_w, "conv2d kernel_w")?;
    checked_u32(stride_h, "conv2d stride_h")?;
    checked_u32(stride_w, "conv2d stride_w")?;
    checked_u32(padding_h, "conv2d padding_h")?;
    checked_u32(padding_w, "conv2d padding_w")?;
    checked_u32(dilation_h, "conv2d dilation_h")?;
    checked_u32(dilation_w, "conv2d dilation_w")?;
    let input_plane = checked_mul(in_h, in_w, "conv2d input plane")?;
    let output_plane = checked_mul(out_h, out_w, "conv2d output plane")?;
    let kernel_plane = checked_mul(kernel_h, kernel_w, "conv2d kernel plane")?;
    checked_u32(input_plane, "conv2d input plane")?;
    checked_u32(output_plane, "conv2d output plane")?;
    checked_u32(kernel_plane, "conv2d kernel plane")?;
    let input_len = checked_mul(in_ch, input_plane, "conv2d input")?;
    let weight_per_output = checked_mul(in_ch / groups, kernel_plane, "conv2d weight per output")?;
    let weight_len = checked_mul(out_ch, weight_per_output, "conv2d weight")?;
    let output_len = checked_mul(out_ch, output_plane, "conv2d output")?;
    checked_u32(input_len, "conv2d input")?;
    checked_u32(weight_len, "conv2d weight")?;
    checked_u32(output_len, "conv2d output")?;
    checked_u32(
        checked_mul(out_ch, out_h, "conv2d dispatch rows")?,
        "conv2d dispatch rows",
    )?;
    expect_len("conv2d input", input.len(), input_len)?;
    expect_len("conv2d weight", weight.len(), weight_len)?;
    expect_len("conv2d out", out.len(), output_len)?;
    if let Some(bias) = bias {
        expect_len("conv2d bias", bias.len(), out_ch)?;
    }
    Ok((out_h, out_w))
}

/// Validates PyTorch-layout dense/grouped ConvTranspose2d and returns output
/// spatial dimensions. Following PyTorch/ATen, output padding must be less
/// than either stride or dilation on each axis before any device work is
/// allocated.
#[allow(clippy::too_many_arguments)]
fn validate_conv_transpose2d(
    input: &[f32],
    in_ch: usize,
    in_h: usize,
    in_w: usize,
    weight: &[f32],
    out_ch: usize,
    kernel_h: usize,
    kernel_w: usize,
    bias: Option<&[f32]>,
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
    output_padding: (usize, usize),
    groups: usize,
    out: &[f32],
) -> Result<(usize, usize)> {
    let (stride_h, stride_w) = stride;
    let (padding_h, padding_w) = padding;
    let (dilation_h, dilation_w) = dilation;
    let (output_padding_h, output_padding_w) = output_padding;
    if in_ch == 0 || in_h == 0 || in_w == 0 || out_ch == 0 || kernel_h == 0 || kernel_w == 0 {
        return Err(VokraError::InvalidArgument(
            "conv_transpose2d channels, spatial dimensions, and kernel dimensions must be > 0"
                .into(),
        ));
    }
    if groups == 0 || in_ch % groups != 0 || out_ch % groups != 0 {
        return Err(VokraError::InvalidArgument(
            "conv_transpose2d groups must be > 0 and divide both channel counts".into(),
        ));
    }
    if stride_h == 0 || stride_w == 0 || dilation_h == 0 || dilation_w == 0 {
        return Err(VokraError::InvalidArgument(
            "conv_transpose2d stride and dilation dimensions must be > 0".into(),
        ));
    }
    if (output_padding_h >= stride_h && output_padding_h >= dilation_h)
        || (output_padding_w >= stride_w && output_padding_w >= dilation_w)
    {
        return Err(VokraError::InvalidArgument(
            "conv_transpose2d output_padding must be smaller than either stride or dilation on each axis".into(),
        ));
    }
    let effective_h = checked_mul(
        kernel_h - 1,
        dilation_h,
        "conv_transpose2d effective kernel height",
    )?
    .checked_add(1)
    .ok_or_else(|| {
        VokraError::InvalidArgument("conv_transpose2d effective kernel height overflow".into())
    })?;
    let effective_w = checked_mul(
        kernel_w - 1,
        dilation_w,
        "conv_transpose2d effective kernel width",
    )?
    .checked_add(1)
    .ok_or_else(|| {
        VokraError::InvalidArgument("conv_transpose2d effective kernel width overflow".into())
    })?;
    let base_h = checked_mul(in_h - 1, stride_h, "conv_transpose2d base height")?
        .checked_add(effective_h)
        .and_then(|value| value.checked_add(output_padding_h))
        .ok_or_else(|| {
            VokraError::InvalidArgument("conv_transpose2d base height overflow".into())
        })?;
    let base_w = checked_mul(in_w - 1, stride_w, "conv_transpose2d base width")?
        .checked_add(effective_w)
        .and_then(|value| value.checked_add(output_padding_w))
        .ok_or_else(|| {
            VokraError::InvalidArgument("conv_transpose2d base width overflow".into())
        })?;
    let trim_h = checked_mul(2, padding_h, "conv_transpose2d 2*padding_h")?;
    let trim_w = checked_mul(2, padding_w, "conv_transpose2d 2*padding_w")?;
    if trim_h >= base_h || trim_w >= base_w {
        return Err(VokraError::InvalidArgument(
            "conv_transpose2d padding removes the complete output extent".into(),
        ));
    }
    // The MSL inverse-index calculation forms `output + padding` and
    // `kernel_tap * dilation` in `uint`; keep the complete pre-trim extent
    // representable so those intermediate expressions cannot wrap.
    checked_u32(effective_h, "conv_transpose2d effective kernel height")?;
    checked_u32(effective_w, "conv_transpose2d effective kernel width")?;
    checked_u32(base_h, "conv_transpose2d base height")?;
    checked_u32(base_w, "conv_transpose2d base width")?;
    let out_h = base_h - trim_h;
    let out_w = base_w - trim_w;
    checked_u32(in_ch, "conv_transpose2d in_ch")?;
    checked_u32(in_h, "conv_transpose2d in_h")?;
    checked_u32(in_w, "conv_transpose2d in_w")?;
    checked_u32(out_ch, "conv_transpose2d out_ch")?;
    checked_u32(out_h, "conv_transpose2d out_h")?;
    checked_u32(out_w, "conv_transpose2d out_w")?;
    checked_u32(kernel_h, "conv_transpose2d kernel_h")?;
    checked_u32(kernel_w, "conv_transpose2d kernel_w")?;
    checked_u32(stride_h, "conv_transpose2d stride_h")?;
    checked_u32(stride_w, "conv_transpose2d stride_w")?;
    checked_u32(padding_h, "conv_transpose2d padding_h")?;
    checked_u32(padding_w, "conv_transpose2d padding_w")?;
    checked_u32(dilation_h, "conv_transpose2d dilation_h")?;
    checked_u32(dilation_w, "conv_transpose2d dilation_w")?;
    checked_u32(output_padding_h, "conv_transpose2d output_padding_h")?;
    checked_u32(output_padding_w, "conv_transpose2d output_padding_w")?;
    let input_plane = checked_mul(in_h, in_w, "conv_transpose2d input plane")?;
    let output_plane = checked_mul(out_h, out_w, "conv_transpose2d output plane")?;
    let kernel_plane = checked_mul(kernel_h, kernel_w, "conv_transpose2d kernel plane")?;
    checked_u32(input_plane, "conv_transpose2d input plane")?;
    checked_u32(output_plane, "conv_transpose2d output plane")?;
    checked_u32(kernel_plane, "conv_transpose2d kernel plane")?;
    let input_len = checked_mul(in_ch, input_plane, "conv_transpose2d input")?;
    let weight_per_input = checked_mul(
        out_ch / groups,
        kernel_plane,
        "conv_transpose2d weight per input",
    )?;
    let weight_len = checked_mul(in_ch, weight_per_input, "conv_transpose2d weight")?;
    let output_len = checked_mul(out_ch, output_plane, "conv_transpose2d output")?;
    checked_u32(input_len, "conv_transpose2d input")?;
    checked_u32(weight_len, "conv_transpose2d weight")?;
    checked_u32(output_len, "conv_transpose2d output")?;
    checked_u32(
        checked_mul(out_ch, out_h, "conv_transpose2d dispatch rows")?,
        "conv_transpose2d dispatch rows",
    )?;
    expect_len("conv_transpose2d input", input.len(), input_len)?;
    expect_len("conv_transpose2d weight", weight.len(), weight_len)?;
    expect_len("conv_transpose2d out", out.len(), output_len)?;
    if let Some(bias) = bias {
        expect_len("conv_transpose2d bias", bias.len(), out_ch)?;
    }
    Ok((out_h, out_w))
}

/// Validates the stride/dilation Conv1d host wrapper and returns its exact
/// output length.  This mirrors `MetalContext::conv1d_dev` and the CPU
/// `conv1d_f32_dilated` contract so invalid dimensions fail before any buffer
/// allocation or dispatch.
#[allow(clippy::too_many_arguments)]
fn validate_conv1d_dilated(
    input: &[f32],
    in_ch: usize,
    in_len: usize,
    weight: &[f32],
    out_ch: usize,
    kernel: usize,
    bias: Option<&[f32]>,
    stride: usize,
    dilation: usize,
    padding: usize,
    out: &[f32],
) -> Result<usize> {
    if dilation == 0 {
        return Err(VokraError::InvalidArgument(
            "conv1d dilation must be >= 1".to_owned(),
        ));
    }
    if in_ch == 0 || out_ch == 0 || in_len == 0 || stride == 0 || kernel == 0 {
        return Err(VokraError::InvalidArgument(
            "conv1d in_ch, out_ch, in_len, stride, and kernel must be > 0".to_owned(),
        ));
    }
    let effective = checked_mul(kernel - 1, dilation, "conv1d effective kernel")?
        .checked_add(1)
        .ok_or_else(|| VokraError::InvalidArgument("conv1d effective kernel overflow".into()))?;
    let padded = in_len
        .checked_add(checked_mul(2, padding, "conv1d 2*padding")?)
        .ok_or_else(|| VokraError::InvalidArgument("conv1d padded length overflow".into()))?;
    if padded < effective {
        return Err(VokraError::InvalidArgument(format!(
            "conv1d padded length {padded} is smaller than effective kernel {effective}"
        )));
    }
    let out_len = (padded - effective) / stride + 1;
    checked_u32(in_ch, "conv1d in_ch")?;
    checked_u32(in_len, "conv1d in_len")?;
    checked_u32(out_ch, "conv1d out_ch")?;
    checked_u32(kernel, "conv1d kernel")?;
    checked_u32(stride, "conv1d stride")?;
    checked_u32(dilation, "conv1d dilation")?;
    checked_u32(padding, "conv1d padding")?;
    checked_u32(padded, "conv1d padded length")?;
    checked_u32(out_len, "conv1d out_len")?;
    expect_len(
        "conv1d input",
        input.len(),
        checked_mul(in_ch, in_len, "conv1d in_ch*in_len")?,
    )?;
    expect_len(
        "conv1d weight",
        weight.len(),
        checked_mul(
            checked_mul(out_ch, in_ch, "conv1d out_ch*in_ch")?,
            kernel,
            "conv1d out_ch*in_ch*kernel",
        )?,
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

/// Validates a host-facing PyTorch-layout ConvTranspose1d and returns its
/// exact output length.  `output_padding` is explicit and must be smaller than
/// `stride`, matching the Metal device kernel.
#[allow(clippy::too_many_arguments)]
fn validate_conv_transpose1d(
    input: &[f32],
    in_ch: usize,
    in_len: usize,
    weight: &[f32],
    out_ch: usize,
    kernel: usize,
    bias: Option<&[f32]>,
    stride: usize,
    padding: usize,
    output_padding: usize,
    out: &[f32],
) -> Result<usize> {
    if in_ch == 0 || out_ch == 0 || in_len == 0 || stride == 0 || kernel == 0 {
        return Err(VokraError::InvalidArgument(
            "conv_transpose1d in_ch, out_ch, in_len, stride, and kernel must be > 0".into(),
        ));
    }
    if output_padding >= stride {
        return Err(VokraError::InvalidArgument(format!(
            "conv_transpose1d output_padding {output_padding} must be < stride {stride}"
        )));
    }
    expect_len(
        "conv_transpose1d input",
        input.len(),
        checked_mul(in_ch, in_len, "conv_transpose1d in_ch*in_len")?,
    )?;
    expect_len(
        "conv_transpose1d weight",
        weight.len(),
        checked_mul(
            checked_mul(in_ch, out_ch, "conv_transpose1d in_ch*out_ch")?,
            kernel,
            "conv_transpose1d weight",
        )?,
    )?;
    if let Some(bias) = bias {
        expect_len("conv_transpose1d bias", bias.len(), out_ch)?;
    }
    let full_out = checked_mul(in_len - 1, stride, "conv_transpose1d output")?
        .checked_add(kernel)
        .and_then(|value| value.checked_add(output_padding))
        .ok_or_else(|| {
            VokraError::InvalidArgument("conv_transpose1d output length overflow".into())
        })?;
    let trim = checked_mul(2, padding, "conv_transpose1d padding")?;
    if trim >= full_out {
        return Err(VokraError::InvalidArgument(format!(
            "conv_transpose1d padding {trim} removes the complete output extent {full_out}"
        )));
    }
    let out_len = full_out - trim;
    checked_u32(in_ch, "conv_transpose1d in_ch")?;
    checked_u32(in_len, "conv_transpose1d in_len")?;
    checked_u32(out_ch, "conv_transpose1d out_ch")?;
    checked_u32(kernel, "conv_transpose1d kernel")?;
    checked_u32(stride, "conv_transpose1d stride")?;
    checked_u32(padding, "conv_transpose1d padding")?;
    checked_u32(output_padding, "conv_transpose1d output_padding")?;
    checked_u32(out_len, "conv_transpose1d out_len")?;
    expect_len(
        "conv_transpose1d out",
        out.len(),
        checked_mul(out_ch, out_len, "conv_transpose1d out_ch*out_len")?,
    )?;
    Ok(out_len)
}

/// Validates the grouped-convolution extension of [`validate_conv1d`].
#[allow(clippy::too_many_arguments)]
fn validate_grouped_conv1d(
    input: &[f32],
    in_ch: usize,
    in_len: usize,
    weight: &[f32],
    out_ch: usize,
    kernel: usize,
    bias: Option<&[f32]>,
    stride: usize,
    padding: usize,
    groups: usize,
    out: &[f32],
) -> Result<usize> {
    if groups == 0 {
        return Err(VokraError::InvalidArgument(
            "grouped_conv1d groups must be >= 1".to_owned(),
        ));
    }
    if in_ch % groups != 0 || out_ch % groups != 0 {
        return Err(VokraError::InvalidArgument(format!(
            "grouped_conv1d in_ch {in_ch} and out_ch {out_ch} must both be divisible by groups {groups}"
        )));
    }
    if stride == 0 {
        return Err(VokraError::InvalidArgument(
            "grouped_conv1d stride must be >= 1".to_owned(),
        ));
    }
    if kernel == 0 {
        return Err(VokraError::InvalidArgument(
            "grouped_conv1d kernel must be >= 1".to_owned(),
        ));
    }
    let padded = in_len
        .checked_add(checked_mul(2, padding, "grouped_conv1d 2*padding")?)
        .ok_or_else(|| {
            VokraError::InvalidArgument("grouped_conv1d padded length overflow".to_owned())
        })?;
    if padded < kernel {
        return Err(VokraError::InvalidArgument(format!(
            "grouped_conv1d padded length {padded} is smaller than kernel {kernel}"
        )));
    }
    let out_len = (padded - kernel) / stride + 1;
    expect_len(
        "grouped_conv1d input",
        input.len(),
        checked_mul(in_ch, in_len, "grouped_conv1d in_ch*in_len")?,
    )?;
    let in_per = in_ch / groups;
    let k = checked_mul(in_per, kernel, "grouped_conv1d in_per*kernel")?;
    expect_len(
        "grouped_conv1d weight",
        weight.len(),
        checked_mul(out_ch, k, "grouped_conv1d out_ch*k")?,
    )?;
    expect_len(
        "grouped_conv1d out",
        out.len(),
        checked_mul(out_ch, out_len, "grouped_conv1d out_ch*out_len")?,
    )?;
    if let Some(bias) = bias {
        expect_len("grouped_conv1d bias", bias.len(), out_ch)?;
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
