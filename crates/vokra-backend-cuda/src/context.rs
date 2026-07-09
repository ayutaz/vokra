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
use core::ffi::{c_char, c_int, c_uint, c_void};
use core::marker::PhantomData;

use vokra_core::{DecoderLayerView, PrenormLayer, Result, VokraError};

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
// NVRTC does not include <math.h> by default, so `INFINITY` from ISO C is not
// visible. Define it once here using the IEEE 754 bit pattern for +∞ (identical
// to `__int_as_float(0x7f800000)`; kept as a straight `#define` so the value is
// a constant expression usable in `-INFINITY` initializers). Fixes the
// "identifier "INFINITY" is undefined" NVRTC error observed on CUDA 12.6.
#ifndef INFINITY
#define INFINITY __int_as_float(0x7f800000)
#endif

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

// ---- softmax_causal: row-wise softmax over the causally-visible key prefix ---
// The decoder self-attention mask, fused into the softmax so the causal decode
// step needs no separate mask write. Row `r` (query at absolute position
// `q_offset + r`) attends keys `[0, q_offset + r]`; keys beyond that are the
// "future" the causal mask hides. This is BIT-IDENTICAL to writing -INF into
// scores[r, j>last] and running the plain softmax above (same IEEE-754 argument
// as the Metal `vokra_softmax_causal_f32` kernel: masked columns contribute
// exp(-INF - m) = 0, and adding 0.0 leaves the accumulator unchanged; masked
// output columns get exactly 0.0 as `0 * inv`).
// For a single new token (t_q = 1) `last = q_offset = t_kv - 1`, so ALL keys are
// visible and this is the plain softmax bit-for-bit; the mask only bites on the
// multi-token prefix step (t_q > 1).
extern "C" __global__ void vokra_softmax_causal_f32(
    const float* inp,
    float* out,
    unsigned int rows,
    unsigned int cols,
    unsigned int q_offset)
{
    unsigned int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) {
        return;
    }
    unsigned int base = r * cols;
    // Last visible key column for this row (clamped; the caller guarantees
    // last < cols, so the clamp is defensive only).
    unsigned int last = q_offset + r;
    if (last >= cols) {
        last = cols - 1u;
    }
    float m = inp[base]; // column 0 is always visible (0 <= q_offset + r)
    for (unsigned int j = 1u; j <= last; ++j) {
        m = fmaxf(m, inp[base + j]);
    }
    float sum = 0.0f;
    for (unsigned int j = 0u; j <= last; ++j) {
        float e = expf(inp[base + j] - m);
        out[base + j] = e;
        sum += e;
    }
    float inv = 1.0f / sum;
    for (unsigned int j = 0u; j <= last; ++j) {
        out[base + j] *= inv;
    }
    for (unsigned int j = last + 1u; j < cols; ++j) {
        out[base + j] = 0.0f; // future keys -> 0 (exactly as the host mask does)
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

// ---- M3-01 graph-executor element-wise Add / Mul (out = a op b) --------------
// Distinct from `vokra_add_assign_f32` (in-place residual, used by the encoder
// device-resident path): these are OUT-OF-PLACE element-wise kernels backing the
// graph-level `OpKind::Add` and `OpKind::Mul` on the CUDA arm of
// `crate::eval::eval_cuda_op`. Bit-identical to
// `vokra_backend_cpu::kernels::{add_f32, mul_f32}` (the differential oracle at
// FP32 `atol = 0.01`, NFR-QL-01). One thread per element, ragged-tail guarded.
extern "C" __global__ void vokra_add_f32(
    const float* a,
    const float* b,
    float* out,
    unsigned int n)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) {
        return;
    }
    out[gid] = a[gid] + b[gid];
}

extern "C" __global__ void vokra_mul_f32(
    const float* a,
    const float* b,
    float* out,
    unsigned int n)
{
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) {
        return;
    }
    out[gid] = a[gid] * b[gid];
}

// ---- FlashAttention-v2 causal, FP32 (M2-03 follow-up RTF<0.1) ---------------
//
// Fused (Q·Kᵀ · softmax · P·V) attention kernel. Semantics — including causal
// mask via `q_offset` vs key column index, softmax normalisation, and the
// `scale` factor folded into the scores — are the exact fp32 contract of the
// decomposed `launch_attn_chain` path (`gemm + softmax_causal + gemm`) up to
// online-softmax rescale round-off (NFR-QL-01 atol=0.01).
//
// Layout: row-major Q[t_q, d_head], K[t_kv, d_head], V[t_kv, d_head], out
// O[t_q, d_head]. One CUDA block covers Br query rows; the block streams the
// key/value dimension in Bc-wide tiles, keeping the running max `m_i`, the
// running exp-sum `l_i`, and the running output accumulator `O_i` per query
// row (all in fp32 register space — the log-sum-exp trick keeps everything
// numerically stable for t_kv up to Whisper large-v3's 1500).
//
// Tile sizes: Br=16, Bc=64, 128 threads/block (4 warps). Shared memory:
//   Q_tile[Br*d_head]  + KV_tile[Bc*d_head*2] + S_tile[Br*Bc]
// = for d_head=64: 16·64·4 + 64·64·2·4 + 16·64·4 = 4096 + 32768 + 4096
// = 40 960 B ≈ 40 KB, inside the 48 KB / SM Ampere/Ada baseline.
//
// Decoder-step specialisation: when t_q==1, only the first thread's row of Q
// is loaded / used; the same recurrence still yields the correct result but
// operates on Br=1 effective queries (shared memory drops to ~34 KB, register
// pressure eases). We keep a single kernel + a compile-time template branch
// on `(t_q > 1)` via a runtime `if` — no separate entry point, so
// `cuModuleGetFunction` still resolves one symbol.
//
// Grid: (⌈t_q / Br⌉, n_head, 1) — n_head via grid.y also solves the
// inter-head-overlap follow-up (O3) with no extra launches.
extern "C" __global__ void vokra_flash_attn_v2_causal_f32(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int t_q,
    int t_kv,
    int d_head,
    int q_offset,
    bool causal,
    float scale)
{
    // Block tile sizes (must match the host launcher).
    const int BR = 16;
    const int BC = 64;

    // Which head + which Br-tile of queries are we processing?
    // grid.y = n_head → Q/K/V/O for head h live at [h * t_* * d_head + ...].
    // The host launcher passes head-relative pointers, so we treat everything
    // as a single head here. That keeps the kernel signature small and lets
    // the launcher advance the base pointers by `h * stride` on the host.

    int q_tile = blockIdx.x;          // 0 .. ⌈t_q / BR⌉ - 1
    int q_row_base = q_tile * BR;     // first query row this block owns
    int tid = threadIdx.x;            // 0 .. 127

    // Effective Br for the last (possibly ragged) query tile.
    int br_eff = (q_row_base + BR <= t_q) ? BR : (t_q - q_row_base);
    if (br_eff <= 0) {
        return;
    }

    extern __shared__ float smem[];
    // Layout inside smem:
    //   Q_tile: BR * d_head          (queries for this block)
    //   K_tile: BC * d_head          (current key tile)
    //   V_tile: BC * d_head          (current value tile)
    //   S_tile: BR * BC              (scores for the current (q_tile, k_tile))
    float* Q_tile = smem;
    float* K_tile = Q_tile + BR * d_head;
    float* V_tile = K_tile + BC * d_head;
    float* S_tile = V_tile + BC * d_head;

    // --- Load Q_tile (once per block). Each thread strides d_head-per-row. ----
    // We only need br_eff rows; unused rows are left uninitialised (S_tile
    // guards against them via the br_eff check below).
    for (int idx = tid; idx < br_eff * d_head; idx += blockDim.x) {
        int r = idx / d_head;
        int c = idx - r * d_head;
        Q_tile[r * d_head + c] = Q[(q_row_base + r) * d_head + c];
    }
    __syncthreads();

    // Per-thread running softmax state, one entry per Br row. We use one
    // thread per row for the reduction; extra threads keep O/L/M in
    // shared-memory-parallel form via the register file (kept per row).
    // For simplicity we let thread `r` (0 <= r < br_eff) own row r's state.
    // Threads with tid >= br_eff still participate in the tile GEMMs.
    float m_i = -INFINITY;
    float l_i = 0.0f;
    // Per-row output accumulator, stored in registers (d_head <= 64 in Whisper).
    // We cap at 128 to give a compile-time bound; d_head values above 128 are
    // rejected by the host launcher (FR-EX-08).
    float O_i[128];
    if (tid < br_eff) {
        for (int c = 0; c < d_head; ++c) {
            O_i[c] = 0.0f;
        }
    }

    // Iterate over K/V tiles of width BC along the key dimension.
    int n_kv_tiles = (t_kv + BC - 1) / BC;
    for (int kt = 0; kt < n_kv_tiles; ++kt) {
        int k_col_base = kt * BC;
        int bc_eff = (k_col_base + BC <= t_kv) ? BC : (t_kv - k_col_base);

        // --- Load K_tile + V_tile cooperatively. -----------------------------
        for (int idx = tid; idx < bc_eff * d_head; idx += blockDim.x) {
            int r = idx / d_head;
            int c = idx - r * d_head;
            K_tile[r * d_head + c] = K[(k_col_base + r) * d_head + c];
            V_tile[r * d_head + c] = V[(k_col_base + r) * d_head + c];
        }
        __syncthreads();

        // --- S_tile = Q_tile · K_tileᵀ · scale (BR × BC) --------------------
        // Thread layout: 128 threads → one thread per (row, col) pair for the
        // dense BR·BC=1024 output. We stride over the tile so that any tile
        // size is handled.
        for (int idx = tid; idx < br_eff * BC; idx += blockDim.x) {
            int r = idx / BC;
            int c = idx - r * BC;
            if (c >= bc_eff) {
                S_tile[r * BC + c] = -INFINITY;
                continue;
            }
            float acc = 0.0f;
            const float* q_row = Q_tile + r * d_head;
            const float* k_row = K_tile + c * d_head;
            for (int d = 0; d < d_head; ++d) {
                acc += q_row[d] * k_row[d];
            }
            acc *= scale;
            if (causal) {
                int q_abs = q_offset + (q_row_base + r);
                int k_abs = k_col_base + c;
                if (k_abs > q_abs) {
                    acc = -INFINITY;
                }
            }
            S_tile[r * BC + c] = acc;
        }
        __syncthreads();

        // --- Online softmax rescale + O accumulation (per-row, thread-r) ----
        if (tid < br_eff) {
            // Row-local max over this tile.
            float m_tile = -INFINITY;
            const float* s_row = S_tile + tid * BC;
            for (int c = 0; c < bc_eff; ++c) {
                float s = s_row[c];
                if (s > m_tile) {
                    m_tile = s;
                }
            }
            // New running max (log-sum-exp trick keeps subsequent expf in
            // range even for very negative scores).
            float m_new = (m_i > m_tile) ? m_i : m_tile;
            // Scale factor to bring the old accumulator to the new max.
            float alpha = (m_i == -INFINITY) ? 0.0f : expf(m_i - m_new);
            // Row-local exp-sum with the new max subtracted.
            float l_tile = 0.0f;
            for (int c = 0; c < bc_eff; ++c) {
                float p = (s_row[c] == -INFINITY) ? 0.0f : expf(s_row[c] - m_new);
                // Reuse s_row as P_tile (it is not read again this tile).
                S_tile[tid * BC + c] = p;
                l_tile += p;
            }
            float l_new = l_i * alpha + l_tile;

            // O_i = O_i * alpha + P_tile · V_tile   (per row).
            for (int d = 0; d < d_head; ++d) {
                float acc = O_i[d] * alpha;
                for (int c = 0; c < bc_eff; ++c) {
                    acc += S_tile[tid * BC + c] * V_tile[c * d_head + d];
                }
                O_i[d] = acc;
            }
            m_i = m_new;
            l_i = l_new;
        }
        __syncthreads();
    }

    // --- Write final O = O_i / l_i to global memory ------------------------
    if (tid < br_eff) {
        float inv_l = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
        for (int d = 0; d < d_head; ++d) {
            O[(q_row_base + tid) * d_head + d] = O_i[d] * inv_l;
        }
    }
}

// ---- M3-04 fused KV-cache dequant + GEMV kernels ----------------------------
//
// One thread per output row. Each block of 32 quantised values is dequantised
// in-register (no shared / global scratch) and directly multiplied against 32
// entries of the query vector `x` — the "fused" property. Byte layout mirrors
// `vokra_core::kv_quant::dequantize_bytes` exactly (Q4_0 = 18 B, Q5_0 = 22 B,
// Q8_0 = 34 B), so the same on-wire block payload feeds the CPU differential
// oracle (`dequant_gemv_scalar`) and this GPU kernel.
//
// FP16 → FP32 for the block scale `d` is done inline here to avoid pulling in
// `<cuda_fp16.h>` (kept out of the NVRTC compile keeps the compile hermetic).
// The semantics match the CPU `vokra_core::kv_quant::half::f16_bits_to_f32`
// helper verbatim; the small helper below is the same shape.
__device__ float vokra_kv_f16_to_f32(unsigned short h) {
    unsigned int sign = (h >> 15) & 1u;
    unsigned int exp  = (h >> 10) & 0x1Fu;
    unsigned int mant = h & 0x3FFu;
    float sign_f = (sign == 1u) ? -1.0f : 1.0f;
    if (exp == 0u) {
        // Subnormal / zero (matches CPU: sign_f * mant * 2^-24). Practical KV
        // scales never reach here; we keep the branch so a corrupted zero-mant
        // half decodes to 0 rather than an undefined value.
        return sign_f * (float)mant * ldexpf(1.0f, -24);
    }
    if (exp == 0x1Fu) {
        // +/- inf if mantissa == 0, NaN otherwise. Also unreachable for a
        // healthy quantised scale, but pinned so a corrupt scale surfaces as
        // inf / NaN downstream instead of undefined.
        if (mant == 0u) {
            return sign_f * INFINITY;
        }
        return 0.0f / 0.0f;
    }
    return sign_f * (1.0f + (float)mant / 1024.0f) * ldexpf(1.0f, (int)exp - 15);
}

// Q4_0: 32 elems / block, 18 B (2 B FP16 scale + 16 B nibbles biased +8).
// `qs[i]` low nibble = elem 2·i, high nibble = elem 2·i+1; each nibble decodes
// as `(nib - 8) * d`. Symmetric quantisation (`_0` suffix), no zero-point.
extern "C" __global__ void vokra_dequant_gemv_q4_0_f32(
    const unsigned char* blocks,
    const float*         x,
    float*               y,
    unsigned int         n_rows,
    unsigned int         n_blocks_per_row)
{
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) {
        return;
    }
    const unsigned int block_bytes = 18u;
    unsigned int per_row_bytes = n_blocks_per_row * block_bytes;
    unsigned int row_start = row * per_row_bytes;

    float acc = 0.0f;
    for (unsigned int b = 0; b < n_blocks_per_row; ++b) {
        unsigned int block_off = row_start + b * block_bytes;
        unsigned short d_bits = (unsigned short)blocks[block_off]
                              | ((unsigned short)blocks[block_off + 1u] << 8);
        float d = vokra_kv_f16_to_f32(d_bits);
        unsigned int x_base = b * 32u;
        // 16 packed bytes -> 32 nibbles -> 32 dequantised values.
        for (unsigned int i = 0; i < 16u; ++i) {
            unsigned char byte = blocks[block_off + 2u + i];
            int lo = (int)(byte & 0x0Fu) - 8;
            int hi = (int)((byte >> 4) & 0x0Fu) - 8;
            acc += (float)lo * d * x[x_base + 2u * i];
            acc += (float)hi * d * x[x_base + 2u * i + 1u];
        }
    }
    y[row] = acc;
}

// Q5_0: 32 elems / block, 22 B (2 B FP16 scale + 4 B `qh` high bits + 16 B
// `qs` low 4 bits). Elem `i` decodes as `((qh_bit(i) << 4) | qs_lo4(i)) - 16`
// multiplied by `d`. Symmetric quantisation.
extern "C" __global__ void vokra_dequant_gemv_q5_0_f32(
    const unsigned char* blocks,
    const float*         x,
    float*               y,
    unsigned int         n_rows,
    unsigned int         n_blocks_per_row)
{
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) {
        return;
    }
    const unsigned int block_bytes = 22u;
    unsigned int per_row_bytes = n_blocks_per_row * block_bytes;
    unsigned int row_start = row * per_row_bytes;

    float acc = 0.0f;
    for (unsigned int b = 0; b < n_blocks_per_row; ++b) {
        unsigned int block_off = row_start + b * block_bytes;
        unsigned short d_bits = (unsigned short)blocks[block_off]
                              | ((unsigned short)blocks[block_off + 1u] << 8);
        float d = vokra_kv_f16_to_f32(d_bits);
        unsigned int qh_base = block_off + 2u; // 4 bytes, one high bit per elem
        unsigned int qs_base = block_off + 6u; // 16 bytes, two lo4 nibbles each
        unsigned int x_base  = b * 32u;
        for (unsigned int i = 0; i < 32u; ++i) {
            unsigned char lo4_byte = blocks[qs_base + (i >> 1)];
            unsigned int lo4 = ((i & 1u) != 0u)
                                   ? ((lo4_byte >> 4) & 0x0Fu)
                                   : (lo4_byte & 0x0Fu);
            unsigned char hi1_byte = blocks[qh_base + (i >> 3)];
            unsigned int hi1 = (hi1_byte >> (i & 7u)) & 0x01u;
            unsigned int biased = (hi1 << 4) | lo4;
            int signed_v = (int)biased - 16;
            acc += (float)signed_v * d * x[x_base + i];
        }
    }
    y[row] = acc;
}

// Q8_0: 32 elems / block, 34 B (2 B FP16 scale + 32 B i8 qs). Elem `i` decodes
// as `qs[i] * d`. Symmetric quantisation.
extern "C" __global__ void vokra_dequant_gemv_q8_0_f32(
    const unsigned char* blocks,
    const float*         x,
    float*               y,
    unsigned int         n_rows,
    unsigned int         n_blocks_per_row)
{
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) {
        return;
    }
    const unsigned int block_bytes = 34u;
    unsigned int per_row_bytes = n_blocks_per_row * block_bytes;
    unsigned int row_start = row * per_row_bytes;

    float acc = 0.0f;
    for (unsigned int b = 0; b < n_blocks_per_row; ++b) {
        unsigned int block_off = row_start + b * block_bytes;
        unsigned short d_bits = (unsigned short)blocks[block_off]
                              | ((unsigned short)blocks[block_off + 1u] << 8);
        float d = vokra_kv_f16_to_f32(d_bits);
        unsigned int x_base = b * 32u;
        for (unsigned int i = 0; i < 32u; ++i) {
            signed char q = (signed char)blocks[block_off + 2u + i];
            acc += (float)q * d * x[x_base + i];
        }
    }
    y[row] = acc;
}
"#;

/// 16×16 thread block (matches the Metal GEMM launch); the kernel guards the
/// ragged tail against `M`/`N`. Also the 2-D conv1d block dim.
const BLOCK: u32 = 16;

/// 1-D thread block for the row/element kernels (gemv / softmax / layer_norm /
/// gelu), matching the Metal `grid_1d` threadgroup width (256).
const BLOCK_1D: u32 = 256;

/// Minimum opt-in shared-memory budget (in bytes) the Flash-Attention v2
/// fused kernel needs per thread block to hold its Q + K/V + S tiles for the
/// Whisper `d_head = 64` decoder self-attention. Sourced from
/// `docs/adr/M2-03-followup-rtf.md` §2 D3 (`Br=16, Bc=64` tile budget
/// `16·64·4 + 64·64·4·2 + 16·64·4 ≈ 40 KB`). Kept as a shared constant so
/// [`CudaDecodeSession::new`]'s device probe and the kernel launch agree by
/// construction.
const FLASH_ATTN_V2_MIN_SHARED_BYTES: c_int = 40 * 1024;

/// CUDA driver attribute enum value for the opt-in per-block shared-memory
/// cap (`CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN`, cuda.h
/// `CUdevice_attribute` ordinal 97). Named locally rather than inflating
/// `sys` — a single-caller (`CudaContext::max_shared_memory_per_block_optin`)
/// constant is not worth a new re-export.
const CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN: c_int = 97;

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

/// A `+`-owned device allocation that carries its own `cuMemFree` fn pointer
/// (a copy of the driver's, since fn pointers are `Copy`), so it drops without a
/// live borrow of the driver — the CUDA sibling of `vokra-backend-metal`'s
/// `OwnedBuf`. Used by [`CudaDecodeSession`], which owns both the buffers and
/// the [`CudaContext`] whose driver they were allocated from: `DeviceBuf<'ctx>`
/// would be a self-referential struct in that setting (a field borrowing another
/// field), so the session uses `OwnedDeviceBuf` instead. Drop order is enforced
/// by putting `ctx` last in the session struct (Rust drops fields top-to-bottom,
/// so every buffer's `cuMemFree` runs before the context's `cuCtxDestroy` and
/// the `dlclose` in `_lib` that unloads `libcuda`).
///
/// `ptr = 0` is the empty sentinel (never allocated / already freed); Drop skips
/// it, matching [`DeviceBuf`].
struct OwnedDeviceBuf {
    ptr: CUdeviceptr,
    /// Element count (f32s) held in the buffer — only used for shape validation
    /// inside the session; the drop only needs `ptr`.
    len: usize,
    /// Snapshot of the driver's `cuMemFree` fn pointer (copied at allocation).
    /// Fn pointers are `Copy` and stay valid as long as `libcuda` is loaded; the
    /// owning session drops its [`CudaContext`] (and hence `_lib`) only AFTER
    /// every `OwnedDeviceBuf` has run its Drop.
    free_fn: sys::FnCuMemFree,
}

impl Drop for OwnedDeviceBuf {
    fn drop(&mut self) {
        if self.ptr == 0 {
            return;
        }
        // SAFETY: `ptr` is a live device allocation from `cuMemAlloc`, freed once
        // via the driver-loaded `cuMemFree` fn pointer. `libcuda` is guaranteed
        // loaded because the owning session's [`CudaContext`] (which owns the
        // `DynLib` handle) is dropped strictly after this buffer (field order).
        unsafe { (self.free_fn)(self.ptr) };
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

/// A device-resident autoregressive self-attention key/value cache — the
/// decoder-step Phase 2 primitive (create with [`CudaContext::new_kv_cache`],
/// grow with [`CudaContext::kv_append`], read with [`CudaContext::kv_download`]),
/// the CUDA analogue of `vokra-backend-metal`'s `MetalKvCache`.
///
/// Two `[cap_rows, width]` row-major buffers are reserved **once** to the hard
/// `cap_rows` bound (the decoder's `n_text_ctx`); each decode step appends its new
/// `[t, width]` rows by launching the k/v-projection GEMM with its output pointer
/// advanced to row `len`, so the cache never reallocates or copies mid-decode —
/// matching the host [`vokra_core::KvCache`] semantics on the device (same GEMM,
/// same bytes, only the destination is the resident buffer at a row offset). The
/// fixed **cross**-attention encoder K/V is uploaded once with
/// [`CudaContext::upload`] instead; it needs no reserve/append.
///
/// Holds two [`CudaDeviceTensor`]s, so it borrows the context like every other
/// device handle and cannot outlive it (`cuMemFree` needs the creating context).
pub struct CudaKvCache<'ctx> {
    /// Key rows `[cap_rows, width]`, filled `[0, len)` from row 0 up.
    k: CudaDeviceTensor<'ctx>,
    /// Value rows `[cap_rows, width]`, filled in lockstep with `k`.
    v: CudaDeviceTensor<'ctx>,
    /// Reserved row capacity — the hard bound `kv_append` never exceeds.
    cap_rows: usize,
    /// Width (hidden size) of one cached row.
    width: usize,
    /// Committed rows (positions) currently in the cache.
    len: usize,
}

impl CudaKvCache<'_> {
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
/// [`CudaContext::run_attn`], [`CudaContext::attn_dev`],
/// [`CudaContext::encode_prenorm_stack`] and the decoder-step Phase-3b
/// `CudaDecodeSession`. `scale = head_dim^-0.5` is folded into the qh gather.
/// `causal = true` swaps in `vokra_softmax_causal_f32` for the decoder
/// self-attention (`q_offset` = the absolute position of query row 0); every
/// other pass — gather, transpose, both GEMMs, scatter — is byte-for-byte
/// identical to the non-causal chain, so the numerics stay single-sourced. On
/// `causal = false` the plain `vokra_softmax_f32` runs and `q_offset` is
/// ignored.
struct AttnChainDims {
    t_q: usize,
    t_kv: usize,
    d: usize,
    n_head: usize,
    scale: f32,
    has_q_bias: bool,
    has_out_bias: bool,
    /// Whether the softmax over each query row masks the causal future
    /// (`vokra_softmax_causal_f32`); `false` = the plain softmax (encoder /
    /// cross-attention path).
    causal: bool,
    /// Absolute position of query row 0 (only read when `causal`): row `i`
    /// attends keys `[0, q_offset + i]`. For a steady-state single-token step
    /// `t_q == 1` and `q_offset == t_kv - 1`; for a prefix step (`t_q > 1`)
    /// `q_offset == t_kv - t_q`.
    q_offset: usize,
    /// Whether [`CudaContext::launch_attn_chain`] should route the whole chain
    /// through the fused Flash-Attention v2 kernel
    /// ([`CudaContext::launch_flash_attn_v2`]) instead of the per-head
    /// `2 + 7·n_head` launches. Default `false` (byte-for-byte the current
    /// decomposed path — Kokoro / piper-plus / every host-in/out entrypoint is
    /// unaffected). Set to `true` only when the constructor has verified the
    /// device supports the required opt-in shared-memory budget (`d_head == 64`
    /// and `MAX_SHARED_MEMORY_PER_BLOCK_OPTIN ≥ 40 KB`), so a `true` value is a
    /// promise the FA v2 launch will succeed. Silent CPU fallback is forbidden
    /// (FR-EX-08); with `use_flash_attn = true` the wrapper either launches or
    /// returns an explicit error.
    use_flash_attn: bool,
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
    /// The ordinal-0 device handle (`CUdevice = c_int`); kept so runtime
    /// capability probes — such as [`Self::max_shared_memory_per_block_optin`]
    /// used by [`CudaDecodeSession::new`] to gate `AttnChainDims::use_flash_attn`
    /// on ≥ 40 KB of shared memory — can query
    /// [`sys::CudaDriver::cu_device_get_attribute`] without re-resolving the
    /// device. The value `0` is a valid ordinal, not a null pointer.
    device: sys::CUdevice,
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
    /// Causal softmax fused with the decoder self-attention mask (decoder-step
    /// Phase 3b, byte-for-byte parity with the Metal `vokra_softmax_causal_f32`
    /// kernel). Distinct kernel — never a runtime branch inside `softmax`.
    softmax_causal: CUfunction,
    layer_norm: CUfunction,
    gelu: CUfunction,
    conv1d: CUfunction,
    col_gather: CUfunction,
    col_gather_t: CUfunction,
    col_scatter: CUfunction,
    add_assign: CUfunction,
    /// M3-01 out-of-place element-wise Add / Mul kernel handles
    /// (`vokra_add_f32`, `vokra_mul_f32`). Distinct from [`Self::add_assign`]
    /// (in-place residual) — these back the graph-executor `OpKind::Add` /
    /// `OpKind::Mul` arms in [`crate::eval::eval_cuda_op`] with the same
    /// semantic contract as `vokra_backend_cpu::kernels::{add_f32, mul_f32}`.
    add: CUfunction,
    mul: CUfunction,
    /// FA v2 fused causal attention kernel handle (`vokra_flash_attn_v2_causal_f32`).
    /// Always resolved at context construction (its symbol is baked into the
    /// same PTX as the other Phase-5 attention kernels); actually dispatched
    /// only when the [`CudaDecodeSession`] probe (`hd == 64` +
    /// `MAX_SHARED_MEMORY_PER_BLOCK_OPTIN ≥ 40 KB`) flips
    /// [`AttnChainDims::use_flash_attn`] `true`. Owned via `kernels_module`.
    flash_attn_v2: CUfunction,
    /// M3-04 fused KV-cache dequant + GEMV kernel handles, one per quant format
    /// (`vokra_dequant_gemv_q4_0_f32` / `_q5_0_f32` / `_q8_0_f32`). Symmetric
    /// with the Metal `dequant_gemv_*` MSL pipelines; each is the GPU
    /// implementation of the [`vokra_core::KvQuantDequantGemvOps`] trait, whose
    /// CPU differential oracle is
    /// [`vokra_core::kv_quant::dequant_gemm::dequant_gemv_scalar`]. Owned via
    /// `kernels_module`.
    dequant_gemv_q4_0: CUfunction,
    dequant_gemv_q5_0: CUfunction,
    dequant_gemv_q8_0: CUfunction,
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
                device: dev,
                context,
                stream,
                gemm_module: m.gemm_module,
                kernels_module: m.kernels_module,
                gemm: m.gemm,
                gemv: m.gemv,
                softmax: m.softmax,
                softmax_causal: m.softmax_causal,
                layer_norm: m.layer_norm,
                gelu: m.gelu,
                conv1d: m.conv1d,
                col_gather: m.col_gather,
                col_gather_t: m.col_gather_t,
                col_scatter: m.col_scatter,
                add_assign: m.add_assign,
                add: m.add,
                mul: m.mul,
                flash_attn_v2: m.flash_attn_v2,
                dequant_gemv_q4_0: m.dequant_gemv_q4_0,
                dequant_gemv_q5_0: m.dequant_gemv_q5_0,
                dequant_gemv_q8_0: m.dequant_gemv_q8_0,
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

    /// Reads
    /// `CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN` (ordinal 97)
    /// for the owned device — the largest per-block shared-memory
    /// allocation the driver will grant when a kernel opts in via
    /// `cuFuncSetAttribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES)`.
    /// Used by [`CudaDecodeSession::new`] to gate the FA v2 seam
    /// ([`AttnChainDims::use_flash_attn`]): the fused kernel needs
    /// ≥ [`FLASH_ATTN_V2_MIN_SHARED_BYTES`] per block, and if that budget
    /// is unavailable the session stays on the decomposed
    /// [`Self::launch_attn_chain`] path.
    ///
    /// # Errors
    ///
    /// [`VokraError::BackendUnavailable`] if the driver call fails; callers
    /// that only need a best-effort probe collapse that to
    /// `unwrap_or(0)` (a zero budget disables FA v2 exactly like an
    /// unsupported device).
    fn max_shared_memory_per_block_optin(&self) -> Result<c_int> {
        let mut val: c_int = 0;
        // SAFETY: `driver.cu_device_get_attribute` is the resolved
        // `cuDeviceGetAttribute` entry point; `&mut val` is a valid writable
        // c_int, the attribute ordinal is the documented enum value, and
        // `self.device` is the ordinal-0 device handle written by
        // `cuDeviceGet` in `Self::new`. No handles or lifetimes escape.
        let r = unsafe {
            (self.driver.cu_device_get_attribute)(
                &mut val,
                CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
                self.device,
            )
        };
        sys::check(
            &self.driver,
            r,
            "cuDeviceGetAttribute(MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)",
        )?;
        Ok(val)
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

    // ---- M3-04 fused KV-cache dequant + GEMV ------------------------------

    /// GPU-side fused dequantisation + row-wise GEMV over a quantised KV block
    /// matrix — the CUDA implementation of the
    /// [`KvQuantDequantGemvOps`](vokra_core::KvQuantDequantGemvOps) seam
    /// (M3-04-T09).
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
    /// backend parity test (`parity_kernels_cuda::dequant_gemv_matches_cpu`)
    /// pins this to `atol = 1e-4`.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on shape mismatch or `mode ==
    /// KvQuant::Fp32`; [`VokraError::BackendUnavailable`] on a device launch
    /// failure.
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
        self.run_dequant_gemv(mode, blocks_bytes, n_rows, n_blocks_per_row, x)
    }

    fn run_dequant_gemv(
        &self,
        mode: vokra_core::KvQuant,
        blocks_bytes: &[u8],
        n_rows: usize,
        n_blocks_per_row: usize,
        x: &[f32],
    ) -> Result<Vec<f32>> {
        let kernel = match mode {
            vokra_core::KvQuant::Q4_0 => self.dequant_gemv_q4_0,
            vokra_core::KvQuant::Q5_0 => self.dequant_gemv_q5_0,
            vokra_core::KvQuant::Q8_0 => self.dequant_gemv_q8_0,
            vokra_core::KvQuant::Fp32 => {
                // Guarded by `validate_dequant_gemv`; keep as an explicit error
                // (never a silent fallback, FR-EX-08).
                return Err(VokraError::InvalidArgument(
                    "dequant_gemv_f32: mode=Fp32 rejected".to_owned(),
                ));
            }
        };

        // Device buffers: packed on-wire bytes (input), FP32 x, FP32 output.
        // A zero-length input is impossible here (`n_rows == 0` early-returned),
        // so `alloc` always receives a positive byte count.
        let blocks_buf = self.alloc(blocks_bytes.len())?;
        self.htod_bytes(&blocks_buf, blocks_bytes)?;
        let x_buf = self.alloc(size_of_val(x))?;
        self.htod(&x_buf, x)?;
        let out_buf = self.alloc(n_rows * size_of::<f32>())?;

        // Kernel signature: (blocks, x, y, n_rows, n_blocks_per_row).
        let n_rows_u = n_rows as c_uint;
        let n_bpr_u = n_blocks_per_row as c_uint;
        let mut params: [*mut c_void; 5] = [
            ptr_arg(&blocks_buf.ptr),
            ptr_arg(&x_buf.ptr),
            ptr_arg(&out_buf.ptr),
            uint_arg(&n_rows_u),
            uint_arg(&n_bpr_u),
        ];
        let grid_x = n_rows.div_ceil(BLOCK_1D as usize) as c_uint;
        let launch_tag = match mode {
            vokra_core::KvQuant::Q4_0 => "cuLaunchKernel(vokra_dequant_gemv_q4_0_f32)",
            vokra_core::KvQuant::Q5_0 => "cuLaunchKernel(vokra_dequant_gemv_q5_0_f32)",
            vokra_core::KvQuant::Q8_0 => "cuLaunchKernel(vokra_dequant_gemv_q8_0_f32)",
            vokra_core::KvQuant::Fp32 => unreachable!("guarded above"),
        };
        self.launch(
            kernel,
            (grid_x, 1, 1),
            (BLOCK_1D, 1, 1),
            &mut params,
            launch_tag,
        )?;

        let mut out = vec![0.0f32; n_rows];
        self.dtoh(&out_buf, &mut out)?;
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

    /// Element-wise `out = a + b` on the GPU — the exact contract of
    /// `vokra_backend_cpu::kernels::add_f32`. Backs the graph-executor
    /// `OpKind::Add` arm (M3-01-T06); distinct from [`Self::residual_add_dev`]
    /// (in-place device-to-device residual used by the encoder-resident chain).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on a length mismatch;
    /// [`VokraError::BackendUnavailable`] on a device failure.
    pub fn add_f32(&self, a: &[f32], b: &[f32], out: &mut [f32]) -> Result<()> {
        validate_binary(a, b, out)?;
        if out.is_empty() {
            return Ok(());
        }
        self.run_binary(self.add, a, b, out, "cuLaunchKernel(vokra_add_f32)")
    }

    /// Element-wise `out = a * b` on the GPU — the exact contract of
    /// `vokra_backend_cpu::kernels::mul_f32`. Backs the graph-executor
    /// `OpKind::Mul` arm (M3-01-T06).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on a length mismatch;
    /// [`VokraError::BackendUnavailable`] on a device failure.
    pub fn mul_f32(&self, a: &[f32], b: &[f32], out: &mut [f32]) -> Result<()> {
        validate_binary(a, b, out)?;
        if out.is_empty() {
            return Ok(());
        }
        self.run_binary(self.mul, a, b, out, "cuLaunchKernel(vokra_mul_f32)")
    }

    /// Shared runner for the M3-01 element-wise binary kernels
    /// (`vokra_add_f32` / `vokra_mul_f32`): upload two host operands, launch
    /// the resolved kernel, download the output. One thread per element,
    /// ragged-tail guarded, so `out.len() = a.len() = b.len()` is the only
    /// shape constraint (checked by [`validate_binary`] at the entry point).
    fn run_binary(
        &self,
        kernel: CUfunction,
        a: &[f32],
        b: &[f32],
        out: &mut [f32],
        launch_label: &str,
    ) -> Result<()> {
        let a_buf = self.alloc(size_of_val(a))?;
        self.htod(&a_buf, a)?;
        let b_buf = self.alloc(size_of_val(b))?;
        self.htod(&b_buf, b)?;
        let out_buf = self.alloc(size_of_val(out))?;

        let n_u = out.len() as c_uint;
        let mut params: [*mut c_void; 4] = [
            ptr_arg(&a_buf.ptr),
            ptr_arg(&b_buf.ptr),
            ptr_arg(&out_buf.ptr),
            uint_arg(&n_u),
        ];
        let grid_x = out.len().div_ceil(BLOCK_1D as usize) as c_uint;
        self.launch(
            kernel,
            (grid_x, 1, 1),
            (BLOCK_1D, 1, 1),
            &mut params,
            launch_label,
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
                causal: false,
                q_offset: 0,
                // Host-in/out `attn_f32` stays on the byte-for-byte decomposed
                // path (M2-03 parity, no FA v2 opt-in): only
                // `CudaDecodeSession::new`'s d_head/shared-memory probe flips
                // this true for the decoder-step self-attention.
                use_flash_attn: false,
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
        // FA v2 opt-in seam. The caller (only the decoder-step session, whose
        // constructor probed `d_head == 64` **and** the opt-in shared-memory
        // budget) sets `dims.use_flash_attn = true` to route the whole chain
        // (q-proj → fused causal Flash-Attention v2 → out-proj) through
        // `launch_flash_attn_v2` in ONE `cuLaunchKernel` per phase, folding
        // per-head parallelism into `grid.z = n_head` (O3). Every other call
        // site — `attn_f32`, `attn_dev`, `encode_prenorm_stack`, the
        // cross-attention of the decoder step — leaves the flag `false` and
        // gets the byte-for-byte decomposed `2 + 7·n_head` chain that
        // Kokoro / piper-plus and the M2-03 parity suite depend on. Silent
        // CPU fallback is forbidden (NFR-RL-06, FR-EX-08); FA v2 is a
        // GPU-only alternate path, never a runtime CPU escape.
        //
        // t_q gate (Approach A, M2-03-followup-rtf T-follow-04): the FA v2
        // kernel's query tile is `BR = 16`, so a launch with `t_q < 16`
        // wastes at least half of every score-compute tile (`BR·BC = 1024`
        // entries, only `t_q·BC ≤ 15·64 = 960` valid) and drives the online
        // softmax through a single thread. The Whisper decoder's
        // steady-state hot path is `t_q == 1` — the FA v2 fusion cost
        // (extra gather + kh reshape) then dominates, and the wrapper
        // regresses vs the decomposed path. The gate keeps the code alive
        // for prefix steps (`t_q > 1`) and non-Whisper models where
        // `t_q ≫ BR` will amortise the fusion, but skips it whenever the
        // tile would be wasted. Not a CPU fallback — falls through to the
        // decomposed GPU path below (FR-EX-08 preserved).
        const FA_V2_MIN_TQ: usize = 16; // BR tile size — below this ≥50% of the tile is wasted
        if dims.use_flash_attn && dims.t_q >= FA_V2_MIN_TQ {
            return self.launch_flash_attn_v2(dims, ptrs);
        }
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
            // probs = softmax_rows(scores). Causal decoder self-attention masks
            // the future in the fused `vokra_softmax_causal_f32` (the ONLY pass
            // that differs from the non-causal chain); everything else — gather,
            // transpose, both GEMMs, scatter — is byte-for-byte identical, so the
            // numerics stay single-sourced (Metal parity: mirror of
            // `encode_attn_passes`'s causal branch).
            if dims.causal {
                let q_offset_u = dims.q_offset as c_uint;
                let mut p_soft_c: [*mut c_void; 5] = [
                    ptr_arg(&ptrs.scores),
                    ptr_arg(&ptrs.probs),
                    uint_arg(&t_q_u),
                    uint_arg(&t_kv_u),
                    uint_arg(&q_offset_u),
                ];
                self.launch_async(
                    self.softmax_causal,
                    lin_grid(t_q),
                    lin_block,
                    &mut p_soft_c,
                    "cuLaunchKernel(vokra_softmax_causal_f32 attn)",
                )?;
            } else {
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
            }
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

    /// M2-03-followup-rtf-sub-0.1 FA v2 fused-attention path (D3/D4 of
    /// `docs/adr/M2-03-followup-rtf.md`). Same chain shape as the decomposed
    /// [`Self::launch_attn_chain`] — q-proj GEMM → per-head {qh/vh/kh gather →
    /// **fused causal Flash-Attention v2** → scatter} → out-proj GEMM — but
    /// the middle three per-head launches (scores GEMM + causal softmax +
    /// context GEMM) collapse into ONE `vokra_flash_attn_v2_causal_f32` launch
    /// per head. Launch count per attention block drops from
    /// `2 + 7·n_head` (decomposed) to `2 + 5·n_head` (gather qh + gather vh +
    /// gather kh + FA v2 + scatter). The tile budget (`Br=16, Bc=64`) matches
    /// `FLASH_ATTN_V2_MIN_SHARED_BYTES` and the constructor's shared-memory
    /// probe.
    ///
    /// The `qh` gather pre-multiplies Q by `dims.scale`, and the FA v2 kernel
    /// receives `scale = 1.0` — mathematically equivalent to un-scaled Q with
    /// `scale = dims.scale`, chosen so the qh gather kernel is reused verbatim
    /// (byte-identical numerics with the decomposed path's Q pre-scaling
    /// convention). The `kh_t` scratch buffer is reused as non-transposed `kh`
    /// (`vokra_col_gather_f32` writes it in `[t_kv, hd]` row-major layout, the
    /// shape FA v2 consumes; the transposed variant `col_gather_t` used by the
    /// decomposed scores GEMM is never launched on this path).
    ///
    /// Handles both `causal = true` (self-attention with `q_offset`) and
    /// `causal = false` (cross-attention) — the FA v2 kernel branches on the
    /// `causal` parameter internally, and both variants come out of the same
    /// probe (the session's `use_flash_attn` flag routes both attention chains
    /// per decoder step).
    ///
    /// # Errors
    ///
    /// [`VokraError::BackendUnavailable`] on a driver launch failure. Never a
    /// silent CPU fall back (NFR-RL-06, FR-EX-08); a device without the FA v2
    /// tile budget must keep the session's probe result `use_flash_attn = false`
    /// so this method is not reached at all.
    fn launch_flash_attn_v2(&self, dims: &AttnChainDims, ptrs: &AttnChainPtrs) -> Result<()> {
        let (t_q, t_kv, d, n_head) = (dims.t_q, dims.t_kv, dims.d, dims.n_head);
        let hd = d / n_head;

        // Tile constants — MUST match the kernel-side `const int BR/BC` in
        // `vokra_flash_attn_v2_causal_f32`. Encoded once as `usize` for
        // arithmetic then narrowed at the launch dims / shared-memory budget.
        const BR_HOST: usize = 16;
        const BC_HOST: usize = 64;
        // Dynamic shared memory: `BR·hd + BC·hd + BC·hd + BR·BC` floats.
        // For hd=64 this is `16·64 + 64·64 + 64·64 + 16·64 = 10240 floats =
        // 40 KiB`, matching [`FLASH_ATTN_V2_MIN_SHARED_BYTES`]. Under the
        // per-block default (48 KiB, compute capability ≥ 2.0) so no opt-in
        // via `cuFuncSetAttribute` is needed for the current Whisper shapes.
        let shared_bytes: c_uint =
            ((BR_HOST * hd + BC_HOST * hd + BC_HOST * hd + BR_HOST * BC_HOST) * 4) as c_uint;

        // Pre-cast scalars: every c_int / c_uint / f32 the launches read via
        // pointer must outlive the whole per-head loop (the driver reads them
        // during `cuLaunchKernel`), so they live on this function's stack.
        let tq_hd_n = t_q * hd;
        let tkv_hd_n = t_kv * hd;
        let t_q_u = t_q as c_uint;
        let t_kv_u = t_kv as c_uint;
        let d_u = d as c_uint;
        let hd_u = hd as c_uint;
        let has_bias_q: c_uint = u32::from(dims.has_q_bias);
        let has_bias_out: c_uint = u32::from(dims.has_out_bias);
        let scale_v = dims.scale;
        let one_v = 1.0f32;

        // FA v2 kernel scalar args. `scale = 1.0` because Q is pre-scaled in
        // the qh gather (matches the decomposed path's numerics convention).
        let t_q_i = t_q as c_int;
        let t_kv_i = t_kv as c_int;
        let hd_i = hd as c_int;
        let q_offset_i = dims.q_offset as c_int;
        let causal_b = dims.causal;
        let fa_scale = 1.0f32;

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

        // 1. q = xq · q_w (+q_bias) — byte-for-byte identical to the decomposed
        // path's q-proj (same kernel, same grid/block, same params).
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
            "cuLaunchKernel(vokra_gemm_f32 attn q-proj [FA v2])",
        )?;

        // 2. Per-head: gather → FA v2 → scatter. Stream ordering guarantees
        // head `h+1`'s gather sees head `h`'s scatter completion in the shared
        // qh/vh/kh_t/ctx_h scratches (the same reuse pattern the decomposed
        // path relies on).
        for h in 0..n_head {
            let c0_u = (h * hd) as c_uint;

            // qh[i,c] = q[i, c0+c] * scale — pre-scale Q so FA v2 uses scale=1.
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
                "cuLaunchKernel(vokra_col_gather_f32 attn qh [FA v2])",
            )?;
            // vh[j,c] = v[j, c0+c] (scale = 1) — identical to the decomposed path.
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
                "cuLaunchKernel(vokra_col_gather_f32 attn vh [FA v2])",
            )?;
            // kh[j,c] = k[j, c0+c] — NON-transposed K gather (FA v2's contract).
            // Reuses the `kh_t` buffer as `kh` (same element count `t_kv·hd`);
            // the transposed variant is never used on this code path.
            let mut p_kh: [*mut c_void; 7] = [
                ptr_arg(&ptrs.k),
                ptr_arg(&ptrs.kh_t),
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
                &mut p_kh,
                "cuLaunchKernel(vokra_col_gather_f32 attn kh [FA v2])",
            )?;
            // FA v2 fused per-head launch. Grid: `⌈t_q / BR⌉` blocks along
            // queries; head parallelism is expressed by the per-head loop (the
            // kernel treats Q/K/V/O as single-head via head-relative pointers).
            // Block: 128 threads is inside the kernel's stride-loop contract
            // (`for idx = tid; idx < ...; idx += blockDim.x`); no strict
            // `blockDim.x == BC` constraint anywhere in the body. Dynamic
            // shared memory sized above.
            let mut p_fa: [*mut c_void; 10] = [
                ptr_arg(&ptrs.qh),
                ptr_arg(&ptrs.kh_t), // reused: non-transposed kh, [t_kv, hd].
                ptr_arg(&ptrs.vh),
                ptr_arg(&ptrs.ctx_h),
                int_arg(&t_q_i),
                int_arg(&t_kv_i),
                int_arg(&hd_i),
                int_arg(&q_offset_i),
                bool_arg(&causal_b),
                f32_arg(&fa_scale),
            ];
            let fa_grid = (t_q.div_ceil(BR_HOST) as c_uint, 1, 1);
            let fa_block: (c_uint, c_uint, c_uint) = (128, 1, 1);
            self.launch_async_shared(
                self.flash_attn_v2,
                fa_grid,
                fa_block,
                shared_bytes,
                &mut p_fa,
                "cuLaunchKernel(vokra_flash_attn_v2_causal_f32 attn)",
            )?;
            // context[i, c0+c] = ctx_h[i,c] — same scatter as decomposed path.
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
                "cuLaunchKernel(vokra_col_scatter_f32 attn [FA v2])",
            )?;
        }

        // 3. out = context · out_w (+out_bias) — byte-for-byte identical to
        // the decomposed path's out-proj (same kernel, same params).
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
            "cuLaunchKernel(vokra_gemm_f32 attn out-proj [FA v2])",
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

    /// Allocates an uninitialised `len`-f32 [`OwnedDeviceBuf`] — the
    /// lifetime-free sibling of [`Self::alloc_dev`], used by [`CudaDecodeSession`]
    /// which owns both the buffers and this context.
    fn alloc_owned(&self, len: usize) -> Result<OwnedDeviceBuf> {
        let bytes = checked_mul(len, size_of::<f32>(), "alloc_owned bytes")?;
        let buf = self.alloc(bytes)?;
        // Defuse the borrowed-driver `DeviceBuf` into an `OwnedDeviceBuf` that
        // carries a copy of the free fn: the raw device ptr moves ownership, and
        // `core::mem::forget(buf)` cancels the borrowed-driver `Drop`.
        let ptr = buf.ptr;
        core::mem::forget(buf);
        Ok(OwnedDeviceBuf {
            ptr,
            len,
            free_fn: self.driver.cu_mem_free,
        })
    }

    /// H2D upload of `data` into a fresh [`OwnedDeviceBuf`] (allocate + copy).
    fn upload_owned(&self, data: &[f32]) -> Result<OwnedDeviceBuf> {
        let buf = self.alloc_owned(data.len())?;
        // Use `htod` with a temporary borrowed `DeviceBuf` view; `htod` only
        // reads `ptr`, so this borrows the driver just for the copy.
        let view = DeviceBuf {
            driver: &self.driver,
            ptr: buf.ptr,
        };
        let r = self.htod(&view, data);
        core::mem::forget(view); // buf owns the alloc; don't double-free
        r?;
        Ok(buf)
    }

    /// Optional upload: `Some(slice) -> Some(OwnedDeviceBuf)`, `None -> None`.
    fn upload_owned_opt(&self, data: Option<&[f32]>) -> Result<Option<OwnedDeviceBuf>> {
        data.map(|d| self.upload_owned(d)).transpose()
    }

    // ---- Decoder-step Phase 2: device-resident self-attention K/V cache ------

    /// Reserves a device-resident autoregressive self-attention K/V cache
    /// ([`CudaKvCache`]): two `[cap_rows, width]` buffers allocated **once** to the
    /// hard `cap_rows` bound (the decoder's `n_text_ctx`), starting empty — the
    /// CUDA analogue of `MetalContext::new_kv_cache`. The fixed cross-attention
    /// encoder K/V is uploaded once with [`Self::upload`] instead.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] if `cap_rows` or `width` is zero;
    /// [`VokraError::BackendUnavailable`] if a device buffer cannot be allocated.
    pub fn new_kv_cache(&self, cap_rows: usize, width: usize) -> Result<CudaKvCache<'_>> {
        if cap_rows == 0 || width == 0 {
            return Err(VokraError::InvalidArgument(
                "kv cache cap_rows and width must both be >= 1".to_owned(),
            ));
        }
        let cap = checked_mul(cap_rows, width, "kv cache cap_rows*width")?;
        Ok(CudaKvCache {
            k: self.alloc_dev(cap)?,
            v: self.alloc_dev(cap)?,
            cap_rows,
            width,
            len: 0,
        })
    }

    /// Appends one decode step's `t` new rows to `cache`, projected from the
    /// device-resident `x` `[t, d]` by the key / value weight matrices
    /// `k_w` / `v_w` `[d, width]` (+ optional `[width]` bias): the two projection
    /// GEMMs launch with their output pointer **advanced to row `cache.len`** of the
    /// resident K / V buffers (one stream, one synchronise), then the committed
    /// length advances by `t`.
    ///
    /// Bit-identical to a host `project_kv` + [`vokra_core::KvCache`] `append`: the
    /// same GEMM kernel and operands, the only difference being the destination is a
    /// resident device buffer at a row byte-offset (`cache.len * width * 4`) rather
    /// than a fresh host buffer. The reserve is a hard bound — appending past
    /// `cache.capacity_rows()` is an explicit error, never a realloc.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on a zero `t`/`d`, an operand-shape mismatch,
    /// or an append exceeding the reserved capacity;
    /// [`VokraError::BackendUnavailable`] on a device launch / sync failure.
    #[allow(clippy::too_many_arguments)] // k/v projection operand set (x + two weights + biases)
    pub fn kv_append(
        &self,
        cache: &mut CudaKvCache<'_>,
        t: usize,
        d: usize,
        x: &CudaDeviceTensor<'_>,
        k_w: &CudaDeviceTensor<'_>,
        k_bias: Option<&CudaDeviceTensor<'_>>,
        v_w: &CudaDeviceTensor<'_>,
        v_bias: Option<&CudaDeviceTensor<'_>>,
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
        // bound: a device cache cannot grow between launches).
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
        // Byte offset of row `len` in each resident buffer — the k/v GEMM writes
        // its `[t, width]` output starting there.
        let off = checked_mul(cache.len, width, "kv_append len*width")?;
        let off_bytes =
            checked_mul(off, size_of::<f32>(), "kv_append offset bytes")? as CUdeviceptr;
        let k_out = cache.k.buf.ptr.checked_add(off_bytes).ok_or_else(|| {
            VokraError::InvalidArgument("kv_append device offset overflow".to_owned())
        })?;
        let v_out = cache.v.buf.ptr.checked_add(off_bytes).ok_or_else(|| {
            VokraError::InvalidArgument("kv_append device offset overflow".to_owned())
        })?;
        // A 1-float never-read device dummy bound where a bias is absent (the
        // kernel reads bias only when has_bias != 0).
        let dummy = self.alloc_dev(1)?;
        let k_bias_ptr = k_bias.map_or(dummy.buf.ptr, |b| b.buf.ptr);
        let v_bias_ptr = v_bias.map_or(dummy.buf.ptr, |b| b.buf.ptr);
        // K = x[t,d] @ k_w[d,width] (+k_bias) at row `len`; V likewise. Two
        // stream-ordered launches into distinct buffers, synchronised once.
        self.launch_gemm_async(
            x.buf.ptr,
            k_w.buf.ptr,
            k_bias_ptr,
            k_out,
            t,
            width,
            d,
            k_bias.is_some(),
        )?;
        self.launch_gemm_async(
            x.buf.ptr,
            v_w.buf.ptr,
            v_bias_ptr,
            v_out,
            t,
            width,
            d,
            v_bias.is_some(),
        )?;
        self.sync_stream("cuStreamSynchronize")?;
        cache.len = end;
        Ok(())
    }

    /// Reads the committed `[len, width]` key and value rows back into host buffers
    /// (`k_out` / `v_out`, each `len * width` f32). Appended rows occupy the front
    /// of the reserved buffers (growth from row 0), so this is a prefix copy; call
    /// after the last [`Self::kv_append`] (which synchronises, so the rows are
    /// readable).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] if either output length differs from
    /// `cache.len() * cache.width()`; [`VokraError::BackendUnavailable`] on a copy
    /// failure.
    pub fn kv_download(
        &self,
        cache: &CudaKvCache<'_>,
        k_out: &mut [f32],
        v_out: &mut [f32],
    ) -> Result<()> {
        let committed = checked_mul(cache.len, cache.width, "kv_download len*width")?;
        expect_len("kv_download k_out", k_out.len(), committed)?;
        expect_len("kv_download v_out", v_out.len(), committed)?;
        if committed == 0 {
            return Ok(());
        }
        self.dtoh(&cache.k.buf, k_out)?;
        self.dtoh(&cache.v.buf, v_out)
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
                causal: false,
                q_offset: 0,
                // Device-in/out `attn_dev` shares the same decomposed launch
                // chain as `attn_f32`; the FA v2 seam is gated on the session
                // constructor's probe, not on the entrypoint.
                use_flash_attn: false,
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
                    causal: false,
                    q_offset: 0,
                    // Whole-encoder residency uses the same decomposed chain
                    // as `attn_dev`; the FA v2 opt-in belongs to the
                    // decoder-step session, which is a distinct entrypoint.
                    use_flash_attn: false,
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

    /// Issues a single bias-less gemv launch on the stream (no synchronise):
    /// `out[i] = Σ_l A[i·k + l] · x[l]` for `i in 0..m` (has_bias = 0). Used by
    /// the decoder-step Phase-3b `CudaDecodeSession` tied-logits head — the
    /// caller pre-advances `x` / `out` to the current decoded row's byte offset
    /// (`x + i·d·4`, `out + i·n_vocab·4`), so ALL `[t, n_vocab]` rows are
    /// produced in ONE synchronise while each row remains a plain per-row
    /// reduction (the same math the CPU
    /// `whisper::decoder::project_logits_into`'s `t == 1` fast path runs on its
    /// single row). `bias` is bound but never read (`has_bias = 0`); the caller
    /// passes a valid dummy pointer.
    #[allow(clippy::too_many_arguments)] // intrinsic bias-less gemv parameter set
    fn launch_gemv_async(
        &self,
        a: CUdeviceptr,
        x: CUdeviceptr,
        bias: CUdeviceptr,
        out: CUdeviceptr,
        m: usize,
        k: usize,
    ) -> Result<()> {
        let m_u = m as c_uint;
        let k_u = k as c_uint;
        let hb: c_uint = 0; // tied logits head is bias-less
        let mut params: [*mut c_void; 7] = [
            ptr_arg(&a),
            ptr_arg(&x),
            ptr_arg(&bias),
            ptr_arg(&out),
            uint_arg(&m_u),
            uint_arg(&k_u),
            uint_arg(&hb),
        ];
        let grid = (m.div_ceil(BLOCK_1D as usize) as c_uint, 1, 1);
        self.launch_async(
            self.gemv,
            grid,
            (BLOCK_1D, 1, 1),
            &mut params,
            "cuLaunchKernel(vokra_gemv_f32 decode logits)",
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

    /// [`Self::launch_async`] variant that reserves `shared_bytes` bytes of
    /// dynamic per-block shared memory for the launch. Used by the FA v2 fused
    /// causal attention kernel (`vokra_flash_attn_v2_causal_f32`), which sizes
    /// its Q + K/V + S tiles at launch time via `extern __shared__`. Every
    /// other kernel keeps [`Self::launch_async`] (0 shared bytes).
    fn launch_async_shared(
        &self,
        func: CUfunction,
        grid: (c_uint, c_uint, c_uint),
        block: (c_uint, c_uint, c_uint),
        shared_bytes: c_uint,
        params: &mut [*mut c_void],
        what: &str,
    ) -> Result<()> {
        let d = &self.driver;
        // SAFETY: identical to [`Self::launch_async`] except the shared-memory
        // argument is non-zero. `shared_bytes` must be within the device's
        // per-block shared-memory budget (the caller — the decoder-step session
        // constructor — already checked
        // `MAX_SHARED_MEMORY_PER_BLOCK_OPTIN ≥ 40 KB` in the FA v2 probe).
        let r = unsafe {
            (d.cu_launch_kernel)(
                func,
                grid.0,
                grid.1,
                grid.2,
                block.0,
                block.1,
                block.2,
                shared_bytes,
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

    /// Byte-oriented sibling of [`Self::htod`] for the M3-04 packed KV block
    /// payload (a byte slice, not `[f32]`). Kept as its own method so a
    /// mistyped call site cannot silently upload the wrong element count.
    fn htod_bytes(&self, buf: &DeviceBuf<'_>, host: &[u8]) -> Result<()> {
        // SAFETY: `buf.ptr` is a device allocation of at least `host.len()`
        // bytes (the caller allocates `blocks_bytes.len()`); `host.as_ptr()` is
        // valid for that many bytes.
        let r = unsafe {
            (self.driver.cu_memcpy_htod)(buf.ptr, host.as_ptr().cast::<c_void>(), host.len())
        };
        sys::check(&self.driver, r, "cuMemcpyHtoD (bytes)")
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

// =====================================================================
// Decoder-step Phase 3b: device-resident CUDA decode session
// =====================================================================
//
// The autoregressive-decode sibling of [`CudaContext::encode_prenorm_stack`],
// bit-for-bit mirroring `vokra-backend-metal`'s `MetalDecodeSession` (Phase 3a):
// every decoder weight + the pre-projected cross-K/V is uploaded ONCE at
// construction, the self-attention K/V cache is reserved ONCE to the hard
// `n_text_ctx` bound, and each [`CudaDecodeSession::step`] issues the whole
// decode step's launches on the owned stream and synchronises ONCE + reads back
// the full `[t, n_vocab]` tied-head logits — no per-step allocation, no per-op
// synchronise.
//
// The same math the CPU per-op decode step runs (whisper::decoder::step), just
// re-emitted as launches on the single CUDA stream: `n_text_layer × [ln → self-
// attn (causal, KV-append via GEMM writing at row offset `start`) → residual →
// ln → cross-attn (non-causal, resident cross K/V) → residual → ln → mlp →
// residual]` + final ln + `t × gemv` into the resident logits scratch. The
// per-head attention chain is the exact `launch_attn_chain` the encoder /
// per-op `attn_dev` use (single-sourced kernel numerics), with the softmax
// swapped for `vokra_softmax_causal_f32` on the self-attention pass (`causal =
// true, q_offset = start`). The FP32 accumulation order / reduction shape are
// unchanged, so the session is bit-for-bit within the FP32 bound of the per-op
// GPU path AND the CPU decoder (parity: same as Metal Phase 3a, proven on M1).

/// One decoder block's weights + its per-layer self-attention K/V cache,
/// resident on the device — the CUDA sibling of the Metal `DevDecoderLayer`
/// used by `MetalDecodeSession`. Every field is an [`OwnedDeviceBuf`] so the
/// session can own both the buffers and the [`CudaContext`] whose driver
/// allocated them (Drop order: buffers first, ctx last — enforced by field
/// order in [`CudaDecodeSession`]). Absent biases (Whisper's `k`) stay `None`
/// and bind the shared 1-float dummy at launch time.
struct DevDecoderLayer {
    // Pre-self-attention LayerNorm (γ, β) — `[d]`.
    self_ln_g: OwnedDeviceBuf,
    self_ln_b: OwnedDeviceBuf,
    // Self-attention projections (`[d, d]`, biases `[d]`; Whisper `k` has no bias).
    self_q_w: OwnedDeviceBuf,
    self_q_bias: Option<OwnedDeviceBuf>,
    self_k_w: OwnedDeviceBuf,
    self_k_bias: Option<OwnedDeviceBuf>,
    self_v_w: OwnedDeviceBuf,
    self_v_bias: Option<OwnedDeviceBuf>,
    self_out_w: OwnedDeviceBuf,
    self_out_bias: Option<OwnedDeviceBuf>,
    // Pre-cross-attention LayerNorm + cross Q / out projections (`[d, d]`).
    cross_ln_g: OwnedDeviceBuf,
    cross_ln_b: OwnedDeviceBuf,
    cross_q_w: OwnedDeviceBuf,
    cross_q_bias: Option<OwnedDeviceBuf>,
    cross_out_w: OwnedDeviceBuf,
    cross_out_bias: Option<OwnedDeviceBuf>,
    // Pre-projected cross-attention K/V (`[n_ctx, d]`; computed by the model
    // layer from the encoder output once and uploaded once here).
    cross_k: OwnedDeviceBuf,
    cross_v: OwnedDeviceBuf,
    // Pre-MLP LayerNorm + MLP linears.
    mlp_ln_g: OwnedDeviceBuf,
    mlp_ln_b: OwnedDeviceBuf,
    fc1_w: OwnedDeviceBuf,
    fc1_bias: Option<OwnedDeviceBuf>,
    fc2_w: OwnedDeviceBuf,
    fc2_bias: Option<OwnedDeviceBuf>,
    // Resident self-attention KV cache `[n_text_ctx, d]`, filled `[0, pos)`
    // (the k/v-proj GEMMs write at row `start` each step; see `step`).
    self_k: OwnedDeviceBuf,
    self_v: OwnedDeviceBuf,
}

/// A device-resident Whisper decoder-step driver — the CUDA (M2 Phase 3b)
/// sibling of `vokra-backend-metal`'s `MetalDecodeSession`.
///
/// Built once at [`CudaDecodeSession::new`]: every decoder weight (all
/// `n_text_layer` blocks, the tied logits head, the final LayerNorm) and the
/// pre-projected cross-attention K/V are H2D uploaded, the self-attention KV
/// cache is reserved to the hard `n_text_ctx` bound, and the per-step scratch
/// (`[max_t_q, ·]` intermediates for `h`, `ln`, `q`, per-head `qh/vh/kh_t`,
/// `scores/probs`, `mlp_h/mlp_a`, and the `[max_t_q, n_vocab]` logits buffer)
/// is allocated ONCE. Each [`Self::step`] then advances the decode
/// device-resident: writes the caller's `[t, d]` token+positional embedding
/// into the resident `h`, issues every layer's launches on the owned CUDA
/// stream (no synchronise between them), fires `t` bias-less gemvs for the
/// tied-logits head, and does ONE `cuStreamSynchronize` + ONE D2H of the
/// `[t · n_vocab]` logits prefix into [`Self::all_logits`] / [`Self::last_logits`].
///
/// # `Send`, thread-affine at use
///
/// The session **owns** its [`CudaContext`] and holds only [`OwnedDeviceBuf`]
/// device buffers (a raw `CUdeviceptr` + a copy of `cuMemFree`; no
/// `CudaDeviceTensor<'ctx>`, so no self-referential lifetime). Even though the
/// `CudaContext` handles (`CUcontext`, `CUstream`, `CUmodule`, `CUfunction` —
/// each `*mut c_void`) are `!Send` at the Rust type level, moving the whole
/// session across threads is safe: an [`OwnedDeviceBuf`]'s `ptr` is a `u64` and
/// its `free_fn` is a Copy fn pointer (both `Send`), and no CUDA state is held
/// mid-flight between calls — every launch is enqueued and awaited inside a
/// single [`Self::step`] call. Callers moving a session between threads must
/// ensure the owned [`CudaContext`] is bound to the calling thread (via
/// `cuCtxSetCurrent`) before the first `step` on the new thread — the same
/// contract the CUDA driver documents for its context handles. `Send` is
/// asserted here (in the backend crate, whose `#![allow(unsafe_code)]` opt-out
/// permits it) so the model-layer `DecoderState` (whose
/// `assert_send::<DecoderState>()` compile-time bound + cross-thread decode
/// test both cover the cuda feature via CI's `--features cuda` matrix) stays
/// `Send`. `Sync` is deliberately NOT asserted: an autoregressive step depends
/// on the previous step's KV cache write and the session sits behind a `&mut`
/// on `DecoderState`, so Rust's ownership rules already enforce
/// single-thread-at-a-time access.
///
/// The device buffers are declared **before** `ctx` so Rust drops them first
/// (every `OwnedDeviceBuf`'s `cuMemFree` runs before `CudaContext`'s
/// `cuCtxDestroy` + the `dlclose` in `_lib` that unloads `libcuda`).
pub struct CudaDecodeSession {
    layers: Vec<DevDecoderLayer>,
    /// Tied logits head `[n_vocab, d]` — also the token embedding table, but
    /// the token gather is a host op, so only the logits projection needs it
    /// on the device.
    token_emb: OwnedDeviceBuf,
    /// Final decoder LayerNorm (γ, β), each `[d]`.
    ln_post_g: OwnedDeviceBuf,
    ln_post_b: OwnedDeviceBuf,
    /// A 1-float never-read device buffer bound where a bias is absent
    /// (`has_bias = 0`).
    dummy: OwnedDeviceBuf,
    /// Residual hidden stream `[max_t_q, d]` (each step's `[t, d]` embedding is
    /// written here, then the residual adds mutate it in place).
    h: OwnedDeviceBuf,
    ln: OwnedDeviceBuf,
    block_out: OwnedDeviceBuf,
    normed: OwnedDeviceBuf,
    q: OwnedDeviceBuf,
    context: OwnedDeviceBuf,
    qh: OwnedDeviceBuf,
    ctx_h: OwnedDeviceBuf,
    vh: OwnedDeviceBuf,
    kh_t: OwnedDeviceBuf,
    scores: OwnedDeviceBuf,
    probs: OwnedDeviceBuf,
    mlp_h: OwnedDeviceBuf,
    mlp_a: OwnedDeviceBuf,
    /// Resident `[max_t_q, n_vocab]` logits (row-major). Each step's tied head
    /// produces every decoded row (`[t, n_vocab]`); the readback pulls only the
    /// `t · n_vocab` prefix `step` wrote.
    logits: OwnedDeviceBuf,
    /// Host copy of the last step's `[max_t_q, n_vocab]` logits scratch.
    /// [`Self::last_logits`] returns the last row; [`Self::all_logits`] returns
    /// the `[last_t, n_vocab]` prefix `step` wrote.
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
    /// Row count the last [`Self::step`] wrote (`0` before the first step).
    last_t: usize,
    /// Whether the session is allowed to route the decoder self-attention
    /// through the fused Flash-Attention v2 kernel
    /// ([`CudaContext::launch_flash_attn_v2`]) instead of the decomposed
    /// `2 + 7·n_head` chain. Set at [`Self::new`] time from a device probe:
    /// `d_head == 64` **and**
    /// `MAX_SHARED_MEMORY_PER_BLOCK_OPTIN ≥ 40 KB` (the minimum tile budget
    /// the FA v2 kernel needs to hold Q + K/V + S). All other configurations
    /// stay on the decomposed path (Kokoro / piper-plus / non-64 `d_head`
    /// unaffected), so the `AttnChainDims::use_flash_attn` seam is opt-in and
    /// M2-03 default-path numerics are 1-bit-identical to the release before
    /// this follow-up.
    use_flash_attn: bool,
    /// Owned last so it drops **after** every device buffer above.
    ctx: CudaContext,
}

// SAFETY: The session owns a [`CudaContext`] and a set of [`OwnedDeviceBuf`]
// (raw device pointers + a Copy fn pointer). `CudaContext` is `!Send` at the
// Rust type level because it holds `*mut c_void` module / function / context /
// stream handles, but every one of those handles refers to CUDA driver state
// that is thread-transferable: the driver contract lets a context be made
// current on any thread via `cuCtxSetCurrent`, streams / modules / functions
// are just handles into the (transferable) context, and there is no interior
// non-thread-safe state held **between** [`Self::step`] calls — each step
// enqueues launches on the owned stream and awaits them inside the same call.
// So moving the whole session across threads is safe: the caller ensures the
// owned context is bound to the new thread before the first `step` there (the
// documented CUDA contract). This `Send` impl lets the model-layer
// `DecoderState` (which holds `Option<DecoderStepSession>`) stay `Send` — the
// compile-time `assert_send::<DecoderState>()` bound and the cross-thread
// decode test both stay green under `--features cuda`, without either
// reuploading every weight per step or duplicating attention math in
// `compute.rs`. `Sync` is deliberately NOT asserted: every step depends on the
// previous step's KV write and the caller borrows the session `&mut`, so
// shared-borrow concurrency has no meaning here.
unsafe impl Send for CudaDecodeSession {}

impl CudaDecodeSession {
    /// Builds a decode session: creates its own [`CudaContext`], uploads every
    /// decoder weight + the pre-projected cross-attention K/V (from `layers`) +
    /// the tied logits head, and reserves the self-attention KV cache to the
    /// hard `n_text_ctx` bound and the per-step scratch to `max_t_q` × the key
    /// window — all **once**. `max_t_q` is the widest single step (the
    /// forced-prefix width; steady-state steps decode one token).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on a zero / mismatched dimension or a
    /// weight-slice shape mismatch; [`VokraError::BackendUnavailable`] if there
    /// is no CUDA driver / GPU or a device allocation / copy fails.
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
    ) -> Result<CudaDecodeSession> {
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

        let ctx = CudaContext::new()?;
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
        let (mut buffers, dummy) = built?;

        // FA v2 shared-memory probe (M2-03 follow-up, RTF < 0.1). Only Whisper
        // shapes with `d_head == 64` are supported by the fused kernel
        // (`vokra_flash_attn_v2_causal_f32`), and the tile budget it needs is
        // ≥ 40 KB per block. A device that satisfies both flips
        // `AttnChainDims::use_flash_attn = true` on the session's decoder
        // self-attention launches; any other device (older SM, or a model with
        // `d_head != 64` — e.g. Kokoro / piper-plus) keeps the byte-for-byte
        // decomposed path. This probe is best-effort: if the driver call fails
        // we conservatively disable FA v2 rather than error out (the
        // decomposed path is always correct); silent-CPU-fallback (NFR-RL-06,
        // FR-EX-08) does NOT apply here — both paths are on the GPU.
        let hd = d / n_head;
        let opt_in_shared = ctx.max_shared_memory_per_block_optin().unwrap_or(0);
        let mut use_flash_attn = hd == 64 && opt_in_shared >= FLASH_ATTN_V2_MIN_SHARED_BYTES;
        // Escape hatch for the M2-14 sanity gate on vast.ai / any other host
        // where the FA v2 kernel launcher (`launch_flash_attn_v2`) still
        // returns `BackendUnavailable` (the stub state, per T-follow-02/03).
        // Setting `VOKRA_CUDA_DISABLE_FA_V2=1` forces the session onto the
        // decomposed `2 + 7·n_head` chain, which is always correct — the
        // measured RTF is then the honest steady-state number for the
        // decomposed path (never a silent CPU escape; FR-EX-08 stays intact
        // because both branches remain on the GPU).
        if use_flash_attn && std::env::var_os("VOKRA_CUDA_DISABLE_FA_V2").is_some() {
            use_flash_attn = false;
        }

        Ok(CudaDecodeSession {
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
            use_flash_attn,
            ctx,
        })
    }

    /// Uploads all weights + the pre-projected cross-KV, reserves the self-KV
    /// cache and the per-step scratch. Factored out of [`Self::new`] so the
    /// whole H2D + allocation burst is a single pass; a failure mid-way frees
    /// every buffer already allocated (their [`OwnedDeviceBuf`] `Drop`s do).
    #[allow(clippy::too_many_arguments)]
    fn build(
        ctx: &CudaContext,
        d: usize,
        n_head: usize,
        ff: usize,
        n_text_ctx: usize,
        n_vocab: usize,
        n_ctx: usize,
        max_t_q: usize,
        layers: &[DecoderLayerView<'_>],
        token_emb: &[f32],
        ln_post_gamma: &[f32],
        ln_post_beta: &[f32],
    ) -> Result<(CudaSessionBuffers, OwnedDeviceBuf)> {
        let up = |s: &[f32]| ctx.upload_owned(s);
        let up_opt = |s: Option<&[f32]>| ctx.upload_owned_opt(s);
        let hd = d / n_head;
        let max_tkv = n_text_ctx.max(n_ctx);
        // Reserve amounts (all fit — validated in `new`).
        let ntc_d = checked_mul(n_text_ctx, d, "decode n_text_ctx*d")?;
        let td = checked_mul(max_t_q, d, "decode max_t_q*d")?;
        let thd = checked_mul(max_t_q, hd, "decode max_t_q*hd")?;
        let tkvhd = checked_mul(max_tkv, hd, "decode max_tkv*hd")?;
        let ttkv = checked_mul(max_t_q, max_tkv, "decode max_t_q*max_tkv")?;
        let tff = checked_mul(max_t_q, ff, "decode max_t_q*ff")?;
        // `[max_t_q, n_vocab]` — the tied head produces every decoded row, so
        // the model-layer path can compare against the CPU decoder's
        // `[t, n_vocab]` output (not just the greedy last-row read).
        let tv = checked_mul(max_t_q, n_vocab, "decode max_t_q*n_vocab")?;

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
                self_k: ctx.alloc_owned(ntc_d)?,
                self_v: ctx.alloc_owned(ntc_d)?,
            });
        }
        let dummy = ctx.upload_owned(&[0.0f32])?;
        let buffers = CudaSessionBuffers {
            layers: dev_layers,
            token_emb: Some(up(token_emb)?),
            ln_post_g: Some(up(ln_post_gamma)?),
            ln_post_b: Some(up(ln_post_beta)?),
            h: Some(ctx.alloc_owned(td)?),
            ln: Some(ctx.alloc_owned(td)?),
            block_out: Some(ctx.alloc_owned(td)?),
            normed: Some(ctx.alloc_owned(td)?),
            q: Some(ctx.alloc_owned(td)?),
            context: Some(ctx.alloc_owned(td)?),
            qh: Some(ctx.alloc_owned(thd)?),
            ctx_h: Some(ctx.alloc_owned(thd)?),
            vh: Some(ctx.alloc_owned(tkvhd)?),
            kh_t: Some(ctx.alloc_owned(tkvhd)?),
            scores: Some(ctx.alloc_owned(ttkv)?),
            probs: Some(ctx.alloc_owned(ttkv)?),
            mlp_h: Some(ctx.alloc_owned(tff)?),
            mlp_a: Some(ctx.alloc_owned(tff)?),
            logits: Some(ctx.alloc_owned(tv)?),
        };
        Ok((buffers, dummy))
    }

    /// Advances the decode by the `t` tokens whose `[t, d]` token+positional
    /// embedding is `embedded` (the host gather; `t <= max_t_q`), starting at
    /// committed position `start`. Runs the whole step device-resident on the
    /// owned stream and reads back the full `[t, n_vocab]` tied-head logits
    /// (one row per decoded token, row-major) into [`Self::all_logits`];
    /// [`Self::last_logits`] reads the last of those rows for the greedy /
    /// argmax path.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on a bad `t` / `start` / `embedded`
    /// length; [`VokraError::BackendUnavailable`] on a CUDA failure.
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

        // H2D the `[t, d]` embedding into the resident `h` buffer (borrowed
        // driver just for the copy; `h` owns the alloc).
        self.htod_owned(&self.h, embedded)?;

        self.run_decode_step(t, start, t_kv)?;

        // Single per-step readback of ALL `[t, n_vocab]` rows the tied head
        // wrote (only the `t · n_vocab` prefix — the `max_t_q` tail past `t` is
        // left untouched and never observed).
        let take = checked_mul(t, self.n_vocab, "decode step t*n_vocab")?;
        // Split-borrow the ctx (immutable, for the driver + `dtoh`) and the
        // host mirror (mutable, the readback target) explicitly — they are
        // disjoint fields of `self`.
        {
            let logits_ptr = self.logits.ptr;
            let ctx = &self.ctx;
            let host_slice = &mut self.logits_host[..take];
            let view = DeviceBuf {
                driver: &ctx.driver,
                ptr: logits_ptr,
            };
            let r = ctx.dtoh(&view, host_slice);
            core::mem::forget(view); // `logits` owns the alloc
            r?;
        }
        self.pos = t_kv;
        self.last_t = t;
        Ok(())
    }

    /// Encodes the whole decode step (`n_text_layer` blocks + final LayerNorm +
    /// `t` tied-logits gemvs) on the owned CUDA stream and issues ONE
    /// `cuStreamSynchronize`. `t_kv = start + t`.
    fn run_decode_step(&self, t: usize, start: usize, t_kv: usize) -> Result<()> {
        let d = self.d;
        let n_head = self.n_head;
        let scale = self.scale;
        let eps = self.eps;
        let td = t * d;
        let f = size_of::<f32>();
        // Byte offset of row `start` inside a `[cap, d]` KV buffer — the k/v
        // projection GEMMs write their `[t, d]` output at this offset each
        // step, matching the CPU `KvCache::append` semantics on the device.
        let kv_off_bytes = (start * d * f) as CUdeviceptr;

        for layer in &self.layers {
            // --- causal self-attention over the growing KV cache ---
            // ln = layer_norm(h, self_ln)
            self.ctx.launch_layer_norm_async(
                self.h.ptr,
                layer.self_ln_g.ptr,
                layer.self_ln_b.ptr,
                self.ln.ptr,
                t,
                d,
                eps,
            )?;
            // Append this step's k/v rows AT cache row `start` (GEMM writes at
            // the resident-buffer's row-`start` byte offset — the same trick
            // `kv_append` uses on the encoder-KV cache).
            let self_k_out = layer.self_k.ptr.checked_add(kv_off_bytes).ok_or_else(|| {
                VokraError::InvalidArgument("decode step: self_k offset overflow".to_owned())
            })?;
            let self_v_out = layer.self_v.ptr.checked_add(kv_off_bytes).ok_or_else(|| {
                VokraError::InvalidArgument("decode step: self_v offset overflow".to_owned())
            })?;
            self.ctx.launch_gemm_async(
                self.ln.ptr,
                layer.self_k_w.ptr,
                bias_ptr_owned(layer.self_k_bias.as_ref(), self.dummy.ptr),
                self_k_out,
                t,
                d,
                d,
                layer.self_k_bias.is_some(),
            )?;
            self.ctx.launch_gemm_async(
                self.ln.ptr,
                layer.self_v_w.ptr,
                bias_ptr_owned(layer.self_v_bias.as_ref(), self.dummy.ptr),
                self_v_out,
                t,
                d,
                d,
                layer.self_v_bias.is_some(),
            )?;
            // Causal fused attention over the whole cache `[0, t_kv)`.
            // Session-scoped FA v2 opt-in: the constructor's probe (`d_head
            // == 64` + `MAX_SHARED_MEMORY_PER_BLOCK_OPTIN ≥ 40 KB`) already
            // decided whether this chain routes through
            // `launch_flash_attn_v2` — the flag is the only branch, so both
            // callsites in the session (this one and the cross-attention
            // below) read the same session-lifetime capability.
            self.ctx.launch_attn_chain(
                &AttnChainDims {
                    t_q: t,
                    t_kv,
                    d,
                    n_head,
                    scale,
                    has_q_bias: layer.self_q_bias.is_some(),
                    has_out_bias: layer.self_out_bias.is_some(),
                    causal: true,
                    q_offset: start,
                    use_flash_attn: self.use_flash_attn,
                },
                &AttnChainPtrs {
                    xq: self.ln.ptr,
                    q_w: layer.self_q_w.ptr,
                    q_bias: bias_ptr_owned(layer.self_q_bias.as_ref(), self.dummy.ptr),
                    k: layer.self_k.ptr,
                    v: layer.self_v.ptr,
                    out_w: layer.self_out_w.ptr,
                    out_bias: bias_ptr_owned(layer.self_out_bias.as_ref(), self.dummy.ptr),
                    q: self.q.ptr,
                    context: self.context.ptr,
                    qh: self.qh.ptr,
                    vh: self.vh.ptr,
                    kh_t: self.kh_t.ptr,
                    scores: self.scores.ptr,
                    probs: self.probs.ptr,
                    ctx_h: self.ctx_h.ptr,
                    out: self.block_out.ptr,
                },
            )?;
            self.ctx
                .launch_residual_add_async(self.h.ptr, self.block_out.ptr, td)?;

            // --- cross-attention over the (fixed) encoder output ---
            self.ctx.launch_layer_norm_async(
                self.h.ptr,
                layer.cross_ln_g.ptr,
                layer.cross_ln_b.ptr,
                self.ln.ptr,
                t,
                d,
                eps,
            )?;
            self.ctx.launch_attn_chain(
                &AttnChainDims {
                    t_q: t,
                    t_kv: self.n_ctx,
                    d,
                    n_head,
                    scale,
                    has_q_bias: layer.cross_q_bias.is_some(),
                    has_out_bias: layer.cross_out_bias.is_some(),
                    causal: false,
                    q_offset: 0,
                    use_flash_attn: self.use_flash_attn,
                },
                &AttnChainPtrs {
                    xq: self.ln.ptr,
                    q_w: layer.cross_q_w.ptr,
                    q_bias: bias_ptr_owned(layer.cross_q_bias.as_ref(), self.dummy.ptr),
                    k: layer.cross_k.ptr,
                    v: layer.cross_v.ptr,
                    out_w: layer.cross_out_w.ptr,
                    out_bias: bias_ptr_owned(layer.cross_out_bias.as_ref(), self.dummy.ptr),
                    q: self.q.ptr,
                    context: self.context.ptr,
                    qh: self.qh.ptr,
                    vh: self.vh.ptr,
                    kh_t: self.kh_t.ptr,
                    scores: self.scores.ptr,
                    probs: self.probs.ptr,
                    ctx_h: self.ctx_h.ptr,
                    out: self.block_out.ptr,
                },
            )?;
            self.ctx
                .launch_residual_add_async(self.h.ptr, self.block_out.ptr, td)?;

            // --- MLP ---
            self.ctx.launch_layer_norm_async(
                self.h.ptr,
                layer.mlp_ln_g.ptr,
                layer.mlp_ln_b.ptr,
                self.ln.ptr,
                t,
                d,
                eps,
            )?;
            self.ctx.launch_mlp_chain(
                &MlpChainDims {
                    t,
                    d,
                    ffn: self.ff,
                    has_fc1_bias: layer.fc1_bias.is_some(),
                    has_fc2_bias: layer.fc2_bias.is_some(),
                },
                &MlpChainPtrs {
                    x: self.ln.ptr,
                    fc1_w: layer.fc1_w.ptr,
                    fc1_bias: bias_ptr_owned(layer.fc1_bias.as_ref(), self.dummy.ptr),
                    fc2_w: layer.fc2_w.ptr,
                    fc2_bias: bias_ptr_owned(layer.fc2_bias.as_ref(), self.dummy.ptr),
                    h: self.mlp_h.ptr,
                    a: self.mlp_a.ptr,
                    out: self.block_out.ptr,
                },
            )?;
            self.ctx
                .launch_residual_add_async(self.h.ptr, self.block_out.ptr, td)?;
        }

        // Final LayerNorm into `normed`, then the tied-logits head on EVERY
        // decoded row (`t` gemvs into `logits[i·n_vocab .. (i+1)·n_vocab]`,
        // reading `normed[i·d .. (i+1)·d]`). One gemv per row keeps each
        // reduction identical to the CPU decoder's `t == 1` fast path — the
        // same math, just repeated `t` times inside the same stream, so the
        // whole step still synchronises exactly once at the end.
        self.ctx.launch_layer_norm_async(
            self.h.ptr,
            self.ln_post_g.ptr,
            self.ln_post_b.ptr,
            self.normed.ptr,
            t,
            d,
            eps,
        )?;
        for i in 0..t {
            let x_off = (i * d * f) as CUdeviceptr;
            let out_off = (i * self.n_vocab * f) as CUdeviceptr;
            let x_ptr = self.normed.ptr.checked_add(x_off).ok_or_else(|| {
                VokraError::InvalidArgument("decode step: normed offset overflow".to_owned())
            })?;
            let out_ptr = self.logits.ptr.checked_add(out_off).ok_or_else(|| {
                VokraError::InvalidArgument("decode step: logits offset overflow".to_owned())
            })?;
            self.ctx.launch_gemv_async(
                self.token_emb.ptr,
                x_ptr,
                self.dummy.ptr,
                out_ptr,
                self.n_vocab,
                d,
            )?;
        }
        self.ctx.sync_stream("cuStreamSynchronize decode step")
    }

    /// H2D copy `data` into an [`OwnedDeviceBuf`]'s prefix (used by [`Self::step`]
    /// to write the per-step embedding into the resident `h` buffer).
    fn htod_owned(&self, buf: &OwnedDeviceBuf, data: &[f32]) -> Result<()> {
        expect_len("decode htod", data.len().min(buf.len), data.len())?;
        let view = DeviceBuf {
            driver: &self.ctx.driver,
            ptr: buf.ptr,
        };
        let r = self.ctx.htod(&view, data);
        core::mem::forget(view); // buf owns the alloc
        r
    }

    /// The last decoded row of the last [`Self::step`] — `[n_vocab]` logits,
    /// the greedy / argmax read. Empty before any step (`last_t == 0`).
    #[must_use]
    pub fn last_logits(&self) -> &[f32] {
        if self.last_t == 0 {
            return &[];
        }
        let v = self.n_vocab;
        let start = (self.last_t - 1) * v;
        &self.logits_host[start..start + v]
    }

    /// All `[t, n_vocab]` rows the last [`Self::step`] wrote, row-major (row
    /// `i` at offset `i·n_vocab`). This is the full-row output the model-layer
    /// path compares against the CPU decoder's `[t, n_vocab]` logits (not just
    /// the last row). Empty before any step.
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

    /// The six load-bearing dims this session was built with — the exact
    /// match key for [`crate::session_pool::CudaDecodeSessionPool::acquire`]
    /// (M2-03-followup §D5, §R4). Every other constructor knob (`max_t_q`,
    /// `eps`, and the weight slices) is a property of the *first* build; a
    /// matching dim tuple guarantees the resident device buffers can be
    /// reused as-is after a [`Self::reset`].
    #[must_use]
    pub(crate) fn session_dims(&self) -> crate::session_pool::SessionDims {
        crate::session_pool::SessionDims {
            d: self.d,
            n_head: self.n_head,
            ff: self.ff,
            n_text_ctx: self.n_text_ctx,
            n_vocab: self.n_vocab,
            n_ctx: self.n_ctx,
        }
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

    /// Stream synchronisations issued through the owned context — one per
    /// [`Self::step`] (the session's construction issues none — every H2D
    /// upload is a synchronous `cuMemcpyHtoD` that does not touch the stream).
    #[must_use]
    pub fn submission_count(&self) -> u64 {
        self.ctx.submission_count()
    }
}

/// Owned-buffer holder used only while [`CudaDecodeSession::new`] assembles the
/// session: every scratch / weight buffer starts here (as `Option`, `take`n
/// into the final struct) so the whole allocation burst can be validated /
/// unwound as one — a `?` mid-way runs every already-taken `OwnedDeviceBuf`'s
/// Drop, releasing its device allocation.
struct CudaSessionBuffers {
    layers: Vec<DevDecoderLayer>,
    token_emb: Option<OwnedDeviceBuf>,
    ln_post_g: Option<OwnedDeviceBuf>,
    ln_post_b: Option<OwnedDeviceBuf>,
    h: Option<OwnedDeviceBuf>,
    ln: Option<OwnedDeviceBuf>,
    block_out: Option<OwnedDeviceBuf>,
    normed: Option<OwnedDeviceBuf>,
    q: Option<OwnedDeviceBuf>,
    context: Option<OwnedDeviceBuf>,
    qh: Option<OwnedDeviceBuf>,
    ctx_h: Option<OwnedDeviceBuf>,
    vh: Option<OwnedDeviceBuf>,
    kh_t: Option<OwnedDeviceBuf>,
    scores: Option<OwnedDeviceBuf>,
    probs: Option<OwnedDeviceBuf>,
    mlp_h: Option<OwnedDeviceBuf>,
    mlp_a: Option<OwnedDeviceBuf>,
    logits: Option<OwnedDeviceBuf>,
}

/// Picks the real `bias` buffer's device pointer or the shared 1-float `dummy`
/// (for an absent bias, bound but never read because `has_bias = 0`) — the
/// `OwnedDeviceBuf` sibling of [`bias_ptr`].
fn bias_ptr_owned(bias: Option<&OwnedDeviceBuf>, dummy: CUdeviceptr) -> CUdeviceptr {
    bias.map_or(dummy, |b| b.ptr)
}

/// The compiled modules + resolved kernel functions built by [`load_modules`]
/// (everything the [`CudaContext`] owns except the driver / context / stream).
struct Modules {
    gemm_module: CUmodule,
    kernels_module: CUmodule,
    gemm: CUfunction,
    gemv: CUfunction,
    softmax: CUfunction,
    softmax_causal: CUfunction,
    layer_norm: CUfunction,
    gelu: CUfunction,
    conv1d: CUfunction,
    col_gather: CUfunction,
    col_gather_t: CUfunction,
    col_scatter: CUfunction,
    add_assign: CUfunction,
    /// M3-01 out-of-place element-wise Add / Mul kernel handles
    /// (`vokra_add_f32`, `vokra_mul_f32`) backing the graph-executor arms of
    /// `OpKind::Add` / `OpKind::Mul` (see `crate::eval::eval_cuda_op`). These
    /// are DISTINCT from `add_assign` (which is in-place for the encoder
    /// residual chain); a graph-level Add reads two inputs and writes a fresh
    /// third, matching the CPU backend's `kernels::add_f32` contract.
    add: CUfunction,
    mul: CUfunction,
    /// M2-03 follow-up FA v2 fused causal attention kernel handle
    /// (`vokra_flash_attn_v2_causal_f32`, defined in `KERNELS_CUDA`). Resolved
    /// alongside the Phase-5 attention kernels so a single module load carries
    /// the whole decoder-step attention chain; the decoder-step session's
    /// `d_head == 64` + shared-memory probe decides whether it is dispatched.
    flash_attn_v2: CUfunction,
    /// M3-04 fused KV-cache dequant + GEMV kernel handles, one per quant format
    /// (`vokra_dequant_gemv_q4_0_f32` / `_q5_0_f32` / `_q8_0_f32`).
    dequant_gemv_q4_0: CUfunction,
    dequant_gemv_q5_0: CUfunction,
    dequant_gemv_q8_0: CUfunction,
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
    let softmax_causal = get_function(driver, kernels_module.module, c"vokra_softmax_causal_f32")?;
    let layer_norm = get_function(driver, kernels_module.module, c"vokra_layer_norm_f32")?;
    let gelu = get_function(driver, kernels_module.module, c"vokra_gelu_f32")?;
    let conv1d = get_function(driver, kernels_module.module, c"vokra_conv1d_f32")?;
    // The three Phase-5 attention column-mover kernels share the same module.
    let col_gather = get_function(driver, kernels_module.module, c"vokra_col_gather_f32")?;
    let col_gather_t = get_function(driver, kernels_module.module, c"vokra_col_gather_t_f32")?;
    let col_scatter = get_function(driver, kernels_module.module, c"vokra_col_scatter_f32")?;
    // The Phase-5-follow-on residual-add kernel shares the same module.
    let add_assign = get_function(driver, kernels_module.module, c"vokra_add_assign_f32")?;
    // M3-01 element-wise Add / Mul kernels for the graph-executor arm.
    let add = get_function(driver, kernels_module.module, c"vokra_add_f32")?;
    let mul = get_function(driver, kernels_module.module, c"vokra_mul_f32")?;
    // M2-03 follow-up: the FA v2 fused causal attention kernel lives in the
    // same module. The kernel is always resolved (a missing symbol means the
    // NVRTC compile lost it, which is a hard error — never a silent CPU
    // fallback, FR-EX-08); whether it is actually dispatched is decided by the
    // decoder-step session probe (`hd == 64` + shared-memory budget).
    let flash_attn_v2 = get_function(
        driver,
        kernels_module.module,
        c"vokra_flash_attn_v2_causal_f32",
    )?;
    // M3-04 fused KV-cache dequant + GEMV kernels — one per Q_0 format. Every
    // symbol is baked into the same PTX as the Phase-5 attention kernels; a
    // missing symbol means the NVRTC compile lost it, which is a hard error
    // (never a silent CPU fallback, FR-EX-08).
    let dequant_gemv_q4_0 = get_function(
        driver,
        kernels_module.module,
        c"vokra_dequant_gemv_q4_0_f32",
    )?;
    let dequant_gemv_q5_0 = get_function(
        driver,
        kernels_module.module,
        c"vokra_dequant_gemv_q5_0_f32",
    )?;
    let dequant_gemv_q8_0 = get_function(
        driver,
        kernels_module.module,
        c"vokra_dequant_gemv_q8_0_f32",
    )?;

    // All resolved: defuse the guards into the owned handle set.
    Ok(Modules {
        gemm_module: gemm_module.into_raw(),
        kernels_module: kernels_module.into_raw(),
        gemm,
        gemv,
        softmax,
        softmax_causal,
        layer_norm,
        gelu,
        conv1d,
        col_gather,
        col_gather_t,
        col_scatter,
        add_assign,
        add,
        mul,
        flash_attn_v2,
        dequant_gemv_q4_0,
        dequant_gemv_q5_0,
        dequant_gemv_q8_0,
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

/// A kernel-argument pointer to a `c_int` scalar (must outlive the launch).
/// The FA v2 kernel declares its integer args as `int` (host `c_int`, 4 bytes
/// on every supported target), so it consumes this shape verbatim.
fn int_arg(p: &c_int) -> *mut c_void {
    (p as *const c_int).cast::<c_void>().cast_mut()
}

/// A kernel-argument pointer to a `bool` scalar (must outlive the launch).
/// The FA v2 kernel declares its `causal` arg as `bool`; CUDA / C++ `bool`
/// matches the host `bool` (1-byte, non-zero = true) when the parameter is
/// passed via the `cuLaunchKernel` pointer-of-arg array.
fn bool_arg(p: &bool) -> *mut c_void {
    (p as *const bool).cast::<c_void>().cast_mut()
}

/// NVRTC-compiles a CUDA C `source` to a PTX byte buffer (NUL-terminated),
/// naming the translation unit `unit` and using `what` in any error. The program
/// handle is always destroyed before returning.
///
/// # M3-01-T07: `--gpu-architecture` gencode pin
///
/// The NVRTC options list is derived from the caller-visible probe
/// (`compute_capability_major.minor`) so the emitted PTX is targeted at the
/// running GPU rather than NVRTC's silent default (which floats across
/// toolkits — CUDA 12.6 defaults to `compute_52`, i.e. Maxwell, on an RTX 4090
/// host, wasting Ada SIMT features). The primary pin is `compute_89` (Ada,
/// SM 8.9 — RTX 4090); Ampere (SM 8.6) / Hopper (SM 9.0) resolve to their
/// own `compute_XX` value via [`gencode_flag`]. The env variable
/// `VOKRA_NVRTC_GPU_ARCH` overrides this for A/B testing (e.g. force
/// `compute_86` on a 4090 to validate parity across gencodes).
///
/// Explicitly *not* Hopper's WGMMA / TMA-specialised path — FA v3 code is
/// forbidden in this WP (ADR M3-01 (b), setting a Hopper gencode alone is safe
/// because the FA v2 kernel makes no WGMMA-only assumptions).
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

/// Resolves the NVRTC `--gpu-architecture=compute_XX` flag to pass this
/// compile (M3-01-T07). Priority order:
///
/// 1. `VOKRA_NVRTC_GPU_ARCH` env var (owner escape hatch — e.g. set to
///    `compute_86` on a 4090 to validate parity across gencodes).
/// 2. Best-effort probe: `vokra_cuda_probe` reports `compute_capability_major.minor`
///    — but that returns a `CudaCapabilities` and we do not want to pay a
///    dlopen ping inside every NVRTC compile. Instead we hard-code the M3-01
///    ADR primary target `compute_89` (Ada / RTX 4090); Ampere (SM 8.6) /
///    Hopper (SM 9.0) hosts get the same PTX in the current slice because
///    NVRTC does forward-compatible PTX → SASS translation at cuModuleLoadData
///    time (a compute_89 PTX loads fine on SM 9.0). The env-var escape hatch
///    covers A/B testing without a code change.
///
/// Returns an owned `CString` that must outlive the raw pointer put into the
/// NVRTC options array.
fn gencode_flag() -> std::ffi::CString {
    // SM 8.9 (Ada / RTX 4090) — the M3-01 ADR primary target. Anyone on a
    // different arch overrides with VOKRA_NVRTC_GPU_ARCH=compute_86 (Ampere) /
    // compute_90 (Hopper) etc.
    const DEFAULT_ARCH: &str = "compute_89";
    let arch = std::env::var("VOKRA_NVRTC_GPU_ARCH").unwrap_or_else(|_| DEFAULT_ARCH.to_owned());
    // Reject any interior NUL — else fall back to the default (an env-var typo
    // must never crash the compile).
    let sanitized = if arch.bytes().any(|b| b == 0) {
        DEFAULT_ARCH.to_owned()
    } else {
        arch
    };
    let flag = format!("--gpu-architecture={sanitized}");
    // `flag` is ASCII (`--gpu-architecture=` + `compute_XX`), so `CString::new`
    // will not fail; fall back to the default flag on the impossible NUL case
    // rather than propagating.
    std::ffi::CString::new(flag)
        .unwrap_or_else(|_| std::ffi::CString::new("--gpu-architecture=compute_89").unwrap())
}

/// Compiles `prog` with the M3-01-T07 gencode pin (`--gpu-architecture=compute_89`
/// by default, `VOKRA_NVRTC_GPU_ARCH` overrides) and extracts its PTX. On a
/// compile failure the NVRTC log is surfaced in the error (labelled `what`).
fn compile_and_extract_ptx(nvrtc: &Nvrtc, prog: sys::NvrtcProgram, what: &str) -> Result<Vec<u8>> {
    let arch = gencode_flag();
    // `options` holds raw `*const c_char` pointers into `arch`'s owned buffer;
    // `arch` must live until nvrtcCompileProgram returns.
    let options: [*const c_char; 1] = [arch.as_ptr()];
    // SAFETY: `prog` is a valid program; `options` is a 1-slot array of
    // NUL-terminated C strings owned by `arch` (alive for the call).
    let compile_res = unsafe { (nvrtc.compile_program)(prog, 1, options.as_ptr()) };
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

/// Element-wise binary op shape validator (M3-01: [`CudaContext::add_f32`] /
/// [`CudaContext::mul_f32`]). Both operands and the output must be the same
/// length — the exact contract of `vokra_backend_cpu::kernels::{add_f32,
/// mul_f32}` (no broadcast in the FP32 MVP; the graph-executor rejects
/// mismatched shapes before `eval_op` reaches this kernel).
fn validate_binary(a: &[f32], b: &[f32], out: &[f32]) -> Result<()> {
    expect_len("binary b", b.len(), a.len())?;
    expect_len("binary out", out.len(), a.len())
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

// =====================================================================
// M3-04 fused KV-cache dequant + GEMV trait impl (CUDA backend arm)
// =====================================================================
//
// The concrete GPU implementation of the
// [`vokra_core::KvQuantDequantGemvOps`] trait: dispatches into
// [`CudaContext::dequant_gemv_f32`] (defined above). Kept at the bottom of
// the file so it sits alongside the other trait impls / helpers rather than
// inside the impl block that owns the launcher — keeps grep-locality with the
// Metal analogue.
impl vokra_core::KvQuantDequantGemvOps for CudaContext {
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
