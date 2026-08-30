//! CPU compute kernels and their safe public wrappers (M0-08-T02, T05..T16).
//!
//! # Confirmed spike kernel set (M0-08-T02)
//!
//! Back-derived from the ops Whisper base (M0-06) and Silero VAD (M0-05)
//! need. dtype is **f32 only** in the spike (aligned with the FP32 parity
//! bound NFR-QL-01 atol = 0.01; f16 / K-quant kernels are FR-QT-01 = v0.1
//! MVP and later). Threading is intentionally **not** introduced in M0 (the
//! rayon / OpenMP-alternative decision is deferred — NFR-LC-03).
//!
//! | kernel | SIMD? | rationale |
//! |--------|-------|-----------|
//! | [`gemm_f32`] (bias = linear) | yes | dominant Whisper attention / FFN cost |
//! | [`gemv_f32`] (bias = per-row) | yes | tied logits head `token_emb[v,d] @ h[d]` (the `gemm` `n=1` scalar-tail case, M1) |
//! | [`add_f32`] / [`mul_f32`] | yes | residual add, gating |
//! | [`relu_f32`] | yes | Silero VAD conv stack |
//! | [`elu_f32`] | scalar | Bark's embedded EnCodec decoder (`alpha = 1`) |
//! | [`sigmoid_f32`] | scalar-backed; SIMD under `simd-transcendental` | VAD output / LSTM gate; exp-bound (`vexp`, M1-05-EXP) |
//! | [`tanh_f32`] | scalar-backed; SIMD under `simd-transcendental` | LSTM cell; exp-bound (`vexp`, M1-05-EXP) |
//! | [`gelu_f32`] | scalar-backed; SIMD under `simd-transcendental` | Whisper MLP (exact/erf form); exp-bound (`vexp`, M1-05-EXP) |
//! | [`gelu_new_f32`] | scalar | GPT-2 / Transformers tanh-approximation GELU |
//! | [`softmax_f32`] | yes (exp scalar; SIMD under `simd-transcendental`) | Whisper attention |
//! | [`layer_norm_f32`] | yes | Whisper pre-norm blocks |
//! | [`scale_norm_f32`] | scalar reduction | MossFormer2 FLASH projections |
//! | [`conv1d_f32`] | via GEMM | Whisper encoder stem; im2col + [`gemm_f32`] |
//! | [`conv1d_f32_dilated`] / [`conv_transpose1d_f32`] | scalar | vocoder convolution seam |
//!
//! **Deliberately not SIMD kernels here** (memory-bound / structural, left to
//! scalar or the model layer's `vokra-ops` reference — M0-06-T03): embedding
//! lookup, transpose, reshape.
//!
//! `conv1d_f32` has no dedicated SIMD kernel: it lowers to im2col + the
//! dispatched [`gemm_f32`], so it inherits AVX2 / NEON automatically
//! (M0-08-T08/T12/T15).
//!
//! # Boundary with `vokra-ops`
//!
//! This crate owns the **dispatch-target compute kernels** (the functions
//! below). `vokra-ops` owns the **operator definitions** (front-end / speech
//! ops and their attributes) and any scalar op *reference* used by the parity
//! harness. New "missing op" requests raised by M0-06-T02 are folded in by
//! appending to the table above and adding the kernel, up to (but not after)
//! WP completion (M0-08-T19), to avoid re-opening a finished WP.
//!
//! # Function boundary for M0-06
//!
//! M0-06's encoder / decoder call these safe wrappers directly:
//! [`gemm_f32`], [`add_f32`], [`mul_f32`], [`relu_f32`], [`elu_f32`], [`sigmoid_f32`],
//! [`tanh_f32`], [`gelu_f32`], [`softmax_f32`], [`layer_norm_f32`],
//! [`conv1d_f32`], plus [`crate::active_isa`] for the demo's ISA log. Each
//! validates its shapes at the boundary and returns
//! [`VokraError::InvalidArgument`] on a mismatch (NFR-RL-07); the `*_on`
//! variants force a specific [`IsaPath`] for differential testing.

pub(crate) mod scalar;

// M5-14 Wave-1 (T05/T09/T10): packed cache-blocked GEMM driver — routing,
// pack routines and thread-local scratch. Per-ISA micro-kernels live in the
// arch modules below; ISAs without them keep the legacy kernels bit-for-bit.
pub(crate) mod gemm_driver;

// Native vectorized `exp` shared by the AVX2 / NEON transcendental kernels
// (M1-05-EXP). Compiled only under the `simd-transcendental` feature; without
// it, `sigmoid` / `tanh` / `gelu` / softmax-exp stay scalar-backed and this
// module is not built.
#[cfg(feature = "simd-transcendental")]
pub(crate) mod vexp;

// M5-03-T06 (Wave 1: exp/tanh/sqrt) + WP-06 (2026-08-09: sin/cos/log/log1p):
// self-contained scalar transcendentals in pure `core` arithmetic (no `std`,
// no `libm`). **WP-07** hoisted the implementations into the new `vokra-math`
// crate — this module is now a thin `pub(crate) use vokra_math::…` re-export
// shim so any internal caller reaching them via
// `crate::kernels::scalar_transcendental::{…}` keeps compiling unchanged.
// `#[allow(dead_code)]` covers the interim: internal callers migrate to
// naming `vokra_math::*` directly (SBV2 hot path, WP-05 owner decision),
// leaving the shim's re-exports unused in non-test builds; the wrap of
// `#[allow(unused_imports)]` on the `use` line inside the shim is the peer
// tolerance for that. Production wiring (Silero forward T08, SBV2 kernels)
// still calls the primitives at runtime — through `vokra-math` after WP-07.
#[allow(dead_code)]
pub(crate) mod scalar_transcendental;

#[cfg(target_arch = "x86_64")]
pub(crate) mod avx2;

// M4-17-T07..T11: AVX-512 f32 kernel tier (F/DQ/BW/VL bundle) + the VNNI
// INT8 / BF16 matmul cores. Compiled only on x86-64; runtime entry is gated
// by `CpuFeatures::supports(IsaPath::Avx512*)` (the SIGILL guard), and the
// binary itself stays at the SSE2 baseline — the AVX-512 encodings exist
// only inside the per-function `#[target_feature]` cores (ADR M4-17 §(g)).
#[cfg(target_arch = "x86_64")]
pub(crate) mod avx512;

#[cfg(target_arch = "aarch64")]
pub(crate) mod neon;

// M4-17-T13: AVX-VNNI 256-bit INT8 group-sum core (Alder Lake+ client INT8
// main path; AVX-512 is fused off on those parts, ADR M4-17 §(d)).
#[cfg(target_arch = "x86_64")]
pub(crate) mod avxvnni256;

// M4-17-T14..T17: ARM64 server-tier cores. dotprod / i8mm / fp16 / bf16
// vector intrinsics are unstable on the pinned rustc, so these use
// `core::arch::asm!` with `.arch_extension` fences (M3-13 RVV precedent,
// ADR M4-17 §(g)); each encoding is reached only after the corresponding
// `CpuFeatures::supports` gate (SIGILL guard).
#[cfg(target_arch = "aarch64")]
pub(crate) mod neon_bf16;
#[cfg(target_arch = "aarch64")]
pub(crate) mod neon_dotprod;
#[cfg(target_arch = "aarch64")]
pub(crate) mod neon_fp16;
#[cfg(target_arch = "aarch64")]
pub(crate) mod neon_i8mm;

// M4-17-T10..T17: K-quants SIMD dequant fusion (bit-identical to the
// vokra-core scalar reference) + the specialized INT8 / BF16 / FP16
// dispatch surface. Target-independent orchestration (the per-arch cores
// live in the modules above / in `avx512`).
pub(crate) mod kquant;

pub use kquant::{
    BF16_REL, FP16_REL, GemmF32Bf16BitsScratch, KQUANT_GROUP, KQuantDtype, bf16_to_f32,
    dot_precision_bound, f16_to_f32 as f16_bits_to_f32, f32_to_bf16_rne, f32_to_f16_rne,
    f64_to_f16_rne, fp16_fma_emu, gemm_bf16_bits_on, gemm_bf16_on, gemm_f32_bf16_bits,
    gemm_f32_bf16_bits_on, gemm_f32_bf16_bits_on_with_scratch,
    gemm_f32_bf16_bits_on_with_scratch_strided, gemm_f32_bf16_bits_with_scratch, gemm_fp16_on,
    int8_error_bound, kquant_dequant_on, kquant_gemm_i8, kquant_gemm_i8_on, kquant_gemv_i8,
    kquant_gemv_i8_on, kquant_gemv2_i8_on, kquant_gemvn_i8_on,
};

// M3-13-T04..T09: RISC-V RVV 1.0 base kernels + Zvfh feature-gated fp16 path.
// Compiled only on `target_arch = "riscv64"` — the runtime dispatch layer
// gates entry via [`crate::features::CpuFeatures::supports`] (needs
// `rvv_v = true`), so no other target ever routes through this module. See
// `docs/adr/M3-13-riscv-rvv-1.0.md` for the scaffold vs. full-kernel split
// (`add` uses inline RVV asm; remaining ops delegate to `scalar::*` pending
// M4+ inline-asm rewrites).
#[cfg(target_arch = "riscv64")]
pub(crate) mod rvv;

// M4-08-T07/T08: RISC-V RVV **draft 0.7.1** kernels (T-Head C910/C906 =
// LicheePi 4A / Milk-V Duo). A peer tier to `rvv` — NOT a subset of it: the
// 0.7.1 and ratified-1.0 instruction encodings are incompatible, so the two
// modules share no instruction bytes (ADR M4-08 §d). Compiled only on
// `target_arch = "riscv64"`; the runtime dispatch layer gates entry via
// `CpuFeatures::supports` (needs `rvv_071 = true` — the xtheadvector /
// cpu-vector detection signals with the RVV 1.0 misdetection guard).
// Scaffold split mirrors `rvv`: `add` emits real 0.7.1 words via `.insn`,
// the rest delegate to `scalar::*` pending M4+/M5 rewrites.
#[cfg(target_arch = "riscv64")]
pub(crate) mod rvv071;

// M4-01-T04/T05: WASM SIMD128 f32x4 kernels (`core::arch::wasm32` intrinsics,
// std-builtin — no external crate). Compiled ONLY when the wasm32 artifact is
// built with `-C target-feature=+simd128`: WASM has no runtime CPU feature
// detection (SIMD acceptance is decided at module validation), so the
// simd/base split is a 2-artifact distribution + JS loader select
// (scripts/build-wasm.sh + web/pkg/index.js `WebAssembly.validate` probe —
// ADR M4-01-webgpu-wasm §4), NOT an AVX2/NEON-style runtime dispatch.
// Relaxed SIMD is not adopted (deterministic mul + add only, NFR-QL-01).
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub(crate) mod wasm_simd128;

// Fused log-mel inner kernel (M2-04-T06): AVX2 8-lane FMA `_mm256_fmadd_ps`
// over the filterbank weights row + `vlog10_avx2` polynomial approximation.
// This is the CPU-side SIMD path for the log-mel front-end fusion. NEON
// counterpart lives in a companion module; a portable scalar reference is
// bundled alongside the AVX2 kernel so the ISA-parity test in
// `tests/fused_logmel_isa_parity.rs` can cross-check without touching
// vokra-ops (zero-dep at the crate boundary).
#[cfg(target_arch = "x86_64")]
pub(crate) mod fused_logmel_avx2;

// NEON companion (M2-04-T06): four-lane `vfmaq_f32` mel accumulation plus
// `vlog10_neon` polynomial approximation reusing the `vexp_neon` IEEE-754
// exponent-field pattern. Compiled only on AArch64.
#[cfg(target_arch = "aarch64")]
pub(crate) mod fused_logmel_neon;

use vokra_core::{Result, VokraError};

use crate::dispatch;
use crate::features::IsaPath;

// ---- production GEMM / GEMV execution (row-parallel when `parallel` is on) ----
//
// The `*_f32` public wrappers below route through these. GEMM goes through
// the M5-14 packed driver (`kernels::gemm_driver`): m == 1 takes the ISA row
// kernel, large shapes the packed cache-blocked path (pool-parallel over
// disjoint output tiles), everything else the legacy pool row-split — all
// three routes are **bit-identical** per element to the legacy kernel, so
// parity is preserved (asserted by `tests/gemm_packed_parity.rs`). The
// `*_f32_on` differential entry points deliberately do NOT use the driver or
// the pool: they stay on the forced-ISA single-thread legacy kernel, which is
// the numeric reference the production path is compared against. Off
// `parallel` (or on WASM / single core) the driver runs its loops inline.

#[allow(clippy::too_many_arguments)]
fn run_gemm(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    bias: Option<&[f32]>,
    out: &mut [f32],
) {
    gemm_driver::run(m, n, k, a, b, bias, out);
}

#[cfg(all(feature = "parallel", not(target_family = "wasm")))]
fn run_gemv(m: usize, k: usize, a: &[f32], x: &[f32], bias: Option<&[f32]>, out: &mut [f32]) {
    crate::pool::parallel_gemv(dispatch::table().gemv, m, k, a, x, bias, out);
}

#[cfg(not(all(feature = "parallel", not(target_family = "wasm"))))]
fn run_gemv(m: usize, k: usize, a: &[f32], x: &[f32], bias: Option<&[f32]>, out: &mut [f32]) {
    (dispatch::table().gemv)(m, k, a, x, bias, out);
}

/// Default layer-norm epsilon (PyTorch `nn.LayerNorm` default `1e-5`, which
/// OpenAI Whisper inherits). Exposed for M0-06 call sites.
pub const LAYER_NORM_DEFAULT_EPS: f32 = scalar::LAYER_NORM_DEFAULT_EPS;

// ---- boundary validation helpers (NFR-RL-07) ----

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

fn validate_binary(a: &[f32], b: &[f32], out: &[f32]) -> Result<()> {
    expect_len("binary b", b.len(), a.len())?;
    expect_len("binary out", out.len(), a.len())
}

fn validate_unary(x: &[f32], out: &[f32]) -> Result<()> {
    expect_len("unary out", out.len(), x.len())
}

fn validate_rows_cols(input: &[f32], out: &[f32], rows: usize, cols: usize) -> Result<()> {
    let total = checked_mul(rows, cols, "rows*cols")?;
    expect_len("input", input.len(), total)?;
    expect_len("out", out.len(), total)
}

// ---- dot product & GEMM (M0-08-T05) ----

/// Dot product of two equal-length f32 slices.
///
/// A scalar building block (no dispatch table entry); a length mismatch is an
/// explicit [`VokraError::InvalidArgument`].
pub fn vec_dot_f32(a: &[f32], b: &[f32]) -> Result<f32> {
    expect_len("vec_dot b", b.len(), a.len())?;
    Ok(scalar::vec_dot(a, b))
}

/// Row-major GEMM with optional per-column bias (bias = affine `linear`):
/// `out[i, j] = bias[j] + sum_l a[i, l] * b[l, j]`.
///
/// `a` is `m x k`, `b` is `k x n`, `out` is `m x n`, and `bias` (when `Some`)
/// has length `n`. Runs on [`crate::active_isa`]. A shape mismatch is an
/// explicit [`VokraError::InvalidArgument`].
pub fn gemm_f32(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    bias: Option<&[f32]>,
    out: &mut [f32],
) -> Result<()> {
    validate_gemm(m, n, k, a, b, bias, out)?;
    run_gemm(m, n, k, a, b, bias, out);
    Ok(())
}

/// Row-major GEMM against a **K-quantized** weight matrix, with an optional
/// per-column bias: `out[t, j] = bias[j] + Σ_l a[t, l] · dequant(wq[j, l])`
/// (M5-15-T33, FR-QT-01).
///
/// This is the fused dequant-dot counterpart of [`gemm_f32`] for an
/// `nn.Linear` whose weight stayed in its on-disk K-quant form:
///
/// - `a` is `[m, k]` (activations / tokens), `out` is `[m, n]` — identical to
///   [`gemm_f32`];
/// - `wq` is the **untransposed** `[n, k]` GGUF payload (`n` output features ×
///   `k / 256` super-blocks per row), *not* the `[k, n]` `b` [`gemm_f32`] takes.
///   The quant path therefore skips the `[out, in] → [in, out]` transpose the
///   f32 loader performs;
/// - `k` must be a positive multiple of 256 (`QK_K`) so super-blocks never
///   straddle a weight row — an explicit error otherwise, never a silent
///   widen (FR-EX-08).
///
/// **Not bit-identical to [`gemm_f32`] over the dequantized weight**: the INT8
/// surface is bounded by the activation-quantization band
/// ([`int8_error_bound`]), not by bit identity, so enabling this route changes
/// model output. See `docs/adr/M5-15-quant.md`.
#[allow(clippy::too_many_arguments)] // mirrors gemm_f32 plus the weight dtype
pub fn gemm_q_f32(
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    wq: &[u8],
    dtype: KQuantDtype,
    bias: Option<&[f32]>,
    out: &mut [f32],
) -> Result<()> {
    expect_len("gemm_q a", a.len(), checked_mul(m, k, "gemm_q m*k")?)?;
    expect_len("gemm_q out", out.len(), checked_mul(m, n, "gemm_q m*n")?)?;
    if let Some(bias) = bias {
        expect_len("gemm_q bias", bias.len(), n)?;
    }
    // `wq`'s length is checked against `(n, k, dtype)` by the kernel's own
    // `validate_gemv_i8`, which owns the super-block arithmetic.
    gemm_driver::run_q(dtype, m, n, k, a, wq, bias, out)
}

/// [`gemm_f32`] forced onto a specific `isa` (differential testing).
///
/// Always single-thread (never the pool): this is the numeric reference the
/// row-parallel production path is checked bit-for-bit against.
#[allow(clippy::too_many_arguments)] // mirrors gemm_f32 plus the forced isa
pub fn gemm_f32_on(
    isa: IsaPath,
    m: usize,
    n: usize,
    k: usize,
    a: &[f32],
    b: &[f32],
    bias: Option<&[f32]>,
    out: &mut [f32],
) -> Result<()> {
    validate_gemm(m, n, k, a, b, bias, out)?;
    (dispatch::table_for(isa)?.gemm)(m, n, k, a, b, bias, out);
    Ok(())
}

/// Row-major matrix-vector product with an optional per-row bias:
/// `out[i] = bias[i] + sum_l a[i, l] * x[l]`.
///
/// `a` is `m x k`, `x` has length `k`, `out` has length `m`, and `bias` (when
/// `Some`) has length `m`. This is the `n = 1` case of [`gemm_f32`], but rather
/// than falling through that kernel's scalar column tail it streams each row of
/// `a` contiguously and reduces it with a wide SIMD FMA + horizontal sum. It is
/// the fast path for Whisper's tied logits head (`token_emb[v, d] @ h[d]`, the
/// single biggest per-token decode matmul). Runs on [`crate::active_isa`]; a
/// shape mismatch is an explicit [`VokraError::InvalidArgument`].
pub fn gemv_f32(
    m: usize,
    k: usize,
    a: &[f32],
    x: &[f32],
    bias: Option<&[f32]>,
    out: &mut [f32],
) -> Result<()> {
    validate_gemv(m, k, a, x, bias, out)?;
    run_gemv(m, k, a, x, bias, out);
    Ok(())
}

/// [`gemv_f32`] forced onto a specific `isa` (differential testing).
///
/// Always single-thread (never the pool): the numeric reference for the
/// row-parallel production path.
#[allow(clippy::too_many_arguments)] // mirrors gemv_f32 plus the forced isa
pub fn gemv_f32_on(
    isa: IsaPath,
    m: usize,
    k: usize,
    a: &[f32],
    x: &[f32],
    bias: Option<&[f32]>,
    out: &mut [f32],
) -> Result<()> {
    validate_gemv(m, k, a, x, bias, out)?;
    (dispatch::table_for(isa)?.gemv)(m, k, a, x, bias, out);
    Ok(())
}

// ---- element-wise & activations (M0-08-T06) ----

macro_rules! binary_wrapper {
    ($name:ident, $name_on:ident, $field:ident, $doc:literal) => {
        #[doc = $doc]
        ///
        /// `a`, `b`, `out` must have equal length. Runs on
        /// [`crate::active_isa`].
        pub fn $name(a: &[f32], b: &[f32], out: &mut [f32]) -> Result<()> {
            validate_binary(a, b, out)?;
            (dispatch::table().$field)(a, b, out);
            Ok(())
        }

        #[doc = concat!("[`", stringify!($name), "`] forced onto a specific `isa`.")]
        pub fn $name_on(isa: IsaPath, a: &[f32], b: &[f32], out: &mut [f32]) -> Result<()> {
            validate_binary(a, b, out)?;
            (dispatch::table_for(isa)?.$field)(a, b, out);
            Ok(())
        }
    };
}

macro_rules! unary_wrapper {
    ($name:ident, $name_on:ident, $field:ident, $doc:literal) => {
        #[doc = $doc]
        ///
        /// `x` and `out` must have equal length. Runs on
        /// [`crate::active_isa`].
        pub fn $name(x: &[f32], out: &mut [f32]) -> Result<()> {
            validate_unary(x, out)?;
            (dispatch::table().$field)(x, out);
            Ok(())
        }

        #[doc = concat!("[`", stringify!($name), "`] forced onto a specific `isa`.")]
        pub fn $name_on(isa: IsaPath, x: &[f32], out: &mut [f32]) -> Result<()> {
            validate_unary(x, out)?;
            (dispatch::table_for(isa)?.$field)(x, out);
            Ok(())
        }
    };
}

binary_wrapper!(add_f32, add_f32_on, add, "Element-wise `out = a + b`.");
binary_wrapper!(mul_f32, mul_f32_on, mul, "Element-wise `out = a * b`.");
unary_wrapper!(
    relu_f32,
    relu_f32_on,
    relu,
    "Element-wise ReLU `out = max(0, x)`."
);

/// Element-wise ELU with the EnCodec/Bark default `alpha = 1`.
///
/// `out = x` for positive inputs and `out = exp(x) - 1` otherwise. This
/// scalar implementation is the portable CPU reference for the dedicated
/// Metal kernel; the public boundary rejects shape mismatches explicitly.
pub fn elu_f32(x: &[f32], out: &mut [f32]) -> Result<()> {
    validate_unary(x, out)?;
    scalar::elu(x, out);
    Ok(())
}

unary_wrapper!(
    sigmoid_f32,
    sigmoid_f32_on,
    sigmoid,
    "Element-wise logistic sigmoid `out = 1 / (1 + exp(-x))`."
);
unary_wrapper!(
    tanh_f32,
    tanh_f32_on,
    tanh,
    "Element-wise hyperbolic tangent."
);
unary_wrapper!(
    gelu_f32,
    gelu_f32_on,
    gelu,
    "Element-wise exact (erf-based) GELU, matching Whisper's `nn.GELU()`."
);

/// Element-wise GPT-2 / Transformers `gelu_new` tanh approximation.
///
/// Kept separate from [`gelu_f32`] because substituting exact/erf GELU changes
/// the released model numerics. The scalar kernel is the portable CPU
/// reference; Metal has a matching dedicated kernel.
pub fn gelu_new_f32(x: &[f32], out: &mut [f32]) -> Result<()> {
    validate_unary(x, out)?;
    scalar::gelu_new(x, out);
    Ok(())
}

// ---- softmax (M0-08-T07) ----

/// Row-wise softmax over the innermost dimension of a `rows x cols`
/// row-major buffer (numerically stabilised by the row max). Each output row
/// sums to 1 within FP32 rounding. Runs on [`crate::active_isa`].
pub fn softmax_f32(input: &[f32], out: &mut [f32], rows: usize, cols: usize) -> Result<()> {
    validate_rows_cols(input, out, rows, cols)?;
    (dispatch::table().softmax)(input, out, rows, cols);
    Ok(())
}

/// [`softmax_f32`] forced onto a specific `isa`.
pub fn softmax_f32_on(
    isa: IsaPath,
    input: &[f32],
    out: &mut [f32],
    rows: usize,
    cols: usize,
) -> Result<()> {
    validate_rows_cols(input, out, rows, cols)?;
    (dispatch::table_for(isa)?.softmax)(input, out, rows, cols);
    Ok(())
}

// ---- layer norm (M0-08-T07) ----

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

/// Row-wise layer normalisation with affine parameters over the innermost
/// dimension: `out[r, c] = (x[r, c] - mean_r) / sqrt(var_r + eps) *
/// gamma[c] + beta[c]`, using the biased (population) variance to match
/// PyTorch `nn.LayerNorm`. `gamma` / `beta` have length `cols`. See
/// [`LAYER_NORM_DEFAULT_EPS`]. Runs on [`crate::active_isa`].
pub fn layer_norm_f32(
    input: &[f32],
    out: &mut [f32],
    rows: usize,
    cols: usize,
    gamma: &[f32],
    beta: &[f32],
    eps: f32,
) -> Result<()> {
    validate_layer_norm(input, out, rows, cols, gamma, beta)?;
    (dispatch::table().layer_norm)(input, out, rows, cols, gamma, beta, eps);
    Ok(())
}

/// [`layer_norm_f32`] forced onto a specific `isa`.
#[allow(clippy::too_many_arguments)] // mirrors layer_norm_f32 plus the forced isa
pub fn layer_norm_f32_on(
    isa: IsaPath,
    input: &[f32],
    out: &mut [f32],
    rows: usize,
    cols: usize,
    gamma: &[f32],
    beta: &[f32],
    eps: f32,
) -> Result<()> {
    validate_layer_norm(input, out, rows, cols, gamma, beta)?;
    (dispatch::table_for(isa)?.layer_norm)(input, out, rows, cols, gamma, beta, eps);
    Ok(())
}

// ---- one-group GroupNorm (SepFormer mask network) ---------------------------

/// Affine GroupNorm with one group over channel-major `[channels, positions]`.
///
/// The reduction uses 256 strided partial sums followed by a fixed pairwise
/// tree.  This avoids feeding SepFormer's 130k–384k-element group through the
/// ordinary one-row LayerNorm accumulator, whose long FP32 left fold loses
/// enough precision to be amplified by the dual-path stack.  The Metal sibling
/// uses the same reduction topology.
#[allow(clippy::too_many_arguments)]
pub fn group_norm_f32(
    input: &[f32],
    out: &mut [f32],
    channels: usize,
    positions: usize,
    gamma: &[f32],
    beta: &[f32],
    eps: f32,
) -> Result<()> {
    const PARTIALS: usize = 256;

    if channels == 0 || positions == 0 {
        return Err(VokraError::InvalidArgument(format!(
            "group_norm channels and positions must be non-zero, got {channels}x{positions}"
        )));
    }
    let total = checked_mul(channels, positions, "group_norm channels*positions")?;
    expect_len("group_norm input", input.len(), total)?;
    expect_len("group_norm out", out.len(), total)?;
    expect_len("group_norm gamma", gamma.len(), channels)?;
    expect_len("group_norm beta", beta.len(), channels)?;

    let mut partial = [0.0f32; PARTIALS];
    for (lane, lane_sum) in partial.iter_mut().enumerate() {
        let mut index = lane;
        while index < total {
            *lane_sum += input[index];
            index += PARTIALS;
        }
    }
    let mut width = PARTIALS / 2;
    while width > 0 {
        for index in 0..width {
            partial[index] += partial[index + width];
        }
        width /= 2;
    }
    let mean = partial[0] / total as f32;

    partial.fill(0.0);
    for (lane, lane_sum) in partial.iter_mut().enumerate() {
        let mut index = lane;
        while index < total {
            let delta = input[index] - mean;
            *lane_sum += delta * delta;
            index += PARTIALS;
        }
    }
    let mut width = PARTIALS / 2;
    while width > 0 {
        for index in 0..width {
            partial[index] += partial[index + width];
        }
        width /= 2;
    }
    let inv_std = 1.0 / (partial[0] / total as f32 + eps).sqrt();
    for channel in 0..channels {
        for position in 0..positions {
            let index = channel * positions + position;
            out[index] = (input[index] - mean) * inv_std * gamma[channel] + beta[channel];
        }
    }
    Ok(())
}

// ---- ScaleNorm (MossFormer2 FLASH projections) -----------------------------

/// Row-wise ScaleNorm:
/// `out[r,c] = input[r,c] / max(||row||₂ · cols⁻¹ᐟ², eps) · gain`.
///
/// This is deliberately distinct from RMSNorm. ScaleNorm clamps the completed
/// norm to `eps`, whereas RMSNorm adds epsilon inside the square root. Keeping
/// a separate kernel preserves the released ClearerVoice-Studio equation and
/// lets the Metal backend execute the reduction without a host fallback.
pub fn scale_norm_f32(
    input: &[f32],
    out: &mut [f32],
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
    validate_rows_cols(input, out, rows, cols)?;
    let dimension_scale = (cols as f64).sqrt().recip() as f32;
    for row in 0..rows {
        let start = row * cols;
        let source = &input[start..start + cols];
        let squared_norm = source.iter().map(|value| value * value).sum::<f32>();
        let denominator = (squared_norm.sqrt() * dimension_scale).max(eps);
        for col in 0..cols {
            out[start + col] = source[col] / denominator * gain;
        }
    }
    Ok(())
}

// ---- conv1d via im2col + GEMM (M0-08-T08) ----

/// 1-D convolution via im2col + [`gemm_f32`], so it rides the dispatched
/// SIMD GEMM (no dedicated conv SIMD kernel).
///
/// Layout: `input` is `in_ch x in_len` row-major, `weight` is
/// `out_ch x in_ch x kernel`, optional `bias` has length `out_ch`, and `out`
/// is `out_ch x out_len` where
/// `out_len = (in_len + 2 * padding - kernel) / stride + 1`. `stride` and
/// `kernel` must be non-zero and the padded length must be at least `kernel`;
/// any shape mismatch is an explicit [`VokraError::InvalidArgument`]. The
/// im2col buffer is allocated per call in M0; a static arena (M1-04, FR-EX-05)
/// can replace it later without changing this signature.
#[allow(clippy::too_many_arguments)] // convolution's intrinsic parameter set
pub fn conv1d_f32(
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
    conv1d_dispatch(
        None, input, in_ch, in_len, weight, out_ch, kernel, bias, stride, padding, out,
    )
}

/// Stride/dilation-aware 1-D convolution with PyTorch's channel-major layout.
///
/// This is the same zero-padded cross-correlation contract as [`conv1d_f32`],
/// with the effective kernel width `1 + (kernel - 1) * dilation`.  The dense
/// `dilation == 1` path intentionally delegates to the established im2col +
/// GEMM implementation; the dilated path keeps the omitted taps out of the
/// accumulation rather than materialising a sparse expanded weight matrix.
#[allow(clippy::too_many_arguments)]
pub fn conv1d_f32_dilated(
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
    if dilation == 0 {
        return Err(VokraError::InvalidArgument(
            "conv1d dilation must be >= 1".into(),
        ));
    }
    if in_ch == 0 || out_ch == 0 || in_len == 0 {
        return Err(VokraError::InvalidArgument(
            "conv1d in_ch, out_ch, and in_len must be > 0".into(),
        ));
    }
    if dilation == 1 {
        return conv1d_f32(
            input, in_ch, in_len, weight, out_ch, kernel, bias, stride, padding, out,
        );
    }
    if stride == 0 || kernel == 0 {
        return Err(VokraError::InvalidArgument(
            "conv1d stride and kernel must be > 0".into(),
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
    let input_end = padding
        .checked_add(in_len)
        .ok_or_else(|| VokraError::InvalidArgument("conv1d input extent overflow".into()))?;

    for oc in 0..out_ch {
        for t in 0..out_len {
            let mut acc = bias.map_or(0.0, |values| values[oc]);
            let window_start = t.checked_mul(stride).ok_or_else(|| {
                VokraError::InvalidArgument("conv1d dispatch index overflow".into())
            })?;
            for ic in 0..in_ch {
                for k in 0..kernel {
                    let padded_index = window_start
                        .checked_add(checked_mul(k, dilation, "conv1d tap")?)
                        .ok_or_else(|| {
                            VokraError::InvalidArgument("conv1d tap index overflow".into())
                        })?;
                    if padded_index < padding || padded_index >= input_end {
                        continue;
                    }
                    let input_index = padded_index - padding;
                    let weight_index = (oc * in_ch + ic) * kernel + k;
                    acc += input[ic * in_len + input_index] * weight[weight_index];
                }
            }
            out[oc * out_len + t] = acc;
        }
    }
    Ok(())
}

/// PyTorch-layout `ConvTranspose1d` (`[in_ch, out_ch, kernel]`).
///
/// The output extent is `(in_len - 1) * stride + kernel + output_padding -
/// 2 * padding`.  Output padding is explicit and must be smaller than stride;
/// no implicit cropping or shape correction is performed.
#[allow(clippy::too_many_arguments)]
pub fn conv_transpose1d_f32(
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
    expect_len(
        "conv_transpose1d out",
        out.len(),
        checked_mul(out_ch, out_len, "conv_transpose1d out_ch*out_len")?,
    )?;

    for oc in 0..out_ch {
        for t in 0..out_len {
            let mut acc = bias.map_or(0.0, |values| values[oc]);
            let target = t.checked_add(padding).ok_or_else(|| {
                VokraError::InvalidArgument("conv_transpose1d index overflow".into())
            })?;
            for ic in 0..in_ch {
                for input_t in 0..in_len {
                    let Some(tap) =
                        target.checked_sub(input_t.checked_mul(stride).ok_or_else(|| {
                            VokraError::InvalidArgument("conv_transpose1d index overflow".into())
                        })?)
                    else {
                        continue;
                    };
                    if tap >= kernel {
                        continue;
                    }
                    let weight_index = (ic * out_ch + oc) * kernel + tap;
                    acc += input[ic * in_len + input_t] * weight[weight_index];
                }
            }
            out[oc * out_len + t] = acc;
        }
    }
    Ok(())
}

/// [`conv1d_f32`] forced onto a specific `isa` (drives the GEMM path).
#[allow(clippy::too_many_arguments)] // convolution's intrinsic parameter set
pub fn conv1d_f32_on(
    isa: IsaPath,
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
    conv1d_dispatch(
        Some(isa),
        input,
        in_ch,
        in_len,
        weight,
        out_ch,
        kernel,
        bias,
        stride,
        padding,
        out,
    )
}

/// Grouped 1-D convolution composed from the dispatched dense
/// [`conv1d_f32`] kernel.
///
/// Layout is PyTorch-compatible: `input = [in_ch, in_len]`, `weight =
/// [out_ch, in_ch / groups, kernel]`, and `out = [out_ch, out_len]`.
/// Each group is a contiguous channel slab, so no input or weight copy is
/// needed. `groups == 1` is bit-identical to [`conv1d_f32`].
///
/// # Errors
///
/// Returns [`VokraError::InvalidArgument`] for a zero or indivisible group
/// count, mismatched buffers, or invalid convolution extents.
#[allow(clippy::too_many_arguments)] // convolution's intrinsic parameter set
pub fn grouped_conv1d_f32(
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
    let fail = |what: String| {
        Err(VokraError::InvalidArgument(format!(
            "grouped_conv1d: {what}"
        )))
    };
    if groups == 0 {
        return fail("groups must be > 0".into());
    }
    if in_ch % groups != 0 || out_ch % groups != 0 {
        return fail(format!(
            "in_ch {in_ch} and out_ch {out_ch} must both be divisible by groups {groups}"
        ));
    }
    if stride == 0 || kernel == 0 {
        return fail(format!(
            "stride and kernel must be > 0 (got {stride} / {kernel})"
        ));
    }
    let in_per = in_ch / groups;
    let out_per = out_ch / groups;
    let weight_len = out_ch
        .checked_mul(in_per)
        .and_then(|n| n.checked_mul(kernel))
        .ok_or_else(|| {
            VokraError::InvalidArgument("grouped_conv1d weight extent overflow".into())
        })?;
    if weight.len() != weight_len {
        return fail(format!(
            "weight length {} != out_ch × (in_ch / groups) × kernel = {}",
            weight.len(),
            weight_len
        ));
    }
    if let Some(b) = bias {
        if b.len() != out_ch {
            return fail(format!("bias length {} != out_ch {out_ch}", b.len()));
        }
    }
    let input_len = in_ch.checked_mul(in_len).ok_or_else(|| {
        VokraError::InvalidArgument("grouped_conv1d input extent overflow".into())
    })?;
    if input.len() != input_len {
        return fail(format!(
            "input length {} != in_ch × in_len = {}",
            input.len(),
            input_len
        ));
    }
    let padded =
        in_len
            .checked_add(padding.checked_mul(2).ok_or_else(|| {
                VokraError::InvalidArgument("grouped_conv1d padding overflow".into())
            })?)
            .ok_or_else(|| {
                VokraError::InvalidArgument("grouped_conv1d padded length overflow".into())
            })?;
    if padded < kernel {
        return fail(format!(
            "padded input length {padded} is shorter than kernel {kernel}"
        ));
    }
    let out_len = (padded - kernel) / stride + 1;
    let output_len = out_ch.checked_mul(out_len).ok_or_else(|| {
        VokraError::InvalidArgument("grouped_conv1d output extent overflow".into())
    })?;
    if out.len() != output_len {
        return fail(format!(
            "out length {} != out_ch × out_len = {}",
            out.len(),
            output_len
        ));
    }

    for g in 0..groups {
        let in_slice = &input[g * in_per * in_len..(g + 1) * in_per * in_len];
        let w_slice = &weight[g * out_per * in_per * kernel..(g + 1) * out_per * in_per * kernel];
        let b_slice = bias.map(|b| &b[g * out_per..(g + 1) * out_per]);
        let out_slice = &mut out[g * out_per * out_len..(g + 1) * out_per * out_len];
        conv1d_f32(
            in_slice, in_per, in_len, w_slice, out_per, kernel, b_slice, stride, padding, out_slice,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)] // convolution's intrinsic parameter set
fn conv1d_dispatch(
    force: Option<IsaPath>,
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
    if stride == 0 {
        return Err(VokraError::InvalidArgument(
            "conv1d stride must be >= 1".into(),
        ));
    }
    if kernel == 0 {
        return Err(VokraError::InvalidArgument(
            "conv1d kernel must be >= 1".into(),
        ));
    }
    let padded = in_len
        .checked_add(checked_mul(2, padding, "conv1d 2*padding")?)
        .ok_or_else(|| VokraError::InvalidArgument("conv1d padded length overflow".into()))?;
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

    // im2col: col is [in_ch*kernel, out_len] row-major.
    let mut col = vec![0.0f32; k * out_len];
    for c in 0..in_ch {
        for kk in 0..kernel {
            let row = c * kernel + kk;
            for t in 0..out_len {
                let pos = t * stride + kk;
                if pos >= padding && pos < padding + in_len {
                    col[row * out_len + t] = input[c * in_len + (pos - padding)];
                }
            }
        }
    }

    // weight [out_ch, k] * col [k, out_len] = out [out_ch, out_len].
    match force {
        Some(isa) => gemm_f32_on(isa, out_ch, out_len, k, weight, &col, None, out)?,
        None => gemm_f32(out_ch, out_len, k, weight, &col, None, out)?,
    }

    // Per-output-channel bias (broadcast over out_len).
    if let Some(bias) = bias {
        for (oc, &b) in bias.iter().enumerate() {
            for v in &mut out[oc * out_len..oc * out_len + out_len] {
                *v += b;
            }
        }
    }
    Ok(())
}

/// PyTorch-layout dense/grouped Conv2d on channel-major host buffers.
///
/// `input` is `[in_ch, in_h, in_w]`, `weight` is
/// `[out_ch, in_ch / groups, kernel_h, kernel_w]`, and `out` is
/// `[out_ch, out_h, out_w]`. All arithmetic is shape-explicit and checked;
/// the accumulation order is output-channel, input-channel, row, column.
#[allow(clippy::too_many_arguments)]
pub fn conv2d_f32(
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
    let out_h = (padded_h - effective_h) / stride_h + 1;
    let out_w = (padded_w - effective_w) / stride_w + 1;
    let input_plane = checked_mul(in_h, in_w, "conv2d input plane")?;
    let output_plane = checked_mul(out_h, out_w, "conv2d output plane")?;
    expect_len(
        "conv2d input",
        input.len(),
        checked_mul(in_ch, input_plane, "conv2d input")?,
    )?;
    let in_per_group = in_ch / groups;
    let kernel_plane = checked_mul(kernel_h, kernel_w, "conv2d kernel plane")?;
    let weight_per_output = checked_mul(in_per_group, kernel_plane, "conv2d weight per output")?;
    expect_len(
        "conv2d weight",
        weight.len(),
        checked_mul(out_ch, weight_per_output, "conv2d weight")?,
    )?;
    expect_len(
        "conv2d out",
        out.len(),
        checked_mul(out_ch, output_plane, "conv2d out")?,
    )?;
    if let Some(bias) = bias {
        expect_len("conv2d bias", bias.len(), out_ch)?;
    }
    let input_end_h = padding_h
        .checked_add(in_h)
        .ok_or_else(|| VokraError::InvalidArgument("conv2d input height extent overflow".into()))?;
    let input_end_w = padding_w
        .checked_add(in_w)
        .ok_or_else(|| VokraError::InvalidArgument("conv2d input width extent overflow".into()))?;
    let out_per_group = out_ch / groups;
    for oc in 0..out_ch {
        let group = oc / out_per_group;
        for oh in 0..out_h {
            for ow in 0..out_w {
                let mut acc = bias.map_or(0.0, |values| values[oc]);
                for ic_local in 0..in_per_group {
                    let ic = group * in_per_group + ic_local;
                    for kh in 0..kernel_h {
                        let input_h = oh * stride_h + kh * dilation_h;
                        if input_h < padding_h || input_h >= input_end_h {
                            continue;
                        }
                        let input_h = input_h - padding_h;
                        for kw in 0..kernel_w {
                            let input_w = ow * stride_w + kw * dilation_w;
                            if input_w < padding_w || input_w >= input_end_w {
                                continue;
                            }
                            let input_w = input_w - padding_w;
                            let input_index = ic * input_plane + input_h * in_w + input_w;
                            let weight_index =
                                (oc * in_per_group + ic_local) * kernel_plane + kh * kernel_w + kw;
                            acc += input[input_index] * weight[weight_index];
                        }
                    }
                }
                out[oc * output_plane + oh * out_w + ow] = acc;
            }
        }
    }
    Ok(())
}

/// PyTorch-layout dense/grouped ConvTranspose2d on channel-major host buffers.
/// The weight layout is `[in_ch, out_ch / groups, kernel_h, kernel_w]` and the
/// output extent is `(in - 1) * stride - 2 * padding + dilation * (kernel - 1)
/// + output_padding + 1` per spatial axis. As in PyTorch/ATen,
/// `output_padding` must be smaller than either the corresponding stride or
/// dilation.
#[allow(clippy::too_many_arguments)]
pub fn conv_transpose2d_f32(
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
    let out_h = base_h - trim_h;
    let out_w = base_w - trim_w;
    let input_plane = checked_mul(in_h, in_w, "conv_transpose2d input plane")?;
    let output_plane = checked_mul(out_h, out_w, "conv_transpose2d output plane")?;
    expect_len(
        "conv_transpose2d input",
        input.len(),
        checked_mul(in_ch, input_plane, "conv_transpose2d input")?,
    )?;
    let out_per_group = out_ch / groups;
    let kernel_plane = checked_mul(kernel_h, kernel_w, "conv_transpose2d kernel plane")?;
    let weight_per_input = checked_mul(
        out_per_group,
        kernel_plane,
        "conv_transpose2d weight per input",
    )?;
    expect_len(
        "conv_transpose2d weight",
        weight.len(),
        checked_mul(in_ch, weight_per_input, "conv_transpose2d weight")?,
    )?;
    expect_len(
        "conv_transpose2d out",
        out.len(),
        checked_mul(out_ch, output_plane, "conv_transpose2d out")?,
    )?;
    if let Some(bias) = bias {
        expect_len("conv_transpose2d bias", bias.len(), out_ch)?;
    }
    let in_per_group = in_ch / groups;
    for oc in 0..out_ch {
        let group = oc / out_per_group;
        let oc_local = oc % out_per_group;
        for oh in 0..out_h {
            for ow in 0..out_w {
                let mut acc = bias.map_or(0.0, |values| values[oc]);
                for ic_local in 0..in_per_group {
                    let ic = group * in_per_group + ic_local;
                    for kh in 0..kernel_h {
                        let numerator_h = oh + padding_h;
                        let tap_h = kh * dilation_h;
                        if numerator_h < tap_h {
                            continue;
                        }
                        let source_h = numerator_h - tap_h;
                        if source_h % stride_h != 0 || source_h / stride_h >= in_h {
                            continue;
                        }
                        let source_h = source_h / stride_h;
                        for kw in 0..kernel_w {
                            let numerator_w = ow + padding_w;
                            let tap_w = kw * dilation_w;
                            if numerator_w < tap_w {
                                continue;
                            }
                            let source_w = numerator_w - tap_w;
                            if source_w % stride_w != 0 || source_w / stride_w >= in_w {
                                continue;
                            }
                            let source_w = source_w / stride_w;
                            let input_index = ic * input_plane + source_h * in_w + source_w;
                            let weight_index =
                                (ic * out_per_group + oc_local) * kernel_plane + kh * kernel_w + kw;
                            acc += input[input_index] * weight[weight_index];
                        }
                    }
                }
                out[oc * output_plane + oh * out_w + ow] = acc;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemm_rejects_bad_shapes() {
        let a = [1.0, 2.0];
        let b = [1.0, 2.0];
        let mut out = [0.0; 4];
        // a should be m*k = 2*2 = 4 long, but it is 2 → explicit error.
        let err = gemm_f32(2, 2, 2, &a, &b, None, &mut out).unwrap_err();
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn gemm_on_scalar_matches_hand_value() {
        // [[1,2],[3,4]] * [[1,0],[0,1]] + bias[100,200].
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [1.0, 0.0, 0.0, 1.0];
        let bias = [100.0, 200.0];
        let mut out = [0.0; 4];
        gemm_f32_on(IsaPath::Scalar, 2, 2, 2, &a, &b, Some(&bias), &mut out).unwrap();
        assert_eq!(out, [101.0, 202.0, 103.0, 204.0]);
    }

    #[test]
    fn gemv_on_scalar_matches_hand_value() {
        // a = [[1,2,3],[4,5,6]] (2x3), x = [1,0,-1], bias = [100, 200].
        // row0 = 1*1 + 2*0 + 3*-1 = -2 (+100 = 98); row1 = 4 - 6 = -2 (+200 = 198).
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x = [1.0, 0.0, -1.0];
        let bias = [100.0, 200.0];
        let mut out = [0.0; 2];
        gemv_f32_on(IsaPath::Scalar, 2, 3, &a, &x, Some(&bias), &mut out).unwrap();
        assert_eq!(out, [98.0, 198.0]);
        // No-bias variant: exactly the n=1 column of gemm on the same data.
        gemv_f32_on(IsaPath::Scalar, 2, 3, &a, &x, None, &mut out).unwrap();
        assert_eq!(out, [-2.0, -2.0]);
    }

    #[test]
    fn gemv_matches_gemm_n1_column() {
        // gemv(m,k) must equal gemm(m, n=1, k) with x as the single b-column
        // (this is the equivalence the tied-logits-head routing relies on).
        let a = [0.5, -1.0, 2.0, 3.0, 0.25, -0.5]; // 2x3
        let x = [0.1, -0.2, 0.3];
        let mut g_out = [0.0; 2];
        gemm_f32(2, 1, 3, &a, &x, None, &mut g_out).unwrap();
        let mut v_out = [0.0; 2];
        gemv_f32_on(IsaPath::Scalar, 2, 3, &a, &x, None, &mut v_out).unwrap();
        assert_eq!(g_out, v_out);
    }

    #[test]
    fn gemv_rejects_bad_shapes() {
        // m=2, k=3: a needs 6, x needs 3, out needs 2, bias needs 2.
        let a = [0.0; 6];
        let x = [0.0; 3];
        let mut out = [0.0; 2];
        // `a` too short (5 != m*k = 6).
        assert!(gemv_f32(2, 3, &[0.0; 5], &x, None, &mut out).is_err());
        // `x` length != k (2 != 3).
        assert!(gemv_f32(2, 3, &a, &[0.0; 2], None, &mut out).is_err());
        // `out` length != m (3 != 2).
        assert!(gemv_f32(2, 3, &a, &x, None, &mut [0.0; 3]).is_err());
        // `bias` length != m (1 != 2).
        assert!(gemv_f32(2, 3, &a, &x, Some(&[0.0; 1]), &mut out).is_err());
        // m*k overflows usize -> explicit error via checked_mul (no kernel run).
        assert!(gemv_f32(usize::MAX, 2, &[], &[], None, &mut []).is_err());
    }

    #[test]
    fn binary_and_unary_reject_length_mismatch() {
        let mut out2 = [0.0; 2];
        assert!(add_f32(&[1.0, 2.0], &[1.0], &mut out2).is_err());
        let mut out1 = [0.0; 1];
        assert!(relu_f32(&[1.0, 2.0], &mut out1).is_err());
        assert!(elu_f32(&[1.0, 2.0], &mut out1).is_err());
    }

    #[test]
    fn elu_matches_transformers_alpha_one_points() {
        let x = [f32::NEG_INFINITY, -4.0, -1.0, -0.0, 0.0, 0.5, 8.0];
        let mut out = [f32::NAN; 7];
        elu_f32(&x, &mut out).expect("valid ELU shape");

        for (index, (&input, &actual)) in x.iter().zip(&out).enumerate() {
            let expected = if input > 0.0 {
                input
            } else {
                input.exp() - 1.0
            };
            assert_eq!(actual.to_bits(), expected.to_bits(), "index {index}");
        }
    }

    #[test]
    fn conv1d_single_channel_hand_fixture() {
        // input [1,5] = 1..5, weight [1,1,3] = [1,1,1], stride 1, pad 0.
        // out_len = 5-3+1 = 3; sliding sums: 1+2+3, 2+3+4, 3+4+5.
        let input = [1.0, 2.0, 3.0, 4.0, 5.0];
        let weight = [1.0, 1.0, 1.0];
        let mut out = [0.0; 3];
        conv1d_f32(&input, 1, 5, &weight, 1, 3, None, 1, 0, &mut out).unwrap();
        assert_eq!(out, [6.0, 9.0, 12.0]);
    }

    #[test]
    fn conv1d_padding_and_bias() {
        // input [1,3] = [1,2,3], weight [1,1,3] = [1,0,-1], pad 1, stride 1,
        // bias [10]. padded = [0,1,2,3,0], out_len = (5-3)/1+1 = 3.
        // windows: [0,1,2]·[1,0,-1] = -2; [1,2,3] = -2; [2,3,0] = 2. +bias 10.
        let input = [1.0, 2.0, 3.0];
        let weight = [1.0, 0.0, -1.0];
        let bias = [10.0];
        let mut out = [0.0; 3];
        conv1d_f32(&input, 1, 3, &weight, 1, 3, Some(&bias), 1, 1, &mut out).unwrap();
        assert_eq!(out, [8.0, 8.0, 12.0]);
    }

    #[test]
    fn conv1d_rejects_kernel_larger_than_padded_input() {
        let input = [1.0, 2.0];
        let weight = [1.0; 5];
        let mut out = [0.0; 1];
        let err = conv1d_f32(&input, 1, 2, &weight, 1, 5, None, 1, 0, &mut out).unwrap_err();
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn conv1d_multichannel_stride2() {
        // 2 in-ch, len 4; 1 out-ch; kernel 2; stride 2; no pad.
        // in ch0 = [1,2,3,4], ch1 = [10,20,30,40].
        // weight [1,2,2] = ch0:[1,1], ch1:[1,1].
        // out_len = (4-2)/2+1 = 2.
        // t=0: ch0(1+2)+ch1(10+20)=3+30=33; t=1: ch0(3+4)+ch1(30+40)=7+70=77.
        let input = [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
        let weight = [1.0, 1.0, 1.0, 1.0];
        let mut out = [0.0; 2];
        conv1d_f32(&input, 2, 4, &weight, 1, 2, None, 2, 0, &mut out).unwrap();
        assert_eq!(out, [33.0, 77.0]);
    }

    #[test]
    fn grouped_conv1d_keeps_channel_groups_isolated() {
        // Two single-channel groups, kernel width 2.  Output channel 0 sees
        // only input channel 0 with weight [1, 1]; output channel 1 sees only
        // input channel 1 with weight [2, -1].
        let input = [1.0, 2.0, 3.0, 10.0, 20.0, 40.0];
        let weight = [1.0, 1.0, 2.0, -1.0];
        let bias = [0.5, -0.5];
        let mut out = [0.0; 4];
        grouped_conv1d_f32(&input, 2, 3, &weight, 2, 2, Some(&bias), 1, 0, 2, &mut out).unwrap();
        assert_eq!(out, [3.5, 5.5, -0.5, -0.5]);
    }

    #[test]
    fn grouped_conv1d_rejects_invalid_groups_and_extent_overflow() {
        let mut out = [0.0; 1];
        assert!(grouped_conv1d_f32(&[1.0], 1, 1, &[1.0], 1, 1, None, 1, 0, 0, &mut out).is_err());
        assert!(
            grouped_conv1d_f32(&[1.0], 1, usize::MAX, &[1.0], 1, 1, None, 1, 0, 1, &mut out)
                .is_err()
        );
    }

    #[test]
    fn vec_dot_hand_value_and_length_mismatch() {
        // 1*-1 + 2*0 + 3*2 = -1 + 0 + 6 = 5 (all terms exactly representable).
        let dot = vec_dot_f32(&[1.0, 2.0, 3.0], &[-1.0, 0.0, 2.0]).unwrap();
        assert!((dot - 5.0).abs() < 1e-6, "dot = {dot}, want 5.0");
        // Unequal lengths are an explicit error.
        let err = vec_dot_f32(&[1.0, 2.0], &[1.0]).unwrap_err();
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn conv1d_nondivisible_stride_drops_trailing_window() {
        // input [1,6] = 1..=6, weight [1,1,2] = [1,1], stride 3, pad 0.
        // out_len = (6-2)/3+1 = 2. Windows begin at pos 0 and 3:
        // [1,2]·[1,1] = 3, [4,5]·[1,1] = 9. input[2]=3 and input[5]=6 lie past
        // the last full window and are intentionally dropped (im2col guard).
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let weight = [1.0, 1.0];
        let mut out = [0.0; 2];
        conv1d_f32(&input, 1, 6, &weight, 1, 2, None, 3, 0, &mut out).unwrap();
        assert_eq!(out, [3.0, 9.0]);
    }

    #[test]
    fn conv1d_padding_ge_in_len_zeros_outside_real_input() {
        // input [1,2] = [2,3], weight [1,1,1] = [1], stride 1, pad 2.
        // padded = 2 + 2*2 = 6, out_len = (6-1)/1+1 = 6. Only positions 2 and 3
        // fall inside the real input; the rest are pure zero-padding.
        let input = [2.0, 3.0];
        let weight = [1.0];
        let mut out = [0.0; 6];
        conv1d_f32(&input, 1, 2, &weight, 1, 1, None, 1, 2, &mut out).unwrap();
        assert_eq!(out, [0.0, 0.0, 2.0, 3.0, 0.0, 0.0]);
    }

    #[test]
    fn conv1d_rejects_zero_stride_and_zero_kernel() {
        // stride == 0 would divide-by-zero in the out_len formula.
        let mut out = [0.0; 1];
        let err = conv1d_f32(&[1.0, 2.0], 1, 2, &[1.0], 1, 1, None, 0, 0, &mut out).unwrap_err();
        assert!(matches!(err, VokraError::InvalidArgument(_)));
        // kernel == 0 would mis-size the im2col matrix.
        let err = conv1d_f32(&[1.0, 2.0], 1, 2, &[], 1, 0, None, 1, 0, &mut out).unwrap_err();
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn conv1d_dilated_matches_explicit_sparse_window() {
        // One channel, kernel 3, dilation 2, symmetric padding 2: logical
        // input index is `t + k*2 - 2`, so both left and right zero padding
        // are exercised.
        let input = [1.0, 2.0, 3.0, 4.0, 5.0];
        let weight = [1.0, 10.0, 100.0];
        let mut out = [0.0; 5];
        conv1d_f32_dilated(&input, 1, 5, &weight, 1, 3, None, 1, 2, 2, &mut out).unwrap();
        assert_eq!(out, [310.0, 420.0, 531.0, 42.0, 53.0]);
    }

    #[test]
    fn conv1d_dilated_rejects_invalid_extent_and_dilation() {
        let mut out = [0.0; 1];
        assert!(conv1d_f32_dilated(&[1.0], 1, 1, &[1.0], 1, 1, None, 1, 0, 0, &mut out).is_err());
        assert!(
            conv1d_f32_dilated(&[1.0], 1, 1, &[1.0, 2.0], 1, 2, None, 1, 2, 0, &mut out).is_err()
        );
    }

    #[test]
    fn conv_transpose1d_matches_hand_fixture_and_rejects_output_padding() {
        // input [1,2], kernel [1,2], stride 2, output_padding 1:
        // transposed output = [1,2,2,4,0] (the final slot is explicit padding).
        let mut out = [0.0; 5];
        conv_transpose1d_f32(
            &[1.0, 2.0],
            1,
            2,
            &[1.0, 2.0],
            1,
            2,
            None,
            2,
            0,
            1,
            &mut out,
        )
        .unwrap();
        assert_eq!(out, [1.0, 2.0, 2.0, 4.0, 0.0]);

        let mut one = [0.0; 1];
        assert!(conv_transpose1d_f32(&[1.0], 1, 1, &[1.0], 1, 1, None, 2, 0, 2, &mut one).is_err());
    }

    #[test]
    fn grouped_conv2d_handles_dilation_asymmetric_axes_and_bias() {
        // Two independent groups, one output channel each. The width uses
        // dilation=2 and left/right padding=1, while stride_w=1 here keeps
        // all three valid output columns visible.
        let input = [
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0,
        ];
        let weight = [1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, -1.0];
        let bias = [0.5, -1.0];
        let mut out = [0.0; 6];
        conv2d_f32(
            &input,
            2,
            2,
            3,
            &weight,
            2,
            2,
            2,
            Some(&bias),
            (1, 1),
            (0, 1),
            (1, 2),
            2,
            &mut out,
        )
        .unwrap();
        assert_eq!(out, [5.5, 7.5, 2.5, -51.0, -41.0, 39.0]);
    }

    #[test]
    fn grouped_conv_transpose2d_handles_dilation_output_padding_and_bias() {
        // The second kernel column is skipped by dilation=2 in this fixture;
        // output_padding_h=1 exposes the extra trailing output row.
        let input = [1.0, 2.0, 10.0, 20.0];
        let weight = [1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0];
        let bias = [0.5, -1.0];
        let mut out = [0.0; 16];
        conv_transpose2d_f32(
            &input,
            2,
            1,
            2,
            &weight,
            2,
            2,
            2,
            Some(&bias),
            (2, 2),
            (0, 0),
            (1, 2),
            (1, 1),
            2,
            &mut out,
        )
        .unwrap();
        assert_eq!(
            out,
            [
                1.5, 0.5, 2.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 19.0, -1.0, 39.0, -1.0
            ]
        );
    }

    #[test]
    fn conv2d_and_conv_transpose2d_reject_invalid_shapes_and_overflow() {
        let mut out = [0.0; 1];
        assert!(
            conv2d_f32(
                &[1.0],
                1,
                1,
                1,
                &[1.0],
                1,
                1,
                1,
                None,
                (0, 1),
                (0, 0),
                (1, 1),
                1,
                &mut out
            )
            .is_err()
        );
        assert!(
            conv2d_f32(
                &[1.0],
                2,
                1,
                1,
                &[1.0],
                1,
                1,
                1,
                None,
                (1, 1),
                (0, 0),
                (1, 1),
                2,
                &mut out
            )
            .is_err()
        );
        assert!(
            conv2d_f32(
                &[1.0],
                1,
                1,
                1,
                &[],
                1,
                2,
                1,
                None,
                (1, 1),
                (0, 0),
                (1, 1),
                1,
                &mut out
            )
            .is_err()
        );
        assert!(
            conv2d_f32(
                &[],
                1,
                usize::MAX,
                2,
                &[],
                1,
                1,
                1,
                None,
                (1, 1),
                (0, 0),
                (1, 1),
                1,
                &mut []
            )
            .is_err()
        );
        assert!(
            conv_transpose2d_f32(
                &[1.0],
                1,
                1,
                1,
                &[1.0],
                1,
                1,
                1,
                None,
                (2, 1),
                (0, 0),
                (1, 1),
                (2, 0),
                1,
                &mut out
            )
            .is_err()
        );
        // ATen permits output_padding == stride when it is still smaller
        // than dilation on that axis.
        let mut dilation_valid = [0.0; 2];
        conv_transpose2d_f32(
            &[3.0],
            1,
            1,
            1,
            &[2.0],
            1,
            1,
            1,
            None,
            (1, 1),
            (0, 0),
            (2, 1),
            (1, 0),
            1,
            &mut dilation_valid,
        )
        .unwrap();
        assert_eq!(dilation_valid, [6.0, 0.0]);
        // It is rejected only once output_padding reaches both bounds.
        assert!(
            conv_transpose2d_f32(
                &[1.0],
                1,
                1,
                1,
                &[1.0],
                1,
                1,
                1,
                None,
                (1, 1),
                (0, 0),
                (2, 1),
                (2, 0),
                1,
                &mut [0.0; 3]
            )
            .is_err()
        );
        assert!(
            conv_transpose2d_f32(
                &[1.0],
                2,
                1,
                1,
                &[1.0],
                1,
                1,
                1,
                None,
                (1, 1),
                (0, 0),
                (1, 1),
                (0, 0),
                2,
                &mut out
            )
            .is_err()
        );
        assert!(
            conv_transpose2d_f32(
                &[],
                1,
                usize::MAX,
                2,
                &[],
                1,
                1,
                1,
                None,
                (1, 1),
                (0, 0),
                (1, 1),
                (0, 0),
                1,
                &mut []
            )
            .is_err()
        );
    }

    #[test]
    fn softmax_and_layer_norm_reject_shape_mismatch() {
        // softmax: length 6 does not match rows*cols = 2*4 = 8.
        let err = softmax_f32(&[0.0; 6], &mut [0.0; 6], 2, 4).unwrap_err();
        assert!(matches!(err, VokraError::InvalidArgument(_)));
        // layer_norm: rows*cols is consistent (8), but gamma length 3 != cols 4.
        let err = layer_norm_f32(
            &[0.0; 8],
            &mut [0.0; 8],
            2,
            4,
            &[1.0; 3],
            &[0.0; 4],
            LAYER_NORM_DEFAULT_EPS,
        )
        .unwrap_err();
        assert!(matches!(err, VokraError::InvalidArgument(_)));
        // ... and a beta length != cols is rejected by the same validator.
        let err = layer_norm_f32(
            &[0.0; 8],
            &mut [0.0; 8],
            2,
            4,
            &[1.0; 4],
            &[0.0; 3],
            LAYER_NORM_DEFAULT_EPS,
        )
        .unwrap_err();
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn group_norm_one_group_matches_hand_fixture() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let gamma = [2.0, 0.5];
        let beta = [1.0, -1.0];
        let mut out = [0.0; 4];
        group_norm_f32(&input, &mut out, 2, 2, &gamma, &beta, 0.0).unwrap();
        let inv_std = 1.0 / 1.25f32.sqrt();
        let expected = [
            (1.0 - 2.5) * inv_std * 2.0 + 1.0,
            (2.0 - 2.5) * inv_std * 2.0 + 1.0,
            (3.0 - 2.5) * inv_std * 0.5 - 1.0,
            (4.0 - 2.5) * inv_std * 0.5 - 1.0,
        ];
        for (actual, expected) in out.iter().zip(expected) {
            assert!((actual - expected).abs() <= 1e-6);
        }
    }

    #[test]
    fn group_norm_rejects_invalid_shapes() {
        assert!(group_norm_f32(&[0.0; 4], &mut [0.0; 4], 0, 2, &[], &[], 1e-8).is_err());
        assert!(group_norm_f32(&[0.0; 4], &mut [0.0; 4], 2, 2, &[1.0], &[0.0; 2], 1e-8,).is_err());
    }

    #[test]
    fn scale_norm_matches_released_equation_and_clamp() {
        let input = [3.0f32, 4.0, 0.0, 0.0];
        let mut out = [f32::NAN; 4];
        scale_norm_f32(&input, &mut out, 2, 2, 1.5, 1.0e-5).unwrap();
        let denominator = 5.0 * (2.0f64).sqrt().recip() as f32;
        assert!((out[0] - 3.0 / denominator * 1.5).abs() <= f32::EPSILON);
        assert!((out[1] - 4.0 / denominator * 1.5).abs() <= f32::EPSILON);
        assert_eq!(out[2], 0.0);
        assert_eq!(out[3], 0.0);
    }

    #[test]
    fn scale_norm_rejects_invalid_contract() {
        assert!(scale_norm_f32(&[], &mut [], 0, 2, 1.0, 1.0e-5).is_err());
        assert!(scale_norm_f32(&[1.0], &mut [0.0], 1, 1, f32::NAN, 1.0e-5).is_err());
        assert!(scale_norm_f32(&[1.0], &mut [0.0], 1, 1, 1.0, 0.0).is_err());
        assert!(scale_norm_f32(&[1.0; 2], &mut [0.0; 1], 1, 2, 1.0, 1.0e-5).is_err());
    }

    #[test]
    fn gemm_rejects_bad_b_out_bias_and_overflow() {
        // m=2, n=2, k=2: a needs 4, b needs k*n=4, out needs m*n=4, bias needs n=2.
        let a = [0.0; 4];
        let good_b = [0.0; 4];
        let mut out = [0.0; 4];
        // `b` too short (3 != k*n = 4).
        let err = gemm_f32(2, 2, 2, &a, &[0.0; 3], None, &mut out).unwrap_err();
        assert!(matches!(err, VokraError::InvalidArgument(_)));
        // `out` too short (3 != m*n = 4).
        let mut short_out = [0.0; 3];
        let err = gemm_f32(2, 2, 2, &a, &good_b, None, &mut short_out).unwrap_err();
        assert!(matches!(err, VokraError::InvalidArgument(_)));
        // `bias` length != n (1 != 2).
        let err = gemm_f32(2, 2, 2, &a, &good_b, Some(&[0.0; 1]), &mut out).unwrap_err();
        assert!(matches!(err, VokraError::InvalidArgument(_)));
        // m*k overflows usize -> explicit error via the checked_mul guard (and
        // no kernel is ever entered with a bogus dimension product).
        let err = gemm_f32(usize::MAX, 1, 2, &[], &[], None, &mut []).unwrap_err();
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }
}
