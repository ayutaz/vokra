//! Kernel dispatch table and process-wide ISA selection (M0-08-T04).
//!
//! Dispatch is the llama.cpp / OpenBLAS style (FR-EX-06): a table of function
//! pointers to statically compiled kernels, selected **once** per process
//! from [`crate::features::resolve_isa`] and cached in a [`OnceLock`]. There
//! is no JIT and no runtime code generation (NFR-RL-05) — only a choice of
//! which already-compiled function to call.
//!
//! # Relationship to FR-EX-08
//!
//! FR-EX-08 ("uniform op coverage; unsupported ops are an explicit error, no
//! silent CPU fallback") is a rule **between backends**. The ISA selection
//! here is **inside** the CPU backend: every path (`Scalar` / `Avx2` /
//! `Neon`) computes the *same* op with the *same* result (within FP32
//! rounding). Selecting a SIMD path is therefore never a silent fallback.

use std::sync::OnceLock;

use vokra_core::{Result, VokraError};

use crate::features::{self, CpuFeatures, IsaPath};
use crate::kernels::scalar;

/// GEMM kernel signature (see [`scalar::gemm`]); inputs are pre-validated.
pub(crate) type GemmKernel = fn(usize, usize, usize, &[f32], &[f32], Option<&[f32]>, &mut [f32]);
/// GEMV (matrix-vector) kernel signature (see [`scalar::gemv`]); inputs are
/// pre-validated. `(m, k, a[m*k], x[k], bias?[m], out[m])`.
pub(crate) type GemvKernel = fn(usize, usize, &[f32], &[f32], Option<&[f32]>, &mut [f32]);
/// Element-wise binary kernel signature (`add` / `mul`).
pub(crate) type BinaryKernel = fn(&[f32], &[f32], &mut [f32]);
/// Element-wise unary kernel signature (`relu` / `sigmoid` / `tanh` / `gelu`).
pub(crate) type UnaryKernel = fn(&[f32], &mut [f32]);
/// Row-wise softmax kernel signature.
pub(crate) type SoftmaxKernel = fn(&[f32], &mut [f32], usize, usize);
/// Row-wise layer-norm kernel signature.
pub(crate) type LayerNormKernel = fn(&[f32], &mut [f32], usize, usize, &[f32], &[f32], f32);
/// Fused log-mel per-frame kernel signature (M2-04-T06). Applies the mel
/// filterbank to one frame's power spectrum and writes `n_mels`
/// `log10(max(·, floor))` values. `(weights[n_mels*n_bins], power[n_bins],
/// n_mels, n_bins, floor, out_log[n_mels])`.
pub(crate) type FusedLogmelKernel = fn(&[f32], &[f32], usize, usize, f32, &mut [f32]);

/// A bundle of function pointers, one per kernel kind, all resolved to the
/// same [`IsaPath`]. Populated by [`build_table`] and cached in [`selected`].
#[derive(Clone, Copy)]
pub(crate) struct KernelTable {
    pub(crate) gemm: GemmKernel,
    pub(crate) gemv: GemvKernel,
    pub(crate) add: BinaryKernel,
    pub(crate) mul: BinaryKernel,
    pub(crate) relu: UnaryKernel,
    pub(crate) sigmoid: UnaryKernel,
    pub(crate) tanh: UnaryKernel,
    pub(crate) gelu: UnaryKernel,
    pub(crate) softmax: SoftmaxKernel,
    pub(crate) layer_norm: LayerNormKernel,
    /// M2-04-T06 fused log-mel inner (mel-band accumulate + `log10(max(·, floor))`).
    /// Scalar / NEON currently share the portable scalar reference; AVX2 uses
    /// the eight-lane FMA + `vlog10_avx2` polynomial path. All three compute
    /// the same op within FP32 rounding (within-CPU-backend dispatch, not a
    /// cross-backend fallback — FR-EX-08 unaffected).
    pub(crate) fused_logmel: FusedLogmelKernel,
}

fn scalar_table() -> KernelTable {
    KernelTable {
        gemm: scalar::gemm,
        gemv: scalar::gemv,
        add: scalar::add,
        mul: scalar::mul,
        relu: scalar::relu,
        sigmoid: scalar::sigmoid,
        tanh: scalar::tanh,
        gelu: scalar::gelu,
        softmax: scalar::softmax,
        layer_norm: scalar::layer_norm,
        fused_logmel: scalar_fused_logmel,
    }
}

/// Portable scalar reference for the fused log-mel per-frame kernel
/// (M2-04-T06). Bit-close to
/// `vokra_ops::fused_log_mel_scalar`'s inner mel-band accumulate + `log10`
/// step; used by the `Scalar` dispatch table entry and as the parity oracle
/// for the SIMD paths. Kept target-agnostic so aarch64 hosts that select
/// `Scalar` (e.g. via `VOKRA_CPU_ISA=scalar` for a forced-path test) can
/// still populate the table.
fn scalar_fused_logmel(
    weights: &[f32],
    power: &[f32],
    n_mels: usize,
    n_bins: usize,
    floor: f32,
    out_log: &mut [f32],
) {
    assert_eq!(weights.len(), n_mels * n_bins, "weights shape mismatch");
    assert_eq!(power.len(), n_bins, "power length mismatch");
    assert_eq!(out_log.len(), n_mels, "out_log length mismatch");
    for m in 0..n_mels {
        let row = &weights[m * n_bins..(m + 1) * n_bins];
        let mut acc = 0.0f32;
        for (w, p) in row.iter().zip(power) {
            acc += w * p;
        }
        let clamped = if acc > floor { acc } else { floor };
        out_log[m] = clamped.log10();
    }
}

#[cfg(target_arch = "x86_64")]
fn avx2_table() -> KernelTable {
    use crate::kernels::avx2;
    use crate::kernels::fused_logmel_avx2;
    KernelTable {
        gemm: avx2::gemm,
        gemv: avx2::gemv,
        add: avx2::add,
        mul: avx2::mul,
        relu: avx2::relu,
        sigmoid: avx2::sigmoid,
        tanh: avx2::tanh,
        gelu: avx2::gelu,
        softmax: avx2::softmax,
        layer_norm: avx2::layer_norm,
        fused_logmel: fused_logmel_avx2::fused_logmel_apply_frame_avx2,
    }
}

#[cfg(not(target_arch = "x86_64"))]
fn avx2_table() -> KernelTable {
    // Unreachable: `features::select_isa` never yields `Avx2` off x86-64, and
    // `table_for` rejects it via `CpuFeatures::supports`.
    unreachable!("AVX2 kernel table requested on a non-x86-64 target")
}

#[cfg(target_arch = "aarch64")]
fn neon_table() -> KernelTable {
    use crate::kernels::fused_logmel_neon;
    use crate::kernels::neon;
    KernelTable {
        gemm: neon::gemm,
        gemv: neon::gemv,
        add: neon::add,
        mul: neon::mul,
        relu: neon::relu,
        sigmoid: neon::sigmoid,
        tanh: neon::tanh,
        gelu: neon::gelu,
        softmax: neon::softmax,
        layer_norm: neon::layer_norm,
        fused_logmel: fused_logmel_neon::fused_logmel_apply_frame_neon,
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn neon_table() -> KernelTable {
    // Unreachable: `features::select_isa` never yields `Neon` off aarch64, and
    // `table_for` rejects it via `CpuFeatures::supports`.
    unreachable!("NEON kernel table requested on a non-aarch64 target")
}

// M3-13-T03: RISC-V RVV 1.0 dispatch tier. Compiled only on riscv64 (the
// crate uses `#[cfg(target_arch = "riscv64")]` on the `rvv` kernels module).
// `features::select_isa` cannot select `Rvv` off riscv64 (probe returns
// `rvv_v = false`), and `table_for` rejects `Rvv` via `CpuFeatures::supports`
// on non-riscv64 hosts — so the unreachable! stub below is genuinely
// unreachable in all production paths.
#[cfg(target_arch = "riscv64")]
fn rvv_table() -> KernelTable {
    use crate::kernels::rvv;
    KernelTable {
        gemm: rvv::gemm,
        gemv: rvv::gemv,
        add: rvv::add,
        mul: rvv::mul,
        relu: rvv::relu,
        sigmoid: rvv::sigmoid,
        tanh: rvv::tanh,
        gelu: rvv::gelu,
        softmax: rvv::softmax,
        layer_norm: rvv::layer_norm,
        fused_logmel: rvv::fused_logmel,
    }
}

#[cfg(not(target_arch = "riscv64"))]
fn rvv_table() -> KernelTable {
    // Unreachable: `features::select_isa` never yields `Rvv` off riscv64, and
    // `table_for` rejects it via `CpuFeatures::supports`.
    unreachable!("RVV kernel table requested on a non-riscv64 target")
}

// M4-08-T09: RISC-V RVV draft-0.7.1 dispatch tier (T-Head C910/C906 =
// LicheePi 4A / Milk-V Duo). Compiled only on riscv64, exactly like the
// RVV 1.0 tier — but the two are encoding-incompatible peers, so this table
// routes to `kernels::rvv071` (`.insn` raw words), never to `kernels::rvv`.
// `features::select_isa` cannot select `Rvv071` unless the host probed
// `rvv_071 = true` (xtheadvector token / cpu-vector 0.7.1 signal with the
// RVV 1.0 misdetection guard, ADR M4-08 §b), and `table_for` rejects it via
// `CpuFeatures::supports` everywhere else — so the unreachable! stub below
// is genuinely unreachable in all production paths.
#[cfg(target_arch = "riscv64")]
fn rvv071_table() -> KernelTable {
    use crate::kernels::rvv071;
    KernelTable {
        gemm: rvv071::gemm,
        gemv: rvv071::gemv,
        add: rvv071::add,
        mul: rvv071::mul,
        relu: rvv071::relu,
        sigmoid: rvv071::sigmoid,
        tanh: rvv071::tanh,
        gelu: rvv071::gelu,
        softmax: rvv071::softmax,
        layer_norm: rvv071::layer_norm,
        fused_logmel: rvv071::fused_logmel,
    }
}

#[cfg(not(target_arch = "riscv64"))]
fn rvv071_table() -> KernelTable {
    // Unreachable: `features::select_isa` never yields `Rvv071` off riscv64
    // (the probe is /proc/cpuinfo-based and riscv64-gated), and `table_for`
    // rejects it via `CpuFeatures::supports`.
    unreachable!("RVV 0.7.1 kernel table requested on a non-riscv64 target")
}

// M4-01-T04: WASM SIMD128 dispatch tier. Compiled only when the wasm32
// artifact is built WITH `-C target-feature=+simd128` (the crate gates the
// `wasm_simd128` kernels module on the same cfg). This is COMPILE-TIME
// dispatch: WASM has no runtime CPU feature detection (SIMD acceptance is a
// module-validation decision), so unlike AVX2/NEON there is no CPUID-style
// probe — `CpuFeatures::detect().wasm_simd128` is `cfg!(target_feature =
// "simd128")` and the base (no-SIMD) artifact selects `Scalar`. Distribution
// is a 2-artifact build + JS loader `WebAssembly.validate` select
// (scripts/build-wasm.sh, ADR M4-01-webgpu-wasm §4).
//
// Relaxed SIMD is NOT adopted (Safari-partial per the CLAUDE.md quarterly
// watch + relaxed-fma nondeterminism vs NFR-QL-01); the kernels use
// deterministic mul + add only.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
fn wasm_simd128_table() -> KernelTable {
    use crate::kernels::wasm_simd128;
    KernelTable {
        gemm: wasm_simd128::gemm,
        gemv: wasm_simd128::gemv,
        add: wasm_simd128::add,
        mul: wasm_simd128::mul,
        // First slice (M4-01-T05) vectorizes the GEMM/GEMV/add/mul hot path;
        // the transcendental + reduction kernels delegate to the portable
        // scalar reference for now (same posture as the M3-13 RVV scaffold —
        // within-CPU-backend dispatch, not a cross-backend fallback, so
        // FR-EX-08 is unaffected). SIMD rewrites are a follow-up.
        relu: scalar::relu,
        sigmoid: scalar::sigmoid,
        tanh: scalar::tanh,
        gelu: scalar::gelu,
        softmax: scalar::softmax,
        layer_norm: scalar::layer_norm,
        fused_logmel: scalar_fused_logmel,
    }
}

#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
fn wasm_simd128_table() -> KernelTable {
    // Unreachable: `features::select_isa` never yields `WasmSimd128` off a
    // simd128-enabled wasm32 build (`CpuFeatures::detect().wasm_simd128` is a
    // compile-time `cfg!` constant), and `table_for` rejects it via
    // `CpuFeatures::supports`.
    unreachable!("WASM SIMD128 kernel table requested off a simd128-enabled wasm32 build")
}

// M4-17-T05: x86-64 AVX-512 f32 dispatch tier. The `avx512` kernels module
// is compiled only on x86-64; `features::select_isa` cannot yield `Avx512`
// elsewhere (the probe fields stay false) and `table_for` rejects it via
// `CpuFeatures::supports`, so the stub is genuinely unreachable.
//
// Transcendental activations (sigmoid / tanh / gelu) delegate to the AVX2
// kernels: `supports(Avx512)` includes AVX2+FMA (ADR M4-17 §(b)-4), any
// AVX-512 host can run them, and the delegation keeps the
// `simd-transcendental` feature posture automatically in sync with the AVX2
// tier (avx2 kernels are scalar-backed by default, vexp under the feature).
#[cfg(target_arch = "x86_64")]
fn avx512_table() -> KernelTable {
    use crate::kernels::avx2;
    use crate::kernels::avx512;
    KernelTable {
        gemm: avx512::gemm,
        gemv: avx512::gemv,
        add: avx512::add,
        mul: avx512::mul,
        relu: avx512::relu,
        sigmoid: avx2::sigmoid,
        tanh: avx2::tanh,
        gelu: avx2::gelu,
        softmax: avx512::softmax,
        layer_norm: avx512::layer_norm,
        fused_logmel: avx512::fused_logmel,
    }
}

#[cfg(not(target_arch = "x86_64"))]
fn avx512_table() -> KernelTable {
    // Unreachable: `features::select_isa` never yields `Avx512` off x86-64,
    // and `table_for` rejects it via `CpuFeatures::supports`.
    unreachable!("AVX-512 kernel table requested on a non-x86-64 target")
}

// M4-17-T05: specialized-tier f32 tables. The INT8 / BF16 / FP16 kernels are
// a separate dispatch surface (`kernels::kquant_gemv_i8*` /
// `kernels::gemm_bf16_on` / `kernels::gemm_fp16_on` — ADR M4-17 §(b)-2), so
// selecting one of these tiers as the process-wide path installs the best
// f32 kernels its `supports` gate guarantees (thin delegation, no second
// kernel implementation):
//
// - `Avx512Vnni` / `Avx512Bf16` → the AVX-512 f32 kernels (their gate
//   includes the full F/DQ/BW/VL bundle);
// - `AvxVnni256` → the AVX2 kernels (AVX-VNNI parts are AVX2+FMA parts);
// - `NeonFp16` / `NeonDotprod` / `NeonI8mm` / `NeonBf16` → the NEON baseline
//   kernels (NEON is unconditional on AArch64).
//
// Within-CPU-backend dispatch, not a cross-backend fallback: every table
// computes the same f32 ops within FP32 rounding (FR-EX-08 unaffected).
#[cfg(target_arch = "x86_64")]
fn avx512vnni_table() -> KernelTable {
    avx512_table()
}

#[cfg(not(target_arch = "x86_64"))]
fn avx512vnni_table() -> KernelTable {
    // Unreachable: probe fields are x86-64-only (see avx512_table stub).
    unreachable!("AVX-512 VNNI kernel table requested on a non-x86-64 target")
}

#[cfg(target_arch = "x86_64")]
fn avx512bf16_table() -> KernelTable {
    avx512_table()
}

#[cfg(not(target_arch = "x86_64"))]
fn avx512bf16_table() -> KernelTable {
    // Unreachable: probe fields are x86-64-only (see avx512_table stub).
    unreachable!("AVX-512 BF16 kernel table requested on a non-x86-64 target")
}

#[cfg(target_arch = "x86_64")]
fn avxvnni256_table() -> KernelTable {
    avx2_table()
}

#[cfg(not(target_arch = "x86_64"))]
fn avxvnni256_table() -> KernelTable {
    // Unreachable: probe fields are x86-64-only (see avx512_table stub).
    unreachable!("AVX-VNNI-256 kernel table requested on a non-x86-64 target")
}

#[cfg(target_arch = "aarch64")]
fn neon_ext_table() -> KernelTable {
    // Shared by NeonFp16 / NeonDotprod / NeonI8mm / NeonBf16 — all four
    // delegate their f32 table to the NEON baseline kernels (the specialized
    // kernels live on the separate dispatch surface, see above).
    neon_table()
}

#[cfg(not(target_arch = "aarch64"))]
fn neon_ext_table() -> KernelTable {
    // Unreachable: `features::select_isa` never yields a Neon* tier off
    // aarch64, and `table_for` rejects them via `CpuFeatures::supports`.
    unreachable!("NEON server-tier kernel table requested on a non-aarch64 target")
}

/// Maps an [`IsaPath`] to its kernel table — the single source of truth for
/// the ISA → implementation mapping (used by both production dispatch and the
/// `*_on` test entry points).
///
/// Deliberately an **exhaustive** match even though `IsaPath` is
/// `#[non_exhaustive]` (the attribute has no effect within the defining
/// crate): a future variant added without a table arm must be a compile
/// error here, never a runtime surprise (M4-17-T05).
fn build_table(isa: IsaPath) -> KernelTable {
    match isa {
        IsaPath::Scalar => scalar_table(),
        IsaPath::Avx2 => avx2_table(),
        IsaPath::Neon => neon_table(),
        IsaPath::Rvv => rvv_table(),
        IsaPath::Rvv071 => rvv071_table(),
        IsaPath::WasmSimd128 => wasm_simd128_table(),
        IsaPath::Avx512 => avx512_table(),
        IsaPath::Avx512Vnni => avx512vnni_table(),
        IsaPath::Avx512Bf16 => avx512bf16_table(),
        IsaPath::AvxVnni256 => avxvnni256_table(),
        IsaPath::NeonFp16 | IsaPath::NeonDotprod | IsaPath::NeonI8mm | IsaPath::NeonBf16 => {
            neon_ext_table()
        }
    }
}

struct Selected {
    isa: IsaPath,
    table: KernelTable,
}

static SELECTED: OnceLock<Selected> = OnceLock::new();

fn selected() -> &'static Selected {
    SELECTED.get_or_init(|| {
        // A malformed / unsupported `VOKRA_CPU_ISA` is a hard configuration
        // error: fail fast and loudly (an explicit panic, not a silent
        // fallback to another path — FR-EX-08 principle). The pure
        // `resolve_isa` path is unit-tested in `features`; production only
        // reaches this panic on genuine misconfiguration.
        let isa = features::resolve_isa()
            .unwrap_or_else(|e| panic!("invalid {} override: {e}", features::ENV_ISA_OVERRIDE));
        Selected {
            isa,
            table: build_table(isa),
        }
    })
}

/// The ISA path this process selected (host default, or the `VOKRA_CPU_ISA`
/// override). Fixed on first use and stable thereafter.
///
/// Intended for diagnostics: the ASR demo's one-line ISA log (M0-06-T26) and
/// the CI assertion that a binary picks the runner's path (M0-08-T18).
pub fn active_isa() -> IsaPath {
    selected().isa
}

/// The cached production kernel table for [`active_isa`].
pub(crate) fn table() -> &'static KernelTable {
    &selected().table
}

/// Builds a kernel table for an explicitly requested `isa`, used by the
/// `*_on` entry points (differential tests, forced-path comparison).
///
/// Requesting a path the host cannot run is an explicit
/// [`VokraError::BackendUnavailable`] (FR-EX-08 principle), never a silent
/// switch to another path.
pub(crate) fn table_for(isa: IsaPath) -> Result<KernelTable> {
    if CpuFeatures::detect().supports(isa) {
        Ok(build_table(isa))
    } else {
        Err(VokraError::BackendUnavailable(format!(
            "the {isa} kernel path is not available on this host CPU"
        )))
    }
}

// ---- fused log-mel per-frame dispatch (M2-04-T06) ---------------------------

/// Applies the fused mel-filterbank + `log10(max(·, floor))` for one frame's
/// power spectrum, selecting the fastest kernel supported by this host
/// (`Scalar` / `Avx2` / `Neon`) via [`active_isa`].
///
/// `weights` is row-major `[n_mels, n_bins]` (matching
/// `vokra_ops::mel::MelFilterbank::weights`), `power` has length `n_bins`,
/// and `out` has length `n_mels`. `floor` is the numerical clamp applied
/// before `log10` (typically `1e-10`, matching the Whisper front-end).
///
/// # Errors
/// Returns [`VokraError::InvalidArgument`] on any shape mismatch, matching
/// the boundary-validation regime of the rest of this crate's public
/// wrappers (NFR-RL-07). This is the safe entry point — the SIMD kernels'
/// `unsafe` + `#[target_feature]` boundary stays inside the crate.
///
/// # FR-EX-08 note
/// Scalar / AVX2 / NEON all compute the same op with the same result within
/// FP32 rounding. Choosing between them here is a within-CPU-backend
/// dispatch, orthogonal to the cross-backend explicit-op-error rule
/// FR-EX-08.
pub fn fused_log_mel_dispatch(
    pcm: &[f32],
    stft: &[f32],
    mel_fb: &[f32],
    n_frames: usize,
    out: &mut [f32],
) -> Result<()> {
    // The Whisper front-end floor (`log10(1e-10) = -10`) — same clamp used
    // by `vokra_ops::fused_log_mel_scalar` and by every SIMD path.
    const FLOOR: f32 = 1e-10;

    if pcm.is_empty() {
        return Err(VokraError::InvalidArgument(
            "fused_log_mel_dispatch: pcm (power) must be non-empty".into(),
        ));
    }
    if n_frames == 0 {
        return Err(VokraError::InvalidArgument(
            "fused_log_mel_dispatch: n_frames must be >= 1".into(),
        ));
    }
    // Interpret the arguments per the ticket's public wrapper contract:
    //   pcm    : one frame's power spectrum, length `n_bins`
    //   stft   : (unused at this layer — reserved for a future
    //            multi-frame streaming variant; a caller passing an empty
    //            slice keeps the wrapper cheap)
    //   mel_fb : row-major mel filterbank weights of length n_mels * n_bins
    //   out    : per-frame log-mel output of length n_mels
    // Shape validation:
    let n_bins = pcm.len();
    if !mel_fb.len().is_multiple_of(n_bins) {
        return Err(VokraError::InvalidArgument(format!(
            "fused_log_mel_dispatch: mel_fb.len() {} is not a multiple of n_bins {}",
            mel_fb.len(),
            n_bins
        )));
    }
    let n_mels = mel_fb.len() / n_bins;
    if out.len() != n_mels {
        return Err(VokraError::InvalidArgument(format!(
            "fused_log_mel_dispatch: out.len() {} != n_mels {} (derived from mel_fb / pcm)",
            out.len(),
            n_mels
        )));
    }
    // `stft` is currently reserved (unused by the per-frame kernel — the
    // caller has already applied window + FFT + `|·|²`). Silence the
    // unused-parameter lint without a name change.
    let _ = stft;

    (table().fused_logmel)(mel_fb, pcm, n_mels, n_bins, FLOOR, out);
    Ok(())
}

/// [`fused_log_mel_dispatch`] forced onto a specific `isa` — differential
/// test entry (mirrors the `*_on` pattern used by the other kernels).
///
/// Requesting a path the host cannot run is an explicit
/// [`VokraError::BackendUnavailable`] (never a silent switch — FR-EX-08).
#[allow(dead_code)] // available for M2-04-T06 parity harness; not used in production
pub(crate) fn fused_log_mel_dispatch_on(
    isa: IsaPath,
    pcm: &[f32],
    mel_fb: &[f32],
    out: &mut [f32],
) -> Result<()> {
    const FLOOR: f32 = 1e-10;
    let n_bins = pcm.len();
    if n_bins == 0 || !mel_fb.len().is_multiple_of(n_bins) {
        return Err(VokraError::InvalidArgument(
            "fused_log_mel_dispatch_on: bad pcm / mel_fb shape".into(),
        ));
    }
    let n_mels = mel_fb.len() / n_bins;
    if out.len() != n_mels {
        return Err(VokraError::InvalidArgument(format!(
            "fused_log_mel_dispatch_on: out.len() {} != n_mels {}",
            out.len(),
            n_mels
        )));
    }
    (table_for(isa)?.fused_logmel)(mel_fb, pcm, n_mels, n_bins, FLOOR, out);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn active_isa_is_stable_and_host_supported() {
        let a = active_isa();
        let b = active_isa();
        assert_eq!(a, b, "selection must be fixed after the first call");
        assert!(CpuFeatures::detect().supports(a));

        // On this test process (no override) the active path equals the host
        // best path.
        assert_eq!(a, CpuFeatures::detect().best_isa());
    }

    #[test]
    fn table_for_scalar_always_available() {
        assert!(table_for(IsaPath::Scalar).is_ok());
    }

    #[test]
    fn table_for_rejects_unavailable_path() {
        // Exactly one of AVX2 / NEON / RVV can be true on any given host arch
        // (they are arch-exclusive); every unsupported path must be an
        // explicit BackendUnavailable, never a silent fallback (FR-EX-08).
        let feats = CpuFeatures::detect();
        for isa in IsaPath::ALL_SIMD {
            if !feats.supports(isa) {
                assert!(matches!(
                    table_for(isa),
                    Err(VokraError::BackendUnavailable(_))
                ));
            }
        }
    }

    // -------------------------------------------------------------------
    // M2-04-T06 fused log-mel dispatch tests
    // -------------------------------------------------------------------

    /// The scalar `fused_logmel` entry in the table computes the same output
    /// as the hand reference (1 band × 3 unit-weight bins over [1, 2, 4] →
    /// log10(7)).
    #[test]
    fn fused_logmel_scalar_table_matches_hand_value() {
        let table = table_for(IsaPath::Scalar).expect("scalar always available");
        let weights = [1.0f32, 1.0, 1.0];
        let power = [1.0f32, 2.0, 4.0];
        let mut out = [0.0f32; 1];
        (table.fused_logmel)(&weights, &power, 1, 3, 1e-10, &mut out);
        let want = 7.0f32.log10();
        assert!(
            (out[0] - want).abs() < 1e-6,
            "fused_logmel scalar table: got {} want {want}",
            out[0]
        );
    }

    /// The safe public wrapper routes shape errors as explicit
    /// [`VokraError::InvalidArgument`], never silently truncating or
    /// swapping to a fallback (NFR-RL-07, FR-EX-08 principle).
    #[test]
    fn fused_log_mel_dispatch_rejects_bad_shapes() {
        // Empty power spectrum.
        let mut out = [0.0f32; 4];
        assert!(matches!(
            fused_log_mel_dispatch(&[], &[], &[0.0; 4], 1, &mut out),
            Err(VokraError::InvalidArgument(_))
        ));
        // mel_fb length not a multiple of n_bins.
        let pcm = [1.0f32, 2.0, 3.0]; // n_bins = 3
        let mel_fb = [1.0f32; 7]; // 7 % 3 != 0
        assert!(matches!(
            fused_log_mel_dispatch(&pcm, &[], &mel_fb, 1, &mut out),
            Err(VokraError::InvalidArgument(_))
        ));
        // out.len() does not match derived n_mels.
        let mel_fb = [1.0f32; 6]; // n_bins=3 → n_mels=2
        let mut bad_out = [0.0f32; 4];
        assert!(matches!(
            fused_log_mel_dispatch(&pcm, &[], &mel_fb, 1, &mut bad_out),
            Err(VokraError::InvalidArgument(_))
        ));
        // n_frames == 0.
        let mut ok_out = [0.0f32; 2];
        assert!(matches!(
            fused_log_mel_dispatch(&pcm, &[], &mel_fb, 0, &mut ok_out),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    /// End-to-end via the safe wrapper: identity-ish filterbank + a positive
    /// power spectrum produces finite `log10` values in the expected band.
    #[test]
    fn fused_log_mel_dispatch_end_to_end_smoke() {
        // 2 mel bands × 5 bins; band 0 sums bins 0..3, band 1 sums bins 2..5.
        let weights: Vec<f32> = vec![
            1.0, 1.0, 1.0, 0.0, 0.0, // band 0
            0.0, 0.0, 1.0, 1.0, 1.0, // band 1
        ];
        let power = [0.5f32, 1.0, 2.0, 4.0, 8.0];
        let mut out = [0.0f32; 2];
        fused_log_mel_dispatch(&power, &[], &weights, 1, &mut out).expect("well-formed inputs");
        // band 0: 0.5+1+2 = 3.5, log10 ≈ 0.5441; band 1: 2+4+8 = 14, log10 ≈ 1.1461.
        // Every ISA path stays inside the plan-spec 1e-5 SIMD ceiling; the
        // scalar path is bit-close to the hand value.
        assert!((out[0] - 3.5_f32.log10()).abs() < 1e-5);
        assert!((out[1] - 14.0_f32.log10()).abs() < 1e-5);
    }

    /// Forcing an unavailable ISA is an explicit
    /// [`VokraError::BackendUnavailable`] — never a silent switch (FR-EX-08).
    #[test]
    fn fused_log_mel_dispatch_on_rejects_unavailable_isa() {
        let feats = CpuFeatures::detect();
        let pcm = [1.0f32, 2.0, 3.0];
        let mel_fb = [1.0f32; 3];
        let mut out = [0.0f32; 1];
        for isa in IsaPath::ALL_SIMD {
            if !feats.supports(isa) {
                assert!(matches!(
                    fused_log_mel_dispatch_on(isa, &pcm, &mel_fb, &mut out),
                    Err(VokraError::BackendUnavailable(_))
                ));
            }
        }
    }
}
