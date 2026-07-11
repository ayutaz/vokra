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

/// Maps an [`IsaPath`] to its kernel table — the single source of truth for
/// the ISA → implementation mapping (used by both production dispatch and the
/// `*_on` test entry points).
fn build_table(isa: IsaPath) -> KernelTable {
    match isa {
        IsaPath::Scalar => scalar_table(),
        IsaPath::Avx2 => avx2_table(),
        IsaPath::Neon => neon_table(),
        IsaPath::Rvv => rvv_table(),
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
    if mel_fb.len() % n_bins != 0 {
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
    if n_bins == 0 || mel_fb.len() % n_bins != 0 {
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
        for isa in [IsaPath::Avx2, IsaPath::Neon, IsaPath::Rvv] {
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
        for isa in [IsaPath::Avx2, IsaPath::Neon, IsaPath::Rvv] {
            if !feats.supports(isa) {
                assert!(matches!(
                    fused_log_mel_dispatch_on(isa, &pcm, &mel_fb, &mut out),
                    Err(VokraError::BackendUnavailable(_))
                ));
            }
        }
    }
}
