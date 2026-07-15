//! CosyVoice2 quality-degradation gate (M3-09-T23; NFR-QL-02).
//!
//! Compares a `cosyvoice2` Vokra-native synthesis against a `reference`
//! waveform (typically the upstream PyTorch checkpoint on the same input)
//! using the shared log-mel L1 loss and asserts the NFR-QL-02 5 % gate.
//! This is the policy-level entry point the M3-09 T23 spec pins:
//!
//! ```text
//! docs/tickets/m3/M3-09-cosyvoice2.md §T23
//! "vokra-eval crate（M1、FR-TL-03）の mel_loss / UTMOS / DNSMOS を
//!  CosyVoice2 出力に適用し、PyTorch reference との劣化が 5% 未満
//!  （NFR-QL-02）であることを CI で機械検証する。"
//! ```
//!
//! The threshold policy is fixed at [`COSYVOICE2_MEL_LOSS_THRESHOLD`] and
//! documented so the CI gate cannot silently drift.
//!
//! # Zero-dep / partial gate policy (NFR-DS-02, M4-18 T08)
//!
//! The weight-free gate uses **mel_loss only**: the same
//! `mel_loss_only = true` posture the M2-08 [`check_degradation`] helper
//! carries applies to [`check_cosyvoice2_degradation`], so a caller (CI
//! job / audit script) can surface the partial-gate caveat. The
//! UTMOS-augmented entry point
//! ([`check_cosyvoice2_degradation_with_utmos`]) takes an **injected**
//! [`AudioMosMetric`] scorer (M4-18 T08 — the real UTMOS weights are
//! owner-deferred, so no weight path is hard-coded here) plus an explicit
//! [`MosDomain`]; because CosyVoice2 synthesizes through the Mimi codec,
//! the codec/streaming domain is advisory-only until the owner-side
//! calibration study clears it (see `degradation` module docs). DNSMOS
//! remains fail-closed (M4-18 T03).
//!
//! # Frontend spec (bit-exact MEL, CLAUDE.md STFT ≠ FFT)
//!
//! The mel-loss uses the M2-08 default librosa attrs (`n_fft=1024`,
//! `hop_length=256`, `n_mels=80`) so the CosyVoice2 24 kHz output is
//! comparable to the PyTorch reference under identical STFT / mel
//! filters. A caller who needs a **model-specific** frontend (matching
//! `vokra.cosyvoice2.frontend.*` bit-for-bit — T05 follow-on) composes
//! [`MelLoss::from_attrs`] directly; the policy-level helper here uses
//! the librosa default that upstream CosyVoice2's own eval harness
//! ships (upstream `evaluation_metric.py`).
//!
//! # No silent fallback (FR-EX-08)
//!
//! - Sample rate mismatch → loud `InvalidArgument` (via `MelLoss::eval_audio`).
//! - Non-finite / non-positive threshold → loud `InvalidArgument`.
//! - Too-short inputs → propagates `MelLoss` error verbatim.

use crate::degradation::{DegradationReport, MosDomain};
use crate::metrics::AudioMosMetric;
use crate::{MelLoss, check_degradation};
use vokra_core::{Result, VokraError};

/// The NFR-QL-02 5 % gate as an f64 threshold.
///
/// Kept as a named constant so a change touches this file (and the
/// docstring caller) — never a scattered `0.05` literal anywhere in the
/// CosyVoice2 CI path.
pub const COSYVOICE2_MEL_LOSS_THRESHOLD: f64 = 0.05;

/// CosyVoice2 output sample rate (Hz). Fixed at 24 kHz — the upstream
/// model card and the Mimi codec's native rate. Kept here as a constant
/// so the CI gate can validate the reference matches without importing
/// [`CosyVoice2Config`] from `vokra-models` (which would create a
/// vokra-eval → vokra-models dep — banned by the crate layer: eval
/// depends only on vokra-core + vokra-ops).
pub const COSYVOICE2_SAMPLE_RATE: u32 = 24_000;

/// Runs the CosyVoice2 mel-loss degradation gate.
///
/// `reference` and `hypothesis` are both mono PCM at
/// [`COSYVOICE2_SAMPLE_RATE`] (24 kHz — mismatch is a loud error, no
/// silent resample). The report's `passes_5pct_gate` field is the
/// NFR-QL-02 verdict.
///
/// This is a thin policy wrapper around [`check_degradation`] fixing
/// (a) the sample rate to 24 kHz, and (b) the threshold to
/// [`COSYVOICE2_MEL_LOSS_THRESHOLD`]. Callers who need a different rate
/// or threshold reach for [`check_degradation`] directly.
///
/// # Errors
///
/// [`VokraError::InvalidArgument`] on non-24 kHz PCM or MEL front-end
/// error (too-short inputs).
pub fn check_cosyvoice2_degradation(
    reference: &[f32],
    hypothesis: &[f32],
    sample_rate: u32,
) -> Result<DegradationReport> {
    if sample_rate != COSYVOICE2_SAMPLE_RATE {
        return Err(VokraError::InvalidArgument(format!(
            "check_cosyvoice2_degradation: sample_rate {sample_rate} != {COSYVOICE2_SAMPLE_RATE} \
             (CosyVoice2 output is fixed at {COSYVOICE2_SAMPLE_RATE} Hz — Mimi codec native \
             rate; the gate does not silently resample, FR-EX-08)"
        )));
    }
    check_degradation(
        reference,
        hypothesis,
        sample_rate,
        COSYVOICE2_MEL_LOSS_THRESHOLD,
    )
}

/// UTMOS-augmented CosyVoice2 gate (M4-18 T08).
///
/// Mirrors [`check_cosyvoice2_degradation`] (fixed 24 kHz rate + the
/// [`COSYVOICE2_MEL_LOSS_THRESHOLD`] policy) but additionally runs the
/// injected `mos` scorer through
/// [`check_degradation_with_utmos`](crate::check_degradation_with_utmos),
/// returning a report with `mel_loss_only = false`.
///
/// The scorer is **injected** (`&dyn AudioMosMetric`): the M4-18 kickoff
/// gate deferred the real UTMOS weights, so this crate never hard-codes a
/// weight path — a weight-less caller uses
/// [`check_cosyvoice2_degradation`] and inherits its honest partial-gate
/// flag. `domain` must be stated explicitly; note CosyVoice2 synthesizes
/// **through the Mimi codec**, so until the owner-side calibration study
/// validates the codec domain, [`MosDomain::CodecStreaming`] (advisory-only)
/// is the honest choice — [`MosDomain::TtsSynthesis`] turns the MOS half
/// into a hard gate.
///
/// # Errors
///
/// [`VokraError::InvalidArgument`] on non-24 kHz PCM (before any scoring),
/// plus everything `check_degradation_with_utmos` rejects (scorer errors,
/// non-positive reference score, MEL front-end errors).
pub fn check_cosyvoice2_degradation_with_utmos(
    reference: &[f32],
    hypothesis: &[f32],
    sample_rate: u32,
    mos: &dyn AudioMosMetric,
    domain: MosDomain,
) -> Result<DegradationReport> {
    if sample_rate != COSYVOICE2_SAMPLE_RATE {
        return Err(VokraError::InvalidArgument(format!(
            "check_cosyvoice2_degradation_with_utmos: sample_rate {sample_rate} != \
             {COSYVOICE2_SAMPLE_RATE} (CosyVoice2 output is fixed at {COSYVOICE2_SAMPLE_RATE} Hz \
             — Mimi codec native rate; the gate does not silently resample, FR-EX-08)"
        )));
    }
    crate::check_degradation_with_utmos(
        reference,
        hypothesis,
        sample_rate,
        COSYVOICE2_MEL_LOSS_THRESHOLD,
        mos,
        domain,
    )
}

/// Convenience: builds a shared [`MelLoss`] pre-configured for CosyVoice2's
/// output rate.
///
/// A caller that needs to compute the mel loss directly (e.g. a per-tensor
/// parity assertion that reports the raw loss value alongside the pass /
/// fail verdict) uses this instead of the pass/fail wrapper.
#[must_use]
pub fn cosyvoice2_mel_loss() -> MelLoss {
    // M2-08 librosa-defaults — the same shape `check_degradation` uses
    // internally. Keeping the two in sync means a caller's raw mel-loss
    // number matches the wrapper's `mel_loss_quant` exactly.
    MelLoss::new(COSYVOICE2_SAMPLE_RATE, 1024, 256, 80)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::utmos::{ConvActivation, HeadPool, TransformerNorm, Utmos, UtmosConfig};

    const N: usize = 24_000; // 1 s of PCM at 24 kHz

    fn tone(freq: f32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| {
                (2.0 * std::f32::consts::PI * freq * i as f32 / COSYVOICE2_SAMPLE_RATE as f32).sin()
            })
            .collect()
    }

    fn noise(n: usize) -> Vec<f32> {
        // Deterministic pseudo-noise (no RNG dep — NFR-DS-02).
        let mut state: u32 = 0xC0FFEE_u32;
        (0..n)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (state >> 8) as f32 / (1u32 << 23) as f32 - 1.0
            })
            .collect()
    }

    fn add(a: &[f32], b: &[f32], scale: f32) -> Vec<f32> {
        a.iter().zip(b).map(|(x, y)| x + scale * y).collect()
    }

    #[test]
    fn identical_inputs_pass_with_zero_relative_delta() {
        // The reference-vs-reference case is exactly 0 — bit-identical
        // audio must not trip the gate.
        let ref_pcm = tone(220.0, N);
        let report = check_cosyvoice2_degradation(&ref_pcm, &ref_pcm, COSYVOICE2_SAMPLE_RATE)
            .expect("identical inputs succeed");
        assert_eq!(report.mel_loss_ref, 0.0);
        assert_eq!(report.mel_loss_quant, 0.0);
        assert_eq!(report.relative_delta, 0.0);
        assert!(report.passes_5pct_gate);
        assert!(report.mel_loss_only, "UTMOS unavailable (M1-09b)");
    }

    #[test]
    fn large_noise_fails_the_5pct_gate() {
        // A large perturbation (0.5 · noise) must fail — the raw mel-loss
        // is > 0 and the epsilon-floored denominator makes the ratio
        // astronomically large.
        let ref_pcm = tone(220.0, N);
        let hyp = add(&ref_pcm, &noise(N), 0.5);
        let report = check_cosyvoice2_degradation(&ref_pcm, &hyp, COSYVOICE2_SAMPLE_RATE)
            .expect("noisy comparison succeeds");
        assert!(report.mel_loss_quant > 0.0);
        assert!(!report.passes_5pct_gate);
    }

    #[test]
    fn sample_rate_mismatch_fails_loudly() {
        // FR-EX-08: no silent resample. 16 kHz PCM must be rejected before
        // any MEL processing runs.
        let ref_pcm = tone(220.0, N);
        let hyp = tone(220.0, N);
        let err = check_cosyvoice2_degradation(&ref_pcm, &hyp, 16_000)
            .expect_err("wrong sample rate must fail");
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn threshold_constant_is_five_percent() {
        // The 5 % threshold is fixed by NFR-QL-02; if it ever changes,
        // this test forces a documentation update in the same commit.
        assert!((COSYVOICE2_MEL_LOSS_THRESHOLD - 0.05).abs() < f64::EPSILON);
    }

    #[test]
    fn sample_rate_constant_is_24khz() {
        // The 24 kHz rate is the Mimi codec native rate + the CosyVoice2
        // model card constant; same doc-update tripwire as above.
        assert_eq!(COSYVOICE2_SAMPLE_RATE, 24_000);
    }

    /// A tiny real UTMOS skeleton at the CosyVoice2 rate (24 kHz) whose
    /// affine shifts scores into a MOS-like positive band.
    fn tiny_utmos_24k() -> Utmos {
        let config = UtmosConfig {
            sample_rate: COSYVOICE2_SAMPLE_RATE,
            conv_channels: vec![4, 6],
            conv_kernels: vec![5, 3],
            conv_strides: vec![3, 2],
            conv_activation: ConvActivation::Gelu,
            n_layer: 1,
            n_head: 2,
            hidden_dim: 6,
            ffn_dim: 12,
            norm: TransformerNorm::Post,
            ln_eps: 1e-5,
            head_dims: vec![4, 1],
            head_pool: HeadPool::MeanAfter,
            head_scale: 1.0,
            head_offset: 3.0,
        };
        Utmos::synthesized(config, 0x4D34_5F31_385F_5531).expect("tiny utmos 24k")
    }

    #[test]
    fn utmos_gate_runs_with_injected_scorer_and_clears_partial_flag() {
        let ref_pcm = tone(220.0, N);
        let m = tiny_utmos_24k();
        // Identical audio through a deterministic scorer: exact zero MOS
        // decrease, and the mel half is 0 — the full gate passes with the
        // partial flag cleared. CodecStreaming is the honest domain for
        // Mimi-codec output (advisory-only until the owner study).
        let report = check_cosyvoice2_degradation_with_utmos(
            &ref_pcm,
            &ref_pcm,
            COSYVOICE2_SAMPLE_RATE,
            &m,
            MosDomain::CodecStreaming,
        )
        .expect("UTMOS-augmented gate runs");
        assert!(!report.mel_loss_only);
        let a = report.utmos.expect("assessment present");
        assert_eq!(a.rel_decrease, 0.0);
        assert!(a.advisory_only, "codec domain is advisory-only");
        assert!(report.passes_5pct_gate);
    }

    #[test]
    fn utmos_gate_rejects_wrong_sample_rate_before_scoring() {
        // FR-EX-08: the 24 kHz policy check fires before any scoring —
        // even with a 16 kHz-config scorer the error is the rate policy.
        let ref_pcm = tone(220.0, N);
        let m = tiny_utmos_24k();
        let err = check_cosyvoice2_degradation_with_utmos(
            &ref_pcm,
            &ref_pcm,
            16_000,
            &m,
            MosDomain::CodecStreaming,
        )
        .expect_err("wrong sample rate must fail");
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn cosyvoice2_mel_loss_shape_matches_wrapper_defaults() {
        // The convenience factory returns a MelLoss with sample rate
        // matching the wrapper's fixed 24 kHz — a caller composing raw
        // mel-loss numbers against the wrapper's verdict must not see
        // a rate drift.
        let ml = cosyvoice2_mel_loss();
        assert_eq!(ml.sample_rate(), COSYVOICE2_SAMPLE_RATE);
    }

    #[test]
    fn cosyvoice2_mel_loss_agrees_with_wrapper_on_identical_inputs() {
        // Raw mel-loss == wrapper's `mel_loss_quant` on identical inputs
        // (both = 0 exactly) — the internal-oracle contract linking the
        // library factory and the wrapper's frontend.
        let ref_pcm = tone(220.0, N);
        let ml = cosyvoice2_mel_loss();
        let raw = ml
            .loss(&ref_pcm, &ref_pcm)
            .expect("mel_loss on identical inputs");
        let report =
            check_cosyvoice2_degradation(&ref_pcm, &ref_pcm, COSYVOICE2_SAMPLE_RATE).unwrap();
        assert_eq!(raw, report.mel_loss_quant);
        assert_eq!(raw, 0.0);
    }
}
