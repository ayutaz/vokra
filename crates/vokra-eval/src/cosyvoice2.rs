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
//! # Zero-dep, weight-free (NFR-DS-02 / partial gate policy)
//!
//! The library gate uses **mel_loss only** — UTMOS / DNSMOS are neural MOS
//! predictors that need model weights (M1-09b, blocked). The same
//! `mel_loss_only = true` posture the M2-08 [`check_degradation`] helper
//! carries applies here: [`check_cosyvoice2_degradation`] sets the flag,
//! so a caller (CI job / audit script) can surface the partial-gate
//! caveat. The API surface leaves a symmetric UTMOS entry point
//! ([`check_cosyvoice2_degradation_with_utmos`]) that returns
//! [`VokraError::NotImplemented`] today — the follow-on session drops the
//! weights in without an API break.
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

use crate::degradation::DegradationReport;
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

/// UTMOS + DNSMOS-augmented CosyVoice2 gate (M1-09b; blocked on weights).
///
/// Placeholder entry point mirroring [`check_cosyvoice2_degradation`]:
/// returns [`VokraError::NotImplemented`] today so the CI wiring can be
/// laid out now without a follow-up API break. Once UTMOS weights land
/// this body drops in a `check_degradation_with_utmos` call and returns
/// a full report with `mel_loss_only = false`.
pub fn check_cosyvoice2_degradation_with_utmos(
    _reference: &[f32],
    _hypothesis: &[f32],
    _sample_rate: u32,
) -> Result<DegradationReport> {
    Err(VokraError::NotImplemented(
        "check_cosyvoice2_degradation_with_utmos: UTMOS weights not delivered (M1-09b); \
         mel-loss-only gate available via check_cosyvoice2_degradation",
    ))
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

    #[test]
    fn utmos_gate_returns_not_implemented() {
        let ref_pcm = tone(220.0, N);
        let err =
            check_cosyvoice2_degradation_with_utmos(&ref_pcm, &ref_pcm, COSYVOICE2_SAMPLE_RATE)
                .expect_err("UTMOS weights not delivered");
        assert!(matches!(err, VokraError::NotImplemented(_)));
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
