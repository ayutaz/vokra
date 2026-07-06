//! HiFi-GAN INT8 opt-in verification gate (M2-08-T12).
//!
//! This module is the *runtime* half of the HiFi-GAN INT8 opt-in contract:
//! when a [`QuantPolicy`] flips `hifigan_int8_opt_in=true` (via the atomic
//! [`QuantPolicy::with_hifigan_int8_opt_in`] path, T10) AND the loaded model
//! actually exercises the HiFi-GAN op, the session / bench flow is required
//! to attach a fresh [`DegradationReport`] before construction succeeds.
//!
//! Three branches:
//!
//! 1. **opt-in + eval pass** (`passes_5pct_gate = true`) → construction allowed.
//! 2. **opt-in + eval not run** (no [`DegradationReport`] attached) →
//!    [`VokraError::HifiganInt8VerifyMissing`].
//! 3. **opt-in + eval fail** (`passes_5pct_gate = false`) →
//!    [`VokraError::HifiganInt8DegradationExceeded { delta, threshold }`].
//!
//! Vocos / BigVGAN (registry `DowngradePolicy::Forbidden`, T09) are rejected
//! by [`validate_policy_against_model`](super::validate) *before* execution
//! ever reaches this verify step — they never observe [`DegradationReport`].
//! Only HiFi-GAN, whose registry `DowngradePolicy::HifiganOptIn` opens the
//! opt-in door, funnels through here.
//!
//! # Zero-dep leaf
//!
//! [`DegradationReport`] is a pure data type — computing it requires a mel
//! metric that lives in `vokra-eval` (T11); `vokra-core` never depends on
//! `vokra-eval` (NFR-DS-02). Callers (`vokra-cli`, `vokra-models`) compute
//! the report and pass it here.

use crate::error::{Result, VokraError};
use crate::quant::policy::QuantPolicy;

/// NFR-QL-02: MEL loss / UTMOS degradation gate threshold (5%).
pub const DEGRADATION_THRESHOLD: f64 = 0.05;

/// Output of a `vokra_eval::check_degradation` (T11) run, threaded into the
/// M2-08-T12 verify gate as a data type so `vokra-core` stays a zero-dep
/// leaf.
///
/// The runtime side of the HiFi-GAN INT8 opt-in contract only needs to know
/// whether the eval *ran* and whether it *passed*; the raw MEL-loss numbers
/// are carried through so an error message can surface the delta.
#[derive(Debug, Clone, PartialEq)]
pub struct DegradationReport {
    /// Baseline MEL-loss (typically `mel_loss(fp32_ref, fp32_ref) = 0.0`, or
    /// against a curated reference clip).
    pub mel_loss_ref: f64,
    /// Quantized-path MEL-loss against the same reference.
    pub mel_loss_quant: f64,
    /// `(mel_loss_quant - mel_loss_ref) / max(mel_loss_ref, ε)`. Same as
    /// `vokra_eval::degradation::DegradationReport.relative_delta`.
    pub relative_delta: f64,
    /// Whether the run cleared the NFR-QL-02 5% gate — the sole flag the
    /// verify branch uses to distinguish "pass" from "fail". Duplicated from
    /// `relative_delta <= threshold` on the eval side so callers can encode
    /// non-mel-loss gates (e.g. UTMOS once weights land) without changing
    /// this shape.
    pub passes_5pct_gate: bool,
}

impl DegradationReport {
    /// Convenience: build a report from raw MEL-loss numbers and a threshold.
    ///
    /// `epsilon` is used as the denominator floor when `mel_loss_ref` is
    /// (numerically) zero — matches the `vokra_eval::degradation` helper (T11).
    pub fn from_mel_loss(mel_loss_ref: f64, mel_loss_quant: f64, threshold: f64) -> Self {
        // Match the `vokra_eval::degradation::check_degradation` epsilon
        // (T11) exactly — 1e-10 keeps the numeric shape stable when the
        // baseline collapses to zero.
        let denom = mel_loss_ref.max(1e-10);
        let relative_delta = (mel_loss_quant - mel_loss_ref) / denom;
        Self {
            mel_loss_ref,
            mel_loss_quant,
            relative_delta,
            passes_5pct_gate: relative_delta <= threshold,
        }
    }
}

/// Enforces the HiFi-GAN INT8 opt-in verification gate (M2-08-T12).
///
/// Call this at session / bench construction, *after* the policy has been
/// resolved but *before* any inference. `hifigan_op_in_use` is `true` when
/// the loaded model exercises the HiFi-GAN op (e.g. piper-plus voices in
/// `vokra-models::piper_plus`) — pass `false` for models with no HiFi-GAN
/// (Whisper, Silero VAD, CAM++) and the check short-circuits.
///
/// # Branches
///
/// | policy.opt_in | hifigan in use | report | outcome |
/// |---|---|---|---|
/// | `false` | any | any | `Ok(())` — HiFi-GAN INT8 not opted in |
/// | `true` | `false` | any | `Ok(())` — the model does not exercise HiFi-GAN |
/// | `true` | `true` | `None` | [`VokraError::HifiganInt8VerifyMissing`] |
/// | `true` | `true` | `Some(passes_5pct_gate=true)` | `Ok(())` |
/// | `true` | `true` | `Some(passes_5pct_gate=false)` | [`VokraError::HifiganInt8DegradationExceeded`] |
///
/// Vocos / BigVGAN never reach here — [`validate_policy_against_model`] (T09)
/// rejects them at a higher altitude.
pub fn verify_hifigan_int8(
    policy: &QuantPolicy,
    hifigan_op_in_use: bool,
    report: Option<&DegradationReport>,
) -> Result<()> {
    // Not opting into INT8 for HiFi-GAN, or the model doesn't run HiFi-GAN.
    // Either way this gate is a no-op.
    if !policy.hifigan_int8_opt_in() || !hifigan_op_in_use {
        return Ok(());
    }
    // Opt-in + HiFi-GAN in use: an eval report is mandatory (FR-EX-08 — no
    // silent shipping of an unverified INT8 vocoder).
    let Some(report) = report else {
        return Err(VokraError::HifiganInt8VerifyMissing);
    };
    if report.passes_5pct_gate {
        return Ok(());
    }
    Err(VokraError::HifiganInt8DegradationExceeded {
        delta: report.relative_delta,
        threshold: DEGRADATION_THRESHOLD,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::policy::CalibrationRef;
    use crate::quant::scheme::QuantScheme;

    fn opt_in_policy() -> QuantPolicy {
        QuantPolicy::new(QuantScheme::Fp16)
            .with_hifigan_int8_opt_in(CalibrationRef::new("hifigan-int8-cal-v1"))
    }

    fn no_opt_in_policy() -> QuantPolicy {
        QuantPolicy::new(QuantScheme::Fp16)
    }

    #[test]
    fn no_opt_in_short_circuits_regardless_of_report() {
        // opt-in disabled → gate is a no-op even when HiFi-GAN op is in use.
        verify_hifigan_int8(&no_opt_in_policy(), true, None).unwrap();
        verify_hifigan_int8(&no_opt_in_policy(), false, None).unwrap();
    }

    #[test]
    fn opt_in_but_no_hifigan_op_short_circuits() {
        // A whisper-only session (no HiFi-GAN) opts into INT8 for future
        // vocoder use — the gate does not fire while the current model does
        // not exercise HiFi-GAN.
        verify_hifigan_int8(&opt_in_policy(), false, None).unwrap();
    }

    #[test]
    fn opt_in_plus_hifigan_without_report_errors_verify_missing() {
        // Branch (b): opt-in + HiFi-GAN in use + no attached report → hard error.
        let err = verify_hifigan_int8(&opt_in_policy(), true, None).unwrap_err();
        assert!(
            matches!(err, VokraError::HifiganInt8VerifyMissing),
            "got: {err}"
        );
    }

    #[test]
    fn opt_in_plus_hifigan_with_passing_report_ok() {
        // Branch (a): opt-in + HiFi-GAN in use + passing report → allowed.
        // Passing = relative delta 2% <= 5% threshold.
        let report = DegradationReport::from_mel_loss(1.0, 1.02, DEGRADATION_THRESHOLD);
        assert!(report.passes_5pct_gate);
        verify_hifigan_int8(&opt_in_policy(), true, Some(&report)).unwrap();
    }

    #[test]
    fn opt_in_plus_hifigan_with_failing_report_errors_degradation_exceeded() {
        // Branch (c): opt-in + HiFi-GAN in use + failing report → hard error
        // carrying the observed delta and the threshold.
        let report = DegradationReport::from_mel_loss(1.0, 1.20, DEGRADATION_THRESHOLD);
        assert!(!report.passes_5pct_gate);
        let err = verify_hifigan_int8(&opt_in_policy(), true, Some(&report)).unwrap_err();
        match err {
            VokraError::HifiganInt8DegradationExceeded { delta, threshold } => {
                // delta ≈ 0.20, threshold == 0.05.
                assert!((delta - 0.20).abs() < 1e-9, "delta={delta}");
                assert!((threshold - 0.05).abs() < 1e-12, "threshold={threshold}");
            }
            other => panic!("unexpected variant: {other}"),
        }
    }

    #[test]
    fn degradation_report_from_mel_loss_handles_zero_baseline() {
        // A zero baseline must not divide by zero — the eval helper (T11)
        // uses the same 1e-10 floor.
        let report = DegradationReport::from_mel_loss(0.0, 0.0, DEGRADATION_THRESHOLD);
        assert!(report.relative_delta.is_finite());
        assert!(report.passes_5pct_gate);
    }

    #[test]
    fn threshold_constant_matches_nfr_ql_02() {
        // Sanity: the crate-wide constant is the NFR-QL-02 gate value.
        assert!((DEGRADATION_THRESHOLD - 0.05).abs() < f64::EPSILON);
    }
}
