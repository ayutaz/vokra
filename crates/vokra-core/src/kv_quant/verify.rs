//! FR-QT-04 quality-verification pipeline hooks for KV cache quantization
//! (M3-04-T12 / T13).
//!
//! # Scope
//!
//! This module carries the **policy shape** of NFR-QL-02: given an FP32
//! baseline and a set of measured GEMV / decoder outputs under each
//! [`KvQuant`] mode, decide whether any mode exceeds the 5% degradation gate
//! and produce a report a CLI / CI job can render.
//!
//! The **metric bodies** (MEL loss, WER, UTMOS, DNSMOS) live in the
//! `vokra-eval` crate; this module deliberately does not depend on it — the
//! zero-dep leaf constraint of `vokra-core` forbids reaching up the stack.
//! Downstream integration (`vokra-cli`, `crates/vokra-eval/tests/`) wires an
//! evaluation into a [`KvQuantVerifyReport`] via the setter methods below.
//!
//! # NFR-QL-02 threshold
//!
//! The gate is **5% relative degradation** vs the FP32 baseline. The value is
//! symmetric across metric direction:
//!
//! - **MEL loss**: `(mel_loss_q - mel_loss_fp32) / mel_loss_fp32 <= 0.05`
//!   (loss increases → bad).
//! - **WER**: same as MEL loss; higher is worse.
//! - **UTMOS / DNSMOS**: `(score_fp32 - score_q) / score_fp32 <= 0.05`
//!   (perceptual score decreases → bad).
//!
//! # UTMOS / DNSMOS placeholder
//!
//! Per T13, this module exposes an API surface for UTMOS / DNSMOS but the
//! scorers themselves are M4 push-outs — weights are not yet distributable
//! (M1-07 land time carry-over). The [`KvQuantVerifyReport::utmos_unavailable`]
//! flag surfaces the state so a CI summary can report "M4 gate" honestly
//! rather than silent-skipping.

use super::KvQuant;

/// The NFR-QL-02 degradation gate threshold: 5%.
pub const DEGRADATION_THRESHOLD: f32 = 0.05;

/// Per-quantization-mode degradation metrics.
#[derive(Debug, Clone, Copy)]
pub struct KvQuantMetric {
    /// Which quantization mode this metric refers to.
    pub mode: KvQuant,
    /// Relative MEL loss delta vs FP32 baseline, computed by `vokra-eval`.
    /// `None` = not measured (e.g. this model is not a TTS model, or the
    /// caller ran a WER-only ASR eval).
    pub mel_loss_rel_delta: Option<f32>,
    /// Relative WER delta vs FP32 baseline (Whisper / ASR only).
    pub wer_rel_delta: Option<f32>,
    /// Relative UTMOS score decrease vs FP32 baseline. `None` = not measured
    /// (UTMOS weights unavailable, T13 M4 push-out).
    pub utmos_rel_delta: Option<f32>,
    /// Relative DNSMOS score decrease vs FP32 baseline. `None` = not measured.
    pub dnsmos_rel_delta: Option<f32>,
}

impl KvQuantMetric {
    /// Constructs an empty metric for `mode` (every optional slot = None).
    #[must_use]
    pub const fn empty(mode: KvQuant) -> Self {
        Self {
            mode,
            mel_loss_rel_delta: None,
            wer_rel_delta: None,
            utmos_rel_delta: None,
            dnsmos_rel_delta: None,
        }
    }

    /// Records the MEL loss relative delta.
    #[must_use]
    pub const fn with_mel_loss(mut self, rel_delta: f32) -> Self {
        self.mel_loss_rel_delta = Some(rel_delta);
        self
    }

    /// Records the WER relative delta.
    #[must_use]
    pub const fn with_wer(mut self, rel_delta: f32) -> Self {
        self.wer_rel_delta = Some(rel_delta);
        self
    }

    /// Records the UTMOS relative decrease.
    #[must_use]
    pub const fn with_utmos(mut self, rel_delta: f32) -> Self {
        self.utmos_rel_delta = Some(rel_delta);
        self
    }

    /// Records the DNSMOS relative decrease.
    #[must_use]
    pub const fn with_dnsmos(mut self, rel_delta: f32) -> Self {
        self.dnsmos_rel_delta = Some(rel_delta);
        self
    }

    /// The worst (max absolute) recorded delta across every measured metric.
    /// `None` if nothing was measured.
    #[must_use]
    pub fn worst_delta(&self) -> Option<f32> {
        [
            self.mel_loss_rel_delta,
            self.wer_rel_delta,
            self.utmos_rel_delta,
            self.dnsmos_rel_delta,
        ]
        .iter()
        .filter_map(|x| *x)
        .fold(None::<f32>, |acc, d| Some(acc.map_or(d, |a| a.max(d))))
    }

    /// True iff every measured metric is within [`DEGRADATION_THRESHOLD`].
    /// A metric with `None` value is treated as *not measured*, not as pass.
    #[must_use]
    pub fn passes_gate(&self) -> bool {
        self.worst_delta()
            .is_none_or(|d| d <= DEGRADATION_THRESHOLD)
    }
}

/// FR-QT-04 verification report bundling every KV quantization mode plus
/// the surface that the FR-QT-04 CI job renders.
#[derive(Debug, Clone)]
pub struct KvQuantVerifyReport {
    /// Model / test identifier for the CI summary.
    pub label: String,
    /// One [`KvQuantMetric`] per non-FP32 mode present in the evaluation.
    pub per_mode: Vec<KvQuantMetric>,
    /// `true` when the caller could not obtain UTMOS weights (T13 M4
    /// push-out); the CI summary should print "UTMOS gate: M4" rather than
    /// pass.
    pub utmos_unavailable: bool,
    /// `true` when DNSMOS was similarly unavailable.
    pub dnsmos_unavailable: bool,
}

impl KvQuantVerifyReport {
    /// Constructs an empty report for `label`.
    #[must_use]
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            per_mode: Vec::new(),
            utmos_unavailable: false,
            dnsmos_unavailable: false,
        }
    }

    /// Attaches a per-mode metric.
    #[must_use]
    pub fn with_metric(mut self, metric: KvQuantMetric) -> Self {
        self.per_mode.push(metric);
        self
    }

    /// Marks UTMOS as unavailable in this environment (T13).
    #[must_use]
    pub const fn utmos_unavailable(mut self) -> Self {
        self.utmos_unavailable = true;
        self
    }

    /// Marks UTMOS as available again — the M4-18 T08 flip: a pipeline that
    /// defaults to the honest "unavailable" posture (the UTMOS weights are
    /// owner-deferred) flips this once a real scorer instance produced the
    /// deltas fed via [`KvQuantMetric::with_utmos`]. Never call this without
    /// such a scorer: an unavailable-but-claimed-available report is exactly
    /// the fabricated pass NFR-QL-04 bans.
    #[must_use]
    pub const fn utmos_available(mut self) -> Self {
        self.utmos_unavailable = false;
        self
    }

    /// Marks DNSMOS as unavailable in this environment (T13).
    #[must_use]
    pub const fn dnsmos_unavailable(mut self) -> Self {
        self.dnsmos_unavailable = true;
        self
    }

    /// The set of quantization modes that failed the [`DEGRADATION_THRESHOLD`]
    /// gate.
    ///
    /// Empty vec = all measured modes passed. UTMOS / DNSMOS unavailability
    /// (T13 push-out) does not itself constitute a failure — the CI reports
    /// them separately.
    #[must_use]
    pub fn failing_modes(&self) -> Vec<KvQuant> {
        self.per_mode
            .iter()
            .filter(|m| !m.passes_gate())
            .map(|m| m.mode)
            .collect()
    }

    /// True iff every measured mode passes the gate. Note: this returns `true`
    /// when nothing was measured (the caller has to check `per_mode.is_empty()`
    /// separately if that matters).
    #[must_use]
    pub fn all_pass_gate(&self) -> bool {
        self.failing_modes().is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_metric_passes_by_default() {
        let m = KvQuantMetric::empty(KvQuant::Q8_0);
        assert!(m.passes_gate());
        assert!(m.worst_delta().is_none());
    }

    #[test]
    fn mel_loss_within_gate_passes() {
        let m = KvQuantMetric::empty(KvQuant::Q8_0).with_mel_loss(0.02);
        assert!(m.passes_gate());
        assert_eq!(m.worst_delta(), Some(0.02));
    }

    #[test]
    fn mel_loss_over_gate_fails() {
        let m = KvQuantMetric::empty(KvQuant::Q4_0).with_mel_loss(0.08);
        assert!(!m.passes_gate());
    }

    #[test]
    fn worst_delta_picks_max() {
        let m = KvQuantMetric::empty(KvQuant::Q5_0)
            .with_mel_loss(0.02)
            .with_wer(0.04)
            .with_utmos(0.06);
        assert_eq!(m.worst_delta(), Some(0.06));
        assert!(!m.passes_gate());
    }

    #[test]
    fn report_aggregates_and_reports_failures() {
        let r = KvQuantVerifyReport::new("whisper-base")
            .with_metric(KvQuantMetric::empty(KvQuant::Q8_0).with_wer(0.01))
            .with_metric(KvQuantMetric::empty(KvQuant::Q5_0).with_wer(0.03))
            .with_metric(KvQuantMetric::empty(KvQuant::Q4_0).with_wer(0.10));
        let failing = r.failing_modes();
        assert_eq!(failing, vec![KvQuant::Q4_0]);
        assert!(!r.all_pass_gate());
    }

    #[test]
    fn report_all_pass_when_no_metric_over_gate() {
        let r = KvQuantVerifyReport::new("piper-plus")
            .with_metric(KvQuantMetric::empty(KvQuant::Q8_0).with_mel_loss(0.01))
            .with_metric(KvQuantMetric::empty(KvQuant::Q5_0).with_mel_loss(0.03))
            .with_metric(KvQuantMetric::empty(KvQuant::Q4_0).with_mel_loss(0.049));
        assert!(r.all_pass_gate());
    }

    #[test]
    fn utmos_and_dnsmos_unavailable_flags() {
        let r = KvQuantVerifyReport::new("kokoro")
            .utmos_unavailable()
            .dnsmos_unavailable();
        assert!(r.utmos_unavailable);
        assert!(r.dnsmos_unavailable);
        // Unavailability does not by itself flip the gate — the CI summary
        // renders it as an M4 push-out annotation.
        assert!(r.all_pass_gate());
    }

    /// The M4-18 T08 flip: a pipeline that defaults to the honest
    /// "unavailable" posture flips UTMOS back once a real scorer exists.
    /// DNSMOS stays untouched (its availability is a separate, owner-gated
    /// decision — license fail-closed).
    #[test]
    fn utmos_available_flips_the_flag_back() {
        let r = KvQuantVerifyReport::new("m4-18")
            .utmos_unavailable()
            .dnsmos_unavailable()
            .utmos_available();
        assert!(!r.utmos_unavailable);
        assert!(r.dnsmos_unavailable, "the UTMOS flip must not touch DNSMOS");
    }

    /// The NFR-QL-02 5% threshold is not something a caller can talk itself
    /// into loosening; the constant is on the wire.
    #[test]
    fn degradation_threshold_pin() {
        assert_eq!(DEGRADATION_THRESHOLD, 0.05);
    }

    /// Boundary case: exactly at the threshold is a pass (not a fail).
    /// Documented invariant of the gate.
    #[test]
    fn exact_threshold_is_pass() {
        let m = KvQuantMetric::empty(KvQuant::Q4_0).with_mel_loss(DEGRADATION_THRESHOLD);
        assert!(m.passes_gate());
    }
}
