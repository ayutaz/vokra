//! The NFR-QL-02 **5 % degradation gate runner** (M5-15 T21).
//!
//! [`crate::degradation`] answers "did *this pair of waveforms* degrade?".
//! This module answers the question a release actually asks: "**did this
//! artifact regress**, and *by which measure*?" — which depends on what the
//! artifact emits.
//!
//! # Two axes, because one metric cannot cover both artifact classes
//!
//! NFR-QL-02 is worded for generative audio ("MEL loss / UTMOS 劣化 5 % 未満").
//! That axis is uncomputable for an ASR model: Whisper and Voxtral emit
//! *text*. [`MelLoss`](crate::MelLoss) needs two waveforms and
//! [`Utmos`](crate::metrics::Utmos) needs one — neither has a waveform to
//! score. So the runner carries two axes and picks by [`ArtifactClass`]:
//!
//! | class                          | axis                    |
//! |--------------------------------|-------------------------|
//! | [`ArtifactClass::GenerativeAudio`] | UTMOS + mel-loss    |
//! | [`ArtifactClass::AsrText`]         | WER + CER           |
//!
//! # The axis that did not run is reported as *not run*, never as "passed"
//!
//! The single most important property here (NFR-QL-04): a
//! [`QualityGateReport`] never lets an unrun axis read as a pass.
//! [`AxisOutcome::NotRun`] carries the reason, [`QualityGateReport::summary`]
//! prints it, and [`QualityGateReport::passed`] is only `true` when the axis
//! that *did* run stayed within the threshold. A caller that wants "everything
//! measurable was measured" asks [`QualityGateReport::fully_covered`].
//!
//! This is the same posture as
//! [`DegradationReport::mel_loss_only`](crate::degradation::DegradationReport)
//! and the Kokoro `PROSODY_F0_ATOL` precedent: a number that cannot honestly
//! gate is surfaced, never laundered into a verdict.

use crate::degradation::{
    DegradationReport, MosDomain, check_degradation, check_degradation_with_utmos,
};
use crate::metrics::{AudioMosMetric, Cer, TextMetric, Wer};
use vokra_core::{Result, VokraError};

/// What the artifact under test emits — which decides the gating axis.
///
/// There is deliberately **no default**: guessing would let an ASR model be
/// "gated" on an axis that silently never ran.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArtifactClass {
    /// TTS / vocoder / codec output — a waveform. Gated on UTMOS + mel-loss
    /// (the NFR-QL-02 axis as written; GA DoD item 2 and the M4-05 / M4-06
    /// completion conditions depend on this one).
    GenerativeAudio,
    /// ASR output — text. Gated on WER + CER, because the audio-domain
    /// metrics are not computable for it (see the module docs).
    AsrText,
}

impl ArtifactClass {
    /// Human-readable axis name, for reports.
    #[must_use]
    pub const fn axis_name(self) -> &'static str {
        match self {
            Self::GenerativeAudio => "utmos+mel_loss",
            Self::AsrText => "wer+cer",
        }
    }
}

/// One axis's result.
#[derive(Debug, Clone, PartialEq)]
pub enum AxisOutcome {
    /// The axis ran on generative-audio artifacts.
    Audio(Box<DegradationReport>),
    /// The axis ran on ASR text: `(wer_delta, cer_delta)` are the *relative*
    /// degradations of the quantized transcript against the reference
    /// transcript, both measured against the ground truth.
    Text {
        /// WER of the reference (f32/f16) transcript vs ground truth.
        wer_ref: f64,
        /// WER of the quantized transcript vs ground truth.
        wer_quant: f64,
        /// CER of the reference transcript vs ground truth.
        cer_ref: f64,
        /// CER of the quantized transcript vs ground truth.
        cer_quant: f64,
        /// `wer_quant - wer_ref` — an **absolute** error-rate increase.
        /// Absolute, not relative, because a reference WER of `0.0` (which
        /// the campaign's Whisper legs actually hit) makes a ratio undefined,
        /// and clamping it would invent a verdict.
        wer_increase: f64,
        /// `cer_quant - cer_ref`.
        cer_increase: f64,
        /// `max(wer_increase, cer_increase) <= threshold`.
        within_threshold: bool,
    },
    /// The axis did **not** run, with the reason. Never a pass.
    NotRun {
        /// Why — surfaced verbatim in [`QualityGateReport::summary`].
        reason: String,
    },
}

impl AxisOutcome {
    /// `true` only when the axis ran **and** stayed within the threshold.
    /// An unrun axis is never a pass.
    #[must_use]
    pub fn passed(&self) -> bool {
        match self {
            Self::Audio(r) => r.passes_5pct_gate,
            Self::Text {
                within_threshold, ..
            } => *within_threshold,
            Self::NotRun { .. } => false,
        }
    }

    /// `true` when the axis produced a measurement at all.
    #[must_use]
    pub fn ran(&self) -> bool {
        !matches!(self, Self::NotRun { .. })
    }
}

/// The gate verdict for one artifact.
#[derive(Debug, Clone, PartialEq)]
pub struct QualityGateReport {
    /// The artifact class the caller declared.
    pub class: ArtifactClass,
    /// A label for the artifact (model + quantization, e.g.
    /// `"whisper-base Q4_K"`), echoed in [`Self::summary`].
    pub label: String,
    /// The relative/absolute degradation bound (`0.05` for the 5 % gate).
    pub threshold: f64,
    /// The audio axis (UTMOS + mel-loss).
    pub audio: AxisOutcome,
    /// The text axis (WER + CER).
    pub text: AxisOutcome,
}

impl QualityGateReport {
    /// The overall verdict: the axis matching [`Self::class`] must have run
    /// **and** passed.
    ///
    /// The non-matching axis is irrelevant to the verdict but is still
    /// reported (as [`AxisOutcome::NotRun`] with a reason) so a reader can
    /// see it was not silently skipped.
    #[must_use]
    pub fn passed(&self) -> bool {
        match self.class {
            ArtifactClass::GenerativeAudio => self.audio.passed(),
            ArtifactClass::AsrText => self.text.passed(),
        }
    }

    /// `true` when **both** axes produced a measurement. Almost never true by
    /// construction — it exists so a future artifact that really does emit
    /// both (an S2S model with a transcript side-channel) can assert it.
    #[must_use]
    pub fn fully_covered(&self) -> bool {
        self.audio.ran() && self.text.ran()
    }

    /// A one-artifact report block naming the axis that gated, the numbers,
    /// and — explicitly — the axis that did not run and why.
    #[must_use]
    pub fn summary(&self) -> String {
        let mut s = format!(
            "{}: class={:?} gating-axis={} threshold={:.4} verdict={}\n",
            self.label,
            self.class,
            self.class.axis_name(),
            self.threshold,
            if self.passed() { "PASS" } else { "FAIL" }
        );
        match &self.audio {
            AxisOutcome::Audio(r) => {
                s.push_str(&format!(
                    "  utmos+mel_loss: mel_loss_quant={:.6e} rel_delta={:.6e}",
                    r.mel_loss_quant, r.relative_delta
                ));
                match &r.utmos {
                    Some(a) => s.push_str(&format!(
                        " utmos_ref={:.4} utmos_quant={:.4} rel_decrease={:.4}{}\n",
                        a.score_ref,
                        a.score_quant,
                        a.rel_decrease,
                        if a.advisory_only {
                            " (ADVISORY — out-of-distribution domain, does not gate)"
                        } else {
                            ""
                        }
                    )),
                    None => s.push_str(" utmos=NOT RUN (no scorer injected)\n"),
                }
            }
            AxisOutcome::NotRun { reason } => {
                s.push_str(&format!("  utmos+mel_loss: NOT RUN — {reason}\n"));
            }
            AxisOutcome::Text { .. } => unreachable!("audio slot holds an audio outcome"),
        }
        match &self.text {
            AxisOutcome::Text {
                wer_ref,
                wer_quant,
                cer_ref,
                cer_quant,
                wer_increase,
                cer_increase,
                ..
            } => s.push_str(&format!(
                "  wer+cer: wer {wer_ref:.6}→{wer_quant:.6} (+{wer_increase:.6}) cer \
                 {cer_ref:.6}→{cer_quant:.6} (+{cer_increase:.6})\n"
            )),
            AxisOutcome::NotRun { reason } => {
                s.push_str(&format!("  wer+cer: NOT RUN — {reason}\n"));
            }
            AxisOutcome::Audio(_) => unreachable!("text slot holds a text outcome"),
        }
        s
    }
}

/// Runs the gate over a **generative-audio** artifact.
///
/// `mos` is optional: without a scorer the audio axis still runs on mel-loss
/// alone and the report carries
/// [`DegradationReport::mel_loss_only`](crate::degradation::DegradationReport)
/// — the honest partial-gate marker, not a pass.
///
/// # Errors
///
/// Propagates every [`check_degradation`] / scorer error verbatim
/// (FR-EX-08 — a sample-rate mismatch is never silently resampled).
pub fn gate_generative_audio(
    label: impl Into<String>,
    reference: &[f32],
    quantized: &[f32],
    sample_rate: u32,
    threshold: f64,
    mos: Option<(&dyn AudioMosMetric, MosDomain)>,
    ground_truth_available: bool,
) -> Result<QualityGateReport> {
    let report = match mos {
        Some((scorer, domain)) => check_degradation_with_utmos(
            reference,
            quantized,
            sample_rate,
            threshold,
            scorer,
            domain,
        )?,
        None => check_degradation(reference, quantized, sample_rate, threshold)?,
    };
    Ok(QualityGateReport {
        class: ArtifactClass::GenerativeAudio,
        label: label.into(),
        threshold,
        audio: AxisOutcome::Audio(Box::new(report)),
        text: AxisOutcome::NotRun {
            reason: if ground_truth_available {
                "artifact emits audio, not text".to_owned()
            } else {
                "artifact emits audio, not text (and no reference transcript was supplied)"
                    .to_owned()
            },
        },
    })
}

/// Runs the gate over an **ASR text** artifact.
///
/// Both transcripts are scored against the same `ground_truth`; the gate is on
/// the *increase* in error rate that quantization caused.
///
/// # Errors
///
/// [`VokraError::InvalidArgument`] when `threshold` is not finite and
/// positive, or when `ground_truth` is empty (every rate would be degenerate).
pub fn gate_asr_text(
    label: impl Into<String>,
    ground_truth: &str,
    reference_hyp: &str,
    quantized_hyp: &str,
    threshold: f64,
) -> Result<QualityGateReport> {
    if !threshold.is_finite() || threshold <= 0.0 {
        return Err(VokraError::InvalidArgument(format!(
            "gate_asr_text: threshold must be a finite positive number, got {threshold}"
        )));
    }
    if ground_truth.trim().is_empty() {
        return Err(VokraError::InvalidArgument(
            "gate_asr_text: empty ground truth — every error rate would be degenerate, and a \
             degenerate rate must not be turned into a verdict (FR-EX-08)"
                .to_owned(),
        ));
    }
    let (wer, cer) = (Wer, Cer);
    let wer_ref = wer.eval_text(reference_hyp, ground_truth);
    let wer_quant = wer.eval_text(quantized_hyp, ground_truth);
    let cer_ref = cer.eval_text(reference_hyp, ground_truth);
    let cer_quant = cer.eval_text(quantized_hyp, ground_truth);
    let wer_increase = wer_quant - wer_ref;
    let cer_increase = cer_quant - cer_ref;
    let within_threshold = wer_increase.max(cer_increase) <= threshold;
    Ok(QualityGateReport {
        class: ArtifactClass::AsrText,
        label: label.into(),
        threshold,
        audio: AxisOutcome::NotRun {
            reason: "artifact emits text, not audio — UTMOS scores one waveform and mel-loss \
                     compares two, so neither is computable here"
                .to_owned(),
        },
        text: AxisOutcome::Text {
            wer_ref,
            wer_quant,
            cer_ref,
            cer_quant,
            wer_increase,
            cer_increase,
            within_threshold,
        },
    })
}

/// Renders a multi-artifact gate run, with an explicit coverage line.
///
/// The trailing line states how many artifacts were gated on each axis and how
/// many failed — so a run in which the audio axis never fired (because every
/// artifact was ASR) reads as exactly that, not as "UTMOS passed".
#[must_use]
pub fn render_run(reports: &[QualityGateReport]) -> String {
    let mut s = String::from("NFR-QL-02 quality gate run\n");
    for r in reports {
        s.push_str(&r.summary());
    }
    let audio_gated = reports
        .iter()
        .filter(|r| r.class == ArtifactClass::GenerativeAudio)
        .count();
    let text_gated = reports.len() - audio_gated;
    let failed = reports.iter().filter(|r| !r.passed()).count();
    s.push_str(&format!(
        "coverage: {audio_gated} artifact(s) gated on utmos+mel_loss, {text_gated} on wer+cer; \
         {failed} failed\n"
    ));
    if audio_gated == 0 {
        s.push_str(
            "note: the UTMOS/mel-loss axis did not fire in this run (no generative-audio \
             artifact). NFR-QL-02's audio axis is therefore UNMEASURED here — it is not \
             satisfied by the text axis passing.\n",
        );
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::{Direction, Metric};
    use std::cell::RefCell;

    const SR: u32 = 16_000;
    const T: f64 = 0.05;

    fn tone(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / SR as f32).sin())
            .collect()
    }

    struct ScriptedMos(RefCell<Vec<f64>>);
    impl Metric for ScriptedMos {
        fn name(&self) -> &str {
            "scripted"
        }
        fn direction(&self) -> Direction {
            Direction::HigherIsBetter
        }
    }
    impl AudioMosMetric for ScriptedMos {
        fn eval_mos(&self, _a: &[f32], _sr: u32) -> Result<f64> {
            self.0
                .borrow_mut()
                .pop()
                .ok_or_else(|| VokraError::InvalidArgument("exhausted".into()))
        }
    }

    #[test]
    fn asr_artifact_reports_the_audio_axis_as_not_run_never_as_passed() {
        let r = gate_asr_text(
            "whisper-base Q4_K",
            "the quick brown fox",
            "the quick brown fox",
            "the quick brown fox",
            T,
        )
        .unwrap();
        assert!(r.passed(), "identical transcripts must pass");
        // The critical property: the axis that could not run must not read
        // as a pass anywhere.
        assert!(!r.audio.ran());
        assert!(!r.audio.passed(), "an unrun axis is never a pass");
        assert!(!r.fully_covered());
        let s = r.summary();
        assert!(s.contains("NOT RUN"), "summary must say so: {s}");
        assert!(s.contains("gating-axis=wer+cer"), "{s}");
    }

    #[test]
    fn asr_gate_fails_on_a_real_transcript_regression() {
        // Ground truth 4 words; the quantized transcript gets one wrong →
        // WER increase 0.25, over the 5 % gate.
        let r = gate_asr_text(
            "whisper-base Q4_K",
            "the quick brown fox",
            "the quick brown fox",
            "the quick brown box",
            T,
        )
        .unwrap();
        match &r.text {
            AxisOutcome::Text {
                wer_ref,
                wer_increase,
                ..
            } => {
                assert_eq!(*wer_ref, 0.0, "reference transcript is exact");
                assert!(
                    (*wer_increase - 0.25).abs() < 1e-12,
                    "one of four words wrong: {wer_increase}"
                );
            }
            other => panic!("expected a text outcome, got {other:?}"),
        }
        assert!(!r.passed());
    }

    #[test]
    fn asr_gate_uses_absolute_increase_so_a_zero_reference_wer_stays_defined() {
        // The campaign's Whisper legs really do hit WER 0.0; a *relative*
        // delta would be 0/0. An absolute increase stays defined, and a tiny
        // increase still passes.
        let r = gate_asr_text(
            "x",
            "a b c d e f g h i j",
            "a b c d e f g h i j",
            "a b c d e f g h i j",
            T,
        )
        .unwrap();
        assert!(r.passed());
        match r.text {
            AxisOutcome::Text { wer_increase, .. } => assert_eq!(wer_increase, 0.0),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn generative_audio_artifact_reports_the_text_axis_as_not_run() {
        let x = tone(16_000);
        let mos = ScriptedMos(RefCell::new(vec![3.92, 4.0])); // ref 4.0, quant 3.92
        let r = gate_generative_audio(
            "kokoro Q8_0",
            &x,
            &x,
            SR,
            T,
            Some((&mos, MosDomain::TtsSynthesis)),
            false,
        )
        .unwrap();
        assert!(r.passed());
        assert!(r.audio.ran());
        assert!(!r.text.ran());
        assert!(!r.text.passed());
        let s = r.summary();
        assert!(s.contains("gating-axis=utmos+mel_loss"), "{s}");
        assert!(s.contains("wer+cer: NOT RUN"), "{s}");
    }

    #[test]
    fn advisory_only_domain_is_labelled_in_the_summary() {
        // An out-of-distribution MOS must be visible as advisory, so nobody
        // reads the PASS as "UTMOS passed" (NFR-QL-04).
        let x = tone(16_000);
        let mos = ScriptedMos(RefCell::new(vec![3.0, 4.0])); // 25 % drop
        let r = gate_generative_audio(
            "moshi Q8_0",
            &x,
            &x,
            SR,
            T,
            Some((&mos, MosDomain::CodecStreaming)),
            false,
        )
        .unwrap();
        assert!(r.passed(), "advisory MOS must not flip the verdict");
        let s = r.summary();
        assert!(s.contains("ADVISORY"), "must be labelled advisory: {s}");
    }

    #[test]
    fn mel_only_audio_gate_surfaces_the_missing_scorer() {
        let x = tone(16_000);
        let r = gate_generative_audio("kokoro Q8_0", &x, &x, SR, T, None, false).unwrap();
        let s = r.summary();
        assert!(s.contains("utmos=NOT RUN"), "{s}");
    }

    #[test]
    fn a_run_with_no_audio_artifact_says_the_audio_axis_is_unmeasured() {
        // This is the M5-15 situation exactly: every quantized artifact in
        // the WP is ASR, so NFR-QL-02's own axis never fires. The run report
        // must say so rather than implying the requirement was met.
        let reports = vec![
            gate_asr_text("whisper-base Q4_K", "a b c", "a b c", "a b c", T).unwrap(),
            gate_asr_text("whisper-small Q6_K", "a b c", "a b c", "a b c", T).unwrap(),
        ];
        let out = render_run(&reports);
        assert!(out.contains("2 on wer+cer"), "{out}");
        assert!(out.contains("UNMEASURED"), "must not imply coverage: {out}");
    }

    #[test]
    fn rejects_degenerate_inputs() {
        assert!(
            gate_asr_text("x", "", "a", "a", T).is_err(),
            "empty ground truth"
        );
        assert!(
            gate_asr_text("x", "a", "a", "a", 0.0).is_err(),
            "zero threshold"
        );
        assert!(
            gate_asr_text("x", "a", "a", "a", f64::NAN).is_err(),
            "NaN threshold"
        );
    }
}
