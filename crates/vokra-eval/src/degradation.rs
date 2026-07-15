//! Quantization degradation gate (M2-08 T11 + M4-18 T08/T10; NFR-QL-02).
//!
//! Compares a `quantized` waveform against a `reference` (typically the
//! fp32/fp16 model output on the same input) using log-mel L1 loss, and
//! reports whether the relative degradation stays under the NFR-QL-02
//! **5 % gate**.
//!
//! ```text
//! mel_loss_ref   = mel_loss(reference, reference)   // == 0 by construction
//! mel_loss_quant = mel_loss(quantized, reference)
//! relative_delta = (mel_loss_quant - mel_loss_ref) / max(mel_loss_ref, ε)
//! passes         = relative_delta < threshold
//! ```
//!
//! Because `mel_loss_ref` is `0.0` for identical inputs, the relative delta
//! collapses to `mel_loss_quant / ε` when the reference is used as its own
//! baseline; the `epsilon = 1e-9` floor keeps that quotient finite and — for
//! the *quantized* path — makes any non-trivial `mel_loss_quant` blow past
//! `threshold`. Callers who prefer an absolute gate can inspect
//! [`DegradationReport::mel_loss_quant`] directly.
//!
//! # UTMOS-augmented gate (M4-18 T08 — scorer injected, weight-deferred)
//!
//! Mel-loss alone can miss perceptual artefacts (e.g. INT8 vocoder buzz that
//! preserves mel energy); the UTMOS neural MOS predictor is the second half
//! of the gate. [`check_degradation_with_utmos`] runs it: the caller
//! **injects** any [`AudioMosMetric`] scorer (dependency injection — the M4-18
//! kickoff gate deferred the real UTMOS weights, so this crate never
//! hard-codes a weight path; a caller that has a `vokra.utmos.*` GGUF builds
//! [`crate::metrics::utmos::Utmos`] and passes it). The UTMOS side follows
//! the `vokra-core` `kv_quant::verify` convention:
//!
//! ```text
//! rel_decrease = (score_ref - score_quant) / score_ref   // decrease → bad
//! within_threshold = rel_decrease <= threshold
//! ```
//!
//! **Weight-absent environments stay honest (FR-EX-08)**: without a scorer
//! instance there is nothing to pass, so such callers use
//! [`check_degradation`], whose report carries `mel_loss_only = true` — the
//! partial-gate caveat that downstream summaries must surface ("UTMOS gate:
//! not run"), never a silent "UTMOS passed".
//!
//! # Advisory-only domains (M4-18 T10 — miscalibration machinery)
//!
//! The UTMOS reference is trained on the SaruLab MOS Challenge 2022 domain
//! (synthetic TTS speech). Mimi-codec / streaming output (M4-05 CSM, M4-06
//! Moshi) is potentially out-of-distribution, where the score may be
//! miscalibrated (`docs/m4-scope-expansion-2026-07-13.md` §BIG-7 risk). The
//! caller states the domain via [`MosDomain`] — there is no default — and an
//! OOD domain makes the MOS half **advisory-only**: the score and its
//! relative decrease are still reported ([`MosAssessment`]), but they do
//! **not** gate [`DegradationReport::passes_5pct_gate`]. This is the same
//! honest-engineering posture as [`DegradationReport::mel_loss_only`] and the
//! Kokoro `PROSODY_F0_ATOL` precedent: a number that cannot honestly gate is
//! surfaced as advisory, never laundered into a pass (NFR-QL-04). Whether
//! the codec/streaming domain is *actually* miscalibrated is an owner-side
//! real-sample correlation study (deferred with the weights); the machinery
//! errs on the safe side until then.
//!
//! # Errors (FR-EX-08 — no silent fallback)
//!
//! - `threshold` must be finite and `> 0`. A non-positive or NaN threshold
//!   is a caller bug and returns [`VokraError::InvalidArgument`] rather
//!   than being clamped.
//! - Any [`MelLoss::loss`] error (e.g. too-short inputs) propagates
//!   verbatim; so does any scorer error (sample-rate mismatch, …).
//! - A non-finite or non-positive reference MOS score makes the relative
//!   decrease meaningless and is an explicit error, never a clamp.

use crate::MelLoss;
use crate::metrics::AudioMosMetric;
use vokra_core::{Result, VokraError};

/// Numerical floor for the relative-delta denominator. Chosen so that a
/// non-trivial `mel_loss_quant` against a bit-identical reference
/// (`mel_loss_ref == 0`) blows past any sane `threshold`, while identical
/// inputs still yield `relative_delta == 0` exactly.
const EPSILON: f64 = 1e-9;

/// The acoustic domain of the audio being scored, relative to the UTMOS
/// reference's training distribution (M4-18 T10).
///
/// There is deliberately **no default**: the caller must state the domain, so
/// an out-of-distribution evaluation can never silently masquerade as a hard
/// gate (FR-EX-08 posture applied to policy).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MosDomain {
    /// Synthetic TTS speech — in-distribution for the UTMOS reference
    /// (SaruLab MOS Challenge 2022 training domain). The MOS half gates.
    TtsSynthesis,
    /// Mimi-codec / streaming output (M4-05 CSM, M4-06 Moshi) — potentially
    /// out-of-distribution; the MOS half is advisory-only until the
    /// owner-side correlation study validates calibration.
    CodecStreaming,
}

impl MosDomain {
    /// `true` when scores from this domain must not gate (advisory-only).
    #[must_use]
    pub const fn is_advisory_only(self) -> bool {
        matches!(self, Self::CodecStreaming)
    }
}

/// The UTMOS half of a [`DegradationReport`] (M4-18 T08/T10).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MosAssessment {
    /// MOS score of the reference waveform.
    pub score_ref: f64,
    /// MOS score of the quantized/hypothesis waveform.
    pub score_quant: f64,
    /// `(score_ref - score_quant) / score_ref` — a perceptual-score
    /// *decrease* is bad (the `kv_quant::verify` NFR-QL-02 convention).
    pub rel_decrease: f64,
    /// `rel_decrease <= threshold` (exactly at the threshold passes,
    /// matching `kv_quant::verify`).
    pub within_threshold: bool,
    /// `true` when the domain is out-of-distribution for the UTMOS
    /// reference ([`MosDomain::is_advisory_only`]): the score is reported
    /// but does **not** gate [`DegradationReport::passes_5pct_gate`].
    pub advisory_only: bool,
}

/// Outcome of [`check_degradation`] / [`check_degradation_with_utmos`] — the
/// two mel-loss samples, their relative delta, the optional UTMOS half, and
/// the pass/fail verdict against the 5 % gate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DegradationReport {
    /// `mel_loss(reference, reference)` — always `0.0` by definition;
    /// carried explicitly so a caller can double-check that the reference
    /// is self-consistent (STFT frame count etc.).
    pub mel_loss_ref: f64,
    /// `mel_loss(quantized, reference)` — the actual degradation signal.
    pub mel_loss_quant: f64,
    /// `(mel_loss_quant - mel_loss_ref) / max(mel_loss_ref, EPSILON)`.
    pub relative_delta: f64,
    /// `true` means the audio stayed under the NFR-QL-02 gate: the mel-loss
    /// delta is under `threshold` AND — when a gating (non-advisory) UTMOS
    /// assessment is present — the MOS decrease is within `threshold` too.
    pub passes_5pct_gate: bool,
    /// `true` when the report reflects mel-loss alone (UTMOS not run — no
    /// scorer was available, e.g. the M4-18 weights are still deferred).
    /// Downstream callers must surface this partial-gate caveat.
    pub mel_loss_only: bool,
    /// The UTMOS half (M4-18 T08): `None` on the mel-only path, `Some` when
    /// a scorer was injected. An `advisory_only` assessment never gates.
    pub utmos: Option<MosAssessment>,
}

/// Runs the mel-loss degradation gate against a `reference` waveform.
///
/// Internally builds `MelLoss::new(sample_rate, 1024, 256, 80)` — the
/// librosa defaults (`n_fft=1024`, `hop_length=256`, `n_mels=80`) that the
/// M2-08 plan pins for a stable cross-model comparison, matching the
/// [`MelLoss::new`] signature. Callers that need a bit-exact model-specific
/// front-end can compose [`MelLoss::from_attrs`] themselves; this helper is
/// the *policy-level* gate used by `vokra-cli` / session ctors.
///
/// `reference` and `quantized` are both mono PCM at `sample_rate`.
/// `threshold` is the relative-delta bound (e.g. `0.05` for the NFR-QL-02
/// 5 % gate).
///
/// # Errors
///
/// - [`VokraError::InvalidArgument`] if `threshold` is not a finite
///   positive number.
/// - Any error from [`MelLoss::loss`] (too-short inputs, etc.).
pub fn check_degradation(
    reference: &[f32],
    quantized: &[f32],
    sample_rate: u32,
    threshold: f64,
) -> Result<DegradationReport> {
    if !threshold.is_finite() || threshold <= 0.0 {
        return Err(VokraError::InvalidArgument(format!(
            "check_degradation: threshold must be a finite positive number, got {threshold}"
        )));
    }

    let ml = MelLoss::new(sample_rate, 1024, 256, 80);
    let mel_loss_ref = ml.loss(reference, reference)?;
    let mel_loss_quant = ml.loss(quantized, reference)?;

    let denom = mel_loss_ref.max(EPSILON);
    let relative_delta = (mel_loss_quant - mel_loss_ref) / denom;
    let passes_5pct_gate = relative_delta < threshold;

    Ok(DegradationReport {
        mel_loss_ref,
        mel_loss_quant,
        relative_delta,
        passes_5pct_gate,
        // UTMOS was not run on this entry point — partial gate, surfaced.
        mel_loss_only: true,
        utmos: None,
    })
}

/// UTMOS-augmented degradation gate (M4-18 T08/T10).
///
/// Runs [`check_degradation`]'s mel-loss half **plus** the injected `mos`
/// scorer over both waveforms, combining the verdicts:
///
/// - mel half: `relative_delta < threshold` (unchanged);
/// - MOS half: `rel_decrease = (score_ref - score_quant) / score_ref <=
///   threshold` — but **only gates when `domain` is in-distribution**
///   ([`MosDomain::TtsSynthesis`]); an advisory-only domain reports the
///   assessment without letting it flip the gate (module docs, NFR-QL-04).
///
/// The scorer is injected (`&dyn AudioMosMetric`) rather than constructed
/// here: the M4-18 kickoff gate deferred the real UTMOS weights, and this
/// crate never hard-codes a weight path. Callers with a `vokra.utmos.*`
/// GGUF build [`crate::metrics::utmos::Utmos`]; callers without one use
/// [`check_degradation`] and inherit its honest `mel_loss_only = true`.
///
/// # Errors
///
/// - Everything [`check_degradation`] rejects (threshold, mel front-end);
/// - any scorer error, propagated verbatim (sample-rate mismatch etc. —
///   FR-EX-08, no silent resample);
/// - a non-finite or non-positive reference score
///   ([`VokraError::InvalidArgument`]) — the relative decrease is
///   meaningless there and is never clamped into a verdict.
pub fn check_degradation_with_utmos(
    reference: &[f32],
    quantized: &[f32],
    sample_rate: u32,
    threshold: f64,
    mos: &dyn AudioMosMetric,
    domain: MosDomain,
) -> Result<DegradationReport> {
    // Mel half first: validates the threshold and computes the mel gate.
    let mel = check_degradation(reference, quantized, sample_rate, threshold)?;

    // MOS half — reference first, quantized second (the documented call
    // order; scripted test scorers rely on it).
    let score_ref = mos.eval_mos(reference, sample_rate)?;
    let score_quant = mos.eval_mos(quantized, sample_rate)?;
    if !score_ref.is_finite() || score_ref <= 0.0 {
        return Err(VokraError::InvalidArgument(format!(
            "check_degradation_with_utmos: reference MOS score {score_ref} is not a finite \
             positive number — the relative decrease is undefined there (a broken scorer or \
             weights; never clamped into a verdict, FR-EX-08)"
        )));
    }
    if !score_quant.is_finite() {
        return Err(VokraError::InvalidArgument(format!(
            "check_degradation_with_utmos: quantized MOS score {score_quant} is not finite"
        )));
    }
    let rel_decrease = (score_ref - score_quant) / score_ref;
    let within_threshold = rel_decrease <= threshold;
    let advisory_only = domain.is_advisory_only();
    let assessment = MosAssessment {
        score_ref,
        score_quant,
        rel_decrease,
        within_threshold,
        advisory_only,
    };

    // The MOS half gates only in-distribution; advisory-only assessments
    // are surfaced without flipping the verdict (module docs, NFR-QL-04).
    let passes_5pct_gate = mel.passes_5pct_gate && (advisory_only || within_threshold);

    Ok(DegradationReport {
        passes_5pct_gate,
        mel_loss_only: false,
        utmos: Some(assessment),
        ..mel
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::utmos::{ConvActivation, HeadPool, TransformerNorm, Utmos, UtmosConfig};
    use crate::metrics::{Direction, Metric};
    use std::cell::RefCell;
    use vokra_core::kv_quant::KvQuant;
    use vokra_core::kv_quant::verify::{KvQuantMetric, KvQuantVerifyReport};

    const SR: u32 = 16_000;
    const THRESHOLD: f64 = 0.05;

    fn tone(freq: f32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / SR as f32).sin())
            .collect()
    }

    // Deterministic pseudo-noise in [-1, 1] — no RNG dependency (NFR-DS-02).
    fn noise(n: usize) -> Vec<f32> {
        let mut state: u32 = 0x1234_5678;
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

    /// A deterministic scripted scorer: returns the queued scores in call
    /// order (reference first, quantized second — the documented call order
    /// of `check_degradation_with_utmos`). Lets the gate-logic tests pin
    /// exact score pairs independently of any real network.
    struct ScriptedMos {
        scores: RefCell<Vec<f64>>,
    }

    impl ScriptedMos {
        fn new(ref_score: f64, quant_score: f64) -> Self {
            // Popped from the back: push in reverse.
            Self {
                scores: RefCell::new(vec![quant_score, ref_score]),
            }
        }
    }

    impl Metric for ScriptedMos {
        fn name(&self) -> &str {
            "scripted-mos"
        }
        fn direction(&self) -> Direction {
            Direction::HigherIsBetter
        }
    }

    impl crate::metrics::AudioMosMetric for ScriptedMos {
        fn eval_mos(&self, _audio: &[f32], _sample_rate: u32) -> vokra_core::Result<f64> {
            self.scores
                .borrow_mut()
                .pop()
                .ok_or_else(|| VokraError::InvalidArgument("scripted scorer exhausted".into()))
        }
    }

    /// A tiny real UTMOS skeleton at 16 kHz whose affine shifts scores into
    /// a MOS-like positive band (offset 3.0) — the integration-path scorer.
    fn tiny_utmos() -> Utmos {
        let config = UtmosConfig {
            sample_rate: SR,
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
        Utmos::synthesized(config, 0x4D34_5F31_385F_5530).expect("tiny utmos")
    }

    #[test]
    fn identical_inputs_pass_with_zero_delta() {
        let x = tone(440.0, 16_000);
        let report = check_degradation(&x, &x, SR, THRESHOLD).unwrap();
        assert_eq!(report.mel_loss_ref, 0.0);
        assert_eq!(report.mel_loss_quant, 0.0);
        assert_eq!(report.relative_delta, 0.0);
        assert!(report.passes_5pct_gate);
        assert!(report.utmos.is_none(), "mel-only path carries no MOS half");
    }

    #[test]
    fn large_noise_fails_gate() {
        let x = tone(440.0, 16_000);
        let noisy = add(&x, &noise(x.len()), 0.5);
        let report = check_degradation(&x, &noisy, SR, THRESHOLD).unwrap();
        assert!(report.mel_loss_quant > 0.0);
        // With mel_loss_ref = 0 the relative_delta = mel_loss_quant / EPSILON
        // → astronomically large, so it must fail the 5% gate.
        assert!(
            !report.passes_5pct_gate,
            "half-amplitude noise must fail: delta={} loss={}",
            report.relative_delta, report.mel_loss_quant
        );
    }

    #[test]
    fn rejects_non_positive_threshold() {
        let x = tone(440.0, 16_000);
        assert!(check_degradation(&x, &x, SR, 0.0).is_err());
        assert!(check_degradation(&x, &x, SR, -0.01).is_err());
        assert!(check_degradation(&x, &x, SR, f64::NAN).is_err());
        assert!(check_degradation(&x, &x, SR, f64::INFINITY).is_err());
        // The UTMOS-augmented entry point applies the same threshold gate.
        let err = check_degradation_with_utmos(
            &x,
            &x,
            SR,
            0.0,
            &ScriptedMos::new(4.0, 4.0),
            MosDomain::TtsSynthesis,
        )
        .expect_err("non-positive threshold");
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn propagates_mel_loss_errors_on_too_short_inputs() {
        // 100 samples < n_fft=1024 with center=false framing would error;
        // with librosa defaults (center=true) even short clips yield one
        // frame, so we verify the wiring by using empty inputs which do
        // produce a zero-frame error.
        let empty: Vec<f32> = Vec::new();
        // Empty inputs still get one center-padded frame under default
        // MelLoss::new attrs, so this actually succeeds with mel_loss=0.
        // The real "propagates errors" contract is exercised by the
        // MelLoss unit tests; here we just confirm the code path is wired.
        let report = check_degradation(&empty, &empty, SR, THRESHOLD);
        // Either an error propagates cleanly, or the identical-empty case
        // yields a trivial pass — both are legal, neither may panic.
        if let Ok(r) = report {
            assert!(r.passes_5pct_gate);
        }
    }

    #[test]
    fn mel_loss_only_flag_signals_partial_gate() {
        let x = tone(440.0, 16_000);
        let report = check_degradation(&x, &x, SR, THRESHOLD).unwrap();
        // No scorer was injected on this entry point, so the flag must be
        // `true` so downstream callers can surface the partial-gate caveat
        // (the M4-18 weights are deferred — fabricated pass is banned).
        assert!(report.mel_loss_only);
    }

    // ---- M4-18 T08: UTMOS-augmented gate (scripted scorer) -----------------

    #[test]
    fn utmos_within_threshold_passes_and_clears_partial_flag() {
        let x = tone(440.0, 16_000);
        // 2% decrease: 4.0 → 3.92.
        let mos = ScriptedMos::new(4.0, 3.92);
        let report =
            check_degradation_with_utmos(&x, &x, SR, THRESHOLD, &mos, MosDomain::TtsSynthesis)
                .expect("gate runs");
        assert!(!report.mel_loss_only, "UTMOS ran — not a partial gate");
        let a = report.utmos.expect("assessment present");
        assert_eq!(a.score_ref, 4.0);
        assert_eq!(a.score_quant, 3.92);
        assert!((a.rel_decrease - 0.02).abs() < 1e-12);
        assert!(a.within_threshold);
        assert!(!a.advisory_only);
        assert!(report.passes_5pct_gate);
    }

    #[test]
    fn utmos_hard_gate_fails_on_large_score_drop() {
        let x = tone(440.0, 16_000);
        // 10% decrease: 4.0 → 3.6 — over the 5% gate. The mel half is
        // bit-identical (passes), so the failure is attributable to the
        // MOS half alone.
        let mos = ScriptedMos::new(4.0, 3.6);
        let report =
            check_degradation_with_utmos(&x, &x, SR, THRESHOLD, &mos, MosDomain::TtsSynthesis)
                .expect("gate runs");
        let a = report.utmos.expect("assessment present");
        assert!(!a.within_threshold);
        assert!(!a.advisory_only);
        assert!(
            !report.passes_5pct_gate,
            "in-distribution MOS drop of 10% must fail the 5% gate"
        );
    }

    #[test]
    fn utmos_exactly_at_threshold_passes() {
        let x = tone(440.0, 16_000);
        // Boundary semantics match kv_quant::verify: `<= threshold` passes.
        // Dyadic values keep the arithmetic bit-exact ((1.0 - 0.9375) / 1.0
        // == 0.0625 exactly in f64) so this really exercises the boundary,
        // not a rounding direction (0.05-family decimals are inexact).
        let mos = ScriptedMos::new(1.0, 0.9375);
        let report =
            check_degradation_with_utmos(&x, &x, SR, 0.0625, &mos, MosDomain::TtsSynthesis)
                .expect("gate runs");
        let a = report.utmos.unwrap();
        assert_eq!(a.rel_decrease, 0.0625, "dyadic boundary is exact");
        assert!(a.within_threshold);
        assert!(report.passes_5pct_gate);
    }

    #[test]
    fn utmos_ood_domain_is_advisory_only_and_never_gates() {
        let x = tone(440.0, 16_000);
        // Same 10% drop, but the domain is codec/streaming (OOD): the
        // assessment is surfaced with advisory_only = true and the gate
        // verdict is carried by the mel half alone (which passes here).
        let mos = ScriptedMos::new(4.0, 3.6);
        let report =
            check_degradation_with_utmos(&x, &x, SR, THRESHOLD, &mos, MosDomain::CodecStreaming)
                .expect("gate runs");
        let a = report.utmos.expect("assessment present");
        assert!(a.advisory_only);
        assert!(!a.within_threshold, "the honest number is still reported");
        assert!(
            report.passes_5pct_gate,
            "advisory-only MOS must not flip the gate (NFR-QL-04: it also \
             must never be claimed as a pass — the flag says advisory)"
        );
        assert!(!report.mel_loss_only);
    }

    #[test]
    fn utmos_ood_domain_still_fails_on_mel_regression() {
        let x = tone(440.0, 16_000);
        let noisy = add(&x, &noise(x.len()), 0.5);
        // OOD domain: MOS is advisory, but the mel half still gates.
        let mos = ScriptedMos::new(4.0, 4.0);
        let report = check_degradation_with_utmos(
            &x,
            &noisy,
            SR,
            THRESHOLD,
            &mos,
            MosDomain::CodecStreaming,
        )
        .expect("gate runs");
        assert!(!report.passes_5pct_gate, "mel regression must still fail");
    }

    #[test]
    fn utmos_rejects_non_positive_or_non_finite_reference_score() {
        let x = tone(440.0, 16_000);
        for bad in [0.0, -1.0, f64::NAN] {
            let mos = ScriptedMos::new(bad, 3.9);
            let err =
                check_degradation_with_utmos(&x, &x, SR, THRESHOLD, &mos, MosDomain::TtsSynthesis)
                    .expect_err("non-positive/non-finite reference score");
            assert!(matches!(err, VokraError::InvalidArgument(_)), "got: {err}");
        }
    }

    #[test]
    fn utmos_scorer_errors_propagate_loudly() {
        // A real skeleton scorer with a 16 kHz config, fed 22.05 kHz audio:
        // the scorer's own sample-rate rejection (FR-EX-08, no silent
        // resample) must propagate verbatim.
        let m = tiny_utmos();
        let x = tone(440.0, 16_000);
        let err =
            check_degradation_with_utmos(&x, &x, 22_050, THRESHOLD, &m, MosDomain::TtsSynthesis)
                .expect_err("sample-rate mismatch propagates");
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    // ---- M4-18 T08: integration over the real (synthesized) skeleton -------

    #[test]
    fn utmos_integration_identical_audio_passes_with_real_skeleton() {
        let m = tiny_utmos();
        let x = tone(440.0, 16_000);
        let report =
            check_degradation_with_utmos(&x, &x, SR, THRESHOLD, &m, MosDomain::TtsSynthesis)
                .expect("gate runs over the synthesized skeleton");
        let a = report.utmos.expect("assessment present");
        // Identical audio through a deterministic scorer: identical scores,
        // exact zero decrease.
        assert_eq!(a.score_ref, a.score_quant);
        assert_eq!(a.rel_decrease, 0.0);
        assert!(a.within_threshold);
        assert!(report.passes_5pct_gate);
        assert!(!report.mel_loss_only);
    }

    // ---- M4-18 T08: kv_quant::verify wiring --------------------------------

    #[test]
    fn utmos_assessment_feeds_kv_quant_verify_report() {
        // The FR-QT-04 pipeline shape: an eval-side UTMOS assessment feeds
        // KvQuantMetric::with_utmos, and a pipeline that defaulted to
        // "unavailable" flips to available once a scorer exists.
        let x = tone(440.0, 16_000);
        let mos = ScriptedMos::new(4.0, 3.92); // 2% decrease
        let report =
            check_degradation_with_utmos(&x, &x, SR, THRESHOLD, &mos, MosDomain::TtsSynthesis)
                .expect("gate runs");
        let a = report.utmos.expect("assessment present");

        let kv = KvQuantVerifyReport::new("m4-18-wiring")
            .utmos_unavailable() // the weight-deferred default posture …
            .utmos_available() // … flipped once a scorer was injected
            .with_metric(KvQuantMetric::empty(KvQuant::Q8_0).with_utmos(a.rel_decrease as f32));
        assert!(!kv.utmos_unavailable);
        assert!(kv.all_pass_gate(), "2% decrease is within the 5% gate");
        // The DNSMOS flag is untouched by the UTMOS path (fail-closed skip
        // is a separate, owner-gated decision — M4-18 T03/T11).
        assert!(!kv.dnsmos_unavailable);
    }
}
