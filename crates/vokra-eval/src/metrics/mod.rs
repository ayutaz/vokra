//! The [`Metric`] interface and the algorithmic (weight-free) metrics.
//!
//! Metrics are split by the kind of input they score so a runner can pick the
//! right one without downcasting:
//!
//! - [`TextMetric`] — hypothesis string vs reference string ([`Wer`], [`Cer`]);
//! - [`AudioRefMetric`] — hypothesis waveform vs reference waveform
//!   ([`MelLoss`]);
//! - [`AudioMosMetric`] — a **reference-free** neural MOS predictor
//!   (UTMOS / DNSMOS). The trait was the reserved M1-09b slot; M4-18 wired in
//!   the first implementor, [`utmos::Utmos`] — a weight-deferred wav2vec2
//!   skeleton (real UTMOS weights are still owner-sourced, see the module
//!   docs). The wiring was additive — no existing caller of [`Metric`]
//!   changed. DNSMOS remains unimplemented (license fail-closed, M4-18 T03).

pub mod mel_loss;
pub mod utmos;
pub mod wer;

pub use mel_loss::MelLoss;
pub use utmos::{Utmos, UtmosConfig, UtmosWeights};
pub use wer::{Cer, Wer, edit_distance};

use vokra_core::Result;

/// Whether a metric reads better when its score is higher or lower.
///
/// Lets a runner rank/aggregate heterogeneous metrics consistently (error
/// rates go down, MOS goes up).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    /// Lower scores are better — the error-rate metrics (WER, CER, mel-loss).
    LowerIsBetter,
    /// Higher scores are better — the MOS predictors (UTMOS / DNSMOS, M1-09b).
    HigherIsBetter,
}

/// Shared metadata common to every evaluation metric.
pub trait Metric {
    /// Stable identifier used in reports and on the CLI (`wer`, `cer`,
    /// `mel_loss`).
    fn name(&self) -> &str;

    /// Score orientation (see [`Direction`]).
    fn direction(&self) -> Direction;
}

/// A metric scoring a hypothesis string against a reference string — the
/// transcription metrics ([`Wer`], [`Cer`]).
pub trait TextMetric: Metric {
    /// Scores `hyp` against `reference`. A total function; the empty-reference
    /// convention is documented per metric.
    fn eval_text(&self, hyp: &str, reference: &str) -> f64;
}

/// A metric scoring a hypothesis waveform against a reference waveform at a
/// shared sample rate — the reference-based audio metrics ([`MelLoss`]).
pub trait AudioRefMetric: Metric {
    /// Scores `hyp` against `reference` (mono PCM in `[-1, 1]`, both already at
    /// `sample_rate`).
    ///
    /// # Errors
    ///
    /// Fails on a front-end/shape mismatch (e.g. `sample_rate` disagreeing with
    /// the metric's configured rate, or inputs too short to yield a frame).
    fn eval_audio(&self, hyp: &[f32], reference: &[f32], sample_rate: u32) -> Result<f64>;
}

/// A **reference-free** neural MOS metric (UTMOS / DNSMOS).
///
/// Reserved as the M1-09b slot; the first implementor is [`utmos::Utmos`]
/// (M4-18) — a config-driven wav2vec2 + regression-head skeleton whose real
/// weights are still owner-sourced (the kickoff gate deferred them, see
/// `utmos` module docs). The trait itself is unchanged from M1-09a, so the
/// wiring was additive for every existing [`Metric`] caller. DNSMOS has no
/// implementor (license fail-closed until the owner's T03 verification).
pub trait AudioMosMetric: Metric {
    /// Predicts a mean-opinion score for a single `audio` clip at
    /// `sample_rate`.
    ///
    /// # Errors
    ///
    /// Fails on a front-end/shape mismatch or (once implemented) a model error.
    fn eval_mos(&self, audio: &[f32], sample_rate: u32) -> Result<f64>;
}
