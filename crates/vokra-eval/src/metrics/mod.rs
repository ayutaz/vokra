//! The [`Metric`] interface and the algorithmic (weight-free) metrics.
//!
//! Metrics are split by the kind of input they score so a runner can pick the
//! right one without downcasting:
//!
//! - [`TextMetric`] — hypothesis string vs reference string ([`Wer`], [`Cer`]);
//! - [`AudioRefMetric`] — hypothesis waveform vs reference waveform
//!   ([`MelLoss`]);
//! - [`AudioMosMetric`] — a **reference-free** neural MOS predictor
//!   (UTMOS / DNSMOS). This trait is the reserved slot for M1-09b and is
//!   deliberately left without an implementation here: those metrics are neural
//!   networks that need model weights, which are not available yet. Wiring one
//!   in later is additive — no existing caller of [`Metric`] changes.

pub mod mel_loss;
pub mod wer;

pub use mel_loss::MelLoss;
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
/// Reserved slot for M1-09b — intentionally without any implementation in this
/// WP because those metrics are neural networks whose weights are not yet
/// available. Once the weights land, a type implementing this trait plugs into
/// the same [`Metric`] machinery with no change to existing callers.
pub trait AudioMosMetric: Metric {
    /// Predicts a mean-opinion score for a single `audio` clip at
    /// `sample_rate`.
    ///
    /// # Errors
    ///
    /// Fails on a front-end/shape mismatch or (once implemented) a model error.
    fn eval_mos(&self, audio: &[f32], sample_rate: u32) -> Result<f64>;
}
