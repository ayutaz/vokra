//! # vokra-eval
//!
//! Evaluation metrics for the Vokra speech runtime (M1-09a; FR-TL-03,
//! NFR-QL-04) plus the `vokra-eval` CLI binary.
//!
//! Two things live here:
//!
//! - a small, reusable **metric library** — a pluggable [`metrics::Metric`]
//!   interface and the *algorithmic* metrics that need no model weights:
//!   [`metrics::MelLoss`] (log-mel L1), [`metrics::Wer`] and [`metrics::Cer`]
//!   (Levenshtein edit-rate). It is `std`-only and reuses the first-party
//!   `vokra-ops` STFT / mel op path (never a third-party crate — NFR-DS-02).
//! - the `vokra-eval` binary (`src/main.rs`) that runs one metric over a single
//!   hypothesis/reference pair or a [`manifest`] of them and prints the scores
//!   in a `key=value` report.
//!
//! Neural mean-opinion-score predictors (UTMOS / DNSMOS) are **not** here: they
//! are networks that need model weights (M1-09b, blocked on those weights). The
//! [`metrics::AudioMosMetric`] trait reserves their slot so they drop in later
//! without changing any caller of [`metrics::Metric`].

pub mod cosyvoice2;
pub mod degradation;
pub mod manifest;
pub mod metrics;
pub mod wav;

pub use cosyvoice2::{
    COSYVOICE2_MEL_LOSS_THRESHOLD, COSYVOICE2_SAMPLE_RATE, check_cosyvoice2_degradation,
    check_cosyvoice2_degradation_with_utmos, cosyvoice2_mel_loss,
};
pub use degradation::{DegradationReport, check_degradation, check_degradation_with_utmos};
pub use manifest::{Manifest, Record};
pub use metrics::{
    AudioMosMetric, AudioRefMetric, Cer, Direction, MelLoss, Metric, TextMetric, Wer, edit_distance,
};
pub use wav::{Wav, read_wav};
