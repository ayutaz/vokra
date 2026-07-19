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
//! Neural mean-opinion-score predictors: [`metrics::utmos::Utmos`] (M4-18)
//! implements the [`metrics::AudioMosMetric`] slot as a **weight-deferred
//! skeleton** — a config-driven wav2vec2-SSL + regression-head forward that
//! runs over synthesized (seed-deterministic) weights or a `vokra.utmos.*`
//! GGUF. The real UTMOS checkpoint + license are still owner-sourced (the
//! M4-18 kickoff gate auto-deferred them to a v1.0.x patch), so **no upstream
//! numerical claim is made yet**; scoring with real weights only needs the
//! flip-time converter, no API change. DNSMOS stays unimplemented (license
//! fail-closed).

pub mod cosyvoice2;
pub mod degradation;
pub mod gate;
pub mod manifest;
pub mod metrics;
pub mod nn;
pub mod wav;

pub use cosyvoice2::{
    COSYVOICE2_MEL_LOSS_THRESHOLD, COSYVOICE2_SAMPLE_RATE, check_cosyvoice2_degradation,
    check_cosyvoice2_degradation_with_utmos, cosyvoice2_mel_loss,
};
pub use degradation::{
    DegradationReport, MosAssessment, MosDomain, check_degradation, check_degradation_with_utmos,
};
pub use manifest::{Manifest, Record};
pub use metrics::{
    AudioMosMetric, AudioRefMetric, Cer, Direction, MelLoss, Metric, TextMetric, Wer, edit_distance,
};
pub use wav::{Wav, read_wav};
