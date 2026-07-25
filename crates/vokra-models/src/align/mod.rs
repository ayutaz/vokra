//! Forced-alignment operators for Vokra.
//!
//! # Scope
//!
//! Given the per-frame log-probabilities emitted by a CTC-trained encoder
//! and the ground-truth token sequence, forced alignment recovers the
//! **time boundary** (`start_sec`, `end_sec`) of every token in the input
//! audio. It is a pure host-side algorithm — there are no external weights
//! (SoTA plan Phase X, `ctc_segmentation` — CAN be fully implemented in
//! Rust).
//!
//! # Op catalogue
//!
//! - [`ctc_segmentation`] — the classical CTC-based forced-alignment
//!   algorithm from Kürzinger et al., "CTC-Segmentation of Large Corpora
//!   for German End-to-end Speech Recognition", Interspeech 2020
//!   (arXiv:2007.09127; reference implementation
//!   `github.com/lumaku/ctc-segmentation`, Apache-2.0). The Rust
//!   re-implementation is a Viterbi walk over the standard CTC extended
//!   token sequence (blank between every pair of real tokens) so the
//!   algorithm covers word / sub-word / character tokens uniformly.
//!
//! # Output
//!
//! Both entry points return a `Vec<AlignedToken>` — one record per input
//! token with the frame-derived seconds and a per-token confidence score
//! in `(0, 1]`.

/// A single token's forced-alignment record.
///
/// # Fields
///
/// * `text` — the token label passed in by the caller (echoed back so the
///   caller does not have to keep its own parallel vector).
/// * `start_sec` — inclusive lower time boundary in seconds.
/// * `end_sec` — exclusive upper time boundary in seconds; always strictly
///   greater than `start_sec`.
/// * `confidence` — a heuristic score in `(0, 1]` computed as the mean
///   posterior probability along the aligned path frames for this token
///   (softmax-normalised per frame across the vocabulary).
#[derive(Debug, Clone, PartialEq)]
pub struct AlignedToken {
    /// Token label echoed back from the caller-supplied input.
    pub text: String,
    /// Inclusive lower time boundary in seconds.
    pub start_sec: f32,
    /// Exclusive upper time boundary in seconds; always strictly greater
    /// than [`AlignedToken::start_sec`].
    pub end_sec: f32,
    /// Heuristic per-token confidence in `(0, 1]`.
    pub confidence: f32,
}

pub mod ctc_segmentation;
