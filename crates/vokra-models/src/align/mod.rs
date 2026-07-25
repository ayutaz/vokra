//! Forced-alignment operators for Vokra.
//!
//! # Scope
//!
//! Forced alignment maps a known text / phoneme transcript onto its
//! time-alignment in the source audio — the canonical use cases are lyric
//! alignment (SVS), subtitle timing, mouth-shape / lip-sync generation for
//! game / VTuber pipelines, and prosody-transfer TTS
//! (CLAUDE.md 音声特化オペレータ §"Alignment / Duration / Prosody" —
//! `force_align` is the third-party operator in that section, distinct from
//! `mas` which is the VITS-family MAS alignment used inside the model).
//!
//! Given the per-frame log-probabilities emitted by a CTC-trained encoder
//! and the ground-truth token sequence, forced alignment recovers the
//! **time boundary** (`start_sec`, `end_sec`) of every token in the input
//! audio.
//!
//! # Op catalogue
//!
//! - [`ctc_segmentation`] — the classical CTC-based forced-alignment
//!   algorithm from Kürzinger et al., "CTC-Segmentation of Large Corpora
//!   for German End-to-end Speech Recognition", Interspeech 2020
//!   (arXiv:2007.09127; reference implementation
//!   `github.com/lumaku/ctc-segmentation`, Apache-2.0). Pure host-side
//!   algorithm — a Viterbi walk over the standard CTC extended token
//!   sequence (blank between every pair of real tokens) so the algorithm
//!   covers word / sub-word / character tokens uniformly. No external
//!   weights.
//! - [`charsiu`] — Wav2Vec2-based neural forced aligner (skeleton). A
//!   small subgraph loaded from a Vokra GGUF (`vokra.align.*` metadata +
//!   tensor weights); reports one [`AlignedToken`] per input phoneme.
//!
//! # Current status (skeleton for Charsiu)
//!
//! Charsiu is a landed skeleton — real wav2vec2 + CTC forced-alignment
//! inference is a follow-up WP; the skeleton only guarantees the load-error
//! and API-surface contracts so downstream consumers can wire the surface
//! without waiting on weights.
//!
//! # Output
//!
//! Both entry points return a `Vec<AlignedToken>` — one record per input
//! token with the frame-derived seconds and a per-token confidence score
//! in `(0, 1]`.

use std::path::PathBuf;

pub mod charsiu;
pub mod ctc_segmentation;

/// A single token's forced-alignment record.
///
/// # Fields
///
/// * `text` — the token label passed in by the caller (echoed back so the
///   caller does not have to keep its own parallel vector).
/// * `start_sec` — inclusive lower time boundary in seconds.
/// * `end_sec` — exclusive upper time boundary in seconds; always strictly
///   greater than `start_sec` for CTC segmentation, though skeleton
///   aligners may not enforce this until the real forward is landed.
/// * `confidence` — a heuristic score in `(0, 1]` computed as the mean
///   posterior probability along the aligned path frames for this token
///   (softmax-normalised per frame across the vocabulary).
#[derive(Debug, Clone, PartialEq)]
pub struct AlignedToken {
    /// Token label echoed back from the caller-supplied input.
    pub text: String,
    /// Inclusive lower time boundary in seconds.
    pub start_sec: f32,
    /// Exclusive upper time boundary in seconds.
    pub end_sec: f32,
    /// Heuristic per-token confidence in `(0, 1]`.
    pub confidence: f32,
}

/// An error produced while loading a forced aligner from a Vokra GGUF file.
///
/// Scoped to the `align::*` op surface — not a variant of the crate-wide
/// [`vokra_core::VokraError`]. Downstream consumers should map this to their
/// own error type at the integration boundary.
///
/// Note: [`ctc_segmentation`] is a pure host-side algorithm with no external
/// weights and does not surface `LoadError`; [`charsiu`] loads its wav2vec2
/// weights from a Vokra GGUF and returns this type.
#[derive(Debug)]
pub enum LoadError {
    /// The path did not exist or could not be opened.
    FileNotFound(PathBuf),
    /// The file was opened but the GGUF payload was malformed (or does not
    /// carry the expected `vokra.align.*` metadata / tensors).
    Gguf(String),
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoadError::FileNotFound(p) => write!(f, "align GGUF not found: {}", p.display()),
            LoadError::Gguf(m) => write!(f, "align GGUF error: {m}"),
        }
    }
}

impl std::error::Error for LoadError {}
