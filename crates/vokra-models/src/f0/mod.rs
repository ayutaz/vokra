//! F0 (fundamental frequency / pitch) extraction (FR-OP-83).
//!
//! # Purpose
//!
//! Pitch extraction is a required input to voice-conversion / singing-voice
//! models (RVC v2, GPT-SoVITS) and a common conditioning signal for
//! high-quality TTS. Each F0 extractor is a per-model native subgraph — the
//! same 1:1-preserved pattern as [`crate::silero_vad`], never lowered to
//! generic audio-dialect ops (FR-LD-06 / NFR-QL-05). Unified across the
//! reference candidates (RMVPE / FCPE / CREPE / PyIN / Harvest — CLAUDE.md
//! 音声特化オペレータ §"F0 / Pitch 抽出").
//!
//! Each per-model skeleton is loaded from a Vokra GGUF
//! (`vokra.f0.*` metadata + tensor weights) and reports pitch per-frame
//! through the shared [`F0Frame`] type.
//!
//! # Members
//!
//! - [`rmvpe`] — Robust Model for Vocal Pitch Estimation
//!   (<https://github.com/Dream-High/RMVPE>, apache-2.0); polyphonic vocal
//!   pitch with V/UV detection; required by RVC.
//! - [`fcpe`] — Fast Context-based Pitch Estimation (skeleton).
//!
//! # Current status (skeleton)
//!
//! FCPE is a landed skeleton — real CNN / attention / autocorrelation
//! inference is a follow-up WP; the current skeleton guarantees only the
//! **frame-count contract** so downstream consumers can wire the API surface
//! without waiting on weights.
//!
//! # Frame shape
//!
//! Every F0 extractor emits [`F0Frame`] rows on a per-hop timebase. The frame
//! itself is model-agnostic (`time_sec` / `hz` / `voiced` / `confidence`), so
//! downstream consumers (VC / TTS conditioners) can share one shape across
//! extractors.

use std::path::PathBuf;

pub mod fcpe;
pub mod rmvpe;

/// A single frame of F0 (pitch) output.
///
/// One row per analysis hop; consumers align [`time_sec`](Self::time_sec)
/// against the source PCM to build a per-frame prosody / conditioning stream.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct F0Frame {
    /// Frame time in seconds from the start of the PCM buffer (the hop-aligned
    /// left edge of the analysis window).
    pub time_sec: f32,
    /// Estimated fundamental frequency in Hz. `0.0` when
    /// [`voiced`](Self::voiced) is `false`.
    pub hz: f32,
    /// Whether the frame carries voiced (harmonic) speech.
    pub voiced: bool,
    /// Model confidence in `[0.0, 1.0]` (`1.0` = maximally confident,
    /// `0.0` = no evidence).
    pub confidence: f32,
}

/// An error produced while loading an F0 extractor from a Vokra GGUF file.
///
/// Scoped to the `f0::*` op surface — not a variant of the crate-wide
/// [`vokra_core::VokraError`]. Downstream consumers should map this to their
/// own error type at the integration boundary.
///
/// Note: `RMVPE::from_gguf` intentionally maps its errors to the crate-wide
/// `vokra_core::VokraError` (see [`rmvpe`]'s design-note comment) and does
/// not use this type; `FCPE::from_gguf` uses this local error type.
#[derive(Debug)]
pub enum LoadError {
    /// The path did not exist or could not be opened.
    FileNotFound(PathBuf),
    /// The file was opened but the GGUF payload was malformed (or does not
    /// carry the expected `vokra.f0.*` metadata / tensors).
    Gguf(String),
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoadError::FileNotFound(p) => write!(f, "f0 GGUF not found: {}", p.display()),
            LoadError::Gguf(m) => write!(f, "f0 GGUF error: {m}"),
        }
    }
}

impl std::error::Error for LoadError {}
