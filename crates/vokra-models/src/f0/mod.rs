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
//! Each per-model extractor is loaded from a Vokra GGUF
//! (`vokra.f0.*` metadata + tensor weights) and reports pitch per-frame
//! through the shared [`F0Frame`] type.
//!
//! # Members
//!
//! - [`rmvpe`] — Robust Model for Vocal Pitch Estimation
//!   (<https://github.com/Dream-High/RMVPE>, apache-2.0); polyphonic vocal
//!   pitch with V/UV detection; required by RVC.
//! - [`fcpe`] — Fast Context-based Pitch Estimation (CFNaiveMelPEInfer,
//!   CNChTu/FCPE, MIT); a GLU-conv encoder over log-mel.
//! - [`crepe`] — Convolutional Representation for Pitch Estimation
//!   (Kim et al. 2018, MIT), a monophonic 6-block CNN over raw 16 kHz audio.
//!
//! # Shared API shape (all three, since 2026-08-15)
//!
//! - `extract(&pcm, sample_rate) -> Result<Vec<F0Frame>, VokraError>` — the
//!   obvious name, and the one that measures. Delegates to `extract_real`.
//! - `extract_real(..) -> Result<Vec<F0Frame>, VokraError>` — the real
//!   forward, under the name the parity harnesses call.
//! - `frame_times(pcm_len, sample_rate) -> Vec<f32>` — the analysis
//!   timestamps alone, for sizing and aligning buffers. It returns bare
//!   seconds rather than [`F0Frame`] rows precisely so that nothing it
//!   returns can be mistaken for a pitch estimate.
//!
//! **No extractor answers a failure with a zero-filled track** (FR-EX-08).
//! Unbound weights, a sample rate the checkpoint is not defined at, and a
//! front-end that cannot run are three distinct named errors. Each names what
//! it received against what it needs. A frame-count-correct all-zero `F0Frame`
//! track would be indistinguishable downstream from "this audio is entirely
//! unvoiced", and silently wrong pitch propagates straight into a vocoder or
//! a VC pipeline — worse than no pitch at all. None of them resamples on the
//! caller's behalf either: refusing is the point.
//!
//! # Frame shape
//!
//! Every F0 extractor emits [`F0Frame`] rows on a per-hop timebase, with
//! `frames.len() == pcm.len() / hop`. The frame itself is model-agnostic
//! (`time_sec` / `hz` / `voiced` / `confidence`), so downstream consumers
//! (VC / TTS conditioners) can share one shape across extractors.

use std::path::PathBuf;

pub mod crepe;
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
/// not use this type; `FCPE::from_gguf` and `CREPE::from_gguf` use this
/// local error type.
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
