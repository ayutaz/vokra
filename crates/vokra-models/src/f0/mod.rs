//! F0 (fundamental frequency / pitch) extraction (FR-OP-83).
//!
//! # Purpose
//!
//! Pitch extraction is a required input to voice-conversion / singing-voice
//! models (RVC v2, GPT-SoVITS) and a common conditioning signal for
//! high-quality TTS. Each F0 extractor is a per-model native subgraph — the
//! same 1:1-preserved pattern as [`crate::silero_vad`], never lowered to
//! generic audio-dialect ops (FR-LD-06 / NFR-QL-05).
//!
//! # Members
//!
//! - [`rmvpe`] — Robust Model for Vocal Pitch Estimation
//!   (<https://github.com/Dream-High/RMVPE>, apache-2.0); polyphonic vocal
//!   pitch with V/UV detection; required by RVC.
//!
//! # Frame shape
//!
//! Every F0 extractor emits [`F0Frame`] rows on a per-hop timebase. The frame
//! itself is model-agnostic (`time_sec` / `hz` / `voiced` / `confidence`), so
//! downstream consumers (VC / TTS conditioners) can share one shape across
//! extractors.

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
