//! FCPE — **Fast Context-based Pitch Estimator** (lightweight CNN-based F0
//! extractor).
//!
//! - Upstream: <https://github.com/CNChTu/FCPE>
//! - License: MIT (Permissive; no runtime-side attribution obligation).
//!
//! FCPE is one of the reference F0 candidates for the Vokra `f0::*` op
//! surface (FR-OP-83 / CLAUDE.md 音声特化オペレータ §"F0 / Pitch 抽出").
//! This module is the FCPE **skeleton** for the surface. Real CNN inference
//! (context-based convolutional pitch classifier + soft-argmax over a
//! bins-per-octave logit table) is a follow-up WP — see [`FCPE::extract`].

use std::path::Path;

use vokra_core::gguf::{GgufError, GgufFile};

use super::{F0Frame, LoadError};

/// Default hop length in samples (`vokra.f0.fcpe.hop`).
const DEFAULT_HOP: u32 = 160;
/// Default minimum-detectable pitch in Hz (`vokra.f0.fcpe.fmin`).
const DEFAULT_FMIN: f32 = 50.0;
/// Default maximum-detectable pitch in Hz (`vokra.f0.fcpe.fmax`).
const DEFAULT_FMAX: f32 = 1100.0;

/// FCPE F0 (pitch) extractor.
///
/// Construct with [`from_gguf`](Self::from_gguf); pitch is emitted per hop
/// via [`extract`](Self::extract).
#[derive(Debug)]
pub struct FCPE {
    hop: u32,
    fmin: f32,
    fmax: f32,
}

impl FCPE {
    /// Binds an FCPE from a Vokra GGUF checkpoint.
    ///
    /// Reads the three configuration metadata keys and falls back to their
    /// canonical defaults if absent:
    /// - `vokra.f0.fcpe.hop` (u32, default 160)
    /// - `vokra.f0.fcpe.fmin` (f32, default 50.0)
    /// - `vokra.f0.fcpe.fmax` (f32, default 1100.0)
    ///
    /// Returns [`LoadError::FileNotFound`] if the path cannot be opened and
    /// [`LoadError::Gguf`] if the file is not a valid GGUF.
    pub fn from_gguf(path: &Path) -> Result<Self, LoadError> {
        let gguf = GgufFile::open(path).map_err(|e| map_gguf_err(path, e))?;
        let hop = gguf
            .get("vokra.f0.fcpe.hop")
            .and_then(|v| v.as_u64())
            .and_then(|v| u32::try_from(v).ok())
            .unwrap_or(DEFAULT_HOP);
        let fmin = gguf
            .get("vokra.f0.fcpe.fmin")
            .and_then(|v| v.as_f64())
            .map(|v| v as f32)
            .unwrap_or(DEFAULT_FMIN);
        let fmax = gguf
            .get("vokra.f0.fcpe.fmax")
            .and_then(|v| v.as_f64())
            .map(|v| v as f32)
            .unwrap_or(DEFAULT_FMAX);
        Ok(Self { hop, fmin, fmax })
    }

    /// Extracts an F0 track from PCM samples.
    ///
    /// SKELETON: real fcpe CNN inference is a follow-up WP; this only
    /// guarantees the frame-count contract.
    pub fn extract(&self, pcm: &[f32], sample_rate: u32) -> Vec<F0Frame> {
        let hop = self.hop as usize;
        if hop == 0 {
            return Vec::new();
        }
        let n_frames = pcm.len() / hop;
        let sr = sample_rate.max(1) as f32;
        (0..n_frames)
            .map(|i| F0Frame {
                time_sec: (i * hop) as f32 / sr,
                hz: 0.0,
                voiced: false,
                confidence: 0.0,
            })
            .collect()
    }

    /// Returns the configured hop length (samples per frame).
    pub fn hop(&self) -> u32 {
        self.hop
    }

    /// Returns the configured minimum-detectable pitch in Hz.
    pub fn fmin(&self) -> f32 {
        self.fmin
    }

    /// Returns the configured maximum-detectable pitch in Hz.
    pub fn fmax(&self) -> f32 {
        self.fmax
    }
}

/// Maps a [`GgufError`] into the local [`LoadError`], collapsing an I/O
/// "not found" into the dedicated [`LoadError::FileNotFound`] variant.
fn map_gguf_err(path: &Path, e: GgufError) -> LoadError {
    match e {
        GgufError::Io(io) if io.kind() == std::io::ErrorKind::NotFound => {
            LoadError::FileNotFound(path.to_path_buf())
        }
        other => LoadError::Gguf(format!("{other:?}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fcpe_load_stub_reports_load_error() {
        // A path that cannot exist -> Err(LoadError::*), never a panic.
        let missing = Path::new("/nonexistent/vokra-fcpe-does-not-exist.gguf");
        let err = FCPE::from_gguf(missing).expect_err("missing GGUF must be LoadError");
        assert!(
            matches!(err, LoadError::FileNotFound(_) | LoadError::Gguf(_)),
            "unexpected LoadError variant: {err:?}",
        );
    }

    #[test]
    fn fcpe_extract_frame_count_matches_hop() {
        // Once implemented: extract(pcm, 16 kHz).len() == pcm.len() / hop
        // (default hop = 160). Constructing the skeleton directly with
        // module-private fields keeps the frame-count contract testable
        // before real weights land.
        let hop = 160usize;
        let pcm = vec![0.0f32; hop * 10]; // ten frames' worth (100 ms @ 16 kHz)
        let fcpe = FCPE {
            hop: hop as u32,
            fmin: 50.0,
            fmax: 1100.0,
        };
        let frames = fcpe.extract(&pcm, 16_000);
        assert_eq!(frames.len(), pcm.len() / hop);
    }
}
