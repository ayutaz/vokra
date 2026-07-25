//! CREPE — Convolutional Representation for Pitch Estimation (Kim et al. 2018).
//!
//! Upstream reference: <https://github.com/marl/crepe> (license: **MIT**).
//! CREPE is a monophonic F0 extractor whose front-end is a stack of dilated 1D
//! convolutions over the raw waveform, followed by a 360-bin classifier over
//! a log-frequency grid; the argmax bin is decoded into Hz with a local
//! centroid.
//!
//! # Status
//!
//! This module is an **HONEST UNIMPLEMENTED skeleton**. [`CREPE::from_gguf`]
//! reads the three tuning knobs (hop, fmin, fmax) from GGUF metadata and
//! reports [`LoadError`] on a missing / malformed file, and [`CREPE::extract`]
//! guarantees the **frame-count contract** (`pcm.len() / hop` frames per
//! call, hop=160 by default). The real CNN forward — dilated conv stack,
//! 360-bin softmax classifier, local-centroid decode into Hz — is a follow-up
//! WP.
//!
//! # No ONNX (permanent)
//!
//! The upstream `crepe` package ships a Keras / TensorFlow model; the model
//! definition will be re-implemented natively (whisper.cpp 型, CLAUDE.md
//! 設計判断 4) once the follow-up wave lands. This module never touches ONNX.

use std::path::Path;

use vokra_core::gguf::GgufFile;

use super::{F0Frame, LoadError};

/// Default hop between output frames, in samples. Matches the upstream
/// `crepe.predict(..., step_size=10)` default at the canonical 16 kHz input
/// rate (10 ms → 160 samples).
const DEFAULT_HOP: u32 = 160;

/// Default lower bound of the pitch search grid (Hz). Matches the upstream
/// CREPE classifier's low log-frequency edge.
const DEFAULT_FMIN: f32 = 50.0;

/// Default upper bound of the pitch search grid (Hz). Matches the upstream
/// CREPE classifier's high log-frequency edge.
const DEFAULT_FMAX: f32 = 1100.0;

/// The CREPE F0 extractor (Convolutional Representation for Pitch Estimation).
///
/// Acronym-cased per the F0 op family (FR-OP-83) — the load / extract surface
/// is the same across siblings (PyIN, FCPE, Harvest, RMVPE), so
/// `CREPE::extract` names the extractor rather than being noun-cased.
#[allow(clippy::upper_case_acronyms)]
pub struct CREPE {
    hop: u32,
    #[allow(dead_code)] // consumed by the real CNN forward (follow-up WP).
    fmin: f32,
    #[allow(dead_code)] // consumed by the real CNN forward (follow-up WP).
    fmax: f32,
}

impl CREPE {
    /// Loads CREPE from a GGUF file on disk.
    ///
    /// Reads three OPTIONAL metadata keys and falls back to the upstream
    /// defaults if a key is absent:
    ///
    /// - `vokra.f0.crepe.hop`  (u32, default `160`  — 10 ms at 16 kHz)
    /// - `vokra.f0.crepe.fmin` (f32, default `50.0` Hz)
    /// - `vokra.f0.crepe.fmax` (f32, default `1100.0` Hz)
    ///
    /// Returns [`LoadError`] if the path cannot be opened / parsed, or if a
    /// key is present with the wrong type.
    pub fn from_gguf(path: &Path) -> Result<Self, LoadError> {
        let file = GgufFile::open(path).map_err(|e| LoadError::Gguf(format!("{e:?}")))?;

        let hop = read_opt_u32(&file, "vokra.f0.crepe.hop")?.unwrap_or(DEFAULT_HOP);
        let fmin = read_opt_f32(&file, "vokra.f0.crepe.fmin")?.unwrap_or(DEFAULT_FMIN);
        let fmax = read_opt_f32(&file, "vokra.f0.crepe.fmax")?.unwrap_or(DEFAULT_FMAX);

        Ok(Self { hop, fmin, fmax })
    }
}

impl CREPE {
    /// SKELETON: real crepe CNN inference is a follow-up WP; this only
    /// guarantees the frame-count contract.
    ///
    /// Every frame is emitted with `hz = 0.0`, `voiced = false` and
    /// `confidence = 0.0` — an honestly-unimplemented placeholder that
    /// nonetheless lets callers size buffers and align timelines around the
    /// real forward's future output.
    pub fn extract(&self, pcm: &[f32], sample_rate: u32) -> Vec<F0Frame> {
        let hop = self.hop.max(1) as usize;
        let n_frames = pcm.len() / hop;
        // `sample_rate == 0` would be a nonsense input; guard it so the
        // timestamp column stays finite (`0.0`) rather than NaN / ±inf.
        let sr = (sample_rate as f32).max(1.0);
        (0..n_frames)
            .map(|i| F0Frame {
                time_sec: (i * hop) as f32 / sr,
                hz: 0.0,
                voiced: false,
                confidence: 0.0,
            })
            .collect()
    }
}

fn read_opt_u32(file: &GgufFile, key: &str) -> Result<Option<u32>, LoadError> {
    match file.get(key) {
        Some(v) => match v.as_u64().and_then(|n| u32::try_from(n).ok()) {
            Some(n) => Ok(Some(n)),
            None => Err(LoadError::Gguf(format!(
                "crepe metadata `{key}` is not a u32-range integer",
            ))),
        },
        None => Ok(None),
    }
}

fn read_opt_f32(file: &GgufFile, key: &str) -> Result<Option<f32>, LoadError> {
    match file.get(key) {
        Some(v) => match v.as_f64() {
            Some(n) => Ok(Some(n as f32)),
            None => Err(LoadError::Gguf(format!(
                "crepe metadata `{key}` is not a float",
            ))),
        },
        None => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::GgufBuilder;

    /// A GGUF path that does not exist must produce a [`LoadError`] rather
    /// than a panic or a silent success.
    #[test]
    fn crepe_load_stub_reports_load_error() {
        let path = Path::new("/vokra-nonexistent-crepe-fixture.gguf");
        let result = CREPE::from_gguf(path);
        assert!(
            result.is_err(),
            "expected LoadError for nonexistent path, got Ok",
        );
    }

    /// The frame-count contract holds: one output frame per `hop` input
    /// samples (hop=160 by default).
    #[test]
    fn crepe_extract_frame_count_matches_hop() {
        let tmp = std::env::temp_dir().join(format!(
            "vokra-crepe-skeleton-frame-count-{}.gguf",
            std::process::id(),
        ));
        let bytes = GgufBuilder::new().to_bytes().unwrap();
        std::fs::write(&tmp, &bytes).unwrap();

        let crepe = CREPE::from_gguf(&tmp).expect("load skeleton GGUF");
        let pcm = vec![0.0f32; 1_600];
        let frames = crepe.extract(&pcm, 16_000);
        assert_eq!(frames.len(), pcm.len() / 160);

        let _ = std::fs::remove_file(&tmp);
    }
}
