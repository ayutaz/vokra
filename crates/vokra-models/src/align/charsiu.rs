//! Charsiu — **Wav2Vec2 neural forced aligner**.
//!
//! - Upstream: <https://github.com/lingjzhu/charsiu>
//! - License: MIT (Permissive; no runtime-side attribution obligation).
//!
//! Charsiu is one of the reference forced-alignment candidates for the
//! Vokra `align::*` op surface (CLAUDE.md 音声特化オペレータ §"Alignment /
//! Duration / Prosody" — `force_align`). This module is the Charsiu
//! **skeleton** for the surface.
//!
//! SKELETON: real wav2vec2 CTC alignment is a follow-up WP. The wav2vec2
//! acoustic encoder + CTC head weights are external and are not shipped
//! with the skeleton; only the load-error and API-surface contracts are
//! wired today so downstream consumers can compile against the shape
//! before real weights land.

use std::path::Path;

use super::{AlignedToken, LoadError};

/// Charsiu forced aligner.
///
/// Construct with [`from_gguf`](Self::from_gguf); alignments are emitted
/// per input phoneme via [`align`](Self::align).
///
/// SKELETON: real wav2vec2 CTC alignment is a follow-up WP.
#[derive(Debug)]
pub struct Charsiu {
    _private: (),
}

impl Charsiu {
    /// Binds a Charsiu aligner from a Vokra GGUF checkpoint.
    ///
    /// SKELETON: real wav2vec2 CTC alignment is a follow-up WP; the
    /// wav2vec2 weights are external and are not shipped with the
    /// skeleton. This entry point currently:
    ///
    /// * Returns [`LoadError::FileNotFound`] if the path does not exist —
    ///   the load-error contract downstream consumers can rely on today.
    /// * Panics with [`unimplemented!`] if the path exists (once real
    ///   weights land this becomes the wav2vec2 GGUF binder).
    pub fn from_gguf(path: &Path) -> Result<Self, LoadError> {
        if !path.exists() {
            return Err(LoadError::FileNotFound(path.to_path_buf()));
        }
        unimplemented!(
            "SKELETON: real wav2vec2 CTC alignment is a follow-up WP; the wav2vec2 weights are external and are not shipped with the skeleton."
        );
    }

    /// Forced-aligns a phoneme sequence to the source PCM.
    ///
    /// SKELETON: real wav2vec2 CTC alignment is a follow-up WP.
    pub fn align(
        &self,
        _pcm: &[f32],
        _sample_rate: u32,
        _phonemes: &[String],
    ) -> Vec<AlignedToken> {
        unimplemented!(
            "SKELETON: real wav2vec2 CTC alignment is a follow-up WP; the wav2vec2 weights are external and are not shipped with the skeleton."
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn charsiu_load_stub_reports_load_error() {
        // A path that cannot exist -> Err(LoadError::*), never a panic.
        let missing = Path::new("/nonexistent/vokra-charsiu-does-not-exist.gguf");
        let err = Charsiu::from_gguf(missing).expect_err("missing GGUF must be LoadError");
        assert!(
            matches!(err, LoadError::FileNotFound(_) | LoadError::Gguf(_)),
            "unexpected LoadError variant: {err:?}",
        );
    }
}
