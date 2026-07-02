//! Task API facades (M0-02-T12).
//!
//! FR-API-02 defines the task facades `session.asr().transcribe()`,
//! `session.tts().synthesize()` and `session.s2s().dialog()`. M0 ships the
//! API shapes as stubs returning
//! [`VokraError::NotImplemented`]; the real wiring belongs to later work
//! packages:
//!
//! - VAD: **M0-05** (Silero VAD subgraph)
//! - ASR: **M0-06** (Whisper base — encoder / decoder / beam search)
//! - TTS: **M0-07** (piper-plus native TTS, MB-iSTFT-VITS2)
//! - S2S: supported models are **v1.0+** (SRS §2.9 / CLAUDE.md model table),
//!   so [`S2s`] is a *long-term* stub.

use crate::error::{Result, VokraError};
use crate::session::Session;

impl Session {
    /// ASR (speech-to-text) facade — FR-API-02: `session.asr().transcribe()`.
    pub fn asr(&self) -> Asr<'_> {
        Asr { session: self }
    }

    /// TTS (text-to-speech) facade — FR-API-02: `session.tts().synthesize()`.
    pub fn tts(&self) -> Tts<'_> {
        Tts { session: self }
    }

    /// S2S (speech-to-speech) facade — FR-API-02: `session.s2s().dialog()`.
    pub fn s2s(&self) -> S2s<'_> {
        S2s { session: self }
    }
}

/// ASR facade borrowed from a [`Session`] (FR-API-02).
#[derive(Debug)]
pub struct Asr<'a> {
    #[allow(dead_code)] // read once ASR wiring lands (M0-06); unused in the M0 skeleton
    session: &'a Session,
}

impl Asr<'_> {
    /// Transcribes mono `f32` PCM samples to text.
    ///
    /// M0 stub: always returns
    /// [`VokraError::NotImplemented`]. Wiring: Whisper base in **M0-06**
    /// (VAD in M0-05).
    pub fn transcribe(&self, _samples: &[f32]) -> Result<Transcription> {
        Err(VokraError::NotImplemented(
            "ASR wiring lands in M0-06 (Whisper base)",
        ))
    }
}

/// TTS facade borrowed from a [`Session`] (FR-API-02).
#[derive(Debug)]
pub struct Tts<'a> {
    #[allow(dead_code)] // read once TTS wiring lands (M0-07); unused in the M0 skeleton
    session: &'a Session,
}

impl Tts<'_> {
    /// Synthesizes speech audio from `text`.
    ///
    /// M0 stub: always returns
    /// [`VokraError::NotImplemented`]. Wiring: piper-plus native TTS
    /// (MB-iSTFT-VITS2) in **M0-07**.
    pub fn synthesize(&self, _text: &str) -> Result<SynthesizedAudio> {
        Err(VokraError::NotImplemented(
            "TTS wiring lands in M0-07 (piper-plus native TTS)",
        ))
    }
}

/// S2S facade borrowed from a [`Session`] (FR-API-02).
///
/// S2S-capable models (CosyVoice2 / Sesame CSM / Moshi) are v1.0+ scope
/// (SRS §2.9 / CLAUDE.md model table), so this facade stays a stub well
/// beyond M0.
#[derive(Debug)]
pub struct S2s<'a> {
    #[allow(dead_code)] // read once S2S models land (v1.0+); unused in the M0 skeleton
    session: &'a Session,
}

impl S2s<'_> {
    /// Runs one speech-to-speech dialog turn over mono `f32` PCM input.
    ///
    /// Long-term stub: always returns
    /// [`VokraError::NotImplemented`] until S2S models arrive (v1.0+).
    pub fn dialog(&self, _samples: &[f32]) -> Result<DialogTurn> {
        Err(VokraError::NotImplemented(
            "S2S models are v1.0+ scope (long-term stub)",
        ))
    }
}

/// Placeholder result of [`Asr::transcribe`] (fields grow with M0-06:
/// timestamps, n-best, ...).
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct Transcription {
    /// Recognized text.
    pub text: String,
}

/// Placeholder result of [`Tts::synthesize`] (fields grow with M0-07).
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct SynthesizedAudio {
    /// Mono PCM samples in `[-1.0, 1.0]`.
    pub samples: Vec<f32>,
    /// Sample rate in Hz.
    pub sample_rate: u32,
}

/// Placeholder result of [`S2s::dialog`] (v1.0+ scope).
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct DialogTurn {
    /// Text form of the reply (inner monologue / transcript).
    pub text: String,
    /// Synthesized reply audio, when produced.
    pub audio: Option<SynthesizedAudio>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::tests::TempModelFile;

    fn session(tag: &str) -> (TempModelFile, Session) {
        let file = TempModelFile::new(tag);
        let session = Session::from_file(&file.0).build().expect("session builds");
        (file, session)
    }

    #[test]
    fn asr_facade_has_fr_api_02_shape_and_is_stubbed() {
        let (_file, session) = session("asr");
        let result = session.asr().transcribe(&[0.0f32; 160]);
        assert!(matches!(result, Err(VokraError::NotImplemented(_))));
    }

    #[test]
    fn tts_facade_has_fr_api_02_shape_and_is_stubbed() {
        let (_file, session) = session("tts");
        let result = session.tts().synthesize("hello vokra");
        assert!(matches!(result, Err(VokraError::NotImplemented(_))));
    }

    #[test]
    fn s2s_facade_has_fr_api_02_shape_and_is_stubbed() {
        let (_file, session) = session("s2s");
        let result = session.s2s().dialog(&[0.0f32; 160]);
        assert!(matches!(result, Err(VokraError::NotImplemented(_))));
    }
}
