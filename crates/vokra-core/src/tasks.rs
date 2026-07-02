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

use crate::engines::SynthesisRequest;
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
    session: &'a Session,
}

impl Asr<'_> {
    /// Transcribes mono `f32` PCM samples to text.
    ///
    /// Delegates to the [`AsrEngine`](crate::engines::AsrEngine) injected via
    /// [`Session::with_asr_engine`](crate::Session::with_asr_engine) (Whisper
    /// base = M0-06). Without an injected engine it returns
    /// [`VokraError::NotImplemented`].
    pub fn transcribe(&self, samples: &[f32]) -> Result<Transcription> {
        match self.session.asr_engine() {
            Some(engine) => engine.transcribe(samples),
            None => Err(VokraError::NotImplemented(
                "no ASR engine injected (Whisper base = M0-06)",
            )),
        }
    }
}

/// TTS facade borrowed from a [`Session`] (FR-API-02).
#[derive(Debug)]
pub struct Tts<'a> {
    session: &'a Session,
}

impl Tts<'_> {
    /// Synthesizes speech audio from `text` (voice defaults).
    ///
    /// The FR-API-02 verbatim shape. Delegates to the
    /// [`TtsEngine`](crate::engines::TtsEngine) injected via
    /// [`Session::with_tts_engine`](crate::Session::with_tts_engine)
    /// (piper-plus native TTS = M0-07) with a default
    /// [`SynthesisRequest`](crate::engines::SynthesisRequest). Use
    /// [`synthesize_request`](Self::synthesize_request) for explicit options.
    /// Without an injected engine it returns [`VokraError::NotImplemented`].
    pub fn synthesize(&self, text: &str) -> Result<SynthesizedAudio> {
        self.synthesize_request(&SynthesisRequest::new(text))
    }

    /// Synthesizes speech audio for an explicit
    /// [`SynthesisRequest`](crate::engines::SynthesisRequest) (language,
    /// determinism, ...).
    pub fn synthesize_request(&self, request: &SynthesisRequest) -> Result<SynthesizedAudio> {
        match self.session.tts_engine() {
            Some(engine) => engine.synthesize(request),
            None => Err(VokraError::NotImplemented(
                "no TTS engine injected (piper-plus native TTS = M0-07)",
            )),
        }
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

/// Result of [`Asr::transcribe`] (fields grow with M0-06: timestamps,
/// n-best, ...).
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct Transcription {
    /// Recognized text.
    pub text: String,
}

impl Transcription {
    /// A transcription carrying just `text` (the M0 shape). Provided so
    /// engine crates (`vokra-models`) can construct this `#[non_exhaustive]`
    /// type across the crate boundary.
    pub fn new(text: impl Into<String>) -> Self {
        Self { text: text.into() }
    }
}

/// Result of [`Tts::synthesize`] (fields grow with M0-07).
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct SynthesizedAudio {
    /// Mono PCM samples in `[-1.0, 1.0]`.
    pub samples: Vec<f32>,
    /// Sample rate in Hz.
    pub sample_rate: u32,
}

impl SynthesizedAudio {
    /// Mono synthesized audio at `sample_rate` Hz. Provided so engine crates
    /// (`vokra-models`) can construct this `#[non_exhaustive]` type across the
    /// crate boundary.
    pub fn new(samples: Vec<f32>, sample_rate: u32) -> Self {
        Self {
            samples,
            sample_rate,
        }
    }
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

    #[test]
    fn facades_delegate_to_injected_engines() {
        use crate::engines::{AsrEngine, TtsEngine};
        use std::sync::Arc;

        struct DummyAsr;
        impl AsrEngine for DummyAsr {
            fn transcribe(&self, _pcm: &[f32]) -> Result<Transcription> {
                Ok(Transcription::new("delegated"))
            }
        }
        struct DummyTts;
        impl TtsEngine for DummyTts {
            fn synthesize(&self, req: &SynthesisRequest) -> Result<SynthesizedAudio> {
                // Echo the deterministic flag length so the test can observe
                // the request reached the engine.
                let n = if req.deterministic { 2 } else { 1 };
                Ok(SynthesizedAudio::new(vec![0.0; n], 22_050))
            }
        }

        let (_file, session) = session("delegate");
        let session = session
            .with_asr_engine(Arc::new(DummyAsr))
            .with_tts_engine(Arc::new(DummyTts));

        assert_eq!(
            session.asr().transcribe(&[0.0; 160]).unwrap().text,
            "delegated"
        );
        assert_eq!(session.tts().synthesize("hi").unwrap().sample_rate, 22_050);
        assert_eq!(
            session
                .tts()
                .synthesize_request(&SynthesisRequest::new("hi").deterministic())
                .unwrap()
                .samples
                .len(),
            2
        );
    }
}
