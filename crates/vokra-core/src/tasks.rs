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
//! - S2S: **M4-05** (Sesame CSM-1B, v1.0-rc window) — the facade delegates
//!   to the injected [`S2sEngine`](crate::engines::S2sEngine).

use crate::engines::{DialogRequest, SynthesisRequest};
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
/// Wired in **M4-05** (v1.0-rc window — the pre-M4 "v1.0+" label was the
/// old v-label scheme): delegates to the
/// [`S2sEngine`](crate::engines::S2sEngine) injected via
/// [`Session::with_s2s_engine`](crate::Session::with_s2s_engine)
/// (Sesame CSM-1B = M4-05; Moshi = M4-06). Without an injected engine it
/// returns [`VokraError::NotImplemented`].
#[derive(Debug)]
pub struct S2s<'a> {
    session: &'a Session,
}

impl S2s<'_> {
    /// Runs one speech-to-speech dialog turn over mono `f32` PCM input —
    /// the FR-API-02 verbatim shape.
    ///
    /// The samples become [`DialogRequest::input_audio`] with every other
    /// field at its default — in particular `reply_text` is **empty**, and
    /// an engine that requires caller-supplied reply text (CSM — ADR
    /// M4-05 §D1-(b): the model does not generate text) rejects that with
    /// a loud [`VokraError::InvalidArgument`] telling the caller to use
    /// [`dialog_request`](Self::dialog_request). Engines that derive their
    /// own reply (Moshi inner monologue = M4-06) can accept it.
    pub fn dialog(&self, samples: &[f32]) -> Result<DialogTurn> {
        self.dialog_request(&DialogRequest::new("").with_input_audio(samples.to_vec()))
    }

    /// Runs one dialog turn for an explicit [`DialogRequest`] (context
    /// turns, reply text, speaker, determinism — the M4-05 surface).
    pub fn dialog_request(&self, request: &DialogRequest) -> Result<DialogTurn> {
        match self.session.s2s_engine() {
            Some(engine) => engine.dialog(request),
            None => Err(VokraError::NotImplemented(
                "no S2S engine injected (Sesame CSM-1B = M4-05, v1.0-rc window)",
            )),
        }
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

/// Result of [`S2s::dialog`] / [`S2s::dialog_request`] (M4-05).
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct DialogTurn {
    /// Text form of the reply. For CSM this echoes the caller-supplied
    /// [`DialogRequest::reply_text`] verbatim (the model does not generate
    /// text — ADR M4-05 §D1-(b); Moshi's inner monologue is the M4-06
    /// producer of engine-derived text).
    pub text: String,
    /// Synthesized reply audio, when produced.
    pub audio: Option<SynthesizedAudio>,
}

impl DialogTurn {
    /// Constructs a turn (engine crates build this `#[non_exhaustive]`
    /// type across the crate boundary — the [`Transcription::new`]
    /// pattern).
    pub fn new(text: impl Into<String>, audio: Option<SynthesizedAudio>) -> Self {
        Self {
            text: text.into(),
            audio,
        }
    }
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
    fn s2s_facade_has_fr_api_02_shape_and_errors_without_an_engine() {
        let (_file, session) = session("s2s");
        let result = session.s2s().dialog(&[0.0f32; 160]);
        assert!(matches!(result, Err(VokraError::NotImplemented(_))));
        let result = session
            .s2s()
            .dialog_request(&crate::engines::DialogRequest::new("hi"));
        assert!(matches!(result, Err(VokraError::NotImplemented(_))));
    }

    #[test]
    fn s2s_facade_delegates_to_an_injected_engine() {
        use crate::engines::S2sEngine;
        use std::sync::Arc;

        struct EchoS2s;
        impl S2sEngine for EchoS2s {
            fn dialog(&self, request: &DialogRequest) -> Result<DialogTurn> {
                Ok(DialogTurn::new(
                    request.reply_text.clone(),
                    Some(SynthesizedAudio::new(
                        vec![0.0; request.input_audio.as_ref().map_or(1, Vec::len)],
                        24_000,
                    )),
                ))
            }
        }

        let (_file, session) = session("s2s-delegate");
        let session = session.with_s2s_engine(Arc::new(EchoS2s));
        let turn = session
            .s2s()
            .dialog_request(&DialogRequest::new("hello").with_reply_speaker(1))
            .unwrap();
        assert_eq!(turn.text, "hello");
        assert_eq!(turn.audio.unwrap().sample_rate, 24_000);
        // The plain-PCM shape maps samples into input_audio.
        let turn = session.s2s().dialog(&[0.0f32; 160]).unwrap();
        assert_eq!(turn.audio.unwrap().samples.len(), 160);
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
