//! Pluggable inference-engine injection points for the task facades.
//!
//! The concrete model implementations live in `vokra-models` (Silero VAD =
//! M0-05, Whisper base = M0-06, piper-plus native TTS = M0-07). To keep
//! `vokra-core` free of any model/graph specifics — and free of external
//! dependencies (NFR-DS-02) — the models are injected into a [`Session`] as
//! trait objects through these interfaces.
//!
//! The task facades ([`crate::tasks`]) delegate to the injected engine when
//! present and otherwise return [`VokraError::NotImplemented`](crate::VokraError).
//! Engines are attached at build time with
//! [`Session::with_asr_engine`](crate::Session::with_asr_engine),
//! [`Session::with_tts_engine`](crate::Session::with_tts_engine) and
//! [`Session::with_vad_engine`](crate::Session::with_vad_engine) (M0-07-T10
//! for TTS; the ASR / VAD injection points are the M0-06 / M0-05 counterparts).

use crate::error::Result;
use crate::tasks::{SynthesizedAudio, Transcription};

/// A speech-to-text engine (implemented natively in `vokra-models`, e.g.
/// Whisper base = M0-06).
pub trait AsrEngine: Send + Sync {
    /// Transcribes mono `f32` PCM (typically 16 kHz) to text.
    fn transcribe(&self, pcm: &[f32]) -> Result<Transcription>;
}

/// A text-to-speech engine (implemented natively in `vokra-models`, e.g.
/// piper-plus MB-iSTFT-VITS2 = M0-07).
pub trait TtsEngine: Send + Sync {
    /// Synthesizes speech audio for `request`.
    fn synthesize(&self, request: &SynthesisRequest) -> Result<SynthesizedAudio>;
}

/// A voice-activity-detection engine (implemented natively in `vokra-models`,
/// e.g. Silero VAD v5 = M0-05).
///
/// VAD is inherently streaming: each engine hands out a stateful
/// [`VadStreamHandle`] that carries the recurrent state (LSTM `h`/`c`, the
/// carried context samples and the pseudo-STFT) hidden inside it (FR-LD-06).
pub trait VadEngine: Send + Sync {
    /// Opens a fresh streaming handle with zero-initialised recurrent state.
    fn open_stream(&self) -> Box<dyn VadStreamHandle + Send>;
}

/// A stateful VAD stream: push PCM, get per-frame speech probabilities.
///
/// The handle hides all recurrent state (FR-LD-06); callers only push samples
/// and read probabilities. [`reset`](Self::reset) returns it to the initial
/// state so a fresh utterance reproduces the first run bit-for-bit.
pub trait VadStreamHandle {
    /// Pushes PCM at `sample_rate` Hz (8 kHz or 16 kHz) and returns the speech
    /// probability of each fixed-size frame that completed.
    fn push_pcm(&mut self, pcm: &[f32], sample_rate: u32) -> Result<Vec<f32>>;

    /// Clears the recurrent state, returning the handle to its initial state.
    fn reset(&mut self);
}

/// Inputs to [`TtsEngine::synthesize`].
///
/// Carries the text plus an optional language hint, a determinism knob used by
/// parity tests (M0-07-T20: fix the VITS noise so the native output matches the
/// piper-plus reference) and the zero-shot conditioning inputs the v7 voice
/// accepts — an external speaker embedding and per-phoneme prosody features
/// (M1). All conditioning fields are optional and default to the voice's
/// zero-shot defaults; the voice itself comes from the loaded GGUF. Fields grow
/// under `#[non_exhaustive]`.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct SynthesisRequest {
    /// Text to synthesize (already normalized by the caller if needed).
    pub text: String,
    /// Optional language tag (e.g. `"ja"`, `"en"`); `None` = the voice default
    /// (language id 0). The engine maps it to the voice's language id.
    pub language: Option<String>,
    /// When `true`, disable stochastic components (noise scale → 0) so the
    /// output is deterministic for reference parity (M0-07-T20).
    pub deterministic: bool,
    /// Optional external zero-shot **speaker embedding** (`speaker_embedding_dim`
    /// floats — 192 for the v7 voice). `None` uses the zero vector, the
    /// deterministic zero-shot default; note the voice's speaker projection maps
    /// even a zero embedding to a non-zero conditioning contribution
    /// (bias / LayerNorm / GELU). A wrong-length vector is treated as zeros.
    pub speaker_embedding: Option<Vec<f32>>,
    /// Optional per-phoneme **prosody features** — one `(A1, A2, A3)` accent
    /// triple per phoneme (piper-plus JA path). `None`, or any non-JA language,
    /// leaves the prosody projection at its bias. When present the length must
    /// match the phoneme count the engine's tokenizer / phonemizer produces, or
    /// synthesis fails with a clear error.
    pub prosody_features: Option<Vec<[i64; 3]>>,
}

impl SynthesisRequest {
    /// A request for `text` with the voice defaults (non-deterministic, no
    /// explicit language, zero-shot conditioning defaults).
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            language: None,
            deterministic: false,
            speaker_embedding: None,
            prosody_features: None,
        }
    }

    /// Sets the language hint.
    #[must_use]
    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = Some(language.into());
        self
    }

    /// Forces deterministic synthesis (noise disabled) for parity comparisons.
    #[must_use]
    pub fn deterministic(mut self) -> Self {
        self.deterministic = true;
        self
    }

    /// Sets the external zero-shot speaker embedding (`speaker_embedding_dim`
    /// floats; the voice's zero vector is used when unset).
    #[must_use]
    pub fn with_speaker_embedding(mut self, embedding: impl Into<Vec<f32>>) -> Self {
        self.speaker_embedding = Some(embedding.into());
        self
    }

    /// Sets the per-phoneme prosody features — one `(A1, A2, A3)` accent triple
    /// per phoneme, honoured only for the JA language of a prosody-aware voice.
    #[must_use]
    pub fn with_prosody_features(mut self, features: impl Into<Vec<[i64; 3]>>) -> Self {
        self.prosody_features = Some(features.into());
        self
    }
}
