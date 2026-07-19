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

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::error::Result;
use crate::tasks::{DialogTurn, SynthesizedAudio, Transcription};

/// A speech-to-text engine (implemented natively in `vokra-models`, e.g.
/// Whisper base = M0-06).
pub trait AsrEngine: Send + Sync {
    /// Transcribes mono `f32` PCM (typically 16 kHz) to text.
    fn transcribe(&self, pcm: &[f32]) -> Result<Transcription>;
}

/// A speech-to-speech dialog engine (implemented natively in
/// `vokra-models` — Sesame CSM-1B = M4-05; Moshi = M4-06).
///
/// The trait mirrors [`TtsEngine`]'s minimal shape: one blocking
/// `dialog` over an explicit [`DialogRequest`]. Streaming handles are
/// engine-specific surfaces (the CSM streaming session lives in
/// `vokra-models::csm` and rides the M1 SPSC ring + M3-14
/// [`crate::stream::Stream`] interrupt); the trait deliberately does not
/// force a streaming shape onto engines that batch.
pub trait S2sEngine: Send + Sync {
    /// Runs one dialog turn.
    fn dialog(&self, request: &DialogRequest) -> Result<DialogTurn>;
}

/// A **full-duplex** speech-to-speech engine (Moshi = M4-06, FR-MD-09):
/// continuous, simultaneous audio in both directions over one session —
/// unlike the turn-based [`S2sEngine::dialog`].
///
/// The `Arc<Self>` receiver keeps the trait object-safe while letting the
/// returned handle own everything it needs (`'static` — the C ABI holds
/// handles across calls). Injected with
/// [`Session::with_s2s_duplex_engine`](crate::Session::with_s2s_duplex_engine);
/// the facade entry is [`S2s::duplex`](crate::tasks::S2s::duplex).
pub trait S2sDuplexEngine: Send + Sync {
    /// Opens a full-duplex session (mic → model → speaker pipeline).
    ///
    /// Engines must honor the [`DuplexSessionConfig`] echo contract:
    /// without [`DuplexSessionConfig::aec_disabled_explicitly`] a session
    /// whose acoustic-echo canceller is not wired is a **loud error**
    /// (FR-OP-60 / FR-EX-08 — AEC 無しの Moshi/CSM は自己エコーで即崩壊,
    /// CLAUDE.md レビュアー C 指摘 #3), and the explicit opt-in must leave
    /// an observable warning on the handle — never a silent skip.
    fn open_duplex(
        self: Arc<Self>,
        config: &DuplexSessionConfig,
    ) -> Result<Box<dyn S2sDuplexHandle + Send>>;
}

/// A live full-duplex session: push mic frames and pull model frames
/// continuously (wall-clock-free — file-driven tests and real-time
/// callers use the same API; M4-06-T16).
pub trait S2sDuplexHandle {
    /// Feeds one mic frame (`frame_hop()` mono samples at
    /// [`Self::sample_rate`]). Runs the input front (AEC unless
    /// explicitly disabled) and one model step; the returned report makes
    /// every degraded mode visible (FR-EX-08).
    fn push_mic_frame(&mut self, pcm: &[f32]) -> Result<DuplexPushReport>;

    /// Pops the next model frame for playback (`None` = nothing pending
    /// — e.g. during the model's delay warmup or after an interrupt
    /// flush). Pulling *is* the playback hand-off: the engine stamps the
    /// frame into its echo-reference queue at this moment.
    fn pull_model_frame(&mut self) -> Result<Option<Vec<f32>>>;

    /// The inner monologue accumulated so far (Moshi's self-generated
    /// transcript; display-rule filtered — M4-06-T14). Engines without a
    /// text stream return an empty string.
    fn monologue_text(&self) -> Result<String>;

    /// A cross-thread barge-in handle (M3-14 semantics: set the flag from
    /// any thread; the session flushes pending model output and resets
    /// its generation state at the next push/pull boundary, then clears
    /// the flag — mic intake continues).
    fn interrupt_handle(&self) -> DuplexInterruptHandle;

    /// Construction-time warnings (e.g. the explicit AEC opt-out). Empty
    /// on a default (AEC-enabled) session.
    fn warnings(&self) -> &[String];

    /// Mono samples per push/pull frame.
    fn frame_hop(&self) -> usize;

    /// PCM sample rate (Hz) of both directions.
    fn sample_rate(&self) -> u32;
}

/// Per-push observability for [`S2sDuplexHandle::push_mic_frame`]
/// (FR-EX-08: degraded modes are visible, never silent). Constructed by
/// engines (`vokra-models`), so the struct is exhaustive by design — new
/// fields are a semver-visible engine-contract change.
#[derive(Debug, Clone, Copy)]
pub struct DuplexPushReport {
    /// `true` once the model emitted a frame for this push (post-warmup).
    pub step_emitted: bool,
    /// `true` when the AEC actually ran on this frame.
    pub aec_applied: bool,
    /// RMS of the raw mic frame (echo-cancellation observability).
    pub raw_rms: f32,
    /// RMS of the frame after the input front (== `raw_rms` on the
    /// explicit bypass).
    pub cleaned_rms: f32,
}

/// Cross-thread duplex barge-in flag (`Arc<AtomicBool>` — the M3-14
/// [`crate::stream::InterruptHandle`] contract mirrored for duplex
/// sessions; M4-06-T18).
#[derive(Debug, Clone)]
pub struct DuplexInterruptHandle {
    flag: Arc<AtomicBool>,
}

impl DuplexInterruptHandle {
    /// Wraps a shared flag (engine-side constructor).
    #[must_use]
    pub fn new(flag: Arc<AtomicBool>) -> Self {
        Self { flag }
    }

    /// Requests barge-in: the session flushes and resets at its next
    /// push/pull boundary (set-then-handle; Release ordering).
    pub fn interrupt(&self) {
        self.flag.store(true, Ordering::Release);
    }

    /// Whether an interrupt is pending (not yet acknowledged).
    #[must_use]
    pub fn is_pending(&self) -> bool {
        self.flag.load(Ordering::Acquire)
    }
}

/// Options for [`S2sDuplexEngine::open_duplex`]. Engine-specific knobs
/// (AEC filter shape, queue capacity, ...) belong to engine
/// construction; this carries only the session-generic contract.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct DuplexSessionConfig {
    /// `true` → both decode channels sample greedily (reproducible —
    /// parity / demo anchor).
    pub deterministic: bool,
    /// Stochastic sampling seed (ignored when `deterministic`).
    pub seed: u64,
    /// **Explicit** opt-out of echo cancellation (recorded-input /
    /// loopback-free rigs only). Defaults to `false`; setting it makes
    /// the engine record a loud warning on the handle instead of
    /// silently skipping the canceller (FR-EX-08).
    pub aec_disabled_explicitly: bool,
    /// Playback-latency compensation added to the echo-reference clock
    /// when frames are pulled (owner-tunable on real hardware —
    /// M4-06-T17).
    pub playback_offset_samples: u64,
}

impl DuplexSessionConfig {
    /// The default (AEC-required, stochastic) config.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Forces deterministic (greedy) sampling.
    #[must_use]
    pub fn deterministic(mut self) -> Self {
        self.deterministic = true;
        self
    }

    /// Sets the stochastic seed.
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// **Explicitly** disables the echo canceller — only for inputs with
    /// no acoustic echo path (recorded files). The engine keeps a loud
    /// warning on the handle; there is no silent variant of this switch
    /// (FR-EX-08 / FR-OP-60).
    #[must_use]
    pub fn with_aec_disabled_explicitly(mut self) -> Self {
        self.aec_disabled_explicitly = true;
        self
    }

    /// Sets the playback-latency compensation (samples).
    #[must_use]
    pub fn with_playback_offset_samples(mut self, samples: u64) -> Self {
        self.playback_offset_samples = samples;
        self
    }
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

/// One prior turn of dialog context an [`S2sEngine`] conditions on.
///
/// CSM-1B conditions on interleaved text + audio segments per speaker
/// (ADR M4-05 §D2 `Segment`); either side may be absent for a turn the
/// caller only has one modality for.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct DialogContextTurn {
    /// Speaker id (model-defined numbering; CSM uses small integers).
    pub speaker: u32,
    /// The turn's text, when known.
    pub text: Option<String>,
    /// The turn's audio (mono PCM at the engine's sample rate), when known.
    pub audio: Option<Vec<f32>>,
}

impl DialogContextTurn {
    /// A text-only context turn.
    pub fn text(speaker: u32, text: impl Into<String>) -> Self {
        Self {
            speaker,
            text: Some(text.into()),
            audio: None,
        }
    }

    /// An audio-only context turn.
    pub fn audio(speaker: u32, audio: Vec<f32>) -> Self {
        Self {
            speaker,
            text: None,
            audio: Some(audio),
        }
    }
}

/// Inputs to [`S2sEngine::dialog`].
///
/// # The `reply_text` contract (ADR M4-05 §D1-(b))
///
/// CSM-1B is a **speech generation** model conditioned on dialog context —
/// it does not run ASR and does not generate reply text. `reply_text` is
/// therefore **caller-supplied** (an upstream text LLM or a human); the
/// engine speaks it in context and echoes it back in
/// [`DialogTurn::text`](crate::tasks::DialogTurn). An engine that cannot
/// proceed without it rejects an empty `reply_text` with a loud
/// [`crate::VokraError::InvalidArgument`] — never a silent empty reply
/// (FR-EX-08).
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct DialogRequest {
    /// Prior turns, oldest first.
    pub context: Vec<DialogContextTurn>,
    /// The text the engine speaks this turn (caller-supplied — see the
    /// struct docs).
    pub reply_text: String,
    /// Speaker id the reply is voiced as.
    pub reply_speaker: u32,
    /// The current incoming utterance (mono PCM at the engine's sample
    /// rate) — for CSM this is the mic audio *after* the AEC front (or an
    /// explicitly bypassed recorded file — `vokra-models::csm::EchoPath`).
    pub input_audio: Option<Vec<f32>>,
    /// When `true`, the engine samples with temperature 0 (or its
    /// documented deterministic mode) so the turn is reproducible for
    /// parity / quality gates.
    pub deterministic: bool,
    /// Sampling seed for the stochastic mode (ignored when
    /// `deterministic`).
    pub seed: u64,
    /// Cap on generated audio frames (`None` = the engine default).
    pub max_frames: Option<usize>,
}

impl DialogRequest {
    /// A request speaking `reply_text` as speaker 0 with no context.
    pub fn new(reply_text: impl Into<String>) -> Self {
        Self {
            context: Vec::new(),
            reply_text: reply_text.into(),
            reply_speaker: 0,
            input_audio: None,
            deterministic: false,
            seed: 0,
            max_frames: None,
        }
    }

    /// Appends a context turn.
    #[must_use]
    pub fn with_context_turn(mut self, turn: DialogContextTurn) -> Self {
        self.context.push(turn);
        self
    }

    /// Sets the reply speaker id.
    #[must_use]
    pub fn with_reply_speaker(mut self, speaker: u32) -> Self {
        self.reply_speaker = speaker;
        self
    }

    /// Attaches the current incoming utterance.
    #[must_use]
    pub fn with_input_audio(mut self, pcm: Vec<f32>) -> Self {
        self.input_audio = Some(pcm);
        self
    }

    /// Forces the deterministic sampling mode.
    #[must_use]
    pub fn deterministic(mut self) -> Self {
        self.deterministic = true;
        self
    }

    /// Sets the stochastic sampling seed.
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Caps the generated frame count.
    #[must_use]
    pub fn with_max_frames(mut self, max_frames: usize) -> Self {
        self.max_frames = Some(max_frames);
        self
    }
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
