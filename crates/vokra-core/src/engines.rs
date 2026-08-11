//! Pluggable inference-engine injection points for the task facades.
//!
//! The concrete model implementations live in `vokra-models` (Silero VAD =
//! M0-05, Whisper base = M0-06, piper-plus native TTS = M0-07). To keep
//! `vokra-core` free of any model/graph specifics — and free of external
//! dependencies (NFR-DS-02) — the models are injected into a [`crate::Session`] as
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

use crate::error::{Result, VokraError};
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

/// A keyword-spotting / wake-word engine (implemented natively in
/// `vokra-models`, e.g. openWakeWord = 2026-08-05).
///
/// KWS is inherently streaming: the model consumes a rolling embedding
/// window (~775 ms at 16 kHz for openWakeWord) and emits one
/// probability per wake-word every ~80 ms hop, so the trait exposes a
/// single **stateful** `push_pcm16k` that carries the internal window
/// buffer + per-wake-word rolling probabilities across calls. Callers
/// only push 16 kHz mono PCM and receive `(wakeword_name, probability)`
/// pairs for the wake-words whose latest score is fresh this push.
///
/// The trait deliberately hides the two-stage internal architecture
/// (mel front-end → shared embedding extractor → per-wake-word
/// classifier MLP) — a change to that architecture is a
/// `vokra_models::kws` implementation detail, not an engine-contract
/// break.
pub trait KwsEngine: Send + Sync {
    /// Names of the wake-words this engine can recognise, in the order
    /// [`push_pcm16k`](Self::push_pcm16k) reports them. Names are the
    /// upstream openWakeWord model names (e.g. `"alexa"`, `"hey_jarvis"`,
    /// `"hey_mycroft"`) transcribed verbatim into the GGUF's
    /// `vokra.openwakeword.wakeword_names` array chunk.
    fn wakeword_names(&self) -> &[String];

    /// Pushes 16 kHz mono `f32` PCM and returns the wake-word
    /// probabilities that completed on this push. Each entry is
    /// `(wakeword_name, probability ∈ [0, 1])`.
    ///
    /// The engine buffers samples internally; `push_pcm16k` may return
    /// an empty vector when the rolling embedding window has not yet
    /// filled, and multiple entries when several 80 ms hops complete on
    /// one push. Callers threshold the returned probabilities
    /// downstream (typically at `0.5`).
    fn push_pcm16k(&mut self, samples: &[f32]) -> Result<Vec<(String, f32)>>;
}

/// A text-to-speech engine (implemented natively in `vokra-models`, e.g.
/// piper-plus MB-iSTFT-VITS2 = M0-07).
///
/// # Capability discovery (WP-23)
///
/// The three defaulted helpers below let a caller — most notably the
/// cross-engine [`SynthesisRequest`] adapter path — discover, at runtime,
/// whether an engine can honor the *optional* [`SynthesisRequest::style_vec`]
/// / [`SynthesisRequest::speaker_id`] fields *before* it constructs a
/// request that carries them. An engine that ignores the fields keeps the
/// defaults (`supports_style_vec() = supports_multi_speaker() = false`);
/// an engine that reads them overrides the corresponding method to return
/// `true`. The unified request shape stays the same across engines — only
/// the capability advertisement (and the internal reject-loudly-if-set
/// path an engine chooses on the not-supported branch, per FR-EX-08)
/// differs.
///
/// [`synthesize_stream`](Self::synthesize_stream) is the placeholder
/// streaming-synthesis entry (real streaming lands in a separate WP —
/// piper-plus M4-03 / SBV2 streaming). Its default impl loudly returns
/// [`VokraError::UnsupportedOp`] so a caller can probe capability without
/// silently getting a synchronous full-utterance render disguised as a
/// stream.
pub trait TtsEngine: Send + Sync {
    /// Synthesizes speech audio for `request`.
    fn synthesize(&self, request: &SynthesisRequest) -> Result<SynthesizedAudio>;

    /// Returns `true` iff [`synthesize`](Self::synthesize) reads
    /// [`SynthesisRequest::style_vec`] as the AdaIN-style per-utterance
    /// style conditioning vector (SBV2 / VITS2-family), rather than
    /// silently discarding it. Default: `false`.
    ///
    /// An engine that returns `true` must (i) accept a well-shaped
    /// `style_vec` and thread it into its own native request shape and
    /// (ii) loudly reject a wrong-length vector with
    /// [`VokraError::InvalidArgument`] (FR-EX-08 — no silent shape
    /// coercion). An engine that returns `false` should either ignore
    /// `style_vec = None` (the common case) or loudly reject
    /// `style_vec = Some(_)` when the engine cannot honor it.
    fn supports_style_vec(&self) -> bool {
        false
    }

    /// Returns `true` iff [`synthesize`](Self::synthesize) reads
    /// [`SynthesisRequest::speaker_id`] as a discrete speaker-table
    /// index (SBV2 / VITS-family multi-speaker voices), rather than
    /// silently discarding it. Default: `false`.
    ///
    /// An engine that returns `true` must thread the id into its own
    /// native request shape and let its speaker-table lookup validate
    /// the id (out-of-range → [`VokraError::InvalidArgument`], FR-EX-08).
    /// A single-speaker engine keeps the default `false`.
    fn supports_multi_speaker(&self) -> bool {
        false
    }

    /// Opens an incremental TTS stream for `request` (placeholder — WP-23).
    ///
    /// Real streaming synthesis is a separate WP (piper-plus streaming
    /// M4-03 / SBV2 streaming) and no engine returns `Ok(..)` from this
    /// default impl today — [`VokraError::UnsupportedOp`] is the loud
    /// probe answer, never a silent full-utterance synthesis disguised
    /// as a stream (FR-EX-08). An engine that lands real streaming
    /// later overrides this method and returns a
    /// `Box<dyn TtsStreamHandle + Send>`; callers pull PCM chunks via
    /// [`TtsStreamHandle::next_pcm_chunk`].
    fn synthesize_stream(
        &self,
        _request: &SynthesisRequest,
    ) -> Result<Box<dyn TtsStreamHandle + Send>> {
        Err(VokraError::UnsupportedOp(
            "TtsEngine::synthesize_stream: incremental streaming synthesis is a separate WP \
             (piper-plus M4-03 / SBV2 streaming) not yet implemented for this engine — \
             the default trait impl loudly refuses per FR-EX-08 (no silent full-utterance \
             render disguised as a stream)"
                .to_string(),
        ))
    }
}

/// A stateful TTS stream: pull PCM chunks until the utterance completes
/// (WP-23 placeholder).
///
/// No engine returns one from [`TtsEngine::synthesize_stream`] today —
/// the trait exists to pin the incremental-streaming *shape* so a later
/// WP (piper-plus M4-03 / SBV2 streaming) can wire it without a
/// [`TtsEngine`] contract break.
pub trait TtsStreamHandle {
    /// Pulls the next PCM chunk (`Some(chunk)`), or `None` once the
    /// utterance is fully synthesized.
    fn next_pcm_chunk(&mut self) -> Result<Option<Vec<f32>>>;

    /// PCM sample rate of the emitted chunks (Hz).
    fn sample_rate(&self) -> u32;
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

/// An **acoustic echo cancellation** engine (neural side of the audio
/// dialect §"Speech Enhancement / AGC / AEC" — implemented natively in
/// `vokra-models`, e.g. NKF-AEC = 2026-08-05).
///
/// AEC is inherently paired-streaming: each engine hands out a stateful
/// [`AecStreamHandle`] that carries every recurrent state (per-bin
/// Kalman filter taps, GRU hidden vectors, iSTFT overlap-add tail,
/// pending PCM residues) hidden inside it (FR-LD-06). The engine
/// consumes sample-aligned mic + far-end PCM and emits echo-cancelled
/// PCM.
///
/// Orthogonal to the algorithmic `vokra_ops::aec` path (M4-03 SpeexDSP
/// / WebRTC AEC3 Rust port surfaced through the `vokra_aec_*` C ABI) —
/// both live side-by-side; a duplex engine (Moshi / CSM) can choose
/// either through this trait.
pub trait AecEngine: Send + Sync {
    /// Opens a fresh streaming handle bound to `sample_rate` Hz with
    /// zero-initialised recurrent state.
    ///
    /// # Errors
    ///
    /// Returns [`VokraError::InvalidArgument`](crate::VokraError) when
    /// `sample_rate` does not match the model's trained rate — the
    /// engine never silently resamples (FR-EX-08).
    fn open_stream(&self, sample_rate: u32) -> Result<Box<dyn AecStreamHandle + Send>>;
}

/// A stateful AEC stream: push paired mic + far-end PCM, get
/// echo-cancelled PCM back.
///
/// The handle hides every recurrent state (FR-LD-06); callers only push
/// sample-aligned mic + far-end samples and read the cleaned output.
/// [`reset`](Self::reset) returns it to the initial state so a fresh
/// utterance reproduces the first run bit-for-bit.
pub trait AecStreamHandle {
    /// Pushes sample-aligned mic + far-end mono `f32` PCM at the
    /// stream's bound sample rate and returns whatever cleaned samples
    /// the internal STFT + Kalman + iSTFT pipeline can commit on this
    /// push. The returned slice length depends on the hop / overlap-add
    /// tail geometry; a starving call may return an empty vec.
    ///
    /// # Errors
    ///
    /// Returns [`VokraError::InvalidArgument`](crate::VokraError) when
    /// `mic.len() != farend.len()` — the two streams are strictly
    /// sample-aligned in AEC (silent trim / repeat is a correctness bug,
    /// not a convenience — FR-EX-08).
    fn push_paired(&mut self, mic: &[f32], farend: &[f32]) -> Result<Vec<f32>>;

    /// Clears every recurrent state, returning the handle to its
    /// initial state.
    fn reset(&mut self);
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

/// A speech-enhancement / denoise engine (implemented natively in
/// `vokra-models`, e.g. Xiph RNNoise v0.2 = 2026-08-05, follow-up
/// DeepFilterNet3 wrapper for the algorithmic
/// `vokra_ops::denoise::denoise` fn).
///
/// Denoise is inherently streaming: the engine consumes one mono PCM
/// stream and emits an enhanced PCM stream on the same clock. Each
/// engine hands out a stateful [`DenoiseStreamHandle`] that carries
/// every recurrent state (per-block GRU hidden vectors, STFT tail,
/// pitch analysis buffer, prev-frame Bark energies) hidden inside it
/// (FR-LD-06).
///
/// This trait absorbs both the neural RNNoise session (real GRU
/// weights) and a future thin wrapper around the existing
/// `vokra_ops::denoise` fn (DeepFilterNet3) so the dispatch layer
/// sees one shape.
pub trait DenoiseEngine: Send + Sync {
    /// Opens a fresh streaming handle with zero-initialised recurrent
    /// state, bound to `sample_rate` Hz.
    ///
    /// # Errors
    ///
    /// Returns [`VokraError::InvalidArgument`](crate::VokraError) when
    /// `sample_rate` does not match the model's trained rate — the
    /// engine never silently resamples (FR-EX-08).
    fn open_stream(&self, sample_rate: u32) -> Result<Box<dyn DenoiseStreamHandle + Send>>;
}

/// A stateful denoise stream: push mono PCM, get enhanced PCM back.
///
/// The handle hides every recurrent state (FR-LD-06); callers only push
/// samples and read the cleaned output. [`reset`](Self::reset) returns
/// it to the initial state so a fresh utterance reproduces the first
/// run bit-for-bit.
pub trait DenoiseStreamHandle {
    /// Pushes mono `f32` PCM at the stream's bound sample rate and
    /// returns whatever enhanced samples the internal STFT + iSTFT
    /// pipeline can commit on this push. The returned slice length
    /// depends on the hop / overlap-add tail geometry; a starving call
    /// may return an empty vec.
    fn push_pcm(&mut self, pcm: &[f32]) -> Result<Vec<f32>>;

    /// Clears every recurrent state, returning the handle to its
    /// initial state.
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
    /// (bias / LayerNorm / GELU).
    ///
    /// **Per-engine wrong-length policy** — this field crosses the cross-engine
    /// [`TtsEngine`] boundary, so a `Some(..)` whose length differs from the
    /// loaded voice's `speaker_embedding_dim` is handled per-engine, not
    /// centrally:
    /// - `piper_plus::PiperPlusTts` falls back to the zero vector (its
    ///   documented zero-shot default; see `piper_plus::conditioning::g`).
    /// - `sbv2::SbV2Model`'s [`TtsEngine`] adapter returns
    ///   [`VokraError::InvalidArgument`] for any `Some(..)` (SBV2 selects
    ///   speakers through a discrete id, not a continuous embedding, so
    ///   honoring the field at all would silently discard caller data — FR-EX-08).
    ///
    /// Callers wanting portable behavior should either supply a correctly-sized
    /// vector or leave the field `None`.
    pub speaker_embedding: Option<Vec<f32>>,
    /// Optional per-phoneme **prosody features** — one `(A1, A2, A3)` accent
    /// triple per phoneme (piper-plus JA path). `None`, or any non-JA language,
    /// leaves the prosody projection at its bias. When present the length must
    /// match the phoneme count the engine's tokenizer / phonemizer produces, or
    /// synthesis fails with a clear error.
    pub prosody_features: Option<Vec<[i64; 3]>>,
    /// Optional per-utterance **AdaIN-style style-vector** (SBV2 / VITS2
    /// style conditioning — WP-23). `None` = the engine's zero-shot
    /// default (typically an all-zero vector sized from the loaded
    /// voice's style width, which is the identity injection).
    ///
    /// An engine that advertises
    /// [`TtsEngine::supports_style_vec`]`() == true` (SBV2) reads this,
    /// validates its length matches the loaded voice's style width, and
    /// loudly rejects a wrong-length vector with
    /// [`crate::VokraError::InvalidArgument`] (FR-EX-08 — no silent
    /// truncate / zero-pad). An engine that advertises `false` (default)
    /// either ignores `None` and errors on `Some(_)`, or silently ignores
    /// the field — it MUST NOT pretend the style was applied.
    pub style_vec: Option<Vec<f32>>,
    /// Optional discrete **speaker id** for multi-speaker voices
    /// (SBV2 / VITS multi-speaker table lookup — WP-23). `None` = the
    /// engine's default speaker (speaker id `0`).
    ///
    /// An engine that advertises
    /// [`TtsEngine::supports_multi_speaker`]`() == true` (SBV2) reads
    /// this and threads it into its speaker-table lookup, which loudly
    /// rejects out-of-range ids with
    /// [`crate::VokraError::InvalidArgument`] (FR-EX-08). A
    /// single-speaker engine (`supports_multi_speaker() == false`) either
    /// ignores `None` and errors on `Some(_)`, or silently ignores the
    /// field — it MUST NOT silently substitute speaker 0 for a nonzero
    /// id.
    pub speaker_id: Option<u32>,
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
            style_vec: None,
            speaker_id: None,
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

    /// Sets the AdaIN-style style-vector (SBV2 / VITS2, WP-23).
    ///
    /// The engine validates the vector's length against the loaded voice's
    /// style width and loudly rejects a mismatch with
    /// [`crate::VokraError::InvalidArgument`] (FR-EX-08). Only engines that
    /// advertise [`TtsEngine::supports_style_vec`]`() == true` read the
    /// field — see that method's doc for the reject-loudly-on-`Some(_)`
    /// contract other engines follow.
    #[must_use]
    pub fn with_style_vec(mut self, style: impl Into<Vec<f32>>) -> Self {
        self.style_vec = Some(style.into());
        self
    }

    /// Sets the discrete speaker id for multi-speaker voices (SBV2 /
    /// VITS multi-speaker, WP-23).
    ///
    /// The engine threads the id into its speaker-table lookup, which
    /// loudly rejects an out-of-range id with
    /// [`crate::VokraError::InvalidArgument`] (FR-EX-08). Only engines
    /// that advertise [`TtsEngine::supports_multi_speaker`]`() == true`
    /// read the field.
    #[must_use]
    pub fn with_speaker_id(mut self, speaker_id: u32) -> Self {
        self.speaker_id = Some(speaker_id);
        self
    }
}

/// A reference-free neural **MOS (Mean Opinion Score) scorer** — the
/// engine-facing surface for MOS predictors like DNSMOS P.808 / P.835
/// (`vokra_models::dnsmos_p808_p835`, 2026-08-05) and the future UTMOS
/// runtime binder.
///
/// Unlike the metrics-side `AudioMosMetric` (in `vokra_eval::metrics`)
/// (which lives in `vokra-eval` and is a single-scalar per-clip abstraction
/// with a stable string `name()`), this trait sits in `vokra-core` alongside
/// the other engine seams so a runtime session (or a C ABI handle) can drive
/// a MOS scorer without pulling `vokra-eval` in. It reports **all four**
/// DNSMOS-style dimensions (P.808 overall + P.835 sig/bak/ovrl) in one call
/// so a caller that binds a bundle GGUF (both variants in one artefact)
/// does not pay the mel front-end cost twice.
pub trait MosScorerEngine: Send + Sync {
    /// The MOS variants this engine can score, in canonical order.
    /// DNSMOS returns some subset of `["p808", "p835"]`; a partial bundle
    /// (only P.808 or only P.835 flattened into the GGUF) advertises only
    /// the truthful subset here so a caller can gate their pipeline
    /// without walking metadata (FR-EX-08).
    fn variants(&self) -> &[&'static str];

    /// Scores a 16 kHz mono `f32` PCM clip. Every field of [`MosScore`] is
    /// `None` unless the engine's variants set includes it — a partial
    /// bundle that only carries P.808 weights returns `p808 = Some(...)`
    /// and `sig = bak = ovrl = None`, never a fabricated `0.0`.
    fn score(&self, pcm16k: &[f32]) -> Result<MosScore>;
}

/// A [`MosScorerEngine`] result. Fields are `None` for variants the
/// engine does not advertise; every advertised variant yields
/// `Some(mos ∈ [1.0, 5.0])`.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct MosScore {
    /// ITU-T P.808 overall quality MOS (single scalar).
    pub p808: Option<f32>,
    /// ITU-T P.835 signal-quality MOS.
    pub sig: Option<f32>,
    /// ITU-T P.835 background-noise MOS.
    pub bak: Option<f32>,
    /// ITU-T P.835 overall MOS.
    pub ovrl: Option<f32>,
}

#[cfg(test)]
mod tts_engine_extension_tests {
    //! WP-23: `SynthesisRequest::style_vec` / `speaker_id` + `TtsEngine`
    //! capability advertisement + `synthesize_stream` placeholder tests.
    //!
    //! These trait-level tests use a spy [`TtsEngine`] implementation that
    //! stores the last received request. They prove the two new
    //! `SynthesisRequest` fields are (a) reachable through the trait
    //! (the spy sees them) and (b) their defaults are `None` (so an
    //! existing caller that never touches the new builders keeps the
    //! pre-WP-23 behavior byte-for-byte). They also prove the trait's
    //! three new defaulted methods behave as documented on an engine
    //! that does not override them (all three defaults hold for a
    //! minimal implementor).

    use super::*;
    use std::sync::Mutex;

    /// A spy engine that stores the last `SynthesisRequest` it saw so the
    /// tests below can assert what threaded through the trait boundary.
    struct SpyTts {
        last: Mutex<Option<SynthesisRequest>>,
    }

    impl SpyTts {
        fn new() -> Self {
            Self {
                last: Mutex::new(None),
            }
        }
        fn take_last(&self) -> Option<SynthesisRequest> {
            self.last.lock().unwrap().take()
        }
    }

    impl TtsEngine for SpyTts {
        fn synthesize(&self, request: &SynthesisRequest) -> Result<SynthesizedAudio> {
            *self.last.lock().unwrap() = Some(request.clone());
            Ok(SynthesizedAudio::new(vec![0.0; 1], 22_050))
        }
    }

    #[test]
    fn new_request_defaults_both_new_fields_to_none() {
        // Backward-compatible default: an existing caller that only touched
        // the pre-WP-23 builders sees `None` for both new optional fields,
        // so downstream engines see exactly the pre-WP-23 shape.
        let req = SynthesisRequest::new("hi");
        assert!(req.style_vec.is_none());
        assert!(req.speaker_id.is_none());
    }

    #[test]
    fn with_style_vec_and_with_speaker_id_thread_through_trait_boundary() {
        // The spy engine sees exactly what the builder set — proving the
        // fields cross the trait boundary intact (not silently dropped by
        // the request's clone / adapter path).
        let engine = SpyTts::new();
        let style = vec![0.1, 0.2, 0.3, 0.4];
        let req = SynthesisRequest::new("hi")
            .with_style_vec(style.clone())
            .with_speaker_id(7);

        engine.synthesize(&req).unwrap();
        let observed = engine
            .take_last()
            .expect("spy engine must have seen the request");

        assert_eq!(observed.style_vec.as_deref(), Some(style.as_slice()));
        assert_eq!(observed.speaker_id, Some(7));
    }

    #[test]
    fn tts_engine_defaults_advertise_no_style_vec_no_multi_speaker() {
        // A minimal implementor keeps the two capability defaults, so an
        // engine author only overrides them when they truthfully honor
        // the corresponding request field.
        let engine = SpyTts::new();
        assert!(!TtsEngine::supports_style_vec(&engine));
        assert!(!TtsEngine::supports_multi_speaker(&engine));
    }

    #[test]
    fn tts_engine_synthesize_stream_default_is_loud_unsupported_op() {
        // The default streaming impl loudly refuses per FR-EX-08 — never a
        // silent full-utterance render disguised as a stream. `Box<dyn
        // TtsStreamHandle + Send>` does not implement `Debug` (the trait
        // deliberately omits it — the WP that lands real streaming
        // decides its handle's debug shape), so `expect_err` is not
        // usable here; matching directly is the shape-neutral way.
        let engine = SpyTts::new();
        let req = SynthesisRequest::new("hi");
        match TtsEngine::synthesize_stream(&engine, &req) {
            Err(VokraError::UnsupportedOp(_)) => {}
            Err(other) => panic!("expected UnsupportedOp, got {other:?}"),
            Ok(_) => panic!("default synthesize_stream must be Err (FR-EX-08)"),
        }
    }
}
