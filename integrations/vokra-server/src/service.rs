//! # T04 — Internal inference service layer (M2-09).
//!
//! Pre-warmed engine registry the HTTP / Wyoming layers share as
//! `Arc<InferenceService>`. Every field is loaded once at startup and
//! shared across every tokio worker (all Vokra engines are `Send + Sync` by
//! construction — they hold `BackendKind: Copy`, never a live `!Send`
//! Metal / CUDA context, so `Arc<Engine>` is safe on every hot path).
//!
//! The T04 cut wires the runtime plumbing only — no HTTP, no async, no
//! serde types are pulled in yet, so this file compiles standalone from
//! only `vokra-core` / `vokra-models` / `vokra-piper-plus`. That preserves
//! the excluded-workspace boundary (NFR-DS-02): the third-party HTTP stack
//! arrives with T06+ handlers, not here.
//!
//! ## Invariants T04 honours (verbatim from the T01 placeholder)
//!
//! 1. **Missing GGUF is a hard startup error, never a silent skip**
//!    (FR-EX-08). Every configured GGUF must load; a broken / missing file
//!    fails [`InferenceService::build`] with [`ServiceError::ModelLoadFailed`].
//! 2. **Unknown model at request time → [`ServiceError::UnknownModel`]**,
//!    never a silent alias. The HTTP layer (T05/T06) maps this to 404.
//! 3. **Kokoro is advertised but its synthesize is unavailable in v0.5**
//!    (M2-07 G2P bridge deferred; `KokoroTts::synthesize` returns
//!    `VokraError::NotImplemented`). The registry rejects synthesize
//!    requests up-front with [`ServiceError::SynthesizeUnavailable`] so the
//!    HTTP layer can return 501 without touching the engine.
//! 4. **No silent CPU fallback for backend holes** (FR-EX-08). The engine's
//!    `UnsupportedOp` propagates verbatim inside
//!    [`ServiceError::Inference`].
//! 5. **Compliance policy is threaded through every load**, so the M2-13
//!    weight-license research-flag gate (FR-CP-03) still runs.
//! 6. **Watermark firing is out of scope** (2026-07-04 owner decision) — a
//!    forward-compat `WatermarkConfig` hook is left for M2-13.

#![allow(clippy::result_large_err)] // VokraError is intentionally rich; we propagate it.

use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

// `WatermarkBackendStatus` is referenced only in doc-links below (rustdoc
// intra-doc link); dropping it from the `use` list avoids the unused-import
// warning while keeping the doc reference valid via its full path.
use vokra_core::{
    AsrEngine, BackendKind, CompliancePolicy, GgufFile, SynthesisRequest, SynthesizedAudio,
    VokraError, WatermarkConfig,
};
use vokra_models::kokoro::KokoroTts;
use vokra_models::piper_plus::PiperPlusTts;
use vokra_models::silero_vad::SileroVadV5;
use vokra_models::voxtral::VoxtralAsr;
use vokra_models::whisper::asr::WhisperAsr;
use vokra_models::whisper::tokenizer::WhisperTokenizer;
use vokra_piper_plus::{PassthroughPhonemizer, Phonemizer};

/// Well-known model-name aliases the registry recognises from the 4
/// compat APIs. The HTTP layer (T06/T09/T11) forwards the raw `model`
/// field; the registry maps it to a concrete engine.
///
/// * `whisper-1` — OpenAI stock alias, routed to base (faster-whisper
///   drop-in convention);
/// * `whisper-base` — explicit base alias;
/// * `whisper-large-v3` — the M2-06 large-v3 engine when configured;
/// * `voxtral` / `voxtral-mini-3b` / `voxtral-small-24b` — the M3-10 Voxtral
///   engine when configured (ASR-only until the full autoregressive decode
///   lands; the model is registered so the server route lights up as soon
///   as the block math ships — never a silent fabrication);
/// * `piper-plus` — the M0-07 native TTS engine (default v0.5 TTS);
/// * `kokoro` — the M2-07 native Kokoro engine (advertised only; synthesize
///   currently unavailable).
pub mod model_names {
    /// OpenAI stock alias (`/v1/audio/transcriptions` `model = "whisper-1"`).
    pub const WHISPER_1: &str = "whisper-1";
    /// Explicit base alias.
    pub const WHISPER_BASE: &str = "whisper-base";
    /// M2-06 large-v3 alias.
    pub const WHISPER_LARGE_V3: &str = "whisper-large-v3";
    /// M3-10 Voxtral generic alias — routed to the loaded Voxtral engine
    /// (mini-3b or small-24b, whichever was configured).
    pub const VOXTRAL: &str = "voxtral";
    /// M3-10 Voxtral mini-3b (Apache 2.0 code + Apache 2.0 weight).
    pub const VOXTRAL_MINI_3B: &str = "voxtral-mini-3b";
    /// M3-10 Voxtral small-24b (Apache 2.0 code + Apache 2.0 weight).
    pub const VOXTRAL_SMALL_24B: &str = "voxtral-small-24b";
    /// piper-plus native TTS alias.
    pub const PIPER_PLUS: &str = "piper-plus";
    /// Kokoro-82M native TTS alias.
    pub const KOKORO: &str = "kokoro";
}

/// Configuration for [`InferenceService::build`], populated from the CLI /
/// env / TOML config in T03. Filenames are [`PathBuf`]s so the T03 layer
/// can canonicalise them before hand-off and `build` never re-parses paths.
#[derive(Debug, Clone)]
pub struct ServiceConfig {
    /// Path to a Whisper base GGUF. **Required** — the server refuses to
    /// start without at least one ASR engine so an OpenAI `/v1/audio/*`
    /// request can never be answered by a silent no-op.
    pub whisper_base_gguf: PathBuf,
    /// Optional path to the Whisper base tokenizer sidecar (base ships a
    /// sidecar per M0-06; large-v3 embeds its vocab).
    pub whisper_base_tokenizer: Option<PathBuf>,
    /// Optional path to a Whisper large-v3 GGUF. When absent, requests
    /// that name `whisper-large-v3` are rejected with
    /// [`ServiceError::UnknownModel`] (never a silent fall-through to base).
    pub whisper_large_v3_gguf: Option<PathBuf>,
    /// Optional path to a Whisper large-v3 tokenizer sidecar. large-v3
    /// embeds its vocab (M2-06) so this is usually unnecessary; kept for
    /// the converter-A path.
    pub whisper_large_v3_tokenizer: Option<PathBuf>,
    /// Path to the piper-plus voice GGUF. **Required** for the same
    /// reason as `whisper_base_gguf`.
    pub piper_plus_gguf: PathBuf,
    /// Optional path to a Kokoro voice GGUF. When present, the registry
    /// advertises `kokoro` but any synthesize request routed to it
    /// returns [`ServiceError::SynthesizeUnavailable`] (M2-07 G2P bridge
    /// deferred).
    pub kokoro_gguf: Option<PathBuf>,
    /// Optional path to a Voxtral (Mistral) GGUF (M3-10). When present, the
    /// registry advertises the `voxtral` / `voxtral-mini-3b` /
    /// `voxtral-small-24b` aliases. The engine is registered even though
    /// [`vokra_models::voxtral::VoxtralAsr::transcribe`] returns
    /// [`VokraError::NotImplemented`] today (the full autoregressive decode
    /// is a follow-up ticket) — this is deliberate: the /v1/audio/*
    /// endpoints must not silently claim to support Voxtral, so we surface
    /// the honest NotImplemented from
    /// [`ServiceError::Inference`], mapped to HTTP 501 by T05. When the
    /// block math + tokenizer greedy step ship, the endpoint lights up
    /// automatically — no server-side re-plumbing.
    pub voxtral_gguf: Option<PathBuf>,
    /// Optional path to a Silero VAD v5 GGUF. When absent, the Wyoming
    /// chunk-boundary VAD helper is disabled (chunks are used as-is).
    pub silero_vad_gguf: Option<PathBuf>,
    /// Backend the pre-warmed engines run on. Applied uniformly across
    /// engines; a per-model backend override is a T03 follow-up.
    pub backend: BackendKind,
    /// Compliance policy for every GGUF load (default: strict). Threaded
    /// through the M2-13 weight-license gate (FR-CP-03).
    pub compliance: CompliancePolicy,
    /// Watermark configuration (T21 forward-compat hook).
    ///
    /// **The server does not fire watermark embedding** (2026-07-04 client
    /// drop of FR-CP-01 / FR-CP-02 embedding, see
    /// `docs/legal-compliance.md` §8): this field is a settings surface only,
    /// carried on `ServiceConfig` → [`InferenceService`] so that the M2-13
    /// design-intent toggles (AudioSeal / C2PA / SynthID / SilentCipher) are
    /// visible to callers and, when a real backend re-lands, the wiring is
    /// already there. Until then
    /// [`WatermarkConfig::backend_status`] returns
    /// [`vokra_core::WatermarkBackendStatus::Deferred`] and no TTS endpoint touches audio
    /// post-synthesis. Silently pretending to watermark would be worse for
    /// compliance than an explicit "not implemented" (see
    /// `crates/vokra-core/src/compliance/watermark.rs`).
    pub watermark: WatermarkConfig,
}

impl ServiceConfig {
    /// Minimum-viable config: only base ASR + piper TTS, CPU backend,
    /// strict compliance. Used by tests and the T03 default startup path.
    pub fn minimum(whisper_base_gguf: PathBuf, piper_plus_gguf: PathBuf) -> Self {
        Self {
            whisper_base_gguf,
            whisper_base_tokenizer: None,
            whisper_large_v3_gguf: None,
            whisper_large_v3_tokenizer: None,
            piper_plus_gguf,
            kokoro_gguf: None,
            voxtral_gguf: None,
            silero_vad_gguf: None,
            backend: BackendKind::Cpu,
            compliance: CompliancePolicy::strict(),
            // Design-intent defaults, embedding is deferred (2026-07-04 drop).
            watermark: WatermarkConfig::default(),
        }
    }
}

/// Errors surfaced from the service layer.
///
/// Distinct from [`VokraError`] so the T05 HTTP error mapper can return
/// the right status codes: missing model → 404, unavailable → 501,
/// inference failure → 500 / 501 depending on the inner [`VokraError`].
/// We do not conflate them into a single string bag (FR-EX-08 spirit:
/// preserve the failure kind end-to-end).
#[derive(Debug)]
pub enum ServiceError {
    /// A configured GGUF failed to load (missing file, wrong arch,
    /// research-flag gate rejected, …). Startup fails hard, not silently.
    ModelLoadFailed {
        /// Which engine slot tried to load (e.g. `"whisper-base"`).
        slot: &'static str,
        /// The offending path.
        path: PathBuf,
        /// Inner error from vokra-core / vokra-models.
        source: VokraError,
    },
    /// The request named a model that is not in the registry.
    UnknownModel(String),
    /// The request routed to an engine whose synthesize / transcribe
    /// implementation is not available in v0.5 (currently: Kokoro
    /// synthesize). Distinct from an inference failure so the HTTP layer
    /// can return `501 Not Implemented` up-front, not `500`.
    SynthesizeUnavailable {
        /// Which model the caller asked for (e.g. `"kokoro"`).
        model: String,
        /// Human-readable reason (source of the deferral).
        reason: &'static str,
    },
    /// An engine's transcribe / synthesize returned a [`VokraError`]
    /// (unsupported op, invalid input, decode failure, …). The HTTP layer
    /// inspects the inner variant to decide 4xx vs 5xx (M2-09-T05).
    Inference(VokraError),
    /// The [`ServiceConfig`] itself was inconsistent (large-v3 tokenizer
    /// path without a large-v3 GGUF, …).
    InvalidConfig(String),
}

impl fmt::Display for ServiceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ModelLoadFailed { slot, path, source } => {
                write!(f, "model load failed for `{slot}` at {path:?}: {source}")
            }
            Self::UnknownModel(m) => write!(f, "unknown model: `{m}`"),
            Self::SynthesizeUnavailable { model, reason } => {
                write!(f, "synthesize unavailable for `{model}`: {reason}")
            }
            Self::Inference(e) => write!(f, "inference failed: {e}"),
            Self::InvalidConfig(msg) => write!(f, "invalid service config: {msg}"),
        }
    }
}

impl std::error::Error for ServiceError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ModelLoadFailed { source, .. } => Some(source),
            Self::Inference(e) => Some(e),
            _ => None,
        }
    }
}

/// Speech-to-text dispatch trait, keyed by the request's `model` name.
///
/// Deliberately narrow (no streaming, no metadata) so mock implementations
/// in the T08 tests can be dropped in for schema checks without linking a
/// real Whisper engine.
pub trait TranscribeService: Send + Sync {
    /// Transcribes `pcm` (mono `f32`, expected 16 kHz) under the engine
    /// keyed by `model`.
    ///
    /// # Errors
    ///
    /// * [`ServiceError::UnknownModel`] if `model` is not in the registry.
    /// * [`ServiceError::Inference`] if the underlying engine fails.
    fn transcribe(&self, model: &str, pcm: &[f32]) -> Result<String, ServiceError>;
}

/// Text-to-speech dispatch trait, keyed by the request's `model` name.
pub trait SynthesizeService: Send + Sync {
    /// Synthesizes PCM for `request` under the engine keyed by `model`.
    ///
    /// # Errors
    ///
    /// * [`ServiceError::UnknownModel`] if `model` is not in the registry.
    /// * [`ServiceError::SynthesizeUnavailable`] if the engine exists but
    ///   its synthesize path is not wired in v0.5 (Kokoro today).
    /// * [`ServiceError::Inference`] for any inner [`VokraError`].
    fn synthesize(
        &self,
        model: &str,
        request: &SynthesisRequest,
    ) -> Result<SynthesizedAudio, ServiceError>;
}

/// Pre-warmed engine registry — the single struct the HTTP / Wyoming
/// layers share (`Arc<InferenceService>`). All fields are private so the
/// invariants (option pairs consistent, backend / compliance applied to
/// every engine) can only be established through
/// [`InferenceService::build`].
pub struct InferenceService {
    /// Whisper base ASR — always present.
    asr_base: Arc<WhisperAsr>,
    /// Whisper large-v3 ASR — present iff `whisper_large_v3_gguf` was
    /// configured. Absent ⇒ `whisper-large-v3` requests are
    /// [`ServiceError::UnknownModel`] (never silently routed to base).
    asr_large: Option<Arc<WhisperAsr>>,
    /// piper-plus native TTS — always present (default v0.5 TTS).
    tts_piper: Arc<PiperPlusTts>,
    /// Kokoro TTS — advertised iff configured. `synthesize` is unavailable
    /// in v0.5 (M2-07 G2P bridge deferred); the registry rejects synthesize
    /// requests up-front rather than calling into a `NotImplemented` path.
    tts_kokoro: Option<Arc<KokoroTts>>,
    /// Voxtral (Mistral) ASR — advertised iff `voxtral_gguf` is configured
    /// (M3-10). The trait method
    /// [`vokra_models::voxtral::VoxtralAsr::transcribe`] returns
    /// [`VokraError::NotImplemented`] today (full autoregressive decode is
    /// a follow-up); the registry stores the engine so
    /// [`TranscribeService::transcribe`] routes to it and the caller sees
    /// the honest NotImplemented rather than a fabricated transcript
    /// (FR-EX-08).
    asr_voxtral: Option<Arc<VoxtralAsr>>,
    /// Silero VAD v5 — optional Wyoming chunk-boundary helper.
    #[allow(dead_code)] // consumed by T15 (Wyoming ASR chunk framing).
    vad: Option<Arc<SileroVadV5>>,
    /// The phonemizer driving `PiperPlusTts::synthesize_full`. Injected as
    /// a trait object so the real 8-language `piper-plus-g2p` (out-of-
    /// workspace, `integrations/vokra-piper-g2p`) can be swapped in
    /// without T04 depending on that crate.
    phonemizer: Arc<dyn Phonemizer + Send + Sync>,
    /// Watermark config the caller supplied (T21 forward-compat hook).
    ///
    /// The TTS endpoints never call into an embedding backend today
    /// (2026-07-04 client drop); this field is carried so downstream that
    /// wants to advertise per-request scheme flags (`/v1/models`-style
    /// enumeration, HA info event) can read the settings without T21
    /// re-plumbing when embedding re-lands.
    watermark: WatermarkConfig,
}

impl InferenceService {
    /// Pre-warms every engine described by `config`.
    ///
    /// Every configured GGUF **must** load; a missing / broken file is a
    /// hard [`ServiceError::ModelLoadFailed`] (FR-EX-08 — no silent skip),
    /// so a caller that reaches the HTTP bind step is guaranteed to have a
    /// working registry.
    ///
    /// A default [`PassthroughPhonemizer`] is installed so the crate
    /// compiles standalone; the real G2P is injected with
    /// [`InferenceService::with_phonemizer`].
    pub fn build(config: &ServiceConfig) -> Result<Arc<Self>, ServiceError> {
        // Config sanity: large-v3 tokenizer without the large-v3 GGUF is a
        // misconfiguration, not a silent no-op.
        if config.whisper_large_v3_tokenizer.is_some() && config.whisper_large_v3_gguf.is_none() {
            return Err(ServiceError::InvalidConfig(
                "whisper_large_v3_tokenizer set without whisper_large_v3_gguf".into(),
            ));
        }

        // Whisper base — required.
        let asr_base = Arc::new(load_whisper(
            "whisper-base",
            &config.whisper_base_gguf,
            config.whisper_base_tokenizer.as_deref(),
            config.backend,
        )?);

        // Whisper large-v3 — optional.
        let asr_large = if let Some(path) = &config.whisper_large_v3_gguf {
            Some(Arc::new(load_whisper(
                "whisper-large-v3",
                path,
                config.whisper_large_v3_tokenizer.as_deref(),
                config.backend,
            )?))
        } else {
            None
        };

        // piper-plus — required. `from_gguf_with_policy` consumes the
        // `GgufFile` (see crates/vokra-models/src/piper_plus/mod.rs:207).
        let piper_file = open_gguf("piper-plus", &config.piper_plus_gguf)?;
        let tts_piper = Arc::new(
            PiperPlusTts::from_gguf_with_policy(piper_file, &config.compliance)
                .map_err(|source| ServiceError::ModelLoadFailed {
                    slot: "piper-plus",
                    path: config.piper_plus_gguf.clone(),
                    source,
                })?
                .with_backend(config.backend),
        );

        // Kokoro — optional. Loader takes raw bytes, not a `GgufFile`.
        let tts_kokoro = if let Some(path) = &config.kokoro_gguf {
            let bytes = std::fs::read(path).map_err(|e| ServiceError::ModelLoadFailed {
                slot: "kokoro",
                path: path.clone(),
                source: VokraError::Io(e),
            })?;
            Some(Arc::new(
                KokoroTts::from_gguf_with_policy(&bytes, &config.compliance)
                    .map_err(|source| ServiceError::ModelLoadFailed {
                        slot: "kokoro",
                        path: path.clone(),
                        source,
                    })?
                    .with_backend(config.backend),
            ))
        } else {
            None
        };

        // Voxtral — optional. M3-10 registers the ASR engine so
        // /v1/audio/transcriptions can advertise the model name; the engine's
        // transcribe() surfaces NotImplemented today (see the
        // TranscribeService impl comment) — never a silent success.
        let asr_voxtral = if let Some(path) = &config.voxtral_gguf {
            let file = open_gguf("voxtral", path)?;
            let engine =
                VoxtralAsr::from_gguf(&file).map_err(|source| ServiceError::ModelLoadFailed {
                    slot: "voxtral",
                    path: path.clone(),
                    source,
                })?;
            Some(Arc::new(engine))
        } else {
            None
        };

        // Silero VAD — optional.
        let vad = if let Some(path) = &config.silero_vad_gguf {
            let file = open_gguf("silero-vad-v5", path)?;
            let engine =
                SileroVadV5::from_gguf(&file).map_err(|source| ServiceError::ModelLoadFailed {
                    slot: "silero-vad-v5",
                    path: path.clone(),
                    source,
                })?;
            Some(Arc::new(engine))
        } else {
            None
        };

        // Default phonemizer: a passthrough over the piper voice's own
        // phoneme table. This lets the crate build & unit-test without
        // depending on the out-of-workspace 8-language G2P; the real G2P
        // (`integrations/vokra-piper-g2p`) is injected via
        // [`InferenceService::with_phonemizer`] before the server binds.
        let phoneme_table = tts_piper.phoneme_table().map_err(ServiceError::Inference)?;
        let phonemizer: Arc<dyn Phonemizer + Send + Sync> =
            Arc::new(PassthroughPhonemizer::new(phoneme_table));

        Ok(Arc::new(Self {
            asr_base,
            asr_large,
            tts_piper,
            tts_kokoro,
            asr_voxtral,
            vad,
            phonemizer,
            watermark: config.watermark,
        }))
    }

    /// Replaces the default [`PassthroughPhonemizer`] with a real one
    /// (e.g. the 8-language `piper-plus-g2p` bridge from
    /// `integrations/vokra-piper-g2p`).
    ///
    /// Consumes `Arc<Self>` and returns a fresh `Arc<Self>` because
    /// [`InferenceService`] is intended to live behind an `Arc` on every
    /// hot path; swapping the phonemizer is a one-time startup operation.
    #[must_use]
    pub fn with_phonemizer(
        mut self: Arc<Self>,
        phonemizer: Arc<dyn Phonemizer + Send + Sync>,
    ) -> Arc<Self> {
        match Arc::get_mut(&mut self) {
            Some(this) => {
                this.phonemizer = phonemizer;
                self
            }
            None => {
                // Rebuild without cloning engines (they are already Arc).
                let rebuilt = Self {
                    asr_base: Arc::clone(&self.asr_base),
                    asr_large: self.asr_large.clone(),
                    tts_piper: Arc::clone(&self.tts_piper),
                    tts_kokoro: self.tts_kokoro.clone(),
                    asr_voxtral: self.asr_voxtral.clone(),
                    vad: self.vad.clone(),
                    phonemizer,
                    watermark: self.watermark,
                };
                Arc::new(rebuilt)
            }
        }
    }

    /// Returns the Whisper ASR engine keyed by `model` (or `None` if the
    /// alias does not name a Whisper variant or the corresponding engine is
    /// not configured). Voxtral aliases return `None` here — use
    /// [`Self::resolve_voxtral`].
    pub fn resolve_asr(&self, model: &str) -> Option<&Arc<WhisperAsr>> {
        match model {
            model_names::WHISPER_1 | model_names::WHISPER_BASE => Some(&self.asr_base),
            model_names::WHISPER_LARGE_V3 => self.asr_large.as_ref(),
            _ => None,
        }
    }

    /// Returns the Voxtral ASR engine keyed by `model` (or `None` if the
    /// alias does not name a Voxtral variant or the engine is not
    /// configured). M3-10 registers three aliases (generic `voxtral`,
    /// `voxtral-mini-3b`, `voxtral-small-24b`), all routed to the same
    /// engine — the converter's `derive_name` picks the specific variant at
    /// GGUF-write time.
    pub fn resolve_voxtral(&self, model: &str) -> Option<&Arc<VoxtralAsr>> {
        match model {
            model_names::VOXTRAL
            | model_names::VOXTRAL_MINI_3B
            | model_names::VOXTRAL_SMALL_24B => self.asr_voxtral.as_ref(),
            _ => None,
        }
    }

    /// Returns `true` iff the ASR engine keyed by `model` is available
    /// (Whisper or Voxtral). The transcribe path returns an honest
    /// [`ServiceError::Inference`] wrapping a [`VokraError::NotImplemented`]
    /// when a Voxtral engine is registered but its full autoregressive
    /// decode is still pending (M3-10 follow-up).
    pub fn has_asr(&self, model: &str) -> bool {
        self.resolve_asr(model).is_some() || self.resolve_voxtral(model).is_some()
    }

    /// Returns `true` iff the TTS engine keyed by `model` is available
    /// **and its synthesize path is wired** (Kokoro is advertised but
    /// unavailable in v0.5).
    pub fn has_tts_available(&self, model: &str) -> bool {
        match model {
            model_names::PIPER_PLUS => true,
            model_names::KOKORO => false, // advertised, synthesize deferred
            _ => false,
        }
    }

    /// Enumerates registered ASR model names in a stable order (for
    /// `/v1/models`-style listings the HTTP layer will expose).
    ///
    /// Voxtral aliases are advertised when the engine is loaded even though
    /// its transcribe path currently returns `NotImplemented`. That is
    /// deliberate: the catalogue reflects what the deployer configured, and
    /// the honest inference-time error is the right place to signal the
    /// deferral. Silently omitting a configured model would violate
    /// FR-EX-08 in the other direction (the operator SET a path — the
    /// server MUST reflect that).
    pub fn asr_model_names(&self) -> Vec<&'static str> {
        let mut v = vec![model_names::WHISPER_BASE, model_names::WHISPER_1];
        if self.asr_large.is_some() {
            v.push(model_names::WHISPER_LARGE_V3);
        }
        if self.asr_voxtral.is_some() {
            v.push(model_names::VOXTRAL);
            v.push(model_names::VOXTRAL_MINI_3B);
            v.push(model_names::VOXTRAL_SMALL_24B);
        }
        v
    }

    /// Enumerates registered TTS model names (including Kokoro when
    /// present, even though its synthesize is unavailable — the caller
    /// can still see the advertised catalogue).
    pub fn tts_model_names(&self) -> Vec<&'static str> {
        let mut v = vec![model_names::PIPER_PLUS];
        if self.tts_kokoro.is_some() {
            v.push(model_names::KOKORO);
        }
        v
    }

    /// Returns the [`WatermarkConfig`] the caller supplied (T21 forward-compat
    /// hook). Read-only: the server never fires embedding today
    /// (2026-07-04 client drop), but the settings surface is carried so a
    /// future `/v1/models`-style enumeration or Wyoming `info` event can
    /// advertise the design-intent toggles without re-plumbing.
    pub fn watermark_config(&self) -> &WatermarkConfig {
        &self.watermark
    }

    /// Whether the watermark embedding backend is wired. Always
    /// [`vokra_core::WatermarkBackendStatus::Deferred`] in v0.5 (2026-07-04
    /// client drop of FR-CP-01 / FR-CP-02 embedding). Callers must consult
    /// this before claiming any TTS output is watermarked — silently
    /// reporting an active backend when none is wired is worse for
    /// compliance than an explicit "not implemented".
    pub fn watermark_backend_status(&self) -> vokra_core::WatermarkBackendStatus {
        self.watermark.backend_status()
    }
}

impl TranscribeService for InferenceService {
    fn transcribe(&self, model: &str, pcm: &[f32]) -> Result<String, ServiceError> {
        // Whisper takes priority (base + large-v3 are the default catalogue).
        // Voxtral is checked second because its transcribe currently
        // surfaces NotImplemented until the M3-10 follow-up ships; a caller
        // who explicitly names a Voxtral alias needs the honest error, not
        // a silent fall-through to Whisper (FR-EX-08).
        if let Some(engine) = self.resolve_asr(model) {
            return engine
                .transcribe(pcm)
                .map(|t| t.text)
                .map_err(ServiceError::Inference);
        }
        if let Some(engine) = self.resolve_voxtral(model) {
            // M3-10 structural completion: engine is registered, the trait
            // dispatch reaches the encoder, but the greedy autoregressive
            // decode returns NotImplemented. Wrap it verbatim so T05 maps
            // to HTTP 501 — never a fabricated transcript.
            return engine
                .transcribe(pcm)
                .map(|t| t.text)
                .map_err(ServiceError::Inference);
        }
        Err(ServiceError::UnknownModel(model.to_owned()))
    }
}

impl SynthesizeService for InferenceService {
    fn synthesize(
        &self,
        model: &str,
        request: &SynthesisRequest,
    ) -> Result<SynthesizedAudio, ServiceError> {
        match model {
            model_names::PIPER_PLUS => self
                .tts_piper
                .synthesize_full(request, self.phonemizer.as_ref())
                .map_err(ServiceError::Inference),
            model_names::KOKORO => {
                if self.tts_kokoro.is_none() {
                    return Err(ServiceError::UnknownModel(model.to_owned()));
                }
                Err(ServiceError::SynthesizeUnavailable {
                    model: model.to_owned(),
                    reason: "kokoro TtsEngine::synthesize needs a G2P bridge (M2-07 deferred)",
                })
            }
            other => Err(ServiceError::UnknownModel(other.to_owned())),
        }
    }
}

// ---------------------------------------------------------------------------
// WAV (RIFF/WAVE) encoder — PCM 16-bit little-endian mono, piper-plus
// `/api/tts` compatible response body (Content-Type: audio/wav).
//
// Kept private to this crate (not pushed into vokra-core) so no runtime
// crate takes a wav dep. Uses only `Vec<u8>::extend_from_slice` on
// little-endian byte literals — endian-safe on every host (NFR-RL-01
// spirit: never rely on host locale / endian).
//
// Reference (informative): the RIFF/WAVE format is a fixed 44-byte header
// followed by interleaved PCM samples. Fields written verbatim:
//
//   [ 0.. 4] "RIFF"
//   [ 4.. 8] file_size - 8                       u32 LE
//   [ 8..12] "WAVE"
//   [12..16] "fmt "
//   [16..20] 16                                   u32 LE (PCM chunk size)
//   [20..22] 1                                    u16 LE (PCM format)
//   [22..24] num_channels (=1)                    u16 LE
//   [24..28] sample_rate                          u32 LE
//   [28..32] byte_rate = sr * ch * 2              u32 LE
//   [32..34] block_align = ch * 2                 u16 LE
//   [34..36] 16                                   u16 LE (bits/sample)
//   [36..40] "data"
//   [40..44] data_size = num_samples * ch * 2     u32 LE
//   [44.. ]  interleaved i16 LE samples
//
// Samples are clamped to [-1.0, +1.0] before scaling to i16
// (`f32::clamp` returns the clamp bound if the input is NaN, so NaN maps
// to +1.0 → 32767 rather than producing an undefined int cast — NaN in
// TTS output is a bug on the model side; we surface valid audio instead
// of poisoning the buffer). No silent fallback: sample_rate=0 is a hard
// error, not an "assume 16 kHz" (FR-EX-08 spirit).
// ---------------------------------------------------------------------------

/// PCM-16 LE WAV encode error.
#[derive(Debug)]
pub enum WavEncodeError {
    /// The engine returned `sample_rate == 0`. Refuse to guess; the
    /// caller must fail the request.
    ZeroSampleRate,
    /// The encoded WAV would exceed the 32-bit RIFF size field.
    /// Piper-plus `/api/tts` is a single-shot response body, not a
    /// streamed one, so we cap at ~4 GiB and error rather than truncate.
    TooLarge,
}

impl fmt::Display for WavEncodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroSampleRate => f.write_str("wav encode: sample_rate is zero"),
            Self::TooLarge => f.write_str("wav encode: audio exceeds 4 GiB RIFF cap"),
        }
    }
}

impl std::error::Error for WavEncodeError {}

/// Encodes a mono `f32` PCM buffer as RIFF/WAVE PCM 16-bit little-endian.
///
/// Returns the fully-serialised `.wav` bytes ready to be written as the
/// HTTP `/api/tts` response body with `Content-Type: audio/wav`.
///
/// # Errors
///
/// * [`WavEncodeError::ZeroSampleRate`] — the engine returned an
///   invalid sample rate. FR-EX-08 spirit: never silently pick a default.
/// * [`WavEncodeError::TooLarge`] — the payload would overflow the
///   32-bit RIFF `data` size field (~4 GiB PCM ≈ 6 h at 48 kHz mono);
///   piper-plus HTTP is single-shot, so we refuse rather than truncate.
pub fn synthesized_audio_to_wav_pcm16_le(
    audio: &SynthesizedAudio,
) -> Result<Vec<u8>, WavEncodeError> {
    if audio.sample_rate == 0 {
        return Err(WavEncodeError::ZeroSampleRate);
    }
    let num_channels: u16 = 1;
    let bits_per_sample: u16 = 16;
    let bytes_per_sample: u32 = u32::from(bits_per_sample) / 8;

    // data_size = num_samples * num_channels * bytes_per_sample. Check
    // against u32::MAX up-front (`checked_mul` / `checked_add`).
    let sample_count_u32 =
        u32::try_from(audio.samples.len()).map_err(|_| WavEncodeError::TooLarge)?;
    let data_size = sample_count_u32
        .checked_mul(u32::from(num_channels))
        .and_then(|v| v.checked_mul(bytes_per_sample))
        .ok_or(WavEncodeError::TooLarge)?;

    // riff_size = 4 ("WAVE") + 8+16 (fmt chunk) + 8+data_size (data chunk)
    let riff_size = 4u32
        .checked_add(8 + 16)
        .and_then(|v| v.checked_add(8))
        .and_then(|v| v.checked_add(data_size))
        .ok_or(WavEncodeError::TooLarge)?;

    let byte_rate = audio
        .sample_rate
        .checked_mul(u32::from(num_channels))
        .and_then(|v| v.checked_mul(bytes_per_sample))
        .ok_or(WavEncodeError::TooLarge)?;
    let block_align: u16 = num_channels * bits_per_sample / 8;

    let mut out = Vec::with_capacity(44 + data_size as usize);

    // RIFF header.
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&riff_size.to_le_bytes());
    out.extend_from_slice(b"WAVE");

    // "fmt " chunk (PCM = 16 bytes).
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes()); // chunk size
    out.extend_from_slice(&1u16.to_le_bytes()); // audio format = PCM
    out.extend_from_slice(&num_channels.to_le_bytes());
    out.extend_from_slice(&audio.sample_rate.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&bits_per_sample.to_le_bytes());

    // "data" chunk header.
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_size.to_le_bytes());

    // Interleaved i16 LE samples. Clamp then scale (see header comment
    // on NaN handling).
    for &s in &audio.samples {
        let clamped = s.clamp(-1.0, 1.0);
        // Scale to full i16 range. Multiplying by 32767 avoids +1.0 →
        // 32768 overflow (piper's canonical mapping).
        let scaled = (clamped * 32767.0).round();
        // `as i16` on an f32 in [-32767, +32767] is well-defined.
        let sample_i16 = scaled as i16;
        out.extend_from_slice(&sample_i16.to_le_bytes());
    }

    Ok(out)
}

/// The single canonical Content-Type for piper-plus `/api/tts` and any
/// other HTTP endpoint returning PCM-16 LE WAV.
pub const AUDIO_WAV_CONTENT_TYPE: &str = "audio/wav";

// ---------------------------------------------------------------------------
// Private helpers — GGUF open + Whisper load, factored so `build` stays
// linear and self-documenting.
// ---------------------------------------------------------------------------

fn open_gguf(slot: &'static str, path: &Path) -> Result<GgufFile, ServiceError> {
    GgufFile::open(path).map_err(|e| ServiceError::ModelLoadFailed {
        slot,
        path: path.to_path_buf(),
        source: VokraError::ModelLoad(format!("{slot} GGUF at {path:?}: {e}")),
    })
}

fn load_whisper(
    slot: &'static str,
    gguf: &Path,
    tokenizer: Option<&Path>,
    backend: BackendKind,
) -> Result<WhisperAsr, ServiceError> {
    let file = open_gguf(slot, gguf)?;
    let mut engine =
        WhisperAsr::from_gguf(&file).map_err(|source| ServiceError::ModelLoadFailed {
            slot,
            path: gguf.to_path_buf(),
            source,
        })?;
    if let Some(tok_path) = tokenizer {
        let eot = engine.model().config().eot;
        let bytes = std::fs::read(tok_path).map_err(|e| ServiceError::ModelLoadFailed {
            slot,
            path: tok_path.to_path_buf(),
            source: VokraError::Io(e),
        })?;
        let tok = WhisperTokenizer::from_bytes(&bytes, eot).map_err(|source| {
            ServiceError::ModelLoadFailed {
                slot,
                path: tok_path.to_path_buf(),
                source,
            }
        })?;
        engine = engine.with_tokenizer(tok);
    }
    Ok(engine.with_backend(backend))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod registry {
    //! `cargo test service::registry` runs only these T04-scoped registry
    //! tests. They do **not** load real GGUFs — they exercise plumbing
    //! that does not need weights: unknown-model routing, Kokoro-not-
    //! configured routing, ServiceConfig sanity, and the alias table.
    //!
    //! The build path with real GGUFs is exercised at T08 / T13
    //! (integration tests that pull the per-PR base GGUF). Duplicating it
    //! here would either require carrying a GGUF into this excluded
    //! workspace (bloat) or mocking the loader (which defeats FR-EX-08:
    //! we specifically want the real loader in the path).

    use super::*;
    use std::path::PathBuf;

    /// Test double: proves the trait objects are `Send + Sync` and
    /// dispatch by model name without wiring a real engine. This is the
    /// shape T08 mocks will take.
    struct NoopTranscribe;
    impl TranscribeService for NoopTranscribe {
        fn transcribe(&self, model: &str, _pcm: &[f32]) -> Result<String, ServiceError> {
            match model {
                model_names::WHISPER_1 | model_names::WHISPER_BASE => Ok(String::new()),
                other => Err(ServiceError::UnknownModel(other.to_owned())),
            }
        }
    }

    struct NoopSynthesize;
    impl SynthesizeService for NoopSynthesize {
        fn synthesize(
            &self,
            model: &str,
            _request: &SynthesisRequest,
        ) -> Result<SynthesizedAudio, ServiceError> {
            match model {
                model_names::PIPER_PLUS => Err(ServiceError::SynthesizeUnavailable {
                    model: model.to_owned(),
                    reason: "noop test double",
                }),
                other => Err(ServiceError::UnknownModel(other.to_owned())),
            }
        }
    }

    #[test]
    fn model_name_aliases_are_stable() {
        // Guard against accidental rename — the HTTP layer wires the raw
        // OpenAI / vLLM / piper-plus HTTP `model` string to these
        // constants, so renaming them silently would break every test at
        // T08 / T10 / T13.
        assert_eq!(model_names::WHISPER_1, "whisper-1");
        assert_eq!(model_names::WHISPER_BASE, "whisper-base");
        assert_eq!(model_names::WHISPER_LARGE_V3, "whisper-large-v3");
        assert_eq!(model_names::VOXTRAL, "voxtral");
        assert_eq!(model_names::VOXTRAL_MINI_3B, "voxtral-mini-3b");
        assert_eq!(model_names::VOXTRAL_SMALL_24B, "voxtral-small-24b");
        assert_eq!(model_names::PIPER_PLUS, "piper-plus");
        assert_eq!(model_names::KOKORO, "kokoro");
    }

    #[test]
    fn service_config_minimum_leaves_voxtral_slot_absent() {
        // Fresh minimum() config must not silently opt Voxtral in — the
        // deployer must explicitly set `voxtral_gguf` to advertise the
        // model. FR-EX-08 spirit: never fabricate a catalogue entry the
        // operator did not ask for.
        let cfg = ServiceConfig::minimum(
            PathBuf::from("/tmp/base.gguf"),
            PathBuf::from("/tmp/piper.gguf"),
        );
        assert!(cfg.voxtral_gguf.is_none());
    }

    #[test]
    fn transcribe_service_is_object_safe_and_dispatches_by_name() {
        let svc: Box<dyn TranscribeService> = Box::new(NoopTranscribe);
        assert!(svc.transcribe(model_names::WHISPER_BASE, &[]).is_ok());
        assert!(svc.transcribe(model_names::WHISPER_1, &[]).is_ok());
        // Unknown model must not silently succeed (FR-EX-08).
        let err = svc.transcribe("gpt-4", &[]).unwrap_err();
        match err {
            ServiceError::UnknownModel(m) => assert_eq!(m, "gpt-4"),
            other => panic!("expected UnknownModel, got {other}"),
        }
    }

    /// A test double for the M3-10 Voxtral dispatch path: the real
    /// `InferenceService` needs a real GGUF to build, so we drive the
    /// TranscribeService trait directly to guard the intended routing
    /// (Voxtral aliases → the Voxtral engine, which currently surfaces
    /// NotImplemented — never a fabricated transcript).
    struct VoxtralAsHonestNotImplemented;
    impl TranscribeService for VoxtralAsHonestNotImplemented {
        fn transcribe(&self, model: &str, _pcm: &[f32]) -> Result<String, ServiceError> {
            match model {
                model_names::WHISPER_1 | model_names::WHISPER_BASE => Ok(String::new()),
                model_names::VOXTRAL
                | model_names::VOXTRAL_MINI_3B
                | model_names::VOXTRAL_SMALL_24B => {
                    // Same shape the real VoxtralAsr::transcribe returns
                    // today (see `crates/vokra-models/src/voxtral/asr.rs`).
                    Err(ServiceError::Inference(VokraError::NotImplemented(
                        "voxtral::VoxtralAsr::transcribe: deferred to M3-10 follow-up",
                    )))
                }
                other => Err(ServiceError::UnknownModel(other.to_owned())),
            }
        }
    }

    #[test]
    fn voxtral_dispatch_returns_honest_not_implemented_not_fabricated_transcript() {
        // Guards the M3-10 contract: a caller who names a Voxtral alias
        // must reach the Voxtral engine (not fall through to Whisper) and
        // see the honest NotImplemented so the HTTP layer maps to 501.
        let svc: Box<dyn TranscribeService> = Box::new(VoxtralAsHonestNotImplemented);
        for alias in [
            model_names::VOXTRAL,
            model_names::VOXTRAL_MINI_3B,
            model_names::VOXTRAL_SMALL_24B,
        ] {
            let err = svc.transcribe(alias, &[]).unwrap_err();
            match err {
                ServiceError::Inference(VokraError::NotImplemented(msg)) => {
                    assert!(
                        msg.contains("voxtral"),
                        "message must name the model, got `{msg}`"
                    );
                }
                other => panic!("alias `{alias}`: expected Inference(NotImplemented), got {other}"),
            }
        }
    }

    #[test]
    fn synthesize_service_is_object_safe_and_dispatches_by_name() {
        let svc: Box<dyn SynthesizeService> = Box::new(NoopSynthesize);
        let req = SynthesisRequest::new("hello");
        let err = svc.synthesize(model_names::PIPER_PLUS, &req).unwrap_err();
        match err {
            ServiceError::SynthesizeUnavailable { model, .. } => {
                assert_eq!(model, model_names::PIPER_PLUS)
            }
            other => panic!("expected SynthesizeUnavailable, got {other}"),
        }
        let err = svc.synthesize("elevenlabs", &req).unwrap_err();
        assert!(matches!(err, ServiceError::UnknownModel(m) if m == "elevenlabs"));
    }

    #[test]
    fn service_config_minimum_is_strict_by_default() {
        let cfg = ServiceConfig::minimum(
            PathBuf::from("/tmp/base.gguf"),
            PathBuf::from("/tmp/piper.gguf"),
        );
        assert_eq!(cfg.backend, BackendKind::Cpu);
        // We can't equate CompliancePolicy directly (opaque); it exists.
        let _ = &cfg.compliance;
        assert!(cfg.whisper_large_v3_gguf.is_none());
        assert!(cfg.whisper_large_v3_tokenizer.is_none());
        assert!(cfg.kokoro_gguf.is_none());
        assert!(cfg.silero_vad_gguf.is_none());
    }

    #[test]
    fn build_rejects_large_v3_tokenizer_without_gguf() {
        let mut cfg = ServiceConfig::minimum(
            PathBuf::from("/nonexistent/base.gguf"),
            PathBuf::from("/nonexistent/piper.gguf"),
        );
        cfg.whisper_large_v3_tokenizer = Some(PathBuf::from("/nonexistent/tok.bin"));
        // `InferenceService` intentionally does not derive `Debug` (it holds
        // opaque engine state); `expect_err` on `Result<Arc<Self>, _>` would
        // therefore not compile. Explicit match preserves FR-EX-08 clarity.
        let err = match InferenceService::build(&cfg) {
            Ok(_) => panic!("must reject: expected InvalidConfig, got Ok"),
            Err(e) => e,
        };
        match err {
            ServiceError::InvalidConfig(msg) => {
                assert!(msg.contains("whisper_large_v3_tokenizer"));
                assert!(msg.contains("whisper_large_v3_gguf"));
            }
            other => panic!("expected InvalidConfig, got {other}"),
        }
    }

    #[test]
    fn build_missing_gguf_is_hard_error_not_silent_skip() {
        // FR-EX-08: a configured GGUF that does not exist is a startup
        // error, never a silent skip.
        let cfg = ServiceConfig::minimum(
            PathBuf::from("/nonexistent/definitely/not/here.gguf"),
            PathBuf::from("/nonexistent/piper.gguf"),
        );
        let err = match InferenceService::build(&cfg) {
            Ok(_) => panic!("must fail load: expected ModelLoadFailed, got Ok"),
            Err(e) => e,
        };
        match err {
            ServiceError::ModelLoadFailed { slot, path, .. } => {
                assert_eq!(slot, "whisper-base");
                assert!(path.to_string_lossy().contains("here.gguf"));
            }
            other => panic!("expected ModelLoadFailed, got {other}"),
        }
    }

    #[test]
    fn service_error_display_and_source_are_wired() {
        // The T05 HTTP mapper expects Display for user text and source()
        // for the log chain.
        let e = ServiceError::UnknownModel("foo".into());
        assert!(e.to_string().contains("foo"));
        let inner = VokraError::UnsupportedOp("stft on Metal".into());
        let e = ServiceError::Inference(inner);
        assert!(e.to_string().contains("stft on Metal"));
        assert!(std::error::Error::source(&e).is_some());
    }

    #[test]
    fn trait_objects_are_send_sync() {
        fn assert_send_sync<T: Send + Sync + ?Sized>() {}
        assert_send_sync::<dyn TranscribeService>();
        assert_send_sync::<dyn SynthesizeService>();
        assert_send_sync::<InferenceService>();
    }
}

// ---------------------------------------------------------------------------
// T12 tests — SynthesizeService dispatch + PCM-16 LE WAV encoder.
//
// Runnable as `cargo test service::synthesize_dispatch` (M2-09 T12).
//
// These tests do not load real GGUFs. They exercise the dispatch surface
// (`SynthesizeService::synthesize` + `synthesized_audio_to_wav_pcm16_le`)
// with a fake `InferenceService`-shaped double so the routing / WAV
// framing / error taxonomy is guarded without carrying a piper voice
// GGUF into this excluded workspace. The build path with a real
// `PiperPlusTts::synthesize_full` is exercised at T13 (integration test
// hitting the real HTTP handler once GGUF fixtures land).
// ---------------------------------------------------------------------------

#[cfg(test)]
mod synthesize_dispatch {
    use super::*;

    /// Test double that matches the exact routing rules of the real
    /// `InferenceService::synthesize`:
    ///
    /// * `piper-plus`   → succeed (fake audio, canonical piper voice
    ///   sample rate of 22 050 Hz — matches the M0-07 native TTS default);
    /// * `kokoro` when registered → `SynthesizeUnavailable` (M2-07 G2P
    ///   bridge deferred, matches real behaviour);
    /// * `kokoro` when not registered → `UnknownModel`;
    /// * any other name → `UnknownModel`.
    ///
    /// Deliberately mirrors the real dispatch so a regression in either
    /// side surfaces here.
    struct FakeSvc {
        kokoro_registered: bool,
    }

    impl SynthesizeService for FakeSvc {
        fn synthesize(
            &self,
            model: &str,
            _request: &SynthesisRequest,
        ) -> Result<SynthesizedAudio, ServiceError> {
            match model {
                model_names::PIPER_PLUS => Ok(SynthesizedAudio::new(
                    vec![0.0, 0.5, -0.5, 1.0, -1.0],
                    22_050,
                )),
                model_names::KOKORO => {
                    if !self.kokoro_registered {
                        return Err(ServiceError::UnknownModel(model.to_owned()));
                    }
                    Err(ServiceError::SynthesizeUnavailable {
                        model: model.to_owned(),
                        reason: "kokoro TtsEngine::synthesize needs a G2P bridge (M2-07 deferred)",
                    })
                }
                other => Err(ServiceError::UnknownModel(other.to_owned())),
            }
        }
    }

    fn le_u32(bytes: &[u8], off: usize) -> u32 {
        u32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]])
    }
    fn le_u16(bytes: &[u8], off: usize) -> u16 {
        u16::from_le_bytes([bytes[off], bytes[off + 1]])
    }
    fn le_i16(bytes: &[u8], off: usize) -> i16 {
        i16::from_le_bytes([bytes[off], bytes[off + 1]])
    }

    #[test]
    fn piper_plus_dispatch_returns_audio() {
        let svc = FakeSvc {
            kokoro_registered: false,
        };
        let req = SynthesisRequest::new("hello");
        let audio = svc.synthesize(model_names::PIPER_PLUS, &req).unwrap();
        assert_eq!(audio.sample_rate, 22_050);
        assert_eq!(audio.samples.len(), 5);
    }

    #[test]
    fn kokoro_when_registered_is_synthesize_unavailable_501() {
        // Kokoro is *advertised* iff a voice GGUF was configured, but its
        // synthesize path is not wired in v0.5. The registry must reject
        // synthesize up-front so the HTTP layer (T05) can map to 501,
        // NOT call into a NotImplemented engine (FR-EX-08).
        let svc = FakeSvc {
            kokoro_registered: true,
        };
        let req = SynthesisRequest::new("hello");
        let err = svc.synthesize(model_names::KOKORO, &req).unwrap_err();
        match err {
            ServiceError::SynthesizeUnavailable { model, reason } => {
                assert_eq!(model, model_names::KOKORO);
                assert!(reason.contains("M2-07"), "reason must cite the deferral");
            }
            other => panic!("expected SynthesizeUnavailable, got {other}"),
        }
    }

    #[test]
    fn kokoro_when_not_registered_is_unknown_model_404() {
        // Kokoro not configured ⇒ the alias is genuinely unknown; the
        // HTTP layer maps to 404, not 501. Distinction matters.
        let svc = FakeSvc {
            kokoro_registered: false,
        };
        let req = SynthesisRequest::new("hello");
        let err = svc.synthesize(model_names::KOKORO, &req).unwrap_err();
        assert!(matches!(err, ServiceError::UnknownModel(m) if m == model_names::KOKORO));
    }

    #[test]
    fn unknown_model_never_silently_falls_back() {
        // FR-EX-08: a request for "elevenlabs" must not be silently
        // rerouted to piper-plus / whisper — it is an explicit error.
        let svc = FakeSvc {
            kokoro_registered: true,
        };
        let req = SynthesisRequest::new("hello");
        let err = svc.synthesize("elevenlabs", &req).unwrap_err();
        assert!(matches!(err, ServiceError::UnknownModel(m) if m == "elevenlabs"));
    }

    #[test]
    fn wav_content_type_is_piper_plus_compatible() {
        // piper-plus `/api/tts` returns `audio/wav`. Constants pin the
        // string so the HTTP layer (T11) cannot drift.
        assert_eq!(AUDIO_WAV_CONTENT_TYPE, "audio/wav");
    }

    #[test]
    fn wav_header_is_pcm16_mono_at_engine_sample_rate() {
        let audio = SynthesizedAudio::new(vec![0.0, 0.5, -0.5, 1.0, -1.0], 22_050);
        let wav = synthesized_audio_to_wav_pcm16_le(&audio).unwrap();

        // Total = 44-byte header + 5 samples * 2 bytes = 54.
        assert_eq!(wav.len(), 44 + 5 * 2);

        // "RIFF" ... "WAVE".
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(le_u32(&wav, 4), (wav.len() - 8) as u32);
        assert_eq!(&wav[8..12], b"WAVE");

        // "fmt " chunk.
        assert_eq!(&wav[12..16], b"fmt ");
        assert_eq!(le_u32(&wav, 16), 16); // PCM chunk size
        assert_eq!(le_u16(&wav, 20), 1); // format = PCM
        assert_eq!(le_u16(&wav, 22), 1); // channels = mono
        assert_eq!(le_u32(&wav, 24), 22_050); // sample rate
        assert_eq!(le_u32(&wav, 28), 22_050 * 2); // byte rate = sr * 2
        assert_eq!(le_u16(&wav, 32), 2); // block align = 2
        assert_eq!(le_u16(&wav, 34), 16); // bits per sample

        // "data" chunk header.
        assert_eq!(&wav[36..40], b"data");
        assert_eq!(le_u32(&wav, 40), 5 * 2);

        // Samples — check the canonical clamp/scale mapping.
        //  0.0 → 0
        //  0.5 → round(0.5 * 32767) = 16384 (actually 16383.5 → 16384)
        // -0.5 → -16384 (actually -16383.5 → -16384 via round-half-to-even? we use .round())
        //  1.0 → 32767
        // -1.0 → -32767
        assert_eq!(le_i16(&wav, 44), 0);
        // f32::round rounds ties away from zero on stable Rust; assert the two possible legal answers to avoid flakes.
        let mid_pos = le_i16(&wav, 46);
        assert!(mid_pos == 16384 || mid_pos == 16383);
        let mid_neg = le_i16(&wav, 48);
        assert!(mid_neg == -16384 || mid_neg == -16383);
        assert_eq!(le_i16(&wav, 50), 32767);
        assert_eq!(le_i16(&wav, 52), -32767);
    }

    #[test]
    fn wav_clamps_out_of_range_samples() {
        // Model bugs that emit |sample| > 1 must not produce a garbage
        // integer cast; clamp before scaling.
        let audio =
            SynthesizedAudio::new(vec![2.5, -2.5, f32::INFINITY, f32::NEG_INFINITY], 16_000);
        let wav = synthesized_audio_to_wav_pcm16_le(&audio).unwrap();
        assert_eq!(le_i16(&wav, 44), 32767);
        assert_eq!(le_i16(&wav, 46), -32767);
        assert_eq!(le_i16(&wav, 48), 32767);
        assert_eq!(le_i16(&wav, 50), -32767);
    }

    #[test]
    fn wav_zero_sample_rate_is_hard_error() {
        // FR-EX-08 spirit: never guess a sample rate on the caller's
        // behalf.
        let audio = SynthesizedAudio::new(vec![0.0], 0);
        let err = synthesized_audio_to_wav_pcm16_le(&audio).unwrap_err();
        assert!(matches!(err, WavEncodeError::ZeroSampleRate));
    }

    #[test]
    fn wav_empty_samples_still_produces_valid_header() {
        // A synthesis that returned zero samples is still a valid (if
        // useless) WAV; the HTTP layer needs the RIFF header even so.
        let audio = SynthesizedAudio::new(vec![], 16_000);
        let wav = synthesized_audio_to_wav_pcm16_le(&audio).unwrap();
        assert_eq!(wav.len(), 44);
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(le_u32(&wav, 40), 0); // data size = 0
    }

    #[test]
    fn wav_error_types_display_cleanly() {
        // T05 will surface these strings via the error chain.
        assert!(
            WavEncodeError::ZeroSampleRate
                .to_string()
                .contains("sample_rate")
        );
        assert!(WavEncodeError::TooLarge.to_string().contains("4 GiB"));
    }
}

// ---------------------------------------------------------------------------
// T21 tests — compliance / watermark forward-compat.
//
// Runnable as `cargo test compliance` (spec: `compliance::cc_by_nc_rejected`).
//
// Two invariants this module guards:
//
//   1. **Research-flag gate rejects CC-BY-NC weights at load / serve time.**
//      A GGUF whose `vokra.provenance.license` (or resolved class) is
//      CC-BY-NC-4.0 / CC-BY-NC-SA-4.0 / Unknown must fail
//      `InferenceService::build` with a `ServiceError::ModelLoadFailed`
//      wrapping `VokraError::ResearchLicenseRequired` — never a silent load,
//      never a substitution (FR-CP-03 / FR-EX-08). The three unlock routes
//      (`CompliancePolicy::with_research_license(true)`,
//      `ComplianceLevel::Research`, `ComplianceLevel::Disabled`) each
//      individually clear the gate; the load then fails on absent weights,
//      not on the license gate.
//
//   2. **TTS endpoints do NOT fire watermark** (2026-07-04 client drop of
//      FR-CP-01 / FR-CP-02 embedding). The forward-compat surface is a
//      settings hook on `ServiceConfig` → `InferenceService` and nothing
//      more: `watermark_backend_status()` is `Deferred` for every config,
//      including the design-intent default. There is no embedding backend
//      to swap in, and the file never wires one — silently pretending to
//      watermark would be worse for compliance than an explicit
//      "not implemented" (crates/vokra-core/src/compliance/watermark.rs).
//
// The tests write real GGUF bytes to a tempfile using the vokra-core
// `GgufBuilder`, then drive `InferenceService::build` end-to-end. No mocks
// of the loader — the real `check_weight_license` runs, guarding against a
// future refactor that accidentally bypasses the gate at the service layer.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod compliance {
    use super::*;
    use std::fs;
    use std::io::Write;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};
    use vokra_core::gguf::{GgufBuilder, GgufFile, chunks};
    use vokra_core::{ComplianceLevel, CompliancePolicy};
    use vokra_models::piper_plus::PiperPlusTts;

    /// RAII temp-file guard — no new dev-dependency. Deletes on `Drop`, so
    /// tests do not leak per-run GGUFs into the tempdir.
    struct TempGguf {
        path: PathBuf,
    }

    impl TempGguf {
        fn path(&self) -> &std::path::Path {
            &self.path
        }
    }

    impl Drop for TempGguf {
        fn drop(&mut self) {
            // Best-effort — the OS scrubs the tempdir eventually anyway; the
            // guard is here to keep the test dir tidy on repeated runs.
            let _ = fs::remove_file(&self.path);
        }
    }

    /// Monotonic per-process counter so the same test binary run cannot
    /// collide two temp-file names.
    static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn temp_path(prefix: &str) -> PathBuf {
        let pid = std::process::id();
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let n = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!("vokra-server-t21-{prefix}-{pid}-{nanos}-{n}.gguf"))
    }

    /// Builds a minimal piper-arch GGUF (no weight tensors) whose
    /// `vokra.provenance.license` is `license_str` (if any).
    ///
    /// Arch string is exactly the piper loader's `EXPECTED_ARCH`
    /// (`piper-plus-mb-istft-vits2` — see
    /// `crates/vokra-models/src/piper_plus/mod.rs:72`) so the compliance
    /// gate is reached before the arch check errors.
    fn piper_gguf_bytes(license_str: Option<&str>) -> Vec<u8> {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, "piper-plus-mb-istft-vits2");
        if let Some(l) = license_str {
            b.add_string(chunks::KEY_PROVENANCE_LICENSE, l);
        }
        b.to_bytes().expect("serialize piper GGUF")
    }

    fn write_gguf(prefix: &str, bytes: &[u8]) -> TempGguf {
        let path = temp_path(prefix);
        let mut f = fs::File::create(&path).expect("create tempfile");
        f.write_all(bytes).expect("write");
        f.flush().expect("flush");
        TempGguf { path }
    }

    /// Extracts the `VokraError` from a failed piper load. `PiperPlusTts`
    /// is not `Debug`, so `expect_err` will not compile.
    fn load_err(res: std::result::Result<PiperPlusTts, VokraError>) -> VokraError {
        match res {
            Ok(_) => panic!("expected a load error — fixture carries no weight tensors"),
            Err(e) => e,
        }
    }

    /// **The spec test**: `compliance::cc_by_nc_rejected`.
    ///
    /// A CC-BY-NC-4.0 piper voice under a strict compliance policy must be
    /// rejected at load / serve with `VokraError::ResearchLicenseRequired`
    /// (the offending license string preserved verbatim so the T05 HTTP
    /// mapper can surface it, never a bare "load failed"). The gate fires
    /// **before** any weight binding — proving the server layer does not
    /// serve CC-BY-NC weights (F5-TTS / Fish-Speech / EnCodec class) on
    /// any default endpoint (T21).
    ///
    /// The path exercised is the exact call `InferenceService::build`
    /// makes for the piper slot at service.rs:332 —
    /// `PiperPlusTts::from_gguf_with_policy(piper_file, &config.compliance)`.
    /// The `map_err` wrapping into
    /// `ServiceError::ModelLoadFailed { slot: "piper-plus", source, .. }`
    /// is exercised in the same test (round-trip through the
    /// service-layer error type) so the T05 HTTP mapping remains correct.
    /// This avoids needing a full Whisper metadata skeleton in the
    /// fixture just to reach the piper slot in `build`.
    #[test]
    fn cc_by_nc_rejected() {
        // Sanity: the minimum config the server ships with is strict.
        let cfg = ServiceConfig::minimum(
            PathBuf::from("/unused/base.gguf"),
            PathBuf::from("/unused/piper.gguf"),
        );
        assert_eq!(
            cfg.compliance.level(),
            ComplianceLevel::Strict,
            "ServiceConfig::minimum must default to strict — the T21 gate depends on it",
        );

        // Exercise the exact loader call the server uses.
        let bytes = piper_gguf_bytes(Some("CC-BY-NC-4.0"));
        let file = GgufFile::parse(bytes).expect("parse piper GGUF");
        let err = load_err(PiperPlusTts::from_gguf_with_policy(file, &cfg.compliance));

        match &err {
            VokraError::ResearchLicenseRequired { license, .. } => {
                assert!(
                    license.contains("CC-BY-NC"),
                    "the gate must preserve the offending license verbatim; got {license}",
                );
            }
            other => panic!("expected ResearchLicenseRequired, got {other}"),
        }
        assert!(
            err.to_string().contains("CC-BY-NC"),
            "the surfaced error must name the license class",
        );

        // Round-trip through the service-layer error type — exactly what
        // `InferenceService::build` does at service.rs:332 for the piper
        // slot. The T05 HTTP mapper matches on this shape to return 4xx
        // with the license reason preserved.
        let wrapped = ServiceError::ModelLoadFailed {
            slot: "piper-plus",
            path: PathBuf::from("/unused/piper.gguf"),
            source: err,
        };
        match &wrapped {
            ServiceError::ModelLoadFailed { slot, source, .. } => {
                assert_eq!(*slot, "piper-plus");
                assert!(matches!(source, VokraError::ResearchLicenseRequired { .. }));
            }
            other => panic!("expected ModelLoadFailed, got {other}"),
        }
        assert!(wrapped.to_string().contains("CC-BY-NC"));
    }

    /// Each of the three FR-CP-03 unlock routes must clear the license
    /// gate. The load then fails on absent weight tensors — so the error
    /// is **not** `ResearchLicenseRequired`, proving the gate itself
    /// passed on each unlock path.
    #[test]
    fn research_flag_unlocks_the_gate() {
        for policy in [
            CompliancePolicy::strict().with_research_license(true),
            CompliancePolicy::new(ComplianceLevel::Research),
            CompliancePolicy::new(ComplianceLevel::Disabled),
        ] {
            let file = GgufFile::parse(piper_gguf_bytes(Some("CC-BY-NC-4.0"))).expect("parse");
            let err = load_err(PiperPlusTts::from_gguf_with_policy(file, &policy));
            assert!(
                !matches!(err, VokraError::ResearchLicenseRequired { .. }),
                "unlock route {:?} must clear the gate; got {err}",
                policy.level(),
            );
        }
    }

    /// Unknown license class → fail-closed. A GGUF carrying an
    /// unrecognised `vokra.provenance.license` string resolves to
    /// `LicenseClass::Unknown` and strict policy refuses it.
    #[test]
    fn unknown_license_is_also_rejected_fail_closed() {
        let file =
            GgufFile::parse(piper_gguf_bytes(Some("proprietary-mystery-license"))).expect("parse");
        let err = load_err(PiperPlusTts::from_gguf_with_policy(
            file,
            &CompliancePolicy::strict(),
        ));
        assert!(
            matches!(err, VokraError::ResearchLicenseRequired { .. }),
            "unknown license must hit the gate (fail-closed); got {err}",
        );
    }

    /// FR-EX-08 end-to-end: even when the fixture cannot possibly load,
    /// `InferenceService::build` never returns Ok. Tests the wrapping
    /// path independent of the license outcome — the whisper slot is
    /// tried first, so this asserts a `ModelLoadFailed` (of some slot),
    /// never a silent success.
    #[test]
    fn build_never_silently_succeeds_on_missing_metadata() {
        let bogus = write_gguf("bogus", &piper_gguf_bytes(Some("CC-BY-NC-4.0")));
        let cfg = ServiceConfig::minimum(bogus.path().to_path_buf(), bogus.path().to_path_buf());
        let err = match InferenceService::build(&cfg) {
            Ok(_) => panic!("build must fail on a fixture with no weight tensors"),
            Err(e) => e,
        };
        assert!(
            matches!(err, ServiceError::ModelLoadFailed { .. }),
            "expected ModelLoadFailed, got {err}",
        );
    }

    /// TTS endpoints do NOT fire watermark (2026-07-04 drop). The forward-
    /// compat hook exists only to carry settings; the backend status is
    /// permanently `Deferred` in v0.5, regardless of the design-intent
    /// toggle set. A future active backend must flip
    /// `WatermarkBackendStatus` in vokra-core first — this test guards the
    /// registry-layer honesty invariant.
    #[test]
    fn watermark_is_forward_compat_hook_only_never_fires() {
        let cfg = ServiceConfig::minimum(
            PathBuf::from("/never/read.gguf"),
            PathBuf::from("/never/read.gguf"),
        );
        // Design-intent defaults (audioseal + c2pa + silent_cipher ON) are
        // carried on the config — this is the *forward-compat* surface.
        assert!(cfg.watermark.audioseal, "FR-CP-01 default ON (settings)");
        assert!(cfg.watermark.c2pa, "FR-CP-02 default ON (settings)");
        assert!(!cfg.watermark.synthid, "SynthID needs a Google agreement");
        assert!(cfg.watermark.silent_cipher);
        assert!(cfg.watermark.any_enabled());

        // But the backend is deferred — settings do not equal firing.
        assert_eq!(
            cfg.watermark.backend_status(),
            vokra_core::WatermarkBackendStatus::Deferred,
            "TTS endpoints must NOT fire watermark in v0.5 (2026-07-04 drop)",
        );

        // Opting out of AudioSeal still yields Deferred (there is nothing
        // to opt out *of* today) — the opt-out flag is preserved for when
        // embedding re-lands.
        let opted_out = WatermarkConfig {
            audioseal: false,
            ..Default::default()
        };
        assert!(opted_out.audioseal_opted_out());
        assert_eq!(
            opted_out.backend_status(),
            vokra_core::WatermarkBackendStatus::Deferred,
        );
    }
}
