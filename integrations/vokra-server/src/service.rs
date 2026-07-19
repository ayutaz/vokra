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
use vokra_core::decode::{BeamHypothesis, BeamSearchConfig};
use vokra_core::{
    AsrEngine, BackendKind, CompliancePolicy, GgufFile, SynthesisRequest, SynthesizedAudio,
    VokraError, WatermarkConfig,
};
use vokra_models::kokoro::KokoroTts;
use vokra_models::piper_plus::PiperPlusTts;
use vokra_models::silero_vad::SileroVadV5;
use vokra_models::voxtral::VoxtralAsr;
use vokra_models::whisper::asr::WhisperAsr;
use vokra_models::whisper::greedy::DEFAULT_MAX_NEW_TOKENS as WHISPER_DEFAULT_MAX_NEW_TOKENS;
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
    /// cc-39 small alias — matches the GGUF's own `vokra.model.name`.
    pub const WHISPER_SMALL: &str = "whisper-small";
    /// cc-39 medium alias.
    pub const WHISPER_MEDIUM: &str = "whisper-medium";
    /// cc-39 turbo alias (the GGUF's `vokra.model.name`).
    pub const WHISPER_TURBO: &str = "whisper-turbo";
    /// cc-39 turbo alias under its upstream Hugging Face id
    /// (`openai/whisper-large-v3-turbo`) — the spelling most clients have.
    /// Routed to the SAME engine as [`WHISPER_TURBO`]; both are advertised
    /// so the catalogue never implies two distinct engines exist.
    pub const WHISPER_LARGE_V3_TURBO: &str = "whisper-large-v3-turbo";
    /// M3-10 Voxtral generic alias — routed to the loaded Voxtral engine
    /// (mini-3b or small-24b, whichever was configured).
    pub const VOXTRAL: &str = "voxtral";
    /// M3-10 Voxtral mini-3b (Apache 2.0 code + Apache 2.0 weight).
    pub const VOXTRAL_MINI_3B: &str = "voxtral-mini-3b";
    /// M3-10 Voxtral small-24b (Apache 2.0 code + Apache 2.0 weight).
    pub const VOXTRAL_SMALL_24B: &str = "voxtral-small-24b";
    /// piper-plus native TTS alias.
    pub const PIPER_PLUS: &str = "piper-plus";
    /// OpenAI stock TTS alias (`/v1/audio/speech` `model = "tts-1"`), routed
    /// to [`PIPER_PLUS`] — the same convention as [`WHISPER_1`] → base
    /// (cc-38, 2026-07-19 M4-residual audit).
    ///
    /// Only the base tier is aliased. `tts-1-hd` is deliberately NOT mapped:
    /// it names a higher-quality tier this server does not have, so accepting
    /// it would be a quality claim we cannot back — it stays an explicit 404.
    pub const TTS_1: &str = "tts-1";
    /// Kokoro-82M native TTS alias.
    pub const KOKORO: &str = "kokoro";
}

/// One pre-warmed engine slot, as named by `--model-backend <SLOT>=<BACKEND>`
/// (cc-30, 2026-07-19 M4-residual audit).
///
/// The string spellings match the model-path flags (`--whisper-base` ⇒
/// `whisper-base`) so an operator never has to learn a second vocabulary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelSlot {
    /// `--whisper-base`.
    WhisperBase,
    /// `--whisper-small` (cc-39).
    WhisperSmall,
    /// `--whisper-medium` (cc-39).
    WhisperMedium,
    /// `--whisper-turbo` (cc-39).
    WhisperTurbo,
    /// `--whisper-large-v3`.
    WhisperLargeV3,
    /// `--piper-plus`.
    PiperPlus,
    /// `--kokoro`.
    Kokoro,
    /// `--voxtral`.
    Voxtral,
    /// `--silero-vad`.
    SileroVad,
}

impl ModelSlot {
    /// Every slot, in the order [`Self::ALL_NAMES`] lists them. Used by the
    /// startup path to walk overrides and by the config layer's error text.
    pub const ALL: [Self; 9] = [
        Self::WhisperBase,
        Self::WhisperSmall,
        Self::WhisperMedium,
        Self::WhisperTurbo,
        Self::WhisperLargeV3,
        Self::PiperPlus,
        Self::Kokoro,
        Self::Voxtral,
        Self::SileroVad,
    ];

    /// Accepted `<SLOT>` spellings, for `--help` and error messages.
    pub const ALL_NAMES: [&'static str; 9] = [
        "whisper-base",
        "whisper-small",
        "whisper-medium",
        "whisper-turbo",
        "whisper-large-v3",
        "piper-plus",
        "kokoro",
        "voxtral",
        "silero-vad",
    ];

    /// Parse a `<SLOT>` token. `None` = unknown slot (the config layer turns
    /// that into an explicit error listing [`Self::ALL_NAMES`] — never a
    /// silently-dropped override, FR-EX-08).
    pub fn parse(s: &str) -> Option<Self> {
        Self::ALL.iter().copied().find(|slot| slot.as_str() == s)
    }

    /// The canonical spelling of this slot.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::WhisperBase => "whisper-base",
            Self::WhisperSmall => "whisper-small",
            Self::WhisperMedium => "whisper-medium",
            Self::WhisperTurbo => "whisper-turbo",
            Self::WhisperLargeV3 => "whisper-large-v3",
            Self::PiperPlus => "piper-plus",
            Self::Kokoro => "kokoro",
            Self::Voxtral => "voxtral",
            Self::SileroVad => "silero-vad",
        }
    }
}

/// Per-slot backend overrides (cc-30). A slot left `None` runs on
/// [`ServiceConfig::backend`].
///
/// Stored as one `Option` per slot rather than a map so the set of valid
/// slots is exhaustively checked by the compiler when a new engine lands
/// (a `HashMap<String, _>` would let a new slot silently miss its override).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BackendOverrides {
    whisper_base: Option<BackendKind>,
    whisper_small: Option<BackendKind>,
    whisper_medium: Option<BackendKind>,
    whisper_turbo: Option<BackendKind>,
    whisper_large_v3: Option<BackendKind>,
    piper_plus: Option<BackendKind>,
    kokoro: Option<BackendKind>,
    voxtral: Option<BackendKind>,
    silero_vad: Option<BackendKind>,
}

impl BackendOverrides {
    /// The override for `slot`, if any.
    pub fn get(&self, slot: ModelSlot) -> Option<BackendKind> {
        match slot {
            ModelSlot::WhisperBase => self.whisper_base,
            ModelSlot::WhisperSmall => self.whisper_small,
            ModelSlot::WhisperMedium => self.whisper_medium,
            ModelSlot::WhisperTurbo => self.whisper_turbo,
            ModelSlot::WhisperLargeV3 => self.whisper_large_v3,
            ModelSlot::PiperPlus => self.piper_plus,
            ModelSlot::Kokoro => self.kokoro,
            ModelSlot::Voxtral => self.voxtral,
            ModelSlot::SileroVad => self.silero_vad,
        }
    }

    /// Set (or replace) the override for `slot`. The config layer checks for
    /// a duplicate *within one precedence layer* before calling this; the
    /// cross-layer merge ([`Self::or`]) deliberately replaces.
    pub fn set(&mut self, slot: ModelSlot, backend: BackendKind) {
        let cell = match slot {
            ModelSlot::WhisperBase => &mut self.whisper_base,
            ModelSlot::WhisperSmall => &mut self.whisper_small,
            ModelSlot::WhisperMedium => &mut self.whisper_medium,
            ModelSlot::WhisperTurbo => &mut self.whisper_turbo,
            ModelSlot::WhisperLargeV3 => &mut self.whisper_large_v3,
            ModelSlot::PiperPlus => &mut self.piper_plus,
            ModelSlot::Kokoro => &mut self.kokoro,
            ModelSlot::Voxtral => &mut self.voxtral,
            ModelSlot::SileroVad => &mut self.silero_vad,
        };
        *cell = Some(backend);
    }

    /// Slot-by-slot `Option::or`: keeps `self`'s value where it has one and
    /// adopts `lower`'s otherwise. Used to fold CLI > env > TOML.
    #[must_use]
    pub fn or(&self, lower: &Self) -> Self {
        let mut out = *self;
        for slot in ModelSlot::ALL {
            if out.get(slot).is_none() {
                if let Some(b) = lower.get(slot) {
                    out.set(slot, b);
                }
            }
        }
        out
    }

    /// Whether any slot carries an override.
    pub fn is_empty(&self) -> bool {
        ModelSlot::ALL.iter().all(|s| self.get(*s).is_none())
    }
}

/// Verify that `backend` is usable by THIS binary, before any engine loads
/// (cc-30, 2026-07-19 M4-residual audit).
///
/// [`vokra_models::Compute::for_backend`] is the authority: a backend whose
/// Cargo feature was not compiled in returns
/// [`VokraError::BackendUnavailable`], and a compiled-in backend whose device
/// cannot be opened fails in its context constructor. Both are hard startup
/// errors here rather than per-request 500s — a server that binds a port must
/// be able to answer on the backend it was told to use (FR-EX-08: never a
/// silent CPU fall back under a GPU label).
///
/// The probe passes an EMPTY required-op set, so it answers exactly one
/// question: "can this binary open this backend at all?". Per-op coverage
/// holes are a different failure and still surface per request as an explicit
/// `UnsupportedOp` from the model hot path — this check does not (and must
/// not) pretend to validate them.
///
/// # Errors
///
/// [`ServiceError::InvalidConfig`] naming the Cargo feature to rebuild with.
pub fn ensure_backend_available(backend: BackendKind) -> Result<(), ServiceError> {
    vokra_models::Compute::for_backend(backend, &[])
        .map(drop)
        .map_err(|source| {
            ServiceError::InvalidConfig(format!(
                "backend `{}` was requested but is not usable by this vokra-server binary: \
                 {source}. GPU backends are opt-in Cargo features that forward to \
                 vokra-models — rebuild with `cargo build --release --features {}` (and \
                 make sure the device/driver is present). Refusing to start on a \
                 different backend than the one requested (FR-EX-08).",
                backend_flag_name(backend),
                backend_feature_name(backend),
            ))
        })
}

/// The `--backend` spelling of a [`BackendKind`], for error text.
fn backend_flag_name(backend: BackendKind) -> &'static str {
    match backend {
        BackendKind::Cpu => "cpu",
        BackendKind::Metal => "metal",
        BackendKind::Cuda => "cuda",
        BackendKind::Vulkan => "vulkan",
        other => {
            // `BackendKind` is `#[non_exhaustive]`-in-spirit (WebGpu, and NPU
            // delegates land in M5): fall back to the Debug spelling rather
            // than inventing a flag name that does not exist.
            debug_assert!(false, "unmapped BackendKind {other:?}");
            "<unknown>"
        }
    }
}

/// The Cargo feature that compiles a [`BackendKind`] into this binary.
/// `vokra-server`'s features forward 1:1 to `vokra-models`' (see its
/// `Cargo.toml`), so the name is the same on both sides.
fn backend_feature_name(backend: BackendKind) -> &'static str {
    match backend {
        // CPU is always compiled in; naming a feature here would be wrong,
        // but the arm must exist — a CPU probe failure is a real bug, not a
        // missing feature.
        BackendKind::Cpu => "<none — cpu is always built in>",
        BackendKind::Metal => "metal",
        BackendKind::Cuda => "cuda",
        BackendKind::Vulkan => "vulkan",
        _ => "<unknown>",
    }
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
    /// Optional path to a Whisper **small** GGUF (cc-39, 2026-07-19
    /// M4-residual audit). Model-side support landed with M4-14 and all four
    /// sizes transcribe byte-identically to onnxruntime
    /// (`docs/bench-baselines/m1-real-weight-eval-2026-07-16/report.md`), but
    /// the server only ever had base + large-v3 slots. Absent ⇒
    /// `whisper-small` requests are [`ServiceError::UnknownModel`], never
    /// silently served by a different size (FR-EX-08).
    pub whisper_small_gguf: Option<PathBuf>,
    /// Optional Whisper small tokenizer sidecar. Every converter-B GGUF
    /// embeds `vokra.tokenizer.model` (verified on the real
    /// `whisper-small.gguf`), so this is normally unnecessary — kept for the
    /// converter-A path, exactly like large-v3's.
    pub whisper_small_tokenizer: Option<PathBuf>,
    /// Optional path to a Whisper **medium** GGUF (cc-39).
    pub whisper_medium_gguf: Option<PathBuf>,
    /// Optional Whisper medium tokenizer sidecar (see
    /// [`Self::whisper_small_tokenizer`]).
    pub whisper_medium_tokenizer: Option<PathBuf>,
    /// Optional path to a Whisper **large-v3-turbo** GGUF (cc-39).
    pub whisper_turbo_gguf: Option<PathBuf>,
    /// Optional Whisper turbo tokenizer sidecar (see
    /// [`Self::whisper_small_tokenizer`]).
    pub whisper_turbo_tokenizer: Option<PathBuf>,
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
    /// `voxtral-small-24b` aliases and
    /// [`vokra_models::voxtral::VoxtralAsr::transcribe`] runs the M3-10
    /// autoregressive greedy path (log-mel front-end + audio encoder +
    /// Mistral text decoder + KV cache + tokenizer decode) on a
    /// dispatched request, returning HTTP 200 with the decoded text.
    /// Missing tokenizer chunk / shape-only converter / uncovered
    /// backend still surface as explicit
    /// [`ServiceError::Inference`] errors mapped to the appropriate
    /// 4xx/5xx codes by T05 — never a silent fabrication (FR-EX-08).
    /// Note the ASR-quality honest scope in
    /// `crate::voxtral::asr` module docs: the audio-adapter follow-up
    /// (real audio conditioning) is a downstream ticket.
    pub voxtral_gguf: Option<PathBuf>,
    /// Optional path to a Silero VAD v5 GGUF. When absent, the Wyoming
    /// chunk-boundary VAD helper is disabled (chunks are used as-is).
    pub silero_vad_gguf: Option<PathBuf>,
    /// Default backend the pre-warmed engines run on. A slot with an entry
    /// in [`Self::backend_overrides`] uses that instead.
    ///
    /// Every distinct backend actually selected is probed once at
    /// [`InferenceService::build`] time via [`ensure_backend_available`]: a
    /// backend not compiled into this binary is a hard startup error, never a
    /// silent CPU fall back (FR-EX-08).
    pub backend: BackendKind,
    /// Per-model backend overrides (cc-30, 2026-07-19 M4-residual audit —
    /// this is the "a per-model backend override is a T03 follow-up" note
    /// that sat on `backend` since M2-09-T04).
    ///
    /// Empty = every engine runs on [`Self::backend`], which is exactly the
    /// pre-cc-30 behaviour.
    pub backend_overrides: BackendOverrides,
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
            whisper_small_gguf: None,
            whisper_small_tokenizer: None,
            whisper_medium_gguf: None,
            whisper_medium_tokenizer: None,
            whisper_turbo_gguf: None,
            whisper_turbo_tokenizer: None,
            piper_plus_gguf,
            kokoro_gguf: None,
            voxtral_gguf: None,
            silero_vad_gguf: None,
            backend: BackendKind::Cpu,
            backend_overrides: BackendOverrides::default(),
            compliance: CompliancePolicy::strict(),
            // Design-intent defaults, embedding is deferred (2026-07-04 drop).
            watermark: WatermarkConfig::default(),
        }
    }

    /// The backend `slot` will actually run on (cc-30): its per-model
    /// override if it has one, otherwise [`Self::backend`].
    ///
    /// **Silero VAD is always CPU.** `SileroVadV5` exposes no backend
    /// selector at all (it is a small LSTM subgraph with a hand-written CPU
    /// kernel — unlike Whisper / piper / Kokoro / Voxtral it has no
    /// `with_backend`), so no value here could be honoured. Returning `Cpu`
    /// is therefore a statement of fact, not a fallback: an *explicit*
    /// `--model-backend silero-vad=…` is rejected outright by
    /// [`Self::validate_backend_overrides`] (FR-EX-08 — an override the
    /// runtime cannot honour must fail loudly), and a non-CPU *global*
    /// default is announced at startup rather than silently ignored.
    pub fn backend_for(&self, slot: ModelSlot) -> BackendKind {
        if slot == ModelSlot::SileroVad {
            return BackendKind::Cpu;
        }
        self.backend_overrides.get(slot).unwrap_or(self.backend)
    }

    /// Every config-consistency check, run BEFORE any weight is read.
    ///
    /// Ordering matters and is load-bearing: a half-configured slot must be
    /// reported as the misconfiguration it is, not masked by whichever file
    /// happens to fail to open first. `build` therefore calls this before
    /// touching the filesystem, so `whisper_large_v3_tokenizer` without its
    /// GGUF surfaces as that error even when the base GGUF path is also
    /// wrong. (Pinned by `registry::build_rejects_large_v3_tokenizer_without_gguf`.)
    ///
    /// # Errors
    ///
    /// [`ServiceError::InvalidConfig`] naming the offending field.
    pub fn validate(&self) -> Result<(), ServiceError> {
        // An optional Whisper slot given a tokenizer sidecar but no GGUF is
        // a typo'd path, not a no-op (FR-EX-08). Checked for all four
        // optional sizes, not just large-v3 (cc-39).
        for (slot, gguf, tok) in [
            (
                "whisper-large-v3",
                &self.whisper_large_v3_gguf,
                &self.whisper_large_v3_tokenizer,
            ),
            (
                "whisper-small",
                &self.whisper_small_gguf,
                &self.whisper_small_tokenizer,
            ),
            (
                "whisper-medium",
                &self.whisper_medium_gguf,
                &self.whisper_medium_tokenizer,
            ),
            (
                "whisper-turbo",
                &self.whisper_turbo_gguf,
                &self.whisper_turbo_tokenizer,
            ),
        ] {
            if gguf.is_none() {
                if let Some(tok) = tok {
                    return Err(orphan_tokenizer_error(slot, tok));
                }
            }
        }
        self.validate_backend_overrides()
    }

    /// Reject per-model overrides that no engine could honour (cc-30).
    ///
    /// Only Silero VAD is in this category today. Accepting the flag and
    /// quietly running CPU would be exactly the silent-substitution failure
    /// FR-EX-08 exists to prevent.
    ///
    /// # Errors
    ///
    /// [`ServiceError::InvalidConfig`] naming the slot and the reason.
    pub fn validate_backend_overrides(&self) -> Result<(), ServiceError> {
        if let Some(requested) = self.backend_overrides.get(ModelSlot::SileroVad) {
            return Err(ServiceError::InvalidConfig(format!(
                "--model-backend silero-vad={} cannot be honoured: the Silero VAD v5 engine \
                 has no backend selector (it is a CPU-only LSTM subgraph — `SileroVadV5` \
                 exposes no `with_backend`). Drop the override rather than have the server \
                 accept it and run CPU anyway (FR-EX-08).",
                backend_flag_name(requested),
            )));
        }
        Ok(())
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

    /// Beam-search transcribe with an n-best response payload (M3-15 Wave
    /// 10 A). Falls through to greedy [`Self::transcribe`] when
    /// `req.beam_size` is `None` or `Some(1)` — the response then carries
    /// an empty `alternatives` list, preserving backward-compat with the
    /// legacy top-1 shape.
    ///
    /// Providers that DO NOT support beam search MUST NOT silently ignore
    /// `beam_size > 1` (FR-EX-08). The stock default implementation below
    /// enforces that by returning an explicit
    /// [`ServiceError::Inference`]([`VokraError::UnsupportedOp`]) — the
    /// HTTP layer maps that to 501. An engine that CAN honour beam search
    /// (Voxtral today) overrides this method to route through its beam
    /// path.
    ///
    /// # Errors
    ///
    /// Same taxonomy as [`Self::transcribe`], plus
    /// [`ServiceError::Inference`] wrapping [`VokraError::UnsupportedOp`]
    /// when a caller requests beam search from an engine that only
    /// supports greedy through this trait.
    fn transcribe_beam(
        &self,
        model: &str,
        pcm: &[f32],
        req: &TranscribeBeamRequest,
    ) -> Result<TranscribeBeamResponse, ServiceError> {
        // Default behaviour: fall through to greedy for beam_size 0..=1,
        // hard-error for beam_size > 1. Concrete implementations override
        // this to route their supported engines through a real beam path.
        //
        // cc-19 (2026-07-19 M4-residual audit): a word-timestamps ask on an
        // implementation that has not overridden this method MUST NOT fold
        // into greedy-with-empty-words — that would be a silently fabricated
        // "no words" alignment (FR-EX-08 / NFR-RL-06). Explicit error instead.
        if req.word_timestamps == Some(true) {
            return Err(ServiceError::Inference(VokraError::UnsupportedOp(format!(
                "transcribe_beam: model `{model}` does not support word_timestamps on this \
                 TranscribeService implementation (FR-EX-08 — word timings are never fabricated)",
            ))));
        }
        match req.beam_size {
            None | Some(0..=1) => {
                let text = self.transcribe(model, pcm)?;
                Ok(TranscribeBeamResponse {
                    text,
                    alternatives: Vec::new(),
                    words: Vec::new(),
                })
            }
            Some(_) => Err(ServiceError::Inference(VokraError::UnsupportedOp(format!(
                "transcribe_beam: model `{model}` does not support beam search on this \
                 TranscribeService implementation (FR-EX-08 — no silent fall back to greedy)",
            )))),
        }
    }
}

/// Beam-search request knobs (M3-15 Wave 10 A). Passed as a value to
/// [`TranscribeService::transcribe_beam`]. Every field is optional — an
/// entirely-default request (all `None`) is greedy and matches the
/// legacy [`TranscribeService::transcribe`] shape exactly.
///
/// # Backward compat
///
/// * `beam_size` `None` or `Some(0)` or `Some(1)` → greedy. The response's
///   `alternatives` list is empty (the top-1 lives in `text`).
/// * `beam_size` `Some(n)` for `n > 1` → beam search when supported.
/// * `length_penalty` / `no_repeat_ngram` are honoured by engines that
///   support them (Voxtral, and — since Wave 12 core-side plumbing — Whisper
///   via `BeamSearchConfig::no_repeat_ngram_size`). Providers that only
///   advertise greedy surface an explicit error when the caller requests
///   `beam_size > 1` — never a silent no-op (FR-EX-08).
#[derive(Debug, Clone, Default)]
pub struct TranscribeBeamRequest {
    /// Beam width. `None` or `Some(0..=1)` = greedy. `> 1` = beam search
    /// on engines that support it.
    pub beam_size: Option<usize>,
    /// GNMT length-penalty exponent. Ignored on the greedy path. Defaults
    /// on the engine to `0.6` when unset (matches
    /// [`vokra_models::voxtral::BeamConfig::with_beam_size`]).
    pub length_penalty: Option<f32>,
    /// Block repeated n-grams of this length. `None` or `Some(0)` disables
    /// blocking.
    pub no_repeat_ngram: Option<usize>,
    /// Upper bound on generated tokens. `None` picks the engine default
    /// (`DEFAULT_MAX_NEW_TOKENS`).
    pub max_new_tokens: Option<usize>,
    /// Emit word-level timestamps (M4-20, FR-OP-40). `None`/`Some(false)`
    /// leaves [`TranscribeBeamResponse::words`] empty (backward compat). When
    /// `Some(true)`, the Whisper backend runs the cross-attention alignment
    /// (via [`vokra_core::decode::BeamSearchConfig::word_timestamps`]) even at
    /// `beam_size <= 1` — greedy `transcribe` produces no alignment — and
    /// surfaces one entry per word. A model without alignment heads makes this
    /// an explicit error, never a silent empty list (FR-EX-08).
    pub word_timestamps: Option<bool>,
}

/// One n-best alternative in the beam-search response.
///
/// Kept a plain struct (not `serde`-derived here) so the T04 layer stays
/// free of `serde` on the internal dispatch surface; the OpenAI /
/// vLLM-compatible HTTP layer wraps this into a serializable shape when
/// the beam endpoint lands.
#[derive(Debug, Clone, PartialEq)]
pub struct TranscribeAlternative {
    /// Decoded text for this beam.
    pub text: String,
    /// Cumulative sum of per-step `log_softmax` values (raw,
    /// unnormalized). See
    /// [`vokra_models::voxtral::BeamResult::log_prob`].
    pub log_prob: f64,
    /// GNMT length-normalized ranking score — the value the response
    /// list is sorted by, descending. See
    /// [`vokra_models::voxtral::BeamResult::length_normalized_score`].
    pub length_normalized_score: f64,
}

/// One word-level timestamp entry (M4-20, FR-OP-40). The `word` text is the
/// detokenization of the aligned token span; `start` / `end` are seconds into
/// the audio.
#[derive(Debug, Clone, PartialEq)]
pub struct WordTimestamp {
    /// Detokenized word text (the aligned token span, spaces included as the
    /// tokenizer produced them).
    pub word: String,
    /// Word start time in seconds.
    pub start: f64,
    /// Word end time in seconds.
    pub end: f64,
}

/// n-best transcribe response (M3-15 Wave 10 A). `text` always carries
/// the top-1 (matches the legacy [`TranscribeService::transcribe`]
/// response shape exactly). `alternatives` is empty on the greedy path,
/// non-empty and ranked descending on the beam-search path — the top
/// entry equals `text`.
#[derive(Debug, Clone)]
pub struct TranscribeBeamResponse {
    /// Top-1 decoded text (matches [`TranscribeService::transcribe`]).
    pub text: String,
    /// Full n-best list ranked by descending
    /// [`TranscribeAlternative::length_normalized_score`]. Empty when
    /// the request was greedy — that preserves the legacy top-1 shape
    /// verbatim so a caller who never sets `beam_size` sees no change.
    pub alternatives: Vec<TranscribeAlternative>,
    /// Word-level timestamps for the top-1 hypothesis (M4-20). Empty unless
    /// [`TranscribeBeamRequest::word_timestamps`] was `Some(true)` and the
    /// backend produced an alignment. Additive field — callers that never
    /// request word timestamps see an empty list.
    pub words: Vec<WordTimestamp>,
}

/// Model-catalogue view for `GET /v1/models` (cc-18, 2026-07-19 M4-residual
/// audit — the enumeration helpers below existed since T04 but no HTTP route
/// ever exposed them).
///
/// Kept as its own narrow trait (rather than widening [`TranscribeService`])
/// so the route can be driven by a mock in tests and so non-registry servers
/// (health-only boot) simply do not mount the route — an honest 404, never an
/// empty fabricated catalogue (FR-EX-08).
pub trait ModelCatalog: Send + Sync {
    /// Every model id the HTTP routes accept, in a stable order. Aliases that
    /// route to the same engine (e.g. `whisper-base` / `whisper-1`) are each
    /// listed — the catalogue's contract is "these ids are accepted", not
    /// "these engines are distinct".
    fn model_ids(&self) -> Vec<String>;
}

impl ModelCatalog for InferenceService {
    fn model_ids(&self) -> Vec<String> {
        let mut v: Vec<String> = self
            .asr_model_names()
            .into_iter()
            .map(str::to_owned)
            .collect();
        v.extend(self.tts_model_names().into_iter().map(str::to_owned));
        v
    }
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
    /// Whisper small ASR (cc-39) — present iff configured. Same
    /// no-substitution rule as `asr_large`.
    asr_small: Option<Arc<WhisperAsr>>,
    /// Whisper medium ASR (cc-39) — present iff configured.
    asr_medium: Option<Arc<WhisperAsr>>,
    /// Whisper large-v3-turbo ASR (cc-39) — present iff configured.
    asr_turbo: Option<Arc<WhisperAsr>>,
    /// piper-plus native TTS — always present (default v0.5 TTS).
    tts_piper: Arc<PiperPlusTts>,
    /// Kokoro TTS — advertised iff configured. `synthesize` is unavailable
    /// in v0.5 (M2-07 G2P bridge deferred); the registry rejects synthesize
    /// requests up-front rather than calling into a `NotImplemented` path.
    tts_kokoro: Option<Arc<KokoroTts>>,
    /// Voxtral (Mistral) ASR — advertised iff `voxtral_gguf` is configured
    /// (M3-10). The trait method
    /// [`vokra_models::voxtral::VoxtralAsr::transcribe`] runs the
    /// autoregressive greedy path (M3-10 T13) and returns
    /// `Ok(Transcription)` on success; errors surface as
    /// [`ServiceError::Inference`] and the HTTP layer maps them to
    /// 4xx/5xx codes as appropriate (FR-EX-08 — no fabricated output).
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
        // Config consistency first, before any filesystem access: the
        // operator must see the flag they typed named back at them, not a
        // load error for whichever slot happened to be read first.
        config.validate()?;

        // cc-30: every backend this config actually selects must be usable by
        // THIS binary before a single weight is read. Probing up-front keeps
        // the failure at startup (where the operator sees it) instead of on
        // the first request, and the probe is once-per-distinct-backend so a
        // 9-slot config does not open 9 GPU contexts.
        let mut probed: Vec<BackendKind> = Vec::new();
        for slot in ModelSlot::ALL {
            let backend = config.backend_for(slot);
            if !probed.contains(&backend) {
                ensure_backend_available(backend)?;
                probed.push(backend);
            }
        }

        // Whisper base — required.
        let asr_base = Arc::new(load_whisper(
            "whisper-base",
            &config.whisper_base_gguf,
            config.whisper_base_tokenizer.as_deref(),
            config.backend_for(ModelSlot::WhisperBase),
        )?);

        // Whisper large-v3 + the three cc-39 sizes — all optional, all with
        // the same "tokenizer sidecar without its GGUF is a misconfiguration"
        // rule (a silent no-op would hide a typo'd path).
        let asr_large = load_optional_whisper(
            "whisper-large-v3",
            config.whisper_large_v3_gguf.as_deref(),
            config.whisper_large_v3_tokenizer.as_deref(),
            config.backend_for(ModelSlot::WhisperLargeV3),
        )?;
        let asr_small = load_optional_whisper(
            "whisper-small",
            config.whisper_small_gguf.as_deref(),
            config.whisper_small_tokenizer.as_deref(),
            config.backend_for(ModelSlot::WhisperSmall),
        )?;
        let asr_medium = load_optional_whisper(
            "whisper-medium",
            config.whisper_medium_gguf.as_deref(),
            config.whisper_medium_tokenizer.as_deref(),
            config.backend_for(ModelSlot::WhisperMedium),
        )?;
        let asr_turbo = load_optional_whisper(
            "whisper-turbo",
            config.whisper_turbo_gguf.as_deref(),
            config.whisper_turbo_tokenizer.as_deref(),
            config.backend_for(ModelSlot::WhisperTurbo),
        )?;

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
                .with_backend(config.backend_for(ModelSlot::PiperPlus)),
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
                    .with_backend(config.backend_for(ModelSlot::Kokoro)),
            ))
        } else {
            None
        };

        // Voxtral — optional. M3-10 T13 landed autoregressive greedy so
        // /v1/audio/transcriptions returns 200 on a well-formed dispatch;
        // registration side still surfaces ModelLoad errors up-front (never
        // a silent skip).
        let asr_voxtral = if let Some(path) = &config.voxtral_gguf {
            let file = open_gguf("voxtral", path)?;
            let engine =
                VoxtralAsr::from_gguf(&file).map_err(|source| ServiceError::ModelLoadFailed {
                    slot: "voxtral",
                    path: path.clone(),
                    source,
                })?;
            // cc-30: Voxtral DOES have a backend selector
            // (`VoxtralAsr::with_backend`, M3-10 Wave 9) but this registry
            // never applied it — so before cc-30 a Voxtral engine silently
            // stayed on its `BackendKind::Cpu` default even when the rest of
            // the registry was configured otherwise. Applying it here makes
            // `ServiceConfig::backend`'s "applied uniformly" doc true.
            Some(Arc::new(
                engine.with_backend(config.backend_for(ModelSlot::Voxtral)),
            ))
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
            asr_small,
            asr_medium,
            asr_turbo,
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
                    asr_small: self.asr_small.clone(),
                    asr_medium: self.asr_medium.clone(),
                    asr_turbo: self.asr_turbo.clone(),
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

    /// Returns the loaded piper-plus voice (always present — it is part of
    /// the required startup minimum).
    ///
    /// The production startup path uses this to derive the real 8-language
    /// G2P from the voice's own GGUF metadata
    /// (`vokra_piper_g2p::PiperPlusG2p::from_voice` reads
    /// `vokra.piper.phoneme_symbols` / `language_codes`) before swapping it
    /// in via [`Self::with_phonemizer`] — the out-of-workspace G2P crate is
    /// deliberately NOT imported here (T04 stays HTTP/G2P-free), only the
    /// engine handle is exposed.
    pub fn piper_voice(&self) -> &Arc<PiperPlusTts> {
        &self.tts_piper
    }

    /// Returns the Whisper ASR engine keyed by `model` (or `None` if the
    /// alias does not name a Whisper variant or the corresponding engine is
    /// not configured). Voxtral aliases return `None` here — use
    /// [`Self::resolve_voxtral`].
    pub fn resolve_asr(&self, model: &str) -> Option<&Arc<WhisperAsr>> {
        match model {
            model_names::WHISPER_1 | model_names::WHISPER_BASE => Some(&self.asr_base),
            model_names::WHISPER_LARGE_V3 => self.asr_large.as_ref(),
            // cc-39. An unconfigured size returns `None` here and becomes
            // `ServiceError::UnknownModel` → HTTP 404 at the caller. It must
            // NEVER fall through to `asr_base`: a caller who asked for medium
            // and silently got base would have no way to detect the swap
            // (RED-LINE, FR-EX-08).
            model_names::WHISPER_SMALL => self.asr_small.as_ref(),
            model_names::WHISPER_MEDIUM => self.asr_medium.as_ref(),
            model_names::WHISPER_TURBO | model_names::WHISPER_LARGE_V3_TURBO => {
                self.asr_turbo.as_ref()
            }
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
    /// (Whisper or Voxtral). Voxtral's transcribe returns `Ok` on a
    /// well-formed dispatch (M3-10 T13 autoregressive greedy path).
    pub fn has_asr(&self, model: &str) -> bool {
        self.resolve_asr(model).is_some() || self.resolve_voxtral(model).is_some()
    }

    /// Returns `true` iff the TTS engine keyed by `model` is available
    /// **and its synthesize path is wired** (Kokoro is advertised but
    /// unavailable in v0.5).
    pub fn has_tts_available(&self, model: &str) -> bool {
        match model {
            model_names::PIPER_PLUS | model_names::TTS_1 => true,
            model_names::KOKORO => false, // advertised, synthesize deferred
            _ => false,
        }
    }

    /// Enumerates registered ASR model names in a stable order (for
    /// `/v1/models`-style listings the HTTP layer will expose).
    ///
    /// Voxtral aliases are advertised when the engine is loaded — the
    /// M3-10 T13 greedy path returns 200 on dispatch. The catalogue
    /// reflects what the deployer configured; silently omitting a
    /// configured model would violate FR-EX-08 (the operator SET a
    /// path — the server MUST reflect that).
    pub fn asr_model_names(&self) -> Vec<&'static str> {
        let mut v = vec![model_names::WHISPER_BASE, model_names::WHISPER_1];
        // cc-39: sizes are advertised in ascending order after base so
        // `GET /v1/models` reads naturally; each appears only when its GGUF
        // was actually configured.
        if self.asr_small.is_some() {
            v.push(model_names::WHISPER_SMALL);
        }
        if self.asr_medium.is_some() {
            v.push(model_names::WHISPER_MEDIUM);
        }
        if self.asr_turbo.is_some() {
            v.push(model_names::WHISPER_TURBO);
            v.push(model_names::WHISPER_LARGE_V3_TURBO);
        }
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
        // `tts-1` is listed alongside `piper-plus` for the same reason
        // `whisper-1` is listed alongside `whisper-base`: the catalogue's
        // contract is "these ids are accepted", not "these engines are
        // distinct" (cc-38).
        let mut v = vec![model_names::PIPER_PLUS, model_names::TTS_1];
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
        // Voxtral is checked second — its M3-10 greedy path now returns a
        // Transcription on success (HTTP 200); mapping-side errors
        // (missing tokenizer, backend not covered, shape-only converter,
        // etc.) still bubble up as explicit ServiceError::Inference for
        // the T05 HTTP layer to render (FR-EX-08 — no silent fall-through
        // to Whisper).
        if let Some(engine) = self.resolve_asr(model) {
            return engine
                .transcribe(pcm)
                .map(|t| t.text)
                .map_err(ServiceError::Inference);
        }
        if let Some(engine) = self.resolve_voxtral(model) {
            return engine
                .transcribe(pcm)
                .map(|t| t.text)
                .map_err(ServiceError::Inference);
        }
        Err(ServiceError::UnknownModel(model.to_owned()))
    }

    /// Beam-search transcribe (M3-15 Wave 10 A / Wave 11 whisper wiring).
    /// Routing rules:
    ///
    /// * Whisper (`whisper-*`) — greedy for `beam_size == None || Some(0..=1)`
    ///   (via `AsrEngine::transcribe`). `beam_size > 1` now routes through
    ///   [`vokra_models::whisper::asr::WhisperAsr::transcribe_tokens_beam_nbest`]
    ///   returning the full n-best list (Wave 11 lift of the "not threaded
    ///   through AsrEngine" limitation), then detokenizes each hypothesis via
    ///   [`WhisperAsr::render_ids`]. `length_penalty` maps to the
    ///   [`vokra_core::decode::BeamSearchConfig::length_normalization`] α
    ///   attribute (HF `length_penalty`). `no_repeat_ngram` maps to
    ///   [`vokra_core::decode::BeamSearchConfig::no_repeat_ngram_size`] (Wave
    ///   12 core-side plumbing, M3-15 follow-up — was an explicit
    ///   `UnsupportedOp` under Wave 11 while the core-side field was absent).
    /// * Voxtral (`voxtral*`) — greedy on `beam_size <= 1`; `> 1` routes
    ///   through
    ///   [`vokra_models::voxtral::VoxtralAsr::transcribe_beam_with_config_overrides`],
    ///   which honours `length_penalty` and `no_repeat_ngram` (FR-EX-08 —
    ///   accepting the fields in the schema and silently ignoring them
    ///   would be a fabrication).
    /// * Unknown model — [`ServiceError::UnknownModel`].
    ///
    /// The `text` field of the response is always the top-1 (matches
    /// [`Self::transcribe`]). The `alternatives` list is empty on greedy
    /// (backward-compat) and ranked descending on beam.
    fn transcribe_beam(
        &self,
        model: &str,
        pcm: &[f32],
        req: &TranscribeBeamRequest,
    ) -> Result<TranscribeBeamResponse, ServiceError> {
        // ---- Whisper: greedy + beam wired end-to-end (M3-15 Wave 11;
        // M4-20 word timestamps). ----------------------------------------
        if let Some(engine) = self.resolve_asr(model) {
            let want_words = req.word_timestamps.unwrap_or(false);
            let beam_size_effective = req.beam_size.unwrap_or(0);

            // Greedy fast-path only when word timestamps are NOT requested:
            // greedy `transcribe` produces no cross-attention alignment, so a
            // caller who wants word timestamps must go through the beam/align
            // path even at width 1. Backward compat is otherwise preserved —
            // a request that sets neither `beam_size > 1` nor `word_timestamps`
            // is byte-for-byte the legacy greedy shape.
            if matches!(beam_size_effective, 0 | 1) && !want_words {
                let text = engine
                    .transcribe(pcm)
                    .map(|t| t.text)
                    .map_err(ServiceError::Inference)?;
                return Ok(TranscribeBeamResponse {
                    text,
                    alternatives: Vec::new(),
                    words: Vec::new(),
                });
            }

            // Beam / alignment path. `n == 1` is greedy-equivalent, used when
            // only word timestamps were requested (`beam_size <= 1`).
            let n = beam_size_effective.max(1);
            let cfg = whisper_beam_config(n, req, model)?;
            let hyps = engine
                .transcribe_tokens_beam_nbest(pcm, &cfg)
                .map_err(ServiceError::Inference)?;
            // FR-EX-08: an empty n-best list is a bug in the beam driver, not
            // a "return greedy" fallback.
            if hyps.is_empty() {
                return Err(ServiceError::Inference(VokraError::ModelLoad(
                    "transcribe_beam: whisper beam search produced no hypothesis".into(),
                )));
            }
            let best = &hyps[0];
            // Top-1 mirrored into `text` (backward-compat top-1 shape).
            let text = engine
                .render_ids(&best.tokens)
                .map_err(ServiceError::Inference)?;
            // `alternatives` stays empty for greedy-equivalent widths
            // (`beam_size <= 1`) so the legacy top-1 shape is preserved; it is
            // populated + ranked only for a genuine beam request (`> 1`).
            let alternatives = if beam_size_effective > 1 {
                let mut v: Vec<TranscribeAlternative> = Vec::with_capacity(hyps.len());
                for h in &hyps {
                    let t = engine
                        .render_ids(&h.tokens)
                        .map_err(ServiceError::Inference)?;
                    v.push(TranscribeAlternative {
                        text: t,
                        log_prob: h.score as f64,
                        length_normalized_score: h.normalized_score as f64,
                    });
                }
                v
            } else {
                Vec::new()
            };
            // Word timestamps come from the best (first) hypothesis only —
            // `beam_search` aligns only the best. Detokenize each word span
            // with the engine's tokenizer.
            let words = if want_words {
                whisper_word_timestamps(best, |ids| engine.render_ids(ids))?
            } else {
                Vec::new()
            };
            return Ok(TranscribeBeamResponse {
                text,
                alternatives,
                words,
            });
        }

        // ---- Voxtral: greedy + beam wired end-to-end. ------------------
        if let Some(engine) = self.resolve_voxtral(model) {
            // Word timestamps are a Whisper-only surface (cross-attention DTW,
            // M4-20). Voxtral has no such alignment here, so accepting the flag
            // and returning an empty list would be a silent fabrication —
            // reject it explicitly (FR-EX-08).
            if req.word_timestamps == Some(true) {
                return Err(ServiceError::Inference(VokraError::UnsupportedOp(format!(
                    "transcribe_beam: model `{model}` (voxtral backend) does not support \
                     word_timestamps (Whisper-only cross-attention alignment, M4-20)",
                ))));
            }
            let beam_size_effective = req.beam_size.unwrap_or(0);
            match beam_size_effective {
                0 | 1 => {
                    let text = engine
                        .transcribe(pcm)
                        .map(|t| t.text)
                        .map_err(ServiceError::Inference)?;
                    return Ok(TranscribeBeamResponse {
                        text,
                        alternatives: Vec::new(),
                        words: Vec::new(),
                    });
                }
                _ => {
                    // The Voxtral engine handles greedy-through-beam-config
                    // itself (beam_size == 1 is equivalent). Wire
                    // length_penalty (default 0.6 — matches BeamConfig::with_beam_size)
                    // and no_repeat_ngram (default 0) explicitly, do NOT
                    // silently coerce them.
                    let length_penalty = req.length_penalty.unwrap_or(0.6);
                    let no_repeat_ngram = req.no_repeat_ngram.unwrap_or(0);
                    let max_new = req.max_new_tokens.unwrap_or(0);
                    let beams = engine
                        .transcribe_beam_with_config_overrides(
                            pcm,
                            beam_size_effective,
                            length_penalty,
                            no_repeat_ngram,
                            max_new,
                        )
                        .map_err(ServiceError::Inference)?;
                    let alternatives: Vec<TranscribeAlternative> = beams
                        .iter()
                        .map(|b| TranscribeAlternative {
                            text: b.text.clone(),
                            log_prob: b.result.log_prob,
                            length_normalized_score: b.result.length_normalized_score,
                        })
                        .collect();
                    let text = alternatives
                        .first()
                        .map(|a| a.text.clone())
                        .unwrap_or_default();
                    return Ok(TranscribeBeamResponse {
                        text,
                        alternatives,
                        words: Vec::new(),
                    });
                }
            }
        }

        Err(ServiceError::UnknownModel(model.to_owned()))
    }
}

/// Builds the Whisper [`BeamSearchConfig`] from a beam request (M3-15 length
/// penalty / n-best / n-gram + M4-20 `word_timestamps`). Extracted from
/// [`InferenceService::transcribe_beam`] so the request→config passthrough —
/// including the M4-20 `word_timestamps` bit — is unit-testable without a real
/// engine.
///
/// # Errors
///
/// [`ServiceError::Inference`]([`VokraError::InvalidArgument`]) if
/// `length_penalty` is negative or non-finite (the GNMT ranking would be
/// undefined, FR-EX-08).
fn whisper_beam_config(
    n: usize,
    req: &TranscribeBeamRequest,
    model: &str,
) -> Result<BeamSearchConfig, ServiceError> {
    // length_penalty defaults to 1.0 to match `BeamSearchConfig::new` (HF
    // length_penalty), NOT the Voxtral default of 0.6 — the two engines use
    // different ranking primitives.
    let length_penalty = req.length_penalty.unwrap_or(1.0);
    if !length_penalty.is_finite() || length_penalty < 0.0 {
        return Err(ServiceError::Inference(VokraError::InvalidArgument(
            format!(
                "transcribe_beam: model `{model}` (whisper backend) requires length_penalty to be a \
             non-negative finite float (maps to BeamSearchConfig::length_normalization), got \
             {length_penalty}",
            ),
        )));
    }
    let max_new = req
        .max_new_tokens
        .filter(|&v| v > 0)
        .unwrap_or(WHISPER_DEFAULT_MAX_NEW_TOKENS);
    // `BeamSearchConfig::new(n, max_new)` seeds `length_normalization = 1.0`,
    // `n_best = 1`, `early_stopping = true`, `word_timestamps = false`,
    // `no_repeat_ngram_size = 0`. Overlay the caller fields.
    let mut cfg = BeamSearchConfig::new(n, max_new);
    cfg.length_normalization = length_penalty;
    cfg.n_best = n;
    cfg.no_repeat_ngram_size = req.no_repeat_ngram.unwrap_or(0);
    cfg.word_timestamps = req.word_timestamps.unwrap_or(false);
    Ok(cfg)
}

/// Maps a best hypothesis's word timings (M4-20) into the response DTO,
/// detokenizing each aligned token span with `detokenize`. The
/// [`vokra_core::decode::word_timing::WordTiming::token_start`] /
/// `token_end` are absolute indices into `hyp.tokens` (the Whisper consumer
/// emits them so, see `whisper::beam_glue`).
///
/// `detokenize` is passed as a closure (the caller supplies
/// [`WhisperAsr::render_ids`]) so the span→DTO mapping is unit-testable
/// without a real engine.
///
/// Returns an empty list when the hypothesis carries no alignment: when word
/// timestamps were requested, [`vokra_core::decode::beam_search`] already
/// raised the explicit FR-EX-08 error for a model without alignment heads
/// before we reach here, so this branch is only hit off the word-timestamp
/// path.
///
/// # Errors
///
/// [`ServiceError::Inference`] if a timing's token span is out of range
/// (an alignment/decoder bug, surfaced not swallowed) or detokenization fails.
fn whisper_word_timestamps(
    hyp: &BeamHypothesis,
    detokenize: impl Fn(&[u32]) -> vokra_core::Result<String>,
) -> Result<Vec<WordTimestamp>, ServiceError> {
    let Some(timings) = &hyp.word_timestamps else {
        return Ok(Vec::new());
    };
    let mut out = Vec::with_capacity(timings.len());
    for w in timings {
        let span = hyp.tokens.get(w.token_start..w.token_end).ok_or_else(|| {
            ServiceError::Inference(VokraError::InvalidArgument(format!(
                "transcribe_beam: word-timing token span {}..{} out of range for {} tokens",
                w.token_start,
                w.token_end,
                hyp.tokens.len(),
            )))
        })?;
        let word = detokenize(span).map_err(ServiceError::Inference)?;
        out.push(WordTimestamp {
            word,
            start: w.start as f64,
            end: w.end as f64,
        });
    }
    Ok(out)
}

impl SynthesizeService for InferenceService {
    fn synthesize(
        &self,
        model: &str,
        request: &SynthesisRequest,
    ) -> Result<SynthesizedAudio, ServiceError> {
        match model {
            // `tts-1` is the OpenAI stock alias for the default TTS engine
            // (cc-38) — same treatment as `whisper-1` on the ASR side.
            model_names::PIPER_PLUS | model_names::TTS_1 => {
                // cc-18 (2026-07-19 M4-residual audit): a `language` the voice
                // does not support must be an explicit error at the service
                // boundary. The engine's `synthesize_full` maps an unknown
                // code to `None` and silently falls back to the phonemizer's
                // detected language (crates/vokra-models/src/piper_plus/mod.rs
                // `language_id(..).unwrap_or(utt.lid)`) — honest surfaces
                // must reject BEFORE that fallback (FR-EX-08). The supported
                // set comes from the voice GGUF's own
                // `vokra.piper.language_codes` metadata.
                if let Some(lang) = request.language.as_deref() {
                    let cfg = self.tts_piper.config();
                    if cfg.language_id(lang).is_none() {
                        return Err(ServiceError::Inference(VokraError::InvalidArgument(
                            format!(
                                "synthesize: language `{lang}` is not supported by the loaded \
                                 piper-plus voice (supported: [{}]) — refusing to silently fall \
                                 back to the voice default (FR-EX-08)",
                                cfg.language_codes.join(", "),
                            ),
                        )));
                    }
                }
                self.tts_piper
                    .synthesize_full(request, self.phonemizer.as_ref())
                    .map_err(ServiceError::Inference)
            }
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

/// The one canonical "tokenizer sidecar without its GGUF" error, shared by
/// the up-front [`ServiceConfig::validate`] pass and the loader below so the
/// two can never word it differently.
///
/// `slot` is the hyphenated engine name (`whisper-large-v3`); the message
/// quotes the underscore FIELD names, which is what an operator reading a
/// TOML file or a `ServiceConfig` literal is looking at.
fn orphan_tokenizer_error(slot: &str, tokenizer: &Path) -> ServiceError {
    let field = slot.replace('-', "_");
    ServiceError::InvalidConfig(format!(
        "{field}_tokenizer set without {field}_gguf (orphan sidecar: {tokenizer:?}); \
         refusing to start with a half-configured slot"
    ))
}

/// Load an OPTIONAL Whisper slot (large-v3 + the cc-39 small / medium /
/// turbo sizes).
///
/// Returns `Ok(None)` when the slot is simply not configured. The
/// orphan-tokenizer arm is normally unreachable — [`ServiceConfig::validate`]
/// rejects that combination before `build` reads any file — but it is kept
/// rather than `unreachable!()`d so this function is total on its own terms:
/// a future caller cannot turn a half-configured slot into a silent `None`.
fn load_optional_whisper(
    slot: &'static str,
    gguf: Option<&Path>,
    tokenizer: Option<&Path>,
    backend: BackendKind,
) -> Result<Option<Arc<WhisperAsr>>, ServiceError> {
    match (gguf, tokenizer) {
        (Some(path), tok) => Ok(Some(Arc::new(load_whisper(slot, path, tok, backend)?))),
        (None, Some(tok)) => Err(orphan_tokenizer_error(slot, tok)),
        (None, None) => Ok(None),
    }
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

    /// Test double for the M3-10 Voxtral dispatch path: the real
    /// `InferenceService` needs a real GGUF to build, so we drive the
    /// TranscribeService trait directly to guard the intended routing
    /// (Voxtral aliases → the Voxtral engine → a `Transcription` payload
    /// on success). The real `VoxtralAsr::transcribe` returns
    /// `Ok(Transcription)` on any well-formed dispatch (T13 landed
    /// autoregressive decode + tokenizer); the double mirrors that
    /// shape.
    struct VoxtralAsFakeGreedy;
    impl TranscribeService for VoxtralAsFakeGreedy {
        fn transcribe(&self, model: &str, pcm: &[f32]) -> Result<String, ServiceError> {
            match model {
                model_names::WHISPER_1 | model_names::WHISPER_BASE => Ok(String::new()),
                model_names::VOXTRAL
                | model_names::VOXTRAL_MINI_3B
                | model_names::VOXTRAL_SMALL_24B => {
                    // Empty PCM must still route to InvalidArgument (matches
                    // real VoxtralAsr::transcribe surface).
                    if pcm.is_empty() {
                        return Err(ServiceError::Inference(VokraError::InvalidArgument(
                            "pcm slice is empty".into(),
                        )));
                    }
                    // Synthesized transcript: this is the 200 shape the M3-10
                    // greedy + tokenizer path produces. Content is a stub in
                    // the double; the real path produces LM-prior tokens
                    // (documented in `VoxtralAsr::transcribe`) until the audio
                    // adapter follow-up lands.
                    Ok(format!("voxtral://{model}/{}samples", pcm.len()))
                }
                other => Err(ServiceError::UnknownModel(other.to_owned())),
            }
        }
    }

    #[test]
    fn voxtral_dispatch_returns_ok_transcription_on_valid_pcm() {
        // Guards the M3-10 501 → 200 acceptance contract: a caller who
        // names a Voxtral alias reaches the Voxtral engine and, given a
        // non-empty PCM, receives Ok(String) — never a fabricated
        // NotImplemented (FR-EX-08 spirit — the T13 greedy path landed).
        let svc: Box<dyn TranscribeService> = Box::new(VoxtralAsFakeGreedy);
        let pcm = vec![0.1f32; 16_000]; // 1 s @ 16 kHz
        for alias in [
            model_names::VOXTRAL,
            model_names::VOXTRAL_MINI_3B,
            model_names::VOXTRAL_SMALL_24B,
        ] {
            let out = svc.transcribe(alias, &pcm).unwrap_or_else(|e| {
                panic!("alias `{alias}`: expected Ok, got error {e}");
            });
            assert!(
                !out.is_empty(),
                "alias `{alias}`: transcript must not be empty"
            );
        }
    }

    #[test]
    fn voxtral_dispatch_surfaces_invalid_argument_on_empty_pcm() {
        // 4xx path still routes cleanly — the 501 → 200 change must not
        // silently swallow validation errors.
        let svc: Box<dyn TranscribeService> = Box::new(VoxtralAsFakeGreedy);
        for alias in [
            model_names::VOXTRAL,
            model_names::VOXTRAL_MINI_3B,
            model_names::VOXTRAL_SMALL_24B,
        ] {
            let err = svc.transcribe(alias, &[]).unwrap_err();
            match err {
                ServiceError::Inference(VokraError::InvalidArgument(_)) => {}
                other => panic!("alias `{alias}`: expected InvalidArgument, got {other}"),
            }
        }
    }

    // -----------------------------------------------------------------
    // M3-15 Wave 10 A — beam-search dispatch + n-best response
    // -----------------------------------------------------------------
    //
    // Guards the beam-search surface added to `TranscribeService`:
    //
    //  1. Default trait method: `beam_size == None || Some(0..=1)` folds
    //     to greedy `transcribe`, `alternatives` empty (backward compat).
    //  2. Default trait method: `beam_size > 1` on a greedy-only engine
    //     surfaces `Inference(UnsupportedOp)` — FR-EX-08, no silent fall
    //     back to greedy.
    //  3. Override: an engine that supports beam search returns a
    //     populated `alternatives` list ranked descending, with the
    //     top-1 mirrored in `text`.
    //  4. `TranscribeBeamRequest` default is genuinely greedy (all
    //     fields None).
    //  5. Unknown model → `UnknownModel` (routing invariant preserved).

    /// Test double: both `voxtral*` and `whisper*` support beam search.
    /// Mirrors the shape the real `InferenceService::transcribe_beam`
    /// override implements after Wave 12:
    ///
    /// * Whisper — greedy on `beam_size <= 1`; beam decode on `> 1`,
    ///   `length_penalty` maps to `BeamSearchConfig::length_normalization`,
    ///   `no_repeat_ngram` maps to `BeamSearchConfig::no_repeat_ngram_size`
    ///   (Wave 12 core-side plumbing; the Wave 11 explicit `UnsupportedOp`
    ///   has been lifted).
    /// * Voxtral — greedy on `beam_size <= 1`; beam decode on `> 1`,
    ///   `length_penalty` and `no_repeat_ngram` both honoured.
    ///
    /// Emitted alternatives are synthesized but ranked (and, on whisper,
    /// carry both `length_penalty` and `no_repeat_ngram` in the top text so
    /// the honor tests can inspect the round-trip without a real engine).
    struct VoxtralAsBeamCapable;
    impl TranscribeService for VoxtralAsBeamCapable {
        fn transcribe(&self, model: &str, pcm: &[f32]) -> Result<String, ServiceError> {
            match model {
                model_names::WHISPER_1 | model_names::WHISPER_BASE => Ok("whisper-greedy".into()),
                model_names::VOXTRAL
                | model_names::VOXTRAL_MINI_3B
                | model_names::VOXTRAL_SMALL_24B => {
                    if pcm.is_empty() {
                        return Err(ServiceError::Inference(VokraError::InvalidArgument(
                            "pcm slice is empty".into(),
                        )));
                    }
                    Ok(format!("voxtral-greedy://{model}"))
                }
                other => Err(ServiceError::UnknownModel(other.to_owned())),
            }
        }

        fn transcribe_beam(
            &self,
            model: &str,
            pcm: &[f32],
            req: &TranscribeBeamRequest,
        ) -> Result<TranscribeBeamResponse, ServiceError> {
            // Whisper: beam supported. Wave 12 lifts the Wave 11
            // `no_repeat_ngram > 0` gate — the field now maps to
            // `BeamSearchConfig::no_repeat_ngram_size`.
            if matches!(model, model_names::WHISPER_1 | model_names::WHISPER_BASE) {
                let bs = req.beam_size.unwrap_or(0);
                if bs <= 1 {
                    return Ok(TranscribeBeamResponse {
                        text: self.transcribe(model, pcm)?,
                        alternatives: Vec::new(),
                        words: Vec::new(),
                    });
                }
                let ngram = req.no_repeat_ngram.unwrap_or(0);
                let length_penalty = req.length_penalty.unwrap_or(1.0);
                if !length_penalty.is_finite() || length_penalty < 0.0 {
                    return Err(ServiceError::Inference(VokraError::InvalidArgument(
                        format!(
                            "whisper beam: length_penalty must be non-negative finite float, got \
                         {length_penalty}",
                        ),
                    )));
                }
                // Emit `bs` synthetic ranked alternatives — the top-1 text
                // carries both `length_penalty` and `no_repeat_ngram` verbatim
                // so the honor tests can observe they flowed through
                // unchanged (FR-EX-08 no silent drop).
                let alternatives: Vec<TranscribeAlternative> = (0..bs)
                    .map(|i| TranscribeAlternative {
                        text: format!(
                            "whisper-beam[{i}]@lp={length_penalty}@ngram={ngram}://{model}",
                        ),
                        log_prob: -(i as f64),
                        length_normalized_score: -(i as f64) * 0.5,
                    })
                    .collect();
                let text = alternatives[0].text.clone();
                return Ok(TranscribeBeamResponse {
                    text,
                    alternatives,
                    words: Vec::new(),
                });
            }
            // Voxtral: beam supported (Wave 10).
            if matches!(
                model,
                model_names::VOXTRAL
                    | model_names::VOXTRAL_MINI_3B
                    | model_names::VOXTRAL_SMALL_24B
            ) {
                let bs = req.beam_size.unwrap_or(0);
                if bs <= 1 {
                    return Ok(TranscribeBeamResponse {
                        text: self.transcribe(model, pcm)?,
                        alternatives: Vec::new(),
                        words: Vec::new(),
                    });
                }
                // Emit `bs` synthetic ranked alternatives so the response
                // schema can be inspected.
                let alternatives: Vec<TranscribeAlternative> = (0..bs)
                    .map(|i| TranscribeAlternative {
                        text: format!("voxtral-beam[{i}]://{model}"),
                        // Rank descending: higher i → lower score.
                        log_prob: -(i as f64),
                        length_normalized_score: -(i as f64) * 0.5,
                    })
                    .collect();
                let text = alternatives[0].text.clone();
                return Ok(TranscribeBeamResponse {
                    text,
                    alternatives,
                    words: Vec::new(),
                });
            }
            Err(ServiceError::UnknownModel(model.to_owned()))
        }
    }

    #[test]
    fn beam_request_default_is_genuinely_greedy() {
        // A default-constructed request must be a greedy request — no
        // silent beam-size default. This is FR-EX-08 spirit at the
        // schema layer: a caller who never sets beam_size sees the
        // legacy greedy behaviour.
        let req = TranscribeBeamRequest::default();
        assert!(req.beam_size.is_none());
        assert!(req.length_penalty.is_none());
        assert!(req.no_repeat_ngram.is_none());
        assert!(req.max_new_tokens.is_none());
    }

    #[test]
    fn beam_default_trait_folds_to_greedy_when_beam_size_none_or_one() {
        // A greedy-only service (NoopTranscribe) whose default
        // `transcribe_beam` folds to greedy must accept:
        //   * None
        //   * Some(0)
        //   * Some(1)
        // and return an empty `alternatives` list — that is the backward
        // compat contract with legacy top-1 callers.
        let svc: Box<dyn TranscribeService> = Box::new(NoopTranscribe);
        for beam_size in [None, Some(0usize), Some(1usize)] {
            let req = TranscribeBeamRequest {
                beam_size,
                ..Default::default()
            };
            let resp = svc
                .transcribe_beam(model_names::WHISPER_BASE, &[], &req)
                .expect("greedy fold must succeed");
            assert!(
                resp.alternatives.is_empty(),
                "greedy fold must not populate alternatives (backward compat, beam_size = {beam_size:?})",
            );
            // NoopTranscribe returns empty; we assert on the shape.
            assert!(resp.text.is_empty());
        }
    }

    #[test]
    fn beam_default_trait_hard_errors_on_beam_size_gt_one() {
        // Default `transcribe_beam` on a greedy-only engine must NOT
        // silently downgrade beam_size > 1 to greedy. FR-EX-08 spirit.
        let svc: Box<dyn TranscribeService> = Box::new(NoopTranscribe);
        let req = TranscribeBeamRequest {
            beam_size: Some(4),
            ..Default::default()
        };
        let err = svc
            .transcribe_beam(model_names::WHISPER_BASE, &[], &req)
            .unwrap_err();
        match err {
            ServiceError::Inference(VokraError::UnsupportedOp(msg)) => {
                assert!(
                    msg.contains("beam"),
                    "error message must name beam search: {msg}"
                );
            }
            other => panic!(
                "expected Inference(UnsupportedOp), got {other} (no silent fall back to greedy)"
            ),
        }
    }

    #[test]
    fn beam_override_populates_ranked_alternatives_on_voxtral() {
        // The Voxtral-capable double emits `beam_size` synthetic
        // alternatives ranked descending, mirroring the real Voxtral
        // dispatch shape.
        let svc: Box<dyn TranscribeService> = Box::new(VoxtralAsBeamCapable);
        let pcm = vec![0.1f32; 16_000];
        for alias in [
            model_names::VOXTRAL,
            model_names::VOXTRAL_MINI_3B,
            model_names::VOXTRAL_SMALL_24B,
        ] {
            let req = TranscribeBeamRequest {
                beam_size: Some(3),
                length_penalty: Some(0.6),
                no_repeat_ngram: Some(2),
                max_new_tokens: Some(16),
                word_timestamps: None,
            };
            let resp = svc
                .transcribe_beam(alias, &pcm, &req)
                .expect("beam decode must succeed on voxtral alias");
            assert_eq!(resp.alternatives.len(), 3, "alias `{alias}`");
            // Ranked descending by length_normalized_score.
            for pair in resp.alternatives.windows(2) {
                assert!(
                    pair[0].length_normalized_score >= pair[1].length_normalized_score,
                    "alias `{alias}`: alternatives must be ranked descending",
                );
            }
            // Top-1 mirrored into `text`.
            assert_eq!(resp.text, resp.alternatives[0].text);
        }
    }

    #[test]
    fn beam_override_folds_to_greedy_on_beam_size_one_on_voxtral() {
        // A Voxtral-capable engine with beam_size = 1 must still fold to
        // greedy (no alternatives) — matches the top-1 legacy shape.
        let svc: Box<dyn TranscribeService> = Box::new(VoxtralAsBeamCapable);
        let pcm = vec![0.1f32; 16_000];
        let req = TranscribeBeamRequest {
            beam_size: Some(1),
            ..Default::default()
        };
        let resp = svc
            .transcribe_beam(model_names::VOXTRAL, &pcm, &req)
            .unwrap();
        assert!(resp.alternatives.is_empty());
        assert!(!resp.text.is_empty());
    }

    // -----------------------------------------------------------------
    // M3-15 Wave 11 — Whisper beam surface tests. Mirrors the shape of the
    // Wave 10 A Voxtral tests one-to-one; the mock (`VoxtralAsBeamCapable`)
    // implements whisper beam with the same semantic as the real
    // `InferenceService::transcribe_beam` override (Wave 11).
    // -----------------------------------------------------------------

    /// `beam_size = 1` must match greedy — the response carries the top-1
    /// text and an empty `alternatives` list (backward-compat with the
    /// pre-beam top-1 shape). Bit-identical to the greedy path.
    #[test]
    fn whisper_beam_size_1_matches_greedy() {
        let svc: Box<dyn TranscribeService> = Box::new(VoxtralAsBeamCapable);
        let pcm = vec![0.1f32; 16_000];
        let greedy_text = svc.transcribe(model_names::WHISPER_BASE, &pcm).unwrap();

        // beam_size = 1 -> same greedy text, no alternatives.
        for beam_size in [None, Some(0usize), Some(1usize)] {
            let req = TranscribeBeamRequest {
                beam_size,
                ..Default::default()
            };
            let resp = svc
                .transcribe_beam(model_names::WHISPER_BASE, &pcm, &req)
                .expect("beam_size <= 1 must fold to greedy");
            assert_eq!(
                resp.text, greedy_text,
                "beam_size = {beam_size:?} must match greedy tokens bit-identically",
            );
            assert!(
                resp.alternatives.is_empty(),
                "beam_size = {beam_size:?} must not populate alternatives (backward compat)",
            );
        }
    }

    /// `beam_size = 4` must return the full ranked n-best list, sorted
    /// descending by `length_normalized_score`, with the top-1 mirrored
    /// into `text`.
    #[test]
    fn whisper_beam_size_4_returns_ranked_alternatives() {
        let svc: Box<dyn TranscribeService> = Box::new(VoxtralAsBeamCapable);
        let pcm = vec![0.1f32; 16_000];
        let req = TranscribeBeamRequest {
            beam_size: Some(4),
            length_penalty: Some(1.0),
            no_repeat_ngram: None,
            max_new_tokens: None,
            word_timestamps: None,
        };
        let resp = svc
            .transcribe_beam(model_names::WHISPER_BASE, &pcm, &req)
            .expect("whisper beam must return n-best");
        assert_eq!(resp.alternatives.len(), 4);
        // Ranked descending by length_normalized_score.
        for pair in resp.alternatives.windows(2) {
            assert!(
                pair[0].length_normalized_score >= pair[1].length_normalized_score,
                "n-best must be ranked descending: {} vs {}",
                pair[0].length_normalized_score,
                pair[1].length_normalized_score,
            );
        }
        // Top-1 mirrored into `text`.
        assert_eq!(resp.text, resp.alternatives[0].text);
    }

    /// Wave 12 lift of the Wave 11 gate: `no_repeat_ngram > 0` on whisper is
    /// now dispatched into
    /// [`vokra_core::decode::BeamSearchConfig::no_repeat_ngram_size`] — the
    /// core-side beam search ported the mask from Voxtral. The test double
    /// echoes the value into the top-1 text so we can verify the field flowed
    /// through unchanged (guards against a silent drop / rename on the
    /// service→engine boundary, FR-EX-08).
    #[test]
    fn whisper_no_repeat_ngram_positive_now_honored() {
        let svc: Box<dyn TranscribeService> = Box::new(VoxtralAsBeamCapable);
        let pcm = vec![0.1f32; 16_000];
        let req = TranscribeBeamRequest {
            beam_size: Some(3),
            no_repeat_ngram: Some(2),
            ..Default::default()
        };
        let resp = svc
            .transcribe_beam(model_names::WHISPER_BASE, &pcm, &req)
            .expect(
                "no_repeat_ngram > 0 must now dispatch (Wave 12 core-side plumbing, M3-15 \
                 follow-up); the Wave 11 UnsupportedOp gate was lifted",
            );
        // Response shape matches the ranked-alternatives path (bs = 3).
        assert_eq!(resp.alternatives.len(), 3);
        for pair in resp.alternatives.windows(2) {
            assert!(pair[0].length_normalized_score >= pair[1].length_normalized_score);
        }
        assert!(
            resp.text.contains("ngram=2"),
            "no_repeat_ngram must reach the engine verbatim; got text `{}`",
            resp.text,
        );
        // Top-1 mirrored into `text`.
        assert_eq!(resp.text, resp.alternatives[0].text);
    }

    /// `no_repeat_ngram = 0` (and `None`, via the schema default) must
    /// produce the same response as any pre-Wave-12 caller that never set
    /// the field — i.e. bit-identical `text` and `alternatives` shape.
    /// Guards against a stray "if ngram >= 0" that would flip the mask on
    /// for the disabled case.
    #[test]
    fn whisper_no_repeat_ngram_zero_matches_previous_behavior() {
        let svc: Box<dyn TranscribeService> = Box::new(VoxtralAsBeamCapable);
        let pcm = vec![0.1f32; 16_000];
        // Baseline: pre-Wave-12 shape (no_repeat_ngram omitted entirely).
        let req_omitted = TranscribeBeamRequest {
            beam_size: Some(2),
            length_penalty: Some(0.6),
            ..Default::default()
        };
        let resp_omitted = svc
            .transcribe_beam(model_names::WHISPER_BASE, &pcm, &req_omitted)
            .expect("baseline (ngram omitted) must succeed");
        // Wave 12: explicit `no_repeat_ngram = 0` — must be bit-identical.
        let req_zero = TranscribeBeamRequest {
            beam_size: Some(2),
            length_penalty: Some(0.6),
            no_repeat_ngram: Some(0),
            ..Default::default()
        };
        let resp_zero = svc
            .transcribe_beam(model_names::WHISPER_BASE, &pcm, &req_zero)
            .expect("no_repeat_ngram = 0 must succeed");
        assert_eq!(
            resp_omitted.text, resp_zero.text,
            "no_repeat_ngram = 0 must produce the same text as omitting the field",
        );
        assert_eq!(
            resp_omitted.alternatives.len(),
            resp_zero.alternatives.len(),
            "no_repeat_ngram = 0 must produce the same alternatives shape as omitting the field",
        );
        for (a, b) in resp_omitted
            .alternatives
            .iter()
            .zip(resp_zero.alternatives.iter())
        {
            assert_eq!(a.text, b.text);
            assert_eq!(a.log_prob, b.log_prob);
            assert_eq!(a.length_normalized_score, b.length_normalized_score);
        }
    }

    /// A finite, non-negative `length_penalty` must actually reach the
    /// engine's `BeamSearchConfig::length_normalization` — the mock echoes
    /// the value verbatim into the top-1 text so the assertion can
    /// observe the round-trip (guards against silently ignoring the field
    /// — FR-EX-08).
    #[test]
    fn whisper_length_penalty_finite_is_honored() {
        let svc: Box<dyn TranscribeService> = Box::new(VoxtralAsBeamCapable);
        let pcm = vec![0.1f32; 16_000];
        let req = TranscribeBeamRequest {
            beam_size: Some(2),
            length_penalty: Some(0.7),
            ..Default::default()
        };
        let resp = svc
            .transcribe_beam(model_names::WHISPER_BASE, &pcm, &req)
            .expect("finite non-negative length_penalty must be honoured");
        assert!(
            resp.text.contains("lp=0.7"),
            "length_penalty must reach the engine verbatim; got text `{}`",
            resp.text,
        );

        // Negative length_penalty must hard-error — the ranking would be
        // undefined.
        let req_neg = TranscribeBeamRequest {
            beam_size: Some(2),
            length_penalty: Some(-0.1),
            ..Default::default()
        };
        let err = svc
            .transcribe_beam(model_names::WHISPER_BASE, &pcm, &req_neg)
            .unwrap_err();
        match err {
            ServiceError::Inference(VokraError::InvalidArgument(msg)) => {
                assert!(
                    msg.contains("length_penalty"),
                    "error message must name length_penalty: {msg}",
                );
            }
            other => panic!("expected InvalidArgument, got {other}"),
        }
    }

    #[test]
    fn beam_override_unknown_model_never_silently_succeeds() {
        // A caller who names an unknown model on the beam surface must
        // still hit UnknownModel — the beam path must not fabricate a
        // route around the greedy check.
        let svc: Box<dyn TranscribeService> = Box::new(VoxtralAsBeamCapable);
        let pcm = vec![0.1f32; 16_000];
        let req = TranscribeBeamRequest {
            beam_size: Some(2),
            ..Default::default()
        };
        let err = svc.transcribe_beam("elevenlabs", &pcm, &req).unwrap_err();
        match err {
            ServiceError::UnknownModel(m) => assert_eq!(m, "elevenlabs"),
            other => panic!("expected UnknownModel, got {other}"),
        }
    }

    // -----------------------------------------------------------------
    // M4-20 — word-timestamp request → BeamSearchConfig → response DTO
    // -----------------------------------------------------------------

    /// The M4-20 wiring the real `InferenceService` whisper branch relies on:
    /// `req.word_timestamps` must reach `BeamSearchConfig::word_timestamps`.
    /// `whisper_beam_config` is the exact helper the branch calls, so this
    /// exercises the real passthrough (incl. the M3-15 length-penalty /
    /// n-gram / n-best fields) without needing a real Whisper GGUF.
    #[test]
    fn whisper_beam_config_passes_word_timestamps_and_beam_fields() {
        // word_timestamps = Some(true) flows into the config flag.
        let req = TranscribeBeamRequest {
            word_timestamps: Some(true),
            ..Default::default()
        };
        let cfg = whisper_beam_config(3, &req, model_names::WHISPER_BASE).unwrap();
        assert!(cfg.word_timestamps, "word_timestamps must reach the config");
        assert_eq!(cfg.beam_width, 3);

        // Defaults off, and the other beam knobs still pass through.
        let req2 = TranscribeBeamRequest {
            beam_size: Some(4),
            length_penalty: Some(0.7),
            no_repeat_ngram: Some(2),
            ..Default::default()
        };
        let cfg2 = whisper_beam_config(4, &req2, model_names::WHISPER_BASE).unwrap();
        assert!(!cfg2.word_timestamps, "word_timestamps defaults off");
        assert_eq!(cfg2.n_best, 4);
        assert_eq!(cfg2.no_repeat_ngram_size, 2);
        assert!((cfg2.length_normalization - 0.7).abs() < 1e-6);

        // A negative length_penalty is still rejected (FR-EX-08).
        let bad = TranscribeBeamRequest {
            length_penalty: Some(-0.5),
            ..Default::default()
        };
        assert!(matches!(
            whisper_beam_config(2, &bad, model_names::WHISPER_BASE),
            Err(ServiceError::Inference(VokraError::InvalidArgument(_))),
        ));
    }

    /// The span→DTO mapping (`whisper_word_timestamps`) turns the best
    /// hypothesis's per-word `WordTiming`s into `WordTimestamp`s, detokenizing
    /// each absolute token span. Driven with a synthetic hypothesis + a fake
    /// detokenizer, so it proves the wiring independent of any real model.
    #[test]
    fn whisper_word_timestamps_maps_spans_to_dto() {
        use vokra_core::decode::word_timing::WordTiming;
        // prefix (1) + 3 content tokens + eot; two words spanning [1,3) and [3,4).
        let hyp = BeamHypothesis {
            tokens: vec![50258, 100, 101, 102, 50257],
            score: -1.0,
            normalized_score: -0.5,
            word_timestamps: Some(vec![
                WordTiming {
                    token_start: 1,
                    token_end: 3,
                    start: 0.0,
                    end: 0.5,
                },
                WordTiming {
                    token_start: 3,
                    token_end: 4,
                    start: 0.5,
                    end: 1.0,
                },
            ]),
        };
        // Fake detokenizer: joins the span ids so we can see which span each
        // word covers (a real engine would return words).
        let words = whisper_word_timestamps(&hyp, |ids| {
            Ok(ids.iter().map(u32::to_string).collect::<Vec<_>>().join("_"))
        })
        .unwrap();
        assert_eq!(
            words,
            vec![
                WordTimestamp {
                    word: "100_101".into(),
                    start: 0.0,
                    end: 0.5,
                },
                WordTimestamp {
                    word: "102".into(),
                    start: 0.5,
                    end: 1.0,
                },
            ],
        );
    }

    /// No alignment on the hypothesis → an empty list (not an error): the
    /// explicit FR-EX-08 error for a model without alignment heads is raised
    /// upstream in `beam_search`, before this mapping runs.
    #[test]
    fn whisper_word_timestamps_none_alignment_is_empty() {
        let hyp = BeamHypothesis {
            tokens: vec![1, 2, 3],
            score: 0.0,
            normalized_score: 0.0,
            word_timestamps: None,
        };
        let words = whisper_word_timestamps(&hyp, |_| Ok(String::new())).unwrap();
        assert!(words.is_empty());
    }

    /// An out-of-range token span surfaces an explicit error — never a
    /// silently-truncated or fabricated word (FR-EX-08).
    #[test]
    fn whisper_word_timestamps_out_of_range_span_errors() {
        use vokra_core::decode::word_timing::WordTiming;
        let hyp = BeamHypothesis {
            tokens: vec![1, 2],
            score: 0.0,
            normalized_score: 0.0,
            word_timestamps: Some(vec![WordTiming {
                token_start: 1,
                token_end: 5, // past the end of `tokens`
                start: 0.0,
                end: 1.0,
            }]),
        };
        let err = whisper_word_timestamps(&hyp, |_| Ok(String::new())).unwrap_err();
        assert!(matches!(
            err,
            ServiceError::Inference(VokraError::InvalidArgument(_)),
        ));
    }

    /// Test double honouring `word_timestamps` the way the real whisper branch
    /// does — surfacing per-word entries only when the flag is set. Proves the
    /// request field flows all the way to `TranscribeBeamResponse::words`
    /// independent of a real model (the "synthetic scorer returns Some(timings)"
    /// wiring check).
    struct WhisperWordTsCapable;
    impl TranscribeService for WhisperWordTsCapable {
        fn transcribe(&self, model: &str, _pcm: &[f32]) -> Result<String, ServiceError> {
            match model {
                model_names::WHISPER_1 | model_names::WHISPER_BASE => Ok("hello world".into()),
                other => Err(ServiceError::UnknownModel(other.to_owned())),
            }
        }

        fn transcribe_beam(
            &self,
            model: &str,
            _pcm: &[f32],
            req: &TranscribeBeamRequest,
        ) -> Result<TranscribeBeamResponse, ServiceError> {
            if !matches!(model, model_names::WHISPER_1 | model_names::WHISPER_BASE) {
                return Err(ServiceError::UnknownModel(model.to_owned()));
            }
            let words = if req.word_timestamps.unwrap_or(false) {
                vec![
                    WordTimestamp {
                        word: "hello".into(),
                        start: 0.0,
                        end: 0.5,
                    },
                    WordTimestamp {
                        word: "world".into(),
                        start: 0.5,
                        end: 1.0,
                    },
                ]
            } else {
                Vec::new()
            };
            Ok(TranscribeBeamResponse {
                text: "hello world".into(),
                alternatives: Vec::new(),
                words,
            })
        }
    }

    #[test]
    fn word_timestamps_request_flows_to_response_words() {
        let svc: Box<dyn TranscribeService> = Box::new(WhisperWordTsCapable);
        let pcm = vec![0.0f32; 16];

        // Default request (word_timestamps None) → empty words (backward compat).
        let r0 = svc
            .transcribe_beam(
                model_names::WHISPER_BASE,
                &pcm,
                &TranscribeBeamRequest::default(),
            )
            .unwrap();
        assert!(
            r0.words.is_empty(),
            "a request that never sets word_timestamps must surface no words",
        );

        // word_timestamps = Some(true) → the response carries per-word entries.
        let req = TranscribeBeamRequest {
            word_timestamps: Some(true),
            ..Default::default()
        };
        let r1 = svc
            .transcribe_beam(model_names::WHISPER_BASE, &pcm, &req)
            .unwrap();
        assert_eq!(r1.words.len(), 2, "word_timestamps must surface words");
        assert_eq!(r1.words[0].word, "hello");
        assert_eq!(r1.words[1].word, "world");
        assert!(r1.words[0].start <= r1.words[0].end);
    }

    #[test]
    fn beam_response_types_are_send_sync() {
        // The beam response propagates across HTTP handlers, so keeping
        // Send + Sync on the payload is load-bearing.
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<TranscribeBeamRequest>();
        assert_send_sync::<TranscribeBeamResponse>();
        assert_send_sync::<TranscribeAlternative>();
        assert_send_sync::<WordTimestamp>();
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

// ---------------------------------------------------------------------------
// cc-39 / cc-30 — whisper size slots + per-model backend selection.
// (2026-07-19 M4-residual audit.)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod size_slots_and_backends {
    //! Plumbing tests that need no weights: alias stability, the
    //! no-substitution rule for unconfigured sizes, slot→backend resolution,
    //! and the two config-rejection paths (uncompiled backend, un-honourable
    //! override).
    //!
    //! The real-GGUF counterpart (a wired server actually answering on each
    //! slot) lives in `tests/real_gguf_slots.rs`, env-gated — see cc-40.

    use super::*;
    use std::path::PathBuf;

    fn cfg() -> ServiceConfig {
        ServiceConfig::minimum(
            PathBuf::from("/tmp/base.gguf"),
            PathBuf::from("/tmp/piper.gguf"),
        )
    }

    #[test]
    fn cc39_size_aliases_are_stable_and_distinct() {
        assert_eq!(model_names::WHISPER_SMALL, "whisper-small");
        assert_eq!(model_names::WHISPER_MEDIUM, "whisper-medium");
        assert_eq!(model_names::WHISPER_TURBO, "whisper-turbo");
        assert_eq!(
            model_names::WHISPER_LARGE_V3_TURBO,
            "whisper-large-v3-turbo"
        );
        // Every advertised ASR alias must be unique — two names collapsing to
        // one string would make the catalogue lie about what it accepts.
        let all = [
            model_names::WHISPER_1,
            model_names::WHISPER_BASE,
            model_names::WHISPER_SMALL,
            model_names::WHISPER_MEDIUM,
            model_names::WHISPER_TURBO,
            model_names::WHISPER_LARGE_V3_TURBO,
            model_names::WHISPER_LARGE_V3,
        ];
        let mut sorted = all.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), all.len(), "aliases must be distinct: {all:?}");
    }

    #[test]
    fn cc39_minimum_config_leaves_every_size_slot_absent() {
        let c = cfg();
        assert!(c.whisper_small_gguf.is_none());
        assert!(c.whisper_medium_gguf.is_none());
        assert!(c.whisper_turbo_gguf.is_none());
        assert!(c.whisper_small_tokenizer.is_none());
        assert!(c.whisper_medium_tokenizer.is_none());
        assert!(c.whisper_turbo_tokenizer.is_none());
    }

    /// RED-LINE (cc-39): a tokenizer sidecar with no matching GGUF is a
    /// misconfiguration for EVERY optional size, not just large-v3. Silently
    /// ignoring it would hide a typo'd path.
    #[test]
    fn cc39_tokenizer_without_gguf_is_rejected_for_every_size() {
        // `WhisperAsr` is not `Debug`, so the Ok arm cannot go through
        // `expect_err` — match instead.
        for slot in ["whisper-small", "whisper-medium", "whisper-turbo"] {
            match load_optional_whisper(
                slot,
                None,
                Some(Path::new("/tmp/orphan.tok")),
                BackendKind::Cpu,
            ) {
                Err(ServiceError::InvalidConfig(msg)) => {
                    let field = slot.replace('-', "_");
                    assert!(
                        msg.contains(&field) && msg.contains("orphan.tok"),
                        "{slot}: the error must name the field and the orphan path; got: {msg}"
                    );
                }
                Err(other) => panic!("{slot}: expected InvalidConfig, got {other:?}"),
                Ok(_) => panic!("{slot}: orphan tokenizer must be rejected, not accepted"),
            }
        }
        // Neither set = simply not configured, which is fine.
        match load_optional_whisper("test-slot", None, None, BackendKind::Cpu) {
            Ok(None) => {}
            Ok(Some(_)) => panic!("an unconfigured slot must not produce an engine"),
            Err(e) => panic!("an unconfigured slot is not an error, got {e:?}"),
        }
    }

    /// The orphan-sidecar check must fire BEFORE any file is opened, for
    /// every size — otherwise a config typo is masked by whichever GGUF
    /// happens to fail to load first. This is the cc-39 generalisation of
    /// `registry::build_rejects_large_v3_tokenizer_without_gguf`, which pins
    /// the same ordering for large-v3; both paths are non-existent here, so
    /// only the up-front validation can produce `InvalidConfig`.
    #[test]
    fn cc39_orphan_tokenizer_is_caught_before_any_file_is_read() {
        /// (field stem, setter for that size's tokenizer sidecar).
        type SidecarSetter = (&'static str, fn(&mut ServiceConfig, PathBuf));
        let sizes: [SidecarSetter; 3] = [
            ("whisper_small", |c, p| c.whisper_small_tokenizer = Some(p)),
            ("whisper_medium", |c, p| {
                c.whisper_medium_tokenizer = Some(p)
            }),
            ("whisper_turbo", |c, p| c.whisper_turbo_tokenizer = Some(p)),
        ];
        for (field, set) in sizes {
            let mut c = cfg(); // both required paths point at /tmp/*.gguf, which do not exist
            set(&mut c, PathBuf::from("/nonexistent/tok.bin"));
            match InferenceService::build(&c) {
                Err(ServiceError::InvalidConfig(msg)) => {
                    assert!(
                        msg.contains(&format!("{field}_tokenizer"))
                            && msg.contains(&format!("{field}_gguf")),
                        "{field}: the config error must name both fields; got: {msg}"
                    );
                }
                Err(other) => panic!(
                    "{field}: config validation must precede file I/O, but got a load error: \
                     {other}"
                ),
                Ok(_) => panic!("{field}: orphan sidecar must be rejected"),
            }
        }
    }

    /// cc-30: a slot with no override inherits the global default; a slot
    /// with one uses it. Silero is the documented exception.
    #[test]
    fn cc30_backend_for_resolves_override_then_default() {
        let mut c = cfg();
        c.backend = BackendKind::Metal;
        assert_eq!(c.backend_for(ModelSlot::WhisperBase), BackendKind::Metal);
        assert_eq!(c.backend_for(ModelSlot::PiperPlus), BackendKind::Metal);

        c.backend_overrides
            .set(ModelSlot::WhisperBase, BackendKind::Cpu);
        assert_eq!(
            c.backend_for(ModelSlot::WhisperBase),
            BackendKind::Cpu,
            "an override must beat the global default"
        );
        assert_eq!(
            c.backend_for(ModelSlot::PiperPlus),
            BackendKind::Metal,
            "other slots keep the global default"
        );

        // Silero has no backend selector at all — it is CPU whatever the
        // global default says (and the startup path announces that).
        assert_eq!(
            c.backend_for(ModelSlot::SileroVad),
            BackendKind::Cpu,
            "silero-vad has no `with_backend`; it can only be CPU"
        );
    }

    /// cc-30 FR-EX-08: an override the runtime cannot honour is rejected,
    /// never accepted-then-ignored.
    #[test]
    fn cc30_silero_backend_override_is_an_explicit_error() {
        let mut c = cfg();
        assert!(c.validate_backend_overrides().is_ok(), "no override = fine");

        c.backend_overrides
            .set(ModelSlot::SileroVad, BackendKind::Metal);
        let err = c
            .validate_backend_overrides()
            .expect_err("silero override must be rejected");
        let msg = err.to_string();
        assert!(
            matches!(err, ServiceError::InvalidConfig(_)),
            "expected InvalidConfig, got {err:?}"
        );
        assert!(
            msg.contains("silero-vad") && msg.contains("no backend selector"),
            "the error must name the slot and the reason; got: {msg}"
        );
    }

    /// cc-30: CPU is always compiled in, so it always probes clean. This is
    /// the control for the negative case below.
    #[test]
    fn cc30_cpu_backend_is_always_available() {
        ensure_backend_available(BackendKind::Cpu).expect("cpu must always be selectable");
    }

    /// cc-30 core contract: a backend that was NOT compiled into this binary
    /// is an explicit error naming the Cargo feature to rebuild with — never
    /// a silent downgrade to CPU.
    ///
    /// Which backends are compiled in depends on this build's features, so
    /// the test asserts the *shape* of the outcome per backend rather than
    /// hard-coding a verdict: either the probe succeeds (feature on, device
    /// present) or it fails with an actionable rebuild instruction. A silent
    /// `Ok` for a backend that cannot run is the only forbidden outcome, and
    /// that is exactly what `Compute::for_backend` rules out.
    #[test]
    fn cc30_uncompiled_backend_error_names_its_cargo_feature() {
        for (backend, feature) in [
            (BackendKind::Metal, "metal"),
            (BackendKind::Cuda, "cuda"),
            (BackendKind::Vulkan, "vulkan"),
        ] {
            match ensure_backend_available(backend) {
                Ok(()) => {
                    // Compiled in AND the device opened. Nothing to assert
                    // beyond "it did not lie" — the engines will really run
                    // there. Printed so `--nocapture` records WHICH arm this
                    // build took (the assertion alone cannot tell you).
                    eprintln!("cc30: {backend:?} probe = AVAILABLE (feature `{feature}` on)");
                }
                Err(e) => {
                    let msg = e.to_string();
                    assert!(
                        msg.contains(&format!("--features {feature}")),
                        "{backend:?} rejection must tell the operator which feature to \
                         rebuild with; got: {msg}"
                    );
                    assert!(
                        msg.contains("FR-EX-08"),
                        "{backend:?} rejection must state the no-fallback rule; got: {msg}"
                    );
                    eprintln!("cc30: {backend:?} probe = REJECTED (actionable): {msg}");
                }
            }
        }
    }

    /// The default build is CPU-only, so `ServiceConfig::minimum` must keep
    /// selecting CPU — cc-30 adds a knob, it does not change the default.
    #[test]
    fn cc30_minimum_config_still_defaults_to_cpu_everywhere() {
        let c = cfg();
        assert_eq!(c.backend, BackendKind::Cpu);
        assert!(c.backend_overrides.is_empty());
        for slot in ModelSlot::ALL {
            assert_eq!(
                c.backend_for(slot),
                BackendKind::Cpu,
                "slot {} must default to CPU",
                slot.as_str()
            );
        }
    }
}
