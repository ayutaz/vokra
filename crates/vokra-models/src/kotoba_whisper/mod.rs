//! Kotoba Technologies **kotoba-whisper** — Whisper large-v3 encoder +
//! a 2-layer decoder distilled on Japanese ReazonSpeech audio (SoTA plan
//! Phase 5 JA-ASR-2, 2026-07-24).
//!
//! # What kotoba-whisper is (primary source)
//!
//! `kotoba-tech/kotoba-whisper-v2.0` is a Japanese-distilled Whisper
//! checkpoint: the **large-v3 encoder is kept intact** (32 layers,
//! d_model=1280, n_mels=128, encoder_attention_heads=20) and the
//! **decoder is shrunk to 2 layers** (same width / head count as
//! large-v3). The tokenizer is the large-v3 multilingual byte-level
//! BPE (`vocab_size=51866`, `eos_token_id=50257`,
//! `decoder_start_token_id=50258`).
//!
//! Every hparam below is transcribed **verbatim** from
//! `huggingface.co/kotoba-tech/kotoba-whisper-v2.0/raw/main/config.json`
//! (fetched 2026-07-24 — CLAUDE.md「ハルシネーション厳禁」):
//!
//! - **Model type** (`model_type`): `"whisper"`
//!   (`architectures = ["WhisperForConditionalGeneration"]`). The
//!   underlying architecture is Whisper, only the decoder depth changes,
//!   which is why this module is a thin scaffold over the existing
//!   [`crate::whisper`] plumbing — every op / kernel / attention detail
//!   is shared verbatim.
//! - **Encoder** (identical to Whisper `large-v3`):
//!   - `d_model` = 1280,
//!   - `encoder_layers` = 32,
//!   - `encoder_attention_heads` = 20 (`head_dim = 1280 / 20 = 64`,
//!     the Whisper family invariant),
//!   - `encoder_ffn_dim` = 5120,
//!   - `num_mel_bins` = 128 (large-v3 log-mel front-end),
//!   - `max_source_positions` = 1500 (encoder positional length —
//!     `n_audio_ctx`).
//! - **Decoder** (the distil axis — this is where it differs from
//!   large-v3):
//!   - `decoder_layers` = **2** (large-v3 has 32),
//!   - `decoder_attention_heads` = 20 (unchanged),
//!   - `decoder_ffn_dim` = 5120 (unchanged),
//!   - `max_target_positions` = 448 (decoder positional length —
//!     `n_text_ctx`).
//! - **Tokenizer**:
//!   - `vocab_size` = 51866 (large-v3 multilingual +1 for `<|yue|>`),
//!   - `eos_token_id` = 50257 (`<|endoftext|>`),
//!   - `decoder_start_token_id` = 50258 (`<|startoftranscript|>`),
//!   - `pad_token_id` = 50256.
//! - **Audio boundary**: `sample_rate` = 16 000 (Whisper convention).
//! - **Weight license**: **Apache-2.0** per every HF model card in the
//!   family (`kotoba-tech/kotoba-whisper-v1.0`, `-v1.1`, `-v2.0`,
//!   `-v2.1`, `-bilingual-v1.0`) — resolves to
//!   [`vokra_core::LicenseClass::Permissive`] via the `kotoba-whisper-`
//!   family walk, so the M2-13 gate passes commercially without any
//!   attribution obligation on the runtime side.
//!
//! # Distinct from distil-whisper (same shape, different upstream)
//!
//! `kotoba-whisper` and `distil-whisper/distil-large-v3.5` share the
//! **exact same architectural shape** (`d_model=1280`, `n_audio_layer=32`,
//! `n_text_layer=2`, `n_mels=128`, `vocab_size=51866`,
//! `ffn_dim=5120`). Both are Whisper large-v3 with a shrunk 2-layer
//! decoder. The differentiators are:
//!
//! - **Distillation corpus**: kotoba-whisper is distilled on
//!   ReazonSpeech Japanese audio (7000+ hours); distil-large-v3.5
//!   is distilled on multilingual audio.
//! - **License**: kotoba-whisper is **Apache-2.0** (upstream); distil-
//!   whisper is **MIT** (upstream). The compliance registry resolves
//!   both to Permissive, but the GGUF provenance stamp differs.
//! - **Upstream identifier**: `kotoba-tech/kotoba-whisper-vN.N` vs
//!   `distil-whisper/distil-large-v3.5`.
//! - **Language specialization**: kotoba-whisper is Japanese-specialized
//!   (JA CER 6-9% on the eval triad); distil-whisper covers many
//!   languages with more moderate quality per language.
//!
//! Because the *runtime forward* is identical (same op inventory, same
//! tensor names, same shape flow), both share the [`crate::whisper`]
//! plumbing. This module carries the primary-source config + a distinct
//! arch tag ("kotoba-whisper") so provenance / telemetry / model cards
//! label the loaded model correctly.
//!
//! # Very-cheap follow-on — reuses Whisper verbatim
//!
//! Because the topology is a Whisper checkpoint whose only difference is
//! `n_text_layer = 2`, kotoba-whisper **does not add any new op**
//! (`vokra-ops`) or backend kernel: the same STFT / mel filterbank / GEMM /
//! GEMV / softmax / layer-norm / GELU / conv1d inventory Whisper base
//! consumes (see [`crate::whisper`] docstring §Operator inventory) is
//! also what kotoba-whisper uses. The runtime forward is a follow-up
//! wave (T29-equivalent — the Moshi / CSM / Zonos / Kyutai STT /
//! Parakeet-CTC / distil-whisper pattern): when it lands it will
//! delegate to [`crate::whisper::WhisperModel`] with an appropriately-
//! shrunk `WhisperConfig`, since the checkpoint's tensor names follow
//! the upstream HF Whisper convention verbatim
//! (`model.encoder.layers.*` / `model.decoder.layers.*`) and the
//! converter (`vokra-convert::models::kotoba_whisper`) writes them
//! through unchanged.
//!
//! # What lands in this Phase 5 slice
//!
//! - [`KotobaWhisperConfig`] — every hparam transcribed from the primary
//!   source (v2.0 canonical), plus a `distil_invariant` sanity check
//!   (`n_text_layer < n_audio_layer`) that catches a checkpoint whose
//!   decoder depth was left at the source (large-v3 = 32) instead of the
//!   shrunk kotoba count.
//! - [`KotobaWhisperAsr`] — engine handle with two construction paths.
//!   [`KotobaWhisperAsr::from_gguf`] binds a converted GGUF through
//!   [`crate::whisper::WhisperAsr`] and [`KotobaWhisperAsr::transcribe`]
//!   then runs the **real** forward (log-mel front-end → 32-layer
//!   encoder → 2-layer decoder → BPE detokenize), shared verbatim with
//!   vanilla Whisper; the [`AsrEngine`] impl below exposes the same
//!   forward behind the session facade. The config-only shell
//!   [`KotobaWhisperAsr::new`] is the only path that hard-errors with
//!   [`VokraError::NotImplemented`] — it binds no weights and exists to
//!   exercise shape / invariant flow.
//!
//! # No ONNX (permanent)
//!
//! `kotoba-tech/kotoba-whisper-*` ships PyTorch safetensors; the
//! pipeline is re-implemented natively via [`crate::whisper`]
//! (whisper.cpp 型, CLAUDE.md 設計判断 4). This module never touches ONNX.

use vokra_core::engines::AsrEngine;
use vokra_core::gguf::GgufFile;
use vokra_core::tasks::Transcription;
use vokra_core::{BackendKind, Result, VokraError};

use crate::whisper::{WhisperAsr, WhisperTokenizer};

#[cfg(feature = "coreml")]
use crate::whisper::CoreMlArtifact;

/// `vokra.model.arch` a kotoba-whisper GGUF must carry. Written by
/// `vokra-convert::models::kotoba_whisper::ARCH`; the compliance registry
/// (`vokra_core::compliance`) knows `kotoba-whisper` / `kotoba-whisper-v1.0` /
/// `kotoba-whisper-v2.0` (and every family variant that lands later) as
/// [`vokra_core::LicenseClass::Permissive`] via the `kotoba-whisper-`
/// family prefix walk (apache-2.0 — the M2-13 gate passes commercially).
///
/// This arch string is intentionally **distinct** from Whisper's
/// (`"whisper"`) and distil-whisper's (`"distil-whisper"`) so the runtime
/// can label the loaded model correctly in telemetry / logs / model cards
/// while still consuming the same `vokra.whisper.*` hparam chunk schema
/// and Whisper decoder plumbing — the "very cheap follow-on" contract
/// in the task.
pub const EXPECTED_ARCH: &str = "kotoba-whisper";

/// PCM sample rate kotoba-whisper expects. Same as vanilla Whisper —
/// 16 kHz mono, per the openai/whisper convention (not written directly
/// in `config.json` but inherited from the Whisper feature extractor
/// preprocessor).
pub const KOTOBA_WHISPER_SAMPLE_RATE: u32 = 16_000;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// kotoba-whisper architectural hyperparameters.
///
/// A deliberate subset of the Whisper `config.json` schema — every field
/// maps 1-to-1 to the corresponding Whisper axis (see
/// [`crate::whisper::WhisperConfig`]). The distinguishing invariant is
/// **`n_text_layer < n_audio_layer`** ("the decoder is smaller than the
/// encoder"), enforced by [`Self::validate_for_forward`] — a real
/// non-distil checkpoint would have `n_text_layer == n_audio_layer`
/// (Whisper base…large-v3) or `n_text_layer < n_audio_layer` only for
/// the turbo variant (which is a separate arch, `"whisper"`, at
/// `whisper-turbo`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KotobaWhisperConfig {
    /// Mel input channels (encoder conv1 in-channels). **128** for
    /// kotoba-whisper v1.x/v2.x (matching large-v3's 128-bin front-end).
    pub n_mels: usize,
    /// Hidden width `d_model` shared by encoder and decoder — 1280 for
    /// kotoba-whisper v1.x/v2.x.
    pub d_model: usize,
    /// Encoder positional length (`max_source_positions`), 1500.
    pub n_audio_ctx: usize,
    /// Encoder attention heads — 20 for kotoba-whisper v1.x/v2.x
    /// (`head_dim = 1280 / 20 = 64`).
    pub n_audio_head: usize,
    /// Encoder block count. **32** for kotoba-whisper v1.x/v2.x — the
    /// family keeps the large-v3 encoder intact.
    pub n_audio_layer: usize,
    /// Decoder positional length (`max_target_positions`), 448.
    pub n_text_ctx: usize,
    /// Decoder attention heads — same as `n_audio_head` for kotoba-whisper.
    pub n_text_head: usize,
    /// Decoder block count. **2** for kotoba-whisper v1.x/v2.x — the
    /// distil axis (large-v3 has 32). This is the JA-ASR-2 axis: the
    /// converter and runtime must both honor it as data-driven from
    /// GGUF metadata (never hard-coded to 32).
    pub n_text_layer: usize,
    /// Token vocabulary size — **51 866** for kotoba-whisper (the
    /// large-v3 multilingual vocab including `<|yue|>`).
    pub n_vocab: usize,
    /// Feed-forward inner width — 5120 for kotoba-whisper v1.x/v2.x.
    pub ffn_dim: usize,
    /// End-of-transcript token id (decode stop condition) — 50257 for
    /// the Whisper multilingual tokenizer.
    pub eot: u32,
    /// Start-of-transcript token id (decoder prompt seed) — 50258 for
    /// the Whisper multilingual tokenizer.
    pub sot: u32,
    /// PCM sample rate — 16 000 (Whisper convention).
    pub sample_rate: u32,
}

impl KotobaWhisperConfig {
    /// Per-head width. Whisper fixes this at 64 across every family
    /// size, so it is simply `d_model / n_audio_head` (validated
    /// non-zero and exact in [`Self::validate_for_forward`]).
    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.d_model.checked_div(self.n_audio_head).unwrap_or(0)
    }

    /// Primary-source `kotoba-tech/kotoba-whisper-v2.0` config (every
    /// value transcribed verbatim from the upstream `config.json` — see
    /// module docstring). The v1.0 / v1.1 / v2.1 releases share the
    /// same architectural quintuple; only the distilled weights differ.
    #[must_use]
    pub fn kotoba_whisper_v2_0() -> Self {
        Self {
            n_mels: 128,
            d_model: 1280,
            n_audio_ctx: 1500,
            n_audio_head: 20,
            n_audio_layer: 32,
            n_text_ctx: 448,
            n_text_head: 20,
            n_text_layer: 2,
            n_vocab: 51_866,
            ffn_dim: 5120,
            eot: 50_257,
            sot: 50_258,
            sample_rate: KOTOBA_WHISPER_SAMPLE_RATE,
        }
    }

    /// Miniature well-formed config for shape / stability tests. Dims
    /// are tiny so synthesized-weight builds fit in KB; the *shape
    /// relationships* (encoder deeper than decoder, MHA head split,
    /// even head_dim, vocab non-zero) mirror the real model.
    #[must_use]
    pub fn tiny_for_tests() -> Self {
        Self {
            n_mels: 16,
            d_model: 16,
            n_audio_ctx: 24,
            n_audio_head: 4,
            n_audio_layer: 4,
            n_text_ctx: 16,
            n_text_head: 4,
            n_text_layer: 2,
            n_vocab: 32,
            ffn_dim: 32,
            eot: 30,
            sot: 31,
            sample_rate: KOTOBA_WHISPER_SAMPLE_RATE,
        }
    }

    /// Rejects `0`-placeholder / ill-formed configs before any forward
    /// runs (FR-EX-08 — a shape-only converter path fails loudly here,
    /// not deep inside a GEMM).
    ///
    /// Enforces the Whisper cross-checks (`d_model % n_head == 0` for
    /// both encoder and decoder, non-zero axes, EOT/SOT inside the
    /// vocab) plus the distil invariant
    /// (`n_text_layer < n_audio_layer`).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] naming the offending field.
    pub fn validate_for_forward(&self) -> Result<()> {
        if self.d_model == 0
            || self.n_mels == 0
            || self.n_audio_ctx == 0
            || self.n_audio_layer == 0
            || self.n_text_ctx == 0
            || self.n_text_layer == 0
            || self.n_vocab == 0
            || self.ffn_dim == 0
            || self.n_audio_head == 0
            || self.n_text_head == 0
            || self.sample_rate == 0
        {
            return Err(VokraError::InvalidArgument(
                "kotoba-whisper config: every architectural axis must be > 0".to_owned(),
            ));
        }
        if self.d_model % self.n_audio_head != 0 {
            return Err(VokraError::InvalidArgument(format!(
                "kotoba-whisper config: n_audio_head ({}) must divide d_model ({})",
                self.n_audio_head, self.d_model,
            )));
        }
        if self.d_model % self.n_text_head != 0 {
            return Err(VokraError::InvalidArgument(format!(
                "kotoba-whisper config: n_text_head ({}) must divide d_model ({})",
                self.n_text_head, self.d_model,
            )));
        }
        if self.head_dim() % 2 != 0 {
            return Err(VokraError::InvalidArgument(format!(
                "kotoba-whisper config: head_dim {} must be even (attention K/V pair layout)",
                self.head_dim(),
            )));
        }
        // The distil axis: a kotoba-whisper checkpoint has fewer decoder
        // layers than encoder layers. A checkpoint where the two are
        // equal is (a) a real Whisper (large-v3, medium, etc.) that
        // landed on the kotoba path by mistake, or (b) a mis-flattened
        // checkpoint where the decoder tensors were duplicated to the
        // encoder count. Either way this must fail loudly (FR-EX-08).
        if self.n_text_layer >= self.n_audio_layer {
            return Err(VokraError::InvalidArgument(format!(
                "kotoba-whisper config: n_text_layer ({}) must be < n_audio_layer ({}); \
                 kotoba-whisper is Japanese-distilled from Whisper large-v3 with a shrunk \
                 decoder, so equal or larger decoder depth means this is not a kotoba-whisper \
                 checkpoint (use --model whisper for vanilla Whisper sizes)",
                self.n_text_layer, self.n_audio_layer,
            )));
        }
        if (self.eot as usize) >= self.n_vocab {
            return Err(VokraError::InvalidArgument(format!(
                "kotoba-whisper config: eot ({}) must be < n_vocab ({})",
                self.eot, self.n_vocab,
            )));
        }
        if (self.sot as usize) >= self.n_vocab {
            return Err(VokraError::InvalidArgument(format!(
                "kotoba-whisper config: sot ({}) must be < n_vocab ({})",
                self.sot, self.n_vocab,
            )));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

/// kotoba-whisper ASR engine handle.
///
/// Carries the resolved config. [`Self::transcribe`] is the primary
/// waveform → text entry point; until real weights are bound (see the
/// module docstring) it returns [`VokraError::NotImplemented`] with a
/// message naming the blocker (FR-EX-08 — never a silent zero-fill or
/// empty transcript).
///
/// # Data-driven decoder depth (JA-ASR-2)
///
/// The [`config`](Self::config) carries `n_text_layer = 2` and the
/// engine honors it end-to-end — the runtime never assumes 32 layers
/// (that would be a large-v3-specific hard-code). This is the JA-ASR-2
/// entry point: the shared [`crate::whisper::WhisperConfig`] loader
/// reads `n_text_layer` from GGUF metadata, and every downstream
/// component (weight binding, KV cache, greedy loop, beam search)
/// iterates over `w.layers.len()` — no fixed constant.
pub struct KotobaWhisperAsr {
    cfg: KotobaWhisperConfig,
    /// Inner Whisper engine (present iff loaded from GGUF). Config-only
    /// constructors ([`Self::new`]) build a shape-flow shell without weights
    /// so [`Self::transcribe`] hard-errors with a message pointing at
    /// [`Self::from_gguf`] as the fix.
    inner: Option<WhisperAsr>,
}

impl KotobaWhisperAsr {
    /// Assembles an engine from `cfg`. Config is cross-checked so a
    /// mismatched shape fails loudly here rather than deep inside a
    /// forward.
    ///
    /// **This constructor does not bind weights.** The returned handle
    /// exercises the shape-flow / config-invariant path but any
    /// [`Self::transcribe`] call hard-errors with a message pointing at
    /// [`Self::from_gguf`] (the constructor that binds a real GGUF) as
    /// the fix. Real ASR requires either `from_gguf` or a follow-up
    /// weight-binding path.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] from `cfg.validate_for_forward`.
    pub fn new(cfg: KotobaWhisperConfig) -> Result<Self> {
        cfg.validate_for_forward()?;
        Ok(Self { cfg, inner: None })
    }

    /// Loads a real kotoba-whisper GGUF and binds the full weight set.
    ///
    /// **kotoba-whisper is architecturally a Whisper checkpoint whose only
    /// difference is `n_text_layer < n_audio_layer`** (see module docs).
    /// The upstream converter (`vokra-convert::models::kotoba_whisper`)
    /// therefore writes the standard `vokra.whisper.*` hparam chunk and
    /// keeps HF Whisper tensor names verbatim, so this delegates the
    /// forward to the shared [`crate::whisper::WhisperAsr`] plumbing —
    /// same op set (STFT / mel filterbank / GEMM / GEMV / softmax /
    /// layer-norm / GELU / conv1d), same kernels, same greedy /
    /// beam-search paths.
    ///
    /// The **distil invariant** (`n_text_layer < n_audio_layer`) is
    /// enforced on the loaded config: a checkpoint whose decoder-layer
    /// count equals or exceeds the encoder count is either vanilla
    /// Whisper (large-v3 = 32/32) or a mis-flattened distil, and this
    /// fails loudly (FR-EX-08) rather than mis-labeling a Whisper GGUF
    /// as kotoba-whisper.
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] via the delegate load path (missing
    ///   `vokra.whisper.*` metadata, missing / mis-shaped weight tensors,
    ///   or the front-end chunk check).
    /// - [`VokraError::ModelLoad`] if the loaded config violates the
    ///   kotoba-whisper distil invariant (`n_text_layer >= n_audio_layer`).
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        let inner = WhisperAsr::from_gguf(file)?;
        let wc = inner.model().config();
        if wc.n_text_layer >= wc.n_audio_layer {
            return Err(VokraError::ModelLoad(format!(
                "kotoba-whisper: loaded GGUF has n_text_layer ({}) >= n_audio_layer ({}); \
                 kotoba-whisper is a Japanese-distilled Whisper checkpoint whose decoder \
                 is strictly smaller than the encoder — equal or larger decoder depth \
                 means this GGUF is vanilla Whisper (use --model whisper) or a \
                 mis-flattened distil (decoder tensors duplicated to the encoder \
                 count). This is a loud-fail contract (FR-EX-08), never a silent mis-label.",
                wc.n_text_layer, wc.n_audio_layer,
            )));
        }
        // Build a `KotobaWhisperConfig` snapshot from the loaded config so the
        // public [`Self::config`] surface stays stable. Every field mirrors the
        // corresponding Whisper axis; `sot` comes from the first entry of
        // `decoder_start_ids` (Whisper's `<|startoftranscript|>` prefix), which
        // the loader guarantees is non-empty.
        let cfg = KotobaWhisperConfig {
            n_mels: wc.n_mels,
            d_model: wc.d_model,
            n_audio_ctx: wc.n_audio_ctx,
            n_audio_head: wc.n_audio_head,
            n_audio_layer: wc.n_audio_layer,
            n_text_ctx: wc.n_text_ctx,
            n_text_head: wc.n_text_head,
            n_text_layer: wc.n_text_layer,
            n_vocab: wc.n_vocab,
            ffn_dim: wc.ffn_dim,
            eot: wc.eot,
            sot: wc
                .decoder_start_ids
                .first()
                .copied()
                .unwrap_or(50_258 /* Whisper `<|startoftranscript|>` fallback */),
            sample_rate: KOTOBA_WHISPER_SAMPLE_RATE,
        };
        cfg.validate_for_forward()?;
        Ok(Self {
            cfg,
            inner: Some(inner),
        })
    }

    /// Attaches a detokenizer for [`Self::transcribe`] to convert token ids
    /// back to text. Whisper family GGUFs may embed the tokenizer as a
    /// `vokra.tokenizer.model` blob (see [`WhisperTokenizer`]) — this
    /// overrides / attaches one from a side-car fixture.
    ///
    /// No-op when no inner engine is bound (config-only shell).
    #[must_use]
    pub fn with_tokenizer(mut self, tokenizer: WhisperTokenizer) -> Self {
        self.inner = self.inner.map(|w| w.with_tokenizer(tokenizer));
        self
    }

    /// Selects the backend the transcription forward runs on (default
    /// [`BackendKind::Cpu`]).
    ///
    /// No-op when no inner engine is bound (config-only shell); the backend
    /// selection is honored by the shared Whisper engine when
    /// [`Self::from_gguf`] was used to build this handle. FR-EX-08: an
    /// unsupported backend (e.g. Metal today) surfaces as an explicit
    /// [`VokraError::UnsupportedOp`] at [`Self::transcribe`] time (via the
    /// delegate), never a silent CPU fall back.
    #[must_use]
    pub fn with_backend(mut self, backend: BackendKind) -> Self {
        self.inner = self.inner.map(|w| w.with_backend(backend));
        self
    }

    /// Binds the verified whole-encoder CoreML sidecar to the shared Whisper
    /// delegate path. The config-only shell rejects it instead of silently
    /// discarding the requested backend artifact.
    #[cfg(feature = "coreml")]
    pub fn with_coreml_artifact(mut self, artifact: CoreMlArtifact) -> Result<Self> {
        let inner = self.inner.take().ok_or_else(|| {
            VokraError::UnsupportedOp(
                "kotoba-whisper CoreML artifact requires from_gguf; the config-only shell has no executable weights"
                    .to_owned(),
            )
        })?;
        self.inner = Some(inner.with_coreml_artifact(artifact)?);
        Ok(self)
    }

    /// The resolved configuration.
    #[must_use]
    pub fn config(&self) -> &KotobaWhisperConfig {
        &self.cfg
    }

    /// Whether a real GGUF weight set was bound ([`Self::from_gguf`] was
    /// used to build this handle). Test-oriented predicate — production
    /// callers should just call [`Self::transcribe`] and let the loud
    /// error path point at [`Self::from_gguf`] when the shell was built
    /// via [`Self::new`].
    #[must_use]
    pub fn has_weights(&self) -> bool {
        self.inner.is_some()
    }

    /// Transcribes a mono `f32` PCM slice at [`Self::config`]'s sample
    /// rate (16 kHz — [`KOTOBA_WHISPER_SAMPLE_RATE`]).
    ///
    /// When built via [`Self::from_gguf`] this delegates to the shared
    /// Whisper greedy decode (log-mel front-end → 32-layer encoder →
    /// kotoba-shrunk decoder → byte-level BPE ids). The **data-driven
    /// decoder depth** (JA-ASR-2 axis) is honored end-to-end: the shared
    /// [`crate::whisper::WhisperConfig`] loader reads `n_text_layer` from
    /// GGUF metadata, weight binding / KV cache / greedy loop / beam
    /// search iterate over `w.layers.len()` — no fixed constant.
    ///
    /// When built via [`Self::new`] (config-only shell) this hard-errors
    /// with a [`VokraError::NotImplemented`] message pointing at
    /// [`Self::from_gguf`] as the fix (FR-EX-08 — never a silent
    /// zero-fill or fabricated transcript).
    ///
    /// # Errors
    ///
    /// - [`VokraError::InvalidArgument`] if `pcm` is empty.
    /// - [`VokraError::NotImplemented`] if this handle was built via
    ///   [`Self::new`] (no weights bound).
    /// - Any error from [`WhisperAsr::transcribe_tokens`] (backend
    ///   unsupported, decoder failure, etc.).
    pub fn transcribe(&self, pcm: &[f32]) -> Result<Vec<u32>> {
        if pcm.is_empty() {
            return Err(VokraError::InvalidArgument(
                "kotoba-whisper transcribe: pcm slice is empty".to_owned(),
            ));
        }
        match &self.inner {
            Some(asr) => asr.transcribe_tokens(pcm),
            None => Err(VokraError::NotImplemented(
                "kotoba-whisper transcribe: this handle was built from a config-only \
                 KotobaWhisperConfig via KotobaWhisperAsr::new (no weights bound). \
                 kotoba-whisper is architecturally a Whisper checkpoint (JA-ASR-2 axis: \
                 shrunk n_text_layer), so real transcription delegates to the shared \
                 crate::whisper::WhisperAsr plumbing — bind real weights via \
                 KotobaWhisperAsr::from_gguf(&GgufFile) instead. The op set (STFT / mel \
                 filterbank / GEMM / GEMV / softmax / layer-norm / GELU / conv1d) is \
                 shared with vanilla Whisper (FR-EX-08 — never a silent zero-fill).",
            )),
        }
    }

    /// Detokenizes `ids` via the attached detokenizer, or renders them as a
    /// bracketed id list when none is attached. Delegates to
    /// [`WhisperAsr::render_ids`] when an inner engine is bound; otherwise
    /// falls back to the bracketed id form (matching the Whisper convention).
    pub fn render_ids(&self, ids: &[u32]) -> Result<String> {
        match &self.inner {
            Some(asr) => asr.render_ids(ids),
            None => Ok(format!(
                "[no tokenizer; token ids: {}]",
                ids.iter().map(u32::to_string).collect::<Vec<_>>().join(" ")
            )),
        }
    }

    /// Test-only wrapper: build a weights-bound handle around an already-loaded
    /// [`WhisperAsr`] **without** enforcing the [`Self::from_gguf`] distil
    /// invariant (`n_text_layer < n_audio_layer`). Tests that exercise the
    /// [`AsrEngine`] trait dispatch (composition, empty-PCM early return) only
    /// need a handle whose `transcribe` funnels through the shared Whisper
    /// engine — they do not exercise the mislabel-refusal path, which has its
    /// own dedicated `from_gguf_rejects_non_distil_shape_via_delegate_chain`
    /// coverage below.
    ///
    /// Mirrors `crate::distil_whisper::DistilWhisperAsr::from_whisper_asr_for_test`.
    /// The config surfaced through [`Self::config`] mirrors the inner Whisper
    /// config verbatim (same shape as [`Self::from_gguf`]), so
    /// [`Self::has_weights`] is `true` and the handle behaves indistinguishably
    /// from a real GGUF load to code that only reads the introspection surface.
    ///
    /// Not part of the public API (compiled only under `cfg(test)`).
    #[cfg(test)]
    pub(crate) fn from_whisper_asr_for_test(inner: WhisperAsr) -> Self {
        let wc = inner.model().config();
        let cfg = KotobaWhisperConfig {
            n_mels: wc.n_mels,
            d_model: wc.d_model,
            n_audio_ctx: wc.n_audio_ctx,
            n_audio_head: wc.n_audio_head,
            n_audio_layer: wc.n_audio_layer,
            n_text_ctx: wc.n_text_ctx,
            n_text_head: wc.n_text_head,
            n_text_layer: wc.n_text_layer,
            n_vocab: wc.n_vocab,
            ffn_dim: wc.ffn_dim,
            eot: wc.eot,
            sot: wc.decoder_start_ids.first().copied().unwrap_or(50_258),
            sample_rate: KOTOBA_WHISPER_SAMPLE_RATE,
        };
        Self {
            cfg,
            inner: Some(inner),
        }
    }
}

/// [`AsrEngine`] wiring so a kotoba-whisper handle can be injected via
/// [`vokra_core::Session::with_asr_engine`] and drive
/// `session.asr().transcribe()` end-to-end — which is exactly how
/// `vokra-cli run` reaches this model.
///
/// Composition — verbatim the [`WhisperAsr`] / [`crate::distil_whisper`]
/// pattern:
/// 1. call the inherent [`KotobaWhisperAsr::transcribe`] for raw token ids
///    (GGUF path → [`WhisperAsr::transcribe_tokens`] greedy; config-only
///    shell → loud [`VokraError::NotImplemented`]),
/// 2. render them through [`KotobaWhisperAsr::render_ids`] (GGUF path →
///    [`WhisperAsr::render_ids`]; shell → the bracketed-id fallback),
/// 3. wrap the resulting `String` in a [`Transcription`].
///
/// The inherent method and this trait method share the receiver + argument
/// shape, so method resolution inside the trait body picks the inherent one
/// (returning `Result<Vec<u32>>`) — the composition leg we want, with no
/// accidental recursion.
///
/// The empty-PCM guard lives in the inherent method and therefore also
/// governs this one (FR-EX-08 — never a silent empty transcription).
impl AsrEngine for KotobaWhisperAsr {
    fn transcribe(&self, pcm: &[f32]) -> Result<Transcription> {
        let ids = self.transcribe(pcm)?;
        Ok(Transcription::new(self.render_ids(&ids)?))
    }

    /// Asks the delegate rather than storing a second copy: the backend is
    /// set through [`KotobaWhisperAsr::with_backend`], which forwards to the
    /// inner [`WhisperAsr`], so a duplicate field here could disagree with
    /// the engine that actually runs.
    ///
    /// The unbound arm reports `Cpu`, which cannot mislead in the way the
    /// trait warns about: with no inner engine there is no forward, so no
    /// execution exists anywhere else to contradict the answer.
    fn backend(&self) -> BackendKind {
        self.inner
            .as_ref()
            .map_or(BackendKind::Cpu, AsrEngine::backend)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Every hparam matches `huggingface.co/kotoba-tech/kotoba-whisper-v2.0/
    /// raw/main/config.json` (fetched 2026-07-24).
    #[test]
    fn kotoba_whisper_v2_0_matches_primary_source_config_json() {
        let c = KotobaWhisperConfig::kotoba_whisper_v2_0();
        // Encoder — identical to whisper-large-v3.
        assert_eq!(c.d_model, 1280);
        assert_eq!(c.n_audio_layer, 32);
        assert_eq!(c.n_audio_head, 20);
        assert_eq!(c.ffn_dim, 5120);
        assert_eq!(c.n_mels, 128);
        assert_eq!(c.n_audio_ctx, 1500);
        // Decoder — the JA-ASR-2 axis.
        assert_eq!(
            c.n_text_layer, 2,
            "kotoba-whisper v2.0 shrinks decoder to 2 layers"
        );
        assert_eq!(c.n_text_head, 20);
        assert_eq!(c.n_text_ctx, 448);
        // Tokenizer — large-v3 multilingual (+1 vocab for <|yue|>).
        assert_eq!(c.n_vocab, 51_866);
        assert_eq!(c.eot, 50_257);
        assert_eq!(c.sot, 50_258);
        // Audio boundary.
        assert_eq!(c.sample_rate, 16_000);
        // Derived — head_dim is the Whisper invariant.
        assert_eq!(c.head_dim(), 64, "kotoba-whisper head_dim = 1280/20 = 64");
        // Distil invariant holds.
        assert!(
            c.n_text_layer < c.n_audio_layer,
            "kotoba-whisper must have decoder < encoder depth"
        );
        c.validate_for_forward()
            .expect("kotoba-whisper v2.0 is well-formed");
    }

    #[test]
    fn tiny_config_is_well_formed() {
        KotobaWhisperConfig::tiny_for_tests()
            .validate_for_forward()
            .expect("tiny config is well-formed");
    }

    /// A checkpoint whose decoder depth equals the encoder depth is
    /// **not** kotoba-whisper — it is vanilla Whisper. The validator
    /// must catch this so a mis-flattened checkpoint (decoder tensors
    /// duplicated to the encoder count) fails loudly at
    /// `KotobaWhisperAsr::new`, not silently deep in a forward.
    #[test]
    fn config_rejects_equal_encoder_decoder_depth() {
        let mut c = KotobaWhisperConfig::tiny_for_tests();
        c.n_text_layer = c.n_audio_layer;
        let err = c
            .validate_for_forward()
            .expect_err("equal depth is not kotoba");
        assert!(
            matches!(err, VokraError::InvalidArgument(_)),
            "expected InvalidArgument, got {err:?}"
        );
    }

    #[test]
    fn config_rejects_decoder_larger_than_encoder() {
        let mut c = KotobaWhisperConfig::tiny_for_tests();
        c.n_text_layer = c.n_audio_layer + 1;
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn config_rejects_zero_axis() {
        for mutate in [
            |c: &mut KotobaWhisperConfig| c.d_model = 0,
            |c: &mut KotobaWhisperConfig| c.n_mels = 0,
            |c: &mut KotobaWhisperConfig| c.n_audio_ctx = 0,
            |c: &mut KotobaWhisperConfig| c.n_audio_layer = 0,
            |c: &mut KotobaWhisperConfig| c.n_text_ctx = 0,
            |c: &mut KotobaWhisperConfig| c.n_text_layer = 0,
            |c: &mut KotobaWhisperConfig| c.n_vocab = 0,
            |c: &mut KotobaWhisperConfig| c.ffn_dim = 0,
            |c: &mut KotobaWhisperConfig| c.n_audio_head = 0,
            |c: &mut KotobaWhisperConfig| c.n_text_head = 0,
            |c: &mut KotobaWhisperConfig| c.sample_rate = 0,
        ] {
            let mut c = KotobaWhisperConfig::tiny_for_tests();
            mutate(&mut c);
            assert!(matches!(
                c.validate_for_forward(),
                Err(VokraError::InvalidArgument(_))
            ));
        }
    }

    #[test]
    fn config_rejects_head_not_dividing_d_model() {
        let mut c = KotobaWhisperConfig::tiny_for_tests();
        c.n_audio_head = 3; // 16 % 3 != 0
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn config_rejects_odd_head_dim() {
        let mut c = KotobaWhisperConfig::tiny_for_tests();
        // 12 / 4 = 3 (odd).
        c.d_model = 12;
        c.n_audio_head = 4;
        c.n_text_head = 4;
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn config_rejects_eot_or_sot_outside_vocab() {
        let mut c = KotobaWhisperConfig::tiny_for_tests();
        c.eot = c.n_vocab as u32;
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));

        let mut c = KotobaWhisperConfig::tiny_for_tests();
        c.sot = c.n_vocab as u32 + 10;
        assert!(matches!(
            c.validate_for_forward(),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn asr_new_accepts_well_formed_config() {
        let c = KotobaWhisperConfig::tiny_for_tests();
        let asr = KotobaWhisperAsr::new(c.clone()).expect("kotoba-whisper asr");
        assert_eq!(asr.config().d_model, c.d_model);
        assert_eq!(asr.config().n_text_layer, c.n_text_layer);
    }

    #[test]
    fn transcribe_rejects_empty_pcm() {
        let c = KotobaWhisperConfig::tiny_for_tests();
        let asr = KotobaWhisperAsr::new(c).expect("kotoba-whisper asr");
        assert!(matches!(
            asr.transcribe(&[]),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    /// The primary NotImplemented path names the follow-up blocker
    /// (FR-EX-08 — never a silent zero-fill / hallucinated transcript).
    #[test]
    fn transcribe_is_loud_not_implemented_until_real_forward_lands() {
        let c = KotobaWhisperConfig::tiny_for_tests();
        let asr = KotobaWhisperAsr::new(c).expect("kotoba-whisper asr");
        let pcm = vec![0.0f32; 1024];
        let err = asr.transcribe(&pcm).unwrap_err();
        match err {
            VokraError::NotImplemented(msg) => {
                assert!(
                    msg.contains("kotoba-whisper"),
                    "message must name the model: {msg}"
                );
                assert!(
                    msg.contains("JA-ASR-2"),
                    "message must reference JA-ASR-2 axis: {msg}"
                );
            }
            other => panic!("expected NotImplemented, got {other:?}"),
        }
    }

    #[test]
    fn expected_arch_is_kotoba_whisper() {
        assert_eq!(EXPECTED_ARCH, "kotoba-whisper");
    }

    #[test]
    fn expected_arch_is_distinct_from_siblings() {
        assert_ne!(EXPECTED_ARCH, "whisper");
        assert_ne!(EXPECTED_ARCH, crate::distil_whisper::EXPECTED_ARCH);
    }

    #[test]
    fn sample_rate_matches_whisper_convention() {
        assert_eq!(KOTOBA_WHISPER_SAMPLE_RATE, 16_000);
    }

    // ---------- from_gguf delegation tests (Wave 7 Part A RUNTIME-NOTIMPL) ----------

    use vokra_core::gguf::{GgufArray, GgufBuilder, GgufFile, GgufMetadataValue, GgufValueType};

    /// Builds a GGUF carrying a `vokra.whisper.*` chunk with a **kotoba-shrunk**
    /// decoder (n_text_layer < n_audio_layer). No weight tensors — the delegate
    /// load path then fails on the front-end check (Whisper requires a
    /// `vokra.frontend.*` chunk), which is exactly the loud error we want to
    /// observe: the delegate is live, config parsing works, and the ordering
    /// is correct.
    fn write_kotoba_shape_config(b: &mut GgufBuilder, n_audio_layer: u32, n_text_layer: u32) {
        b.add_u32("vokra.whisper.n_mels", 128);
        b.add_u32("vokra.whisper.n_audio_ctx", 1500);
        b.add_u32("vokra.whisper.n_audio_state", 1280);
        b.add_u32("vokra.whisper.n_audio_head", 20);
        b.add_u32("vokra.whisper.n_audio_layer", n_audio_layer);
        b.add_u32("vokra.whisper.n_text_ctx", 448);
        b.add_u32("vokra.whisper.n_text_state", 1280);
        b.add_u32("vokra.whisper.n_text_head", 20);
        b.add_u32("vokra.whisper.n_text_layer", n_text_layer);
        b.add_u32("vokra.whisper.n_vocab", 51_866);
        b.add_u32("vokra.whisper.ffn_dim", 5120);
        b.add_u32("vokra.whisper.eot", 50_257);
        b.add_metadata(
            "vokra.whisper.decoder_start_ids",
            GgufMetadataValue::Array(GgufArray {
                element_type: GgufValueType::U32,
                values: [50_258u32, 50_259, 50_359, 50_363]
                    .iter()
                    .map(|&id| GgufMetadataValue::U32(id))
                    .collect(),
            }),
        );
    }

    /// [`KotobaWhisperAsr::new`] builds a config-only shell; [`Self::has_weights`]
    /// is `false` and [`Self::transcribe`] hard-errors with the migration hint.
    #[test]
    fn new_builds_a_shell_without_weights() {
        let asr = KotobaWhisperAsr::new(KotobaWhisperConfig::tiny_for_tests())
            .expect("tiny config is well-formed");
        assert!(!asr.has_weights(), "new() must not bind weights");
        let err = asr.transcribe(&[0.0f32; 512]).unwrap_err();
        match err {
            VokraError::NotImplemented(msg) => {
                assert!(msg.contains("from_gguf"), "hint must name the fix: {msg}");
                assert!(msg.contains("kotoba-whisper"));
            }
            other => panic!("expected NotImplemented, got {other:?}"),
        }
    }

    /// [`Self::from_gguf`] delegates the load to [`WhisperAsr::from_gguf`],
    /// which requires the `vokra.frontend.*` chunk (M1-03). A GGUF without
    /// the chunk fails as [`VokraError::ModelLoad`] before any weight bind —
    /// this observes the delegation wiring is live (the error surfaces
    /// from the shared Whisper loader, not from a shape-only stub).
    #[test]
    fn from_gguf_delegates_and_reports_missing_frontend_chunk() {
        let mut b = GgufBuilder::new();
        write_kotoba_shape_config(&mut b, 32, 2);
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();

        // Delegate reports ModelLoad from the front-end check (Whisper requires
        // it; no weights reached).
        match KotobaWhisperAsr::from_gguf(&file) {
            Err(VokraError::ModelLoad(msg)) => {
                // Any Whisper-loader error is acceptable here; we only assert
                // the delegation reached the shared loader.
                assert!(!msg.is_empty());
            }
            Err(other) => {
                panic!("expected ModelLoad from the delegate Whisper load path, got {other:?}")
            }
            Ok(_) => panic!(
                "expected ModelLoad from the delegate Whisper load path, got Ok(_) \
                 — but this GGUF carries no weights (front-end check should fire)"
            ),
        }
    }

    /// A GGUF whose decoder is NOT smaller than the encoder is **not** a
    /// kotoba-whisper (vanilla Whisper large-v3 has 32/32). The distil
    /// invariant must fire — FR-EX-08, loud mislabel refusal.
    ///
    /// Uses a hand-built GGUF whose `vokra.whisper.n_text_layer` ==
    /// `n_audio_layer` and covers the whole delegate chain up to the point
    /// of the invariant check. Because the invariant fires *after* the
    /// underlying [`WhisperAsr::from_gguf`], a shape-only fixture would
    /// fail earlier at the front-end check — so the *specific* invariant
    /// path is only observable with a fixture that reaches the invariant.
    /// The direct-invariant path is covered by
    /// [`KotobaWhisperConfig::validate_for_forward`] already; here we
    /// simply confirm the from_gguf path fails loudly on a matched-depth
    /// GGUF (whichever check fires first).
    #[test]
    fn from_gguf_rejects_non_distil_shape_via_delegate_chain() {
        let mut b = GgufBuilder::new();
        // Vanilla-shape: 6/6 (matches whisper base). This is NOT kotoba.
        write_kotoba_shape_config(&mut b, 6, 6);
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();

        // Some delegate error must fire (either the front-end check, the
        // distil invariant, or a downstream weight bind). What we assert is
        // that this GGUF does NOT load successfully — FR-EX-08 loud-fail.
        assert!(
            KotobaWhisperAsr::from_gguf(&file).is_err(),
            "matched-depth GGUF must not load as kotoba-whisper (FR-EX-08 mislabel refusal)"
        );
    }

    /// The M2-13 compliance registry must resolve every canonical
    /// kotoba-whisper id to Permissive (Apache-2.0). Cross-crate test
    /// to keep this module's registry-side contract honest.
    ///
    /// Scout A-5 follow-up (2026-07-29): `kotoba-whisper-v2.2` is the
    /// slug the parity-CI workflow (`parity-whisper-extras-real.yml`)
    /// pins today via `env.KOTOBA_WHISPER_REPO`. It resolves Permissive
    /// transitively via the `kotoba-whisper-` prefix walk in
    /// `vokra_core::compliance::license_class`, but pinning it here
    /// makes the workflow-pinned literal a machine-checked invariant so
    /// a future prefix-walk removal surfaces red on any `cargo test`
    /// rather than only during a paid HF-download workflow run.
    #[test]
    fn registry_lookup_maps_kotoba_whisper_to_permissive_apache_2_0() {
        use vokra_core::compliance::{LicenseClass, registry_lookup};
        for id in [
            "kotoba-whisper",
            "kotoba-whisper-v1.0",
            "kotoba-whisper-v1.1",
            "kotoba-whisper-v2.0",
            "kotoba-whisper-v2.1",
            // v2.2 = workflow-pinned literal (see rustdoc above).
            // Prefix walk covers it today; the pin binds the contract.
            "kotoba-whisper-v2.2",
            "kotoba-whisper-bilingual",
            "kotoba-whisper-bilingual-v1.0",
        ] {
            assert_eq!(
                registry_lookup(id),
                Some(LicenseClass::Permissive),
                "registry must map `{id}` to Permissive (Apache-2.0)"
            );
        }
    }

    // -------- AsrEngine trait dispatch tests --------
    //
    // These prove the `impl AsrEngine for KotobaWhisperAsr` (a) reaches the
    // shared Whisper delegate on the weights-bound arm rather than the
    // config-only `NotImplemented` arm, (b) honors the empty-PCM early return
    // through the trait method, and (c) composes to the same text as the
    // inherent `.transcribe(...)` → `.render_ids(...)` pipeline.
    //
    // The trait impl is what `vokra-cli run` consumes (the dispatch injects
    // this handle into the session's ASR slot), so an untested impl here would
    // leave the CLI routing resting on nothing. Mirrors the distil-whisper
    // trait tests, against the same synthetic fixture.

    use crate::whisper::decoder::test_support::tiny_model_distil;

    /// Builds a weights-bound `KotobaWhisperAsr` over a whisper-shape synthetic
    /// model (`n_audio_ctx = 1500` so the encoder passes its output-length
    /// check; 2 encoder layers, 1 decoder layer keeps the distil axis honest
    /// even though the test-only ctor bypasses the invariant check).
    fn delegate_asr() -> KotobaWhisperAsr {
        let model = tiny_model_distil(2, 1);
        let inner = WhisperAsr::from_model_for_test(model);
        KotobaWhisperAsr::from_whisper_asr_for_test(inner)
    }

    /// (a) `AsrEngine::transcribe` reaches the shared Whisper delegate — never
    /// the config-only `NotImplemented` arm — and returns a bounded
    /// [`Transcription`]. `Ok` here is itself the proof: the unbound arm
    /// returns `Err`.
    #[test]
    fn asr_engine_transcribe_delegate_returns_finite_transcription() {
        let asr = delegate_asr();
        assert!(
            asr.has_weights(),
            "the test fixture must be the weights-bound arm"
        );
        // 1024 mono samples: the WhisperAsr log-mel front-end zero-pads to its
        // fixed 30 s window regardless, so any non-empty PCM exercises the full
        // PCM → mel → encoder → decoder path.
        let pcm = vec![0.0f32; 1024];
        let out: Transcription = <KotobaWhisperAsr as AsrEngine>::transcribe(&asr, &pcm)
            .expect("bound AsrEngine::transcribe must return Ok(Transcription)");
        // Bounded (never unbounded / DoS): greedy stops on eot within
        // DEFAULT_MAX_NEW_TOKENS = 224 iterations, so the bracketed-ids render
        // is at most a few KB even in the worst case.
        assert!(
            out.text.len() < 16 * 1024,
            "transcription text must stay bounded; got {} bytes",
            out.text.len()
        );
    }

    /// (b) The trait method honors the empty-PCM early return the inherent
    /// method enforces, so a caller behind `dyn AsrEngine` (i.e.
    /// `session.asr().transcribe(&[])`, which is what the CLI holds) sees the
    /// same loud `InvalidArgument` — never a silent empty transcript
    /// (FR-EX-08).
    #[test]
    fn asr_engine_transcribe_rejects_empty_pcm() {
        let asr = delegate_asr();
        let Err(err) = <KotobaWhisperAsr as AsrEngine>::transcribe(&asr, &[]) else {
            panic!("expected an error when the trait method is handed empty PCM");
        };
        match err {
            VokraError::InvalidArgument(msg) => {
                assert!(
                    msg.contains("kotoba-whisper"),
                    "error must name the model: {msg}"
                );
                assert!(msg.contains("empty"), "error must name the blocker: {msg}");
            }
            other => panic!("expected InvalidArgument, got {other:?}"),
        }
    }

    /// (c) The trait method is exactly
    /// `Transcription::new(self.render_ids(&self.transcribe(pcm)?)?)` — the
    /// text is byte-identical to the manual pipeline, proving it introduced no
    /// separate beam / sampling branch and no post-processing beyond
    /// `render_ids`.
    #[test]
    fn asr_engine_transcribe_composes_with_inherent_transcribe() {
        let asr = delegate_asr();
        let pcm = vec![0.0f32; 1024];

        let via_trait = <KotobaWhisperAsr as AsrEngine>::transcribe(&asr, &pcm)
            .expect("trait transcribe must succeed on the bound path");

        // `WhisperAsr::transcribe_tokens` is idempotent per call (fresh KV
        // cache, no RNG on greedy), so re-running over the same PCM reproduces
        // the same ids deterministically.
        let ids = asr
            .transcribe(&pcm)
            .expect("inherent transcribe must succeed on the bound path");
        let text = asr
            .render_ids(&ids)
            .expect("render_ids must succeed on the bound path");

        assert_eq!(
            via_trait.text, text,
            "trait method must be a straight composition of inherent transcribe + render_ids",
        );
    }
}
