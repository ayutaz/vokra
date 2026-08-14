//! NVIDIA **Canary-1B-Flash** — FastConformer encoder + Transformer AED
//! decoder, multitask multilingual ASR / AST runtime binder (Wave C1,
//! 2026-08-15; loud-partial per the `canary` / `canary_qwen` /
//! `parakeet_ctc` / `emotion2vec` precedent — CLAUDE.md 教訓 (a):
//! 「loud-partial は fake-complete より honest」).
//!
//! # Why this module exists
//!
//! `crates/vokra-convert/src/models/canary_1b_flash.rs` produces a GGUF
//! stamped `vokra.model.arch = "canary-1b-flash"`, but before this module
//! landed **nothing in the workspace read that arch string**: the weights
//! could be converted and then no loader would accept them. This binder
//! closes that gap — it is the runtime half of the converter's contract.
//!
//! # Primary sources
//!
//! - HF release / model card: <https://huggingface.co/nvidia/canary-1b-flash>
//!   — 883 M parameters, multitask ASR + AST (speech translation) +
//!   timestamps across **four** European languages (English / German /
//!   French / Spanish), 16 kHz mono `.wav` / `.flac`, weights **CC-BY 4.0**.
//!   Transcribed by the converter on 2026-08-03 and mirrored here verbatim
//!   (CLAUDE.md「ハルシネーション厳禁」).
//! - Family reference config (every axis the card does **not** state):
//!   `github.com/NVIDIA-NeMo/Speech/blob/main/examples/asr/conf/speech_multitask/fast-conformer_aed.yaml`
//!   — the shared FastConformer-Transformer AED reference the whole Canary
//!   family reuses. Its variant table records `canary-1b-flash` explicitly:
//!   `model_defaults.asr_enc_hidden` / `.lm_dec_hidden` = `1024`, decoder
//!   `max_sequence_length` = `1024`. Those two are **directly attested for
//!   the flash variant**, not extrapolated (see
//!   [`crate::canary`]'s module docstring, which transcribes the same table).
//! - Checkpoint bridge: the release ships as a `.nemo` tarball; the generic
//!   `tools/parity/nemo_pt_to_safetensors.py` (uv-managed, Python 3.12)
//!   flattens it to safetensors before `vokra-cli convert --model
//!   canary-1b-flash` runs. The runtime never sees Python / torch / `.nemo`
//!   (FR-LD-05).
//!
//! # Architecture (transcribed, never invented)
//!
//! ```text
//! PCM (mono f32, 16 kHz)
//!   -> 128-bin log-mel front-end                 ← primitive EXISTS (`vokra_ops::waveform_frontend`)
//!   -> FastConformer encoder, 32 layers          ← primitive EXISTS (`vokra_ops::conformer`,
//!                                                   `ConvSubsampleKind::Stacking { factor: 8 }`)
//!   -> (optional) encoder->decoder width proj    ← identity for the flash widths (1024 == 1024)
//!   -> Transformer AED decoder, **4 layers**     ← per-step wiring NOT LANDED (shared gap with `canary`)
//!        (pre-norm self-attn + cross-attn to the
//!         encoder-out + FFN, driven by a task-token
//!         prompt prefix)
//!   -> vocab head -> beam search                 ← primitive EXISTS (`vokra_core::decode::beam_search`)
//!   -> SentencePiece detokenize                  ← tokenizer NOT AVAILABLE (`.nemo` extraction owed)
//! ```
//!
//! The **4-layer decoder** is the whole point of the Flash variant: it is the
//! distillation of Canary-1B-v2's 8-layer decoder (and Canary-1B-v1's 24), and
//! is the axis that unlocks the model card's "1000+ RTFx" throughput claim.
//! That is also why [`ARCH`] must stay distinct from `"canary"` — see
//! "Sibling family distinctness" below.
//!
//! # Loud-partial classification (CLAUDE.md 教訓 (a))
//!
//! **Real in this WP** (nothing here is a stub):
//!
//! - Strict `vokra.model.arch == "canary-1b-flash"` verification, with a
//!   sibling-mis-route diagnostic naming the whole Canary neighbourhood.
//! - [`Canary1bFlashConfig`] — every axis transcribed from a primary source,
//!   with the axes that no primary source states left as **`0` placeholder
//!   sentinels** (`head.vocab_size` / `pad` / `bos` / `eos`) that
//!   [`Canary1bFlashConfig::validate_for_forward`] **refuses**, so a caller
//!   cannot silently run a hallucinated forward.
//! - Forward-compatible `vokra.canary_1b_flash.*` axis overrides: the current
//!   converter stamps **none** of them (it writes only the `vokra.model.*` /
//!   `vokra.provenance.*` / `vokra.schema.*` groups plus the verbatim tensor
//!   payload), so a real converted GGUF resolves to
//!   [`Canary1bFlashConfigSource::FamilyAnchored`]. Any override that *is*
//!   present is honoured, and a present-but-wrong-dtype key fails loud.
//! - [`Canary1bFlashWeights`] — the tensor manifest discovered on disk under
//!   the verbatim upstream names the converter passes through, with a
//!   non-empty gate, name lookup ([`Canary1bFlashWeights::require_tensor`])
//!   and shape checking ([`Canary1bFlashWeights::require_tensor_dims`]) that
//!   fail loud naming the tensor.
//! - Weight-license + FR-MD-09 attribution surfacing from the provenance
//!   chunks, fail-closed to [`LicenseClass::Unknown`] when unstamped.
//!
//! **Loud-partial in this WP**: [`Canary1bFlashAsr::transcribe`] /
//! [`Canary1bFlashAsr::transcribe_with_task`] return
//! [`VokraError::UnsupportedOp`] naming four concrete blockers:
//!
//! 1. **No tensor-name manifest.** The converter copies every float tensor
//!    under its verbatim upstream safetensors name; nothing in-repo
//!    transcribes NeMo's `EncDecMultiTaskModel` `state_dict` naming, and the
//!    `.nemo` tarball is owner-gated. Walking guessed names into typed slots
//!    would fabricate the layout.
//! 2. **No tokenizer.** The unified Canary SentencePiece model, its vocab
//!    width, and the concrete `pad` / `bos` / `eos` / `<taskname>` ids live
//!    inside the `.nemo` tarball. The model card does not enumerate them.
//! 3. **`head.vocab_size` is a `0` sentinel** as a direct consequence of (2)
//!    — the head width is unknown, so no logits array can be shaped.
//! 4. **The AED decoder step is not wired.** The encoder body
//!    (`vokra_ops::conformer`), the front-end
//!    (`vokra_ops::waveform_frontend`) and the search
//!    (`vokra_core::decode::beam_search`) all exist; the per-step
//!    self-attn + cross-attn + FFN decoder loop with a task-prompt prefix
//!    does not — a gap shared with [`crate::canary`], not specific to Flash.
//!
//! **No fabricated token ids are ever emitted** (FR-EX-08 — no silent partial
//! output, no silent zero-fill, no empty-transcript-as-success).
//!
//! # Sibling family distinctness
//!
//! [`ARCH`] = `"canary-1b-flash"` is deliberately distinct from every sibling
//! in the Canary / NeMo-ASR neighbourhood:
//!
//! - `canary` (Canary-1B-v2) — same FastConformer encoder depth (32) but an
//!   **8-layer** Transformer AED decoder and a 25-language vocabulary;
//! - `canary-qwen` (Canary-Qwen-2.5B) — same encoder, but the decoder is a
//!   **Qwen LLM** consuming the encoder-out as a soft-prompt prefix (GQA +
//!   RoPE + SwiGLU + RMSNorm), not an AED decoder with cross-attention;
//! - `parakeet-ctc` / `parakeet-tdt` — FastConformer encoders with **CTC /
//!   RNN-T** heads (no decoder stack at all, English-only);
//! - `whisper` / `voxtral` / `kyutai-stt` — unrelated ASR topologies.
//!
//! A loader that binds a 4-layer decoder manifest against an 8-layer
//! expectation does not crash — it silently mis-reads. FR-EX-08 forbids that
//! silent misroute, hence the strict arch gate.
//!
//! # Structure reuse (no third shape)
//!
//! The config axes reuse [`crate::canary`]'s types verbatim
//! ([`CanaryEncoderConfig`] / [`CanaryDecoderConfig`] / [`CanaryHeadConfig`])
//! and [`Canary1bFlashConfig::validate_for_forward`] **delegates** to the
//! shared Canary-family validator rather than duplicating ~130 lines of shape
//! algebra. This mirrors [`crate::canary_qwen`], which re-exports the same
//! encoder types. Flash-specific state is only what genuinely differs: the
//! 4-layer decoder anchor, the 4-language subset, and the
//! [`Canary1bFlashConfigSource`] provenance marker.
//!
//! # Cross-crate constant duplication
//!
//! [`ARCH`] / [`NAME`] / [`CATEGORY`] / [`UPSTREAM_HF`] / [`DEFAULT_LICENSE`]
//! mirror the converter's constants by value (pinned by a test) so
//! `vokra-models` does not gain a dependency edge onto `vokra-convert`,
//! preserving the layered convention: `vokra-ops` → nothing GGUF-aware,
//! `vokra-core` → GGUF reader, `vokra-models` → GGUF binder, `vokra-convert`
//! → GGUF writer.
//!
//! # Licensing posture
//!
//! The converter stamps `cc-by-4.0` → [`LicenseClass::AttributionRequired`]
//! (commercial use permitted, attribution required — the FR-MD-09 surface
//! activates and [`Canary1bFlashAsr::attribution`] returns the stamped text).
//! This binder only **surfaces** whatever class the artifact carries and
//! fail-closes to [`LicenseClass::Unknown`] when nothing is stamped. The
//! `docs/license-audit.md` §3.1 sign-off stays **blank** (owner-only per
//! `[[feedback-license-signoff-primary-source]]` — the converter's default is
//! not a sign-off).
//!
//! # No ONNX (permanent)
//!
//! Canary-1B-Flash ships as a `.nemo` tarball / PyTorch pipeline; the pipeline
//! is re-implemented natively here (whisper.cpp 型, CLAUDE.md 設計判断 4).
//! This module never touches ONNX (FR-LD-05).

use vokra_core::gguf::{GgufFile, chunks};
use vokra_core::{CompliancePolicy, LicenseClass, Result, VokraError, check_weight_license};

pub use crate::canary::{CanaryDecoderConfig, CanaryEncoderConfig, CanaryHeadConfig};

// ---------------------------------------------------------------------------
// Contract constants — mirror of `crates/vokra-convert/src/models/canary_1b_flash.rs`.
// See the module docstring for the cross-crate duplication rationale.
// ---------------------------------------------------------------------------

/// Expected `vokra.model.arch` value, written by
/// `vokra-cli convert --model canary-1b-flash`.
///
/// Deliberately distinct from `crate::canary::EXPECTED_ARCH` (`"canary"`) and
/// `crate::canary_qwen::EXPECTED_ARCH` (`"canary-qwen"`): the decoder-layer
/// axis differs (4 vs 8) and the decoder *class* differs (AED vs Qwen LLM).
/// Silently sharing an arch tag would let a loader walk the wrong tensor
/// manifest without crashing (FR-EX-08 — no silent misroute).
pub const ARCH: &str = "canary-1b-flash";

/// Expected `vokra.model.name` value written by the converter.
pub const NAME: &str = "canary-1b-flash";

/// Expected `vokra.model.category` value — the `"asr"` tier, shared with
/// `canary` / `canary-qwen` / `parakeet` / `parakeet-ctc` / `kyutai-stt`.
pub const CATEGORY: &str = "asr";

/// Upstream HuggingFace repository slug recorded under
/// `vokra.provenance.upstream_hf` by the converter. Echoed in loud-partial
/// diagnostics so a reader never has to re-fetch a manifest to find it.
pub const UPSTREAM_HF: &str = "nvidia/canary-1b-flash";

/// Canonical weight-license SPDX the converter stamps by default
/// (`cc-by-4.0` → [`LicenseClass::AttributionRequired`]). A caller-supplied
/// `--license` override on the converter side replaces it; this binder reads
/// whatever the artifact actually carries and never assumes.
pub const DEFAULT_LICENSE: &str = "cc-by-4.0";

/// PCM sample rate Canary-1B-Flash expects — **16 000 Hz** mono
/// (model card: "16kHz Audio, .wav and .flac audio formats, Monochannel").
pub const CANARY_1B_FLASH_SAMPLE_RATE: u32 = 16_000;

/// FastConformer encoder depth — **32 layers** (model card, transcribed by
/// the converter 2026-08-03). Identical to Canary-1B-v2: the Flash
/// distillation shrinks the *decoder*, not the encoder.
pub const ENCODER_N_LAYER: usize = 32;

/// Transformer AED decoder depth — **4 layers** (model card). This is the
/// Flash-specific shrinkage (Canary-1B-v2: 8, Canary-1B-v1: 24) and the axis
/// behind the "1000+ RTFx" throughput claim. Load-bearing: a loader that
/// walks 8 decoder blocks against a 4-block manifest mis-reads silently.
pub const DECODER_N_LAYER: usize = 4;

/// The four languages Canary-1B-Flash covers (model card): English, German,
/// French, Spanish — a strict subset of Canary-1B-v2's 25.
///
/// Recorded as ISO 639-1 codes. The concrete `<source_lang>` /
/// `<target_lang>` **token spellings** are *not* pinned here: they live in the
/// `.nemo` SentencePiece model and no primary source enumerates them, so
/// asserting them would be fabrication.
pub const SUPPORTED_LANGUAGES: [&str; 4] = ["en", "de", "fr", "es"];

/// The task-token families the unified Canary SentencePiece vocabulary
/// carries (converter docstring, transcribed from the model card).
///
/// These are the token *families*, verbatim as the upstream documents name
/// them — not their integer ids, which the `.nemo` tokenizer owns.
pub const TASK_TOKENS: [&str; 8] = [
    "<source_lang>",
    "<target_lang>",
    "<taskname>",
    "<pnc>",
    "<itn>",
    "<timestamp>",
    "<diarize>",
    "<emotion>",
];

/// Primary-source anchor: the HF model card.
pub const PRIMARY_SOURCE_HF: &str = "https://huggingface.co/nvidia/canary-1b-flash";

/// Primary-source anchor: the shared FastConformer-Transformer AED reference
/// config whose variant table records `canary-1b-flash` explicitly.
pub const PRIMARY_SOURCE_FAMILY_YAML: &str = "github.com/NVIDIA-NeMo/Speech/blob/main/examples/asr/conf/speech_multitask/\
     fast-conformer_aed.yaml";

/// Primary-source anchor: the in-repo `.nemo` → safetensors bridge a
/// downstream runs before the converter (uv-managed, Python 3.12; the runtime
/// itself never sees Python — FR-LD-05 / NFR-DS-02).
pub const PRIMARY_SOURCE_NEMO_PREP: &str = "tools/parity/nemo_pt_to_safetensors.py";

// ---------------------------------------------------------------------------
// Forward-compatible `vokra.canary_1b_flash.*` axis-override keys.
//
// The CURRENT converter stamps NONE of these — it writes only the
// `vokra.model.*` / `vokra.provenance.*` / `vokra.schema.*` groups plus the
// verbatim tensor payload. They are defined here so that (a) the naming
// convention is fixed by the reader before a writer exists (the sibling
// convention is `vokra.<arch-with-underscores>.*`, cf. `vokra.canary.*` /
// `vokra.canary_qwen.*` / `vokra.parakeet_ctc.*`), and (b) the moment a
// converter revision starts stamping real `.nemo`-extracted axes, this binder
// honours them with no runtime change.
//
// Booleans ride as `u32` (0 / 1) — the sibling convention; `hidden_act` rides
// as a string.
// ---------------------------------------------------------------------------

/// Optional override for [`Canary1bFlashConfig::sample_rate`].
pub const GGUF_KEY_SAMPLE_RATE: &str = "vokra.canary_1b_flash.sample_rate";

/// Optional override for the FastConformer encoder layer count.
pub const GGUF_KEY_ENC_N_LAYER: &str = "vokra.canary_1b_flash.arch.encoder.n_layer";
/// Optional override for the FastConformer encoder residual width.
pub const GGUF_KEY_ENC_D_MODEL: &str = "vokra.canary_1b_flash.arch.encoder.d_model";
/// Optional override for the encoder Q-head count.
pub const GGUF_KEY_ENC_N_HEAD: &str = "vokra.canary_1b_flash.arch.encoder.n_head";
/// Optional override for the encoder KV-head count (MHA when equal to
/// `n_head`).
pub const GGUF_KEY_ENC_N_HEAD_KV: &str = "vokra.canary_1b_flash.arch.encoder.n_head_kv";
/// Optional override for the encoder FFN inner width.
pub const GGUF_KEY_ENC_FFN_DIM: &str = "vokra.canary_1b_flash.arch.encoder.ffn_dim";
/// Optional override for the FastConformer depthwise convolution kernel size.
pub const GGUF_KEY_ENC_CONV_KERNEL: &str = "vokra.canary_1b_flash.arch.encoder.conv_kernel_size";
/// Optional override for the log-mel channel count on the encoder input.
pub const GGUF_KEY_ENC_IN_DIM: &str = "vokra.canary_1b_flash.arch.encoder.in_dim";
/// Optional override for the FastConformer subsampling factor.
pub const GGUF_KEY_ENC_SUBSAMPLING_FACTOR: &str =
    "vokra.canary_1b_flash.arch.encoder.subsampling_factor";
/// Optional override for the subsample-stem convolution kernel size.
pub const GGUF_KEY_ENC_SUB_CONV_KERNEL: &str =
    "vokra.canary_1b_flash.arch.encoder.subsampling_conv_kernel_size";
/// Optional override for the subsample-stem convolution stride.
pub const GGUF_KEY_ENC_SUB_CONV_STRIDE: &str =
    "vokra.canary_1b_flash.arch.encoder.subsampling_conv_stride";
/// Optional override for the subsample-stem convolution channel count.
pub const GGUF_KEY_ENC_SUB_CONV_CHANNELS: &str =
    "vokra.canary_1b_flash.arch.encoder.subsampling_conv_channels";
/// Optional override for the encoder positional-embedding upper bound.
pub const GGUF_KEY_ENC_MAX_POS: &str = "vokra.canary_1b_flash.arch.encoder.max_position_embeddings";
/// Optional override (`0` / `1`) for the encoder attention-bias flag.
pub const GGUF_KEY_ENC_ATTN_BIAS: &str = "vokra.canary_1b_flash.arch.encoder.attention_bias";
/// Optional override (`0` / `1`) for the encoder convolution-bias flag.
pub const GGUF_KEY_ENC_CONV_BIAS: &str = "vokra.canary_1b_flash.arch.encoder.convolution_bias";
/// Optional override (`0` / `1`) for the subsample-stem `xscaling` flag.
pub const GGUF_KEY_ENC_SCALE_INPUT: &str = "vokra.canary_1b_flash.arch.encoder.scale_input";

/// Optional override for the Transformer AED decoder layer count.
pub const GGUF_KEY_DEC_N_LAYER: &str = "vokra.canary_1b_flash.arch.decoder.n_layer";
/// Optional override for the decoder residual width.
pub const GGUF_KEY_DEC_D_MODEL: &str = "vokra.canary_1b_flash.arch.decoder.d_model";
/// Optional override for the decoder attention-head count.
pub const GGUF_KEY_DEC_N_HEAD: &str = "vokra.canary_1b_flash.arch.decoder.n_head";
/// Optional override for the decoder FFN inner width.
pub const GGUF_KEY_DEC_FFN_DIM: &str = "vokra.canary_1b_flash.arch.decoder.ffn_dim";
/// Optional override for the decoder maximum sequence length.
pub const GGUF_KEY_DEC_MAX_SEQ: &str = "vokra.canary_1b_flash.arch.decoder.max_sequence_length";
/// Optional override (`0` / `1`) for the decoder pre-LayerNorm flag.
pub const GGUF_KEY_DEC_PRE_LN: &str = "vokra.canary_1b_flash.arch.decoder.pre_ln";
/// Optional override (string) for the decoder FFN activation name.
pub const GGUF_KEY_DEC_HIDDEN_ACT: &str = "vokra.canary_1b_flash.arch.decoder.hidden_act";

/// Optional override for the vocabulary / head width.
pub const GGUF_KEY_HEAD_VOCAB_SIZE: &str = "vokra.canary_1b_flash.head.vocab_size";
/// Optional override for the tokenizer pad-token id.
pub const GGUF_KEY_HEAD_PAD_ID: &str = "vokra.canary_1b_flash.head.pad_token_id";
/// Optional override for the decoder beginning-of-sequence token id.
pub const GGUF_KEY_HEAD_BOS_ID: &str = "vokra.canary_1b_flash.head.bos_token_id";
/// Optional override for the decoder end-of-sequence token id.
pub const GGUF_KEY_HEAD_EOS_ID: &str = "vokra.canary_1b_flash.head.eos_token_id";

// ---------------------------------------------------------------------------
// Task surface
// ---------------------------------------------------------------------------

/// The multitask modes Canary-1B-Flash exposes (model card: "multi-task
/// ASR / AST").
///
/// The enum records **what the model can be asked to do**; the concrete
/// `<taskname>` token spelling that selects a mode lives in the `.nemo`
/// SentencePiece model and is deliberately **not** pinned here (no primary
/// source enumerates it — asserting one would be fabrication).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Canary1bFlashTask {
    /// Automatic speech recognition — transcribe in the source language.
    Asr,
    /// Automatic speech translation — translate speech into the target
    /// language (any ordered pair over [`SUPPORTED_LANGUAGES`]).
    Ast,
}

impl Canary1bFlashTask {
    /// Stable lower-case identifier for diagnostics and CLI surfacing.
    ///
    /// This is a **Vokra-side** label, not the upstream `<taskname>` token
    /// value — see the type docstring.
    #[inline]
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Asr => "asr",
            Self::Ast => "ast",
        }
    }
}

/// Where the axes in a [`Canary1bFlashConfig`] came from.
///
/// This marker is the honest answer to "did the artifact tell us its shape, or
/// did we anchor it to the published family reference?". The current converter
/// stamps no `vokra.canary_1b_flash.*` axes, so every real converted GGUF
/// resolves to [`Self::FamilyAnchored`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Canary1bFlashConfigSource {
    /// No `vokra.canary_1b_flash.*` axis chunk was present; every axis comes
    /// from [`Canary1bFlashConfig::canary_1b_flash`], i.e. from the model card
    /// plus the family reference YAML.
    FamilyAnchored,
    /// At least one `vokra.canary_1b_flash.*` axis chunk was present and
    /// overrode its family-anchored default.
    GgufStamped,
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Resolved Canary-1B-Flash hparam snapshot.
///
/// The three axis groups reuse [`crate::canary`]'s types verbatim — the Flash
/// variant is the *same* FastConformer + Transformer AED shape with a shallower
/// decoder, so inventing a third set of structs would only create drift.
#[derive(Debug, Clone, PartialEq)]
pub struct Canary1bFlashConfig {
    /// FastConformer encoder axes (32 layers for Flash).
    pub encoder: CanaryEncoderConfig,
    /// Transformer AED decoder axes (**4 layers** for Flash).
    pub decoder: CanaryDecoderConfig,
    /// Vocabulary / special-token / head axes. `vocab_size` and the three
    /// token ids are `0` placeholder sentinels until the `.nemo` tokenizer is
    /// extracted — [`Self::validate_for_forward`] refuses them.
    pub head: CanaryHeadConfig,
    /// PCM sample rate the model expects (16 000 Hz).
    pub sample_rate: u32,
    /// Whether these axes were read off the artifact or anchored to the
    /// published family reference.
    pub source: Canary1bFlashConfigSource,
}

impl Canary1bFlashConfig {
    /// Primary-source Canary-1B-Flash axes.
    ///
    /// Provenance per field:
    ///
    /// - **Model card** (`nvidia/canary-1b-flash`, transcribed 2026-08-03):
    ///   `encoder.n_layer = 32`, `decoder.n_layer = 4`, `sample_rate = 16 kHz`.
    /// - **Family reference YAML variant table**, which lists
    ///   `canary-1b-flash` by name: `encoder.d_model = 1024`
    ///   (`model_defaults.asr_enc_hidden`), `decoder.d_model = 1024`
    ///   (`.lm_dec_hidden`), `decoder.max_sequence_length = 1024`.
    /// - **Family defaults** shared by every Canary variant in the same YAML:
    ///   `n_head = 8`, `ffn_dim = 4 × d_model = 4096`, `conv_kernel_size = 9`,
    ///   `in_dim = 128` (`preprocessor.features`), `subsampling_factor = 8`
    ///   with stride-2 kernel-3 `dw_striding` stages and 256 channels,
    ///   `pos_emb_max_len = 5000`, `untie_biases = true` → `attention_bias`,
    ///   bias-free convolutions, `xscaling = false`, `pre_ln = true`,
    ///   `hidden_act = "relu"`.
    /// - **No source at all**: `head.vocab_size` / `pad` / `bos` / `eos`. The
    ///   model card describes the tokenizer as "the 4-language subset of the
    ///   unified Canary SentencePiece" without stating a width, and the ids
    ///   live inside the `.nemo` tarball. They stay `0` — a sentinel
    ///   [`Self::validate_for_forward`] rejects — rather than being copied
    ///   from Canary-1B-v2's 25-language 16 384 (a different tokenizer).
    ///
    /// The `.nemo` `model_config.yaml` is the ultimate authority; a divergence
    /// surfaces through the shape gate, never through a silent widen
    /// (FR-EX-08).
    #[must_use]
    pub fn canary_1b_flash() -> Self {
        // Start from the shared family encoder (identical axes: Flash keeps
        // Canary-1B-v2's 32-layer / 1024-wide FastConformer verbatim) so the
        // two cannot drift apart silently.
        let family = crate::canary::CanaryConfig::canary_1b_v2();
        Self {
            encoder: CanaryEncoderConfig {
                n_layer: ENCODER_N_LAYER,
                ..family.encoder
            },
            decoder: CanaryDecoderConfig {
                // The Flash-specific shrinkage — the ONLY topology axis that
                // differs from Canary-1B-v2.
                n_layer: DECODER_N_LAYER,
                // `max_sequence_length = 1024` is attested for
                // `canary-1b-flash` BY NAME in the family YAML variant table
                // (unlike Canary-1B-v2, where 1024 is adopted by family
                // convention). Same value, stronger provenance.
                max_sequence_length: 1024,
                ..family.decoder
            },
            head: CanaryHeadConfig {
                // `0` placeholder sentinels — see the fn docstring. NOT copied
                // from Canary-1B-v2: that is a 25-language tokenizer, this is
                // a 4-language one.
                vocab_size: 0,
                pad_token_id: 0,
                bos_token_id: 0,
                eos_token_id: 0,
            },
            sample_rate: CANARY_1B_FLASH_SAMPLE_RATE,
            source: Canary1bFlashConfigSource::FamilyAnchored,
        }
    }

    /// Miniature well-formed config for shape / stability tests.
    ///
    /// Dimensions are tiny so shape algebra can be exercised in KB, but the
    /// *relationships* (MHA head split, even head dims, 4-layer decoder,
    /// cross-attn width coupling) mirror the real model. Unlike
    /// [`Self::canary_1b_flash`] the head axes are real values, so this config
    /// **passes** [`Self::validate_for_forward`].
    #[must_use]
    pub fn tiny_for_tests() -> Self {
        let family = crate::canary::CanaryConfig::tiny_for_tests();
        Self {
            encoder: family.encoder,
            decoder: CanaryDecoderConfig {
                n_layer: DECODER_N_LAYER,
                ..family.decoder
            },
            head: family.head,
            sample_rate: CANARY_1B_FLASH_SAMPLE_RATE,
            source: Canary1bFlashConfigSource::FamilyAnchored,
        }
    }

    /// True iff every axis came from the published family reference rather
    /// than from a `vokra.canary_1b_flash.*` stamp on the artifact.
    #[inline]
    #[must_use]
    pub const fn is_family_anchored(&self) -> bool {
        matches!(self.source, Canary1bFlashConfigSource::FamilyAnchored)
    }

    /// Rejects `0`-placeholder / ill-formed configs before any forward runs.
    ///
    /// **Delegates** to the shared Canary-family validator
    /// (`crate::canary::CanaryConfig::validate_for_forward`) — the Flash
    /// variant has the same shape algebra, so duplicating it here would only
    /// create drift. The delegated message is re-prefixed so a reader sees
    /// which model surfaced the failure.
    ///
    /// Note that [`Self::canary_1b_flash`] **fails** this check on
    /// `head.vocab_size == 0`. That is deliberate and is the mechanism that
    /// stops a caller from running a hallucinated forward on a tokenizer that
    /// has not been extracted yet (FR-EX-08).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] naming the offending field.
    pub fn validate_for_forward(&self) -> Result<()> {
        let family = crate::canary::CanaryConfig {
            encoder: self.encoder.clone(),
            decoder: self.decoder.clone(),
            head: self.head.clone(),
            sample_rate: self.sample_rate,
        };
        family.validate_for_forward().map_err(|e| match e {
            VokraError::InvalidArgument(m) => {
                // Cosmetic: strip the delegate's own prefix so the composed
                // message does not read "canary-1b-flash config: canary
                // config: ...". Falls back to the full message if the
                // delegate's wording ever changes.
                let detail = m.strip_prefix("canary config: ").unwrap_or(&m);
                VokraError::InvalidArgument(format!(
                    "canary-1b-flash config (shared Canary-family validator): {detail} \
                     — primary source: {PRIMARY_SOURCE_HF}"
                ))
            }
            other => other,
        })
    }

    /// Resolves the axes for a Canary-1B-Flash GGUF.
    ///
    /// Starts from [`Self::canary_1b_flash`] (the primary-source anchor) and
    /// applies any `vokra.canary_1b_flash.*` override the artifact carries.
    /// The **current converter stamps none of them**, so a real converted GGUF
    /// resolves to [`Canary1bFlashConfigSource::FamilyAnchored`] — which is
    /// exactly why this reader is permissive about absence and strict about
    /// malformation:
    ///
    /// - an **absent** key is normal (the writer does not emit it yet);
    /// - a **present but wrong-dtype** key is a corrupted / hand-assembled
    ///   artifact and fails loud (FR-EX-08 — never silently ignored).
    ///
    /// # Loud-partial posture
    ///
    /// This reader deliberately does **not** call
    /// [`Self::validate_for_forward`]. The family-anchored config carries `0`
    /// head sentinels; validating here would make the GGUF unloadable and
    /// prevent [`Canary1bFlashAsr::transcribe`] from firing its specific
    /// loud-partial message. Same posture as `canary_qwen`.
    ///
    /// # Errors
    ///
    /// [`VokraError::ModelLoad`] naming the key when a `vokra.canary_1b_flash.*`
    /// chunk is present with the wrong dtype.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        let mut cfg = Self::canary_1b_flash();
        let mut stamped = 0usize;

        // ---- sample rate --------------------------------------------------
        if let Some(v) = opt_u32(file, GGUF_KEY_SAMPLE_RATE)? {
            cfg.sample_rate = v;
            stamped += 1;
        }

        // ---- encoder ------------------------------------------------------
        if let Some(v) = opt_usize(file, GGUF_KEY_ENC_N_LAYER)? {
            cfg.encoder.n_layer = v;
            stamped += 1;
        }
        if let Some(v) = opt_usize(file, GGUF_KEY_ENC_D_MODEL)? {
            cfg.encoder.d_model = v;
            stamped += 1;
        }
        if let Some(v) = opt_usize(file, GGUF_KEY_ENC_N_HEAD)? {
            cfg.encoder.n_head = v;
            stamped += 1;
        }
        if let Some(v) = opt_usize(file, GGUF_KEY_ENC_N_HEAD_KV)? {
            cfg.encoder.n_head_kv = v;
            stamped += 1;
        }
        if let Some(v) = opt_usize(file, GGUF_KEY_ENC_FFN_DIM)? {
            cfg.encoder.ffn_dim = v;
            stamped += 1;
        }
        if let Some(v) = opt_usize(file, GGUF_KEY_ENC_CONV_KERNEL)? {
            cfg.encoder.conv_kernel_size = v;
            stamped += 1;
        }
        if let Some(v) = opt_usize(file, GGUF_KEY_ENC_IN_DIM)? {
            cfg.encoder.in_dim = v;
            stamped += 1;
        }
        if let Some(v) = opt_usize(file, GGUF_KEY_ENC_SUBSAMPLING_FACTOR)? {
            cfg.encoder.subsampling_factor = v;
            stamped += 1;
        }
        if let Some(v) = opt_usize(file, GGUF_KEY_ENC_SUB_CONV_KERNEL)? {
            cfg.encoder.subsampling_conv_kernel_size = v;
            stamped += 1;
        }
        if let Some(v) = opt_usize(file, GGUF_KEY_ENC_SUB_CONV_STRIDE)? {
            cfg.encoder.subsampling_conv_stride = v;
            stamped += 1;
        }
        if let Some(v) = opt_usize(file, GGUF_KEY_ENC_SUB_CONV_CHANNELS)? {
            cfg.encoder.subsampling_conv_channels = v;
            stamped += 1;
        }
        if let Some(v) = opt_usize(file, GGUF_KEY_ENC_MAX_POS)? {
            cfg.encoder.max_position_embeddings = v;
            stamped += 1;
        }
        if let Some(v) = opt_u32(file, GGUF_KEY_ENC_ATTN_BIAS)? {
            cfg.encoder.attention_bias = v != 0;
            stamped += 1;
        }
        if let Some(v) = opt_u32(file, GGUF_KEY_ENC_CONV_BIAS)? {
            cfg.encoder.convolution_bias = v != 0;
            stamped += 1;
        }
        if let Some(v) = opt_u32(file, GGUF_KEY_ENC_SCALE_INPUT)? {
            cfg.encoder.scale_input = v != 0;
            stamped += 1;
        }

        // ---- decoder ------------------------------------------------------
        if let Some(v) = opt_usize(file, GGUF_KEY_DEC_N_LAYER)? {
            cfg.decoder.n_layer = v;
            stamped += 1;
        }
        if let Some(v) = opt_usize(file, GGUF_KEY_DEC_D_MODEL)? {
            cfg.decoder.d_model = v;
            stamped += 1;
        }
        if let Some(v) = opt_usize(file, GGUF_KEY_DEC_N_HEAD)? {
            cfg.decoder.n_head = v;
            stamped += 1;
        }
        if let Some(v) = opt_usize(file, GGUF_KEY_DEC_FFN_DIM)? {
            cfg.decoder.ffn_dim = v;
            stamped += 1;
        }
        if let Some(v) = opt_usize(file, GGUF_KEY_DEC_MAX_SEQ)? {
            cfg.decoder.max_sequence_length = v;
            stamped += 1;
        }
        if let Some(v) = opt_u32(file, GGUF_KEY_DEC_PRE_LN)? {
            cfg.decoder.pre_ln = v != 0;
            stamped += 1;
        }
        if let Some(v) = opt_string(file, GGUF_KEY_DEC_HIDDEN_ACT)? {
            cfg.decoder.hidden_act = v;
            stamped += 1;
        }

        // ---- head ---------------------------------------------------------
        if let Some(v) = opt_usize(file, GGUF_KEY_HEAD_VOCAB_SIZE)? {
            cfg.head.vocab_size = v;
            stamped += 1;
        }
        if let Some(v) = opt_u32(file, GGUF_KEY_HEAD_PAD_ID)? {
            cfg.head.pad_token_id = v;
            stamped += 1;
        }
        if let Some(v) = opt_u32(file, GGUF_KEY_HEAD_BOS_ID)? {
            cfg.head.bos_token_id = v;
            stamped += 1;
        }
        if let Some(v) = opt_u32(file, GGUF_KEY_HEAD_EOS_ID)? {
            cfg.head.eos_token_id = v;
            stamped += 1;
        }

        cfg.source = if stamped == 0 {
            Canary1bFlashConfigSource::FamilyAnchored
        } else {
            Canary1bFlashConfigSource::GgufStamped
        };
        Ok(cfg)
    }
}

/// Reads an **optional** `u32`-range integer chunk.
///
/// `None` when the key is absent (normal — the converter stamps no axis
/// chunks). A present value that is not a `u32`-range unsigned integer is a
/// loud [`VokraError::ModelLoad`]: silently ignoring a malformed override
/// would run the model on the family default while the artifact claimed
/// otherwise (FR-EX-08).
fn opt_u32(file: &GgufFile, key: &str) -> Result<Option<u32>> {
    let Some(value) = file.get(key) else {
        return Ok(None);
    };
    value
        .as_u64()
        .and_then(|n| u32::try_from(n).ok())
        .map(Some)
        .ok_or_else(|| {
            VokraError::ModelLoad(format!(
                "canary-1b-flash: metadata key `{key}` is present but is not a \
                 u32-range unsigned integer (got {value:?}). This axis-override \
                 group is optional — the canonical converter stamps none of it — \
                 but a key that IS present must be well-formed; ignoring it would \
                 silently run the family-anchored default while the artifact \
                 claimed a different shape (FR-EX-08). Primary source: \
                 {PRIMARY_SOURCE_HF}"
            ))
        })
}

/// [`opt_u32`] widened to `usize` for the dimension axes.
fn opt_usize(file: &GgufFile, key: &str) -> Result<Option<usize>> {
    Ok(opt_u32(file, key)?.map(|v| v as usize))
}

/// Reads an **optional** string chunk, with the same present-but-malformed
/// loud posture as [`opt_u32`].
fn opt_string(file: &GgufFile, key: &str) -> Result<Option<String>> {
    let Some(value) = file.get(key) else {
        return Ok(None);
    };
    value.as_str().map(|s| Some(s.to_owned())).ok_or_else(|| {
        VokraError::ModelLoad(format!(
            "canary-1b-flash: metadata key `{key}` is present but is not a string \
                 (got {value:?}). Ignoring a malformed override would silently run the \
                 family-anchored default (FR-EX-08). Primary source: {PRIMARY_SOURCE_HF}"
        ))
    })
}

// ---------------------------------------------------------------------------
// Weights — verbatim-upstream-name tensor manifest
// ---------------------------------------------------------------------------

/// Tensor manifest bound from a Canary-1B-Flash GGUF.
///
/// The converter passes **every float tensor through under its verbatim
/// upstream safetensors name** (the name produced by
/// `tools/parity/nemo_pt_to_safetensors.py` flattening the `.nemo`
/// `state_dict`). Nothing in-repo transcribes NeMo's `EncDecMultiTaskModel`
/// naming, so this store deliberately does **not** walk names into typed
/// encoder / decoder slots: a guessed manifest would bind shape-valid garbage.
/// Instead it records what is actually on disk and offers loud lookups
/// ([`Self::require_tensor`] / [`Self::require_tensor_dims`]) that the
/// follow-up real-weight wave uses once the manifest is known.
///
/// **Contract**: [`Self::from_gguf`] refuses a zero-tensor GGUF — an
/// 883 M-parameter FastConformer + AED checkpoint always carries hundreds of
/// tensors, so an empty manifest is always a mis-produced artifact, never a
/// valid one (FR-EX-08 — no all-zero forward).
#[derive(Debug, Clone)]
pub struct Canary1bFlashWeights {
    /// `(upstream name, GGUF dims)` in on-disk order. Order is preserved so
    /// diagnostics are deterministic across runs.
    tensors: Vec<(String, Vec<usize>)>,
}

impl Canary1bFlashWeights {
    /// Scans `file` for the checkpoint tensors.
    ///
    /// # Errors
    ///
    /// [`VokraError::ModelLoad`] when the GGUF carries zero tensors.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        let tensors: Vec<(String, Vec<usize>)> = file
            .tensors()
            .iter()
            .map(|info| {
                let dims: Vec<usize> = info.dimensions.iter().map(|&d| d as usize).collect();
                (info.name.clone(), dims)
            })
            .collect();

        if tensors.is_empty() {
            return Err(VokraError::ModelLoad(format!(
                "canary-1b-flash: GGUF carries zero tensors — refusing to bind an \
                 all-zero forward (FR-EX-08). A legitimate Canary-1B-Flash checkpoint \
                 is 883 M parameters (arch={ARCH}, name={NAME}): a 32-layer \
                 FastConformer encoder plus a 4-layer Transformer AED decoder carry \
                 hundreds of Linear / LayerNorm / Conv1D tensors, so zero tensors \
                 always signals a mis-produced GGUF. Re-run \
                 `vokra-cli convert --model canary-1b-flash` against an upstream \
                 `{UPSTREAM_HF}` checkpoint prepared with `{PRIMARY_SOURCE_NEMO_PREP}`. \
                 Primary source: {PRIMARY_SOURCE_HF}"
            )));
        }
        Ok(Self { tensors })
    }

    /// Number of tensors discovered on disk.
    #[inline]
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    /// Every discovered tensor name, in on-disk order.
    #[must_use]
    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensors.iter().map(|(n, _)| n.as_str()).collect()
    }

    /// GGUF dimensions of `name`, or `None` when it is absent.
    #[must_use]
    pub fn tensor_dims(&self, name: &str) -> Option<&[usize]> {
        self.tensors
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, d)| d.as_slice())
    }

    /// How many discovered tensors start with `prefix`.
    ///
    /// A pure observation over what is on disk — it asserts **no** naming
    /// scheme (the upstream NeMo prefixes are not transcribed anywhere
    /// in-repo). The follow-up real-weight wave uses it to sanity-check a
    /// manifest once the naming *is* known.
    #[must_use]
    pub fn count_with_prefix(&self, prefix: &str) -> usize {
        self.tensors
            .iter()
            .filter(|(n, _)| n.starts_with(prefix))
            .count()
    }

    /// Looks up `name`, failing loud when it is absent.
    ///
    /// The error names the missing tensor and lists up to five sibling names
    /// that share its first dotted segment (or, failing that, the first five
    /// names on disk) so a reader diagnosing a manifest mismatch can see what
    /// the artifact *does* contain without dumping the whole GGUF.
    ///
    /// # Errors
    ///
    /// [`VokraError::ModelLoad`] naming the missing tensor.
    pub fn require_tensor(&self, name: &str) -> Result<&[usize]> {
        if let Some(dims) = self.tensor_dims(name) {
            return Ok(dims);
        }
        let segment = name.split('.').next().unwrap_or(name);
        let mut near: Vec<&str> = self
            .tensors
            .iter()
            .filter(|(n, _)| n.starts_with(segment))
            .map(|(n, _)| n.as_str())
            .take(5)
            .collect();
        if near.is_empty() {
            near = self
                .tensors
                .iter()
                .map(|(n, _)| n.as_str())
                .take(5)
                .collect();
        }
        Err(VokraError::ModelLoad(format!(
            "canary-1b-flash: required tensor `{name}` is absent from the GGUF \
             ({count} tensors present; nearest names on disk: {near:?}). The converter \
             passes upstream safetensors names through verbatim, so a mismatch means \
             either the checkpoint was prepared with a different \
             `{PRIMARY_SOURCE_NEMO_PREP}` invocation (e.g. a `--tensor-prefix-strip` \
             that removed a prefix) or the caller is walking a manifest transcribed \
             from a different Canary variant. Refusing to substitute a zero tensor \
             (FR-EX-08). Primary source: {PRIMARY_SOURCE_HF}",
            count = self.tensors.len(),
        )))
    }

    /// Looks up `name` and checks its dimensions against `expected`.
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] naming the tensor when it is absent
    ///   (via [`Self::require_tensor`]).
    /// - [`VokraError::ModelLoad`] naming the tensor plus **both** the
    ///   expected and the actual dims on a shape mismatch — never a silent
    ///   reshape or truncation (FR-EX-08).
    pub fn require_tensor_dims(&self, name: &str, expected: &[usize]) -> Result<()> {
        let actual = self.require_tensor(name)?;
        if actual != expected {
            return Err(VokraError::ModelLoad(format!(
                "canary-1b-flash: tensor `{name}` has dims {actual:?} but the resolved \
                 config expects {expected:?} — refusing to reshape or truncate \
                 silently (FR-EX-08). Either the GGUF was produced from a different \
                 Canary variant (Canary-1B-v2 ships an 8-layer decoder, \
                 Canary-Qwen-2.5B a Qwen LLM decoder) or the axis overrides in the \
                 `vokra.canary_1b_flash.*` group disagree with the payload. Primary \
                 source: {PRIMARY_SOURCE_HF}"
            )));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

/// Canary-1B-Flash ASR / AST engine handle.
///
/// Bind with [`Self::from_gguf`] (or the compliance-gated
/// [`Self::from_gguf_with_policy`] / [`Self::from_path`]), then call
/// [`Self::transcribe`] / [`Self::transcribe_with_task`]. See the module
/// docstring for the loud-partial contract on the forward path.
#[derive(Debug, Clone)]
pub struct Canary1bFlashAsr {
    cfg: Canary1bFlashConfig,
    weights: Canary1bFlashWeights,
    weight_license: LicenseClass,
    attribution: Option<String>,
}

impl Canary1bFlashAsr {
    /// Binds a Canary-1B-Flash GGUF: verifies arch strictly, resolves the
    /// axes, discovers the tensor manifest, and surfaces the weight-license
    /// class plus the FR-MD-09 attribution text.
    ///
    /// Every failure is a distinct [`VokraError::ModelLoad`] naming the exact
    /// key or tensor at fault, so a reader diagnosing a mis-produced GGUF has
    /// one place to walk (FR-EX-08 — never a silent partial bind).
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] when `vokra.model.arch` is absent, or is
    ///   not `"canary-1b-flash"` (the message names both the found and the
    ///   expected tag and enumerates the Canary neighbourhood).
    /// - [`VokraError::ModelLoad`] when a present `vokra.canary_1b_flash.*`
    ///   override has the wrong dtype
    ///   ([`Canary1bFlashConfig::from_gguf`]).
    /// - [`VokraError::ModelLoad`] when the GGUF carries zero tensors
    ///   ([`Canary1bFlashWeights::from_gguf`]).
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        // 1. Arch first, so a mis-routed model surfaces a specific message
        //    instead of a downstream missing-tensor trail.
        verify_arch(file)?;

        // 2. Axes. Deliberately NOT validate_for_forward'd — the
        //    family-anchored head sentinels are `0` pending .nemo extraction,
        //    and a strict validate here would make every real artifact
        //    unloadable and suppress the specific loud-partial message.
        let cfg = Canary1bFlashConfig::from_gguf(file)?;

        // 3. Tensor manifest with the non-emptiness gate.
        let weights = Canary1bFlashWeights::from_gguf(file)?;

        // 4. Provenance surfacing. The converter stamps
        //    `AttributionRequired` (CC-BY 4.0); a GGUF missing the stamp
        //    reads back as `Unknown` — fail-closed at the M2-13 gate.
        let weight_license = file
            .get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
            .and_then(|v| v.as_str())
            .and_then(LicenseClass::from_class_str)
            .unwrap_or(LicenseClass::Unknown);
        let attribution = file
            .get(chunks::KEY_PROVENANCE_ATTRIBUTION)
            .and_then(|v| v.as_str())
            .map(str::to_owned);

        Ok(Self {
            cfg,
            weights,
            weight_license,
            attribution,
        })
    }

    /// Loads a Canary-1B-Flash GGUF from raw bytes under `policy` (the M2-13
    /// weight-license gate).
    ///
    /// Canary-1B-Flash ships **CC-BY 4.0** →
    /// [`LicenseClass::AttributionRequired`], which is commercially permitted,
    /// so a correctly stamped artifact passes under
    /// [`CompliancePolicy::strict`] without a research opt-in. An artifact
    /// with no provenance stamp resolves to [`LicenseClass::Unknown`] and is
    /// refused by the gate — fail-closed, never a silent substitution.
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] on GGUF parse failure, or on a wrong /
    ///   missing `vokra.model.arch`.
    /// - `VokraError::ResearchLicenseRequired` from the compliance gate when
    ///   the weight class is gated and `policy` grants no research opt-in.
    /// - See [`Self::from_gguf`] for the remaining bind errors.
    pub fn from_gguf_with_policy(bytes: &[u8], policy: &CompliancePolicy) -> Result<Self> {
        let file = GgufFile::parse(bytes.to_vec())
            .map_err(|e| VokraError::ModelLoad(format!("canary-1b-flash GGUF: {e}")))?;
        // Arch before the compliance gate so a mis-routed artifact reports the
        // arch mismatch (the actionable fact) rather than a licence verdict
        // about a model the caller never meant to load.
        verify_arch(&file)?;
        check_weight_license(&file, policy)?;
        Self::from_gguf(&file)
    }

    /// Loads a Canary-1B-Flash GGUF from a path under
    /// [`CompliancePolicy::strict`].
    ///
    /// # Errors
    ///
    /// - `VokraError::Io` on read failure.
    /// - See [`Self::from_gguf_with_policy`].
    pub fn from_path(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let bytes = std::fs::read(path.as_ref()).map_err(VokraError::Io)?;
        Self::from_gguf_with_policy(&bytes, &CompliancePolicy::strict())
    }

    /// The resolved configuration.
    #[inline]
    #[must_use]
    pub fn config(&self) -> &Canary1bFlashConfig {
        &self.cfg
    }

    /// The bound tensor manifest.
    #[inline]
    #[must_use]
    pub fn weights(&self) -> &Canary1bFlashWeights {
        &self.weights
    }

    /// Number of tensors discovered on disk.
    #[inline]
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.weights.tensor_count()
    }

    /// PCM sample rate the bound model expects.
    #[inline]
    #[must_use]
    pub const fn sample_rate(&self) -> u32 {
        self.cfg.sample_rate
    }

    /// The weight-license class surfaced from
    /// `vokra.provenance.weight_license`.
    ///
    /// [`LicenseClass::AttributionRequired`] for a correctly stamped
    /// Canary-1B-Flash artifact (CC-BY 4.0); [`LicenseClass::Unknown`] when
    /// the stamp is absent (fail-closed).
    #[inline]
    #[must_use]
    pub const fn weight_license(&self) -> LicenseClass {
        self.weight_license
    }

    /// The FR-MD-09 attribution text stamped under
    /// `vokra.provenance.attribution`, if any.
    ///
    /// CC-BY 4.0 requires a downstream to display attribution alongside the
    /// model output, so this is surfaced rather than buried: a consumer that
    /// ships Canary-1B-Flash output must render this string. `None` means the
    /// artifact carries no stamp (e.g. it was converted with an explicit
    /// `--license` override, which suppresses the CC-BY wording).
    #[inline]
    #[must_use]
    pub fn attribution(&self) -> Option<&str> {
        self.attribution.as_deref()
    }

    /// Transcribes a mono `f32` PCM slice at [`Self::sample_rate`]
    /// (ASR mode — equivalent to
    /// `transcribe_with_task(pcm, Canary1bFlashTask::Asr, false)`).
    ///
    /// # Errors
    ///
    /// See [`Self::transcribe_with_task`].
    pub fn transcribe(&self, pcm: &[f32]) -> Result<Vec<u32>> {
        self.transcribe_with_task(pcm, Canary1bFlashTask::Asr, false)
    }

    /// Runs the multitask forward: ASR or AST (speech translation), optionally
    /// with segment timestamps (the `<timestamp>` task token).
    ///
    /// # Loud-partial (this WP)
    ///
    /// Returns [`VokraError::UnsupportedOp`] naming the four blockers listed
    /// in the module docstring — the missing `.nemo` tensor-name manifest, the
    /// missing SentencePiece tokenizer, the resulting `0`-sentinel head width,
    /// and the unwired AED decoder step. The message names the primitives that
    /// *do* exist (`vokra_ops::waveform_frontend`, `vokra_ops::conformer`,
    /// `vokra_core::decode::beam_search`) so the follow-up wave knows exactly
    /// what is left to write. **No fabricated token ids are ever emitted**
    /// (FR-EX-08 — CLAUDE.md 教訓 (a)).
    ///
    /// # Errors
    ///
    /// - [`VokraError::InvalidArgument`] when `pcm` is empty.
    /// - [`VokraError::UnsupportedOp`] — the loud-partial gate.
    pub fn transcribe_with_task(
        &self,
        pcm: &[f32],
        task: Canary1bFlashTask,
        timestamps: bool,
    ) -> Result<Vec<u32>> {
        if pcm.is_empty() {
            return Err(VokraError::InvalidArgument(
                "canary-1b-flash transcribe: pcm slice is empty".to_owned(),
            ));
        }
        Err(transcribe_loud_partial(
            &self.cfg,
            &self.weights,
            task,
            timestamps,
        ))
    }
}

/// Strict `vokra.model.arch` verification shared by every entry point.
///
/// Canonical STRICT posture (the `emotion2vec` precedent): a missing tag and a
/// foreign tag get **different** messages, and the foreign-tag message names
/// both the found and the expected value plus the sibling neighbourhood, so a
/// mis-routed GGUF is diagnosable from the error alone.
fn verify_arch(file: &GgufFile) -> Result<()> {
    match file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()) {
        Some(a) if a == ARCH => Ok(()),
        Some(other) => Err(VokraError::ModelLoad(format!(
            "canary-1b-flash: GGUF arch is `{other}`, expected `{ARCH}` (was this GGUF \
             produced by `vokra-cli convert --model canary-1b-flash`?). The Canary \
             neighbourhood shares an encoder but NOT a decoder manifest: `canary` \
             (Canary-1B-v2) carries an **8-layer** Transformer AED decoder, \
             `canary-qwen` carries a **Qwen LLM** decoder consuming the encoder-out as \
             a soft-prompt prefix, `parakeet-ctc` / `parakeet-tdt` carry a CTC / RNN-T \
             head with no decoder stack at all, and `whisper` / `voxtral` / \
             `kyutai-stt` are unrelated topologies. Canary-1B-Flash's decoder is \
             **{DECODER_N_LAYER} layers** — binding a 4-layer manifest against an \
             8-layer expectation does not crash, it silently mis-reads, so the arch \
             tags stay distinct (FR-EX-08 — no silent misroute). Primary source: \
             {PRIMARY_SOURCE_HF}"
        ))),
        None => Err(VokraError::ModelLoad(format!(
            "canary-1b-flash: GGUF is missing `vokra.model.arch` — this is not a \
             Vokra-native canary-1b-flash GGUF (was it produced by `vokra-cli convert \
             --model canary-1b-flash`?). Primary source: {PRIMARY_SOURCE_HF}"
        ))),
    }
}

/// Builds the loud-partial [`VokraError::UnsupportedOp`] returned by the
/// transcribe entry points.
///
/// Kept as a free function so the message has exactly one definition and the
/// tests can assert against the same text every caller sees.
fn transcribe_loud_partial(
    cfg: &Canary1bFlashConfig,
    weights: &Canary1bFlashWeights,
    task: Canary1bFlashTask,
    timestamps: bool,
) -> VokraError {
    VokraError::UnsupportedOp(format!(
        "canary-1b-flash transcribe (loud-partial, task={task}, timestamps={timestamps}): \
         the full ASR / AST forward is deferred. Four blockers must land before real \
         token ids can be emitted: \
         (1) NO TENSOR-NAME MANIFEST — the converter copies every float tensor under \
         its verbatim upstream safetensors name and nothing in-repo transcribes NeMo's \
         `EncDecMultiTaskModel` state_dict naming, so walking guessed names into typed \
         encoder / decoder slots would bind shape-valid garbage ({count} tensors are \
         present on disk and can be inspected via \
         `Canary1bFlashWeights::tensor_names`); \
         (2) NO TOKENIZER — the unified Canary SentencePiece model, its vocabulary \
         width and the concrete pad / bos / eos / `<taskname>` ids live inside the \
         `.nemo` tarball and are not stated on the model card; \
         (3) HEAD WIDTH IS A PLACEHOLDER — head.vocab_size={vocab} (a `0` sentinel is \
         the direct consequence of (2); it is deliberately NOT copied from \
         Canary-1B-v2's 25-language 16384, which is a different tokenizer), so no \
         logits array can even be shaped; \
         (4) THE AED DECODER STEP IS NOT WIRED — the {dec_layers}-layer pre-norm \
         self-attn + cross-attn + FFN loop driven by a task-token prompt prefix has no \
         implementation (a gap shared with `crate::canary`, not specific to Flash). \
         The surrounding primitives DO exist and are what the follow-up wave composes: \
         `vokra_ops::waveform_frontend` (128-bin log-mel front-end), \
         `vokra_ops::conformer` with ConvSubsampleKind::Stacking {{ factor: {sub} }} \
         (the {enc_layers}-layer FastConformer encoder, shared with Canary-1B-v2 / \
         Parakeet), and `vokra_core::decode::beam_search` (the search, shared with \
         Whisper / Voxtral). Config source: {source:?} (the current converter stamps \
         no `vokra.canary_1b_flash.*` axes, so the axes above are anchored to the \
         published family reference). Bind a real Canary-1B-Flash checkpoint — \
         `{upstream}` (CC-BY 4.0), distributed as a `.nemo` tarball, prepared with \
         `{prep}` — then re-run `vokra-cli convert --model canary-1b-flash`. Primary \
         sources: model card {hf}; family reference {yaml}. Loud-partial per CLAUDE.md \
         教訓 (a) — no fabricated token ids are ever emitted (FR-EX-08, no silent \
         partial output).",
        task = task.as_str(),
        count = weights.tensor_count(),
        vocab = cfg.head.vocab_size,
        dec_layers = cfg.decoder.n_layer,
        enc_layers = cfg.encoder.n_layer,
        sub = cfg.encoder.subsampling_factor,
        source = cfg.source,
        upstream = UPSTREAM_HF,
        prep = PRIMARY_SOURCE_NEMO_PREP,
        hf = PRIMARY_SOURCE_HF,
        yaml = PRIMARY_SOURCE_FAMILY_YAML,
    ))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    //! Tests for the Canary-1B-Flash runtime binder.
    //!
    //! # What is honestly testable here
    //!
    //! On a real 16 kHz waveform the round-trip would be `transcribe(...)`
    //! returning token ids, but the `.nemo` tensor manifest + SentencePiece
    //! tokenizer are owner-gated (see the module doc). Fabricating a transcript
    //! would violate CLAUDE.md 教訓 (a). What IS testable, and is tested:
    //!
    //! 1. **Contract-constant pins** — `ARCH` / `NAME` / `CATEGORY` /
    //!    `UPSTREAM_HF` / `DEFAULT_LICENSE` match the converter by value, and
    //!    the arch tag is distinct from every Canary sibling.
    //! 2. **Primary-source axis pins** — 32 encoder layers, **4** decoder
    //!    layers, 1024 widths, 16 kHz, and the `0` head sentinels that the
    //!    validator refuses.
    //! 3. **Metadata round-trip** — a synthetic GGUF built exactly the way the
    //!    converter builds one binds, and its licence / attribution surface.
    //! 4. **Loud negative space** — missing arch, foreign arch, empty
    //!    manifest, missing tensor, wrong dims, malformed override: each fires
    //!    at its documented surface in its documented variant.
    //! 5. **Loud-partial contract** — `transcribe` names every blocker and
    //!    every primitive the follow-up wave must compose.

    use super::*;
    use vokra_core::gguf::{GgmlType, GgufBuilder};

    /// Builds a GGUF the way `convert_canary_1b_flash_file` builds one: the
    /// `vokra.model.*` group, the provenance group, and float tensors carrying
    /// verbatim upstream-style names. No `vokra.canary_1b_flash.*` axis chunk
    /// — mirroring the real converter, which stamps none.
    fn flash_gguf(weight_license_class: Option<LicenseClass>, attribution: bool) -> GgufFile {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, NAME);
        b.add_string("vokra.model.category", CATEGORY);
        b.add_string("vokra.provenance.upstream_hf", UPSTREAM_HF);
        b.add_string(chunks::KEY_PROVENANCE_LICENSE, DEFAULT_LICENSE);
        if let Some(cls) = weight_license_class {
            b.add_string(chunks::KEY_PROVENANCE_WEIGHT_LICENSE, cls.as_str());
        }
        if attribution {
            b.add_string(
                chunks::KEY_PROVENANCE_ATTRIBUTION,
                "This application uses NVIDIA Canary-1B-Flash. Model weights are \
                 licensed under CC-BY 4.0. Copyright (c) NVIDIA. Source: \
                 https://huggingface.co/nvidia/canary-1b-flash",
            );
        }
        // Two representative float tensors under the same verbatim
        // upstream-style names the converter's own fixture uses, so the two
        // test suites describe the same artifact.
        b.add_tensor(
            "encoder.blocks.0.attn.qkv_proj.weight",
            GgmlType::F32,
            vec![2, 3],
            vec![0u8; 2 * 3 * 4],
        )
        .expect("add encoder tensor");
        b.add_tensor(
            "decoder.blocks.0.self_attn.qkv.weight",
            GgmlType::F32,
            vec![1, 4],
            vec![0u8; 4 * 4],
        )
        .expect("add decoder tensor");
        GgufFile::parse(b.to_bytes().expect("serialize")).expect("parse")
    }

    // -----------------------------------------------------------------------
    // 1 — contract-constant pins + sibling distinctness
    // -----------------------------------------------------------------------

    #[test]
    fn contract_constants_mirror_the_converter() {
        assert_eq!(ARCH, "canary-1b-flash", "arch tag pin");
        assert_eq!(NAME, "canary-1b-flash", "model name pin");
        assert_eq!(CATEGORY, "asr", "category tier pin");
        assert_eq!(UPSTREAM_HF, "nvidia/canary-1b-flash", "upstream slug pin");
        assert_eq!(DEFAULT_LICENSE, "cc-by-4.0", "default weight SPDX pin");
        assert_eq!(CANARY_1B_FLASH_SAMPLE_RATE, 16_000, "model card: 16 kHz");
        assert_eq!(SUPPORTED_LANGUAGES, ["en", "de", "fr", "es"]);
        assert_eq!(TASK_TOKENS.len(), 8, "eight task-token families");
        assert!(
            TASK_TOKENS.contains(&"<timestamp>"),
            "timestamps are a task"
        );
    }

    #[test]
    fn arch_is_distinct_from_every_canary_sibling() {
        // The whole point of a separate arch tag: the decoder manifests
        // differ, so a shared tag would silently misroute.
        assert_ne!(ARCH, crate::canary::EXPECTED_ARCH);
        assert_ne!(ARCH, crate::canary_qwen::EXPECTED_ARCH);
        assert_eq!(crate::canary::EXPECTED_ARCH, "canary");
        assert_eq!(crate::canary_qwen::EXPECTED_ARCH, "canary-qwen");
    }

    // -----------------------------------------------------------------------
    // 2 — primary-source axis pins
    // -----------------------------------------------------------------------

    #[test]
    fn config_matches_primary_sources() {
        let c = Canary1bFlashConfig::canary_1b_flash();
        // Model card.
        assert_eq!(c.encoder.n_layer, 32, "model card: 32 FastConformer layers");
        assert_eq!(c.decoder.n_layer, 4, "model card: 4 decoder layers (Flash)");
        assert_eq!(c.sample_rate, 16_000);
        // Family YAML variant table (records canary-1b-flash by name).
        assert_eq!(c.encoder.d_model, 1024, "asr_enc_hidden = 1024");
        assert_eq!(c.decoder.d_model, 1024, "lm_dec_hidden = 1024");
        assert_eq!(c.decoder.max_sequence_length, 1024, "flash row = 1024");
        // Family defaults.
        assert_eq!(c.encoder.n_head, 8);
        assert_eq!(c.encoder.n_head_kv, 8, "MHA, no GQA broadcast");
        assert_eq!(c.encoder.ffn_dim, 4096, "4 x d_model");
        assert_eq!(c.encoder.conv_kernel_size, 9);
        assert_eq!(c.encoder.in_dim, 128, "preprocessor.features = 128");
        assert_eq!(c.encoder.subsampling_factor, 8);
        assert!(c.encoder.attention_bias, "untie_biases = true");
        assert!(!c.encoder.scale_input, "xscaling = false");
        assert!(c.decoder.pre_ln);
        assert_eq!(c.decoder.hidden_act, "relu");
        assert_eq!(c.encoder.head_dim(), 128);
        // Provenance marker.
        assert!(c.is_family_anchored());
    }

    #[test]
    fn decoder_depth_differs_from_canary_1b_v2() {
        // The Flash distillation IS this axis. A regression that silently
        // aligned the two would defeat the reason the arch tags are distinct.
        let flash = Canary1bFlashConfig::canary_1b_flash();
        let v2 = crate::canary::CanaryConfig::canary_1b_v2();
        assert_eq!(flash.decoder.n_layer, DECODER_N_LAYER);
        assert_eq!(v2.decoder.n_layer, 8, "Canary-1B-v2 decoder depth");
        assert_ne!(flash.decoder.n_layer, v2.decoder.n_layer);
        // ... while the encoder is deliberately identical.
        assert_eq!(flash.encoder.n_layer, v2.encoder.n_layer);
        assert_eq!(flash.encoder, v2.encoder, "shared FastConformer encoder");
    }

    #[test]
    fn primary_source_config_refuses_to_run_on_placeholder_head() {
        // vocab_size / pad / bos / eos are `0` sentinels because NO primary
        // source states them. The validator must refuse — this is the
        // mechanism that stops a hallucinated forward (FR-EX-08).
        let c = Canary1bFlashConfig::canary_1b_flash();
        assert_eq!(c.head.vocab_size, 0, "no primary source states the width");
        let Err(err) = c.validate_for_forward() else {
            panic!("placeholder head must be refused, not accepted");
        };
        match err {
            VokraError::InvalidArgument(m) => {
                assert!(m.contains("vocab_size"), "must name the field: {m}");
                assert!(
                    m.contains("canary-1b-flash config"),
                    "must be attributed to this model: {m}"
                );
            }
            other => panic!("expected InvalidArgument, got {other:?}"),
        }
    }

    #[test]
    fn tiny_config_is_well_formed_and_keeps_the_flash_decoder_depth() {
        let c = Canary1bFlashConfig::tiny_for_tests();
        assert_eq!(c.decoder.n_layer, DECODER_N_LAYER);
        c.validate_for_forward()
            .expect("tiny config must validate (real head axes)");
    }

    #[test]
    fn validator_delegation_rejects_an_ill_formed_encoder() {
        let mut c = Canary1bFlashConfig::tiny_for_tests();
        c.encoder.n_head = 3; // 16 % 3 != 0
        let Err(err) = c.validate_for_forward() else {
            panic!("ill-formed head split must be refused");
        };
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    // -----------------------------------------------------------------------
    // 3 — metadata round-trip
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_binds_a_converter_shaped_artifact() {
        let file = flash_gguf(Some(LicenseClass::AttributionRequired), true);
        let asr = Canary1bFlashAsr::from_gguf(&file).expect("valid GGUF must bind");

        assert_eq!(
            asr.weight_license(),
            LicenseClass::AttributionRequired,
            "cc-by-4.0 must surface as AttributionRequired"
        );
        assert_eq!(asr.tensor_count(), 2, "both fixture tensors bound");
        assert_eq!(asr.sample_rate(), 16_000);
        // The converter stamps no axis chunks, so the axes are family-anchored.
        assert!(
            asr.config().is_family_anchored(),
            "no vokra.canary_1b_flash.* chunk => FamilyAnchored"
        );
        assert_eq!(asr.config().decoder.n_layer, DECODER_N_LAYER);
        // FR-MD-09 attribution surface (CC-BY 4.0 obligation).
        let attr = asr.attribution().expect("attribution stamp must surface");
        assert!(
            attr.contains("NVIDIA") && attr.contains("CC-BY 4.0"),
            "attribution must name NVIDIA + CC-BY 4.0: {attr}"
        );
        // Manifest lookups over the real (verbatim upstream) names.
        assert_eq!(
            asr.weights()
                .require_tensor("encoder.blocks.0.attn.qkv_proj.weight")
                .expect("present tensor"),
            &[2, 3]
        );
        assert_eq!(asr.weights().count_with_prefix("decoder."), 1);
        assert_eq!(asr.weights().tensor_names().len(), 2);
    }

    #[test]
    fn missing_license_stamp_fails_closed_to_unknown() {
        let file = flash_gguf(None, false);
        let asr = Canary1bFlashAsr::from_gguf(&file).expect("arch + tensors are the bind gates");
        assert_eq!(
            asr.weight_license(),
            LicenseClass::Unknown,
            "absent stamp must fail-closed"
        );
        assert!(asr.attribution().is_none(), "no stamp => no attribution");
    }

    #[test]
    fn from_gguf_with_policy_accepts_cc_by_under_strict() {
        // CC-BY 4.0 is commercially permitted, so the M2-13 gate passes under
        // the fail-closed strict policy without a research opt-in.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, NAME);
        b.add_string(chunks::KEY_PROVENANCE_LICENSE, DEFAULT_LICENSE);
        b.add_string(
            chunks::KEY_PROVENANCE_WEIGHT_LICENSE,
            LicenseClass::AttributionRequired.as_str(),
        );
        b.add_tensor("encoder.probe", GgmlType::F32, vec![2, 2], vec![0u8; 16])
            .expect("add_tensor");
        let bytes = b.to_bytes().expect("serialize");
        let asr = Canary1bFlashAsr::from_gguf_with_policy(&bytes, &CompliancePolicy::strict())
            .expect("CC-BY 4.0 must pass the strict gate");
        assert_eq!(asr.weight_license(), LicenseClass::AttributionRequired);
    }

    // -----------------------------------------------------------------------
    // 4 — loud negative space
    // -----------------------------------------------------------------------

    #[test]
    fn from_gguf_rejects_missing_arch() {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_NAME, "something-else");
        b.add_tensor("some.tensor", GgmlType::F32, vec![2, 2], vec![0u8; 16])
            .expect("add_tensor");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let Err(err) = Canary1bFlashAsr::from_gguf(&file) else {
            panic!("expected ModelLoad on missing arch");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("`vokra.model.arch`"),
                    "must name the missing key: {m}"
                );
                assert!(
                    m.contains("not a Vokra-native canary-1b-flash GGUF"),
                    "must name the surface: {m}"
                );
            }
            other => panic!("expected ModelLoad, got {other:?}"),
        }
    }

    #[test]
    fn from_gguf_rejects_a_base_canary_artifact() {
        // The most dangerous confusion in the neighbourhood: Canary-1B-v2 has
        // the SAME encoder and a DIFFERENT (8-layer) decoder, so a silent bind
        // would mis-read rather than crash.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, crate::canary::EXPECTED_ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, "canary-1b-v2");
        b.add_tensor("encoder.probe", GgmlType::F32, vec![4, 4], vec![0u8; 64])
            .expect("add_tensor");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let Err(err) = Canary1bFlashAsr::from_gguf(&file) else {
            panic!("expected ModelLoad on foreign arch");
        };
        match err {
            VokraError::ModelLoad(m) => {
                // BOTH tags named.
                assert!(m.contains("`canary`"), "must name the found arch: {m}");
                assert!(
                    m.contains("`canary-1b-flash`"),
                    "must name the expected arch: {m}"
                );
                // Sibling neighbourhood enumerated.
                for sibling in ["canary-qwen", "parakeet-ctc", "whisper", "voxtral"] {
                    assert!(m.contains(sibling), "must name sibling {sibling}: {m}");
                }
                assert!(m.contains("FR-EX-08"), "must cite FR-EX-08: {m}");
            }
            other => panic!("expected ModelLoad, got {other:?}"),
        }
    }

    #[test]
    fn from_gguf_rejects_an_empty_tensor_manifest() {
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, NAME);
        // No tensors.
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let Err(err) = Canary1bFlashAsr::from_gguf(&file) else {
            panic!("expected ModelLoad on empty manifest");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(m.contains("zero tensors"), "must name the gap: {m}");
                assert!(m.contains("FR-EX-08"), "must cite FR-EX-08: {m}");
                assert!(
                    m.contains("vokra-cli convert --model canary-1b-flash"),
                    "must include the repro command: {m}"
                );
            }
            other => panic!("expected ModelLoad, got {other:?}"),
        }
    }

    #[test]
    fn require_tensor_names_the_missing_tensor() {
        let file = flash_gguf(Some(LicenseClass::AttributionRequired), false);
        let asr = Canary1bFlashAsr::from_gguf(&file).expect("bind");
        let Err(err) = asr
            .weights()
            .require_tensor("encoder.blocks.31.ff2_fc2.weight")
        else {
            panic!("absent tensor must fail loud");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains("encoder.blocks.31.ff2_fc2.weight"),
                    "must name the missing tensor: {m}"
                );
                assert!(
                    m.contains("encoder.blocks.0.attn.qkv_proj.weight"),
                    "must list nearest on-disk names: {m}"
                );
                assert!(m.contains("FR-EX-08"), "must cite FR-EX-08: {m}");
            }
            other => panic!("expected ModelLoad, got {other:?}"),
        }
    }

    #[test]
    fn require_tensor_dims_names_expected_and_actual() {
        let file = flash_gguf(Some(LicenseClass::AttributionRequired), false);
        let asr = Canary1bFlashAsr::from_gguf(&file).expect("bind");
        // Correct dims pass.
        asr.weights()
            .require_tensor_dims("decoder.blocks.0.self_attn.qkv.weight", &[1, 4])
            .expect("matching dims must pass");
        // Wrong dims fail loud, naming both sides.
        let Err(err) = asr
            .weights()
            .require_tensor_dims("decoder.blocks.0.self_attn.qkv.weight", &[8, 8])
        else {
            panic!("dim mismatch must fail loud");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(m.contains("[1, 4]"), "must name the actual dims: {m}");
                assert!(m.contains("[8, 8]"), "must name the expected dims: {m}");
                assert!(
                    m.contains("decoder.blocks.0.self_attn.qkv.weight"),
                    "must name the tensor: {m}"
                );
            }
            other => panic!("expected ModelLoad, got {other:?}"),
        }
    }

    #[test]
    fn present_axis_override_is_honoured() {
        // Forward-compatibility: the current converter stamps nothing, but a
        // future one that stamps real .nemo-extracted axes must be honoured
        // without a runtime change.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, NAME);
        b.add_u32(GGUF_KEY_DEC_N_LAYER, 6);
        b.add_u32(GGUF_KEY_HEAD_VOCAB_SIZE, 16_384);
        b.add_tensor("encoder.probe", GgmlType::F32, vec![2, 2], vec![0u8; 16])
            .expect("add_tensor");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let asr = Canary1bFlashAsr::from_gguf(&file).expect("bind");
        assert_eq!(
            asr.config().decoder.n_layer,
            6,
            "stamp overrides the anchor"
        );
        assert_eq!(asr.config().head.vocab_size, 16_384);
        assert_eq!(
            asr.config().source,
            Canary1bFlashConfigSource::GgufStamped,
            "a present stamp must be reported as such"
        );
        assert!(!asr.config().is_family_anchored());
        // Untouched axes still come from the family anchor.
        assert_eq!(asr.config().encoder.n_layer, ENCODER_N_LAYER);
    }

    #[test]
    fn malformed_axis_override_fails_loud() {
        // A key that is present but of the wrong dtype must NOT be silently
        // ignored — that would run the family default under a false claim.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, NAME);
        b.add_string(GGUF_KEY_DEC_N_LAYER, "four");
        b.add_tensor("encoder.probe", GgmlType::F32, vec![2, 2], vec![0u8; 16])
            .expect("add_tensor");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let Err(err) = Canary1bFlashAsr::from_gguf(&file) else {
            panic!("malformed override must fail loud");
        };
        match err {
            VokraError::ModelLoad(m) => {
                assert!(
                    m.contains(GGUF_KEY_DEC_N_LAYER),
                    "must name the offending key: {m}"
                );
                assert!(m.contains("FR-EX-08"), "must cite FR-EX-08: {m}");
            }
            other => panic!("expected ModelLoad, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------
    // 5 — loud-partial contract
    // -----------------------------------------------------------------------

    #[test]
    fn transcribe_rejects_empty_pcm_before_the_loud_partial() {
        let file = flash_gguf(Some(LicenseClass::AttributionRequired), false);
        let asr = Canary1bFlashAsr::from_gguf(&file).expect("bind");
        let Err(err) = asr.transcribe(&[]) else {
            panic!("empty pcm must be rejected");
        };
        match err {
            VokraError::InvalidArgument(m) => {
                assert!(m.contains("pcm slice is empty"), "message: {m}");
            }
            other => panic!("expected InvalidArgument, got {other:?}"),
        }
    }

    #[test]
    fn transcribe_loud_partials_naming_every_blocker() {
        let file = flash_gguf(Some(LicenseClass::AttributionRequired), false);
        let asr = Canary1bFlashAsr::from_gguf(&file).expect("bind");
        let pcm = vec![0.0_f32; 16_000]; // 1 s of silence at 16 kHz mono.
        let Err(err) = asr.transcribe(&pcm) else {
            panic!("transcribe must loud-partial (no tokenizer, no manifest)");
        };
        match err {
            VokraError::UnsupportedOp(m) => {
                assert!(m.contains("canary-1b-flash transcribe"), "surface: {m}");
                assert!(m.contains("loud-partial"), "posture label: {m}");
                // The four blockers.
                assert!(m.contains("NO TENSOR-NAME MANIFEST"), "blocker 1: {m}");
                assert!(m.contains("NO TOKENIZER"), "blocker 2: {m}");
                assert!(m.contains("head.vocab_size=0"), "blocker 3: {m}");
                assert!(
                    m.contains("AED DECODER STEP IS NOT WIRED"),
                    "blocker 4: {m}"
                );
                assert!(m.contains(".nemo"), "must name the checkpoint format: {m}");
                assert!(m.contains("SentencePiece"), "must name the tokenizer: {m}");
                // The primitives the follow-up wave composes — all of which
                // genuinely exist today.
                for primitive in [
                    "vokra_ops::waveform_frontend",
                    "vokra_ops::conformer",
                    "vokra_core::decode::beam_search",
                ] {
                    assert!(m.contains(primitive), "must name {primitive}: {m}");
                }
                // Primary sources + honesty clause.
                assert!(m.contains(PRIMARY_SOURCE_HF), "must cite the card: {m}");
                assert!(
                    m.contains(PRIMARY_SOURCE_NEMO_PREP),
                    "must cite the prep script: {m}"
                );
                assert!(m.contains("FR-EX-08"), "must cite FR-EX-08: {m}");
            }
            other => panic!("expected UnsupportedOp, got {other:?}"),
        }
    }

    #[test]
    fn transcribe_with_task_reports_the_requested_mode() {
        let file = flash_gguf(Some(LicenseClass::AttributionRequired), false);
        let asr = Canary1bFlashAsr::from_gguf(&file).expect("bind");
        let pcm = vec![0.0_f32; 1_600];

        let Err(VokraError::UnsupportedOp(ast)) =
            asr.transcribe_with_task(&pcm, Canary1bFlashTask::Ast, true)
        else {
            panic!("AST must loud-partial too");
        };
        assert!(ast.contains("task=ast"), "must report the task: {ast}");
        assert!(
            ast.contains("timestamps=true"),
            "must report the timestamp request: {ast}"
        );

        let Err(VokraError::UnsupportedOp(asr_msg)) =
            asr.transcribe_with_task(&pcm, Canary1bFlashTask::Asr, false)
        else {
            panic!("ASR must loud-partial too");
        };
        assert!(
            asr_msg.contains("task=asr"),
            "must report the task: {asr_msg}"
        );
        assert_eq!(Canary1bFlashTask::Asr.as_str(), "asr");
        assert_eq!(Canary1bFlashTask::Ast.as_str(), "ast");
    }
}
