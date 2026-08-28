//! NVIDIA Canary-1B-Flash native multilingual ASR / AST.
//!
//! The runtime binds the exact released `.nemo` inference contract: 1,374
//! float tensors, a 32-layer FastConformer encoder, a four-layer pre-LayerNorm
//! Transformer AED decoder, the 5,248-entry aggregate SentencePiece
//! vocabulary, and the nine-token `canary2` prompt. The historical public
//! `vokra/canary-1b-flash` GGUF contains only the 1,292 encoder tensors and is
//! rejected explicitly; no decoder or tokenizer is synthesized.
//!
//! Learned matrix, normalization, attention, activation, and grouped-conv
//! work is routed through [`crate::compute::Compute`]. CPU is the default;
//! Metal is selected explicitly and must cover the complete declared hot-op
//! set before inference begins. Other backends fail without CPU fallback.
//!
//! The upstream weights are CC-BY-4.0. Runtime provenance and attribution are
//! surfaced from GGUF metadata, while publication remains a separate gated
//! operation. No ONNX or third-party runtime dependency is used.

mod tokenizer;

use std::sync::Arc;

use vokra_core::gguf::{GgufFile, GgufMetadataValue, chunks};
use vokra_core::{
    AsrEngine, BackendKind, CompliancePolicy, LicenseClass, Result, Transcription, VokraError,
    check_weight_license,
};

use crate::compute::{Compute, HotOp};

use crate::canary_aed_bound::{CanaryAedReleaseSpec, CanaryBoundWeights};

pub use tokenizer::{
    BOS_ID, Canary1bFlashOptions, CanaryEmotion, CanaryLanguage, CanaryTokenizer, EOS_ID,
    KEY_TOKENIZER_VOCAB, KEY_TOKENIZER_VOCAB_SHA256, PAD_ID, SPECIAL_VOCAB_SIZE,
    TOKENIZER_VOCAB_SHA256, VOCAB_SIZE,
};

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
/// `vokra.provenance.upstream_hf` by the converter.
pub const UPSTREAM_HF: &str = "nvidia/canary-1b-flash";

/// Canonical weight-license SPDX the converter stamps by default
/// (`cc-by-4.0` → [`LicenseClass::AttributionRequired`]). A caller-supplied
/// `--license` override on the converter side replaces it; this binder reads
/// whatever the artifact actually carries and never assumes.
pub const DEFAULT_LICENSE: &str = "cc-by-4.0";

/// PCM sample rate Canary-1B-Flash expects — **16 000 Hz** mono
/// (model card: "16kHz Audio, .wav and .flac audio formats, Monochannel").
pub const CANARY_1B_FLASH_SAMPLE_RATE: u32 = 16_000;

/// Complete backend-dispatched learned-op set used by the native encoder and
/// decoder. `Compute::for_backend` checks this set as one unit, so selecting a
/// backend can never execute an uncovered operation on the CPU silently.
pub const CANARY_1B_FLASH_HOT_OPS: &[HotOp] = &[
    HotOp::Gemm,
    HotOp::Gemv,
    HotOp::Softmax,
    HotOp::LayerNorm,
    HotOp::Relu,
    HotOp::GroupedConv1d,
];

const RELEASE_TENSOR_COUNT: usize = 1_374;
const RELEASE_MANIFEST_SHA256: [u8; 32] = [
    0xf7, 0x6f, 0x4c, 0x3d, 0x28, 0x14, 0x7b, 0x41, 0x87, 0x05, 0xc8, 0x27, 0x2a, 0x81, 0xda, 0xb5,
    0x34, 0x25, 0xe3, 0xbd, 0x26, 0x4b, 0x8a, 0x20, 0x40, 0xff, 0xb0, 0xde, 0x03, 0x38, 0x5c, 0xb6,
];
const RELEASE_SPEC: CanaryAedReleaseSpec = CanaryAedReleaseSpec::new(
    "Canary-1B-Flash",
    ARCH,
    NAME,
    RELEASE_TENSOR_COUNT,
    RELEASE_MANIFEST_SHA256,
    CANARY_1B_FLASH_SAMPLE_RATE,
    VOCAB_SIZE,
    EOS_ID,
);

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
/// Recorded as ISO 639-1 codes. Their concrete prompt spellings and ids are
/// authenticated separately by the aggregate vocabulary hash in
/// [`CanaryTokenizer`].
pub const SUPPORTED_LANGUAGES: [&str; 4] = ["en", "de", "fr", "es"];

/// The eight variable slots in the released `canary2` user prompt.
///
/// `decodercontext` is empty in the one-shot API. ASR versus AST is selected
/// by equal versus different source/target languages; Canary2 has no separate
/// task-name slot. The historical constant name is retained for Rust-source
/// compatibility even though these are prompt slots rather than eight literal
/// task tokens.
pub const TASK_TOKENS: [&str; 8] = [
    "<decodercontext>",
    "<emotion>",
    "<source_lang>",
    "<target_lang>",
    "<pnc>",
    "<itn>",
    "<timestamp>",
    "<diarize>",
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
pub const PRIMARY_SOURCE_NEMO_PREP: &str = "tools/parity/canary_1b_flash_prepare_checkpoint.py";

// ---------------------------------------------------------------------------
// Immutable-release `vokra.canary_1b_flash.*` axis keys.
//
// The authenticated converter stamps the complete group. The reader still
// accepts an absent group for the historical encoder-only artifact only far
// enough to emit its exact manifest error; any present conflicting value is
// refused by `validate_release_contract` before weights execute.
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

const GGUF_KEY_SOURCE_REVISION: &str = "vokra.canary_1b_flash.source_revision";
const GGUF_KEY_SOURCE_NEMO_SHA256: &str = "vokra.canary_1b_flash.source_nemo_sha256";
const GGUF_KEY_MODEL_CONFIG_SHA256: &str = "vokra.canary_1b_flash.model_config_sha256";
const GGUF_KEY_DATA_PICKLE_SHA256: &str = "vokra.canary_1b_flash.data_pickle_sha256";
const GGUF_KEY_TENSOR_MANIFEST_SHA256: &str = "vokra.canary_1b_flash.tensor_manifest_sha256";
const GGUF_KEY_FRONTEND_N_FFT: &str = "vokra.canary_1b_flash.frontend.n_fft";
const GGUF_KEY_FRONTEND_HOP: &str = "vokra.canary_1b_flash.frontend.hop_length";
const GGUF_KEY_FRONTEND_WIN: &str = "vokra.canary_1b_flash.frontend.win_length";
const GGUF_KEY_FRONTEND_N_MELS: &str = "vokra.canary_1b_flash.frontend.n_mels";
const GGUF_KEY_FRONTEND_PREEMPHASIS: &str = "vokra.canary_1b_flash.frontend.preemphasis";
const GGUF_KEY_FRONTEND_WINDOW: &str = "vokra.canary_1b_flash.frontend.window";
const GGUF_KEY_FRONTEND_WINDOW_PERIODIC: &str = "vokra.canary_1b_flash.frontend.window_periodic";
const GGUF_KEY_FRONTEND_NORMALIZE: &str = "vokra.canary_1b_flash.frontend.normalize";
const GGUF_KEY_FRONTEND_PAD_MODE: &str = "vokra.canary_1b_flash.frontend.pad_mode";

const SOURCE_REVISION: &str = "2b6e4d2dacb11cc1b1724de31bb48fe68c26c12e";
const SOURCE_NEMO_SHA256: &str = "3887cce1afdd425429cfc5109575a8f2cffeb07c02c503a9faff7612bd74e324";
const MODEL_CONFIG_SHA256: &str =
    "42d71aebc1f4b9f387a20902db71e00128b324ff5156bdac63897e1afad55ff9";
const DATA_PICKLE_SHA256: &str = "a60784f60aa5cea26d3c11d62c3ed7270e5c7bf52844d99b553656d9498a3617";
const TENSOR_MANIFEST_SHA256: &str =
    "f76f4c3d28147b418705c8272a81dab53425e3bd264b8a2040ffb0de03385cb6";

const RELEASE_STRING_METADATA: &[(&str, &str)] = &[
    ("vokra.model.category", CATEGORY),
    ("vokra.provenance.upstream_hf", UPSTREAM_HF),
    (GGUF_KEY_SOURCE_REVISION, SOURCE_REVISION),
    (GGUF_KEY_SOURCE_NEMO_SHA256, SOURCE_NEMO_SHA256),
    (GGUF_KEY_MODEL_CONFIG_SHA256, MODEL_CONFIG_SHA256),
    (GGUF_KEY_DATA_PICKLE_SHA256, DATA_PICKLE_SHA256),
    (GGUF_KEY_TENSOR_MANIFEST_SHA256, TENSOR_MANIFEST_SHA256),
    (GGUF_KEY_DEC_HIDDEN_ACT, "relu"),
    (GGUF_KEY_FRONTEND_WINDOW, "hann"),
    (GGUF_KEY_FRONTEND_NORMALIZE, "per_feature"),
    (GGUF_KEY_FRONTEND_PAD_MODE, "constant"),
];

const RELEASE_U32_METADATA: &[(&str, u32)] = &[
    (GGUF_KEY_SAMPLE_RATE, 16_000),
    (GGUF_KEY_ENC_N_LAYER, 32),
    (GGUF_KEY_ENC_D_MODEL, 1_024),
    (GGUF_KEY_ENC_N_HEAD, 8),
    (GGUF_KEY_ENC_N_HEAD_KV, 8),
    (GGUF_KEY_ENC_FFN_DIM, 4_096),
    (GGUF_KEY_ENC_CONV_KERNEL, 9),
    (GGUF_KEY_ENC_IN_DIM, 128),
    (GGUF_KEY_ENC_SUBSAMPLING_FACTOR, 8),
    (GGUF_KEY_ENC_SUB_CONV_KERNEL, 3),
    (GGUF_KEY_ENC_SUB_CONV_STRIDE, 2),
    (GGUF_KEY_ENC_SUB_CONV_CHANNELS, 256),
    (GGUF_KEY_ENC_MAX_POS, 5_000),
    (GGUF_KEY_ENC_ATTN_BIAS, 1),
    (GGUF_KEY_ENC_CONV_BIAS, 1),
    (GGUF_KEY_ENC_SCALE_INPUT, 0),
    (GGUF_KEY_DEC_N_LAYER, 4),
    (GGUF_KEY_DEC_D_MODEL, 1_024),
    (GGUF_KEY_DEC_N_HEAD, 8),
    (GGUF_KEY_DEC_FFN_DIM, 4_096),
    (GGUF_KEY_DEC_MAX_SEQ, 1_024),
    (GGUF_KEY_DEC_PRE_LN, 1),
    (GGUF_KEY_HEAD_VOCAB_SIZE, 5_248),
    (GGUF_KEY_HEAD_PAD_ID, 2),
    (GGUF_KEY_HEAD_BOS_ID, 4),
    (GGUF_KEY_HEAD_EOS_ID, 3),
    (GGUF_KEY_FRONTEND_N_FFT, 512),
    (GGUF_KEY_FRONTEND_HOP, 160),
    (GGUF_KEY_FRONTEND_WIN, 400),
    (GGUF_KEY_FRONTEND_N_MELS, 128),
    (GGUF_KEY_FRONTEND_WINDOW_PERIODIC, 0),
];

// ---------------------------------------------------------------------------
// Task surface
// ---------------------------------------------------------------------------

/// The multitask modes Canary-1B-Flash exposes (model card: "multi-task
/// ASR / AST").
///
/// The enum records **what the model can be asked to do**. The released
/// Canary2 formatter selects ASR when source and target language are equal and
/// AST when they differ; it does not carry a separate `<taskname>` token.
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
/// did we anchor it to the immutable released family reference?". Canonical
/// new conversions stamp the complete group.
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
    /// Vocabulary / special-token / head axes from the released aggregate
    /// tokenizer and loss configuration.
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
    ///   biased convolutions (`ConformerEncoder::use_bias` defaults true),
    ///   `xscaling = false`, `pre_ln = true`,
    ///   `hidden_act = "relu"`.
    /// - **Released `.nemo` tokenizer/config**: `head.vocab_size = 5248`,
    ///   `pad = 2`, `bos = 4`, `eos = 3`.
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
                convolution_bias: true,
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
                vocab_size: VOCAB_SIZE,
                pad_token_id: PAD_ID,
                bos_token_id: BOS_ID,
                eos_token_id: EOS_ID,
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

    /// Rejects incomplete or ill-formed configs before any forward runs.
    ///
    /// **Delegates** to the shared Canary-family validator
    /// (`crate::canary::CanaryConfig::validate_for_forward`) — the Flash
    /// variant has the same shape algebra, so duplicating it here would only
    /// create drift. The delegated message is re-prefixed so a reader sees
    /// which model surfaced the failure.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] naming the offending field.
    pub fn validate_for_forward(&self) -> Result<()> {
        let family = self.to_family_config();
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

    fn to_family_config(&self) -> crate::canary::CanaryConfig {
        crate::canary::CanaryConfig {
            encoder: self.encoder.clone(),
            decoder: self.decoder.clone(),
            head: self.head.clone(),
            sample_rate: self.sample_rate,
        }
    }

    /// Ensures metadata cannot change a non-shape inference axis while still
    /// binding the same immutable released tensor manifest.
    pub(crate) fn validate_release_contract(&self) -> Result<()> {
        let expected = Self::canary_1b_flash();
        if self.encoder != expected.encoder
            || self.decoder != expected.decoder
            || self.head != expected.head
            || self.sample_rate != expected.sample_rate
        {
            return Err(VokraError::ModelLoad(format!(
                "canary-1b-flash: resolved runtime metadata does not match the pinned released `.nemo` topology (expected encoder={:?}, decoder={:?}, head={:?}, sample_rate={}; found encoder={:?}, decoder={:?}, head={:?}, sample_rate={}). Refusing to run an immutable 1,374-tensor checkpoint under conflicting axes (FR-EX-08)",
                expected.encoder,
                expected.decoder,
                expected.head,
                expected.sample_rate,
                self.encoder,
                self.decoder,
                self.head,
                self.sample_rate,
            )));
        }
        Ok(())
    }

    /// Resolves the axes for a Canary-1B-Flash GGUF.
    ///
    /// Starts from [`Self::canary_1b_flash`] (the primary-source anchor) and
    /// applies any `vokra.canary_1b_flash.*` override the artifact carries.
    /// Canonical new conversions stamp the full group. Absence remains
    /// representable for an authenticated legacy artifact, while any partial
    /// or conflicting group is rejected by the release contract:
    ///
    /// - an **absent** key is normal (the writer does not emit it yet);
    /// - a **present but wrong-dtype** key is a corrupted / hand-assembled
    ///   artifact and fails loud (FR-EX-08 — never silently ignored).
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
/// `None` when the key is absent. A present value that is not a `u32`-range unsigned integer is a
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
                 group is optional only for an authenticated legacy artifact, but a \
                 key that IS present must be well-formed; ignoring it would \
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
/// Instead it records what is actually on disk and offers diagnostic lookups;
/// executable binding is handled by the exact typed manifest in `bound`.
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
    /// scheme. The strict typed binder is the executable source of truth.
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
/// [`Self::transcribe`] / [`Self::transcribe_with_options`].
#[derive(Debug, Clone)]
pub struct Canary1bFlashAsr {
    cfg: Canary1bFlashConfig,
    runtime_cfg: crate::canary::CanaryConfig,
    weights: Canary1bFlashWeights,
    bound: Arc<CanaryBoundWeights>,
    tokenizer: CanaryTokenizer,
    backend: BackendKind,
    attribution: Option<String>,
}

impl Canary1bFlashAsr {
    /// Strictly binds the complete released checkpoint and aggregate
    /// tokenizer, selecting CPU by default.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        Self::from_gguf_with_backend(file, BackendKind::Cpu)
    }

    /// Strictly binds the complete released checkpoint and records an
    /// explicit backend choice. Backend availability and whole-model hot-op
    /// coverage are checked before the multi-GB tensor payload is decoded and
    /// rechecked at the inference boundary.
    pub fn from_gguf_with_backend(file: &GgufFile, backend: BackendKind) -> Result<Self> {
        verify_arch(file)?;
        let cfg = Canary1bFlashConfig::from_gguf(file)?;
        cfg.validate_for_forward()?;
        cfg.validate_release_contract()?;
        let runtime_cfg = cfg.to_family_config();

        // Fail an unavailable/uncovered backend before materializing the
        // 3.54 GB release into typed vectors. This is both FR-EX-08 and an
        // important Mac memory boundary: `--backend metal` on a CPU-only build
        // must not spend gigabytes loading weights before reporting that Metal
        // is unavailable. Inference constructs its own dispatcher again so a
        // device loss between bind and execution is still surfaced.
        let _backend_preflight = Compute::for_backend(backend, CANARY_1B_FLASH_HOT_OPS)?;

        // Authenticate the manifest before decoding any large tensor. This
        // makes the historical encoder-only public artifact fail cheaply and
        // explicitly at 1,292 != 1,374 tensors.
        CanaryBoundWeights::verify_manifest(file, RELEASE_SPEC)?;
        validate_runtime_metadata(file)?;
        let tokenizer = CanaryTokenizer::from_gguf(file)?;
        let weights = Canary1bFlashWeights::from_gguf(file)?;
        let bound = Arc::new(CanaryBoundWeights::from_gguf(
            file,
            &runtime_cfg,
            RELEASE_SPEC,
        )?);
        let attribution = file
            .get(chunks::KEY_PROVENANCE_ATTRIBUTION)
            .and_then(|v| v.as_str())
            .map(str::to_owned);

        Ok(Self {
            cfg,
            runtime_cfg,
            weights,
            bound,
            tokenizer,
            backend,
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

    /// Selects the backend used by subsequent encoder and decoder calls.
    #[must_use]
    pub fn with_backend(mut self, backend: BackendKind) -> Self {
        self.backend = backend;
        self
    }

    /// Returns the explicitly selected backend.
    #[must_use]
    pub const fn backend(&self) -> BackendKind {
        self.backend
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
        self.bound.tensor_count()
    }

    /// Authenticated upstream model name from the strict checkpoint binder.
    #[must_use]
    pub fn model_name(&self) -> &str {
        self.bound.model_name()
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
    pub fn weight_license(&self) -> LicenseClass {
        self.bound.weight_license()
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

    /// Transcribes 16 kHz mono PCM to Canary aggregate token IDs using the
    /// released English-ASR prompt defaults.
    pub fn transcribe(&self, pcm: &[f32]) -> Result<Vec<u32>> {
        self.transcribe_with_options(pcm, Canary1bFlashOptions::default())
    }

    /// Runs multilingual ASR or AST according to the exact `canary2` prompt
    /// fields in `options`.
    pub fn transcribe_with_options(
        &self,
        pcm: &[f32],
        options: Canary1bFlashOptions,
    ) -> Result<Vec<u32>> {
        if pcm.is_empty() {
            return Err(VokraError::InvalidArgument(
                "canary-1b-flash transcribe: pcm slice is empty".to_owned(),
            ));
        }
        let compute = Compute::for_backend(self.backend, CANARY_1B_FLASH_HOT_OPS)?;
        let (encoder, frames) = self.bound.encode_pcm(&compute, pcm)?;
        let prompt = options.prompt_tokens();
        self.bound.decode_tokens(
            &compute,
            &encoder,
            frames,
            &self.runtime_cfg,
            &prompt,
            options.max_new_tokens,
        )
    }

    /// Compatibility wrapper for the earlier task-only API. ASR uses the
    /// released English defaults. AST requires explicit source and target
    /// languages through [`Self::transcribe_with_options`] and is never
    /// guessed here.
    pub fn transcribe_with_task(
        &self,
        pcm: &[f32],
        task: Canary1bFlashTask,
        timestamps: bool,
    ) -> Result<Vec<u32>> {
        match task {
            Canary1bFlashTask::Asr => self.transcribe_with_options(
                pcm,
                Canary1bFlashOptions {
                    timestamps,
                    ..Canary1bFlashOptions::default()
                },
            ),
            Canary1bFlashTask::Ast => Err(VokraError::InvalidArgument(
                "canary-1b-flash AST requires explicit source_language and target_language; use transcribe_with_options instead of guessing a translation target"
                    .to_owned(),
            )),
        }
    }

    /// Runs the configured forward and decodes aggregate SentencePiece IDs to
    /// text. Special prompt/timestamp/diarization tokens are omitted from the
    /// text surface; callers that need them should use the token-ID method.
    pub fn transcribe_text_with_options(
        &self,
        pcm: &[f32],
        options: Canary1bFlashOptions,
    ) -> Result<String> {
        let tokens = self.transcribe_with_options(pcm, options)?;
        self.tokenizer.decode(&tokens)
    }
}

impl AsrEngine for Canary1bFlashAsr {
    fn transcribe(&self, pcm: &[f32]) -> Result<Transcription> {
        Ok(Transcription::new(self.transcribe_text_with_options(
            pcm,
            Canary1bFlashOptions::default(),
        )?))
    }

    fn backend(&self) -> BackendKind {
        self.backend
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

/// Authenticates every non-tensor inference condition stamped by the complete
/// converter. The shared frontend reproduces the released mel filter and Hann
/// window rather than reading their checkpoint buffers, so accepting a
/// conflicting FFT/window/normalization stamp would otherwise execute a
/// different graph while appearing self-describing.
fn validate_runtime_metadata(file: &GgufFile) -> Result<()> {
    for &(key, expected) in RELEASE_STRING_METADATA {
        require_release_string(file, key, expected)?;
    }

    for &(key, expected) in RELEASE_U32_METADATA {
        require_release_u32(file, key, expected)?;
    }

    match file.get(GGUF_KEY_FRONTEND_PREEMPHASIS) {
        Some(GgufMetadataValue::F32(value)) if value.to_bits() == 0.97f32.to_bits() => Ok(()),
        Some(other) => Err(VokraError::ModelLoad(format!(
            "canary-1b-flash: `{GGUF_KEY_FRONTEND_PREEMPHASIS}` must be f32 0.97, found {other:?}; refusing a frontend that differs from the authenticated release (FR-EX-08)"
        ))),
        None => Err(VokraError::ModelLoad(format!(
            "canary-1b-flash: missing required release metadata `{GGUF_KEY_FRONTEND_PREEMPHASIS}`; reconvert the complete pinned `.nemo` checkpoint"
        ))),
    }
}

fn require_release_string(file: &GgufFile, key: &str, expected: &str) -> Result<()> {
    match file.get(key) {
        Some(GgufMetadataValue::String(value)) if value == expected => Ok(()),
        Some(other) => Err(VokraError::ModelLoad(format!(
            "canary-1b-flash: `{key}` must be {expected:?}, found {other:?}; refusing metadata from a different release (FR-EX-08)"
        ))),
        None => Err(VokraError::ModelLoad(format!(
            "canary-1b-flash: missing required release metadata `{key}`; reconvert the complete pinned `.nemo` checkpoint"
        ))),
    }
}

fn require_release_u32(file: &GgufFile, key: &str, expected: u32) -> Result<()> {
    match file.get(key) {
        Some(GgufMetadataValue::U32(value)) if *value == expected => Ok(()),
        Some(other) => Err(VokraError::ModelLoad(format!(
            "canary-1b-flash: `{key}` must be u32 {expected}, found {other:?}; refusing metadata from a different release (FR-EX-08)"
        ))),
        None => Err(VokraError::ModelLoad(format!(
            "canary-1b-flash: missing required release metadata `{key}`; reconvert the complete pinned `.nemo` checkpoint"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    //! Tests for the Canary-1B-Flash runtime binder.
    //!
    //! Synthetic files exercise fail-closed metadata and manifest boundaries.
    //! The real 3.54 GB `.nemo` forward/parity suite is VAST-only and must use
    //! the independent upstream NeMo implementation as its oracle.

    use super::*;
    use vokra_core::gguf::{GgmlType, GgufBuilder};

    /// Builds a deliberately partial GGUF for negative and diagnostic tests.
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

    fn runtime_metadata_gguf(n_fft: u32) -> GgufFile {
        let mut builder = GgufBuilder::new();
        for &(key, value) in RELEASE_STRING_METADATA {
            builder.add_string(key, value);
        }
        for &(key, value) in RELEASE_U32_METADATA {
            builder.add_u32(
                key,
                if key == GGUF_KEY_FRONTEND_N_FFT {
                    n_fft
                } else {
                    value
                },
            );
        }
        builder.add_f32(GGUF_KEY_FRONTEND_PREEMPHASIS, 0.97);
        GgufFile::parse(builder.to_bytes().expect("serialize metadata")).expect("parse metadata")
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
        assert_eq!(TASK_TOKENS.len(), 8, "eight Canary2 prompt slots");
        assert!(
            !TASK_TOKENS.contains(&"<taskname>"),
            "Canary2 selects ASR/AST from the language pair"
        );
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
        assert!(
            c.encoder.convolution_bias,
            "ConformerEncoder::use_bias defaults true and released conv tensors carry biases"
        );
        assert!(!c.encoder.scale_input, "xscaling = false");
        assert!(c.decoder.pre_ln);
        assert_eq!(c.decoder.hidden_act, "relu");
        assert_eq!(c.encoder.head_dim(), 128);
        assert_eq!(c.head.vocab_size, VOCAB_SIZE);
        assert_eq!(c.head.pad_token_id, PAD_ID);
        assert_eq!(c.head.bos_token_id, BOS_ID);
        assert_eq!(c.head.eos_token_id, EOS_ID);
        c.validate_for_forward()
            .expect("released axes must validate");
        c.validate_release_contract()
            .expect("released axes must match the immutable contract");
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
        // ... while encoder depth and widths remain shared.
        assert_eq!(flash.encoder.n_layer, v2.encoder.n_layer);
        assert_eq!(flash.encoder.d_model, v2.encoder.d_model);
        assert_eq!(flash.encoder.ffn_dim, v2.encoder.ffn_dim);
    }

    #[test]
    fn release_contract_rejects_conflicting_non_shape_metadata() {
        let mut c = Canary1bFlashConfig::canary_1b_flash();
        c.head.vocab_size = 0;
        let Err(err) = c.validate_for_forward() else {
            panic!("shared family validation must reject an empty head");
        };
        assert!(matches!(err, VokraError::InvalidArgument(_)));

        let mut c = Canary1bFlashConfig::canary_1b_flash();
        c.sample_rate = 8_000;
        let Err(VokraError::ModelLoad(message)) = c.validate_release_contract() else {
            panic!("conflicting inference metadata must fail the release contract");
        };
        assert!(message.contains("1,374-tensor"), "message: {message}");
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
    fn strict_binder_rejects_the_historical_encoder_only_shape() {
        let file = flash_gguf(Some(LicenseClass::AttributionRequired), true);
        let Err(VokraError::ModelLoad(message)) = Canary1bFlashAsr::from_gguf(&file) else {
            panic!("a partial checkpoint must not bind");
        };
        assert!(message.contains("tensor count 2"), "message: {message}");
        assert!(message.contains("expected 1374"), "message: {message}");
    }

    #[test]
    fn diagnostic_manifest_view_remains_available_without_binding() {
        let file = flash_gguf(None, false);
        let weights = Canary1bFlashWeights::from_gguf(&file).expect("non-empty manifest");
        assert_eq!(
            weights
                .require_tensor("encoder.blocks.0.attn.qkv_proj.weight")
                .expect("present tensor"),
            &[2, 3]
        );
        assert_eq!(weights.count_with_prefix("decoder."), 1);
        assert_eq!(weights.tensor_names().len(), 2);
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
        let Err(VokraError::ModelLoad(message)) =
            Canary1bFlashAsr::from_gguf_with_policy(&bytes, &CompliancePolicy::strict())
        else {
            panic!("the licence gate passes, then the one-tensor manifest must fail");
        };
        assert!(message.contains("expected 1374"), "message: {message}");
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
                assert!(m.contains("tensor count 0"), "must name the gap: {m}");
                assert!(m.contains("expected 1374"), "must name the contract: {m}");
            }
            other => panic!("expected ModelLoad, got {other:?}"),
        }
    }

    #[test]
    fn require_tensor_names_the_missing_tensor() {
        let file = flash_gguf(Some(LicenseClass::AttributionRequired), false);
        let weights = Canary1bFlashWeights::from_gguf(&file).expect("scan");
        let Err(err) = weights.require_tensor("encoder.blocks.31.ff2_fc2.weight") else {
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
        let weights = Canary1bFlashWeights::from_gguf(&file).expect("scan");
        // Correct dims pass.
        weights
            .require_tensor_dims("decoder.blocks.0.self_attn.qkv.weight", &[1, 4])
            .expect("matching dims must pass");
        // Wrong dims fail loud, naming both sides.
        let Err(err) =
            weights.require_tensor_dims("decoder.blocks.0.self_attn.qkv.weight", &[8, 8])
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
        // The generic metadata reader reflects the stamp, while the immutable
        // release contract rejects the conflict before inference.
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        b.add_string(chunks::KEY_MODEL_NAME, NAME);
        b.add_u32(GGUF_KEY_DEC_N_LAYER, 6);
        b.add_u32(GGUF_KEY_HEAD_VOCAB_SIZE, 16_384);
        b.add_tensor("encoder.probe", GgmlType::F32, vec![2, 2], vec![0u8; 16])
            .expect("add_tensor");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let config = Canary1bFlashConfig::from_gguf(&file).expect("read metadata");
        assert_eq!(config.decoder.n_layer, 6, "stamp overrides the anchor");
        assert_eq!(config.head.vocab_size, 16_384);
        assert_eq!(
            config.source,
            Canary1bFlashConfigSource::GgufStamped,
            "a present stamp must be reported as such"
        );
        assert!(!config.is_family_anchored());
        // Untouched axes still come from the family anchor.
        assert_eq!(config.encoder.n_layer, ENCODER_N_LAYER);
        let Err(VokraError::ModelLoad(message)) = config.validate_release_contract() else {
            panic!("the immutable release must reject conflicting axes");
        };
        assert!(message.contains("1,374-tensor"), "message: {message}");
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
        let Err(err) = Canary1bFlashConfig::from_gguf(&file) else {
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
    // 5 — backend and task contracts
    // -----------------------------------------------------------------------

    #[test]
    fn cpu_covers_the_complete_canary_hot_op_registry() {
        Compute::for_backend(BackendKind::Cpu, CANARY_1B_FLASH_HOT_OPS)
            .expect("CPU must cover every declared Canary op");
    }

    #[test]
    fn complete_runtime_metadata_is_required_exactly() {
        validate_runtime_metadata(&runtime_metadata_gguf(512)).expect("canonical release metadata");
        let Err(VokraError::ModelLoad(message)) =
            validate_runtime_metadata(&runtime_metadata_gguf(1_024))
        else {
            panic!("a conflicting frontend FFT must fail closed");
        };
        assert!(message.contains(GGUF_KEY_FRONTEND_N_FFT), "{message}");
        assert!(message.contains("u32 512"), "{message}");
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    fn metal_covers_the_complete_canary_hot_op_registry() {
        Compute::for_backend(BackendKind::Metal, CANARY_1B_FLASH_HOT_OPS)
            .expect("Metal must cover every declared Canary op without CPU fallback");
    }

    #[test]
    fn hot_op_registry_covers_decoder_and_fastconformer_kernels() {
        for op in [
            HotOp::Gemm,
            HotOp::Gemv,
            HotOp::Softmax,
            HotOp::LayerNorm,
            HotOp::Relu,
            HotOp::GroupedConv1d,
        ] {
            assert!(CANARY_1B_FLASH_HOT_OPS.contains(&op), "missing {op:?}");
        }
    }

    #[test]
    fn task_labels_remain_stable() {
        assert_eq!(Canary1bFlashTask::Asr.as_str(), "asr");
        assert_eq!(Canary1bFlashTask::Ast.as_str(), "ast");
    }

    /// VAST-only flip-the-switch gate against the independent official NeMo
    /// hypothesis. No fixture values are embedded or synthesized here: the
    /// three paths must come from `canary_1b_flash_dump_reference.py` and the
    /// authenticated complete converter output in the same remote run.
    #[test]
    #[ignore = "VAST-only: loads the 3.54 GB released checkpoint and independent NeMo fixture"]
    fn released_checkpoint_matches_official_nemo_greedy_tokens() {
        let gguf_path = std::env::var("VOKRA_CANARY_REAL_GGUF")
            .expect("set VOKRA_CANARY_REAL_GGUF on the provisioned VAST host");
        let pcm_path = std::env::var("VOKRA_CANARY_REFERENCE_PCM")
            .expect("set VOKRA_CANARY_REFERENCE_PCM from the NeMo dumper");
        let tokens_path = std::env::var("VOKRA_CANARY_REFERENCE_TOKENS")
            .expect("set VOKRA_CANARY_REFERENCE_TOKENS from the NeMo dumper");

        let gguf_bytes = std::fs::read(&gguf_path).expect("read complete Canary GGUF on VAST");
        let gguf = GgufFile::parse(gguf_bytes).expect("parse complete Canary GGUF");
        let model = Canary1bFlashAsr::from_gguf_with_backend(&gguf, BackendKind::Cpu)
            .expect("bind complete Canary release on CPU");

        let pcm_bytes = std::fs::read(&pcm_path).expect("read NeMo input PCM");
        assert_eq!(
            pcm_bytes.len() % 4,
            0,
            "reference PCM must be raw little-endian f32"
        );
        let pcm = pcm_bytes
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four-byte f32")))
            .collect::<Vec<_>>();
        let expected = std::fs::read_to_string(&tokens_path)
            .expect("read official NeMo token fixture")
            .split_whitespace()
            .map(|token| token.parse::<u32>().expect("decimal token id"))
            .collect::<Vec<_>>();
        assert!(
            !expected.is_empty(),
            "official NeMo tokens must not be empty"
        );

        let language = |variable: &str, default: CanaryLanguage| {
            let Ok(code) = std::env::var(variable) else {
                return default;
            };
            match code.as_str() {
                "en" => CanaryLanguage::English,
                "de" => CanaryLanguage::German,
                "es" => CanaryLanguage::Spanish,
                "fr" => CanaryLanguage::French,
                other => panic!("{variable}={other:?} must be one of en/de/es/fr"),
            }
        };
        let options = Canary1bFlashOptions {
            source_language: language("VOKRA_CANARY_SOURCE_LANGUAGE", CanaryLanguage::English),
            target_language: language("VOKRA_CANARY_TARGET_LANGUAGE", CanaryLanguage::English),
            ..Canary1bFlashOptions::default()
        };

        let actual = model
            .transcribe_with_options(&pcm, options)
            .expect("run Canary CPU forward");
        assert_eq!(
            actual, expected,
            "Vokra greedy token sequence must exactly match official NeMo"
        );
    }
}
