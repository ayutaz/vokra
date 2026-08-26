//! Strict released-checkpoint binding for Alibaba Qwen3-ASR.
//!
//! The public Vokra artifacts preserve the official Hugging Face tensor names
//! and BF16 shapes verbatim.  This module authenticates both released sizes
//! without eagerly widening either checkpoint: 0.6B carries 612 tensors and
//! 1.7B carries 708 tensors.  The complete sorted `(name, shape)` manifests
//! were audited with bounded HTTP Range reads of the public GGUF headers on
//! 2026-08-27; tensor payloads were not downloaded for that audit.
//!
//! The exact variable-length frontend, three-convolution audio stem,
//! 18/24-layer audio Transformer, projector and fixed-revision Qwen2 BPE/chat
//! contract and bounded-memory 28-layer Qwen3 decoder are executable without
//! runtime downloads. Every learned audio and text op uses one preflighted CPU
//! or Metal backend; unsupported backends fail explicitly rather than silently
//! selecting the CPU.

use std::path::Path;
use std::sync::Arc;

use vokra_core::backend::BackendKind;
use vokra_core::gguf::{GgufFile, GgufMetadataValue, chunks};
use vokra_core::{LicenseClass, Result, VokraError};

use crate::strict_checkpoint::{StrictCheckpoint, StrictCheckpointSpec};

mod audio_encoder;
mod frontend;
mod text_decoder;
mod tokenizer;
mod weights;

use audio_encoder::Qwen3AsrAudioRuntime;
pub use audio_encoder::{QWEN3_ASR_AUDIO_HOT_OPS, Qwen3AsrAudioEmbeddings};
use text_decoder::Qwen3AsrTextRuntime;
pub use text_decoder::{QWEN3_ASR_HOT_OPS, QWEN3_ASR_TEXT_HOT_OPS, Qwen3AsrGenerationOptions};
pub use tokenizer::{
    ASR_TEXT_TOKEN_ID, AUDIO_END_TOKEN_ID, AUDIO_PAD_TOKEN_ID, AUDIO_START_TOKEN_ID,
    END_OF_TEXT_TOKEN_ID, IM_END_TOKEN_ID, IM_START_TOKEN_ID, Qwen3AsrTokenizer,
    Qwen3AsrTranscription, SUPPORTED_LANGUAGES,
};
use weights::Qwen3AsrMappedDescriptors;

/// `vokra.model.arch` shared by the two Qwen3-ASR release sizes.
pub const EXPECTED_ARCH: &str = "qwen3_asr";
/// Official waveform sample rate from the pinned Qwen3-ASR preprocessor.
pub const SAMPLE_RATE: u32 = 16_000;

const LABEL: &str = "qwen3_asr";
const CATEGORY: &str = "asr";
const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
const KEY_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";
const KEY_UPSTREAM_REVISION: &str = "vokra.provenance.upstream_revision";
const KEY_SOURCE_REVISION: &str = "vokra.qwen3_asr.source_revision";

const KEY_AUDIO_D_MODEL: &str = "vokra.qwen3_asr.audio.d_model";
const KEY_AUDIO_N_LAYER: &str = "vokra.qwen3_asr.audio.n_layer";
const KEY_AUDIO_N_HEAD: &str = "vokra.qwen3_asr.audio.n_head";
const KEY_AUDIO_FFN_DIM: &str = "vokra.qwen3_asr.audio.ffn_dim";
const KEY_AUDIO_N_MELS: &str = "vokra.qwen3_asr.audio.n_mels";
const KEY_AUDIO_MAX_SOURCE_POSITIONS: &str = "vokra.qwen3_asr.audio.max_source_positions";
const KEY_AUDIO_OUTPUT_DIM: &str = "vokra.qwen3_asr.audio.output_dim";
const KEY_AUDIO_DOWNSAMPLE_HIDDEN_SIZE: &str = "vokra.qwen3_asr.audio.downsample_hidden_size";
const KEY_AUDIO_CONV_CHUNKSIZE: &str = "vokra.qwen3_asr.audio.conv_chunksize";
const KEY_AUDIO_N_WINDOW: &str = "vokra.qwen3_asr.audio.n_window";
const KEY_AUDIO_N_WINDOW_INFER: &str = "vokra.qwen3_asr.audio.n_window_infer";
const KEY_AUDIO_LAYER_NORM_EPS: &str = "vokra.qwen3_asr.audio.layer_norm_eps";
const KEY_AUDIO_ACTIVATION_FUNCTION: &str = "vokra.qwen3_asr.audio.activation_function";
const KEY_AUDIO_SCALE_EMBEDDING: &str = "vokra.qwen3_asr.audio.scale_embedding";

const KEY_TEXT_HIDDEN_SIZE: &str = "vokra.qwen3_asr.text.hidden_size";
const KEY_TEXT_N_LAYER: &str = "vokra.qwen3_asr.text.n_layer";
const KEY_TEXT_N_HEAD: &str = "vokra.qwen3_asr.text.n_head";
const KEY_TEXT_N_KV_HEAD: &str = "vokra.qwen3_asr.text.n_kv_head";
const KEY_TEXT_HEAD_DIM: &str = "vokra.qwen3_asr.text.head_dim";
const KEY_TEXT_FFN_DIM: &str = "vokra.qwen3_asr.text.ffn_dim";
const KEY_TEXT_MAX_POSITION_EMBEDDINGS: &str = "vokra.qwen3_asr.text.max_position_embeddings";
const KEY_TEXT_ROPE_THETA: &str = "vokra.qwen3_asr.text.rope_theta";
const KEY_TEXT_RMS_NORM_EPS: &str = "vokra.qwen3_asr.text.rms_norm_eps";
const KEY_TEXT_VOCAB_SIZE: &str = "vokra.qwen3_asr.text.vocab_size";
const KEY_TEXT_TIE_WORD_EMBEDDINGS: &str = "vokra.qwen3_asr.text.tie_word_embeddings";
const KEY_TEXT_ATTENTION_BIAS: &str = "vokra.qwen3_asr.text.attention_bias";

const KEY_AUDIO_START_TOKEN_ID: &str = "vokra.qwen3_asr.audio_start_token_id";
const KEY_AUDIO_END_TOKEN_ID: &str = "vokra.qwen3_asr.audio_end_token_id";
const KEY_AUDIO_TOKEN_ID: &str = "vokra.qwen3_asr.audio_token_id";

const SPEC_06: StrictCheckpointSpec = StrictCheckpointSpec {
    label: LABEL,
    arch: EXPECTED_ARCH,
    model_name: "qwen3-asr-0.6b",
    model_name_alias: None,
    tensor_count: 612,
    manifest_sha256: [
        0x8f, 0xf0, 0x41, 0xc0, 0x12, 0x25, 0xc0, 0xc7, 0x43, 0xaf, 0x73, 0x86, 0x97, 0x8c, 0xa5,
        0x16, 0xaf, 0xc6, 0x33, 0xe0, 0x00, 0xb1, 0x81, 0xf4, 0xd4, 0x9d, 0x77, 0x5b, 0x8e, 0x99,
        0xf9, 0x1b,
    ],
};

const SPEC_17: StrictCheckpointSpec = StrictCheckpointSpec {
    label: LABEL,
    arch: EXPECTED_ARCH,
    model_name: "qwen3-asr-1.7b",
    model_name_alias: None,
    tensor_count: 708,
    manifest_sha256: [
        0x91, 0x36, 0xbf, 0x1d, 0xe4, 0x2a, 0x32, 0x48, 0xfb, 0x1e, 0xa5, 0x58, 0x77, 0xdc, 0xed,
        0x61, 0x13, 0xa8, 0xb1, 0xe5, 0xa9, 0x8f, 0xca, 0xe0, 0x8b, 0x01, 0xb6, 0x7f, 0x10, 0xa5,
        0x23, 0xee,
    ],
};

/// One of the two official Qwen3-ASR release sizes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qwen3AsrVariant {
    /// `Qwen/Qwen3-ASR-0.6B`.
    B06,
    /// `Qwen/Qwen3-ASR-1.7B`.
    B17,
}

impl Qwen3AsrVariant {
    /// Exact `vokra.model.name` admitted by the strict binder.
    #[must_use]
    pub const fn model_name(self) -> &'static str {
        match self {
            Self::B06 => "qwen3-asr-0.6b",
            Self::B17 => "qwen3-asr-1.7b",
        }
    }

    /// Immutable upstream Hugging Face repository for this size.
    #[must_use]
    pub const fn upstream_hf(self) -> &'static str {
        match self {
            Self::B06 => "Qwen/Qwen3-ASR-0.6B",
            Self::B17 => "Qwen/Qwen3-ASR-1.7B",
        }
    }

    /// Immutable upstream snapshot revision admitted for this size.
    #[must_use]
    pub const fn source_revision(self) -> &'static str {
        match self {
            Self::B06 => "5eb144179a02acc5e5ba31e748d22b0cf3e303b0",
            Self::B17 => "7278e1e70fe206f11671096ffdd38061171dd6e5a",
        }
    }

    /// Complete tensor count in the released checkpoint.
    #[must_use]
    pub const fn tensor_count(self) -> usize {
        match self {
            Self::B06 => SPEC_06.tensor_count,
            Self::B17 => SPEC_17.tensor_count,
        }
    }

    /// Canonical lowercase SHA-256 of the sorted `(name, shape)` manifest.
    #[must_use]
    pub const fn manifest_sha256(self) -> &'static str {
        match self {
            Self::B06 => "8ff041c01225c0c743af7386978ca516afc633e000b181f4d49d775b8e99f91b",
            Self::B17 => "9136bf1de42a3248fb1ea55877dced6113a8b1e5a98fcae08b01b67f10a523ee",
        }
    }

    const fn spec(self) -> StrictCheckpointSpec {
        match self {
            Self::B06 => SPEC_06,
            Self::B17 => SPEC_17,
        }
    }

    /// Exact architecture axes for this release.
    #[must_use]
    pub const fn config(self) -> Qwen3AsrConfig {
        match self {
            Self::B06 => Qwen3AsrConfig {
                audio: Qwen3AsrAudioConfig {
                    d_model: 896,
                    n_layer: 18,
                    n_head: 14,
                    ffn_dim: 3_584,
                    n_mels: 128,
                    max_source_positions: 1_500,
                    output_dim: 1_024,
                    downsample_hidden_size: 480,
                    conv_chunksize: 500,
                    n_window: 50,
                    n_window_infer: 800,
                    layer_norm_eps: 1.0e-5,
                    scale_embedding: false,
                },
                text: Qwen3AsrTextConfig {
                    hidden_size: 1_024,
                    n_layer: 28,
                    n_head: 16,
                    n_kv_head: 8,
                    head_dim: 128,
                    ffn_dim: 3_072,
                    max_position_embeddings: 65_536,
                    rope_theta: 1_000_000.0,
                    rms_norm_eps: 1.0e-6,
                    vocab_size: 151_936,
                    tie_word_embeddings: true,
                    attention_bias: false,
                },
                audio_start_token_id: 151_669,
                audio_end_token_id: 151_670,
                audio_token_id: 151_676,
            },
            Self::B17 => Qwen3AsrConfig {
                audio: Qwen3AsrAudioConfig {
                    d_model: 1_024,
                    n_layer: 24,
                    n_head: 16,
                    ffn_dim: 4_096,
                    n_mels: 128,
                    max_source_positions: 1_500,
                    output_dim: 2_048,
                    downsample_hidden_size: 480,
                    conv_chunksize: 500,
                    n_window: 50,
                    n_window_infer: 800,
                    layer_norm_eps: 1.0e-5,
                    scale_embedding: false,
                },
                text: Qwen3AsrTextConfig {
                    hidden_size: 2_048,
                    n_layer: 28,
                    n_head: 16,
                    n_kv_head: 8,
                    head_dim: 128,
                    ffn_dim: 6_144,
                    max_position_embeddings: 65_536,
                    rope_theta: 1_000_000.0,
                    rms_norm_eps: 1.0e-6,
                    vocab_size: 151_936,
                    tie_word_embeddings: true,
                    attention_bias: false,
                },
                audio_start_token_id: 151_669,
                audio_end_token_id: 151_670,
                audio_token_id: 151_676,
            },
        }
    }
}

/// Official audio-tower axes persisted in the GGUF contract.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Qwen3AsrAudioConfig {
    /// Hidden width of the audio Transformer.
    pub d_model: u32,
    /// Number of audio Transformer layers.
    pub n_layer: u32,
    /// Number of audio self-attention heads.
    pub n_head: u32,
    /// Width of each audio feed-forward block.
    pub ffn_dim: u32,
    /// Number of log-mel input bins.
    pub n_mels: u32,
    /// Maximum source positions accepted by the audio tower.
    pub max_source_positions: u32,
    /// Width projected into the text decoder.
    pub output_dim: u32,
    /// Channel width of the convolutional downsampler.
    pub downsample_hidden_size: u32,
    /// Training-time convolution chunk size recorded by the release.
    pub conv_chunksize: u32,
    /// Training-time local attention window.
    pub n_window: u32,
    /// Inference-time local attention window.
    pub n_window_infer: u32,
    /// Epsilon of every affine audio LayerNorm.
    pub layer_norm_eps: f32,
    /// Whether convolutional embeddings are scaled by `sqrt(d_model)`.
    pub scale_embedding: bool,
}

/// Official Qwen3 text-decoder axes persisted in the GGUF contract.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Qwen3AsrTextConfig {
    /// Hidden width of the Qwen3 decoder.
    pub hidden_size: u32,
    /// Number of autoregressive decoder layers.
    pub n_layer: u32,
    /// Number of query heads.
    pub n_head: u32,
    /// Number of grouped key/value heads.
    pub n_kv_head: u32,
    /// Per-head query/key/value width.
    pub head_dim: u32,
    /// Width of each SwiGLU feed-forward block.
    pub ffn_dim: u32,
    /// Maximum RoPE position count.
    pub max_position_embeddings: u32,
    /// Base frequency used by rotary embeddings.
    pub rope_theta: f32,
    /// Epsilon used by RMS normalization.
    pub rms_norm_eps: f32,
    /// Number of text and control tokens.
    pub vocab_size: u32,
    /// Whether the release ties input and output token embeddings.
    pub tie_word_embeddings: bool,
    /// Whether attention projection layers carry bias terms.
    pub attention_bias: bool,
}

/// Full Qwen3-ASR topology read from one strict GGUF.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Qwen3AsrConfig {
    /// Audio-tower topology.
    pub audio: Qwen3AsrAudioConfig,
    /// Autoregressive text-decoder topology.
    pub text: Qwen3AsrTextConfig,
    /// Token opening an injected audio span.
    pub audio_start_token_id: u32,
    /// Token closing an injected audio span.
    pub audio_end_token_id: u32,
    /// Placeholder token replaced by projected audio features.
    pub audio_token_id: u32,
}

impl Qwen3AsrConfig {
    fn from_gguf(file: &GgufFile) -> Result<Self> {
        optional_string_value(file, KEY_AUDIO_ACTIVATION_FUNCTION, "gelu")?;
        Ok(Self {
            audio: Qwen3AsrAudioConfig {
                d_model: required_u32(file, KEY_AUDIO_D_MODEL)?,
                n_layer: required_u32(file, KEY_AUDIO_N_LAYER)?,
                n_head: required_u32(file, KEY_AUDIO_N_HEAD)?,
                ffn_dim: required_u32(file, KEY_AUDIO_FFN_DIM)?,
                n_mels: required_u32(file, KEY_AUDIO_N_MELS)?,
                max_source_positions: required_u32(file, KEY_AUDIO_MAX_SOURCE_POSITIONS)?,
                output_dim: required_u32(file, KEY_AUDIO_OUTPUT_DIM)?,
                downsample_hidden_size: required_u32(file, KEY_AUDIO_DOWNSAMPLE_HIDDEN_SIZE)?,
                conv_chunksize: required_u32(file, KEY_AUDIO_CONV_CHUNKSIZE)?,
                n_window: required_u32(file, KEY_AUDIO_N_WINDOW)?,
                n_window_infer: required_u32(file, KEY_AUDIO_N_WINDOW_INFER)?,
                layer_norm_eps: optional_f32(file, KEY_AUDIO_LAYER_NORM_EPS)?.unwrap_or(1.0e-5),
                scale_embedding: optional_bool(file, KEY_AUDIO_SCALE_EMBEDDING)?.unwrap_or(false),
            },
            text: Qwen3AsrTextConfig {
                hidden_size: required_u32(file, KEY_TEXT_HIDDEN_SIZE)?,
                n_layer: required_u32(file, KEY_TEXT_N_LAYER)?,
                n_head: required_u32(file, KEY_TEXT_N_HEAD)?,
                n_kv_head: required_u32(file, KEY_TEXT_N_KV_HEAD)?,
                head_dim: required_u32(file, KEY_TEXT_HEAD_DIM)?,
                ffn_dim: required_u32(file, KEY_TEXT_FFN_DIM)?,
                max_position_embeddings: required_u32(file, KEY_TEXT_MAX_POSITION_EMBEDDINGS)?,
                rope_theta: required_f32(file, KEY_TEXT_ROPE_THETA)?,
                rms_norm_eps: required_f32(file, KEY_TEXT_RMS_NORM_EPS)?,
                vocab_size: required_u32(file, KEY_TEXT_VOCAB_SIZE)?,
                tie_word_embeddings: required_bool(file, KEY_TEXT_TIE_WORD_EMBEDDINGS)?,
                attention_bias: required_bool(file, KEY_TEXT_ATTENTION_BIAS)?,
            },
            audio_start_token_id: required_u32(file, KEY_AUDIO_START_TOKEN_ID)?,
            audio_end_token_id: required_u32(file, KEY_AUDIO_END_TOKEN_ID)?,
            audio_token_id: required_u32(file, KEY_AUDIO_TOKEN_ID)?,
        })
    }

    fn validate_for_variant(self, variant: Qwen3AsrVariant) -> Result<()> {
        let expected = variant.config();
        if self != expected {
            return Err(VokraError::ModelLoad(format!(
                "qwen3_asr: stamped topology {self:?} does not match {:?} for {}",
                expected,
                variant.model_name()
            )));
        }
        if self.audio.d_model % self.audio.n_head != 0 {
            return Err(VokraError::ModelLoad(format!(
                "qwen3_asr: audio d_model {} is not divisible by {} heads",
                self.audio.d_model, self.audio.n_head
            )));
        }
        if self.text.n_head % self.text.n_kv_head != 0 || self.text.n_head * self.text.head_dim == 0
        {
            return Err(VokraError::ModelLoad(
                "qwen3_asr: invalid text GQA/head topology".to_owned(),
            ));
        }
        Ok(())
    }
}

/// Cheap proof that a GGUF is exactly one released Qwen3-ASR checkpoint.
///
/// [`Self::open_mapped`] and [`Self::from_gguf_mapped`] additionally retain
/// the original mmap and a structured descriptor for every tensor. The
/// multi-gigabyte checkpoint is never widened during binding.
#[derive(Clone)]
pub struct Qwen3AsrCheckpoint {
    checkpoint: StrictCheckpoint,
    variant: Qwen3AsrVariant,
    config: Qwen3AsrConfig,
    mapped: Option<Arc<Qwen3AsrMappedDescriptors>>,
    tokenizer: Option<Arc<Qwen3AsrTokenizer>>,
}

impl std::fmt::Debug for Qwen3AsrCheckpoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Qwen3AsrCheckpoint")
            .field("variant", &self.variant)
            .field("tensor_count", &self.tensor_count())
            .field("weight_license", &self.weight_license())
            .field("mapped", &self.mapped.is_some())
            .field("tokenizer", &self.tokenizer.is_some())
            .finish()
    }
}

impl Qwen3AsrCheckpoint {
    /// Opens the GGUF through the true mmap loader, strictly binds every
    /// descriptor and authenticates the embedded tokenizer contract without
    /// decoding tensor payloads.
    pub fn open_mapped(path: impl AsRef<Path>) -> Result<Self> {
        let file = vokra_mmap::open_gguf(path.as_ref()).map_err(VokraError::from)?;
        Self::from_gguf_mapped(Arc::new(file))
    }

    /// Strictly binds an already mmap-backed GGUF and retains its mapping for
    /// layer-at-a-time execution.
    ///
    /// Passing a buffered [`GgufFile::open`] inside the `Arc` is valid but
    /// defeats the bounded-memory contract; normal callers should use
    /// [`Self::open_mapped`] or `vokra_mmap::open_gguf`.
    pub fn from_gguf_mapped(file: Arc<GgufFile>) -> Result<Self> {
        let mut checkpoint = Self::from_gguf(&file)?;
        frontend::check_frontend_spec(&file)?;
        require_string_value(&file, KEY_AUDIO_ACTIVATION_FUNCTION, "gelu")?;
        require_f32_value(&file, KEY_AUDIO_LAYER_NORM_EPS, 1.0e-5)?;
        require_bool_value(&file, KEY_AUDIO_SCALE_EMBEDDING, false)?;
        let tokenizer = Arc::new(Qwen3AsrTokenizer::from_gguf(&file)?);
        let mapped = Qwen3AsrMappedDescriptors::bind(file, checkpoint.config)?;
        checkpoint.mapped = Some(Arc::new(mapped));
        checkpoint.tokenizer = Some(tokenizer);
        Ok(checkpoint)
    }

    /// Validates identity, provenance, all topology chunks and the complete
    /// release-specific tensor name/shape manifest.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        let model_name = required_string(file, chunks::KEY_MODEL_NAME)?;
        let variant = match model_name {
            "qwen3-asr-0.6b" => Qwen3AsrVariant::B06,
            "qwen3-asr-1.7b" => Qwen3AsrVariant::B17,
            other => {
                return Err(VokraError::ModelLoad(format!(
                    "qwen3_asr: unsupported `{}`={other:?}; expected {:?} or {:?}",
                    chunks::KEY_MODEL_NAME,
                    Qwen3AsrVariant::B06.model_name(),
                    Qwen3AsrVariant::B17.model_name()
                )));
            }
        };
        require_string_value(file, KEY_MODEL_CATEGORY, CATEGORY)?;
        require_string_value(file, KEY_UPSTREAM_HF, variant.upstream_hf())?;
        require_string_value(file, KEY_UPSTREAM_REVISION, variant.source_revision())?;
        require_string_value(file, KEY_SOURCE_REVISION, variant.source_revision())?;
        let checkpoint = StrictCheckpoint::bind(file, variant.spec())?;
        if checkpoint.weight_license() != LicenseClass::Permissive {
            return Err(VokraError::ModelLoad(format!(
                "qwen3_asr: `{}` must classify as `permissive` for the pinned Apache-2.0 release, got {:?}",
                chunks::KEY_PROVENANCE_WEIGHT_LICENSE,
                checkpoint.weight_license()
            )));
        }
        let config = Qwen3AsrConfig::from_gguf(file)?;
        config.validate_for_variant(variant)?;
        Ok(Self {
            checkpoint,
            variant,
            config,
            mapped: None,
            tokenizer: None,
        })
    }

    /// Release size selected by the authenticated manifest.
    #[must_use]
    pub const fn variant(&self) -> Qwen3AsrVariant {
        self.variant
    }

    /// Exact topology parsed from the GGUF metadata.
    #[must_use]
    pub const fn config(&self) -> &Qwen3AsrConfig {
        &self.config
    }

    /// Exact model name validated by the strict binder.
    #[must_use]
    pub fn model_name(&self) -> &str {
        self.checkpoint.model_name()
    }

    /// Stamped weight-license class after the fail-closed Apache check.
    #[must_use]
    pub const fn weight_license(&self) -> LicenseClass {
        self.checkpoint.weight_license()
    }

    /// Number of tensor descriptors authenticated by the manifest SHA.
    #[must_use]
    pub const fn tensor_count(&self) -> usize {
        self.checkpoint.tensor_count()
    }

    /// Whether this checkpoint retains a true mapped payload for execution.
    #[must_use]
    pub const fn is_mapped(&self) -> bool {
        self.mapped.is_some()
    }

    pub(super) fn mapped(&self) -> Result<&Qwen3AsrMappedDescriptors> {
        self.mapped.as_deref().ok_or_else(|| {
            VokraError::ModelLoad(
                "qwen3_asr: executable inference requires Qwen3AsrCheckpoint::open_mapped or from_gguf_mapped; from_gguf performs descriptor-only validation"
                    .to_owned(),
            )
        })
    }

    fn tokenizer(&self) -> Result<&Arc<Qwen3AsrTokenizer>> {
        self.tokenizer.as_ref().ok_or_else(|| {
            VokraError::ModelLoad(
                "qwen3_asr: executable inference requires the five exact embedded tokenizer/chat/generation sidecars; re-convert the pinned release"
                    .to_owned(),
            )
        })
    }

    /// Descriptor-only diagnostic entry point. Construct [`Qwen3Asr`] through
    /// its mapped constructor to select and preflight an execution backend.
    pub fn transcribe(&self, pcm: &[f32], sample_rate: u32) -> Result<String> {
        if pcm.is_empty() {
            return Err(VokraError::InvalidArgument(
                "qwen3_asr transcribe: PCM input is empty".to_owned(),
            ));
        }
        if sample_rate != SAMPLE_RATE {
            return Err(VokraError::InvalidArgument(format!(
                "qwen3_asr transcribe: sample_rate={sample_rate}, expected {SAMPLE_RATE} Hz"
            )));
        }
        Err(VokraError::UnsupportedOp(format!(
            "qwen3_asr transcribe: the exact {} {}-tensor checkpoint is descriptor-only and has no selected backend; construct Qwen3Asr::open_mapped to authenticate the tokenizer and run its complete {}-layer audio tower plus 28-layer decoder on one explicit CPU or Metal backend",
            self.variant.model_name(),
            self.tensor_count(),
            self.config.audio.n_layer
        )))
    }
}

/// Executable Qwen3-ASR model with one explicit CPU or Metal backend.
#[derive(Clone)]
pub struct Qwen3Asr {
    checkpoint: Qwen3AsrCheckpoint,
    backend: BackendKind,
    audio: Arc<Qwen3AsrAudioRuntime>,
    text: Arc<Qwen3AsrTextRuntime>,
    tokenizer: Arc<Qwen3AsrTokenizer>,
}

impl std::fmt::Debug for Qwen3Asr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Qwen3Asr")
            .field("variant", &self.checkpoint.variant())
            .field("backend", &self.backend)
            .finish_non_exhaustive()
    }
}

impl Qwen3Asr {
    /// Opens an exact dense GGUF through mmap, authenticates the tokenizer and
    /// preflights the complete audio/text graph on `backend`.
    pub fn open_mapped(path: impl AsRef<Path>, backend: BackendKind) -> Result<Self> {
        Self::from_checkpoint(Qwen3AsrCheckpoint::open_mapped(path)?, backend)
    }

    /// Builds an executable model from a mapped strict checkpoint.
    pub fn from_checkpoint(checkpoint: Qwen3AsrCheckpoint, backend: BackendKind) -> Result<Self> {
        checkpoint.mapped()?;
        let tokenizer = Arc::clone(checkpoint.tokenizer()?);
        let _ = crate::compute::Compute::for_backend(backend, QWEN3_ASR_HOT_OPS)?;
        Ok(Self {
            checkpoint,
            backend,
            audio: Arc::new(Qwen3AsrAudioRuntime::default()),
            text: Arc::new(Qwen3AsrTextRuntime::default()),
            tokenizer,
        })
    }

    /// Selected backend for every learned audio and decoder operation.
    #[must_use]
    pub const fn backend(&self) -> BackendKind {
        self.backend
    }

    /// Authenticated mapped checkpoint.
    #[must_use]
    pub const fn checkpoint(&self) -> &Qwen3AsrCheckpoint {
        &self.checkpoint
    }

    /// Fixed-revision Qwen2 byte-BPE and Qwen3-ASR prompt contract.
    #[must_use]
    pub fn tokenizer(&self) -> &Qwen3AsrTokenizer {
        &self.tokenizer
    }

    /// Runs the exact variable-length frontend, convolutional stem,
    /// 18/24-layer audio Transformer and text-width projector.
    pub fn encode_audio(&self, pcm: &[f32], sample_rate: u32) -> Result<Qwen3AsrAudioEmbeddings> {
        validate_audio_input(pcm, sample_rate)?;
        audio_encoder::encode(&self.checkpoint, self.backend, &self.audio, pcm)
    }

    /// Runs deterministic end-to-end ASR with the released generation
    /// defaults and returns the transcript text.
    pub fn transcribe(&self, pcm: &[f32], sample_rate: u32) -> Result<String> {
        Ok(self
            .transcribe_with_options(pcm, sample_rate, &Qwen3AsrGenerationOptions::default())?
            .text)
    }

    /// Runs end-to-end ASR with explicit context, language and length controls.
    pub fn transcribe_with_options(
        &self,
        pcm: &[f32],
        sample_rate: u32,
        options: &Qwen3AsrGenerationOptions,
    ) -> Result<Qwen3AsrTranscription> {
        validate_audio_input(pcm, sample_rate)?;
        let audio = audio_encoder::encode(&self.checkpoint, self.backend, &self.audio, pcm)?;
        text_decoder::transcribe(
            &self.checkpoint,
            self.backend,
            &self.text,
            &self.tokenizer,
            &audio,
            options,
        )
    }
}

fn validate_audio_input(pcm: &[f32], sample_rate: u32) -> Result<()> {
    if pcm.is_empty() {
        return Err(VokraError::InvalidArgument(
            "qwen3_asr: PCM input is empty".to_owned(),
        ));
    }
    if sample_rate != SAMPLE_RATE {
        return Err(VokraError::InvalidArgument(format!(
            "qwen3_asr: sample_rate={sample_rate}, expected {SAMPLE_RATE} Hz"
        )));
    }
    Ok(())
}

fn required_string<'a>(file: &'a GgufFile, key: &str) -> Result<&'a str> {
    file.get(key)
        .and_then(GgufMetadataValue::as_str)
        .ok_or_else(|| VokraError::ModelLoad(format!("qwen3_asr: missing/non-string `{key}`")))
}

fn require_string_value(file: &GgufFile, key: &str, expected: &str) -> Result<()> {
    let value = required_string(file, key)?;
    if value != expected {
        return Err(VokraError::ModelLoad(format!(
            "qwen3_asr: `{key}`={value:?}, expected {expected:?}"
        )));
    }
    Ok(())
}

fn optional_string_value(file: &GgufFile, key: &str, expected: &str) -> Result<()> {
    match file.get(key) {
        None => Ok(()),
        Some(GgufMetadataValue::String(value)) if value == expected => Ok(()),
        Some(value) => Err(VokraError::ModelLoad(format!(
            "qwen3_asr: `{key}`={value:?}, expected string {expected:?}"
        ))),
    }
}

fn required_u32(file: &GgufFile, key: &str) -> Result<u32> {
    let value = file
        .get(key)
        .and_then(GgufMetadataValue::as_u64)
        .ok_or_else(|| VokraError::ModelLoad(format!("qwen3_asr: missing/non-u32 `{key}`")))?;
    u32::try_from(value)
        .map_err(|_| VokraError::ModelLoad(format!("qwen3_asr: `{key}`={value} exceeds u32")))
}

fn required_f32(file: &GgufFile, key: &str) -> Result<f32> {
    let value = file
        .get(key)
        .and_then(GgufMetadataValue::as_f64)
        .ok_or_else(|| VokraError::ModelLoad(format!("qwen3_asr: missing/non-float `{key}`")))?;
    if !value.is_finite() || value < f32::MIN as f64 || value > f32::MAX as f64 {
        return Err(VokraError::ModelLoad(format!(
            "qwen3_asr: `{key}`={value} is not a finite f32"
        )));
    }
    Ok(value as f32)
}

fn optional_f32(file: &GgufFile, key: &str) -> Result<Option<f32>> {
    match file.get(key) {
        None => Ok(None),
        Some(value) if value.as_f64().is_some() => required_f32(file, key).map(Some),
        Some(value) => Err(VokraError::ModelLoad(format!(
            "qwen3_asr: `{key}`={value:?}, expected float"
        ))),
    }
}

fn require_f32_value(file: &GgufFile, key: &str, expected: f32) -> Result<()> {
    let value = required_f32(file, key)?;
    if value.to_bits() != expected.to_bits() {
        return Err(VokraError::ModelLoad(format!(
            "qwen3_asr: `{key}`={value}, expected {expected}"
        )));
    }
    Ok(())
}

fn required_bool(file: &GgufFile, key: &str) -> Result<bool> {
    file.get(key)
        .and_then(GgufMetadataValue::as_bool)
        .ok_or_else(|| VokraError::ModelLoad(format!("qwen3_asr: missing/non-bool `{key}`")))
}

fn optional_bool(file: &GgufFile, key: &str) -> Result<Option<bool>> {
    match file.get(key) {
        None => Ok(None),
        Some(GgufMetadataValue::Bool(value)) => Ok(Some(*value)),
        Some(value) => Err(VokraError::ModelLoad(format!(
            "qwen3_asr: `{key}`={value:?}, expected bool"
        ))),
    }
}

fn require_bool_value(file: &GgufFile, key: &str, expected: bool) -> Result<()> {
    let value = required_bool(file, key)?;
    if value != expected {
        return Err(VokraError::ModelLoad(format!(
            "qwen3_asr: `{key}`={value}, expected {expected}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::strict_checkpoint::sha256_bytes;

    fn expected_manifest(variant: Qwen3AsrVariant) -> BTreeMap<String, Vec<u64>> {
        weights::tensor_contract(variant.config())
            .into_iter()
            .collect()
    }

    fn manifest_sha256(manifest: &BTreeMap<String, Vec<u64>>) -> [u8; 32] {
        let mut canonical = Vec::new();
        for (name, shape) in manifest {
            canonical.extend_from_slice(name.as_bytes());
            canonical.push(0);
            canonical.extend_from_slice(&(shape.len() as u64).to_le_bytes());
            for dimension in shape {
                canonical.extend_from_slice(&dimension.to_le_bytes());
            }
        }
        sha256_bytes(&canonical)
    }

    #[test]
    fn generated_manifests_match_range_audited_public_headers() {
        for variant in [Qwen3AsrVariant::B06, Qwen3AsrVariant::B17] {
            let manifest = expected_manifest(variant);
            assert_eq!(manifest.len(), variant.tensor_count());
            assert_eq!(manifest_sha256(&manifest), variant.spec().manifest_sha256);
        }
    }

    #[test]
    fn variant_axes_pin_nonstandard_q_projection_width() {
        let b06 = expected_manifest(Qwen3AsrVariant::B06);
        assert_eq!(
            b06["thinker.model.layers.0.self_attn.q_proj.weight"],
            vec![2_048, 1_024]
        );
        assert_eq!(
            b06["thinker.model.layers.0.self_attn.k_proj.weight"],
            vec![1_024, 1_024]
        );
        let b17 = expected_manifest(Qwen3AsrVariant::B17);
        assert_eq!(b17["thinker.audio_tower.proj2.weight"], vec![2_048, 1_024]);
    }

    #[test]
    fn configs_are_forward_safe_and_distinct() {
        let b06 = Qwen3AsrVariant::B06.config();
        let b17 = Qwen3AsrVariant::B17.config();
        b06.validate_for_variant(Qwen3AsrVariant::B06)
            .expect("0.6B config");
        b17.validate_for_variant(Qwen3AsrVariant::B17)
            .expect("1.7B config");
        assert!(b06.validate_for_variant(Qwen3AsrVariant::B17).is_err());
        assert_ne!(b06, b17);
    }

    #[test]
    fn source_revisions_are_full_lowercase_commits_and_distinct() {
        let b06 = Qwen3AsrVariant::B06.source_revision();
        let b17 = Qwen3AsrVariant::B17.source_revision();
        for revision in [b06, b17] {
            assert_eq!(revision.len(), 40);
            assert!(revision.bytes().all(|byte| byte.is_ascii_hexdigit()));
            assert_eq!(revision, revision.to_ascii_lowercase());
        }
        assert_ne!(b06, b17);
    }
}
