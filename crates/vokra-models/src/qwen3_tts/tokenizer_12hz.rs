//! Strict binder for the official Qwen3-TTS 12 Hz neural waveform decoder.
//!
//! This is a companion to the Qwen3-TTS codec LM, not a replacement for its
//! talker or code predictor. The GGUF contains only the 271 `decoder.*`
//! tensors required for code-to-wave synthesis; the 225 audio-encoder tensors
//! from the upstream tokenizer checkpoint are intentionally removed by the
//! authenticated converter.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use vokra_core::backend::BackendKind;
use vokra_core::gguf::{GgmlType, GgufFile, GgufMetadataValue, chunks};
use vokra_core::{CompliancePolicy, LicenseClass, Result, VokraError, check_weight_license};

use crate::compute::Compute;

use super::tokenizer_12hz_forward::{MappedDecoder, QWEN3_TTS_TOKENIZER_12HZ_HOT_OPS};

/// Runtime architecture written by the decode-only converter.
pub const EXPECTED_ARCH: &str = "qwen3_tts_tokenizer_12hz";
/// Exact decode-only model name written by the converter.
pub const MODEL_NAME: &str = "qwen3-tts-tokenizer-12hz-decoder";
/// Pinned official model repository.
pub const UPSTREAM_HF: &str = "Qwen/Qwen3-TTS-Tokenizer-12Hz";
/// Revision that introduced the immutable weight and config used here.
pub const UPSTREAM_REVISION: &str = "a87c50897bb00837eb857d0538b29d117541d7f6";
/// Repository tip whose model-card-only change was audited with the weight.
pub const REPOSITORY_REVISION: &str = "7dd38ad4e9bad454aae9cd937d0cd577604fe229";
/// Official Qwen3-TTS source revision used to transcribe the native graph.
pub const SOURCE_REVISION: &str = "022e286b98fbec7e1e916cb940cdf532cd9f488e";
/// Whole-file SHA-256 of the official 496-tensor source checkpoint.
pub const CHECKPOINT_SHA256: &str =
    "836b7b357f5ea43e889936a3709af68dfe3751881acefe4ecf0dbd30ba571258";
/// SHA-256 of the pinned official tokenizer config.
pub const CONFIG_SHA256: &str = "ee65bb901c876664ab8707c487157aa1a6ee57c65969b28fb5ec9dc211e68167";
/// SHA-256 of the pinned official modeling source.
pub const MODELING_SOURCE_SHA256: &str =
    "844e8dd8c0182ef9c6463c874631c22ef3c5a4fd1899dd657016164cc5379628";
/// SHA-256 of the pinned official configuration source.
pub const CONFIGURATION_SOURCE_SHA256: &str =
    "9e30c24394b00cb0366d7da3482b7436468acc1cd3da1a6fe614a1d34653a5e3";
/// Canonical name/shape manifest SHA-256 for the emitted 271 tensors.
pub const DECODER_MANIFEST_SHA256: &str =
    "501397728761b1d97763ec1817f5b36dbbf0132ba272bf8756999b8b1e7f8803";
/// Number of tensors in the decode-only GGUF.
pub const DECODER_TENSOR_COUNT: usize = 271;

const KEY_INPUT_SAMPLE_RATE: &str = "vokra.qwen3_tts_tokenizer_12hz.input_sample_rate";
const KEY_OUTPUT_SAMPLE_RATE: &str = "vokra.qwen3_tts_tokenizer_12hz.output_sample_rate";
const KEY_DECODE_UPSAMPLE_RATE: &str = "vokra.qwen3_tts_tokenizer_12hz.decode_upsample_rate";
const KEY_NUM_QUANTIZERS: &str = "vokra.qwen3_tts_tokenizer_12hz.num_quantizers";
const KEY_NUM_SEMANTIC_QUANTIZERS: &str = "vokra.qwen3_tts_tokenizer_12hz.num_semantic_quantizers";
const KEY_CODEBOOK_SIZE: &str = "vokra.qwen3_tts_tokenizer_12hz.codebook_size";
const KEY_CONFIGURED_SEMANTIC_VOCAB_SIZE: &str =
    "vokra.qwen3_tts_tokenizer_12hz.configured_semantic_vocab_size";
const KEY_CODEBOOK_DIM: &str = "vokra.qwen3_tts_tokenizer_12hz.codebook_dim";
const KEY_QUANTIZER_DIM: &str = "vokra.qwen3_tts_tokenizer_12hz.quantizer_dim";
const KEY_LATENT_DIM: &str = "vokra.qwen3_tts_tokenizer_12hz.latent_dim";
const KEY_HIDDEN_SIZE: &str = "vokra.qwen3_tts_tokenizer_12hz.transformer.hidden_size";
const KEY_INTERMEDIATE_SIZE: &str = "vokra.qwen3_tts_tokenizer_12hz.transformer.intermediate_size";
const KEY_NUM_HIDDEN_LAYERS: &str = "vokra.qwen3_tts_tokenizer_12hz.transformer.num_hidden_layers";
const KEY_NUM_ATTENTION_HEADS: &str =
    "vokra.qwen3_tts_tokenizer_12hz.transformer.num_attention_heads";
const KEY_NUM_KEY_VALUE_HEADS: &str =
    "vokra.qwen3_tts_tokenizer_12hz.transformer.num_key_value_heads";
const KEY_HEAD_DIM: &str = "vokra.qwen3_tts_tokenizer_12hz.transformer.head_dim";
const KEY_RMS_NORM_EPS: &str = "vokra.qwen3_tts_tokenizer_12hz.transformer.rms_norm_eps";
const KEY_ROPE_THETA: &str = "vokra.qwen3_tts_tokenizer_12hz.transformer.rope_theta";
const KEY_SLIDING_WINDOW: &str = "vokra.qwen3_tts_tokenizer_12hz.transformer.sliding_window";
const KEY_LAYER_SCALE_INITIAL: &str =
    "vokra.qwen3_tts_tokenizer_12hz.transformer.layer_scale_initial";
const KEY_DECODER_DIM: &str = "vokra.qwen3_tts_tokenizer_12hz.decoder_dim";
const KEY_UPSAMPLING_RATIOS: &str = "vokra.qwen3_tts_tokenizer_12hz.upsampling_ratios";
const KEY_UPSAMPLE_RATES: &str = "vokra.qwen3_tts_tokenizer_12hz.upsample_rates";
const KEY_CHUNK_SIZE: &str = "vokra.qwen3_tts_tokenizer_12hz.chunk_size";
const KEY_LEFT_CONTEXT: &str = "vokra.qwen3_tts_tokenizer_12hz.left_context";
const KEY_CHECKPOINT_SHA256: &str = "vokra.provenance.checkpoint_sha256";
const KEY_UPSTREAM_REVISION: &str = "vokra.provenance.upstream_revision";
const KEY_REPOSITORY_REVISION: &str = "vokra.provenance.repository_revision";
const KEY_CONFIG_SHA256: &str = "vokra.qwen3_tts_tokenizer_12hz.config_sha256";
const KEY_SOURCE_REVISION: &str = "vokra.qwen3_tts_tokenizer_12hz.source_revision";
const KEY_MODELING_SOURCE_SHA256: &str = "vokra.qwen3_tts_tokenizer_12hz.modeling_source_sha256";
const KEY_CONFIGURATION_SOURCE_SHA256: &str =
    "vokra.qwen3_tts_tokenizer_12hz.configuration_source_sha256";
const KEY_MANIFEST_SHA256: &str = "vokra.qwen3_tts_tokenizer_12hz.decoder_manifest_sha256";

/// Exact topology transcribed from the official immutable config.
#[derive(Debug, Clone, PartialEq)]
pub struct Qwen3TtsTokenizer12HzConfig {
    /// Source audio sample rate (encoder half; retained for handshake).
    pub input_sample_rate: u32,
    /// PCM output sample rate.
    pub output_sample_rate: u32,
    /// PCM samples emitted per 12.5 Hz code frame.
    pub decode_upsample_rate: usize,
    /// Total residual codebooks expected on every decode call.
    pub num_quantizers: usize,
    /// Number of semantic codebooks at the head of the matrix.
    pub num_semantic_quantizers: usize,
    /// Real row count of every decoder RVQ table.
    pub codebook_size: usize,
    /// Config-only semantic vocabulary value. The decoder constructor does
    /// not use this for the first RVQ table; its real row count is 2048.
    pub configured_semantic_vocab_size: usize,
    /// Feature width after the split-RVQ output projections.
    pub codebook_dim: usize,
    /// Width of each Euclidean codebook row.
    pub quantizer_dim: usize,
    /// Width of the pre-transformer convolution and waveform upsampler.
    pub latent_dim: usize,
    /// Transformer residual width.
    pub hidden_size: usize,
    /// Transformer SwiGLU inner width.
    pub intermediate_size: usize,
    /// Number of pre-transformer layers.
    pub num_hidden_layers: usize,
    /// Query attention heads.
    pub num_attention_heads: usize,
    /// Key/value attention heads.
    pub num_key_value_heads: usize,
    /// Per-head width.
    pub head_dim: usize,
    /// RMSNorm epsilon.
    pub rms_norm_eps: f32,
    /// RoPE base.
    pub rope_theta: f32,
    /// Sliding attention window in code frames.
    pub sliding_window: usize,
    /// Learned residual layer-scale initialization value.
    pub layer_scale_initial: f32,
    /// First waveform-decoder channel width.
    pub decoder_dim: usize,
    /// Pre-decoder temporal upsampling factors.
    pub upsampling_ratios: [usize; 2],
    /// Waveform decoder temporal upsampling factors.
    pub upsample_rates: [usize; 4],
    /// Official bounded-memory chunk size in code frames.
    pub chunk_size: usize,
    /// Official left context in code frames.
    pub left_context: usize,
}

impl Qwen3TtsTokenizer12HzConfig {
    /// Pinned official decoder topology.
    #[must_use]
    pub const fn canonical() -> Self {
        Self {
            input_sample_rate: 24_000,
            output_sample_rate: 24_000,
            decode_upsample_rate: 1_920,
            num_quantizers: 16,
            num_semantic_quantizers: 1,
            codebook_size: 2_048,
            configured_semantic_vocab_size: 4_096,
            codebook_dim: 512,
            quantizer_dim: 256,
            latent_dim: 1_024,
            hidden_size: 512,
            intermediate_size: 1_024,
            num_hidden_layers: 8,
            num_attention_heads: 16,
            num_key_value_heads: 16,
            head_dim: 64,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            sliding_window: 72,
            layer_scale_initial: 0.01,
            decoder_dim: 1_536,
            upsampling_ratios: [2, 2],
            upsample_rates: [8, 5, 4, 3],
            chunk_size: 300,
            left_context: 25,
        }
    }

    /// Total waveform expansion per code frame.
    #[must_use]
    pub fn total_upsample(&self) -> usize {
        self.upsampling_ratios
            .iter()
            .chain(self.upsample_rates.iter())
            .product()
    }
}

/// Strictly bound decode-only companion.
#[derive(Debug, Clone)]
pub struct Qwen3TtsTokenizer12HzDecoder {
    config: Qwen3TtsTokenizer12HzConfig,
    weight_license: LicenseClass,
    tensor_count: usize,
    backend: BackendKind,
    mapped: Option<MappedDecoder>,
}

impl Qwen3TtsTokenizer12HzDecoder {
    /// Validates exact metadata, license, dtype, tensor names and tensor shapes
    /// under the fail-closed default compliance policy.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        Self::from_gguf_with_policy(file, &CompliancePolicy::strict())
    }

    /// Validates the exact artifact contract under an explicit compliance
    /// policy. The official release is Apache-2.0, so a non-permissive or
    /// malformed provenance stamp is refused even when a research policy
    /// would otherwise admit it.
    pub fn from_gguf_with_policy(file: &GgufFile, policy: &CompliancePolicy) -> Result<Self> {
        require_string_value(file, chunks::KEY_MODEL_ARCH, EXPECTED_ARCH)?;
        require_string_value(file, chunks::KEY_MODEL_NAME, MODEL_NAME)?;
        require_string_value(
            file,
            chunks::KEY_PROVENANCE_WEIGHT_LICENSE,
            LicenseClass::Permissive.as_str(),
        )?;
        require_string_value(file, chunks::KEY_PROVENANCE_MODEL_ID, MODEL_NAME)?;
        require_string_value(file, chunks::KEY_PROVENANCE_LICENSE, "apache-2.0")?;
        require_string_value(
            file,
            chunks::KEY_PROVENANCE_SOURCE,
            &format!(
                "{UPSTREAM_HF}@{UPSTREAM_REVISION}/model.safetensors sha256:{CHECKPOINT_SHA256}"
            ),
        )?;
        require_string_value(file, KEY_UPSTREAM_REVISION, UPSTREAM_REVISION)?;
        require_string_value(file, KEY_REPOSITORY_REVISION, REPOSITORY_REVISION)?;
        require_string_value(file, KEY_CHECKPOINT_SHA256, CHECKPOINT_SHA256)?;
        require_string_value(file, KEY_CONFIG_SHA256, CONFIG_SHA256)?;
        require_string_value(file, KEY_SOURCE_REVISION, SOURCE_REVISION)?;
        require_string_value(file, KEY_MODELING_SOURCE_SHA256, MODELING_SOURCE_SHA256)?;
        require_string_value(
            file,
            KEY_CONFIGURATION_SOURCE_SHA256,
            CONFIGURATION_SOURCE_SHA256,
        )?;
        require_string_value(file, KEY_MANIFEST_SHA256, DECODER_MANIFEST_SHA256)?;

        let config = config_from_gguf(file)?;
        let expected = Qwen3TtsTokenizer12HzConfig::canonical();
        if config != expected {
            return Err(load_error(format!(
                "metadata topology mismatch: found {config:?}, expected {expected:?}"
            )));
        }
        if config.total_upsample() != config.decode_upsample_rate {
            return Err(load_error(format!(
                "upsample product {} disagrees with decode_upsample_rate {}",
                config.total_upsample(),
                config.decode_upsample_rate
            )));
        }
        validate_manifest(file, &config)?;

        let license = check_weight_license(file, policy)?;
        if license.class != LicenseClass::Permissive {
            return Err(load_error(format!(
                "weight license resolves to {}, expected Permissive for the authenticated Apache-2.0 release",
                license.class.as_str()
            )));
        }
        Ok(Self {
            config,
            weight_license: license.class,
            tensor_count: file.tensors().len(),
            backend: BackendKind::Cpu,
            mapped: None,
        })
    }

    /// Validates a borrowed artifact and preflights one backend. This form is
    /// intentionally validation-only because the borrow cannot keep a 682 MB
    /// checkpoint alive for decode; use [`Self::open_mapped_with_backend`] for
    /// executable weights.
    pub fn from_gguf_with_backend(file: &GgufFile, backend: BackendKind) -> Result<Self> {
        let mut decoder = Self::from_gguf(file)?;
        Compute::for_backend(backend, QWEN3_TTS_TOKENIZER_12HZ_HOT_OPS)?;
        decoder.backend = backend;
        Ok(decoder)
    }

    /// Memory-maps and strictly binds the official decoder for CPU execution.
    /// No tensor payload is copied at bind time.
    pub fn open_mapped(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_mapped_with_policy_and_backend(
            path,
            &CompliancePolicy::strict(),
            BackendKind::Cpu,
        )
    }

    /// Memory-maps and strictly binds the official decoder for one selected
    /// backend. Backend coverage is checked before any tensor page is read.
    pub fn open_mapped_with_backend(path: impl AsRef<Path>, backend: BackendKind) -> Result<Self> {
        Self::open_mapped_with_policy_and_backend(path, &CompliancePolicy::strict(), backend)
    }

    /// Policy-explicit mapping-owning constructor.
    pub fn open_mapped_with_policy_and_backend(
        path: impl AsRef<Path>,
        policy: &CompliancePolicy,
        backend: BackendKind,
    ) -> Result<Self> {
        let file = vokra_mmap::open_gguf(path.as_ref()).map_err(VokraError::from)?;
        Self::from_gguf_mapped_with_policy_and_backend(Arc::new(file), policy, backend)
    }

    /// Binds an existing mapping under the strict policy for one backend.
    pub fn from_gguf_mapped_with_backend(
        file: Arc<GgufFile>,
        backend: BackendKind,
    ) -> Result<Self> {
        Self::from_gguf_mapped_with_policy_and_backend(file, &CompliancePolicy::strict(), backend)
    }

    /// Binds an existing mapping under an explicit compliance policy. The
    /// authenticated release remains required to resolve to Apache-2.0 /
    /// permissive even when a looser research policy is supplied.
    pub fn from_gguf_mapped_with_policy_and_backend(
        file: Arc<GgufFile>,
        policy: &CompliancePolicy,
        backend: BackendKind,
    ) -> Result<Self> {
        let mut decoder = Self::from_gguf_with_policy(&file, policy)?;
        Compute::for_backend(backend, QWEN3_TTS_TOKENIZER_12HZ_HOT_OPS)?;
        decoder.mapped = Some(MappedDecoder::bind(file)?);
        decoder.backend = backend;
        Ok(decoder)
    }

    /// Returns the authenticated topology.
    #[must_use]
    pub const fn config(&self) -> &Qwen3TtsTokenizer12HzConfig {
        &self.config
    }

    /// Returns the stamped fail-closed license class.
    #[must_use]
    pub const fn weight_license(&self) -> LicenseClass {
        self.weight_license
    }

    /// Returns the number of tensors admitted by the strict binder.
    #[must_use]
    pub const fn tensor_count(&self) -> usize {
        self.tensor_count
    }

    /// Returns the selected whole-model execution backend.
    #[must_use]
    pub const fn backend(&self) -> BackendKind {
        self.backend
    }

    /// Returns the decoder's complete learned hot-op inventory.
    #[must_use]
    pub const fn required_hot_ops(&self) -> &'static [crate::compute::HotOp] {
        QWEN3_TTS_TOKENIZER_12HZ_HOT_OPS
    }

    /// Validates a sixteen-row code matrix before neural decoding.
    pub fn validate_codes(&self, codes: &[Vec<u32>]) -> Result<usize> {
        if codes.len() != self.config.num_quantizers {
            return Err(VokraError::InvalidArgument(format!(
                "qwen3_tts_tokenizer_12hz: expected {} codebook rows, got {}",
                self.config.num_quantizers,
                codes.len()
            )));
        }
        let frames = codes.first().map_or(0, Vec::len);
        if frames == 0 {
            return Err(VokraError::InvalidArgument(
                "qwen3_tts_tokenizer_12hz: code matrix has zero frames".to_owned(),
            ));
        }
        for (row, values) in codes.iter().enumerate() {
            if values.len() != frames {
                return Err(VokraError::InvalidArgument(format!(
                    "qwen3_tts_tokenizer_12hz: codebook row {row} has {} frames, expected {frames}",
                    values.len()
                )));
            }
            if let Some((frame, value)) = values
                .iter()
                .copied()
                .enumerate()
                .find(|(_, value)| *value as usize >= self.config.codebook_size)
            {
                return Err(VokraError::InvalidArgument(format!(
                    "qwen3_tts_tokenizer_12hz: codebook row {row} frame {frame} id {value} is outside 0..{}; main-model EOS/control ids must be removed before waveform decode",
                    self.config.codebook_size
                )));
            }
        }
        Ok(frames)
    }

    /// Decodes sixteen codebook rows to 24 kHz mono PCM using the official
    /// bounded-memory 300-frame / 25-frame-left-context schedule.
    pub fn decode_codes(&self, codes: &[Vec<u32>]) -> Result<Vec<f32>> {
        let _frames = self.validate_codes(codes)?;
        let mapped = self.mapped.as_ref().ok_or_else(|| {
            VokraError::UnsupportedOp(
                "qwen3_tts_tokenizer_12hz: decode requires the mapping-owning `open_mapped_with_backend` or `from_gguf_mapped_with_backend` constructor; a borrowed bind cannot retain the 682 MB artifact and Vokra will not create a resident copy or silently fall back to CPU"
                    .to_owned(),
            )
        })?;
        mapped.decode_codes(self.backend, codes, &self.config)
    }
}

fn load_error(message: impl Into<String>) -> VokraError {
    VokraError::ModelLoad(format!("qwen3_tts_tokenizer_12hz: {}", message.into()))
}

fn required_string<'a>(file: &'a GgufFile, key: &str) -> Result<&'a str> {
    file.get(key)
        .and_then(GgufMetadataValue::as_str)
        .ok_or_else(|| load_error(format!("missing/non-string metadata `{key}`")))
}

fn require_string_value(file: &GgufFile, key: &str, expected: &str) -> Result<()> {
    let actual = required_string(file, key)?;
    if actual != expected {
        return Err(load_error(format!(
            "metadata `{key}` is {actual:?}, expected {expected:?}"
        )));
    }
    Ok(())
}

fn required_usize(file: &GgufFile, key: &str) -> Result<usize> {
    file.get(key)
        .and_then(GgufMetadataValue::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| load_error(format!("missing/non-usize metadata `{key}`")))
}

fn required_u32(file: &GgufFile, key: &str) -> Result<u32> {
    file.get(key)
        .and_then(GgufMetadataValue::as_u64)
        .and_then(|value| u32::try_from(value).ok())
        .ok_or_else(|| load_error(format!("missing/non-u32 metadata `{key}`")))
}

fn required_f32(file: &GgufFile, key: &str) -> Result<f32> {
    match file.get(key) {
        Some(GgufMetadataValue::F32(value)) => Ok(*value),
        _ => Err(load_error(format!("missing/non-f32 metadata `{key}`"))),
    }
}

fn required_usize_array<const N: usize>(file: &GgufFile, key: &str) -> Result<[usize; N]> {
    let values = file
        .get(key)
        .and_then(GgufMetadataValue::as_array)
        .ok_or_else(|| load_error(format!("missing/non-array metadata `{key}`")))?;
    if values.values.len() != N {
        return Err(load_error(format!(
            "metadata `{key}` length is {}, expected {N}",
            values.values.len()
        )));
    }
    let parsed = values
        .values
        .iter()
        .map(|value| {
            value
                .as_u64()
                .and_then(|value| usize::try_from(value).ok())
                .ok_or_else(|| load_error(format!("metadata `{key}` contains a non-usize value")))
        })
        .collect::<Result<Vec<_>>>()?;
    parsed.try_into().map_err(|_| {
        load_error(format!(
            "metadata `{key}` could not be converted to a fixed array"
        ))
    })
}

fn config_from_gguf(file: &GgufFile) -> Result<Qwen3TtsTokenizer12HzConfig> {
    Ok(Qwen3TtsTokenizer12HzConfig {
        input_sample_rate: required_u32(file, KEY_INPUT_SAMPLE_RATE)?,
        output_sample_rate: required_u32(file, KEY_OUTPUT_SAMPLE_RATE)?,
        decode_upsample_rate: required_usize(file, KEY_DECODE_UPSAMPLE_RATE)?,
        num_quantizers: required_usize(file, KEY_NUM_QUANTIZERS)?,
        num_semantic_quantizers: required_usize(file, KEY_NUM_SEMANTIC_QUANTIZERS)?,
        codebook_size: required_usize(file, KEY_CODEBOOK_SIZE)?,
        configured_semantic_vocab_size: required_usize(file, KEY_CONFIGURED_SEMANTIC_VOCAB_SIZE)?,
        codebook_dim: required_usize(file, KEY_CODEBOOK_DIM)?,
        quantizer_dim: required_usize(file, KEY_QUANTIZER_DIM)?,
        latent_dim: required_usize(file, KEY_LATENT_DIM)?,
        hidden_size: required_usize(file, KEY_HIDDEN_SIZE)?,
        intermediate_size: required_usize(file, KEY_INTERMEDIATE_SIZE)?,
        num_hidden_layers: required_usize(file, KEY_NUM_HIDDEN_LAYERS)?,
        num_attention_heads: required_usize(file, KEY_NUM_ATTENTION_HEADS)?,
        num_key_value_heads: required_usize(file, KEY_NUM_KEY_VALUE_HEADS)?,
        head_dim: required_usize(file, KEY_HEAD_DIM)?,
        rms_norm_eps: required_f32(file, KEY_RMS_NORM_EPS)?,
        rope_theta: required_f32(file, KEY_ROPE_THETA)?,
        sliding_window: required_usize(file, KEY_SLIDING_WINDOW)?,
        layer_scale_initial: required_f32(file, KEY_LAYER_SCALE_INITIAL)?,
        decoder_dim: required_usize(file, KEY_DECODER_DIM)?,
        upsampling_ratios: required_usize_array(file, KEY_UPSAMPLING_RATIOS)?,
        upsample_rates: required_usize_array(file, KEY_UPSAMPLE_RATES)?,
        chunk_size: required_usize(file, KEY_CHUNK_SIZE)?,
        left_context: required_usize(file, KEY_LEFT_CONTEXT)?,
    })
}

fn validate_manifest(file: &GgufFile, config: &Qwen3TtsTokenizer12HzConfig) -> Result<()> {
    let expected = expected_manifest(config);
    let actual = file
        .tensors()
        .iter()
        .map(|tensor| {
            (
                tensor.name.clone(),
                tensor
                    .dimensions
                    .iter()
                    .map(|&dimension| dimension as usize)
                    .collect::<Vec<_>>(),
            )
        })
        .collect::<BTreeMap<_, _>>();
    if actual != expected {
        let missing = expected
            .keys()
            .filter(|name| !actual.contains_key(*name))
            .take(8)
            .collect::<Vec<_>>();
        let extra = actual
            .keys()
            .filter(|name| !expected.contains_key(*name))
            .take(8)
            .collect::<Vec<_>>();
        let wrong = expected
            .iter()
            .filter_map(|(name, expected_shape)| {
                actual
                    .get(name)
                    .filter(|actual_shape| *actual_shape != expected_shape)
                    .map(|actual_shape| (name, actual_shape, expected_shape))
            })
            .take(8)
            .collect::<Vec<_>>();
        return Err(load_error(format!(
            "tensor manifest mismatch (expected {}, found {}); missing={missing:?}, extra={extra:?}, wrong_shape={wrong:?}",
            expected.len(),
            actual.len()
        )));
    }
    if let Some(tensor) = file
        .tensors()
        .iter()
        .find(|tensor| tensor.dtype != GgmlType::F32)
    {
        return Err(load_error(format!(
            "tensor {:?} has {:?}; the authenticated decode-only checkpoint is F32",
            tensor.name, tensor.dtype
        )));
    }
    Ok(())
}

fn expected_manifest(config: &Qwen3TtsTokenizer12HzConfig) -> BTreeMap<String, Vec<usize>> {
    let mut out = BTreeMap::new();
    add_quantizer_manifest(&mut out, config);
    add_pre_transformer_manifest(&mut out, config);
    add_upsample_manifest(&mut out, config);
    add_wave_decoder_manifest(&mut out, config);
    debug_assert_eq!(out.len(), DECODER_TENSOR_COUNT);
    out
}

fn insert(out: &mut BTreeMap<String, Vec<usize>>, name: impl Into<String>, shape: &[usize]) {
    let old = out.insert(name.into(), shape.to_vec());
    debug_assert!(old.is_none());
}

fn add_quantizer_manifest(
    out: &mut BTreeMap<String, Vec<usize>>,
    config: &Qwen3TtsTokenizer12HzConfig,
) {
    for branch in ["rvq_first", "rvq_rest"] {
        insert(
            out,
            format!("decoder.quantizer.{branch}.input_proj.weight"),
            &[config.quantizer_dim, config.codebook_dim, 1],
        );
        insert(
            out,
            format!("decoder.quantizer.{branch}.output_proj.weight"),
            &[config.codebook_dim, config.quantizer_dim, 1],
        );
    }
    for layer in 0..config.num_quantizers {
        let (branch, layer) = if layer == 0 {
            ("rvq_first", 0)
        } else {
            ("rvq_rest", layer - 1)
        };
        let prefix = format!("decoder.quantizer.{branch}.vq.layers.{layer}._codebook");
        insert(
            out,
            format!("{prefix}.cluster_usage"),
            &[config.codebook_size],
        );
        insert(
            out,
            format!("{prefix}.embedding_sum"),
            &[config.codebook_size, config.quantizer_dim],
        );
    }
}

fn add_pre_transformer_manifest(
    out: &mut BTreeMap<String, Vec<usize>>,
    config: &Qwen3TtsTokenizer12HzConfig,
) {
    insert(
        out,
        "decoder.pre_conv.conv.weight",
        &[config.latent_dim, config.codebook_dim, 3],
    );
    insert(out, "decoder.pre_conv.conv.bias", &[config.latent_dim]);
    for (name, shape) in [
        (
            "decoder.pre_transformer.input_proj.weight",
            vec![config.hidden_size, config.latent_dim],
        ),
        (
            "decoder.pre_transformer.input_proj.bias",
            vec![config.hidden_size],
        ),
        (
            "decoder.pre_transformer.output_proj.weight",
            vec![config.latent_dim, config.hidden_size],
        ),
        (
            "decoder.pre_transformer.output_proj.bias",
            vec![config.latent_dim],
        ),
        (
            "decoder.pre_transformer.norm.weight",
            vec![config.hidden_size],
        ),
    ] {
        insert(out, name, &shape);
    }
    let attention_width = config.num_attention_heads * config.head_dim;
    let key_value_width = config.num_key_value_heads * config.head_dim;
    for layer in 0..config.num_hidden_layers {
        let prefix = format!("decoder.pre_transformer.layers.{layer}");
        for suffix in [
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "self_attn_layer_scale.scale",
            "mlp_layer_scale.scale",
        ] {
            insert(out, format!("{prefix}.{suffix}"), &[config.hidden_size]);
        }
        insert(
            out,
            format!("{prefix}.self_attn.q_proj.weight"),
            &[attention_width, config.hidden_size],
        );
        for projection in ["k_proj", "v_proj"] {
            insert(
                out,
                format!("{prefix}.self_attn.{projection}.weight"),
                &[key_value_width, config.hidden_size],
            );
        }
        insert(
            out,
            format!("{prefix}.self_attn.o_proj.weight"),
            &[config.hidden_size, attention_width],
        );
        for projection in ["gate_proj", "up_proj"] {
            insert(
                out,
                format!("{prefix}.mlp.{projection}.weight"),
                &[config.intermediate_size, config.hidden_size],
            );
        }
        insert(
            out,
            format!("{prefix}.mlp.down_proj.weight"),
            &[config.hidden_size, config.intermediate_size],
        );
    }
}

fn add_upsample_manifest(
    out: &mut BTreeMap<String, Vec<usize>>,
    config: &Qwen3TtsTokenizer12HzConfig,
) {
    for stage in 0..config.upsampling_ratios.len() {
        let prefix = format!("decoder.upsample.{stage}");
        insert(
            out,
            format!("{prefix}.0.conv.weight"),
            &[config.latent_dim, config.latent_dim, 2],
        );
        insert(out, format!("{prefix}.0.conv.bias"), &[config.latent_dim]);
        insert(
            out,
            format!("{prefix}.1.dwconv.conv.weight"),
            &[config.latent_dim, 1, 7],
        );
        insert(
            out,
            format!("{prefix}.1.dwconv.conv.bias"),
            &[config.latent_dim],
        );
        for suffix in ["norm.weight", "norm.bias", "gamma"] {
            insert(out, format!("{prefix}.1.{suffix}"), &[config.latent_dim]);
        }
        insert(
            out,
            format!("{prefix}.1.pwconv1.weight"),
            &[4 * config.latent_dim, config.latent_dim],
        );
        insert(
            out,
            format!("{prefix}.1.pwconv1.bias"),
            &[4 * config.latent_dim],
        );
        insert(
            out,
            format!("{prefix}.1.pwconv2.weight"),
            &[config.latent_dim, 4 * config.latent_dim],
        );
        insert(
            out,
            format!("{prefix}.1.pwconv2.bias"),
            &[config.latent_dim],
        );
    }
}

fn add_wave_decoder_manifest(
    out: &mut BTreeMap<String, Vec<usize>>,
    config: &Qwen3TtsTokenizer12HzConfig,
) {
    insert(
        out,
        "decoder.decoder.0.conv.weight",
        &[config.decoder_dim, config.latent_dim, 7],
    );
    insert(out, "decoder.decoder.0.conv.bias", &[config.decoder_dim]);
    for (stage, rate) in config.upsample_rates.iter().copied().enumerate() {
        let prefix = format!("decoder.decoder.{}.block", stage + 1);
        let in_dim = config.decoder_dim / (1_usize << stage);
        let out_dim = in_dim / 2;
        for parameter in ["alpha", "beta"] {
            insert(out, format!("{prefix}.0.{parameter}"), &[in_dim]);
        }
        insert(
            out,
            format!("{prefix}.1.conv.weight"),
            &[in_dim, out_dim, 2 * rate],
        );
        insert(out, format!("{prefix}.1.conv.bias"), &[out_dim]);
        for residual in 2..5 {
            for activation in ["act1", "act2"] {
                for parameter in ["alpha", "beta"] {
                    insert(
                        out,
                        format!("{prefix}.{residual}.{activation}.{parameter}"),
                        &[out_dim],
                    );
                }
            }
            insert(
                out,
                format!("{prefix}.{residual}.conv1.conv.weight"),
                &[out_dim, out_dim, 7],
            );
            insert(
                out,
                format!("{prefix}.{residual}.conv1.conv.bias"),
                &[out_dim],
            );
            insert(
                out,
                format!("{prefix}.{residual}.conv2.conv.weight"),
                &[out_dim, out_dim, 1],
            );
            insert(
                out,
                format!("{prefix}.{residual}.conv2.conv.bias"),
                &[out_dim],
            );
        }
    }
    let output_dim = config.decoder_dim / (1_usize << config.upsample_rates.len());
    for parameter in ["alpha", "beta"] {
        insert(out, format!("decoder.decoder.5.{parameter}"), &[output_dim]);
    }
    insert(out, "decoder.decoder.6.conv.weight", &[1, output_dim, 7]);
    insert(out, "decoder.decoder.6.conv.bias", &[1]);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_geometry_is_self_consistent() {
        let config = Qwen3TtsTokenizer12HzConfig::canonical();
        assert_eq!(config.total_upsample(), 1_920);
        assert_eq!(expected_manifest(&config).len(), DECODER_TENSOR_COUNT);
    }

    #[test]
    fn code_contract_rejects_control_ids() {
        let decoder = Qwen3TtsTokenizer12HzDecoder {
            config: Qwen3TtsTokenizer12HzConfig::canonical(),
            weight_license: LicenseClass::Permissive,
            tensor_count: DECODER_TENSOR_COUNT,
            backend: BackendKind::Cpu,
            mapped: None,
        };
        let mut codes = vec![vec![0_u32; 2]; 16];
        codes[0][1] = 2_048;
        let error = decoder.validate_codes(&codes).unwrap_err();
        assert!(error.to_string().contains("outside 0..2048"));
    }
}
