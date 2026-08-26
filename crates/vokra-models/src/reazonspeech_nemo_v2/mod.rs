//! Native ReazonSpeech NeMo v2 Japanese long-form ASR runtime.
//!
//! The immutable official NeMo 1.21 archive is a 24-layer FastConformer with
//! Longformer-style relative local attention (`[128, 128]`, one global token),
//! an 8x depthwise-striding frontend, and a two-layer 640-wide RNN-T prediction
//! network. The complete 965-tensor public Vokra GGUF manifest is pinned below;
//! no shape, source revision, or decoder convention is inferred at runtime.
//!
//! CPU and Metal share the imperative [`Compute`] seam. Unsupported backends
//! fail through `Compute::for_backend`; there is no silent CPU fallback.

use vokra_core::gguf::{GgufFile, GgufMetadataValue};
use vokra_core::{AsrEngine, BackendKind, LicenseClass, Result, Transcription, VokraError};

use crate::compute::{Compute, HotOp};
use crate::parakeet::{
    FastConformerAttentionContext, FastConformerConvNorm, ParakeetBoundEncoderBlock,
    ParakeetBoundLstmLayer, ParakeetBoundNorm, ParakeetBoundSubsampling, ParakeetEncoderConfig,
    ParakeetTokenizer, conformer_block_forward_with_context, local_relative_positions,
    parakeet_logmel, subsampling_forward, transpose_out_in,
};
use crate::strict_checkpoint::{StrictCheckpoint, StrictCheckpointSpec, load_tensor, sha256_bytes};

/// Architecture tag written by `vokra-convert` and the existing public GGUF.
pub const EXPECTED_ARCH: &str = "reazonspeech_nemo_v2";
/// Canonical public model name.
pub const MODEL_NAME: &str = "reazonspeech-nemo-v2";
/// Required PCM rate.
pub const SAMPLE_RATE: u32 = 16_000;
/// Official Hugging Face source revision audited through bounded tar ranges.
pub const SOURCE_REVISION: &str = "33693408be76b7cba9fd4a7546a0a8772430211b";
/// SHA-256 of `model_config.yaml` inside the pinned `.nemo` archive.
pub const MODEL_CONFIG_SHA256: &str =
    "88925d58533c40da62007ad39b8abd702646c7e81627dea5b15961c4ad4f9833";
/// SHA-256 of the official 3,000-line plaintext SentencePiece vocabulary.
pub const TOKENIZER_VOCAB_SHA256: &str =
    "989e4950cf53c0fee66f632cdd966bdd840b851a9e0e812322fd667e4b1c07bb";
/// Optional embedded decode-only tokenizer. The existing public GGUF predates
/// this key and therefore binds for token-level APIs but fails text decoding
/// explicitly until it is replaced through the gated publishing workflow.
pub const KEY_TOKENIZER_VOCAB: &str = "vokra.reazonspeech_nemo_v2.tokenizer.vocab";

const LABEL: &str = "ReazonSpeech-NeMo-v2";
const TENSOR_COUNT: usize = 965;
const MANIFEST_SHA256: [u8; 32] = [
    0x06, 0x63, 0x93, 0x29, 0x75, 0xfb, 0x21, 0x57, 0xd1, 0x1f, 0xa8, 0xce, 0x9d, 0x71, 0x83, 0xc6,
    0x9c, 0x00, 0xa3, 0xd3, 0xf3, 0xf0, 0xe9, 0x16, 0xaf, 0xf1, 0xca, 0xb0, 0x55, 0x04, 0x01, 0xab,
];
const SPEC: StrictCheckpointSpec = StrictCheckpointSpec {
    label: LABEL,
    arch: EXPECTED_ARCH,
    model_name: MODEL_NAME,
    model_name_alias: None,
    tensor_count: TENSOR_COUNT,
    manifest_sha256: MANIFEST_SHA256,
};

const KEY_SOURCE_REVISION: &str = "vokra.reazonspeech_nemo_v2.source_revision";
const KEY_MODEL_CONFIG_SHA256: &str = "vokra.reazonspeech_nemo_v2.model_config_sha256";
const KEY_SAMPLE_RATE: &str = "vokra.reazonspeech_nemo_v2.sample_rate";
const KEY_ENC_N_LAYER: &str = "vokra.reazonspeech_nemo_v2.encoder.n_layer";
const KEY_ENC_D_MODEL: &str = "vokra.reazonspeech_nemo_v2.encoder.d_model";
const KEY_ENC_N_HEAD: &str = "vokra.reazonspeech_nemo_v2.encoder.n_head";
const KEY_ENC_FFN_DIM: &str = "vokra.reazonspeech_nemo_v2.encoder.ffn_dim";
const KEY_ENC_CONV_KERNEL: &str = "vokra.reazonspeech_nemo_v2.encoder.conv_kernel_size";
const KEY_ENC_N_MELS: &str = "vokra.reazonspeech_nemo_v2.encoder.n_mels";
const KEY_ENC_SUB_FACTOR: &str = "vokra.reazonspeech_nemo_v2.encoder.subsampling_factor";
const KEY_ENC_SUB_CHANNELS: &str = "vokra.reazonspeech_nemo_v2.encoder.subsampling_channels";
const KEY_ENC_MAX_POS: &str = "vokra.reazonspeech_nemo_v2.encoder.max_position_embeddings";
const KEY_ENC_LEFT_CONTEXT: &str = "vokra.reazonspeech_nemo_v2.encoder.left_context";
const KEY_ENC_RIGHT_CONTEXT: &str = "vokra.reazonspeech_nemo_v2.encoder.right_context";
const KEY_ENC_GLOBAL_TOKENS: &str = "vokra.reazonspeech_nemo_v2.encoder.global_tokens";
const KEY_ENC_GLOBAL_SPACING: &str = "vokra.reazonspeech_nemo_v2.encoder.global_tokens_spacing";
const KEY_DEC_N_LAYER: &str = "vokra.reazonspeech_nemo_v2.decoder.n_layer";
const KEY_DEC_D_MODEL: &str = "vokra.reazonspeech_nemo_v2.decoder.d_model";
const KEY_JOINT_VOCAB_SIZE: &str = "vokra.reazonspeech_nemo_v2.joint.vocab_size";
const KEY_JOINT_BLANK_ID: &str = "vokra.reazonspeech_nemo_v2.joint.blank_token_id";
const KEY_JOINT_MAX_SYMBOLS: &str = "vokra.reazonspeech_nemo_v2.joint.max_symbols_per_step";
const KEY_TOKENIZER_VOCAB_SHA256: &str = "vokra.reazonspeech_nemo_v2.tokenizer.vocab_sha256";

const RUNTIME_KEYS: &[&str] = &[
    KEY_SOURCE_REVISION,
    KEY_MODEL_CONFIG_SHA256,
    KEY_SAMPLE_RATE,
    KEY_ENC_N_LAYER,
    KEY_ENC_D_MODEL,
    KEY_ENC_N_HEAD,
    KEY_ENC_FFN_DIM,
    KEY_ENC_CONV_KERNEL,
    KEY_ENC_N_MELS,
    KEY_ENC_SUB_FACTOR,
    KEY_ENC_SUB_CHANNELS,
    KEY_ENC_MAX_POS,
    KEY_ENC_LEFT_CONTEXT,
    KEY_ENC_RIGHT_CONTEXT,
    KEY_ENC_GLOBAL_TOKENS,
    KEY_ENC_GLOBAL_SPACING,
    KEY_DEC_N_LAYER,
    KEY_DEC_D_MODEL,
    KEY_JOINT_VOCAB_SIZE,
    KEY_JOINT_BLANK_ID,
    KEY_JOINT_MAX_SYMBOLS,
];

/// Every learned hot operation used by the native encoder and RNN-T decoder.
pub const HOT_OPS: &[HotOp] = &[
    HotOp::Gemm,
    HotOp::Gemv,
    HotOp::Softmax,
    HotOp::LayerNorm,
    HotOp::GroupedConv1d,
];

/// Immutable architecture snapshot from the official `model_config.yaml`.
#[derive(Debug, Clone, PartialEq)]
pub struct ReazonSpeechConfig {
    pub encoder: ParakeetEncoderConfig,
    pub left_context: usize,
    pub right_context: usize,
    pub global_tokens: usize,
    pub global_tokens_spacing: usize,
    pub decoder_layers: usize,
    pub decoder_dim: usize,
    /// Includes the tail blank: 3,000 SentencePiece entries + blank.
    pub vocab_size: usize,
    pub blank_id: u32,
    pub max_symbols_per_step: usize,
    pub sample_rate: u32,
}

impl ReazonSpeechConfig {
    #[must_use]
    pub fn official() -> Self {
        Self {
            encoder: ParakeetEncoderConfig {
                n_layer: 24,
                d_model: 1_024,
                n_head: 8,
                n_head_kv: 8,
                ffn_dim: 4_096,
                conv_kernel_size: 9,
                in_dim: 80,
                subsampling_factor: 8,
                subsampling_conv_kernel_size: 3,
                subsampling_conv_stride: 2,
                subsampling_conv_channels: 256,
                max_position_embeddings: 5_000,
                attention_bias: true,
                convolution_bias: true,
                scale_input: true,
            },
            left_context: 128,
            right_context: 128,
            global_tokens: 1,
            global_tokens_spacing: 1,
            decoder_layers: 2,
            decoder_dim: 640,
            vocab_size: 3_001,
            blank_id: 3_000,
            max_symbols_per_step: 10,
            sample_rate: SAMPLE_RATE,
        }
    }

    fn validate(&self) -> Result<()> {
        if !self.encoder.is_well_formed()
            || self.encoder.n_layer == 0
            || self.encoder.ffn_dim == 0
            || self.encoder.conv_kernel_size == 0
            || self.encoder.conv_kernel_size % 2 == 0
            || self.encoder.in_dim == 0
            || self.encoder.subsampling_factor != 8
            || self.left_context == 0
            || self.right_context == 0
            || self.global_tokens == 0
            || self.global_tokens_spacing == 0
            || self.decoder_layers == 0
            || self.decoder_dim == 0
            || self.vocab_size == 0
            || self.blank_id as usize + 1 != self.vocab_size
            || self.max_symbols_per_step == 0
            || self.sample_rate == 0
        {
            return Err(VokraError::InvalidArgument(format!(
                "{LABEL}: invalid runtime config {self:?}"
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
struct ReazonSpeechWeights {
    subsampling: ParakeetBoundSubsampling,
    encoder: Vec<ParakeetBoundEncoderBlock>,
    encoder_projector_w: Vec<f32>,
    encoder_projector_b: Vec<f32>,
    embedding: Vec<f32>,
    lstm: Vec<ParakeetBoundLstmLayer>,
    decoder_projector_w: Vec<f32>,
    decoder_projector_b: Vec<f32>,
    joint_head_w: Vec<f32>,
    joint_head_b: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct ReazonSpeechNemoV2 {
    config: ReazonSpeechConfig,
    weights: Box<ReazonSpeechWeights>,
    tokenizer: Option<ParakeetTokenizer>,
    weight_license: LicenseClass,
    backend: BackendKind,
}

impl ReazonSpeechNemoV2 {
    /// Strictly binds the one audited 965-tensor public release. Legacy public
    /// GGUFs without model-specific axis chunks are accepted only because the
    /// complete name/shape manifest already authenticates the exact topology.
    /// If any new axis chunk is present, all must be present and canonical.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        let checkpoint = StrictCheckpoint::bind(file, SPEC)?;
        let config = ReazonSpeechConfig::official();
        config.validate()?;
        validate_runtime_metadata(file, &config)?;
        let tokenizer = load_tokenizer(file, &config)?;
        let weights = Box::new(load_weights(file, &config)?);
        Ok(Self {
            config,
            weights,
            tokenizer,
            weight_license: checkpoint.weight_license(),
            backend: BackendKind::Cpu,
        })
    }

    #[must_use]
    pub fn with_backend(mut self, backend: BackendKind) -> Self {
        self.backend = backend;
        self
    }

    #[must_use]
    pub const fn backend(&self) -> BackendKind {
        self.backend
    }

    #[must_use]
    pub const fn config(&self) -> &ReazonSpeechConfig {
        &self.config
    }

    #[must_use]
    pub const fn has_tokenizer(&self) -> bool {
        self.tokenizer.is_some()
    }

    #[must_use]
    pub const fn tensor_count(&self) -> usize {
        TENSOR_COUNT
    }

    #[must_use]
    pub const fn weight_license(&self) -> LicenseClass {
        self.weight_license
    }

    /// Runs the log-mel frontend, depthwise subsampler, and all 24 local/global
    /// FastConformer blocks. Output is row-major `[frames, 1024]` before the
    /// RNN-T encoder projection.
    pub fn encode_pcm(&self, pcm: &[f32]) -> Result<(Vec<f32>, usize)> {
        if pcm.is_empty() {
            return Err(VokraError::InvalidArgument(
                "ReazonSpeech-NeMo-v2: PCM slice is empty".to_owned(),
            ));
        }
        let compute = Compute::for_backend(self.backend, HOT_OPS)?;
        let (features, frames) =
            parakeet_logmel(pcm, self.config.sample_rate, self.config.encoder.in_dim)?;
        let (mut hidden, encoded_frames) = subsampling_forward(
            &compute,
            &features,
            frames,
            self.config.encoder.in_dim,
            &self.weights.subsampling,
            &self.config.encoder,
        )?;
        let scale = (self.config.encoder.d_model as f32).sqrt();
        for value in &mut hidden {
            *value *= scale;
        }
        let positions = local_relative_positions(
            self.config.left_context,
            self.config.right_context,
            self.config.encoder.d_model,
        );
        let context = FastConformerAttentionContext::LongformerLocal {
            left_context: self.config.left_context,
            right_context: self.config.right_context,
            global_tokens: self.config.global_tokens,
            global_tokens_spacing: self.config.global_tokens_spacing,
        };
        for block in &self.weights.encoder {
            conformer_block_forward_with_context(
                &compute,
                &mut hidden,
                encoded_frames,
                block,
                &positions,
                &self.config.encoder,
                context,
            )?;
        }
        Ok((hidden, encoded_frames))
    }

    /// Native greedy RNN-T decoding. Repeated tokens are retained and only the
    /// tail blank advances the encoder frame.
    pub fn transcribe_tokens(&self, pcm: &[f32]) -> Result<Vec<u32>> {
        if pcm.is_empty() {
            return Err(VokraError::InvalidArgument(
                "ReazonSpeech-NeMo-v2: PCM slice is empty".to_owned(),
            ));
        }
        let compute = Compute::for_backend(self.backend, HOT_OPS)?;
        let (encoder, frames) = self.encode_pcm(pcm)?;
        let enc_dim = self.config.encoder.d_model;
        let hidden = self.config.decoder_dim;
        let mut projected = vec![0.0f32; frames * hidden];
        for frame in 0..frames {
            linear_into(
                &compute,
                &encoder[frame * enc_dim..(frame + 1) * enc_dim],
                &self.weights.encoder_projector_w,
                &self.weights.encoder_projector_b,
                &mut projected[frame * hidden..(frame + 1) * hidden],
            )?;
        }

        let mut state = DecoderState::new(self.config.decoder_layers, hidden);
        decoder_step(
            &compute,
            self.config.blank_id,
            &self.weights,
            hidden,
            &mut state,
        )?;
        let mut tokens = Vec::new();
        for frame in 0..frames {
            for _ in 0..self.config.max_symbols_per_step {
                let mut joint = vec![0.0f32; hidden];
                for index in 0..hidden {
                    joint[index] =
                        (projected[frame * hidden + index] + state.projected[index]).max(0.0);
                }
                let mut logits = vec![0.0f32; self.config.vocab_size];
                linear_into(
                    &compute,
                    &joint,
                    &self.weights.joint_head_w,
                    &self.weights.joint_head_b,
                    &mut logits,
                )?;
                let token = argmax_finite(&logits)? as u32;
                if token == self.config.blank_id {
                    break;
                }
                tokens.push(token);
                decoder_step(&compute, token, &self.weights, hidden, &mut state)?;
            }
        }
        Ok(tokens)
    }

    pub fn transcribe_text(&self, pcm: &[f32]) -> Result<String> {
        let tokenizer = self.tokenizer.as_ref().ok_or_else(|| {
            VokraError::ModelLoad(format!(
                "{LABEL}: `{KEY_TOKENIZER_VOCAB}` is absent from the legacy public GGUF; token-level inference is available through `transcribe_tokens`, but text decoding requires a gated replacement converted with the pinned official tokenizer vocabulary"
            ))
        })?;
        let tokens = self.transcribe_tokens(pcm)?;
        tokenizer.decode(&tokens, self.config.blank_id, self.config.blank_id, None)
    }
}

impl AsrEngine for ReazonSpeechNemoV2 {
    fn transcribe(&self, pcm: &[f32]) -> Result<Transcription> {
        Ok(Transcription::new(self.transcribe_text(pcm)?))
    }

    fn backend(&self) -> BackendKind {
        self.backend
    }
}

fn validate_runtime_metadata(file: &GgufFile, config: &ReazonSpeechConfig) -> Result<()> {
    let present = RUNTIME_KEYS
        .iter()
        .filter(|&&key| file.get(key).is_some())
        .count();
    if present == 0 {
        return Ok(());
    }
    if present != RUNTIME_KEYS.len() {
        let missing = RUNTIME_KEYS
            .iter()
            .filter(|&&key| file.get(key).is_none())
            .take(6)
            .collect::<Vec<_>>();
        return Err(VokraError::ModelLoad(format!(
            "{LABEL}: incomplete runtime metadata ({present}/{} keys); missing={missing:?}",
            RUNTIME_KEYS.len()
        )));
    }
    required_string(file, KEY_SOURCE_REVISION, SOURCE_REVISION)?;
    required_string(file, KEY_MODEL_CONFIG_SHA256, MODEL_CONFIG_SHA256)?;
    for (key, expected) in [
        (KEY_SAMPLE_RATE, config.sample_rate),
        (KEY_ENC_N_LAYER, config.encoder.n_layer as u32),
        (KEY_ENC_D_MODEL, config.encoder.d_model as u32),
        (KEY_ENC_N_HEAD, config.encoder.n_head as u32),
        (KEY_ENC_FFN_DIM, config.encoder.ffn_dim as u32),
        (KEY_ENC_CONV_KERNEL, config.encoder.conv_kernel_size as u32),
        (KEY_ENC_N_MELS, config.encoder.in_dim as u32),
        (KEY_ENC_SUB_FACTOR, config.encoder.subsampling_factor as u32),
        (
            KEY_ENC_SUB_CHANNELS,
            config.encoder.subsampling_conv_channels as u32,
        ),
        (
            KEY_ENC_MAX_POS,
            config.encoder.max_position_embeddings as u32,
        ),
        (KEY_ENC_LEFT_CONTEXT, config.left_context as u32),
        (KEY_ENC_RIGHT_CONTEXT, config.right_context as u32),
        (KEY_ENC_GLOBAL_TOKENS, config.global_tokens as u32),
        (KEY_ENC_GLOBAL_SPACING, config.global_tokens_spacing as u32),
        (KEY_DEC_N_LAYER, config.decoder_layers as u32),
        (KEY_DEC_D_MODEL, config.decoder_dim as u32),
        (KEY_JOINT_VOCAB_SIZE, config.vocab_size as u32),
        (KEY_JOINT_BLANK_ID, config.blank_id),
        (KEY_JOINT_MAX_SYMBOLS, config.max_symbols_per_step as u32),
    ] {
        required_u32(file, key, expected)?;
    }
    Ok(())
}

fn required_string(file: &GgufFile, key: &str, expected: &str) -> Result<()> {
    let actual = file
        .get(key)
        .and_then(GgufMetadataValue::as_str)
        .ok_or_else(|| VokraError::ModelLoad(format!("{LABEL}: missing/non-string `{key}`")))?;
    if actual != expected {
        return Err(VokraError::ModelLoad(format!(
            "{LABEL}: `{key}`={actual:?}, expected {expected:?}"
        )));
    }
    Ok(())
}

fn required_u32(file: &GgufFile, key: &str, expected: u32) -> Result<()> {
    let actual = match file.get(key) {
        Some(GgufMetadataValue::U32(value)) => *value,
        Some(GgufMetadataValue::U64(value)) => u32::try_from(*value)
            .map_err(|_| VokraError::ModelLoad(format!("{LABEL}: `{key}`={value} exceeds u32")))?,
        Some(other) => {
            return Err(VokraError::ModelLoad(format!(
                "{LABEL}: `{key}` must be unsigned integer, found {other:?}"
            )));
        }
        None => return Err(VokraError::ModelLoad(format!("{LABEL}: missing `{key}`"))),
    };
    if actual != expected {
        return Err(VokraError::ModelLoad(format!(
            "{LABEL}: `{key}`={actual}, expected {expected}"
        )));
    }
    Ok(())
}

fn load_tokenizer(
    file: &GgufFile,
    config: &ReazonSpeechConfig,
) -> Result<Option<ParakeetTokenizer>> {
    let Some(value) = file.get(KEY_TOKENIZER_VOCAB) else {
        return Ok(None);
    };
    let GgufMetadataValue::Array(array) = value else {
        return Err(VokraError::ModelLoad(format!(
            "{LABEL}: `{KEY_TOKENIZER_VOCAB}` must be a u8 array"
        )));
    };
    let bytes = array
        .values
        .iter()
        .map(|value| match value {
            GgufMetadataValue::U8(byte) => Ok(*byte),
            other => Err(VokraError::ModelLoad(format!(
                "{LABEL}: `{KEY_TOKENIZER_VOCAB}` contains non-u8 element {other:?}"
            ))),
        })
        .collect::<Result<Vec<_>>>()?;
    let actual = hex(&sha256_bytes(&bytes));
    if actual != TOKENIZER_VOCAB_SHA256 {
        return Err(VokraError::ModelLoad(format!(
            "{LABEL}: tokenizer SHA-256 {actual}, expected {TOKENIZER_VOCAB_SHA256}"
        )));
    }
    required_string(file, KEY_TOKENIZER_VOCAB_SHA256, TOKENIZER_VOCAB_SHA256)?;
    ParakeetTokenizer::from_sentencepiece_vocab_bytes(&bytes, config.vocab_size, config.blank_id)
        .map(Some)
}

fn load_weights(file: &GgufFile, config: &ReazonSpeechConfig) -> Result<ReazonSpeechWeights> {
    let enc = &config.encoder;
    let channels = enc.subsampling_conv_channels;
    let kernel = enc.subsampling_conv_kernel_size;
    let tensor = |name: &str, shape: &[usize]| load_tensor(file, LABEL, name, shape);
    let out_frequency = enc.in_dim / enc.subsampling_factor;
    let subsampling = ParakeetBoundSubsampling {
        conv0_w: tensor(
            "encoder.pre_encode.conv.0.weight",
            &[channels, 1, kernel, kernel],
        )?,
        conv0_b: tensor("encoder.pre_encode.conv.0.bias", &[channels])?,
        depthwise_w: [
            tensor(
                "encoder.pre_encode.conv.2.weight",
                &[channels, 1, kernel, kernel],
            )?,
            tensor(
                "encoder.pre_encode.conv.5.weight",
                &[channels, 1, kernel, kernel],
            )?,
        ],
        depthwise_b: [
            tensor("encoder.pre_encode.conv.2.bias", &[channels])?,
            tensor("encoder.pre_encode.conv.5.bias", &[channels])?,
        ],
        pointwise_w_t: [
            transpose_out_in(
                tensor(
                    "encoder.pre_encode.conv.3.weight",
                    &[channels, channels, 1, 1],
                )?,
                channels,
                channels,
            ),
            transpose_out_in(
                tensor(
                    "encoder.pre_encode.conv.6.weight",
                    &[channels, channels, 1, 1],
                )?,
                channels,
                channels,
            ),
        ],
        pointwise_b: [
            tensor("encoder.pre_encode.conv.3.bias", &[channels])?,
            tensor("encoder.pre_encode.conv.6.bias", &[channels])?,
        ],
        linear_w_t: transpose_out_in(
            tensor(
                "encoder.pre_encode.out.weight",
                &[enc.d_model, channels * out_frequency],
            )?,
            enc.d_model,
            channels * out_frequency,
        ),
        linear_b: tensor("encoder.pre_encode.out.bias", &[enc.d_model])?,
    };

    let mut encoder = Vec::with_capacity(enc.n_layer);
    for layer in 0..enc.n_layer {
        let prefix = format!("encoder.layers.{layer}");
        let norm = |name: &str| -> Result<ParakeetBoundNorm> {
            Ok(ParakeetBoundNorm {
                weight: tensor(&format!("{prefix}.{name}.weight"), &[enc.d_model])?,
                bias: tensor(&format!("{prefix}.{name}.bias"), &[enc.d_model])?,
            })
        };
        let ff_weight = |branch: &str, linear: usize, output: usize, input: usize| {
            tensor(
                &format!("{prefix}.{branch}.linear{linear}.weight"),
                &[output, input],
            )
            .map(|weight| transpose_out_in(weight, output, input))
        };
        let ff_bias = |branch: &str, linear: usize, output: usize| {
            tensor(&format!("{prefix}.{branch}.linear{linear}.bias"), &[output]).map(Some)
        };
        let attention_weight = |name: &str| {
            tensor(
                &format!("{prefix}.self_attn.{name}.weight"),
                &[enc.d_model, enc.d_model],
            )
            .map(|weight| transpose_out_in(weight, enc.d_model, enc.d_model))
        };
        let attention_bias = |name: &str| {
            tensor(&format!("{prefix}.self_attn.{name}.bias"), &[enc.d_model]).map(Some)
        };
        encoder.push(ParakeetBoundEncoderBlock {
            ff1_w1_t: ff_weight("feed_forward1", 1, enc.ffn_dim, enc.d_model)?,
            ff1_b1: ff_bias("feed_forward1", 1, enc.ffn_dim)?,
            ff1_w2_t: ff_weight("feed_forward1", 2, enc.d_model, enc.ffn_dim)?,
            ff1_b2: ff_bias("feed_forward1", 2, enc.d_model)?,
            ff2_w1_t: ff_weight("feed_forward2", 1, enc.ffn_dim, enc.d_model)?,
            ff2_b1: ff_bias("feed_forward2", 1, enc.ffn_dim)?,
            ff2_w2_t: ff_weight("feed_forward2", 2, enc.d_model, enc.ffn_dim)?,
            ff2_b2: ff_bias("feed_forward2", 2, enc.d_model)?,
            norm_ff1: norm("norm_feed_forward1")?,
            norm_attn: norm("norm_self_att")?,
            norm_conv: norm("norm_conv")?,
            norm_ff2: norm("norm_feed_forward2")?,
            norm_out: norm("norm_out")?,
            q_w_t: attention_weight("linear_q")?,
            q_b: attention_bias("linear_q")?,
            k_w_t: attention_weight("linear_k")?,
            k_b: attention_bias("linear_k")?,
            v_w_t: attention_weight("linear_v")?,
            v_b: attention_bias("linear_v")?,
            o_w_t: attention_weight("linear_out")?,
            o_b: attention_bias("linear_out")?,
            relative_k_w_t: attention_weight("linear_pos")?,
            bias_u: tensor(
                &format!("{prefix}.self_attn.pos_bias_u"),
                &[enc.n_head, enc.head_dim()],
            )?,
            bias_v: tensor(
                &format!("{prefix}.self_attn.pos_bias_v"),
                &[enc.n_head, enc.head_dim()],
            )?,
            conv_pw1_w_t: transpose_out_in(
                tensor(
                    &format!("{prefix}.conv.pointwise_conv1.weight"),
                    &[2 * enc.d_model, enc.d_model, 1],
                )?,
                2 * enc.d_model,
                enc.d_model,
            ),
            conv_pw1_b: Some(tensor(
                &format!("{prefix}.conv.pointwise_conv1.bias"),
                &[2 * enc.d_model],
            )?),
            conv_dw_w: tensor(
                &format!("{prefix}.conv.depthwise_conv.weight"),
                &[enc.d_model, 1, enc.conv_kernel_size],
            )?,
            conv_dw_b: Some(tensor(
                &format!("{prefix}.conv.depthwise_conv.bias"),
                &[enc.d_model],
            )?),
            conv_inner_norm: FastConformerConvNorm::BatchNorm {
                weight: tensor(&format!("{prefix}.conv.batch_norm.weight"), &[enc.d_model])?,
                bias: tensor(&format!("{prefix}.conv.batch_norm.bias"), &[enc.d_model])?,
                running_mean: tensor(
                    &format!("{prefix}.conv.batch_norm.running_mean"),
                    &[enc.d_model],
                )?,
                running_var: tensor(
                    &format!("{prefix}.conv.batch_norm.running_var"),
                    &[enc.d_model],
                )?,
            },
            conv_pw2_w_t: transpose_out_in(
                tensor(
                    &format!("{prefix}.conv.pointwise_conv2.weight"),
                    &[enc.d_model, enc.d_model, 1],
                )?,
                enc.d_model,
                enc.d_model,
            ),
            conv_pw2_b: Some(tensor(
                &format!("{prefix}.conv.pointwise_conv2.bias"),
                &[enc.d_model],
            )?),
        });
    }

    let mut lstm = Vec::with_capacity(config.decoder_layers);
    for layer in 0..config.decoder_layers {
        let prefix = "decoder.prediction.dec_rnn.lstm";
        lstm.push(ParakeetBoundLstmLayer {
            w_ih: tensor(
                &format!("{prefix}.weight_ih_l{layer}"),
                &[4 * config.decoder_dim, config.decoder_dim],
            )?,
            w_hh: tensor(
                &format!("{prefix}.weight_hh_l{layer}"),
                &[4 * config.decoder_dim, config.decoder_dim],
            )?,
            b_ih: tensor(
                &format!("{prefix}.bias_ih_l{layer}"),
                &[4 * config.decoder_dim],
            )?,
            b_hh: tensor(
                &format!("{prefix}.bias_hh_l{layer}"),
                &[4 * config.decoder_dim],
            )?,
        });
    }

    Ok(ReazonSpeechWeights {
        subsampling,
        encoder,
        encoder_projector_w: tensor("joint.enc.weight", &[config.decoder_dim, enc.d_model])?,
        encoder_projector_b: tensor("joint.enc.bias", &[config.decoder_dim])?,
        embedding: tensor(
            "decoder.prediction.embed.weight",
            &[config.vocab_size, config.decoder_dim],
        )?,
        lstm,
        decoder_projector_w: tensor(
            "joint.pred.weight",
            &[config.decoder_dim, config.decoder_dim],
        )?,
        decoder_projector_b: tensor("joint.pred.bias", &[config.decoder_dim])?,
        joint_head_w: tensor(
            "joint.joint_net.2.weight",
            &[config.vocab_size, config.decoder_dim],
        )?,
        joint_head_b: tensor("joint.joint_net.2.bias", &[config.vocab_size])?,
    })
}

#[derive(Debug)]
struct DecoderState {
    hidden: Vec<Vec<f32>>,
    cell: Vec<Vec<f32>>,
    projected: Vec<f32>,
}

impl DecoderState {
    fn new(layers: usize, hidden: usize) -> Self {
        Self {
            hidden: vec![vec![0.0; hidden]; layers],
            cell: vec![vec![0.0; hidden]; layers],
            projected: vec![0.0; hidden],
        }
    }
}

fn decoder_step(
    compute: &Compute,
    token: u32,
    weights: &ReazonSpeechWeights,
    hidden: usize,
    state: &mut DecoderState,
) -> Result<()> {
    let token = token as usize;
    let offset = token.checked_mul(hidden).ok_or_else(|| {
        VokraError::InvalidArgument("ReazonSpeech decoder embedding offset overflow".to_owned())
    })?;
    let mut input = weights
        .embedding
        .get(offset..offset + hidden)
        .ok_or_else(|| {
            VokraError::InvalidArgument(format!(
                "ReazonSpeech decoder token {token} is outside the embedding table"
            ))
        })?
        .to_vec();
    for (layer_index, layer) in weights.lstm.iter().enumerate() {
        let mut gates = vec![0.0f32; 4 * hidden];
        compute.gemv_f32(
            4 * hidden,
            hidden,
            &layer.w_ih,
            &input,
            Some(&layer.b_ih),
            &mut gates,
        )?;
        let mut recurrent = vec![0.0f32; 4 * hidden];
        compute.gemv_f32(
            4 * hidden,
            hidden,
            &layer.w_hh,
            &state.hidden[layer_index],
            Some(&layer.b_hh),
            &mut recurrent,
        )?;
        let mut next = vec![0.0f32; hidden];
        for index in 0..hidden {
            let input_gate = sigmoid(gates[index] + recurrent[index]);
            let forget_gate = sigmoid(gates[hidden + index] + recurrent[hidden + index]);
            let candidate = (gates[2 * hidden + index] + recurrent[2 * hidden + index]).tanh();
            let output_gate = sigmoid(gates[3 * hidden + index] + recurrent[3 * hidden + index]);
            let cell = forget_gate * state.cell[layer_index][index] + input_gate * candidate;
            state.cell[layer_index][index] = cell;
            next[index] = output_gate * cell.tanh();
        }
        state.hidden[layer_index].clone_from(&next);
        input = next;
    }
    linear_into(
        compute,
        &input,
        &weights.decoder_projector_w,
        &weights.decoder_projector_b,
        &mut state.projected,
    )
}

fn linear_into(
    compute: &Compute,
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    output: &mut [f32],
) -> Result<()> {
    compute.gemv_f32(output.len(), input.len(), weight, input, Some(bias), output)
}

fn argmax_finite(values: &[f32]) -> Result<usize> {
    let mut best: Option<(usize, f32)> = None;
    for (index, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(VokraError::InvalidArgument(format!(
                "ReazonSpeech RNN-T logits contain non-finite value at index {index}: {value}"
            )));
        }
        if best.is_none_or(|(_, current)| value > current) {
            best = Some((index, value));
        }
    }
    best.map(|(index, _)| index).ok_or_else(|| {
        VokraError::InvalidArgument("ReazonSpeech RNN-T logits are empty".to_owned())
    })
}

#[inline]
fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

fn hex(bytes: &[u8; 32]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(64);
    for byte in bytes {
        output.push(char::from(DIGITS[(byte >> 4) as usize]));
        output.push(char::from(DIGITS[(byte & 0x0f) as usize]));
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn official_config_matches_pinned_nemo_yaml() {
        let config = ReazonSpeechConfig::official();
        config.validate().unwrap();
        assert_eq!(config.encoder.n_layer, 24);
        assert_eq!(config.encoder.d_model, 1_024);
        assert_eq!(config.encoder.n_head, 8);
        assert_eq!(config.encoder.ffn_dim, 4_096);
        assert_eq!(config.encoder.in_dim, 80);
        assert_eq!(config.encoder.subsampling_factor, 8);
        assert!(config.encoder.attention_bias);
        assert!(config.encoder.convolution_bias);
        assert!(config.encoder.scale_input);
        assert_eq!((config.left_context, config.right_context), (128, 128));
        assert_eq!((config.global_tokens, config.global_tokens_spacing), (1, 1));
        assert_eq!((config.decoder_layers, config.decoder_dim), (2, 640));
        assert_eq!((config.vocab_size, config.blank_id), (3_001, 3_000));
    }

    #[test]
    fn public_manifest_identity_is_pinned() {
        assert_eq!(TENSOR_COUNT, 965);
        assert_eq!(
            hex(&MANIFEST_SHA256),
            "0663932975fb2157d11fa8ce9d7183c69c00a3d3f3f0e916aff1cab0550401ab"
        );
    }
}
