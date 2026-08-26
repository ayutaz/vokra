//! NVIDIA Nemotron-3.5-ASR-Streaming-0.6B native offline inference.
//!
//! The released checkpoint is a prompt-conditioned cache-aware
//! FastConformer + RNN-T model. This module implements the complete offline
//! PCM-to-text route on the shared [`crate::Compute`] seam. CPU and Metal use
//! the same learned-op inventory; selecting an uncovered backend fails before
//! inference and never falls back to CPU.
//!
//! The public `vokra/nemotron-3.5-asr-streaming-0.6b` GGUF predates the
//! self-describing hparam/tokenizer chunk. Its exact 655-tensor manifest is
//! therefore the only legacy contract accepted. Newly converted artifacts
//! must carry the full `vokra.nemotron_asr.*` metadata group and may embed the
//! official `tokenizer.json`. A legacy public GGUF needs the authenticated
//! tokenizer as an explicit sidecar; token inference remains available when
//! text decoding is not requested.
//!
//! Stateful cache-aware chunk streaming is intentionally not claimed by the
//! offline engine. The causal convolutions and released chunk-limited mask are
//! preserved in the full-utterance forward, while a future stateful stream API
//! must carry convolution and K/V caches explicitly.

use vokra_core::gguf::{GgufFile, GgufMetadataValue};
use vokra_core::ir::graph::{MelAttrs, Normalization, PadMode, StftAttrs, Window, WindowSymmetry};
use vokra_core::{AsrEngine, BackendKind, LicenseClass, Result, Transcription, VokraError};
use vokra_ops::mel::MelFilterbank;
use vokra_ops::stft::stft;

use crate::compute::{Compute, HotOp};
use crate::parakeet::{
    FastConformerAttentionContext, FastConformerConvNorm, FastConformerSubsamplingPadding,
    ParakeetBoundEncoderBlock, ParakeetBoundLstmLayer, ParakeetBoundNorm, ParakeetBoundSubsampling,
    ParakeetEncoderConfig, ParakeetTokenizer, conformer_block_forward_with_context,
    relative_positions, subsampling_forward_with_padding, transpose_out_in,
};
use crate::strict_checkpoint::{StrictCheckpoint, StrictCheckpointSpec, load_tensor};

pub const EXPECTED_ARCH: &str = "nemotron_asr_streaming";
pub const MODEL_NAME: &str = "nemotron-3.5-asr-streaming-0.6b";
pub const SAMPLE_RATE: u32 = 16_000;
pub const BLANK_TOKEN_ID: u32 = 13_087;
pub const PAD_TOKEN_ID: u32 = 0;
pub const VOCAB_SIZE: usize = 13_088;
pub const DEFAULT_PROMPT_ID: u32 = 101;
pub const DEFAULT_LOOKAHEAD_TOKENS: usize = 3;
pub const SUPPORTED_LOOKAHEAD_TOKENS: &[usize] = &[3, 0, 6, 13];

/// Converter/runtime key for the byte-exact official Hugging Face tokenizer.
pub const KEY_TOKENIZER_JSON: &str = "vokra.nemotron_asr.tokenizer.json";

pub const NEMOTRON_HOT_OPS: &[HotOp] = &[
    HotOp::Gemm,
    HotOp::Gemv,
    HotOp::Softmax,
    HotOp::LayerNorm,
    HotOp::GroupedConv1d,
];

const LABEL: &str = "Nemotron-3.5-ASR-Streaming-0.6B";
const TENSOR_COUNT: usize = 655;
const MANIFEST_SHA256: [u8; 32] = [
    0x9d, 0xa8, 0x5a, 0x77, 0x82, 0x91, 0x8f, 0x37, 0x2c, 0x12, 0x06, 0x7c, 0xd0, 0x27, 0xe3, 0xc1,
    0x58, 0x5b, 0x95, 0x8e, 0xb8, 0xb2, 0x9e, 0x2d, 0xc0, 0x50, 0xfd, 0x04, 0xaf, 0x0e, 0xf7, 0xcb,
];
const SPEC: StrictCheckpointSpec = StrictCheckpointSpec {
    label: LABEL,
    arch: EXPECTED_ARCH,
    model_name: MODEL_NAME,
    model_name_alias: None,
    tensor_count: TENSOR_COUNT,
    manifest_sha256: MANIFEST_SHA256,
};

const KEY_SAMPLE_RATE: &str = "vokra.nemotron_asr.sample_rate";
const KEY_N_FFT: &str = "vokra.nemotron_asr.frontend.n_fft";
const KEY_HOP_LENGTH: &str = "vokra.nemotron_asr.frontend.hop_length";
const KEY_WIN_LENGTH: &str = "vokra.nemotron_asr.frontend.win_length";
const KEY_PREEMPHASIS: &str = "vokra.nemotron_asr.frontend.preemphasis";
const KEY_N_MELS: &str = "vokra.nemotron_asr.frontend.n_mels";
const KEY_ENC_N_LAYER: &str = "vokra.nemotron_asr.encoder.n_layer";
const KEY_ENC_D_MODEL: &str = "vokra.nemotron_asr.encoder.d_model";
const KEY_ENC_N_HEAD: &str = "vokra.nemotron_asr.encoder.n_head";
const KEY_ENC_N_HEAD_KV: &str = "vokra.nemotron_asr.encoder.n_head_kv";
const KEY_ENC_FFN_DIM: &str = "vokra.nemotron_asr.encoder.ffn_dim";
const KEY_ENC_CONV_KERNEL: &str = "vokra.nemotron_asr.encoder.conv_kernel_size";
const KEY_ENC_SUB_FACTOR: &str = "vokra.nemotron_asr.encoder.subsampling_factor";
const KEY_ENC_SUB_KERNEL: &str = "vokra.nemotron_asr.encoder.subsampling_conv_kernel_size";
const KEY_ENC_SUB_STRIDE: &str = "vokra.nemotron_asr.encoder.subsampling_conv_stride";
const KEY_ENC_SUB_CHANNELS: &str = "vokra.nemotron_asr.encoder.subsampling_conv_channels";
const KEY_ENC_MAX_POS: &str = "vokra.nemotron_asr.encoder.max_position_embeddings";
const KEY_ENC_SLIDING_WINDOW: &str = "vokra.nemotron_asr.encoder.sliding_window";
const KEY_ENC_DEFAULT_LOOKAHEAD: &str = "vokra.nemotron_asr.encoder.default_lookahead_tokens";
const KEY_ENC_ATTN_BIAS: &str = "vokra.nemotron_asr.encoder.attention_bias";
const KEY_ENC_CONV_BIAS: &str = "vokra.nemotron_asr.encoder.convolution_bias";
const KEY_ENC_SCALE_INPUT: &str = "vokra.nemotron_asr.encoder.scale_input";
const KEY_DEC_N_LAYER: &str = "vokra.nemotron_asr.decoder.n_layer";
const KEY_DEC_D_MODEL: &str = "vokra.nemotron_asr.decoder.d_model";
const KEY_VOCAB_SIZE: &str = "vokra.nemotron_asr.joint.vocab_size";
const KEY_BLANK_ID: &str = "vokra.nemotron_asr.joint.blank_token_id";
const KEY_PAD_ID: &str = "vokra.nemotron_asr.joint.pad_token_id";
const KEY_MAX_SYMBOLS: &str = "vokra.nemotron_asr.joint.max_symbols_per_step";
const KEY_NUM_PROMPTS: &str = "vokra.nemotron_asr.prompt.num_prompts";
const KEY_PROMPT_INTERMEDIATE: &str = "vokra.nemotron_asr.prompt.intermediate_size";
const KEY_DEFAULT_PROMPT: &str = "vokra.nemotron_asr.prompt.default_id";
const KEY_ENCODER_ACT: &str = "vokra.nemotron_asr.encoder.hidden_act";
const KEY_JOINT_ACT: &str = "vokra.nemotron_asr.joint.hidden_act";
const PREFIX_LOOKAHEAD: &str = "vokra.nemotron_asr.encoder.supported_lookahead.";

const CONFIG_U32: &[(&str, u32)] = &[
    (KEY_SAMPLE_RATE, 16_000),
    (KEY_N_FFT, 512),
    (KEY_HOP_LENGTH, 160),
    (KEY_WIN_LENGTH, 400),
    (KEY_N_MELS, 128),
    (KEY_ENC_N_LAYER, 24),
    (KEY_ENC_D_MODEL, 1_024),
    (KEY_ENC_N_HEAD, 8),
    (KEY_ENC_N_HEAD_KV, 8),
    (KEY_ENC_FFN_DIM, 4_096),
    (KEY_ENC_CONV_KERNEL, 9),
    (KEY_ENC_SUB_FACTOR, 8),
    (KEY_ENC_SUB_KERNEL, 3),
    (KEY_ENC_SUB_STRIDE, 2),
    (KEY_ENC_SUB_CHANNELS, 256),
    (KEY_ENC_MAX_POS, 5_000),
    (KEY_ENC_SLIDING_WINDOW, 57),
    (KEY_ENC_DEFAULT_LOOKAHEAD, 3),
    (KEY_ENC_ATTN_BIAS, 0),
    (KEY_ENC_CONV_BIAS, 0),
    (KEY_ENC_SCALE_INPUT, 0),
    (KEY_DEC_N_LAYER, 2),
    (KEY_DEC_D_MODEL, 640),
    (KEY_VOCAB_SIZE, 13_088),
    (KEY_BLANK_ID, 13_087),
    (KEY_PAD_ID, 0),
    (KEY_MAX_SYMBOLS, 10),
    (KEY_NUM_PROMPTS, 128),
    (KEY_PROMPT_INTERMEDIATE, 2_048),
    (KEY_DEFAULT_PROMPT, 101),
];

#[derive(Debug, Clone, PartialEq)]
pub struct NemotronAsrConfig {
    pub encoder: ParakeetEncoderConfig,
    pub decoder_hidden_size: usize,
    pub num_decoder_layers: usize,
    pub vocab_size: usize,
    pub blank_token_id: u32,
    pub pad_token_id: u32,
    pub max_symbols_per_step: usize,
    pub num_prompts: usize,
    pub prompt_intermediate_size: usize,
    pub default_prompt_id: u32,
    pub sliding_window: usize,
    pub default_lookahead_tokens: usize,
    pub sample_rate: u32,
}

impl NemotronAsrConfig {
    #[must_use]
    pub fn canonical() -> Self {
        Self {
            encoder: ParakeetEncoderConfig {
                n_layer: 24,
                d_model: 1_024,
                n_head: 8,
                n_head_kv: 8,
                ffn_dim: 4_096,
                conv_kernel_size: 9,
                in_dim: 128,
                subsampling_factor: 8,
                subsampling_conv_kernel_size: 3,
                subsampling_conv_stride: 2,
                subsampling_conv_channels: 256,
                max_position_embeddings: 5_000,
                attention_bias: false,
                convolution_bias: false,
                scale_input: false,
            },
            decoder_hidden_size: 640,
            num_decoder_layers: 2,
            vocab_size: VOCAB_SIZE,
            blank_token_id: BLANK_TOKEN_ID,
            pad_token_id: PAD_TOKEN_ID,
            max_symbols_per_step: 10,
            num_prompts: 128,
            prompt_intermediate_size: 2_048,
            default_prompt_id: DEFAULT_PROMPT_ID,
            sliding_window: 57,
            default_lookahead_tokens: DEFAULT_LOOKAHEAD_TOKENS,
            sample_rate: SAMPLE_RATE,
        }
    }

    fn validate(&self) -> Result<()> {
        if !self.encoder.is_well_formed()
            || self.encoder.n_layer == 0
            || self.encoder.ffn_dim == 0
            || self.encoder.conv_kernel_size == 0
            || self.encoder.subsampling_conv_kernel_size == 0
            || self.encoder.subsampling_conv_stride == 0
        {
            return Err(VokraError::ModelLoad(format!(
                "{LABEL}: invalid FastConformer geometry: {:?}",
                self.encoder
            )));
        }
        if self.blank_token_id as usize >= self.vocab_size
            || self.pad_token_id as usize >= self.vocab_size
            || self.default_prompt_id as usize >= self.num_prompts
            || self.max_symbols_per_step == 0
            || self.sliding_window == 0
        {
            return Err(VokraError::ModelLoad(format!(
                "{LABEL}: invalid RNN-T/prompt configuration: {self:?}"
            )));
        }
        if !SUPPORTED_LOOKAHEAD_TOKENS.contains(&self.default_lookahead_tokens) {
            return Err(VokraError::ModelLoad(format!(
                "{LABEL}: default lookahead {} is not in {SUPPORTED_LOOKAHEAD_TOKENS:?}",
                self.default_lookahead_tokens
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
struct NemotronBoundWeights {
    subsampling: ParakeetBoundSubsampling,
    encoder: Vec<ParakeetBoundEncoderBlock>,
    prompt_w1_t: Vec<f32>,
    prompt_b1: Vec<f32>,
    prompt_w2_t: Vec<f32>,
    prompt_b2: Vec<f32>,
    encoder_projector_w_t: Vec<f32>,
    encoder_projector_b: Vec<f32>,
    embedding: Vec<f32>,
    lstm: Vec<ParakeetBoundLstmLayer>,
    decoder_projector_w: Vec<f32>,
    decoder_projector_b: Vec<f32>,
    joint_head_w: Vec<f32>,
    joint_head_b: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct NemotronAsr {
    config: NemotronAsrConfig,
    weights: Box<NemotronBoundWeights>,
    tokenizer: Option<ParakeetTokenizer>,
    model_name: String,
    weight_license: LicenseClass,
    tensor_count: usize,
    backend: BackendKind,
}

impl NemotronAsr {
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        Self::bind(file, None)
    }

    pub fn from_gguf_with_tokenizer_bytes(file: &GgufFile, tokenizer: &[u8]) -> Result<Self> {
        Self::bind(file, Some(tokenizer))
    }

    fn bind(file: &GgufFile, tokenizer_override: Option<&[u8]>) -> Result<Self> {
        let checkpoint = StrictCheckpoint::bind(file, SPEC)?;
        validate_metadata_contract(file)?;
        let config = NemotronAsrConfig::canonical();
        config.validate()?;
        let tokenizer = match tokenizer_override {
            Some(bytes) => Some(ParakeetTokenizer::from_bytes(bytes, config.vocab_size)?),
            None => embedded_tokenizer(file, config.vocab_size)?,
        };
        let weights = Box::new(load_bound_weights(file, &config)?);
        Ok(Self {
            config,
            weights,
            tokenizer,
            model_name: checkpoint.model_name().to_owned(),
            weight_license: checkpoint.weight_license(),
            tensor_count: checkpoint.tensor_count(),
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
    pub fn config(&self) -> &NemotronAsrConfig {
        &self.config
    }

    #[must_use]
    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    #[must_use]
    pub const fn weight_license(&self) -> LicenseClass {
        self.weight_license
    }

    #[must_use]
    pub const fn tensor_count(&self) -> usize {
        self.tensor_count
    }

    #[must_use]
    pub fn has_tokenizer(&self) -> bool {
        self.tokenizer.is_some()
    }

    pub fn transcribe_tokens(&self, pcm: &[f32]) -> Result<Vec<u32>> {
        self.transcribe_tokens_with_prompt(pcm, self.config.default_prompt_id)
    }

    pub fn transcribe_tokens_with_prompt(&self, pcm: &[f32], prompt_id: u32) -> Result<Vec<u32>> {
        if prompt_id as usize >= self.config.num_prompts {
            return Err(VokraError::InvalidArgument(format!(
                "{LABEL}: prompt_id={prompt_id} is outside 0..{}",
                self.config.num_prompts
            )));
        }
        let compute = Compute::for_backend(self.backend, NEMOTRON_HOT_OPS)?;
        let (encoded, frames) = self.encode_pcm(&compute, pcm, prompt_id)?;
        greedy_rnnt_decode(&compute, &encoded, frames, &self.weights, &self.config)
    }

    pub fn transcribe_with_prompt(&self, pcm: &[f32], prompt_id: u32) -> Result<Transcription> {
        let tokenizer = self.tokenizer.as_ref().ok_or_else(|| {
            VokraError::ModelLoad(format!(
                "{LABEL}: `{KEY_TOKENIZER_JSON}` is absent; use the authenticated official tokenizer.json as an explicit sidecar or reconvert with tokenizer embedding"
            ))
        })?;
        let ids = self.transcribe_tokens_with_prompt(pcm, prompt_id)?;
        let text = tokenizer.decode(
            &ids,
            self.config.blank_token_id,
            self.config.pad_token_id,
            None,
        )?;
        Ok(Transcription::new(text))
    }

    /// Explicit boundary for callers looking for stateful cache streaming.
    pub fn transcribe_streaming_chunks(&self, _chunks: &[&[f32]]) -> Result<Transcription> {
        Err(VokraError::UnsupportedOp(
            "Nemotron 3.5 cache-aware chunk streaming is not exposed yet: the complete offline causal FastConformer/RNN-T path is available, but a stateful stream must preserve Conv2D/Conv1D padding caches plus per-layer attention K/V caches; no whole-utterance or CPU fallback is substituted"
                .to_owned(),
        ))
    }

    fn encode_pcm(
        &self,
        compute: &Compute,
        pcm: &[f32],
        prompt_id: u32,
    ) -> Result<(Vec<f32>, usize)> {
        let (features, feature_frames) = nemotron_logmel(pcm)?;
        let (mut hidden, encoded_frames) = subsampling_forward_with_padding(
            compute,
            &features,
            feature_frames,
            self.config.encoder.in_dim,
            &self.weights.subsampling,
            &self.config.encoder,
            FastConformerSubsamplingPadding::CausalOffline,
        )?;
        if encoded_frames == 0 || encoded_frames > self.config.encoder.max_position_embeddings {
            return Err(VokraError::InvalidArgument(format!(
                "{LABEL}: encoded frame count {encoded_frames} is outside 1..={} ",
                self.config.encoder.max_position_embeddings
            )));
        }
        let positions = relative_positions(encoded_frames, self.config.encoder.d_model);
        let attention_context = FastConformerAttentionContext::ChunkedLimited {
            left_context: self.config.sliding_window - 1,
            right_context: self.config.default_lookahead_tokens,
        };
        for block in &self.weights.encoder {
            conformer_block_forward_with_context(
                compute,
                &mut hidden,
                encoded_frames,
                block,
                &positions,
                &self.config.encoder,
                attention_context,
            )?;
        }
        let width = self.config.encoder.d_model;
        let prompt_width = width + self.config.num_prompts;
        let mut fused_input = vec![0.0f32; encoded_frames * prompt_width];
        for frame in 0..encoded_frames {
            let input_row = &hidden[frame * width..(frame + 1) * width];
            let output_row = &mut fused_input[frame * prompt_width..(frame + 1) * prompt_width];
            output_row[..width].copy_from_slice(input_row);
            output_row[width + prompt_id as usize] = 1.0;
        }
        let inner = self.config.prompt_intermediate_size;
        let mut prompt_hidden = vec![0.0f32; encoded_frames * inner];
        compute.gemm_f32(
            encoded_frames,
            inner,
            prompt_width,
            &fused_input,
            &self.weights.prompt_w1_t,
            Some(&self.weights.prompt_b1),
            &mut prompt_hidden,
        )?;
        for value in &mut prompt_hidden {
            *value = value.max(0.0);
        }
        let mut prompted = vec![0.0f32; encoded_frames * width];
        compute.gemm_f32(
            encoded_frames,
            width,
            inner,
            &prompt_hidden,
            &self.weights.prompt_w2_t,
            Some(&self.weights.prompt_b2),
            &mut prompted,
        )?;
        let decoder_width = self.config.decoder_hidden_size;
        let mut projected = vec![0.0f32; encoded_frames * decoder_width];
        compute.gemm_f32(
            encoded_frames,
            decoder_width,
            width,
            &prompted,
            &self.weights.encoder_projector_w_t,
            Some(&self.weights.encoder_projector_b),
            &mut projected,
        )?;
        Ok((projected, encoded_frames))
    }
}

impl AsrEngine for NemotronAsr {
    fn transcribe(&self, pcm: &[f32]) -> Result<Transcription> {
        self.transcribe_with_prompt(pcm, self.config.default_prompt_id)
    }

    fn backend(&self) -> BackendKind {
        self.backend
    }
}

fn embedded_tokenizer(file: &GgufFile, vocab_size: usize) -> Result<Option<ParakeetTokenizer>> {
    let Some(value) = file.get(KEY_TOKENIZER_JSON) else {
        return Ok(None);
    };
    let GgufMetadataValue::Array(array) = value else {
        return Err(VokraError::ModelLoad(format!(
            "{LABEL}: `{KEY_TOKENIZER_JSON}` must be a u8 array, found {value:?}"
        )));
    };
    let bytes = array
        .values
        .iter()
        .map(|value| match value {
            GgufMetadataValue::U8(byte) => Ok(*byte),
            other => Err(VokraError::ModelLoad(format!(
                "{LABEL}: `{KEY_TOKENIZER_JSON}` contains non-u8 element {other:?}"
            ))),
        })
        .collect::<Result<Vec<_>>>()?;
    ParakeetTokenizer::from_bytes(&bytes, vocab_size).map(Some)
}

fn validate_metadata_contract(file: &GgufFile) -> Result<()> {
    let lookahead_keys = SUPPORTED_LOOKAHEAD_TOKENS
        .iter()
        .enumerate()
        .map(|(index, value)| (format!("{PREFIX_LOOKAHEAD}{index}"), *value as u32))
        .collect::<Vec<_>>();
    let expected_count = CONFIG_U32.len() + lookahead_keys.len() + 3;
    let present_count = CONFIG_U32
        .iter()
        .filter(|(key, _)| file.get(key).is_some())
        .count()
        + lookahead_keys
            .iter()
            .filter(|(key, _)| file.get(key).is_some())
            .count()
        + usize::from(file.get(KEY_PREEMPHASIS).is_some())
        + usize::from(file.get(KEY_ENCODER_ACT).is_some())
        + usize::from(file.get(KEY_JOINT_ACT).is_some());
    if present_count == 0 {
        // Narrow legacy exception: StrictCheckpoint has already authenticated
        // the exact public 655-tensor manifest and immutable model identity.
        return Ok(());
    }
    if present_count != expected_count {
        return Err(VokraError::ModelLoad(format!(
            "{LABEL}: partial `vokra.nemotron_asr.*` metadata group ({present_count}/{expected_count} keys); reconvert from the pinned release instead of mixing legacy and self-describing metadata"
        )));
    }
    for &(key, expected) in CONFIG_U32 {
        require_u32(file, key, expected)?;
    }
    for (key, expected) in &lookahead_keys {
        require_u32(file, key, *expected)?;
    }
    match file.get(KEY_PREEMPHASIS) {
        Some(GgufMetadataValue::F32(value)) if value.to_bits() == 0.97f32.to_bits() => {}
        Some(other) => {
            return Err(VokraError::ModelLoad(format!(
                "{LABEL}: `{KEY_PREEMPHASIS}` must be f32 0.97, found {other:?}"
            )));
        }
        None => unreachable!("complete-group count checked above"),
    }
    require_string(file, KEY_ENCODER_ACT, "silu")?;
    require_string(file, KEY_JOINT_ACT, "relu")?;
    Ok(())
}

fn require_u32(file: &GgufFile, key: &str, expected: u32) -> Result<()> {
    match file.get(key) {
        Some(GgufMetadataValue::U32(value)) if *value == expected => Ok(()),
        Some(other) => Err(VokraError::ModelLoad(format!(
            "{LABEL}: `{key}` must be u32 {expected}, found {other:?}"
        ))),
        None => Err(VokraError::ModelLoad(format!(
            "{LABEL}: missing required metadata `{key}`"
        ))),
    }
}

fn require_string(file: &GgufFile, key: &str, expected: &str) -> Result<()> {
    match file.get(key) {
        Some(GgufMetadataValue::String(value)) if value == expected => Ok(()),
        Some(other) => Err(VokraError::ModelLoad(format!(
            "{LABEL}: `{key}` must be {expected:?}, found {other:?}"
        ))),
        None => Err(VokraError::ModelLoad(format!(
            "{LABEL}: missing required metadata `{key}`"
        ))),
    }
}

fn load_bound_weights(file: &GgufFile, config: &NemotronAsrConfig) -> Result<NemotronBoundWeights> {
    let enc = &config.encoder;
    let dec = config.decoder_hidden_size;
    let channels = enc.subsampling_conv_channels;
    let kernel = enc.subsampling_conv_kernel_size;
    let tensor = |name: &str, shape: &[usize]| load_tensor(file, LABEL, name, shape);

    let subsampling = ParakeetBoundSubsampling {
        conv0_w: tensor(
            "encoder.subsampling.conv_in.weight",
            &[channels, 1, kernel, kernel],
        )?,
        conv0_b: tensor("encoder.subsampling.conv_in.bias", &[channels])?,
        depthwise_w: [
            tensor(
                "encoder.subsampling.layers.0.depthwise_conv.weight",
                &[channels, 1, kernel, kernel],
            )?,
            tensor(
                "encoder.subsampling.layers.1.depthwise_conv.weight",
                &[channels, 1, kernel, kernel],
            )?,
        ],
        depthwise_b: [
            tensor(
                "encoder.subsampling.layers.0.depthwise_conv.bias",
                &[channels],
            )?,
            tensor(
                "encoder.subsampling.layers.1.depthwise_conv.bias",
                &[channels],
            )?,
        ],
        pointwise_w_t: [
            transpose_out_in(
                tensor(
                    "encoder.subsampling.layers.0.pointwise_conv.weight",
                    &[channels, channels, 1, 1],
                )?,
                channels,
                channels,
            ),
            transpose_out_in(
                tensor(
                    "encoder.subsampling.layers.1.pointwise_conv.weight",
                    &[channels, channels, 1, 1],
                )?,
                channels,
                channels,
            ),
        ],
        pointwise_b: [
            tensor(
                "encoder.subsampling.layers.0.pointwise_conv.bias",
                &[channels],
            )?,
            tensor(
                "encoder.subsampling.layers.1.pointwise_conv.bias",
                &[channels],
            )?,
        ],
        linear_w_t: transpose_out_in(
            tensor(
                "encoder.subsampling.linear.weight",
                &[enc.d_model, channels * 17],
            )?,
            enc.d_model,
            channels * 17,
        ),
        linear_b: tensor("encoder.subsampling.linear.bias", &[enc.d_model])?,
    };

    let norm = |prefix: &str, name: &str| -> Result<ParakeetBoundNorm> {
        Ok(ParakeetBoundNorm {
            weight: tensor(&format!("{prefix}.{name}.weight"), &[enc.d_model])?,
            bias: tensor(&format!("{prefix}.{name}.bias"), &[enc.d_model])?,
        })
    };
    let mut encoder = Vec::with_capacity(enc.n_layer);
    for layer in 0..enc.n_layer {
        let prefix = format!("encoder.layers.{layer}");
        let ff = |branch: &str, linear: usize, output: usize, input: usize| {
            tensor(
                &format!("{prefix}.{branch}.linear{linear}.weight"),
                &[output, input],
            )
            .map(|weight| transpose_out_in(weight, output, input))
        };
        let projection = |name: &str| {
            tensor(
                &format!("{prefix}.self_attn.{name}.weight"),
                &[enc.d_model, enc.d_model],
            )
            .map(|weight| transpose_out_in(weight, enc.d_model, enc.d_model))
        };
        encoder.push(ParakeetBoundEncoderBlock {
            ff1_w1_t: ff("feed_forward1", 1, enc.ffn_dim, enc.d_model)?,
            ff1_b1: None,
            ff1_w2_t: ff("feed_forward1", 2, enc.d_model, enc.ffn_dim)?,
            ff1_b2: None,
            ff2_w1_t: ff("feed_forward2", 1, enc.ffn_dim, enc.d_model)?,
            ff2_b1: None,
            ff2_w2_t: ff("feed_forward2", 2, enc.d_model, enc.ffn_dim)?,
            ff2_b2: None,
            norm_ff1: norm(&prefix, "norm_feed_forward1")?,
            norm_attn: norm(&prefix, "norm_self_att")?,
            norm_conv: norm(&prefix, "norm_conv")?,
            norm_ff2: norm(&prefix, "norm_feed_forward2")?,
            norm_out: norm(&prefix, "norm_out")?,
            q_w_t: projection("q_proj")?,
            q_b: None,
            k_w_t: projection("k_proj")?,
            k_b: None,
            v_w_t: projection("v_proj")?,
            v_b: None,
            o_w_t: projection("o_proj")?,
            o_b: None,
            relative_k_w_t: projection("relative_k_proj")?,
            bias_u: tensor(
                &format!("{prefix}.self_attn.bias_u"),
                &[enc.n_head, enc.head_dim()],
            )?,
            bias_v: tensor(
                &format!("{prefix}.self_attn.bias_v"),
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
            conv_pw1_b: None,
            conv_dw_w: tensor(
                &format!("{prefix}.conv.depthwise_conv.weight"),
                &[enc.d_model, 1, enc.conv_kernel_size],
            )?,
            conv_dw_b: None,
            conv_inner_norm: FastConformerConvNorm::LayerNorm(norm(&prefix, "conv.norm")?),
            conv_pw2_w_t: transpose_out_in(
                tensor(
                    &format!("{prefix}.conv.pointwise_conv2.weight"),
                    &[enc.d_model, enc.d_model, 1],
                )?,
                enc.d_model,
                enc.d_model,
            ),
            conv_pw2_b: None,
        });
    }

    let mut lstm = Vec::with_capacity(config.num_decoder_layers);
    for layer in 0..config.num_decoder_layers {
        lstm.push(ParakeetBoundLstmLayer {
            w_ih: tensor(&format!("decoder.lstm.weight_ih_l{layer}"), &[4 * dec, dec])?,
            w_hh: tensor(&format!("decoder.lstm.weight_hh_l{layer}"), &[4 * dec, dec])?,
            b_ih: tensor(&format!("decoder.lstm.bias_ih_l{layer}"), &[4 * dec])?,
            b_hh: tensor(&format!("decoder.lstm.bias_hh_l{layer}"), &[4 * dec])?,
        });
    }
    let prompt_input = enc.d_model + config.num_prompts;
    Ok(NemotronBoundWeights {
        subsampling,
        encoder,
        prompt_w1_t: transpose_out_in(
            tensor(
                "prompt_projector.linear_1.weight",
                &[config.prompt_intermediate_size, prompt_input],
            )?,
            config.prompt_intermediate_size,
            prompt_input,
        ),
        prompt_b1: tensor(
            "prompt_projector.linear_1.bias",
            &[config.prompt_intermediate_size],
        )?,
        prompt_w2_t: transpose_out_in(
            tensor(
                "prompt_projector.linear_2.weight",
                &[enc.d_model, config.prompt_intermediate_size],
            )?,
            enc.d_model,
            config.prompt_intermediate_size,
        ),
        prompt_b2: tensor("prompt_projector.linear_2.bias", &[enc.d_model])?,
        encoder_projector_w_t: transpose_out_in(
            tensor("encoder_projector.weight", &[dec, enc.d_model])?,
            dec,
            enc.d_model,
        ),
        encoder_projector_b: tensor("encoder_projector.bias", &[dec])?,
        embedding: tensor("decoder.embedding.weight", &[config.vocab_size, dec])?,
        lstm,
        decoder_projector_w: tensor("decoder.decoder_projector.weight", &[dec, dec])?,
        decoder_projector_b: tensor("decoder.decoder_projector.bias", &[dec])?,
        joint_head_w: tensor("joint.head.weight", &[config.vocab_size, dec])?,
        joint_head_b: tensor("joint.head.bias", &[config.vocab_size])?,
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
    weights: &NemotronBoundWeights,
    config: &NemotronAsrConfig,
    state: &mut DecoderState,
) -> Result<()> {
    if token as usize >= config.vocab_size {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: decoder token {token} outside 0..{}",
            config.vocab_size
        )));
    }
    let width = config.decoder_hidden_size;
    let offset = token as usize * width;
    let mut input = weights.embedding[offset..offset + width].to_vec();
    for (layer_index, layer) in weights.lstm.iter().enumerate() {
        let mut input_gates = vec![0.0f32; 4 * width];
        compute.gemv_f32(
            4 * width,
            width,
            &layer.w_ih,
            &input,
            Some(&layer.b_ih),
            &mut input_gates,
        )?;
        let mut recurrent_gates = vec![0.0f32; 4 * width];
        compute.gemv_f32(
            4 * width,
            width,
            &layer.w_hh,
            &state.hidden[layer_index],
            Some(&layer.b_hh),
            &mut recurrent_gates,
        )?;
        let mut next = vec![0.0f32; width];
        for index in 0..width {
            let input_gate = sigmoid(input_gates[index] + recurrent_gates[index]);
            let forget_gate = sigmoid(input_gates[width + index] + recurrent_gates[width + index]);
            let candidate =
                (input_gates[2 * width + index] + recurrent_gates[2 * width + index]).tanh();
            let output_gate =
                sigmoid(input_gates[3 * width + index] + recurrent_gates[3 * width + index]);
            let cell = forget_gate * state.cell[layer_index][index] + input_gate * candidate;
            state.cell[layer_index][index] = cell;
            next[index] = output_gate * cell.tanh();
        }
        state.hidden[layer_index].copy_from_slice(&next);
        input = next;
    }
    compute.gemv_f32(
        width,
        width,
        &weights.decoder_projector_w,
        &input,
        Some(&weights.decoder_projector_b),
        &mut state.projected,
    )
}

fn greedy_rnnt_decode(
    compute: &Compute,
    encoded: &[f32],
    frames: usize,
    weights: &NemotronBoundWeights,
    config: &NemotronAsrConfig,
) -> Result<Vec<u32>> {
    let width = config.decoder_hidden_size;
    if encoded.len() != frames * width {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: projected encoder buffer has {} values, expected {} x {width}",
            encoded.len(),
            frames
        )));
    }
    let mut state = DecoderState::new(config.num_decoder_layers, width);
    // The official generation path prepends decoder_start_token_id=blank and
    // runs it once to initialize the LSTM cache.
    decoder_step(compute, config.blank_token_id, weights, config, &mut state)?;
    let capacity = frames
        .checked_mul(config.max_symbols_per_step)
        .ok_or_else(|| VokraError::InvalidArgument(format!("{LABEL}: token capacity overflow")))?;
    let mut tokens = Vec::with_capacity(capacity);
    let mut frame = 0usize;
    let mut symbols_at_frame = 0usize;
    let mut joint = vec![0.0f32; width];
    let mut logits = vec![0.0f32; config.vocab_size];
    while frame < frames {
        let encoder_row = &encoded[frame * width..(frame + 1) * width];
        for index in 0..width {
            joint[index] = (encoder_row[index] + state.projected[index]).max(0.0);
        }
        compute.gemv_f32(
            config.vocab_size,
            width,
            &weights.joint_head_w,
            &joint,
            Some(&weights.joint_head_b),
            &mut logits,
        )?;
        let token = argmax_finite(&logits)? as u32;
        if token == config.blank_token_id {
            frame += 1;
            symbols_at_frame = 0;
            continue;
        }
        tokens.push(token);
        decoder_step(compute, token, weights, config, &mut state)?;
        symbols_at_frame += 1;
        if symbols_at_frame >= config.max_symbols_per_step {
            frame += 1;
            symbols_at_frame = 0;
        }
    }
    Ok(tokens)
}

fn argmax_finite(values: &[f32]) -> Result<usize> {
    let mut best = None;
    for (index, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(VokraError::InvalidArgument(format!(
                "{LABEL}: non-finite joint logit at index {index}: {value}"
            )));
        }
        if best.is_none_or(|(_, current)| value > current) {
            best = Some((index, value));
        }
    }
    best.map(|(index, _)| index)
        .ok_or_else(|| VokraError::InvalidArgument(format!("{LABEL}: empty joint logits")))
}

fn nemotron_logmel(pcm: &[f32]) -> Result<(Vec<f32>, usize)> {
    const N_FFT: usize = 512;
    const HOP: usize = 160;
    const WIN: usize = 400;
    const N_MELS: usize = 128;
    const PREEMPHASIS: f32 = 0.97;
    const LOG_ZERO_GUARD: f32 = 1.0 / 16_777_216.0;

    if pcm.is_empty() {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: PCM input is empty"
        )));
    }
    let frames = pcm.len() / HOP;
    if frames == 0 {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: PCM has {} samples; at least {HOP} are required",
            pcm.len()
        )));
    }
    let mut emphasized = vec![0.0f32; pcm.len()];
    emphasized[0] = pcm[0];
    for index in 1..pcm.len() {
        emphasized[index] = pcm[index] - PREEMPHASIS * pcm[index - 1];
    }
    let spectrum = stft(
        &emphasized,
        &StftAttrs {
            n_fft: N_FFT,
            hop_length: HOP,
            win_length: WIN,
            window: Window::Hann,
            window_symmetry: WindowSymmetry::Symmetric,
            center: true,
            pad_mode: PadMode::Constant,
            normalization: Normalization::Backward,
            causal: false,
            real_input: true,
        },
    )?;
    if spectrum.frames < frames {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: STFT returned {} frames, fewer than valid frame count {frames}",
            spectrum.frames
        )));
    }
    let bins = N_FFT / 2 + 1;
    let mut power = vec![0.0f32; frames * bins];
    for (index, value) in power.iter_mut().enumerate() {
        *value = spectrum.re[index] * spectrum.re[index] + spectrum.im[index] * spectrum.im[index];
    }
    let mel = MelFilterbank::new(&MelAttrs::new(SAMPLE_RATE, N_FFT, N_MELS));
    let mut features = mel.apply(&power, frames);
    for value in &mut features {
        *value = (*value + LOG_ZERO_GUARD).ln();
    }
    // Nemotron's official feature extractor explicitly does not normalize
    // mel features. Parakeet's per-feature normalization is not reused here.
    Ok((features, frames))
}

#[must_use]
pub fn prompt_id_for_language(language: &str) -> Option<u32> {
    Some(match language {
        "en-US" | "en" => 0,
        "en-GB" | "enGB" => 1,
        "es-ES" | "esES" => 2,
        "es-US" | "es" => 3,
        "zh-CN" | "zh-ZH" => 4,
        "zh-TW" => 5,
        "hi-IN" | "hi" | "hi-HI" => 6,
        "ar-AR" | "ar" => 7,
        "fr-FR" | "fr" => 8,
        "de-DE" | "de" => 9,
        "ja-JP" | "ja-JA" => 10,
        "ru-RU" | "ru" => 11,
        "pt-BR" => 12,
        "pt-PT" | "pt" => 13,
        "ko-KR" | "ko" | "ko-KO" => 14,
        "it-IT" | "it" => 15,
        "nl-NL" | "nl" => 16,
        "pl-PL" | "pl" => 17,
        "tr-TR" | "tr" => 18,
        "uk-UA" | "uk" => 19,
        "ro-RO" | "ro" => 20,
        "el-GR" | "el" => 21,
        "cs-CZ" | "cs" => 22,
        "hu-HU" | "hu" => 23,
        "sv-SE" | "sv" => 24,
        "da-DK" | "da" => 25,
        "fi-FI" | "fi" => 26,
        "no-NO" | "no" => 27,
        "sk-SK" | "sk" => 28,
        "hr-HR" | "hr" => 29,
        "bg-BG" | "bg" => 30,
        "lt-LT" | "lt" => 31,
        "th-TH" => 32,
        "vi-VN" => 33,
        "id-ID" => 34,
        "ms-MY" => 35,
        "bn-IN" => 36,
        "ur-PK" => 37,
        "fa-IR" => 38,
        "ta-IN" => 39,
        "te-IN" => 40,
        "mr-IN" => 41,
        "gu-IN" => 42,
        "kn-IN" => 43,
        "ml-IN" => 44,
        "si-LK" => 45,
        "ne-NP" => 46,
        "km-KH" => 47,
        "sw-KE" => 48,
        "am-ET" => 49,
        "ha-NG" => 50,
        "zu-ZA" => 51,
        "yo-NG" => 52,
        "ig-NG" => 53,
        "af-ZA" => 54,
        "rw-RW" => 55,
        "so-SO" => 56,
        "ny-MW" => 57,
        "ln-CD" => 58,
        "or-KE" => 59,
        "et-EE" | "et" => 60,
        "lv-LV" | "lv" => 61,
        "sl-SI" | "sl" => 62,
        "he-IL" => 64,
        "ku-TR" => 65,
        "az-AZ" => 66,
        "ka-GE" => 67,
        "hy-AM" => 68,
        "uz-UZ" => 69,
        "tg-TJ" => 70,
        "ky-KG" => 71,
        "qu-PE" => 80,
        "ay-BO" => 81,
        "gn-PY" => 82,
        "nah-MX" => 83,
        "mi-NZ" => 96,
        "haw-US" => 97,
        "sm-WS" => 98,
        "to-TO" => 99,
        "fr-CA" => 100,
        "auto" => 101,
        "mt-MT" => 102,
        "nb-NO" | "nb" => 103,
        "nn-NO" | "nn" => 104,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_config_matches_released_axes() {
        let config = NemotronAsrConfig::canonical();
        config.validate().expect("canonical config");
        assert_eq!(config.encoder.n_layer, 24);
        assert_eq!(config.encoder.d_model, 1_024);
        assert_eq!(config.encoder.in_dim, 128);
        assert_eq!(config.decoder_hidden_size, 640);
        assert_eq!(config.vocab_size, 13_088);
        assert_eq!(config.blank_token_id, 13_087);
        assert_eq!(config.default_prompt_id, 101);
        assert_eq!(config.sliding_window - 1, 56);
        assert_eq!(config.default_lookahead_tokens, 3);
    }

    #[test]
    fn language_prompt_map_pins_aliases_and_sparse_slots() {
        assert_eq!(prompt_id_for_language("en-US"), Some(0));
        assert_eq!(prompt_id_for_language("ja-JP"), Some(10));
        assert_eq!(prompt_id_for_language("auto"), Some(101));
        assert_eq!(prompt_id_for_language("nb-NO"), Some(103));
        assert_eq!(prompt_id_for_language("nn"), Some(104));
        assert_eq!(prompt_id_for_language("xx-XX"), None);
    }

    #[test]
    fn public_manifest_identity_is_pinned() {
        assert_eq!(TENSOR_COUNT, 655);
        assert_eq!(MANIFEST_SHA256.len(), 32);
        assert_eq!(EXPECTED_ARCH, "nemotron_asr_streaming");
        assert_eq!(MODEL_NAME, "nemotron-3.5-asr-streaming-0.6b");
    }
}
