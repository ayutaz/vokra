//! Strict native Data2Vec Audio Base 960h CTC runtime.
//!
//! The first public Vokra artifact was incorrectly stamped as Wav2Vec2.
//! Its tensors prove a distinct contract: `data2vec_audio.*` names,
//! LayerNorm on every feature-extractor convolution, and five grouped
//! positional convolutions with kernel 19 and parameter-free LayerNorm.
//! This binder accepts that one exact legacy stamp, reports the repair,
//! and also accepts the corrected `data2vec_audio` metadata emitted by
//! the dedicated converter.

use std::path::Path;

use vokra_core::backend::BackendKind;
use vokra_core::engines::AsrEngine;
use vokra_core::gguf::{GgufFile, GgufMetadataValue, chunks};
use vokra_core::tasks::Transcription;
use vokra_core::{Result, VokraError};
use vokra_ops::{
    ConvLayerAttrs, ConvLayerWeights, Norm, WaveformFrontendAttrs, WaveformFrontendWeights,
    ctc_decode_greedy,
};

use crate::align::charsiu::{
    CharsiuBlock, CharsiuConfig, CharsiuFeatureProjection, CharsiuHead, CharsiuPosConv,
    feature_projection_forward_with_compute, layer_norm_with_compute_inplace,
    linear_forward_with_compute, transformer_block_forward_with_valid_keys_and_compute,
};
use crate::compute::{Compute, HotOp};
use crate::wav2vec2_ctc::{
    bind_family_block, decode_tokens, parse_vocab, reject_non_finite, tensor,
    transpose_channel_to_frame, transpose_frame_to_channel, validate_pcm,
    waveform_frontend_with_compute, zero_mean_unit_var,
};

/// Corrected public architecture tag.
pub const ARCH: &str = "data2vec_audio";

/// Complete learned-op registry for CPU/Metal inference.
pub const DATA2VEC_AUDIO_HOT_OPS: &[HotOp] = &[
    HotOp::Gemm,
    HotOp::Softmax,
    HotOp::LayerNorm,
    HotOp::Gelu,
    HotOp::Conv1d,
    HotOp::GroupedConv1d,
];

const LEGACY_ARCH: &str = "wav2vec2_ctc";
const MODEL_ID: &str = "data2vec-audio-base-960h";
const UPSTREAM_HF: &str = "facebook/data2vec-audio-base-960h";
const SAMPLE_RATE: u32 = 16_000;
const HIDDEN: usize = 768;
const LAYERS: usize = 12;
const HEADS: usize = 12;
const FFN: usize = 3072;
const VOCAB: usize = 32;
const FEATURE_DIM: usize = 512;
const POSITION_LAYERS: usize = 5;
const POSITION_KERNEL: usize = 19;
const POSITION_GROUPS: usize = 16;
const LAYER_NORM_EPS: f32 = 1e-5;
const TENSOR_COUNT: usize = 232;
const BLANK_ID: usize = 0;
const CONV_DIM: [usize; 7] = [512; 7];
const CONV_KERNEL: [usize; 7] = [10, 3, 3, 3, 3, 2, 2];
const CONV_STRIDE: [usize; 7] = [5, 2, 2, 2, 2, 2, 2];
const ENGLISH_VOCAB: &[u8] = include_bytes!("../../resources/wav2vec2_ctc/english.json");

const KEY_MODEL_ID: &str = "vokra.provenance.model_id";
const KEY_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";
const KEY_HIDDEN: &str = "vokra.data2vec_audio.hidden_size";
const KEY_LAYERS: &str = "vokra.data2vec_audio.n_layer";
const KEY_HEADS: &str = "vokra.data2vec_audio.n_head";
const KEY_FFN: &str = "vokra.data2vec_audio.intermediate_size";
const KEY_VOCAB: &str = "vokra.data2vec_audio.vocab_size";
const KEY_POS_LAYERS: &str = "vokra.data2vec_audio.num_conv_pos_embeddings";
const KEY_POS_KERNEL: &str = "vokra.data2vec_audio.conv_pos_kernel_size";
const KEY_POS_GROUPS: &str = "vokra.data2vec_audio.num_conv_pos_embedding_groups";

/// Fixed, audited Data2Vec Audio inference configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Data2VecAudioConfig {
    /// Stable public artifact id.
    pub model_id: String,
    /// Transformer width.
    pub hidden_size: usize,
    /// Transformer block count.
    pub n_layer: usize,
    /// Attention head count.
    pub n_head: usize,
    /// Feed-forward inner width.
    pub intermediate_size: usize,
    /// CTC vocabulary width.
    pub vocab_size: usize,
    /// Required PCM sample rate.
    pub sample_rate: u32,
}

#[derive(Debug, Clone)]
struct Data2VecAudioWeights {
    stem_attrs: WaveformFrontendAttrs,
    stem: WaveformFrontendWeights,
    projection: CharsiuFeatureProjection,
    position: Vec<CharsiuPosConv>,
    encoder_norm_gamma: Vec<f32>,
    encoder_norm_beta: Vec<f32>,
    blocks: Vec<CharsiuBlock>,
    head: CharsiuHead,
}

/// Native `facebook/data2vec-audio-base-960h` CTC engine.
#[derive(Debug, Clone)]
pub struct Data2VecAudioCtc {
    config: Data2VecAudioConfig,
    weights: Data2VecAudioWeights,
    vocab: Vec<String>,
    backend: BackendKind,
    legacy_metadata_repaired: bool,
}

impl Data2VecAudioCtc {
    /// Opens and strictly binds a public/corrected GGUF.
    pub fn from_gguf(path: impl AsRef<Path>) -> Result<Self> {
        let file = GgufFile::open(path)?;
        Self::from_file(&file)
    }

    /// Strictly binds an already-open GGUF.
    pub fn from_file(file: &GgufFile) -> Result<Self> {
        let arch = meta_string(file, chunks::KEY_MODEL_ARCH)?;
        let legacy_metadata_repaired = match arch {
            ARCH => {
                validate_corrected_metadata(file)?;
                false
            }
            LEGACY_ARCH => {
                validate_exact_legacy_metadata(file)?;
                true
            }
            other => {
                return Err(VokraError::ModelLoad(format!(
                    "data2vec_audio: expected arch {ARCH:?} or audited legacy {LEGACY_ARCH:?}, got {other:?}"
                )));
            }
        };
        if file.tensors().len() != TENSOR_COUNT {
            return Err(VokraError::ModelLoad(format!(
                "data2vec_audio: {MODEL_ID} has {} tensors, expected exactly {TENSOR_COUNT}",
                file.tensors().len()
            )));
        }

        let stem_attrs = WaveformFrontendAttrs {
            in_channels: 1,
            layers: CONV_DIM
                .iter()
                .zip(CONV_KERNEL)
                .zip(CONV_STRIDE)
                .map(|((&out_channels, kernel), stride)| ConvLayerAttrs {
                    out_channels,
                    kernel,
                    stride,
                })
                .collect(),
            norm: Norm::LayerAll,
            conv_bias: false,
        };
        let mut stem_layers = Vec::with_capacity(CONV_DIM.len());
        let mut in_channels = 1usize;
        for layer in 0..CONV_DIM.len() {
            let prefix = format!("data2vec_audio.feature_extractor.conv_layers.{layer}");
            stem_layers.push(ConvLayerWeights {
                conv_w: tensor(
                    file,
                    &format!("{prefix}.conv.weight"),
                    &[CONV_DIM[layer], in_channels, CONV_KERNEL[layer]],
                )?,
                conv_b: Vec::new(),
                norm_gamma: Some(tensor(
                    file,
                    &format!("{prefix}.layer_norm.weight"),
                    &[FEATURE_DIM],
                )?),
                norm_beta: Some(tensor(
                    file,
                    &format!("{prefix}.layer_norm.bias"),
                    &[FEATURE_DIM],
                )?),
            });
            in_channels = CONV_DIM[layer];
        }
        let stem = WaveformFrontendWeights {
            layers: stem_layers,
        };
        stem.validate(&stem_attrs)?;

        let projection = CharsiuFeatureProjection {
            norm_gamma: Some(tensor(
                file,
                "data2vec_audio.feature_projection.layer_norm.weight",
                &[FEATURE_DIM],
            )?),
            norm_beta: Some(tensor(
                file,
                "data2vec_audio.feature_projection.layer_norm.bias",
                &[FEATURE_DIM],
            )?),
            linear_w: tensor(
                file,
                "data2vec_audio.feature_projection.projection.weight",
                &[HIDDEN, FEATURE_DIM],
            )?,
            linear_b: tensor(
                file,
                "data2vec_audio.feature_projection.projection.bias",
                &[HIDDEN],
            )?,
        };
        let mut position = Vec::with_capacity(POSITION_LAYERS);
        for layer in 0..POSITION_LAYERS {
            let prefix = format!("data2vec_audio.encoder.pos_conv_embed.layers.{layer}.conv");
            position.push(CharsiuPosConv {
                weight: tensor(
                    file,
                    &format!("{prefix}.weight"),
                    &[HIDDEN, HIDDEN / POSITION_GROUPS, POSITION_KERNEL],
                )?,
                bias: tensor(file, &format!("{prefix}.bias"), &[HIDDEN])?,
            });
        }
        let encoder_norm_gamma =
            tensor(file, "data2vec_audio.encoder.layer_norm.weight", &[HIDDEN])?;
        let encoder_norm_beta = tensor(file, "data2vec_audio.encoder.layer_norm.bias", &[HIDDEN])?;
        let mut blocks = Vec::with_capacity(LAYERS);
        for layer in 0..LAYERS {
            blocks.push(bind_family_block(
                file,
                &format!("data2vec_audio.encoder.layers.{layer}"),
                HIDDEN,
                FFN,
            )?);
        }
        let head = CharsiuHead {
            weight: tensor(file, "lm_head.weight", &[VOCAB, HIDDEN])?,
            bias: tensor(file, "lm_head.bias", &[VOCAB])?,
        };
        let vocab = parse_vocab(ENGLISH_VOCAB, VOCAB)?;

        Ok(Self {
            config: Data2VecAudioConfig {
                model_id: MODEL_ID.to_owned(),
                hidden_size: HIDDEN,
                n_layer: LAYERS,
                n_head: HEADS,
                intermediate_size: FFN,
                vocab_size: VOCAB,
                sample_rate: SAMPLE_RATE,
            },
            weights: Data2VecAudioWeights {
                stem_attrs,
                stem,
                projection,
                position,
                encoder_norm_gamma,
                encoder_norm_beta,
                blocks,
                head,
            },
            vocab,
            backend: BackendKind::Cpu,
            legacy_metadata_repaired,
        })
    }

    /// Selects the backend used by every learned operation.
    #[must_use]
    pub fn with_backend(mut self, backend: BackendKind) -> Self {
        self.backend = backend;
        self
    }

    /// Resolved fixed topology.
    pub fn config(&self) -> &Data2VecAudioConfig {
        &self.config
    }

    /// Selected compute backend.
    pub fn backend(&self) -> BackendKind {
        self.backend
    }

    /// Whether the exact audited Wav2Vec2 mis-stamp was repaired.
    pub fn legacy_metadata_repaired(&self) -> bool {
        self.legacy_metadata_repaired
    }

    /// Runs the complete waveform stem and Data2Vec encoder.
    pub fn encode_features(&self, pcm: &[f32]) -> Result<(Vec<f32>, usize)> {
        validate_pcm(pcm)?;
        let compute = Compute::for_backend(self.backend, DATA2VEC_AUDIO_HOT_OPS)?;
        self.encode_with_compute(pcm, &compute)
    }

    /// Runs encoder plus the 32-way CTC head.
    pub fn logits(&self, pcm: &[f32]) -> Result<(Vec<f32>, usize)> {
        let compute = Compute::for_backend(self.backend, DATA2VEC_AUDIO_HOT_OPS)?;
        validate_pcm(pcm)?;
        let (features, frames) = self.encode_with_compute(pcm, &compute)?;
        let logits = linear_forward_with_compute(
            &features,
            frames,
            HIDDEN,
            &self.weights.head.weight,
            &self.weights.head.bias,
            VOCAB,
            &compute,
        )?;
        reject_non_finite("Data2Vec logits", &logits)?;
        Ok((logits, frames))
    }

    /// Returns greedy blank/repeat-folded CTC ids.
    pub fn transcribe_tokens(&self, pcm: &[f32]) -> Result<Vec<u32>> {
        let (logits, frames) = self.logits(pcm)?;
        ctc_decode_greedy(&logits, frames, VOCAB, BLANK_ID)
    }

    /// Decodes complete PCM to text with the official vocabulary.
    pub fn transcribe_text(&self, pcm: &[f32]) -> Result<String> {
        decode_tokens(&self.transcribe_tokens(pcm)?, &self.vocab)
    }

    fn encode_with_compute(&self, pcm: &[f32], compute: &Compute) -> Result<(Vec<f32>, usize)> {
        let normalized = zero_mean_unit_var(pcm);
        let features = waveform_frontend_with_compute(
            &normalized,
            &self.weights.stem_attrs,
            &self.weights.stem,
            compute,
        )?;
        let frames = features.len() / FEATURE_DIM;
        let mut hidden = feature_projection_forward_with_compute(
            &features,
            frames,
            FEATURE_DIM,
            &self.weights.projection,
            HIDDEN,
            true,
            LAYER_NORM_EPS,
            compute,
        )?;
        let position = positional_stack(&hidden, frames, &self.weights.position, compute)?;
        for (value, positional) in hidden.iter_mut().zip(position) {
            *value += positional;
        }
        layer_norm_with_compute_inplace(
            &mut hidden,
            frames,
            HIDDEN,
            &self.weights.encoder_norm_gamma,
            &self.weights.encoder_norm_beta,
            LAYER_NORM_EPS,
            compute,
        )?;
        let cfg = encoder_config();
        for block in &self.weights.blocks {
            transformer_block_forward_with_valid_keys_and_compute(
                &mut hidden,
                frames,
                frames,
                &cfg,
                block,
                compute,
            )?;
        }
        reject_non_finite("Data2Vec encoder output", &hidden)?;
        Ok((hidden, frames))
    }
}

impl AsrEngine for Data2VecAudioCtc {
    fn transcribe(&self, pcm: &[f32]) -> Result<Transcription> {
        Ok(Transcription::new(self.transcribe_text(pcm)?))
    }

    fn backend(&self) -> BackendKind {
        self.backend
    }
}

fn positional_stack(
    hidden: &[f32],
    frames: usize,
    layers: &[CharsiuPosConv],
    compute: &Compute,
) -> Result<Vec<f32>> {
    let mut channel_major = transpose_frame_to_channel(hidden, frames, HIDDEN);
    let gamma = vec![1.0f32; HIDDEN];
    let beta = vec![0.0f32; HIDDEN];
    for layer in layers {
        let mut convolution = vec![0.0f32; HIDDEN * frames];
        compute.grouped_conv1d_f32(
            &channel_major,
            HIDDEN,
            frames,
            &layer.weight,
            HIDDEN,
            POSITION_KERNEL,
            Some(&layer.bias),
            1,
            POSITION_KERNEL / 2,
            POSITION_GROUPS,
            &mut convolution,
        )?;
        let mut frame_major = transpose_channel_to_frame(&convolution, HIDDEN, frames);
        layer_norm_with_compute_inplace(
            &mut frame_major,
            frames,
            HIDDEN,
            &gamma,
            &beta,
            LAYER_NORM_EPS,
            compute,
        )?;
        channel_major = transpose_frame_to_channel(&frame_major, frames, HIDDEN);
        let mut activated = vec![0.0f32; channel_major.len()];
        compute.gelu_f32(&channel_major, &mut activated)?;
        channel_major = activated;
    }
    Ok(transpose_channel_to_frame(&channel_major, HIDDEN, frames))
}

fn encoder_config() -> CharsiuConfig {
    CharsiuConfig {
        hidden_size: HIDDEN,
        n_layer: LAYERS,
        n_head: HEADS,
        ffn_dim: FFN,
        vocab_size: VOCAB,
        silence_id: 0,
        pad_id: 0,
        sample_rate: SAMPLE_RATE,
        frame_shift_sec: 0.02,
        layer_norm_eps: LAYER_NORM_EPS,
        pos_conv_kernel: POSITION_KERNEL,
        pos_conv_groups: POSITION_GROUPS,
        silence_threshold: 1,
        feature_projection_has_layer_norm: true,
        stem_conv_bias: false,
    }
}

fn validate_corrected_metadata(file: &GgufFile) -> Result<()> {
    if meta_is_string(file, chunks::KEY_MODEL_NAME, MODEL_ID)
        && meta_is_string(file, KEY_MODEL_ID, MODEL_ID)
        && meta_is_string(file, KEY_UPSTREAM_HF, UPSTREAM_HF)
        && meta_is_u64(file, KEY_HIDDEN, HIDDEN as u64)
        && meta_is_u64(file, KEY_LAYERS, LAYERS as u64)
        && meta_is_u64(file, KEY_HEADS, HEADS as u64)
        && meta_is_u64(file, KEY_FFN, FFN as u64)
        && meta_is_u64(file, KEY_VOCAB, VOCAB as u64)
        && meta_is_u64(file, KEY_POS_LAYERS, POSITION_LAYERS as u64)
        && meta_is_u64(file, KEY_POS_KERNEL, POSITION_KERNEL as u64)
        && meta_is_u64(file, KEY_POS_GROUPS, POSITION_GROUPS as u64)
    {
        return Ok(());
    }
    Err(VokraError::ModelLoad(
        "data2vec_audio: corrected metadata does not match the audited base-960h topology"
            .to_owned(),
    ))
}

fn validate_exact_legacy_metadata(file: &GgufFile) -> Result<()> {
    if meta_is_string(file, chunks::KEY_MODEL_NAME, MODEL_ID)
        && meta_is_string(file, KEY_MODEL_ID, MODEL_ID)
        && meta_is_string(file, KEY_UPSTREAM_HF, UPSTREAM_HF)
        && meta_is_u64(file, "vokra.wav2vec2_ctc.hidden_size", HIDDEN as u64)
        && meta_is_u64(file, "vokra.wav2vec2_ctc.n_layer", LAYERS as u64)
        && meta_is_u64(file, "vokra.wav2vec2_ctc.n_head", HEADS as u64)
        && meta_is_u64(file, "vokra.wav2vec2_ctc.intermediate_size", FFN as u64)
        && meta_is_u64(file, "vokra.wav2vec2_ctc.vocab_size", VOCAB as u64)
        && meta_is_u64(file, "vokra.wav2vec2_ctc.num_conv_pos_embeddings", 128)
        && meta_is_string(file, "vokra.wav2vec2_ctc.feat_extract_norm", "group")
        && meta_is_bool(file, "vokra.wav2vec2_ctc.do_stable_layer_norm", false)
    {
        return Ok(());
    }
    Err(VokraError::ModelLoad(
        "data2vec_audio: legacy arch stamp is not the exact audited public mis-stamp; refusing an unrecognised repair"
            .to_owned(),
    ))
}

fn meta_string<'a>(file: &'a GgufFile, key: &str) -> Result<&'a str> {
    file.get(key)
        .and_then(GgufMetadataValue::as_str)
        .ok_or_else(|| VokraError::ModelLoad(format!("data2vec_audio: missing string `{key}`")))
}

fn meta_is_string(file: &GgufFile, key: &str, expected: &str) -> bool {
    file.get(key).and_then(GgufMetadataValue::as_str) == Some(expected)
}

fn meta_is_u64(file: &GgufFile, key: &str, expected: u64) -> bool {
    file.get(key).and_then(GgufMetadataValue::as_u64) == Some(expected)
}

fn meta_is_bool(file: &GgufFile, key: &str, expected: bool) -> bool {
    file.get(key).and_then(GgufMetadataValue::as_bool) == Some(expected)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_axes_match_the_audited_public_checkpoint() {
        let cfg = encoder_config();
        assert_eq!((cfg.hidden_size, cfg.n_layer, cfg.n_head), (768, 12, 12));
        assert_eq!(
            (POSITION_LAYERS, POSITION_KERNEL, POSITION_GROUPS),
            (5, 19, 16)
        );
        assert_eq!(DATA2VEC_AUDIO_HOT_OPS.len(), 6);
    }

    #[test]
    fn official_english_vocab_is_dense() {
        let vocab = parse_vocab(ENGLISH_VOCAB, VOCAB).unwrap();
        assert_eq!(vocab.len(), 32);
        assert_eq!(vocab[0], "<pad>");
        assert_eq!(vocab[4], "|");
    }
}
