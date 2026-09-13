//! Strict structural binder for Microsoft VibeVoice-Realtime-0.5B.
//!
//! The Realtime release is not the 1.5B VibeVoice topology with a smaller
//! width.  It has a distinct `vibevoice_streaming` identity and two Qwen2
//! language-model stacks: a four-layer language prefix and a twenty-layer
//! TTS backbone.  This module binds that topology without importing or
//! executing the upstream Python implementation.
//!
//! The authenticated source packet records the complete checkpoint as BF16
//! with 1,017,626,724 parameters, but the exact tensor payload is intentionally
//! not carried in the maintainer checkout.  Consequently the binder performs
//! a strict descriptor/topology gate (names, layer coverage, shapes, dtypes,
//! and required metadata) while synthesis remains an explicit
//! `NotImplemented` boundary until the VAST real-weight manifest and parity
//! packet are available.  A synthetic fixture passing this gate is a contract
//! test only; it is not a model or numerical-parity claim.

use std::collections::BTreeSet;

use vokra_core::gguf::{GgmlType, GgufFile, GgufMetadataValue, GgufValueType, chunks};
use vokra_core::{Result, VokraError};

/// GGUF architecture tag for the streaming Realtime release.
pub const ARCH: &str = "vibevoice_streaming";
/// Canonical Vokra model name for the Realtime release.
pub const NAME: &str = "vibevoice-realtime-0.5b";
/// Authenticated upstream model identifier.
pub const UPSTREAM_HF: &str = "microsoft/VibeVoice-Realtime-0.5B";
/// Authenticated source revision used by the inspection packet.
pub const SOURCE_REVISION: &str = "94da20d98b2fa7688e9cbfaf7692ddb4954f7600";
/// Authenticated HF revision used by the inspection packet.
pub const MODEL_REVISION: &str = "6bce5f06044837fe6d2c5d7a71a84f0416bd57e4";

const FAMILY: &str = "vibevoice_streaming";
const HIDDEN: usize = 896;
const LANGUAGE_LAYERS: usize = 4;
const TTS_LAYERS: usize = 20;
const VOCAB: usize = 151_936;
const FFN: usize = 4_864;
const HEADS: usize = 14;
const KV_HEADS: usize = 2;
const MAX_POSITIONS: usize = 8_192;
const TTS_BACKBONE_LAYERS: usize = 20;
const ACOUSTIC_VAE_DIM: usize = 64;
const FRAME_RATE_HZ: f32 = 7.5;
const SAMPLE_RATE_HZ: usize = 24_000;

const KEY_FAMILY: &str = "vokra.vibevoice.model_family";
const KEY_ACOUSTIC_VAE_DIM: &str = "vokra.vibevoice.acoustic_vae_dim";
const KEY_LM_FRAME_RATE_HZ: &str = "vokra.vibevoice.lm_frame_rate_hz";
const KEY_TTS_BACKBONE_LAYERS: &str = "vokra.vibevoice.tts_backbone_num_hidden_layers";
const KEY_DECODER_HIDDEN: &str = "vokra.vibevoice.decoder.hidden_dim";
const KEY_DECODER_LAYERS: &str = "vokra.vibevoice.decoder.n_layer";
const KEY_DECODER_HEADS: &str = "vokra.vibevoice.decoder.n_head";
const KEY_DECODER_KV_HEADS: &str = "vokra.vibevoice.decoder.n_head_kv";
const KEY_DECODER_FFN: &str = "vokra.vibevoice.decoder.ffn_dim";
const KEY_DECODER_VOCAB: &str = "vokra.vibevoice.decoder.vocab_size";
const KEY_DECODER_MAX_POSITIONS: &str = "vokra.vibevoice.decoder.max_position_embeddings";
const KEY_DECODER_ROPE_BASE: &str = "vokra.vibevoice.decoder.rope_base";
const KEY_DECODER_RMS_EPS: &str = "vokra.vibevoice.decoder.rms_norm_eps";
const KEY_DECODER_DROPOUT: &str = "vokra.vibevoice.decoder.attention_dropout";
const KEY_DECODER_TIED: &str = "vokra.vibevoice.decoder.tie_word_embeddings";
const KEY_DECODER_SLIDING: &str = "vokra.vibevoice.decoder.use_sliding_window";
const KEY_DECODER_WINDOW_LAYERS: &str = "vokra.vibevoice.decoder.max_window_layers";
const KEY_ACOUSTIC_CHANNELS: &str = "vokra.vibevoice.acoustic.channels";
const KEY_ACOUSTIC_CAUSAL: &str = "vokra.vibevoice.acoustic.causal";
const KEY_ACOUSTIC_DIM: &str = "vokra.vibevoice.acoustic.vae_dim";
const KEY_ACOUSTIC_FIX_STD: &str = "vokra.vibevoice.acoustic.fix_std";
const KEY_ACOUSTIC_STD_DIST: &str = "vokra.vibevoice.acoustic.std_dist_type";
const KEY_ACOUSTIC_ENCODER_FILTERS: &str = "vokra.vibevoice.acoustic.encoder_n_filters";
const KEY_ACOUSTIC_DECODER_FILTERS: &str = "vokra.vibevoice.acoustic.decoder_n_filters";
const KEY_ACOUSTIC_ENCODER_RATIOS: &str = "vokra.vibevoice.acoustic.encoder_ratios";
const KEY_ACOUSTIC_DECODER_RATIOS: &str = "vokra.vibevoice.acoustic.decoder_ratios";
const KEY_ACOUSTIC_ENCODER_DEPTHS: &str = "vokra.vibevoice.acoustic.encoder_depths";
const KEY_ACOUSTIC_LAYER_SCALE: &str = "vokra.vibevoice.acoustic.layer_scale_init_value";
const KEY_ACOUSTIC_WEIGHT_INIT: &str = "vokra.vibevoice.acoustic.weight_init_value";
const KEY_ACOUSTIC_LAYERNORM: &str = "vokra.vibevoice.acoustic.layernorm";
const KEY_ACOUSTIC_LAYERNORM_AFFINE: &str = "vokra.vibevoice.acoustic.layernorm_elementwise_affine";
const KEY_ACOUSTIC_LAYERNORM_EPS: &str = "vokra.vibevoice.acoustic.layernorm_eps";
const KEY_ACOUSTIC_MIXER: &str = "vokra.vibevoice.acoustic.mixer_layer";
const KEY_ACOUSTIC_PAD: &str = "vokra.vibevoice.acoustic.pad_mode";
const KEY_ACOUSTIC_DISABLE_LAST_NORM: &str = "vokra.vibevoice.acoustic.disable_last_norm";
const KEY_ACOUSTIC_CONV_NORM: &str = "vokra.vibevoice.acoustic.conv_norm";
const KEY_ACOUSTIC_CONV_BIAS: &str = "vokra.vibevoice.acoustic.conv_bias";
const KEY_ACOUSTIC_CORPUS_NORMALIZE: &str = "vokra.vibevoice.acoustic.corpus_normalize";
const KEY_ACOUSTIC_SAMPLE_RATE: &str = "vokra.vibevoice.acoustic.sample_rate_hz";
const KEY_HEAD_HIDDEN: &str = "vokra.vibevoice.diffusion_head.hidden_size";
const KEY_HEAD_LAYERS: &str = "vokra.vibevoice.diffusion_head.head_layers";
const KEY_HEAD_FFN_RATIO: &str = "vokra.vibevoice.diffusion_head.head_ffn_ratio";
const KEY_HEAD_RMS_EPS: &str = "vokra.vibevoice.diffusion_head.rms_norm_eps";
const KEY_HEAD_LATENT: &str = "vokra.vibevoice.diffusion_head.latent_size";
const KEY_HEAD_SPEECH_LATENT: &str = "vokra.vibevoice.diffusion_head.speech_vae_dim";
const KEY_HEAD_PREDICTION: &str = "vokra.vibevoice.diffusion_head.prediction_type";
const KEY_HEAD_DIFFUSION: &str = "vokra.vibevoice.diffusion_head.diffusion_type";
const KEY_HEAD_STEPS: &str = "vokra.vibevoice.diffusion_head.ddpm_num_steps";
const KEY_HEAD_INFERENCE_STEPS: &str = "vokra.vibevoice.diffusion_head.ddpm_num_inference_steps";
const KEY_HEAD_BETA: &str = "vokra.vibevoice.diffusion_head.ddpm_beta_schedule";
const KEY_HEAD_BATCH_MUL: &str = "vokra.vibevoice.diffusion_head.ddpm_batch_mul";

/// Immutable Realtime topology read from a GGUF's converter metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct VibeVoiceStreamingConfig {
    /// Qwen2 hidden width.
    pub hidden_dim: usize,
    /// Total decoder layer count (4 language + 20 TTS plus shared wrapper).
    pub decoder_layers: usize,
    /// TTS backbone layer count.
    pub tts_backbone_layers: usize,
    /// Qwen2 vocabulary size.
    pub vocab_size: usize,
    /// Acoustic VAE latent width.
    pub acoustic_vae_dim: usize,
}

impl VibeVoiceStreamingConfig {
    /// Strictly reads the complete Realtime metadata group.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        require_string(file, chunks::KEY_MODEL_ARCH, ARCH)?;
        require_string(file, chunks::KEY_MODEL_NAME, NAME)?;
        require_string(file, chunks::KEY_PROVENANCE_MODEL_ID, NAME)?;
        require_string(file, chunks::KEY_PROVENANCE_SOURCE, UPSTREAM_HF)?;
        require_string(file, chunks::KEY_PROVENANCE_LICENSE, "mit")?;
        require_string(file, chunks::KEY_PROVENANCE_WEIGHT_LICENSE, "permissive")?;

        require_string(file, KEY_FAMILY, FAMILY)?;
        require_u32(file, KEY_ACOUSTIC_VAE_DIM, ACOUSTIC_VAE_DIM)?;
        require_f32(file, KEY_LM_FRAME_RATE_HZ, FRAME_RATE_HZ)?;
        require_u32(file, KEY_TTS_BACKBONE_LAYERS, TTS_BACKBONE_LAYERS)?;
        require_u32(file, KEY_DECODER_HIDDEN, HIDDEN)?;
        require_u32(file, KEY_DECODER_LAYERS, 24)?;
        require_u32(file, KEY_DECODER_HEADS, HEADS)?;
        require_u32(file, KEY_DECODER_KV_HEADS, KV_HEADS)?;
        require_u32(file, KEY_DECODER_FFN, FFN)?;
        require_u32(file, KEY_DECODER_VOCAB, VOCAB)?;
        require_u32(file, KEY_DECODER_MAX_POSITIONS, MAX_POSITIONS)?;
        require_f32(file, KEY_DECODER_ROPE_BASE, 1_000_000.0)?;
        require_f32(file, KEY_DECODER_RMS_EPS, 1.0e-6)?;
        require_f32(file, KEY_DECODER_DROPOUT, 0.0)?;
        require_bool(file, KEY_DECODER_TIED, false)?;
        require_bool(file, KEY_DECODER_SLIDING, false)?;
        require_u32(file, KEY_DECODER_WINDOW_LAYERS, 24)?;

        require_u32(file, KEY_ACOUSTIC_CHANNELS, 1)?;
        require_bool(file, KEY_ACOUSTIC_CAUSAL, true)?;
        require_u32(file, KEY_ACOUSTIC_DIM, ACOUSTIC_VAE_DIM)?;
        require_f32(file, KEY_ACOUSTIC_FIX_STD, 0.5)?;
        require_string(file, KEY_ACOUSTIC_STD_DIST, "gaussian")?;
        require_u32(file, KEY_ACOUSTIC_ENCODER_FILTERS, 32)?;
        require_u32(file, KEY_ACOUSTIC_DECODER_FILTERS, 32)?;
        require_u32_array(file, KEY_ACOUSTIC_ENCODER_RATIOS, &[8, 5, 5, 4, 2, 2])?;
        require_u32_array(file, KEY_ACOUSTIC_DECODER_RATIOS, &[8, 5, 5, 4, 2, 2])?;
        require_string(file, KEY_ACOUSTIC_ENCODER_DEPTHS, "3-3-3-3-3-3-8")?;
        require_f32(file, KEY_ACOUSTIC_LAYER_SCALE, 1.0e-6)?;
        require_f32(file, KEY_ACOUSTIC_WEIGHT_INIT, 0.01)?;
        require_string(file, KEY_ACOUSTIC_LAYERNORM, "RMSNorm")?;
        require_bool(file, KEY_ACOUSTIC_LAYERNORM_AFFINE, true)?;
        require_f32(file, KEY_ACOUSTIC_LAYERNORM_EPS, 1.0e-5)?;
        require_string(file, KEY_ACOUSTIC_MIXER, "depthwise_conv")?;
        require_string(file, KEY_ACOUSTIC_PAD, "constant")?;
        require_bool(file, KEY_ACOUSTIC_DISABLE_LAST_NORM, true)?;
        require_string(file, KEY_ACOUSTIC_CONV_NORM, "none")?;
        require_bool(file, KEY_ACOUSTIC_CONV_BIAS, true)?;
        require_f32(file, KEY_ACOUSTIC_CORPUS_NORMALIZE, 0.0)?;
        require_u32(file, KEY_ACOUSTIC_SAMPLE_RATE, SAMPLE_RATE_HZ)?;

        require_u32(file, KEY_HEAD_HIDDEN, HIDDEN)?;
        require_u32(file, KEY_HEAD_LAYERS, 4)?;
        require_f32(file, KEY_HEAD_FFN_RATIO, 3.0)?;
        require_f32(file, KEY_HEAD_RMS_EPS, 1.0e-5)?;
        require_u32(file, KEY_HEAD_LATENT, ACOUSTIC_VAE_DIM)?;
        require_u32(file, KEY_HEAD_SPEECH_LATENT, ACOUSTIC_VAE_DIM)?;
        require_string(file, KEY_HEAD_PREDICTION, "v_prediction")?;
        require_string(file, KEY_HEAD_DIFFUSION, "ddpm")?;
        require_u32(file, KEY_HEAD_STEPS, 1000)?;
        require_u32(file, KEY_HEAD_INFERENCE_STEPS, 20)?;
        require_string(file, KEY_HEAD_BETA, "cosine")?;
        require_u32(file, KEY_HEAD_BATCH_MUL, 4)?;

        if file
            .metadata()
            .iter()
            .any(|(key, _)| key.starts_with("vokra.vibevoice.semantic."))
        {
            return Err(VokraError::ModelLoad(
                "vibevoice-realtime: semantic tokenizer metadata is forbidden; Realtime is acoustic-only"
                    .to_owned(),
            ));
        }
        Ok(Self {
            hidden_dim: HIDDEN,
            decoder_layers: 24,
            tts_backbone_layers: TTS_BACKBONE_LAYERS,
            vocab_size: VOCAB,
            acoustic_vae_dim: ACOUSTIC_VAE_DIM,
        })
    }
}

/// Descriptor-only strict tensor binding for the Realtime composite.
#[derive(Debug, Clone)]
pub struct VibeVoiceStreamingWeights {
    tensor_names: BTreeSet<String>,
}

impl VibeVoiceStreamingWeights {
    /// Validates the two language-model stacks and streaming-only heads.
    ///
    /// This does not claim complete payload authentication.  The exact
    /// payload manifest and independent real-weight parity remain a VAST gate.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        if file.tensors().is_empty() {
            return Err(VokraError::ModelLoad(
                "vibevoice-realtime: GGUF carries zero tensors".to_owned(),
            ));
        }
        let mut names = BTreeSet::new();
        for info in file.tensors() {
            if !matches!(info.dtype, GgmlType::F32 | GgmlType::F16 | GgmlType::BF16) {
                return Err(VokraError::ModelLoad(format!(
                    "vibevoice-realtime: tensor `{}` uses {:?}; expected dense F32/F16/BF16",
                    info.name, info.dtype
                )));
            }
            if !info.name.starts_with("model.") {
                return Err(VokraError::ModelLoad(format!(
                    "vibevoice-realtime: tensor `{}` is outside the authenticated model namespace",
                    info.name
                )));
            }
            if info.name.starts_with("model.semantic") {
                return Err(VokraError::ModelLoad(
                    "vibevoice-realtime: semantic tokenizer tensors are forbidden".to_owned(),
                ));
            }
            names.insert(info.name.clone());
        }
        require_layer_stack(file, "model.language_model", LANGUAGE_LAYERS)?;
        require_layer_stack(file, "model.tts_language_model", TTS_LAYERS)?;
        require_nonempty_ranked_tensor(file, "model.tts_input_types")?;
        let classifier = file
            .tensor_info("model.tts_eos_classifier.weight")
            .ok_or_else(|| {
                VokraError::ModelLoad(
                    "vibevoice-realtime: missing model.tts_eos_classifier.weight".to_owned(),
                )
            })?;
        if classifier.dimensions.len() != 2
            || classifier.dimensions[1] as usize != HIDDEN
            || classifier.dimensions[0] == 0
        {
            return Err(VokraError::ModelLoad(format!(
                "vibevoice-realtime: classifier shape {:?} does not end in hidden width {HIDDEN}",
                classifier.dimensions
            )));
        }
        let bias = file
            .tensor_info("model.tts_eos_classifier.bias")
            .ok_or_else(|| {
                VokraError::ModelLoad(
                    "vibevoice-realtime: missing model.tts_eos_classifier.bias".to_owned(),
                )
            })?;
        if bias.dimensions != [classifier.dimensions[0]] {
            return Err(VokraError::ModelLoad(format!(
                "vibevoice-realtime: classifier bias shape {:?} does not match {:?}",
                bias.dimensions, classifier.dimensions
            )));
        }
        Ok(Self {
            tensor_names: names,
        })
    }

    /// Number of descriptors validated by this handle.
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.tensor_names.len()
    }
}

/// Strict Realtime checkpoint handle.  Forward is intentionally loud-partial.
#[derive(Debug, Clone)]
pub struct VibeVoiceStreamingCheckpoint {
    config: VibeVoiceStreamingConfig,
    weights: VibeVoiceStreamingWeights,
}

impl VibeVoiceStreamingCheckpoint {
    /// Applies compliance, metadata, and descriptor topology gates.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        vokra_core::check_weight_license(file, &vokra_core::CompliancePolicy::strict())?;
        let config = VibeVoiceStreamingConfig::from_gguf(file)?;
        let weights = VibeVoiceStreamingWeights::from_gguf(file)?;
        Ok(Self { config, weights })
    }

    /// Returns the strict Realtime configuration.
    #[must_use]
    pub fn config(&self) -> &VibeVoiceStreamingConfig {
        &self.config
    }

    /// Returns the descriptor count validated at load time.
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.weights.tensor_count()
    }

    /// Synthesis remains blocked until the real VAST forward/parity packet.
    pub fn synthesize(&self, text: &str) -> Result<Vec<f32>> {
        if text.is_empty() {
            return Err(VokraError::InvalidArgument(
                "vibevoice-realtime synthesize: text is empty".to_owned(),
            ));
        }
        Err(VokraError::NotImplemented(
            "vibevoice-realtime synthesize: streaming state/prefill, CFG diffusion, acoustic decoder, tokenizer policy, and independent CPU parity remain VAST follow-up gates; no CPU fallback or synthetic waveform is permitted",
        ))
    }
}

fn require_layer_stack(file: &GgufFile, prefix: &str, layers: usize) -> Result<()> {
    for index in 0..layers {
        let base = format!("{prefix}.layers.{index}");
        for suffix in ["input_layernorm.weight", "post_attention_layernorm.weight"] {
            let name = format!("{base}.{suffix}");
            let Some(info) = file.tensor_info(&name) else {
                return Err(VokraError::ModelLoad(format!(
                    "vibevoice-realtime: missing streaming stack tensor `{name}`"
                )));
            };
            if info.dimensions != [HIDDEN as u64] {
                return Err(VokraError::ModelLoad(format!(
                    "vibevoice-realtime: `{name}` shape {:?}, expected [{HIDDEN}]",
                    info.dimensions
                )));
            }
        }
    }
    Ok(())
}

fn require_nonempty_ranked_tensor(file: &GgufFile, name: &str) -> Result<()> {
    let Some(info) = file.tensor_info(name) else {
        return Err(VokraError::ModelLoad(format!(
            "vibevoice-realtime: missing `{name}` tensor"
        )));
    };
    if info.dimensions.is_empty() || info.dimensions.contains(&0) {
        return Err(VokraError::ModelLoad(format!(
            "vibevoice-realtime: `{name}` has empty shape {:?}",
            info.dimensions
        )));
    }
    Ok(())
}

fn require_string(file: &GgufFile, key: &str, expected: &str) -> Result<()> {
    if file.get(key).and_then(GgufMetadataValue::as_str) != Some(expected) {
        return Err(VokraError::ModelLoad(format!(
            "vibevoice-realtime: metadata `{key}` is not {expected:?}"
        )));
    }
    Ok(())
}

fn require_u32(file: &GgufFile, key: &str, expected: usize) -> Result<()> {
    if !matches!(file.get(key), Some(GgufMetadataValue::U32(value)) if *value == expected as u32) {
        return Err(VokraError::ModelLoad(format!(
            "vibevoice-realtime: metadata `{key}` is not UINT32 {expected}"
        )));
    }
    Ok(())
}

fn require_bool(file: &GgufFile, key: &str, expected: bool) -> Result<()> {
    if file.get(key).and_then(GgufMetadataValue::as_bool) != Some(expected) {
        return Err(VokraError::ModelLoad(format!(
            "vibevoice-realtime: metadata `{key}` is not BOOL {expected}"
        )));
    }
    Ok(())
}

fn require_f32(file: &GgufFile, key: &str, expected: f32) -> Result<()> {
    if !matches!(file.get(key), Some(GgufMetadataValue::F32(value)) if *value == expected) {
        return Err(VokraError::ModelLoad(format!(
            "vibevoice-realtime: metadata `{key}` is not F32 {expected}"
        )));
    }
    Ok(())
}

fn require_u32_array(file: &GgufFile, key: &str, expected: &[u32]) -> Result<()> {
    let Some(GgufMetadataValue::Array(array)) = file.get(key) else {
        return Err(VokraError::ModelLoad(format!(
            "vibevoice-realtime: metadata `{key}` is not a UINT32 array"
        )));
    };
    if array.element_type != GgufValueType::U32
        || array.values.len() != expected.len()
        || array
            .values
            .iter()
            .zip(expected)
            .any(|(actual, expected)| !matches!(actual, GgufMetadataValue::U32(value) if value == expected))
    {
        return Err(VokraError::ModelLoad(format!(
            "vibevoice-realtime: metadata `{key}` does not match UINT32 array {expected:?}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::{GgufBuilder, GgufFile};

    fn add_metadata(builder: &mut GgufBuilder) {
        builder
            .add_string(chunks::KEY_MODEL_ARCH, ARCH)
            .add_string(chunks::KEY_MODEL_NAME, NAME)
            .add_string(chunks::KEY_PROVENANCE_MODEL_ID, NAME)
            .add_string(chunks::KEY_PROVENANCE_SOURCE, UPSTREAM_HF)
            .add_string(chunks::KEY_PROVENANCE_LICENSE, "mit")
            .add_string(chunks::KEY_PROVENANCE_WEIGHT_LICENSE, "permissive")
            .add_string(KEY_FAMILY, FAMILY)
            .add_u32(KEY_ACOUSTIC_VAE_DIM, 64)
            .add_f32(KEY_LM_FRAME_RATE_HZ, 7.5)
            .add_u32(KEY_TTS_BACKBONE_LAYERS, 20)
            .add_u32(KEY_DECODER_HIDDEN, 896)
            .add_u32(KEY_DECODER_LAYERS, 24)
            .add_u32(KEY_DECODER_HEADS, 14)
            .add_u32(KEY_DECODER_KV_HEADS, 2)
            .add_u32(KEY_DECODER_FFN, 4864)
            .add_u32(KEY_DECODER_VOCAB, 151_936)
            .add_u32(KEY_DECODER_MAX_POSITIONS, 8192)
            .add_f32(KEY_DECODER_ROPE_BASE, 1_000_000.0)
            .add_f32(KEY_DECODER_RMS_EPS, 1.0e-6)
            .add_f32(KEY_DECODER_DROPOUT, 0.0)
            .add_bool(KEY_DECODER_TIED, false)
            .add_bool(KEY_DECODER_SLIDING, false)
            .add_u32(KEY_DECODER_WINDOW_LAYERS, 24)
            .add_u32(KEY_ACOUSTIC_CHANNELS, 1)
            .add_bool(KEY_ACOUSTIC_CAUSAL, true)
            .add_u32(KEY_ACOUSTIC_DIM, 64)
            .add_f32(KEY_ACOUSTIC_FIX_STD, 0.5)
            .add_string(KEY_ACOUSTIC_STD_DIST, "gaussian")
            .add_u32(KEY_ACOUSTIC_ENCODER_FILTERS, 32)
            .add_u32(KEY_ACOUSTIC_DECODER_FILTERS, 32)
            .add_metadata(KEY_ACOUSTIC_ENCODER_RATIOS, u32_array(&[8, 5, 5, 4, 2, 2]))
            .add_metadata(KEY_ACOUSTIC_DECODER_RATIOS, u32_array(&[8, 5, 5, 4, 2, 2]))
            .add_string(KEY_ACOUSTIC_ENCODER_DEPTHS, "3-3-3-3-3-3-8")
            .add_f32(KEY_ACOUSTIC_LAYER_SCALE, 1.0e-6)
            .add_f32(KEY_ACOUSTIC_WEIGHT_INIT, 0.01)
            .add_string(KEY_ACOUSTIC_LAYERNORM, "RMSNorm")
            .add_bool(KEY_ACOUSTIC_LAYERNORM_AFFINE, true)
            .add_f32(KEY_ACOUSTIC_LAYERNORM_EPS, 1.0e-5)
            .add_string(KEY_ACOUSTIC_MIXER, "depthwise_conv")
            .add_string(KEY_ACOUSTIC_PAD, "constant")
            .add_bool(KEY_ACOUSTIC_DISABLE_LAST_NORM, true)
            .add_string(KEY_ACOUSTIC_CONV_NORM, "none")
            .add_bool(KEY_ACOUSTIC_CONV_BIAS, true)
            .add_f32(KEY_ACOUSTIC_CORPUS_NORMALIZE, 0.0)
            .add_u32(KEY_ACOUSTIC_SAMPLE_RATE, 24_000)
            .add_u32(KEY_HEAD_HIDDEN, 896)
            .add_u32(KEY_HEAD_LAYERS, 4)
            .add_f32(KEY_HEAD_FFN_RATIO, 3.0)
            .add_f32(KEY_HEAD_RMS_EPS, 1.0e-5)
            .add_u32(KEY_HEAD_LATENT, 64)
            .add_u32(KEY_HEAD_SPEECH_LATENT, 64)
            .add_string(KEY_HEAD_PREDICTION, "v_prediction")
            .add_string(KEY_HEAD_DIFFUSION, "ddpm")
            .add_u32(KEY_HEAD_STEPS, 1000)
            .add_u32(KEY_HEAD_INFERENCE_STEPS, 20)
            .add_string(KEY_HEAD_BETA, "cosine")
            .add_u32(KEY_HEAD_BATCH_MUL, 4);
    }

    fn u32_array(values: &[u32]) -> GgufMetadataValue {
        GgufMetadataValue::Array(vokra_core::gguf::GgufArray {
            element_type: GgufValueType::U32,
            values: values.iter().copied().map(GgufMetadataValue::U32).collect(),
        })
    }

    fn valid_file() -> GgufFile {
        let mut builder = GgufBuilder::new();
        add_metadata(&mut builder);
        for prefix in ["model.language_model", "model.tts_language_model"] {
            let layers =
                if prefix.ends_with("language_model") && !prefix.ends_with("tts_language_model") {
                    LANGUAGE_LAYERS
                } else {
                    TTS_LAYERS
                };
            for layer in 0..layers {
                for suffix in ["input_layernorm.weight", "post_attention_layernorm.weight"] {
                    builder
                        .add_tensor(
                            &format!("{prefix}.layers.{layer}.{suffix}"),
                            GgmlType::F32,
                            vec![HIDDEN as u64],
                            vec![0; HIDDEN * 4],
                        )
                        .unwrap();
                }
            }
        }
        builder
            .add_tensor("model.tts_input_types", GgmlType::F32, vec![2], vec![0; 8])
            .unwrap()
            .add_tensor(
                "model.tts_eos_classifier.weight",
                GgmlType::F32,
                vec![1, HIDDEN as u64],
                vec![0; HIDDEN * 4],
            )
            .unwrap()
            .add_tensor(
                "model.tts_eos_classifier.bias",
                GgmlType::F32,
                vec![1],
                vec![0; 4],
            )
            .unwrap();
        GgufFile::parse(builder.to_bytes().unwrap()).unwrap()
    }

    #[test]
    fn strict_synthetic_realtime_topology_binds() {
        let file = valid_file();
        let checkpoint = VibeVoiceStreamingCheckpoint::from_gguf(&file).unwrap();
        assert_eq!(checkpoint.config().hidden_dim, HIDDEN);
        assert_eq!(checkpoint.config().decoder_layers, 24);
        assert_eq!(checkpoint.config().tts_backbone_layers, TTS_LAYERS);
        assert_eq!(
            checkpoint.tensor_count(),
            2 * (LANGUAGE_LAYERS + TTS_LAYERS) + 3
        );
    }

    #[test]
    fn rejects_base_arch_without_misrouting() {
        let mut builder = GgufBuilder::new();
        add_metadata(&mut builder);
        builder.add_string(chunks::KEY_MODEL_ARCH, "vibevoice");
        let file = GgufFile::parse(builder.to_bytes().unwrap()).unwrap();
        let error = VibeVoiceStreamingCheckpoint::from_gguf(&file).unwrap_err();
        assert!(error.to_string().contains("vibevoice_streaming"));
    }

    #[test]
    fn rejects_semantic_metadata_and_missing_streaming_stack() {
        let mut builder = GgufBuilder::new();
        add_metadata(&mut builder);
        builder.add_u32("vokra.vibevoice.semantic.vae_dim", 128);
        let file = GgufFile::parse(builder.to_bytes().unwrap()).unwrap();
        let error = VibeVoiceStreamingCheckpoint::from_gguf(&file).unwrap_err();
        assert!(error.to_string().contains("semantic"));
    }

    #[test]
    fn synthesis_stays_explicitly_unimplemented() {
        let file = valid_file();
        let checkpoint = VibeVoiceStreamingCheckpoint::from_gguf(&file).unwrap();
        let error = checkpoint.synthesize("hello").unwrap_err();
        assert!(matches!(error, VokraError::NotImplemented(_)));
    }
}
