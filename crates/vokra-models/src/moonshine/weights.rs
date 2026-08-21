use vokra_core::gguf::GgufFile;
use vokra_core::{Result, VokraError};

use super::MoonshineConfig;

#[derive(Debug, Clone)]
pub(super) struct Linear {
    pub w: Vec<f32>,
    pub b: Option<Vec<f32>>,
}

#[derive(Debug, Clone)]
pub(super) struct Attention {
    pub q: Linear,
    pub k: Linear,
    pub v: Linear,
    pub o: Linear,
}

#[derive(Debug, Clone)]
pub(super) struct EncoderLayer {
    pub ln1: Vec<f32>,
    pub attn: Attention,
    pub ln2: Vec<f32>,
    pub fc1: Linear,
    pub fc2: Linear,
}

#[derive(Debug, Clone)]
pub(super) struct DecoderLayer {
    pub ln1: Vec<f32>,
    pub self_attn: Attention,
    pub ln2: Vec<f32>,
    pub cross_attn: Attention,
    pub ln3: Vec<f32>,
    pub fc1: Linear,
    pub fc2: Linear,
}

#[derive(Debug, Clone)]
pub(super) struct MoonshineWeights {
    pub embedding: Vec<f32>,
    pub conv1: Linear,
    pub conv2: Linear,
    pub conv3: Linear,
    pub groupnorm_weight: Vec<f32>,
    pub groupnorm_bias: Vec<f32>,
    pub encoder_layers: Vec<EncoderLayer>,
    pub encoder_norm: Vec<f32>,
    pub decoder_layers: Vec<DecoderLayer>,
    pub decoder_norm: Vec<f32>,
}

impl MoonshineWeights {
    pub(super) fn load(file: &GgufFile, config: &MoonshineConfig) -> Result<Self> {
        let d = config.hidden_size;
        let ff = config.intermediate_size;
        let mut encoder_layers = Vec::with_capacity(config.encoder_layers);
        for layer in 0..config.encoder_layers {
            let p = format!("model.encoder.layers.{layer}");
            encoder_layers.push(EncoderLayer {
                ln1: tensor(file, &format!("{p}.input_layernorm.weight"), &[d])?,
                attn: attention(file, &format!("{p}.self_attn"), d)?,
                ln2: tensor(file, &format!("{p}.post_attention_layernorm.weight"), &[d])?,
                fc1: linear(file, &format!("{p}.mlp.fc1"), d, ff, true)?,
                fc2: linear(file, &format!("{p}.mlp.fc2"), ff, d, true)?,
            });
        }
        let mut decoder_layers = Vec::with_capacity(config.decoder_layers);
        for layer in 0..config.decoder_layers {
            let p = format!("model.decoder.layers.{layer}");
            decoder_layers.push(DecoderLayer {
                ln1: tensor(file, &format!("{p}.input_layernorm.weight"), &[d])?,
                self_attn: attention(file, &format!("{p}.self_attn"), d)?,
                ln2: tensor(file, &format!("{p}.post_attention_layernorm.weight"), &[d])?,
                cross_attn: attention(file, &format!("{p}.encoder_attn"), d)?,
                ln3: tensor(file, &format!("{p}.final_layernorm.weight"), &[d])?,
                fc1: linear(file, &format!("{p}.mlp.fc1"), d, 2 * ff, true)?,
                fc2: linear(file, &format!("{p}.mlp.fc2"), ff, d, true)?,
            });
        }
        Ok(Self {
            embedding: tensor(
                file,
                "model.decoder.embed_tokens.weight",
                &[config.vocab_size, d],
            )?,
            conv1: linear_3d(file, "model.encoder.conv1", 1, d, 127, false)?,
            conv2: linear_3d(file, "model.encoder.conv2", d, 2 * d, 7, true)?,
            conv3: linear_3d(file, "model.encoder.conv3", 2 * d, d, 3, true)?,
            groupnorm_weight: tensor(file, "model.encoder.groupnorm.weight", &[d])?,
            groupnorm_bias: tensor(file, "model.encoder.groupnorm.bias", &[d])?,
            encoder_layers,
            encoder_norm: tensor(file, "model.encoder.layer_norm.weight", &[d])?,
            decoder_layers,
            decoder_norm: tensor(file, "model.decoder.norm.weight", &[d])?,
        })
    }
}

fn attention(file: &GgufFile, prefix: &str, d: usize) -> Result<Attention> {
    Ok(Attention {
        q: linear(file, &format!("{prefix}.q_proj"), d, d, false)?,
        k: linear(file, &format!("{prefix}.k_proj"), d, d, false)?,
        v: linear(file, &format!("{prefix}.v_proj"), d, d, false)?,
        o: linear(file, &format!("{prefix}.o_proj"), d, d, false)?,
    })
}

fn linear(
    file: &GgufFile,
    prefix: &str,
    input: usize,
    output: usize,
    bias: bool,
) -> Result<Linear> {
    Ok(Linear {
        w: tensor(file, &format!("{prefix}.weight"), &[output, input])?,
        b: bias
            .then(|| tensor(file, &format!("{prefix}.bias"), &[output]))
            .transpose()?,
    })
}

fn linear_3d(
    file: &GgufFile,
    prefix: &str,
    input: usize,
    output: usize,
    kernel: usize,
    bias: bool,
) -> Result<Linear> {
    Ok(Linear {
        w: tensor(file, &format!("{prefix}.weight"), &[output, input, kernel])?,
        b: bias
            .then(|| tensor(file, &format!("{prefix}.bias"), &[output]))
            .transpose()?,
    })
}

fn tensor(file: &GgufFile, name: &str, expected: &[usize]) -> Result<Vec<f32>> {
    let info = file.tensor_info(name).ok_or_else(|| {
        VokraError::ModelLoad(format!("moonshine: required tensor `{name}` is missing"))
    })?;
    let actual = info
        .dimensions
        .iter()
        .map(|&dimension| dimension as usize)
        .collect::<Vec<_>>();
    if actual != expected {
        return Err(VokraError::ModelLoad(format!(
            "moonshine: tensor `{name}` has shape {actual:?}, expected {expected:?}"
        )));
    }
    file.tensor_f32(name).map_err(|error| {
        VokraError::ModelLoad(format!("moonshine: tensor `{name}` decode failed: {error}"))
    })
}
