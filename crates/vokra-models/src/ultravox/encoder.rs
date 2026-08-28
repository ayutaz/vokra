//! Variable-length Whisper encoder execution for Ultravox.

use std::sync::Mutex;

use vokra_core::Result;
use vokra_core::VokraError;

use crate::compute::Compute;
use crate::mapped_weights::{lock_scratch, transpose_widen, widen_into};
use crate::whisper::encoder::encoder_block;
use crate::whisper::nn::layer_norm_into;
use crate::whisper::scratch::EncoderScratch;
use crate::whisper::weights::{Attention, EncoderLayer, LayerNorm, Linear, LinearWeight};

use super::projector::{self, ProjectorRuntime, UltravoxAudioEmbeddings};
use super::weights::{AudioLayerDescriptors, LinearDescriptors, UltravoxMappedDescriptors};
use super::{ULTRAVOX_AUDIO_HOT_OPS, UltravoxAudioConfig, UltravoxAudioTower};

#[derive(Default)]
pub(super) struct UltravoxAudioRuntime {
    stem: Mutex<StemWeights>,
    layer: Mutex<AudioLayerBlock>,
    projector: ProjectorRuntime,
}

#[derive(Default)]
struct StemWeights {
    ready: bool,
    conv1_weight: Vec<f32>,
    conv1_bias: Vec<f32>,
    conv2_weight: Vec<f32>,
    conv2_bias: Vec<f32>,
    positions: Vec<f32>,
    final_norm_weight: Vec<f32>,
    final_norm_bias: Vec<f32>,
}

struct AudioLayerBlock {
    layer: EncoderLayer,
}

impl Default for AudioLayerBlock {
    fn default() -> Self {
        let config = UltravoxAudioConfig::OFFICIAL;
        let d = config.hidden_size;
        let ff = config.ffn_dim;
        Self {
            layer: EncoderLayer {
                attn_ln: empty_norm(),
                attn: Attention {
                    q: empty_linear(d, d, true),
                    k: empty_linear(d, d, false),
                    v: empty_linear(d, d, true),
                    out: empty_linear(d, d, true),
                },
                mlp_ln: empty_norm(),
                fc1: empty_linear(d, ff, true),
                fc2: empty_linear(ff, d, true),
            },
        }
    }
}

pub(super) fn encode(
    tower: &UltravoxAudioTower,
    log_mel: &[f32],
    n_frames: usize,
) -> Result<UltravoxAudioEmbeddings> {
    let mapped = &tower.mapped;
    let config = mapped.config();
    validate_input(log_mel, n_frames, config)?;
    let compute = Compute::for_backend(tower.backend, ULTRAVOX_AUDIO_HOT_OPS)?;

    let mut stem = lock_scratch(&tower.runtime.stem, mapped.mapped_model())?;
    if !stem.ready {
        materialize_stem(mapped, &mut stem)?;
        stem.ready = true;
    }
    let (mut hidden, encoded_frames) = stem_forward(&compute, &stem, log_mel, n_frames, config)?;
    drop(stem);

    let mut scratch = EncoderScratch::with_reserve(
        encoded_frames,
        config.hidden_size,
        config.ffn_dim,
        config.n_head,
    );
    let mut layer = lock_scratch(&tower.runtime.layer, mapped.mapped_model())?;
    for index in 0..config.n_layer {
        materialize_layer(mapped, &mapped.layers[index], &mut layer.layer)?;
        encoder_block(
            &compute,
            &mut scratch.block,
            &mut hidden,
            encoded_frames,
            config.hidden_size,
            config.ffn_dim,
            config.n_head,
            &layer.layer,
        )?;
    }
    drop(layer);

    let stem = lock_scratch(&tower.runtime.stem, mapped.mapped_model())?;
    let final_norm = LayerNorm {
        gamma: stem.final_norm_weight.clone(),
        beta: stem.final_norm_bias.clone(),
    };
    let mut normed = Vec::new();
    layer_norm_into(&compute, &mut normed, &hidden, encoded_frames, &final_norm)?;
    drop(stem);
    reject_non_finite("encoder output", &normed)?;

    projector::project(
        &compute,
        mapped,
        &tower.runtime.projector,
        &normed,
        encoded_frames,
    )
}

fn validate_input(log_mel: &[f32], n_frames: usize, config: UltravoxAudioConfig) -> Result<()> {
    if !(1..=config.max_mel_frames).contains(&n_frames) {
        return Err(VokraError::InvalidArgument(format!(
            "ultravox: n_frames must be in 1..={}, got {n_frames}",
            config.max_mel_frames
        )));
    }
    let expected = config.n_mels.checked_mul(n_frames).ok_or_else(|| {
        VokraError::InvalidArgument("ultravox: n_mels*n_frames overflow".to_owned())
    })?;
    if log_mel.len() != expected {
        return Err(VokraError::InvalidArgument(format!(
            "ultravox: log-mel len {} != {}*{n_frames} ({expected})",
            log_mel.len(),
            config.n_mels
        )));
    }
    reject_non_finite("log-mel input", log_mel)
}

fn materialize_stem(mapped: &UltravoxMappedDescriptors, output: &mut StemWeights) -> Result<()> {
    let descriptors = &mapped.stem;
    widen_tensor(mapped, &descriptors.conv1_weight, &mut output.conv1_weight)?;
    widen_tensor(mapped, &descriptors.conv1_bias, &mut output.conv1_bias)?;
    widen_tensor(mapped, &descriptors.conv2_weight, &mut output.conv2_weight)?;
    widen_tensor(mapped, &descriptors.conv2_bias, &mut output.conv2_bias)?;
    widen_tensor(mapped, &descriptors.positions, &mut output.positions)?;
    widen_tensor(
        mapped,
        &descriptors.final_norm_weight,
        &mut output.final_norm_weight,
    )?;
    widen_tensor(
        mapped,
        &descriptors.final_norm_bias,
        &mut output.final_norm_bias,
    )
}

fn materialize_layer(
    mapped: &UltravoxMappedDescriptors,
    descriptors: &AudioLayerDescriptors,
    output: &mut EncoderLayer,
) -> Result<()> {
    let config = mapped.config();
    widen_tensor(
        mapped,
        &descriptors.self_attn_norm_weight,
        &mut output.attn_ln.gamma,
    )?;
    widen_tensor(
        mapped,
        &descriptors.self_attn_norm_bias,
        &mut output.attn_ln.beta,
    )?;
    materialize_linear(mapped, &descriptors.q, &mut output.attn.q)?;
    materialize_linear(mapped, &descriptors.k, &mut output.attn.k)?;
    materialize_linear(mapped, &descriptors.v, &mut output.attn.v)?;
    materialize_linear(mapped, &descriptors.out, &mut output.attn.out)?;
    widen_tensor(
        mapped,
        &descriptors.final_norm_weight,
        &mut output.mlp_ln.gamma,
    )?;
    widen_tensor(
        mapped,
        &descriptors.final_norm_bias,
        &mut output.mlp_ln.beta,
    )?;
    debug_assert_eq!(output.fc1.in_features, config.hidden_size);
    debug_assert_eq!(output.fc1.out_features, config.ffn_dim);
    materialize_linear(mapped, &descriptors.fc1, &mut output.fc1)?;
    materialize_linear(mapped, &descriptors.fc2, &mut output.fc2)
}

fn materialize_linear(
    mapped: &UltravoxMappedDescriptors,
    descriptors: &LinearDescriptors,
    output: &mut Linear,
) -> Result<()> {
    let weight = match &mut output.w {
        LinearWeight::Dense(weight) => weight,
        LinearWeight::KQuant { .. } => unreachable!("Ultravox scratch is always dense"),
    };
    transpose_widen(
        mapped.file().tensor_bytes(&descriptors.weight),
        descriptors.weight.dtype,
        output.out_features,
        output.in_features,
        weight,
        mapped.mapped_model(),
    )?;
    match (&descriptors.bias, &mut output.bias) {
        (Some(info), Some(bias)) => widen_tensor(mapped, info, bias),
        (None, None) => Ok(()),
        _ => Err(VokraError::ModelLoad(
            "ultravox: internal linear bias contract drift".to_owned(),
        )),
    }
}

fn stem_forward(
    compute: &Compute,
    weights: &StemWeights,
    log_mel: &[f32],
    n_frames: usize,
    config: UltravoxAudioConfig,
) -> Result<(Vec<f32>, usize)> {
    let d = config.hidden_size;
    let len1 = conv_out_len(n_frames, 3, 1, 1);
    let mut conv1 = vec![0.0; d * len1];
    compute.conv1d_f32(
        log_mel,
        config.n_mels,
        n_frames,
        &weights.conv1_weight,
        d,
        3,
        Some(&weights.conv1_bias),
        1,
        1,
        &mut conv1,
    )?;
    gelu_in_place(compute, &mut conv1)?;

    let len2 = conv_out_len(len1, 3, 2, 1);
    let mut conv2 = vec![0.0; d * len2];
    compute.conv1d_f32(
        &conv1,
        d,
        len1,
        &weights.conv2_weight,
        d,
        3,
        Some(&weights.conv2_bias),
        2,
        1,
        &mut conv2,
    )?;
    gelu_in_place(compute, &mut conv2)?;

    let max_positions = config.max_mel_frames / 2;
    if len2 > max_positions {
        return Err(VokraError::InvalidArgument(format!(
            "ultravox: post-conv frames {len2} exceed positional table {max_positions}"
        )));
    }
    let mut hidden = vec![0.0; len2 * d];
    for channel in 0..d {
        for frame in 0..len2 {
            hidden[frame * d + channel] =
                conv2[channel * len2 + frame] + weights.positions[frame * d + channel];
        }
    }
    Ok((hidden, len2))
}

fn gelu_in_place(compute: &Compute, values: &mut [f32]) -> Result<()> {
    let mut output = vec![0.0; values.len()];
    compute.gelu_f32(values, &mut output)?;
    values.copy_from_slice(&output);
    Ok(())
}

fn conv_out_len(input: usize, kernel: usize, stride: usize, padding: usize) -> usize {
    (input + 2 * padding - kernel) / stride + 1
}

fn empty_norm() -> LayerNorm {
    LayerNorm {
        gamma: Vec::new(),
        beta: Vec::new(),
    }
}

fn empty_linear(input: usize, output: usize, bias: bool) -> Linear {
    Linear::dense(Vec::new(), input, output, bias.then(Vec::<f32>::new))
}

fn widen_tensor(
    mapped: &UltravoxMappedDescriptors,
    info: &vokra_core::gguf::GgufTensorInfo,
    output: &mut Vec<f32>,
) -> Result<()> {
    widen_into(
        mapped.file().tensor_bytes(info),
        info.dtype,
        output,
        mapped.mapped_model(),
    )
}

fn reject_non_finite(label: &str, values: &[f32]) -> Result<()> {
    if let Some((index, value)) = values
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(VokraError::InvalidArgument(format!(
            "ultravox: {label} contains non-finite value {value} at index {index}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn convolution_lengths_match_modified_whisper() {
        assert_eq!(conv_out_len(3_000, 3, 1, 1), 3_000);
        assert_eq!(conv_out_len(3_000, 3, 2, 1), 1_500);
        assert_eq!(conv_out_len(7, 3, 2, 1), 4);
    }

    #[test]
    fn input_contract_preserves_variable_length() {
        let config = UltravoxAudioConfig::OFFICIAL;
        assert!(validate_input(&vec![0.0; config.n_mels * 17], 17, config).is_ok());
        let error = validate_input(&[], 0, config).unwrap_err();
        assert!(format!("{error}").contains("1..=3000"));
    }
}
