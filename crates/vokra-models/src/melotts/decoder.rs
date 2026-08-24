//! MeloTTS speaker-conditioned HiFi-GAN decoder.

use vokra_core::backend::BackendKind;
use vokra_core::gguf::GgufFile;
use vokra_core::{Result, VokraError};
use vokra_ops::attrs::{HifiGanAttrs, ResBlockType};
use vokra_ops::hifigan::{
    GinCondition, HifiGanConfig, HifiGanConvPadding, HifiGanWeights, MrfBranchWeights,
    ResBlockLayer, UpsampleStageWeights, hifigan_generator_conditioned_with_backend_ops,
};

use crate::compute::{Compute, HotOp};
use crate::hifigan::HifiGanComputeOps;
use crate::strict_checkpoint::load_tensor;

use super::{GIN_CHANNELS, INTER_CHANNELS, LABEL, SAMPLE_RATE, UPSAMPLE_INITIAL_CHANNEL};

const CONV_KERNEL: usize = 7;
const UPSAMPLE_RATES: [usize; 5] = [8, 8, 2, 2, 2];
const UPSAMPLE_KERNELS: [usize; 5] = [16, 16, 8, 2, 2];
const RESBLOCK_KERNELS: [usize; 3] = [3, 7, 11];
const RESBLOCK_DILATIONS: [usize; 3] = [1, 3, 5];

/// Backend operations required by the MeloTTS HiFi-GAN decoder.
pub const MELOTTS_DECODER_HOT_OPS: &[HotOp] = &[HotOp::Conv1d];

/// Loaded MeloTTS speaker-conditioned HiFi-GAN generator.
pub struct MeloDecoder {
    weights: HifiGanWeights,
    attrs: HifiGanAttrs,
}

impl MeloDecoder {
    pub(super) fn from_gguf(file: &GgufFile) -> Result<Self> {
        let initial = UPSAMPLE_INITIAL_CHANNEL as usize;
        let attrs = HifiGanAttrs {
            n_mels: INTER_CHANNELS as usize,
            initial_channel: initial,
            upsample_rates: UPSAMPLE_RATES.to_vec(),
            upsample_kernel_sizes: UPSAMPLE_KERNELS.to_vec(),
            resblock_kernel_sizes: RESBLOCK_KERNELS.to_vec(),
            resblock_dilation_sizes: vec![RESBLOCK_DILATIONS.to_vec(); 3],
            sample_rate: SAMPLE_RATE,
            leaky_relu_slope: 0.1,
            res_block_type: ResBlockType::V1,
        };
        attrs.validate_shape()?;

        let mut upsample_weights = Vec::with_capacity(UPSAMPLE_RATES.len());
        let mut mrf_stage_weights = Vec::with_capacity(UPSAMPLE_RATES.len());
        let mut in_channels = initial;
        for stage in 0..UPSAMPLE_RATES.len() {
            let out_channels = in_channels / 2;
            let kernel = UPSAMPLE_KERNELS[stage];
            upsample_weights.push(UpsampleStageWeights {
                weight: folded_weight(
                    file,
                    &format!("dec.ups.{stage}"),
                    &[in_channels, 1, 1],
                    &[in_channels, out_channels, kernel],
                    in_channels,
                    out_channels * kernel,
                )?,
                bias: tensor(file, &format!("dec.ups.{stage}.bias"), &[out_channels])?,
                in_ch: in_channels,
                out_ch: out_channels,
                kernel,
                stride: UPSAMPLE_RATES[stage],
            });

            let mut branches = Vec::with_capacity(RESBLOCK_KERNELS.len());
            for (branch, kernel) in RESBLOCK_KERNELS.into_iter().enumerate() {
                let upstream_index = stage * RESBLOCK_KERNELS.len() + branch;
                let mut layers = Vec::with_capacity(RESBLOCK_DILATIONS.len());
                for (layer, dilation) in RESBLOCK_DILATIONS.into_iter().enumerate() {
                    let conv1 = format!("dec.resblocks.{upstream_index}.convs1.{layer}");
                    let conv2 = format!("dec.resblocks.{upstream_index}.convs2.{layer}");
                    layers.push(ResBlockLayer {
                        weight: folded_weight(
                            file,
                            &conv1,
                            &[out_channels, 1, 1],
                            &[out_channels, out_channels, kernel],
                            out_channels,
                            out_channels * kernel,
                        )?,
                        bias: tensor(file, &format!("{conv1}.bias"), &[out_channels])?,
                        weight_c2: Some(folded_weight(
                            file,
                            &conv2,
                            &[out_channels, 1, 1],
                            &[out_channels, out_channels, kernel],
                            out_channels,
                            out_channels * kernel,
                        )?),
                        bias_c2: Some(tensor(file, &format!("{conv2}.bias"), &[out_channels])?),
                        dilation,
                        kernel,
                        channels: out_channels,
                    });
                }
                branches.push(MrfBranchWeights { layers });
            }
            mrf_stage_weights.push(branches);
            in_channels = out_channels;
        }

        Ok(Self {
            weights: HifiGanWeights {
                conv_pre_weight: tensor(
                    file,
                    "dec.conv_pre.weight",
                    &[initial, INTER_CHANNELS as usize, CONV_KERNEL],
                )?,
                conv_pre_bias: tensor(file, "dec.conv_pre.bias", &[initial])?,
                conv_pre_kernel: CONV_KERNEL,
                upsample_weights,
                mrf_stage_weights,
                conv_post_weight: tensor(
                    file,
                    "dec.conv_post.weight",
                    &[1, in_channels, CONV_KERNEL],
                )?,
                conv_post_bias: Vec::new(),
                conv_post_kernel: CONV_KERNEL,
                cond: Some(GinCondition {
                    weight: tensor(
                        file,
                        "dec.cond.weight",
                        &[initial, GIN_CHANNELS as usize, 1],
                    )?,
                    bias: tensor(file, "dec.cond.bias", &[initial])?,
                    gin_channels: GIN_CHANNELS as usize,
                }),
            },
            attrs,
        })
    }

    /// Decodes position-major flow latents to 44.1 kHz mono PCM.
    pub fn decode(
        &self,
        latent_position_major: &[f32],
        frame_count: usize,
        global_conditioning: &[f32],
        backend: BackendKind,
    ) -> Result<Vec<f32>> {
        let channels = INTER_CHANNELS as usize;
        if frame_count == 0 {
            return Err(VokraError::InvalidArgument(
                "melotts decoder: frame_count must be positive".to_owned(),
            ));
        }
        if latent_position_major.len() != frame_count * channels {
            return Err(VokraError::InvalidArgument(format!(
                "melotts decoder: expected latent [{frame_count}, {channels}], got {} values",
                latent_position_major.len()
            )));
        }
        if global_conditioning.len() != GIN_CHANNELS as usize {
            return Err(VokraError::InvalidArgument(format!(
                "melotts decoder: expected {} speaker-conditioning values, got {}",
                GIN_CHANNELS,
                global_conditioning.len()
            )));
        }
        let channel_major = position_to_channel(latent_position_major, frame_count, channels);
        let compute = Compute::for_backend(backend, MELOTTS_DECODER_HOT_OPS)?;
        let ops = HifiGanComputeOps { compute: &compute };
        let pcm = hifigan_generator_conditioned_with_backend_ops(
            &channel_major,
            frame_count,
            &self.weights,
            &self.attrs,
            &HifiGanConfig::fp32(),
            global_conditioning,
            HifiGanConvPadding::Zero,
            &ops,
        )?;
        debug_assert_eq!(pcm.len(), frame_count * self.attrs.total_upsample_factor());
        Ok(pcm)
    }

    /// Output sampling rate in Hz.
    #[must_use]
    pub const fn sample_rate(&self) -> u32 {
        SAMPLE_RATE
    }
}

fn folded_weight(
    file: &GgufFile,
    prefix: &str,
    g_shape: &[usize],
    v_shape: &[usize],
    rows: usize,
    row_width: usize,
) -> Result<Vec<f32>> {
    let g = tensor(file, &format!("{prefix}.weight_g"), g_shape)?;
    let v = tensor(file, &format!("{prefix}.weight_v"), v_shape)?;
    fold_weight_norm(&v, &g, rows, row_width, prefix)
}

fn fold_weight_norm(
    v: &[f32],
    g: &[f32],
    rows: usize,
    row_width: usize,
    label: &str,
) -> Result<Vec<f32>> {
    if v.len() != rows * row_width || g.len() != rows {
        return Err(VokraError::ModelLoad(format!(
            "melotts decoder: `{label}` weight-norm lengths v={} g={}, expected {} and {rows}",
            v.len(),
            g.len(),
            rows * row_width
        )));
    }
    let mut weight = vec![0.0; v.len()];
    for row in 0..rows {
        let source = &v[row * row_width..(row + 1) * row_width];
        let norm = source.iter().map(|value| value * value).sum::<f32>().sqrt();
        if !norm.is_finite() || norm == 0.0 || !g[row].is_finite() {
            return Err(VokraError::ModelLoad(format!(
                "melotts decoder: `{label}` has invalid weight-norm row {row}: norm={norm}, g={}",
                g[row]
            )));
        }
        let scale = g[row] / norm;
        for (destination, source) in weight[row * row_width..(row + 1) * row_width]
            .iter_mut()
            .zip(source)
        {
            *destination = *source * scale;
        }
    }
    Ok(weight)
}

fn position_to_channel(input: &[f32], time: usize, channels: usize) -> Vec<f32> {
    let mut output = vec![0.0; input.len()];
    for position in 0..time {
        for channel in 0..channels {
            output[channel * time + position] = input[position * channels + channel];
        }
    }
    output
}

fn tensor(file: &GgufFile, name: &str, expected: &[usize]) -> Result<Vec<f32>> {
    load_tensor(file, LABEL, name, expected)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weight_norm_fold_matches_definition() {
        let folded = fold_weight_norm(&[3.0, 4.0, 0.0, 2.0], &[2.0, 3.0], 2, 2, "test").unwrap();
        assert_eq!(folded, vec![1.2, 1.6, 0.0, 3.0]);
    }

    #[test]
    fn position_channel_transpose_matches_expected_layout() {
        assert_eq!(
            position_to_channel(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3),
            vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
        );
    }
}
