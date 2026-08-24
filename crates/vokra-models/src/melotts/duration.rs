//! MeloTTS stochastic/deterministic duration predictors.

use vokra_core::backend::BackendKind;
use vokra_core::gguf::GgufFile;
use vokra_core::rng::NormalSource;
use vokra_core::{Result, VokraError};

use crate::compute::{Compute, HotOp};
use crate::sbv2::duration::{
    ConvFlow, DDSConv, DP_CONV_LAYERS, DP_KERNEL, ElementwiseAffine, LAYER_NORM_EPS, RQS_NUM_BINS,
    SbV2SDP, SdpLayerNorm,
};
use crate::strict_checkpoint::load_tensor;

use super::{GIN_CHANNELS, HIDDEN_CHANNELS, LABEL};

const DP_FILTER_CHANNELS: usize = 256;

/// Backend operations required by both MeloTTS duration predictors.
pub const MELOTTS_DURATION_HOT_OPS: &[HotOp] = &[
    HotOp::Conv1d,
    HotOp::GroupedConv1d,
    HotOp::LayerNorm,
    HotOp::Gelu,
];

struct DeterministicDurationPredictor {
    condition_weight: Vec<f32>,
    condition_bias: Vec<f32>,
    conv1_weight: Vec<f32>,
    conv1_bias: Vec<f32>,
    norm1_gamma: Vec<f32>,
    norm1_beta: Vec<f32>,
    conv2_weight: Vec<f32>,
    conv2_bias: Vec<f32>,
    norm2_gamma: Vec<f32>,
    norm2_beta: Vec<f32>,
    projection_weight: Vec<f32>,
    projection_bias: Vec<f32>,
}

impl DeterministicDurationPredictor {
    fn load(file: &GgufFile) -> Result<Self> {
        let hidden = HIDDEN_CHANNELS as usize;
        let gin = GIN_CHANNELS as usize;
        Ok(Self {
            condition_weight: tensor(file, "dp.cond.weight", &[hidden, gin, 1])?,
            condition_bias: tensor(file, "dp.cond.bias", &[hidden])?,
            conv1_weight: tensor(file, "dp.conv_1.weight", &[DP_FILTER_CHANNELS, hidden, 3])?,
            conv1_bias: tensor(file, "dp.conv_1.bias", &[DP_FILTER_CHANNELS])?,
            norm1_gamma: tensor(file, "dp.norm_1.gamma", &[DP_FILTER_CHANNELS])?,
            norm1_beta: tensor(file, "dp.norm_1.beta", &[DP_FILTER_CHANNELS])?,
            conv2_weight: tensor(
                file,
                "dp.conv_2.weight",
                &[DP_FILTER_CHANNELS, DP_FILTER_CHANNELS, 3],
            )?,
            conv2_bias: tensor(file, "dp.conv_2.bias", &[DP_FILTER_CHANNELS])?,
            norm2_gamma: tensor(file, "dp.norm_2.gamma", &[DP_FILTER_CHANNELS])?,
            norm2_beta: tensor(file, "dp.norm_2.beta", &[DP_FILTER_CHANNELS])?,
            projection_weight: tensor(file, "dp.proj.weight", &[1, DP_FILTER_CHANNELS, 1])?,
            projection_bias: tensor(file, "dp.proj.bias", &[1])?,
        })
    }

    fn forward(
        &self,
        compute: &Compute,
        hidden_position_major: &[f32],
        sequence_len: usize,
        global_conditioning: &[f32],
    ) -> Result<Vec<f32>> {
        let hidden_width = HIDDEN_CHANNELS as usize;
        let hidden_channel_major =
            position_to_channel(hidden_position_major, sequence_len, hidden_width);
        let condition = conv1d(
            compute,
            global_conditioning,
            GIN_CHANNELS as usize,
            1,
            &self.condition_weight,
            hidden_width,
            1,
            Some(&self.condition_bias),
            0,
        )?;
        let mut conditioned = hidden_channel_major;
        for channel in 0..hidden_width {
            for position in 0..sequence_len {
                conditioned[channel * sequence_len + position] += condition[channel];
            }
        }
        let hidden = conv1d(
            compute,
            &conditioned,
            hidden_width,
            sequence_len,
            &self.conv1_weight,
            DP_FILTER_CHANNELS,
            3,
            Some(&self.conv1_bias),
            1,
        )?;
        let hidden = relu(hidden);
        let hidden = layer_norm_channels(
            compute,
            &hidden,
            DP_FILTER_CHANNELS,
            sequence_len,
            &self.norm1_gamma,
            &self.norm1_beta,
        )?;
        let hidden = conv1d(
            compute,
            &hidden,
            DP_FILTER_CHANNELS,
            sequence_len,
            &self.conv2_weight,
            DP_FILTER_CHANNELS,
            3,
            Some(&self.conv2_bias),
            1,
        )?;
        let hidden = relu(hidden);
        let hidden = layer_norm_channels(
            compute,
            &hidden,
            DP_FILTER_CHANNELS,
            sequence_len,
            &self.norm2_gamma,
            &self.norm2_beta,
        )?;
        conv1d(
            compute,
            &hidden,
            DP_FILTER_CHANNELS,
            sequence_len,
            &self.projection_weight,
            1,
            1,
            Some(&self.projection_bias),
            0,
        )
    }
}

/// Loaded MeloTTS duration stack.
pub struct MeloDurationModel {
    stochastic: SbV2SDP,
    deterministic: DeterministicDurationPredictor,
}

impl MeloDurationModel {
    pub(super) fn from_gguf(file: &GgufFile) -> Result<Self> {
        let hidden = HIDDEN_CHANNELS as usize;
        let gin = GIN_CHANNELS as usize;
        let stochastic = SbV2SDP::from_weights(
            hidden,
            gin,
            tensor(file, "sdp.pre.weight", &[hidden, hidden, 1])?,
            tensor(file, "sdp.pre.bias", &[hidden])?,
            load_dds(file, "sdp.convs", hidden)?,
            tensor(file, "sdp.cond.weight", &[hidden, gin, 1])?,
            tensor(file, "sdp.cond.bias", &[hidden])?,
            tensor(file, "sdp.proj.weight", &[hidden, hidden, 1])?,
            tensor(file, "sdp.proj.bias", &[hidden])?,
            ElementwiseAffine::from_weights(
                tensor(file, "sdp.flows.0.m", &[2, 1])?,
                tensor(file, "sdp.flows.0.logs", &[2, 1])?,
            ),
            [1, 3, 5, 7]
                .into_iter()
                .map(|index| load_conv_flow(file, index, hidden))
                .collect::<Result<Vec<_>>>()?,
        );
        Ok(Self {
            stochastic,
            deterministic: DeterministicDurationPredictor::load(file)?,
        })
    }

    /// Predicts one positive frame duration per text position.
    #[allow(clippy::too_many_arguments)]
    pub fn predict<R: NormalSource>(
        &self,
        hidden: &[f32],
        sequence_len: usize,
        global_conditioning: &[f32],
        sdp_ratio: f32,
        noise_scale_w: f32,
        length_scale: f32,
        rng: &mut R,
        backend: BackendKind,
    ) -> Result<Vec<i32>> {
        let compute = Compute::for_backend(backend, MELOTTS_DURATION_HOT_OPS)?;
        self.predict_with_compute(
            &compute,
            hidden,
            sequence_len,
            global_conditioning,
            sdp_ratio,
            noise_scale_w,
            length_scale,
            rng,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn predict_with_compute<R: NormalSource>(
        &self,
        compute: &Compute,
        hidden: &[f32],
        sequence_len: usize,
        global_conditioning: &[f32],
        sdp_ratio: f32,
        noise_scale_w: f32,
        length_scale: f32,
        rng: &mut R,
    ) -> Result<Vec<i32>> {
        if !(0.0..=1.0).contains(&sdp_ratio) || !sdp_ratio.is_finite() {
            return Err(VokraError::InvalidArgument(format!(
                "melotts duration: sdp_ratio must be finite in [0, 1], got {sdp_ratio}"
            )));
        }
        if !noise_scale_w.is_finite() || noise_scale_w < 0.0 {
            return Err(VokraError::InvalidArgument(format!(
                "melotts duration: noise_scale_w must be finite and non-negative, got {noise_scale_w}"
            )));
        }
        if !length_scale.is_finite() || length_scale <= 0.0 {
            return Err(VokraError::InvalidArgument(format!(
                "melotts duration: length_scale must be finite and positive, got {length_scale}"
            )));
        }
        let deterministic =
            self.deterministic
                .forward(compute, hidden, sequence_len, global_conditioning)?;
        let stochastic = if sdp_ratio == 0.0 {
            vec![0.0; sequence_len]
        } else {
            self.stochastic.sample_log_duration_with_compute(
                compute,
                hidden,
                sequence_len,
                global_conditioning,
                rng,
                noise_scale_w,
            )?
        };
        Ok(stochastic
            .iter()
            .zip(deterministic)
            .map(|(stochastic, deterministic)| {
                let log_duration = stochastic * sdp_ratio + deterministic * (1.0 - sdp_ratio);
                (vokra_math::exp(log_duration) * length_scale)
                    .ceil()
                    .max(1.0) as i32
            })
            .collect())
    }
}

fn load_dds(file: &GgufFile, prefix: &str, channels: usize) -> Result<DDSConv> {
    let mut separated_weights = Vec::with_capacity(DP_CONV_LAYERS);
    let mut separated_biases = Vec::with_capacity(DP_CONV_LAYERS);
    let mut pointwise_weights = Vec::with_capacity(DP_CONV_LAYERS);
    let mut pointwise_biases = Vec::with_capacity(DP_CONV_LAYERS);
    let mut norms1 = Vec::with_capacity(DP_CONV_LAYERS);
    let mut norms2 = Vec::with_capacity(DP_CONV_LAYERS);
    for layer in 0..DP_CONV_LAYERS {
        separated_weights.push(tensor(
            file,
            &format!("{prefix}.convs_sep.{layer}.weight"),
            &[channels, 1, DP_KERNEL],
        )?);
        separated_biases.push(tensor(
            file,
            &format!("{prefix}.convs_sep.{layer}.bias"),
            &[channels],
        )?);
        pointwise_weights.push(tensor(
            file,
            &format!("{prefix}.convs_1x1.{layer}.weight"),
            &[channels, channels, 1],
        )?);
        pointwise_biases.push(tensor(
            file,
            &format!("{prefix}.convs_1x1.{layer}.bias"),
            &[channels],
        )?);
        norms1.push(SdpLayerNorm {
            gamma: tensor(
                file,
                &format!("{prefix}.norms_1.{layer}.gamma"),
                &[channels],
            )?,
            beta: tensor(file, &format!("{prefix}.norms_1.{layer}.beta"), &[channels])?,
        });
        norms2.push(SdpLayerNorm {
            gamma: tensor(
                file,
                &format!("{prefix}.norms_2.{layer}.gamma"),
                &[channels],
            )?,
            beta: tensor(file, &format!("{prefix}.norms_2.{layer}.beta"), &[channels])?,
        });
    }
    Ok(DDSConv::from_weights(
        channels,
        DP_CONV_LAYERS,
        DP_KERNEL,
        separated_weights,
        separated_biases,
        pointwise_weights,
        pointwise_biases,
        norms1,
        norms2,
    ))
}

fn load_conv_flow(file: &GgufFile, index: usize, channels: usize) -> Result<ConvFlow> {
    let prefix = format!("sdp.flows.{index}");
    Ok(ConvFlow::from_weights(
        tensor(file, &format!("{prefix}.pre.weight"), &[channels, 1, 1])?,
        tensor(file, &format!("{prefix}.pre.bias"), &[channels])?,
        load_dds(file, &format!("{prefix}.convs"), channels)?,
        tensor(
            file,
            &format!("{prefix}.proj.weight"),
            &[RQS_NUM_BINS * 3 - 1, channels, 1],
        )?,
        tensor(
            file,
            &format!("{prefix}.proj.bias"),
            &[RQS_NUM_BINS * 3 - 1],
        )?,
        channels,
    ))
}

#[allow(clippy::too_many_arguments)]
fn conv1d(
    compute: &Compute,
    input: &[f32],
    in_channels: usize,
    input_len: usize,
    weight: &[f32],
    out_channels: usize,
    kernel: usize,
    bias: Option<&[f32]>,
    padding: usize,
) -> Result<Vec<f32>> {
    let output_len = input_len + 2 * padding - kernel + 1;
    let mut output = vec![0.0; out_channels * output_len];
    compute.conv1d_f32(
        input,
        in_channels,
        input_len,
        weight,
        out_channels,
        kernel,
        bias,
        1,
        padding,
        &mut output,
    )?;
    Ok(output)
}

fn layer_norm_channels(
    compute: &Compute,
    input: &[f32],
    channels: usize,
    time: usize,
    gamma: &[f32],
    beta: &[f32],
) -> Result<Vec<f32>> {
    let position_major = channel_to_position(input, channels, time);
    let mut normalized = vec![0.0; input.len()];
    compute.layer_norm_f32(
        &position_major,
        &mut normalized,
        time,
        channels,
        gamma,
        beta,
        LAYER_NORM_EPS,
    )?;
    Ok(position_to_channel(&normalized, time, channels))
}

fn relu(mut values: Vec<f32>) -> Vec<f32> {
    for value in &mut values {
        *value = value.max(0.0);
    }
    values
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

fn channel_to_position(input: &[f32], channels: usize, time: usize) -> Vec<f32> {
    let mut output = vec![0.0; input.len()];
    for channel in 0..channels {
        for position in 0..time {
            output[position * channels + channel] = input[channel * time + position];
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
    fn position_channel_transposes_round_trip() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let channel = position_to_channel(&input, 2, 3);
        assert_eq!(channel, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        assert_eq!(channel_to_position(&channel, 3, 2), input);
    }
}
