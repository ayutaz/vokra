//! Vocos ConvNeXt-1D backbone and iSTFT head.
//!
//! This is a native transcription of `vocos==0.1.0`'s
//! `VocosBackbone` / `ConvNeXtBlock` / `ISTFTHead`.  The released models do
//! not use ConvNeXt-V2 GRN: the non-conditional model uses LayerNorm and the
//! Encodec model uses bandwidth-conditioned AdaLayerNorm.  Inputs are
//! channel-major `[input_channels, frames]`; the output is mono PCM.

use vokra_core::ir::graph::IstftAttrs;
use vokra_core::{Result, VokraError};

use crate::{Spectrogram, istft};

/// Padding contract used by the released Vocos iSTFT heads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VocosIstftPadding {
    /// `torch.istft(..., center=true)`; trims `n_fft / 2` at both edges.
    Center,
    /// Vocos custom same padding; trims `(n_fft - hop_length) / 2`.
    Same,
}

/// Shape and DSP axes for one Vocos decoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VocosAttrs {
    /// Feature channels accepted by the embedding convolution.
    pub input_channels: usize,
    /// ConvNeXt hidden width.
    pub dim: usize,
    /// Pointwise feed-forward width.
    pub intermediate_dim: usize,
    /// Number of ConvNeXt blocks.
    pub num_layers: usize,
    /// Number of AdaLayerNorm condition embeddings (`0` for plain LN).
    pub num_conditions: usize,
    /// Fourier transform and Hann-window length.
    pub n_fft: usize,
    /// iSTFT frame hop.
    pub hop_length: usize,
    /// Edge trimming convention.
    pub padding: VocosIstftPadding,
}

impl VocosAttrs {
    /// Validates all structural axes before any allocation or arithmetic.
    pub fn validate(&self) -> Result<()> {
        if self.input_channels == 0
            || self.dim == 0
            || self.intermediate_dim == 0
            || self.num_layers == 0
            || self.n_fft == 0
            || self.hop_length == 0
            || self.hop_length > self.n_fft
        {
            return Err(VokraError::InvalidArgument(format!(
                "vocos: invalid attrs {self:?}"
            )));
        }
        if self.n_fft % 2 != 0 {
            return Err(VokraError::InvalidArgument(format!(
                "vocos: n_fft must be even, got {}",
                self.n_fft
            )));
        }
        Ok(())
    }
}

/// Affine parameters for LayerNorm or bandwidth-conditioned AdaLayerNorm.
#[derive(Debug, Clone)]
pub struct VocosNormWeights {
    /// Row-major `[rows, dim]`; `rows=1` for LayerNorm.
    pub scale: Vec<f32>,
    /// Row-major `[rows, dim]`; `rows=1` for LayerNorm.
    pub shift: Vec<f32>,
}

/// Weights for one released Vocos ConvNeXt 1D block.
#[derive(Debug, Clone)]
pub struct VocosBlockWeights {
    /// Depthwise Conv1d weights `[dim, 1, 7]`.
    pub depthwise_weight: Vec<f32>,
    /// Depthwise Conv1d bias `[dim]`.
    pub depthwise_bias: Vec<f32>,
    /// LayerNorm / AdaLayerNorm affine parameters.
    pub norm: VocosNormWeights,
    /// First pointwise Linear weights `[intermediate_dim, dim]`.
    pub pointwise1_weight: Vec<f32>,
    /// First pointwise Linear bias `[intermediate_dim]`.
    pub pointwise1_bias: Vec<f32>,
    /// Second pointwise Linear weights `[dim, intermediate_dim]`.
    pub pointwise2_weight: Vec<f32>,
    /// Second pointwise Linear bias `[dim]`.
    pub pointwise2_bias: Vec<f32>,
    /// Per-channel LayerScale `[dim]`.
    pub gamma: Vec<f32>,
}

/// Complete Vocos decoder weight bundle.
#[derive(Debug, Clone)]
pub struct VocosWeights {
    /// Input Conv1d `[dim, input_channels, 7]`.
    pub embed_weight: Vec<f32>,
    /// Input Conv1d bias `[dim]`.
    pub embed_bias: Vec<f32>,
    /// Initial LayerNorm / AdaLayerNorm.
    pub norm: VocosNormWeights,
    /// Eight blocks in the released checkpoints.
    pub blocks: Vec<VocosBlockWeights>,
    /// Final plain LayerNorm gain `[dim]`.
    pub final_norm_weight: Vec<f32>,
    /// Final plain LayerNorm bias `[dim]`.
    pub final_norm_bias: Vec<f32>,
    /// iSTFT head Linear weights `[n_fft + 2, dim]`.
    pub head_weight: Vec<f32>,
    /// iSTFT head Linear bias `[n_fft + 2]`.
    pub head_bias: Vec<f32>,
}

impl VocosWeights {
    /// Checks the full tensor topology against `attrs`.
    pub fn validate(&self, attrs: &VocosAttrs) -> Result<()> {
        attrs.validate()?;
        check_len(
            "embed_weight",
            &self.embed_weight,
            attrs.dim * attrs.input_channels * 7,
        )?;
        check_len("embed_bias", &self.embed_bias, attrs.dim)?;
        validate_norm("norm", &self.norm, attrs)?;
        if self.blocks.len() != attrs.num_layers {
            return Err(VokraError::InvalidArgument(format!(
                "vocos: blocks has length {}, expected {}",
                self.blocks.len(),
                attrs.num_layers
            )));
        }
        for (index, block) in self.blocks.iter().enumerate() {
            let tag = format!("blocks[{index}]");
            check_len(
                &format!("{tag}.depthwise_weight"),
                &block.depthwise_weight,
                attrs.dim * 7,
            )?;
            check_len(
                &format!("{tag}.depthwise_bias"),
                &block.depthwise_bias,
                attrs.dim,
            )?;
            validate_norm(&format!("{tag}.norm"), &block.norm, attrs)?;
            check_len(
                &format!("{tag}.pointwise1_weight"),
                &block.pointwise1_weight,
                attrs.intermediate_dim * attrs.dim,
            )?;
            check_len(
                &format!("{tag}.pointwise1_bias"),
                &block.pointwise1_bias,
                attrs.intermediate_dim,
            )?;
            check_len(
                &format!("{tag}.pointwise2_weight"),
                &block.pointwise2_weight,
                attrs.dim * attrs.intermediate_dim,
            )?;
            check_len(
                &format!("{tag}.pointwise2_bias"),
                &block.pointwise2_bias,
                attrs.dim,
            )?;
            check_len(&format!("{tag}.gamma"), &block.gamma, attrs.dim)?;
        }
        check_len("final_norm_weight", &self.final_norm_weight, attrs.dim)?;
        check_len("final_norm_bias", &self.final_norm_bias, attrs.dim)?;
        let head = attrs.n_fft + 2;
        check_len("head_weight", &self.head_weight, head * attrs.dim)?;
        check_len("head_bias", &self.head_bias, head)?;
        Ok(())
    }
}

/// Backend seam for Vocos' learned ConvNeXt operations.
///
/// The public CPU decoder below uses a scalar implementation that preserves
/// its original arithmetic. Model binders may supply a GPU implementation;
/// DSP-only magnitude/phase assembly and iSTFT deliberately remain outside
/// this trait.
pub trait VocosBackendOps {
    /// Dense same-padded channel-major Conv1d.
    #[allow(clippy::too_many_arguments)]
    fn conv1d_same(
        &self,
        input: &[f32],
        input_channels: usize,
        frames: usize,
        output_channels: usize,
        weight: &[f32],
        bias: &[f32],
        kernel: usize,
    ) -> Result<Vec<f32>>;

    /// Depthwise same-padded channel-major Conv1d.
    #[allow(clippy::too_many_arguments)]
    fn depthwise_conv1d_same(
        &self,
        input: &[f32],
        channels: usize,
        frames: usize,
        weight: &[f32],
        bias: &[f32],
        kernel: usize,
    ) -> Result<Vec<f32>>;

    /// LayerNorm over channels independently for every frame.
    fn norm_channel_major(
        &self,
        values: &mut [f32],
        frames: usize,
        dim: usize,
        scale: &[f32],
        shift: &[f32],
    ) -> Result<()>;

    /// A per-frame linear projection represented as a kernel-size-one Conv1d.
    #[allow(clippy::too_many_arguments)]
    fn pointwise(
        &self,
        input: &[f32],
        input_dim: usize,
        frames: usize,
        output_dim: usize,
        weight: &[f32],
        bias: &[f32],
    ) -> Result<Vec<f32>>;

    /// Exact GELU over an arbitrary flat buffer.
    fn gelu_in_place(&self, values: &mut [f32]) -> Result<()>;
}

struct ScalarVocosOps;

impl VocosBackendOps for ScalarVocosOps {
    fn conv1d_same(
        &self,
        input: &[f32],
        input_channels: usize,
        frames: usize,
        output_channels: usize,
        weight: &[f32],
        bias: &[f32],
        kernel: usize,
    ) -> Result<Vec<f32>> {
        Ok(conv1d_same(
            input,
            input_channels,
            frames,
            output_channels,
            weight,
            bias,
            kernel,
        ))
    }

    fn depthwise_conv1d_same(
        &self,
        input: &[f32],
        channels: usize,
        frames: usize,
        weight: &[f32],
        bias: &[f32],
        kernel: usize,
    ) -> Result<Vec<f32>> {
        Ok(depthwise_conv1d_same(
            input, channels, frames, weight, bias, kernel,
        ))
    }

    fn norm_channel_major(
        &self,
        values: &mut [f32],
        frames: usize,
        dim: usize,
        scale: &[f32],
        shift: &[f32],
    ) -> Result<()> {
        norm_channel_major_plain(values, frames, dim, scale, shift);
        Ok(())
    }

    fn pointwise(
        &self,
        input: &[f32],
        input_dim: usize,
        frames: usize,
        output_dim: usize,
        weight: &[f32],
        bias: &[f32],
    ) -> Result<Vec<f32>> {
        let mut output = vec![0.0f32; output_dim * frames];
        let mut row = vec![0.0f32; input_dim];
        let mut projected = vec![0.0f32; output_dim];
        for frame in 0..frames {
            for channel in 0..input_dim {
                row[channel] = input[channel * frames + frame];
            }
            linear_row(&row, weight, bias, output_dim, input_dim, &mut projected);
            for channel in 0..output_dim {
                output[channel * frames + frame] = projected[channel];
            }
        }
        Ok(output)
    }

    fn gelu_in_place(&self, values: &mut [f32]) -> Result<()> {
        for value in values {
            *value = gelu_exact(*value);
        }
        Ok(())
    }
}

/// Runs the released Vocos feature-to-waveform decoder.
///
/// `condition_id` is required for an AdaLayerNorm model and forbidden for a
/// plain LayerNorm model.  The released Encodec model accepts ids `0..4`,
/// corresponding to bandwidths 1.5, 3.0, 6.0 and 12.0 kbps.
pub fn vocos_decode(
    features: &[f32],
    frames: usize,
    condition_id: Option<usize>,
    weights: &VocosWeights,
    attrs: &VocosAttrs,
) -> Result<Vec<f32>> {
    vocos_decode_with_ops(
        features,
        frames,
        condition_id,
        weights,
        attrs,
        &ScalarVocosOps,
    )
}

/// Runs Vocos with a caller-supplied backend for every learned operation.
pub fn vocos_decode_with_ops<O: VocosBackendOps>(
    features: &[f32],
    frames: usize,
    condition_id: Option<usize>,
    weights: &VocosWeights,
    attrs: &VocosAttrs,
    ops: &O,
) -> Result<Vec<f32>> {
    if frames == 0 {
        return Err(VokraError::InvalidArgument(
            "vocos: frames must be positive".to_owned(),
        ));
    }
    check_len("features", features, attrs.input_channels * frames)?;
    let x = ops.conv1d_same(
        features,
        attrs.input_channels,
        frames,
        attrs.dim,
        &weights.embed_weight,
        &weights.embed_bias,
        7,
    )?;
    vocos_decode_from_embedded_with_ops(x, frames, condition_id, weights, attrs, ops)
}

/// Runs the shared Vocos normalization, ConvNeXt and iSTFT head from an
/// already embedded channel-major `[dim, frames]` activation.
///
/// WavTokenizer inserts its released positional ResNet/attention stack
/// between `backbone.embed` and `backbone.norm`; this entry keeps the common
/// downstream arithmetic in one implementation while allowing that exact
/// upstream ordering. The caller must route the embedding and any inserted
/// learned operations through the same selected backend before calling this
/// function.
pub fn vocos_decode_from_embedded_with_ops<O: VocosBackendOps>(
    mut x: Vec<f32>,
    frames: usize,
    condition_id: Option<usize>,
    weights: &VocosWeights,
    attrs: &VocosAttrs,
    ops: &O,
) -> Result<Vec<f32>> {
    weights.validate(attrs)?;
    if frames == 0 {
        return Err(VokraError::InvalidArgument(
            "vocos: frames must be positive".to_owned(),
        ));
    }
    check_len("embedded", &x, attrs.dim * frames)?;
    let condition = match (attrs.num_conditions, condition_id) {
        (0, None) => 0,
        (0, Some(id)) => {
            return Err(VokraError::InvalidArgument(format!(
                "vocos: condition_id {id} supplied to a non-conditional model"
            )));
        }
        (rows, Some(id)) if id < rows => id,
        (rows, Some(id)) => {
            return Err(VokraError::InvalidArgument(format!(
                "vocos: condition_id {id} out of range 0..{rows}"
            )));
        }
        (rows, None) => {
            return Err(VokraError::InvalidArgument(format!(
                "vocos: conditional model requires condition_id in 0..{rows}"
            )));
        }
    };

    norm_channel_major_with_ops(ops, &mut x, frames, attrs.dim, &weights.norm, condition)?;

    for block in &weights.blocks {
        let residual = x.clone();
        x = ops.depthwise_conv1d_same(
            &x,
            attrs.dim,
            frames,
            &block.depthwise_weight,
            &block.depthwise_bias,
            7,
        )?;
        norm_channel_major_with_ops(ops, &mut x, frames, attrs.dim, &block.norm, condition)?;

        let mut hidden = ops.pointwise(
            &x,
            attrs.dim,
            frames,
            attrs.intermediate_dim,
            &block.pointwise1_weight,
            &block.pointwise1_bias,
        )?;
        ops.gelu_in_place(&mut hidden)?;
        let output = ops.pointwise(
            &hidden,
            attrs.intermediate_dim,
            frames,
            attrs.dim,
            &block.pointwise2_weight,
            &block.pointwise2_bias,
        )?;
        for channel in 0..attrs.dim {
            for frame in 0..frames {
                x[channel * frames + frame] = residual[channel * frames + frame]
                    + block.gamma[channel] * output[channel * frames + frame];
            }
        }
    }

    ops.norm_channel_major(
        &mut x,
        frames,
        attrs.dim,
        &weights.final_norm_weight,
        &weights.final_norm_bias,
    )?;

    let bins = attrs.n_fft / 2 + 1;
    let head_dim = attrs.n_fft + 2;
    let mut re = vec![0.0f32; frames * bins];
    let mut im = vec![0.0f32; frames * bins];
    let projected = ops.pointwise(
        &x,
        attrs.dim,
        frames,
        head_dim,
        &weights.head_weight,
        &weights.head_bias,
    )?;
    for frame in 0..frames {
        for bin in 0..bins {
            let magnitude = projected[bin * frames + frame].exp().min(100.0);
            let phase = projected[(bins + bin) * frames + frame];
            re[frame * bins + bin] = magnitude * phase.cos();
            im[frame * bins + bin] = magnitude * phase.sin();
        }
    }

    let spectrogram = Spectrogram {
        frames,
        bins,
        re,
        im,
    };
    let mut istft_attrs = IstftAttrs::new(attrs.n_fft, attrs.hop_length);
    istft_attrs.center = matches!(attrs.padding, VocosIstftPadding::Center);
    let mut pcm = istft(&spectrogram, &istft_attrs)?;
    if matches!(attrs.padding, VocosIstftPadding::Same) {
        let trim = (attrs.n_fft - attrs.hop_length) / 2;
        if 2 * trim > pcm.len() {
            return Err(VokraError::InvalidArgument(format!(
                "vocos: same-padding trim {trim} exceeds iSTFT output length {}",
                pcm.len()
            )));
        }
        pcm = pcm[trim..pcm.len() - trim].to_vec();
    }
    Ok(pcm)
}

fn check_len(name: &str, values: &[f32], expected: usize) -> Result<()> {
    if values.len() != expected {
        return Err(VokraError::InvalidArgument(format!(
            "vocos: {name} has length {}, expected {expected}",
            values.len()
        )));
    }
    Ok(())
}

fn validate_norm(name: &str, norm: &VocosNormWeights, attrs: &VocosAttrs) -> Result<()> {
    let rows = attrs.num_conditions.max(1);
    check_len(&format!("{name}.scale"), &norm.scale, rows * attrs.dim)?;
    check_len(&format!("{name}.shift"), &norm.shift, rows * attrs.dim)
}

fn conv1d_same(
    input: &[f32],
    input_channels: usize,
    frames: usize,
    output_channels: usize,
    weight: &[f32],
    bias: &[f32],
    kernel: usize,
) -> Vec<f32> {
    let pad = kernel / 2;
    let mut output = vec![0.0f32; output_channels * frames];
    for oc in 0..output_channels {
        for frame in 0..frames {
            let mut sum = bias[oc];
            for ic in 0..input_channels {
                for tap in 0..kernel {
                    let source = frame as isize + tap as isize - pad as isize;
                    if source >= 0 && source < frames as isize {
                        sum += weight[(oc * input_channels + ic) * kernel + tap]
                            * input[ic * frames + source as usize];
                    }
                }
            }
            output[oc * frames + frame] = sum;
        }
    }
    output
}

fn depthwise_conv1d_same(
    input: &[f32],
    channels: usize,
    frames: usize,
    weight: &[f32],
    bias: &[f32],
    kernel: usize,
) -> Vec<f32> {
    let pad = kernel / 2;
    let mut output = vec![0.0f32; channels * frames];
    for channel in 0..channels {
        for frame in 0..frames {
            let mut sum = bias[channel];
            for tap in 0..kernel {
                let source = frame as isize + tap as isize - pad as isize;
                if source >= 0 && source < frames as isize {
                    sum +=
                        weight[channel * kernel + tap] * input[channel * frames + source as usize];
                }
            }
            output[channel * frames + frame] = sum;
        }
    }
    output
}

fn norm_channel_major_with_ops<O: VocosBackendOps>(
    ops: &O,
    values: &mut [f32],
    frames: usize,
    dim: usize,
    norm: &VocosNormWeights,
    row: usize,
) -> Result<()> {
    let scale = &norm.scale[row * dim..(row + 1) * dim];
    let shift = &norm.shift[row * dim..(row + 1) * dim];
    ops.norm_channel_major(values, frames, dim, scale, shift)
}

fn norm_channel_major_plain(
    values: &mut [f32],
    frames: usize,
    dim: usize,
    scale: &[f32],
    shift: &[f32],
) {
    const EPS: f32 = 1e-6;
    for frame in 0..frames {
        let mut mean = 0.0f32;
        for channel in 0..dim {
            mean += values[channel * frames + frame];
        }
        mean /= dim as f32;
        let mut variance = 0.0f32;
        for channel in 0..dim {
            let delta = values[channel * frames + frame] - mean;
            variance += delta * delta;
        }
        variance /= dim as f32;
        let inverse = (variance + EPS).sqrt().recip();
        for channel in 0..dim {
            let index = channel * frames + frame;
            values[index] = (values[index] - mean) * inverse * scale[channel] + shift[channel];
        }
    }
}

fn linear_row(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    output_dim: usize,
    input_dim: usize,
    output: &mut [f32],
) {
    for out in 0..output_dim {
        let mut sum = bias[out];
        let row = &weight[out * input_dim..(out + 1) * input_dim];
        for index in 0..input_dim {
            sum += row[index] * input[index];
        }
        output[out] = sum;
    }
}

fn gelu_exact(value: f32) -> f32 {
    let x = f64::from(value);
    (0.5 * x * (1.0 + erf(x / std::f64::consts::SQRT_2))) as f32
}

fn erf(value: f64) -> f64 {
    const P: f64 = 0.327_591_1;
    const A1: f64 = 0.254_829_592;
    const A2: f64 = -0.284_496_736;
    const A3: f64 = 1.421_413_741;
    const A4: f64 = -1.453_152_027;
    const A5: f64 = 1.061_405_429;
    let sign = if value < 0.0 { -1.0 } else { 1.0 };
    let x = value.abs();
    let t = 1.0 / (1.0 + P * x);
    let polynomial = t * (A1 + t * (A2 + t * (A3 + t * (A4 + t * A5))));
    sign * (1.0 - polynomial * (-x * x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny() -> (VocosAttrs, VocosWeights) {
        let attrs = VocosAttrs {
            input_channels: 2,
            dim: 2,
            intermediate_dim: 3,
            num_layers: 1,
            num_conditions: 0,
            n_fft: 8,
            hop_length: 2,
            padding: VocosIstftPadding::Center,
        };
        let norm = VocosNormWeights {
            scale: vec![1.0; 2],
            shift: vec![0.0; 2],
        };
        let weights = VocosWeights {
            embed_weight: vec![0.0; 2 * 2 * 7],
            embed_bias: vec![0.0; 2],
            norm: norm.clone(),
            blocks: vec![VocosBlockWeights {
                depthwise_weight: vec![0.0; 2 * 7],
                depthwise_bias: vec![0.0; 2],
                norm: norm.clone(),
                pointwise1_weight: vec![0.0; 3 * 2],
                pointwise1_bias: vec![0.0; 3],
                pointwise2_weight: vec![0.0; 2 * 3],
                pointwise2_bias: vec![0.0; 2],
                gamma: vec![0.125; 2],
            }],
            final_norm_weight: vec![1.0; 2],
            final_norm_bias: vec![0.0; 2],
            head_weight: vec![0.0; 10 * 2],
            head_bias: vec![0.0; 10],
        };
        (attrs, weights)
    }

    #[test]
    fn zero_fixture_runs_and_has_center_length() {
        let (attrs, weights) = tiny();
        let pcm = vocos_decode(&[0.0; 8], 4, None, &weights, &attrs).unwrap();
        assert_eq!(pcm.len(), 6);
        assert!(pcm.iter().all(|sample| sample.is_finite()));
    }

    #[test]
    fn conditional_contract_is_loud() {
        let (mut attrs, mut weights) = tiny();
        attrs.num_conditions = 4;
        weights.norm.scale = vec![1.0; 8];
        weights.norm.shift = vec![0.0; 8];
        weights.blocks[0].norm.scale = vec![1.0; 8];
        weights.blocks[0].norm.shift = vec![0.0; 8];
        let err = vocos_decode(&[0.0; 8], 4, None, &weights, &attrs).unwrap_err();
        assert!(err.to_string().contains("requires condition_id"));
        assert!(vocos_decode(&[0.0; 8], 4, Some(3), &weights, &attrs).is_ok());
        assert!(vocos_decode(&[0.0; 8], 4, Some(4), &weights, &attrs).is_err());
    }

    #[test]
    fn same_padding_emits_one_hop_per_frame() {
        let (mut attrs, weights) = tiny();
        attrs.padding = VocosIstftPadding::Same;
        let pcm = vocos_decode(&[0.0; 8], 4, None, &weights, &attrs).unwrap();
        assert_eq!(pcm.len(), 8);
    }
}
