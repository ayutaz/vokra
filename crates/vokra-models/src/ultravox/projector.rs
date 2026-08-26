//! Exact Ultravox frame-stack + SwiGLU projector.

use std::sync::Mutex;

use vokra_core::{Result, VokraError};

use crate::compute::Compute;
use crate::mapped_weights::{lock_scratch, transpose_widen, widen_into};

use super::weights::UltravoxMappedDescriptors;

const RMS_NORM_EPS: f32 = 1.0e-6;

/// Audio-prefix embeddings ready to replace Ultravox audio placeholder tokens.
#[derive(Debug, Clone, PartialEq)]
pub struct UltravoxAudioEmbeddings {
    values: Vec<f32>,
    frames: usize,
    hidden_size: usize,
}

impl UltravoxAudioEmbeddings {
    /// Row-major `[frames, hidden_size]` values.
    #[must_use]
    pub fn values(&self) -> &[f32] {
        &self.values
    }

    /// Number of projected audio-prefix rows.
    #[must_use]
    pub const fn frames(&self) -> usize {
        self.frames
    }

    /// Embedding width expected by the Llama companion.
    #[must_use]
    pub const fn hidden_size(&self) -> usize {
        self.hidden_size
    }
}

#[derive(Default)]
pub(super) struct ProjectorRuntime {
    weights: Mutex<ProjectorWeights>,
}

#[derive(Default)]
struct ProjectorWeights {
    ready: bool,
    norm_pre: Vec<f32>,
    linear_1_t: Vec<f32>,
    norm_mid: Vec<f32>,
    linear_2_t: Vec<f32>,
}

pub(super) fn project(
    compute: &Compute,
    mapped: &UltravoxMappedDescriptors,
    runtime: &ProjectorRuntime,
    encoder_hidden: &[f32],
    encoder_frames: usize,
) -> Result<UltravoxAudioEmbeddings> {
    let config = mapped.config();
    if encoder_frames == 0 || encoder_hidden.len() != encoder_frames * config.hidden_size {
        return Err(VokraError::InvalidArgument(format!(
            "ultravox projector: hidden len {} is not {encoder_frames}x{}",
            encoder_hidden.len(),
            config.hidden_size
        )));
    }
    let mut weights = lock_scratch(&runtime.weights, mapped.mapped_model())?;
    if !weights.ready {
        materialize(mapped, &mut weights)?;
        weights.ready = true;
    }

    let (stacked, rows) = stack_frames(
        encoder_hidden,
        encoder_frames,
        config.hidden_size,
        config.stack_factor,
    )?;
    let mut normalized = vec![0.0; stacked.len()];
    compute.rms_norm_f32(
        &stacked,
        &mut normalized,
        rows,
        config.stacked_size,
        &weights.norm_pre,
        RMS_NORM_EPS,
    )?;

    let mut packed = vec![0.0; rows * config.projector_packed_size];
    compute.gemm_f32(
        rows,
        config.projector_packed_size,
        config.stacked_size,
        &normalized,
        &weights.linear_1_t,
        None,
        &mut packed,
    )?;
    let gated = swiglu(compute, &packed, rows, config.text_hidden_size)?;
    let mut mid = vec![0.0; gated.len()];
    compute.rms_norm_f32(
        &gated,
        &mut mid,
        rows,
        config.text_hidden_size,
        &weights.norm_mid,
        RMS_NORM_EPS,
    )?;
    let mut values = vec![0.0; rows * config.text_hidden_size];
    compute.gemm_f32(
        rows,
        config.text_hidden_size,
        config.text_hidden_size,
        &mid,
        &weights.linear_2_t,
        None,
        &mut values,
    )?;
    reject_non_finite(&values)?;
    Ok(UltravoxAudioEmbeddings {
        values,
        frames: rows,
        hidden_size: config.text_hidden_size,
    })
}

fn materialize(mapped: &UltravoxMappedDescriptors, output: &mut ProjectorWeights) -> Result<()> {
    let descriptors = &mapped.projector;
    let config = mapped.config();
    widen_into(
        mapped.file().tensor_bytes(&descriptors.norm_pre),
        descriptors.norm_pre.dtype,
        &mut output.norm_pre,
        mapped.mapped_model(),
    )?;
    transpose_widen(
        mapped.file().tensor_bytes(&descriptors.linear_1),
        descriptors.linear_1.dtype,
        config.projector_packed_size,
        config.stacked_size,
        &mut output.linear_1_t,
        mapped.mapped_model(),
    )?;
    widen_into(
        mapped.file().tensor_bytes(&descriptors.norm_mid),
        descriptors.norm_mid.dtype,
        &mut output.norm_mid,
        mapped.mapped_model(),
    )?;
    transpose_widen(
        mapped.file().tensor_bytes(&descriptors.linear_2),
        descriptors.linear_2.dtype,
        config.text_hidden_size,
        config.text_hidden_size,
        &mut output.linear_2_t,
        mapped.mapped_model(),
    )
}

fn stack_frames(
    input: &[f32],
    frames: usize,
    width: usize,
    factor: usize,
) -> Result<(Vec<f32>, usize)> {
    if frames == 0 || width == 0 || factor == 0 || input.len() != frames.saturating_mul(width) {
        return Err(VokraError::InvalidArgument(format!(
            "ultravox projector: invalid stack shape input={}, frames={frames}, width={width}, factor={factor}",
            input.len()
        )));
    }
    let rows = frames.div_ceil(factor);
    let row_width = width.checked_mul(factor).ok_or_else(|| {
        VokraError::InvalidArgument("ultravox projector: stacked width overflow".to_owned())
    })?;
    let mut output = vec![0.0; rows * row_width];
    for frame in 0..frames {
        let row = frame / factor;
        let within = frame % factor;
        let target = row * row_width + within * width;
        output[target..target + width].copy_from_slice(&input[frame * width..(frame + 1) * width]);
    }
    Ok((output, rows))
}

fn swiglu(compute: &Compute, packed: &[f32], rows: usize, width: usize) -> Result<Vec<f32>> {
    if rows == 0 || width == 0 || packed.len() != rows.saturating_mul(2 * width) {
        return Err(VokraError::InvalidArgument(format!(
            "ultravox projector: packed SwiGLU len {} != {rows}x{}",
            packed.len(),
            2 * width
        )));
    }
    let mut value = vec![0.0; rows * width];
    let mut gate = vec![0.0; rows * width];
    for row in 0..rows {
        let packed_start = row * 2 * width;
        let output_start = row * width;
        value[output_start..output_start + width]
            .copy_from_slice(&packed[packed_start..packed_start + width]);
        gate[output_start..output_start + width]
            .copy_from_slice(&packed[packed_start + width..packed_start + 2 * width]);
    }
    let mut activated = vec![0.0; gate.len()];
    compute.silu_f32(&gate, &mut activated)?;
    for (value, gate) in value.iter_mut().zip(activated) {
        *value *= gate;
    }
    Ok(value)
}

fn reject_non_finite(values: &[f32]) -> Result<()> {
    if let Some((index, value)) = values
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(VokraError::InvalidArgument(format!(
            "ultravox projector: output contains non-finite value {value} at index {index}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stack_zero_pads_only_the_final_incomplete_group() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let (stacked, rows) = stack_frames(&input, 3, 2, 2).unwrap();
        assert_eq!(rows, 2);
        assert_eq!(stacked, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 0.0]);
    }

    #[test]
    fn swiglu_chunks_each_row_as_value_then_gate() {
        let compute = Compute::cpu();
        let packed = [2.0, 3.0, 0.0, 1.0, 4.0, 5.0, -1.0, 0.0];
        let output = swiglu(&compute, &packed, 2, 2).unwrap();
        assert_eq!(output[0], 0.0);
        assert!((output[1] - 3.0 / (1.0 + (-1.0_f32).exp())).abs() < 1.0e-6);
        assert!((output[2] - 4.0 * (-1.0 / (1.0 + 1.0_f32.exp()))).abs() < 1.0e-6);
        assert_eq!(output[3], 0.0);
    }
}
