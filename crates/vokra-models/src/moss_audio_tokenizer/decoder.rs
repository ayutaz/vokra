use vokra_core::backend::BackendKind;
use vokra_core::gguf::GgufFile;
use vokra_core::{Result, VokraError};

use crate::compute::Compute;

use super::MOSS_AUDIO_TOKENIZER_NANO_HOT_OPS;
use super::weights::{
    CODEBOOK_DIM, CODEBOOK_SIZE, MODEL_DIM, NUM_HEADS, NUM_QUANTIZERS, NanoWeights, StageWeights,
};

const LABEL: &str = "moss_audio_tokenizer/nano";
const PATCH_FACTORS: [usize; 5] = [4, 2, 2, 2, 240];
const HEAD_DIM: usize = MODEL_DIM / NUM_HEADS;
const LAYER_NORM_EPS: f32 = 1.0e-5;
const ROPE_MAX_PERIOD: f32 = 10_000.0;

#[derive(Debug, Clone)]
pub(super) struct NanoDecoder {
    weights: NanoWeights,
}

impl NanoDecoder {
    pub(super) fn bind(file: &GgufFile) -> Result<Self> {
        Ok(Self {
            weights: NanoWeights::bind(file)?,
        })
    }

    pub(super) fn decode_frame_major(
        &self,
        backend: BackendKind,
        codes: &[u32],
        frames: usize,
        num_quantizers: usize,
    ) -> Result<Vec<f32>> {
        validate_codes(codes, frames, num_quantizers)?;
        let compute = Compute::for_backend(backend, MOSS_AUDIO_TOKENIZER_NANO_HOT_OPS)?;
        let mut values = self.quantizer_decode(&compute, codes, frames, num_quantizers)?;
        let mut time = frames;

        (values, time) = patch_up(&values, time, PATCH_FACTORS[0])?;
        for (index, stage) in self.weights.stages.iter().enumerate() {
            values = transformer_stage(&compute, &values, time, stage)?;
            (values, time) = patch_up(&values, time, PATCH_FACTORS[index + 1])?;
        }
        if values.len() != time {
            return Err(VokraError::InvalidArgument(format!(
                "{LABEL}: final patched decoder has {} values for {time} scalar interleaved samples",
                values.len()
            )));
        }
        reject_non_finite("decoded interleaved waveform", &values)?;
        Ok(values)
    }

    fn quantizer_decode(
        &self,
        compute: &Compute,
        codes: &[u32],
        frames: usize,
        num_quantizers: usize,
    ) -> Result<Vec<f32>> {
        let quantizer = &self.weights.quantizer;
        let mut rvq = vec![0.0f32; frames * super::weights::RVQ_DIM];
        let mut embedded = vec![0.0f32; frames * CODEBOOK_DIM];
        for quantizer_index in 0..num_quantizers {
            let table = &quantizer.codebooks[quantizer_index];
            for frame in 0..frames {
                let code = codes[frame * num_quantizers + quantizer_index] as usize;
                embedded[frame * CODEBOOK_DIM..(frame + 1) * CODEBOOK_DIM]
                    .copy_from_slice(&table[code * CODEBOOK_DIM..(code + 1) * CODEBOOK_DIM]);
            }
            let projected =
                quantizer.projections[quantizer_index].forward(compute, &embedded, frames)?;
            for (sum, value) in rvq.iter_mut().zip(projected) {
                *sum += value;
            }
        }
        quantizer.output_projection.forward(compute, &rvq, frames)
    }
}

fn transformer_stage(
    compute: &Compute,
    input: &[f32],
    time: usize,
    weights: &StageWeights,
) -> Result<Vec<f32>> {
    let mut hidden = weights.input_projection.forward(compute, input, time)?;
    for layer in &weights.layers {
        let mut normalized = vec![0.0f32; hidden.len()];
        compute.layer_norm_f32(
            &hidden,
            &mut normalized,
            time,
            MODEL_DIM,
            &layer.norm1_weight,
            &layer.norm1_bias,
            LAYER_NORM_EPS,
        )?;
        let attention = attention(
            compute,
            &normalized,
            time,
            weights.spec.context,
            &layer.attention_in,
            &layer.attention_out,
        )?;
        add_scaled_residual(&mut hidden, &attention, &layer.layer_scale1, MODEL_DIM)?;

        compute.layer_norm_f32(
            &hidden,
            &mut normalized,
            time,
            MODEL_DIM,
            &layer.norm2_weight,
            &layer.norm2_bias,
            LAYER_NORM_EPS,
        )?;
        let mut feed_forward = layer.ffn_in.forward(compute, &normalized, time)?;
        let mut activated = vec![0.0f32; feed_forward.len()];
        compute.gelu_f32(&feed_forward, &mut activated)?;
        feed_forward = layer.ffn_out.forward(compute, &activated, time)?;
        add_scaled_residual(&mut hidden, &feed_forward, &layer.layer_scale2, MODEL_DIM)?;
    }
    weights.output_projection.forward(compute, &hidden, time)
}

fn attention(
    compute: &Compute,
    input: &[f32],
    time: usize,
    context: usize,
    input_projection: &super::weights::Linear,
    output_projection: &super::weights::Linear,
) -> Result<Vec<f32>> {
    let projected = input_projection.forward(compute, input, time)?;
    let mut joined = vec![0.0f32; time * MODEL_DIM];
    let scale = (HEAD_DIM as f32).sqrt().recip();

    for head in 0..NUM_HEADS {
        let mut query = vec![0.0f32; time * HEAD_DIM];
        let mut key = vec![0.0f32; time * HEAD_DIM];
        let mut value = vec![0.0f32; time * HEAD_DIM];
        for position in 0..time {
            let source = position * MODEL_DIM * 3 + head * HEAD_DIM;
            let target = position * HEAD_DIM;
            query[target..target + HEAD_DIM].copy_from_slice(&projected[source..source + HEAD_DIM]);
            key[target..target + HEAD_DIM]
                .copy_from_slice(&projected[source + MODEL_DIM..source + MODEL_DIM + HEAD_DIM]);
            value[target..target + HEAD_DIM].copy_from_slice(
                &projected[source + MODEL_DIM * 2..source + MODEL_DIM * 2 + HEAD_DIM],
            );
        }
        apply_rope(&mut query, &mut key, time, HEAD_DIM)?;

        let mut key_t = vec![0.0f32; HEAD_DIM * time];
        for position in 0..time {
            for dimension in 0..HEAD_DIM {
                key_t[dimension * time + position] = key[position * HEAD_DIM + dimension];
            }
        }
        let mut logits = vec![0.0f32; time * time];
        compute.gemm_f32(time, time, HEAD_DIM, &query, &key_t, None, &mut logits)?;
        apply_local_causal_mask(&mut logits, time, context, scale)?;
        let mut probabilities = vec![0.0f32; logits.len()];
        compute.softmax_f32(&logits, &mut probabilities, time, time)?;
        let mut attended = vec![0.0f32; time * HEAD_DIM];
        compute.gemm_f32(
            time,
            HEAD_DIM,
            time,
            &probabilities,
            &value,
            None,
            &mut attended,
        )?;
        for position in 0..time {
            let source = position * HEAD_DIM;
            let target = position * MODEL_DIM + head * HEAD_DIM;
            joined[target..target + HEAD_DIM].copy_from_slice(&attended[source..source + HEAD_DIM]);
        }
    }
    output_projection.forward(compute, &joined, time)
}

fn patch_up(input: &[f32], time: usize, patch: usize) -> Result<(Vec<f32>, usize)> {
    if time == 0 || patch == 0 || input.len() % time != 0 {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: patch-up shape mismatch: values={}, time={time}, patch={patch}",
            input.len()
        )));
    }
    let packed_channels = input.len() / time;
    if packed_channels % patch != 0 {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: patch-up channels {packed_channels} are not divisible by patch {patch}"
        )));
    }
    let channels = packed_channels / patch;
    let output_time = time.checked_mul(patch).ok_or_else(|| {
        VokraError::InvalidArgument(format!(
            "{LABEL}: patch-up time overflows: {time} * {patch}"
        ))
    })?;
    let output_len = output_time.checked_mul(channels).ok_or_else(|| {
        VokraError::InvalidArgument(format!(
            "{LABEL}: patch-up output shape overflows: {output_time} * {channels}"
        ))
    })?;
    let mut output = vec![0.0f32; output_len];
    for position in 0..time {
        for channel in 0..channels {
            for subframe in 0..patch {
                output[(position * patch + subframe) * channels + channel] =
                    input[position * packed_channels + channel * patch + subframe];
            }
        }
    }
    Ok((output, output_time))
}

fn apply_rope(query: &mut [f32], key: &mut [f32], time: usize, dim: usize) -> Result<()> {
    if dim == 0 || !dim.is_multiple_of(2) || query.len() != time * dim || key.len() != query.len() {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: RoPE shape mismatch: q={}, k={}, time={time}, dim={dim}",
            query.len(),
            key.len()
        )));
    }
    for position in 0..time {
        for pair in 0..dim / 2 {
            let frequency = ((pair as f32) * (-ROPE_MAX_PERIOD.ln() * 2.0 / dim as f32)).exp();
            let angle = position as f32 * frequency;
            let (sin, cos) = angle.sin_cos();
            let index = position * dim + pair * 2;
            let q_real = query[index];
            let q_imag = query[index + 1];
            let k_real = key[index];
            let k_imag = key[index + 1];
            query[index] = q_real * cos - q_imag * sin;
            query[index + 1] = q_real * sin + q_imag * cos;
            key[index] = k_real * cos - k_imag * sin;
            key[index + 1] = k_real * sin + k_imag * cos;
        }
    }
    Ok(())
}

fn apply_local_causal_mask(
    logits: &mut [f32],
    time: usize,
    context: usize,
    scale: f32,
) -> Result<()> {
    if time == 0 || context == 0 || logits.len() != time * time || !scale.is_finite() {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: attention-mask shape mismatch: logits={}, time={time}, context={context}, scale={scale}",
            logits.len()
        )));
    }
    for query in 0..time {
        for key in 0..time {
            let delta = query.wrapping_sub(key);
            let value = &mut logits[query * time + key];
            if key > query || delta >= context {
                *value = f32::NEG_INFINITY;
            } else {
                *value *= scale;
            }
        }
    }
    Ok(())
}

fn add_scaled_residual(
    residual: &mut [f32],
    update: &[f32],
    scale: &[f32],
    dim: usize,
) -> Result<()> {
    if residual.len() != update.len() || dim == 0 || residual.len() % dim != 0 || scale.len() != dim
    {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: scaled residual shape mismatch: residual={}, update={}, scale={}, dim={dim}",
            residual.len(),
            update.len(),
            scale.len()
        )));
    }
    for (index, (residual, update)) in residual.iter_mut().zip(update).enumerate() {
        *residual += update * scale[index % dim];
    }
    Ok(())
}

fn validate_codes(codes: &[u32], frames: usize, num_quantizers: usize) -> Result<()> {
    if frames == 0 {
        return Err(VokraError::InvalidArgument(
            "moss_audio_tokenizer/nano: frames must be > 0".to_owned(),
        ));
    }
    if !(1..=NUM_QUANTIZERS).contains(&num_quantizers) {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: num_quantizers {num_quantizers} is outside 1..={NUM_QUANTIZERS}"
        )));
    }
    let expected = frames.checked_mul(num_quantizers).ok_or_else(|| {
        VokraError::InvalidArgument(format!(
            "{LABEL}: frames * num_quantizers overflows: {frames} * {num_quantizers}"
        ))
    })?;
    if codes.len() != expected {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: codes.len() {} != frames {frames} * num_quantizers {num_quantizers} = {expected}",
            codes.len()
        )));
    }
    if let Some((index, code)) = codes
        .iter()
        .copied()
        .enumerate()
        .find(|(_, code)| *code as usize >= CODEBOOK_SIZE)
    {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: codes[{index}]={code} is outside 0..{CODEBOOK_SIZE}"
        )));
    }
    Ok(())
}

fn reject_non_finite(label: &str, values: &[f32]) -> Result<()> {
    if let Some((index, value)) = values
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: {label}[{index}] is not finite ({value})"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn patch_up_matches_upstream_reshape_permute() {
        let (output, time) =
            patch_up(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 2, 2).expect("valid patch");
        assert_eq!(time, 4);
        assert_eq!(output, vec![1.0, 3.0, 2.0, 4.0, 5.0, 7.0, 6.0, 8.0]);
    }

    #[test]
    fn local_causal_mask_keeps_diagonal_and_bounded_history() {
        let mut logits = vec![1.0f32; 16];
        apply_local_causal_mask(&mut logits, 4, 2, 0.5).expect("valid mask");
        assert_eq!(logits[3 * 4 + 3], 0.5);
        assert_eq!(logits[3 * 4 + 2], 0.5);
        assert_eq!(logits[3 * 4 + 1], f32::NEG_INFINITY);
        assert_eq!(logits[6], f32::NEG_INFINITY);
    }

    #[test]
    fn rope_position_zero_is_identity() {
        let mut query = vec![1.0, 2.0, 3.0, 4.0];
        let mut key = vec![5.0, 6.0, 7.0, 8.0];
        apply_rope(&mut query, &mut key, 1, 4).expect("valid RoPE");
        assert_eq!(query, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(key, vec![5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn code_validation_is_explicit() {
        assert!(matches!(
            validate_codes(&[CODEBOOK_SIZE as u32], 1, 1),
            Err(VokraError::InvalidArgument(_))
        ));
        assert!(matches!(
            validate_codes(&[], 0, 1),
            Err(VokraError::InvalidArgument(_))
        ));
    }
}
