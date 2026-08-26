use vokra_core::{Result, VokraError};

use crate::compute::Compute;

use super::weights::{BlockWeights, HEAD_DIM, HIDDEN_DIM, NUM_HEADS, TransformerWeights};

const LABEL: &str = "moss_tts/nano";
const LAYER_NORM_EPS: f32 = 1.0e-5;
const ROPE_BASE: f32 = 10_000.0;

pub(super) fn forward(
    compute: &Compute,
    input: &[f32],
    rows: usize,
    weights: &TransformerWeights,
) -> Result<Vec<f32>> {
    if rows == 0 || input.len() != rows * HIDDEN_DIM {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: transformer input has {} values, expected non-empty [{rows}, {HIDDEN_DIM}]",
            input.len()
        )));
    }
    let mut hidden = input.to_vec();
    for block in &weights.blocks {
        hidden = block_forward(compute, &hidden, rows, block)?;
    }
    let mut output = vec![0.0f32; hidden.len()];
    compute.layer_norm_f32(
        &hidden,
        &mut output,
        rows,
        HIDDEN_DIM,
        &weights.final_norm_weight,
        &weights.final_norm_bias,
        LAYER_NORM_EPS,
    )?;
    reject_non_finite("transformer output", &output)?;
    Ok(output)
}

fn block_forward(
    compute: &Compute,
    input: &[f32],
    rows: usize,
    weights: &BlockWeights,
) -> Result<Vec<f32>> {
    let mut normalized = vec![0.0f32; input.len()];
    compute.layer_norm_f32(
        input,
        &mut normalized,
        rows,
        HIDDEN_DIM,
        &weights.ln1_weight,
        &weights.ln1_bias,
        LAYER_NORM_EPS,
    )?;
    let attention = attention(compute, &normalized, rows, weights)?;
    let mut hidden = add_residual(input, &attention)?;

    compute.layer_norm_f32(
        &hidden,
        &mut normalized,
        rows,
        HIDDEN_DIM,
        &weights.ln2_weight,
        &weights.ln2_bias,
        LAYER_NORM_EPS,
    )?;
    let projected = weights.ffn_in.forward(compute, &normalized, rows)?;
    let mut activated = vec![0.0f32; projected.len()];
    compute.gelu_new_f32(&projected, &mut activated)?;
    let feed_forward = weights.ffn_out.forward(compute, &activated, rows)?;
    for (value, residual) in hidden.iter_mut().zip(feed_forward) {
        *value += residual;
    }
    reject_non_finite("transformer block output", &hidden)?;
    Ok(hidden)
}

fn attention(
    compute: &Compute,
    input: &[f32],
    rows: usize,
    weights: &BlockWeights,
) -> Result<Vec<f32>> {
    let qkv = weights.attention_in.forward(compute, input, rows)?;
    let mut joined = vec![0.0f32; rows * HIDDEN_DIM];
    let scale = (HEAD_DIM as f32).sqrt().recip();

    for head in 0..NUM_HEADS {
        let mut query = vec![0.0f32; rows * HEAD_DIM];
        let mut key = vec![0.0f32; rows * HEAD_DIM];
        let mut value = vec![0.0f32; rows * HEAD_DIM];
        for position in 0..rows {
            let qkv_row = position * 3 * HIDDEN_DIM;
            let head_offset = head * HEAD_DIM;
            let target = position * HEAD_DIM;
            query[target..target + HEAD_DIM]
                .copy_from_slice(&qkv[qkv_row + head_offset..qkv_row + head_offset + HEAD_DIM]);
            key[target..target + HEAD_DIM].copy_from_slice(
                &qkv[qkv_row + HIDDEN_DIM + head_offset
                    ..qkv_row + HIDDEN_DIM + head_offset + HEAD_DIM],
            );
            value[target..target + HEAD_DIM].copy_from_slice(
                &qkv[qkv_row + 2 * HIDDEN_DIM + head_offset
                    ..qkv_row + 2 * HIDDEN_DIM + head_offset + HEAD_DIM],
            );
        }
        apply_adjacent_rope(&mut query, rows, HEAD_DIM, 0)?;
        apply_adjacent_rope(&mut key, rows, HEAD_DIM, 0)?;

        let mut key_t = vec![0.0f32; HEAD_DIM * rows];
        for position in 0..rows {
            for dimension in 0..HEAD_DIM {
                key_t[dimension * rows + position] = key[position * HEAD_DIM + dimension];
            }
        }
        let mut scores = vec![0.0f32; rows * rows];
        compute.gemm_f32(rows, rows, HEAD_DIM, &query, &key_t, None, &mut scores)?;
        apply_causal_mask(&mut scores, rows, scale)?;
        let mut probabilities = vec![0.0f32; scores.len()];
        compute.softmax_f32(&scores, &mut probabilities, rows, rows)?;
        let mut attended = vec![0.0f32; rows * HEAD_DIM];
        compute.gemm_f32(
            rows,
            HEAD_DIM,
            rows,
            &probabilities,
            &value,
            None,
            &mut attended,
        )?;
        for position in 0..rows {
            let source = position * HEAD_DIM;
            let target = position * HIDDEN_DIM + head * HEAD_DIM;
            joined[target..target + HEAD_DIM].copy_from_slice(&attended[source..source + HEAD_DIM]);
        }
    }
    weights.attention_out.forward(compute, &joined, rows)
}

fn apply_adjacent_rope(
    values: &mut [f32],
    rows: usize,
    dimensions: usize,
    position_offset: usize,
) -> Result<()> {
    if dimensions == 0 || !dimensions.is_multiple_of(2) || values.len() != rows * dimensions {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: RoPE shape mismatch: values={}, rows={rows}, dimensions={dimensions}",
            values.len()
        )));
    }
    for position in 0..rows {
        for pair in 0..dimensions / 2 {
            let frequency = ROPE_BASE.powf(-((2 * pair) as f32) / dimensions as f32);
            let angle = (position_offset + position) as f32 * frequency;
            let (sin, cos) = angle.sin_cos();
            let index = position * dimensions + 2 * pair;
            let even = values[index];
            let odd = values[index + 1];
            values[index] = even * cos - odd * sin;
            values[index + 1] = even * sin + odd * cos;
        }
    }
    Ok(())
}

fn apply_causal_mask(scores: &mut [f32], rows: usize, scale: f32) -> Result<()> {
    if rows == 0 || scores.len() != rows * rows || !scale.is_finite() {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: causal mask shape mismatch: scores={}, rows={rows}, scale={scale}",
            scores.len()
        )));
    }
    for query in 0..rows {
        for key in 0..rows {
            let score = &mut scores[query * rows + key];
            if key > query {
                *score = f32::MIN;
            } else {
                *score *= scale;
            }
        }
    }
    Ok(())
}

fn add_residual(left: &[f32], right: &[f32]) -> Result<Vec<f32>> {
    if left.len() != right.len() {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: residual length mismatch: {} != {}",
            left.len(),
            right.len()
        )));
    }
    Ok(left.iter().zip(right).map(|(&a, &b)| a + b).collect())
}

fn reject_non_finite(label: &str, values: &[f32]) -> Result<()> {
    if let Some((index, value)) = values
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: {label} contains non-finite value {value} at index {index}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adjacent_rope_position_zero_is_identity() {
        let mut values = vec![1.0, 2.0, 3.0, 4.0];
        let expected = values.clone();
        apply_adjacent_rope(&mut values, 1, 4, 0).unwrap();
        assert_eq!(values, expected);
    }

    #[test]
    fn causal_mask_keeps_prefix_and_hides_future() {
        let mut scores = vec![1.0; 9];
        apply_causal_mask(&mut scores, 3, 0.5).unwrap();
        assert_eq!(scores[0], 0.5);
        assert_eq!(scores[1], f32::MIN);
        assert_eq!(scores[3], 0.5);
        assert_eq!(scores[4], 0.5);
        assert_eq!(scores[5], f32::MIN);
    }
}
