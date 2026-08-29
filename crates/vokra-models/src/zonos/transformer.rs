//! Compute-dispatched Zonos transformer primitives.
//!
//! This module keeps the incremental state explicit.  It is intentionally
//! independent of the delayed codebook sampler: callers append one already
//! embedded frame at a time and receive the nine head logits for that frame.

use super::{ZonosConfig, ZonosWeights};
use crate::compute::Compute;
use vokra_core::{Result, VokraError};

/// Per-layer causal K/V state for one Zonos sequence.
#[derive(Debug, Clone)]
pub(crate) struct KvCache {
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
    frames: usize,
    weights_validated: bool,
}

impl KvCache {
    /// Allocates an empty cache after validating the exact GQA topology.
    pub(crate) fn new(config: &ZonosConfig) -> Result<Self> {
        config.validate_for_forward()?;
        Ok(Self {
            keys: vec![Vec::new(); config.backbone.n_layer],
            values: vec![Vec::new(); config.backbone.n_layer],
            frames: 0,
            weights_validated: false,
        })
    }

    /// Number of frames already committed to every layer.
    #[must_use]
    pub(crate) fn len(&self) -> usize {
        self.frames
    }

    /// Consumes one row and returns the nine codebook-head logits.
    pub(crate) fn step(
        &mut self,
        config: &ZonosConfig,
        weights: &ZonosWeights,
        input: &[f32],
        compute: &Compute,
    ) -> Result<Vec<Vec<f32>>> {
        // Every cache mutation below is append-only. Save only lengths and
        // scalar state so a late-layer/backend failure can truncate the
        // partial append without cloning the full 26-layer history per token.
        let frame_snapshot = self.frames;
        let validated_snapshot = self.weights_validated;
        let lengths: Vec<(usize, usize)> = self
            .keys
            .iter()
            .zip(&self.values)
            .map(|(keys, values)| (keys.len(), values.len()))
            .collect();
        match self.step_inner(config, weights, input, compute) {
            Ok(logits) => Ok(logits),
            Err(error) => {
                self.frames = frame_snapshot;
                self.weights_validated = validated_snapshot;
                for ((keys, values), (key_len, value_len)) in self
                    .keys
                    .iter_mut()
                    .zip(self.values.iter_mut())
                    .zip(lengths)
                {
                    keys.truncate(key_len);
                    values.truncate(value_len);
                }
                Err(error)
            }
        }
    }

    fn step_inner(
        &mut self,
        config: &ZonosConfig,
        weights: &ZonosWeights,
        input: &[f32],
        compute: &Compute,
    ) -> Result<Vec<Vec<f32>>> {
        let bb = &config.backbone;
        let d = bb.d_model;
        if input.len() != d
            || weights.blocks.len() != bb.n_layer
            || weights.codebook_embeddings.len() != config.num_codebooks
            || weights.logit_heads.len() != config.num_codebooks
        {
            return Err(VokraError::InvalidArgument(
                "zonos incremental transformer input/weight shape mismatch".to_owned(),
            ));
        }
        let position = self.frames;
        let mut hidden = input.to_vec();
        let qh = bb.q_hidden();
        let kvh = bb.kv_hidden();
        let qkv_width = qh + 2 * kvh;
        let groups = bb.num_heads / bb.num_heads_kv;
        let scale = (bb.head_dim() as f32).sqrt().recip();
        let block_shapes_valid = weights.blocks.iter().all(|block| {
            block.norm_1_w.len() == d
                && block.norm_1_b.len() == d
                && block.qkv_proj.len() == d * qkv_width
                && block.o_proj.len() == qh * d
                && block.norm_2_w.len() == d
                && block.norm_2_b.len() == d
                && block.mlp_fc1.len() == d * bb.mlp_fc1_out()
                && block.mlp_fc2.len() == bb.d_intermediate * d
        });
        if !block_shapes_valid
            || weights
                .logit_heads
                .iter()
                .any(|head| head.len() != d * config.head_vocab)
            || weights.norm_f_w.len() != d
            || weights.norm_f_b.len() != d
        {
            return Err(VokraError::InvalidArgument(
                "zonos incremental transformer tensor shape mismatch".to_owned(),
            ));
        }
        if input.iter().any(|value| !value.is_finite()) {
            return Err(VokraError::InvalidArgument(
                "zonos incremental transformer input contains non-finite values".to_owned(),
            ));
        }
        if !self.weights_validated {
            let static_finite = weights.norm_f_w.iter().all(|v| v.is_finite())
                && weights.norm_f_b.iter().all(|v| v.is_finite())
                && weights
                    .logit_heads
                    .iter()
                    .all(|head| head.iter().all(|v| v.is_finite()))
                && weights
                    .codebook_embeddings
                    .iter()
                    .all(|table| table.iter().all(|v| v.is_finite()))
                && weights.blocks.iter().all(|block| {
                    block.norm_1_w.iter().all(|v| v.is_finite())
                        && block.norm_1_b.iter().all(|v| v.is_finite())
                        && block.qkv_proj.iter().all(|v| v.is_finite())
                        && block.o_proj.iter().all(|v| v.is_finite())
                        && block.norm_2_w.iter().all(|v| v.is_finite())
                        && block.norm_2_b.iter().all(|v| v.is_finite())
                        && block.mlp_fc1.iter().all(|v| v.is_finite())
                        && block.mlp_fc2.iter().all(|v| v.is_finite())
                });
            if !static_finite {
                return Err(VokraError::InvalidArgument(
                    "zonos incremental transformer weights contain non-finite values".to_owned(),
                ));
            }
            if let Some(prefix) = &weights.prefix_conditioner {
                prefix.validate(d)?;
            }
            self.weights_validated = true;
        }

        for (layer, block) in weights.blocks.iter().enumerate() {
            let normed = super::layer_norm_rows(
                &hidden,
                &block.norm_1_w,
                &block.norm_1_b,
                1,
                d,
                bb.norm_epsilon,
                compute,
            )?;
            let mut packed = vec![0.0; qkv_width];
            compute.gemm_f32(1, qkv_width, d, &normed, &block.qkv_proj, None, &mut packed)?;
            let (mut query, mut key, value) = unpack_fused_qkv(&packed, qh, kvh)?;
            if query
                .iter()
                .chain(&key)
                .chain(&value)
                .any(|value| !value.is_finite())
            {
                return Err(VokraError::InvalidArgument(
                    "zonos incremental QKV projection is non-finite".to_owned(),
                ));
            }
            apply_rope_position(&mut query, bb.num_heads, bb.head_dim(), position);
            apply_rope_position(&mut key, bb.num_heads_kv, bb.head_dim(), position);

            let keys = &mut self.keys[layer];
            let values = &mut self.values[layer];
            let past = keys.len() / kvh;
            if keys.len() != past * kvh || values.len() != past * kvh || past != position {
                return Err(VokraError::InvalidArgument(
                    "zonos incremental K/V cache topology mismatch".to_owned(),
                ));
            }
            keys.extend_from_slice(&key);
            values.extend_from_slice(&value);

            let mut attended = vec![0.0; qh];
            for head in 0..bb.num_heads {
                let kv_head = head / groups;
                let head_dim = bb.head_dim();
                let mut key_transposed = vec![0.0; head_dim * (position + 1)];
                let mut value_rows = vec![0.0; (position + 1) * head_dim];
                for prior in 0..=position {
                    for lane in 0..head_dim {
                        key_transposed[lane * (position + 1) + prior] =
                            keys[prior * kvh + kv_head * head_dim + lane];
                        value_rows[prior * head_dim + lane] =
                            values[prior * kvh + kv_head * head_dim + lane];
                    }
                }
                let mut scores = vec![0.0; position + 1];
                compute.gemm_f32(
                    1,
                    position + 1,
                    head_dim,
                    &query[head * head_dim..(head + 1) * head_dim],
                    &key_transposed,
                    None,
                    &mut scores,
                )?;
                for score in &mut scores {
                    *score *= scale;
                }
                let mut probabilities = vec![0.0; position + 1];
                compute.softmax_f32(&scores, &mut probabilities, 1, position + 1)?;
                compute.gemm_f32(
                    1,
                    head_dim,
                    position + 1,
                    &probabilities,
                    &value_rows,
                    None,
                    &mut attended[head * head_dim..(head + 1) * head_dim],
                )?;
            }
            let mut attention_out = vec![0.0; d];
            compute.gemm_f32(1, d, qh, &attended, &block.o_proj, None, &mut attention_out)?;
            for (value, update) in hidden.iter_mut().zip(attention_out) {
                *value += update;
            }

            let normed = super::layer_norm_rows(
                &hidden,
                &block.norm_2_w,
                &block.norm_2_b,
                1,
                d,
                bb.norm_epsilon,
                compute,
            )?;
            let mut projected = vec![0.0; bb.mlp_fc1_out()];
            compute.gemm_f32(
                1,
                bb.mlp_fc1_out(),
                d,
                &normed,
                &block.mlp_fc1,
                None,
                &mut projected,
            )?;
            let mut activated = vec![0.0; bb.d_intermediate];
            let mut gate = vec![0.0; bb.d_intermediate];
            compute.silu_f32(&projected[bb.d_intermediate..], &mut gate)?;
            for index in 0..bb.d_intermediate {
                // Official `_torch.py`: y, gate = fc1.chunk(2), then
                // fc2(y * silu(gate)).
                activated[index] = projected[index] * gate[index];
            }
            let mut ffn = vec![0.0; d];
            compute.gemm_f32(
                1,
                d,
                bb.d_intermediate,
                &activated,
                &block.mlp_fc2,
                None,
                &mut ffn,
            )?;
            for (value, update) in hidden.iter_mut().zip(ffn) {
                *value += update;
            }
        }
        self.frames = self.frames.checked_add(1).ok_or_else(|| {
            VokraError::InvalidArgument("zonos incremental frame count overflow".to_owned())
        })?;

        let final_hidden = super::layer_norm_rows(
            &hidden,
            &weights.norm_f_w,
            &weights.norm_f_b,
            1,
            d,
            bb.norm_epsilon,
            compute,
        )?;
        let mut logits = Vec::with_capacity(config.num_codebooks);
        for head in &weights.logit_heads {
            let mut row = vec![0.0; config.head_vocab];
            compute.gemm_f32(1, config.head_vocab, d, &final_hidden, head, None, &mut row)?;
            logits.push(row);
        }
        if logits.iter().flatten().any(|value| !value.is_finite()) {
            return Err(VokraError::InvalidArgument(
                "zonos incremental transformer result is non-finite".to_owned(),
            ));
        }
        Ok(logits)
    }
}

/// Unpacks one row of the row-major fused projection.  A fused GEMM output is
/// laid out as `[q|k|v]` per frame, not as three contiguous frame ranges;
/// keeping this boundary explicit prevents multi-frame QKV interleaving bugs.
fn unpack_fused_qkv(
    packed: &[f32],
    q_hidden: usize,
    kv_hidden: usize,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    let width = q_hidden
        .checked_add(kv_hidden.checked_mul(2).ok_or_else(|| {
            VokraError::InvalidArgument("zonos fused QKV width overflow".to_owned())
        })?)
        .ok_or_else(|| VokraError::InvalidArgument("zonos fused QKV width overflow".to_owned()))?;
    if packed.len() != width {
        return Err(VokraError::InvalidArgument(
            "zonos fused QKV row width mismatch".to_owned(),
        ));
    }
    Ok((
        packed[..q_hidden].to_vec(),
        packed[q_hidden..q_hidden + kv_hidden].to_vec(),
        packed[q_hidden + kv_hidden..].to_vec(),
    ))
}

/// Runs the same causal transformer through one incremental step per row.
pub(crate) fn forward_incremental(
    config: &ZonosConfig,
    weights: &ZonosWeights,
    input: &[f32],
    frames: usize,
    compute: &Compute,
) -> Result<Vec<Vec<f32>>> {
    let d = config.backbone.d_model;
    if frames == 0 || input.len() != frames.saturating_mul(d) {
        return Err(VokraError::InvalidArgument(
            "zonos incremental transformer input shape mismatch".to_owned(),
        ));
    }
    let mut cache = KvCache::new(config)?;
    let mut logits = None;
    for frame in 0..frames {
        logits = Some(cache.step(config, weights, &input[frame * d..(frame + 1) * d], compute)?);
    }
    debug_assert_eq!(cache.len(), frames);
    logits.ok_or_else(|| VokraError::InvalidArgument("zonos empty incremental sequence".to_owned()))
}

fn apply_rope_position(x: &mut [f32], heads: usize, head_dim: usize, position: usize) {
    let width = heads * head_dim;
    debug_assert_eq!(x.len(), width);
    for head in 0..heads {
        for pair in (0..head_dim).step_by(2) {
            let angle = position as f32 / 10_000.0_f32.powf(pair as f32 / head_dim as f32);
            let (sin, cos) = angle.sin_cos();
            let offset = head * head_dim + pair;
            let real = x[offset];
            let imag = x[offset + 1];
            x[offset] = real * cos - imag * sin;
            x[offset + 1] = real * sin + imag * cos;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_causal_path_matches_incremental_kv_path() {
        let config = ZonosConfig::tiny_for_tests();
        let weights = ZonosWeights::synthesized(&config, 0x5eed).unwrap();
        let compute = Compute::cpu();
        let input: Vec<f32> = (0..config.backbone.d_model * 3)
            .map(|index| (index as f32 - 11.0) / 17.0)
            .collect();
        let full =
            super::super::transformer_logits(&config, &weights, &input, 3, &compute).unwrap();
        let incremental = forward_incremental(&config, &weights, &input, 3, &compute).unwrap();
        let mut persistent_cache = KvCache::new(&config).unwrap();
        let mut persistent = None;
        for row in input.chunks_exact(config.backbone.d_model) {
            persistent = Some(
                persistent_cache
                    .step(&config, &weights, row, &compute)
                    .unwrap(),
            );
        }
        let persistent = persistent.expect("persistent cache has a final frame");
        let oracle = scalar_oracle(&config, &weights, &input, 3);
        for (full_head, step_head) in full.iter().zip(incremental.iter()) {
            for (full_value, step_value) in full_head.iter().zip(step_head.iter()) {
                assert!((full_value - step_value).abs() < 2.0e-4);
            }
        }
        assert_eq!(persistent_cache.len(), 3);
        for (step_head, persistent_head) in incremental.iter().zip(persistent.iter()) {
            for (step_value, persistent_value) in step_head.iter().zip(persistent_head.iter()) {
                assert!((step_value - persistent_value).abs() < 2.0e-4);
            }
        }
        for (full_head, oracle_head) in full.iter().zip(oracle.iter()) {
            for (full_value, oracle_value) in full_head.iter().zip(oracle_head.iter()) {
                assert!((full_value - oracle_value).abs() < 2.0e-4);
            }
        }
    }

    #[test]
    fn failed_step_rolls_back_cache_and_rejects_nonfinite_inputs() {
        let config = ZonosConfig::tiny_for_tests();
        let weights = ZonosWeights::synthesized(&config, 0x1234).unwrap();
        let compute = Compute::cpu();
        let mut cache = KvCache::new(&config).unwrap();
        let valid = vec![0.25; config.backbone.d_model];
        cache.step(&config, &weights, &valid, &compute).unwrap();
        let before = cache.clone();
        let mut invalid = valid.clone();
        invalid[0] = f32::NAN;
        assert!(cache.step(&config, &weights, &invalid, &compute).is_err());
        assert_eq!(cache.len(), before.len());
        assert_eq!(cache.keys, before.keys);
        assert_eq!(cache.values, before.values);
    }

    #[test]
    fn failed_late_layer_step_rolls_back_earlier_layer_mutation() {
        let config = ZonosConfig::tiny_for_tests();
        let weights = ZonosWeights::synthesized(&config, 0x4321).unwrap();
        let compute = Compute::cpu();
        let mut cache = KvCache::new(&config).unwrap();
        let valid = vec![0.25; config.backbone.d_model];
        cache.step(&config, &weights, &valid, &compute).unwrap();
        // Corrupt only the second layer's topology. Layer zero will append
        // before layer one rejects it; `step` must restore both layers.
        cache.keys[1].push(0.0);
        let before = cache.clone();
        assert!(cache.step(&config, &weights, &valid, &compute).is_err());
        assert_eq!(cache.frames, before.frames);
        assert_eq!(cache.keys, before.keys);
        assert_eq!(cache.values, before.values);
    }

    #[test]
    fn fused_qkv_unpack_is_row_strided_for_multiple_frames() {
        let qh = 2;
        let kvh = 1;
        let rows = [[10.0, 11.0, 20.0, 30.0], [40.0, 41.0, 50.0, 60.0]];
        let unpacked: Vec<_> = rows
            .iter()
            .map(|row| unpack_fused_qkv(row, qh, kvh).unwrap())
            .collect();
        assert_eq!(unpacked[0], (vec![10.0, 11.0], vec![20.0], vec![30.0]));
        assert_eq!(unpacked[1], (vec![40.0, 41.0], vec![50.0], vec![60.0]));
    }

    fn scalar_oracle(
        config: &ZonosConfig,
        weights: &ZonosWeights,
        input: &[f32],
        frames: usize,
    ) -> Vec<Vec<f32>> {
        let bb = &config.backbone;
        let d = bb.d_model;
        let qh = bb.q_hidden();
        let kvh = bb.kv_hidden();
        let hd = bb.head_dim();
        let mut hidden: Vec<Vec<f32>> = input.chunks_exact(d).map(ToOwned::to_owned).collect();
        assert_eq!(hidden.len(), frames);
        for block in &weights.blocks {
            let normed: Vec<Vec<f32>> = hidden
                .iter()
                .map(|row| scalar_ln(row, &block.norm_1_w, &block.norm_1_b, bb.norm_epsilon))
                .collect();
            let mut queries = Vec::with_capacity(frames);
            let mut keys = Vec::with_capacity(frames);
            let mut values = Vec::with_capacity(frames);
            for (position, row) in normed.iter().enumerate() {
                let packed = scalar_linear(row, &block.qkv_proj, d, qh + 2 * kvh);
                let mut query = packed[..qh].to_vec();
                let mut key = packed[qh..qh + kvh].to_vec();
                scalar_rope(&mut query, bb.num_heads, hd, position);
                scalar_rope(&mut key, bb.num_heads_kv, hd, position);
                queries.push(query);
                keys.push(key);
                values.push(packed[qh + kvh..].to_vec());
            }
            let groups = bb.num_heads / bb.num_heads_kv;
            let scale = (hd as f32).sqrt().recip();
            let mut attended = vec![vec![0.0; qh]; frames];
            for position in 0..frames {
                for head in 0..bb.num_heads {
                    let kv_head = head / groups;
                    let mut scores = Vec::with_capacity(position + 1);
                    for prior in 0..=position {
                        let mut score = 0.0;
                        for lane in 0..hd {
                            score += queries[position][head * hd + lane]
                                * keys[prior][kv_head * hd + lane];
                        }
                        scores.push(score * scale);
                    }
                    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let exp: Vec<f32> = scores.iter().map(|score| (score - max).exp()).collect();
                    let denominator = exp.iter().sum::<f32>();
                    for (prior, value) in exp.iter().enumerate() {
                        let probability = value / denominator;
                        for lane in 0..hd {
                            attended[position][head * hd + lane] +=
                                probability * values[prior][kv_head * hd + lane];
                        }
                    }
                }
            }
            for position in 0..frames {
                let attention = scalar_linear(&attended[position], &block.o_proj, qh, d);
                for (value, update) in hidden[position].iter_mut().zip(attention) {
                    *value += update;
                }
                let normed = scalar_ln(
                    &hidden[position],
                    &block.norm_2_w,
                    &block.norm_2_b,
                    bb.norm_epsilon,
                );
                let projected = scalar_linear(&normed, &block.mlp_fc1, d, 2 * bb.d_intermediate);
                // Keep the oracle's SiLU expression explicit and independent
                // of the production Compute implementation.
                let activated: Vec<f32> = projected[..bb.d_intermediate]
                    .iter()
                    .zip(&projected[bb.d_intermediate..])
                    .map(|(value, gate)| value * (*gate / (1.0 + (-*gate).exp())))
                    .collect();
                let ffn = scalar_linear(&activated, &block.mlp_fc2, bb.d_intermediate, d);
                for (value, update) in hidden[position].iter_mut().zip(ffn) {
                    *value += update;
                }
            }
        }
        let last = scalar_ln(
            hidden.last().expect("non-empty oracle input"),
            &weights.norm_f_w,
            &weights.norm_f_b,
            bb.norm_epsilon,
        );
        weights
            .logit_heads
            .iter()
            .map(|head| scalar_linear(&last, head, d, config.head_vocab))
            .collect()
    }

    fn scalar_linear(
        input: &[f32],
        weight: &[f32],
        input_width: usize,
        output_width: usize,
    ) -> Vec<f32> {
        assert_eq!(input.len(), input_width);
        assert_eq!(weight.len(), input_width * output_width);
        (0..output_width)
            .map(|column| {
                (0..input_width)
                    .map(|row| input[row] * weight[row * output_width + column])
                    .sum()
            })
            .collect()
    }

    fn scalar_ln(input: &[f32], weight: &[f32], bias: &[f32], eps: f32) -> Vec<f32> {
        let mean = input.iter().sum::<f32>() / input.len() as f32;
        let variance = input
            .iter()
            .map(|value| (value - mean) * (value - mean))
            .sum::<f32>()
            / input.len() as f32;
        input
            .iter()
            .zip(weight)
            .zip(bias)
            .map(|((value, gamma), beta)| (value - mean) / (variance + eps).sqrt() * gamma + beta)
            .collect()
    }

    fn scalar_rope(x: &mut [f32], heads: usize, head_dim: usize, position: usize) {
        for head in 0..heads {
            for pair in (0..head_dim).step_by(2) {
                let angle = position as f32 / 10_000.0_f32.powf(pair as f32 / head_dim as f32);
                let (sin, cos) = angle.sin_cos();
                let offset = head * head_dim + pair;
                let real = x[offset];
                let imag = x[offset + 1];
                x[offset] = real * cos - imag * sin;
                x[offset + 1] = real * sin + imag * cos;
            }
        }
    }
}
