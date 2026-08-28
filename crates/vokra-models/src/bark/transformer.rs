//! Bark's three native Transformer stages.
//!
//! Semantic and coarse generation are token-incremental and keep one K/V
//! cache per layer. Their public `[out, in]` F32 matrices can therefore feed
//! [`Compute::gemv_f32`](crate::compute::Compute::gemv_f32) directly without
//! creating a checkpoint-sized transposed copy. Fine generation is non-causal
//! over a 1,024-frame window, so it transposes one mapped matrix at a time into
//! a short-lived scratch buffer and dispatches GEMM. Embedding lookup,
//! residual addition, head packing and cache layout are deterministic host
//! glue; every learned reduction and activation goes through [`Compute`].

use vokra_core::{Result, VokraError};

use crate::compute::Compute;

use super::BarkConfig;
use super::weights::BarkMappedWeights;

const LABEL: &str = "bark";
const LAYER_NORM_EPS: f32 = 1.0e-5;
const FINE_CODEBOOKS: usize = 8;
const FINE_VOCAB: usize = 1_056;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum CausalStage {
    Semantic,
    Coarse,
}

impl CausalStage {
    const fn prefix(self) -> &'static str {
        match self {
            Self::Semantic => "semantic",
            Self::Coarse => "coarse_acoustics",
        }
    }

    const fn input_vocab(self) -> usize {
        match self {
            Self::Semantic => 129_600,
            Self::Coarse => 12_096,
        }
    }

    const fn output_vocab(self) -> usize {
        match self {
            Self::Semantic => 10_048,
            Self::Coarse => 12_096,
        }
    }
}

#[derive(Debug, Clone, Default)]
struct LayerKv {
    /// Position-major `[time, hidden]` K rows.
    key: Vec<f32>,
    /// Position-major `[time, hidden]` V rows.
    value: Vec<f32>,
}

/// Per-request causal cache for either the semantic or coarse model.
#[derive(Debug, Clone)]
pub(super) struct CausalCache {
    stage: CausalStage,
    layers: Vec<LayerKv>,
    next_position: usize,
}

impl CausalCache {
    fn new(stage: CausalStage, num_layers: usize) -> Self {
        Self {
            stage,
            layers: vec![LayerKv::default(); num_layers],
            next_position: 0,
        }
    }

    #[cfg(test)]
    pub(super) const fn len(&self) -> usize {
        self.next_position
    }
}

/// Looks up one causal-stage embedding row without adding a position vector.
pub(super) fn causal_embedding(
    weights: &BarkMappedWeights,
    config: &BarkConfig,
    stage: CausalStage,
    token: u32,
) -> Result<Vec<f32>> {
    let token = token as usize;
    if token >= stage.input_vocab() {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}/{}: token {token} is outside input vocab 0..{}",
            stage.prefix(),
            stage.input_vocab()
        )));
    }
    let table = weights.tensor(
        &format!("{}.input_embeds_layer.weight", stage.prefix()),
        &[stage.input_vocab(), config.hidden_size],
    )?;
    let start = token * config.hidden_size;
    Ok(table[start..start + config.hidden_size].to_vec())
}

/// Prefills one causal model from caller-composed embeddings and returns the
/// logits predicting the first generated token.
pub(super) fn causal_prefill(
    weights: &BarkMappedWeights,
    compute: &Compute,
    config: &BarkConfig,
    stage: CausalStage,
    embeddings: &[f32],
) -> Result<(CausalCache, Vec<f32>)> {
    let hidden = config.hidden_size;
    if embeddings.is_empty() || embeddings.len() % hidden != 0 {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}/{}: prefill embeddings length {} is not non-empty [time, {hidden}]",
            stage.prefix(),
            embeddings.len()
        )));
    }
    let rows = embeddings.len() / hidden;
    if rows > config.block_size {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}/{}: prefill rows {rows} exceed block size {}",
            stage.prefix(),
            config.block_size
        )));
    }
    let mut cache = CausalCache::new(stage, config.num_layers_per_stage);
    let mut logits = Vec::new();
    for row in embeddings.chunks_exact(hidden) {
        logits = causal_step_embedding(weights, compute, config, &mut cache, row)?;
    }
    Ok((cache, logits))
}

/// Appends one generated token to a causal cache and returns next-token logits.
pub(super) fn causal_token_step(
    weights: &BarkMappedWeights,
    compute: &Compute,
    config: &BarkConfig,
    cache: &mut CausalCache,
    token: u32,
) -> Result<Vec<f32>> {
    let embedding = causal_embedding(weights, config, cache.stage, token)?;
    causal_step_embedding(weights, compute, config, cache, &embedding)
}

fn causal_step_embedding(
    weights: &BarkMappedWeights,
    compute: &Compute,
    config: &BarkConfig,
    cache: &mut CausalCache,
    embedding: &[f32],
) -> Result<Vec<f32>> {
    let hidden = config.hidden_size;
    let position = cache.next_position;
    if embedding.len() != hidden || position >= config.block_size {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}/{}: causal step has embedding {} and position {position}; expected hidden {hidden} and position < {}",
            cache.stage.prefix(),
            embedding.len(),
            config.block_size
        )));
    }
    let positions = weights.tensor(
        &format!("{}.position_embeds_layer.weight", cache.stage.prefix()),
        &[config.block_size, hidden],
    )?;
    let mut state = embedding.to_vec();
    let position_row = &positions[position * hidden..(position + 1) * hidden];
    add_assign(&mut state, position_row)?;

    let zero_beta = vec![0.0f32; hidden];
    for layer in 0..config.num_layers_per_stage {
        let base = format!("{}.layers.{layer}", cache.stage.prefix());
        let norm1_weight = weights.tensor(&format!("{base}.layernorm_1.weight"), &[hidden])?;
        let normalized = layer_norm_row(compute, &state, norm1_weight, &zero_beta)?;
        let attended = causal_attention(
            weights,
            compute,
            config,
            &base,
            &normalized,
            &mut cache.layers[layer],
            position,
        )?;
        add_assign(&mut state, &attended)?;

        let norm2_weight = weights.tensor(&format!("{base}.layernorm_2.weight"), &[hidden])?;
        let normalized = layer_norm_row(compute, &state, norm2_weight, &zero_beta)?;
        let feed_forward = mlp_row(weights, compute, hidden, &base, &normalized)?;
        add_assign(&mut state, &feed_forward)?;
    }

    let final_weight = weights.tensor(
        &format!("{}.layernorm_final.weight", cache.stage.prefix()),
        &[hidden],
    )?;
    let normalized = layer_norm_row(compute, &state, final_weight, &zero_beta)?;
    let logits = gemv_tensor(
        weights,
        compute,
        &format!("{}.lm_head.weight", cache.stage.prefix()),
        cache.stage.output_vocab(),
        hidden,
        &normalized,
    )?;
    cache.next_position += 1;
    Ok(logits)
}

#[allow(clippy::too_many_arguments)]
fn causal_attention(
    weights: &BarkMappedWeights,
    compute: &Compute,
    config: &BarkConfig,
    base: &str,
    input: &[f32],
    cache: &mut LayerKv,
    position: usize,
) -> Result<Vec<f32>> {
    let hidden = config.hidden_size;
    let heads = config.num_heads;
    let head_dim = hidden / heads;
    if cache.key.len() != position * hidden || cache.value.len() != position * hidden {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: K/V cache drift at {base}: key={} value={}, expected {}",
            cache.key.len(),
            cache.value.len(),
            position * hidden
        )));
    }
    let qkv = gemv_tensor(
        weights,
        compute,
        &format!("{base}.attn.att_proj.weight"),
        3 * hidden,
        hidden,
        input,
    )?;
    let query = &qkv[..hidden];
    cache.key.extend_from_slice(&qkv[hidden..2 * hidden]);
    cache.value.extend_from_slice(&qkv[2 * hidden..]);
    let sequence = position + 1;
    let scale = (head_dim as f32).sqrt().recip();
    let mut joined = vec![0.0f32; hidden];

    for head in 0..heads {
        let head_start = head * head_dim;
        let mut keys = Vec::with_capacity(sequence * head_dim);
        let mut values = Vec::with_capacity(sequence * head_dim);
        for timestep in 0..sequence {
            let start = timestep * hidden + head_start;
            keys.extend_from_slice(&cache.key[start..start + head_dim]);
            values.extend_from_slice(&cache.value[start..start + head_dim]);
        }
        let mut scores = vec![0.0f32; sequence];
        compute.gemv_f32(
            sequence,
            head_dim,
            &keys,
            &query[head_start..head_start + head_dim],
            None,
            &mut scores,
        )?;
        for score in &mut scores {
            *score *= scale;
        }
        let mut probabilities = vec![0.0f32; sequence];
        compute.softmax_f32(&scores, &mut probabilities, 1, sequence)?;
        let mut context = vec![0.0f32; head_dim];
        compute.gemm_f32(
            1,
            head_dim,
            sequence,
            &probabilities,
            &values,
            None,
            &mut context,
        )?;
        joined[head_start..head_start + head_dim].copy_from_slice(&context);
    }
    gemv_tensor(
        weights,
        compute,
        &format!("{base}.attn.out_proj.weight"),
        hidden,
        hidden,
        &joined,
    )
}

fn mlp_row(
    weights: &BarkMappedWeights,
    compute: &Compute,
    hidden: usize,
    base: &str,
    input: &[f32],
) -> Result<Vec<f32>> {
    let projected = gemv_tensor(
        weights,
        compute,
        &format!("{base}.mlp.in_proj.weight"),
        4 * hidden,
        hidden,
        input,
    )?;
    let mut activated = vec![0.0f32; projected.len()];
    compute.gelu_f32(&projected, &mut activated)?;
    gemv_tensor(
        weights,
        compute,
        &format!("{base}.mlp.out_proj.weight"),
        hidden,
        4 * hidden,
        &activated,
    )
}

/// Non-causal Fine-stage forward for one target codebook.
///
/// `input_ids` is frame-major `[rows, 8]`. Embeddings 0 through
/// `codebook_idx` are summed exactly like the official model. Returned logits
/// are row-major `[rows, 1056]`.
pub(super) fn fine_logits(
    weights: &BarkMappedWeights,
    compute: &Compute,
    config: &BarkConfig,
    input_ids: &[u32],
    rows: usize,
    codebook_idx: usize,
) -> Result<Vec<f32>> {
    if rows == 0
        || rows > config.block_size
        || input_ids.len() != rows * FINE_CODEBOOKS
        || !(1..FINE_CODEBOOKS).contains(&codebook_idx)
    {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}/fine: invalid forward shape ids={}, rows={rows}, codebook_idx={codebook_idx}, block_size={}",
            input_ids.len(),
            config.block_size
        )));
    }
    let hidden = config.hidden_size;
    let positions = weights.tensor(
        "fine_acoustics.position_embeds_layer.weight",
        &[config.block_size, hidden],
    )?;
    let mut state = vec![0.0f32; rows * hidden];
    for source_codebook in 0..=codebook_idx {
        let table = weights.tensor(
            &format!("fine_acoustics.input_embeds_layers.{source_codebook}.weight"),
            &[FINE_VOCAB, hidden],
        )?;
        for row in 0..rows {
            let token = input_ids[row * FINE_CODEBOOKS + source_codebook] as usize;
            if token >= FINE_VOCAB {
                return Err(VokraError::InvalidArgument(format!(
                    "{LABEL}/fine: input_ids[{row},{source_codebook}]={token} is outside 0..{FINE_VOCAB}"
                )));
            }
            let source = &table[token * hidden..(token + 1) * hidden];
            let target = &mut state[row * hidden..(row + 1) * hidden];
            add_assign(target, source)?;
        }
    }
    for row in 0..rows {
        add_assign(
            &mut state[row * hidden..(row + 1) * hidden],
            &positions[row * hidden..(row + 1) * hidden],
        )?;
    }

    for layer in 0..config.num_layers_per_stage {
        let base = format!("fine_acoustics.layers.{layer}");
        let norm1 = layer_norm_rows(
            weights,
            compute,
            &state,
            rows,
            hidden,
            &format!("{base}.layernorm_1"),
        )?;
        let attended = noncausal_attention(weights, compute, config, &base, &norm1, rows)?;
        add_assign(&mut state, &attended)?;

        let norm2 = layer_norm_rows(
            weights,
            compute,
            &state,
            rows,
            hidden,
            &format!("{base}.layernorm_2"),
        )?;
        let projected = linear_rows(
            weights,
            compute,
            &format!("{base}.mlp.in_proj.weight"),
            &norm2,
            rows,
            hidden,
            4 * hidden,
        )?;
        let mut activated = vec![0.0f32; projected.len()];
        compute.gelu_f32(&projected, &mut activated)?;
        let feed_forward = linear_rows(
            weights,
            compute,
            &format!("{base}.mlp.out_proj.weight"),
            &activated,
            rows,
            4 * hidden,
            hidden,
        )?;
        add_assign(&mut state, &feed_forward)?;
    }

    let final_weight = weights.tensor("fine_acoustics.layernorm_final.weight", &[hidden])?;
    let final_bias = weights.tensor("fine_acoustics.layernorm_final.bias", &[hidden])?;
    let mut normalized = vec![0.0f32; state.len()];
    compute.layer_norm_f32(
        &state,
        &mut normalized,
        rows,
        hidden,
        final_weight,
        final_bias,
        LAYER_NORM_EPS,
    )?;
    linear_rows(
        weights,
        compute,
        &format!("fine_acoustics.lm_heads.{}.weight", codebook_idx - 1),
        &normalized,
        rows,
        hidden,
        FINE_VOCAB,
    )
}

fn noncausal_attention(
    weights: &BarkMappedWeights,
    compute: &Compute,
    config: &BarkConfig,
    base: &str,
    input: &[f32],
    rows: usize,
) -> Result<Vec<f32>> {
    let hidden = config.hidden_size;
    let heads = config.num_heads;
    let head_dim = hidden / heads;
    let qkv = linear_rows(
        weights,
        compute,
        &format!("{base}.attn.att_proj.weight"),
        input,
        rows,
        hidden,
        3 * hidden,
    )?;
    let scale = (head_dim as f32).sqrt().recip();
    let mut joined = vec![0.0f32; rows * hidden];
    for head in 0..heads {
        let head_start = head * head_dim;
        let mut query = vec![0.0f32; rows * head_dim];
        let mut key_t = vec![0.0f32; head_dim * rows];
        let mut value = vec![0.0f32; rows * head_dim];
        for row in 0..rows {
            let qkv_row = row * 3 * hidden;
            let target = row * head_dim;
            query[target..target + head_dim]
                .copy_from_slice(&qkv[qkv_row + head_start..qkv_row + head_start + head_dim]);
            value[target..target + head_dim].copy_from_slice(
                &qkv[qkv_row + 2 * hidden + head_start
                    ..qkv_row + 2 * hidden + head_start + head_dim],
            );
            for dimension in 0..head_dim {
                key_t[dimension * rows + row] = qkv[qkv_row + hidden + head_start + dimension];
            }
        }
        let mut scores = vec![0.0f32; rows * rows];
        compute.gemm_f32(rows, rows, head_dim, &query, &key_t, None, &mut scores)?;
        for score in &mut scores {
            *score *= scale;
        }
        let mut probabilities = vec![0.0f32; scores.len()];
        compute.softmax_f32(&scores, &mut probabilities, rows, rows)?;
        let mut context = vec![0.0f32; rows * head_dim];
        compute.gemm_f32(
            rows,
            head_dim,
            rows,
            &probabilities,
            &value,
            None,
            &mut context,
        )?;
        for row in 0..rows {
            let source = row * head_dim;
            let target = row * hidden + head_start;
            joined[target..target + head_dim].copy_from_slice(&context[source..source + head_dim]);
        }
    }
    linear_rows(
        weights,
        compute,
        &format!("{base}.attn.out_proj.weight"),
        &joined,
        rows,
        hidden,
        hidden,
    )
}

fn layer_norm_rows(
    weights: &BarkMappedWeights,
    compute: &Compute,
    input: &[f32],
    rows: usize,
    hidden: usize,
    prefix: &str,
) -> Result<Vec<f32>> {
    let weight = weights.tensor(&format!("{prefix}.weight"), &[hidden])?;
    let bias = weights.tensor(&format!("{prefix}.bias"), &[hidden])?;
    let mut output = vec![0.0f32; input.len()];
    compute.layer_norm_f32(
        input,
        &mut output,
        rows,
        hidden,
        weight,
        bias,
        LAYER_NORM_EPS,
    )?;
    Ok(output)
}

fn layer_norm_row(
    compute: &Compute,
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
) -> Result<Vec<f32>> {
    let mut output = vec![0.0f32; input.len()];
    compute.layer_norm_f32(
        input,
        &mut output,
        1,
        input.len(),
        weight,
        bias,
        LAYER_NORM_EPS,
    )?;
    Ok(output)
}

fn gemv_tensor(
    weights: &BarkMappedWeights,
    compute: &Compute,
    name: &str,
    output: usize,
    input: usize,
    values: &[f32],
) -> Result<Vec<f32>> {
    if values.len() != input {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: `{name}` input has {}, expected {input}",
            values.len()
        )));
    }
    let weight = weights.tensor(name, &[output, input])?;
    let mut result = vec![0.0f32; output];
    compute.gemv_f32(output, input, weight, values, None, &mut result)?;
    Ok(result)
}

fn linear_rows(
    weights: &BarkMappedWeights,
    compute: &Compute,
    name: &str,
    values: &[f32],
    rows: usize,
    input: usize,
    output: usize,
) -> Result<Vec<f32>> {
    if rows == 0 || values.len() != rows * input {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: `{name}` rows input has {}, expected [{rows}, {input}]",
            values.len()
        )));
    }
    let weight = weights.tensor(name, &[output, input])?;
    let weight_t = transpose_linear(weight, output, input);
    let mut result = vec![0.0f32; rows * output];
    compute.gemm_f32(rows, output, input, values, &weight_t, None, &mut result)?;
    Ok(result)
}

fn transpose_linear(weight: &[f32], output: usize, input: usize) -> Vec<f32> {
    let mut transposed = vec![0.0f32; weight.len()];
    for out in 0..output {
        for inner in 0..input {
            transposed[inner * output + out] = weight[out * input + inner];
        }
    }
    transposed
}

fn add_assign(left: &mut [f32], right: &[f32]) -> Result<()> {
    if left.len() != right.len() {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: residual length mismatch {} != {}",
            left.len(),
            right.len()
        )));
    }
    for (target, &value) in left.iter_mut().zip(right) {
        *target += value;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transpose_linear_matches_hand_fixture() {
        // [out=2, in=3] -> [in=3, out=2]
        assert_eq!(
            transpose_linear(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3),
            vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
        );
    }

    #[test]
    fn cache_starts_empty_for_every_layer() {
        let cache = CausalCache::new(CausalStage::Semantic, 3);
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.layers.len(), 3);
        assert!(
            cache
                .layers
                .iter()
                .all(|layer| layer.key.is_empty() && layer.value.is_empty())
        );
    }
}
