//! Bounded-memory Qwen3 decoder with MOSS-Audio DeepStack injection.

use std::sync::Mutex;

use vokra_core::backend::BackendKind;
use vokra_core::gguf::GgufTensorInfo;
use vokra_core::{KvCache, Result, VokraError};

use crate::compute::{Compute, HotOp};
use crate::mapped_weights::{lock_scratch, transpose_widen, widen_into};

use super::audio_encoder::MossAudioEmbeddings;
use super::weights::MossAudioMappedDescriptors;
use super::{MossAudioCheckpoint, MossAudioTextConfig};

const PREFILL_CHUNK_ROWS: usize = 8;
const HEAD_CHUNK_ROWS: usize = 512;

/// Learned operations required by the Qwen3 text decoder.
pub const MOSS_AUDIO_TEXT_HOT_OPS: &[HotOp] = &[
    HotOp::Gemm,
    HotOp::Gemv,
    HotOp::Softmax,
    HotOp::RmsNorm,
    HotOp::Silu,
];

/// Whole-model CPU/Metal preflight contract.
pub const MOSS_AUDIO_HOT_OPS: &[HotOp] = &[
    HotOp::Gemm,
    HotOp::Gemv,
    HotOp::Softmax,
    HotOp::LayerNorm,
    HotOp::RmsNorm,
    HotOp::Gelu,
    HotOp::Silu,
];

/// Deterministic greedy-generation controls.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MossAudioGenerationOptions {
    /// Maximum number of autoregressive output tokens.
    pub max_new_tokens: usize,
}

impl Default for MossAudioGenerationOptions {
    fn default() -> Self {
        Self {
            max_new_tokens: 512,
        }
    }
}

/// Raw Qwen3 token output from the tokenizer-independent API.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MossAudioTokenOutput {
    token_ids: Vec<u32>,
}

impl MossAudioTokenOutput {
    #[must_use]
    pub fn token_ids(&self) -> &[u32] {
        &self.token_ids
    }

    #[must_use]
    pub fn into_token_ids(self) -> Vec<u32> {
        self.token_ids
    }
}

#[derive(Default)]
pub(super) struct MossAudioTextRuntime {
    block: Mutex<TextBlock>,
    head: Mutex<HeadScratch>,
}

#[derive(Default)]
struct TextBlock {
    input_norm: Vec<f32>,
    q_w_t: Vec<f32>,
    q_norm: Vec<f32>,
    k_w_t: Vec<f32>,
    k_norm: Vec<f32>,
    v_w_t: Vec<f32>,
    o_w_t: Vec<f32>,
    ffn_norm: Vec<f32>,
    gate_w_t: Vec<f32>,
    up_w_t: Vec<f32>,
    down_w_t: Vec<f32>,
}

#[derive(Default)]
struct HeadScratch {
    weights: Vec<f32>,
    logits: Vec<f32>,
}

#[derive(Default)]
struct TextStepScratch {
    hidden: Vec<f32>,
    norm: Vec<f32>,
    embedding_row: Vec<f32>,
    q_raw: Vec<f32>,
    q: Vec<f32>,
    k_raw: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    query: Vec<f32>,
    key_t: Vec<f32>,
    value: Vec<f32>,
    scores: Vec<f32>,
    probabilities: Vec<f32>,
    attended: Vec<f32>,
    attention: Vec<f32>,
    attention_out: Vec<f32>,
    ffn_gate: Vec<f32>,
    ffn_activated: Vec<f32>,
    ffn_up: Vec<f32>,
    ffn_down: Vec<f32>,
}

pub(super) fn generate(
    checkpoint: &MossAudioCheckpoint,
    backend: BackendKind,
    runtime: &MossAudioTextRuntime,
    prompt: &[u32],
    audio: &MossAudioEmbeddings,
    options: &MossAudioGenerationOptions,
) -> Result<MossAudioTokenOutput> {
    let mapped = checkpoint.mapped()?;
    let config = mapped.config().text;
    validate_audio_embeddings(audio, config)?;
    validate_generation_inputs(mapped, prompt, audio, options)?;

    let compute = Compute::for_backend(backend, MOSS_AUDIO_HOT_OPS)?;
    let reserve = prompt
        .len()
        .min(512)
        .saturating_add(options.max_new_tokens.min(128))
        .max(1);
    let mut kv_cache = KvCache::with_reserve(config.n_layer as usize, kv_dim(config), reserve);
    let mut scratch = TextStepScratch::default();
    let mut audio_offset = 0;
    for row_start in (0..prompt.len()).step_by(PREFILL_CHUNK_ROWS) {
        let rows = PREFILL_CHUNK_ROWS.min(prompt.len() - row_start);
        forward_chunk(
            &compute,
            mapped,
            runtime,
            &mut scratch,
            &mut kv_cache,
            &prompt[row_start..row_start + rows],
            audio,
            &mut audio_offset,
        )?;
    }
    if audio_offset != audio.frames() {
        return Err(VokraError::InvalidArgument(format!(
            "moss_audio: prompt consumed {audio_offset} audio rows, expected {}",
            audio.frames()
        )));
    }

    let mut generated = Vec::with_capacity(options.max_new_tokens);
    let mut next = last_argmax(&compute, mapped, runtime, &scratch)?;
    for step in 0..options.max_new_tokens {
        generated.push(next);
        if next == mapped.config().tokens.eos || step + 1 == options.max_new_tokens {
            break;
        }
        forward_chunk(
            &compute,
            mapped,
            runtime,
            &mut scratch,
            &mut kv_cache,
            &[next],
            audio,
            &mut audio_offset,
        )?;
        next = last_argmax(&compute, mapped, runtime, &scratch)?;
    }
    Ok(MossAudioTokenOutput {
        token_ids: generated,
    })
}

fn validate_audio_embeddings(
    audio: &MossAudioEmbeddings,
    config: MossAudioTextConfig,
) -> Result<()> {
    if audio.frames() == 0 {
        return Err(VokraError::InvalidArgument(
            "moss_audio: projected audio contains zero rows".to_owned(),
        ));
    }
    let required = audio.frames() * audio.hidden_size();
    if audio.hidden_size() != config.hidden_size as usize || audio.values().len() != required {
        return Err(VokraError::InvalidArgument(format!(
            "moss_audio: projected audio shape is [{},{}] with {} values; decoder requires hidden {}",
            audio.frames(),
            audio.hidden_size(),
            audio.values().len(),
            config.hidden_size
        )));
    }
    reject_non_finite("primary audio embeddings", audio.values())?;
    for (index, values) in audio.deepstack_values().into_iter().enumerate() {
        if values.len() != required {
            return Err(VokraError::InvalidArgument(format!(
                "moss_audio: DeepStack embedding {index} has {} values, expected {required}",
                values.len()
            )));
        }
        reject_non_finite("DeepStack audio embeddings", values)?;
    }
    Ok(())
}

fn validate_generation_inputs(
    mapped: &MossAudioMappedDescriptors,
    prompt: &[u32],
    audio: &MossAudioEmbeddings,
    options: &MossAudioGenerationOptions,
) -> Result<()> {
    let config = mapped.config();
    if prompt.is_empty() {
        return Err(VokraError::InvalidArgument(
            "moss_audio: prompt token sequence is empty".to_owned(),
        ));
    }
    if options.max_new_tokens == 0 {
        return Err(VokraError::InvalidArgument(
            "moss_audio: max_new_tokens must be greater than zero".to_owned(),
        ));
    }
    let required_positions = prompt
        .len()
        .checked_add(options.max_new_tokens - 1)
        .ok_or_else(|| {
            VokraError::InvalidArgument("moss_audio: generation position count overflows".into())
        })?;
    if required_positions > config.text.max_position_embeddings as usize {
        return Err(VokraError::InvalidArgument(format!(
            "moss_audio: prompt {} + at most {} forwarded generation rows exceeds max positions {}",
            prompt.len(),
            options.max_new_tokens - 1,
            config.text.max_position_embeddings
        )));
    }
    let placeholders = prompt
        .iter()
        .filter(|&&token| token == config.tokens.audio)
        .count();
    if placeholders != audio.frames() {
        return Err(VokraError::InvalidArgument(format!(
            "moss_audio: prompt has {placeholders} audio placeholders, but encoder produced {} rows",
            audio.frames()
        )));
    }
    if let Some((index, token)) = prompt
        .iter()
        .copied()
        .enumerate()
        .find(|(_, token)| *token >= config.text.vocab_size)
    {
        return Err(VokraError::InvalidArgument(format!(
            "moss_audio: prompt token {token} at row {index} is outside vocabulary 0..{}",
            config.text.vocab_size
        )));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn forward_chunk(
    compute: &Compute,
    mapped: &MossAudioMappedDescriptors,
    runtime: &MossAudioTextRuntime,
    scratch: &mut TextStepScratch,
    kv_cache: &mut KvCache,
    tokens: &[u32],
    audio: &MossAudioEmbeddings,
    audio_offset: &mut usize,
) -> Result<()> {
    let config = mapped.config().text;
    let rows = tokens.len();
    if rows == 0 {
        return Err(VokraError::InvalidArgument(
            "moss_audio: decoder chunk is empty".to_owned(),
        ));
    }
    let position_offset = kv_cache.positions();
    if position_offset + rows > config.max_position_embeddings as usize {
        return Err(VokraError::InvalidArgument(format!(
            "moss_audio: decode position {} exceeds max positions {}",
            position_offset + rows,
            config.max_position_embeddings
        )));
    }

    let chunk_audio_rows = embed_tokens(mapped, tokens, audio, audio_offset, scratch)?;
    let hidden = config.hidden_size as usize;
    let q_width = q_dim(config);
    let kv_width = kv_dim(config);
    let ffn = config.ffn_dim as usize;
    resize_zero(&mut scratch.norm, rows * hidden);
    resize_zero(&mut scratch.q_raw, rows * q_width);
    resize_zero(&mut scratch.q, rows * q_width);
    resize_zero(&mut scratch.k_raw, rows * kv_width);
    resize_zero(&mut scratch.k, rows * kv_width);
    resize_zero(&mut scratch.v, rows * kv_width);
    resize_zero(&mut scratch.query, rows * config.head_dim as usize);
    resize_zero(&mut scratch.attention, rows * q_width);
    resize_zero(&mut scratch.attention_out, rows * hidden);
    resize_zero(&mut scratch.ffn_gate, rows * ffn);
    resize_zero(&mut scratch.ffn_activated, rows * ffn);
    resize_zero(&mut scratch.ffn_up, rows * ffn);
    resize_zero(&mut scratch.ffn_down, rows * hidden);

    let mut block = lock_scratch(&runtime.block, mapped.mapped_model())?;
    for layer in 0..config.n_layer as usize {
        materialize_layer(mapped, layer, &mut block)?;
        compute.rms_norm_f32(
            &scratch.hidden,
            &mut scratch.norm,
            rows,
            hidden,
            &block.input_norm,
            config.rms_norm_eps,
        )?;
        compute.gemm_f32(
            rows,
            q_width,
            hidden,
            &scratch.norm,
            &block.q_w_t,
            None,
            &mut scratch.q_raw,
        )?;
        compute.gemm_f32(
            rows,
            kv_width,
            hidden,
            &scratch.norm,
            &block.k_w_t,
            None,
            &mut scratch.k_raw,
        )?;
        compute.gemm_f32(
            rows,
            kv_width,
            hidden,
            &scratch.norm,
            &block.v_w_t,
            None,
            &mut scratch.v,
        )?;
        compute.rms_norm_f32(
            &scratch.q_raw,
            &mut scratch.q,
            rows * config.n_head as usize,
            config.head_dim as usize,
            &block.q_norm,
            config.rms_norm_eps,
        )?;
        compute.rms_norm_f32(
            &scratch.k_raw,
            &mut scratch.k,
            rows * config.n_kv_head as usize,
            config.head_dim as usize,
            &block.k_norm,
            config.rms_norm_eps,
        )?;
        apply_half_split_rope(
            &mut scratch.q,
            rows,
            config.n_head as usize,
            config.head_dim as usize,
            config.rope_theta,
            position_offset,
        )?;
        apply_half_split_rope(
            &mut scratch.k,
            rows,
            config.n_kv_head as usize,
            config.head_dim as usize,
            config.rope_theta,
            position_offset,
        )?;

        kv_cache.append(layer, &scratch.k, &scratch.v);
        attention(
            compute,
            scratch,
            kv_cache.k(layer),
            kv_cache.v(layer),
            rows,
            position_offset,
            config,
        )?;
        compute.gemm_f32(
            rows,
            hidden,
            q_width,
            &scratch.attention,
            &block.o_w_t,
            None,
            &mut scratch.attention_out,
        )?;
        for (value, residual) in scratch.hidden.iter_mut().zip(&scratch.attention_out) {
            *value += residual;
        }

        compute.rms_norm_f32(
            &scratch.hidden,
            &mut scratch.norm,
            rows,
            hidden,
            &block.ffn_norm,
            config.rms_norm_eps,
        )?;
        compute.gemm_f32(
            rows,
            ffn,
            hidden,
            &scratch.norm,
            &block.gate_w_t,
            None,
            &mut scratch.ffn_gate,
        )?;
        compute.gemm_f32(
            rows,
            ffn,
            hidden,
            &scratch.norm,
            &block.up_w_t,
            None,
            &mut scratch.ffn_up,
        )?;
        compute.silu_f32(&scratch.ffn_gate, &mut scratch.ffn_activated)?;
        for (activated, up) in scratch.ffn_activated.iter_mut().zip(&scratch.ffn_up) {
            *activated *= up;
        }
        compute.gemm_f32(
            rows,
            hidden,
            ffn,
            &scratch.ffn_activated,
            &block.down_w_t,
            None,
            &mut scratch.ffn_down,
        )?;
        for (value, residual) in scratch.hidden.iter_mut().zip(&scratch.ffn_down) {
            *value += residual;
        }
        if layer < mapped.config().deepstack_num_inject_layers as usize {
            inject_deepstack(&mut scratch.hidden, hidden, &chunk_audio_rows, audio, layer)?;
        }
    }
    kv_cache.advance(rows);

    let mut final_norm = Vec::new();
    widen_tensor(mapped, mapped.text_final_norm(), &mut final_norm)?;
    compute.rms_norm_f32(
        &scratch.hidden,
        &mut scratch.norm,
        rows,
        hidden,
        &final_norm,
        config.rms_norm_eps,
    )?;
    reject_non_finite("final decoder hidden", &scratch.norm)
}

fn embed_tokens(
    mapped: &MossAudioMappedDescriptors,
    tokens: &[u32],
    audio: &MossAudioEmbeddings,
    audio_offset: &mut usize,
    scratch: &mut TextStepScratch,
) -> Result<Vec<(usize, usize)>> {
    let config = mapped.config();
    let hidden = config.text.hidden_size as usize;
    resize_zero(&mut scratch.hidden, tokens.len() * hidden);
    let mut audio_rows = Vec::new();
    for (row, &token) in tokens.iter().enumerate() {
        if token >= config.text.vocab_size {
            return Err(VokraError::InvalidArgument(format!(
                "moss_audio: text token {token} at row {row} is outside vocabulary 0..{}",
                config.text.vocab_size
            )));
        }
        let target = &mut scratch.hidden[row * hidden..(row + 1) * hidden];
        if token == config.tokens.audio {
            if *audio_offset >= audio.frames() {
                return Err(VokraError::InvalidArgument(format!(
                    "moss_audio: audio placeholder at decoder row {row} exceeds {} projected rows",
                    audio.frames()
                )));
            }
            let source = &audio.values()[*audio_offset * hidden..(*audio_offset + 1) * hidden];
            target.copy_from_slice(source);
            audio_rows.push((row, *audio_offset));
            *audio_offset += 1;
        } else {
            widen_row(
                mapped,
                mapped.text_embedding(),
                token as usize,
                &mut scratch.embedding_row,
            )?;
            target.copy_from_slice(&scratch.embedding_row);
        }
    }
    reject_non_finite("decoder embeddings", &scratch.hidden)?;
    Ok(audio_rows)
}

fn inject_deepstack(
    hidden_values: &mut [f32],
    hidden: usize,
    chunk_audio_rows: &[(usize, usize)],
    audio: &MossAudioEmbeddings,
    layer: usize,
) -> Result<()> {
    let deepstack = audio.deepstack_values();
    let values = deepstack.get(layer).ok_or_else(|| {
        VokraError::ModelLoad(format!(
            "moss_audio: decoder layer {layer} has no matching DeepStack adapter"
        ))
    })?;
    for &(row, audio_row) in chunk_audio_rows {
        let target = hidden_values
            .get_mut(row * hidden..(row + 1) * hidden)
            .ok_or_else(|| {
                VokraError::InvalidArgument(
                    "moss_audio: DeepStack target row is outside decoder chunk".to_owned(),
                )
            })?;
        let source = values
            .get(audio_row * hidden..(audio_row + 1) * hidden)
            .ok_or_else(|| {
                VokraError::InvalidArgument(
                    "moss_audio: DeepStack source row is outside audio embeddings".to_owned(),
                )
            })?;
        for (target, source) in target.iter_mut().zip(source) {
            *target += source;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn attention(
    compute: &Compute,
    scratch: &mut TextStepScratch,
    key_cache: &[f32],
    value_cache: &[f32],
    rows: usize,
    position_offset: usize,
    config: MossAudioTextConfig,
) -> Result<()> {
    let q_width = q_dim(config);
    let kv_width = kv_dim(config);
    let head_dim = config.head_dim as usize;
    let total_rows = position_offset + rows;
    let expected = total_rows * kv_width;
    if key_cache.len() != expected || value_cache.len() != expected {
        return Err(VokraError::InvalidArgument(format!(
            "moss_audio: KV cache length mismatch: key={}, value={}, expected={expected}",
            key_cache.len(),
            value_cache.len()
        )));
    }
    resize_zero(&mut scratch.key_t, head_dim * total_rows);
    resize_zero(&mut scratch.value, total_rows * head_dim);
    resize_zero(&mut scratch.scores, rows * total_rows);
    resize_zero(&mut scratch.probabilities, rows * total_rows);
    resize_zero(&mut scratch.attended, rows * head_dim);
    scratch.attention.fill(0.0);
    let groups = config.n_head as usize / config.n_kv_head as usize;
    let scale = (head_dim as f32).sqrt().recip();

    for kv_head in 0..config.n_kv_head as usize {
        for position in 0..total_rows {
            let source = position * kv_width + kv_head * head_dim;
            for dimension in 0..head_dim {
                scratch.key_t[dimension * total_rows + position] = key_cache[source + dimension];
                scratch.value[position * head_dim + dimension] = value_cache[source + dimension];
            }
        }
        for group in 0..groups {
            let q_head = kv_head * groups + group;
            for row in 0..rows {
                let source = row * q_width + q_head * head_dim;
                scratch.query[row * head_dim..(row + 1) * head_dim]
                    .copy_from_slice(&scratch.q[source..source + head_dim]);
            }
            compute.gemm_f32(
                rows,
                total_rows,
                head_dim,
                &scratch.query,
                &scratch.key_t,
                None,
                &mut scratch.scores,
            )?;
            scale_and_mask(
                &mut scratch.scores,
                rows,
                total_rows,
                position_offset,
                scale,
            )?;
            compute.softmax_f32(
                &scratch.scores,
                &mut scratch.probabilities,
                rows,
                total_rows,
            )?;
            compute.gemm_f32(
                rows,
                head_dim,
                total_rows,
                &scratch.probabilities,
                &scratch.value,
                None,
                &mut scratch.attended,
            )?;
            for row in 0..rows {
                let target = row * q_width + q_head * head_dim;
                scratch.attention[target..target + head_dim]
                    .copy_from_slice(&scratch.attended[row * head_dim..(row + 1) * head_dim]);
            }
        }
    }
    Ok(())
}

fn scale_and_mask(
    scores: &mut [f32],
    rows: usize,
    total_rows: usize,
    position_offset: usize,
    scale: f32,
) -> Result<()> {
    if rows == 0
        || total_rows < rows
        || position_offset + rows != total_rows
        || scores.len() != rows * total_rows
        || !scale.is_finite()
    {
        return Err(VokraError::InvalidArgument(format!(
            "moss_audio: causal mask shape mismatch: scores={}, rows={rows}, total_rows={total_rows}, offset={position_offset}, scale={scale}",
            scores.len()
        )));
    }
    for row in 0..rows {
        let last_visible = position_offset + row;
        for column in 0..total_rows {
            let score = &mut scores[row * total_rows + column];
            if column > last_visible {
                *score = f32::MIN;
            } else {
                *score *= scale;
            }
        }
    }
    Ok(())
}

fn apply_half_split_rope(
    values: &mut [f32],
    rows: usize,
    heads: usize,
    head_dim: usize,
    rope_base: f32,
    position_offset: usize,
) -> Result<()> {
    if rows == 0
        || heads == 0
        || !head_dim.is_multiple_of(2)
        || !rope_base.is_finite()
        || rope_base <= 0.0
        || values.len() != rows * heads * head_dim
    {
        return Err(VokraError::InvalidArgument(format!(
            "moss_audio: RoPE shape mismatch: values={}, rows={rows}, heads={heads}, head_dim={head_dim}, rope_base={rope_base}",
            values.len()
        )));
    }
    let half = head_dim / 2;
    for row in 0..rows {
        let position = (position_offset + row) as f32;
        for head in 0..heads {
            let base = (row * heads + head) * head_dim;
            for pair in 0..half {
                let frequency = rope_base.powf(-(2 * pair) as f32 / head_dim as f32);
                let angle = position * frequency;
                let (sin, cos) = angle.sin_cos();
                let first = values[base + pair];
                let second = values[base + half + pair];
                values[base + pair] = first * cos - second * sin;
                values[base + half + pair] = first * sin + second * cos;
            }
        }
    }
    Ok(())
}

fn materialize_layer(
    mapped: &MossAudioMappedDescriptors,
    layer: usize,
    block: &mut TextBlock,
) -> Result<()> {
    let config = mapped.config().text;
    let hidden = config.hidden_size as usize;
    let q_width = q_dim(config);
    let kv_width = kv_dim(config);
    let ffn = config.ffn_dim as usize;
    let descriptors = mapped.text_layer(layer);
    widen_tensor(mapped, descriptors.input_norm, &mut block.input_norm)?;
    transpose_tensor(mapped, descriptors.q, q_width, hidden, &mut block.q_w_t)?;
    widen_tensor(mapped, descriptors.q_norm, &mut block.q_norm)?;
    transpose_tensor(mapped, descriptors.k, kv_width, hidden, &mut block.k_w_t)?;
    widen_tensor(mapped, descriptors.k_norm, &mut block.k_norm)?;
    transpose_tensor(mapped, descriptors.v, kv_width, hidden, &mut block.v_w_t)?;
    transpose_tensor(mapped, descriptors.o, hidden, q_width, &mut block.o_w_t)?;
    widen_tensor(mapped, descriptors.ffn_norm, &mut block.ffn_norm)?;
    transpose_tensor(mapped, descriptors.gate, ffn, hidden, &mut block.gate_w_t)?;
    transpose_tensor(mapped, descriptors.up, ffn, hidden, &mut block.up_w_t)?;
    transpose_tensor(mapped, descriptors.down, hidden, ffn, &mut block.down_w_t)?;
    Ok(())
}

fn last_argmax(
    compute: &Compute,
    mapped: &MossAudioMappedDescriptors,
    runtime: &MossAudioTextRuntime,
    scratch: &TextStepScratch,
) -> Result<u32> {
    let config = mapped.config().text;
    let hidden_width = config.hidden_size as usize;
    if scratch.norm.len() < hidden_width {
        return Err(VokraError::InvalidArgument(
            "moss_audio: no final decoder hidden row is available".to_owned(),
        ));
    }
    let hidden = &scratch.norm[scratch.norm.len() - hidden_width..];
    let mut head = lock_scratch(&runtime.head, mapped.mapped_model())?;
    let HeadScratch { weights, logits } = &mut *head;
    let mut best_token = 0_u32;
    let mut best_value = f32::NEG_INFINITY;
    let vocab = config.vocab_size as usize;
    let mut first_row = 0;
    while first_row < vocab {
        let rows = HEAD_CHUNK_ROWS.min(vocab - first_row);
        widen_rows(mapped, mapped.text_head(), first_row, rows, weights)?;
        logits.clear();
        logits.resize(rows, 0.0);
        compute.gemv_f32(rows, hidden_width, weights, hidden, None, logits)?;
        reject_non_finite("vocabulary logits", logits)?;
        for (offset, &value) in logits.iter().enumerate() {
            if value > best_value {
                best_value = value;
                best_token = (first_row + offset) as u32;
            }
        }
        first_row += rows;
    }
    Ok(best_token)
}

fn q_dim(config: MossAudioTextConfig) -> usize {
    config.n_head as usize * config.head_dim as usize
}

fn kv_dim(config: MossAudioTextConfig) -> usize {
    config.n_kv_head as usize * config.head_dim as usize
}

fn resize_zero(values: &mut Vec<f32>, len: usize) {
    values.clear();
    values.resize(len, 0.0);
}

fn widen_row(
    mapped: &MossAudioMappedDescriptors,
    info: &GgufTensorInfo,
    row: usize,
    output: &mut Vec<f32>,
) -> Result<()> {
    widen_rows(mapped, info, row, 1, output)
}

fn widen_rows(
    mapped: &MossAudioMappedDescriptors,
    info: &GgufTensorInfo,
    first_row: usize,
    rows: usize,
    output: &mut Vec<f32>,
) -> Result<()> {
    let hidden = mapped.config().text.hidden_size as usize;
    let element_size = info.dtype.type_size();
    let bytes = mapped.file().tensor_bytes(info);
    let start = first_row
        .checked_mul(hidden)
        .and_then(|value| value.checked_mul(element_size))
        .ok_or_else(|| VokraError::InvalidArgument("moss_audio: row offset overflow".into()))?;
    let len = rows
        .checked_mul(hidden)
        .and_then(|value| value.checked_mul(element_size))
        .ok_or_else(|| VokraError::InvalidArgument("moss_audio: row length overflow".into()))?;
    let end = start
        .checked_add(len)
        .ok_or_else(|| VokraError::InvalidArgument("moss_audio: row range overflow".into()))?;
    let source = bytes.get(start..end).ok_or_else(|| {
        VokraError::ModelLoad(format!(
            "moss_audio: tensor `{}` row range {first_row}..{} exceeds {} bytes",
            info.name,
            first_row + rows,
            bytes.len()
        ))
    })?;
    widen_into(source, info.dtype, output, mapped.mapped_model())
}

fn widen_tensor(
    mapped: &MossAudioMappedDescriptors,
    info: &GgufTensorInfo,
    output: &mut Vec<f32>,
) -> Result<()> {
    widen_into(
        mapped.file().tensor_bytes(info),
        info.dtype,
        output,
        mapped.mapped_model(),
    )
}

fn transpose_tensor(
    mapped: &MossAudioMappedDescriptors,
    info: &GgufTensorInfo,
    rows: usize,
    columns: usize,
    output: &mut Vec<f32>,
) -> Result<()> {
    transpose_widen(
        mapped.file().tensor_bytes(info),
        info.dtype,
        rows,
        columns,
        output,
        mapped.mapped_model(),
    )
}

fn reject_non_finite(value_label: &str, values: &[f32]) -> Result<()> {
    if let Some((index, value)) = values
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(VokraError::InvalidArgument(format!(
            "moss_audio: {value_label} contains non-finite value {value} at index {index}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complete_backend_contract_covers_audio_and_decoder() {
        for op in super::super::MOSS_AUDIO_ENCODER_HOT_OPS
            .iter()
            .chain(MOSS_AUDIO_TEXT_HOT_OPS)
        {
            assert!(MOSS_AUDIO_HOT_OPS.contains(op));
        }
        Compute::for_backend(BackendKind::Cpu, MOSS_AUDIO_HOT_OPS)
            .expect("CPU covers complete MOSS-Audio graph");
        #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
        match Compute::for_backend(BackendKind::Metal, MOSS_AUDIO_HOT_OPS) {
            Ok(compute) => assert_eq!(compute.backend_name(), "metal"),
            Err(VokraError::BackendUnavailable(_)) => {}
            Err(error) => panic!("MOSS-Audio has a Metal coverage gap: {error}"),
        }
    }

    #[test]
    fn half_split_rope_position_zero_is_identity() {
        let mut values = (0..128).map(|value| value as f32).collect::<Vec<_>>();
        let expected = values.clone();
        apply_half_split_rope(&mut values, 1, 1, 128, 1_000_000.0, 0).expect("rope");
        assert_eq!(values, expected);
    }

    #[test]
    fn causal_mask_respects_cached_prefix() {
        let mut scores = vec![2.0; 6];
        scale_and_mask(&mut scores, 2, 3, 1, 0.5).expect("mask");
        assert_eq!(&scores[..3], &[1.0, 1.0, f32::MIN]);
        assert_eq!(&scores[3..], &[1.0, 1.0, 1.0]);
    }

    #[test]
    fn dimensions_preserve_qwen3_projection_widths() {
        for variant in [
            super::super::MossAudioVariant::B4Instruct,
            super::super::MossAudioVariant::B8Instruct,
        ] {
            let config = variant.config().text;
            assert_eq!(q_dim(config), 4_096);
            assert_eq!(kv_dim(config), 1_024);
        }
    }
}
