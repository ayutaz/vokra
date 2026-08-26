//! Bounded-memory native Qwen3 autoregressive decoder for Qwen3-ASR.
//!
//! Dense checkpoint tensors remain in the GGUF mapping. One decoder layer is
//! widened and transposed into a reused scratch block, while embeddings and
//! the vocabulary head are read by row/chunk. Every learned projection,
//! reduction and activation dispatches through one selected [`Compute`]
//! backend. Embedding gather/replacement, RoPE, causal masking, residual adds
//! and argmax remain deterministic host glue.

use std::sync::Mutex;

use vokra_core::backend::BackendKind;
use vokra_core::gguf::GgufTensorInfo;
use vokra_core::{KvCache, Result, VokraError};

use crate::compute::{Compute, HotOp};
use crate::mapped_weights::{lock_scratch, transpose_widen, widen_into};

use super::audio_encoder::Qwen3AsrAudioEmbeddings;
use super::tokenizer::{AUDIO_PAD_TOKEN_ID, Qwen3AsrTokenizer, Qwen3AsrTranscription, is_eos};
use super::weights::Qwen3AsrMappedDescriptors;
use super::{Qwen3AsrCheckpoint, Qwen3AsrTextConfig};

const PREFILL_CHUNK_ROWS: usize = 8;
const HEAD_CHUNK_ROWS: usize = 512;

/// Every learned op required by the Qwen3 text decoder.
pub const QWEN3_ASR_TEXT_HOT_OPS: &[HotOp] = &[
    HotOp::Gemm,
    HotOp::Gemv,
    HotOp::Softmax,
    HotOp::RmsNorm,
    HotOp::Silu,
];

/// Whole-model Qwen3-ASR learned-op contract.
///
/// Backend preflight uses this union before either the audio tower or text
/// decoder runs, so no stage can silently drop to CPU.
pub const QWEN3_ASR_HOT_OPS: &[HotOp] = &[
    HotOp::Gemm,
    HotOp::Gemv,
    HotOp::Softmax,
    HotOp::LayerNorm,
    HotOp::RmsNorm,
    HotOp::Gelu,
    HotOp::Silu,
];

/// Deterministic Qwen3-ASR generation controls.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen3AsrGenerationOptions {
    /// Optional system context inserted verbatim into the official template.
    pub context: String,
    /// Optional official language name. When set, the prompt forces text-only
    /// generation after `language X<asr_text>`.
    pub language: Option<String>,
    /// Maximum autoregressive tokens. The released wrapper defaults to 512.
    pub max_new_tokens: usize,
}

impl Default for Qwen3AsrGenerationOptions {
    fn default() -> Self {
        Self {
            context: String::new(),
            language: None,
            max_new_tokens: 512,
        }
    }
}

#[derive(Default)]
pub(super) struct Qwen3AsrTextRuntime {
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

pub(super) fn transcribe(
    checkpoint: &Qwen3AsrCheckpoint,
    backend: BackendKind,
    runtime: &Qwen3AsrTextRuntime,
    tokenizer: &Qwen3AsrTokenizer,
    audio: &Qwen3AsrAudioEmbeddings,
    options: &Qwen3AsrGenerationOptions,
) -> Result<Qwen3AsrTranscription> {
    let mapped = checkpoint.mapped()?;
    let config = mapped.config().text;
    validate_audio_embeddings(audio, config, mapped.mapped_model().name)?;
    let prompt = tokenizer.prompt_ids(
        audio.frames(),
        Some(&options.context),
        options.language.as_deref(),
    )?;
    validate_generation_inputs(&prompt, audio, options, config, mapped.mapped_model().name)?;

    let compute = Compute::for_backend(backend, QWEN3_ASR_HOT_OPS)?;
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
            "{}: prompt consumed {audio_offset} audio rows, expected {}",
            mapped.mapped_model().name,
            audio.frames()
        )));
    }

    let mut generated = Vec::with_capacity(options.max_new_tokens);
    let mut next = last_argmax(&compute, mapped, runtime, &scratch)?;
    for step in 0..options.max_new_tokens {
        if is_eos(next) {
            break;
        }
        generated.push(next);
        if step + 1 == options.max_new_tokens {
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
    tokenizer.parse_generated_ids(&generated, options.language.as_deref())
}

fn validate_audio_embeddings(
    audio: &Qwen3AsrAudioEmbeddings,
    config: Qwen3AsrTextConfig,
    label: &str,
) -> Result<()> {
    if audio.frames() == 0 {
        return Err(VokraError::InvalidArgument(format!(
            "{label}: projected audio contains zero rows"
        )));
    }
    if audio.hidden_size() != config.hidden_size as usize
        || audio.values().len() != audio.frames() * audio.hidden_size()
    {
        return Err(VokraError::InvalidArgument(format!(
            "{label}: projected audio shape is [{},{}] with {} values; decoder requires hidden {}",
            audio.frames(),
            audio.hidden_size(),
            audio.values().len(),
            config.hidden_size
        )));
    }
    reject_non_finite(label, "projected audio", audio.values())
}

fn validate_generation_inputs(
    prompt: &[u32],
    audio: &Qwen3AsrAudioEmbeddings,
    options: &Qwen3AsrGenerationOptions,
    config: Qwen3AsrTextConfig,
    label: &str,
) -> Result<()> {
    if prompt.is_empty() {
        return Err(VokraError::InvalidArgument(format!(
            "{label}: generated ChatML prompt is empty"
        )));
    }
    if options.max_new_tokens == 0 {
        return Err(VokraError::InvalidArgument(format!(
            "{label}: max_new_tokens must be greater than zero"
        )));
    }
    let required_positions = prompt
        .len()
        .checked_add(options.max_new_tokens - 1)
        .ok_or_else(|| {
            VokraError::InvalidArgument(format!("{label}: generation position count overflows"))
        })?;
    if required_positions > config.max_position_embeddings as usize {
        return Err(VokraError::InvalidArgument(format!(
            "{label}: prompt {} + at most {} forwarded generation rows exceeds max positions {}",
            prompt.len(),
            options.max_new_tokens - 1,
            config.max_position_embeddings
        )));
    }
    let placeholders = prompt
        .iter()
        .filter(|&&token| token == AUDIO_PAD_TOKEN_ID)
        .count();
    if placeholders != audio.frames() {
        return Err(VokraError::InvalidArgument(format!(
            "{label}: prompt has {placeholders} audio placeholders, but encoder produced {} rows",
            audio.frames()
        )));
    }
    if let Some((index, token)) = prompt
        .iter()
        .copied()
        .enumerate()
        .find(|(_, token)| *token >= config.vocab_size)
    {
        return Err(VokraError::InvalidArgument(format!(
            "{label}: prompt token {token} at row {index} is outside vocabulary 0..{}",
            config.vocab_size
        )));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn forward_chunk(
    compute: &Compute,
    mapped: &Qwen3AsrMappedDescriptors,
    runtime: &Qwen3AsrTextRuntime,
    scratch: &mut TextStepScratch,
    kv_cache: &mut KvCache,
    tokens: &[u32],
    audio: &Qwen3AsrAudioEmbeddings,
    audio_offset: &mut usize,
) -> Result<()> {
    let config = mapped.config().text;
    let rows = tokens.len();
    let label = mapped.mapped_model().name;
    if rows == 0 {
        return Err(VokraError::InvalidArgument(format!(
            "{label}: decoder chunk is empty"
        )));
    }
    let position_offset = kv_cache.positions();
    if position_offset + rows > config.max_position_embeddings as usize {
        return Err(VokraError::InvalidArgument(format!(
            "{label}: decode position {} exceeds max positions {}",
            position_offset + rows,
            config.max_position_embeddings
        )));
    }

    embed_tokens(mapped, tokens, audio, audio_offset, scratch)?;
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
            label,
        )?;
        apply_half_split_rope(
            &mut scratch.k,
            rows,
            config.n_kv_head as usize,
            config.head_dim as usize,
            config.rope_theta,
            position_offset,
            label,
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
            label,
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
    reject_non_finite(label, "final decoder hidden", &scratch.norm)
}

fn embed_tokens(
    mapped: &Qwen3AsrMappedDescriptors,
    tokens: &[u32],
    audio: &Qwen3AsrAudioEmbeddings,
    audio_offset: &mut usize,
    scratch: &mut TextStepScratch,
) -> Result<()> {
    let config = mapped.config();
    let hidden = config.text.hidden_size as usize;
    let label = mapped.mapped_model().name;
    resize_zero(&mut scratch.hidden, tokens.len() * hidden);
    for (row, &token) in tokens.iter().enumerate() {
        if token >= config.text.vocab_size {
            return Err(VokraError::InvalidArgument(format!(
                "{label}: text token {token} at row {row} is outside vocabulary 0..{}",
                config.text.vocab_size
            )));
        }
        let target = &mut scratch.hidden[row * hidden..(row + 1) * hidden];
        if token == config.audio_token_id {
            if *audio_offset >= audio.frames() {
                return Err(VokraError::InvalidArgument(format!(
                    "{label}: audio placeholder at decoder row {row} exceeds {} projected rows",
                    audio.frames()
                )));
            }
            let source = &audio.values()[*audio_offset * hidden..(*audio_offset + 1) * hidden];
            target.copy_from_slice(source);
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
    reject_non_finite(label, "decoder embeddings", &scratch.hidden)
}

#[allow(clippy::too_many_arguments)]
fn attention(
    compute: &Compute,
    scratch: &mut TextStepScratch,
    key_cache: &[f32],
    value_cache: &[f32],
    rows: usize,
    position_offset: usize,
    config: Qwen3AsrTextConfig,
    label: &str,
) -> Result<()> {
    let q_width = q_dim(config);
    let kv_width = kv_dim(config);
    let head_dim = config.head_dim as usize;
    let total_rows = position_offset + rows;
    let expected = total_rows * kv_width;
    if key_cache.len() != expected || value_cache.len() != expected {
        return Err(VokraError::InvalidArgument(format!(
            "{label}: KV cache length mismatch: key={}, value={}, expected={expected}",
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
                label,
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
    label: &str,
) -> Result<()> {
    if rows == 0
        || total_rows < rows
        || position_offset + rows != total_rows
        || scores.len() != rows * total_rows
        || !scale.is_finite()
    {
        return Err(VokraError::InvalidArgument(format!(
            "{label}: causal mask shape mismatch: scores={}, rows={rows}, total_rows={total_rows}, offset={position_offset}, scale={scale}",
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
    label: &str,
) -> Result<()> {
    if rows == 0
        || heads == 0
        || !head_dim.is_multiple_of(2)
        || !rope_base.is_finite()
        || rope_base <= 0.0
        || values.len() != rows * heads * head_dim
    {
        return Err(VokraError::InvalidArgument(format!(
            "{label}: RoPE shape mismatch: values={}, rows={rows}, heads={heads}, head_dim={head_dim}, rope_base={rope_base}",
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
    mapped: &Qwen3AsrMappedDescriptors,
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
    mapped: &Qwen3AsrMappedDescriptors,
    runtime: &Qwen3AsrTextRuntime,
    scratch: &TextStepScratch,
) -> Result<u32> {
    let config = mapped.config().text;
    let hidden_width = config.hidden_size as usize;
    let label = mapped.mapped_model().name;
    if scratch.norm.len() < hidden_width {
        return Err(VokraError::InvalidArgument(format!(
            "{label}: no final decoder hidden row is available"
        )));
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
        reject_non_finite(label, "vocabulary logits", logits)?;
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

fn q_dim(config: Qwen3AsrTextConfig) -> usize {
    config.n_head as usize * config.head_dim as usize
}

fn kv_dim(config: Qwen3AsrTextConfig) -> usize {
    config.n_kv_head as usize * config.head_dim as usize
}

fn resize_zero(values: &mut Vec<f32>, len: usize) {
    values.clear();
    values.resize(len, 0.0);
}

fn widen_row(
    mapped: &Qwen3AsrMappedDescriptors,
    info: &GgufTensorInfo,
    row: usize,
    output: &mut Vec<f32>,
) -> Result<()> {
    widen_rows(mapped, info, row, 1, output)
}

fn widen_rows(
    mapped: &Qwen3AsrMappedDescriptors,
    info: &GgufTensorInfo,
    first_row: usize,
    rows: usize,
    output: &mut Vec<f32>,
) -> Result<()> {
    let hidden = mapped.config().text.hidden_size as usize;
    let label = mapped.mapped_model().name;
    let element_size = info.dtype.type_size();
    let bytes = mapped.file().tensor_bytes(info);
    let start = first_row
        .checked_mul(hidden)
        .and_then(|value| value.checked_mul(element_size))
        .ok_or_else(|| VokraError::InvalidArgument(format!("{label}: row offset overflow")))?;
    let len = rows
        .checked_mul(hidden)
        .and_then(|value| value.checked_mul(element_size))
        .ok_or_else(|| VokraError::InvalidArgument(format!("{label}: row length overflow")))?;
    let end = start
        .checked_add(len)
        .ok_or_else(|| VokraError::InvalidArgument(format!("{label}: row range overflow")))?;
    let source = bytes.get(start..end).ok_or_else(|| {
        VokraError::ModelLoad(format!(
            "{label}: tensor `{}` row range {first_row}..{} exceeds {} bytes",
            info.name,
            first_row + rows,
            bytes.len()
        ))
    })?;
    widen_into(source, info.dtype, output, mapped.mapped_model())
}

fn widen_tensor(
    mapped: &Qwen3AsrMappedDescriptors,
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
    mapped: &Qwen3AsrMappedDescriptors,
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

fn reject_non_finite(label: &str, value_label: &str, values: &[f32]) -> Result<()> {
    if let Some((index, value)) = values
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(VokraError::InvalidArgument(format!(
            "{label}: {value_label} contains non-finite value {value} at index {index}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complete_backend_contract_covers_audio_and_decoder() {
        for op in super::super::QWEN3_ASR_AUDIO_HOT_OPS
            .iter()
            .chain(QWEN3_ASR_TEXT_HOT_OPS)
        {
            assert!(
                QWEN3_ASR_HOT_OPS.contains(op),
                "whole-model contract omits {op:?}"
            );
        }
        Compute::for_backend(BackendKind::Cpu, QWEN3_ASR_HOT_OPS)
            .expect("CPU covers complete Qwen3-ASR");
        #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
        match Compute::for_backend(BackendKind::Metal, QWEN3_ASR_HOT_OPS) {
            Ok(compute) => assert_eq!(compute.backend_name(), "metal"),
            Err(VokraError::BackendUnavailable(_)) => {}
            Err(error) => panic!("Qwen3-ASR has a Metal coverage gap: {error}"),
        }
    }

    #[test]
    fn half_split_rope_position_zero_is_identity() {
        let mut values = (0..128).map(|value| value as f32).collect::<Vec<_>>();
        let expected = values.clone();
        apply_half_split_rope(&mut values, 1, 1, 128, 1_000_000.0, 0, "qwen3_asr").expect("rope");
        assert_eq!(values, expected);
    }

    #[test]
    fn causal_mask_respects_cached_prefix() {
        let mut scores = vec![2.0; 6];
        scale_and_mask(&mut scores, 2, 3, 1, 0.5, "qwen3_asr").expect("mask");
        assert_eq!(&scores[..3], &[1.0, 1.0, f32::MIN]);
        assert_eq!(&scores[3..], &[1.0, 1.0, 1.0]);
    }

    #[test]
    fn dimensions_preserve_qwen3_nonstandard_q_width() {
        let config = super::super::Qwen3AsrVariant::B06.config().text;
        assert_eq!(q_dim(config), 2_048);
        assert_eq!(kv_dim(config), 1_024);
        assert_eq!(config.hidden_size, 1_024);
    }
}
