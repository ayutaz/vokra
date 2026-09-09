//! Bounded-memory Llama 3.2 decoder for the separately acquired companion.
//!
//! The tied embedding table stays mmap-backed and doubles as the vocabulary
//! head. CPU and Metal routes consume mapped BF16 linear weights through
//! bounded transposed panels; only the two norm vectors are widened per layer.
//! Other GPU routes retain their established dense-F32 path. The caller
//! supplies an exact pre-tokenized prompt plus the
//! consecutive span whose ordinary token embeddings are replaced by projected
//! Ultravox audio.

use std::sync::Mutex;

use vokra_backend_cpu::kernels::GemmF32Bf16BitsScratch;
use vokra_core::backend::BackendKind;
use vokra_core::gguf::{GgmlType, GgufTensorInfo};
use vokra_core::{KvCache, Result, VokraError};

use crate::compute::Compute;
use crate::mapped_weights::{lock_scratch, transpose_widen, widen_into};

use super::companion::{
    ULTRAVOX_LLAMA_HOT_OPS, UltravoxGeneration, UltravoxGenerationOptions, UltravoxLlamaConfig,
};
use super::companion_weights::UltravoxLlamaMappedDescriptors;
use super::projector::UltravoxAudioEmbeddings;

const PREFILL_CHUNK_ROWS: usize = 8;
const HEAD_CHUNK_ROWS: usize = 512;
const BF16_CPU_PANEL_COLUMNS: usize = 8;
const LABEL: &str = "ultravox_llama_companion";

#[derive(Default)]
pub(super) struct UltravoxLlamaDecoderRuntime {
    block: Mutex<DecoderBlock>,
    head: Mutex<HeadScratch>,
}

#[derive(Default)]
struct DecoderBlock {
    input_norm: Vec<f32>,
    q_w_t: Vec<f32>,
    k_w_t: Vec<f32>,
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
    mixed: MixedLinearScratch,
    logits: Vec<f32>,
}

#[derive(Default)]
struct MixedLinearScratch {
    /// Transposed raw BF16 `[input, panel_columns]` payload. CPU keeps this at
    /// at most 8 columns; Metal reuses it for one complete logical linear.
    panel: Vec<u16>,
    /// Reusable CPU kernel scratch for the widened panel and SIMD output tile.
    /// Metal leaves this scratch untouched and writes directly to its output.
    kernel: GemmF32Bf16BitsScratch,
}

#[derive(Default)]
struct StepScratch {
    hidden: Vec<f32>,
    norm: Vec<f32>,
    embedding_row: Vec<f32>,
    q: Vec<f32>,
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
    mixed: MixedLinearScratch,
}

#[derive(Clone, Copy)]
struct AudioPlacement<'a> {
    start: usize,
    embeddings: &'a UltravoxAudioEmbeddings,
}

pub(super) fn generate(
    mapped: &UltravoxLlamaMappedDescriptors,
    backend: BackendKind,
    runtime: &UltravoxLlamaDecoderRuntime,
    prompt: &[u32],
    audio_start: usize,
    audio: &UltravoxAudioEmbeddings,
    options: &UltravoxGenerationOptions,
) -> Result<UltravoxGeneration> {
    let config = mapped.config();
    let compute = Compute::for_backend(backend, ULTRAVOX_LLAMA_HOT_OPS)?;
    let inv_freqs = llama3_inv_freqs(config)?;
    let reserve = prompt
        .len()
        .min(512)
        .saturating_add(options.max_new_tokens.min(128))
        .max(1);
    let mut kv_cache = KvCache::with_reserve(config.n_layer as usize, kv_dim(config), reserve);
    let mut scratch = StepScratch::default();
    let placement = AudioPlacement {
        start: audio_start,
        embeddings: audio,
    };

    for row_start in (0..prompt.len()).step_by(PREFILL_CHUNK_ROWS) {
        let rows = PREFILL_CHUNK_ROWS.min(prompt.len() - row_start);
        forward_chunk(
            &compute,
            mapped,
            runtime,
            &mut scratch,
            &mut kv_cache,
            &prompt[row_start..row_start + rows],
            placement,
            &inv_freqs,
        )?;
    }

    let mut token_ids = Vec::with_capacity(options.max_new_tokens);
    let mut logits = Vec::new();
    let mut stop_token = None;
    for step in 0..options.max_new_tokens {
        fill_logits(&compute, mapped, runtime, &scratch, &mut logits)?;
        let next = argmax(&logits)?;
        token_ids.push(next);
        if options.stop_token_ids.contains(&next) {
            stop_token = Some(next);
            break;
        }
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
            placement,
            &inv_freqs,
        )?;
    }
    Ok(UltravoxGeneration {
        token_ids,
        stop_token,
    })
}

pub(super) fn next_token_logits(
    mapped: &UltravoxLlamaMappedDescriptors,
    backend: BackendKind,
    runtime: &UltravoxLlamaDecoderRuntime,
    prompt: &[u32],
    audio_start: usize,
    audio: &UltravoxAudioEmbeddings,
) -> Result<Vec<f32>> {
    let config = mapped.config();
    let compute = Compute::for_backend(backend, ULTRAVOX_LLAMA_HOT_OPS)?;
    let inv_freqs = llama3_inv_freqs(config)?;
    let reserve = prompt.len().clamp(1, 512);
    let mut kv_cache = KvCache::with_reserve(config.n_layer as usize, kv_dim(config), reserve);
    let mut scratch = StepScratch::default();
    let placement = AudioPlacement {
        start: audio_start,
        embeddings: audio,
    };
    for row_start in (0..prompt.len()).step_by(PREFILL_CHUNK_ROWS) {
        let rows = PREFILL_CHUNK_ROWS.min(prompt.len() - row_start);
        forward_chunk(
            &compute,
            mapped,
            runtime,
            &mut scratch,
            &mut kv_cache,
            &prompt[row_start..row_start + rows],
            placement,
            &inv_freqs,
        )?;
    }
    let mut logits = Vec::new();
    fill_logits(&compute, mapped, runtime, &scratch, &mut logits)?;
    Ok(logits)
}

#[allow(clippy::too_many_arguments)]
fn forward_chunk(
    compute: &Compute,
    mapped: &UltravoxLlamaMappedDescriptors,
    runtime: &UltravoxLlamaDecoderRuntime,
    scratch: &mut StepScratch,
    kv_cache: &mut KvCache,
    tokens: &[u32],
    audio: AudioPlacement<'_>,
    inv_freqs: &[f32],
) -> Result<()> {
    let config = mapped.config();
    let rows = tokens.len();
    if rows == 0 {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: decoder chunk is empty"
        )));
    }
    let position_offset = kv_cache.positions();
    if position_offset + rows > config.max_positions as usize {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: decode position {} exceeds max positions {}",
            position_offset + rows,
            config.max_positions
        )));
    }

    embed_tokens(mapped, tokens, position_offset, audio, scratch)?;
    let hidden = config.hidden_size as usize;
    let q_width = q_dim(config);
    let kv_width = kv_dim(config);
    let ffn = config.ffn_dim as usize;
    resize_zero(&mut scratch.norm, rows * hidden);
    resize_zero(&mut scratch.q, rows * q_width);
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
        let descriptors = mapped.layer(layer);
        if compute.supports_mixed_bf16() {
            // Norm vectors are small and remain exact f32 widened values. Dense
            // linear weights stay mapped as BF16 bits on CPU and Metal. CPU
            // copies bounded <=8-column panels; Metal reuses one complete raw
            // panel per linear to avoid synchronous per-tile submissions.
            widen_tensor(mapped, descriptors.input_norm, &mut block.input_norm)?;
            widen_tensor(mapped, descriptors.ffn_norm, &mut block.ffn_norm)?;
        } else {
            // GPU routes retain the established full-f32 layer scratch path.
            materialize_layer(mapped, layer, &mut block)?;
        }
        compute.rms_norm_f32(
            &scratch.hidden,
            &mut scratch.norm,
            rows,
            hidden,
            &block.input_norm,
            config.rms_norm_eps,
        )?;
        linear_dispatch(
            compute,
            mapped,
            descriptors.q_weight,
            rows,
            q_width,
            hidden,
            &scratch.norm,
            &block.q_w_t,
            &mut scratch.q,
            &mut scratch.mixed,
        )?;
        linear_dispatch(
            compute,
            mapped,
            descriptors.k_weight,
            rows,
            kv_width,
            hidden,
            &scratch.norm,
            &block.k_w_t,
            &mut scratch.k,
            &mut scratch.mixed,
        )?;
        linear_dispatch(
            compute,
            mapped,
            descriptors.v_weight,
            rows,
            kv_width,
            hidden,
            &scratch.norm,
            &block.v_w_t,
            &mut scratch.v,
            &mut scratch.mixed,
        )?;
        apply_half_split_rope(
            &mut scratch.q,
            rows,
            config.n_head as usize,
            config.head_dim as usize,
            inv_freqs,
            position_offset,
        )?;
        apply_half_split_rope(
            &mut scratch.k,
            rows,
            config.n_kv_head as usize,
            config.head_dim as usize,
            inv_freqs,
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
        linear_dispatch(
            compute,
            mapped,
            descriptors.o_weight,
            rows,
            hidden,
            q_width,
            &scratch.attention,
            &block.o_w_t,
            &mut scratch.attention_out,
            &mut scratch.mixed,
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
        linear_dispatch(
            compute,
            mapped,
            descriptors.gate,
            rows,
            ffn,
            hidden,
            &scratch.norm,
            &block.gate_w_t,
            &mut scratch.ffn_gate,
            &mut scratch.mixed,
        )?;
        linear_dispatch(
            compute,
            mapped,
            descriptors.up,
            rows,
            ffn,
            hidden,
            &scratch.norm,
            &block.up_w_t,
            &mut scratch.ffn_up,
            &mut scratch.mixed,
        )?;
        compute.silu_f32(&scratch.ffn_gate, &mut scratch.ffn_activated)?;
        for (activated, up) in scratch.ffn_activated.iter_mut().zip(&scratch.ffn_up) {
            *activated *= up;
        }
        linear_dispatch(
            compute,
            mapped,
            descriptors.down,
            rows,
            hidden,
            ffn,
            &scratch.ffn_activated,
            &block.down_w_t,
            &mut scratch.ffn_down,
            &mut scratch.mixed,
        )?;
        for (value, residual) in scratch.hidden.iter_mut().zip(&scratch.ffn_down) {
            *value += residual;
        }
    }
    kv_cache.advance(rows);

    let mut final_norm = Vec::new();
    widen_tensor(mapped, mapped.final_norm(), &mut final_norm)?;
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
    mapped: &UltravoxLlamaMappedDescriptors,
    tokens: &[u32],
    position_offset: usize,
    audio: AudioPlacement<'_>,
    scratch: &mut StepScratch,
) -> Result<()> {
    let config = mapped.config();
    let hidden = config.hidden_size as usize;
    resize_zero(&mut scratch.hidden, tokens.len() * hidden);
    let audio_end = audio
        .start
        .checked_add(audio.embeddings.frames())
        .ok_or_else(|| {
            VokraError::InvalidArgument(format!("{LABEL}: audio span end overflows usize"))
        })?;
    for (row, &token) in tokens.iter().enumerate() {
        if token >= config.vocab_size {
            return Err(VokraError::InvalidArgument(format!(
                "{LABEL}: token {token} at absolute row {} is outside vocabulary 0..{}",
                position_offset + row,
                config.vocab_size
            )));
        }
        let absolute_row = position_offset + row;
        let target = &mut scratch.hidden[row * hidden..(row + 1) * hidden];
        if (audio.start..audio_end).contains(&absolute_row) {
            let audio_row = absolute_row - audio.start;
            let source = &audio.embeddings.values()[audio_row * hidden..(audio_row + 1) * hidden];
            target.copy_from_slice(source);
        } else {
            widen_row(
                mapped,
                mapped.embedding(),
                token as usize,
                &mut scratch.embedding_row,
            )?;
            target.copy_from_slice(&scratch.embedding_row);
        }
    }
    reject_non_finite("decoder embeddings", &scratch.hidden)
}

#[allow(clippy::too_many_arguments)]
fn attention(
    compute: &Compute,
    scratch: &mut StepScratch,
    key_cache: &[f32],
    value_cache: &[f32],
    rows: usize,
    position_offset: usize,
    config: UltravoxLlamaConfig,
) -> Result<()> {
    let q_width = q_dim(config);
    let kv_width = kv_dim(config);
    let head_dim = config.head_dim as usize;
    let total_rows = position_offset + rows;
    let expected = total_rows * kv_width;
    if key_cache.len() != expected || value_cache.len() != expected {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: KV cache length mismatch: key={}, value={}, expected={expected}",
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
            "{LABEL}: causal mask shape mismatch: scores={}, rows={rows}, total_rows={total_rows}, offset={position_offset}, scale={scale}",
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

fn llama3_inv_freqs(config: UltravoxLlamaConfig) -> Result<Vec<f32>> {
    let head_dim = config.head_dim as usize;
    if head_dim == 0
        || !head_dim.is_multiple_of(2)
        || !config.rope_theta.is_finite()
        || config.rope_theta <= 0.0
        || !config.rope_factor.is_finite()
        || config.rope_factor <= 0.0
        || !config.rope_low_freq_factor.is_finite()
        || config.rope_low_freq_factor <= 0.0
        || !config.rope_high_freq_factor.is_finite()
        || config.rope_high_freq_factor <= config.rope_low_freq_factor
        || config.rope_original_max_positions == 0
    {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: invalid Llama-3 RoPE config {config:?}"
        )));
    }
    let old_context = config.rope_original_max_positions as f32;
    let low_wavelength = old_context / config.rope_low_freq_factor;
    let high_wavelength = old_context / config.rope_high_freq_factor;
    let mut frequencies = Vec::with_capacity(head_dim / 2);
    for pair in 0..head_dim / 2 {
        let frequency = config
            .rope_theta
            .powf(-((2 * pair) as f32) / head_dim as f32);
        let wavelength = std::f32::consts::TAU / frequency;
        let scaled = if wavelength < high_wavelength {
            frequency
        } else if wavelength > low_wavelength {
            frequency / config.rope_factor
        } else {
            let smooth = (old_context / wavelength - config.rope_low_freq_factor)
                / (config.rope_high_freq_factor - config.rope_low_freq_factor);
            (1.0 - smooth) * frequency / config.rope_factor + smooth * frequency
        };
        frequencies.push(scaled);
    }
    Ok(frequencies)
}

fn apply_half_split_rope(
    values: &mut [f32],
    rows: usize,
    heads: usize,
    head_dim: usize,
    inv_freqs: &[f32],
    position_offset: usize,
) -> Result<()> {
    if rows == 0
        || heads == 0
        || !head_dim.is_multiple_of(2)
        || inv_freqs.len() != head_dim / 2
        || values.len() != rows * heads * head_dim
    {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: RoPE shape mismatch: values={}, rows={rows}, heads={heads}, head_dim={head_dim}, inv_freqs={}",
            values.len(),
            inv_freqs.len()
        )));
    }
    let half = head_dim / 2;
    for row in 0..rows {
        let position = (position_offset + row) as f32;
        for head in 0..heads {
            let base = (row * heads + head) * head_dim;
            for pair in 0..half {
                let angle = position * inv_freqs[pair];
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
    mapped: &UltravoxLlamaMappedDescriptors,
    layer: usize,
    block: &mut DecoderBlock,
) -> Result<()> {
    let config = mapped.config();
    let hidden = config.hidden_size as usize;
    let q_width = q_dim(config);
    let kv_width = kv_dim(config);
    let ffn = config.ffn_dim as usize;
    let descriptors = mapped.layer(layer);
    widen_tensor(mapped, descriptors.input_norm, &mut block.input_norm)?;
    transpose_tensor(
        mapped,
        descriptors.q_weight,
        q_width,
        hidden,
        &mut block.q_w_t,
    )?;
    transpose_tensor(
        mapped,
        descriptors.k_weight,
        kv_width,
        hidden,
        &mut block.k_w_t,
    )?;
    transpose_tensor(
        mapped,
        descriptors.v_weight,
        kv_width,
        hidden,
        &mut block.v_w_t,
    )?;
    transpose_tensor(
        mapped,
        descriptors.o_weight,
        hidden,
        q_width,
        &mut block.o_w_t,
    )?;
    widen_tensor(mapped, descriptors.ffn_norm, &mut block.ffn_norm)?;
    transpose_tensor(mapped, descriptors.gate, ffn, hidden, &mut block.gate_w_t)?;
    transpose_tensor(mapped, descriptors.up, ffn, hidden, &mut block.up_w_t)?;
    transpose_tensor(mapped, descriptors.down, hidden, ffn, &mut block.down_w_t)
}

/// Dispatches a dense Llama linear through the established F32 route on
/// CUDA/WebGPU, or through the mapped-BF16 seam on CPU/Metal. `dense_weight`
/// is only read on the dense backends; CPU transposes bounded output-row
/// panels and Metal uploads one raw-BF16 logical panel directly from the
/// validated little-endian payload.
#[allow(clippy::too_many_arguments)]
fn linear_dispatch(
    compute: &Compute,
    mapped: &UltravoxLlamaMappedDescriptors,
    info: &GgufTensorInfo,
    rows: usize,
    output_width: usize,
    input_width: usize,
    input: &[f32],
    dense_weight: &[f32],
    output: &mut [f32],
    mixed_scratch: &mut MixedLinearScratch,
) -> Result<()> {
    if compute.supports_mixed_bf16() {
        mixed_linear(
            compute,
            mapped,
            info,
            output_width,
            input_width,
            0,
            output_width,
            rows,
            input,
            output,
            mixed_scratch,
        )
    } else {
        compute.gemm_f32(
            rows,
            output_width,
            input_width,
            input,
            dense_weight,
            None,
            output,
        )
    }
}

/// Computes a contiguous output-column chunk from a GGUF `[output, input]`
/// BF16 matrix. CPU transposes only `BF16_CPU_PANEL_COLUMNS` output rows into the
/// mixed GEMM's `[input, columns]` layout at once. Metal transposes the full
/// requested logical matrix as raw `u16` and issues one device submission;
/// this keeps the large panel bounded in raw storage without creating an F32
/// weight mirror or thousands of synchronous GPU round trips.
#[allow(clippy::too_many_arguments)]
fn mixed_linear(
    compute: &Compute,
    mapped: &UltravoxLlamaMappedDescriptors,
    info: &GgufTensorInfo,
    source_rows: usize,
    source_columns: usize,
    first_output: usize,
    output_width: usize,
    rows: usize,
    input: &[f32],
    output: &mut [f32],
    mixed_scratch: &mut MixedLinearScratch,
) -> Result<()> {
    let expected_input = rows.checked_mul(source_columns).ok_or_else(|| {
        VokraError::InvalidArgument(format!("{LABEL}: mixed linear input shape overflow"))
    })?;
    let expected_output = rows.checked_mul(output_width).ok_or_else(|| {
        VokraError::InvalidArgument(format!("{LABEL}: mixed linear output shape overflow"))
    })?;
    if input.len() != expected_input || output.len() != expected_output {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: mixed linear shape mismatch: input={} (want {}), output={} (want {})",
            input.len(),
            expected_input,
            output.len(),
            expected_output
        )));
    }
    let end_output = first_output.checked_add(output_width).ok_or_else(|| {
        VokraError::InvalidArgument(format!("{LABEL}: mixed linear output range overflow"))
    })?;
    if end_output > source_rows {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: mixed linear output range {first_output}..{end_output} exceeds {source_rows}"
        )));
    }
    if info.dtype != GgmlType::BF16 {
        return Err(VokraError::ModelLoad(format!(
            "{LABEL}: mixed linear tensor `{}` has {:?}, expected BF16",
            info.name, info.dtype
        )));
    }

    // CPU SIMD retains the established <=8-column bounded tile. Metal keeps
    // one complete logical linear as a raw-BF16 panel for one submission;
    // widening a complete F32 matrix would defeat the mapped-weight contract.
    let panel_width = mixed_panel_width(compute.is_cpu(), output_width);
    if !compute.is_cpu() {
        transpose_bf16_rows(
            mapped,
            info,
            source_rows,
            source_columns,
            first_output,
            panel_width,
            &mut mixed_scratch.panel,
        )?;
        return compute.gemm_f32_bf16_bits(
            rows,
            panel_width,
            source_columns,
            input,
            &mixed_scratch.panel,
            output,
        );
    }

    let mut column = 0;
    while column < output_width {
        let columns = mixed_panel_width(true, output_width - column);
        let panel_first = first_output.checked_add(column).ok_or_else(|| {
            VokraError::InvalidArgument(format!("{LABEL}: mixed linear panel offset overflow"))
        })?;
        transpose_bf16_rows(
            mapped,
            info,
            source_rows,
            source_columns,
            panel_first,
            columns,
            &mut mixed_scratch.panel,
        )?;
        let output_panel = output.get_mut(column..).ok_or_else(|| {
            VokraError::InvalidArgument(format!("{LABEL}: mixed linear output offset overflow"))
        })?;
        compute.gemm_f32_bf16_bits_strided_with_scratch(
            rows,
            columns,
            source_columns,
            input,
            &mixed_scratch.panel,
            output_panel,
            output_width,
            &mut mixed_scratch.kernel,
        )?;
        column += columns;
    }
    Ok(())
}

/// Selects the bounded panel width for the mapped-BF16 backend policy. CPU
/// keeps the reusable eight-column SIMD tile; Metal consumes the requested
/// logical range in one raw-BF16 panel. This function is deliberately pure so
/// the policy stays pinned even when Metal tests are unavailable on CI hosts.
fn mixed_panel_width(is_cpu: bool, requested: usize) -> usize {
    if is_cpu {
        BF16_CPU_PANEL_COLUMNS.min(requested)
    } else {
        requested
    }
}

/// Transposes a bounded range of rows from a validated GGUF `[rows, columns]`
/// BF16 payload into `[columns, selected_rows]` numeric `u16` bit patterns.
/// The reader has already authenticated the tensor range; this helper repeats
/// dtype, shape and byte-length checks before touching the destination.
fn transpose_bf16_rows(
    mapped: &UltravoxLlamaMappedDescriptors,
    info: &GgufTensorInfo,
    source_rows: usize,
    source_columns: usize,
    first_row: usize,
    rows: usize,
    output: &mut Vec<u16>,
) -> Result<()> {
    if info.dtype != GgmlType::BF16 {
        return Err(VokraError::ModelLoad(format!(
            "{LABEL}: tensor `{}` has {:?}, expected BF16",
            info.name, info.dtype
        )));
    }
    let expected_dimensions = [
        u64::try_from(source_rows).map_err(|_| {
            VokraError::InvalidArgument(format!("{LABEL}: BF16 row dimension exceeds u64"))
        })?,
        u64::try_from(source_columns).map_err(|_| {
            VokraError::InvalidArgument(format!("{LABEL}: BF16 column dimension exceeds u64"))
        })?,
    ];
    if info.dimensions.as_slice() != expected_dimensions {
        return Err(VokraError::ModelLoad(format!(
            "{LABEL}: tensor `{}` shape {:?}, expected {:?}",
            info.name, info.dimensions, expected_dimensions
        )));
    }
    let bytes = mapped.file().tensor_bytes(info);
    transpose_bf16_payload(bytes, source_rows, source_columns, first_row, rows, output)
}

fn transpose_bf16_payload(
    bytes: &[u8],
    source_rows: usize,
    source_columns: usize,
    first_row: usize,
    rows: usize,
    output: &mut Vec<u16>,
) -> Result<()> {
    let elements = source_rows.checked_mul(source_columns).ok_or_else(|| {
        VokraError::InvalidArgument(format!("{LABEL}: BF16 tensor shape overflow"))
    })?;
    let expected_bytes = elements.checked_mul(2).ok_or_else(|| {
        VokraError::InvalidArgument(format!("{LABEL}: BF16 tensor byte length overflow"))
    })?;
    if bytes.len() != expected_bytes {
        return Err(VokraError::ModelLoad(format!(
            "{LABEL}: BF16 payload has {} bytes, expected {expected_bytes}",
            bytes.len()
        )));
    }
    let end_row = first_row.checked_add(rows).ok_or_else(|| {
        VokraError::InvalidArgument(format!("{LABEL}: BF16 transpose row range overflow"))
    })?;
    if end_row > source_rows {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: BF16 transpose rows {first_row}..{end_row} exceed {source_rows}"
        )));
    }
    let panel_len = source_columns.checked_mul(rows).ok_or_else(|| {
        VokraError::InvalidArgument(format!("{LABEL}: BF16 transpose panel overflow"))
    })?;
    output.clear();
    output.resize(panel_len, 0);
    for column in 0..source_columns {
        for row in 0..rows {
            let source_element = (first_row + row)
                .checked_mul(source_columns)
                .and_then(|value| value.checked_add(column))
                .ok_or_else(|| {
                    VokraError::InvalidArgument(format!(
                        "{LABEL}: BF16 transpose source index overflow"
                    ))
                })?;
            let source_byte = source_element.checked_mul(2).ok_or_else(|| {
                VokraError::InvalidArgument(format!(
                    "{LABEL}: BF16 transpose source byte index overflow"
                ))
            })?;
            output[column * rows + row] =
                u16::from_le_bytes([bytes[source_byte], bytes[source_byte + 1]]);
        }
    }
    Ok(())
}

fn fill_logits(
    compute: &Compute,
    mapped: &UltravoxLlamaMappedDescriptors,
    runtime: &UltravoxLlamaDecoderRuntime,
    scratch: &StepScratch,
    output: &mut Vec<f32>,
) -> Result<()> {
    let config = mapped.config();
    let hidden_width = config.hidden_size as usize;
    if scratch.norm.len() < hidden_width {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: no final decoder hidden row is available"
        )));
    }
    let hidden = &scratch.norm[scratch.norm.len() - hidden_width..];
    let mut head = lock_scratch(&runtime.head, mapped.mapped_model())?;
    output.clear();
    output.reserve(config.vocab_size as usize);
    let vocab = config.vocab_size as usize;
    let mut first_row = 0;
    while first_row < vocab {
        let rows = HEAD_CHUNK_ROWS.min(vocab - first_row);
        head.logits.clear();
        head.logits.resize(rows, 0.0);
        if compute.supports_mixed_bf16() {
            let HeadScratch { mixed, logits, .. } = &mut *head;
            mixed_linear(
                compute,
                mapped,
                mapped.embedding(),
                vocab,
                hidden_width,
                first_row,
                rows,
                1,
                hidden,
                logits,
                mixed,
            )?;
            reject_non_finite("vocabulary logits", logits)?;
            output.extend_from_slice(logits);
        } else {
            widen_rows(
                mapped,
                mapped.embedding(),
                first_row,
                rows,
                &mut head.weights,
            )?;
            let HeadScratch {
                weights, logits, ..
            } = &mut *head;
            compute.gemv_f32(rows, hidden_width, weights, hidden, None, logits)?;
            reject_non_finite("vocabulary logits", logits)?;
            output.extend_from_slice(logits);
        }
        first_row += rows;
    }
    debug_assert_eq!(output.len(), vocab);
    Ok(())
}

fn argmax(logits: &[f32]) -> Result<u32> {
    let Some((&first, rest)) = logits.split_first() else {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: cannot select from empty logits"
        )));
    };
    if !first.is_finite() {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: vocabulary logits contain non-finite value {first} at index 0"
        )));
    }
    let mut best_index = 0usize;
    let mut best_value = first;
    for (offset, &value) in rest.iter().enumerate() {
        if !value.is_finite() {
            return Err(VokraError::InvalidArgument(format!(
                "{LABEL}: vocabulary logits contain non-finite value {value} at index {}",
                offset + 1
            )));
        }
        if value > best_value {
            best_value = value;
            best_index = offset + 1;
        }
    }
    u32::try_from(best_index)
        .map_err(|_| VokraError::InvalidArgument(format!("{LABEL}: vocabulary index exceeds u32")))
}

fn q_dim(config: UltravoxLlamaConfig) -> usize {
    config.n_head as usize * config.head_dim as usize
}

fn kv_dim(config: UltravoxLlamaConfig) -> usize {
    config.n_kv_head as usize * config.head_dim as usize
}

fn resize_zero(values: &mut Vec<f32>, len: usize) {
    values.clear();
    values.resize(len, 0.0);
}

fn widen_row(
    mapped: &UltravoxLlamaMappedDescriptors,
    info: &GgufTensorInfo,
    row: usize,
    output: &mut Vec<f32>,
) -> Result<()> {
    widen_rows(mapped, info, row, 1, output)
}

fn widen_rows(
    mapped: &UltravoxLlamaMappedDescriptors,
    info: &GgufTensorInfo,
    first_row: usize,
    rows: usize,
    output: &mut Vec<f32>,
) -> Result<()> {
    let hidden = mapped.config().hidden_size as usize;
    let element_size = info.dtype.type_size();
    let bytes = mapped.file().tensor_bytes(info);
    let start = first_row
        .checked_mul(hidden)
        .and_then(|value| value.checked_mul(element_size))
        .ok_or_else(|| {
            VokraError::InvalidArgument(format!("{LABEL}: embedding row offset overflow"))
        })?;
    let len = rows
        .checked_mul(hidden)
        .and_then(|value| value.checked_mul(element_size))
        .ok_or_else(|| {
            VokraError::InvalidArgument(format!("{LABEL}: embedding row length overflow"))
        })?;
    let end = start.checked_add(len).ok_or_else(|| {
        VokraError::InvalidArgument(format!("{LABEL}: embedding row range overflow"))
    })?;
    let source = bytes.get(start..end).ok_or_else(|| {
        VokraError::ModelLoad(format!(
            "{LABEL}: tensor `{}` row range {first_row}..{} exceeds {} bytes",
            info.name,
            first_row + rows,
            bytes.len()
        ))
    })?;
    widen_into(source, info.dtype, output, mapped.mapped_model())
}

fn widen_tensor(
    mapped: &UltravoxLlamaMappedDescriptors,
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
    mapped: &UltravoxLlamaMappedDescriptors,
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
            "{LABEL}: {value_label} contains non-finite value {value} at index {index}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_bf16_transpose_preserves_payload_and_orientation() {
        let source: Vec<u16> = (0..15).map(|value| 0x3f80 + value).collect();
        let bytes: Vec<u8> = source.iter().flat_map(|bits| bits.to_le_bytes()).collect();
        let before = bytes.clone();
        let mut scratch = MixedLinearScratch::default();
        scratch.panel.push(0xdead_u16);
        transpose_bf16_payload(&bytes, 3, 5, 1, 2, &mut scratch.panel).expect("transpose");
        let panel_capacity = scratch.panel.capacity();
        let panel_ptr = scratch.panel.as_ptr();
        assert_eq!(
            scratch.panel,
            vec![
                source[5], source[10], source[6], source[11], source[7], source[12], source[8],
                source[13], source[9], source[14],
            ]
        );
        transpose_bf16_payload(&bytes, 3, 5, 0, 1, &mut scratch.panel).expect("tail transpose");
        assert_eq!(scratch.panel.capacity(), panel_capacity);
        assert_eq!(scratch.panel.as_ptr(), panel_ptr);
        assert_eq!(bytes, before);
    }

    #[test]
    fn bounded_bf16_transpose_rejects_bad_length_and_overflow() {
        let mut output = Vec::new();
        assert!(matches!(
            transpose_bf16_payload(&[0; 4], 2, 2, 0, 1, &mut output),
            Err(VokraError::ModelLoad(_))
        ));
        assert!(matches!(
            transpose_bf16_payload(&[], usize::MAX, 2, 0, 0, &mut output),
            Err(VokraError::InvalidArgument(_))
        ));
        let bytes = vec![0; 8];
        assert!(matches!(
            transpose_bf16_payload(&bytes, 2, 2, usize::MAX, 1, &mut output),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn bounded_bf16_transpose_rejects_out_of_range_rows() {
        let bytes = vec![0; 8];
        let mut output = Vec::new();
        assert!(matches!(
            transpose_bf16_payload(&bytes, 2, 2, 2, 1, &mut output),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn mixed_bf16_panel_policy_is_backend_specific() {
        assert_eq!(mixed_panel_width(true, usize::MAX), BF16_CPU_PANEL_COLUMNS);
        assert_eq!(mixed_panel_width(true, 3), 3);
        assert_eq!(mixed_panel_width(false, 8192), 8192);
    }

    #[test]
    fn llama3_scaling_keeps_high_band_and_divides_low_band() {
        let config = UltravoxLlamaConfig::OFFICIAL;
        let frequencies = llama3_inv_freqs(config).expect("frequencies");
        let base_first = 1.0;
        assert_eq!(frequencies[0], base_first);
        let pair = frequencies.len() - 1;
        let base_last = config
            .rope_theta
            .powf(-((2 * pair) as f32) / config.head_dim as f32);
        assert!((frequencies[pair] - base_last / 32.0).abs() <= f32::EPSILON * base_last);
    }

    #[test]
    fn half_split_rope_position_zero_is_identity() {
        let config = UltravoxLlamaConfig::OFFICIAL;
        let frequencies = llama3_inv_freqs(config).expect("frequencies");
        let mut values = (0..config.head_dim)
            .map(|value| value as f32 / config.head_dim as f32)
            .collect::<Vec<_>>();
        let expected = values.clone();
        apply_half_split_rope(&mut values, 1, 1, config.head_dim as usize, &frequencies, 0)
            .expect("rope");
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
    fn argmax_keeps_first_index_on_ties() {
        assert_eq!(argmax(&[1.0, 3.0, 3.0]).expect("argmax"), 1);
        assert!(argmax(&[]).is_err());
        assert!(argmax(&[f32::NAN]).is_err());
    }
}
