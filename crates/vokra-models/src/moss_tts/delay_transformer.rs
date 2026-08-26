//! Bounded-memory native Qwen3 forward for MOSS-TTS Base/v1.5 Delay.
//!
//! The model stays in its BF16/F16/F32 GGUF mapping. Each transformer layer
//! is widened and transposed into one reused scratch block, while embeddings
//! and LM heads are read one row/chunk at a time. Learned reductions and
//! projections all use one selected [`Compute`] backend; only embedding row
//! gather, RoPE, causal masking, layout changes, residual addition and
//! element-wise product remain deterministic host glue.

mod generation;

use std::path::Path;
use std::sync::{Arc, Mutex};

use vokra_core::backend::BackendKind;
use vokra_core::gguf::GgufTensorInfo;
use vokra_core::{KvCache, Result, VokraError};

use crate::compute::{Compute, HotOp};
use crate::mapped_weights::{lock_scratch, transpose_widen, widen_into};

use super::delay::{
    AUDIO_VOCAB_WITH_PAD, DelayMappedDescriptors, FFN_DIM, HEAD_DIM, HIDDEN_DIM, KV_DIM, MAPPED,
    MAX_POSITION_EMBEDDINGS, MossTtsDelayCheckpoint, NUM_AUDIO_CODEBOOKS, NUM_KV_HEADS, NUM_LAYERS,
    NUM_Q_HEADS, Q_DIM, RMS_NORM_EPS, ROPE_BASE, TEXT_VOCAB_SIZE,
};

pub use self::generation::{MossTtsDelayGeneration, MossTtsDelayGenerationOptions};

const LABEL: &str = "moss_tts/delay";
const INPUT_COLUMNS: usize = 1 + NUM_AUDIO_CODEBOOKS;
const PREFILL_CHUNK_ROWS: usize = 8;
const HEAD_CHUNK_ROWS: usize = 512;

/// Every learned operator required by the Base/v1.5 Delay Qwen3 forward.
/// A selected backend must cover the complete set before inference starts.
pub const MOSS_TTS_DELAY_HOT_OPS: &[HotOp] = &[
    HotOp::Gemm,
    HotOp::Gemv,
    HotOp::Softmax,
    HotOp::RmsNorm,
    HotOp::Silu,
];

/// Last-position output of all 33 official Delay heads.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct MossTtsDelayLogits {
    /// Text head (`lm_heads.0`) logits, length 155,648.
    pub text_logits: Vec<f32>,
    /// Flat codebook-major audio logits: 32 rows × 1,025 values. Index 1,024
    /// is the official audio-pad sentinel and remains present for the caller's
    /// generation mask.
    pub audio_logits: Vec<f32>,
}

impl MossTtsDelayLogits {
    /// Returns one audio-codebook head (`0..32`).
    pub fn audio_codebook(&self, codebook: usize) -> Result<&[f32]> {
        if codebook >= NUM_AUDIO_CODEBOOKS {
            return Err(VokraError::InvalidArgument(format!(
                "{LABEL}: audio codebook {codebook} is outside 0..{NUM_AUDIO_CODEBOOKS}"
            )));
        }
        let start = codebook * AUDIO_VOCAB_WITH_PAD;
        Ok(&self.audio_logits[start..start + AUDIO_VOCAB_WITH_PAD])
    }
}

/// Native Base/v1.5 Delay Qwen3 model with an explicit CPU or Metal backend.
///
/// This type exposes the independently testable raw-logits boundary. The
/// delayed-codebook sampling state machine and Full audio-tokenizer companion
/// are layered on top; neither is silently substituted here.
#[derive(Clone)]
pub struct MossTtsDelay {
    checkpoint: MossTtsDelayCheckpoint,
    backend: BackendKind,
    runtime: Arc<DelayRuntimeScratch>,
}

impl std::fmt::Debug for MossTtsDelay {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MossTtsDelay")
            .field("release", &self.checkpoint.release())
            .field("backend", &self.backend)
            .finish_non_exhaustive()
    }
}

impl MossTtsDelay {
    /// Opens and strictly binds a true-mmap checkpoint, then preflights the
    /// complete learned-op set on `backend`.
    pub fn open_mapped(path: impl AsRef<Path>, backend: BackendKind) -> Result<Self> {
        Self::from_checkpoint(MossTtsDelayCheckpoint::open_mapped(path)?, backend)
    }

    /// Builds the executable model from a previously authenticated checkpoint.
    pub fn from_checkpoint(
        checkpoint: MossTtsDelayCheckpoint,
        backend: BackendKind,
    ) -> Result<Self> {
        let _ = Compute::for_backend(backend, MOSS_TTS_DELAY_HOT_OPS)?;
        Ok(Self {
            checkpoint,
            backend,
            runtime: Arc::new(DelayRuntimeScratch::default()),
        })
    }

    /// Selected backend for the complete learned graph.
    pub const fn backend(&self) -> BackendKind {
        self.backend
    }

    /// Strict release checkpoint retained by this model.
    pub const fn checkpoint(&self) -> &MossTtsDelayCheckpoint {
        &self.checkpoint
    }

    /// Runs an explicit upstream-compatible `[rows, 33]` prompt matrix and
    /// returns all last-position text/audio logits.
    ///
    /// Column zero is a text token (`0..155648`); columns 1..32 are audio
    /// codes (`0..1025`, including pad 1024). Raw text is intentionally not
    /// accepted because the GGUF does not embed the official tokenizer or
    /// chat-template assets.
    pub fn forward_prompt_last_logits(&self, prompt_rows: &[u32]) -> Result<MossTtsDelayLogits> {
        if prompt_rows.is_empty() || !prompt_rows.len().is_multiple_of(INPUT_COLUMNS) {
            return Err(VokraError::InvalidArgument(format!(
                "{LABEL}: prompt must be a non-empty [rows,{INPUT_COLUMNS}] u32 matrix, got {} values",
                prompt_rows.len()
            )));
        }
        let rows = prompt_rows.len() / INPUT_COLUMNS;
        if rows > MAX_POSITION_EMBEDDINGS {
            return Err(VokraError::InvalidArgument(format!(
                "{LABEL}: prompt rows {rows} exceed max positions {MAX_POSITION_EMBEDDINGS}"
            )));
        }

        let compute = Compute::for_backend(self.backend, MOSS_TTS_DELAY_HOT_OPS)?;
        let mapped = self.checkpoint.mapped();
        let reserve = rows.min(256).max(1);
        let mut kv_cache = KvCache::with_reserve(NUM_LAYERS, KV_DIM, reserve);
        let mut scratch = DelayStepScratch::default();
        for row_start in (0..rows).step_by(PREFILL_CHUNK_ROWS) {
            let chunk_rows = PREFILL_CHUNK_ROWS.min(rows - row_start);
            let start = row_start * INPUT_COLUMNS;
            let end = start + chunk_rows * INPUT_COLUMNS;
            forward_chunk(
                &compute,
                mapped,
                &self.runtime,
                &mut scratch,
                &mut kv_cache,
                &prompt_rows[start..end],
                chunk_rows,
            )?;
        }
        last_logits(&compute, mapped, &self.runtime, &scratch)
    }
}

#[derive(Default)]
struct DelayRuntimeScratch {
    block: Mutex<DelayBlock>,
    head_chunk: Mutex<Vec<f32>>,
}

#[derive(Default)]
struct DelayBlock {
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
struct DelayStepScratch {
    hidden: Vec<f32>,
    norm: Vec<f32>,
    embed_row: Vec<f32>,
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

fn resize_zero(values: &mut Vec<f32>, len: usize) {
    values.clear();
    values.resize(len, 0.0);
}

fn forward_chunk(
    compute: &Compute,
    mapped: &DelayMappedDescriptors,
    runtime: &DelayRuntimeScratch,
    scratch: &mut DelayStepScratch,
    kv_cache: &mut KvCache,
    prompt: &[u32],
    rows: usize,
) -> Result<()> {
    if rows == 0 || prompt.len() != rows * INPUT_COLUMNS {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: forward chunk shape mismatch: prompt={}, rows={rows}, columns={INPUT_COLUMNS}",
            prompt.len()
        )));
    }
    let position_offset = kv_cache.positions();
    if position_offset + rows > MAX_POSITION_EMBEDDINGS {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: decode position {} exceeds max positions {MAX_POSITION_EMBEDDINGS}",
            position_offset + rows
        )));
    }

    embed_prompt(mapped, prompt, rows, scratch)?;
    resize_zero(&mut scratch.norm, rows * HIDDEN_DIM);
    resize_zero(&mut scratch.q_raw, rows * Q_DIM);
    resize_zero(&mut scratch.q, rows * Q_DIM);
    resize_zero(&mut scratch.k_raw, rows * KV_DIM);
    resize_zero(&mut scratch.k, rows * KV_DIM);
    resize_zero(&mut scratch.v, rows * KV_DIM);
    resize_zero(&mut scratch.query, rows * HEAD_DIM);
    resize_zero(&mut scratch.attention, rows * Q_DIM);
    resize_zero(&mut scratch.attention_out, rows * HIDDEN_DIM);
    resize_zero(&mut scratch.ffn_gate, rows * FFN_DIM);
    resize_zero(&mut scratch.ffn_activated, rows * FFN_DIM);
    resize_zero(&mut scratch.ffn_up, rows * FFN_DIM);
    resize_zero(&mut scratch.ffn_down, rows * HIDDEN_DIM);

    let mut block = lock_scratch(&runtime.block, MAPPED)?;
    for layer in 0..NUM_LAYERS {
        materialize_layer(mapped, layer, &mut block)?;

        compute.rms_norm_f32(
            &scratch.hidden,
            &mut scratch.norm,
            rows,
            HIDDEN_DIM,
            &block.input_norm,
            RMS_NORM_EPS,
        )?;
        compute.gemm_f32(
            rows,
            Q_DIM,
            HIDDEN_DIM,
            &scratch.norm,
            &block.q_w_t,
            None,
            &mut scratch.q_raw,
        )?;
        compute.gemm_f32(
            rows,
            KV_DIM,
            HIDDEN_DIM,
            &scratch.norm,
            &block.k_w_t,
            None,
            &mut scratch.k_raw,
        )?;
        compute.gemm_f32(
            rows,
            KV_DIM,
            HIDDEN_DIM,
            &scratch.norm,
            &block.v_w_t,
            None,
            &mut scratch.v,
        )?;
        compute.rms_norm_f32(
            &scratch.q_raw,
            &mut scratch.q,
            rows * NUM_Q_HEADS,
            HEAD_DIM,
            &block.q_norm,
            RMS_NORM_EPS,
        )?;
        compute.rms_norm_f32(
            &scratch.k_raw,
            &mut scratch.k,
            rows * NUM_KV_HEADS,
            HEAD_DIM,
            &block.k_norm,
            RMS_NORM_EPS,
        )?;
        apply_half_split_rope(&mut scratch.q, rows, NUM_Q_HEADS, position_offset)?;
        apply_half_split_rope(&mut scratch.k, rows, NUM_KV_HEADS, position_offset)?;

        kv_cache.append(layer, &scratch.k, &scratch.v);
        attention(
            compute,
            scratch,
            kv_cache.k(layer),
            kv_cache.v(layer),
            rows,
            position_offset,
        )?;
        compute.gemm_f32(
            rows,
            HIDDEN_DIM,
            Q_DIM,
            &scratch.attention,
            &block.o_w_t,
            None,
            &mut scratch.attention_out,
        )?;
        for (hidden, &residual) in scratch.hidden.iter_mut().zip(&scratch.attention_out) {
            *hidden += residual;
        }

        compute.rms_norm_f32(
            &scratch.hidden,
            &mut scratch.norm,
            rows,
            HIDDEN_DIM,
            &block.ffn_norm,
            RMS_NORM_EPS,
        )?;
        compute.gemm_f32(
            rows,
            FFN_DIM,
            HIDDEN_DIM,
            &scratch.norm,
            &block.gate_w_t,
            None,
            &mut scratch.ffn_gate,
        )?;
        compute.gemm_f32(
            rows,
            FFN_DIM,
            HIDDEN_DIM,
            &scratch.norm,
            &block.up_w_t,
            None,
            &mut scratch.ffn_up,
        )?;
        compute.silu_f32(&scratch.ffn_gate, &mut scratch.ffn_activated)?;
        for (activated, &up) in scratch.ffn_activated.iter_mut().zip(&scratch.ffn_up) {
            *activated *= up;
        }
        compute.gemm_f32(
            rows,
            HIDDEN_DIM,
            FFN_DIM,
            &scratch.ffn_activated,
            &block.down_w_t,
            None,
            &mut scratch.ffn_down,
        )?;
        for (hidden, &residual) in scratch.hidden.iter_mut().zip(&scratch.ffn_down) {
            *hidden += residual;
        }
    }
    kv_cache.advance(rows);

    let mut final_norm = Vec::new();
    widen_tensor(mapped, mapped.final_norm(), &mut final_norm)?;
    compute.rms_norm_f32(
        &scratch.hidden,
        &mut scratch.norm,
        rows,
        HIDDEN_DIM,
        &final_norm,
        RMS_NORM_EPS,
    )?;
    reject_non_finite("final hidden", &scratch.norm)
}

fn embed_prompt(
    mapped: &DelayMappedDescriptors,
    prompt: &[u32],
    rows: usize,
    scratch: &mut DelayStepScratch,
) -> Result<()> {
    resize_zero(&mut scratch.hidden, rows * HIDDEN_DIM);
    for row in 0..rows {
        let row_tokens = &prompt[row * INPUT_COLUMNS..(row + 1) * INPUT_COLUMNS];
        let text = row_tokens[0] as usize;
        if text >= TEXT_VOCAB_SIZE {
            return Err(VokraError::InvalidArgument(format!(
                "{LABEL}: text token at row {row} is {text}, outside 0..{TEXT_VOCAB_SIZE}"
            )));
        }
        widen_row(
            mapped,
            mapped.text_embedding(),
            text,
            &mut scratch.embed_row,
        )?;
        let hidden = &mut scratch.hidden[row * HIDDEN_DIM..(row + 1) * HIDDEN_DIM];
        hidden.copy_from_slice(&scratch.embed_row);
        for codebook in 0..NUM_AUDIO_CODEBOOKS {
            let token = row_tokens[1 + codebook] as usize;
            if token >= AUDIO_VOCAB_WITH_PAD {
                return Err(VokraError::InvalidArgument(format!(
                    "{LABEL}: audio token at row {row}, codebook {codebook} is {token}, outside 0..{AUDIO_VOCAB_WITH_PAD}"
                )));
            }
            widen_row(
                mapped,
                mapped.audio_embedding(codebook),
                token,
                &mut scratch.embed_row,
            )?;
            for (value, &embedding) in hidden.iter_mut().zip(&scratch.embed_row) {
                *value += embedding;
            }
        }
    }
    reject_non_finite("prompt embedding", &scratch.hidden)
}

fn attention(
    compute: &Compute,
    scratch: &mut DelayStepScratch,
    key_cache: &[f32],
    value_cache: &[f32],
    rows: usize,
    position_offset: usize,
) -> Result<()> {
    let total_rows = position_offset + rows;
    let expected = total_rows * KV_DIM;
    if key_cache.len() != expected || value_cache.len() != expected {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: KV cache length mismatch: key={}, value={}, expected={expected}",
            key_cache.len(),
            value_cache.len()
        )));
    }
    resize_zero(&mut scratch.key_t, HEAD_DIM * total_rows);
    resize_zero(&mut scratch.value, total_rows * HEAD_DIM);
    resize_zero(&mut scratch.scores, rows * total_rows);
    resize_zero(&mut scratch.probabilities, rows * total_rows);
    resize_zero(&mut scratch.attended, rows * HEAD_DIM);
    scratch.attention.fill(0.0);
    let groups = NUM_Q_HEADS / NUM_KV_HEADS;
    let scale = (HEAD_DIM as f32).sqrt().recip();

    for kv_head in 0..NUM_KV_HEADS {
        for position in 0..total_rows {
            let source = position * KV_DIM + kv_head * HEAD_DIM;
            for dimension in 0..HEAD_DIM {
                scratch.key_t[dimension * total_rows + position] = key_cache[source + dimension];
                scratch.value[position * HEAD_DIM + dimension] = value_cache[source + dimension];
            }
        }
        for group in 0..groups {
            let q_head = kv_head * groups + group;
            for row in 0..rows {
                let source = row * Q_DIM + q_head * HEAD_DIM;
                scratch.query[row * HEAD_DIM..(row + 1) * HEAD_DIM]
                    .copy_from_slice(&scratch.q[source..source + HEAD_DIM]);
            }
            compute.gemm_f32(
                rows,
                total_rows,
                HEAD_DIM,
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
                HEAD_DIM,
                total_rows,
                &scratch.probabilities,
                &scratch.value,
                None,
                &mut scratch.attended,
            )?;
            for row in 0..rows {
                let target = row * Q_DIM + q_head * HEAD_DIM;
                scratch.attention[target..target + HEAD_DIM]
                    .copy_from_slice(&scratch.attended[row * HEAD_DIM..(row + 1) * HEAD_DIM]);
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

fn apply_half_split_rope(
    values: &mut [f32],
    rows: usize,
    heads: usize,
    position_offset: usize,
) -> Result<()> {
    if rows == 0
        || heads == 0
        || !HEAD_DIM.is_multiple_of(2)
        || values.len() != rows * heads * HEAD_DIM
    {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: RoPE shape mismatch: values={}, rows={rows}, heads={heads}, head_dim={HEAD_DIM}",
            values.len()
        )));
    }
    let half = HEAD_DIM / 2;
    for row in 0..rows {
        let position = (position_offset + row) as f32;
        for head in 0..heads {
            let base = (row * heads + head) * HEAD_DIM;
            for pair in 0..half {
                let frequency = ROPE_BASE.powf(-(2 * pair) as f32 / HEAD_DIM as f32);
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
    mapped: &DelayMappedDescriptors,
    layer: usize,
    block: &mut DelayBlock,
) -> Result<()> {
    let descriptors = mapped.layer(layer);
    widen_tensor(mapped, descriptors.input_norm, &mut block.input_norm)?;
    transpose_tensor(mapped, descriptors.q, Q_DIM, HIDDEN_DIM, &mut block.q_w_t)?;
    widen_tensor(mapped, descriptors.q_norm, &mut block.q_norm)?;
    transpose_tensor(mapped, descriptors.k, KV_DIM, HIDDEN_DIM, &mut block.k_w_t)?;
    widen_tensor(mapped, descriptors.k_norm, &mut block.k_norm)?;
    transpose_tensor(mapped, descriptors.v, KV_DIM, HIDDEN_DIM, &mut block.v_w_t)?;
    transpose_tensor(mapped, descriptors.o, HIDDEN_DIM, Q_DIM, &mut block.o_w_t)?;
    widen_tensor(mapped, descriptors.ffn_norm, &mut block.ffn_norm)?;
    transpose_tensor(
        mapped,
        descriptors.gate,
        FFN_DIM,
        HIDDEN_DIM,
        &mut block.gate_w_t,
    )?;
    transpose_tensor(
        mapped,
        descriptors.up,
        FFN_DIM,
        HIDDEN_DIM,
        &mut block.up_w_t,
    )?;
    transpose_tensor(
        mapped,
        descriptors.down,
        HIDDEN_DIM,
        FFN_DIM,
        &mut block.down_w_t,
    )?;
    Ok(())
}

fn last_logits(
    compute: &Compute,
    mapped: &DelayMappedDescriptors,
    runtime: &DelayRuntimeScratch,
    scratch: &DelayStepScratch,
) -> Result<MossTtsDelayLogits> {
    if scratch.norm.len() < HIDDEN_DIM {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: no final hidden row is available"
        )));
    }
    let hidden = &scratch.norm[scratch.norm.len() - HIDDEN_DIM..];
    let mut chunk = lock_scratch(&runtime.head_chunk, MAPPED)?;
    let mut text_logits = vec![0.0; TEXT_VOCAB_SIZE];
    project_head(
        compute,
        mapped,
        mapped.head(0),
        TEXT_VOCAB_SIZE,
        hidden,
        &mut chunk,
        &mut text_logits,
    )?;
    let mut audio_logits = vec![0.0; NUM_AUDIO_CODEBOOKS * AUDIO_VOCAB_WITH_PAD];
    for codebook in 0..NUM_AUDIO_CODEBOOKS {
        let start = codebook * AUDIO_VOCAB_WITH_PAD;
        project_head(
            compute,
            mapped,
            mapped.head(1 + codebook),
            AUDIO_VOCAB_WITH_PAD,
            hidden,
            &mut chunk,
            &mut audio_logits[start..start + AUDIO_VOCAB_WITH_PAD],
        )?;
    }
    reject_non_finite("text logits", &text_logits)?;
    reject_non_finite("audio logits", &audio_logits)?;
    Ok(MossTtsDelayLogits {
        text_logits,
        audio_logits,
    })
}

#[allow(clippy::too_many_arguments)]
fn project_head(
    compute: &Compute,
    mapped: &DelayMappedDescriptors,
    info: &GgufTensorInfo,
    rows: usize,
    hidden: &[f32],
    chunk: &mut Vec<f32>,
    output: &mut [f32],
) -> Result<()> {
    if hidden.len() != HIDDEN_DIM || output.len() != rows {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: head projection shape mismatch: hidden={}, rows={rows}, output={} ",
            hidden.len(),
            output.len()
        )));
    }
    let mut row = 0;
    while row < rows {
        let chunk_rows = HEAD_CHUNK_ROWS.min(rows - row);
        widen_rows(mapped, info, row, chunk_rows, chunk)?;
        compute.gemv_f32(
            chunk_rows,
            HIDDEN_DIM,
            chunk,
            hidden,
            None,
            &mut output[row..row + chunk_rows],
        )?;
        row += chunk_rows;
    }
    Ok(())
}

fn widen_row(
    mapped: &DelayMappedDescriptors,
    info: &GgufTensorInfo,
    row: usize,
    output: &mut Vec<f32>,
) -> Result<()> {
    widen_rows(mapped, info, row, 1, output)
}

fn widen_rows(
    mapped: &DelayMappedDescriptors,
    info: &GgufTensorInfo,
    first_row: usize,
    rows: usize,
    output: &mut Vec<f32>,
) -> Result<()> {
    let element_size = info.dtype.type_size();
    let bytes = mapped.file().tensor_bytes(info);
    let start = first_row
        .checked_mul(HIDDEN_DIM)
        .and_then(|value| value.checked_mul(element_size))
        .ok_or_else(|| VokraError::InvalidArgument(format!("{LABEL}: row offset overflow")))?;
    let len = rows
        .checked_mul(HIDDEN_DIM)
        .and_then(|value| value.checked_mul(element_size))
        .ok_or_else(|| VokraError::InvalidArgument(format!("{LABEL}: row length overflow")))?;
    let end = start
        .checked_add(len)
        .ok_or_else(|| VokraError::InvalidArgument(format!("{LABEL}: row range overflow")))?;
    let source = bytes.get(start..end).ok_or_else(|| {
        VokraError::ModelLoad(format!(
            "{LABEL}: tensor `{}` row range {first_row}..{} exceeds {} bytes",
            info.name,
            first_row + rows,
            bytes.len()
        ))
    })?;
    widen_into(source, info.dtype, output, MAPPED)
}

fn widen_tensor(
    mapped: &DelayMappedDescriptors,
    info: &GgufTensorInfo,
    output: &mut Vec<f32>,
) -> Result<()> {
    widen_into(mapped.file().tensor_bytes(info), info.dtype, output, MAPPED)
}

fn transpose_tensor(
    mapped: &DelayMappedDescriptors,
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
        MAPPED,
    )
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
    fn complete_backend_contract_includes_attention_and_head_reductions() {
        Compute::for_backend(BackendKind::Cpu, MOSS_TTS_DELAY_HOT_OPS)
            .expect("CPU covers the complete Delay graph");
        #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
        match Compute::for_backend(BackendKind::Metal, MOSS_TTS_DELAY_HOT_OPS) {
            Ok(compute) => assert_eq!(compute.backend_name(), "metal"),
            Err(VokraError::BackendUnavailable(_)) => {}
            Err(error) => panic!("MOSS-TTS Delay has a Metal coverage gap: {error}"),
        }
    }

    #[test]
    fn half_split_rope_position_zero_is_identity() {
        let mut values = (0..HEAD_DIM).map(|value| value as f32).collect::<Vec<_>>();
        let expected = values.clone();
        apply_half_split_rope(&mut values, 1, 1, 0).unwrap();
        assert_eq!(values, expected);
    }

    #[test]
    fn causal_mask_respects_cached_prefix() {
        let mut scores = vec![2.0; 6];
        scale_and_mask(&mut scores, 2, 3, 1, 0.5).unwrap();
        assert_eq!(&scores[..3], &[1.0, 1.0, f32::MIN]);
        assert_eq!(&scores[3..], &[1.0, 1.0, 1.0]);
    }
}
