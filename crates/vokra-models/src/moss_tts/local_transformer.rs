//! Native MOSS-TTS Local Transformer v1.5 generation.
//!
//! The global Qwen3 forward reuses the mapped Delay implementation under the
//! authenticated Local tensor layout. For each generated frame, one local
//! GPT-2 block autoregressively selects the assistant/end decision and twelve
//! RVQ codes. Every learned reduction uses one selected [`Compute`] backend;
//! no CPU codec or local-decoder fallback is hidden behind Metal selection.

use std::collections::BTreeSet;
use std::path::Path;
use std::sync::{Arc, Mutex};

use vokra_core::backend::BackendKind;
use vokra_core::{KvCache, Result, Sampler, SamplerConfig, VokraError};

use crate::compute::{Compute, HotOp};
use crate::mapped_weights::lock_scratch;

use super::delay::DelayMappedDescriptors;
use super::delay_transformer::{
    DelayRuntimeScratch, DelayStepScratch, PREFILL_CHUNK_ROWS, forward_chunk, last_hidden,
    project_head, reject_non_finite, widen_row, widen_tensor,
};
use super::local::{
    LABEL, LOCAL_CACHE_CAPACITY, LOCAL_FFN_DIM, LOCAL_HEAD_DIM, LOCAL_LAYER_NORM_EPS,
    LOCAL_NUM_HEADS, LOCAL_ROPE_BASE, LOCAL_TOPOLOGY, LocalMappedDescriptors,
    MossTtsLocalCheckpoint,
};

const AUDIO_START_TOKEN_ID: u32 = 151_669;
const AUDIO_END_TOKEN_ID: u32 = 151_670;
const AUDIO_ASSISTANT_SLOT_TOKEN_ID: u32 = 151_656;
const AUDIO_PAD_TOKEN_ID: u32 = 1_024;

/// Complete learned-op contract for global Qwen3 plus local GPT-2.
pub const MOSS_TTS_LOCAL_HOT_OPS: &[HotOp] = &[
    HotOp::Gemm,
    HotOp::Gemv,
    HotOp::Softmax,
    HotOp::RmsNorm,
    HotOp::LayerNorm,
    HotOp::Silu,
];

/// Sampling controls for the fixed official Local `generate` signature.
#[derive(Debug, Clone)]
pub struct MossTtsLocalGenerationOptions {
    /// Maximum appended audio frames. The official default is 4096.
    pub max_new_frames: usize,
    /// Binary assistant/end sampler. Defaults to temperature 1, top-k 50 and
    /// top-p 1 (represented as disabled).
    pub text_sampler: SamplerConfig,
    /// Per-codebook audio sampler. Defaults to temperature 1, top-k 50 and
    /// top-p 0.95. Vokra uses a fixed first-party seed.
    pub audio_sampler: SamplerConfig,
}

impl Default for MossTtsLocalGenerationOptions {
    fn default() -> Self {
        Self {
            max_new_frames: 4_096,
            text_sampler: SamplerConfig {
                temperature: 1.0,
                top_k: Some(50),
                top_p: None,
                repetition_penalty: None,
                seed: 0,
            },
            audio_sampler: SamplerConfig {
                temperature: 1.0,
                top_k: Some(50),
                top_p: Some(0.95),
                repetition_penalty: None,
                seed: 1,
            },
        }
    }
}

impl MossTtsLocalGenerationOptions {
    /// Deterministic upstream `do_sample=false` equivalent.
    #[must_use]
    pub fn greedy(max_new_frames: usize) -> Self {
        Self {
            max_new_frames,
            text_sampler: SamplerConfig::greedy(),
            audio_sampler: SamplerConfig::greedy(),
        }
    }
}

/// Official single-sequence Local generation boundary.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MossTtsLocalGeneration {
    /// Row-major `[rows,13]` values starting at the prompt's last audio-start
    /// row and including all generated assistant/audio rows. This matches the
    /// fixed processor's decode input and preserves continuation context.
    pub rows_from_audio_start: Vec<u32>,
    /// Number of prompt audio rows following the last audio-start marker.
    pub start_length: usize,
    /// Number of newly appended 12-codebook frames.
    pub generated_frames: usize,
}

impl MossTtsLocalGeneration {
    pub const fn column_count(&self) -> usize {
        1 + LOCAL_TOPOLOGY.num_audio_codebooks
    }

    pub fn row_count(&self) -> usize {
        self.rows_from_audio_start.len() / self.column_count()
    }

    pub fn row(&self, index: usize) -> Result<&[u32]> {
        if index >= self.row_count() {
            return Err(VokraError::InvalidArgument(format!(
                "{LABEL}: generated row {index} is outside 0..{}",
                self.row_count()
            )));
        }
        let columns = self.column_count();
        let start = index * columns;
        Ok(&self.rows_from_audio_start[start..start + columns])
    }
}

/// Native Local Transformer model on one explicit CPU or Metal backend.
#[derive(Clone)]
pub struct MossTtsLocal {
    checkpoint: MossTtsLocalCheckpoint,
    backend: BackendKind,
    qwen_runtime: Arc<DelayRuntimeScratch>,
    local_runtime: Arc<LocalRuntimeScratch>,
}

impl std::fmt::Debug for MossTtsLocal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MossTtsLocal")
            .field("backend", &self.backend)
            .field(
                "requires_metadata_repair",
                &self.checkpoint.requires_metadata_repair(),
            )
            .finish_non_exhaustive()
    }
}

impl MossTtsLocal {
    /// Opens and strictly binds the mmap checkpoint before backend preflight.
    pub fn open_mapped(path: impl AsRef<Path>, backend: BackendKind) -> Result<Self> {
        Self::from_checkpoint(MossTtsLocalCheckpoint::open_mapped(path)?, backend)
    }

    pub fn from_checkpoint(
        checkpoint: MossTtsLocalCheckpoint,
        backend: BackendKind,
    ) -> Result<Self> {
        let _ = Compute::for_backend(backend, MOSS_TTS_LOCAL_HOT_OPS)?;
        Ok(Self {
            checkpoint,
            backend,
            qwen_runtime: Arc::new(DelayRuntimeScratch::default()),
            local_runtime: Arc::new(LocalRuntimeScratch::default()),
        })
    }

    pub const fn backend(&self) -> BackendKind {
        self.backend
    }

    pub const fn checkpoint(&self) -> &MossTtsLocalCheckpoint {
        &self.checkpoint
    }

    pub fn default_generation_options(&self) -> MossTtsLocalGenerationOptions {
        MossTtsLocalGenerationOptions::default()
    }

    /// Runs global Qwen3 and the first local GPT-2 step, returning the exact
    /// binary `[assistant_slot, audio_end]` logits for parity tests.
    pub fn forward_prompt_local_text_logits(&self, prompt_rows: &[u32]) -> Result<[f32; 2]> {
        let compute = Compute::for_backend(self.backend, MOSS_TTS_LOCAL_HOT_OPS)?;
        let mapped = self.checkpoint.mapped();
        let global = prefill_global(&compute, mapped.qwen(), &self.qwen_runtime, prompt_rows)?;
        let global_hidden = last_hidden(mapped.qwen(), &global.scratch)?.to_vec();
        let mut local_cache = LocalKvCache::default();
        let mut local_scratch = LocalStepScratch::default();
        let mut block_guard = lock_scratch(&self.local_runtime.block, super::local::MAPPED)?;
        let block = ensure_local_block(mapped, &mut block_guard)?;
        let local_hidden = local_decode_step(
            &compute,
            block,
            &global_hidden,
            &mut local_cache,
            &mut local_scratch,
        )?;
        let mut logits = [0.0; 2];
        project_mapped_head(
            &compute,
            mapped,
            mapped.local_text_head(),
            2,
            local_hidden,
            &self.local_runtime,
            &mut logits,
        )?;
        Ok(logits)
    }

    /// Runs the official single-sequence global/local generation loop from an
    /// explicit upstream-compatible `[rows,13]` prompt matrix.
    pub fn generate_rows(
        &self,
        prompt_rows: &[u32],
        options: &MossTtsLocalGenerationOptions,
    ) -> Result<MossTtsLocalGeneration> {
        validate_generation_options(prompt_rows, options)?;
        let compute = Compute::for_backend(self.backend, MOSS_TTS_LOCAL_HOT_OPS)?;
        let mapped = self.checkpoint.mapped();
        let columns = LOCAL_TOPOLOGY.input_columns();
        let prompt_count = prompt_rows.len() / columns;
        let audio_start_row = (0..prompt_count)
            .rev()
            .find(|&row| prompt_rows[row * columns] == AUDIO_START_TOKEN_ID)
            .ok_or_else(|| {
                VokraError::InvalidArgument(format!(
                    "{LABEL}: every generation prompt must contain audio-start token {AUDIO_START_TOKEN_ID}"
                ))
            })?;
        let start_length = prompt_count - audio_start_row - 1;
        let mut global = prefill_global(&compute, mapped.qwen(), &self.qwen_runtime, prompt_rows)?;
        let mut appended = Vec::with_capacity(options.max_new_frames * columns);
        let mut text_sampler = Sampler::new(options.text_sampler.clone());
        let mut audio_config = options.audio_sampler.clone();
        let audio_repetition_penalty = audio_config.repetition_penalty.take();
        let mut audio_sampler = Sampler::new(audio_config);
        let mut audio_history = Vec::with_capacity(options.max_new_frames * 12);
        let mut local_scratch = LocalStepScratch::default();
        let mut block_guard = lock_scratch(&self.local_runtime.block, super::local::MAPPED)?;
        let block = ensure_local_block(mapped, &mut block_guard)?;

        for frame in 0..options.max_new_frames {
            let global_hidden = last_hidden(mapped.qwen(), &global.scratch)?.to_vec();
            let mut local_cache = LocalKvCache::default();
            let mut local_hidden = local_decode_step(
                &compute,
                block,
                &global_hidden,
                &mut local_cache,
                &mut local_scratch,
            )?
            .to_vec();

            let mut text_logits = [0.0; 2];
            project_mapped_head(
                &compute,
                mapped,
                mapped.local_text_head(),
                2,
                &local_hidden,
                &self.local_runtime,
                &mut text_logits,
            )?;
            let text_choice = text_sampler.sample(&mut text_logits);
            let text_token = match text_choice {
                0 => AUDIO_ASSISTANT_SLOT_TOKEN_ID,
                1 => AUDIO_END_TOKEN_ID,
                other => {
                    return Err(VokraError::InvalidArgument(format!(
                        "{LABEL}: binary local text sampler returned class {other}"
                    )));
                }
            };
            if text_token == AUDIO_END_TOKEN_ID {
                break;
            }

            let mut row = vec![AUDIO_PAD_TOKEN_ID; columns];
            row[0] = AUDIO_ASSISTANT_SLOT_TOKEN_ID;
            for codebook in 0..LOCAL_TOPOLOGY.num_audio_codebooks {
                let mut logits = vec![0.0; LOCAL_TOPOLOGY.audio_vocab_with_pad];
                project_mapped_head(
                    &compute,
                    mapped,
                    mapped.qwen().head(1 + codebook),
                    LOCAL_TOPOLOGY.audio_vocab_with_pad,
                    &local_hidden,
                    &self.local_runtime,
                    &mut logits,
                )?;
                if let Some(penalty) = audio_repetition_penalty {
                    apply_audio_repetition_penalty(&mut logits, &audio_history, codebook, penalty)?;
                }
                let token = audio_sampler.sample(&mut logits);
                row[1 + codebook] = token;
                if codebook + 1 < LOCAL_TOPOLOGY.num_audio_codebooks {
                    let mut embedding = Vec::new();
                    widen_row(
                        mapped.qwen(),
                        mapped.qwen().audio_embedding(codebook),
                        token as usize,
                        &mut embedding,
                    )?;
                    local_hidden = local_decode_step(
                        &compute,
                        block,
                        &embedding,
                        &mut local_cache,
                        &mut local_scratch,
                    )?
                    .to_vec();
                }
            }

            audio_history.extend_from_slice(&row[1..]);
            appended.extend_from_slice(&row);
            if frame + 1 < options.max_new_frames {
                forward_chunk(
                    &compute,
                    mapped.qwen(),
                    &self.qwen_runtime,
                    &mut global.scratch,
                    &mut global.kv_cache,
                    &row,
                    1,
                )?;
            }
        }

        let mut rows_from_audio_start = prompt_rows[audio_start_row * columns..].to_vec();
        rows_from_audio_start.extend_from_slice(&appended);
        Ok(MossTtsLocalGeneration {
            rows_from_audio_start,
            start_length,
            generated_frames: appended.len() / columns,
        })
    }
}

struct GlobalState {
    kv_cache: KvCache,
    scratch: DelayStepScratch,
}

fn prefill_global(
    compute: &Compute,
    mapped: &DelayMappedDescriptors,
    runtime: &DelayRuntimeScratch,
    prompt_rows: &[u32],
) -> Result<GlobalState> {
    let columns = LOCAL_TOPOLOGY.input_columns();
    if prompt_rows.is_empty() || !prompt_rows.len().is_multiple_of(columns) {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: prompt must be a non-empty [rows,{columns}] u32 matrix, got {} values",
            prompt_rows.len()
        )));
    }
    let rows = prompt_rows.len() / columns;
    if rows > LOCAL_TOPOLOGY.max_position_embeddings {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: prompt rows {rows} exceed max positions {}",
            LOCAL_TOPOLOGY.max_position_embeddings
        )));
    }
    let mut kv_cache = KvCache::with_reserve(
        LOCAL_TOPOLOGY.num_layers,
        LOCAL_TOPOLOGY.kv_dim(),
        rows.min(256).max(1),
    );
    let mut scratch = DelayStepScratch::default();
    for row_start in (0..rows).step_by(PREFILL_CHUNK_ROWS) {
        let chunk_rows = PREFILL_CHUNK_ROWS.min(rows - row_start);
        let start = row_start * columns;
        let end = start + chunk_rows * columns;
        forward_chunk(
            compute,
            mapped,
            runtime,
            &mut scratch,
            &mut kv_cache,
            &prompt_rows[start..end],
            chunk_rows,
        )?;
    }
    Ok(GlobalState { kv_cache, scratch })
}

#[derive(Default)]
struct LocalRuntimeScratch {
    block: Mutex<Option<LocalBlock>>,
    head_chunk: Mutex<Vec<f32>>,
}

#[derive(Default)]
struct LocalBlock {
    qkv_bias: Vec<f32>,
    qkv_weight: Vec<f32>,
    projection_bias: Vec<f32>,
    projection_weight: Vec<f32>,
    norm1_bias: Vec<f32>,
    norm1_weight: Vec<f32>,
    norm2_bias: Vec<f32>,
    norm2_weight: Vec<f32>,
    ffn_in_bias: Vec<f32>,
    ffn_in_weight: Vec<f32>,
    ffn_out_bias: Vec<f32>,
    ffn_out_weight: Vec<f32>,
    final_norm_bias: Vec<f32>,
    final_norm_weight: Vec<f32>,
}

fn ensure_local_block<'a>(
    mapped: &LocalMappedDescriptors,
    slot: &'a mut Option<LocalBlock>,
) -> Result<&'a LocalBlock> {
    if slot.is_none() {
        let descriptors = mapped.local_block();
        let qwen = mapped.qwen();
        let mut block = LocalBlock::default();
        widen_tensor(qwen, descriptors.qkv_bias, &mut block.qkv_bias)?;
        widen_tensor(qwen, descriptors.qkv_weight, &mut block.qkv_weight)?;
        widen_tensor(
            qwen,
            descriptors.projection_bias,
            &mut block.projection_bias,
        )?;
        widen_tensor(
            qwen,
            descriptors.projection_weight,
            &mut block.projection_weight,
        )?;
        widen_tensor(qwen, descriptors.norm1_bias, &mut block.norm1_bias)?;
        widen_tensor(qwen, descriptors.norm1_weight, &mut block.norm1_weight)?;
        widen_tensor(qwen, descriptors.norm2_bias, &mut block.norm2_bias)?;
        widen_tensor(qwen, descriptors.norm2_weight, &mut block.norm2_weight)?;
        widen_tensor(qwen, descriptors.ffn_in_bias, &mut block.ffn_in_bias)?;
        widen_tensor(qwen, descriptors.ffn_in_weight, &mut block.ffn_in_weight)?;
        widen_tensor(qwen, descriptors.ffn_out_bias, &mut block.ffn_out_bias)?;
        widen_tensor(qwen, descriptors.ffn_out_weight, &mut block.ffn_out_weight)?;
        widen_tensor(
            qwen,
            descriptors.final_norm_bias,
            &mut block.final_norm_bias,
        )?;
        widen_tensor(
            qwen,
            descriptors.final_norm_weight,
            &mut block.final_norm_weight,
        )?;
        *slot = Some(block);
    }
    slot.as_ref().ok_or_else(|| {
        VokraError::InvalidArgument(format!(
            "{LABEL}: local block materialization completed without a block"
        ))
    })
}

#[derive(Default)]
struct LocalKvCache {
    key: Vec<f32>,
    value: Vec<f32>,
    positions: usize,
}

impl LocalKvCache {
    fn append(&mut self, key: &[f32], value: &[f32]) -> Result<()> {
        if key.len() != LOCAL_TOPOLOGY.hidden_dim || value.len() != LOCAL_TOPOLOGY.hidden_dim {
            return Err(VokraError::InvalidArgument(format!(
                "{LABEL}: local KV row shape mismatch: key={}, value={}, expected {}",
                key.len(),
                value.len(),
                LOCAL_TOPOLOGY.hidden_dim
            )));
        }
        if self.positions >= LOCAL_CACHE_CAPACITY {
            return Err(VokraError::InvalidArgument(format!(
                "{LABEL}: local KV cache exceeds fixed capacity {LOCAL_CACHE_CAPACITY}"
            )));
        }
        self.key.extend_from_slice(key);
        self.value.extend_from_slice(value);
        self.positions += 1;
        Ok(())
    }
}

#[derive(Default)]
struct LocalStepScratch {
    hidden: Vec<f32>,
    norm: Vec<f32>,
    qkv: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    key_t: Vec<f32>,
    values: Vec<f32>,
    scores: Vec<f32>,
    probabilities: Vec<f32>,
    attended: Vec<f32>,
    attention: Vec<f32>,
    projection: Vec<f32>,
    ffn: Vec<f32>,
    ffn_activated: Vec<f32>,
    ffn_out: Vec<f32>,
    final_hidden: Vec<f32>,
}

fn resize_zero(values: &mut Vec<f32>, len: usize) {
    values.clear();
    values.resize(len, 0.0);
}

fn local_decode_step<'a>(
    compute: &Compute,
    block: &LocalBlock,
    input: &[f32],
    cache: &mut LocalKvCache,
    scratch: &'a mut LocalStepScratch,
) -> Result<&'a [f32]> {
    let hidden = LOCAL_TOPOLOGY.hidden_dim;
    if input.len() != hidden {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: local input has {} values, expected {hidden}",
            input.len()
        )));
    }
    scratch.hidden.clear();
    scratch.hidden.extend_from_slice(input);
    resize_zero(&mut scratch.norm, hidden);
    resize_zero(&mut scratch.qkv, 3 * hidden);
    resize_zero(&mut scratch.q, hidden);
    resize_zero(&mut scratch.k, hidden);
    resize_zero(&mut scratch.v, hidden);
    resize_zero(&mut scratch.attention, hidden);
    resize_zero(&mut scratch.projection, hidden);
    resize_zero(&mut scratch.ffn, LOCAL_FFN_DIM);
    resize_zero(&mut scratch.ffn_activated, LOCAL_FFN_DIM);
    resize_zero(&mut scratch.ffn_out, hidden);
    resize_zero(&mut scratch.final_hidden, hidden);

    compute.layer_norm_f32(
        &scratch.hidden,
        &mut scratch.norm,
        1,
        hidden,
        &block.norm1_weight,
        &block.norm1_bias,
        LOCAL_LAYER_NORM_EPS,
    )?;
    compute.gemv_f32(
        3 * hidden,
        hidden,
        &block.qkv_weight,
        &scratch.norm,
        Some(&block.qkv_bias),
        &mut scratch.qkv,
    )?;
    scratch.q.copy_from_slice(&scratch.qkv[..hidden]);
    scratch.k.copy_from_slice(&scratch.qkv[hidden..2 * hidden]);
    scratch.v.copy_from_slice(&scratch.qkv[2 * hidden..]);
    apply_adjacent_rope(
        &mut scratch.q,
        LOCAL_NUM_HEADS,
        LOCAL_HEAD_DIM,
        LOCAL_ROPE_BASE,
        cache.positions,
    )?;
    apply_adjacent_rope(
        &mut scratch.k,
        LOCAL_NUM_HEADS,
        LOCAL_HEAD_DIM,
        LOCAL_ROPE_BASE,
        cache.positions,
    )?;
    cache.append(&scratch.k, &scratch.v)?;
    local_attention(compute, cache, scratch)?;
    compute.gemv_f32(
        hidden,
        hidden,
        &block.projection_weight,
        &scratch.attention,
        Some(&block.projection_bias),
        &mut scratch.projection,
    )?;
    for (value, &residual) in scratch.hidden.iter_mut().zip(&scratch.projection) {
        *value += residual;
    }

    compute.layer_norm_f32(
        &scratch.hidden,
        &mut scratch.norm,
        1,
        hidden,
        &block.norm2_weight,
        &block.norm2_bias,
        LOCAL_LAYER_NORM_EPS,
    )?;
    compute.gemv_f32(
        LOCAL_FFN_DIM,
        hidden,
        &block.ffn_in_weight,
        &scratch.norm,
        Some(&block.ffn_in_bias),
        &mut scratch.ffn,
    )?;
    compute.silu_f32(&scratch.ffn, &mut scratch.ffn_activated)?;
    compute.gemv_f32(
        hidden,
        LOCAL_FFN_DIM,
        &block.ffn_out_weight,
        &scratch.ffn_activated,
        Some(&block.ffn_out_bias),
        &mut scratch.ffn_out,
    )?;
    for (value, &residual) in scratch.hidden.iter_mut().zip(&scratch.ffn_out) {
        *value += residual;
    }
    compute.layer_norm_f32(
        &scratch.hidden,
        &mut scratch.final_hidden,
        1,
        hidden,
        &block.final_norm_weight,
        &block.final_norm_bias,
        LOCAL_LAYER_NORM_EPS,
    )?;
    reject_non_finite(LABEL, "local final hidden", &scratch.final_hidden)?;
    Ok(&scratch.final_hidden)
}

fn local_attention(
    compute: &Compute,
    cache: &LocalKvCache,
    scratch: &mut LocalStepScratch,
) -> Result<()> {
    let positions = cache.positions;
    let hidden = LOCAL_TOPOLOGY.hidden_dim;
    if positions == 0
        || cache.key.len() != positions * hidden
        || cache.value.len() != positions * hidden
    {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: local attention cache shape mismatch"
        )));
    }
    resize_zero(&mut scratch.key_t, LOCAL_HEAD_DIM * positions);
    resize_zero(&mut scratch.values, positions * LOCAL_HEAD_DIM);
    resize_zero(&mut scratch.scores, positions);
    resize_zero(&mut scratch.probabilities, positions);
    resize_zero(&mut scratch.attended, LOCAL_HEAD_DIM);
    scratch.attention.fill(0.0);
    let scale = (LOCAL_HEAD_DIM as f32).sqrt().recip();
    for head in 0..LOCAL_NUM_HEADS {
        for position in 0..positions {
            let source = position * hidden + head * LOCAL_HEAD_DIM;
            for dimension in 0..LOCAL_HEAD_DIM {
                scratch.key_t[dimension * positions + position] = cache.key[source + dimension];
                scratch.values[position * LOCAL_HEAD_DIM + dimension] =
                    cache.value[source + dimension];
            }
        }
        let query_start = head * LOCAL_HEAD_DIM;
        compute.gemm_f32(
            1,
            positions,
            LOCAL_HEAD_DIM,
            &scratch.q[query_start..query_start + LOCAL_HEAD_DIM],
            &scratch.key_t,
            None,
            &mut scratch.scores,
        )?;
        for score in &mut scratch.scores {
            *score *= scale;
        }
        compute.softmax_f32(&scratch.scores, &mut scratch.probabilities, 1, positions)?;
        compute.gemm_f32(
            1,
            LOCAL_HEAD_DIM,
            positions,
            &scratch.probabilities,
            &scratch.values,
            None,
            &mut scratch.attended,
        )?;
        scratch.attention[query_start..query_start + LOCAL_HEAD_DIM]
            .copy_from_slice(&scratch.attended);
    }
    Ok(())
}

fn apply_adjacent_rope(
    values: &mut [f32],
    heads: usize,
    head_dim: usize,
    rope_base: f32,
    position: usize,
) -> Result<()> {
    if heads == 0
        || !head_dim.is_multiple_of(2)
        || values.len() != heads * head_dim
        || !rope_base.is_finite()
        || rope_base <= 0.0
    {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: adjacent RoPE shape mismatch"
        )));
    }
    for head in 0..heads {
        let base = head * head_dim;
        for pair in 0..head_dim / 2 {
            let frequency = rope_base.powf(-(2 * pair) as f32 / head_dim as f32);
            let angle = position as f32 * frequency;
            let (sin, cos) = angle.sin_cos();
            let even_index = base + 2 * pair;
            let odd_index = even_index + 1;
            let even = values[even_index];
            let odd = values[odd_index];
            values[even_index] = even * cos - odd * sin;
            values[odd_index] = even * sin + odd * cos;
        }
    }
    Ok(())
}

fn project_mapped_head(
    compute: &Compute,
    mapped: &LocalMappedDescriptors,
    info: &vokra_core::gguf::GgufTensorInfo,
    rows: usize,
    hidden: &[f32],
    runtime: &LocalRuntimeScratch,
    output: &mut [f32],
) -> Result<()> {
    let mut chunk = lock_scratch(&runtime.head_chunk, super::local::MAPPED)?;
    project_head(
        compute,
        mapped.qwen(),
        info,
        rows,
        hidden,
        &mut chunk,
        output,
    )
}

fn apply_audio_repetition_penalty(
    logits: &mut [f32],
    history: &[u32],
    codebook: usize,
    penalty: f32,
) -> Result<()> {
    let columns = LOCAL_TOPOLOGY.num_audio_codebooks;
    if !penalty.is_finite() || penalty <= 0.0 || !history.len().is_multiple_of(columns) {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: invalid audio repetition penalty/history"
        )));
    }
    let mut unique = BTreeSet::new();
    for frame in history.chunks_exact(columns) {
        unique.insert(frame[codebook] as usize);
    }
    for token in unique {
        if token < logits.len() {
            logits[token] = if logits[token] < 0.0 {
                logits[token] * penalty
            } else {
                logits[token] / penalty
            };
        }
    }
    Ok(())
}

fn validate_generation_options(
    prompt_rows: &[u32],
    options: &MossTtsLocalGenerationOptions,
) -> Result<()> {
    let columns = LOCAL_TOPOLOGY.input_columns();
    if prompt_rows.is_empty() || !prompt_rows.len().is_multiple_of(columns) {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: prompt must be a non-empty [rows,{columns}] matrix"
        )));
    }
    let prompt_count = prompt_rows.len() / columns;
    if prompt_count
        .checked_add(options.max_new_frames)
        .is_none_or(|rows| rows > LOCAL_TOPOLOGY.max_position_embeddings)
    {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: prompt plus frame budget exceeds max positions {}",
            LOCAL_TOPOLOGY.max_position_embeddings
        )));
    }
    validate_sampler("text", &options.text_sampler)?;
    if options.text_sampler.repetition_penalty.is_some() {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: the fixed upstream binary text sampler has no repetition-penalty input"
        )));
    }
    validate_sampler("audio", &options.audio_sampler)
}

fn validate_sampler(kind: &str, config: &SamplerConfig) -> Result<()> {
    if !config.temperature.is_finite() || config.temperature < 0.0 {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: {kind} temperature must be finite and >= 0"
        )));
    }
    if config.top_k == Some(0)
        || config
            .top_p
            .is_some_and(|value| !value.is_finite() || value <= 0.0 || value > 1.0)
        || config
            .repetition_penalty
            .is_some_and(|value| !value.is_finite() || value <= 0.0)
    {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: invalid {kind} sampler configuration"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hot_op_contract_extends_qwen_with_affine_layer_norm() {
        for op in super::super::delay_transformer::MOSS_TTS_DELAY_HOT_OPS {
            assert!(MOSS_TTS_LOCAL_HOT_OPS.contains(op));
        }
        assert!(MOSS_TTS_LOCAL_HOT_OPS.contains(&HotOp::LayerNorm));
    }

    #[test]
    fn cpu_covers_complete_local_learned_graph() {
        Compute::for_backend(BackendKind::Cpu, MOSS_TTS_LOCAL_HOT_OPS)
            .expect("CPU covers global Qwen3 plus local GPT-2");
    }

    #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
    #[test]
    fn metal_covers_complete_local_learned_graph() {
        match Compute::for_backend(BackendKind::Metal, MOSS_TTS_LOCAL_HOT_OPS) {
            Ok(_) => {}
            Err(error) => panic!("MOSS-TTS Local has a Metal coverage gap: {error}"),
        }
    }

    #[test]
    fn defaults_match_fixed_official_generate_signature() {
        let options = MossTtsLocalGenerationOptions::default();
        assert_eq!(options.max_new_frames, 4_096);
        assert_eq!(options.text_sampler.temperature, 1.0);
        assert_eq!(options.text_sampler.top_k, Some(50));
        assert_eq!(options.text_sampler.top_p, None);
        assert_eq!(options.audio_sampler.temperature, 1.0);
        assert_eq!(options.audio_sampler.top_k, Some(50));
        assert_eq!(options.audio_sampler.top_p, Some(0.95));
        assert_eq!(options.audio_sampler.repetition_penalty, None);
    }

    #[test]
    fn adjacent_rope_position_zero_is_identity() {
        let mut values = (0..LOCAL_TOPOLOGY.hidden_dim)
            .map(|index| index as f32 / 100.0)
            .collect::<Vec<_>>();
        let expected = values.clone();
        apply_adjacent_rope(
            &mut values,
            LOCAL_NUM_HEADS,
            LOCAL_HEAD_DIM,
            LOCAL_ROPE_BASE,
            0,
        )
        .unwrap();
        assert_eq!(values, expected);
    }

    #[test]
    fn audio_repetition_penalty_is_per_codebook() {
        let mut logits = vec![2.0, -2.0, 3.0];
        let mut history = vec![0; 2 * LOCAL_TOPOLOGY.num_audio_codebooks];
        history[0] = 1;
        history[LOCAL_TOPOLOGY.num_audio_codebooks] = 1;
        history[1] = 2;
        apply_audio_repetition_penalty(&mut logits, &history, 0, 2.0).unwrap();
        assert_eq!(logits, vec![2.0, -4.0, 3.0]);
    }

    #[test]
    fn generation_rows_preserve_official_thirteen_column_boundary() {
        let generation = MossTtsLocalGeneration {
            rows_from_audio_start: vec![0; 2 * LOCAL_TOPOLOGY.input_columns()],
            start_length: 1,
            generated_frames: 1,
        };
        assert_eq!(generation.column_count(), 13);
        assert_eq!(generation.row_count(), 2);
        assert_eq!(generation.row(1).unwrap().len(), 13);
    }

    #[test]
    fn pad_token_is_not_a_generated_audio_class() {
        assert_eq!(
            AUDIO_PAD_TOKEN_ID as usize,
            LOCAL_TOPOLOGY.audio_vocab_with_pad
        );
    }
}
