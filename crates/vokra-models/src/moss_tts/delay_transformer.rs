//! Bounded-memory native Qwen3 forward for MOSS-TTS Delay-class releases.
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
use crate::moss_audio_tokenizer::{
    MossAudioTokenizer, MossAudioTokenizerVariant, MossDecodedAudio,
};

use super::delay::{DelayMappedDescriptors, DelayTopology, MossTtsDelayCheckpoint};
use super::voice_generator::MossVoiceGeneratorCheckpoint;

pub use self::generation::{MossTtsDelayGeneration, MossTtsDelayGenerationOptions};

const LABEL: &str = "moss_tts/delay";
pub(super) const PREFILL_CHUNK_ROWS: usize = 8;
const HEAD_CHUNK_ROWS: usize = 512;

/// Every learned operator required by the Delay-class Qwen3 forward.
/// A selected backend must cover the complete set before inference starts.
pub const MOSS_TTS_DELAY_HOT_OPS: &[HotOp] = &[
    HotOp::Gemm,
    HotOp::Gemv,
    HotOp::Softmax,
    HotOp::RmsNorm,
    HotOp::Silu,
];

/// Last-position output of every authenticated Delay-class head.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct MossTtsDelayLogits {
    /// Text head (`lm_heads.0`) logits, length 155,648.
    pub text_logits: Vec<f32>,
    /// Flat codebook-major audio logits: `num_audio_codebooks` rows × 1,025
    /// values. Index 1,024 is the official audio-pad sentinel and remains
    /// present for the caller's generation mask.
    pub audio_logits: Vec<f32>,
    /// Authenticated number of audio-codebook heads (32 or 16).
    pub num_audio_codebooks: usize,
    /// Logit width per audio-codebook head, including pad (1,025).
    pub audio_vocab_with_pad: usize,
    model_label: &'static str,
}

/// One independently decoded audio segment from an official delayed stream.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct MossTtsDelayAudioSegment {
    /// De-delayed frame-major `[frames, num_audio_codebooks]` Full-codec values.
    pub codes: Vec<u32>,
    /// Codec-frame count before continuation trimming.
    pub frames: usize,
    /// 24 kHz mono waveform. For a continued first segment, the official
    /// waveform-ratio trim has already been applied.
    pub audio: MossDecodedAudio,
    /// Number of leading PCM samples removed after full causal decode.
    pub trimmed_prefix_samples: usize,
}

/// Complete Delay-class generation plus zero or more official audio segments.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct MossTtsDelaySynthesis {
    /// Raw appended delayed rows, retained for text/continuation callers.
    pub generated: MossTtsDelayGeneration,
    /// Consecutive non-pad de-delayed segments, each decoded independently.
    pub segments: Vec<MossTtsDelayAudioSegment>,
}

impl MossTtsDelayLogits {
    /// Returns one authenticated audio-codebook head.
    pub fn audio_codebook(&self, codebook: usize) -> Result<&[f32]> {
        if codebook >= self.num_audio_codebooks {
            return Err(VokraError::InvalidArgument(format!(
                "{}: audio codebook {codebook} is outside 0..{}",
                self.model_label, self.num_audio_codebooks
            )));
        }
        let start = codebook * self.audio_vocab_with_pad;
        let end = start
            .checked_add(self.audio_vocab_with_pad)
            .ok_or_else(|| {
                VokraError::InvalidArgument(format!(
                    "{}: audio head {codebook} range overflows",
                    self.model_label
                ))
            })?;
        self.audio_logits.get(start..end).ok_or_else(|| {
            VokraError::InvalidArgument(format!(
                "{}: audio head {codebook} range {start}..{end} exceeds {} logits",
                self.model_label,
                self.audio_logits.len()
            ))
        })
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

    /// Official Base/v1.5 generation defaults.
    pub fn default_generation_options(&self) -> MossTtsDelayGenerationOptions {
        MossTtsDelayGenerationOptions::default()
    }

    /// Runs an explicit upstream-compatible `[rows, 33]` prompt matrix and
    /// returns all last-position text/audio logits.
    ///
    /// Column zero is a text token (`0..155648`); columns 1..32 are audio
    /// codes (`0..1025`, including pad 1024). Raw text is intentionally not
    /// accepted because the GGUF does not embed the official tokenizer or
    /// chat-template assets.
    pub fn forward_prompt_last_logits(&self, prompt_rows: &[u32]) -> Result<MossTtsDelayLogits> {
        forward_prompt_last_logits(self, prompt_rows)
    }

    /// Runs the official delayed-codebook state machine.
    pub fn generate_delay_rows(
        &self,
        prompt_rows: &[u32],
        options: &MossTtsDelayGenerationOptions,
    ) -> Result<MossTtsDelayGeneration> {
        generation::generate_delay_rows(self, prompt_rows, options)
    }

    /// Generates delayed Base/v1.5 rows, restores official 32-codebook frame
    /// order, splits all-pad separators and decodes every segment through the
    /// exact Full MOSS Audio Tokenizer companion.
    ///
    /// The prompt must contain the official last `im_start` framing token so
    /// continuation length can be recovered without guessing. The LLM and
    /// codec must select the same backend; neither stage may fall back to CPU.
    pub fn synthesize_prompt_rows(
        &self,
        codec: &MossAudioTokenizer,
        prompt_rows: &[u32],
        options: &MossTtsDelayGenerationOptions,
    ) -> Result<MossTtsDelaySynthesis> {
        synthesize_prompt_rows(self, codec, prompt_rows, options)
    }
}

/// Native MOSS-VoiceGenerator Qwen3-1.7B model on CPU or Metal.
#[derive(Clone)]
pub struct MossVoiceGenerator {
    checkpoint: MossVoiceGeneratorCheckpoint,
    backend: BackendKind,
    runtime: Arc<DelayRuntimeScratch>,
}

impl std::fmt::Debug for MossVoiceGenerator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MossVoiceGenerator")
            .field("backend", &self.backend)
            .field(
                "requires_metadata_repair",
                &self.checkpoint.requires_metadata_repair(),
            )
            .finish_non_exhaustive()
    }
}

impl MossVoiceGenerator {
    /// Opens and strictly binds the true-mmap VoiceGenerator checkpoint.
    pub fn open_mapped(path: impl AsRef<Path>, backend: BackendKind) -> Result<Self> {
        Self::from_checkpoint(MossVoiceGeneratorCheckpoint::open_mapped(path)?, backend)
    }

    /// Builds the executable model from an authenticated VoiceGenerator map.
    pub fn from_checkpoint(
        checkpoint: MossVoiceGeneratorCheckpoint,
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

    /// Strict VoiceGenerator checkpoint retained by this model.
    pub const fn checkpoint(&self) -> &MossVoiceGeneratorCheckpoint {
        &self.checkpoint
    }

    /// Official VoiceGenerator release defaults. These intentionally differ
    /// from Base/v1.5 audio sampling.
    pub fn default_generation_options(&self) -> MossTtsDelayGenerationOptions {
        MossTtsDelayGenerationOptions::voice_generator()
    }

    /// Runs explicit upstream-compatible `[rows,17]` prompt IDs.
    pub fn forward_prompt_last_logits(&self, prompt_rows: &[u32]) -> Result<MossTtsDelayLogits> {
        forward_prompt_last_logits(self, prompt_rows)
    }

    /// Generates raw delayed VoiceGenerator rows.
    pub fn generate_delay_rows(
        &self,
        prompt_rows: &[u32],
        options: &MossTtsDelayGenerationOptions,
    ) -> Result<MossTtsDelayGeneration> {
        generation::generate_delay_rows(self, prompt_rows, options)
    }

    /// Generates, de-delays and decodes through the exact Full codec.
    pub fn synthesize_prompt_rows(
        &self,
        codec: &MossAudioTokenizer,
        prompt_rows: &[u32],
        options: &MossTtsDelayGenerationOptions,
    ) -> Result<MossTtsDelaySynthesis> {
        synthesize_prompt_rows(self, codec, prompt_rows, options)
    }
}

trait DelayRuntimeAccess {
    fn backend(&self) -> BackendKind;
    fn mapped(&self) -> &DelayMappedDescriptors;
    fn runtime(&self) -> &DelayRuntimeScratch;
}

impl DelayRuntimeAccess for MossTtsDelay {
    fn backend(&self) -> BackendKind {
        self.backend
    }

    fn mapped(&self) -> &DelayMappedDescriptors {
        self.checkpoint.mapped()
    }

    fn runtime(&self) -> &DelayRuntimeScratch {
        &self.runtime
    }
}

impl DelayRuntimeAccess for MossVoiceGenerator {
    fn backend(&self) -> BackendKind {
        self.backend
    }

    fn mapped(&self) -> &DelayMappedDescriptors {
        self.checkpoint.mapped()
    }

    fn runtime(&self) -> &DelayRuntimeScratch {
        &self.runtime
    }
}

fn forward_prompt_last_logits(
    model: &impl DelayRuntimeAccess,
    prompt_rows: &[u32],
) -> Result<MossTtsDelayLogits> {
    let mapped = model.mapped();
    let topology = mapped.topology();
    let columns = topology.input_columns();
    let label = mapped.mapped_model().name;
    if prompt_rows.is_empty() || !prompt_rows.len().is_multiple_of(columns) {
        return Err(VokraError::InvalidArgument(format!(
            "{label}: prompt must be a non-empty [rows,{columns}] u32 matrix, got {} values",
            prompt_rows.len()
        )));
    }
    let rows = prompt_rows.len() / columns;
    if rows > topology.max_position_embeddings {
        return Err(VokraError::InvalidArgument(format!(
            "{label}: prompt rows {rows} exceed max positions {}",
            topology.max_position_embeddings
        )));
    }

    let compute = Compute::for_backend(model.backend(), MOSS_TTS_DELAY_HOT_OPS)?;
    let reserve = rows.clamp(1, 256);
    let mut kv_cache = KvCache::with_reserve(topology.num_layers, topology.kv_dim(), reserve);
    let mut scratch = DelayStepScratch::default();
    for row_start in (0..rows).step_by(PREFILL_CHUNK_ROWS) {
        let chunk_rows = PREFILL_CHUNK_ROWS.min(rows - row_start);
        let start = row_start * columns;
        let end = start + chunk_rows * columns;
        forward_chunk(
            &compute,
            mapped,
            model.runtime(),
            &mut scratch,
            &mut kv_cache,
            &prompt_rows[start..end],
            chunk_rows,
        )?;
    }
    last_logits(&compute, mapped, model.runtime(), &scratch)
}

fn synthesize_prompt_rows(
    model: &impl DelayRuntimeAccess,
    codec: &MossAudioTokenizer,
    prompt_rows: &[u32],
    options: &MossTtsDelayGenerationOptions,
) -> Result<MossTtsDelaySynthesis> {
    let mapped = model.mapped();
    let label = mapped.mapped_model().name;
    let num_audio_codebooks = mapped.topology().num_audio_codebooks;
    if codec.variant() != MossAudioTokenizerVariant::Full {
        return Err(VokraError::UnsupportedOp(format!(
            "{label}: synthesis requires the exact MOSS Audio Tokenizer Full companion; got {:?}",
            codec.variant()
        )));
    }
    if codec.backend() != model.backend() {
        return Err(VokraError::InvalidArgument(format!(
            "{label}: LLM backend {:?} does not match Full codec backend {:?}; the composed graph must select one backend and never hide a CPU fallback",
            model.backend(),
            codec.backend()
        )));
    }

    let generated = generation::generate_delay_rows(model, prompt_rows, options)?;
    let generation::DeDelayedAudio {
        start_length,
        segments: code_segments,
    } = generation::de_delay_audio_segments(prompt_rows, &generated, label)?;
    let mut segments = Vec::with_capacity(code_segments.len());
    for (index, code_segment) in code_segments.into_iter().enumerate() {
        let mut audio = codec.decode_frame_major(
            &code_segment.codes,
            code_segment.frames,
            num_audio_codebooks,
        )?;
        let mut trimmed_prefix_samples = 0;
        if index == 0 && start_length > 0 {
            if start_length >= code_segment.frames {
                continue;
            }
            trimmed_prefix_samples = audio
                .samples_per_channel
                .checked_mul(start_length)
                .ok_or_else(|| {
                    VokraError::InvalidArgument(format!(
                        "{label}: continuation trim sample count overflows"
                    ))
                })?
                / code_segment.frames;
            let interleaved = trimmed_prefix_samples
                .checked_mul(audio.channels)
                .ok_or_else(|| {
                    VokraError::InvalidArgument(format!(
                        "{label}: continuation interleaved trim overflows"
                    ))
                })?;
            audio.pcm.drain(..interleaved);
            audio.samples_per_channel -= trimmed_prefix_samples;
        }
        segments.push(MossTtsDelayAudioSegment {
            codes: code_segment.codes,
            frames: code_segment.frames,
            audio,
            trimmed_prefix_samples,
        });
    }
    Ok(MossTtsDelaySynthesis {
        generated,
        segments,
    })
}

#[derive(Default)]
pub(super) struct DelayRuntimeScratch {
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
pub(super) struct DelayStepScratch {
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

pub(super) fn forward_chunk(
    compute: &Compute,
    mapped: &DelayMappedDescriptors,
    runtime: &DelayRuntimeScratch,
    scratch: &mut DelayStepScratch,
    kv_cache: &mut KvCache,
    prompt: &[u32],
    rows: usize,
) -> Result<()> {
    let topology = mapped.topology();
    let columns = topology.input_columns();
    let mapped_model = mapped.mapped_model();
    let label = mapped_model.name;
    let q_dim = topology.q_dim();
    let kv_dim = topology.kv_dim();
    if rows == 0 || prompt.len() != rows * columns {
        return Err(VokraError::InvalidArgument(format!(
            "{label}: forward chunk shape mismatch: prompt={}, rows={rows}, columns={columns}",
            prompt.len()
        )));
    }
    let position_offset = kv_cache.positions();
    if position_offset + rows > topology.max_position_embeddings {
        return Err(VokraError::InvalidArgument(format!(
            "{label}: decode position {} exceeds max positions {}",
            position_offset + rows,
            topology.max_position_embeddings
        )));
    }

    embed_prompt(mapped, prompt, rows, scratch)?;
    resize_zero(&mut scratch.norm, rows * topology.hidden_dim);
    resize_zero(&mut scratch.q_raw, rows * q_dim);
    resize_zero(&mut scratch.q, rows * q_dim);
    resize_zero(&mut scratch.k_raw, rows * kv_dim);
    resize_zero(&mut scratch.k, rows * kv_dim);
    resize_zero(&mut scratch.v, rows * kv_dim);
    resize_zero(&mut scratch.query, rows * topology.head_dim);
    resize_zero(&mut scratch.attention, rows * q_dim);
    resize_zero(&mut scratch.attention_out, rows * topology.hidden_dim);
    resize_zero(&mut scratch.ffn_gate, rows * topology.ffn_dim);
    resize_zero(&mut scratch.ffn_activated, rows * topology.ffn_dim);
    resize_zero(&mut scratch.ffn_up, rows * topology.ffn_dim);
    resize_zero(&mut scratch.ffn_down, rows * topology.hidden_dim);

    let mut block = lock_scratch(&runtime.block, mapped_model)?;
    for layer in 0..topology.num_layers {
        materialize_layer(mapped, layer, &mut block)?;

        compute.rms_norm_f32(
            &scratch.hidden,
            &mut scratch.norm,
            rows,
            topology.hidden_dim,
            &block.input_norm,
            topology.rms_norm_eps,
        )?;
        compute.gemm_f32(
            rows,
            q_dim,
            topology.hidden_dim,
            &scratch.norm,
            &block.q_w_t,
            None,
            &mut scratch.q_raw,
        )?;
        compute.gemm_f32(
            rows,
            kv_dim,
            topology.hidden_dim,
            &scratch.norm,
            &block.k_w_t,
            None,
            &mut scratch.k_raw,
        )?;
        compute.gemm_f32(
            rows,
            kv_dim,
            topology.hidden_dim,
            &scratch.norm,
            &block.v_w_t,
            None,
            &mut scratch.v,
        )?;
        compute.rms_norm_f32(
            &scratch.q_raw,
            &mut scratch.q,
            rows * topology.num_q_heads,
            topology.head_dim,
            &block.q_norm,
            topology.rms_norm_eps,
        )?;
        compute.rms_norm_f32(
            &scratch.k_raw,
            &mut scratch.k,
            rows * topology.num_kv_heads,
            topology.head_dim,
            &block.k_norm,
            topology.rms_norm_eps,
        )?;
        apply_half_split_rope(
            &mut scratch.q,
            rows,
            topology.num_q_heads,
            topology.head_dim,
            topology.rope_base,
            position_offset,
            label,
        )?;
        apply_half_split_rope(
            &mut scratch.k,
            rows,
            topology.num_kv_heads,
            topology.head_dim,
            topology.rope_base,
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
            topology,
            label,
        )?;
        compute.gemm_f32(
            rows,
            topology.hidden_dim,
            q_dim,
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
            topology.hidden_dim,
            &block.ffn_norm,
            topology.rms_norm_eps,
        )?;
        compute.gemm_f32(
            rows,
            topology.ffn_dim,
            topology.hidden_dim,
            &scratch.norm,
            &block.gate_w_t,
            None,
            &mut scratch.ffn_gate,
        )?;
        compute.gemm_f32(
            rows,
            topology.ffn_dim,
            topology.hidden_dim,
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
            topology.hidden_dim,
            topology.ffn_dim,
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
        topology.hidden_dim,
        &final_norm,
        topology.rms_norm_eps,
    )?;
    reject_non_finite(label, "final hidden", &scratch.norm)
}

fn embed_prompt(
    mapped: &DelayMappedDescriptors,
    prompt: &[u32],
    rows: usize,
    scratch: &mut DelayStepScratch,
) -> Result<()> {
    let topology = mapped.topology();
    let columns = topology.input_columns();
    let label = mapped.mapped_model().name;
    resize_zero(&mut scratch.hidden, rows * topology.hidden_dim);
    for row in 0..rows {
        let row_tokens = &prompt[row * columns..(row + 1) * columns];
        let text = row_tokens[0] as usize;
        if text >= topology.text_vocab_size {
            return Err(VokraError::InvalidArgument(format!(
                "{label}: text token at row {row} is {text}, outside 0..{}",
                topology.text_vocab_size
            )));
        }
        widen_row(
            mapped,
            mapped.text_embedding(),
            text,
            &mut scratch.embed_row,
        )?;
        let hidden =
            &mut scratch.hidden[row * topology.hidden_dim..(row + 1) * topology.hidden_dim];
        hidden.copy_from_slice(&scratch.embed_row);
        for codebook in 0..topology.num_audio_codebooks {
            let token = row_tokens[1 + codebook] as usize;
            if !topology.accepts_audio_token(token) {
                return Err(VokraError::InvalidArgument(format!(
                    "{label}: audio token at row {row}, codebook {codebook} is {token}, outside learned rows 0..{} and authenticated pad id {}",
                    topology.audio_vocab_with_pad, topology.audio_pad_token_id
                )));
            }
            if let Some(embedding_row) = topology.audio_embedding_row(token) {
                widen_row(
                    mapped,
                    mapped.audio_embedding(codebook),
                    embedding_row,
                    &mut scratch.embed_row,
                )?;
                for (value, &embedding) in hidden.iter_mut().zip(&scratch.embed_row) {
                    *value += embedding;
                }
            }
        }
    }
    reject_non_finite(label, "prompt embedding", &scratch.hidden)
}

// Cache geometry and authenticated topology stay explicit at this hot-kernel boundary.
#[allow(clippy::too_many_arguments)]
fn attention(
    compute: &Compute,
    scratch: &mut DelayStepScratch,
    key_cache: &[f32],
    value_cache: &[f32],
    rows: usize,
    position_offset: usize,
    topology: DelayTopology,
    label: &str,
) -> Result<()> {
    let q_dim = topology.q_dim();
    let kv_dim = topology.kv_dim();
    let total_rows = position_offset + rows;
    let expected = total_rows * kv_dim;
    if key_cache.len() != expected || value_cache.len() != expected {
        return Err(VokraError::InvalidArgument(format!(
            "{label}: KV cache length mismatch: key={}, value={}, expected={expected}",
            key_cache.len(),
            value_cache.len()
        )));
    }
    resize_zero(&mut scratch.key_t, topology.head_dim * total_rows);
    resize_zero(&mut scratch.value, total_rows * topology.head_dim);
    resize_zero(&mut scratch.scores, rows * total_rows);
    resize_zero(&mut scratch.probabilities, rows * total_rows);
    resize_zero(&mut scratch.attended, rows * topology.head_dim);
    scratch.attention.fill(0.0);
    let groups = topology.num_q_heads / topology.num_kv_heads;
    let scale = (topology.head_dim as f32).sqrt().recip();

    for kv_head in 0..topology.num_kv_heads {
        for position in 0..total_rows {
            let source = position * kv_dim + kv_head * topology.head_dim;
            for dimension in 0..topology.head_dim {
                scratch.key_t[dimension * total_rows + position] = key_cache[source + dimension];
                scratch.value[position * topology.head_dim + dimension] =
                    value_cache[source + dimension];
            }
        }
        for group in 0..groups {
            let q_head = kv_head * groups + group;
            for row in 0..rows {
                let source = row * q_dim + q_head * topology.head_dim;
                scratch.query[row * topology.head_dim..(row + 1) * topology.head_dim]
                    .copy_from_slice(&scratch.q[source..source + topology.head_dim]);
            }
            compute.gemm_f32(
                rows,
                total_rows,
                topology.head_dim,
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
                topology.head_dim,
                total_rows,
                &scratch.probabilities,
                &scratch.value,
                None,
                &mut scratch.attended,
            )?;
            for row in 0..rows {
                let target = row * q_dim + q_head * topology.head_dim;
                scratch.attention[target..target + topology.head_dim].copy_from_slice(
                    &scratch.attended[row * topology.head_dim..(row + 1) * topology.head_dim],
                );
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
                let frequency = rope_base.powf(-((2 * pair) as f32) / head_dim as f32);
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
    let topology = mapped.topology();
    let q_dim = topology.q_dim();
    let kv_dim = topology.kv_dim();
    let descriptors = mapped.layer(layer);
    widen_tensor(mapped, descriptors.input_norm, &mut block.input_norm)?;
    transpose_tensor(
        mapped,
        descriptors.q,
        q_dim,
        topology.hidden_dim,
        &mut block.q_w_t,
    )?;
    widen_tensor(mapped, descriptors.q_norm, &mut block.q_norm)?;
    transpose_tensor(
        mapped,
        descriptors.k,
        kv_dim,
        topology.hidden_dim,
        &mut block.k_w_t,
    )?;
    widen_tensor(mapped, descriptors.k_norm, &mut block.k_norm)?;
    transpose_tensor(
        mapped,
        descriptors.v,
        kv_dim,
        topology.hidden_dim,
        &mut block.v_w_t,
    )?;
    transpose_tensor(
        mapped,
        descriptors.o,
        topology.hidden_dim,
        q_dim,
        &mut block.o_w_t,
    )?;
    widen_tensor(mapped, descriptors.ffn_norm, &mut block.ffn_norm)?;
    transpose_tensor(
        mapped,
        descriptors.gate,
        topology.ffn_dim,
        topology.hidden_dim,
        &mut block.gate_w_t,
    )?;
    transpose_tensor(
        mapped,
        descriptors.up,
        topology.ffn_dim,
        topology.hidden_dim,
        &mut block.up_w_t,
    )?;
    transpose_tensor(
        mapped,
        descriptors.down,
        topology.hidden_dim,
        topology.ffn_dim,
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
    let topology = mapped.topology();
    let label = mapped.mapped_model().name;
    if scratch.norm.len() < topology.hidden_dim {
        return Err(VokraError::InvalidArgument(format!(
            "{label}: no final hidden row is available"
        )));
    }
    let hidden = &scratch.norm[scratch.norm.len() - topology.hidden_dim..];
    let mut chunk = lock_scratch(&runtime.head_chunk, mapped.mapped_model())?;
    let mut text_logits = vec![0.0; topology.text_vocab_size];
    project_head(
        compute,
        mapped,
        mapped.head(0),
        topology.text_vocab_size,
        hidden,
        &mut chunk,
        &mut text_logits,
    )?;
    let mut audio_logits = vec![0.0; topology.num_audio_codebooks * topology.audio_vocab_with_pad];
    for codebook in 0..topology.num_audio_codebooks {
        let start = codebook * topology.audio_vocab_with_pad;
        project_head(
            compute,
            mapped,
            mapped.head(1 + codebook),
            topology.audio_vocab_with_pad,
            hidden,
            &mut chunk,
            &mut audio_logits[start..start + topology.audio_vocab_with_pad],
        )?;
    }
    reject_non_finite(label, "text logits", &text_logits)?;
    reject_non_finite(label, "audio logits", &audio_logits)?;
    Ok(MossTtsDelayLogits {
        text_logits,
        audio_logits,
        num_audio_codebooks: topology.num_audio_codebooks,
        audio_vocab_with_pad: topology.audio_vocab_with_pad,
        model_label: label,
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn project_head(
    compute: &Compute,
    mapped: &DelayMappedDescriptors,
    info: &GgufTensorInfo,
    rows: usize,
    hidden: &[f32],
    chunk: &mut Vec<f32>,
    output: &mut [f32],
) -> Result<()> {
    let topology = mapped.topology();
    let label = mapped.mapped_model().name;
    if hidden.len() != topology.hidden_dim || output.len() != rows {
        return Err(VokraError::InvalidArgument(format!(
            "{label}: head projection shape mismatch: hidden={}, rows={rows}, output={} ",
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
            topology.hidden_dim,
            chunk,
            hidden,
            None,
            &mut output[row..row + chunk_rows],
        )?;
        row += chunk_rows;
    }
    Ok(())
}

pub(super) fn widen_row(
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
    let topology = mapped.topology();
    let label = mapped.mapped_model().name;
    let element_size = info.dtype.type_size();
    let bytes = mapped.file().tensor_bytes(info);
    let start = first_row
        .checked_mul(topology.hidden_dim)
        .and_then(|value| value.checked_mul(element_size))
        .ok_or_else(|| VokraError::InvalidArgument(format!("{label}: row offset overflow")))?;
    let len = rows
        .checked_mul(topology.hidden_dim)
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

pub(super) fn widen_tensor(
    mapped: &DelayMappedDescriptors,
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
        mapped.mapped_model(),
    )
}

pub(super) fn last_hidden<'a>(
    mapped: &DelayMappedDescriptors,
    scratch: &'a DelayStepScratch,
) -> Result<&'a [f32]> {
    let topology = mapped.topology();
    let label = mapped.mapped_model().name;
    if scratch.norm.len() < topology.hidden_dim {
        return Err(VokraError::InvalidArgument(format!(
            "{label}: no final hidden row is available"
        )));
    }
    Ok(&scratch.norm[scratch.norm.len() - topology.hidden_dim..])
}

pub(super) fn reject_non_finite(
    model_label: &str,
    value_label: &str,
    values: &[f32],
) -> Result<()> {
    if let Some((index, value)) = values
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(VokraError::InvalidArgument(format!(
            "{model_label}: {value_label} contains non-finite value {value} at index {index}"
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
        let head_dim = 128;
        let mut values = (0..head_dim).map(|value| value as f32).collect::<Vec<_>>();
        let expected = values.clone();
        apply_half_split_rope(&mut values, 1, 1, head_dim, 1_000_000.0, 0, LABEL).unwrap();
        assert_eq!(values, expected);
    }

    #[test]
    fn causal_mask_respects_cached_prefix() {
        let mut scores = vec![2.0; 6];
        scale_and_mask(&mut scores, 2, 3, 1, 0.5, LABEL).unwrap();
        assert_eq!(&scores[..3], &[1.0, 1.0, f32::MIN]);
        assert_eq!(&scores[3..], &[1.0, 1.0, 1.0]);
    }
}
