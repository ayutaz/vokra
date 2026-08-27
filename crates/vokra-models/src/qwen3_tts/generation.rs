//! Bounded-memory native Qwen3-TTS talker prefill and cached decode.
//!
//! Dense tensors remain mmap-backed. One layer is widened into a reused
//! scratch block, and every learned projection/reduction/activation runs on
//! one preflighted [`Compute`] backend. Embedding construction and the
//! 15-row code predictor are layered on this raw talker seam.

use std::path::Path;
use std::sync::{Arc, Mutex};

use vokra_core::backend::BackendKind;
use vokra_core::gguf::GgufTensorInfo;
use vokra_core::{KvCache, Result, Sampler, SamplerConfig, VokraError};

use crate::compute::{Compute, HotOp};
use crate::mapped_weights::{lock_scratch, transpose_widen, widen_into};

use super::weights::{DecoderLayerDescriptors, Qwen3TtsMappedDescriptors};
use super::{
    CODEC_BOS_TOKEN_ID, CODEC_EOS_TOKEN_ID, CODEC_NOTHINK_TOKEN_ID, CODEC_PAD_TOKEN_ID,
    CODEC_THINK_BOS_TOKEN_ID, CODEC_THINK_EOS_TOKEN_ID, CODEC_THINK_TOKEN_ID,
    QWEN3_TTS_NUM_CODE_GROUPS, Qwen3TtsCheckpoint, Qwen3TtsCheckpointVariant,
    Qwen3TtsCodePredictorConfig, Qwen3TtsTalkerConfig, Qwen3TtsTokenizer, TTS_BOS_TOKEN_ID,
    TTS_EOS_TOKEN_ID, TTS_PAD_TOKEN_ID,
};

const LABEL: &str = "qwen3_tts";
const PREFILL_CHUNK_ROWS: usize = 8;
const HEAD_CHUNK_ROWS: usize = 512;
const OFFICIAL_MAX_NEW_TOKENS: usize = 8_192;
const OFFICIAL_MIN_NEW_TOKENS: usize = 2;

/// Official high-level generation controls for every released Qwen3-TTS
/// checkpoint variant.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct Qwen3TtsGenerationOptions {
    /// Language name accepted by the embedded fixed-revision tokenizer.
    pub language: String,
    /// Fixed CustomVoice speaker. Required only by CustomVoice variants.
    pub speaker: Option<String>,
    /// Natural-language voice/style instruction. Supported by VoiceDesign and
    /// 1.7B CustomVoice; the 0.6B CustomVoice release rejects it explicitly.
    pub instruction: Option<String>,
    /// Precomputed Base speaker x-vector in talker-hidden width. Base variants
    /// require this until the separately authenticated reference-audio speaker
    /// encoder frontend is joined.
    pub speaker_embedding: Option<Vec<f32>>,
    /// `None` selects the official wrapper default: streaming-simulation for
    /// Base and non-streaming text prefill for CustomVoice/VoiceDesign.
    pub non_streaming_mode: Option<bool>,
    /// Maximum first-codebook tokens, including terminal EOS when emitted.
    pub max_new_tokens: usize,
    /// Minimum tokens before first-codebook EOS may be sampled.
    pub min_new_tokens: usize,
    /// First-codebook sampling temperature; zero selects greedy generation.
    pub temperature: f32,
    /// First-codebook top-k candidate count.
    pub top_k: Option<usize>,
    /// First-codebook nucleus threshold. `None` is equivalent to official
    /// `top_p=1.0` without an unnecessary filtering pass.
    pub top_p: Option<f32>,
    /// First-codebook CTRL repetition penalty.
    pub repetition_penalty: Option<f32>,
    /// Code-predictor sampling temperature.
    pub predictor_temperature: f32,
    /// Code-predictor top-k candidate count.
    pub predictor_top_k: Option<usize>,
    /// Code-predictor nucleus threshold; `None` matches official `top_p=1.0`.
    pub predictor_top_p: Option<f32>,
    /// Deterministic Vokra seed for first-codebook sampling.
    pub seed: u64,
    /// Deterministic Vokra seed for the nested code predictor.
    pub predictor_seed: u64,
}

impl Default for Qwen3TtsGenerationOptions {
    fn default() -> Self {
        Self {
            language: "Auto".to_owned(),
            speaker: None,
            instruction: None,
            speaker_embedding: None,
            non_streaming_mode: None,
            max_new_tokens: OFFICIAL_MAX_NEW_TOKENS,
            min_new_tokens: OFFICIAL_MIN_NEW_TOKENS,
            temperature: 0.9,
            top_k: Some(50),
            top_p: None,
            repetition_penalty: Some(1.05),
            predictor_temperature: 0.9,
            predictor_top_k: Some(50),
            predictor_top_p: None,
            seed: 0,
            predictor_seed: 1,
        }
    }
}

impl Qwen3TtsGenerationOptions {
    /// Deterministic configuration for numerical and backend parity runs.
    #[must_use]
    pub fn greedy(max_new_tokens: usize) -> Self {
        Self {
            max_new_tokens,
            min_new_tokens: 0,
            temperature: 0.0,
            top_k: None,
            top_p: None,
            repetition_penalty: None,
            predictor_temperature: 0.0,
            predictor_top_k: None,
            predictor_top_p: None,
            ..Self::default()
        }
    }

    fn talker_sampler(&self) -> SamplerConfig {
        SamplerConfig {
            temperature: self.temperature,
            top_k: self.top_k,
            top_p: self.top_p,
            repetition_penalty: self.repetition_penalty,
            seed: self.seed,
        }
    }

    fn predictor_sampler(&self) -> SamplerConfig {
        SamplerConfig {
            temperature: self.predictor_temperature,
            top_k: self.predictor_top_k,
            top_p: self.predictor_top_p,
            repetition_penalty: None,
            seed: self.predictor_seed,
        }
    }
}

/// Generated Qwen3-TTS codes in compact frame-major `[frames,16]` order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Qwen3TtsGeneratedCodes {
    frame_major: Vec<u32>,
    frames: usize,
    ended: bool,
}

impl Qwen3TtsGeneratedCodes {
    /// Number of complete 12-Hz frames, excluding terminal EOS.
    #[must_use]
    pub const fn frames(&self) -> usize {
        self.frames
    }

    /// Every frame has exactly sixteen codec rows.
    #[must_use]
    pub const fn num_codebooks(&self) -> usize {
        QWEN3_TTS_NUM_CODE_GROUPS as usize
    }

    /// Whether first-codebook EOS was observed before the configured cap.
    #[must_use]
    pub const fn ended(&self) -> bool {
        self.ended
    }

    /// Frame-major `[frames,16]` codes, all valid for the 12-Hz decoder.
    #[must_use]
    pub fn as_frame_major(&self) -> &[u32] {
        &self.frame_major
    }

    /// Converts to the codebook-major matrix accepted by the waveform decoder.
    #[must_use]
    pub fn to_codebook_rows(&self) -> Vec<Vec<u32>> {
        let groups = QWEN3_TTS_NUM_CODE_GROUPS as usize;
        let mut rows = (0..groups)
            .map(|_| Vec::with_capacity(self.frames))
            .collect::<Vec<_>>();
        for frame in self.frame_major.chunks_exact(groups) {
            for (row, &code) in rows.iter_mut().zip(frame) {
                row.push(code);
            }
        }
        rows
    }
}

/// Complete learned-op contract for the main talker and code predictor.
pub const QWEN3_TTS_MAIN_HOT_OPS: &[HotOp] = &[
    HotOp::Gemm,
    HotOp::Gemv,
    HotOp::Softmax,
    HotOp::RmsNorm,
    HotOp::Silu,
];

/// Executable mmap-backed Qwen3-TTS main checkpoint on one CPU or Metal
/// backend.
#[derive(Clone)]
pub struct Qwen3TtsMain {
    checkpoint: Qwen3TtsCheckpoint,
    backend: BackendKind,
    runtime: Arc<Qwen3TtsRuntime>,
    tokenizer: Arc<Qwen3TtsTokenizer>,
}

impl std::fmt::Debug for Qwen3TtsMain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Qwen3TtsMain")
            .field("variant", &self.checkpoint.variant())
            .field("backend", &self.backend)
            .finish_non_exhaustive()
    }
}

impl Qwen3TtsMain {
    /// Opens and strictly binds the main GGUF through mmap, authenticates the
    /// embedded tokenizer assets and preflights the complete learned graph.
    pub fn open_mapped(path: impl AsRef<Path>, backend: BackendKind) -> Result<Self> {
        Self::from_checkpoint(Qwen3TtsCheckpoint::open_mapped(path)?, backend)
    }

    /// Builds an executable model from an authenticated mapped checkpoint.
    pub fn from_checkpoint(checkpoint: Qwen3TtsCheckpoint, backend: BackendKind) -> Result<Self> {
        checkpoint.mapped()?;
        let tokenizer = Arc::clone(checkpoint.embedded_tokenizer()?);
        let _ = Compute::for_backend(backend, QWEN3_TTS_MAIN_HOT_OPS)?;
        Ok(Self {
            checkpoint,
            backend,
            runtime: Arc::new(Qwen3TtsRuntime::default()),
            tokenizer,
        })
    }

    /// Explicit backend used by every learned operation.
    #[must_use]
    pub const fn backend(&self) -> BackendKind {
        self.backend
    }

    /// Strict mapped checkpoint retained by this model.
    #[must_use]
    pub const fn checkpoint(&self) -> &Qwen3TtsCheckpoint {
        &self.checkpoint
    }

    /// Authenticated fixed-revision tokenizer and prompt builder.
    #[must_use]
    pub fn tokenizer(&self) -> &Qwen3TtsTokenizer {
        &self.tokenizer
    }

    /// Starts a raw talker session from caller-built `[rows, hidden]`
    /// embeddings. This is the independently testable boundary below the
    /// official text/language/speaker prompt constructor.
    pub fn start_talker_session(
        &self,
        prompt_embeddings: &[f32],
    ) -> Result<Qwen3TtsTalkerSession<'_>> {
        Qwen3TtsTalkerSession::start(self, prompt_embeddings)
    }

    /// Tokenizes `text`, builds the exact variant-specific official prompt and
    /// generates complete sixteen-row 12-Hz codec frames.
    pub fn generate_codes(
        &self,
        text: &str,
        options: &Qwen3TtsGenerationOptions,
    ) -> Result<Qwen3TtsGeneratedCodes> {
        generate_codes(self, text, options)
    }

    /// Generates the fifteen code-predictor rows for one sampled first-codebook
    /// token. The returned frame is ordered exactly as the companion 12-Hz
    /// waveform decoder expects: talker row first, predictor rows 1 through 15
    /// after it.
    ///
    /// The predictor KV cache is intentionally fresh for every frame, matching
    /// the official nested `code_predictor.generate` call. `sampler` remains
    /// caller-owned so its RNG can advance continuously across frames.
    pub fn predict_code_frame(
        &self,
        talker_hidden: &[f32],
        first_code: u32,
        sampler: &mut Sampler,
    ) -> Result<[u32; QWEN3_TTS_NUM_CODE_GROUPS as usize]> {
        predict_code_frame(self, talker_hidden, first_code, sampler)
    }
}

/// Last-row normalized hidden state and first-codebook logits.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct Qwen3TtsTalkerOutput {
    /// Final-normalized talker hidden row, used as the code-predictor prefix.
    pub hidden: Vec<f32>,
    /// First-codebook logits, exactly `talker.vocab_size` values.
    pub logits: Vec<f32>,
}

/// Stateful KV-cached talker session.
///
/// A session owns the selected [`Compute`] context. Metal therefore remains
/// device-affine and is never hidden inside the `Send + Sync` model handle.
pub struct Qwen3TtsTalkerSession<'a> {
    model: &'a Qwen3TtsMain,
    compute: Compute,
    kv_cache: KvCache,
    scratch: DecoderStepScratch,
    output: Qwen3TtsTalkerOutput,
}

impl std::fmt::Debug for Qwen3TtsTalkerSession<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Qwen3TtsTalkerSession")
            .field("backend", &self.model.backend)
            .field("positions", &self.kv_cache.positions())
            .finish_non_exhaustive()
    }
}

impl<'a> Qwen3TtsTalkerSession<'a> {
    fn start(model: &'a Qwen3TtsMain, prompt_embeddings: &[f32]) -> Result<Self> {
        let mapped = model.checkpoint.mapped()?;
        let config = &mapped.config().talker;
        let hidden = config.hidden_dim as usize;
        if prompt_embeddings.is_empty() || !prompt_embeddings.len().is_multiple_of(hidden) {
            return Err(VokraError::InvalidArgument(format!(
                "{LABEL}: prompt embeddings must be a non-empty [rows,{hidden}] matrix, got {} values",
                prompt_embeddings.len()
            )));
        }
        reject_non_finite(LABEL, "prompt embeddings", prompt_embeddings)?;
        let rows = prompt_embeddings.len() / hidden;
        if rows > config.max_position_embeddings as usize {
            return Err(VokraError::InvalidArgument(format!(
                "{LABEL}: prompt rows {rows} exceed talker max positions {}",
                config.max_position_embeddings
            )));
        }
        let compute = Compute::for_backend(model.backend, QWEN3_TTS_MAIN_HOT_OPS)?;
        let reserve = rows.saturating_add(64).min(512).max(1);
        let mut kv_cache =
            KvCache::with_reserve(config.n_layer as usize, kv_width(config), reserve);
        let mut scratch = DecoderStepScratch::default();
        for row_start in (0..rows).step_by(PREFILL_CHUNK_ROWS) {
            let chunk_rows = PREFILL_CHUNK_ROWS.min(rows - row_start);
            let start = row_start * hidden;
            let end = start + chunk_rows * hidden;
            forward_talker_chunk(
                &compute,
                mapped,
                &model.runtime,
                &mut scratch,
                &mut kv_cache,
                &prompt_embeddings[start..end],
                chunk_rows,
            )?;
        }
        let output = last_talker_output(&compute, mapped, &model.runtime, &scratch)?;
        Ok(Self {
            model,
            compute,
            kv_cache,
            scratch,
            output,
        })
    }

    /// Current cached position count, including the prefill.
    #[must_use]
    pub fn positions(&self) -> usize {
        self.kv_cache.positions()
    }

    /// Current last-row normalized hidden state and first-codebook logits.
    #[must_use]
    pub const fn output(&self) -> &Qwen3TtsTalkerOutput {
        &self.output
    }

    /// Appends one already-composed talker embedding and executes one cached
    /// autoregressive step on the same backend.
    pub fn decode_embedding(&mut self, embedding: &[f32]) -> Result<&Qwen3TtsTalkerOutput> {
        let mapped = self.model.checkpoint.mapped()?;
        let hidden = mapped.config().talker.hidden_dim as usize;
        if embedding.len() != hidden {
            return Err(VokraError::InvalidArgument(format!(
                "{LABEL}: decode embedding has {} values, expected hidden {hidden}",
                embedding.len()
            )));
        }
        reject_non_finite(LABEL, "decode embedding", embedding)?;
        forward_talker_chunk(
            &self.compute,
            mapped,
            &self.model.runtime,
            &mut self.scratch,
            &mut self.kv_cache,
            embedding,
            1,
        )?;
        self.output =
            last_talker_output(&self.compute, mapped, &self.model.runtime, &self.scratch)?;
        Ok(&self.output)
    }
}

#[derive(Default)]
struct Qwen3TtsRuntime {
    block: Mutex<DecoderBlock>,
    head: Mutex<HeadScratch>,
    text_projection: Mutex<TextProjectionScratch>,
    predictor_block: Mutex<DecoderBlock>,
    predictor_head: Mutex<HeadScratch>,
    predictor_projection: Mutex<ProjectionScratch>,
}

#[derive(Default)]
struct DecoderBlock {
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
struct ProjectionScratch {
    weights_t: Vec<f32>,
    bias: Vec<f32>,
    output: Vec<f32>,
}

#[derive(Default)]
struct TextProjectionScratch {
    embedding: Vec<f32>,
    embedding_row: Vec<f32>,
    fc1_weights_t: Vec<f32>,
    fc1_bias: Vec<f32>,
    fc1_hidden: Vec<f32>,
    activated: Vec<f32>,
    fc2_weights_t: Vec<f32>,
    fc2_bias: Vec<f32>,
    output: Vec<f32>,
}

#[derive(Default)]
struct DecoderStepScratch {
    hidden: Vec<f32>,
    norm: Vec<f32>,
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

struct PromptEmbeddings {
    prompt: Vec<f32>,
    trailing_text: Vec<f32>,
    tts_pad: Vec<f32>,
}

fn generate_codes(
    model: &Qwen3TtsMain,
    text: &str,
    options: &Qwen3TtsGenerationOptions,
) -> Result<Qwen3TtsGeneratedCodes> {
    validate_generation_options(model, options)?;
    let prompt = build_prompt_embeddings(model, text, options)?;
    let mapped = model.checkpoint.mapped()?;
    let hidden = mapped.config().talker.hidden_dim as usize;
    let prompt_rows = prompt.prompt.len() / hidden;
    let max_positions = mapped.config().talker.max_position_embeddings as usize;
    if prompt_rows
        .checked_add(options.max_new_tokens.saturating_sub(1))
        .is_none_or(|rows| rows > max_positions)
    {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: prompt rows {prompt_rows} plus up to {} cached decode rows exceed talker max positions {max_positions}",
            options.max_new_tokens.saturating_sub(1)
        )));
    }

    let mut session = model.start_talker_session(&prompt.prompt)?;
    let mut talker_sampler = Sampler::new(options.talker_sampler());
    let mut predictor_sampler = Sampler::new(options.predictor_sampler());
    let mut frame_major = Vec::with_capacity(
        options
            .max_new_tokens
            .saturating_mul(QWEN3_TTS_NUM_CODE_GROUPS as usize),
    );
    let mut ended = false;
    for generated in 0..options.max_new_tokens {
        let mut logits = session.output().logits.clone();
        suppress_talker_control_logits(&mut logits, generated, options.min_new_tokens)?;
        let first_code = talker_sampler.sample(&mut logits);
        if first_code == CODEC_EOS_TOKEN_ID {
            ended = true;
            break;
        }
        let frame = model.predict_code_frame(
            &session.output().hidden,
            first_code,
            &mut predictor_sampler,
        )?;
        frame_major.extend_from_slice(&frame);
        if generated + 1 == options.max_new_tokens {
            break;
        }
        let embedding = compose_next_talker_embedding(
            model,
            &frame,
            &prompt.trailing_text,
            &prompt.tts_pad,
            generated,
        )?;
        session.decode_embedding(&embedding)?;
    }
    Ok(Qwen3TtsGeneratedCodes {
        frames: frame_major.len() / QWEN3_TTS_NUM_CODE_GROUPS as usize,
        frame_major,
        ended,
    })
}

fn validate_generation_options(
    model: &Qwen3TtsMain,
    options: &Qwen3TtsGenerationOptions,
) -> Result<()> {
    if options.max_new_tokens == 0 {
        return Err(VokraError::InvalidArgument(
            "qwen3_tts: max_new_tokens must be positive".to_owned(),
        ));
    }
    if options.min_new_tokens > options.max_new_tokens {
        return Err(VokraError::InvalidArgument(format!(
            "qwen3_tts: min_new_tokens {} exceeds max_new_tokens {}",
            options.min_new_tokens, options.max_new_tokens
        )));
    }
    let mapped = model.checkpoint.mapped()?;
    validate_sampler_options(
        "talker",
        options.temperature,
        options.top_k,
        options.top_p,
        options.repetition_penalty,
        mapped.config().talker.vocab_size as usize,
    )?;
    validate_sampler_options(
        "code predictor",
        options.predictor_temperature,
        options.predictor_top_k,
        options.predictor_top_p,
        None,
        mapped.config().code_predictor.vocab_size as usize,
    )?;
    model.tokenizer.language_id(&options.language)?;

    let has_speaker = options
        .speaker
        .as_deref()
        .is_some_and(|speaker| !speaker.trim().is_empty());
    let has_instruction = options
        .instruction
        .as_deref()
        .is_some_and(|instruction| !instruction.is_empty());
    match model.checkpoint.variant() {
        Qwen3TtsCheckpointVariant::Base0_6B | Qwen3TtsCheckpointVariant::Base1_7B => {
            if has_speaker {
                return Err(VokraError::InvalidArgument(
                    "qwen3_tts: Base variants take a speaker_embedding, not a fixed CustomVoice speaker name"
                        .to_owned(),
                ));
            }
            if has_instruction {
                return Err(VokraError::InvalidArgument(
                    "qwen3_tts: Base voice-clone generation does not accept an instruction"
                        .to_owned(),
                ));
            }
            let embedding = options.speaker_embedding.as_deref().ok_or_else(|| {
                VokraError::InvalidArgument(
                    "qwen3_tts: Base generation requires a precomputed speaker_embedding; reference-audio speaker-encoder execution is a separate explicit frontend"
                        .to_owned(),
                )
            })?;
            let expected = mapped.config().talker.hidden_dim as usize;
            if embedding.len() != expected {
                return Err(VokraError::InvalidArgument(format!(
                    "qwen3_tts: Base speaker_embedding has {} values, expected talker hidden width {expected}",
                    embedding.len()
                )));
            }
            reject_non_finite(LABEL, "Base speaker embedding", embedding)?;
        }
        Qwen3TtsCheckpointVariant::CustomVoice0_6B | Qwen3TtsCheckpointVariant::CustomVoice1_7B => {
            let speaker = options.speaker.as_deref().ok_or_else(|| {
                VokraError::InvalidArgument(
                    "qwen3_tts: CustomVoice generation requires a fixed speaker name".to_owned(),
                )
            })?;
            if speaker.trim().is_empty() {
                return Err(VokraError::InvalidArgument(
                    "qwen3_tts: CustomVoice speaker name must not be empty".to_owned(),
                ));
            }
            model.tokenizer.speaker_id(speaker)?;
            if options.speaker_embedding.is_some() {
                return Err(VokraError::InvalidArgument(
                    "qwen3_tts: CustomVoice uses its fixed speaker table and rejects speaker_embedding"
                        .to_owned(),
                ));
            }
            if model.checkpoint.variant() == Qwen3TtsCheckpointVariant::CustomVoice0_6B
                && has_instruction
            {
                return Err(VokraError::InvalidArgument(
                    "qwen3_tts: the 0.6B CustomVoice release does not support instruction control"
                        .to_owned(),
                ));
            }
        }
        Qwen3TtsCheckpointVariant::VoiceDesign1_7B => {
            if has_speaker || options.speaker_embedding.is_some() {
                return Err(VokraError::InvalidArgument(
                    "qwen3_tts: VoiceDesign creates a voice from instruction text and rejects speaker/speaker_embedding"
                        .to_owned(),
                ));
            }
        }
    }
    Ok(())
}

fn validate_sampler_options(
    kind: &str,
    temperature: f32,
    top_k: Option<usize>,
    top_p: Option<f32>,
    repetition_penalty: Option<f32>,
    vocab: usize,
) -> Result<()> {
    if !temperature.is_finite() || temperature < 0.0 {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: {kind} temperature must be finite and non-negative, got {temperature}"
        )));
    }
    if top_k.is_some_and(|top_k| top_k == 0 || top_k > vocab) {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: {kind} top_k must be in 1..={vocab} when present"
        )));
    }
    if top_p.is_some_and(|top_p| !top_p.is_finite() || !(0.0 < top_p && top_p <= 1.0)) {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: {kind} top_p must be finite and in (0,1] when present"
        )));
    }
    if repetition_penalty.is_some_and(|penalty| !penalty.is_finite() || penalty <= 0.0) {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: {kind} repetition_penalty must be finite and positive when present"
        )));
    }
    Ok(())
}

fn build_prompt_embeddings(
    model: &Qwen3TtsMain,
    text: &str,
    options: &Qwen3TtsGenerationOptions,
) -> Result<PromptEmbeddings> {
    let mapped = model.checkpoint.mapped()?;
    let hidden = mapped.config().talker.hidden_dim as usize;
    let input_ids = model.tokenizer.assistant_ids(text)?;
    if input_ids.len() < 9 {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: assistant wrapper produced {} ids, expected at least 9 for role/text/suffix slicing",
            input_ids.len()
        )));
    }
    let text_end = input_ids.len() - 5;
    if text_end <= 3 {
        return Err(VokraError::InvalidArgument(
            "qwen3_tts: assistant wrapper contains no TTS text tokens".to_owned(),
        ));
    }

    let special = project_text_ids(
        model,
        &[TTS_BOS_TOKEN_ID, TTS_EOS_TOKEN_ID, TTS_PAD_TOKEN_ID],
    )?;
    let tts_bos = special[..hidden].to_vec();
    let tts_eos = special[hidden..2 * hidden].to_vec();
    let tts_pad = special[2 * hidden..].to_vec();

    let speaker_name = options
        .speaker
        .as_deref()
        .filter(|value| !value.trim().is_empty());
    let language_id = model
        .tokenizer
        .language_id_for_speaker(&options.language, speaker_name)?;
    let mut codec_prefix_ids = if let Some(language_id) = language_id {
        vec![
            CODEC_THINK_TOKEN_ID,
            CODEC_THINK_BOS_TOKEN_ID,
            language_id,
            CODEC_THINK_EOS_TOKEN_ID,
        ]
    } else {
        vec![
            CODEC_NOTHINK_TOKEN_ID,
            CODEC_THINK_BOS_TOKEN_ID,
            CODEC_THINK_EOS_TOKEN_ID,
        ]
    };
    codec_prefix_ids.extend([CODEC_PAD_TOKEN_ID, CODEC_BOS_TOKEN_ID]);
    let mut codec = codec_embedding_ids(model, &codec_prefix_ids)?;
    if let Some(speaker_row) = variant_speaker_embedding(model, options)? {
        let insertion_row = codec_prefix_ids.len() - 2;
        codec.splice(insertion_row * hidden..insertion_row * hidden, speaker_row);
    }
    let codec_rows = codec.len() / hidden;
    if codec_rows < 3 {
        return Err(VokraError::ModelLoad(
            "qwen3_tts: codec prompt must contain markers plus pad/BOS".to_owned(),
        ));
    }

    let mut aligned_text = Vec::with_capacity((codec_rows - 1) * hidden);
    for _ in 0..codec_rows - 2 {
        aligned_text.extend_from_slice(&tts_pad);
    }
    aligned_text.extend_from_slice(&tts_bos);
    let aligned_codec = &codec[..(codec_rows - 1) * hidden];
    add_in_place(&mut aligned_text, aligned_codec);

    let mut prompt = Vec::new();
    if let Some(instruction) = options
        .instruction
        .as_deref()
        .filter(|instruction| !instruction.is_empty())
    {
        let instruction_ids = model.tokenizer.instruction_ids(instruction)?;
        prompt.extend(project_text_ids(model, &instruction_ids)?);
    }
    prompt.extend(project_text_ids(model, &input_ids[..3])?);
    prompt.extend(aligned_text);

    let non_streaming = options.non_streaming_mode.unwrap_or(!matches!(
        model.checkpoint.variant(),
        Qwen3TtsCheckpointVariant::Base0_6B | Qwen3TtsCheckpointVariant::Base1_7B
    ));
    let trailing_text;
    if non_streaming {
        let mut text_rows = project_text_ids(model, &input_ids[3..text_end])?;
        text_rows.extend_from_slice(&tts_eos);
        for row in text_rows.chunks_exact_mut(hidden) {
            add_in_place(row, &codec_embedding_id(model, CODEC_PAD_TOKEN_ID)?);
        }
        prompt.extend(text_rows);
        let mut final_row = tts_pad.clone();
        add_in_place(
            &mut final_row,
            &codec_embedding_id(model, CODEC_BOS_TOKEN_ID)?,
        );
        prompt.extend(final_row);
        trailing_text = tts_pad.clone();
    } else {
        let mut first_text = project_text_ids(model, &input_ids[3..4])?;
        add_in_place(&mut first_text, &codec[(codec_rows - 1) * hidden..]);
        prompt.extend(first_text);
        trailing_text = {
            let mut rows = project_text_ids(model, &input_ids[4..text_end])?;
            rows.extend_from_slice(&tts_eos);
            rows
        };
    }
    reject_non_finite(LABEL, "official prompt embeddings", &prompt)?;
    Ok(PromptEmbeddings {
        prompt,
        trailing_text,
        tts_pad,
    })
}

fn variant_speaker_embedding(
    model: &Qwen3TtsMain,
    options: &Qwen3TtsGenerationOptions,
) -> Result<Option<Vec<f32>>> {
    match model.checkpoint.variant() {
        Qwen3TtsCheckpointVariant::Base0_6B | Qwen3TtsCheckpointVariant::Base1_7B => {
            Ok(options.speaker_embedding.clone())
        }
        Qwen3TtsCheckpointVariant::CustomVoice0_6B | Qwen3TtsCheckpointVariant::CustomVoice1_7B => {
            options
                .speaker
                .as_deref()
                .map(|speaker| model.tokenizer.speaker_id(speaker))
                .transpose()?
                .map(|id| codec_embedding_id(model, id))
                .transpose()
        }
        Qwen3TtsCheckpointVariant::VoiceDesign1_7B => Ok(None),
    }
}

fn project_text_ids(model: &Qwen3TtsMain, ids: &[u32]) -> Result<Vec<f32>> {
    if ids.is_empty() {
        return Ok(Vec::new());
    }
    let mapped = model.checkpoint.mapped()?;
    let config = &mapped.config().talker;
    let text_hidden = config.text_hidden_size as usize;
    let hidden = config.hidden_dim as usize;
    let compute = Compute::for_backend(model.backend, QWEN3_TTS_MAIN_HOT_OPS)?;
    let mut scratch = lock_scratch(&model.runtime.text_projection, mapped.mapped_model())?;
    scratch.embedding.clear();
    for &id in ids {
        widen_embedding_row(
            mapped,
            mapped.text_embedding(),
            text_hidden,
            id,
            config.text_vocab_size,
            "text embedding",
            &mut scratch.embedding_row,
        )?;
        let row = scratch.embedding_row.clone();
        scratch.embedding.extend_from_slice(&row);
    }
    if scratch.fc1_weights_t.is_empty() {
        transpose_tensor(
            mapped,
            mapped.text_projection_fc1_weight(),
            text_hidden,
            text_hidden,
            &mut scratch.fc1_weights_t,
        )?;
        widen_tensor(
            mapped,
            mapped.text_projection_fc1_bias(),
            &mut scratch.fc1_bias,
        )?;
        transpose_tensor(
            mapped,
            mapped.text_projection_fc2_weight(),
            hidden,
            text_hidden,
            &mut scratch.fc2_weights_t,
        )?;
        widen_tensor(
            mapped,
            mapped.text_projection_fc2_bias(),
            &mut scratch.fc2_bias,
        )?;
    }
    let rows = ids.len();
    let TextProjectionScratch {
        embedding,
        fc1_weights_t,
        fc1_bias,
        fc1_hidden,
        activated,
        fc2_weights_t,
        fc2_bias,
        output,
        ..
    } = &mut *scratch;
    resize_zero(fc1_hidden, rows * text_hidden);
    compute.gemm_f32(
        rows,
        text_hidden,
        text_hidden,
        embedding,
        fc1_weights_t,
        Some(fc1_bias),
        fc1_hidden,
    )?;
    compute.silu_f32(fc1_hidden, activated)?;
    resize_zero(output, rows * hidden);
    compute.gemm_f32(
        rows,
        hidden,
        text_hidden,
        activated,
        fc2_weights_t,
        Some(fc2_bias),
        output,
    )?;
    reject_non_finite(LABEL, "projected text embeddings", output)?;
    Ok(output.clone())
}

fn codec_embedding_ids(model: &Qwen3TtsMain, ids: &[u32]) -> Result<Vec<f32>> {
    let mapped = model.checkpoint.mapped()?;
    let hidden = mapped.config().talker.hidden_dim as usize;
    let mut output = Vec::with_capacity(ids.len() * hidden);
    for &id in ids {
        output.extend(codec_embedding_id(model, id)?);
    }
    Ok(output)
}

fn codec_embedding_id(model: &Qwen3TtsMain, id: u32) -> Result<Vec<f32>> {
    let mapped = model.checkpoint.mapped()?;
    let config = &mapped.config().talker;
    let mut output = Vec::new();
    widen_embedding_row(
        mapped,
        mapped.talker_codec_embedding(),
        config.hidden_dim as usize,
        id,
        config.vocab_size,
        "talker codec embedding",
        &mut output,
    )?;
    Ok(output)
}

fn compose_next_talker_embedding(
    model: &Qwen3TtsMain,
    frame: &[u32; QWEN3_TTS_NUM_CODE_GROUPS as usize],
    trailing_text: &[f32],
    tts_pad: &[f32],
    generated: usize,
) -> Result<Vec<f32>> {
    let mapped = model.checkpoint.mapped()?;
    let hidden = mapped.config().talker.hidden_dim as usize;
    let mut output = codec_embedding_id(model, frame[0])?;
    let mut row = Vec::new();
    for (group, &token) in frame[1..].iter().enumerate() {
        widen_embedding_row(
            mapped,
            mapped.code_predictor_embedding(group),
            hidden,
            token,
            mapped.config().code_predictor.vocab_size,
            "code-predictor codec embedding",
            &mut row,
        )?;
        add_in_place(&mut output, &row);
    }
    let trailing_rows = trailing_text.len() / hidden;
    let text_row = if generated < trailing_rows {
        &trailing_text[generated * hidden..(generated + 1) * hidden]
    } else {
        tts_pad
    };
    add_in_place(&mut output, text_row);
    reject_non_finite(LABEL, "composed talker decode embedding", &output)?;
    Ok(output)
}

fn suppress_talker_control_logits(
    logits: &mut [f32],
    generated: usize,
    min_new_tokens: usize,
) -> Result<()> {
    if logits.len() <= 1_024 || CODEC_EOS_TOKEN_ID as usize >= logits.len() {
        return Err(VokraError::ModelLoad(format!(
            "{LABEL}: talker logits width {} cannot apply the official control-token mask",
            logits.len()
        )));
    }
    let first_control = logits.len() - 1_024;
    for (token, logit) in logits.iter_mut().enumerate().skip(first_control) {
        if token != CODEC_EOS_TOKEN_ID as usize {
            *logit = f32::NEG_INFINITY;
        }
    }
    if generated < min_new_tokens {
        logits[CODEC_EOS_TOKEN_ID as usize] = f32::NEG_INFINITY;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn forward_talker_chunk(
    compute: &Compute,
    mapped: &Qwen3TtsMappedDescriptors,
    runtime: &Qwen3TtsRuntime,
    scratch: &mut DecoderStepScratch,
    kv_cache: &mut KvCache,
    embeddings: &[f32],
    rows: usize,
) -> Result<()> {
    let config = &mapped.config().talker;
    let topology = DecoderTopology::from_talker(config);
    forward_decoder_chunk(
        compute,
        mapped,
        &runtime.block,
        scratch,
        kv_cache,
        embeddings,
        rows,
        topology,
        |layer| mapped.talker_layer(layer),
        mapped.talker_final_norm(),
        "talker",
    )
}

#[derive(Clone, Copy)]
struct DecoderTopology {
    hidden: usize,
    layers: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    ffn: usize,
    max_positions: usize,
    rope_base: f32,
    rms_norm_eps: f32,
}

impl DecoderTopology {
    fn from_talker(config: &Qwen3TtsTalkerConfig) -> Self {
        Self {
            hidden: config.hidden_dim as usize,
            layers: config.n_layer as usize,
            heads: config.n_head as usize,
            kv_heads: config.n_head_kv as usize,
            head_dim: config.head_dim as usize,
            ffn: config.ffn_dim as usize,
            max_positions: config.max_position_embeddings as usize,
            rope_base: config.rope_base,
            rms_norm_eps: config.rms_norm_eps,
        }
    }

    fn from_predictor(config: &Qwen3TtsCodePredictorConfig) -> Self {
        Self {
            hidden: config.hidden_dim as usize,
            layers: config.n_layer as usize,
            heads: config.n_head as usize,
            kv_heads: config.n_head_kv as usize,
            head_dim: config.head_dim as usize,
            ffn: config.ffn_dim as usize,
            max_positions: config.max_position_embeddings as usize,
            rope_base: config.rope_base,
            rms_norm_eps: config.rms_norm_eps,
        }
    }

    const fn q_width(self) -> usize {
        self.heads * self.head_dim
    }

    const fn kv_width(self) -> usize {
        self.kv_heads * self.head_dim
    }
}

#[allow(clippy::too_many_arguments)]
fn forward_decoder_chunk<'a>(
    compute: &Compute,
    mapped: &'a Qwen3TtsMappedDescriptors,
    block_scratch: &Mutex<DecoderBlock>,
    scratch: &mut DecoderStepScratch,
    kv_cache: &mut KvCache,
    embeddings: &[f32],
    rows: usize,
    topology: DecoderTopology,
    layer_descriptors: impl Fn(usize) -> DecoderLayerDescriptors<'a>,
    final_norm_info: &GgufTensorInfo,
    decoder_label: &str,
) -> Result<()> {
    let hidden = topology.hidden;
    if rows == 0 || embeddings.len() != rows * hidden {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: {decoder_label} chunk shape mismatch: rows={rows}, values={}, hidden={hidden}",
            embeddings.len()
        )));
    }
    let position_offset = kv_cache.positions();
    if position_offset + rows > topology.max_positions {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: {decoder_label} decode position {} exceeds max positions {}",
            position_offset + rows,
            topology.max_positions
        )));
    }
    scratch.hidden.clear();
    scratch.hidden.extend_from_slice(embeddings);
    let q_width = topology.q_width();
    let kv_width = topology.kv_width();
    let ffn = topology.ffn;
    resize_zero(&mut scratch.norm, rows * hidden);
    resize_zero(&mut scratch.q_raw, rows * q_width);
    resize_zero(&mut scratch.q, rows * q_width);
    resize_zero(&mut scratch.k_raw, rows * kv_width);
    resize_zero(&mut scratch.k, rows * kv_width);
    resize_zero(&mut scratch.v, rows * kv_width);
    resize_zero(&mut scratch.query, rows * topology.head_dim);
    resize_zero(&mut scratch.attention, rows * q_width);
    resize_zero(&mut scratch.attention_out, rows * hidden);
    resize_zero(&mut scratch.ffn_gate, rows * ffn);
    resize_zero(&mut scratch.ffn_activated, rows * ffn);
    resize_zero(&mut scratch.ffn_up, rows * ffn);
    resize_zero(&mut scratch.ffn_down, rows * hidden);

    let mut block = lock_scratch(block_scratch, mapped.mapped_model())?;
    for layer in 0..topology.layers {
        materialize_layer(mapped, layer_descriptors(layer), topology, &mut block)?;
        compute.rms_norm_f32(
            &scratch.hidden,
            &mut scratch.norm,
            rows,
            hidden,
            &block.input_norm,
            topology.rms_norm_eps,
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
            rows * topology.heads,
            topology.head_dim,
            &block.q_norm,
            topology.rms_norm_eps,
        )?;
        compute.rms_norm_f32(
            &scratch.k_raw,
            &mut scratch.k,
            rows * topology.kv_heads,
            topology.head_dim,
            &block.k_norm,
            topology.rms_norm_eps,
        )?;
        // TTS-only prompts have no image/video axes: temporal, height and
        // width position ids are identical, so official interleaved mRoPE is
        // exactly this half-split rotation for every section.
        apply_half_split_rope(
            &mut scratch.q,
            rows,
            topology.heads,
            topology.head_dim,
            topology.rope_base,
            position_offset,
            LABEL,
        )?;
        apply_half_split_rope(
            &mut scratch.k,
            rows,
            topology.kv_heads,
            topology.head_dim,
            topology.rope_base,
            position_offset,
            LABEL,
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
            &format!("{LABEL} {decoder_label}"),
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
        add_in_place(&mut scratch.hidden, &scratch.attention_out);

        compute.rms_norm_f32(
            &scratch.hidden,
            &mut scratch.norm,
            rows,
            hidden,
            &block.ffn_norm,
            topology.rms_norm_eps,
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
        add_in_place(&mut scratch.hidden, &scratch.ffn_down);
    }
    kv_cache.advance(rows);

    let mut final_norm = Vec::new();
    widen_tensor(mapped, final_norm_info, &mut final_norm)?;
    compute.rms_norm_f32(
        &scratch.hidden,
        &mut scratch.norm,
        rows,
        hidden,
        &final_norm,
        topology.rms_norm_eps,
    )?;
    reject_non_finite(
        LABEL,
        &format!("final {decoder_label} hidden"),
        &scratch.norm,
    )
}

fn last_talker_output(
    compute: &Compute,
    mapped: &Qwen3TtsMappedDescriptors,
    runtime: &Qwen3TtsRuntime,
    scratch: &DecoderStepScratch,
) -> Result<Qwen3TtsTalkerOutput> {
    let config = &mapped.config().talker;
    let hidden_width = config.hidden_dim as usize;
    if scratch.norm.len() < hidden_width {
        return Err(VokraError::InvalidArgument(
            "qwen3_tts: no final talker hidden row is available".to_owned(),
        ));
    }
    let hidden = scratch.norm[scratch.norm.len() - hidden_width..].to_vec();
    let mut head = lock_scratch(&runtime.head, mapped.mapped_model())?;
    let HeadScratch { weights, logits } = &mut *head;
    let vocab = config.vocab_size as usize;
    let mut output_logits = vec![0.0; vocab];
    let mut first_row = 0;
    while first_row < vocab {
        let rows = HEAD_CHUNK_ROWS.min(vocab - first_row);
        widen_rows(
            mapped,
            mapped.talker_codec_head(),
            hidden_width,
            first_row,
            rows,
            weights,
        )?;
        resize_zero(logits, rows);
        compute.gemv_f32(rows, hidden_width, weights, &hidden, None, logits)?;
        output_logits[first_row..first_row + rows].copy_from_slice(logits);
        first_row += rows;
    }
    reject_non_finite(LABEL, "talker codec logits", &output_logits)?;
    Ok(Qwen3TtsTalkerOutput {
        hidden,
        logits: output_logits,
    })
}

fn predict_code_frame(
    model: &Qwen3TtsMain,
    talker_hidden: &[f32],
    first_code: u32,
    sampler: &mut Sampler,
) -> Result<[u32; QWEN3_TTS_NUM_CODE_GROUPS as usize]> {
    let mapped = model.checkpoint.mapped()?;
    let talker = &mapped.config().talker;
    let predictor = &mapped.config().code_predictor;
    let talker_hidden_width = talker.hidden_dim as usize;
    if talker_hidden.len() != talker_hidden_width {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: code predictor received {} talker-hidden values, expected {talker_hidden_width}",
            talker_hidden.len()
        )));
    }
    reject_non_finite(LABEL, "code-predictor talker hidden", talker_hidden)?;
    if first_code >= predictor.vocab_size {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: first codebook id {first_code} is not an audio code in 0..{}; EOS/control ids must stop talker generation before code prediction",
            predictor.vocab_size
        )));
    }
    let groups = predictor.num_code_groups as usize;
    if groups != QWEN3_TTS_NUM_CODE_GROUPS as usize {
        return Err(VokraError::ModelLoad(format!(
            "{LABEL}: mapped code predictor has {groups} groups, expected {QWEN3_TTS_NUM_CODE_GROUPS}"
        )));
    }

    let compute = Compute::for_backend(model.backend, QWEN3_TTS_MAIN_HOT_OPS)?;
    let topology = DecoderTopology::from_predictor(predictor);
    let mut kv_cache = KvCache::with_reserve(
        topology.layers,
        topology.kv_width(),
        groups.saturating_add(1),
    );
    let mut decoder_scratch = DecoderStepScratch::default();
    let mut embedding = Vec::new();
    widen_embedding_row(
        mapped,
        mapped.talker_codec_embedding(),
        talker_hidden_width,
        first_code,
        talker.vocab_size,
        "talker codec embedding",
        &mut embedding,
    )?;
    let mut initial = Vec::with_capacity(talker_hidden_width * 2);
    initial.extend_from_slice(talker_hidden);
    initial.extend_from_slice(&embedding);

    let mut projection = lock_scratch(&model.runtime.predictor_projection, mapped.mapped_model())?;
    project_predictor_inputs(&compute, mapped, &initial, 2, &mut projection)?;
    forward_decoder_chunk(
        &compute,
        mapped,
        &model.runtime.predictor_block,
        &mut decoder_scratch,
        &mut kv_cache,
        &projection.output,
        2,
        topology,
        |layer| mapped.code_predictor_layer(layer),
        mapped.code_predictor_final_norm(),
        "code predictor",
    )?;

    let mut codes = [0_u32; QWEN3_TTS_NUM_CODE_GROUPS as usize];
    codes[0] = first_code;
    for group in 0..groups - 1 {
        let mut logits =
            last_predictor_logits(&compute, mapped, &model.runtime, &decoder_scratch, group)?;
        let token = sampler.sample(&mut logits);
        codes[group + 1] = token;
        if group + 1 == groups - 1 {
            break;
        }

        widen_embedding_row(
            mapped,
            mapped.code_predictor_embedding(group),
            talker_hidden_width,
            token,
            predictor.vocab_size,
            "code-predictor codec embedding",
            &mut embedding,
        )?;
        project_predictor_inputs(&compute, mapped, &embedding, 1, &mut projection)?;
        forward_decoder_chunk(
            &compute,
            mapped,
            &model.runtime.predictor_block,
            &mut decoder_scratch,
            &mut kv_cache,
            &projection.output,
            1,
            topology,
            |layer| mapped.code_predictor_layer(layer),
            mapped.code_predictor_final_norm(),
            "code predictor",
        )?;
    }
    Ok(codes)
}

fn project_predictor_inputs(
    compute: &Compute,
    mapped: &Qwen3TtsMappedDescriptors,
    inputs: &[f32],
    rows: usize,
    scratch: &mut ProjectionScratch,
) -> Result<()> {
    let talker_hidden = mapped.config().talker.hidden_dim as usize;
    let predictor_hidden = mapped.config().code_predictor.hidden_dim as usize;
    if rows == 0 || inputs.len() != rows * talker_hidden {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: code-predictor projection shape mismatch: rows={rows}, values={}, talker_hidden={talker_hidden}",
            inputs.len()
        )));
    }
    if let Some((weight, bias)) = mapped.code_predictor_projection() {
        if scratch.weights_t.is_empty() {
            transpose_tensor(
                mapped,
                weight,
                predictor_hidden,
                talker_hidden,
                &mut scratch.weights_t,
            )?;
        }
        if scratch.bias.is_empty() {
            widen_tensor(mapped, bias, &mut scratch.bias)?;
        }
        resize_zero(&mut scratch.output, rows * predictor_hidden);
        compute.gemm_f32(
            rows,
            predictor_hidden,
            talker_hidden,
            inputs,
            &scratch.weights_t,
            Some(&scratch.bias),
            &mut scratch.output,
        )?;
    } else {
        scratch.output.clear();
        scratch.output.extend_from_slice(inputs);
    }
    reject_non_finite(LABEL, "projected code-predictor input", &scratch.output)
}

fn last_predictor_logits(
    compute: &Compute,
    mapped: &Qwen3TtsMappedDescriptors,
    runtime: &Qwen3TtsRuntime,
    scratch: &DecoderStepScratch,
    group: usize,
) -> Result<Vec<f32>> {
    let config = &mapped.config().code_predictor;
    let hidden_width = config.hidden_dim as usize;
    if scratch.norm.len() < hidden_width {
        return Err(VokraError::InvalidArgument(
            "qwen3_tts: no final code-predictor hidden row is available".to_owned(),
        ));
    }
    let hidden = &scratch.norm[scratch.norm.len() - hidden_width..];
    let mut head = lock_scratch(&runtime.predictor_head, mapped.mapped_model())?;
    let HeadScratch { weights, logits } = &mut *head;
    let vocab = config.vocab_size as usize;
    let mut output_logits = vec![0.0; vocab];
    let mut first_row = 0;
    while first_row < vocab {
        let rows = HEAD_CHUNK_ROWS.min(vocab - first_row);
        widen_rows(
            mapped,
            mapped.code_predictor_head(group),
            hidden_width,
            first_row,
            rows,
            weights,
        )?;
        resize_zero(logits, rows);
        compute.gemv_f32(rows, hidden_width, weights, hidden, None, logits)?;
        output_logits[first_row..first_row + rows].copy_from_slice(logits);
        first_row += rows;
    }
    reject_non_finite(LABEL, "code-predictor logits", &output_logits)?;
    Ok(output_logits)
}

fn materialize_layer(
    mapped: &Qwen3TtsMappedDescriptors,
    descriptors: DecoderLayerDescriptors<'_>,
    topology: DecoderTopology,
    block: &mut DecoderBlock,
) -> Result<()> {
    let hidden = topology.hidden;
    let q_width = topology.q_width();
    let kv_width = topology.kv_width();
    let ffn = topology.ffn;
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

#[allow(clippy::too_many_arguments)]
fn attention(
    compute: &Compute,
    scratch: &mut DecoderStepScratch,
    key_cache: &[f32],
    value_cache: &[f32],
    rows: usize,
    position_offset: usize,
    topology: DecoderTopology,
    label: &str,
) -> Result<()> {
    let q_width = topology.q_width();
    let kv_width = topology.kv_width();
    let head_dim = topology.head_dim;
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
    let groups = topology.heads / topology.kv_heads;
    let scale = (head_dim as f32).sqrt().recip();

    for kv_head in 0..topology.kv_heads {
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

fn kv_width(config: &Qwen3TtsTalkerConfig) -> usize {
    DecoderTopology::from_talker(config).kv_width()
}

fn add_in_place(values: &mut [f32], residual: &[f32]) {
    for (value, residual) in values.iter_mut().zip(residual) {
        *value += residual;
    }
}

fn resize_zero(values: &mut Vec<f32>, len: usize) {
    values.clear();
    values.resize(len, 0.0);
}

fn widen_rows(
    mapped: &Qwen3TtsMappedDescriptors,
    info: &GgufTensorInfo,
    row_width: usize,
    first_row: usize,
    rows: usize,
    output: &mut Vec<f32>,
) -> Result<()> {
    let label = mapped.mapped_model().name;
    let element_size = info.dtype.type_size();
    let bytes = mapped.file().tensor_bytes(info);
    let start = first_row
        .checked_mul(row_width)
        .and_then(|value| value.checked_mul(element_size))
        .ok_or_else(|| VokraError::InvalidArgument(format!("{label}: row offset overflow")))?;
    let len = rows
        .checked_mul(row_width)
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

#[allow(clippy::too_many_arguments)]
fn widen_embedding_row(
    mapped: &Qwen3TtsMappedDescriptors,
    info: &GgufTensorInfo,
    row_width: usize,
    token: u32,
    vocab_size: u32,
    embedding_label: &str,
    output: &mut Vec<f32>,
) -> Result<()> {
    if token >= vocab_size {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: {embedding_label} token {token} is outside 0..{vocab_size}"
        )));
    }
    widen_rows(mapped, info, row_width, token as usize, 1, output)
}

fn widen_tensor(
    mapped: &Qwen3TtsMappedDescriptors,
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
    mapped: &Qwen3TtsMappedDescriptors,
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
    fn complete_backend_contract_has_cpu_and_no_metal_gap() {
        Compute::for_backend(BackendKind::Cpu, QWEN3_TTS_MAIN_HOT_OPS)
            .expect("CPU covers the complete Qwen3-TTS main graph");
        #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
        match Compute::for_backend(BackendKind::Metal, QWEN3_TTS_MAIN_HOT_OPS) {
            Ok(compute) => assert_eq!(compute.backend_name(), "metal"),
            Err(VokraError::BackendUnavailable(_)) => {}
            Err(error) => panic!("Qwen3-TTS main graph has a Metal coverage gap: {error}"),
        }
    }

    #[test]
    fn half_split_rope_position_zero_is_identity() {
        let mut values = (0..128).map(|value| value as f32).collect::<Vec<_>>();
        let expected = values.clone();
        apply_half_split_rope(&mut values, 1, 1, 128, 1_000_000.0, 0, LABEL).expect("rope");
        assert_eq!(values, expected);
    }

    #[test]
    fn causal_mask_respects_cached_prefix() {
        let mut scores = vec![2.0; 10];
        scale_and_mask(&mut scores, 2, 5, 3, 0.5, LABEL).expect("mask");
        assert_eq!(&scores[..5], &[1.0, 1.0, 1.0, 1.0, f32::MIN]);
        assert_eq!(&scores[5..], &[1.0, 1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn official_generation_defaults_are_transcribed_exactly() {
        let options = Qwen3TtsGenerationOptions::default();
        assert_eq!(options.language, "Auto");
        assert_eq!(options.max_new_tokens, 8_192);
        assert_eq!(options.min_new_tokens, 2);
        assert_eq!(options.temperature, 0.9);
        assert_eq!(options.top_k, Some(50));
        assert_eq!(options.top_p, None);
        assert_eq!(options.repetition_penalty, Some(1.05));
        assert_eq!(options.predictor_temperature, 0.9);
        assert_eq!(options.predictor_top_k, Some(50));
        assert_eq!(options.predictor_top_p, None);
    }

    #[test]
    fn talker_control_mask_preserves_only_eos() {
        let mut logits = (0..3_072).map(|value| value as f32).collect::<Vec<_>>();
        suppress_talker_control_logits(&mut logits, 2, 2).expect("mask");
        assert_eq!(logits[2_047], 2_047.0);
        assert_eq!(logits[CODEC_EOS_TOKEN_ID as usize], 2_150.0);
        assert!(logits[2_048].is_infinite() && logits[2_048].is_sign_negative());
        assert!(logits[3_071].is_infinite() && logits[3_071].is_sign_negative());

        suppress_talker_control_logits(&mut logits, 0, 2).expect("minimum mask");
        assert!(logits[CODEC_EOS_TOKEN_ID as usize].is_infinite());
    }

    #[test]
    fn generated_codes_transpose_to_decoder_rows() {
        let generated = Qwen3TtsGeneratedCodes {
            frame_major: (0..32).collect(),
            frames: 2,
            ended: true,
        };
        let rows = generated.to_codebook_rows();
        assert_eq!(rows.len(), 16);
        assert_eq!(rows[0], vec![0, 16]);
        assert_eq!(rows[15], vec![15, 31]);
        assert!(generated.ended());
    }

    #[test]
    fn predictor_topology_carries_official_position_ceiling() {
        let config = Qwen3TtsCodePredictorConfig::qwen3_tts_0_6b_base();
        let topology = DecoderTopology::from_predictor(&config);
        assert_eq!(topology.max_positions, 65_536);
        assert_eq!(topology.layers, 5);
        assert_eq!(topology.hidden, 1_024);
    }
}
