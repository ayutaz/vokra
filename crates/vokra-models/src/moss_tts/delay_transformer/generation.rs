//! Official single-sequence delayed-codebook generation state machine.

use std::collections::BTreeSet;

use vokra_core::{Sampler, SamplerConfig};

use super::*;

const PAD_TOKEN_ID: u32 = 151_643;
const IM_START_TOKEN_ID: u32 = 151_644;
const IM_END_TOKEN_ID: u32 = 151_645;
const AUDIO_START_TOKEN_ID: u32 = 151_652;
const AUDIO_END_TOKEN_ID: u32 = 151_653;
const AUDIO_ASSISTANT_GEN_SLOT_TOKEN_ID: u32 = 151_656;
const AUDIO_ASSISTANT_DELAY_SLOT_TOKEN_ID: u32 = 151_662;
const AUDIO_PAD_CODE: u32 = 1_024;

/// Sampling controls for the official MOSS-TTS Delay state machine.
///
/// The default temperatures/top-k/top-p values match the fixed upstream
/// `generate` signature. Vokra uses its seedable first-party sampler rather
/// than PyTorch's process-global RNG, so the same seeds reproduce a sequence.
#[derive(Debug, Clone)]
pub struct MossTtsDelayGenerationOptions {
    /// Maximum number of appended `[text + 32 audio]` rows.
    pub max_new_tokens: usize,
    /// Text-head sampler. Default: temperature 1.5, top-k 50, top-p disabled
    /// (`1.0` upstream), seed 0.
    pub text_sampler: SamplerConfig,
    /// Audio-head sampler. Default: temperature 1.7, top-k 25, top-p 0.8,
    /// repetition penalty disabled (`1.0` upstream), seed 1.
    pub audio_sampler: SamplerConfig,
}

impl Default for MossTtsDelayGenerationOptions {
    fn default() -> Self {
        Self {
            max_new_tokens: 1_000,
            text_sampler: SamplerConfig {
                temperature: 1.5,
                top_k: Some(50),
                top_p: None,
                repetition_penalty: None,
                seed: 0,
            },
            audio_sampler: SamplerConfig {
                temperature: 1.7,
                top_k: Some(25),
                top_p: Some(0.8),
                repetition_penalty: None,
                seed: 1,
            },
        }
    }
}

/// Raw rows appended by the official delayed-codebook generator.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MossTtsDelayGeneration {
    /// Flat row-major `[generated_rows, 33]` values. Column zero is text;
    /// columns 1..32 are delayed audio codes/pad. The explicit raw boundary
    /// preserves continuation context and lets the Full codec stage perform
    /// the official de-delay/segment operation without guessing.
    pub generated_rows: Vec<u32>,
}

impl MossTtsDelayGeneration {
    /// Number of generated 33-column rows.
    pub fn row_count(&self) -> usize {
        self.generated_rows.len() / INPUT_COLUMNS
    }

    /// One generated row by index.
    pub fn row(&self, index: usize) -> Result<&[u32]> {
        if index >= self.row_count() {
            return Err(VokraError::InvalidArgument(format!(
                "{LABEL}: generated row {index} is outside 0..{}",
                self.row_count()
            )));
        }
        let start = index * INPUT_COLUMNS;
        Ok(&self.generated_rows[start..start + INPUT_COLUMNS])
    }
}

pub(super) struct DeDelayedAudio {
    pub(super) start_length: usize,
    pub(super) segments: Vec<DeDelayedCodeSegment>,
}

pub(super) struct DeDelayedCodeSegment {
    pub(super) codes: Vec<u32>,
    pub(super) frames: usize,
}

/// Ports `MossTTSDelayProcessor.apply_de_delay_pattern` and the following
/// all-pad segment split from the fixed official processor source.
pub(super) fn de_delay_audio_segments(
    prompt: &[u32],
    generated: &MossTtsDelayGeneration,
) -> Result<DeDelayedAudio> {
    if prompt.is_empty() || !prompt.len().is_multiple_of(INPUT_COLUMNS) {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: de-delay prompt must be non-empty [rows,{INPUT_COLUMNS}], got {} values",
            prompt.len()
        )));
    }
    if !generated.generated_rows.len().is_multiple_of(INPUT_COLUMNS) {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: generated de-delay rows have {} values, not a multiple of {INPUT_COLUMNS}",
            generated.generated_rows.len()
        )));
    }
    let prompt_rows = prompt.len() / INPUT_COLUMNS;
    let im_start = prompt
        .chunks_exact(INPUT_COLUMNS)
        .rposition(|row| row[0] == IM_START_TOKEN_ID)
        .ok_or_else(|| {
            VokraError::InvalidArgument(format!(
                "{LABEL}: synthesis prompt has no official im_start token {IM_START_TOKEN_ID}; continuation trim cannot be inferred"
            ))
        })?;
    let assistant_start = im_start.checked_add(3).ok_or_else(|| {
        VokraError::InvalidArgument(format!("{LABEL}: assistant start index overflows"))
    })?;
    if assistant_start > prompt_rows {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: official assistant slice starts at row {assistant_start}, beyond prompt rows {prompt_rows}"
        )));
    }
    let start_length = prompt_rows - assistant_start;
    let generated_rows = generated.row_count();
    let delay_rows = start_length
        .checked_add(generated_rows)
        .ok_or_else(|| VokraError::InvalidArgument(format!("{LABEL}: de-delay rows overflow")))?;
    if delay_rows < NUM_AUDIO_CODEBOOKS {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: de-delay requires at least {NUM_AUDIO_CODEBOOKS} assistant rows, got {delay_rows}; increase max_new_tokens so the codebook drain can finish"
        )));
    }

    let audio_row = |row: usize, codebook: usize| -> u32 {
        if row < start_length {
            prompt[(assistant_start + row) * INPUT_COLUMNS + 1 + codebook]
        } else {
            generated.generated_rows[(row - start_length) * INPUT_COLUMNS + 1 + codebook]
        }
    };
    let frames = delay_rows - NUM_AUDIO_CODEBOOKS + 1;
    let mut de_delayed = Vec::with_capacity(frames * NUM_AUDIO_CODEBOOKS);
    for frame in 0..frames {
        for codebook in 0..NUM_AUDIO_CODEBOOKS {
            de_delayed.push(audio_row(frame + codebook, codebook));
        }
    }

    let mut segments = Vec::new();
    let mut segment_start = None;
    for frame in 0..frames {
        let row = &de_delayed[frame * NUM_AUDIO_CODEBOOKS..(frame + 1) * NUM_AUDIO_CODEBOOKS];
        let all_pad = row.iter().all(|code| *code == AUDIO_PAD_CODE);
        if let Some((codebook, code)) = row
            .iter()
            .copied()
            .enumerate()
            .find(|(_, code)| !all_pad && *code >= AUDIO_PAD_CODE)
        {
            return Err(VokraError::InvalidArgument(format!(
                "{LABEL}: de-delayed frame {frame} codebook {codebook} contains pad/out-of-range code {code}; generation ended before a complete 32-codebook segment"
            )));
        }
        match (segment_start, all_pad) {
            (None, false) => segment_start = Some(frame),
            (Some(start), true) => {
                let codes =
                    de_delayed[start * NUM_AUDIO_CODEBOOKS..frame * NUM_AUDIO_CODEBOOKS].to_vec();
                segments.push(DeDelayedCodeSegment {
                    frames: frame - start,
                    codes,
                });
                segment_start = None;
            }
            _ => {}
        }
    }
    if let Some(start) = segment_start {
        let codes = de_delayed[start * NUM_AUDIO_CODEBOOKS..].to_vec();
        segments.push(DeDelayedCodeSegment {
            frames: frames - start,
            codes,
        });
    }
    Ok(DeDelayedAudio {
        start_length,
        segments,
    })
}

impl MossTtsDelay {
    /// Runs the official delayed-codebook state machine from an explicit
    /// upstream-compatible `[rows,33]` prompt.
    ///
    /// This is single-sequence generation: no left-padding attention mask is
    /// inferred. Text/audio tokenizer assets remain explicit companions. The
    /// return value contains only appended rows, not the prompt.
    pub fn generate_delay_rows(
        &self,
        prompt_rows: &[u32],
        options: &MossTtsDelayGenerationOptions,
    ) -> Result<MossTtsDelayGeneration> {
        validate_generation_inputs(prompt_rows, options)?;
        let prompt_count = prompt_rows.len() / INPUT_COLUMNS;
        let compute = Compute::for_backend(self.backend, MOSS_TTS_DELAY_HOT_OPS)?;
        let mapped = self.checkpoint.mapped();
        let reserve = (prompt_count + options.max_new_tokens.min(256)).min(512);
        let mut kv_cache = KvCache::with_reserve(NUM_LAYERS, KV_DIM, reserve.max(1));
        let mut scratch = DelayStepScratch::default();
        for row_start in (0..prompt_count).step_by(PREFILL_CHUNK_ROWS) {
            let chunk_rows = PREFILL_CHUNK_ROWS.min(prompt_count - row_start);
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
        let mut logits = last_logits(&compute, mapped, &self.runtime, &scratch)?;
        let mut state = DelayGenerationState::from_prompt(prompt_rows);
        let mut text_sampler = Sampler::new(options.text_sampler.clone());
        let mut audio_config = options.audio_sampler.clone();
        let audio_repetition_penalty = audio_config.repetition_penalty.take();
        let mut audio_sampler = Sampler::new(audio_config);
        let mut history = prompt_rows.to_vec();
        let mut generated = Vec::with_capacity(options.max_new_tokens * INPUT_COLUMNS);

        for time_step in 0..options.max_new_tokens {
            let row = next_row(
                &mut logits,
                &mut state,
                time_step,
                &history,
                &mut text_sampler,
                &mut audio_sampler,
                audio_repetition_penalty,
            )?;
            let should_stop = row[0] == IM_END_TOKEN_ID;
            generated.extend_from_slice(&row);
            history.extend_from_slice(&row);
            if should_stop || time_step + 1 == options.max_new_tokens {
                break;
            }
            forward_chunk(
                &compute,
                mapped,
                &self.runtime,
                &mut scratch,
                &mut kv_cache,
                &row,
                1,
            )?;
            logits = last_logits(&compute, mapped, &self.runtime, &scratch)?;
        }
        Ok(MossTtsDelayGeneration {
            generated_rows: generated,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DelayGenerationState {
    audio_length: usize,
    delayed_length: Option<usize>,
    is_audio: bool,
}

impl DelayGenerationState {
    fn from_prompt(prompt: &[u32]) -> Self {
        let rows = prompt.len() / INPUT_COLUMNS;
        let last_text = prompt[(rows - 1) * INPUT_COLUMNS];
        let continuation = matches!(
            last_text,
            AUDIO_START_TOKEN_ID | AUDIO_ASSISTANT_GEN_SLOT_TOKEN_ID
        );
        let last_audio_start = prompt
            .chunks_exact(INPUT_COLUMNS)
            .rposition(|row| row[0] == AUDIO_START_TOKEN_ID);
        let (is_audio, audio_length) = match (continuation, last_audio_start) {
            (true, Some(index)) => (true, rows - index),
            _ => (false, 0),
        };
        Self {
            audio_length,
            delayed_length: None,
            is_audio,
        }
    }

    fn audio_sampling_mask(self, codebook: usize) -> bool {
        self.audio_length > codebook
            && self
                .delayed_length
                .is_none_or(|delayed| codebook >= delayed)
    }

    fn advance(&mut self, text_token: u32) {
        if matches!(
            text_token,
            AUDIO_START_TOKEN_ID
                | AUDIO_ASSISTANT_GEN_SLOT_TOKEN_ID
                | AUDIO_ASSISTANT_DELAY_SLOT_TOKEN_ID
        ) {
            self.audio_length += 1;
        }
        if text_token == AUDIO_END_TOKEN_ID {
            self.audio_length = 0;
        }
        if self.delayed_length.is_none() && text_token == AUDIO_ASSISTANT_DELAY_SLOT_TOKEN_ID {
            self.delayed_length = Some(0);
        }
        if let Some(delayed) = &mut self.delayed_length {
            *delayed += 1;
            if *delayed > NUM_AUDIO_CODEBOOKS {
                self.delayed_length = None;
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn next_row(
    logits: &mut MossTtsDelayLogits,
    state: &mut DelayGenerationState,
    time_step: usize,
    history: &[u32],
    text_sampler: &mut Sampler,
    audio_sampler: &mut Sampler,
    audio_repetition_penalty: Option<f32>,
) -> Result<[u32; INPUT_COLUMNS]> {
    let delayed_before = state.delayed_length;
    let audio_length_before = state.audio_length;
    let text_token = match delayed_before {
        Some(delayed) if delayed < NUM_AUDIO_CODEBOOKS => AUDIO_ASSISTANT_DELAY_SLOT_TOKEN_ID,
        Some(delayed) if delayed == NUM_AUDIO_CODEBOOKS => {
            state.is_audio = false;
            AUDIO_END_TOKEN_ID
        }
        Some(delayed) => {
            return Err(VokraError::InvalidArgument(format!(
                "{LABEL}: delayed length {delayed} exceeds {NUM_AUDIO_CODEBOOKS}"
            )));
        }
        None => {
            mask_text_logits(&mut logits.text_logits, state.is_audio, time_step)?;
            text_sampler.sample(&mut logits.text_logits)
        }
    };
    if text_token == AUDIO_START_TOKEN_ID {
        state.is_audio = true;
    }

    let mask_state = DelayGenerationState {
        audio_length: audio_length_before,
        delayed_length: delayed_before,
        is_audio: state.is_audio,
    };
    let mut row = [AUDIO_PAD_CODE; INPUT_COLUMNS];
    row[0] = text_token;
    for codebook in 0..NUM_AUDIO_CODEBOOKS {
        if !mask_state.audio_sampling_mask(codebook) {
            continue;
        }
        let head = logits.audio_codebook(codebook)?;
        let mut audio_logits = head.to_vec();
        audio_logits[AUDIO_PAD_CODE as usize] = f32::NEG_INFINITY;
        if let Some(penalty) = audio_repetition_penalty {
            apply_repetition_penalty(&mut audio_logits, history, codebook, penalty)?;
        }
        row[1 + codebook] = audio_sampler.sample(&mut audio_logits);
    }
    state.advance(text_token);
    Ok(row)
}

fn mask_text_logits(logits: &mut [f32], is_audio: bool, time_step: usize) -> Result<()> {
    if logits.len() != TEXT_VOCAB_SIZE {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: text logits length {} != {TEXT_VOCAB_SIZE}",
            logits.len()
        )));
    }
    if is_audio {
        for (token, logit) in logits.iter_mut().enumerate() {
            if token as u32 != AUDIO_ASSISTANT_GEN_SLOT_TOKEN_ID
                && token as u32 != AUDIO_ASSISTANT_DELAY_SLOT_TOKEN_ID
            {
                *logit = f32::NEG_INFINITY;
            }
        }
    } else {
        for token in [
            PAD_TOKEN_ID,
            AUDIO_ASSISTANT_GEN_SLOT_TOKEN_ID,
            AUDIO_ASSISTANT_DELAY_SLOT_TOKEN_ID,
            AUDIO_END_TOKEN_ID,
        ] {
            logits[token as usize] = f32::NEG_INFINITY;
        }
    }
    if time_step == 0 {
        logits[AUDIO_ASSISTANT_DELAY_SLOT_TOKEN_ID as usize] = f32::NEG_INFINITY;
    }
    if time_step <= NUM_AUDIO_CODEBOOKS {
        logits[IM_END_TOKEN_ID as usize] = f32::NEG_INFINITY;
    }
    Ok(())
}

fn apply_repetition_penalty(
    logits: &mut [f32],
    history: &[u32],
    codebook: usize,
    penalty: f32,
) -> Result<()> {
    if !penalty.is_finite() || penalty <= 0.0 {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: repetition penalty must be finite and positive, got {penalty}"
        )));
    }
    let unique: BTreeSet<usize> = history
        .chunks_exact(INPUT_COLUMNS)
        .map(|row| row[1 + codebook] as usize)
        .filter(|&token| token < logits.len())
        .collect();
    for token in unique {
        logits[token] = if logits[token] > 0.0 {
            logits[token] / penalty
        } else {
            logits[token] * penalty
        };
    }
    Ok(())
}

fn validate_generation_inputs(
    prompt: &[u32],
    options: &MossTtsDelayGenerationOptions,
) -> Result<()> {
    if prompt.is_empty() || !prompt.len().is_multiple_of(INPUT_COLUMNS) {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: generation prompt must be non-empty [rows,{INPUT_COLUMNS}], got {} values",
            prompt.len()
        )));
    }
    if options.max_new_tokens == 0 {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: max_new_tokens must be positive"
        )));
    }
    let prompt_rows = prompt.len() / INPUT_COLUMNS;
    if prompt_rows
        .checked_add(options.max_new_tokens)
        .is_none_or(|total| total > MAX_POSITION_EMBEDDINGS)
    {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: prompt rows {prompt_rows} + max_new_tokens {} exceed max positions {MAX_POSITION_EMBEDDINGS}",
            options.max_new_tokens
        )));
    }
    validate_sampler("text", &options.text_sampler)?;
    validate_sampler("audio", &options.audio_sampler)
}

fn validate_sampler(label: &str, config: &SamplerConfig) -> Result<()> {
    if !config.temperature.is_finite() || config.temperature < 0.0 {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: {label} temperature must be finite and non-negative, got {}",
            config.temperature
        )));
    }
    if config.top_k == Some(0) {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: {label} top_k must be positive when present"
        )));
    }
    if config
        .top_p
        .is_some_and(|top_p| !top_p.is_finite() || top_p <= 0.0 || top_p > 1.0)
    {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: {label} top_p must be in (0,1] when present"
        )));
    }
    if config
        .repetition_penalty
        .is_some_and(|penalty| !penalty.is_finite() || penalty <= 0.0)
    {
        return Err(VokraError::InvalidArgument(format!(
            "{LABEL}: {label} repetition penalty must be finite and positive"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prompt_before_assistant_audio() -> Vec<u32> {
        let mut prompt = vec![AUDIO_PAD_CODE; 3 * INPUT_COLUMNS];
        prompt[0] = IM_START_TOKEN_ID;
        prompt[INPUT_COLUMNS] = PAD_TOKEN_ID;
        prompt[2 * INPUT_COLUMNS] = PAD_TOKEN_ID;
        prompt
    }

    fn delay_frames(frames: &[u32], frame_count: usize) -> MossTtsDelayGeneration {
        assert_eq!(frames.len(), frame_count * NUM_AUDIO_CODEBOOKS);
        let delay_rows = frame_count + NUM_AUDIO_CODEBOOKS - 1;
        let mut generated_rows = vec![AUDIO_PAD_CODE; delay_rows * INPUT_COLUMNS];
        for row in generated_rows.chunks_exact_mut(INPUT_COLUMNS) {
            row[0] = AUDIO_ASSISTANT_GEN_SLOT_TOKEN_ID;
        }
        for frame in 0..frame_count {
            for codebook in 0..NUM_AUDIO_CODEBOOKS {
                generated_rows[(frame + codebook) * INPUT_COLUMNS + 1 + codebook] =
                    frames[frame * NUM_AUDIO_CODEBOOKS + codebook];
            }
        }
        MossTtsDelayGeneration { generated_rows }
    }

    #[test]
    fn delay_drain_advances_one_codebook_per_row() {
        let mut state = DelayGenerationState {
            audio_length: 40,
            delayed_length: None,
            is_audio: true,
        };
        assert!((0..NUM_AUDIO_CODEBOOKS).all(|codebook| state.audio_sampling_mask(codebook)));
        state.advance(AUDIO_ASSISTANT_DELAY_SLOT_TOKEN_ID);
        assert_eq!(state.delayed_length, Some(1));
        assert!(!state.audio_sampling_mask(0));
        assert!(state.audio_sampling_mask(1));
        for _ in 1..NUM_AUDIO_CODEBOOKS {
            state.advance(AUDIO_ASSISTANT_DELAY_SLOT_TOKEN_ID);
        }
        assert_eq!(state.delayed_length, Some(NUM_AUDIO_CODEBOOKS));
        assert!(!(0..NUM_AUDIO_CODEBOOKS).any(|codebook| state.audio_sampling_mask(codebook)));
        state.advance(AUDIO_END_TOKEN_ID);
        assert_eq!(state.delayed_length, None);
        assert_eq!(state.audio_length, 0);
    }

    #[test]
    fn audio_text_mask_keeps_only_generation_and_delay_slots() {
        let mut logits = vec![0.0; TEXT_VOCAB_SIZE];
        mask_text_logits(&mut logits, true, 1).unwrap();
        assert_eq!(logits[AUDIO_ASSISTANT_GEN_SLOT_TOKEN_ID as usize], 0.0);
        assert_eq!(logits[AUDIO_ASSISTANT_DELAY_SLOT_TOKEN_ID as usize], 0.0);
        assert_eq!(logits[AUDIO_START_TOKEN_ID as usize], f32::NEG_INFINITY);
    }

    #[test]
    fn repetition_penalty_is_unique_per_codebook() {
        let mut logits = vec![2.0, -2.0, 4.0];
        let mut history = vec![AUDIO_PAD_CODE; 3 * INPUT_COLUMNS];
        history[1] = 0;
        history[INPUT_COLUMNS + 1] = 0;
        history[2 * INPUT_COLUMNS + 1] = 1;
        apply_repetition_penalty(&mut logits, &history, 0, 2.0).unwrap();
        assert_eq!(logits, vec![1.0, -4.0, 4.0]);
    }

    #[test]
    fn official_de_delay_restores_frame_major_codes_and_pad_segments() {
        let mut codes = (0..3 * NUM_AUDIO_CODEBOOKS)
            .map(|index| (index % 1_024) as u32)
            .collect::<Vec<_>>();
        codes[NUM_AUDIO_CODEBOOKS..2 * NUM_AUDIO_CODEBOOKS].fill(AUDIO_PAD_CODE);
        let generated = delay_frames(&codes, 3);
        let parsed = de_delay_audio_segments(&prompt_before_assistant_audio(), &generated).unwrap();
        assert_eq!(parsed.start_length, 0);
        assert_eq!(parsed.segments.len(), 2);
        assert_eq!(parsed.segments[0].frames, 1);
        assert_eq!(parsed.segments[0].codes, codes[..NUM_AUDIO_CODEBOOKS]);
        assert_eq!(parsed.segments[1].frames, 1);
        assert_eq!(parsed.segments[1].codes, codes[2 * NUM_AUDIO_CODEBOOKS..]);
    }

    #[test]
    fn de_delay_includes_prompt_prefix_for_continuation() {
        let codes = (0..2 * NUM_AUDIO_CODEBOOKS)
            .map(|index| (index % 1_024) as u32)
            .collect::<Vec<_>>();
        let full = delay_frames(&codes, 2);
        let mut prompt = prompt_before_assistant_audio();
        prompt.extend_from_slice(&full.generated_rows[..INPUT_COLUMNS]);
        let generated = MossTtsDelayGeneration {
            generated_rows: full.generated_rows[INPUT_COLUMNS..].to_vec(),
        };
        let parsed = de_delay_audio_segments(&prompt, &generated).unwrap();
        assert_eq!(parsed.start_length, 1);
        assert_eq!(parsed.segments.len(), 1);
        assert_eq!(parsed.segments[0].frames, 2);
        assert_eq!(parsed.segments[0].codes, codes);
    }
}
