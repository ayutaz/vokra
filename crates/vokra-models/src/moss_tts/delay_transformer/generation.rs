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

/// Sampling controls for the official MOSS-TTS Delay-class state machine.
///
/// [`Default`] matches Base/v1.5. [`Self::voice_generator`] selects the
/// separately audited VoiceGenerator values. Vokra uses its seedable
/// first-party sampler rather than PyTorch's process-global RNG.
#[derive(Debug, Clone)]
pub struct MossTtsDelayGenerationOptions {
    /// Maximum number of appended `[text + release codebooks]` rows.
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

impl MossTtsDelayGenerationOptions {
    /// Official MOSS-VoiceGenerator sampling defaults.
    pub fn voice_generator() -> Self {
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
                temperature: 1.5,
                top_k: Some(50),
                top_p: Some(0.6),
                repetition_penalty: Some(1.1),
                seed: 1,
            },
        }
    }
}

/// Raw rows appended by the official delayed-codebook generator.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MossTtsDelayGeneration {
    /// Flat row-major `[generated_rows, 1 + num_audio_codebooks]` values.
    /// Column zero is text and the remaining columns are delayed audio
    /// codes/pad. The explicit raw boundary preserves continuation context and
    /// lets the Full codec stage de-delay without guessing.
    pub generated_rows: Vec<u32>,
    /// Authenticated release codebook count (32 for Base/v1.5, 16 for
    /// VoiceGenerator).
    pub num_audio_codebooks: usize,
}

impl MossTtsDelayGeneration {
    /// Number of values in one generated row.
    pub const fn column_count(&self) -> usize {
        1 + self.num_audio_codebooks
    }

    /// Number of generated rows.
    pub fn row_count(&self) -> usize {
        self.generated_rows.len() / self.column_count()
    }

    /// One generated row by index.
    pub fn row(&self, index: usize) -> Result<&[u32]> {
        if index >= self.row_count() {
            return Err(VokraError::InvalidArgument(format!(
                "{LABEL}: generated row {index} is outside 0..{}",
                self.row_count()
            )));
        }
        let columns = self.column_count();
        let start = index * columns;
        Ok(&self.generated_rows[start..start + columns])
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
    label: &str,
) -> Result<DeDelayedAudio> {
    let num_audio_codebooks = generated.num_audio_codebooks;
    if num_audio_codebooks == 0 {
        return Err(VokraError::InvalidArgument(format!(
            "{label}: de-delay requires at least one audio codebook"
        )));
    }
    let columns = generated.column_count();
    if prompt.is_empty() || !prompt.len().is_multiple_of(columns) {
        return Err(VokraError::InvalidArgument(format!(
            "{label}: de-delay prompt must be non-empty [rows,{columns}], got {} values",
            prompt.len()
        )));
    }
    if !generated.generated_rows.len().is_multiple_of(columns) {
        return Err(VokraError::InvalidArgument(format!(
            "{label}: generated de-delay rows have {} values, not a multiple of {columns}",
            generated.generated_rows.len()
        )));
    }
    let prompt_rows = prompt.len() / columns;
    let im_start = prompt
        .chunks_exact(columns)
        .rposition(|row| row[0] == IM_START_TOKEN_ID)
        .ok_or_else(|| {
            VokraError::InvalidArgument(format!(
                "{label}: synthesis prompt has no official im_start token {IM_START_TOKEN_ID}; continuation trim cannot be inferred"
            ))
        })?;
    let assistant_start = im_start.checked_add(3).ok_or_else(|| {
        VokraError::InvalidArgument(format!("{label}: assistant start index overflows"))
    })?;
    if assistant_start > prompt_rows {
        return Err(VokraError::InvalidArgument(format!(
            "{label}: official assistant slice starts at row {assistant_start}, beyond prompt rows {prompt_rows}"
        )));
    }
    let start_length = prompt_rows - assistant_start;
    let generated_rows = generated.row_count();
    let delay_rows = start_length
        .checked_add(generated_rows)
        .ok_or_else(|| VokraError::InvalidArgument(format!("{label}: de-delay rows overflow")))?;
    if delay_rows < num_audio_codebooks {
        return Err(VokraError::InvalidArgument(format!(
            "{label}: de-delay requires at least {num_audio_codebooks} assistant rows, got {delay_rows}; increase max_new_tokens so the codebook drain can finish"
        )));
    }

    let audio_row = |row: usize, codebook: usize| -> u32 {
        if row < start_length {
            prompt[(assistant_start + row) * columns + 1 + codebook]
        } else {
            generated.generated_rows[(row - start_length) * columns + 1 + codebook]
        }
    };
    let frames = delay_rows - num_audio_codebooks + 1;
    let mut de_delayed = Vec::with_capacity(frames * num_audio_codebooks);
    for frame in 0..frames {
        for codebook in 0..num_audio_codebooks {
            de_delayed.push(audio_row(frame + codebook, codebook));
        }
    }

    let mut segments = Vec::new();
    let mut segment_start = None;
    for frame in 0..frames {
        let row = &de_delayed[frame * num_audio_codebooks..(frame + 1) * num_audio_codebooks];
        let all_pad = row.iter().all(|code| *code == AUDIO_PAD_CODE);
        if let Some((codebook, code)) = row
            .iter()
            .copied()
            .enumerate()
            .find(|(_, code)| !all_pad && *code >= AUDIO_PAD_CODE)
        {
            return Err(VokraError::InvalidArgument(format!(
                "{label}: de-delayed frame {frame} codebook {codebook} contains pad/out-of-range code {code}; generation ended before a complete {num_audio_codebooks}-codebook segment"
            )));
        }
        match (segment_start, all_pad) {
            (None, false) => segment_start = Some(frame),
            (Some(start), true) => {
                let codes =
                    de_delayed[start * num_audio_codebooks..frame * num_audio_codebooks].to_vec();
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
        let codes = de_delayed[start * num_audio_codebooks..].to_vec();
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

pub(super) fn generate_delay_rows(
    model: &impl DelayRuntimeAccess,
    prompt_rows: &[u32],
    options: &MossTtsDelayGenerationOptions,
) -> Result<MossTtsDelayGeneration> {
    let mapped = model.mapped();
    let topology = mapped.topology();
    let columns = topology.input_columns();
    let label = mapped.mapped_model().name;
    validate_generation_inputs(prompt_rows, options, topology, label)?;
    let prompt_count = prompt_rows.len() / columns;
    let compute = Compute::for_backend(model.backend(), MOSS_TTS_DELAY_HOT_OPS)?;
    let reserve = (prompt_count + options.max_new_tokens.min(256)).min(512);
    let mut kv_cache =
        KvCache::with_reserve(topology.num_layers, topology.kv_dim(), reserve.max(1));
    let mut scratch = DelayStepScratch::default();
    for row_start in (0..prompt_count).step_by(PREFILL_CHUNK_ROWS) {
        let chunk_rows = PREFILL_CHUNK_ROWS.min(prompt_count - row_start);
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
    let mut logits = last_logits(&compute, mapped, model.runtime(), &scratch)?;
    let mut state = DelayGenerationState::from_prompt(prompt_rows, columns);
    let mut text_sampler = Sampler::new(options.text_sampler.clone());
    let mut audio_config = options.audio_sampler.clone();
    let audio_repetition_penalty = audio_config.repetition_penalty.take();
    let mut audio_sampler = Sampler::new(audio_config);
    let mut history = prompt_rows.to_vec();
    let mut generated = Vec::with_capacity(options.max_new_tokens * columns);

    for time_step in 0..options.max_new_tokens {
        let row = next_row(
            &mut logits,
            &mut state,
            time_step,
            &history,
            &mut text_sampler,
            &mut audio_sampler,
            audio_repetition_penalty,
            topology.num_audio_codebooks,
            columns,
            topology.text_vocab_size,
            label,
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
            model.runtime(),
            &mut scratch,
            &mut kv_cache,
            &row,
            1,
        )?;
        logits = last_logits(&compute, mapped, model.runtime(), &scratch)?;
    }
    Ok(MossTtsDelayGeneration {
        generated_rows: generated,
        num_audio_codebooks: topology.num_audio_codebooks,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DelayGenerationState {
    audio_length: usize,
    delayed_length: Option<usize>,
    is_audio: bool,
}

impl DelayGenerationState {
    fn from_prompt(prompt: &[u32], columns: usize) -> Self {
        let rows = prompt.len() / columns;
        let last_text = prompt[(rows - 1) * columns];
        let continuation = matches!(
            last_text,
            AUDIO_START_TOKEN_ID | AUDIO_ASSISTANT_GEN_SLOT_TOKEN_ID
        );
        let last_audio_start = prompt
            .chunks_exact(columns)
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

    fn advance(&mut self, text_token: u32, num_audio_codebooks: usize) {
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
            if *delayed > num_audio_codebooks {
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
    num_audio_codebooks: usize,
    columns: usize,
    text_vocab_size: usize,
    label: &str,
) -> Result<Vec<u32>> {
    let delayed_before = state.delayed_length;
    let audio_length_before = state.audio_length;
    let text_token = match delayed_before {
        Some(delayed) if delayed < num_audio_codebooks => AUDIO_ASSISTANT_DELAY_SLOT_TOKEN_ID,
        Some(delayed) if delayed == num_audio_codebooks => {
            state.is_audio = false;
            AUDIO_END_TOKEN_ID
        }
        Some(delayed) => {
            return Err(VokraError::InvalidArgument(format!(
                "{label}: delayed length {delayed} exceeds {num_audio_codebooks}"
            )));
        }
        None => {
            mask_text_logits(
                &mut logits.text_logits,
                state.is_audio,
                time_step,
                text_vocab_size,
                num_audio_codebooks,
                label,
            )?;
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
    let mut row = vec![AUDIO_PAD_CODE; columns];
    row[0] = text_token;
    for codebook in 0..num_audio_codebooks {
        if !mask_state.audio_sampling_mask(codebook) {
            continue;
        }
        let head = logits.audio_codebook(codebook)?;
        let mut audio_logits = head.to_vec();
        if audio_logits.len() <= AUDIO_PAD_CODE as usize {
            return Err(VokraError::InvalidArgument(format!(
                "{label}: audio head {codebook} has {} logits, expected pad index {AUDIO_PAD_CODE}",
                audio_logits.len()
            )));
        }
        audio_logits[AUDIO_PAD_CODE as usize] = f32::NEG_INFINITY;
        if let Some(penalty) = audio_repetition_penalty {
            apply_repetition_penalty(
                &mut audio_logits,
                history,
                codebook,
                columns,
                penalty,
                label,
            )?;
        }
        row[1 + codebook] = audio_sampler.sample(&mut audio_logits);
    }
    state.advance(text_token, num_audio_codebooks);
    Ok(row)
}

fn mask_text_logits(
    logits: &mut [f32],
    is_audio: bool,
    time_step: usize,
    text_vocab_size: usize,
    num_audio_codebooks: usize,
    label: &str,
) -> Result<()> {
    if logits.len() != text_vocab_size {
        return Err(VokraError::InvalidArgument(format!(
            "{label}: text logits length {} != {text_vocab_size}",
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
    if time_step <= num_audio_codebooks {
        logits[IM_END_TOKEN_ID as usize] = f32::NEG_INFINITY;
    }
    Ok(())
}

fn apply_repetition_penalty(
    logits: &mut [f32],
    history: &[u32],
    codebook: usize,
    columns: usize,
    penalty: f32,
    label: &str,
) -> Result<()> {
    if !penalty.is_finite() || penalty <= 0.0 {
        return Err(VokraError::InvalidArgument(format!(
            "{label}: repetition penalty must be finite and positive, got {penalty}"
        )));
    }
    let unique: BTreeSet<usize> = history
        .chunks_exact(columns)
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
    topology: DelayTopology,
    label: &str,
) -> Result<()> {
    let columns = topology.input_columns();
    if prompt.is_empty() || !prompt.len().is_multiple_of(columns) {
        return Err(VokraError::InvalidArgument(format!(
            "{label}: generation prompt must be non-empty [rows,{columns}], got {} values",
            prompt.len()
        )));
    }
    if options.max_new_tokens == 0 {
        return Err(VokraError::InvalidArgument(format!(
            "{label}: max_new_tokens must be positive"
        )));
    }
    if topology.num_audio_codebooks == 0
        || topology.audio_vocab_with_pad <= AUDIO_PAD_CODE as usize
        || topology.text_vocab_size <= AUDIO_ASSISTANT_DELAY_SLOT_TOKEN_ID as usize
    {
        return Err(VokraError::InvalidArgument(format!(
            "{label}: authenticated generation topology cannot represent the official token/code sentinels"
        )));
    }
    let prompt_rows = prompt.len() / columns;
    if prompt_rows
        .checked_add(options.max_new_tokens)
        .is_none_or(|total| total > topology.max_position_embeddings)
    {
        return Err(VokraError::InvalidArgument(format!(
            "{label}: prompt rows {prompt_rows} + max_new_tokens {} exceed max positions {}",
            options.max_new_tokens, topology.max_position_embeddings
        )));
    }
    validate_sampler("text", &options.text_sampler, label)?;
    validate_sampler("audio", &options.audio_sampler, label)
}

fn validate_sampler(kind: &str, config: &SamplerConfig, label: &str) -> Result<()> {
    if !config.temperature.is_finite() || config.temperature < 0.0 {
        return Err(VokraError::InvalidArgument(format!(
            "{label}: {kind} temperature must be finite and non-negative, got {}",
            config.temperature
        )));
    }
    if config.top_k == Some(0) {
        return Err(VokraError::InvalidArgument(format!(
            "{label}: {kind} top_k must be positive when present"
        )));
    }
    if config
        .top_p
        .is_some_and(|top_p| !top_p.is_finite() || top_p <= 0.0 || top_p > 1.0)
    {
        return Err(VokraError::InvalidArgument(format!(
            "{label}: {kind} top_p must be in (0,1] when present"
        )));
    }
    if config
        .repetition_penalty
        .is_some_and(|penalty| !penalty.is_finite() || penalty <= 0.0)
    {
        return Err(VokraError::InvalidArgument(format!(
            "{label}: {kind} repetition penalty must be finite and positive"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_N_VQ: usize = 32;
    const TEST_COLUMNS: usize = 1 + TEST_N_VQ;
    const TEST_TEXT_VOCAB: usize = 155_648;

    fn prompt_before_assistant_audio(num_audio_codebooks: usize) -> Vec<u32> {
        let columns = 1 + num_audio_codebooks;
        let mut prompt = vec![AUDIO_PAD_CODE; 3 * columns];
        prompt[0] = IM_START_TOKEN_ID;
        prompt[columns] = PAD_TOKEN_ID;
        prompt[2 * columns] = PAD_TOKEN_ID;
        prompt
    }

    fn delay_frames(
        frames: &[u32],
        frame_count: usize,
        num_audio_codebooks: usize,
    ) -> MossTtsDelayGeneration {
        let columns = 1 + num_audio_codebooks;
        assert_eq!(frames.len(), frame_count * num_audio_codebooks);
        let delay_rows = frame_count + num_audio_codebooks - 1;
        let mut generated_rows = vec![AUDIO_PAD_CODE; delay_rows * columns];
        for row in generated_rows.chunks_exact_mut(columns) {
            row[0] = AUDIO_ASSISTANT_GEN_SLOT_TOKEN_ID;
        }
        for frame in 0..frame_count {
            for codebook in 0..num_audio_codebooks {
                generated_rows[(frame + codebook) * columns + 1 + codebook] =
                    frames[frame * num_audio_codebooks + codebook];
            }
        }
        MossTtsDelayGeneration {
            generated_rows,
            num_audio_codebooks,
        }
    }

    #[test]
    fn delay_drain_advances_one_codebook_per_row() {
        let mut state = DelayGenerationState {
            audio_length: 40,
            delayed_length: None,
            is_audio: true,
        };
        assert!((0..TEST_N_VQ).all(|codebook| state.audio_sampling_mask(codebook)));
        state.advance(AUDIO_ASSISTANT_DELAY_SLOT_TOKEN_ID, TEST_N_VQ);
        assert_eq!(state.delayed_length, Some(1));
        assert!(!state.audio_sampling_mask(0));
        assert!(state.audio_sampling_mask(1));
        for _ in 1..TEST_N_VQ {
            state.advance(AUDIO_ASSISTANT_DELAY_SLOT_TOKEN_ID, TEST_N_VQ);
        }
        assert_eq!(state.delayed_length, Some(TEST_N_VQ));
        assert!(!(0..TEST_N_VQ).any(|codebook| state.audio_sampling_mask(codebook)));
        state.advance(AUDIO_END_TOKEN_ID, TEST_N_VQ);
        assert_eq!(state.delayed_length, None);
        assert_eq!(state.audio_length, 0);
    }

    #[test]
    fn audio_text_mask_keeps_only_generation_and_delay_slots() {
        let mut logits = vec![0.0; TEST_TEXT_VOCAB];
        mask_text_logits(&mut logits, true, 1, TEST_TEXT_VOCAB, TEST_N_VQ, LABEL).unwrap();
        assert_eq!(logits[AUDIO_ASSISTANT_GEN_SLOT_TOKEN_ID as usize], 0.0);
        assert_eq!(logits[AUDIO_ASSISTANT_DELAY_SLOT_TOKEN_ID as usize], 0.0);
        assert_eq!(logits[AUDIO_START_TOKEN_ID as usize], f32::NEG_INFINITY);
    }

    #[test]
    fn repetition_penalty_is_unique_per_codebook() {
        let mut logits = vec![2.0, -2.0, 4.0];
        let mut history = vec![AUDIO_PAD_CODE; 3 * TEST_COLUMNS];
        history[1] = 0;
        history[TEST_COLUMNS + 1] = 0;
        history[2 * TEST_COLUMNS + 1] = 1;
        apply_repetition_penalty(&mut logits, &history, 0, TEST_COLUMNS, 2.0, LABEL).unwrap();
        assert_eq!(logits, vec![1.0, -4.0, 4.0]);
    }

    #[test]
    fn official_de_delay_restores_frame_major_codes_and_pad_segments() {
        let mut codes = (0..3 * TEST_N_VQ)
            .map(|index| (index % 1_024) as u32)
            .collect::<Vec<_>>();
        codes[TEST_N_VQ..2 * TEST_N_VQ].fill(AUDIO_PAD_CODE);
        let generated = delay_frames(&codes, 3, TEST_N_VQ);
        let parsed =
            de_delay_audio_segments(&prompt_before_assistant_audio(TEST_N_VQ), &generated, LABEL)
                .unwrap();
        assert_eq!(parsed.start_length, 0);
        assert_eq!(parsed.segments.len(), 2);
        assert_eq!(parsed.segments[0].frames, 1);
        assert_eq!(parsed.segments[0].codes, codes[..TEST_N_VQ]);
        assert_eq!(parsed.segments[1].frames, 1);
        assert_eq!(parsed.segments[1].codes, codes[2 * TEST_N_VQ..]);
    }

    #[test]
    fn de_delay_includes_prompt_prefix_for_continuation() {
        let codes = (0..2 * TEST_N_VQ)
            .map(|index| (index % 1_024) as u32)
            .collect::<Vec<_>>();
        let full = delay_frames(&codes, 2, TEST_N_VQ);
        let mut prompt = prompt_before_assistant_audio(TEST_N_VQ);
        prompt.extend_from_slice(&full.generated_rows[..TEST_COLUMNS]);
        let generated = MossTtsDelayGeneration {
            generated_rows: full.generated_rows[TEST_COLUMNS..].to_vec(),
            num_audio_codebooks: TEST_N_VQ,
        };
        let parsed = de_delay_audio_segments(&prompt, &generated, LABEL).unwrap();
        assert_eq!(parsed.start_length, 1);
        assert_eq!(parsed.segments.len(), 1);
        assert_eq!(parsed.segments[0].frames, 2);
        assert_eq!(parsed.segments[0].codes, codes);
    }

    #[test]
    fn de_delay_uses_authenticated_voice_generator_codebook_count() {
        let num_audio_codebooks = 16;
        let codes = (0..2 * num_audio_codebooks)
            .map(|index| (index % 1_024) as u32)
            .collect::<Vec<_>>();
        let generated = delay_frames(&codes, 2, num_audio_codebooks);
        let parsed = de_delay_audio_segments(
            &prompt_before_assistant_audio(num_audio_codebooks),
            &generated,
            "moss_tts/voice_generator",
        )
        .unwrap();
        assert_eq!(parsed.segments.len(), 1);
        assert_eq!(parsed.segments[0].frames, 2);
        assert_eq!(parsed.segments[0].codes, codes);
    }

    #[test]
    fn voice_generator_defaults_match_fixed_upstream_signature() {
        let options = MossTtsDelayGenerationOptions::voice_generator();
        assert_eq!(options.max_new_tokens, 1_000);
        assert_eq!(options.text_sampler.temperature, 1.5);
        assert_eq!(options.text_sampler.top_k, Some(50));
        assert_eq!(options.text_sampler.top_p, None);
        assert_eq!(options.audio_sampler.temperature, 1.5);
        assert_eq!(options.audio_sampler.top_k, Some(50));
        assert_eq!(options.audio_sampler.top_p, Some(0.6));
        assert_eq!(options.audio_sampler.repetition_penalty, Some(1.1));
    }
}
