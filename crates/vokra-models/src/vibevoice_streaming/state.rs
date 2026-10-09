//! Source-authenticated input/state boundary for VibeVoice Realtime.
//!
//! The pinned upstream processor does not accept the ordinary processor call
//! for realtime input.  Its `process_input_with_cached_prompt` path creates
//! pseudo input IDs from the cached LM/TTS hidden-state lengths, then exposes
//! a separate newline-terminated text-token sequence to the staged model.
//! This module binds exactly that input contract without pretending to own a
//! Transformer KV cache or to execute the model.
//!
//! The implementation is intentionally limited to the semantics that are
//! visible in the fixed upstream source:
//!
//! - one text input per state step;
//! - pad-filled LM and TTS-LM pseudo inputs sized by the cached prompt;
//! - all-one attention masks and an all-false speech-position mask;
//! - `text.strip() + "\\n"` tokenization through the authenticated sidecar;
//! - no audio input: the pinned cached-prompt method sets `speech_inputs` to
//!   `None`.
//!
//! KV cache updates, TTS-LM forward, EOS classification, diffusion, acoustic
//! decoding, and numerical parity remain explicit follow-up gates.

use super::MAX_POSITIONS;
#[cfg(test)]
use super::tokenizer::SPEECH_START_ID;
use super::tokenizer::{VOCAB_SIZE, VibeVoiceRealtimeTokenizer};
use vokra_core::{Result, VokraError};

/// Number of text tokens consumed by one upstream Realtime text window.
///
/// Source: `vibevoice/modular/modeling_vibevoice_streaming_inference.py` at
/// `94da20d98b2fa7688e9cbfaf7692ddb4954f7600`, lines 25 and 667-669.
pub const TTS_TEXT_WINDOW_SIZE: usize = 5;

/// Maximum number of speech iterations interleaved after each upstream text
/// window before the upstream loop evaluates the next text window.
///
/// Source: `vibevoice/modular/modeling_vibevoice_streaming_inference.py` at
/// `94da20d98b2fa7688e9cbfaf7692ddb4954f7600`, lines 26 and 705-706.
pub const TTS_SPEECH_WINDOW_SIZE: usize = 6;

/// Lengths of the two cached prompt hidden-state sequences.
///
/// The upstream processor reads these lengths from
/// `cached_prompt["lm"]["last_hidden_state"]` and
/// `cached_prompt["tts_lm"]["last_hidden_state"]`; it does not derive them
/// from text.  The native boundary therefore accepts them explicitly and
/// never fabricates KV tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VibeVoiceStreamingPrompt {
    lm_cached_len: usize,
    tts_lm_cached_len: usize,
    streaming_pad_id: u32,
}

impl VibeVoiceStreamingPrompt {
    /// Creates a prompt descriptor from an authenticated tokenizer and cached
    /// hidden-state lengths.
    #[must_use]
    const fn new(lm_cached_len: usize, tts_lm_cached_len: usize, streaming_pad_id: u32) -> Self {
        Self {
            lm_cached_len,
            tts_lm_cached_len,
            streaming_pad_id,
        }
    }

    /// Creates a prompt descriptor using the pad ID authenticated by Qwen.
    pub fn from_tokenizer(
        tokenizer: &VibeVoiceRealtimeTokenizer,
        lm_cached_len: usize,
        tts_lm_cached_len: usize,
    ) -> Result<Self> {
        let prompt = Self::new(
            lm_cached_len,
            tts_lm_cached_len,
            tokenizer.streaming_pad_id(),
        );
        prompt.validate().map(|()| prompt)
    }

    /// Number of pseudo input positions for the base language LM.
    #[must_use]
    pub const fn lm_cached_len(self) -> usize {
        self.lm_cached_len
    }

    /// Number of pseudo input positions for the TTS LM.
    #[must_use]
    pub const fn tts_lm_cached_len(self) -> usize {
        self.tts_lm_cached_len
    }

    /// Authenticated VibeVoice streaming pad ID used in pseudo inputs.
    #[must_use]
    pub const fn streaming_pad_id(self) -> u32 {
        self.streaming_pad_id
    }

    fn validate(self) -> Result<()> {
        if self.lm_cached_len > MAX_POSITIONS || self.tts_lm_cached_len > MAX_POSITIONS {
            return Err(VokraError::InvalidArgument(format!(
                "vibevoice-realtime streaming cached prompt lengths must not exceed {MAX_POSITIONS}"
            )));
        }
        if self.streaming_pad_id >= VOCAB_SIZE as u32 {
            return Err(VokraError::InvalidArgument(format!(
                "vibevoice-realtime streaming prompt pad ID {} is outside vocabulary size {VOCAB_SIZE}",
                self.streaming_pad_id
            )));
        }
        Ok(())
    }
}

/// One source-authenticated text input prepared for a staged Realtime step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VibeVoiceStreamingInput {
    /// Pad-filled pseudo IDs for the base language LM.
    pub input_ids: Vec<u32>,
    /// All-one mask matching [`Self::input_ids`].
    pub attention_mask: Vec<bool>,
    /// Pad-filled pseudo IDs for the TTS LM.
    pub tts_lm_input_ids: Vec<u32>,
    /// All-one mask matching [`Self::tts_lm_input_ids`].
    pub tts_lm_attention_mask: Vec<bool>,
    /// Newline-terminated text IDs supplied to the streaming TTS stage.
    pub tts_text_ids: Vec<u32>,
    /// False for every pseudo TTS position in this text-only input.
    pub speech_input_mask: Vec<bool>,
}

impl VibeVoiceStreamingInput {
    fn from_text_ids(prompt: VibeVoiceStreamingPrompt, text_ids: Vec<u32>) -> Result<Self> {
        prompt.validate()?;
        validate_text_capacity(prompt, text_ids.len())?;
        validate_text_ids(&text_ids, prompt.streaming_pad_id)?;
        Ok(Self {
            input_ids: vec![prompt.streaming_pad_id; prompt.lm_cached_len],
            attention_mask: vec![true; prompt.lm_cached_len],
            tts_lm_input_ids: vec![prompt.streaming_pad_id; prompt.tts_lm_cached_len],
            tts_lm_attention_mask: vec![true; prompt.tts_lm_cached_len],
            tts_text_ids: text_ids,
            speech_input_mask: vec![false; prompt.tts_lm_cached_len],
        })
    }
}

/// One source-authenticated text window in the Realtime generation loop.
///
/// This is a finite text-plan descriptor only. It does not mean that
/// generation terminates after this window, and it does not contain a
/// Transformer KV cache, hidden states, speech latents, or audio. The caller
/// must still run the corresponding positive/negative LM, diffusion, EOS, and
/// acoustic stages with the correct backend and explicit error handling.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VibeVoiceStreamingTextWindow {
    index: usize,
    text_ids: Vec<u32>,
    next_text_window_size: usize,
}

impl VibeVoiceStreamingTextWindow {
    /// Zero-based text-window index in the upstream generation loop.
    #[must_use]
    pub const fn index(&self) -> usize {
        self.index
    }

    /// Authenticated text IDs consumed by this window.
    #[must_use]
    pub fn text_ids(&self) -> &[u32] {
        &self.text_ids
    }

    /// Number of text IDs consumed by this window.
    #[must_use]
    pub const fn text_window_size(&self) -> usize {
        self.text_ids.len()
    }

    /// Number of text IDs in the next window, or zero for the final window.
    ///
    /// The upstream loop uses this value when updating the next text cache
    /// position; it is exposed as data and is not interpreted as a cache.
    #[must_use]
    pub const fn next_text_window_size(&self) -> usize {
        self.next_text_window_size
    }

    /// Maximum number of speech iterations in the upstream loop after this
    /// text window, before the next loop condition is evaluated.
    #[must_use]
    pub const fn max_speech_steps_after_text(&self) -> usize {
        TTS_SPEECH_WINDOW_SIZE
    }

    /// Whether this is the last text window in the supplied script.
    ///
    /// This is not a generation-complete signal. The upstream `while True`
    /// loop may execute further zero-text speech iterations after this window.
    #[must_use]
    pub const fn is_last_text_window(&self) -> bool {
        self.next_text_window_size == 0
    }
}

/// The termination condition for the zero-text continuation after text IDs
/// are exhausted.
///
/// Source: the pinned upstream loop checks external stop and finished/EOS
/// state before slicing the next text window, and separately enforces the
/// model maximum length.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VibeVoiceStreamingZeroTextContinuation {
    /// Continue up to six speech-only iterations until EOS, max length, or
    /// external stop.
    UntilEosMaxLengthOrExternalStop,
}

impl VibeVoiceStreamingZeroTextContinuation {
    /// Maximum number of speech iterations in each zero-text continuation
    /// loop.
    #[must_use]
    pub const fn max_speech_steps_per_iteration(self) -> usize {
        TTS_SPEECH_WINDOW_SIZE
    }
}

/// Source-authenticated, model-free text-window plan for one Realtime script.
///
/// This plan is intentionally finite because it describes text windows only.
/// [`Self::zero_text_continuation`] records the separate runtime phase that
/// can follow the last text window; it is not expanded into an invented number
/// of speech steps.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VibeVoiceStreamingTextPlan {
    windows: Vec<VibeVoiceStreamingTextWindow>,
    zero_text_continuation: VibeVoiceStreamingZeroTextContinuation,
}

impl VibeVoiceStreamingTextPlan {
    /// Returns text windows in the exact order consumed by the upstream loop.
    #[must_use]
    pub fn text_windows(&self) -> &[VibeVoiceStreamingTextWindow] {
        &self.windows
    }

    /// Number of text windows in this schedule.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.windows.len()
    }

    /// Whether the schedule has no windows.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.windows.is_empty()
    }

    /// Returns the explicit post-text continuation condition.
    #[must_use]
    pub const fn zero_text_continuation(&self) -> VibeVoiceStreamingZeroTextContinuation {
        self.zero_text_continuation
    }
}

/// Native bookkeeping for repeated cached-prompt input steps.
///
/// This is deliberately not a KV cache.  It records only the prompt shape and
/// successfully prepared text-step counts, so callers cannot mistake an
/// input contract for model execution.  State is advanced only after all
/// token and prompt validation succeeds.
#[derive(Debug, Clone)]
pub struct VibeVoiceStreamingState {
    prompt: VibeVoiceStreamingPrompt,
    text_steps: usize,
    text_tokens: usize,
}

impl VibeVoiceStreamingState {
    /// Starts a state machine from the cached prompt descriptor.
    #[must_use]
    pub const fn new(prompt: VibeVoiceStreamingPrompt) -> Self {
        Self {
            prompt,
            text_steps: 0,
            text_tokens: 0,
        }
    }

    /// Returns the fixed prompt descriptor for this stream.
    #[must_use]
    pub const fn prompt(&self) -> VibeVoiceStreamingPrompt {
        self.prompt
    }

    /// Number of successfully prepared text steps.
    #[must_use]
    pub const fn text_steps(&self) -> usize {
        self.text_steps
    }

    /// Number of text IDs prepared across successful steps.
    #[must_use]
    pub const fn text_tokens(&self) -> usize {
        self.text_tokens
    }

    /// Tokenizes and prepares one upstream streaming text step.
    pub fn prepare_text(
        &mut self,
        tokenizer: &VibeVoiceRealtimeTokenizer,
        text: &str,
    ) -> Result<VibeVoiceStreamingInput> {
        let text_ids = tokenizer.streaming_text_ids(text)?;
        self.prepare_text_ids(text_ids)
    }

    /// Plans the upstream text windows for authenticated text IDs without
    /// executing either language model.
    ///
    /// The pinned upstream implementation slices `tts_text_ids` into windows
    /// of five IDs and runs six speech iterations after each window. This
    /// method records that ordering and the next-window size only. After the
    /// last window, callers must honor the explicit zero-text continuation
    /// condition; this method does not claim to update a KV cache or provide
    /// synthesis/parity.
    pub fn plan_text_windows(&self, text_ids: &[u32]) -> Result<VibeVoiceStreamingTextPlan> {
        self.prompt.validate()?;
        validate_text_capacity(self.prompt, text_ids.len())?;
        validate_text_ids(text_ids, self.prompt.streaming_pad_id)?;

        let mut windows = Vec::with_capacity(text_ids.len().div_ceil(TTS_TEXT_WINDOW_SIZE));
        for (index, current) in text_ids.chunks(TTS_TEXT_WINDOW_SIZE).enumerate() {
            let next_text_window_size = text_ids
                .get((index + 1) * TTS_TEXT_WINDOW_SIZE..)
                .map_or(0, |remaining| remaining.len().min(TTS_TEXT_WINDOW_SIZE));
            windows.push(VibeVoiceStreamingTextWindow {
                index,
                text_ids: current.to_vec(),
                next_text_window_size,
            });
        }
        Ok(VibeVoiceStreamingTextPlan {
            windows,
            zero_text_continuation:
                VibeVoiceStreamingZeroTextContinuation::UntilEosMaxLengthOrExternalStop,
        })
    }

    fn prepare_text_ids(&mut self, text_ids: Vec<u32>) -> Result<VibeVoiceStreamingInput> {
        let input = VibeVoiceStreamingInput::from_text_ids(self.prompt, text_ids)?;
        let next_steps = self.text_steps.checked_add(1).ok_or_else(|| {
            VokraError::InvalidArgument("vibevoice-realtime streaming step counter overflow".into())
        })?;
        let next_tokens = self
            .text_tokens
            .checked_add(input.tts_text_ids.len())
            .ok_or_else(|| {
                VokraError::InvalidArgument(
                    "vibevoice-realtime streaming text counter overflow".into(),
                )
            })?;
        self.text_steps = next_steps;
        self.text_tokens = next_tokens;
        Ok(input)
    }
}

fn validate_text_capacity(prompt: VibeVoiceStreamingPrompt, text_len: usize) -> Result<()> {
    let lm_total = prompt.lm_cached_len.checked_add(text_len).ok_or_else(|| {
        VokraError::InvalidArgument(
            "vibevoice-realtime streaming prompt + text length overflow".into(),
        )
    })?;
    let tts_lm_total = prompt
        .tts_lm_cached_len
        .checked_add(text_len)
        .ok_or_else(|| {
            VokraError::InvalidArgument(
                "vibevoice-realtime streaming prompt + text length overflow".into(),
            )
        })?;
    if lm_total > MAX_POSITIONS || tts_lm_total > MAX_POSITIONS {
        return Err(VokraError::InvalidArgument(format!(
            "vibevoice-realtime streaming prompt plus text must not exceed {MAX_POSITIONS} positions"
        )));
    }
    Ok(())
}

fn validate_text_ids(text_ids: &[u32], streaming_pad_id: u32) -> Result<()> {
    if text_ids.is_empty() {
        return Err(VokraError::InvalidArgument(
            "vibevoice-realtime streaming text IDs must not be empty".into(),
        ));
    }
    if text_ids.iter().any(|&id| id >= VOCAB_SIZE as u32) {
        return Err(VokraError::InvalidArgument(
            "vibevoice-realtime streaming text ID is outside the authenticated vocabulary".into(),
        ));
    }
    if text_ids
        .iter()
        .any(|&id| VibeVoiceRealtimeTokenizer::is_speech_boundary(id))
    {
        return Err(VokraError::InvalidArgument(
            "vibevoice-realtime streaming text IDs contain a reserved speech boundary".into(),
        ));
    }
    if text_ids.contains(&streaming_pad_id) {
        return Err(VokraError::InvalidArgument(
            "vibevoice-realtime streaming text IDs contain reserved <|image_pad|>".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
pub(crate) fn synthetic_generation_test_state() -> VibeVoiceStreamingState {
    VibeVoiceStreamingState::new(VibeVoiceStreamingPrompt::new(1, 1, 7))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cached_prompt_input_matches_processor_contract() {
        let prompt = VibeVoiceStreamingPrompt::new(3, 5, 7);
        let mut state = VibeVoiceStreamingState::new(prompt);
        let input = state.prepare_text_ids(vec![11, 12, 13]).unwrap();

        assert_eq!(input.input_ids, vec![7; 3]);
        assert_eq!(input.attention_mask, vec![true; 3]);
        assert_eq!(input.tts_lm_input_ids, vec![7; 5]);
        assert_eq!(input.tts_lm_attention_mask, vec![true; 5]);
        assert_eq!(input.tts_text_ids, vec![11, 12, 13]);
        assert_eq!(input.speech_input_mask, vec![false; 5]);
        assert_eq!(state.text_steps(), 1);
        assert_eq!(state.text_tokens(), 3);
    }

    #[test]
    fn invalid_step_does_not_advance_state() {
        let mut state = VibeVoiceStreamingState::new(VibeVoiceStreamingPrompt::new(1, 1, 7));
        let error = state.prepare_text_ids(vec![SPEECH_START_ID]).unwrap_err();
        assert!(error.to_string().contains("reserved speech boundary"));
        assert_eq!(state.text_steps(), 0);
        assert_eq!(state.text_tokens(), 0);
    }

    #[test]
    fn reserved_streaming_pad_id_does_not_enter_text() {
        let mut state = VibeVoiceStreamingState::new(VibeVoiceStreamingPrompt::new(1, 1, 7));
        let error = state.prepare_text_ids(vec![7]).unwrap_err();
        assert!(error.to_string().contains("<|image_pad|>"));
        assert_eq!(state.text_steps(), 0);
        assert_eq!(state.text_tokens(), 0);
    }

    #[test]
    fn counters_commit_atomically_on_overflow() {
        let mut state = VibeVoiceStreamingState::new(VibeVoiceStreamingPrompt::new(1, 1, 7));
        state.text_steps = usize::MAX;
        let error = state.prepare_text_ids(vec![1]).unwrap_err();
        assert!(error.to_string().contains("step counter overflow"));
        assert_eq!(state.text_steps, usize::MAX);
        assert_eq!(state.text_tokens, 0);

        state.text_steps = 0;
        state.text_tokens = usize::MAX;
        let error = state.prepare_text_ids(vec![1]).unwrap_err();
        assert!(error.to_string().contains("text counter overflow"));
        assert_eq!(state.text_steps, 0);
        assert_eq!(state.text_tokens, usize::MAX);
    }

    #[test]
    fn prompt_rejects_out_of_range_pad_id() {
        let mut state =
            VibeVoiceStreamingState::new(VibeVoiceStreamingPrompt::new(1, 1, VOCAB_SIZE as u32));
        let error = state.prepare_text_ids(vec![1]).unwrap_err();
        assert!(error.to_string().contains("outside vocabulary"));
        assert_eq!(state.text_steps(), 0);
    }

    #[test]
    fn oversized_cached_prompt_is_rejected_before_allocation() {
        let mut state =
            VibeVoiceStreamingState::new(VibeVoiceStreamingPrompt::new(MAX_POSITIONS + 1, 1, 7));
        let error = state.prepare_text_ids(vec![1]).unwrap_err();
        assert!(error.to_string().contains("cached prompt lengths"));
        assert_eq!(state.text_steps(), 0);
    }

    #[test]
    fn oversized_text_ids_are_rejected_before_pseudo_input_allocation() {
        let mut state = VibeVoiceStreamingState::new(VibeVoiceStreamingPrompt::new(1, 1, 7));
        let error = state
            .prepare_text_ids(vec![1; MAX_POSITIONS + 1])
            .unwrap_err();
        assert!(error.to_string().contains("prompt plus text"));
        assert_eq!(state.text_steps(), 0);
    }

    #[test]
    fn source_text_plan_slices_text_and_marks_speech_steps() {
        let state = VibeVoiceStreamingState::new(VibeVoiceStreamingPrompt::new(1, 1, 7));
        let text_ids: Vec<u32> = (101..=112).collect();
        let plan = state.plan_text_windows(&text_ids).unwrap();

        assert_eq!(plan.len(), 3);
        assert_eq!(plan.text_windows()[0].index(), 0);
        assert_eq!(
            plan.text_windows()[0].text_ids(),
            &[101, 102, 103, 104, 105]
        );
        assert_eq!(plan.text_windows()[0].text_window_size(), 5);
        assert_eq!(plan.text_windows()[0].next_text_window_size(), 5);
        assert_eq!(plan.text_windows()[0].max_speech_steps_after_text(), 6);
        assert!(!plan.text_windows()[0].is_last_text_window());

        assert_eq!(plan.text_windows()[1].index(), 1);
        assert_eq!(
            plan.text_windows()[1].text_ids(),
            &[106, 107, 108, 109, 110]
        );
        assert_eq!(plan.text_windows()[1].next_text_window_size(), 2);

        assert_eq!(plan.text_windows()[2].index(), 2);
        assert_eq!(plan.text_windows()[2].text_ids(), &[111, 112]);
        assert_eq!(plan.text_windows()[2].next_text_window_size(), 0);
        assert!(plan.text_windows()[2].is_last_text_window());

        // A last text window is not generation completion: the upstream loop
        // can continue with zero text until EOS, max length, or external stop.
        assert_eq!(
            plan.zero_text_continuation(),
            VibeVoiceStreamingZeroTextContinuation::UntilEosMaxLengthOrExternalStop
        );
        assert_eq!(
            plan.zero_text_continuation()
                .max_speech_steps_per_iteration(),
            6
        );

        // Planning is model-free and does not consume the input state.
        assert_eq!(state.text_steps(), 0);
        assert_eq!(state.text_tokens(), 0);
    }

    #[test]
    fn source_text_plan_preserves_short_last_window() {
        let state = VibeVoiceStreamingState::new(VibeVoiceStreamingPrompt::new(1, 1, 7));
        let plan = state.plan_text_windows(&[1, 2, 3, 4, 5]).unwrap();

        assert_eq!(plan.len(), 1);
        assert_eq!(plan.text_windows()[0].text_window_size(), 5);
        assert_eq!(plan.text_windows()[0].next_text_window_size(), 0);
        assert_eq!(plan.text_windows()[0].max_speech_steps_after_text(), 6);
        assert_eq!(
            plan.zero_text_continuation(),
            VibeVoiceStreamingZeroTextContinuation::UntilEosMaxLengthOrExternalStop
        );
        assert_eq!(
            plan.zero_text_continuation()
                .max_speech_steps_per_iteration(),
            6
        );
    }

    #[test]
    fn source_text_plan_rejects_invalid_text_without_advancing_state() {
        let state = VibeVoiceStreamingState::new(VibeVoiceStreamingPrompt::new(1, 1, 7));
        let error = state.plan_text_windows(&[1, SPEECH_START_ID]).unwrap_err();

        assert!(error.to_string().contains("reserved speech boundary"));
        assert_eq!(state.text_steps(), 0);
        assert_eq!(state.text_tokens(), 0);
    }

    #[test]
    fn source_text_plan_checks_prompt_plus_text_capacity() {
        let state = VibeVoiceStreamingState::new(VibeVoiceStreamingPrompt::new(
            MAX_POSITIONS,
            MAX_POSITIONS,
            7,
        ));
        let error = state.plan_text_windows(&[1]).unwrap_err();

        assert!(error.to_string().contains("prompt plus text"));
    }

    #[test]
    #[ignore = "requires exact VAST-recovered Qwen sidecars; no local tokenizer execution"]
    fn fixed_sidecars_bind_cached_prompt_state() {
        let root = std::env::var_os("VOKRA_VIBEVOICE_TOKENIZER_DIR")
            .map(std::path::PathBuf::from)
            .expect("VOKRA_VIBEVOICE_TOKENIZER_DIR must point at exact sidecars");
        let tokenizer = VibeVoiceRealtimeTokenizer::from_files(&root).unwrap();
        // Independently obtained from the pinned upstream
        // VibeVoiceTextTokenizerFast.pad_id property and checked against the
        // same fixed sidecar snapshot on VAST.
        assert_eq!(tokenizer.streaming_pad_id(), 151_655);

        let prompt = VibeVoiceStreamingPrompt::from_tokenizer(&tokenizer, 3, 5).unwrap();
        let mut state = VibeVoiceStreamingState::new(prompt);
        let input = state.prepare_text(&tokenizer, "hello").unwrap();

        assert_eq!(input.input_ids, vec![151_655; 3]);
        assert_eq!(input.attention_mask, vec![true; 3]);
        assert_eq!(input.tts_lm_input_ids, vec![151_655; 5]);
        assert_eq!(input.tts_lm_attention_mask, vec![true; 5]);
        assert_eq!(input.tts_text_ids, vec![14_990, 198]);
        assert_eq!(input.speech_input_mask, vec![false; 5]);
        assert_eq!(state.text_steps(), 1);
        assert_eq!(state.text_tokens(), 2);
    }
}
