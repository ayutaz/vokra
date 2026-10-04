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
        if text_ids.len() > MAX_POSITIONS {
            return Err(VokraError::InvalidArgument(format!(
                "vibevoice-realtime streaming text IDs must not exceed {MAX_POSITIONS}"
            )));
        }
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
        assert!(error.to_string().contains("text IDs must not exceed"));
        assert_eq!(state.text_steps(), 0);
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
