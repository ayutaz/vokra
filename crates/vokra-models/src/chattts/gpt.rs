//! Source-aligned ChatTTS GPT prompt and sampling contracts.
//!
//! The fixed v0.2.5 release uses a Llama-shaped 20-layer decoder, but its
//! text/audio embeddings are external composite inputs.  This module keeps
//! those boundaries typed and deterministic without pretending that a
//! GPT-only GGUF is a complete ChatTTS checkpoint.

use vokra_core::backend::BackendKind;
use vokra_core::{Result, VokraError};

/// ChatTTS text vocabulary size from the authenticated GPT config.
pub const TEXT_VOCAB_SIZE: usize = 21_178;
/// ChatTTS audio vocabulary size from the authenticated GPT config.
pub const AUDIO_VOCAB_SIZE: usize = 626;
/// Number of audio VQ codebooks emitted per GPT position.
pub const AUDIO_CODEBOOKS: usize = 4;
/// GPT hidden width from the authenticated GPT config.
pub const HIDDEN_SIZE: usize = 768;
/// GPT intermediate width from the authenticated GPT config.
pub const INTERMEDIATE_SIZE: usize = 3_072;
/// Number of decoder layers from the authenticated GPT config.
pub const NUM_LAYERS: usize = 20;
/// Number of attention heads from the authenticated GPT config.
pub const NUM_HEADS: usize = 12;
/// Maximum source position from the authenticated GPT config.
pub const MAX_POSITION_EMBEDDINGS: usize = 4_096;
/// Source EOS code for each generated audio row.
pub const EOS_AUDIO_CODE: u32 = 625;

/// Fixed source GPT topology; external embedding tables remain required.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChatTtsGptConfig {
    /// External text embedding width.
    pub text_vocab: usize,
    /// External audio embedding width.
    pub audio_vocab: usize,
    /// Number of audio codebook rows.
    pub audio_codebooks: usize,
    /// Decoder hidden width.
    pub hidden: usize,
    /// Decoder intermediate width.
    pub intermediate: usize,
    /// Decoder layer count.
    pub layers: usize,
    /// Attention head count.
    pub heads: usize,
    /// Maximum causal position.
    pub max_positions: usize,
    /// Speaker replacement embedding width from the source GPT config.
    pub speaker_embedding_dim: usize,
}

impl Default for ChatTtsGptConfig {
    fn default() -> Self {
        Self {
            text_vocab: TEXT_VOCAB_SIZE,
            audio_vocab: AUDIO_VOCAB_SIZE,
            audio_codebooks: AUDIO_CODEBOOKS,
            hidden: HIDDEN_SIZE,
            intermediate: INTERMEDIATE_SIZE,
            layers: NUM_LAYERS,
            heads: NUM_HEADS,
            max_positions: MAX_POSITION_EMBEDDINGS,
            speaker_embedding_dim: 192,
        }
    }
}

impl ChatTtsGptConfig {
    /// Validates the exact source topology, rejecting substitution defaults.
    pub fn validate(&self) -> Result<()> {
        if *self != Self::default() {
            return Err(VokraError::ModelLoad(
                "chattts: GPT topology differs from the authenticated v0.2.5 config".to_owned(),
            ));
        }
        Ok(())
    }
}

/// Source sampling controls. Every random value is supplied by the caller.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChatTtsSamplingConfig {
    /// Minimum generated positions before EOS is accepted.
    pub min_new_tokens: usize,
    /// Maximum generated positions.
    pub max_new_tokens: usize,
    /// Top-p nucleus threshold.
    pub top_p: f32,
    /// Top-k cutoff; zero disables top-k.
    pub top_k: usize,
    /// Repetition penalty applied before top-k/top-p filtering.
    pub repetition_penalty: f32,
    /// Per-codebook temperatures.
    pub temperatures: [f32; AUDIO_CODEBOOKS],
    /// Whether the source retries an immediate EOS when no seed was supplied.
    pub ensure_non_empty: bool,
    /// Number of generated positions between streaming boundaries.
    pub stream_batch: usize,
    /// Maximum waveform samples exposed per streaming boundary.
    pub stream_speed: usize,
    /// Number of initial streaming batches withheld by the source.
    pub pass_first_n_batches: usize,
}

impl Default for ChatTtsSamplingConfig {
    fn default() -> Self {
        Self {
            min_new_tokens: 0,
            max_new_tokens: 2_048,
            top_p: 0.7,
            top_k: 20,
            repetition_penalty: 1.05,
            temperatures: [0.3; AUDIO_CODEBOOKS],
            ensure_non_empty: true,
            stream_batch: 24,
            stream_speed: 12_000,
            pass_first_n_batches: 2,
        }
    }
}

impl ChatTtsSamplingConfig {
    /// Validates source-compatible ranges and all four temperatures.
    pub fn validate(&self) -> Result<()> {
        if self.max_new_tokens == 0 || self.min_new_tokens > self.max_new_tokens {
            return Err(VokraError::InvalidArgument(
                "chattts: invalid min/max generation bounds".to_owned(),
            ));
        }
        if self.stream_batch == 0 || self.stream_speed == 0 {
            return Err(VokraError::InvalidArgument(
                "chattts: streaming boundaries must be positive".to_owned(),
            ));
        }
        if !self.top_p.is_finite() || !(0.0..=1.0).contains(&self.top_p) {
            return Err(VokraError::InvalidArgument(
                "chattts: top_p must be finite and in [0,1]".to_owned(),
            ));
        }
        if !self.repetition_penalty.is_finite() || self.repetition_penalty <= 0.0 {
            return Err(VokraError::InvalidArgument(
                "chattts: repetition_penalty must be finite and positive".to_owned(),
            ));
        }
        if self
            .temperatures
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err(VokraError::InvalidArgument(
                "chattts: all four codebook temperatures must be finite and positive".to_owned(),
            ));
        }
        Ok(())
    }
}

/// External prompt embeddings and masks required by the composite source.
#[derive(Debug, Clone)]
pub struct ChatTtsPrompt {
    /// Text token IDs from the authenticated WordPiece tokenizer.
    pub text_tokens: Vec<u32>,
    /// Audio code rows, flattened position-major with four codebooks.
    pub audio_tokens: Vec<[u32; AUDIO_CODEBOOKS]>,
    /// Text/audio mask matching the combined embedding sequence.
    pub text_mask: Vec<bool>,
    /// Optional source speaker embedding replacement.
    pub speaker_embedding: Option<Vec<f32>>,
}

impl ChatTtsPrompt {
    /// Validates prompt bounds without filling missing embeddings or masks.
    pub fn validate(&self, config: ChatTtsGptConfig) -> Result<()> {
        config.validate()?;
        if self.text_tokens.len() + self.audio_tokens.len() > config.max_positions {
            return Err(VokraError::InvalidArgument(
                "chattts: prompt exceeds authenticated GPT context".to_owned(),
            ));
        }
        if self
            .text_tokens
            .iter()
            .any(|token| *token as usize >= config.text_vocab)
            || self
                .audio_tokens
                .iter()
                .flatten()
                .any(|token| *token as usize >= config.audio_vocab)
        {
            return Err(VokraError::InvalidArgument(
                "chattts: prompt token is outside the authenticated vocabulary".to_owned(),
            ));
        }
        if self.text_mask.len() != self.text_tokens.len() + self.audio_tokens.len() {
            return Err(VokraError::InvalidArgument(
                "chattts: text/audio mask must match the combined prompt sequence".to_owned(),
            ));
        }
        if let Some(embedding) = &self.speaker_embedding {
            if embedding.len() != config.speaker_embedding_dim
                || embedding.iter().any(|value| !value.is_finite())
            {
                return Err(VokraError::InvalidArgument(
                    "chattts: speaker embedding must be finite width 192".to_owned(),
                ));
            }
        }
        Ok(())
    }
}

/// A future authenticated GPT session; public construction remains gated.
#[derive(Debug)]
pub struct ChatTtsGptSession {
    backend: BackendKind,
    config: ChatTtsGptConfig,
}

impl ChatTtsGptSession {
    /// Refuses an unauthenticated GPT-only artifact rather than synthesizing.
    pub fn from_authenticated_bundle(_bundle: &[u8], backend: BackendKind) -> Result<Self> {
        let _ = backend;
        Err(VokraError::UnsupportedOp(
            "chattts: composite GPT session requires VAST-authenticated Embed/DVAE/Decoder/Vocos/tokenizer evidence".to_owned(),
        ))
    }

    /// Returns the selected backend for future learned-op dispatch.
    #[must_use]
    pub const fn backend(&self) -> BackendKind {
        self.backend
    }

    /// Returns the fixed GPT topology.
    #[must_use]
    pub const fn config(&self) -> ChatTtsGptConfig {
        self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_source_topology_and_sampling_contract() {
        ChatTtsGptConfig::default().validate().unwrap();
        ChatTtsSamplingConfig::default().validate().unwrap();
        assert_eq!(ChatTtsGptConfig::default().audio_codebooks, 4);
        assert_eq!(EOS_AUDIO_CODE, AUDIO_VOCAB_SIZE as u32 - 1);
        assert_eq!(ChatTtsSamplingConfig::default().max_new_tokens, 2_048);
        assert_eq!(ChatTtsSamplingConfig::default().repetition_penalty, 1.05);
    }

    #[test]
    fn prompt_rejects_substitution_and_unsafe_tokens() {
        let mut prompt = ChatTtsPrompt {
            text_tokens: vec![TEXT_VOCAB_SIZE as u32],
            audio_tokens: Vec::new(),
            text_mask: vec![true],
            speaker_embedding: None,
        };
        assert!(prompt.validate(ChatTtsGptConfig::default()).is_err());
        prompt.text_tokens[0] = 1;
        assert!(prompt.validate(ChatTtsGptConfig::default()).is_ok());
        prompt.text_mask.clear();
        assert!(prompt.validate(ChatTtsGptConfig::default()).is_err());
        let mut config = ChatTtsSamplingConfig::default();
        config.temperatures[3] = 0.0;
        assert!(config.validate().is_err());
    }
}
