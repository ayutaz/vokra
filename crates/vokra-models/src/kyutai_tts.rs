//! Source-authenticated pure contract seams for Kyutai TTS 1.6B EN/FR.
//!
//! This module intentionally does not implement a GGUF binder, tensor topology,
//! model forward, Mimi decode, or audio generation. The checked-in inspector
//! authenticates artifact/source identities and scalar config facts, but it
//! does not authenticate tensor names/shapes or a native runtime. Those gates
//! remain explicit follow-up work.

use std::collections::VecDeque;

use vokra_core::{Result, VokraError};

/// Authenticated Hugging Face model identity.
pub const HF_REPOSITORY: &str = "kyutai/tts-1.6b-en_fr";
/// Authenticated immutable revision of [`HF_REPOSITORY`].
pub const HF_REVISION: &str = "f65439609986c392cb12df63938abcc550c3fb15";
/// Authenticated TTS weight filename.
pub const TTS_FILE: &str = "dsm_tts_1e68beda@240.safetensors";
/// Authenticated byte size of [`TTS_FILE`].
pub const TTS_BYTES: u64 = 3_683_719_712;
/// Authenticated SHA-256 digest of [`TTS_FILE`].
pub const TTS_SHA256: &str = "726ddadd90a080c89cbc6b217745296ef32d8e25666d30f81a09e8ae5c9e0f0c";
/// Authenticated Mimi checkpoint filename.
pub const MIMI_FILE: &str = "tokenizer-e351c8d8-checkpoint125.safetensors";
/// Authenticated byte size of [`MIMI_FILE`].
pub const MIMI_BYTES: u64 = 384_644_900;
/// Authenticated SHA-256 digest of [`MIMI_FILE`].
pub const MIMI_SHA256: &str = "09b782f0629851a271227fb9d36db65c041790365f11bbe5d3d59369cf863f50";
/// Authenticated SentencePiece model filename.
pub const SPM_FILE: &str = "tokenizer_spm_8k_en_fr_audio.model";
/// Authenticated byte size of [`SPM_FILE`].
pub const SPM_BYTES: u64 = 120_378;
/// Authenticated SHA-256 digest of [`SPM_FILE`].
pub const SPM_SHA256: &str = "cd87dd5d17169151782ac700280ec057e5d658a9afbe238a048ea5ff318cce69";
/// Authenticated voice repository identity.
pub const VOICE_REPOSITORY: &str = "kyutai/tts-voices";
/// Authenticated immutable revision of [`VOICE_REPOSITORY`].
pub const VOICE_REVISION: &str = "323332d33f997de8394f24a193e1a76df720e01a";
/// Authenticated voice tensor filename.
pub const VOICE_FILE: &str = "voice-donations/robert.wav.1e68beda@240.safetensors";
/// Authenticated byte size of [`VOICE_FILE`].
pub const VOICE_BYTES: u64 = 256_136;
/// Authenticated SHA-256 digest of [`VOICE_FILE`].
pub const VOICE_SHA256: &str = "bc79b0162c94862aadd6c5d351b5b4984274af0616e3a56b0df9973ff7c793c7";
/// Authenticated Moshi source repository URL.
pub const MOSHI_SOURCE: &str = "https://github.com/kyutai-labs/moshi.git";
/// Authenticated immutable Moshi source revision.
pub const MOSHI_REVISION: &str = "e6a55d2722a65870ef52a6c9f6ecfc0e90f38362";
/// Authenticated delayed-streams-modeling source repository URL.
pub const DSM_SOURCE: &str = "https://github.com/kyutai-labs/delayed-streams-modeling.git";
/// Authenticated immutable delayed-streams-modeling source revision.
pub const DSM_REVISION: &str = "4c4f65e147df056adf3346290d64c7b9649b18c9";

/// Number of authenticated audio codebook channels.
pub const N_Q: usize = 32;
/// Total stream channels, including the text channel.
pub const STREAM_CHANNELS: usize = N_Q + 1;
/// Authenticated audio token-card size.
pub const CARD: u32 = 2048;
/// Authenticated text token-card size.
pub const TEXT_CARD: u32 = 8000;
/// Authenticated text padding token ID.
pub const TEXT_PADDING_ID: u32 = 3;
/// Authenticated lead of the delayed second stream, in token steps.
pub const SECOND_STREAM_AHEAD: u32 = 2;

const DEPFORMER_WEIGHT_SCHEDULE: [u32; N_Q] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10,
    10, 10,
];

const fn authenticated_delays() -> [u32; STREAM_CHANNELS] {
    let mut delays = [SECOND_STREAM_AHEAD; STREAM_CHANNELS];
    delays[0] = 0;
    delays[1] = 0;
    delays
}

/// Scalar config facts transcribed from the checked-in inspector.
#[derive(Debug, Clone, PartialEq)]
pub struct KyutaiTtsConfig {
    /// Authenticated model type string.
    pub model_type: String,
    /// Audio token-card size.
    pub card: u32,
    /// Number of audio codebook channels.
    pub n_q: usize,
    /// Number of depformer codebook channels.
    pub dep_q: usize,
    /// Main transformer dimension.
    pub dim: usize,
    /// Text token-card size.
    pub text_card: u32,
    /// Text padding token ID.
    pub text_padding_id: u32,
    /// Number of main transformer attention heads.
    pub num_heads: usize,
    /// Number of main transformer layers.
    pub num_layers: usize,
    /// Main transformer hidden scale.
    pub hidden_scale: f32,
    /// Main transformer context length.
    pub context: usize,
    /// Main transformer rotary-embedding period limit.
    pub max_period: u32,
    /// Whether the main transformer is causal.
    pub causal: bool,
    /// Main transformer gating function name.
    pub gating: String,
    /// Main transformer normalization name.
    pub norm: String,
    /// Main transformer positional-embedding name.
    pub positional_embedding: String,
    /// Depformer hidden dimension.
    pub depformer_dim: usize,
    /// Number of depformer attention heads.
    pub depformer_num_heads: usize,
    /// Number of depformer layers.
    pub depformer_num_layers: usize,
    /// Depformer feed-forward dimension.
    pub depformer_dim_feedforward: usize,
    /// Whether depformer projections are multi-linear.
    pub depformer_multi_linear: bool,
    /// Depformer positional-embedding name.
    pub depformer_pos_emb: String,
    /// Whether depformer weights vary by step.
    pub depformer_weights_per_step: bool,
    /// Depformer low-rank embedding dimension.
    pub depformer_low_rank_embeddings: usize,
    /// Authenticated second-stream demux flag from the inspector.
    pub demux_second_stream: bool,
    /// Whether cross-attention is enabled.
    pub cross_attention: bool,
    /// Authenticated audio delay in seconds.
    pub audio_delay: f32,
    /// Authenticated second-stream lead in token steps.
    pub second_stream_ahead: u32,
    /// Per-channel delay values used by the pure alignment seam.
    pub delays: Vec<u32>,
    /// Depformer weight index for each step.
    pub depformer_weight_schedule: Vec<u32>,
}

impl KyutaiTtsConfig {
    /// Returns the scalar configuration authenticated by the checked-in inspector.
    #[must_use]
    pub fn authenticated() -> Self {
        Self {
            model_type: "tts".to_owned(),
            card: CARD,
            n_q: N_Q,
            dep_q: N_Q,
            dim: 2048,
            text_card: TEXT_CARD,
            text_padding_id: TEXT_PADDING_ID,
            num_heads: 16,
            num_layers: 16,
            hidden_scale: 4.125,
            context: 500,
            max_period: 10_000,
            causal: true,
            gating: "silu".to_owned(),
            norm: "rms_norm_f32".to_owned(),
            positional_embedding: "rope".to_owned(),
            depformer_dim: 1024,
            depformer_num_heads: 16,
            depformer_num_layers: 4,
            depformer_dim_feedforward: 3072,
            depformer_multi_linear: true,
            depformer_pos_emb: "none".to_owned(),
            depformer_weights_per_step: true,
            depformer_low_rank_embeddings: 128,
            demux_second_stream: true,
            cross_attention: true,
            audio_delay: 1.28,
            second_stream_ahead: SECOND_STREAM_AHEAD,
            delays: authenticated_delays().to_vec(),
            depformer_weight_schedule: DEPFORMER_WEIGHT_SCHEDULE.to_vec(),
        }
    }

    /// Rejects any config that is not exactly the inspector's fixed contract.
    pub fn validate(&self) -> Result<()> {
        let expected = Self::authenticated();
        let exact = self.model_type == expected.model_type
            && self.card == expected.card
            && self.n_q == expected.n_q
            && self.dep_q == expected.dep_q
            && self.dim == expected.dim
            && self.text_card == expected.text_card
            && self.text_padding_id == expected.text_padding_id
            && self.num_heads == expected.num_heads
            && self.num_layers == expected.num_layers
            && self.hidden_scale.to_bits() == expected.hidden_scale.to_bits()
            && self.context == expected.context
            && self.max_period == expected.max_period
            && self.causal == expected.causal
            && self.gating == expected.gating
            && self.norm == expected.norm
            && self.positional_embedding == expected.positional_embedding
            && self.depformer_dim == expected.depformer_dim
            && self.depformer_num_heads == expected.depformer_num_heads
            && self.depformer_num_layers == expected.depformer_num_layers
            && self.depformer_dim_feedforward == expected.depformer_dim_feedforward
            && self.depformer_multi_linear == expected.depformer_multi_linear
            && self.depformer_pos_emb == expected.depformer_pos_emb
            && self.depformer_weights_per_step == expected.depformer_weights_per_step
            && self.depformer_low_rank_embeddings == expected.depformer_low_rank_embeddings
            && self.demux_second_stream == expected.demux_second_stream
            && self.cross_attention == expected.cross_attention
            && self.audio_delay.to_bits() == expected.audio_delay.to_bits()
            && self.second_stream_ahead == expected.second_stream_ahead
            && self.delays == expected.delays
            && self.depformer_weight_schedule == expected.depformer_weight_schedule;
        if exact {
            Ok(())
        } else {
            Err(VokraError::InvalidArgument(
                "kyutai-tts config does not match authenticated inspector contract".to_owned(),
            ))
        }
    }

    /// Returns the largest configured channel delay.
    #[must_use]
    pub fn max_delay(&self) -> u32 {
        self.delays.iter().copied().max().unwrap_or(0)
    }
}

/// Pure delay alignment only; this is not the source-level second-stream demux.
///
/// The inspector authenticates the delay vector but not the detailed demux
/// implementation. This seam therefore aligns delayed channels without
/// claiming demux, model forward, Mimi decode, or audio-generation behavior.
#[derive(Debug)]
pub struct KyutaiTtsDelayedStreamAligner {
    config: KyutaiTtsConfig,
    pending: VecDeque<[u32; STREAM_CHANNELS]>,
}

impl KyutaiTtsDelayedStreamAligner {
    /// Creates an aligner after validating the exact authenticated config.
    pub fn new(config: KyutaiTtsConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            config,
            pending: VecDeque::new(),
        })
    }

    /// Pushes one text-plus-audio token step and emits an aligned step if ready.
    ///
    /// Token values are checked against the text or audio card before they are
    /// queued; this helper performs delay alignment only, not model decoding.
    pub fn push(&mut self, step: [u32; STREAM_CHANNELS]) -> Result<Option<[u32; STREAM_CHANNELS]>> {
        for (channel, &token) in step.iter().enumerate() {
            let limit = if channel == 0 {
                self.config.text_card
            } else {
                self.config.card
            };
            if token >= limit {
                return Err(VokraError::InvalidArgument(format!(
                    "kyutai-tts token {token} is outside channel {channel} card {limit}"
                )));
            }
        }
        self.pending.push_back(step);
        let max_delay = self.config.max_delay() as usize;
        if self.pending.len() <= max_delay {
            return Ok(None);
        }
        let mut output = [0u32; STREAM_CHANNELS];
        for (channel, slot) in output.iter_mut().enumerate() {
            let delay = self.config.delays[channel] as usize;
            *slot = self.pending[max_delay - delay][channel];
        }
        self.pending.pop_front();
        Ok(Some(output))
    }

    /// Returns the number of queued input steps not yet emitted.
    #[must_use]
    pub fn pending_steps(&self) -> usize {
        self.pending.len()
    }
}

/// Returns the inspector-authenticated depformer weight table entry.
pub fn depformer_weight_index(step: usize) -> Result<u32> {
    DEPFORMER_WEIGHT_SCHEDULE.get(step).copied().ok_or_else(|| {
        VokraError::InvalidArgument(format!("kyutai-tts depformer step {step} is out of range"))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn authenticated_config_and_schedule_are_exact() {
        let config = KyutaiTtsConfig::authenticated();
        config.validate().unwrap();
        assert_eq!(config.delays.len(), STREAM_CHANNELS);
        assert_eq!(config.delays[0..2], [0, 0]);
        assert!(config.delays[2..].iter().all(|delay| *delay == 2));
        assert_eq!(depformer_weight_index(0).unwrap(), 0);
        assert_eq!(depformer_weight_index(31).unwrap(), 10);
        assert!(depformer_weight_index(32).is_err());
    }

    #[test]
    fn config_rejects_wrong_card_nq_depq_or_delay_shape() {
        let mut config = KyutaiTtsConfig::authenticated();
        config.card = CARD - 1;
        assert!(config.validate().is_err());
        config = KyutaiTtsConfig::authenticated();
        config.n_q = 31;
        assert!(config.validate().is_err());
        config = KyutaiTtsConfig::authenticated();
        config.dep_q = 31;
        assert!(config.validate().is_err());
        config = KyutaiTtsConfig::authenticated();
        config.delays.pop();
        assert!(config.validate().is_err());
    }

    #[test]
    fn delay_alignment_withholds_until_max_delay_without_fallback() {
        let mut aligner =
            KyutaiTtsDelayedStreamAligner::new(KyutaiTtsConfig::authenticated()).unwrap();
        let step = |value| [value; STREAM_CHANNELS];
        assert!(aligner.push(step(10)).unwrap().is_none());
        assert!(aligner.push(step(11)).unwrap().is_none());
        let output = aligner.push(step(12)).unwrap().unwrap();
        assert_eq!(output[0], 12);
        assert_eq!(output[1], 12);
        assert_eq!(output[2], 10);
        assert_eq!(aligner.pending_steps(), 2);
        assert!(
            aligner
                .push({
                    let mut invalid = step(0);
                    invalid[0] = TEXT_CARD;
                    invalid
                })
                .is_err()
        );
        assert!(
            aligner
                .push({
                    let mut invalid = step(0);
                    invalid[1] = CARD;
                    invalid
                })
                .is_err()
        );
    }
}
