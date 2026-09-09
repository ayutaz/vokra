//! Pure source contract seams for Kyutai Hibiki 2B.
//!
//! The checked-in inspector authenticates the public artifact identities and
//! scalar config facts below. It does not authenticate a compiled tensor
//! topology or a native translation/Moshi/Mimi runtime, so this module does
//! not add a GGUF binder, forward path, demux, audio generation, or parity
//! claim. Those remain explicit follow-up gates.

use std::collections::VecDeque;

use vokra_core::{Result, VokraError};

/// Authenticated Hugging Face repository identity.
pub const HF_REPOSITORY: &str = "kyutai/hibiki-2b-pytorch-bf16";
/// Authenticated Hugging Face revision.
pub const HF_REVISION: &str = "bd71144c96f26040612f6414716f5f48ee4fce69";
/// Authenticated Hibiki source repository.
pub const HIBIKI_SOURCE: &str = "https://github.com/kyutai-labs/hibiki.git";
/// Authenticated Hibiki source revision.
pub const HIBIKI_REVISION: &str = "f1cf9293e35c1dceffbe60dd325bdd702bc8305e";
/// Authenticated Moshi source repository used by the source contract.
pub const MOSHI_SOURCE: &str = "https://github.com/kyutai-labs/moshi.git";
/// Authenticated Moshi source revision.
pub const MOSHI_REVISION: &str = "e6a55d2722a65870ef52a6c9f6ecfc0e90f38362";

/// Main Hibiki safetensors filename in the authenticated artifact set.
pub const MAIN_FILE: &str = "hibiki-pytorch-ccef4858@200.safetensors";
/// Main Hibiki safetensors byte length.
pub const MAIN_BYTES: u64 = 5_574_762_720;
/// Main Hibiki safetensors SHA-256 digest.
pub const MAIN_SHA256: &str = "0847a768f01f3e78c42ddb779e7aa9c610b7bab71306a71d62d98a7d9cff3bdb";
/// Mimi safetensors filename in the authenticated artifact set.
pub const MIMI_FILE: &str = "mimi-pytorch-e351c8d8@125.safetensors";
/// Mimi safetensors byte length.
pub const MIMI_BYTES: u64 = 384_644_900;
/// Mimi safetensors SHA-256 digest.
pub const MIMI_SHA256: &str = "09b782f0629851a271227fb9d36db65c041790365f11bbe5d3d59369cf863f50";
/// SentencePiece tokenizer filename in the authenticated artifact set.
pub const SPM_FILE: &str = "tokenizer_spm_48k_multi6_2.model";
/// SentencePiece tokenizer byte length.
pub const SPM_BYTES: u64 = 857_314;
/// SentencePiece tokenizer SHA-256 digest.
pub const SPM_SHA256: &str = "c22110fb855aa049e17346ea2e88355bdd664f06cbfd09948380ab5e85b39697";

/// Git blob identity for the main Hibiki artifact.
pub const MAIN_GIT_BLOB_SHA1: &str = "a1f6cf83e90f4cfa83a294d468e5820c2a12ebc6";
/// Git blob identity for the Mimi artifact.
pub const MIMI_GIT_BLOB_SHA1: &str = "c8d5e4cd18a5c1ce05bb89d81144a46cf1b9076c";
/// Git blob identity for the SentencePiece tokenizer.
pub const SPM_GIT_BLOB_SHA1: &str = "e3d3dac8d55cf70915d8a4b1915becbdb89b828a";
/// Git blob identity for the authenticated Hibiki audio I/O source file.
pub const HIBIKI_AUDIO_IO_GIT_BLOB_SHA1: &str = "5625eafbb7b68e4c99f693ae812bac8f7212f070";
/// Git blob identity for the authenticated Hibiki generation source file.
pub const HIBIKI_GEN_GIT_BLOB_SHA1: &str = "42df14d865f8de183a99d592bf97c6b31f6c13de";
/// Git blob identity for the authenticated Hibiki main source file.
pub const HIBIKI_MAIN_GIT_BLOB_SHA1: &str = "c34f6716ffaaa34590cc97825b716780832b48bb";

/// Number of audio codebooks in the fixed Hibiki contract.
pub const N_Q: usize = 32;
/// Number of streamed token channels, including the text channel.
pub const STREAM_CHANNELS: usize = N_Q + 1;
/// Audio codebook cardinality.
pub const CARD: u32 = 2048;
/// Text token cardinality.
pub const TEXT_CARD: u32 = 48_000;
/// Text padding token identifier.
pub const TEXT_PADDING_ID: u32 = 3;

const AUTHENTICATED_DELAYS: [u32; STREAM_CHANNELS] = [
    0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2,
];
const DEPFORMER_WEIGHT_SCHEDULE: [u32; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 8, 8, 8, 8, 8, 8];

#[derive(Debug, Clone, PartialEq)]
/// Authenticated scalar configuration for the Hibiki 2B source contract.
pub struct Hibiki2bConfig {
    /// Model type label.
    pub model_type: String,
    /// Audio codebook cardinality.
    pub card: u32,
    /// Number of audio codebooks.
    pub n_q: usize,
    /// Number of depformer codebooks.
    pub dep_q: usize,
    /// Per-channel stream delays.
    pub delays: Vec<u32>,
    /// Main model feature dimension.
    pub dim: usize,
    /// Text token cardinality.
    pub text_card: u32,
    /// Text padding token identifier.
    pub text_padding_id: u32,
    /// Number of main model attention heads.
    pub num_heads: usize,
    /// Number of main model layers.
    pub num_layers: usize,
    /// Main model hidden-width scale.
    pub hidden_scale: f32,
    /// Whether the main model is causal.
    pub causal: bool,
    /// Whether layer scale is absent.
    pub layer_scale_is_none: bool,
    /// Main model context length.
    pub context: usize,
    /// Positional embedding period.
    pub max_period: f32,
    /// Main model gating mode.
    pub gating: String,
    /// Main model normalization mode.
    pub norm: String,
    /// Main model positional embedding mode.
    pub positional_embedding: String,
    /// Depformer feature dimension.
    pub depformer_dim: usize,
    /// Number of depformer attention heads.
    pub depformer_num_heads: usize,
    /// Number of depformer layers.
    pub depformer_num_layers: usize,
    /// Depformer feed-forward dimension.
    pub depformer_dim_feedforward: usize,
    /// Whether depformer projections are multi-linear.
    pub depformer_multi_linear: bool,
    /// Depformer positional embedding mode.
    pub depformer_pos_emb: String,
    /// Whether depformer weights vary per step.
    pub depformer_weights_per_step: bool,
    /// Depformer low-rank embedding dimension.
    pub depformer_low_rank_embeddings: usize,
    /// Number of description bins.
    pub description_bins: usize,
    /// Description feature dimension.
    pub description_dim: usize,
    /// Description tokenizer mode.
    pub description_tokenizer: String,
    /// Authenticated description values.
    pub description_values: Vec<String>,
    /// Whether cross-attention positional embeddings are enabled in the fuser.
    pub fuser_cross_attention_pos_emb: bool,
    /// Cross-attention positional embedding scale.
    pub fuser_cross_attention_pos_emb_scale: u32,
    /// Fuser sum inputs.
    pub fuser_sum: Vec<String>,
    /// Fuser prepend inputs.
    pub fuser_prepend: Vec<String>,
    /// Fuser cross inputs.
    pub fuser_cross: Vec<String>,
    /// Cross-attention inputs.
    pub cross_attention: Vec<String>,
    /// Authenticated model identifier signature.
    pub model_id_sig: String,
    /// Authenticated model identifier epoch.
    pub model_id_epoch: u32,
    /// Depformer weight schedule.
    pub depformer_weight_schedule: Vec<u32>,
    /// Audio sampling temperature.
    pub temperature: f32,
    /// Text sampling temperature.
    pub text_temperature: f32,
    /// Audio sampling top-k value.
    pub top_k: u32,
    /// Text sampling top-k value.
    pub text_top_k: u32,
}

impl Hibiki2bConfig {
    /// Returns the fixed configuration authenticated by the inspector.
    #[must_use]
    pub fn authenticated() -> Self {
        Self {
            model_type: "hibiki".to_owned(),
            card: CARD,
            n_q: N_Q,
            dep_q: 16,
            delays: AUTHENTICATED_DELAYS.to_vec(),
            dim: 2560,
            text_card: TEXT_CARD,
            text_padding_id: TEXT_PADDING_ID,
            num_heads: 20,
            num_layers: 24,
            hidden_scale: 4.125,
            causal: true,
            layer_scale_is_none: true,
            context: 1500,
            max_period: 100_000.0,
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
            description_bins: 31,
            description_dim: 16,
            description_tokenizer: "noop".to_owned(),
            description_values: ["very_bad", "bad", "neutral", "good", "very_good"]
                .into_iter()
                .map(str::to_owned)
                .collect(),
            fuser_cross_attention_pos_emb: false,
            fuser_cross_attention_pos_emb_scale: 1,
            fuser_sum: vec!["description".to_owned()],
            fuser_prepend: Vec::new(),
            fuser_cross: Vec::new(),
            cross_attention: Vec::new(),
            model_id_sig: "ccef4858".to_owned(),
            model_id_epoch: 200,
            depformer_weight_schedule: DEPFORMER_WEIGHT_SCHEDULE.to_vec(),
            temperature: 0.8,
            text_temperature: 0.8,
            top_k: 250,
            text_top_k: 50,
        }
    }

    /// Rejects any deviation from the inspector's fixed scalar contract.
    pub fn validate(&self) -> Result<()> {
        let expected = Self::authenticated();
        let exact = self.model_type == expected.model_type
            && self.card == expected.card
            && self.n_q == expected.n_q
            && self.dep_q == expected.dep_q
            && self.delays == expected.delays
            && self.dim == expected.dim
            && self.text_card == expected.text_card
            && self.text_padding_id == expected.text_padding_id
            && self.num_heads == expected.num_heads
            && self.num_layers == expected.num_layers
            && self.hidden_scale.to_bits() == expected.hidden_scale.to_bits()
            && self.causal == expected.causal
            && self.layer_scale_is_none == expected.layer_scale_is_none
            && self.context == expected.context
            && self.max_period.to_bits() == expected.max_period.to_bits()
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
            && self.description_bins == expected.description_bins
            && self.description_dim == expected.description_dim
            && self.description_tokenizer == expected.description_tokenizer
            && self.description_values == expected.description_values
            && self.fuser_cross_attention_pos_emb == expected.fuser_cross_attention_pos_emb
            && self.fuser_cross_attention_pos_emb_scale
                == expected.fuser_cross_attention_pos_emb_scale
            && self.fuser_sum == expected.fuser_sum
            && self.fuser_prepend == expected.fuser_prepend
            && self.fuser_cross == expected.fuser_cross
            && self.cross_attention == expected.cross_attention
            && self.model_id_sig == expected.model_id_sig
            && self.model_id_epoch == expected.model_id_epoch
            && self.depformer_weight_schedule == expected.depformer_weight_schedule
            && self.temperature.to_bits() == expected.temperature.to_bits()
            && self.text_temperature.to_bits() == expected.text_temperature.to_bits()
            && self.top_k == expected.top_k
            && self.text_top_k == expected.text_top_k;
        if exact {
            Ok(())
        } else {
            Err(VokraError::InvalidArgument(
                "hibiki-2b config does not match authenticated inspector contract".to_owned(),
            ))
        }
    }

    /// Returns the maximum configured stream delay.
    #[must_use]
    pub fn max_delay(&self) -> u32 {
        self.delays.iter().copied().max().unwrap_or(0)
    }
}

/// Pure delay alignment only; this is not Hibiki's translation or demux runtime.
#[derive(Debug)]
pub struct Hibiki2bDelayedStreamAligner {
    config: Hibiki2bConfig,
    pending: VecDeque<[u32; STREAM_CHANNELS]>,
}

impl Hibiki2bDelayedStreamAligner {
    /// Creates an aligner after validating the authenticated configuration.
    pub fn new(config: Hibiki2bConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            config,
            pending: VecDeque::new(),
        })
    }

    /// Queues one token step and returns an aligned step when available.
    pub fn push(&mut self, step: [u32; STREAM_CHANNELS]) -> Result<Option<[u32; STREAM_CHANNELS]>> {
        for (channel, &token) in step.iter().enumerate() {
            let limit = if channel == 0 {
                self.config.text_card
            } else {
                self.config.card
            };
            if token >= limit {
                return Err(VokraError::InvalidArgument(format!(
                    "hibiki-2b token {token} is outside channel {channel} card {limit}"
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

    /// Returns the number of queued, not-yet-emitted steps.
    #[must_use]
    pub fn pending_steps(&self) -> usize {
        self.pending.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn authenticated_config_and_identities_are_exact() {
        let config = Hibiki2bConfig::authenticated();
        config.validate().unwrap();
        assert_eq!(HF_REVISION.len(), 40);
        assert_eq!(MAIN_SHA256.len(), 64);
        assert_eq!(MIMI_SHA256.len(), 64);
        assert_eq!(SPM_SHA256.len(), 64);
        assert_eq!(config.delays.len(), STREAM_CHANNELS);
        assert_eq!(config.delays[0..2], [0, 0]);
        assert_eq!(config.delays[17], 0);
        assert_eq!(config.depformer_weight_schedule.len(), 16);
    }

    #[test]
    fn config_rejects_wrong_card_nq_depq_delay_or_schedule() {
        let mut config = Hibiki2bConfig::authenticated();
        config.card = CARD - 1;
        assert!(config.validate().is_err());
        config = Hibiki2bConfig::authenticated();
        config.n_q = 31;
        assert!(config.validate().is_err());
        config = Hibiki2bConfig::authenticated();
        config.dep_q = 15;
        assert!(config.validate().is_err());
        config = Hibiki2bConfig::authenticated();
        config.delays.pop();
        assert!(config.validate().is_err());
        config = Hibiki2bConfig::authenticated();
        config.depformer_weight_schedule[0] = 1;
        assert!(config.validate().is_err());
    }

    #[test]
    fn delay_alignment_is_model_free_and_has_no_fallback() {
        let mut aligner =
            Hibiki2bDelayedStreamAligner::new(Hibiki2bConfig::authenticated()).unwrap();
        let step = |value| [value; STREAM_CHANNELS];
        assert!(aligner.push(step(10)).unwrap().is_none());
        assert!(aligner.push(step(11)).unwrap().is_none());
        let output = aligner.push(step(12)).unwrap().unwrap();
        assert_eq!(output[0], 12);
        assert_eq!(output[1], 12);
        assert_eq!(output[2], 10);
        assert_eq!(output[17], 12);
        assert_eq!(output[18], 10);
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
