//! **VibeVoice-1.5B** — Microsoft's long-form, multi-speaker end-to-end
//! diffusion-autoregressive TTS (SoTA plan Phase 4, 2026-07-24). MIT
//! code + weight.
//!
//! # What VibeVoice is (primary source)
//!
//! `microsoft/VibeVoice-1.5B` is a tokenizer-**paired** speech
//! synthesizer whose LM predicts a diffusion head that samples
//! continuous acoustic-VAE latents. Every axis below is transcribed
//! **verbatim** from
//! `huggingface.co/microsoft/VibeVoice-1.5B/raw/main/config.json` and
//! `github.com/microsoft/VibeVoice/blob/main/vibevoice/modular/
//! configuration_vibevoice.py` (fetched 2026-07-24 — CLAUDE.md
//! 「ハルシネーション厳禁」).
//!
//! The stack chains:
//!
//! - A **Qwen2 decoder LM** (`decoder_config` — 28-layer /
//!   `hidden_size=1536` / MHA `num_attention_heads=12` / GQA
//!   `num_key_value_heads=2` for a group ratio of 6 / SwiGLU
//!   `intermediate_size=8960` / RoPE `theta=1_000_000` /
//!   `rms_norm_eps=1e-6`, `vocab_size=151_936`,
//!   `max_position_embeddings=65_536`, `tie_word_embeddings=true`,
//!   `sliding_window=null`, `use_cache=true`).
//! - An **acoustic tokenizer** (`acoustic_tokenizer_config` — σ-VAE
//!   with a mirror-symmetric encoder/decoder, `vae_dim=64`,
//!   `std_dist_type="gaussian"` with `fix_std=0.5`,
//!   `encoder_ratios=[8,5,5,4,2,2]` product 3200 →
//!   `24_000 Hz / 3200 = 7.5 Hz` frame rate,
//!   `encoder_n_filters=decoder_n_filters=32`,
//!   `encoder_depths="3-3-3-3-3-3-8"`, `mixer_layer="depthwise_conv"`,
//!   `layernorm="RMSNorm"`, `layernorm_eps=1e-5`, `causal=true`,
//!   `channels=1`, `conv_bias=true`, `disable_last_norm=true`,
//!   `layer_scale_init_value=1e-6`). Encoder input + decoder output
//!   both run at 24 kHz mono PCM.
//! - A **semantic tokenizer** (`semantic_tokenizer_config` — encoder-
//!   **only** variant of the same causal-Conv1d chain, `vae_dim=128`
//!   with `std_dist_type="none"` and `fix_std=0` — the semantic head
//!   is deterministic and its output is fed to the LM as
//!   conditioning; there is no decoder half). Same
//!   `encoder_ratios=[8,5,5,4,2,2]` product 3200 → 7.5 Hz.
//! - A **diffusion head** (`diffusion_head_config` — 4-layer AdaLN-
//!   modulated MLP with SwiGLU FFN, `hidden_size=1536`,
//!   `head_layers=4`, `head_ffn_ratio=3.0` → `ffn_dim = int(1536 · 3) = 4608`,
//!   `rms_norm_eps=1e-5`, `latent_size=64`, `speech_vae_dim=64`,
//!   `prediction_type="v_prediction"`, `diffusion_type="ddpm"`,
//!   `ddpm_num_steps=1000`, `ddpm_num_inference_steps=20`,
//!   `ddpm_beta_schedule="cosine"`, `ddpm_batch_mul=4`). Consumes the
//!   LM hidden state as AdaLN condition and predicts velocity in the
//!   64-d acoustic VAE latent space.
//!
//! # Distinct topology axis: continuous VAE + **diffusion decoder**
//!
//! VibeVoice is the **second** Vokra target (after VoxCPM-0.5B) whose
//! terminal decoding hop is a continuous-latent feature generator
//! feeding a continuous VAE decoder — but where VoxCPM uses a
//! **flow-matching** sampler ([`vokra_ops::flow_sampler`]), VibeVoice
//! uses a **DDPM** sampler ([`vokra_ops::ddpm_sampler`]) with
//! `v-prediction` and a cosine β schedule. The two axes are
//! irreconcilable inside [`vokra_ops::flow_sampler`] (its DDIM /
//! DPM++ solvers carry `ε`-prediction with a linear α schedule pinned
//! inside the solver per ADR M3-05 §D4) — see
//! [`vokra_ops::ddpm_sampler`]'s crate rustdoc for the full "why a
//! distinct sampler" argument.
//!
//! # Reuses two existing ops + one shared new op
//!
//! - **Acoustic VAE encoder/decoder**: shared
//!   [`vokra_ops::vae_continuous`] primitive (SoTA plan Phase 4 new op
//!   introduced with VoxCPM-0.5B and shared with this VibeVoice
//!   consumer as documented in that module's rustdoc).
//! - **DDPM diffusion sampler**: the SoTA plan Phase 4 new op
//!   [`vokra_ops::ddpm_sampler`] introduced with this model.
//! - **Semantic encoder**: a **local** deterministic
//!   causal-Conv1d chain identical in shape to the acoustic encoder
//!   half. Structurally consumes the same VAE-primitive kernels as
//!   the acoustic side; the runtime treats it as a
//!   [`vokra_ops::vae_continuous::ContinuousVaeEncoder`] with
//!   `latent_dim=128` — but without a decoder counterpart (VibeVoice
//!   does not decode the semantic latents back to audio).
//! - **Qwen2 LM backbone**: reuses the same GQA / RMSNorm / SwiGLU /
//!   RoPE primitives every earlier Qwen / Mistral / MiniCPM sibling
//!   uses (the SIMD kernels of `vokra-backend-cpu`).
//!
//! No new **backend kernel** is added by this model — the diffusion
//! head's MLP+AdaLN body is Linear / RMSNorm / SwiGLU, all covered by
//! the existing kernel inventory.
//!
//! # What lands in this Phase 4 slice
//!
//! - [`VibeVoiceAcousticTokenizerConfig`] / [`VibeVoiceSemanticTokenizerConfig`]
//!   / [`VibeVoiceDecoderConfig`] / [`VibeVoiceDiffusionHeadConfig`] /
//!   [`VibeVoiceConfig`] — every architectural hparam transcribed
//!   verbatim from the primary source. `validate_for_forward` fails
//!   loudly (FR-EX-08) on zeroed axes / broken GQA algebra /
//!   feat_dim ≠ latent_size / semantic-vs-acoustic vae_dim swap.
//! - [`VibeVoiceWeights`] — deterministic
//!   [`VibeVoiceWeights::synthesized`] scaffold fixture (zero-
//!   initialized; only the shape flow is exercised — the LM backbone
//!   weight store is a follow-up wave).
//! - [`VibeVoiceTts`] — engine handle carrying config + weights. The
//!   primary [`VibeVoiceTts::synthesize`] entry point returns
//!   [`VokraError::NotImplemented`] naming the blocker until real
//!   weights are bound and the LM → diffusion-head → DDPM sampler →
//!   AudioVAE decode → 24 kHz PCM chain is wired end-to-end
//!   (T29-equivalent follow-up wave — never a silent zero-fill,
//!   FR-EX-08).
//!
//! # No ONNX (permanent)
//!
//! VibeVoice-1.5B is distributed as safetensors + a Python pipeline
//! (`transformers`); the runtime **never** loads an ONNX graph
//! (FR-LD-05, permanent constraint); the pipeline is re-implemented
//! natively from the safetensors checkpoint (whisper.cpp 型 self
//! re-implementation, CLAUDE.md 設計判断 4).

use vokra_core::{Result, VokraError};

// Public seam re-exports — shared with the VAE + sampler primitives.
pub use vokra_ops::ddpm_sampler::{BetaSchedule, DdpmSamplerConfig, PredictionType};
pub use vokra_ops::vae_continuous::ContinuousVaeConfig;

/// `vokra.model.arch` a VibeVoice-1.5B GGUF must carry. Written by
/// `vokra-convert::models::vibevoice::ARCH`. Intentionally **distinct**
/// from every existing arch tag in this crate — VibeVoice pairs a
/// continuous VAE decoder with a **DDPM** diffusion head, not the
/// UnifiedCFM flow-matching sampler VoxCPM uses. Silently sharing an
/// arch tag with VoxCPM would misroute the runtime dispatch: the
/// sampler for VibeVoice is [`vokra_ops::ddpm_sampler::ddpm_sample`],
/// **not** [`vokra_ops::flow_sampler::flow_sample`].
pub const EXPECTED_ARCH: &str = "vibevoice";

/// Acoustic and semantic tokenizer PCM sample rate (Hz). Both
/// tokenizers consume 24 kHz mono PCM (upstream
/// `configuration_vibevoice.py::VibeVoiceAcousticTokenizerConfig`
/// implicit — the model card states "24kHz input" and the acoustic
/// decoder's ratios mirror the encoder so its output sample rate is
/// also 24 kHz).
pub const VIBEVOICE_ENCODER_SAMPLE_RATE: u32 = 24_000;

/// Frame rate (Hz) at which the LM steps: `24_000 / product(encoder_ratios) =
/// 24_000 / 3200 = 7.5`.
pub const VIBEVOICE_LM_FRAME_RATE_HZ: f32 = 7.5;

// ---------------------------------------------------------------------------
// Decoder LM config — Qwen2.5-1.5B flavour
// ---------------------------------------------------------------------------

/// Qwen2 decoder-LM hparams the VibeVoice `decoder_config` block
/// transcribes.
///
/// Every field is transcribed **verbatim** from
/// `huggingface.co/microsoft/VibeVoice-1.5B/raw/main/config.json`
/// `decoder_config.*` (fetched 2026-07-24). The Qwen2 block is a
/// Llama-family decoder-only transformer with GQA
/// (`num_attention_heads=12`, `num_key_value_heads=2` → group ratio
/// 6), a very long context (`max_position_embeddings=65_536`), and
/// tied word embeddings (`tie_word_embeddings=true` — the LM head and
/// the token embedding table share tensor storage). No sliding
/// window (`sliding_window=null`, `use_sliding_window=false`).
#[derive(Debug, Clone, PartialEq)]
pub struct VibeVoiceDecoderConfig {
    /// Backbone hidden dimension (`hidden_size`). `1536`.
    pub hidden_dim: u32,
    /// Backbone transformer block count (`num_hidden_layers`). `28`.
    pub n_layer: u32,
    /// Backbone attention head count (`num_attention_heads`). `12`.
    pub n_head: u32,
    /// Backbone key/value head count for GQA (`num_key_value_heads`).
    /// `2` — the group ratio is `n_head / n_head_kv = 6` (each K/V
    /// head fans out to 6 Q heads).
    pub n_head_kv: u32,
    /// SwiGLU FFN inner dimension (`intermediate_size`). `8960`.
    pub ffn_dim: u32,
    /// Vocabulary size (`vocab_size`). `151_936` — the Qwen2
    /// tokenizer's shared BPE.
    pub vocab_size: u32,
    /// Max positions the LM can attend over (`max_position_embeddings`).
    /// `65_536`.
    pub max_position_embeddings: u32,
    /// RoPE base θ (`rope_theta`). `1_000_000` — Qwen2 default for
    /// long-context models.
    pub rope_base: f32,
    /// RMSNorm epsilon (`rms_norm_eps`). `1e-6`.
    pub rms_norm_eps: f32,
    /// Attention dropout (`attention_dropout`). `0.0`.
    pub attention_dropout: f32,
    /// Whether the LM head shares tensor storage with the token
    /// embedding (`tie_word_embeddings`). `true`.
    pub tie_word_embeddings: bool,
    /// Whether the sliding-window attention mask is enabled
    /// (`use_sliding_window`). `false` for VibeVoice — the model
    /// carries full causal attention across its 65k-token context.
    pub use_sliding_window: bool,
    /// Cap on how many of the backbone's layers use windowed
    /// attention when `use_sliding_window` is enabled
    /// (`max_window_layers`). Recorded verbatim even though
    /// `use_sliding_window=false` — a future variant that enables
    /// windowing would honor this axis. `28`.
    pub max_window_layers: u32,
}

impl VibeVoiceDecoderConfig {
    /// Canonical VibeVoice-1.5B `decoder_config` (primary source:
    /// `config.json.decoder_config.*`, fetched 2026-07-24).
    #[must_use]
    pub fn vibevoice_1_5b() -> Self {
        Self {
            hidden_dim: 1536,
            n_layer: 28,
            n_head: 12,
            n_head_kv: 2,
            ffn_dim: 8960,
            vocab_size: 151_936,
            max_position_embeddings: 65_536,
            rope_base: 1_000_000.0,
            rms_norm_eps: 1e-6,
            attention_dropout: 0.0,
            tie_word_embeddings: true,
            use_sliding_window: false,
            max_window_layers: 28,
        }
    }

    /// Miniature well-formed decoder config for shape / stability
    /// tests. Every ratio (GQA well-formedness, non-zero vocab,
    /// positive FFN dim) mirrors the real model.
    #[must_use]
    pub fn tiny_for_tests() -> Self {
        Self {
            hidden_dim: 16,
            n_layer: 2,
            n_head: 4,
            n_head_kv: 2,
            ffn_dim: 32,
            vocab_size: 64,
            max_position_embeddings: 128,
            rope_base: 1_000_000.0,
            rms_norm_eps: 1e-6,
            attention_dropout: 0.0,
            tie_word_embeddings: true,
            use_sliding_window: false,
            max_window_layers: 2,
        }
    }
}

// ---------------------------------------------------------------------------
// Tokenizer configs
// ---------------------------------------------------------------------------

/// Acoustic tokenizer (σ-VAE) hparams — the mirror-symmetric
/// encoder/decoder pair whose latent stream the diffusion head
/// predicts.
///
/// Every field is transcribed **verbatim** from
/// `configuration_vibevoice.py::VibeVoiceAcousticTokenizerConfig`
/// (fetched 2026-07-24; the release checkpoint carries the same
/// defaults through `config.json.acoustic_tokenizer_config.*`).
///
/// Every stage-internal knob (`encoder_depths`,
/// `layer_scale_init_value`, `weight_init_value`, `layernorm`) is
/// carried so a future real-weight binding can cross-check the
/// checkpoint layer inventory against the transcription.
#[derive(Debug, Clone, PartialEq)]
pub struct VibeVoiceAcousticTokenizerConfig {
    /// PCM input channels (`channels`). `1` (mono).
    pub channels: u32,
    /// Whether the causal-Conv1d chain is causal
    /// (streaming-friendly, `causal`). `true`.
    pub causal: bool,
    /// VAE latent width (`vae_dim`). `64`.
    pub vae_dim: u32,
    /// Std bootstrap magnitude for the σ-VAE (`fix_std`). `0.5`.
    pub fix_std: f32,
    /// Std distribution kind (`std_dist_type`). `"gaussian"` — the
    /// **stochastic** σ-VAE the diffusion head consumes.
    pub std_dist_type: String,
    /// Encoder base channel count (`encoder_n_filters`). `32`.
    pub encoder_n_filters: u32,
    /// Decoder base channel count (`decoder_n_filters`). `32`.
    /// (Upstream: `decoder_ratios = encoder_ratios` when
    /// `decoder_ratios is None`; VibeVoice explicitly sets both to
    /// `[8, 5, 5, 4, 2, 2]`.)
    pub decoder_n_filters: u32,
    /// Encoder stride list (`encoder_ratios`). `[8, 5, 5, 4, 2, 2]` —
    /// product 3200 → 7.5 Hz frame rate at 24 kHz input.
    pub encoder_ratios: Vec<u32>,
    /// Decoder stride list (`decoder_ratios`). `[8, 5, 5, 4, 2, 2]` —
    /// mirror-symmetric to the encoder.
    pub decoder_ratios: Vec<u32>,
    /// Encoder block-depths as a hyphen-joined string
    /// (`encoder_depths`). `"3-3-3-3-3-3-8"` — the last (deepest)
    /// stage runs 8 blocks; the six others run 3 each.
    pub encoder_depths: String,
    /// LayerScale γ init value (`layer_scale_init_value`). `1e-6`.
    pub layer_scale_init_value: f32,
    /// Weight-init RMS ceiling (`weight_init_value`). `1e-2`.
    pub weight_init_value: f32,
    /// Layer normalization kind (`layernorm`). `"RMSNorm"`.
    pub layernorm: String,
    /// Whether the layer normalization carries a per-channel weight
    /// (`layernorm_elementwise_affine`). `true`.
    pub layernorm_elementwise_affine: bool,
    /// RMSNorm epsilon (`layernorm_eps`). `1e-5`.
    pub layernorm_eps: f32,
    /// Convolution-fusion mixer layer (`mixer_layer`).
    /// `"depthwise_conv"` — maps to the shared
    /// [`ContinuousVaeConfig::depthwise=true`].
    pub mixer_layer: String,
    /// Padding mode for the causal Conv1d (`pad_mode`). `"constant"`.
    pub pad_mode: String,
    /// Whether the terminal decoder LayerNorm is disabled
    /// (`disable_last_norm`). `true`.
    pub disable_last_norm: bool,
    /// Convolution weight normalization mode (`conv_norm`). `"none"`.
    pub conv_norm: String,
    /// Whether Conv1d layers carry a bias (`conv_bias`). `true`.
    pub conv_bias: bool,
    /// Corpus RMS normalization scale (`corpus_normalize`). `0.0`
    /// (disabled).
    pub corpus_normalize: f32,
}

impl VibeVoiceAcousticTokenizerConfig {
    /// Canonical VibeVoice-1.5B `acoustic_tokenizer_config`
    /// (primary source: `config.json.acoustic_tokenizer_config.*`,
    /// fetched 2026-07-24).
    #[must_use]
    pub fn vibevoice_1_5b() -> Self {
        Self {
            channels: 1,
            causal: true,
            vae_dim: 64,
            fix_std: 0.5,
            std_dist_type: "gaussian".to_owned(),
            encoder_n_filters: 32,
            decoder_n_filters: 32,
            encoder_ratios: vec![8, 5, 5, 4, 2, 2],
            decoder_ratios: vec![8, 5, 5, 4, 2, 2],
            encoder_depths: "3-3-3-3-3-3-8".to_owned(),
            layer_scale_init_value: 1e-6,
            weight_init_value: 1e-2,
            layernorm: "RMSNorm".to_owned(),
            layernorm_elementwise_affine: true,
            layernorm_eps: 1e-5,
            mixer_layer: "depthwise_conv".to_owned(),
            pad_mode: "constant".to_owned(),
            disable_last_norm: true,
            conv_norm: "none".to_owned(),
            conv_bias: true,
            corpus_normalize: 0.0,
        }
    }

    /// Miniature well-formed acoustic-tokenizer config for tests.
    #[must_use]
    pub fn tiny_for_tests() -> Self {
        Self {
            channels: 1,
            causal: true,
            vae_dim: 4,
            fix_std: 0.5,
            std_dist_type: "gaussian".to_owned(),
            encoder_n_filters: 4,
            decoder_n_filters: 4,
            encoder_ratios: vec![2, 2],
            decoder_ratios: vec![2, 2],
            encoder_depths: "1-1-1".to_owned(),
            layer_scale_init_value: 1e-6,
            weight_init_value: 1e-2,
            layernorm: "RMSNorm".to_owned(),
            layernorm_elementwise_affine: true,
            layernorm_eps: 1e-5,
            mixer_layer: "depthwise_conv".to_owned(),
            pad_mode: "constant".to_owned(),
            disable_last_norm: true,
            conv_norm: "none".to_owned(),
            conv_bias: true,
            corpus_normalize: 0.0,
        }
    }

    /// Encoder hop length (`product(encoder_ratios)`). PCM samples
    /// per encoded frame. VibeVoice-1.5B: `3200`.
    ///
    /// Returns `None` on `u32` overflow.
    #[must_use]
    pub fn hop_length(&self) -> Option<u32> {
        self.encoder_ratios
            .iter()
            .try_fold(1u32, |acc, r| acc.checked_mul(*r))
    }

    /// Encoded frame rate (Hz) — `sample_rate_hz / hop_length`.
    /// VibeVoice-1.5B: `7.5 Hz` at 24 kHz input.
    ///
    /// Returns `None` when [`Self::hop_length`] does or when
    /// `hop_length` is zero.
    #[must_use]
    pub fn frame_rate_hz(&self, sample_rate_hz: u32) -> Option<f32> {
        let hop = self.hop_length()?;
        if hop == 0 {
            return None;
        }
        Some(sample_rate_hz as f32 / hop as f32)
    }
}

/// Semantic tokenizer hparams — encoder-**only** deterministic
/// causal-Conv1d chain that lifts 24 kHz PCM to a 128-d conditioning
/// stream the LM consumes. **No decoder half**: VibeVoice does not
/// decode the semantic latents back to audio.
///
/// Every field is transcribed **verbatim** from
/// `configuration_vibevoice.py::VibeVoiceSemanticTokenizerConfig`
/// (fetched 2026-07-24; the release checkpoint carries the same
/// defaults through `config.json.semantic_tokenizer_config.*`).
///
/// The key distinguishing axes vs. the acoustic tokenizer:
///
/// - `vae_dim = 128` (double the acoustic 64).
/// - `std_dist_type = "none"` (deterministic — no σ head).
/// - `fix_std = 0` (irrelevant when `std_dist_type = "none"`,
///   transcribed for completeness).
/// - No `decoder_*` axes exist upstream — VibeVoice's semantic
///   tokenizer runs the encoder chain only.
#[derive(Debug, Clone, PartialEq)]
pub struct VibeVoiceSemanticTokenizerConfig {
    /// PCM input channels (`channels`). `1` (mono).
    pub channels: u32,
    /// Whether the causal-Conv1d chain is causal (`causal`). `true`.
    pub causal: bool,
    /// Encoder latent width (`vae_dim`). `128` — distinct from the
    /// acoustic tokenizer's `64`.
    pub vae_dim: u32,
    /// Std bootstrap magnitude (`fix_std`). `0` for the deterministic
    /// semantic head.
    pub fix_std: f32,
    /// Std distribution kind (`std_dist_type`). `"none"` — the head
    /// is **deterministic**; there is no σ output.
    pub std_dist_type: String,
    /// Encoder base channel count (`encoder_n_filters`). `32`.
    pub encoder_n_filters: u32,
    /// Encoder stride list (`encoder_ratios`). `[8, 5, 5, 4, 2, 2]` —
    /// same as the acoustic tokenizer.
    pub encoder_ratios: Vec<u32>,
    /// Encoder block-depths (`encoder_depths`). `"3-3-3-3-3-3-8"`.
    pub encoder_depths: String,
    /// LayerScale γ init value (`layer_scale_init_value`). `1e-6`.
    pub layer_scale_init_value: f32,
    /// Weight-init RMS ceiling (`weight_init_value`). `1e-2`.
    pub weight_init_value: f32,
    /// Layer normalization kind (`layernorm`). `"RMSNorm"`.
    pub layernorm: String,
    /// Whether the layer normalization carries a per-channel weight
    /// (`layernorm_elementwise_affine`). `true`.
    pub layernorm_elementwise_affine: bool,
    /// RMSNorm epsilon (`layernorm_eps`). `1e-5`.
    pub layernorm_eps: f32,
    /// Mixer layer (`mixer_layer`). `"depthwise_conv"`.
    pub mixer_layer: String,
    /// Pad mode for the causal Conv1d (`pad_mode`). `"constant"`.
    pub pad_mode: String,
    /// Whether the terminal encoder LayerNorm is disabled
    /// (`disable_last_norm`). `true`.
    pub disable_last_norm: bool,
    /// Convolution weight normalization mode (`conv_norm`). `"none"`.
    pub conv_norm: String,
    /// Whether Conv1d layers carry a bias (`conv_bias`). `true`.
    pub conv_bias: bool,
    /// Corpus RMS normalization scale (`corpus_normalize`). `0.0`.
    pub corpus_normalize: f32,
}

impl VibeVoiceSemanticTokenizerConfig {
    /// Canonical VibeVoice-1.5B `semantic_tokenizer_config`
    /// (primary source: `config.json.semantic_tokenizer_config.*`,
    /// fetched 2026-07-24).
    #[must_use]
    pub fn vibevoice_1_5b() -> Self {
        Self {
            channels: 1,
            causal: true,
            vae_dim: 128,
            fix_std: 0.0,
            std_dist_type: "none".to_owned(),
            encoder_n_filters: 32,
            encoder_ratios: vec![8, 5, 5, 4, 2, 2],
            encoder_depths: "3-3-3-3-3-3-8".to_owned(),
            layer_scale_init_value: 1e-6,
            weight_init_value: 1e-2,
            layernorm: "RMSNorm".to_owned(),
            layernorm_elementwise_affine: true,
            layernorm_eps: 1e-5,
            mixer_layer: "depthwise_conv".to_owned(),
            pad_mode: "constant".to_owned(),
            disable_last_norm: true,
            conv_norm: "none".to_owned(),
            conv_bias: true,
            corpus_normalize: 0.0,
        }
    }

    /// Miniature well-formed semantic-tokenizer config for tests.
    #[must_use]
    pub fn tiny_for_tests() -> Self {
        Self {
            channels: 1,
            causal: true,
            vae_dim: 8,
            fix_std: 0.0,
            std_dist_type: "none".to_owned(),
            encoder_n_filters: 4,
            encoder_ratios: vec![2, 2],
            encoder_depths: "1-1-1".to_owned(),
            layer_scale_init_value: 1e-6,
            weight_init_value: 1e-2,
            layernorm: "RMSNorm".to_owned(),
            layernorm_elementwise_affine: true,
            layernorm_eps: 1e-5,
            mixer_layer: "depthwise_conv".to_owned(),
            pad_mode: "constant".to_owned(),
            disable_last_norm: true,
            conv_norm: "none".to_owned(),
            conv_bias: true,
            corpus_normalize: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Diffusion head config
// ---------------------------------------------------------------------------

/// Diffusion-head hparams — the per-step 4-layer AdaLN-modulated MLP
/// that predicts velocity in the acoustic-VAE latent space.
///
/// Every field is transcribed **verbatim** from
/// `configuration_vibevoice.py::VibeVoiceDiffusionHeadConfig` and
/// `config.json.diffusion_head_config.*` (fetched 2026-07-24). The
/// upstream `VibeVoiceDiffusionHead` forward chain is (per
/// `modular_vibevoice_diffusion_head.py`):
///
/// 1. `noisy_images_proj: Linear(latent_size, hidden_size, bias=False)`
///    lifts the noisy latent `x_t` to LM width.
/// 2. `cond_proj: Linear(hidden_size, hidden_size, bias=False)` +
///    `t_embedder: TimestepEmbedder(cond_dim=hidden_size,
///    frequency_embedding_size=256)` sum-fuse the LM condition
///    with a sinusoidal timestep embedding.
/// 3. `head_layers × HeadLayer(embed=hidden_size,
///    ffn=int(hidden_size · head_ffn_ratio), cond=hidden_size,
///    norm_eps=rms_norm_eps)`: RMSNorm → shift/scale/gate adaLN →
///    SwiGLU FFN → residual.
/// 4. `FinalLayer(hidden=hidden_size, out=latent_size,
///    cond=hidden_size, norm_eps=rms_norm_eps)`: affine-free
///    RMSNorm → shift/scale adaLN → Linear.
///
/// The sampler axes (`diffusion_type`, `prediction_type`,
/// `beta_schedule`, `num_steps`, `num_inference_steps`,
/// `batch_mul`) map 1:1 onto [`DdpmSamplerConfig`] fields — see
/// [`VibeVoiceConfig::ddpm_sampler_config`].
#[derive(Debug, Clone, PartialEq)]
pub struct VibeVoiceDiffusionHeadConfig {
    /// Head hidden dimension (`hidden_size`). `1536` — pinned equal
    /// to the LM `decoder_config.hidden_size` so `cond_proj` is a
    /// square linear.
    pub hidden_size: u32,
    /// Number of `HeadLayer` blocks (`head_layers`). `4`.
    pub head_layers: u32,
    /// FFN inner dim ratio (`head_ffn_ratio`). `3.0` — the SwiGLU
    /// inner dim is `int(hidden_size · 3.0) = 4608`.
    pub head_ffn_ratio: f32,
    /// RMSNorm epsilon (`rms_norm_eps`). `1e-5`.
    pub rms_norm_eps: f32,
    /// Continuous-VAE latent width the head predicts (`latent_size`).
    /// `64` — MUST equal the acoustic tokenizer's `vae_dim` (the
    /// VAE handshake).
    pub latent_size: u32,
    /// Speech-VAE latent width (`speech_vae_dim`). `64` — carried
    /// separately upstream but always equal to `latent_size` for
    /// VibeVoice-1.5B.
    pub speech_vae_dim: u32,
    /// Diffusion prediction target (`prediction_type`).
    /// `"v_prediction"`.
    pub prediction_type: String,
    /// Diffusion family (`diffusion_type`). `"ddpm"`.
    pub diffusion_type: String,
    /// Full training-time timestep count (`ddpm_num_steps`). `1000`.
    pub ddpm_num_steps: u32,
    /// Reduced-step inference count (`ddpm_num_inference_steps`).
    /// `20`.
    pub ddpm_num_inference_steps: u32,
    /// β schedule (`ddpm_beta_schedule`). `"cosine"`.
    pub ddpm_beta_schedule: String,
    /// Training-side per-sample batch multiplier for the DDPM loss
    /// (`ddpm_batch_mul`). `4`. Kept for provenance completeness —
    /// inference does not consume it.
    pub ddpm_batch_mul: u32,
}

impl VibeVoiceDiffusionHeadConfig {
    /// Canonical VibeVoice-1.5B `diffusion_head_config` (primary
    /// source: `config.json.diffusion_head_config.*`, fetched
    /// 2026-07-24).
    #[must_use]
    pub fn vibevoice_1_5b() -> Self {
        Self {
            hidden_size: 1536,
            head_layers: 4,
            head_ffn_ratio: 3.0,
            rms_norm_eps: 1e-5,
            latent_size: 64,
            speech_vae_dim: 64,
            prediction_type: "v_prediction".to_owned(),
            diffusion_type: "ddpm".to_owned(),
            ddpm_num_steps: 1000,
            ddpm_num_inference_steps: 20,
            ddpm_beta_schedule: "cosine".to_owned(),
            ddpm_batch_mul: 4,
        }
    }

    /// Miniature well-formed diffusion-head config for tests.
    #[must_use]
    pub fn tiny_for_tests() -> Self {
        Self {
            hidden_size: 16,
            head_layers: 2,
            head_ffn_ratio: 3.0,
            rms_norm_eps: 1e-5,
            latent_size: 4,
            speech_vae_dim: 4,
            prediction_type: "v_prediction".to_owned(),
            diffusion_type: "ddpm".to_owned(),
            ddpm_num_steps: 10,
            ddpm_num_inference_steps: 2,
            ddpm_beta_schedule: "cosine".to_owned(),
            ddpm_batch_mul: 4,
        }
    }

    /// The SwiGLU FFN inner dimension (`int(hidden_size · head_ffn_ratio)`).
    /// VibeVoice-1.5B: `4608`.
    #[must_use]
    pub fn ffn_inner_dim(&self) -> u32 {
        (self.hidden_size as f32 * self.head_ffn_ratio) as u32
    }
}

// ---------------------------------------------------------------------------
// Top-level config
// ---------------------------------------------------------------------------

/// Full VibeVoice-1.5B config: acoustic tokenizer + semantic
/// tokenizer + Qwen2 decoder LM + DDPM diffusion head + top-level
/// `{acoustic,semantic}_vae_dim` shortcuts.
///
/// The two tokenizer VAE seams and the diffusion head are
/// cross-checked at [`Self::validate_for_forward`]: the diffusion
/// head's `latent_size` MUST equal the acoustic tokenizer's `vae_dim`
/// (silent mismatch would drop or duplicate channels between the head
/// and the VAE decoder). The `acoustic_vae_dim` / `semantic_vae_dim`
/// top-level shortcuts are the same axes carried again for parity
/// with upstream's `config.json`.
#[derive(Debug, Clone, PartialEq)]
pub struct VibeVoiceConfig {
    /// Acoustic tokenizer sub-config (σ-VAE, mirror-symmetric enc/dec).
    pub acoustic: VibeVoiceAcousticTokenizerConfig,
    /// Semantic tokenizer sub-config (encoder-only deterministic).
    pub semantic: VibeVoiceSemanticTokenizerConfig,
    /// Qwen2 decoder-LM sub-config.
    pub decoder: VibeVoiceDecoderConfig,
    /// DDPM diffusion-head sub-config.
    pub diffusion_head: VibeVoiceDiffusionHeadConfig,
    /// Top-level `acoustic_vae_dim`. `64` — MUST equal
    /// `acoustic.vae_dim`.
    pub acoustic_vae_dim: u32,
    /// Top-level `semantic_vae_dim`. `128` — MUST equal
    /// `semantic.vae_dim`.
    pub semantic_vae_dim: u32,
}

impl VibeVoiceConfig {
    /// Canonical VibeVoice-1.5B config (primary source: `config.json`,
    /// fetched 2026-07-24).
    #[must_use]
    pub fn vibevoice_1_5b() -> Self {
        Self {
            acoustic: VibeVoiceAcousticTokenizerConfig::vibevoice_1_5b(),
            semantic: VibeVoiceSemanticTokenizerConfig::vibevoice_1_5b(),
            decoder: VibeVoiceDecoderConfig::vibevoice_1_5b(),
            diffusion_head: VibeVoiceDiffusionHeadConfig::vibevoice_1_5b(),
            acoustic_vae_dim: 64,
            semantic_vae_dim: 128,
        }
    }

    /// Miniature well-formed config for shape / stability tests.
    /// Every axis is proportional to the canonical release: the
    /// diffusion head's `latent_size` matches the acoustic tokenizer's
    /// `vae_dim`, the LM's `hidden_size` matches the diffusion head's
    /// `hidden_size`, and the GQA algebra is preserved.
    #[must_use]
    pub fn tiny_for_tests() -> Self {
        Self {
            acoustic: VibeVoiceAcousticTokenizerConfig::tiny_for_tests(),
            semantic: VibeVoiceSemanticTokenizerConfig::tiny_for_tests(),
            decoder: VibeVoiceDecoderConfig::tiny_for_tests(),
            diffusion_head: VibeVoiceDiffusionHeadConfig::tiny_for_tests(),
            acoustic_vae_dim: 4,
            semantic_vae_dim: 8,
        }
    }

    /// True iff every architectural axis is at its `0` sentinel — the
    /// shape-only conversion path the runtime tolerates as
    /// inspectable-but-not-forward-ready.
    #[must_use]
    pub fn is_placeholder_shape(&self) -> bool {
        self.decoder.hidden_dim == 0
            && self.decoder.n_layer == 0
            && self.diffusion_head.hidden_size == 0
            && self.diffusion_head.head_layers == 0
            && self.acoustic.vae_dim == 0
            && self.semantic.vae_dim == 0
    }

    /// Builds a shared [`ContinuousVaeConfig`] for the **acoustic**
    /// tokenizer (encoder + decoder — mirror-symmetric).
    ///
    /// Both `sample_rate_hz` and `out_sample_rate_hz` are 24 kHz
    /// (VibeVoice's mirror-symmetric acoustic VAE does not upsample
    /// like VoxCPM's 16 → 48 kHz asymmetric decoder does).
    #[must_use]
    pub fn acoustic_vae_config(&self) -> ContinuousVaeConfig {
        ContinuousVaeConfig {
            sample_rate_hz: VIBEVOICE_ENCODER_SAMPLE_RATE,
            out_sample_rate_hz: VIBEVOICE_ENCODER_SAMPLE_RATE,
            encoder_dim: self.acoustic.encoder_n_filters,
            encoder_rates: self.acoustic.encoder_ratios.clone(),
            latent_dim: self.acoustic.vae_dim,
            decoder_dim: self.acoustic.decoder_n_filters,
            decoder_rates: self.acoustic.decoder_ratios.clone(),
            depthwise: self.acoustic.mixer_layer == "depthwise_conv",
            use_noise_block: false,
        }
    }

    /// Builds the [`DdpmSamplerConfig`] the head samples with.
    ///
    /// Every axis is transcribed from `diffusion_head_config.*`; the
    /// CFG axes default to `CfgMode::None` / `CfgScaleProfile::Constant(1.0)`
    /// (see [`DdpmSamplerConfig::vibevoice_defaults`] for the
    /// rationale).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] iff the diffusion-head config
    /// carries an unrecognised `prediction_type` or
    /// `ddpm_beta_schedule` string (a future variant may add axes;
    /// silently mapping an unknown string to the default would
    /// mis-route the sampler per FR-EX-08).
    pub fn ddpm_sampler_config(&self) -> Result<DdpmSamplerConfig> {
        let prediction_type = match self.diffusion_head.prediction_type.as_str() {
            "v_prediction" => PredictionType::VPrediction,
            "epsilon" => PredictionType::Epsilon,
            "sample" => PredictionType::Sample,
            other => {
                return Err(VokraError::InvalidArgument(format!(
                    "vibevoice diffusion_head_config: prediction_type = {other:?} is not \
                     one of {{\"v_prediction\", \"epsilon\", \"sample\"}}"
                )));
            }
        };
        let beta_schedule = match self.diffusion_head.ddpm_beta_schedule.as_str() {
            "cosine" => BetaSchedule::Cosine,
            "linear" => BetaSchedule::Linear,
            other => {
                return Err(VokraError::InvalidArgument(format!(
                    "vibevoice diffusion_head_config: ddpm_beta_schedule = {other:?} is not \
                     one of {{\"cosine\", \"linear\"}}"
                )));
            }
        };
        Ok(DdpmSamplerConfig {
            num_train_steps: self.diffusion_head.ddpm_num_steps,
            num_inference_steps: self.diffusion_head.ddpm_num_inference_steps,
            prediction_type,
            beta_schedule,
            ..DdpmSamplerConfig::vibevoice_defaults()
        })
    }

    /// Rejects `0`-placeholder / ill-formed configs before any
    /// forward runs.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] naming the offending field.
    pub fn validate_for_forward(&self) -> Result<()> {
        // Decoder axes.
        let d = &self.decoder;
        if d.hidden_dim == 0
            || d.n_layer == 0
            || d.n_head == 0
            || d.n_head_kv == 0
            || d.ffn_dim == 0
            || d.vocab_size == 0
            || d.max_position_embeddings == 0
        {
            return Err(VokraError::InvalidArgument(
                "vibevoice decoder config: every architectural axis must be > 0 (bind a real \
                 checkpoint or use VibeVoiceDecoderConfig::tiny_for_tests for shape tests)"
                    .to_owned(),
            ));
        }
        if d.n_head % d.n_head_kv != 0 {
            return Err(VokraError::InvalidArgument(format!(
                "vibevoice decoder config: n_head_kv ({}) must divide n_head ({}) — GQA requires \
                 an integer group ratio",
                d.n_head_kv, d.n_head,
            )));
        }
        if !(d.rope_base.is_finite() && d.rope_base > 0.0) {
            return Err(VokraError::InvalidArgument(format!(
                "vibevoice decoder config: rope_base must be a positive finite f32 (got {})",
                d.rope_base,
            )));
        }
        if !(d.rms_norm_eps.is_finite() && d.rms_norm_eps > 0.0) {
            return Err(VokraError::InvalidArgument(format!(
                "vibevoice decoder config: rms_norm_eps must be a positive finite f32 (got {})",
                d.rms_norm_eps,
            )));
        }
        if !(d.attention_dropout.is_finite() && d.attention_dropout >= 0.0) {
            return Err(VokraError::InvalidArgument(format!(
                "vibevoice decoder config: attention_dropout must be a non-negative finite f32 \
                 (got {})",
                d.attention_dropout,
            )));
        }

        // Acoustic tokenizer axes.
        let a = &self.acoustic;
        if a.channels == 0 || a.vae_dim == 0 || a.encoder_n_filters == 0 || a.decoder_n_filters == 0
        {
            return Err(VokraError::InvalidArgument(
                "vibevoice acoustic tokenizer: channels / vae_dim / encoder_n_filters / \
                 decoder_n_filters must all be > 0"
                    .to_owned(),
            ));
        }
        if a.encoder_ratios.is_empty() || a.decoder_ratios.is_empty() {
            return Err(VokraError::InvalidArgument(
                "vibevoice acoustic tokenizer: encoder_ratios / decoder_ratios must be non-empty"
                    .to_owned(),
            ));
        }
        for (i, r) in a.encoder_ratios.iter().enumerate() {
            if *r == 0 {
                return Err(VokraError::InvalidArgument(format!(
                    "vibevoice acoustic tokenizer: encoder_ratios[{i}] must be > 0"
                )));
            }
        }
        for (i, r) in a.decoder_ratios.iter().enumerate() {
            if *r == 0 {
                return Err(VokraError::InvalidArgument(format!(
                    "vibevoice acoustic tokenizer: decoder_ratios[{i}] must be > 0"
                )));
            }
        }
        if !(a.layernorm_eps.is_finite() && a.layernorm_eps > 0.0) {
            return Err(VokraError::InvalidArgument(format!(
                "vibevoice acoustic tokenizer: layernorm_eps must be a positive finite f32 \
                 (got {})",
                a.layernorm_eps,
            )));
        }

        // Semantic tokenizer axes.
        let s = &self.semantic;
        if s.channels == 0 || s.vae_dim == 0 || s.encoder_n_filters == 0 {
            return Err(VokraError::InvalidArgument(
                "vibevoice semantic tokenizer: channels / vae_dim / encoder_n_filters must all \
                 be > 0"
                    .to_owned(),
            ));
        }
        if s.encoder_ratios.is_empty() {
            return Err(VokraError::InvalidArgument(
                "vibevoice semantic tokenizer: encoder_ratios must be non-empty".to_owned(),
            ));
        }
        for (i, r) in s.encoder_ratios.iter().enumerate() {
            if *r == 0 {
                return Err(VokraError::InvalidArgument(format!(
                    "vibevoice semantic tokenizer: encoder_ratios[{i}] must be > 0"
                )));
            }
        }
        if s.std_dist_type != "none" && s.std_dist_type != "gaussian" {
            return Err(VokraError::InvalidArgument(format!(
                "vibevoice semantic tokenizer: std_dist_type = {:?} is not one of {{\"none\", \
                 \"gaussian\"}} (VibeVoice-1.5B ships \"none\" — a deterministic head)",
                s.std_dist_type,
            )));
        }

        // Diffusion head axes.
        let h = &self.diffusion_head;
        if h.hidden_size == 0 || h.head_layers == 0 || h.latent_size == 0 || h.speech_vae_dim == 0 {
            return Err(VokraError::InvalidArgument(
                "vibevoice diffusion head: hidden_size / head_layers / latent_size / \
                 speech_vae_dim must all be > 0"
                    .to_owned(),
            ));
        }
        if h.ddpm_num_steps == 0 || h.ddpm_num_inference_steps == 0 {
            return Err(VokraError::InvalidArgument(
                "vibevoice diffusion head: ddpm_num_steps / ddpm_num_inference_steps must be > 0"
                    .to_owned(),
            ));
        }
        if h.ddpm_num_inference_steps > h.ddpm_num_steps {
            return Err(VokraError::InvalidArgument(format!(
                "vibevoice diffusion head: ddpm_num_inference_steps ({}) must be <= \
                 ddpm_num_steps ({})",
                h.ddpm_num_inference_steps, h.ddpm_num_steps,
            )));
        }
        if !(h.head_ffn_ratio.is_finite() && h.head_ffn_ratio > 0.0) {
            return Err(VokraError::InvalidArgument(format!(
                "vibevoice diffusion head: head_ffn_ratio must be a positive finite f32 (got {})",
                h.head_ffn_ratio,
            )));
        }
        if !(h.rms_norm_eps.is_finite() && h.rms_norm_eps > 0.0) {
            return Err(VokraError::InvalidArgument(format!(
                "vibevoice diffusion head: rms_norm_eps must be a positive finite f32 (got {})",
                h.rms_norm_eps,
            )));
        }

        // Top-level shortcut consistency.
        if self.acoustic_vae_dim != self.acoustic.vae_dim {
            return Err(VokraError::InvalidArgument(format!(
                "vibevoice config: acoustic_vae_dim ({}) != acoustic.vae_dim ({}) — the two axes \
                 are the same architectural quantity carried by upstream `config.json` at two \
                 levels and MUST agree",
                self.acoustic_vae_dim, self.acoustic.vae_dim,
            )));
        }
        if self.semantic_vae_dim != self.semantic.vae_dim {
            return Err(VokraError::InvalidArgument(format!(
                "vibevoice config: semantic_vae_dim ({}) != semantic.vae_dim ({}) — the two axes \
                 are the same architectural quantity carried by upstream `config.json` at two \
                 levels and MUST agree",
                self.semantic_vae_dim, self.semantic.vae_dim,
            )));
        }

        // Cross-config handshakes.
        // (1) Diffusion head runs in the acoustic VAE latent space.
        if h.latent_size != a.vae_dim {
            return Err(VokraError::InvalidArgument(format!(
                "vibevoice config: diffusion_head.latent_size ({}) != acoustic.vae_dim ({}) — \
                 the diffusion head predicts velocity in the acoustic VAE latent space; a \
                 silent mismatch would drop or duplicate channels between the head and the \
                 acoustic decoder",
                h.latent_size, a.vae_dim,
            )));
        }
        if h.speech_vae_dim != a.vae_dim {
            return Err(VokraError::InvalidArgument(format!(
                "vibevoice config: diffusion_head.speech_vae_dim ({}) != acoustic.vae_dim ({}) — \
                 the two axes carry the same architectural quantity",
                h.speech_vae_dim, a.vae_dim,
            )));
        }
        // (2) Diffusion head hidden_size == LM hidden_size (so cond_proj
        //     is a square linear — the upstream code assumes this).
        if h.hidden_size != d.hidden_dim {
            return Err(VokraError::InvalidArgument(format!(
                "vibevoice config: diffusion_head.hidden_size ({}) != decoder.hidden_dim ({}) — \
                 the head's cond_proj is a square linear so the LM hidden state feeds AdaLN \
                 without an extra projection",
                h.hidden_size, d.hidden_dim,
            )));
        }
        // (3) Acoustic ≠ semantic vae_dim (a common transcription error).
        if a.vae_dim == s.vae_dim {
            return Err(VokraError::InvalidArgument(format!(
                "vibevoice config: acoustic.vae_dim ({}) == semantic.vae_dim ({}) — the two \
                 tokenizers carry DIFFERENT latent widths (VibeVoice-1.5B: 64 vs 128); a match \
                 here is almost certainly a transcription error, not a legitimate variant",
                a.vae_dim, s.vae_dim,
            )));
        }

        // Sampler-config axis derivation must succeed.
        self.ddpm_sampler_config()?;

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Weight-store scaffold
// ---------------------------------------------------------------------------

/// VibeVoice-1.5B weight store scaffold.
///
/// Real binding is a follow-up wave (T29-equivalent — the Qwen2 LM
/// walk, the acoustic tokenizer walk, the semantic tokenizer walk,
/// and the diffusion-head walk all defer to the T29 tensor-name
/// manifest fetch). This scaffold carries only aggregate byte
/// bundles so downstream shape flow / handshake tests are unblocked;
/// the sole invariant this slice pins is that
/// `is_synthesized = true` prevents a spurious synthesize call from
/// returning zero audio (FR-EX-08 — the loud
/// [`VibeVoiceTts::synthesize`] guard).
#[derive(Debug, Clone)]
pub struct VibeVoiceWeights {
    /// Placeholder for the Qwen2 decoder-LM tensor bytes (aggregate).
    /// Real binding walks the upstream `model.layers.*` naming.
    pub decoder: Vec<f32>,
    /// Placeholder for the acoustic tokenizer tensor bytes
    /// (aggregate). Real binding walks the upstream
    /// `acoustic_tokenizer.*` naming.
    pub acoustic: Vec<f32>,
    /// Placeholder for the semantic tokenizer tensor bytes
    /// (aggregate). Real binding walks the upstream
    /// `semantic_tokenizer.*` naming.
    pub semantic: Vec<f32>,
    /// Placeholder for the diffusion-head tensor bytes (aggregate).
    /// Real binding walks the upstream `prediction_head.*` naming
    /// (`noisy_images_proj` / `cond_proj` / `t_embedder.mlp.*` /
    /// `layers.*.{ffn,norm,adaLN_modulation}.*` /
    /// `final_layer.{norm_final,linear,adaLN_modulation}.*`).
    pub diffusion_head: Vec<f32>,
    /// `true` when built by [`Self::synthesized`] — never a real
    /// upstream checkpoint.
    pub is_synthesized: bool,
}

impl VibeVoiceWeights {
    /// Builds a deterministic zero-initialized fixture (shape
    /// scaffold only — every slot is `Vec::new()`).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] if
    /// `config.validate_for_forward` fails.
    pub fn synthesized(config: &VibeVoiceConfig) -> Result<Self> {
        config.validate_for_forward()?;
        Ok(Self {
            decoder: Vec::new(),
            acoustic: Vec::new(),
            semantic: Vec::new(),
            diffusion_head: Vec::new(),
            is_synthesized: true,
        })
    }
}

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

/// VibeVoice-1.5B TTS engine handle.
///
/// Carries the resolved config + weight store + a derived
/// [`ContinuousVaeConfig`] (acoustic VAE) + a derived
/// [`DdpmSamplerConfig`]. [`Self::synthesize`] is the primary text →
/// PCM entry point; until real weights are bound and the LM →
/// diffusion-head → DDPM sampler → AudioVAE decode → 24 kHz PCM
/// chain is wired end-to-end (T29-equivalent follow-up wave), it
/// returns [`VokraError::NotImplemented`] naming the blocker
/// (FR-EX-08 — never a silent zero-fill or empty audio buffer).
#[derive(Debug, Clone)]
pub struct VibeVoiceTts {
    cfg: VibeVoiceConfig,
    weights: VibeVoiceWeights,
    acoustic_vae: ContinuousVaeConfig,
    sampler: DdpmSamplerConfig,
}

impl VibeVoiceTts {
    /// Assembles an engine from `cfg` and `weights`. Cross-checks the
    /// config, the acoustic VAE seam, and the DDPM sampler seam at
    /// construction time so a mismatched trio fails loudly here
    /// rather than deep inside a forward.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] from
    /// `cfg.validate_for_forward`, `cfg.acoustic_vae_config`, or
    /// `cfg.ddpm_sampler_config`.
    pub fn new(cfg: VibeVoiceConfig, weights: VibeVoiceWeights) -> Result<Self> {
        cfg.validate_for_forward()?;
        let acoustic_vae = cfg.acoustic_vae_config();
        acoustic_vae.validate_for_forward()?;
        let sampler = cfg.ddpm_sampler_config()?;
        Ok(Self {
            cfg,
            weights,
            acoustic_vae,
            sampler,
        })
    }

    /// The resolved configuration.
    #[must_use]
    pub fn config(&self) -> &VibeVoiceConfig {
        &self.cfg
    }

    /// The active acoustic VAE config
    /// (canonical: `latent_dim=64`, 24 kHz in/out).
    #[must_use]
    pub fn acoustic_vae_config(&self) -> &ContinuousVaeConfig {
        &self.acoustic_vae
    }

    /// The active DDPM sampler config
    /// (canonical: 20-step DDIM v-prediction, cosine β).
    #[must_use]
    pub fn ddpm_sampler_config(&self) -> &DdpmSamplerConfig {
        &self.sampler
    }

    /// True iff the weight store was built by
    /// [`VibeVoiceWeights::synthesized`] (never a real upstream
    /// checkpoint).
    #[must_use]
    pub fn is_synthesized(&self) -> bool {
        self.weights.is_synthesized
    }

    /// Synthesizes PCM for `text` at 24 kHz mono.
    ///
    /// This is the primary text → PCM entry point. **Real weights
    /// required**: synthesized-weight builds cannot produce
    /// meaningful audio, so this returns
    /// [`VokraError::NotImplemented`] naming the blocker (FR-EX-08 —
    /// never a silent zero-fill or empty audio buffer).
    ///
    /// # Errors
    ///
    /// - [`VokraError::InvalidArgument`] if `text` is empty.
    /// - [`VokraError::NotImplemented`] otherwise (real forward not
    ///   yet bound — FR-EX-08).
    pub fn synthesize(&self, text: &str) -> Result<Vec<f32>> {
        if text.is_empty() {
            return Err(VokraError::InvalidArgument(
                "vibevoice synthesize: text is empty".to_owned(),
            ));
        }
        if self.weights.is_synthesized {
            return Err(VokraError::NotImplemented(
                "vibevoice synthesize: this engine holds synthesized weights \
                 (deterministic scaffold fixture from VibeVoiceWeights::synthesized) — \
                 synthesized-weight audio would be a hallucinated waveform, not real speech. \
                 Bind real VibeVoice-1.5B weights (MIT, huggingface.co/microsoft/VibeVoice-1.5B) \
                 before invoking synthesize. The shape flow (config validation, weight-store \
                 construction, text-empty check, VAE + sampler handshake) is exercised through \
                 VibeVoiceTts::new; real-checkpoint binding lands in a follow-up wave \
                 (T29-equivalent).",
            ));
        }
        Err(VokraError::NotImplemented(
            "vibevoice synthesize: real weights are bound but the Qwen2 decoder LM → diffusion \
             head → DDPM sampler (vokra_ops::ddpm_sample, v-prediction / cosine β / 20 inference \
             steps) → acoustic VAE decode (vokra_ops::vae_continuous_decode, 24 kHz PCM out) \
             forward path has not landed yet. Follow-up wave (T29-equivalent): (1) run the \
             Qwen2 LM (GQA 12 Q ÷ 2 KV / RoPE θ=1_000_000 / RMSNorm ε=1e-6 / SwiGLU / tied \
             word embeddings / max_position_embeddings=65_536) with the tokenizer prompt; (2) \
             at each acoustic frame, drive vokra_ops::ddpm_sample with the 4-layer diffusion \
             head as the v-prediction closure (AdaLN-modulated Linear+SwiGLU MLP receiving the \
             LM hidden state as `c` and the sinusoidal `timestep_embedding(t)` fused via \
             `t_embedder.mlp`); (3) decode the recovered continuous acoustic latent through \
             vokra_ops::vae_continuous_decode → 24 kHz PCM (the shared Phase 4 primitive). \
             The semantic tokenizer runs the encoder chain on the audio prompt (24 kHz PCM \
             → 7.5 Hz continuous 128-d latents) as LM conditioning; VibeVoice does NOT decode \
             the semantic latents back to audio.",
        ))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Arch tag distinctness ------------------------------------------

    #[test]
    fn expected_arch_is_vibevoice() {
        assert_eq!(EXPECTED_ARCH, "vibevoice");
    }

    #[test]
    fn arch_is_distinct_from_neighbouring_families() {
        // VibeVoice's terminal decoding hop is the acoustic VAE decoder
        // driven by the DDPM sampler — neither vocoder-LM (HiFTChain) nor
        // codec-LM (any RVQ / FSQ codec) nor VoxCPM's UnifiedCFM flow-
        // matching sampler. Silently sharing an arch tag would misroute
        // the runtime dispatch.
        assert_ne!(EXPECTED_ARCH, "voxcpm2");
        assert_ne!(EXPECTED_ARCH, crate::voxcpm2::EXPECTED_ARCH);
        assert_ne!(EXPECTED_ARCH, crate::cosyvoice3::EXPECTED_ARCH);
        assert_ne!(EXPECTED_ARCH, crate::qwen3_tts::EXPECTED_ARCH);
        assert_ne!(EXPECTED_ARCH, crate::chatterbox::EXPECTED_ARCH);
        assert_ne!(EXPECTED_ARCH, crate::chatterbox_nano::EXPECTED_ARCH);
        assert_ne!(EXPECTED_ARCH, crate::chatterbox_turbo::EXPECTED_ARCH);
        assert_ne!(EXPECTED_ARCH, crate::dia::EXPECTED_ARCH);
        assert_ne!(EXPECTED_ARCH, crate::zonos::EXPECTED_ARCH);
    }

    // ---- Sample-rate + frame-rate constants -----------------------------

    #[test]
    fn encoder_sample_rate_matches_upstream() {
        assert_eq!(VIBEVOICE_ENCODER_SAMPLE_RATE, 24_000);
    }

    #[test]
    fn lm_frame_rate_matches_upstream() {
        // 24_000 / (8 * 5 * 5 * 4 * 2 * 2) = 24_000 / 3200 = 7.5 Hz.
        assert!((VIBEVOICE_LM_FRAME_RATE_HZ - 7.5).abs() < 1e-6);
        let a = VibeVoiceAcousticTokenizerConfig::vibevoice_1_5b();
        assert_eq!(a.hop_length(), Some(3200));
        assert!((a.frame_rate_hz(VIBEVOICE_ENCODER_SAMPLE_RATE).unwrap() - 7.5).abs() < 1e-6,);
    }

    // ---- Primary-source pins --------------------------------------------

    /// Every hparam carries its primary-source value transcribed from
    /// `microsoft/VibeVoice-1.5B/config.json` (fetched 2026-07-24).
    #[test]
    fn decoder_config_matches_primary_source() {
        let d = VibeVoiceDecoderConfig::vibevoice_1_5b();
        assert_eq!(d.hidden_dim, 1536);
        assert_eq!(d.n_layer, 28);
        assert_eq!(d.n_head, 12);
        assert_eq!(d.n_head_kv, 2);
        assert_eq!(d.ffn_dim, 8960);
        assert_eq!(d.vocab_size, 151_936);
        assert_eq!(d.max_position_embeddings, 65_536);
        assert!((d.rope_base - 1_000_000.0).abs() < 1e-3);
        assert!((d.rms_norm_eps - 1e-6).abs() < 1e-9);
        assert!((d.attention_dropout - 0.0).abs() < 1e-9);
        assert!(d.tie_word_embeddings);
        assert!(!d.use_sliding_window);
        assert_eq!(d.max_window_layers, 28);
    }

    #[test]
    fn acoustic_tokenizer_config_matches_primary_source() {
        let a = VibeVoiceAcousticTokenizerConfig::vibevoice_1_5b();
        assert_eq!(a.channels, 1);
        assert!(a.causal);
        assert_eq!(a.vae_dim, 64);
        assert!((a.fix_std - 0.5).abs() < 1e-6);
        assert_eq!(a.std_dist_type, "gaussian");
        assert_eq!(a.encoder_n_filters, 32);
        assert_eq!(a.decoder_n_filters, 32);
        assert_eq!(a.encoder_ratios, vec![8, 5, 5, 4, 2, 2]);
        assert_eq!(a.decoder_ratios, vec![8, 5, 5, 4, 2, 2]);
        assert_eq!(a.encoder_depths, "3-3-3-3-3-3-8");
        assert!((a.layer_scale_init_value - 1e-6).abs() < 1e-9);
        assert!((a.weight_init_value - 1e-2).abs() < 1e-9);
        assert_eq!(a.layernorm, "RMSNorm");
        assert!(a.layernorm_elementwise_affine);
        assert!((a.layernorm_eps - 1e-5).abs() < 1e-9);
        assert_eq!(a.mixer_layer, "depthwise_conv");
        assert_eq!(a.pad_mode, "constant");
        assert!(a.disable_last_norm);
        assert_eq!(a.conv_norm, "none");
        assert!(a.conv_bias);
        assert!((a.corpus_normalize - 0.0).abs() < 1e-9);
    }

    #[test]
    fn semantic_tokenizer_config_matches_primary_source() {
        let s = VibeVoiceSemanticTokenizerConfig::vibevoice_1_5b();
        assert_eq!(s.channels, 1);
        assert!(s.causal);
        assert_eq!(s.vae_dim, 128);
        assert!((s.fix_std - 0.0).abs() < 1e-9);
        assert_eq!(s.std_dist_type, "none");
        assert_eq!(s.encoder_n_filters, 32);
        assert_eq!(s.encoder_ratios, vec![8, 5, 5, 4, 2, 2]);
        assert_eq!(s.encoder_depths, "3-3-3-3-3-3-8");
        assert_eq!(s.layernorm, "RMSNorm");
        assert!((s.layernorm_eps - 1e-5).abs() < 1e-9);
        assert_eq!(s.mixer_layer, "depthwise_conv");
    }

    #[test]
    fn diffusion_head_config_matches_primary_source() {
        let h = VibeVoiceDiffusionHeadConfig::vibevoice_1_5b();
        assert_eq!(h.hidden_size, 1536);
        assert_eq!(h.head_layers, 4);
        assert!((h.head_ffn_ratio - 3.0).abs() < 1e-6);
        assert!((h.rms_norm_eps - 1e-5).abs() < 1e-9);
        assert_eq!(h.latent_size, 64);
        assert_eq!(h.speech_vae_dim, 64);
        assert_eq!(h.prediction_type, "v_prediction");
        assert_eq!(h.diffusion_type, "ddpm");
        assert_eq!(h.ddpm_num_steps, 1000);
        assert_eq!(h.ddpm_num_inference_steps, 20);
        assert_eq!(h.ddpm_beta_schedule, "cosine");
        assert_eq!(h.ddpm_batch_mul, 4);
    }

    #[test]
    fn diffusion_head_ffn_inner_dim_matches_upstream_formula() {
        // int(1536 * 3.0) = 4608 — matches the upstream
        // `HeadLayer(embed_dim, ffn_dim=int(hidden_size * head_ffn_ratio))`.
        let h = VibeVoiceDiffusionHeadConfig::vibevoice_1_5b();
        assert_eq!(h.ffn_inner_dim(), 4608);
    }

    #[test]
    fn top_level_config_matches_primary_source() {
        let c = VibeVoiceConfig::vibevoice_1_5b();
        assert_eq!(c.acoustic_vae_dim, 64);
        assert_eq!(c.semantic_vae_dim, 128);
    }

    // ---- Handshakes hold by construction ---------------------------------

    #[test]
    fn vae_handshake_holds_by_construction_for_canonical_release() {
        let c = VibeVoiceConfig::vibevoice_1_5b();
        // diffusion_head.latent_size == acoustic.vae_dim.
        assert_eq!(c.diffusion_head.latent_size, c.acoustic.vae_dim);
        // diffusion_head.speech_vae_dim == acoustic.vae_dim.
        assert_eq!(c.diffusion_head.speech_vae_dim, c.acoustic.vae_dim);
        // diffusion_head.hidden_size == decoder.hidden_dim (square cond_proj).
        assert_eq!(c.diffusion_head.hidden_size, c.decoder.hidden_dim);
        // Top-level shortcuts agree with the sub-configs.
        assert_eq!(c.acoustic_vae_dim, c.acoustic.vae_dim);
        assert_eq!(c.semantic_vae_dim, c.semantic.vae_dim);
        c.validate_for_forward().expect("canonical must validate");
    }

    #[test]
    fn tiny_config_validates() {
        let c = VibeVoiceConfig::tiny_for_tests();
        c.validate_for_forward().expect("tiny must validate");
    }

    // ---- Derived seams ---------------------------------------------------

    #[test]
    fn acoustic_vae_config_carries_upstream_axes() {
        let c = VibeVoiceConfig::vibevoice_1_5b();
        let vae = c.acoustic_vae_config();
        // Mirror-symmetric — same rate for encoder input and decoder output.
        assert_eq!(vae.sample_rate_hz, 24_000);
        assert_eq!(vae.out_sample_rate_hz, 24_000);
        assert_eq!(vae.encoder_dim, 32);
        assert_eq!(vae.decoder_dim, 32);
        assert_eq!(vae.encoder_rates, vec![8, 5, 5, 4, 2, 2]);
        assert_eq!(vae.decoder_rates, vec![8, 5, 5, 4, 2, 2]);
        assert_eq!(vae.latent_dim, 64);
        assert!(vae.depthwise);
        assert!(!vae.use_noise_block);
        vae.validate_for_forward()
            .expect("derived acoustic VAE config must validate");
    }

    #[test]
    fn ddpm_sampler_config_carries_upstream_axes() {
        let c = VibeVoiceConfig::vibevoice_1_5b();
        let s = c
            .ddpm_sampler_config()
            .expect("v_prediction + cosine map to enum arms");
        assert_eq!(s.num_train_steps, 1000);
        assert_eq!(s.num_inference_steps, 20);
        assert_eq!(s.prediction_type, PredictionType::VPrediction);
        assert_eq!(s.beta_schedule, BetaSchedule::Cosine);
    }

    #[test]
    fn ddpm_sampler_config_rejects_unknown_prediction_type() {
        let mut c = VibeVoiceConfig::vibevoice_1_5b();
        c.diffusion_head.prediction_type = "not-a-real-target".to_owned();
        let err = c.ddpm_sampler_config().unwrap_err();
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn ddpm_sampler_config_rejects_unknown_beta_schedule() {
        let mut c = VibeVoiceConfig::vibevoice_1_5b();
        c.diffusion_head.ddpm_beta_schedule = "not-a-real-schedule".to_owned();
        let err = c.ddpm_sampler_config().unwrap_err();
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    // ---- Validation surface ---------------------------------------------

    #[test]
    fn gqa_algebra_enforced() {
        let mut c = VibeVoiceConfig::tiny_for_tests();
        c.decoder.n_head_kv = 3; // 4 % 3 != 0.
        assert!(c.validate_for_forward().is_err());
    }

    #[test]
    fn zero_axes_rejected() {
        let mut c = VibeVoiceConfig::vibevoice_1_5b();
        c.decoder.hidden_dim = 0;
        assert!(c.validate_for_forward().is_err());

        let mut c = VibeVoiceConfig::vibevoice_1_5b();
        c.acoustic.vae_dim = 0;
        assert!(c.validate_for_forward().is_err());

        let mut c = VibeVoiceConfig::vibevoice_1_5b();
        c.semantic.vae_dim = 0;
        assert!(c.validate_for_forward().is_err());

        let mut c = VibeVoiceConfig::vibevoice_1_5b();
        c.diffusion_head.head_layers = 0;
        assert!(c.validate_for_forward().is_err());
    }

    #[test]
    fn negative_rms_norm_eps_rejected() {
        let mut c = VibeVoiceConfig::vibevoice_1_5b();
        c.decoder.rms_norm_eps = -1e-5;
        assert!(c.validate_for_forward().is_err());
        let mut c = VibeVoiceConfig::vibevoice_1_5b();
        c.diffusion_head.rms_norm_eps = -1e-5;
        assert!(c.validate_for_forward().is_err());
    }

    #[test]
    fn vae_dim_swap_between_tokenizers_rejected() {
        // A silent swap of acoustic ↔ semantic vae_dim would still
        // arithmetically type-check but is almost certainly a
        // transcription error (canonical: 64 vs 128).
        let mut c = VibeVoiceConfig::vibevoice_1_5b();
        c.acoustic.vae_dim = 128;
        c.semantic.vae_dim = 128;
        c.acoustic_vae_dim = 128;
        c.semantic_vae_dim = 128;
        // Also fix the diffusion head so we're specifically testing the
        // acoustic == semantic guard (not the head handshake).
        c.diffusion_head.latent_size = 128;
        c.diffusion_head.speech_vae_dim = 128;
        assert!(c.validate_for_forward().is_err());
    }

    #[test]
    fn top_level_vs_subconfig_vae_dim_drift_rejected() {
        let mut c = VibeVoiceConfig::vibevoice_1_5b();
        c.acoustic_vae_dim = 96; // != acoustic.vae_dim = 64
        assert!(c.validate_for_forward().is_err());

        let mut c = VibeVoiceConfig::vibevoice_1_5b();
        c.semantic_vae_dim = 96; // != semantic.vae_dim = 128
        assert!(c.validate_for_forward().is_err());
    }

    #[test]
    fn diffusion_head_hidden_size_must_match_decoder_hidden_dim() {
        let mut c = VibeVoiceConfig::vibevoice_1_5b();
        c.diffusion_head.hidden_size = 1024; // != decoder.hidden_dim = 1536
        assert!(c.validate_for_forward().is_err());
    }

    #[test]
    fn diffusion_head_latent_size_must_match_acoustic_vae_dim() {
        let mut c = VibeVoiceConfig::vibevoice_1_5b();
        c.diffusion_head.latent_size = 32; // != acoustic.vae_dim = 64
        assert!(c.validate_for_forward().is_err());
    }

    #[test]
    fn ddpm_num_inference_steps_must_not_exceed_ddpm_num_steps() {
        let mut c = VibeVoiceConfig::vibevoice_1_5b();
        c.diffusion_head.ddpm_num_steps = 100;
        c.diffusion_head.ddpm_num_inference_steps = 200;
        assert!(c.validate_for_forward().is_err());
    }

    #[test]
    fn semantic_std_dist_type_bounded_to_known_values() {
        let mut c = VibeVoiceConfig::vibevoice_1_5b();
        c.semantic.std_dist_type = "not-a-real-kind".to_owned();
        assert!(c.validate_for_forward().is_err());
    }

    // ---- Placeholder detection ------------------------------------------

    #[test]
    fn placeholder_shape_detects_zeroed_config() {
        let mut c = VibeVoiceConfig::vibevoice_1_5b();
        c.decoder.hidden_dim = 0;
        c.decoder.n_layer = 0;
        c.diffusion_head.hidden_size = 0;
        c.diffusion_head.head_layers = 0;
        c.acoustic.vae_dim = 0;
        c.semantic.vae_dim = 0;
        assert!(c.is_placeholder_shape());
    }

    // ---- Weight + engine surface ----------------------------------------

    #[test]
    fn synthesized_weights_scaffold_flag_carried() {
        let c = VibeVoiceConfig::vibevoice_1_5b();
        let w = VibeVoiceWeights::synthesized(&c).unwrap();
        assert!(w.is_synthesized);
    }

    #[test]
    fn tts_new_binds_canonical_release() {
        let c = VibeVoiceConfig::vibevoice_1_5b();
        let w = VibeVoiceWeights::synthesized(&c).unwrap();
        let tts = VibeVoiceTts::new(c, w).unwrap();
        assert!(tts.is_synthesized());
        assert_eq!(tts.acoustic_vae_config().latent_dim, 64);
        assert_eq!(tts.acoustic_vae_config().sample_rate_hz, 24_000);
        assert_eq!(tts.ddpm_sampler_config().num_inference_steps, 20);
        assert_eq!(
            tts.ddpm_sampler_config().prediction_type,
            PredictionType::VPrediction
        );
        assert_eq!(
            tts.ddpm_sampler_config().beta_schedule,
            BetaSchedule::Cosine
        );
    }

    #[test]
    fn tts_synthesize_empty_text_rejected_loudly() {
        let c = VibeVoiceConfig::vibevoice_1_5b();
        let w = VibeVoiceWeights::synthesized(&c).unwrap();
        let tts = VibeVoiceTts::new(c, w).unwrap();
        let err = tts.synthesize("").expect_err("empty text");
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn tts_synthesize_on_synth_weights_is_not_implemented_and_says_why() {
        let c = VibeVoiceConfig::vibevoice_1_5b();
        let w = VibeVoiceWeights::synthesized(&c).unwrap();
        let tts = VibeVoiceTts::new(c, w).unwrap();
        let err = tts.synthesize("hello").expect_err("synth synthesize");
        assert!(
            matches!(err, VokraError::NotImplemented(_)),
            "synth synth must be NotImplemented, got {err:?}"
        );
    }

    // ---- Tiny-config plumbing --------------------------------------------

    #[test]
    fn tiny_config_tts_new_produces_ready_engine() {
        // Sanity that the tiny-fixture keeps handshake integrity: the
        // scaled-down tokenizers, LM, and diffusion head all agree on
        // the shared dims.
        let c = VibeVoiceConfig::tiny_for_tests();
        assert_eq!(c.diffusion_head.latent_size, c.acoustic.vae_dim);
        assert_eq!(c.diffusion_head.hidden_size, c.decoder.hidden_dim);
        assert_ne!(c.acoustic.vae_dim, c.semantic.vae_dim);
        let w = VibeVoiceWeights::synthesized(&c).unwrap();
        let tts = VibeVoiceTts::new(c, w).unwrap();
        assert!(tts.is_synthesized());
    }
}
