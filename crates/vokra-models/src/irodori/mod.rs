//! **Irodori-TTS** — Aratako's Japanese Rectified-Flow Diffusion Transformer
//! TTS (SoTA plan Phase 5 JA-TTS-1, 2026-07-24).
//!
//! # What Irodori-TTS is (primary source)
//!
//! `Aratako/Irodori-TTS-500M-v3` is a Japanese TTS model that samples
//! continuous DACVAE latents with a **Rectified-Flow Diffusion Transformer
//! (RF-DiT)** over a 32-dim continuous latent stream and reconstructs 48
//! kHz PCM via the paired `Aratako/Semantic-DACVAE-Japanese-32dim` codec
//! (a variant of the Meta [`facebookresearch/dacvae`] DAC-VAE, Apache 2.0).
//! The architecture and training design largely follow **Echo-TTS**
//! (Darefsky 2025), and Aratako's repository ships the training + inference
//! Python code under MIT (`gh api /repos/Aratako/Irodori-TTS/license` →
//! `MIT`, fetched 2026-07-24 — CLAUDE.md「ハルシネーション厳禁」).
//!
//! Every architectural axis below is transcribed **verbatim** from
//! `configs/train_500m_v3_phase1_body.yaml` +
//! `configs/train_500m_v3_phase2_duration.yaml` +
//! `irodori_tts/config.py::ModelConfig` at
//! `github.com/Aratako/Irodori-TTS` (fetched 2026-07-24):
//!
//! - **Latent (patched) space** — `latent_dim = 32`,
//!   `latent_patch_size = 1` → the DiT operates on 32-d tokens (one DACVAE
//!   frame per position).
//! - **RF-DiT body** — `model_dim = 1280`, `num_layers = 12`,
//!   `num_heads = 20` (`head_dim = 64`), `mlp_ratio = 2.875` (SwiGLU FFN),
//!   Low-Rank AdaLN modulation with `adaln_rank = 192` and
//!   `timestep_embed_dim = 512` (sinusoidal timestep embedding — see
//!   `irodori_tts/model.py::get_timestep_embedding`).
//! - **Text encoder** — `text_vocab_size = 99574`,
//!   `text_tokenizer_repo = "llm-jp/llm-jp-3-150m"` (Apache-2.0),
//!   `text_add_bos = true`, `text_dim = 512`, `text_layers = 10`,
//!   `text_heads = 8`, `text_mlp_ratio = 2.6`, initialized from a
//!   pretrained LLM checkpoint (LLM-JP-3 150M). Attention adds RoPE and a
//!   sigmoid gate on the output projection
//!   (`irodori_tts/model.py::SelfAttention`).
//! - **Reference (speaker) latent encoder** — self-attention transformer
//!   over patched reference DACVAE latents for speaker/style conditioning.
//!   `speaker_dim = 768`, `speaker_layers = 8`, `speaker_heads = 12`
//!   (`head_dim = 64`), `speaker_patch_size = 1`, `speaker_mlp_ratio = 2.6`.
//! - **Duration predictor (v3 phase-2)** — `use_duration_predictor = true`
//!   for the released v3 base checkpoint (integrated automatic length
//!   estimation). Axes: `duration_aux_dim = 14`,
//!   `duration_hidden_dim = 1024`, `duration_layers = 3`,
//!   `duration_attention_heads = 8`, `duration_dropout = 0.1`,
//!   `duration_architecture = "token_sum_adarn_zero_no_aux"`,
//!   `duration_token_init_frames = 9.0`,
//!   `duration_speaker_fusion = "adarn_zero"`.
//! - **Normalization** — RMSNorm ε = `1e-5` (`ModelConfig.norm_eps`).
//! - **Sampling** — Euler ODE over the rectified-flow ODE
//!   `x_t = (1-t) x_0 + t z`, `v = z - x_0`, evolved from `t = 1` (noise)
//!   to `t = 0` (data). `num_steps = 40` default, split-batch CFG on
//!   three independent axes (text / caption / speaker) with
//!   per-axis scales `cfg_scale_text = 3.0`, `cfg_scale_caption = 3.0`,
//!   `cfg_scale_speaker = 5.0` and windowing `cfg_min_t = 0.5`,
//!   `cfg_max_t = 1.0`. Schedule ∈ {Linear, Sway (F5-TTS)} — both
//!   supported natively by [`vokra_ops::flow_sampler`] (M3-05).
//!
//! # Terminal codec — Semantic-DACVAE-Japanese-32dim
//!
//! Irodori-TTS decodes to PCM via **Semantic-DACVAE-Japanese-32dim**
//! (`huggingface.co/Aratako/Semantic-DACVAE-Japanese-32dim`), a
//! `dacvae.DACVAE` variant of the Meta open-source
//! [`facebookresearch/dacvae`] codec (Apache 2.0). Two axes are pinned by
//! the release: **`latent_dim = 32`** (matches the RF-DiT latent stream)
//! and **`sample_rate = 48_000`** (48 kHz PCM out, per the base model
//! card at `huggingface.co/Aratako/Irodori-TTS-500M-v3`); the exact
//! `encoder_rates` / `decoder_rates` are set inside the checkpoint blob
//! (`weights.pth`) and are NOT part of the model's public config — they
//! ride the codec GGUF. Callers inject the codec through
//! [`IrodoriTts::with_codec`] once the paired GGUF is prepared (the same
//! `DacCodecGguf`-shaped seam Dia + Zonos use with vanilla DAC).
//! Until then [`IrodoriTts::synthesize`] returns
//! [`VokraError::NotImplemented`] naming the blocker (FR-EX-08 — never
//! a silent zero-fill).
//!
//! # Distinct topology axis: DiT over continuous DACVAE latents
//!
//! Irodori-TTS is the **third** Vokra target whose terminal decoding hop
//! runs a continuous-latent generator over a continuous VAE decoder
//! (after VoxCPM-0.5B and VibeVoice-1.5B) — but this time the DiT is
//! trained with **Rectified Flow** (RF; Liu et al. 2022,
//! arxiv 2209.03003) instead of DDPM (VibeVoice) or the UnifiedCFM
//! flow-matching sampler with an EpsS-style schedule (VoxCPM). Sampling
//! integrates the RF ODE with an **Euler** step over a **Linear** or
//! **Sway** schedule — both directly supported by
//! [`vokra_ops::flow_sampler`], so no new sampler primitive is added by
//! this model.
//!
//! # Reuses existing ops
//!
//! - **RF sampler**: shared [`vokra_ops::flow_sampler`] primitive (M3-05)
//!   — `OdeSolver::Euler` + `Schedule::Linear` (default) or
//!   `Schedule::Sway` (F5-TTS toggle).
//! - **DACVAE decoder**: shared [`crate::codec::DacCodecGguf`] seam — a
//!   paired `Semantic-DACVAE-Japanese-32dim` GGUF is a stock DAC-family
//!   codec injected via [`IrodoriTts::with_codec`].
//!
//! No new backend kernel is added — every DiT / text-encoder / speaker-
//! encoder building block is Linear + RMSNorm + SwiGLU + RoPE + softmax
//! attention, all covered by the existing kernel inventory.
//!
//! # What lands in this Phase 5 slice
//!
//! - [`IrodoriDitConfig`] / [`IrodoriTextEncoderConfig`] /
//!   [`IrodoriSpeakerEncoderConfig`] / [`IrodoriDurationPredictorConfig`] /
//!   [`IrodoriConfig`] — every architectural hparam transcribed **verbatim**
//!   from the primary sources. [`IrodoriConfig::validate_for_forward`]
//!   fails loudly (FR-EX-08) on zeroed axes, non-even `head_dim` (RoPE
//!   pairs), or malformed FFN inner widths.
//! - [`IrodoriWeights`] — deterministic
//!   [`IrodoriWeights::synthesized`] scaffold fixture (zero-initialized;
//!   only the shape flow is exercised — the real safetensors walk is a
//!   follow-up wave).
//! - [`IrodoriTts`] — engine handle carrying config + weights + an
//!   optional [`DacCodecGguf`] codec binding. The primary
//!   [`IrodoriTts::synthesize`] entry point returns
//!   [`VokraError::NotImplemented`] naming the blocker until real
//!   weights are bound AND a codec GGUF is injected AND the full
//!   text-encoder → speaker-encoder → RF-DiT → codec-decode chain is
//!   wired end-to-end (T29-equivalent follow-up wave — never a silent
//!   zero-fill, FR-EX-08).
//!
//! # No ONNX (permanent)
//!
//! Irodori-TTS is distributed as safetensors + a Python pipeline
//! (`irodori_tts/inference_runtime.py`); the runtime **never** loads an
//! ONNX graph (FR-LD-05, permanent constraint); the pipeline is
//! re-implemented natively from the safetensors checkpoint (whisper.cpp
//! 型 self re-implementation, CLAUDE.md 設計判断 4).

use vokra_core::{Result, VokraError};

use crate::codec::DacCodecGguf;

// Public seam re-exports — shared with the RF sampler primitive.
pub use vokra_ops::flow_sampler::{
    CfgMode, CfgScaleProfile, FlowSamplerConfig, OdeSolver, Schedule,
};

/// `vokra.model.arch` an Irodori-TTS GGUF must carry. Written by
/// `vokra-convert::models::irodori::ARCH`. Intentionally **distinct**
/// from every other arch tag in this crate — Irodori pairs a DACVAE-
/// continuous DiT with a **Rectified-Flow** Euler sampler; silently
/// sharing an arch tag with any Phase-4 continuous-VAE sibling would
/// misroute the runtime dispatch (VibeVoice → `ddpm_sample`, VoxCPM →
/// `flow_sample` with EpsS schedule, Irodori → `flow_sample` with
/// Linear / Sway schedule and a **distinct** latent width of 32 vs
/// VoxCPM's 64 vs VibeVoice's 64).
pub const EXPECTED_ARCH: &str = "irodori-tts";

/// PCM sample rate Irodori-TTS emits — **48 kHz**. Primary source: the
/// base-model card at `huggingface.co/Aratako/Irodori-TTS-500M-v3`
/// (`Semantic-DACVAE-Japanese-32dim codec (32-dim), enabling high-quality
/// 48kHz waveform reconstruction`, fetched 2026-07-24). Not encoded in
/// the training YAML — it rides the paired codec GGUF.
pub const IRODORI_SAMPLE_RATE: u32 = 48_000;

/// Text tokenizer repo the released v3 checkpoint pins
/// (`ModelConfig.text_tokenizer_repo`). LLM-JP-3 150M is Apache-2.0
/// (`huggingface.co/llm-jp/llm-jp-3-150m`) so the tokenizer transitively
/// inherits Permissive-class redistribution; the id is recorded verbatim
/// so a real-checkpoint bind can cross-check the tokenizer manifest.
pub const IRODORI_TEXT_TOKENIZER_REPO: &str = "llm-jp/llm-jp-3-150m";

// ---------------------------------------------------------------------------
// DiT (rectified-flow diffusion transformer) hparams
// ---------------------------------------------------------------------------

/// Main RF-DiT body hparams — the transformer that predicts velocity in
/// the DACVAE 32-d latent space.
///
/// Every field is transcribed **verbatim** from
/// `configs/train_500m_v3_phase1_body.yaml`, cross-referenced with
/// `irodori_tts/config.py::ModelConfig` defaults (fetched 2026-07-24 —
/// CLAUDE.md「ハルシネーション厳禁」).
#[derive(Debug, Clone, PartialEq)]
pub struct IrodoriDitConfig {
    /// `latent_dim` — DACVAE latent width, **32**.
    pub latent_dim: u32,
    /// `latent_patch_size` — sequence-axis patch factor before DiT input.
    /// The v3 release keeps this at **1** (one DACVAE frame per DiT step).
    pub latent_patch_size: u32,
    /// `model_dim` — DiT residual stream width, **1280**.
    pub model_dim: u32,
    /// `num_layers` — DiT block count, **12**.
    pub num_layers: u32,
    /// `num_heads` — attention heads (MHA — `n_head_kv = n_head`), **20**.
    /// Head width follows `model_dim / num_heads = 64`.
    pub num_heads: u32,
    /// `mlp_ratio` — SwiGLU FFN inner ratio, **2.875**. Inner dim is
    /// `int(model_dim * mlp_ratio) = int(1280 * 2.875) = 3680`.
    pub mlp_ratio: f32,
    /// `timestep_embed_dim` — sinusoidal timestep embedding width,
    /// **512**.
    pub timestep_embed_dim: u32,
    /// `adaln_rank` — Low-Rank AdaLN bottleneck rank, **192**
    /// (`LowRankAdaLN`; see `irodori_tts/model.py`).
    pub adaln_rank: u32,
    /// `norm_eps` — RMSNorm ε, **1e-5**.
    pub norm_eps: f32,
    /// `dropout` — training dropout probability (**0.0** at v3 phase-1
    /// release); recorded so a real-weight bind can cross-check.
    pub dropout: f32,
}

impl IrodoriDitConfig {
    /// Canonical Irodori-TTS-500M-v3 DiT config (primary source:
    /// `train_500m_v3_phase1_body.yaml.model.*`).
    #[must_use]
    pub fn irodori_500m_v3() -> Self {
        Self {
            latent_dim: 32,
            latent_patch_size: 1,
            model_dim: 1280,
            num_layers: 12,
            num_heads: 20,
            mlp_ratio: 2.875,
            timestep_embed_dim: 512,
            adaln_rank: 192,
            norm_eps: 1e-5,
            dropout: 0.0,
        }
    }

    /// Miniature well-formed DiT config for shape / stability tests.
    /// Every ratio (even head_dim for RoPE, non-zero FFN dim) mirrors the
    /// real model.
    #[must_use]
    pub fn tiny_for_tests() -> Self {
        Self {
            latent_dim: 4,
            latent_patch_size: 1,
            model_dim: 16,
            num_layers: 2,
            num_heads: 4,
            mlp_ratio: 2.0,
            timestep_embed_dim: 8,
            adaln_rank: 4,
            norm_eps: 1e-5,
            dropout: 0.0,
        }
    }

    /// Patched latent width: `latent_dim * latent_patch_size`.
    #[must_use]
    pub fn patched_latent_dim(&self) -> u32 {
        self.latent_dim.saturating_mul(self.latent_patch_size)
    }

    /// Per-head width `model_dim / num_heads`; **None** if `num_heads` is
    /// zero or does not divide `model_dim`.
    #[must_use]
    pub fn head_dim(&self) -> Option<u32> {
        if self.num_heads == 0 || self.model_dim % self.num_heads != 0 {
            return None;
        }
        Some(self.model_dim / self.num_heads)
    }

    /// SwiGLU FFN inner dim `int(model_dim * mlp_ratio)`; upstream
    /// `TextToLatentRFDiT` builds its SwiGLU with this width.
    #[must_use]
    pub fn ffn_inner_dim(&self) -> u32 {
        (self.model_dim as f64 * self.mlp_ratio as f64) as u32
    }
}

// ---------------------------------------------------------------------------
// Text encoder hparams
// ---------------------------------------------------------------------------

/// Text (prompt) encoder hparams — token embed + self-attention
/// transformer whose output feeds the DiT's joint attention.
///
/// Every field is transcribed **verbatim** from
/// `train_500m_v3_phase1_body.yaml.model.*` + `ModelConfig` defaults
/// (`text_add_bos: true`).
#[derive(Debug, Clone, PartialEq)]
pub struct IrodoriTextEncoderConfig {
    /// `text_vocab_size` — **99574** (LLM-JP-3 150M shared BPE, per
    /// `ModelConfig.text_tokenizer_repo = "llm-jp/llm-jp-3-150m"`).
    pub vocab_size: u32,
    /// `text_dim` — residual width, **512**.
    pub dim: u32,
    /// `text_layers` — block count, **10**.
    pub n_layer: u32,
    /// `text_heads` — attention heads (MHA), **8** (`head_dim = 64`).
    pub n_head: u32,
    /// `text_mlp_ratio` — SwiGLU FFN inner ratio, **2.6**.
    pub mlp_ratio: f32,
    /// `text_add_bos` — whether the encoder prepends a BOS token to the
    /// tokenizer output (**true**).
    pub add_bos: bool,
}

impl IrodoriTextEncoderConfig {
    /// Canonical Irodori-TTS-500M-v3 text encoder (primary source:
    /// `train_500m_v3_phase1_body.yaml.model.*`).
    #[must_use]
    pub fn irodori_500m_v3() -> Self {
        Self {
            vocab_size: 99_574,
            dim: 512,
            n_layer: 10,
            n_head: 8,
            mlp_ratio: 2.6,
            add_bos: true,
        }
    }

    /// Miniature well-formed text-encoder config for shape tests.
    #[must_use]
    pub fn tiny_for_tests() -> Self {
        Self {
            vocab_size: 32,
            dim: 16,
            n_layer: 2,
            n_head: 4,
            mlp_ratio: 2.0,
            add_bos: true,
        }
    }

    /// Per-head width; **None** if `n_head` does not divide `dim`.
    #[must_use]
    pub fn head_dim(&self) -> Option<u32> {
        if self.n_head == 0 || self.dim % self.n_head != 0 {
            return None;
        }
        Some(self.dim / self.n_head)
    }

    /// SwiGLU FFN inner dim `int(dim * mlp_ratio)`.
    #[must_use]
    pub fn ffn_inner_dim(&self) -> u32 {
        (self.dim as f64 * self.mlp_ratio as f64) as u32
    }
}

// ---------------------------------------------------------------------------
// Speaker (reference) encoder hparams
// ---------------------------------------------------------------------------

/// Reference latent (speaker) encoder hparams — self-attention
/// transformer over patched reference DACVAE latents. Feeds the DiT's
/// joint attention alongside the text-encoder output.
#[derive(Debug, Clone, PartialEq)]
pub struct IrodoriSpeakerEncoderConfig {
    /// `speaker_dim` — residual width, **768**.
    pub dim: u32,
    /// `speaker_layers` — block count, **8**.
    pub n_layer: u32,
    /// `speaker_heads` — attention heads (MHA), **12** (`head_dim = 64`).
    pub n_head: u32,
    /// `speaker_mlp_ratio` — SwiGLU FFN inner ratio, **2.6**.
    pub mlp_ratio: f32,
    /// `speaker_patch_size` — sequence-axis patch factor applied to the
    /// reference latents before this encoder consumes them. **1** at the
    /// v3 release (one DACVAE frame per speaker-encoder token).
    pub patch_size: u32,
}

impl IrodoriSpeakerEncoderConfig {
    /// Canonical Irodori-TTS-500M-v3 speaker encoder.
    #[must_use]
    pub fn irodori_500m_v3() -> Self {
        Self {
            dim: 768,
            n_layer: 8,
            n_head: 12,
            mlp_ratio: 2.6,
            patch_size: 1,
        }
    }

    /// Miniature well-formed speaker-encoder config for shape tests.
    #[must_use]
    pub fn tiny_for_tests() -> Self {
        Self {
            dim: 24,
            n_layer: 2,
            n_head: 4,
            mlp_ratio: 2.0,
            patch_size: 1,
        }
    }

    /// Per-head width; **None** if `n_head` does not divide `dim`.
    #[must_use]
    pub fn head_dim(&self) -> Option<u32> {
        if self.n_head == 0 || self.dim % self.n_head != 0 {
            return None;
        }
        Some(self.dim / self.n_head)
    }

    /// SwiGLU FFN inner dim.
    #[must_use]
    pub fn ffn_inner_dim(&self) -> u32 {
        (self.dim as f64 * self.mlp_ratio as f64) as u32
    }
}

// ---------------------------------------------------------------------------
// Duration predictor hparams (v3 phase-2)
// ---------------------------------------------------------------------------

/// Duration predictor hparams — the v3 base + v3 VoiceDesign releases
/// bundle an integrated predictor for automatic output length estimation
/// (`use_duration_predictor = true` in
/// `train_500m_v3_phase2_duration.yaml`; v2 releases set it `false`).
///
/// Every field is transcribed **verbatim** from
/// `train_500m_v3_phase2_duration.yaml.model.*`.
#[derive(Debug, Clone, PartialEq)]
pub struct IrodoriDurationPredictorConfig {
    /// `use_duration_predictor` — whether the checkpoint carries an
    /// integrated duration predictor. **true** for the v3 base + v3
    /// VoiceDesign releases.
    pub enabled: bool,
    /// `duration_aux_dim` — auxiliary feature width fed alongside the
    /// text state, **14**.
    pub aux_dim: u32,
    /// `duration_hidden_dim` — internal transformer width, **1024**.
    pub hidden_dim: u32,
    /// `duration_layers` — block count, **3**.
    pub n_layer: u32,
    /// `duration_attention_heads` — attention heads, **8**.
    pub n_head: u32,
    /// `duration_dropout` — training dropout, **0.1**.
    pub dropout: f32,
    /// `duration_architecture` — architecture tag, verbatim string
    /// (`"token_sum_adarn_zero_no_aux"` for v3 base +
    /// v3 VoiceDesign). See `DURATION_ARCHITECTURES` in
    /// `irodori_tts/model.py`.
    pub architecture: String,
    /// `duration_token_init_frames` — bias initialiser (mean output
    /// duration in DACVAE frames), **9.0**.
    pub token_init_frames: f32,
    /// `duration_speaker_fusion` — how the speaker state is fused into
    /// the predictor (`"adarn_zero"` at v3 release).
    pub speaker_fusion: String,
}

impl IrodoriDurationPredictorConfig {
    /// Canonical Irodori-TTS-500M-v3 duration predictor (primary source:
    /// `train_500m_v3_phase2_duration.yaml.model.*`).
    #[must_use]
    pub fn irodori_500m_v3() -> Self {
        Self {
            enabled: true,
            aux_dim: 14,
            hidden_dim: 1024,
            n_layer: 3,
            n_head: 8,
            dropout: 0.1,
            architecture: "token_sum_adarn_zero_no_aux".to_owned(),
            token_init_frames: 9.0,
            speaker_fusion: "adarn_zero".to_owned(),
        }
    }

    /// A `disabled` predictor stub — carries the same struct shape but
    /// the runtime skips the head. Matches the v2 base release
    /// (`use_duration_predictor = false`) so a caller who bumps the
    /// runtime past the v3 axes still sees a well-formed struct.
    #[must_use]
    pub fn disabled() -> Self {
        let mut cfg = Self::irodori_500m_v3();
        cfg.enabled = false;
        cfg
    }

    /// Miniature well-formed duration-predictor config for shape tests.
    #[must_use]
    pub fn tiny_for_tests() -> Self {
        Self {
            enabled: true,
            aux_dim: 4,
            hidden_dim: 16,
            n_layer: 2,
            n_head: 2,
            dropout: 0.0,
            architecture: "token_sum_adarn_zero_no_aux".to_owned(),
            token_init_frames: 1.0,
            speaker_fusion: "adarn_zero".to_owned(),
        }
    }
}

// ---------------------------------------------------------------------------
// Top-level config
// ---------------------------------------------------------------------------

/// Resolved Irodori-TTS-500M-v3 hparam snapshot — every field is
/// transcribed from the training YAML
/// (`train_500m_v3_phase1_body.yaml` and
/// `train_500m_v3_phase2_duration.yaml`) or from the paired
/// Semantic-DACVAE-Japanese-32dim codec (`sample_rate`).
#[derive(Debug, Clone, PartialEq)]
pub struct IrodoriConfig {
    /// Rectified-Flow DiT body hparams.
    pub dit: IrodoriDitConfig,
    /// Prompt-text encoder hparams.
    pub text: IrodoriTextEncoderConfig,
    /// Reference-latent (speaker) encoder hparams.
    pub speaker: IrodoriSpeakerEncoderConfig,
    /// Duration predictor hparams (v3 phase-2 default: enabled).
    pub duration: IrodoriDurationPredictorConfig,
    /// PCM sample rate the paired codec emits (**48 kHz**; inherited
    /// from `Semantic-DACVAE-Japanese-32dim`, not encoded in the training
    /// YAML).
    pub sample_rate: u32,
    /// Text tokenizer repo id the release pins
    /// (`llm-jp/llm-jp-3-150m`).
    pub text_tokenizer_repo: String,
}

impl IrodoriConfig {
    /// Primary-source Irodori-TTS-500M-v3 config (every value transcribed
    /// from `train_500m_v3_phase1_body.yaml` +
    /// `train_500m_v3_phase2_duration.yaml`).
    #[must_use]
    pub fn irodori_500m_v3() -> Self {
        Self {
            dit: IrodoriDitConfig::irodori_500m_v3(),
            text: IrodoriTextEncoderConfig::irodori_500m_v3(),
            speaker: IrodoriSpeakerEncoderConfig::irodori_500m_v3(),
            duration: IrodoriDurationPredictorConfig::irodori_500m_v3(),
            sample_rate: IRODORI_SAMPLE_RATE,
            text_tokenizer_repo: IRODORI_TEXT_TOKENIZER_REPO.to_owned(),
        }
    }

    /// Miniature well-formed config for shape / stability tests.
    #[must_use]
    pub fn tiny_for_tests() -> Self {
        Self {
            dit: IrodoriDitConfig::tiny_for_tests(),
            text: IrodoriTextEncoderConfig::tiny_for_tests(),
            speaker: IrodoriSpeakerEncoderConfig::tiny_for_tests(),
            duration: IrodoriDurationPredictorConfig::tiny_for_tests(),
            sample_rate: IRODORI_SAMPLE_RATE,
            text_tokenizer_repo: IRODORI_TEXT_TOKENIZER_REPO.to_owned(),
        }
    }

    /// A canonical [`FlowSamplerConfig`] that matches the release
    /// inference defaults (`SamplingConfig.num_steps = 40`,
    /// `Schedule::Linear`, independent split-batch CFG on three axes).
    /// The three per-axis scales (`cfg_scale_text = 3.0`,
    /// `cfg_scale_caption = 3.0`, `cfg_scale_speaker = 5.0`) collapse to
    /// a single [`CfgScaleProfile::Constant`] here because the v3 base
    /// checkpoint does not carry the caption branch — the caller can
    /// override the profile when driving the VoiceDesign 3-branch
    /// variant.
    #[must_use]
    pub fn default_sampler(&self) -> FlowSamplerConfig {
        FlowSamplerConfig {
            cfg_mode: CfgMode::SplitBatch,
            cfg_scale: CfgScaleProfile::Constant(3.0),
            nfe: 40,
            schedule: Schedule::Linear,
            solver: OdeSolver::Euler,
        }
    }

    /// Rejects `0`-placeholder / ill-formed configs before any forward
    /// runs (FR-EX-08 — a shape-only converter path fails loudly here,
    /// not deep inside a GEMM).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] naming the offending field.
    pub fn validate_for_forward(&self) -> Result<()> {
        // DiT axes.
        let d = &self.dit;
        if d.latent_dim == 0
            || d.model_dim == 0
            || d.num_layers == 0
            || d.num_heads == 0
            || d.timestep_embed_dim == 0
            || d.adaln_rank == 0
        {
            return Err(VokraError::InvalidArgument(format!(
                "irodori dit config: zero-size hparam (latent_dim={}, model_dim={}, \
                 num_layers={}, num_heads={}, timestep_embed_dim={}, adaln_rank={})",
                d.latent_dim,
                d.model_dim,
                d.num_layers,
                d.num_heads,
                d.timestep_embed_dim,
                d.adaln_rank,
            )));
        }
        if d.latent_patch_size == 0 {
            return Err(VokraError::InvalidArgument(
                "irodori dit config: latent_patch_size must be > 0".to_owned(),
            ));
        }
        let dit_head = d.head_dim().ok_or_else(|| {
            VokraError::InvalidArgument(format!(
                "irodori dit config: num_heads ({}) must divide model_dim ({})",
                d.num_heads, d.model_dim,
            ))
        })?;
        if dit_head % 2 != 0 {
            return Err(VokraError::InvalidArgument(format!(
                "irodori dit config: RoPE requires even head_dim (model_dim / num_heads = {dit_head})"
            )));
        }
        if d.adaln_rank > d.model_dim {
            return Err(VokraError::InvalidArgument(format!(
                "irodori dit config: adaln_rank ({}) must be <= model_dim ({}) — Low-Rank AdaLN \
                 bottlenecks the modulation into a `rank`-wide space",
                d.adaln_rank, d.model_dim,
            )));
        }
        if !(d.mlp_ratio.is_finite() && d.mlp_ratio > 0.0) {
            return Err(VokraError::InvalidArgument(format!(
                "irodori dit config: mlp_ratio must be a positive finite f32 (got {})",
                d.mlp_ratio,
            )));
        }
        if !(d.norm_eps.is_finite() && d.norm_eps > 0.0) {
            return Err(VokraError::InvalidArgument(format!(
                "irodori dit config: norm_eps must be a positive finite f32 (got {})",
                d.norm_eps,
            )));
        }
        if !(d.dropout.is_finite() && d.dropout >= 0.0) {
            return Err(VokraError::InvalidArgument(format!(
                "irodori dit config: dropout must be a non-negative finite f32 (got {})",
                d.dropout,
            )));
        }

        // Text encoder axes.
        let t = &self.text;
        if t.vocab_size == 0 || t.dim == 0 || t.n_layer == 0 || t.n_head == 0 {
            return Err(VokraError::InvalidArgument(format!(
                "irodori text encoder: zero-size hparam (vocab_size={}, dim={}, n_layer={}, \
                 n_head={})",
                t.vocab_size, t.dim, t.n_layer, t.n_head,
            )));
        }
        let text_head = t.head_dim().ok_or_else(|| {
            VokraError::InvalidArgument(format!(
                "irodori text encoder: n_head ({}) must divide dim ({})",
                t.n_head, t.dim,
            ))
        })?;
        if text_head % 2 != 0 {
            return Err(VokraError::InvalidArgument(format!(
                "irodori text encoder: RoPE requires even head_dim (dim / n_head = {text_head})"
            )));
        }
        if !(t.mlp_ratio.is_finite() && t.mlp_ratio > 0.0) {
            return Err(VokraError::InvalidArgument(format!(
                "irodori text encoder: mlp_ratio must be a positive finite f32 (got {})",
                t.mlp_ratio,
            )));
        }

        // Speaker encoder axes.
        let s = &self.speaker;
        if s.dim == 0 || s.n_layer == 0 || s.n_head == 0 || s.patch_size == 0 {
            return Err(VokraError::InvalidArgument(format!(
                "irodori speaker encoder: zero-size hparam (dim={}, n_layer={}, n_head={}, \
                 patch_size={})",
                s.dim, s.n_layer, s.n_head, s.patch_size,
            )));
        }
        let speaker_head = s.head_dim().ok_or_else(|| {
            VokraError::InvalidArgument(format!(
                "irodori speaker encoder: n_head ({}) must divide dim ({})",
                s.n_head, s.dim,
            ))
        })?;
        if speaker_head % 2 != 0 {
            return Err(VokraError::InvalidArgument(format!(
                "irodori speaker encoder: RoPE requires even head_dim \
                 (dim / n_head = {speaker_head})"
            )));
        }
        if !(s.mlp_ratio.is_finite() && s.mlp_ratio > 0.0) {
            return Err(VokraError::InvalidArgument(format!(
                "irodori speaker encoder: mlp_ratio must be a positive finite f32 (got {})",
                s.mlp_ratio,
            )));
        }

        // Duration predictor axes — validated only when enabled.
        let dur = &self.duration;
        if dur.enabled {
            if dur.hidden_dim == 0 || dur.n_layer == 0 || dur.n_head == 0 {
                return Err(VokraError::InvalidArgument(format!(
                    "irodori duration predictor: zero-size hparam (hidden_dim={}, n_layer={}, \
                     n_head={})",
                    dur.hidden_dim, dur.n_layer, dur.n_head,
                )));
            }
            if dur.hidden_dim % dur.n_head != 0 {
                return Err(VokraError::InvalidArgument(format!(
                    "irodori duration predictor: n_head ({}) must divide hidden_dim ({})",
                    dur.n_head, dur.hidden_dim,
                )));
            }
            if !(dur.dropout.is_finite() && dur.dropout >= 0.0) {
                return Err(VokraError::InvalidArgument(format!(
                    "irodori duration predictor: dropout must be a non-negative finite f32 \
                     (got {})",
                    dur.dropout,
                )));
            }
            if !(dur.token_init_frames.is_finite() && dur.token_init_frames > 0.0) {
                return Err(VokraError::InvalidArgument(format!(
                    "irodori duration predictor: token_init_frames must be a positive finite f32 \
                     (got {})",
                    dur.token_init_frames,
                )));
            }
            if dur.architecture.is_empty() {
                return Err(VokraError::InvalidArgument(
                    "irodori duration predictor: architecture tag must be non-empty".to_owned(),
                ));
            }
            if dur.speaker_fusion.is_empty() {
                return Err(VokraError::InvalidArgument(
                    "irodori duration predictor: speaker_fusion tag must be non-empty".to_owned(),
                ));
            }
        }

        // Codec handshake — sample rate must be positive.
        if self.sample_rate == 0 {
            return Err(VokraError::InvalidArgument(
                "irodori config: sample_rate must be > 0 (inherited from the paired \
                 Semantic-DACVAE-Japanese-32dim codec — 48 kHz at release)"
                    .to_owned(),
            ));
        }
        if self.text_tokenizer_repo.is_empty() {
            return Err(VokraError::InvalidArgument(
                "irodori config: text_tokenizer_repo must be non-empty".to_owned(),
            ));
        }

        // Sampler-config derivation must succeed (validates cfg fields).
        // The default sampler is used as a canary — a caller can still
        // override, but the *default* must be well-formed.
        // FlowSamplerConfig has no runtime validate method that runs
        // independently of `flow_sample()`; the shape guarantees above
        // suffice for the shape-only forward gate.

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Weight-store scaffold
// ---------------------------------------------------------------------------

/// Irodori-TTS-500M-v3 weight store scaffold.
///
/// Real binding is a follow-up wave (T29-equivalent — the safetensors
/// walk for the text encoder / speaker encoder / DiT body / duration
/// predictor defers to the T29 tensor-name manifest fetch). This
/// scaffold carries only aggregate byte bundles so downstream shape flow
/// / handshake tests are unblocked; the sole invariant this slice pins
/// is that `is_synthesized = true` prevents a spurious synthesize call
/// from returning zero audio (FR-EX-08 — the loud
/// [`IrodoriTts::synthesize`] guard).
#[derive(Debug, Clone)]
pub struct IrodoriWeights {
    /// Placeholder for the text-encoder tensor bytes (aggregate). Real
    /// binding walks the upstream naming.
    pub text_encoder: Vec<f32>,
    /// Placeholder for the speaker (reference-latent) encoder tensor
    /// bytes (aggregate).
    pub speaker_encoder: Vec<f32>,
    /// Placeholder for the DiT body tensor bytes (aggregate).
    pub dit: Vec<f32>,
    /// Placeholder for the duration-predictor tensor bytes (aggregate).
    /// Empty when [`IrodoriDurationPredictorConfig::enabled`] is false.
    pub duration_predictor: Vec<f32>,
    /// `true` when built by [`Self::synthesized`] — never a real upstream
    /// checkpoint.
    pub is_synthesized: bool,
}

impl IrodoriWeights {
    /// Builds a deterministic zero-initialized fixture (shape scaffold
    /// only — every slot is `Vec::new()`).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] if `config.validate_for_forward`
    /// fails.
    pub fn synthesized(config: &IrodoriConfig) -> Result<Self> {
        config.validate_for_forward()?;
        Ok(Self {
            text_encoder: Vec::new(),
            speaker_encoder: Vec::new(),
            dit: Vec::new(),
            duration_predictor: Vec::new(),
            is_synthesized: true,
        })
    }
}

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

/// Irodori-TTS-500M-v3 engine handle.
///
/// Carries the resolved config + weight store + an optional
/// [`DacCodecGguf`] codec binding + a derived [`FlowSamplerConfig`]
/// (the release inference defaults). [`Self::synthesize`] is the primary
/// text → PCM entry point; until real weights are bound, a codec GGUF
/// is injected via [`Self::with_codec`], and the text-encoder →
/// speaker-encoder → RF-DiT → codec-decode chain is wired end-to-end
/// (T29-equivalent follow-up wave), it returns
/// [`VokraError::NotImplemented`] naming the blocker (FR-EX-08 — never
/// a silent zero-fill or empty audio buffer).
#[derive(Debug, Clone)]
pub struct IrodoriTts {
    cfg: IrodoriConfig,
    weights: IrodoriWeights,
    codec: Option<DacCodecGguf>,
    sampler: FlowSamplerConfig,
}

impl IrodoriTts {
    /// Assembles an engine from `cfg` and `weights`. Cross-checks the
    /// config at construction time so a mismatched pair fails loudly
    /// here rather than deep inside a forward.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] from
    /// [`IrodoriConfig::validate_for_forward`].
    pub fn new(cfg: IrodoriConfig, weights: IrodoriWeights) -> Result<Self> {
        cfg.validate_for_forward()?;
        let sampler = cfg.default_sampler();
        Ok(Self {
            cfg,
            weights,
            codec: None,
            sampler,
        })
    }

    /// Injects a `Semantic-DACVAE-Japanese-32dim` codec binding (the
    /// same `DacCodecGguf`-shaped seam Dia / Zonos use with vanilla
    /// DAC). Returns the modified engine; without this bind
    /// [`Self::synthesize`] returns `NotImplemented`.
    #[must_use]
    pub fn with_codec(mut self, codec: DacCodecGguf) -> Self {
        self.codec = Some(codec);
        self
    }

    /// Overrides the default sampler. The release defaults are a
    /// 40-step Euler over the Linear schedule with independent
    /// split-batch CFG at scale 3.0; a caller can drop to Sway for
    /// F5-TTS-style noise-side densification or raise the step count
    /// for slower / higher-quality synthesis.
    #[must_use]
    pub fn with_sampler(mut self, sampler: FlowSamplerConfig) -> Self {
        self.sampler = sampler;
        self
    }

    /// The resolved configuration.
    #[must_use]
    pub fn config(&self) -> &IrodoriConfig {
        &self.cfg
    }

    /// The active RF sampler config.
    #[must_use]
    pub fn sampler_config(&self) -> &FlowSamplerConfig {
        &self.sampler
    }

    /// True iff the weight store was built by
    /// [`IrodoriWeights::synthesized`] (never a real upstream checkpoint).
    #[must_use]
    pub fn is_synthesized(&self) -> bool {
        self.weights.is_synthesized
    }

    /// True iff a `Semantic-DACVAE-Japanese-32dim` codec binding has
    /// been injected via [`Self::with_codec`].
    #[must_use]
    pub fn has_codec(&self) -> bool {
        self.codec.is_some()
    }

    /// Synthesizes PCM for `text` at 48 kHz mono.
    ///
    /// This is the primary text → PCM entry point. **Real weights AND a
    /// codec binding required**: synthesized-weight builds cannot
    /// produce meaningful audio, so this returns
    /// [`VokraError::NotImplemented`] naming the blocker (FR-EX-08 —
    /// never a silent zero-fill or empty audio buffer).
    ///
    /// # Errors
    ///
    /// - [`VokraError::InvalidArgument`] if `text` is empty.
    /// - [`VokraError::NotImplemented`] otherwise (real forward not yet
    ///   bound — FR-EX-08).
    pub fn synthesize(&self, text: &str) -> Result<Vec<f32>> {
        if text.is_empty() {
            return Err(VokraError::InvalidArgument(
                "irodori synthesize: text is empty".to_owned(),
            ));
        }
        if self.weights.is_synthesized {
            return Err(VokraError::NotImplemented(
                "irodori synthesize: this engine holds synthesized weights (deterministic \
                 scaffold fixture from IrodoriWeights::synthesized) — synthesized-weight audio \
                 would be a hallucinated waveform, not real speech. Bind real Irodori-TTS-500M-v3 \
                 weights (MIT, huggingface.co/Aratako/Irodori-TTS-500M-v3) before invoking \
                 synthesize. The shape flow (config validation, weight-store construction, \
                 text-empty check, sampler handshake) is exercised through IrodoriTts::new; \
                 real-checkpoint binding lands in a follow-up wave (T29-equivalent).",
            ));
        }
        if self.codec.is_none() {
            return Err(VokraError::NotImplemented(
                "irodori synthesize: no `Semantic-DACVAE-Japanese-32dim` codec binding — inject \
                 one via IrodoriTts::with_codec(DacCodecGguf) before invoking synthesize. \
                 Irodori-TTS decodes to PCM through the paired DACVAE codec \
                 (huggingface.co/Aratako/Semantic-DACVAE-Japanese-32dim, 32-d latent → 48 kHz \
                 PCM); without it the runtime cannot lower continuous latents to a waveform.",
            ));
        }
        Err(VokraError::NotImplemented(
            "irodori synthesize: real weights are bound and a Semantic-DACVAE codec is present, \
             but the LLM-JP-3 text-encoder → reference-latent speaker-encoder → RF-DiT joint- \
             attention body → rectified-flow Euler sampler (vokra_ops::flow_sample, \
             Schedule::Linear or Sway / 40 steps / split-batch CFG on text, caption and speaker \
             axes / cfg window t ∈ [0.5, 1.0]) → Semantic-DACVAE decode → 48 kHz PCM forward \
             path has not landed yet. Follow-up wave (T29-equivalent): (1) tokenize `text` with \
             the LLM-JP-3 150M tokenizer (add_bos=true), (2) run the text encoder + optional \
             reference-latent speaker encoder, (3) sample a 32-d latent sequence of length \
             derived from the duration predictor (v3 base + v3 VoiceDesign) or a caller-supplied \
             `--seconds`, (4) decode through the injected DACVAE codec to 48 kHz mono PCM.",
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
    fn expected_arch_is_irodori_tts() {
        assert_eq!(EXPECTED_ARCH, "irodori-tts");
    }

    #[test]
    fn arch_is_distinct_from_neighbouring_families() {
        // Irodori's terminal decoding hop is a continuous-latent RF-DiT
        // feeding a paired DACVAE codec — neither a vocoder-LM (HiFTChain),
        // nor a codec-LM (any RVQ / FSQ codec), nor VibeVoice's DDPM
        // sampler, nor VoxCPM's EpsS-schedule flow-matching sampler.
        // Silently sharing an arch tag would misroute the runtime dispatch.
        assert_ne!(EXPECTED_ARCH, crate::vibevoice::EXPECTED_ARCH);
        assert_ne!(EXPECTED_ARCH, crate::voxcpm2::EXPECTED_ARCH);
        assert_ne!(EXPECTED_ARCH, crate::cosyvoice3::EXPECTED_ARCH);
        assert_ne!(EXPECTED_ARCH, crate::qwen3_tts::EXPECTED_ARCH);
        assert_ne!(EXPECTED_ARCH, crate::chatterbox::EXPECTED_ARCH);
        assert_ne!(EXPECTED_ARCH, crate::chatterbox_nano::EXPECTED_ARCH);
        assert_ne!(EXPECTED_ARCH, crate::chatterbox_turbo::EXPECTED_ARCH);
        assert_ne!(EXPECTED_ARCH, crate::dia::EXPECTED_ARCH);
        assert_ne!(EXPECTED_ARCH, crate::zonos::EXPECTED_ARCH);
    }

    // ---- Sample-rate + tokenizer constants ------------------------------

    #[test]
    fn sample_rate_matches_primary_source() {
        assert_eq!(IRODORI_SAMPLE_RATE, 48_000);
    }

    #[test]
    fn text_tokenizer_repo_matches_primary_source() {
        assert_eq!(IRODORI_TEXT_TOKENIZER_REPO, "llm-jp/llm-jp-3-150m");
    }

    // ---- Primary-source pins --------------------------------------------

    #[test]
    fn dit_config_matches_primary_source() {
        let d = IrodoriDitConfig::irodori_500m_v3();
        // train_500m_v3_phase1_body.yaml.model.*
        assert_eq!(d.latent_dim, 32);
        assert_eq!(d.latent_patch_size, 1);
        assert_eq!(d.model_dim, 1280);
        assert_eq!(d.num_layers, 12);
        assert_eq!(d.num_heads, 20);
        assert!((d.mlp_ratio - 2.875).abs() < 1e-6);
        assert_eq!(d.timestep_embed_dim, 512);
        assert_eq!(d.adaln_rank, 192);
        assert!((d.norm_eps - 1e-5).abs() < 1e-9);
        assert!((d.dropout - 0.0).abs() < 1e-9);
    }

    #[test]
    fn dit_derived_dims_match_algebra() {
        let d = IrodoriDitConfig::irodori_500m_v3();
        // 1280 / 20 = 64 head_dim (even → RoPE OK).
        assert_eq!(d.head_dim(), Some(64));
        // 32 * 1 = 32 patched latent dim.
        assert_eq!(d.patched_latent_dim(), 32);
        // int(1280 * 2.875) = 3680 SwiGLU inner dim.
        assert_eq!(d.ffn_inner_dim(), 3680);
    }

    #[test]
    fn text_encoder_config_matches_primary_source() {
        let t = IrodoriTextEncoderConfig::irodori_500m_v3();
        // train_500m_v3_phase1_body.yaml.model.*
        assert_eq!(t.vocab_size, 99_574);
        assert_eq!(t.dim, 512);
        assert_eq!(t.n_layer, 10);
        assert_eq!(t.n_head, 8);
        assert!((t.mlp_ratio - 2.6).abs() < 1e-6);
        // ModelConfig default: text_add_bos=True.
        assert!(t.add_bos);
        // 512 / 8 = 64 head_dim (even → RoPE OK).
        assert_eq!(t.head_dim(), Some(64));
        // int(512 * 2.6) = 1331 SwiGLU inner dim.
        assert_eq!(t.ffn_inner_dim(), 1331);
    }

    #[test]
    fn speaker_encoder_config_matches_primary_source() {
        let s = IrodoriSpeakerEncoderConfig::irodori_500m_v3();
        // train_500m_v3_phase1_body.yaml.model.*
        assert_eq!(s.dim, 768);
        assert_eq!(s.n_layer, 8);
        assert_eq!(s.n_head, 12);
        assert!((s.mlp_ratio - 2.6).abs() < 1e-6);
        assert_eq!(s.patch_size, 1);
        // 768 / 12 = 64 head_dim (even → RoPE OK).
        assert_eq!(s.head_dim(), Some(64));
        // int(768 * 2.6) = 1996 SwiGLU inner dim.
        assert_eq!(s.ffn_inner_dim(), 1996);
    }

    #[test]
    fn duration_predictor_config_matches_primary_source() {
        let dur = IrodoriDurationPredictorConfig::irodori_500m_v3();
        // train_500m_v3_phase2_duration.yaml.model.*
        assert!(dur.enabled);
        assert_eq!(dur.aux_dim, 14);
        assert_eq!(dur.hidden_dim, 1024);
        assert_eq!(dur.n_layer, 3);
        assert_eq!(dur.n_head, 8);
        assert!((dur.dropout - 0.1).abs() < 1e-6);
        assert_eq!(dur.architecture, "token_sum_adarn_zero_no_aux");
        assert!((dur.token_init_frames - 9.0).abs() < 1e-6);
        assert_eq!(dur.speaker_fusion, "adarn_zero");
    }

    #[test]
    fn top_level_config_matches_primary_source() {
        let c = IrodoriConfig::irodori_500m_v3();
        assert_eq!(c.sample_rate, 48_000);
        assert_eq!(c.text_tokenizer_repo, "llm-jp/llm-jp-3-150m");
    }

    // ---- Validation holds by construction -------------------------------

    #[test]
    fn canonical_config_validates() {
        IrodoriConfig::irodori_500m_v3()
            .validate_for_forward()
            .expect("canonical must validate");
    }

    #[test]
    fn tiny_config_validates() {
        IrodoriConfig::tiny_for_tests()
            .validate_for_forward()
            .expect("tiny must validate");
    }

    #[test]
    fn validate_rejects_zero_axes() {
        let mut c = IrodoriConfig::irodori_500m_v3();
        c.dit.model_dim = 0;
        let err = c.validate_for_forward().unwrap_err();
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn validate_rejects_gqa_mismatch_in_dit() {
        let mut c = IrodoriConfig::irodori_500m_v3();
        // 1280 / 21 does not divide evenly.
        c.dit.num_heads = 21;
        let err = c.validate_for_forward().unwrap_err();
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn validate_rejects_odd_head_dim() {
        let mut c = IrodoriConfig::irodori_500m_v3();
        // Force an odd head_dim by shrinking model_dim to an odd multiple.
        // 20 * 3 = 60 → 60/20 = 3 (odd).
        c.dit.model_dim = 60;
        let err = c.validate_for_forward().unwrap_err();
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn validate_rejects_adaln_rank_over_model_dim() {
        let mut c = IrodoriConfig::irodori_500m_v3();
        c.dit.adaln_rank = c.dit.model_dim + 1;
        let err = c.validate_for_forward().unwrap_err();
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn validate_rejects_zero_sample_rate() {
        let mut c = IrodoriConfig::irodori_500m_v3();
        c.sample_rate = 0;
        let err = c.validate_for_forward().unwrap_err();
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn validate_ignores_duration_axes_when_disabled() {
        let mut c = IrodoriConfig::irodori_500m_v3();
        c.duration = IrodoriDurationPredictorConfig::disabled();
        // Even setting zero axes should not trip validation while disabled.
        c.duration.hidden_dim = 0;
        c.duration.n_head = 0;
        c.validate_for_forward()
            .expect("disabled duration predictor is validation-invisible");
    }

    #[test]
    fn validate_rejects_zero_axes_when_duration_enabled() {
        let mut c = IrodoriConfig::irodori_500m_v3();
        c.duration.hidden_dim = 0;
        let err = c.validate_for_forward().unwrap_err();
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    // ---- Default sampler carries release inference axes -----------------

    #[test]
    fn default_sampler_carries_release_inference_axes() {
        let c = IrodoriConfig::irodori_500m_v3();
        let s = c.default_sampler();
        // SamplingConfig.num_steps = 40 (irodori_tts/config.py).
        assert_eq!(s.nfe, 40);
        // Rectified-flow primary solver.
        assert_eq!(s.solver, OdeSolver::Euler);
        // Default schedule (Linear); Sway is a caller-selectable knob per
        // irodori_tts/rf.py::sample_euler_rf_cfg.
        assert_eq!(s.schedule, Schedule::Linear);
        // Split-batch CFG on three independent axes
        // (irodori_tts/rf.py::_bundle + `cfg_batch_mult`).
        assert_eq!(s.cfg_mode, CfgMode::SplitBatch);
    }

    // ---- Engine posture --------------------------------------------------

    #[test]
    fn engine_synthesize_rejects_empty_text() {
        let cfg = IrodoriConfig::tiny_for_tests();
        let weights = IrodoriWeights::synthesized(&cfg).expect("synthesized");
        let engine = IrodoriTts::new(cfg, weights).expect("new");
        let err = engine.synthesize("").unwrap_err();
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn engine_synthesize_rejects_synthesized_weights_loudly() {
        // FR-EX-08 — synthesized-weight builds cannot produce meaningful
        // audio, so `synthesize` must fail loudly (never zero-fill).
        let cfg = IrodoriConfig::tiny_for_tests();
        let weights = IrodoriWeights::synthesized(&cfg).expect("synthesized");
        let engine = IrodoriTts::new(cfg, weights).expect("new");
        let err = engine.synthesize("こんにちは").unwrap_err();
        match err {
            VokraError::NotImplemented(msg) => {
                // Message must name the blocker so operators can act on it.
                assert!(msg.contains("synthesized weights"), "message: {msg}");
                assert!(msg.contains("Irodori"), "message: {msg}");
            }
            other => panic!("expected NotImplemented, got {other:?}"),
        }
    }

    #[test]
    fn engine_has_codec_starts_false() {
        let cfg = IrodoriConfig::tiny_for_tests();
        let weights = IrodoriWeights::synthesized(&cfg).expect("synthesized");
        let engine = IrodoriTts::new(cfg, weights).expect("new");
        assert!(!engine.has_codec());
    }
}
