//! **VoxCPM-0.5B** — OpenBMB's end-to-end diffusion-autoregressive TTS
//! (SoTA plan Phase 4, 2026-07-24). Apache 2.0 code + weight.
//!
//! # What VoxCPM is (primary source)
//!
//! `openbmb/VoxCPM-0.5B` is a **tokenizer-free** speech synthesizer whose
//! decoder emits **continuous** feature vectors (not discrete codec
//! indices). It combines:
//!
//! - A **MiniCPM-4** text-semantic LM backbone (`lm_config` — 24-layer /
//!   `hidden_size = 1024` / GQA `num_attention_heads = 16` /
//!   `num_key_value_heads = 2` / SwiGLU `intermediate_size = 4096` /
//!   RoPE `theta = 10000` with **longrope scaling** / `rms_norm_eps = 1e-5`,
//!   `vocab_size = 73448`, `max_position_embeddings = 32_768`,
//!   `scale_emb = 12`, `dim_model_base = 256`, `scale_depth = 1.4`).
//! - A **residual acoustic LM** (`residual_lm_num_layers = 6`, same backbone
//!   family with `vocab_size = 0` and RoPE optionally disabled).
//! - A **local encoder** (`encoder_config` — 4-layer / 1024d / GQA 16 heads,
//!   consumes the continuous VAE feature stream and lifts it to LM width).
//! - A **local DiT** (`dit_config` — 4-layer / 1024d / GQA 16 heads, the
//!   *diffusion decoder* that predicts velocity in the VAE latent space; a
//!   **conditional flow-matching sampler** (`cfm_config`) with
//!   `solver = "euler"`, `sigma_min = 1e-6`, `inference_cfg_rate = 2.0`,
//!   `t_scheduler = "log-norm"` — training-side; inference walks a linear
//!   `t_span` per the upstream `UnifiedCFM.forward`).
//! - A **scalar quantization bottleneck** (`ScalarQuantizationLayer`,
//!   `scalar_quantization_latent_dim = 256`,
//!   `scalar_quantization_scale = 9`) — an *inline* FSQ constraint on the
//!   LM hidden stream (distinct from the FSQ *codec* family that pairs a
//!   discrete index with a decoder — this projection stays continuous).
//! - An **AudioVAE V2** (continuous VAE, `patch_size = 2`, `feat_dim = 64`).
//!   The VAE encoder downsamples 16 kHz PCM by `product(encoder_rates) =
//!   640` → 25 Hz feature frames; the VAE decoder upsamples continuous
//!   latents by `product(decoder_rates) = 1920` → 48 kHz PCM (upstream
//!   `AudioVAEConfig.out_sample_rate = 48_000`). Shared primitive lives
//!   at [`vokra_ops::vae_continuous`] (SoTA plan Phase 4 new op).
//!
//! Every field above is transcribed **verbatim** from
//! `huggingface.co/openbmb/VoxCPM-0.5B/raw/main/config.json` and the
//! upstream `audio_vae_v2.py` `AudioVAEConfig` defaults (fetched 2026-07-24
//! — CLAUDE.md「ハルシネーション厳禁」).
//!
//! # Distinct topology axis: continuous VAE + diffusion decoder
//!
//! VoxCPM is the first Vokra target whose terminal decoding hop is a
//! **flow-matching feature generator** (LocDiT + UnifiedCFM) predicting
//! velocity in a **continuous** VAE latent space. Every earlier model in
//! this crate ends in one of:
//!
//! - a **vocoder-LM** chain that decodes an intermediate representation
//!   through HiFT-GAN / HiFTNet / BigVGAN (CosyVoice2/3, Chatterbox
//!   family), or
//! - a **codec-LM** chain that decodes discrete RVQ / FSQ indices
//!   through Mimi / DAC / SNAC / WavTokenizer / X-Codec 2 / Qwen3-TTS
//!   Codec (Qwen3-TTS, SNAC-family, Kyutai STT, Moshi, CSM, Voxtral).
//!
//! Silently sharing either sibling's `EXPECTED_ARCH` would misroute the
//! runtime dispatch — the terminal step is neither `HiFTChain` nor any
//! discrete codec but the **continuous VAE decoder** consuming the flow-
//! matching sampler output. See [`EXPECTED_ARCH`] for the distinct tag.
//!
//! # Reuses two existing ops — [`vokra_ops::vae_continuous`] and
//! [`vokra_ops::flow_sampler`]
//!
//! - The **AudioVAE V2** encoder / decoder ride the SoTA plan Phase 4
//!   [`vokra_ops::vae_continuous`] primitive introduced with this model
//!   (shared with the planned VibeVoice consumer).
//! - The **local DiT + UnifiedCFM** flow-matching sampler rides the
//!   existing [`vokra_ops::flow_sampler`] (Euler solver / linear schedule
//!   / CFG mode = split-batch or dual-forward — VoxCPM uses the split-
//!   batch form with `inference_cfg_rate = 2.0`).
//!
//! No new **backend kernel** is added by this model — the ops it needs
//! (`vae_continuous` and `flow_sampler`) already live in `vokra-ops`, and
//! the LM backbone reuses the same GQA / RMSNorm / SwiGLU / RoPE
//! primitives every earlier Qwen / MiniCPM / Llama sibling uses (the SIMD
//! kernels of `vokra-backend-cpu`).
//!
//! # What lands in this Phase 4 slice
//!
//! - [`VoxCpm2LmConfig`] + [`VoxCpm2EncoderConfig`] +
//!   [`VoxCpm2DitConfig`] + [`VoxCpm2CfmConfig`] + [`VoxCpm2Config`] —
//!   every architectural hparam transcribed verbatim from the primary
//!   source. `validate_for_forward` fails loudly (FR-EX-08) on zeroed
//!   axes / broken GQA algebra / broken CFG mode.
//! - [`VoxCpm2Weights`] — deterministic
//!   [`VoxCpm2Weights::synthesized`] scaffold fixture (zero-initialized;
//!   only the shape flow is exercised — the SplitMix64 Xavier fixture
//!   the sibling models use is skipped because the LM backbone weight
//!   store is a follow-up wave).
//! - [`VoxCpm2Tts`] — engine handle carrying config + weights. The
//!   primary [`VoxCpm2Tts::synthesize`] entry point returns
//!   [`VokraError::NotImplemented`] naming the blocker until real
//!   weights are bound and the LM → local DiT → CFM sampler →
//!   AudioVAE-decode → 48 kHz PCM chain is wired end-to-end (T29-
//!   equivalent follow-up wave — never a silent zero-fill, FR-EX-08).
//!
//! # No ONNX (permanent)
//!
//! VoxCPM-0.5B is distributed as safetensors + a Python pipeline; the
//! runtime **never** loads an ONNX graph (FR-LD-05, permanent constraint);
//! the pipeline is re-implemented natively from the safetensors checkpoint
//! (whisper.cpp 型 self re-implementation, CLAUDE.md 設計判断 4).

use vokra_core::{Result, VokraError};

// Public seam re-export — shared with the VAE primitive
pub use vokra_ops::vae_continuous::ContinuousVaeConfig;

/// `vokra.model.arch` a VoxCPM-0.5B GGUF must carry. Written by
/// `vokra-convert::models::voxcpm2::ARCH`. Intentionally **distinct**
/// from every existing arch tag in this crate — the terminal decoding
/// hop is a continuous VAE decoder (through [`vokra_ops::vae_continuous`]),
/// not HiFTNet / HiFT-GAN / Mimi / DAC / SNAC / Qwen3-TTS Codec. The
/// compliance registry (`vokra_core::compliance`) knows every
/// `voxcpm*` spelling as [`vokra_core::LicenseClass::Permissive`]
/// (apache-2.0).
pub const EXPECTED_ARCH: &str = "voxcpm2";

/// Encoder input PCM sample rate (Hz). VoxCPM-0.5B: `16_000`.
/// Downstream `AudioVAE V2.out_sample_rate = 48_000` for synthesis
/// output — see [`ContinuousVaeConfig::out_sample_rate_hz`] on the
/// shared VAE seam.
pub const VOXCPM_ENCODER_SAMPLE_RATE: u32 = 16_000;

// ---------------------------------------------------------------------------
// LM backbone config — MiniCPM-4 flavour
// ---------------------------------------------------------------------------

/// MiniCPM-4 backbone hparams the VoxCPM `lm_config` block transcribes.
///
/// Every field is transcribed **verbatim** from the primary source
/// (`config.json.lm_config.*` at
/// `huggingface.co/openbmb/VoxCPM-0.5B/raw/main/config.json`, fetched
/// 2026-07-24 — CLAUDE.md「ハルシネーション厳禁」). The MiniCPM-4 block
/// is a Llama-family decoder-only transformer with (a) very wide GQA
/// (16 Q ÷ 2 KV, ratio 8), (b) **longrope scaling** (`rope_scaling.type =
/// "longrope"`, 32-entry `long_factor` / `short_factor` tables — Note
/// **these tables live on the safetensors side of the checkpoint at load
/// time**; the runtime config carries only the axes needed to validate
/// their presence), (c) MiniCPM-specific **µ-parametrization**-adjacent
/// scale knobs (`scale_emb = 12`, `dim_model_base = 256`,
/// `scale_depth = 1.4`), and (d) the token-id anchors
/// (`bos_token_id = 1`, `eos_token_id = 2`).
#[derive(Debug, Clone, PartialEq)]
pub struct VoxCpm2LmConfig {
    /// Backbone hidden dimension (`hidden_size`). `1024`.
    pub hidden_dim: u32,
    /// Backbone transformer block count (`num_hidden_layers`). `24`.
    pub n_layer: u32,
    /// Backbone attention head count (`num_attention_heads`). `16`.
    pub n_head: u32,
    /// Backbone key/value head count for GQA (`num_key_value_heads`).
    /// `2` — the group ratio is `n_head / n_head_kv = 8` (each K/V head
    /// fans out to 8 Q heads — very wide GQA compared to Qwen2/3's 2/8).
    pub n_head_kv: u32,
    /// SwiGLU FFN inner dimension (`intermediate_size`). `4096`.
    pub ffn_dim: u32,
    /// Vocabulary size (`vocab_size`). `73_448` — the MiniCPM tokenizer's
    /// shared BPE.
    pub vocab_size: u32,
    /// Max positions the LM can attend over (`max_position_embeddings`).
    /// `32_768`.
    pub max_position_embeddings: u32,
    /// RoPE base θ (`rope_theta`). `10_000` — the classic Llama default
    /// (VoxCPM does NOT widen this axis; the long-context extension
    /// rides through longrope scaling instead).
    pub rope_base: f32,
    /// RMSNorm epsilon (`rms_norm_eps`). `1e-5`.
    pub rms_norm_eps: f32,
    /// Whether RoPE scaling is `"longrope"` (the only kind VoxCPM ships).
    /// `true` for the 0.5B release; a future variant that drops the
    /// scaling would set this to `false` and use plain RoPE.
    pub rope_scaling_longrope: bool,
    /// `original_max_position_embeddings` from the longrope block —
    /// carried so downstream binding can cross-check the scaling table
    /// lengths. `32_768` for the 0.5B release.
    pub rope_original_max_position_embeddings: u32,
    /// MiniCPM `scale_emb` scalar (`scale_emb`). `12`.
    pub scale_emb: u32,
    /// MiniCPM `dim_model_base` scalar (`dim_model_base`). `256`.
    pub dim_model_base: u32,
    /// MiniCPM `scale_depth` scalar (`scale_depth`). `1.4`.
    pub scale_depth: f32,
    /// Whether the µP path is enabled (`use_mup`). `false` for the 0.5B
    /// release.
    pub use_mup: bool,
}

impl VoxCpm2LmConfig {
    /// Canonical VoxCPM-0.5B `lm_config` (primary source:
    /// `config.json.lm_config.*`, fetched 2026-07-24).
    #[must_use]
    pub fn voxcpm_0_5b() -> Self {
        Self {
            hidden_dim: 1024,
            n_layer: 24,
            n_head: 16,
            n_head_kv: 2,
            ffn_dim: 4096,
            vocab_size: 73_448,
            max_position_embeddings: 32_768,
            rope_base: 10_000.0,
            rms_norm_eps: 1e-5,
            rope_scaling_longrope: true,
            rope_original_max_position_embeddings: 32_768,
            scale_emb: 12,
            dim_model_base: 256,
            scale_depth: 1.4,
            use_mup: false,
        }
    }

    /// Miniature well-formed LM config for shape / stability tests.
    /// Every ratio (GQA well-formedness, non-zero vocab, positive FFN
    /// dim) mirrors the real model.
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
            rope_base: 10_000.0,
            rms_norm_eps: 1e-5,
            rope_scaling_longrope: true,
            rope_original_max_position_embeddings: 128,
            scale_emb: 12,
            dim_model_base: 8,
            scale_depth: 1.4,
            use_mup: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Encoder / DiT / CFM configs
// ---------------------------------------------------------------------------

/// Local encoder hparams the VoxCPM `encoder_config` block transcribes
/// (`config.json.encoder_config.*`, fetched 2026-07-24).
#[derive(Debug, Clone, PartialEq)]
pub struct VoxCpm2EncoderConfig {
    /// Local encoder hidden dimension (`hidden_dim`). `1024`.
    pub hidden_dim: u32,
    /// SwiGLU FFN inner dimension (`ffn_dim`). `4096`.
    pub ffn_dim: u32,
    /// Attention head count (`num_heads`). `16`.
    pub n_head: u32,
    /// Transformer block count (`num_layers`). `4`.
    pub n_layer: u32,
}

impl VoxCpm2EncoderConfig {
    /// Canonical VoxCPM-0.5B encoder config.
    #[must_use]
    pub fn voxcpm_0_5b() -> Self {
        Self {
            hidden_dim: 1024,
            ffn_dim: 4096,
            n_head: 16,
            n_layer: 4,
        }
    }

    /// Miniature well-formed encoder config for tests.
    #[must_use]
    pub fn tiny_for_tests() -> Self {
        Self {
            hidden_dim: 16,
            ffn_dim: 32,
            n_head: 4,
            n_layer: 2,
        }
    }
}

/// Conditional flow-matching sampler hparams the VoxCPM
/// `dit_config.cfm_config` block transcribes (`config.json.dit_config
/// .cfm_config.*`, fetched 2026-07-24).
#[derive(Debug, Clone, PartialEq)]
pub struct VoxCpm2CfmConfig {
    /// σ_min for the flow-matching path (`sigma_min`). `1e-6`.
    pub sigma_min: f32,
    /// ODE solver identifier (`solver`). `"euler"` for the 0.5B release.
    /// A future release that switches to Heun / DPM++ would swap this
    /// field.
    pub solver: String,
    /// Training-side timestep scheduler (`t_scheduler`). `"log-norm"` —
    /// **training-only**; the inference `t_span` walks linear from `t=1`
    /// down to `t=0` per upstream `UnifiedCFM.forward`.
    pub t_scheduler: String,
    /// Inference classifier-free guidance rate (`inference_cfg_rate`).
    /// `2.0` for the 0.5B release.
    pub inference_cfg_rate: f32,
}

impl VoxCpm2CfmConfig {
    /// Canonical VoxCPM-0.5B CFM config.
    #[must_use]
    pub fn voxcpm_0_5b() -> Self {
        Self {
            sigma_min: 1e-6,
            solver: "euler".to_owned(),
            t_scheduler: "log-norm".to_owned(),
            inference_cfg_rate: 2.0,
        }
    }

    /// Miniature well-formed CFM config for tests (byte-parallel to the
    /// canonical release — solver / schedule / rate are training axes,
    /// so the tiny fixture shares them).
    #[must_use]
    pub fn tiny_for_tests() -> Self {
        Self::voxcpm_0_5b()
    }
}

/// Local DiT hparams the VoxCPM `dit_config` block transcribes
/// (`config.json.dit_config.*`, fetched 2026-07-24). The DiT is a
/// transformer stack that predicts velocity in the VAE latent space —
/// the "diffusion decoder" the SoTA plan Phase 4 note calls out.
#[derive(Debug, Clone, PartialEq)]
pub struct VoxCpm2DitConfig {
    /// DiT hidden dimension (`hidden_dim`). `1024`.
    pub hidden_dim: u32,
    /// SwiGLU FFN inner dimension (`ffn_dim`). `4096`.
    pub ffn_dim: u32,
    /// Attention head count (`num_heads`). `16`.
    pub n_head: u32,
    /// Transformer block count (`num_layers`). `4`.
    pub n_layer: u32,
    /// CFM sampler sub-config.
    pub cfm: VoxCpm2CfmConfig,
}

impl VoxCpm2DitConfig {
    /// Canonical VoxCPM-0.5B DiT config.
    #[must_use]
    pub fn voxcpm_0_5b() -> Self {
        Self {
            hidden_dim: 1024,
            ffn_dim: 4096,
            n_head: 16,
            n_layer: 4,
            cfm: VoxCpm2CfmConfig::voxcpm_0_5b(),
        }
    }

    /// Miniature well-formed DiT config for tests.
    #[must_use]
    pub fn tiny_for_tests() -> Self {
        Self {
            hidden_dim: 16,
            ffn_dim: 32,
            n_head: 4,
            n_layer: 2,
            cfm: VoxCpm2CfmConfig::tiny_for_tests(),
        }
    }
}

// ---------------------------------------------------------------------------
// Top-level config
// ---------------------------------------------------------------------------

/// Full VoxCPM-0.5B config: LM backbone + residual acoustic LM depth +
/// local encoder + local DiT + scalar-quantization bottleneck + AudioVAE V2
/// (through the shared [`ContinuousVaeConfig`] seam).
///
/// The AudioVAE attributes are not carried inline (they live on the shared
/// [`ContinuousVaeConfig`] seam re-exported from
/// [`vokra_ops::vae_continuous`]); [`Self::vae_config`] returns the
/// canonical released-variant VAE config and the handshake gate
/// [`Self::validate_for_forward`] cross-checks the `latent_dim` axis
/// against `feat_dim` (the LM step feature width MUST equal the VAE
/// `latent_dim` — a silent mismatch would drop or duplicate channels
/// entering the DiT).
#[derive(Debug, Clone, PartialEq)]
pub struct VoxCpm2Config {
    /// LM backbone sub-config (MiniCPM-4 flavour).
    pub lm: VoxCpm2LmConfig,
    /// Residual acoustic LM depth (`residual_lm_num_layers`). `6`.
    /// Same backbone family as [`Self::lm`] but with `vocab_size = 0`.
    pub residual_lm_n_layer: u32,
    /// Local encoder sub-config.
    pub encoder: VoxCpm2EncoderConfig,
    /// Local DiT sub-config (includes the CFM sampler settings).
    pub dit: VoxCpm2DitConfig,
    /// LM step feature width (`feat_dim`). `64` — must equal the shared
    /// VAE `latent_dim` (see the module-level docstring).
    pub feat_dim: u32,
    /// LM patch size (`patch_size`). `2` for the 0.5B release — the LM
    /// slots two VAE feature frames per step.
    pub patch_size: u32,
    /// Scalar-quantization latent dimension
    /// (`scalar_quantization_latent_dim`). `256`.
    pub scalar_quantization_latent_dim: u32,
    /// Scalar-quantization scale (`scalar_quantization_scale`). `9`.
    pub scalar_quantization_scale: u32,
    /// Max autoregressive sequence length (`max_length`). `4096`.
    pub max_length: u32,
}

impl VoxCpm2Config {
    /// Canonical VoxCPM-0.5B config (primary source: `config.json`,
    /// fetched 2026-07-24).
    #[must_use]
    pub fn voxcpm_0_5b() -> Self {
        Self {
            lm: VoxCpm2LmConfig::voxcpm_0_5b(),
            residual_lm_n_layer: 6,
            encoder: VoxCpm2EncoderConfig::voxcpm_0_5b(),
            dit: VoxCpm2DitConfig::voxcpm_0_5b(),
            feat_dim: 64,
            patch_size: 2,
            scalar_quantization_latent_dim: 256,
            scalar_quantization_scale: 9,
            max_length: 4096,
        }
    }

    /// Miniature well-formed config for shape / stability tests. Every
    /// ratio (`feat_dim == vae_config.latent_dim`, GQA algebra, positive
    /// FFN dims, positive residual depth) mirrors the real model.
    #[must_use]
    pub fn tiny_for_tests() -> Self {
        // The tiny VAE config has `latent_dim = 4`, so `feat_dim = 4`
        // keeps the handshake exact.
        Self {
            lm: VoxCpm2LmConfig::tiny_for_tests(),
            residual_lm_n_layer: 2,
            encoder: VoxCpm2EncoderConfig::tiny_for_tests(),
            dit: VoxCpm2DitConfig::tiny_for_tests(),
            feat_dim: 4,
            patch_size: 2,
            scalar_quantization_latent_dim: 8,
            scalar_quantization_scale: 3,
            max_length: 64,
        }
    }

    /// Returns the canonical released-variant AudioVAE V2 config
    /// (`ContinuousVaeConfig::voxcpm_0_5b`).
    #[must_use]
    pub fn vae_config(&self) -> ContinuousVaeConfig {
        ContinuousVaeConfig::voxcpm_0_5b()
    }

    /// Returns the tiny AudioVAE V2 config the tiny top-level fixture is
    /// paired with (`ContinuousVaeConfig::tiny_for_tests` — `latent_dim
    /// = 4` matches [`Self::tiny_for_tests`]'s `feat_dim`).
    #[must_use]
    pub fn tiny_vae_config() -> ContinuousVaeConfig {
        ContinuousVaeConfig::tiny_for_tests()
    }

    /// True iff every architectural axis is at its `0` sentinel — the
    /// shape-only conversion path the runtime tolerates as
    /// inspectable-but-not-forward-ready.
    #[must_use]
    pub fn is_placeholder_shape(&self) -> bool {
        self.lm.hidden_dim == 0
            && self.lm.n_layer == 0
            && self.dit.hidden_dim == 0
            && self.dit.n_layer == 0
            && self.encoder.hidden_dim == 0
    }

    /// Rejects `0`-placeholder / ill-formed configs before any forward
    /// runs, cross-checked against the canonical released AudioVAE V2.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] naming the offending field.
    pub fn validate_for_forward(&self) -> Result<()> {
        self.validate_for_forward_with_vae(&self.vae_config())
    }

    /// Variant-aware form of [`Self::validate_for_forward`] — cross-
    /// checks the `feat_dim` handshake against the caller-supplied VAE
    /// config instead of the canonical released one. Useful for a
    /// hypothetical future variant that widens the VAE latent.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] naming the offending field.
    pub fn validate_for_forward_with_vae(&self, vae: &ContinuousVaeConfig) -> Result<()> {
        // LM axes.
        let lm = &self.lm;
        if lm.hidden_dim == 0
            || lm.n_layer == 0
            || lm.n_head == 0
            || lm.n_head_kv == 0
            || lm.ffn_dim == 0
            || lm.vocab_size == 0
            || lm.max_position_embeddings == 0
            || lm.scale_emb == 0
            || lm.dim_model_base == 0
        {
            return Err(VokraError::InvalidArgument(
                "voxcpm2 lm config: every architectural axis must be > 0 (bind a real \
                 checkpoint or use VoxCpm2LmConfig::tiny_for_tests for shape tests)"
                    .to_owned(),
            ));
        }
        if lm.n_head % lm.n_head_kv != 0 {
            return Err(VokraError::InvalidArgument(format!(
                "voxcpm2 lm config: n_head_kv ({}) must divide n_head ({}) — GQA requires \
                 an integer group ratio",
                lm.n_head_kv, lm.n_head,
            )));
        }
        if !(lm.rope_base.is_finite() && lm.rope_base > 0.0) {
            return Err(VokraError::InvalidArgument(format!(
                "voxcpm2 lm config: rope_base must be a positive finite f32 (got {})",
                lm.rope_base,
            )));
        }
        if !(lm.rms_norm_eps.is_finite() && lm.rms_norm_eps > 0.0) {
            return Err(VokraError::InvalidArgument(format!(
                "voxcpm2 lm config: rms_norm_eps must be a positive finite f32 (got {})",
                lm.rms_norm_eps,
            )));
        }
        if !(lm.scale_depth.is_finite() && lm.scale_depth > 0.0) {
            return Err(VokraError::InvalidArgument(format!(
                "voxcpm2 lm config: scale_depth must be a positive finite f32 (got {})",
                lm.scale_depth,
            )));
        }
        if lm.rope_scaling_longrope && lm.rope_original_max_position_embeddings == 0 {
            return Err(VokraError::InvalidArgument(
                "voxcpm2 lm config: rope_scaling_longrope=true requires \
                 rope_original_max_position_embeddings > 0 (the longrope table anchor)"
                    .to_owned(),
            ));
        }

        // Residual LM depth.
        if self.residual_lm_n_layer == 0 {
            return Err(VokraError::InvalidArgument(
                "voxcpm2 config: residual_lm_n_layer must be > 0".to_owned(),
            ));
        }

        // Encoder axes.
        let e = &self.encoder;
        if e.hidden_dim == 0 || e.n_layer == 0 || e.n_head == 0 || e.ffn_dim == 0 {
            return Err(VokraError::InvalidArgument(
                "voxcpm2 encoder config: every architectural axis must be > 0".to_owned(),
            ));
        }

        // DiT axes.
        let d = &self.dit;
        if d.hidden_dim == 0 || d.n_layer == 0 || d.n_head == 0 || d.ffn_dim == 0 {
            return Err(VokraError::InvalidArgument(
                "voxcpm2 dit config: every architectural axis must be > 0".to_owned(),
            ));
        }
        // CFM sampler.
        let cfm = &d.cfm;
        if cfm.solver.is_empty() {
            return Err(VokraError::InvalidArgument(
                "voxcpm2 cfm config: solver name must not be empty".to_owned(),
            ));
        }
        if !(cfm.sigma_min.is_finite() && cfm.sigma_min >= 0.0) {
            return Err(VokraError::InvalidArgument(format!(
                "voxcpm2 cfm config: sigma_min must be a non-negative finite f32 (got {})",
                cfm.sigma_min,
            )));
        }
        if !(cfm.inference_cfg_rate.is_finite() && cfm.inference_cfg_rate >= 0.0) {
            return Err(VokraError::InvalidArgument(format!(
                "voxcpm2 cfm config: inference_cfg_rate must be a non-negative finite f32 \
                 (got {})",
                cfm.inference_cfg_rate,
            )));
        }

        // Top-level axes.
        if self.feat_dim == 0 {
            return Err(VokraError::InvalidArgument(
                "voxcpm2 config: feat_dim must be > 0".to_owned(),
            ));
        }
        if self.patch_size == 0 {
            return Err(VokraError::InvalidArgument(
                "voxcpm2 config: patch_size must be > 0".to_owned(),
            ));
        }
        if self.scalar_quantization_latent_dim == 0 {
            return Err(VokraError::InvalidArgument(
                "voxcpm2 config: scalar_quantization_latent_dim must be > 0".to_owned(),
            ));
        }
        if self.scalar_quantization_scale == 0 {
            return Err(VokraError::InvalidArgument(
                "voxcpm2 config: scalar_quantization_scale must be > 0".to_owned(),
            ));
        }
        if self.max_length == 0 {
            return Err(VokraError::InvalidArgument(
                "voxcpm2 config: max_length must be > 0".to_owned(),
            ));
        }

        // VAE handshake. The DiT operates in the VAE latent space, so
        // `feat_dim` (the LM step feature width) MUST equal
        // `vae.latent_dim` (the VAE decoder input width). A silent
        // mismatch would drop or duplicate channels entering the DiT.
        vae.validate_for_forward()?;
        if self.feat_dim != vae.latent_dim {
            return Err(VokraError::InvalidArgument(format!(
                "voxcpm2 config: feat_dim ({}) != vae.latent_dim ({}) — the DiT operates in \
                 the VAE latent space, so the LM's per-step feature width MUST equal the \
                 VAE latent width",
                self.feat_dim, vae.latent_dim,
            )));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Weight-store scaffold
// ---------------------------------------------------------------------------

/// VoxCPM-0.5B weight store scaffold.
///
/// Real binding is a follow-up wave (T29-equivalent — the LM backbone
/// walk, the residual LM walk, the encoder / DiT walks, and the AudioVAE
/// V2 walk all defer to the T29 tensor-name manifest fetch). This
/// scaffold carries only the placeholder VAE decoder-weight bundle so
/// downstream shape flow / handshake tests are unblocked; the LM /
/// encoder / DiT slots stay `Vec::new()` and the sole invariant this
/// slice pins is that `is_synthesized = true` prevents a spurious
/// synthesize call from returning zero audio (FR-EX-08 — the loud
/// [`VoxCpm2Tts::synthesize`] guard).
#[derive(Debug, Clone)]
pub struct VoxCpm2Weights {
    /// Placeholder for the LM backbone tensor bytes (aggregate). Real
    /// binding walks the upstream `base_lm.*` naming.
    pub lm_backbone: Vec<f32>,
    /// Placeholder for the residual LM tensor bytes (aggregate). Real
    /// binding walks the upstream `residual_lm.*` naming.
    pub residual_lm: Vec<f32>,
    /// Placeholder for the local encoder tensor bytes (aggregate). Real
    /// binding walks the upstream `feat_encoder.*` naming.
    pub encoder: Vec<f32>,
    /// Placeholder for the local DiT tensor bytes (aggregate). Real
    /// binding walks the upstream `feat_decoder.estimator.*` naming.
    pub dit: Vec<f32>,
    /// Placeholder for the fusion / projection tensor bytes.
    pub projections: Vec<f32>,
    /// `true` when built by [`Self::synthesized`] — never a real
    /// upstream checkpoint.
    pub is_synthesized: bool,
}

impl VoxCpm2Weights {
    /// Builds a deterministic zero-initialized fixture (shape scaffold
    /// only — every slot is `Vec::new()`).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] if
    /// `config.validate_for_forward` fails.
    pub fn synthesized(config: &VoxCpm2Config) -> Result<Self> {
        config.validate_for_forward()?;
        Ok(Self {
            lm_backbone: Vec::new(),
            residual_lm: Vec::new(),
            encoder: Vec::new(),
            dit: Vec::new(),
            projections: Vec::new(),
            is_synthesized: true,
        })
    }
}

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

/// VoxCPM-0.5B TTS engine handle.
///
/// Carries the resolved config + weight store + an optional
/// [`ContinuousVaeConfig`] override for the shared
/// [`vokra_ops::vae_continuous`] seam (default = the canonical released
/// variant). [`Self::synthesize`] is the primary text → PCM entry point;
/// until real weights are bound and the LM → residual LM → local
/// encoder → local DiT → CFM sampler → AudioVAE decode → 48 kHz PCM
/// chain is wired end-to-end (T29-equivalent follow-up wave), it returns
/// [`VokraError::NotImplemented`] naming the blocker (FR-EX-08 — never a
/// silent zero-fill or empty audio buffer).
#[derive(Debug, Clone)]
pub struct VoxCpm2Tts {
    cfg: VoxCpm2Config,
    weights: VoxCpm2Weights,
    vae: ContinuousVaeConfig,
}

impl VoxCpm2Tts {
    /// Assembles an engine from `cfg` and `weights`. Cross-checks the
    /// config against the canonical released VAE (see
    /// [`VoxCpm2Config::validate_for_forward`]) so a mismatched pair
    /// fails loudly here rather than deep inside a forward.
    ///
    /// The VAE seam defaults to the canonical released
    /// [`ContinuousVaeConfig::voxcpm_0_5b`]; callers with a hypothetical
    /// future variant chain through [`Self::with_vae_config`].
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] from
    /// `cfg.validate_for_forward`.
    pub fn new(cfg: VoxCpm2Config, weights: VoxCpm2Weights) -> Result<Self> {
        cfg.validate_for_forward()?;
        let vae = cfg.vae_config();
        Ok(Self { cfg, weights, vae })
    }

    /// Assembles an engine from `cfg` + `weights` + an explicit
    /// [`ContinuousVaeConfig`]. Cross-validated against
    /// `cfg.validate_for_forward_with_vae` at construction time.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] if the VAE handshake fails
    /// (`cfg.feat_dim != vae.latent_dim`) or `cfg` is otherwise
    /// ill-formed.
    pub fn new_with_vae(
        cfg: VoxCpm2Config,
        weights: VoxCpm2Weights,
        vae: ContinuousVaeConfig,
    ) -> Result<Self> {
        cfg.validate_for_forward_with_vae(&vae)?;
        Ok(Self { cfg, weights, vae })
    }

    /// Injects a caller-supplied VAE config — cross-validated against
    /// the config's `feat_dim` at assembly time.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] if the VAE handshake fails
    /// (`cfg.feat_dim != vae.latent_dim`).
    pub fn with_vae_config(mut self, vae: ContinuousVaeConfig) -> Result<Self> {
        self.cfg.validate_for_forward_with_vae(&vae)?;
        self.vae = vae;
        Ok(self)
    }

    /// The resolved configuration.
    #[must_use]
    pub fn config(&self) -> &VoxCpm2Config {
        &self.cfg
    }

    /// The active VAE config (default: `voxcpm_0_5b`; overridden via
    /// [`Self::with_vae_config`]).
    #[must_use]
    pub fn vae_config(&self) -> &ContinuousVaeConfig {
        &self.vae
    }

    /// True iff the weight store was built by
    /// [`VoxCpm2Weights::synthesized`] (never a real upstream
    /// checkpoint).
    #[must_use]
    pub fn is_synthesized(&self) -> bool {
        self.weights.is_synthesized
    }

    /// Synthesizes PCM for `text` at [`Self::vae_config`]'s
    /// `out_sample_rate_hz` (48 kHz for the canonical release).
    ///
    /// This is the primary text → PCM entry point. **Real weights
    /// required**: synthesized-weight builds cannot produce meaningful
    /// audio, so this returns [`VokraError::NotImplemented`] naming the
    /// blocker (FR-EX-08 — never a silent zero-fill or empty audio
    /// buffer).
    ///
    /// # Errors
    ///
    /// - [`VokraError::InvalidArgument`] if `text` is empty.
    /// - [`VokraError::NotImplemented`] otherwise (real forward not yet
    ///   bound — FR-EX-08).
    pub fn synthesize(&self, text: &str) -> Result<Vec<f32>> {
        if text.is_empty() {
            return Err(VokraError::InvalidArgument(
                "voxcpm2 synthesize: text is empty".to_owned(),
            ));
        }
        if self.weights.is_synthesized {
            return Err(VokraError::NotImplemented(
                "voxcpm2 synthesize: this engine holds synthesized weights \
                 (deterministic scaffold fixture from VoxCpm2Weights::synthesized) — \
                 synthesized-weight audio would be a hallucinated waveform, not real \
                 speech. Bind real VoxCPM-0.5B weights (apache-2.0, \
                 huggingface.co/openbmb/VoxCPM-0.5B) before invoking synthesize. \
                 The shape flow (config validation, weight-store construction, \
                 text-empty check, VAE handshake) is exercised through VoxCpm2Tts::new; \
                 real-checkpoint binding lands in a follow-up wave (T29-equivalent).",
            ));
        }
        Err(VokraError::NotImplemented(
            "voxcpm2 synthesize: real weights are bound but the MiniCPM-4 LM → residual \
             acoustic LM → local encoder → local DiT → UnifiedCFM sampler → AudioVAE V2 \
             decode → 48 kHz PCM forward path has not landed yet. Follow-up wave \
             (T29-equivalent): (1) run the MiniCPM-4 LM (GQA 16 Q ÷ 2 KV / RoPE θ=10000 \
             with longrope scaling / RMSNorm ε=1e-5 / SwiGLU) with the tokenizer prompt; \
             (2) run the 6-layer residual acoustic LM; (3) run the 4-layer local encoder \
             on the audio prompt (16 kHz mono PCM → AudioVAE V2 encode → 25 Hz continuous \
             latents); (4) drive vokra_ops::flow_sample with the 4-layer local DiT as the \
             velocity estimator (cfg_mode=SplitBatch, cfg_scale=inference_cfg_rate=2.0, \
             solver=Euler, schedule=Linear, nfe=inference_timesteps=10); (5) decode the \
             continuous VAE latents through vokra_ops::vae_continuous_decode → 48 kHz \
             PCM (the shared Phase 4 primitive). The scalar-quantization bottleneck \
             (scalar_quantization_latent_dim=256, scalar_quantization_scale=9) applies \
             inside the LM hidden stream — not the codec.",
        ))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expected_arch_is_voxcpm2() {
        assert_eq!(EXPECTED_ARCH, "voxcpm2");
    }

    #[test]
    fn arch_is_distinct_from_neighbouring_families() {
        // VoxCPM's terminal decoding hop is neither vocoder-LM
        // (HiFTChain) nor codec-LM (any RVQ / FSQ codec) — it is a
        // continuous VAE decoder consuming flow-matching sampler
        // output. Silently sharing an arch tag with any sibling would
        // misroute the runtime dispatch. Cross-checked against the
        // sibling `EXPECTED_ARCH` constants (public ones dereferenced;
        // private ones — cosyvoice2, kokoro, piper_plus — compared
        // against the same string literals the converter stamps into
        // `vokra.model.arch`, the sole cross-crate handshake).
        assert_ne!(EXPECTED_ARCH, "cosyvoice2");
        assert_ne!(EXPECTED_ARCH, crate::cosyvoice3::EXPECTED_ARCH);
        assert_ne!(EXPECTED_ARCH, crate::qwen3_tts::EXPECTED_ARCH);
        assert_ne!(EXPECTED_ARCH, crate::chatterbox::EXPECTED_ARCH);
        assert_ne!(EXPECTED_ARCH, crate::chatterbox_nano::EXPECTED_ARCH);
        assert_ne!(EXPECTED_ARCH, crate::chatterbox_turbo::EXPECTED_ARCH);
        assert_ne!(EXPECTED_ARCH, crate::dia::EXPECTED_ARCH);
        assert_ne!(EXPECTED_ARCH, crate::zonos::EXPECTED_ARCH);
    }

    #[test]
    fn encoder_sample_rate_matches_upstream_audio_vae_v2() {
        // AudioVAE V2 primary source (audio_vae_v2.py) — the encoder
        // consumes 16 kHz PCM.
        assert_eq!(VOXCPM_ENCODER_SAMPLE_RATE, 16_000);
    }

    /// Every hparam carries its primary-source value transcribed from
    /// `openbmb/VoxCPM-0.5B/config.json` (fetched 2026-07-24).
    #[test]
    fn lm_config_matches_primary_source() {
        let lm = VoxCpm2LmConfig::voxcpm_0_5b();
        assert_eq!(lm.hidden_dim, 1024);
        assert_eq!(lm.n_layer, 24);
        assert_eq!(lm.n_head, 16);
        assert_eq!(lm.n_head_kv, 2);
        assert_eq!(lm.ffn_dim, 4096);
        assert_eq!(lm.vocab_size, 73_448);
        assert_eq!(lm.max_position_embeddings, 32_768);
        assert!((lm.rope_base - 10_000.0).abs() < 1e-3);
        assert!((lm.rms_norm_eps - 1e-5).abs() < 1e-9);
        assert!(lm.rope_scaling_longrope);
        assert_eq!(lm.rope_original_max_position_embeddings, 32_768);
        assert_eq!(lm.scale_emb, 12);
        assert_eq!(lm.dim_model_base, 256);
        assert!((lm.scale_depth - 1.4).abs() < 1e-5);
        assert!(!lm.use_mup);
    }

    #[test]
    fn encoder_and_dit_configs_match_primary_source() {
        let e = VoxCpm2EncoderConfig::voxcpm_0_5b();
        assert_eq!(e.hidden_dim, 1024);
        assert_eq!(e.ffn_dim, 4096);
        assert_eq!(e.n_head, 16);
        assert_eq!(e.n_layer, 4);

        let d = VoxCpm2DitConfig::voxcpm_0_5b();
        assert_eq!(d.hidden_dim, 1024);
        assert_eq!(d.ffn_dim, 4096);
        assert_eq!(d.n_head, 16);
        assert_eq!(d.n_layer, 4);

        assert!((d.cfm.sigma_min - 1e-6).abs() < 1e-9);
        assert_eq!(d.cfm.solver, "euler");
        assert_eq!(d.cfm.t_scheduler, "log-norm");
        assert!((d.cfm.inference_cfg_rate - 2.0).abs() < 1e-5);
    }

    #[test]
    fn top_level_config_matches_primary_source() {
        let c = VoxCpm2Config::voxcpm_0_5b();
        assert_eq!(c.residual_lm_n_layer, 6);
        assert_eq!(c.feat_dim, 64);
        assert_eq!(c.patch_size, 2);
        assert_eq!(c.scalar_quantization_latent_dim, 256);
        assert_eq!(c.scalar_quantization_scale, 9);
        assert_eq!(c.max_length, 4096);
    }

    #[test]
    fn vae_handshake_holds_by_construction_for_canonical_release() {
        let c = VoxCpm2Config::voxcpm_0_5b();
        assert_eq!(c.feat_dim, c.vae_config().latent_dim);
        c.validate_for_forward().expect("canonical must validate");
    }

    #[test]
    fn tiny_config_validates_against_tiny_vae() {
        let c = VoxCpm2Config::tiny_for_tests();
        let vae = VoxCpm2Config::tiny_vae_config();
        assert_eq!(c.feat_dim, vae.latent_dim);
        c.validate_for_forward_with_vae(&vae)
            .expect("tiny must validate");
    }

    #[test]
    fn tiny_config_validates_against_canonical_vae_fails_by_design() {
        // The tiny top-level fixture pairs `feat_dim = 4` with the tiny
        // VAE `latent_dim = 4`; validating against the CANONICAL vae
        // (latent_dim = 64) must be rejected — a silent pass would let
        // a mis-paired top-level / VAE combination reach a forward.
        let c = VoxCpm2Config::tiny_for_tests();
        let err = c.validate_for_forward().expect_err("mismatch");
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn gqa_algebra_is_enforced() {
        let mut c = VoxCpm2Config::tiny_for_tests();
        c.lm.n_head_kv = 3; // 4 % 3 != 0
        let vae = VoxCpm2Config::tiny_vae_config();
        assert!(c.validate_for_forward_with_vae(&vae).is_err());
    }

    #[test]
    fn placeholder_shape_detects_zeroed_config() {
        let mut c = VoxCpm2Config::voxcpm_0_5b();
        c.lm.hidden_dim = 0;
        c.lm.n_layer = 0;
        c.dit.hidden_dim = 0;
        c.dit.n_layer = 0;
        c.encoder.hidden_dim = 0;
        assert!(c.is_placeholder_shape());
    }

    #[test]
    fn synthesized_weights_scaffold_flag_carried() {
        let c = VoxCpm2Config::voxcpm_0_5b();
        let w = VoxCpm2Weights::synthesized(&c).unwrap();
        assert!(w.is_synthesized);
    }

    #[test]
    fn tts_new_binds_canonical_pair() {
        let c = VoxCpm2Config::voxcpm_0_5b();
        let w = VoxCpm2Weights::synthesized(&c).unwrap();
        let tts = VoxCpm2Tts::new(c, w).unwrap();
        assert!(tts.is_synthesized());
        assert_eq!(tts.vae_config().latent_dim, 64);
    }

    #[test]
    fn tts_new_with_vae_uses_caller_supplied_vae_axes() {
        // `new_with_vae` validates against the caller's vae rather than
        // the canonical one — the canonical release itself round-trips
        // through this path (equivalent to `new`) so a caller that
        // wants explicit control over the seam gets it. This test pins
        // the equivalence for the canonical pair; the tiny pair's
        // handshake is covered by `tiny_config_validates_against_tiny_vae`
        // at the config level (the weight-store constructor
        // `VoxCpm2Weights::synthesized` walks `validate_for_forward`
        // which is canonical-only by design — a tiny pair therefore
        // exercises the vae seam at the config level rather than
        // through the engine).
        let c = VoxCpm2Config::voxcpm_0_5b();
        let vae = c.vae_config();
        let w = VoxCpm2Weights::synthesized(&c).unwrap();
        let tts = VoxCpm2Tts::new_with_vae(c, w, vae).unwrap();
        assert!(tts.is_synthesized());
        assert_eq!(tts.vae_config().latent_dim, 64);
    }

    #[test]
    fn tts_synthesize_empty_text_rejected_loudly() {
        let c = VoxCpm2Config::voxcpm_0_5b();
        let w = VoxCpm2Weights::synthesized(&c).unwrap();
        let tts = VoxCpm2Tts::new(c, w).unwrap();
        let err = tts.synthesize("").expect_err("empty text");
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn tts_synthesize_on_synth_weights_is_not_implemented_and_says_why() {
        let c = VoxCpm2Config::voxcpm_0_5b();
        let w = VoxCpm2Weights::synthesized(&c).unwrap();
        let tts = VoxCpm2Tts::new(c, w).unwrap();
        let err = tts.synthesize("hello").expect_err("synth synthesize");
        assert!(
            matches!(err, VokraError::NotImplemented(_)),
            "synth synth must be NotImplemented, got {err:?}"
        );
    }

    #[test]
    fn with_vae_config_rejects_mismatched_latent_dim() {
        let c = VoxCpm2Config::voxcpm_0_5b();
        let w = VoxCpm2Weights::synthesized(&c).unwrap();
        let tts = VoxCpm2Tts::new(c, w).unwrap();
        // Handing a tiny VAE (latent_dim=4) to a canonical VoxCPM
        // (feat_dim=64) must be rejected loudly (FR-EX-08).
        let err = tts
            .with_vae_config(VoxCpm2Config::tiny_vae_config())
            .expect_err("mismatched vae");
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn negative_rms_norm_eps_rejected() {
        let mut c = VoxCpm2Config::voxcpm_0_5b();
        c.lm.rms_norm_eps = -1e-5;
        assert!(c.validate_for_forward().is_err());
    }

    #[test]
    fn zero_scalar_quantization_axes_rejected() {
        let mut c = VoxCpm2Config::voxcpm_0_5b();
        c.scalar_quantization_latent_dim = 0;
        assert!(c.validate_for_forward().is_err());

        let mut c = VoxCpm2Config::voxcpm_0_5b();
        c.scalar_quantization_scale = 0;
        assert!(c.validate_for_forward().is_err());
    }

    #[test]
    fn zero_max_length_rejected() {
        let mut c = VoxCpm2Config::voxcpm_0_5b();
        c.max_length = 0;
        assert!(c.validate_for_forward().is_err());
    }

    #[test]
    fn zero_residual_lm_depth_rejected() {
        let mut c = VoxCpm2Config::voxcpm_0_5b();
        c.residual_lm_n_layer = 0;
        assert!(c.validate_for_forward().is_err());
    }
}
