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
//!   `scale_emb = 12` raw config value (effective `1.0` because
//!   `use_mup=false`), `dim_model_base = 256`, `scale_depth = 1.4`).
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
//! - A **continuous AudioVAE** (`patch_size = 2`, `feat_dim = 64`). For the
//!   0.5B release, the pinned `audio_vae.py` topology consumes/produces
//!   16-kHz PCM, with encoder rates `[2,5,8,8]` and decoder rates
//!   `[8,8,5,2]` (640 samples per latent frame). VoxCPM2-2B is a distinct
//!   v2 topology and remains on its separately authenticated route.
//!
//! Every field above is transcribed **verbatim** from
//! `huggingface.co/openbmb/VoxCPM-0.5B/raw/main/config.json` and the
//! upstream `audio_vae.py` AudioVAE defaults for 0.5B (fetched 2026-07-24
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
//! - The 0.5B **AudioVAE** encoder / decoder use the source-shaped
//!   [`AudioVaeDecoder`] kernels below; the shared
//!   [`vokra_ops::vae_continuous`] config remains the metadata seam for the
//!   separately authenticated 2B route.
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
//! - [`VoxCpm2Weights`] — deterministic test-only
//!   [`VoxCpm2Weights::synthesized`] fixture (zero-initialized; it is not a
//!   production checkpoint and cannot synthesize audio).
//! - [`VoxCpm2Tts`] — engine handle carrying config + weights. A native,
//!   source-shaped batch-one LM → local DiT → CFM sampler → AudioVAE decode
//!   route exists behind the crate-private staged seam. The public loading
//!   and synthesis entry points remain fail-closed until an immutable,
//!   complete composite manifest, tokenizer/provenance authentication, and
//!   independent CPU/Metal parity evidence are accepted (never a silent
//!   zero-fill, FR-EX-08).
//!
//! # No ONNX (permanent)
//!
//! VoxCPM-0.5B is distributed as safetensors + a Python pipeline; the
//! runtime **never** loads an ONNX graph (FR-LD-05, permanent constraint);
//! a future authenticated-composite wave (the current public route remains
//! fail-closed; no upstream source implementation is copied into the runtime).

use vokra_core::backend::BackendKind;
use vokra_core::{Result, VokraError};

use crate::compute::{Compute, HotOp};

mod audio_vae;
mod bound;
mod generation;
mod local;
mod minicpm4;
pub(crate) use audio_vae::pad_audio_vae_prompt_pcm;
pub use audio_vae::{
    AUDIO_VAE_DECODER_RATES, AUDIO_VAE_ENCODER_DIM, AUDIO_VAE_ENCODER_RATES, AUDIO_VAE_HOP,
    AUDIO_VAE_HOT_OPS, AUDIO_VAE_LATENT_DIM, AUDIO_VAE_PROMPT_CHUNK, AUDIO_VAE_SAMPLE_RATE,
    AudioVaeDecoder, AudioVaeEncoder, CausalConv1d, CausalConvTranspose1d, DecoderStage,
    EncoderStage, ResidualUnit, Snake,
};
pub use bound::{VoxCpm2Checkpoint, VoxCpm2StopProjection};
pub(crate) use generation::StagedGenerationRuntime;
pub use generation::{
    CausalLanguageState, EulerFlow, FEATURE_PATCHES_PER_STEP, FeatureGenerationLoop,
    LearnedStopController, PrefillState, ScalarQuantizer, StopController, VoxCpm2FlowDraws,
};
pub use local::{LocalDit, LocalDitWeights, LocalEncoder, UnifiedCfm};
pub use minicpm4::{
    MINICPM4_HOT_OPS, MiniCpm4BlockWeights, MiniCpm4Config, MiniCpm4KvCache, MiniCpm4Linear,
    MiniCpm4Model, MiniCpm4Stack, MiniCpm4StackWeights, minicpm4_compute,
};

/// Learned operations required by the complete staged 0.5B route. A single
/// preflight is shared by LM, local DiT, and AudioVAE so a selected Metal
/// backend cannot silently fall back to scalar host execution.
pub const VOXCPM2_HOT_OPS: &[HotOp] = &[
    HotOp::Gemm,
    HotOp::Softmax,
    HotOp::RmsNorm,
    HotOp::Silu,
    HotOp::Conv1d,
    HotOp::GroupedConv1d,
    HotOp::SnakeActivation,
    HotOp::Tanh,
];

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
/// Downstream `AudioVAE.out_sample_rate = 16_000` for synthesis
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
/// "longrope"`, 32-entry `long_factor` / `short_factor` tables). These
/// tables are authenticated values from the config companion (not learned
/// safetensors), and the runtime uses them when selecting the source
/// frequency cache. (c) MiniCPM-specific **µ-parametrization**-adjacent
/// scale knobs (`scale_emb = 12` raw config value, effective `1.0` for the
/// fixed non-µP 0.5B path, `dim_model_base = 256`,
/// `scale_depth = 1.4`), and (d) the token-id anchors
/// (`bos_token_id = 1`, `eos_token_id = 2`).
#[derive(Debug, Clone, PartialEq)]
pub struct VoxCpm2LmConfig {
    /// Backbone hidden dimension (`hidden_size`). `1024` (0.5B) /
    /// `2048` (2B).
    pub hidden_dim: u32,
    /// Backbone transformer block count (`num_hidden_layers`). `24`
    /// (0.5B) / `28` (2B).
    pub n_layer: u32,
    /// Backbone attention head count (`num_attention_heads`). `16`.
    pub n_head: u32,
    /// Backbone key/value head count for GQA (`num_key_value_heads`).
    /// `2` — the group ratio is `n_head / n_head_kv = 8` (each K/V head
    /// fans out to 8 Q heads — very wide GQA compared to Qwen2/3's 2/8).
    pub n_head_kv: u32,
    /// Per-head channel width for GQA (upstream `kv_channels`).
    /// `64` for 0.5B (derived as `hidden_dim / n_head = 1024 / 16 = 64`
    /// — the field is non-explicit in the 0.5B `config.json` and Vokra
    /// records the derived value), `128` for 2B (**explicit** in the
    /// 2B `config.json`, `hidden_dim / n_head = 2048 / 16 = 128`).
    /// Adding this field as an additive axis (default = 64 for the 0.5B
    /// backward-compat path) lets a variant that widens the head channel
    /// beyond `hidden_dim / n_head` (a hypothetical 3B / 5B) land
    /// without a re-conversion of the 0.5B GGUFs already shipped.
    /// `validate_for_forward` refuses `kv_channels == 0` (FR-EX-08).
    pub kv_channels: u32,
    /// SwiGLU FFN inner dimension (`intermediate_size`). `4096` (0.5B)
    /// / `6144` (2B).
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
    /// Raw MiniCPM `scale_emb` scalar (`scale_emb`). `12`. The official
    /// 0.5B forward applies it only on the µP path; because `use_mup=false`,
    /// the effective embedding scale is `1.0`.
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
    /// Return the scale actually applied by the official embedding path.
    #[must_use]
    pub fn effective_scale_emb(&self) -> f32 {
        if self.use_mup {
            self.scale_emb as f32
        } else {
            1.0
        }
    }

    /// Canonical VoxCPM-0.5B `lm_config` (primary source:
    /// `config.json.lm_config.*`, fetched 2026-07-24).
    #[must_use]
    pub fn voxcpm_0_5b() -> Self {
        Self {
            hidden_dim: 1024,
            n_layer: 24,
            n_head: 16,
            n_head_kv: 2,
            // 0.5B `config.json` does not carry `kv_channels`; the derived
            // value is `hidden_dim / n_head = 1024 / 16 = 64`. Vokra
            // records the derived value here so downstream binding does
            // not silently fold the 2B backbone with a stale 64-width KV
            // path (FR-EX-08).
            kv_channels: 64,
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

    /// Canonical **VoxCPM2-2B** `lm_config` (primary source:
    /// `huggingface.co/openbmb/VoxCPM2/raw/main/config.json.lm_config.*`,
    /// fetched 2026-07-28 — CLAUDE.md「ハルシネーション厳禁」).
    ///
    /// **Delta vs [`Self::voxcpm_0_5b`]**:
    /// - `hidden_dim`: 1024 → **2048** (×2)
    /// - `n_layer`: 24 → **28** (+4)
    /// - `ffn_dim`: 4096 → **6144** (×1.5)
    /// - `kv_channels`: 64 (derived) → **128** (explicit,
    ///   `hidden_dim / n_head = 2048 / 16`)
    ///
    /// GQA `n_head_kv = 2` (group ratio 8), vocab, RoPE base / scaling,
    /// scale-family scalars (`scale_emb / dim_model_base / scale_depth`)
    /// and `use_mup` all match 0.5B verbatim.
    #[must_use]
    pub fn voxcpm2_2b() -> Self {
        Self {
            hidden_dim: 2048,
            n_layer: 28,
            n_head: 16,
            n_head_kv: 2,
            kv_channels: 128,
            ffn_dim: 6144,
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
            // Tiny fixture derives `kv_channels = hidden_dim / n_head =
            // 16 / 4 = 4`, mirroring the 0.5B convention.
            kv_channels: 4,
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
    /// Local encoder hidden dimension (`hidden_dim`). `1024` (0.5B and
    /// 2B — width is unchanged; only depth scales).
    pub hidden_dim: u32,
    /// SwiGLU FFN inner dimension (`ffn_dim`). `4096` (0.5B and 2B).
    pub ffn_dim: u32,
    /// Attention head count (`num_heads`). `16`.
    pub n_head: u32,
    /// Transformer block count (`num_layers`). `4` (0.5B) / `12` (2B —
    /// depth ×3).
    pub n_layer: u32,
    /// Per-head channel width for the encoder attention (upstream
    /// `kv_channels`). `64` for 0.5B (derived, non-explicit) / `128` for
    /// 2B (explicit). See [`VoxCpm2LmConfig::kv_channels`] for the
    /// rationale (FR-EX-08 refuses zero at `validate_for_forward`).
    pub kv_channels: u32,
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
            // 0.5B non-explicit: derived `hidden_dim / n_head = 1024 /
            // 16 = 64`. Additive field default preserves the 0.5B binding.
            kv_channels: 64,
        }
    }

    /// Canonical **VoxCPM2-2B** encoder config (primary source:
    /// `openbmb/VoxCPM2/config.json.encoder_config.*`, fetched
    /// 2026-07-28).
    ///
    /// **Delta vs [`Self::voxcpm_0_5b`]**: `n_layer` 4 → 12 (encoder
    /// depth ×3); `kv_channels` 64 → 128 (explicit). All width axes
    /// (`hidden_dim`, `ffn_dim`, `n_head`) match 0.5B verbatim.
    #[must_use]
    pub fn voxcpm2_2b() -> Self {
        Self {
            hidden_dim: 1024,
            ffn_dim: 4096,
            n_head: 16,
            n_layer: 12,
            kv_channels: 128,
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
            kv_channels: 4,
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
    /// DiT hidden dimension (`hidden_dim`). `1024` (0.5B and 2B — width
    /// unchanged; only depth scales).
    pub hidden_dim: u32,
    /// SwiGLU FFN inner dimension (`ffn_dim`). `4096`.
    pub ffn_dim: u32,
    /// Attention head count (`num_heads`). `16`.
    pub n_head: u32,
    /// Transformer block count (`num_layers`). `4` (0.5B) / `12` (2B).
    pub n_layer: u32,
    /// Per-head channel width for the DiT attention (upstream
    /// `kv_channels`). `64` for 0.5B (derived, non-explicit) / `128` for
    /// 2B (explicit). See [`VoxCpm2LmConfig::kv_channels`].
    pub kv_channels: u32,
    /// Whether the DiT trains under **mean-mode** scheduling (upstream
    /// `dit_config.mean_mode`). `false` for both released variants
    /// (0.5B non-explicit, 2B explicit `false`). Recording the axis so
    /// a future variant that flips this to `true` (a training-side
    /// switch that MAY leak into an alternate inference sampler) does
    /// not silently drift the sampler shape — `validate_for_forward` /
    /// downstream forward path can inspect it explicitly.
    ///
    /// The crate-private staged local-DiT route consumes this flag when
    /// assembling time/delta-time conditioning. It remains explicit in the
    /// config so alternate variants cannot silently inherit 0.5B semantics.
    pub mean_mode: bool,
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
            // 0.5B non-explicit: derived `hidden_dim / n_head = 1024 /
            // 16 = 64`.
            kv_channels: 64,
            // 0.5B: `mean_mode` is not declared in `config.json.dit_config`;
            // upstream training-side default is `false`. Scaffold-safe
            // default; runtime binding wave (T29-equivalent) will pin the
            // exact upstream tensor names.
            mean_mode: false,
            cfm: VoxCpm2CfmConfig::voxcpm_0_5b(),
        }
    }

    /// Canonical **VoxCPM2-2B** DiT config (primary source:
    /// `openbmb/VoxCPM2/config.json.dit_config.*`, fetched 2026-07-28).
    ///
    /// **Delta vs [`Self::voxcpm_0_5b`]**: `n_layer` 4 → 12; `kv_channels`
    /// 64 → 128; `mean_mode` non-explicit → explicit `false`. CFM
    /// sampler axes match 0.5B verbatim (`sigma_min = 1e-6`, solver
    /// `"euler"`, scheduler `"log-norm"`, `inference_cfg_rate = 2.0`).
    #[must_use]
    pub fn voxcpm2_2b() -> Self {
        Self {
            hidden_dim: 1024,
            ffn_dim: 4096,
            n_head: 16,
            n_layer: 12,
            kv_channels: 128,
            mean_mode: false,
            // Primary source explicitly pins CFM axes at the same 0.5B
            // values; reuse the 0.5B factory to keep a single-source-of-
            // truth for the sampler axes.
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
            kv_channels: 4,
            mean_mode: false,
            cfm: VoxCpm2CfmConfig::tiny_for_tests(),
        }
    }
}

// ---------------------------------------------------------------------------
// Top-level config
// ---------------------------------------------------------------------------

/// Full VoxCPM-0.5B config: LM backbone + residual acoustic LM depth +
/// local encoder + local DiT + scalar-quantization bottleneck + 0.5B
/// AudioVAE (through the shared [`ContinuousVaeConfig`] seam).
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
    /// Residual acoustic LM depth (`residual_lm_num_layers`). `6` (0.5B)
    /// / `8` (2B). Same backbone family as [`Self::lm`] but with
    /// `vocab_size = 0`.
    pub residual_lm_n_layer: u32,
    /// Whether the residual acoustic LM disables RoPE
    /// (`residual_lm_no_rope`). `false` for 0.5B (RoPE applied through
    /// the residual LM attention path), `true` for 2B (RoPE **skipped**
    /// — Q/K carry no position embedding, purely content-based
    /// attention). Silently ignoring this flag on 2B would let the
    /// residual LM apply RoPE and drift the attention pattern away from
    /// upstream (a hazard the parity harness catches). The 0.5B staged route
    /// requires this to remain `false`; an alternate no-RoPE route must
    /// authenticate and consume the `true` setting before exposure.
    pub residual_lm_no_rope: bool,
    /// Local encoder sub-config.
    pub encoder: VoxCpm2EncoderConfig,
    /// Local DiT sub-config (includes the CFM sampler settings).
    pub dit: VoxCpm2DitConfig,
    /// LM step feature width (`feat_dim`). `64` for both 0.5B and 2B —
    /// must equal the shared VAE `latent_dim` (see the module-level
    /// docstring). Keeping `feat_dim` invariant across the 0.5B ↔ 2B
    /// axis is what preserves the VAE handshake unchanged.
    pub feat_dim: u32,
    /// LM patch size (`patch_size`). `2` for 0.5B (12.5 Hz LM step
    /// rate) / `4` for 2B (6.25 Hz LM step rate — the LM slots four
    /// VAE feature frames per step). Runtime forward path consumers of
    /// this axis MUST read the field from config; hard-coded `2` on the
    /// 2B path silently mis-shapes per-step tensors.
    pub patch_size: u32,
    /// Scalar-quantization latent dimension
    /// (`scalar_quantization_latent_dim`). `256` (0.5B) / `512` (2B).
    pub scalar_quantization_latent_dim: u32,
    /// Scalar-quantization scale (`scalar_quantization_scale`). `9`
    /// (unchanged across 0.5B / 2B).
    pub scalar_quantization_scale: u32,
    /// Max autoregressive sequence length (`max_length`). `4096`
    /// (0.5B) / `8192` (2B).
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
            // 0.5B: residual acoustic LM keeps RoPE; field additive with
            // 0.5B-preserving default.
            residual_lm_no_rope: false,
            encoder: VoxCpm2EncoderConfig::voxcpm_0_5b(),
            dit: VoxCpm2DitConfig::voxcpm_0_5b(),
            feat_dim: 64,
            patch_size: 2,
            scalar_quantization_latent_dim: 256,
            scalar_quantization_scale: 9,
            max_length: 4096,
        }
    }

    /// Canonical **VoxCPM2-2B** top-level config (primary source:
    /// `huggingface.co/openbmb/VoxCPM2/raw/main/config.json`, fetched
    /// 2026-07-28 — CLAUDE.md「ハルシネーション厳禁」).
    ///
    /// **Delta vs [`Self::voxcpm_0_5b`]**:
    /// - `residual_lm_n_layer`: 6 → **8** (+2)
    /// - `residual_lm_no_rope`: false → **true** (RoPE skipped on
    ///   residual acoustic LM Q/K — silent-drift hazard if runtime path
    ///   ignores this)
    /// - `patch_size`: 2 → **4** (LM step rate 12.5 → 6.25 Hz)
    /// - `scalar_quantization_latent_dim`: 256 → **512** (×2)
    /// - `max_length`: 4096 → **8192** (×2)
    /// - `feat_dim`: **64 (unchanged)** — the VAE handshake stays intact
    /// - `scalar_quantization_scale`: **9 (unchanged)**
    ///
    /// Sub-configs delegate to [`VoxCpm2LmConfig::voxcpm2_2b`],
    /// [`VoxCpm2EncoderConfig::voxcpm2_2b`],
    /// [`VoxCpm2DitConfig::voxcpm2_2b`] — the per-block deltas live
    /// there.
    ///
    /// Cross-check via [`Self::validate_for_forward_with_vae`] against
    /// [`ContinuousVaeConfig::voxcpm2_2b`] — the primary-source-pinned
    /// bandwidth-adaptive VAE. The default [`Self::vae_config`] method
    /// remains 0.5B-anchored for backward-compat (see its docstring for
    /// the 2B-aware helper [`Self::vae_config_2b`]).
    #[must_use]
    pub fn voxcpm2_2b() -> Self {
        Self {
            lm: VoxCpm2LmConfig::voxcpm2_2b(),
            residual_lm_n_layer: 8,
            residual_lm_no_rope: true,
            encoder: VoxCpm2EncoderConfig::voxcpm2_2b(),
            dit: VoxCpm2DitConfig::voxcpm2_2b(),
            feat_dim: 64,
            patch_size: 4,
            scalar_quantization_latent_dim: 512,
            scalar_quantization_scale: 9,
            max_length: 8192,
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
            residual_lm_no_rope: false,
            encoder: VoxCpm2EncoderConfig::tiny_for_tests(),
            dit: VoxCpm2DitConfig::tiny_for_tests(),
            feat_dim: 4,
            patch_size: 2,
            scalar_quantization_latent_dim: 8,
            scalar_quantization_scale: 3,
            max_length: 64,
        }
    }

    /// Returns the canonical released 0.5B AudioVAE config
    /// (`ContinuousVaeConfig::voxcpm_0_5b`).
    ///
    /// This method is 0.5B-anchored for backward-compat. Callers holding
    /// a [`Self::voxcpm2_2b`] instance who want the bandwidth-adaptive
    /// 2B VAE (with `sr_bin_boundaries = Some([20_000, 30_000, 40_000])`)
    /// should call [`Self::vae_config_2b`] explicitly. Both 0.5B and 2B
    /// share `latent_dim = 64` so the `feat_dim == vae.latent_dim`
    /// handshake passes through either seam.
    #[must_use]
    pub fn vae_config(&self) -> ContinuousVaeConfig {
        ContinuousVaeConfig::voxcpm_0_5b()
    }

    /// Returns the canonical 2B AudioVAE V2 config — the
    /// bandwidth-adaptive variant with `sr_bin_boundaries =
    /// Some([20_000, 30_000, 40_000])`. Static helper so a caller
    /// paired with [`Self::voxcpm2_2b`] can validate against the 2B VAE
    /// directly:
    ///
    /// ```rust
    /// use vokra_models::voxcpm2::VoxCpm2Config;
    /// let cfg = VoxCpm2Config::voxcpm2_2b();
    /// let vae = VoxCpm2Config::vae_config_2b();
    /// cfg.validate_for_forward_with_vae(&vae)
    ///     .expect("2B ↔ 2B handshake");
    /// ```
    #[must_use]
    pub fn vae_config_2b() -> ContinuousVaeConfig {
        ContinuousVaeConfig::voxcpm2_2b()
    }

    /// Returns the tiny AudioVAE config the tiny top-level fixture is
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
    /// runs, cross-checked against the canonical released 0.5B AudioVAE.
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
        if lm.kv_channels == 0 {
            return Err(VokraError::InvalidArgument(
                "voxcpm2 lm config: kv_channels must be > 0 (0.5B derives 64 from \
                 hidden_dim / n_head; 2B pins 128 explicitly — a zero here would mis-shape \
                 GQA head width). FR-EX-08."
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
        if e.kv_channels == 0 {
            return Err(VokraError::InvalidArgument(
                "voxcpm2 encoder config: kv_channels must be > 0 (FR-EX-08)".to_owned(),
            ));
        }

        // DiT axes.
        let d = &self.dit;
        if d.hidden_dim == 0 || d.n_layer == 0 || d.n_head == 0 || d.ffn_dim == 0 {
            return Err(VokraError::InvalidArgument(
                "voxcpm2 dit config: every architectural axis must be > 0".to_owned(),
            ));
        }
        if d.kv_channels == 0 {
            return Err(VokraError::InvalidArgument(
                "voxcpm2 dit config: kv_channels must be > 0 (FR-EX-08)".to_owned(),
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
/// This compatibility weight store is retained for deterministic shape and
/// handshake fixtures. It is not an authenticated composite loader: public
/// loading still requires the immutable complete main+AudioVAE+tokenizer
/// manifest and provenance gate. The native source-shaped batch-one route is
/// implemented behind crate-private staged constructors, while synthesized
/// fixtures remain explicitly non-production and cannot produce audio
/// (FR-EX-08 — the loud [`VoxCpm2Tts::synthesize`] guard).
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
/// variant). A source-shaped batch-one route now exists behind the
/// crate-private staged constructor. Public [`Self::synthesize`] remains
/// [`VokraError::NotImplemented`] until the immutable complete composite
/// manifest, tokenizer/provenance gate, and independent real-weight parity
/// evidence authorize public loading (FR-EX-08 — never a silent zero-fill or
/// empty audio buffer).
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

    /// Decode already-produced 0.5B continuous latents with a fully bound
    /// source-shaped AudioVAE decoder. This narrow seam does not synthesize
    /// or fabricate decoder weights; text/LM/CFM generation remains guarded
    /// by [`Self::synthesize`]. Samples follow the 0.5B 16 kHz contract.
    pub fn decode_audio_vae(
        &self,
        decoder: &AudioVaeDecoder,
        latents: &[f32],
        time: usize,
    ) -> Result<Vec<f32>> {
        if self.weights.is_synthesized {
            return Err(VokraError::NotImplemented(
                "voxcpm2 AudioVAE decode: synthesized engine weights cannot establish a real decoder binding",
            ));
        }
        if self.vae.out_sample_rate_hz != 16_000 {
            return Err(VokraError::NotImplemented(
                "voxcpm2 AudioVAE decode: source-shaped 0.5B decoder requires the authenticated 16 kHz contract",
            ));
        }
        decoder.decode_with_backend(latents, time, BackendKind::Cpu)
    }

    /// Backend-selected AudioVAE decode. The complete hot-op set is
    /// preflighted before any learned convolution is evaluated.
    pub fn decode_audio_vae_with_backend(
        &self,
        decoder: &AudioVaeDecoder,
        latents: &[f32],
        time: usize,
        backend: BackendKind,
    ) -> Result<Vec<f32>> {
        if self.weights.is_synthesized {
            return Err(VokraError::NotImplemented(
                "voxcpm2 AudioVAE decode: synthesized engine weights cannot establish a real decoder binding",
            ));
        }
        if self.vae.out_sample_rate_hz != 16_000 {
            return Err(VokraError::NotImplemented(
                "voxcpm2 AudioVAE decode: source-shaped 0.5B decoder requires the authenticated 16 kHz contract",
            ));
        }
        decoder.decode_with_backend(latents, time, backend)
    }

    /// True iff the weight store was built by
    /// [`VoxCpm2Weights::synthesized`] (never a real upstream
    /// checkpoint).
    #[must_use]
    pub fn is_synthesized(&self) -> bool {
        self.weights.is_synthesized
    }

    /// Synthesizes PCM for `text` at [`Self::vae_config`]'s
    /// `out_sample_rate_hz` (16 kHz for the canonical 0.5B release).
    ///
    /// This public entry point remains fail-closed. The source-shaped
    /// batch-one route is available only through the crate-private staged
    /// construction path; public loading still needs an immutable complete
    /// composite manifest, tokenizer/provenance authentication, and
    /// independent real-weight parity evidence. Synthesized fixtures never
    /// produce audio (FR-EX-08 — never a silent zero-fill or empty buffer).
    ///
    /// # Errors
    ///
    /// - [`VokraError::InvalidArgument`] if `text` is empty.
    /// - [`VokraError::NotImplemented`] otherwise (public composite gate and
    ///   parity evidence are not yet authorized — FR-EX-08).
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
                 the native source-shaped batch-one route is staged internally, but \
                 public loading still requires the immutable complete-composite \
                 manifest, tokenizer/provenance authentication, and real-weight parity.",
            ));
        }
        Err(VokraError::NotImplemented(
            "voxcpm2 synthesize: public loading remains fail-closed even though the native source-shaped batch-one MiniCPM-4 → residual LM → local encoder/DiT → UnifiedCFM → AudioVAE → 16 kHz PCM route exists internally; immutable complete-composite manifest, tokenizer/provenance authentication, and independent CPU/Metal real-weight parity are required before authorization. The historical main-only checkpoint is diagnostic only, and the scalar-quantization bottleneck remains an LM hidden-stream operation, not a codec.",
        ))
    }
}

/// Batch-one composite orchestration over source-shaped staged components.
/// This crate-private seam requires the caller to supply authenticated
/// tokenizer IDs, prompt audio, CFM draws, and unconditional CFG inputs; it
/// never fabricates any of those artifacts. Public GGUF loading remains
/// fail-closed until the VAST complete-composite manifest is fixed.
#[allow(dead_code)]
pub(crate) struct VoxCpm2Batch1Route {
    generation: StagedGenerationRuntime,
    local_encoder: LocalEncoder,
    local_dit: LocalDit,
    flow: UnifiedCfm,
    audio_encoder: AudioVaeEncoder,
    audio_decoder: AudioVaeDecoder,
}

#[allow(dead_code)] // Prompt packing awaits the authenticated complete composite route.
fn reshape_prompt_latents(latent: &[f32]) -> Result<Vec<f32>> {
    if latent.len() % 64 != 0 || latent.iter().any(|value| !value.is_finite()) {
        return Err(VokraError::InvalidArgument(
            "voxcpm prompt latent must be finite channel-major [64,frames]".to_owned(),
        ));
    }
    let frames = latent.len() / 64;
    if frames < FEATURE_PATCHES_PER_STEP || frames % FEATURE_PATCHES_PER_STEP != 0 {
        return Err(VokraError::InvalidArgument(
            "voxcpm prompt AudioVAE output must contain complete two-frame rows".to_owned(),
        ));
    }
    // Preserve the official boundary rule: reshape to [rows,2,64], then
    // remove the final complete row before matching audio-mask rows.
    let rows = frames / FEATURE_PATCHES_PER_STEP - 1;
    let mut output = vec![0.0; rows * FEATURE_PATCHES_PER_STEP * 64];
    for row in 0..rows {
        for patch in 0..FEATURE_PATCHES_PER_STEP {
            for channel in 0..64 {
                output[row * FEATURE_PATCHES_PER_STEP * 64 + patch * 64 + channel] =
                    latent[channel * frames + row * FEATURE_PATCHES_PER_STEP + patch];
            }
        }
    }
    Ok(output)
}

impl VoxCpm2Batch1Route {
    #[allow(clippy::too_many_arguments)]
    #[allow(dead_code)] // Staged orchestration awaits complete composite authorization.
    pub(crate) fn from_staged_components(
        generation: StagedGenerationRuntime,
        local_encoder: LocalEncoder,
        local_dit: LocalDit,
        flow: UnifiedCfm,
        audio_encoder: AudioVaeEncoder,
        audio_decoder: AudioVaeDecoder,
    ) -> Self {
        Self {
            generation,
            local_encoder,
            local_dit,
            flow,
            audio_encoder,
            audio_decoder,
        }
    }

    /// Run batch-one text/audio prefill, persistent LM feature generation,
    /// endpoint trimming, and 16-kHz AudioVAE decode on one selected backend.
    /// `audio_mask` identifies prompt rows; when a row is audio, the encoded
    /// prompt must contain exactly two usable VAE frames for that row. The
    /// source CFG negative half is built internally from the dynamic prefix
    /// and zero mu.
    #[allow(dead_code)] // Staged orchestration awaits complete composite authorization.
    pub(crate) fn synthesize_batch1(
        &mut self,
        token_ids: &[u32],
        audio_mask: &[bool],
        prompt_pcm: Option<&[f32]>,
        loop_: &FeatureGenerationLoop,
        draws: &VoxCpm2FlowDraws,
        backend: BackendKind,
    ) -> Result<Vec<f32>> {
        if token_ids.is_empty() || token_ids.len() != audio_mask.len() {
            return Err(VokraError::InvalidArgument(
                "voxcpm batch-1 route requires matching non-empty token/audio rows".to_owned(),
            ));
        }
        let compute = Compute::for_backend(backend, VOXCPM2_HOT_OPS)?;
        let text_embeddings = self.generation.embed_tokens_raw(token_ids)?;
        let mut audio_features = vec![0.0; token_ids.len() * FEATURE_PATCHES_PER_STEP * 64];
        let audio_rows = audio_mask.iter().filter(|value| **value).count();
        let mut seed_prefix = vec![0.0; FEATURE_PATCHES_PER_STEP * 64];
        if let Some(pcm) = prompt_pcm {
            if audio_rows == 0 {
                return Err(VokraError::InvalidArgument(
                    "voxcpm prompt PCM requires at least one audio-mask row".to_owned(),
                ));
            }
            let padded_pcm = pad_audio_vae_prompt_pcm(pcm)?;
            let latent =
                self.audio_encoder
                    .encode_with_compute(&padded_pcm, padded_pcm.len(), &compute)?;
            let prompt_features = reshape_prompt_latents(&latent)?;
            if prompt_features.len() != audio_rows * FEATURE_PATCHES_PER_STEP * 64 {
                return Err(VokraError::InvalidArgument(
                    "voxcpm prompt audio rows do not match post-drop AudioVAE features".to_owned(),
                ));
            }
            let mut audio_row = 0;
            for (row, is_audio) in audio_mask.iter().copied().enumerate() {
                if !is_audio {
                    continue;
                }
                audio_features[row * FEATURE_PATCHES_PER_STEP * 64
                    ..(row + 1) * FEATURE_PATCHES_PER_STEP * 64]
                    .copy_from_slice(
                        &prompt_features[audio_row * FEATURE_PATCHES_PER_STEP * 64
                            ..(audio_row + 1) * FEATURE_PATCHES_PER_STEP * 64],
                    );
                audio_row += 1;
            }
            if audio_rows != 0 {
                let last = audio_mask
                    .iter()
                    .rposition(|is_audio| *is_audio)
                    .expect("audio_rows checked nonzero");
                seed_prefix.copy_from_slice(
                    &audio_features[last * FEATURE_PATCHES_PER_STEP * 64
                        ..(last + 1) * FEATURE_PATCHES_PER_STEP * 64],
                );
            }
        } else if audio_rows != 0 {
            return Err(VokraError::InvalidArgument(
                "voxcpm audio rows require caller-owned prompt PCM".to_owned(),
            ));
        }
        let local_encoder = &self.local_encoder;
        let generation = &self.generation;
        let prefill = loop_.assemble_prefill(
            &text_embeddings,
            &audio_features,
            audio_mask,
            VoxCpm2LmConfig::voxcpm_0_5b().effective_scale_emb(),
            |feature| {
                let encoded = local_encoder.forward(feature, 1, 2, &compute)?;
                generation.enc_to_lm(&encoded, 1, &compute)
            },
        )?;
        self.generation.prefill(&prefill, &compute)?;
        let patches = self.generation.generate_batch1(
            loop_,
            &seed_prefix,
            &self.local_encoder,
            &self.local_dit,
            &self.flow,
            draws,
            &compute,
        )?;
        let steps = patches.len() / (FEATURE_PATCHES_PER_STEP * 64);
        let latents = FeatureGenerationLoop::patches_to_latent(
            &patches,
            1,
            steps,
            FEATURE_PATCHES_PER_STEP,
            64,
        )?;
        let latent_time = latents.len() / 64;
        self.audio_decoder
            .decode_with_compute(&latents, latent_time, &compute)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_latent_reshape_drops_final_complete_patch_row() {
        let mut latent = vec![0.0f32; 64 * 4];
        for channel in 0..64 {
            for frame in 0..4 {
                latent[channel * 4 + frame] = channel as f32 * 10.0 + frame as f32;
            }
        }
        let features = reshape_prompt_latents(&latent).unwrap();
        assert_eq!(features.len(), 2 * 64);
        assert_eq!(features[0], 0.0);
        assert_eq!(features[1], 10.0);
        assert_eq!(features[64], 1.0);
        assert_eq!(features[65], 11.0);
        assert!(reshape_prompt_latents(&latent[..64 * 3]).is_err());
    }

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
    fn encoder_sample_rate_matches_upstream_audio_vae() {
        // AudioVAE primary source (audio_vae.py) — the encoder
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
        assert_eq!(
            lm.kv_channels, 64,
            "0.5B derived kv_channels = hidden_dim / n_head = 1024 / 16"
        );
        assert_eq!(lm.ffn_dim, 4096);
        assert_eq!(lm.vocab_size, 73_448);
        assert_eq!(lm.max_position_embeddings, 32_768);
        assert!((lm.rope_base - 10_000.0).abs() < 1e-3);
        assert!((lm.rms_norm_eps - 1e-5).abs() < 1e-9);
        assert!(lm.rope_scaling_longrope);
        assert_eq!(lm.rope_original_max_position_embeddings, 32_768);
        assert_eq!(lm.scale_emb, 12);
        assert_eq!(lm.effective_scale_emb(), 1.0);
        assert_eq!(lm.dim_model_base, 256);
        assert!((lm.scale_depth - 1.4).abs() < 1e-5);
        assert!(!lm.use_mup);
    }

    /// VoxCPM2-2B LM primary-source pin
    /// (`openbmb/VoxCPM2/raw/main/config.json.lm_config.*`,
    /// fetched 2026-07-28). Silently drifting these values would let a
    /// converter that emits `vokra.voxcpm2.lm.*` for the 2B GGUF slip
    /// past parity into a mis-shaped forward.
    #[test]
    fn lm_config_voxcpm2_2b_matches_primary_source() {
        let lm = VoxCpm2LmConfig::voxcpm2_2b();
        assert_eq!(lm.hidden_dim, 2048, "2B hidden_dim ×2 vs 0.5B");
        assert_eq!(lm.n_layer, 28, "2B n_layer +4 vs 0.5B");
        assert_eq!(lm.n_head, 16, "unchanged vs 0.5B");
        assert_eq!(lm.n_head_kv, 2, "unchanged vs 0.5B (GQA ratio 8)");
        assert_eq!(
            lm.kv_channels, 128,
            "2B explicit kv_channels = hidden_dim / n_head = 2048 / 16"
        );
        assert_eq!(lm.ffn_dim, 6144, "2B ffn_dim ×1.5 vs 0.5B");
        assert_eq!(lm.vocab_size, 73_448, "unchanged");
        assert_eq!(lm.max_position_embeddings, 32_768, "unchanged");
        assert!((lm.rope_base - 10_000.0).abs() < 1e-3, "unchanged");
        assert!((lm.rms_norm_eps - 1e-5).abs() < 1e-9, "unchanged");
        assert!(lm.rope_scaling_longrope, "unchanged");
        assert_eq!(
            lm.rope_original_max_position_embeddings, 32_768,
            "unchanged"
        );
        assert_eq!(lm.scale_emb, 12, "unchanged");
        assert_eq!(lm.dim_model_base, 256, "unchanged");
        assert!((lm.scale_depth - 1.4).abs() < 1e-5, "unchanged");
        assert!(!lm.use_mup, "unchanged");
    }

    /// VoxCPM2-2B encoder primary-source pin. Depth scales ×3 (4 → 12).
    #[test]
    fn encoder_config_voxcpm2_2b_matches_primary_source() {
        let e = VoxCpm2EncoderConfig::voxcpm2_2b();
        assert_eq!(e.hidden_dim, 1024, "encoder width unchanged");
        assert_eq!(e.ffn_dim, 4096, "encoder FFN unchanged");
        assert_eq!(e.n_head, 16, "encoder heads unchanged");
        assert_eq!(e.n_layer, 12, "encoder depth ×3 vs 0.5B");
        assert_eq!(e.kv_channels, 128, "encoder kv_channels explicit 128");
    }

    /// VoxCPM2-2B DiT primary-source pin. Depth scales ×3 (4 → 12),
    /// `mean_mode` explicit `false`, CFM sampler axes unchanged.
    #[test]
    fn dit_config_voxcpm2_2b_matches_primary_source() {
        let d = VoxCpm2DitConfig::voxcpm2_2b();
        assert_eq!(d.hidden_dim, 1024);
        assert_eq!(d.ffn_dim, 4096);
        assert_eq!(d.n_head, 16);
        assert_eq!(d.n_layer, 12, "DiT depth ×3 vs 0.5B");
        assert_eq!(d.kv_channels, 128);
        assert!(!d.mean_mode, "2B pins mean_mode = false");
        assert!((d.cfm.sigma_min - 1e-6).abs() < 1e-9);
        assert_eq!(d.cfm.solver, "euler");
        assert_eq!(d.cfm.t_scheduler, "log-norm");
        assert!((d.cfm.inference_cfg_rate - 2.0).abs() < 1e-5);
    }

    /// VoxCPM2-2B top-level primary-source pin. The delta table lives on
    /// [`VoxCpm2Config::voxcpm2_2b`] rustdoc; this test pins every axis
    /// so a converter regression that emits the wrong `vokra.voxcpm2.*`
    /// hparam block cannot pass silently.
    #[test]
    fn voxcpm2_2b_matches_primary_source() {
        let c = VoxCpm2Config::voxcpm2_2b();
        assert_eq!(c.residual_lm_n_layer, 8, "residual LM depth 6 → 8");
        assert!(c.residual_lm_no_rope, "2B: residual LM disables RoPE");
        assert_eq!(c.feat_dim, 64, "feat_dim unchanged — VAE handshake pin");
        assert_eq!(
            c.patch_size, 4,
            "2B patch_size 2 → 4 (LM step 12.5 → 6.25 Hz)"
        );
        assert_eq!(
            c.scalar_quantization_latent_dim, 512,
            "2B scalar_quantization_latent_dim 256 → 512 (×2)"
        );
        assert_eq!(
            c.scalar_quantization_scale, 9,
            "scalar_quantization_scale unchanged"
        );
        assert_eq!(c.max_length, 8192, "2B max_length 4096 → 8192 (×2)");
    }

    /// The 2B top-level + 2B VAE seam handshake must pass — both have
    /// `latent_dim = 64` (the primary-source-pinned invariant).
    #[test]
    fn voxcpm2_2b_config_validates_against_2b_vae() {
        let c = VoxCpm2Config::voxcpm2_2b();
        let vae = VoxCpm2Config::vae_config_2b();
        assert_eq!(c.feat_dim, vae.latent_dim);
        c.validate_for_forward_with_vae(&vae)
            .expect("2B ↔ 2B handshake must validate");
    }

    /// The 2B top-level + 0.5B VAE seam handshake ALSO passes because
    /// `latent_dim == 64` on both variants (the primary-source
    /// invariant). This documents the backward-compat property that lets
    /// [`VoxCpm2Config::vae_config`] stay 0.5B-anchored for existing
    /// callers.
    #[test]
    fn voxcpm2_2b_config_validates_against_0_5b_vae_by_shared_latent_dim() {
        let c = VoxCpm2Config::voxcpm2_2b();
        c.validate_for_forward()
            .expect("2B ↔ 0.5B VAE handshake must pass (latent_dim = 64 shared)");
    }

    #[test]
    fn kv_channels_zero_rejected() {
        // LM
        let mut c = VoxCpm2Config::voxcpm_0_5b();
        c.lm.kv_channels = 0;
        assert!(c.validate_for_forward().is_err());
        // Encoder
        let mut c = VoxCpm2Config::voxcpm_0_5b();
        c.encoder.kv_channels = 0;
        assert!(c.validate_for_forward().is_err());
        // DiT
        let mut c = VoxCpm2Config::voxcpm_0_5b();
        c.dit.kv_channels = 0;
        assert!(c.validate_for_forward().is_err());
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
    fn tts_synthesize_without_composite_manifest_stays_fail_closed() {
        let c = VoxCpm2Config::voxcpm_0_5b();
        let w = VoxCpm2Weights {
            lm_backbone: Vec::new(),
            residual_lm: Vec::new(),
            encoder: Vec::new(),
            dit: Vec::new(),
            projections: Vec::new(),
            is_synthesized: false,
        };
        let tts = VoxCpm2Tts::new(c, w).unwrap();
        let err = tts
            .synthesize("hello")
            .expect_err("fragmentary weights must not authorize synthesis");
        match err {
            VokraError::NotImplemented(message) => {
                assert!(message.contains("complete-composite manifest"));
                assert!(message.contains("tokenizer/provenance"));
                assert!(message.contains("parity"));
                assert!(message.contains("batch-one"));
            }
            other => panic!("expected fail-closed public synthesis, got {other:?}"),
        }
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
