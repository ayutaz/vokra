//! **VoxCPM family** (0.5B + VoxCPM2-2B): safetensors checkpoint → GGUF
//! conversion (SoTA plan Phase 4 initial land 2026-07-24; 2B variant land
//! 2026-07-30, spec `docs/superpowers/specs/2026-07-28-voxcpm2-2b-design.md`
//! Option C hybrid — single converter file with a
//! [`VoxCpm2Variant`] enum + shared arch + name-based runtime dispatch).
//!
//! Inputs (both apache-2.0 end-to-end):
//!
//! - `openbmb/VoxCPM-0.5B` → `model.safetensors` (BF16, single file).
//! - `openbmb/VoxCPM2` → `model.safetensors` (BF16, 4.96 GB, single file at
//!   pinned SHA `bffb3df5a29440629464e5e839f4d214c8714c3d`).
//!
//! Output: a GGUF carrying every float tensor plus the `vokra.voxcpm2.*`,
//! `vokra.vae_continuous.*`, and `vokra.model.*` / `vokra.provenance.*`
//! metadata chunks the native VoxCPM implementation
//! (`crates/vokra-models/src/voxcpm2/`) reads.
//!
//! # Variant detection (single-file converter, no side-car `--config`)
//!
//! The converter has no `--config` path (byte-only `Vec<u8>` interface —
//! the CLI hands it the raw safetensors bytes). Variant selection is
//! therefore driven by the **safetensors payload itself**: the discriminant
//! is the LM backbone hidden dim, which is `1024` for 0.5B and `2048` for
//! the 2B release (primary source `config.json.lm_config.hidden_size`).
//! The token-embedding tensor `base_lm.embed_tokens.weight` carries the
//! discriminant on its shape's last axis (shape `[vocab_size, hidden_dim]`,
//! `vocab_size = 73_448` in both variants — unchanged). See
//! [`detect_variant`].
//!
//! A shape that is neither `1024` nor `2048` is rejected loudly (FR-EX-08 —
//! never a silent default to 0.5B). A safetensors file with no
//! `base_lm.embed_tokens.weight` tensor is also rejected loudly — that
//! signals either a sharded release whose first shard lacks the LM half or
//! a corrupt input, and either case would cause a mis-shaped GGUF if we
//! guessed.
//!
//! # What is transcribed vs. shape-driven
//!
//! - **Transcribed constants** — every hparam of the `vokra.voxcpm2.*`
//!   chunk group is transcribed **verbatim** from the primary source
//!   `config.json` at each variant's HF release. Two `struct VariantHparams`
//!   factories ([`VariantHparams::half_b`] and [`VariantHparams::two_b`])
//!   hold the transcribed constants per variant. CLAUDE.md「ハルシネー
//!   ション厳禁」.
//! - **Nested config blocks** — VoxCPM splits its `config.json` into
//!   `lm_config.*` (MiniCPM-4 backbone), `encoder_config.*`, `dit_config.*`
//!   (with a nested `cfm_config.*`) blocks. All are transcribed in full.
//! - **VAE handshake** — `feat_dim` (top-level `config.json.feat_dim`,
//!   unchanged at `64` across both variants) must equal `vae.latent_dim`
//!   (unchanged at `64`); the runtime rejects a mismatch loudly at load
//!   per FR-EX-08 via
//!   [`vokra_models::voxcpm2::VoxCpm2Config::validate_for_forward_with_vae`].
//!
//! # Tensor naming contract
//!
//! GGUF tensor names are the **upstream safetensors names verbatim**
//! (the CSM / Kokoro / CosyVoice2 / Chatterbox / Qwen3-TTS contract).
//! Real-weight binding is a follow-up wave gated on the upstream tensor-
//! name manifest fetch; this converter passes every F32 / F16 / BF16
//! tensor through unchanged so a future `VoxCpm2Weights::from_gguf` can
//! walk the same names.
//!
//! # BF16 posture
//!
//! Both upstream releases ship in **BF16**
//! (`config.json.dtype = "bfloat16"`). The BF16 pass-through arm added
//! 2026-07-25 (mirror of qwen3-tts / vibevoice / moshi / voxtral) emits
//! GGUF type 30 verbatim; runtime widens BF16 → f32 losslessly on load
//! via the single choke point `crates/vokra-core/src/gguf/quant/mod.rs
//! decode_bf16` (BF16 = top 16 bits of an f32 — `bits << 16` is exact).
//!
//! # No ONNX (permanent)
//!
//! Both releases are distributed as safetensors + a Python pipeline; this
//! converter **never** touches ONNX (FR-LD-05); the pipeline is
//! re-implemented natively in `crates/vokra-models/src/voxcpm2/`
//! (whisper.cpp 型 self re-implementation, CLAUDE.md 設計判断 4).

use vokra_core::LicenseClass;
use vokra_core::gguf::{
    GgmlType, GgufArray, GgufBuilder, GgufMetadataValue, GgufValueType, chunks,
};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` for VoxCPM-family GGUFs — kept in sync with the
/// runtime constant `vokra-models::voxcpm2::EXPECTED_ARCH`.
/// Intentionally **distinct** from every sibling arch tag because VoxCPM's
/// terminal decoding hop is a continuous VAE decoder — not HiFTNet /
/// HiFT-GAN (CosyVoice2/3, Chatterbox) and not any RVQ / FSQ codec
/// (Qwen3-TTS, SNAC family, Kyutai STT, Moshi, CSM, Voxtral, Dia, Zonos).
/// Silently sharing an arch tag would mis-route the runtime dispatch.
///
/// The same arch tag serves both VoxCPM-0.5B and VoxCPM2-2B — the LM
/// backbone, encoder, DiT, CFM sampler and AudioVAE V2 topology are
/// byte-parallel between the two releases (only the hparams change).
/// The variant that produced a specific GGUF is recorded in
/// `vokra.model.name` (see [`half_b_name`] / [`two_b_name`]).
pub(crate) const ARCH: &str = "voxcpm2";

/// `vokra.model.name` value the converter stamps for the canonical
/// **VoxCPM-0.5B** release. **Renamed from `"voxcpm-0.5b"` to
/// `"voxcpm2-0.5b"` on 2026-07-30** (Option C hybrid) so both variants
/// carry the arch-family prefix and the parity harness can dispatch on a
/// single string. The compliance registry
/// `vokra_core::compliance::license_class` was extended with a `voxcpm2-`
/// prefix arm on the same day so the new name resolves permissive without
/// touching the legacy `voxcpm-0.5b` string (kept in the registry for
/// backward compat with any pre-2026-07-30 GGUF on disk).
pub(crate) const fn half_b_name() -> &'static str {
    "voxcpm2-0.5b"
}

/// `vokra.model.name` value the converter stamps for the canonical
/// **VoxCPM2-2B** release (SoTA plan Phase 4 scale-up, spec
/// `docs/superpowers/specs/2026-07-28-voxcpm2-2b-design.md`).
///
/// A GGUF carrying this name activates the 2B parity branch in the
/// tts-continuous-vae harness (variant-aware since 2026-07-30).
pub(crate) const fn two_b_name() -> &'static str {
    "voxcpm2-2b"
}

// --- vokra.voxcpm2.* metadata keys (kept as constants in the converter;
// the runtime side lives in `crates/vokra-models/src/voxcpm2/mod.rs` —
// the two crates share only `vokra-core`, so the cross-crate constant
// duplication rule the CSM / CosyVoice2 / Kokoro / Chatterbox / Qwen3-TTS
// family converters use applies) -----------------------------------------

// Top-level
const KEY_FEAT_DIM: &str = "vokra.voxcpm2.feat_dim";
const KEY_PATCH_SIZE: &str = "vokra.voxcpm2.patch_size";
const KEY_RESIDUAL_LM_N_LAYER: &str = "vokra.voxcpm2.residual_lm_n_layer";
/// Added 2026-07-30 for the 2B variant (0.5B: `false`; 2B: `true`).
/// Documented in `crates/vokra-models/src/voxcpm2/mod.rs`'s
/// `residual_lm_no_rope` docstring.
const KEY_RESIDUAL_LM_NO_ROPE: &str = "vokra.voxcpm2.residual_lm.no_rope";
const KEY_SCALAR_QUANT_LATENT_DIM: &str = "vokra.voxcpm2.scalar_quantization.latent_dim";
const KEY_SCALAR_QUANT_SCALE: &str = "vokra.voxcpm2.scalar_quantization.scale";
const KEY_MAX_LENGTH: &str = "vokra.voxcpm2.max_length";
const KEY_MODEL_FAMILY: &str = "vokra.voxcpm2.model_family";

// LM backbone axes — config.json.lm_config.*
const KEY_LM_HIDDEN_DIM: &str = "vokra.voxcpm2.lm.hidden_dim";
const KEY_LM_N_LAYER: &str = "vokra.voxcpm2.lm.n_layer";
const KEY_LM_N_HEAD: &str = "vokra.voxcpm2.lm.n_head";
const KEY_LM_N_HEAD_KV: &str = "vokra.voxcpm2.lm.n_head_kv";
/// Added 2026-07-30 for the 2B variant. 0.5B: `64` (derived from
/// `hidden_dim / n_head`); 2B: `128` (explicit in
/// `config.json.lm_config.kv_channels`). Documented in
/// `crates/vokra-models/src/voxcpm2/mod.rs`'s `kv_channels` docstring on
/// each of `VoxCpm2LmConfig`, `VoxCpm2EncoderConfig`, `VoxCpm2DitConfig`.
const KEY_LM_KV_CHANNELS: &str = "vokra.voxcpm2.lm.kv_channels";
const KEY_LM_FFN_DIM: &str = "vokra.voxcpm2.lm.ffn_dim";
const KEY_LM_VOCAB_SIZE: &str = "vokra.voxcpm2.lm.vocab_size";
const KEY_LM_MAX_POSITIONS: &str = "vokra.voxcpm2.lm.max_position_embeddings";
const KEY_LM_ROPE_BASE: &str = "vokra.voxcpm2.lm.rope_base";
const KEY_LM_RMS_NORM_EPS: &str = "vokra.voxcpm2.lm.rms_norm_eps";
const KEY_LM_ROPE_SCALING_LONGROPE: &str = "vokra.voxcpm2.lm.rope_scaling.longrope";
const KEY_LM_ROPE_ORIG_MAX_POS: &str =
    "vokra.voxcpm2.lm.rope_scaling.original_max_position_embeddings";
const KEY_LM_SCALE_EMB: &str = "vokra.voxcpm2.lm.scale_emb";
const KEY_LM_DIM_MODEL_BASE: &str = "vokra.voxcpm2.lm.dim_model_base";
const KEY_LM_SCALE_DEPTH: &str = "vokra.voxcpm2.lm.scale_depth";
const KEY_LM_USE_MUP: &str = "vokra.voxcpm2.lm.use_mup";

// Encoder axes — config.json.encoder_config.*
const KEY_ENC_HIDDEN_DIM: &str = "vokra.voxcpm2.encoder.hidden_dim";
const KEY_ENC_FFN_DIM: &str = "vokra.voxcpm2.encoder.ffn_dim";
const KEY_ENC_N_HEAD: &str = "vokra.voxcpm2.encoder.n_head";
const KEY_ENC_N_LAYER: &str = "vokra.voxcpm2.encoder.n_layer";
/// Added 2026-07-30 for the 2B variant. See [`KEY_LM_KV_CHANNELS`].
const KEY_ENC_KV_CHANNELS: &str = "vokra.voxcpm2.encoder.kv_channels";

// DiT axes — config.json.dit_config.*
const KEY_DIT_HIDDEN_DIM: &str = "vokra.voxcpm2.dit.hidden_dim";
const KEY_DIT_FFN_DIM: &str = "vokra.voxcpm2.dit.ffn_dim";
const KEY_DIT_N_HEAD: &str = "vokra.voxcpm2.dit.n_head";
const KEY_DIT_N_LAYER: &str = "vokra.voxcpm2.dit.n_layer";
/// Added 2026-07-30 for the 2B variant. See [`KEY_LM_KV_CHANNELS`].
const KEY_DIT_KV_CHANNELS: &str = "vokra.voxcpm2.dit.kv_channels";
/// Added 2026-07-30 for the 2B variant. Both variants pin `false`
/// (primary sources: 0.5B non-explicit / 2B explicit — see
/// `VoxCpm2DitConfig::mean_mode` docstring for the follow-up rationale).
const KEY_DIT_MEAN_MODE: &str = "vokra.voxcpm2.dit.mean_mode";

// CFM sampler axes — config.json.dit_config.cfm_config.*
const KEY_CFM_SIGMA_MIN: &str = "vokra.voxcpm2.cfm.sigma_min";
const KEY_CFM_SOLVER: &str = "vokra.voxcpm2.cfm.solver";
const KEY_CFM_T_SCHEDULER: &str = "vokra.voxcpm2.cfm.t_scheduler";
const KEY_CFM_INFERENCE_CFG_RATE: &str = "vokra.voxcpm2.cfm.inference_cfg_rate";

// AudioVAE V2 axes — upstream `AudioVAEConfig` defaults (audio_vae_v2.py)
const KEY_VAE_SAMPLE_RATE: &str = "vokra.vae_continuous.sample_rate_hz";
const KEY_VAE_OUT_SAMPLE_RATE: &str = "vokra.vae_continuous.out_sample_rate_hz";
const KEY_VAE_ENCODER_DIM: &str = "vokra.vae_continuous.encoder_dim";
const KEY_VAE_ENCODER_RATES: &str = "vokra.vae_continuous.encoder_rates";
const KEY_VAE_LATENT_DIM: &str = "vokra.vae_continuous.latent_dim";
const KEY_VAE_DECODER_DIM: &str = "vokra.vae_continuous.decoder_dim";
const KEY_VAE_DECODER_RATES: &str = "vokra.vae_continuous.decoder_rates";
const KEY_VAE_DEPTHWISE: &str = "vokra.vae_continuous.depthwise";
const KEY_VAE_USE_NOISE_BLOCK: &str = "vokra.vae_continuous.use_noise_block";
/// Added 2026-07-30 for the 2B variant. 0.5B: `None` (single decoder
/// head, key omitted). 2B: `[20_000, 30_000, 40_000]`
/// (bandwidth-adaptive head — 4 bins). See
/// `ContinuousVaeConfig::voxcpm2_2b` docstring.
const KEY_VAE_SR_BIN_BOUNDARIES: &str = "vokra.vae_continuous.sr_bin_boundaries";

/// Raw upstream text tokenizer, embedded byte-for-byte.  "Tokenizer-free"
/// in VoxCPM describes the acoustic path; the MiniCPM text backbone still
/// consumes the release's tokenizer.json.
pub(crate) const KEY_TOKENIZER_MODEL: &str = "vokra.tokenizer.model";

/// Machine-readable upstream identity in addition to the human-readable
/// `vokra.provenance.source` stamp.
const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";
const KEY_PROVENANCE_UPSTREAM_REVISION: &str = "vokra.provenance.upstream_revision";
const TWO_B_UPSTREAM_HF: &str = "openbmb/VoxCPM2";
const TWO_B_UPSTREAM_REVISION: &str = "bffb3df5a29440629464e5e839f4d214c8714c3d";

// Complete pinned VoxCPM2-2B release after the UV sidecar imports the
// separately shipped audiovae.pth state dict under `audio_vae.`.  These
// counts were read from that immutable HF revision on VAST (2026-08-18).
const TWO_B_MAIN_BF16_TENSORS: usize = 577;
const TWO_B_AUDIOVAE_F32_TENSORS: usize = 311;
const TWO_B_COMPLETE_TENSORS: usize = TWO_B_MAIN_BF16_TENSORS + TWO_B_AUDIOVAE_F32_TENSORS;

/// Model family marker (`config.json.architecture = "voxcpm"`).
/// Distinct from the sibling Qwen family / Llama family etc. Recorded
/// so the runtime can distinguish VoxCPM from other MiniCPM-family
/// releases at telemetry time.
const MODEL_FAMILY: &str = "voxcpm";

/// Which VoxCPM-family release a converter run targets.
///
/// Detected from the safetensors payload itself — see [`detect_variant`].
/// Selects the `struct VariantHparams` factory used to emit the
/// `vokra.voxcpm2.*` + `vokra.vae_continuous.*` chunk groups, and the
/// `vokra.model.name` string. Every downstream consumer (parity harness,
/// runtime binding) uses `vokra.model.name` to route back to the correct
/// runtime factory (`VoxCpm2Config::voxcpm_0_5b` / `voxcpm2_2b`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum VoxCpm2Variant {
    /// Canonical `openbmb/VoxCPM-0.5B` release (LM hidden_dim = 1024,
    /// 24 layers, ffn_dim = 4096, kv_channels = 64 derived, residual LM
    /// depth 6 with RoPE, patch_size 2, SQ latent 256, max_length 4096).
    HalfB,
    /// Canonical `openbmb/VoxCPM2` release — the 2B scale-up (LM
    /// hidden_dim = 2048, 28 layers, ffn_dim = 6144, kv_channels = 128
    /// explicit; encoder + DiT depth 12; residual LM depth 8 with RoPE
    /// **skipped**; patch_size 4; SQ latent 512; max_length 8192;
    /// AudioVAE V2 gains a bandwidth-adaptive decoder head with
    /// `sr_bin_boundaries = [20_000, 30_000, 40_000]`).
    TwoB,
}

impl VoxCpm2Variant {
    /// The `vokra.model.name` string the converter stamps for this
    /// variant. See [`half_b_name`] / [`two_b_name`].
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::HalfB => half_b_name(),
            Self::TwoB => two_b_name(),
        }
    }

    /// Transcribed hparam factory for this variant.
    pub(crate) fn hparams(self) -> VariantHparams {
        match self {
            Self::HalfB => VariantHparams::half_b(),
            Self::TwoB => VariantHparams::two_b(),
        }
    }
}

/// The full set of variant-specific hparams. One instance per variant
/// (see [`VariantHparams::half_b`] / [`VariantHparams::two_b`]).
/// Everything not carried here (RoPE base, RMSNorm eps, VAE sample rates,
/// encoder / decoder rates, etc.) is invariant across variants and lives
/// as module-level constants below.
#[derive(Debug, Clone)]
pub(crate) struct VariantHparams {
    // LM backbone (config.json.lm_config.*)
    pub(crate) lm_hidden_dim: u32,
    pub(crate) lm_n_layer: u32,
    pub(crate) lm_ffn_dim: u32,
    pub(crate) lm_kv_channels: u32,
    // Encoder (config.json.encoder_config.*)
    pub(crate) enc_n_layer: u32,
    pub(crate) enc_kv_channels: u32,
    // DiT (config.json.dit_config.*)
    pub(crate) dit_n_layer: u32,
    pub(crate) dit_kv_channels: u32,
    pub(crate) dit_mean_mode: bool,
    // Top-level (config.json.*)
    pub(crate) residual_lm_n_layer: u32,
    pub(crate) residual_lm_no_rope: bool,
    pub(crate) patch_size: u32,
    pub(crate) scalar_quant_latent_dim: u32,
    pub(crate) max_length: u32,
    // VAE (audio_vae_v2.py `AudioVAEConfig` — 2B adds bandwidth-adaptive)
    pub(crate) vae_sr_bin_boundaries: Option<&'static [u32]>,
}

impl VariantHparams {
    /// Canonical **VoxCPM-0.5B** hparams. Primary source:
    /// `huggingface.co/openbmb/VoxCPM-0.5B/raw/main/config.json` (fetched
    /// 2026-07-24).
    pub(crate) const fn half_b() -> Self {
        Self {
            lm_hidden_dim: 1024,
            lm_n_layer: 24,
            lm_ffn_dim: 4096,
            // Derived: hidden_dim / n_head = 1024 / 16 = 64
            // (non-explicit in 0.5B config.json; the runtime records the
            // derived value — see VoxCpm2LmConfig::kv_channels docstring).
            lm_kv_channels: 64,
            enc_n_layer: 4,
            enc_kv_channels: 64,
            dit_n_layer: 4,
            dit_kv_channels: 64,
            // 0.5B config.json.dit_config is silent on mean_mode; upstream
            // training-side default is false.
            dit_mean_mode: false,
            residual_lm_n_layer: 6,
            // 0.5B: residual acoustic LM keeps RoPE (config additive).
            residual_lm_no_rope: false,
            patch_size: 2,
            scalar_quant_latent_dim: 256,
            max_length: 4096,
            // 0.5B has no bandwidth-adaptive head — single decoder head,
            // full-band output. Key intentionally omitted from GGUF.
            vae_sr_bin_boundaries: None,
        }
    }

    /// Canonical **VoxCPM2-2B** hparams. Primary source:
    /// `huggingface.co/openbmb/VoxCPM2/raw/main/config.json` at pinned
    /// SHA `bffb3df5a29440629464e5e839f4d214c8714c3d` (fetched
    /// 2026-07-28). Field-level rationale is documented on the runtime
    /// factory `VoxCpm2Config::voxcpm2_2b` and its sub-config sibling
    /// factories.
    pub(crate) const fn two_b() -> Self {
        Self {
            lm_hidden_dim: 2048,
            lm_n_layer: 28,
            lm_ffn_dim: 6144,
            // Explicit: config.json.lm_config.kv_channels = 128
            // (= hidden_dim / n_head = 2048 / 16).
            lm_kv_channels: 128,
            enc_n_layer: 12,
            enc_kv_channels: 128,
            dit_n_layer: 12,
            dit_kv_channels: 128,
            // 2B: explicit `mean_mode: false`.
            dit_mean_mode: false,
            residual_lm_n_layer: 8,
            // 2B: residual acoustic LM skips RoPE (config additive; the
            // runtime forward branch that consumes this axis lands in a
            // follow-up wave — see VoxCpm2Config::residual_lm_no_rope
            // docstring).
            residual_lm_no_rope: true,
            patch_size: 4,
            scalar_quant_latent_dim: 512,
            max_length: 8192,
            vae_sr_bin_boundaries: Some(&VAE_SR_BIN_BOUNDARIES_2B),
        }
    }
}

// --- Invariant constants (identical across 0.5B / 2B — primary sources:
// `config.json.lm_config.*` (both variants agree byte-for-byte on these
// axes) + `audio_vae_v2.py` `AudioVAEConfig` defaults) ------------------

// LM backbone (invariant)
const LM_N_HEAD: u32 = 16;
const LM_N_HEAD_KV: u32 = 2;
const LM_VOCAB_SIZE: u32 = 73_448;
const LM_MAX_POSITIONS: u32 = 32_768;
const LM_ROPE_BASE: f32 = 10_000.0;
const LM_RMS_NORM_EPS: f32 = 1e-5;
const LM_ROPE_SCALING_LONGROPE: bool = true;
const LM_ROPE_ORIG_MAX_POS: u32 = 32_768;
const LM_SCALE_EMB: u32 = 12;
const LM_DIM_MODEL_BASE: u32 = 256;
const LM_SCALE_DEPTH: f32 = 1.4;
const LM_USE_MUP: bool = false;

// Encoder (invariant width axes)
const ENC_HIDDEN_DIM: u32 = 1024;
const ENC_FFN_DIM: u32 = 4096;
const ENC_N_HEAD: u32 = 16;

// DiT (invariant width axes)
const DIT_HIDDEN_DIM: u32 = 1024;
const DIT_FFN_DIM: u32 = 4096;
const DIT_N_HEAD: u32 = 16;

// CFM (invariant — 2B primary source pins the same sampler axes)
const CFM_SIGMA_MIN: f32 = 1e-6;
const CFM_SOLVER: &str = "euler";
const CFM_T_SCHEDULER: &str = "log-norm";
const CFM_INFERENCE_CFG_RATE: f32 = 2.0;

// Top-level (invariant)
const FEAT_DIM: u32 = 64;
const SCALAR_QUANT_SCALE: u32 = 9;

// AudioVAE V2 (invariant — 2B primary source pins the same encoder /
// decoder topology; only sr_bin_boundaries differs, carried per-variant
// via `VariantHparams::vae_sr_bin_boundaries`)
const VAE_SAMPLE_RATE: u32 = 16_000;
const VAE_OUT_SAMPLE_RATE: u32 = 48_000;
const VAE_ENCODER_DIM: u32 = 128;
const VAE_ENCODER_RATES: [u32; 4] = [2, 5, 8, 8];
const VAE_LATENT_DIM: u32 = 64;
const VAE_DECODER_DIM: u32 = 2048;
const VAE_DECODER_RATES: [u32; 6] = [8, 6, 5, 2, 2, 2];
const VAE_DEPTHWISE: bool = true;
const VAE_USE_NOISE_BLOCK: bool = false;

/// 2B-only bandwidth-adaptive decoder-head boundaries. Primary source:
/// `openbmb/VoxCPM2/raw/main/config.json`. Four bins covering
/// `(0, 20k] / (20k, 30k] / (30k, 40k] / (40k+ kHz]`.
const VAE_SR_BIN_BOUNDARIES_2B: [u32; 3] = [20_000, 30_000, 40_000];

/// Discriminating LM-backbone token-embedding tensor name. Both
/// variants ship a single-file safetensors whose LM embedding lives at
/// `base_lm.embed_tokens.weight` with shape `[vocab_size=73_448,
/// hidden_dim]`.
const LM_EMBED_TENSOR: &str = "base_lm.embed_tokens.weight";

/// Detects the variant from the safetensors payload.
///
/// The discriminator is the LM backbone hidden dim, read from the last
/// axis of [`LM_EMBED_TENSOR`]'s shape. `1024` → [`VoxCpm2Variant::HalfB`],
/// `2048` → [`VoxCpm2Variant::TwoB`]. Any other value is a loud error
/// (FR-EX-08 — never a silent default to 0.5B).
///
/// A safetensors file with no [`LM_EMBED_TENSOR`] tensor is also
/// rejected loudly. That state signals either (a) a corrupt / stripped
/// input (a build-time synthetic that forgot the LM half) or (b) a
/// sharded release whose first shard lacks the LM half. The workflow
/// (`.github/workflows/parity-tts-continuous-vae-real.yml`) already
/// clean-skips on the shard case; if a caller reaches this converter
/// with a sharded input that lacks the LM half, we refuse rather than
/// invent a variant.
fn detect_variant(st: &SafetensorsFile) -> Result<VoxCpm2Variant, ConvertError> {
    let embed = st.tensor_info(LM_EMBED_TENSOR).ok_or_else(|| {
        ConvertError::Parse(format!(
            "voxcpm2: could not detect variant — safetensors payload has no `{LM_EMBED_TENSOR}` \
             tensor. Both openbmb/VoxCPM-0.5B and openbmb/VoxCPM2 (2B) ship this tensor as \
             their LM token-embedding matrix (shape [{LM_VOCAB_SIZE}, hidden_dim]). This \
             conversion refuses to guess (FR-EX-08 — no silent default to 0.5B) because a \
             wrong variant would silently mis-shape every downstream `vokra.voxcpm2.*` chunk. \
             If this is a sharded release, feed the shard that contains the LM half."
        ))
    })?;
    if embed.shape.len() != 2 {
        return Err(ConvertError::Parse(format!(
            "voxcpm2: `{LM_EMBED_TENSOR}` must have rank 2 (`[vocab_size, hidden_dim]`); \
             got shape {:?}. Cannot proceed (FR-EX-08).",
            embed.shape,
        )));
    }
    if embed.shape[0] != u64::from(LM_VOCAB_SIZE) {
        return Err(ConvertError::Parse(format!(
            "voxcpm2: `{LM_EMBED_TENSOR}` vocab axis is {} — expected {LM_VOCAB_SIZE} \
             (unchanged between 0.5B and 2B). Refusing to proceed (FR-EX-08).",
            embed.shape[0],
        )));
    }
    let hidden = embed.shape[1];
    match hidden {
        1024 => Ok(VoxCpm2Variant::HalfB),
        2048 => Ok(VoxCpm2Variant::TwoB),
        other => Err(ConvertError::Parse(format!(
            "voxcpm2: unrecognised LM hidden_dim {other} from `{LM_EMBED_TENSOR}` (shape \
             [{}, {other}]). Known variants: 1024 (VoxCPM-0.5B) / 2048 (VoxCPM2-2B). \
             Refusing to emit a mis-shaped GGUF (FR-EX-08 — no silent variant fallback).",
            embed.shape[0],
        ))),
    }
}

/// Outcome of a VoxCPM-family conversion.
#[derive(Debug, Default)]
pub(crate) struct VoxCpm2Report {
    /// Float tensors written verbatim (F32 / F16 / BF16 — all three go
    /// through the same byte-copy path since the BF16 pass-through land
    /// 2026-07-25, mirror of `qwen3-tts` / `moshi` / `voxtral`).
    pub(crate) written: usize,
    /// Non-F32 / F16 / BF16 tensors skipped (defensive counter — the
    /// safetensors reader rejects unknown dtypes at parse time; anything
    /// that reaches this arm is a quantized dtype the runtime is not
    /// expected to consume).
    pub(crate) skipped_non_float: usize,
    /// Of the tensors in `written`, how many were BF16 (subset counter).
    /// Emits GGUF type 30 verbatim; runtime widens BF16 → f32 losslessly
    /// via the single choke point `crates/vokra-core/src/gguf/quant/mod.rs
    /// decode_bf16` (BF16 = top 16 bits of an f32 — `bits << 16` is exact).
    pub(crate) bf16_passthrough: usize,
    /// Which variant the converter detected from the payload. Recorded
    /// so callers (and unit tests) can pin the dispatch without
    /// re-parsing the header.
    pub(crate) variant: Option<VoxCpm2Variant>,
    /// Number of float tensors under the separately prepared `audio_vae.`
    /// namespace.  A complete pinned 2B artifact has exactly 311.
    pub(crate) audio_vae_tensors: usize,
    /// Whether the upstream text tokenizer JSON was embedded.
    pub(crate) tokenizer_embedded: bool,
    /// Operator-facing diagnostics (never fail the conversion — the
    /// runtime is the authoritative gate, FR-EX-08).
    pub(crate) notes: Vec<String>,
}

/// Converts a VoxCPM-family safetensors buffer into a populated GGUF
/// builder.
///
/// Variant selection is driven by the payload — see [`detect_variant`].
/// Every F32 / F16 / BF16 tensor passes through under its upstream name;
/// the `vokra.voxcpm2.*` + `vokra.vae_continuous.*` chunk groups are
/// written from the variant's transcribed constants; provenance stamps
/// mark the weight as `Permissive` (apache-2.0 — end-to-end).
#[cfg(test)]
pub(crate) fn convert(bytes: Vec<u8>) -> Result<(GgufBuilder, VoxCpm2Report), ConvertError> {
    convert_impl(bytes, None, false)
}

/// Converts while embedding the release text tokenizer, without enabling the
/// pinned-release completeness gate.  Kept for bounded converter tests and
/// legacy 0.5B tooling; official VoxCPM2-2B conversion uses
/// [`convert_release`].
#[cfg(test)]
pub(crate) fn convert_with_tokenizer(
    bytes: Vec<u8>,
    tokenizer_bytes: Option<Vec<u8>>,
) -> Result<(GgufBuilder, VoxCpm2Report), ConvertError> {
    convert_impl(bytes, tokenizer_bytes, false)
}

/// Converts an official release artifact.  The 2B route refuses unless the
/// UV preparer supplied all 577 main + 311 AudioVAE float tensors and a
/// non-empty tokenizer.  This prevents the old success-shaped conversion of
/// `model.safetensors` alone.
pub(crate) fn convert_release(
    bytes: Vec<u8>,
    tokenizer_bytes: Option<Vec<u8>>,
) -> Result<(GgufBuilder, VoxCpm2Report), ConvertError> {
    convert_impl(bytes, tokenizer_bytes, true)
}

fn convert_impl(
    bytes: Vec<u8>,
    tokenizer_bytes: Option<Vec<u8>>,
    require_complete_release: bool,
) -> Result<(GgufBuilder, VoxCpm2Report), ConvertError> {
    let st = SafetensorsFile::parse(bytes)?;
    let variant = detect_variant(&st)?;
    if require_complete_release && variant == VoxCpm2Variant::TwoB {
        validate_complete_two_b(&st)?;
        if tokenizer_bytes.as_ref().map_or(true, Vec::is_empty) {
            return Err(ConvertError::Parse(
                "voxcpm2-2b: complete release conversion requires the pinned tokenizer.json; \
                 `tokenizer-free` refers to the acoustic path, not the MiniCPM text input. \
                 Pass --tokenizer (FR-EX-08 — no success-shaped tokenizer-less artifact)."
                    .to_owned(),
            ));
        }
    }
    let hparams = variant.hparams();

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, variant.name());
    write_hparams(&mut b, &hparams);
    // Self-describing redistribution: the artifact carries its own licence.
    // Both openbmb/VoxCPM-0.5B and openbmb/VoxCPM2 ship apache-2.0
    // end-to-end (LICENSE + HF model card `license: apache-2.0`, fetched
    // 2026-07-24 / 2026-07-28 — CLAUDE.md「ハルシネーション厳禁」).
    let (weight_ref_name, weight_ref_note): (&str, &str) = match variant {
        VoxCpm2Variant::HalfB => (
            variant.name(),
            "openbmb/VoxCPM-0.5B (apache-2.0 end-to-end)",
        ),
        VoxCpm2Variant::TwoB => (
            variant.name(),
            "openbmb/VoxCPM2@bffb3df5a29440629464e5e839f4d214c8714c3d \
             (model.safetensors + audiovae.pth, apache-2.0 end-to-end)",
        ),
    };
    vokra_core::stamp_provenance(
        &mut b,
        LicenseClass::Permissive,
        "apache-2.0",
        Some(weight_ref_name),
        Some(weight_ref_note),
    );
    match variant {
        VoxCpm2Variant::HalfB => {
            b.add_string(KEY_PROVENANCE_UPSTREAM_HF, "openbmb/VoxCPM-0.5B");
        }
        VoxCpm2Variant::TwoB => {
            b.add_string(KEY_PROVENANCE_UPSTREAM_HF, TWO_B_UPSTREAM_HF);
            b.add_string(KEY_PROVENANCE_UPSTREAM_REVISION, TWO_B_UPSTREAM_REVISION);
        }
    }

    let tokenizer_embedded = match tokenizer_bytes {
        Some(tokenizer) if !tokenizer.is_empty() => {
            b.add_metadata(
                KEY_TOKENIZER_MODEL,
                GgufMetadataValue::Array(GgufArray {
                    element_type: GgufValueType::U8,
                    values: tokenizer.into_iter().map(GgufMetadataValue::U8).collect(),
                }),
            );
            true
        }
        _ => false,
    };

    let mut report = VoxCpm2Report {
        variant: Some(variant),
        tokenizer_embedded,
        ..Default::default()
    };
    for t in st.tensors() {
        match t.dtype {
            // BF16 pass-through added 2026-07-25 (mirror of qwen3-tts +
            // moshi + voxtral): upstream VoxCPM releases ship
            // `dtype: bfloat16` so the release checkpoint hits this arm.
            // Emit as GGUF type 30 verbatim; runtime widens on load via
            // `decode_bf16` (exact, `bits << 16`).
            GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 => {
                b.add_tensor(
                    &t.name,
                    t.dtype,
                    t.shape.clone(),
                    st.tensor_bytes(t).to_vec(),
                )?;
                report.written += 1;
                if t.dtype == GgmlType::BF16 {
                    report.bf16_passthrough += 1;
                }
                if t.name.starts_with("audio_vae.") {
                    report.audio_vae_tensors += 1;
                }
            }
            _ => {
                report.skipped_non_float += 1;
            }
        }
    }
    if report.written == 0 {
        report.notes.push(
            "no float tensors passed through — this GGUF is metadata-only and \
             the runtime will refuse to bind any weights (FR-EX-08). Both \
             upstream releases (VoxCPM-0.5B, VoxCPM2-2B) ship BF16 \
             (config.json `dtype: bfloat16`); the BF16 pass-through path is \
             now wired (2026-07-25), so this state is only reachable when the \
             release contains no F32 / F16 / BF16 float tensors at all."
                .into(),
        );
    }
    if !report.tokenizer_embedded {
        report.notes.push(
            "no tokenizer supplied — `vokra.tokenizer.model` was not embedded; \
             the text path is incomplete even though VoxCPM's acoustic path is \
             tokenizer-free"
                .into(),
        );
    }
    Ok((b, report))
}

fn validate_complete_two_b(st: &SafetensorsFile) -> Result<(), ConvertError> {
    let tensors = st.tensors();
    let bf16 = tensors
        .iter()
        .filter(|tensor| tensor.dtype == GgmlType::BF16)
        .count();
    let f32_count = tensors
        .iter()
        .filter(|tensor| tensor.dtype == GgmlType::F32)
        .count();
    let audio_vae = tensors
        .iter()
        .filter(|tensor| tensor.name.starts_with("audio_vae."))
        .count();
    if tensors.len() != TWO_B_COMPLETE_TENSORS
        || bf16 != TWO_B_MAIN_BF16_TENSORS
        || f32_count != TWO_B_AUDIOVAE_F32_TENSORS
        || audio_vae != TWO_B_AUDIOVAE_F32_TENSORS
    {
        return Err(ConvertError::Parse(format!(
            "voxcpm2-2b: incomplete pinned release checkpoint: got total={} BF16={} F32={} \
             audio_vae.*={}; expected total={TWO_B_COMPLETE_TENSORS} BF16={TWO_B_MAIN_BF16_TENSORS} \
             F32={TWO_B_AUDIOVAE_F32_TENSORS} audio_vae.*={TWO_B_AUDIOVAE_F32_TENSORS}. \
             The upstream audiovae.pth must be imported with \
             tools/parity/voxcpm2_prepare_checkpoint.py before conversion (FR-EX-08).",
            tensors.len(),
            bf16,
            f32_count,
            audio_vae,
        )));
    }
    for (name, shape) in [
        ("audio_vae.encoder.fc_mu.weight_g", &[64, 1, 1][..]),
        ("audio_vae.encoder.fc_logvar.weight_g", &[64, 1, 1][..]),
        ("audio_vae.decoder.model.0.bias", &[64][..]),
        (
            "audio_vae.decoder.sr_cond_model.7.bias_embed.weight",
            &[4, 64][..],
        ),
    ] {
        let tensor = st.tensor_info(name).ok_or_else(|| {
            ConvertError::Parse(format!(
                "voxcpm2-2b: complete checkpoint missing AudioVAE sentinel `{name}`"
            ))
        })?;
        if tensor.dtype != GgmlType::F32 || tensor.shape.as_slice() != shape {
            return Err(ConvertError::Parse(format!(
                "voxcpm2-2b: AudioVAE sentinel `{name}` expected F32 shape {shape:?}, \
                 got {:?} shape {:?}",
                tensor.dtype, tensor.shape,
            )));
        }
    }
    Ok(())
}

/// Writes the `vokra.voxcpm2.*` + `vokra.vae_continuous.*` chunk groups
/// from the variant's transcribed constants + the invariant module-level
/// constants (primary sources: `config.json` per variant and
/// `audio_vae_v2.py` `AudioVAEConfig` defaults).
fn write_hparams(b: &mut GgufBuilder, hp: &VariantHparams) {
    // Top-level
    b.add_string(KEY_MODEL_FAMILY, MODEL_FAMILY);
    b.add_u32(KEY_FEAT_DIM, FEAT_DIM);
    b.add_u32(KEY_PATCH_SIZE, hp.patch_size);
    b.add_u32(KEY_RESIDUAL_LM_N_LAYER, hp.residual_lm_n_layer);
    b.add_bool(KEY_RESIDUAL_LM_NO_ROPE, hp.residual_lm_no_rope);
    b.add_u32(KEY_SCALAR_QUANT_LATENT_DIM, hp.scalar_quant_latent_dim);
    b.add_u32(KEY_SCALAR_QUANT_SCALE, SCALAR_QUANT_SCALE);
    b.add_u32(KEY_MAX_LENGTH, hp.max_length);

    // LM backbone
    b.add_u32(KEY_LM_HIDDEN_DIM, hp.lm_hidden_dim);
    b.add_u32(KEY_LM_N_LAYER, hp.lm_n_layer);
    b.add_u32(KEY_LM_N_HEAD, LM_N_HEAD);
    b.add_u32(KEY_LM_N_HEAD_KV, LM_N_HEAD_KV);
    b.add_u32(KEY_LM_KV_CHANNELS, hp.lm_kv_channels);
    b.add_u32(KEY_LM_FFN_DIM, hp.lm_ffn_dim);
    b.add_u32(KEY_LM_VOCAB_SIZE, LM_VOCAB_SIZE);
    b.add_u32(KEY_LM_MAX_POSITIONS, LM_MAX_POSITIONS);
    b.add_f32(KEY_LM_ROPE_BASE, LM_ROPE_BASE);
    b.add_f32(KEY_LM_RMS_NORM_EPS, LM_RMS_NORM_EPS);
    b.add_bool(KEY_LM_ROPE_SCALING_LONGROPE, LM_ROPE_SCALING_LONGROPE);
    b.add_u32(KEY_LM_ROPE_ORIG_MAX_POS, LM_ROPE_ORIG_MAX_POS);
    b.add_u32(KEY_LM_SCALE_EMB, LM_SCALE_EMB);
    b.add_u32(KEY_LM_DIM_MODEL_BASE, LM_DIM_MODEL_BASE);
    b.add_f32(KEY_LM_SCALE_DEPTH, LM_SCALE_DEPTH);
    b.add_bool(KEY_LM_USE_MUP, LM_USE_MUP);

    // Encoder
    b.add_u32(KEY_ENC_HIDDEN_DIM, ENC_HIDDEN_DIM);
    b.add_u32(KEY_ENC_FFN_DIM, ENC_FFN_DIM);
    b.add_u32(KEY_ENC_N_HEAD, ENC_N_HEAD);
    b.add_u32(KEY_ENC_N_LAYER, hp.enc_n_layer);
    b.add_u32(KEY_ENC_KV_CHANNELS, hp.enc_kv_channels);

    // DiT
    b.add_u32(KEY_DIT_HIDDEN_DIM, DIT_HIDDEN_DIM);
    b.add_u32(KEY_DIT_FFN_DIM, DIT_FFN_DIM);
    b.add_u32(KEY_DIT_N_HEAD, DIT_N_HEAD);
    b.add_u32(KEY_DIT_N_LAYER, hp.dit_n_layer);
    b.add_u32(KEY_DIT_KV_CHANNELS, hp.dit_kv_channels);
    b.add_bool(KEY_DIT_MEAN_MODE, hp.dit_mean_mode);

    // CFM sampler
    b.add_f32(KEY_CFM_SIGMA_MIN, CFM_SIGMA_MIN);
    b.add_string(KEY_CFM_SOLVER, CFM_SOLVER);
    b.add_string(KEY_CFM_T_SCHEDULER, CFM_T_SCHEDULER);
    b.add_f32(KEY_CFM_INFERENCE_CFG_RATE, CFM_INFERENCE_CFG_RATE);

    // AudioVAE V2
    b.add_u32(KEY_VAE_SAMPLE_RATE, VAE_SAMPLE_RATE);
    b.add_u32(KEY_VAE_OUT_SAMPLE_RATE, VAE_OUT_SAMPLE_RATE);
    b.add_u32(KEY_VAE_ENCODER_DIM, VAE_ENCODER_DIM);
    b.add_metadata(
        KEY_VAE_ENCODER_RATES,
        GgufMetadataValue::Array(GgufArray {
            element_type: GgufValueType::U32,
            values: VAE_ENCODER_RATES
                .iter()
                .map(|&r| GgufMetadataValue::U32(r))
                .collect(),
        }),
    );
    b.add_u32(KEY_VAE_LATENT_DIM, VAE_LATENT_DIM);
    b.add_u32(KEY_VAE_DECODER_DIM, VAE_DECODER_DIM);
    b.add_metadata(
        KEY_VAE_DECODER_RATES,
        GgufMetadataValue::Array(GgufArray {
            element_type: GgufValueType::U32,
            values: VAE_DECODER_RATES
                .iter()
                .map(|&r| GgufMetadataValue::U32(r))
                .collect(),
        }),
    );
    b.add_bool(KEY_VAE_DEPTHWISE, VAE_DEPTHWISE);
    b.add_bool(KEY_VAE_USE_NOISE_BLOCK, VAE_USE_NOISE_BLOCK);
    // Bandwidth-adaptive head — 2B only. 0.5B intentionally omits the key
    // (single decoder head, full-band output — a downstream consumer that
    // reads it as `Option<Vec<u32>>` sees `None` from the missing key).
    if let Some(boundaries) = hp.vae_sr_bin_boundaries {
        b.add_metadata(
            KEY_VAE_SR_BIN_BOUNDARIES,
            GgufMetadataValue::Array(GgufArray {
                element_type: GgufValueType::U32,
                values: boundaries
                    .iter()
                    .map(|&r| GgufMetadataValue::U32(r))
                    .collect(),
            }),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::{GgufFile, GgufMetadataValue};

    /// Builds a minimal-but-well-shaped VoxCPM safetensors buffer.
    ///
    /// The `hidden_dim` axis of `base_lm.embed_tokens.weight` drives
    /// variant detection. `1024` selects the 0.5B path, `2048` selects
    /// the 2B path. The tensor is F32 (compact fixture — pass-through
    /// arm fires once).
    fn safetensors_with_lm_hidden(hidden_dim: u32) -> Vec<u8> {
        // Shape `[vocab_size=73_448, hidden_dim]` is what the primary
        // sources declare. For the fixture we compress to a small width
        // so the byte payload stays a few KB, but keep the vocab axis
        // exact — `detect_variant` cross-checks it.
        //
        // Compressed vocab breaks detection cross-check, so we use the
        // real vocab (73_448) as the outer axis + the discriminating
        // hidden_dim as the inner axis, with a zeroed payload
        // (`73_448 * hidden_dim * 4` bytes). For `hidden_dim=1` we would
        // trip the "unrecognised" arm; the fixture uses the real values
        // (1024 or 2048) to exercise the true detection path.
        //
        // For 1024: 73_448 * 1024 * 4 = 301_154_304 bytes ≈ 287 MiB —
        // too heavy for a unit test. Compress by widening the vocab
        // axis: keep vocab exact (73_448) but only fill 1 element of
        // hidden_dim to keep the byte payload tiny? No — safetensors
        // shape must match declared byte count.
        //
        // Solution: declare an OUTER vocab axis of 73_448 and INNER
        // hidden_dim as declared, but produce ONLY the bytes required.
        // safetensors requires `data_offsets` bytes to equal
        // `product(shape) * dtype_size`. So we must produce the full
        // 73_448 * hidden_dim * 4 bytes. That is too much for a unit test.
        //
        // Alternate: teach `detect_variant` to skip the vocab-axis check
        // when the discriminant is unambiguous. The vocab-axis check is
        // a defense against a malformed input; we can move it to a
        // warning note. But the primary source pins vocab_size, so a
        // silent skip would hide legitimate corruption.
        //
        // Compromise: unit test uses a compact vocab (1) plus asserts
        // that the vocab-cross-check fires for non-73_448 payloads in a
        // separate `detect_variant_rejects_wrong_vocab` test. The main
        // round-trip test then targets the shape-agreement leg only.
        // For the round-trip fixture we build a payload with an
        // ambiguous vocab axis (1) and assert that `detect_variant`
        // FAILS on it — validating the guard.
        //
        // For the byte-round-trip tests we hand-build a fixture that
        // asks the detection to pass by writing the compact form and
        // relaxing the vocab-cross-check for a `#[cfg(test)]`
        // fixture-only escape hatch. But adding an escape hatch to
        // detection breaks FR-EX-08.
        //
        // Final: build the FULL 73_448 * hidden_dim * 4 byte payload
        // for the pass-through round-trip test. For hidden_dim=1024
        // this is ~287 MiB — expensive but only inside a single #[test]
        // marked #[ignore] on default runs? No, we need CI to exercise
        // it. Instead, split the tests: the shape/detection tests use
        // an exact-vocab compact fixture (only 4-byte payload
        // difference between variants is possible without honouring
        // the full byte count). This is the byte-cost fundamental
        // that safetensors imposes.
        //
        // Sanity: run the test on hidden_dim ∈ {1024, 2048} with
        // vocab_size = 73_448 but PRETEND the tensor is `hidden_dim`
        // long inside a zero-copy scheme? safetensors strictly
        // enforces `data_offsets`, so this is not achievable without
        // building the real bytes.
        //
        // Practical answer: unit test uses vocab_size = 1 (fixture) +
        // wires a separate unit test that asserts the vocab check
        // catches mismatches on the real fixture. The primary-source
        // gate is still active in production: any real VoxCPM
        // safetensors ships vocab_size = 73_448.
        let vocab = 1u64; // compact fixture — see rationale above
        let byte_len = (vocab as usize) * (hidden_dim as usize) * 4; // f32
        let header = format!(
            r#"{{"{LM_EMBED_TENSOR}":{{"dtype":"F32","shape":[{vocab},{hidden_dim}],"data_offsets":[0,{byte_len}]}}}}"#
        );
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.resize(out.len() + byte_len, 0u8);
        out
    }

    /// A fixture that carries the real vocab (73_448) plus a hidden dim.
    /// Used by the variant-detection tests only (not by the round-trip
    /// tests, since building the full 287 MiB payload for hidden_dim =
    /// 1024 is prohibitively expensive in unit-test CPU time).
    fn safetensors_full_vocab_lm_embed(hidden_dim: u32) -> Vec<u8> {
        let vocab = u64::from(LM_VOCAB_SIZE);
        let byte_len = (vocab as usize) * (hidden_dim as usize) * 4; // f32
        let header = format!(
            r#"{{"{LM_EMBED_TENSOR}":{{"dtype":"F32","shape":[{vocab},{hidden_dim}],"data_offsets":[0,{byte_len}]}}}}"#
        );
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.resize(out.len() + byte_len, 0u8);
        out
    }

    fn safetensors_no_embed() -> Vec<u8> {
        let header = r#"{"some.other.tensor":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}}"#;
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.resize(out.len() + 24, 0u8);
        out
    }

    /// A fixture that carries the real full-vocab LM embedding at F32 (for
    /// variant detection) plus one auxiliary tensor at a caller-chosen
    /// dtype (for exercising the pass-through arm). Second tensor's shape
    /// is `[2, 3]` = 6 elements, byte size = 6 × dtype-size.
    ///
    /// Layout in the safetensors payload: LM embed first (offsets
    /// `[0, embed_bytes]`), aux tensor second (offsets `[embed_bytes,
    /// embed_bytes + aux_bytes]`).
    fn safetensors_lm_plus_aux(
        hidden_dim: u32,
        aux_dtype: &str,
        aux_bytes_per_elem: usize,
    ) -> Vec<u8> {
        let vocab = u64::from(LM_VOCAB_SIZE);
        let embed_bytes = (vocab as usize) * (hidden_dim as usize) * 4;
        let aux_elems = 6usize; // 2 × 3
        let aux_bytes = aux_elems * aux_bytes_per_elem;
        let header = format!(
            r#"{{"{LM_EMBED_TENSOR}":{{"dtype":"F32","shape":[{vocab},{hidden_dim}],"data_offsets":[0,{embed_bytes}]}},"base_lm.some_other.weight":{{"dtype":"{aux_dtype}","shape":[2,3],"data_offsets":[{embed_bytes},{end}]}}}}"#,
            end = embed_bytes + aux_bytes,
        );
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.resize(out.len() + embed_bytes + aux_bytes, 0u8);
        out
    }

    fn safetensors_empty_header() -> Vec<u8> {
        let header = r#"{}"#;
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out
    }

    fn get_u32(file: &GgufFile, key: &str) -> u32 {
        match file.get(key) {
            Some(GgufMetadataValue::U32(v)) => *v,
            other => panic!("{key}: unexpected {other:?}"),
        }
    }

    fn get_f32(file: &GgufFile, key: &str) -> f32 {
        match file.get(key) {
            Some(GgufMetadataValue::F32(v)) => *v,
            other => panic!("{key}: unexpected {other:?}"),
        }
    }

    fn get_bool(file: &GgufFile, key: &str) -> bool {
        match file.get(key) {
            Some(GgufMetadataValue::Bool(v)) => *v,
            other => panic!("{key}: unexpected {other:?}"),
        }
    }

    fn get_string(file: &GgufFile, key: &str) -> String {
        file.get(key)
            .and_then(|v| v.as_str())
            .unwrap_or_else(|| panic!("{key}: missing"))
            .to_owned()
    }

    fn get_u32_array(file: &GgufFile, key: &str) -> Vec<u32> {
        match file.get(key) {
            Some(GgufMetadataValue::Array(a)) => a
                .values
                .iter()
                .map(|v| match v {
                    GgufMetadataValue::U32(x) => *x,
                    other => panic!("{key}: unexpected array element {other:?}"),
                })
                .collect(),
            other => panic!("{key}: unexpected {other:?}"),
        }
    }

    fn get_u8_array(file: &GgufFile, key: &str) -> Vec<u8> {
        match file.get(key) {
            Some(GgufMetadataValue::Array(a)) => {
                assert_eq!(
                    a.element_type,
                    GgufValueType::U8,
                    "{key}: wrong element type"
                );
                a.values
                    .iter()
                    .map(|v| match v {
                        GgufMetadataValue::U8(x) => *x,
                        other => panic!("{key}: unexpected array element {other:?}"),
                    })
                    .collect()
            }
            other => panic!("{key}: unexpected {other:?}"),
        }
    }

    #[test]
    fn arch_string_matches_runtime_constant() {
        // The two crates only share `vokra-core`, so this constant is the
        // sole handshake with `vokra-models::voxcpm2::EXPECTED_ARCH`.
        assert_eq!(ARCH, "voxcpm2");
    }

    #[test]
    fn arch_is_distinct_from_every_sibling_family() {
        // VoxCPM's terminal decoding hop is neither vocoder-LM
        // (HiFTChain) nor codec-LM (any RVQ / FSQ codec) — silently
        // sharing an arch tag with a sibling would misroute the runtime
        // dispatch.
        assert_ne!(ARCH, "cosyvoice2");
        assert_ne!(ARCH, "cosyvoice3");
        assert_ne!(ARCH, "qwen3_tts");
        assert_ne!(ARCH, "chatterbox");
        assert_ne!(ARCH, "chatterbox_turbo");
        assert_ne!(ARCH, "chatterbox_nano");
        assert_ne!(ARCH, "dia");
        assert_ne!(ARCH, "zonos");
        assert_ne!(ARCH, "csm");
        assert_ne!(ARCH, "voxtral");
        assert_ne!(ARCH, "kyutai_stt");
        assert_ne!(ARCH, "moshi");
    }

    /// The two variant names carry the arch prefix so the parity harness
    /// can dispatch on the leading `voxcpm2-` substring without needing
    /// a lookup table (renamed 2026-07-30 — `voxcpm-0.5b` → `voxcpm2-0.5b`
    /// to align both variants under the arch-family prefix).
    #[test]
    fn variant_names_carry_arch_prefix() {
        assert_eq!(half_b_name(), "voxcpm2-0.5b");
        assert_eq!(two_b_name(), "voxcpm2-2b");
        assert!(half_b_name().starts_with("voxcpm2-"));
        assert!(two_b_name().starts_with("voxcpm2-"));
        assert_ne!(half_b_name(), two_b_name());
        // Enum dispatch mirrors the module-level helpers.
        assert_eq!(VoxCpm2Variant::HalfB.name(), half_b_name());
        assert_eq!(VoxCpm2Variant::TwoB.name(), two_b_name());
    }

    /// Every LM axis of the 0.5B hparams must equal the primary source.
    #[test]
    fn half_b_hparams_match_primary_source() {
        let hp = VariantHparams::half_b();
        // LM backbone (config.json.lm_config.*)
        assert_eq!(hp.lm_hidden_dim, 1024);
        assert_eq!(hp.lm_n_layer, 24);
        assert_eq!(hp.lm_ffn_dim, 4096);
        assert_eq!(hp.lm_kv_channels, 64);
        // Encoder
        assert_eq!(hp.enc_n_layer, 4);
        assert_eq!(hp.enc_kv_channels, 64);
        // DiT
        assert_eq!(hp.dit_n_layer, 4);
        assert_eq!(hp.dit_kv_channels, 64);
        assert!(!hp.dit_mean_mode);
        // Top-level
        assert_eq!(hp.residual_lm_n_layer, 6);
        assert!(!hp.residual_lm_no_rope);
        assert_eq!(hp.patch_size, 2);
        assert_eq!(hp.scalar_quant_latent_dim, 256);
        assert_eq!(hp.max_length, 4096);
        // VAE bandwidth-adaptive head — 0.5B has none.
        assert!(hp.vae_sr_bin_boundaries.is_none());
    }

    /// Every LM axis of the 2B hparams must equal the primary source
    /// (`openbmb/VoxCPM2/config.json` @ `bffb3df5…` — see module-level
    /// rustdoc).
    #[test]
    fn two_b_hparams_match_primary_source() {
        let hp = VariantHparams::two_b();
        // LM backbone
        assert_eq!(hp.lm_hidden_dim, 2048, "2B hidden_dim ×2 vs 0.5B");
        assert_eq!(hp.lm_n_layer, 28, "2B n_layer +4 vs 0.5B");
        assert_eq!(hp.lm_ffn_dim, 6144, "2B ffn_dim ×1.5 vs 0.5B");
        assert_eq!(hp.lm_kv_channels, 128, "2B explicit 128 (= 2048/16)");
        // Encoder — depth ×3
        assert_eq!(hp.enc_n_layer, 12);
        assert_eq!(hp.enc_kv_channels, 128);
        // DiT — depth ×3
        assert_eq!(hp.dit_n_layer, 12);
        assert_eq!(hp.dit_kv_channels, 128);
        assert!(!hp.dit_mean_mode);
        // Top-level
        assert_eq!(hp.residual_lm_n_layer, 8, "residual LM 6 → 8");
        assert!(hp.residual_lm_no_rope, "2B: residual LM disables RoPE");
        assert_eq!(hp.patch_size, 4, "2B patch_size 2 → 4");
        assert_eq!(hp.scalar_quant_latent_dim, 512, "SQ latent 256 → 512");
        assert_eq!(hp.max_length, 8192, "max_length 4096 → 8192");
        // VAE bandwidth-adaptive head — 2B pins 3 boundaries.
        assert_eq!(
            hp.vae_sr_bin_boundaries,
            Some(&VAE_SR_BIN_BOUNDARIES_2B[..])
        );
    }

    /// Invariant module-level constants — pinned so a silent drift on
    /// the width axes / RoPE / VAE-topology cannot slip past the tests.
    #[test]
    fn invariant_constants_match_primary_source() {
        // LM
        assert_eq!(LM_N_HEAD, 16);
        assert_eq!(LM_N_HEAD_KV, 2);
        assert_eq!(LM_VOCAB_SIZE, 73_448);
        assert_eq!(LM_MAX_POSITIONS, 32_768);
        assert!((LM_ROPE_BASE - 10_000.0).abs() < 1e-3);
        assert!((LM_RMS_NORM_EPS - 1e-5).abs() < 1e-9);
        #[allow(clippy::assertions_on_constants)]
        {
            assert!(LM_ROPE_SCALING_LONGROPE);
            assert!(!LM_USE_MUP);
        }
        assert_eq!(LM_ROPE_ORIG_MAX_POS, 32_768);
        assert_eq!(LM_SCALE_EMB, 12);
        assert_eq!(LM_DIM_MODEL_BASE, 256);
        assert!((LM_SCALE_DEPTH - 1.4).abs() < 1e-5);
        // Encoder + DiT (widths)
        assert_eq!(ENC_HIDDEN_DIM, 1024);
        assert_eq!(ENC_FFN_DIM, 4096);
        assert_eq!(ENC_N_HEAD, 16);
        assert_eq!(DIT_HIDDEN_DIM, 1024);
        assert_eq!(DIT_FFN_DIM, 4096);
        assert_eq!(DIT_N_HEAD, 16);
        // CFM
        assert!((CFM_SIGMA_MIN - 1e-6).abs() < 1e-9);
        assert_eq!(CFM_SOLVER, "euler");
        assert_eq!(CFM_T_SCHEDULER, "log-norm");
        assert!((CFM_INFERENCE_CFG_RATE - 2.0).abs() < 1e-5);
        // Top-level (invariant)
        assert_eq!(FEAT_DIM, 64);
        assert_eq!(SCALAR_QUANT_SCALE, 9);
        // VAE topology
        assert_eq!(VAE_SAMPLE_RATE, 16_000);
        assert_eq!(VAE_OUT_SAMPLE_RATE, 48_000);
        assert_eq!(VAE_ENCODER_DIM, 128);
        assert_eq!(VAE_ENCODER_RATES, [2, 5, 8, 8]);
        assert_eq!(VAE_LATENT_DIM, 64);
        assert_eq!(VAE_DECODER_DIM, 2048);
        assert_eq!(VAE_DECODER_RATES, [8, 6, 5, 2, 2, 2]);
        #[allow(clippy::assertions_on_constants)]
        {
            assert!(VAE_DEPTHWISE);
            assert!(!VAE_USE_NOISE_BLOCK);
        }
        assert_eq!(VAE_SR_BIN_BOUNDARIES_2B, [20_000, 30_000, 40_000]);
        assert_eq!(MODEL_FAMILY, "voxcpm");

        // Compile-time algebra pins.
        const _: () = {
            // GQA well-formedness (both variants share n_head/n_head_kv).
            assert!(LM_N_HEAD % LM_N_HEAD_KV == 0);
            // VAE handshake — LM step feature width MUST equal VAE latent.
            assert!(FEAT_DIM == VAE_LATENT_DIM);
            // Positive shapes.
            assert!(FEAT_DIM > 0);
            assert!(VAE_LATENT_DIM > 0);
        };
    }

    /// Full round-trip for the 0.5B fixture: `detect_variant` picks
    /// [`VoxCpm2Variant::HalfB`], every 0.5B-anchored hparam round-trips
    /// under the `vokra.voxcpm2.*` / `vokra.vae_continuous.*` prefixes,
    /// `vokra.model.name = "voxcpm2-0.5b"`, and the bandwidth-adaptive
    /// key is **absent** (0.5B has none).
    #[test]
    fn round_trip_half_b_variant() {
        let (builder, report) = convert(safetensors_full_vocab_lm_embed(1024)).expect("convert");
        assert_eq!(report.variant, Some(VoxCpm2Variant::HalfB));
        assert_eq!(report.written, 1);
        assert_eq!(report.skipped_non_float, 0);

        let out = builder.to_bytes().expect("serialize");
        let file = GgufFile::parse(out).expect("parse");
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH)
        );
        assert_eq!(
            file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
            Some("voxcpm2-0.5b")
        );
        assert_eq!(get_string(&file, KEY_MODEL_FAMILY), MODEL_FAMILY);

        // Variant-driven axes.
        assert_eq!(get_u32(&file, KEY_LM_HIDDEN_DIM), 1024);
        assert_eq!(get_u32(&file, KEY_LM_N_LAYER), 24);
        assert_eq!(get_u32(&file, KEY_LM_FFN_DIM), 4096);
        assert_eq!(get_u32(&file, KEY_LM_KV_CHANNELS), 64);
        assert_eq!(get_u32(&file, KEY_ENC_N_LAYER), 4);
        assert_eq!(get_u32(&file, KEY_ENC_KV_CHANNELS), 64);
        assert_eq!(get_u32(&file, KEY_DIT_N_LAYER), 4);
        assert_eq!(get_u32(&file, KEY_DIT_KV_CHANNELS), 64);
        assert!(!get_bool(&file, KEY_DIT_MEAN_MODE));
        assert_eq!(get_u32(&file, KEY_RESIDUAL_LM_N_LAYER), 6);
        assert!(!get_bool(&file, KEY_RESIDUAL_LM_NO_ROPE));
        assert_eq!(get_u32(&file, KEY_PATCH_SIZE), 2);
        assert_eq!(get_u32(&file, KEY_SCALAR_QUANT_LATENT_DIM), 256);
        assert_eq!(get_u32(&file, KEY_MAX_LENGTH), 4096);

        // Invariant axes.
        assert_eq!(get_u32(&file, KEY_FEAT_DIM), FEAT_DIM);
        assert_eq!(get_u32(&file, KEY_SCALAR_QUANT_SCALE), SCALAR_QUANT_SCALE);
        assert_eq!(get_u32(&file, KEY_LM_N_HEAD), 16);
        assert_eq!(get_u32(&file, KEY_LM_N_HEAD_KV), 2);
        assert_eq!(get_u32(&file, KEY_LM_VOCAB_SIZE), LM_VOCAB_SIZE);
        assert_eq!(get_u32(&file, KEY_LM_MAX_POSITIONS), LM_MAX_POSITIONS);
        assert!((get_f32(&file, KEY_LM_ROPE_BASE) - LM_ROPE_BASE).abs() < 1e-3);
        assert!((get_f32(&file, KEY_LM_RMS_NORM_EPS) - LM_RMS_NORM_EPS).abs() < 1e-9);
        assert!((get_f32(&file, KEY_LM_SCALE_DEPTH) - LM_SCALE_DEPTH).abs() < 1e-5);
        assert_eq!(
            get_bool(&file, KEY_LM_ROPE_SCALING_LONGROPE),
            LM_ROPE_SCALING_LONGROPE
        );
        assert_eq!(get_bool(&file, KEY_LM_USE_MUP), LM_USE_MUP);
        assert_eq!(
            get_u32(&file, KEY_LM_ROPE_ORIG_MAX_POS),
            LM_ROPE_ORIG_MAX_POS
        );
        assert_eq!(get_u32(&file, KEY_LM_SCALE_EMB), LM_SCALE_EMB);
        assert_eq!(get_u32(&file, KEY_LM_DIM_MODEL_BASE), LM_DIM_MODEL_BASE);
        assert_eq!(get_u32(&file, KEY_ENC_HIDDEN_DIM), ENC_HIDDEN_DIM);
        assert_eq!(get_u32(&file, KEY_ENC_FFN_DIM), ENC_FFN_DIM);
        assert_eq!(get_u32(&file, KEY_ENC_N_HEAD), ENC_N_HEAD);
        assert_eq!(get_u32(&file, KEY_DIT_HIDDEN_DIM), DIT_HIDDEN_DIM);
        assert_eq!(get_u32(&file, KEY_DIT_FFN_DIM), DIT_FFN_DIM);
        assert_eq!(get_u32(&file, KEY_DIT_N_HEAD), DIT_N_HEAD);
        assert!((get_f32(&file, KEY_CFM_SIGMA_MIN) - CFM_SIGMA_MIN).abs() < 1e-9);
        assert_eq!(get_string(&file, KEY_CFM_SOLVER), CFM_SOLVER);
        assert_eq!(get_string(&file, KEY_CFM_T_SCHEDULER), CFM_T_SCHEDULER);
        assert!((get_f32(&file, KEY_CFM_INFERENCE_CFG_RATE) - CFM_INFERENCE_CFG_RATE).abs() < 1e-5);
        // VAE topology
        assert_eq!(get_u32(&file, KEY_VAE_SAMPLE_RATE), VAE_SAMPLE_RATE);
        assert_eq!(get_u32(&file, KEY_VAE_OUT_SAMPLE_RATE), VAE_OUT_SAMPLE_RATE);
        assert_eq!(get_u32(&file, KEY_VAE_ENCODER_DIM), VAE_ENCODER_DIM);
        assert_eq!(get_u32(&file, KEY_VAE_LATENT_DIM), VAE_LATENT_DIM);
        assert_eq!(get_u32(&file, KEY_VAE_DECODER_DIM), VAE_DECODER_DIM);
        assert_eq!(get_bool(&file, KEY_VAE_DEPTHWISE), VAE_DEPTHWISE);
        assert_eq!(
            get_bool(&file, KEY_VAE_USE_NOISE_BLOCK),
            VAE_USE_NOISE_BLOCK
        );

        // The bandwidth-adaptive key MUST be absent for the 0.5B variant.
        assert!(
            file.get(KEY_VAE_SR_BIN_BOUNDARIES).is_none(),
            "0.5B must omit vokra.vae_continuous.sr_bin_boundaries"
        );

        // Provenance: apache-2.0 permissive (end-to-end).
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_MODEL_ID)
                .and_then(|v| v.as_str()),
            Some("voxcpm2-0.5b")
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some("apache-2.0")
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Permissive.as_str())
        );
    }

    /// Full round-trip for the 2B fixture: `detect_variant` picks
    /// [`VoxCpm2Variant::TwoB`], every 2B-anchored hparam round-trips
    /// under the `vokra.voxcpm2.*` / `vokra.vae_continuous.*` prefixes,
    /// `vokra.model.name = "voxcpm2-2b"`, and the bandwidth-adaptive
    /// key IS present with the primary-source-pinned boundaries.
    #[test]
    fn round_trip_two_b_variant() {
        let tokenizer = br#"{"model":{"type":"BPE"}}"#.to_vec();
        let (builder, report) = convert_with_tokenizer(
            safetensors_full_vocab_lm_embed(2048),
            Some(tokenizer.clone()),
        )
        .expect("convert");
        assert_eq!(report.variant, Some(VoxCpm2Variant::TwoB));
        assert_eq!(report.written, 1);
        assert_eq!(report.skipped_non_float, 0);
        assert!(report.tokenizer_embedded);

        let out = builder.to_bytes().expect("serialize");
        let file = GgufFile::parse(out).expect("parse");
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH),
            "2B and 0.5B share the same arch tag"
        );
        assert_eq!(
            file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
            Some("voxcpm2-2b")
        );

        // Variant-driven axes (2B).
        assert_eq!(get_u32(&file, KEY_LM_HIDDEN_DIM), 2048);
        assert_eq!(get_u32(&file, KEY_LM_N_LAYER), 28);
        assert_eq!(get_u32(&file, KEY_LM_FFN_DIM), 6144);
        assert_eq!(get_u32(&file, KEY_LM_KV_CHANNELS), 128);
        assert_eq!(get_u32(&file, KEY_ENC_N_LAYER), 12);
        assert_eq!(get_u32(&file, KEY_ENC_KV_CHANNELS), 128);
        assert_eq!(get_u32(&file, KEY_DIT_N_LAYER), 12);
        assert_eq!(get_u32(&file, KEY_DIT_KV_CHANNELS), 128);
        assert!(!get_bool(&file, KEY_DIT_MEAN_MODE));
        assert_eq!(get_u32(&file, KEY_RESIDUAL_LM_N_LAYER), 8);
        assert!(get_bool(&file, KEY_RESIDUAL_LM_NO_ROPE));
        assert_eq!(get_u32(&file, KEY_PATCH_SIZE), 4);
        assert_eq!(get_u32(&file, KEY_SCALAR_QUANT_LATENT_DIM), 512);
        assert_eq!(get_u32(&file, KEY_MAX_LENGTH), 8192);

        // Invariant axes match 0.5B verbatim (spot-check a few).
        assert_eq!(get_u32(&file, KEY_FEAT_DIM), FEAT_DIM);
        assert_eq!(get_u32(&file, KEY_LM_VOCAB_SIZE), LM_VOCAB_SIZE);

        // Bandwidth-adaptive head — MUST be present for 2B, exact
        // boundaries from the primary source.
        assert_eq!(
            get_u32_array(&file, KEY_VAE_SR_BIN_BOUNDARIES),
            vec![20_000, 30_000, 40_000]
        );

        // Provenance: apache-2.0 permissive (end-to-end).
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_MODEL_ID)
                .and_then(|v| v.as_str()),
            Some("voxcpm2-2b")
        );
        assert_eq!(
            file.get(KEY_PROVENANCE_UPSTREAM_HF)
                .and_then(|v| v.as_str()),
            Some(TWO_B_UPSTREAM_HF)
        );
        assert_eq!(
            file.get(KEY_PROVENANCE_UPSTREAM_REVISION)
                .and_then(|v| v.as_str()),
            Some(TWO_B_UPSTREAM_REVISION)
        );
        assert_eq!(get_u8_array(&file, KEY_TOKENIZER_MODEL), tokenizer);
    }

    /// The public release path must not turn the upstream main checkpoint
    /// alone into a success-shaped 2B GGUF. The AudioVAE is a separately
    /// shipped required weight file in the pinned release.
    #[test]
    fn release_rejects_two_b_main_weights_without_audiovae() {
        let err = convert_release(
            safetensors_full_vocab_lm_embed(2048),
            Some(br#"{"model":{}}"#.to_vec()),
        )
        .expect_err("main-only 2B release must fail");
        match err {
            ConvertError::Parse(message) => {
                assert!(message.contains("incomplete pinned release"), "{message}");
                assert!(message.contains("audiovae.pth"), "{message}");
                assert!(message.contains("expected total=888"), "{message}");
            }
            other => panic!("expected Parse, got {other:?}"),
        }
    }

    /// Detection must refuse loudly (never a silent default) when the
    /// LM embedding tensor is missing.
    #[test]
    fn detect_rejects_missing_lm_embed() {
        let err = convert(safetensors_no_embed()).expect_err("missing LM embed");
        match err {
            ConvertError::Parse(m) => {
                assert!(m.contains(LM_EMBED_TENSOR), "message names tensor: {m}");
                assert!(m.contains("FR-EX-08"), "message cites red-line: {m}");
            }
            other => panic!("expected Parse, got {other:?}"),
        }
    }

    /// Detection must refuse loudly when the LM embedding shape has the
    /// wrong rank.
    #[test]
    fn detect_rejects_wrong_rank_lm_embed() {
        let header = format!(
            r#"{{"{LM_EMBED_TENSOR}":{{"dtype":"F32","shape":[8],"data_offsets":[0,32]}}}}"#
        );
        let mut buf = Vec::new();
        buf.extend_from_slice(&(header.len() as u64).to_le_bytes());
        buf.extend_from_slice(header.as_bytes());
        buf.resize(buf.len() + 32, 0u8);
        let err = convert(buf).expect_err("wrong rank");
        match err {
            ConvertError::Parse(m) => assert!(
                m.contains("rank 2") || m.contains("`[vocab_size, hidden_dim]`"),
                "rank-2 rejection: {m}"
            ),
            other => panic!("expected Parse, got {other:?}"),
        }
    }

    /// Detection must refuse loudly when the vocab axis mismatches.
    #[test]
    fn detect_rejects_wrong_vocab_axis() {
        let err = convert(safetensors_with_lm_hidden(1024)).expect_err("wrong vocab");
        match err {
            ConvertError::Parse(m) => assert!(
                m.contains(&LM_VOCAB_SIZE.to_string()),
                "cite vocab_size: {m}"
            ),
            other => panic!("expected Parse, got {other:?}"),
        }
    }

    /// Detection must refuse loudly when the hidden_dim is neither 1024
    /// nor 2048.
    #[test]
    fn detect_rejects_unknown_hidden_dim() {
        // Real vocab, unknown hidden.
        let err = convert(safetensors_full_vocab_lm_embed(1536)).expect_err("unknown hidden");
        match err {
            ConvertError::Parse(m) => {
                assert!(m.contains("1536"), "cite the bad value: {m}");
                assert!(m.contains("1024"), "cite 0.5B alternative: {m}");
                assert!(m.contains("2048"), "cite 2B alternative: {m}");
            }
            other => panic!("expected Parse, got {other:?}"),
        }
    }

    /// Empty header — no tensors at all → detect refuses loudly.
    #[test]
    fn empty_header_rejected() {
        let err = convert(safetensors_empty_header()).expect_err("no tensors");
        // Either the safetensors parser rejects the empty header, or
        // detect_variant fires on the missing LM embed. Both are loud
        // Parse errors.
        assert!(matches!(err, ConvertError::Parse(_)), "loud parse: {err:?}");
    }

    /// Round-trip 2B via converter, then check that the shared VAE seam
    /// [`vokra_ops::vae_continuous::ContinuousVaeConfig::voxcpm2_2b`]
    /// carries the same bandwidth-adaptive boundaries the converter
    /// stamps into `vokra.vae_continuous.sr_bin_boundaries`. This is
    /// the only cross-crate handshake exercisable from *within*
    /// `vokra-convert` — the runtime-side `VoxCpm2Config` handshake
    /// lives in `crates/vokra-models/tests/parity_tts_continuous_vae.rs`
    /// (the parity harness) because `vokra-models` is a downstream
    /// dev-dep of this crate, so importing it here would cycle.
    #[test]
    fn two_b_vae_bandwidth_boundaries_agree_with_shared_seam() {
        let (builder, _) = convert(safetensors_full_vocab_lm_embed(2048)).expect("convert");
        let file = GgufFile::parse(builder.to_bytes().expect("bytes")).expect("parse");
        let vae = vokra_ops::vae_continuous::ContinuousVaeConfig::voxcpm2_2b();
        let boundaries = vae.sr_bin_boundaries.expect("2B: sr_bin_boundaries set");
        assert_eq!(get_u32_array(&file, KEY_VAE_SR_BIN_BOUNDARIES), boundaries);
    }

    /// Pins the F16 leg of the `GgmlType::F32 | GgmlType::F16 |
    /// GgmlType::BF16` union match arm — an F16 aux tensor alongside the
    /// F32 LM embed must reach the pass-through arm and land in
    /// `written` (not `skipped_non_float`), and its dtype must be
    /// preserved in the GGUF (no convert-time widening).
    #[test]
    fn f16_aux_tensor_passes_through_verbatim() {
        // 0.5B route: LM embed at F32 (real vocab) + 6-element F16 aux.
        let (builder, report) = convert(safetensors_lm_plus_aux(1024, "F16", 2)).expect("convert");
        assert_eq!(report.variant, Some(VoxCpm2Variant::HalfB));
        assert_eq!(
            report.written, 2,
            "both LM embed (F32) and aux (F16) reach the pass-through arm"
        );
        assert_eq!(report.skipped_non_float, 0);
        assert_eq!(report.bf16_passthrough, 0, "no BF16 in this fixture");
        let out = builder.to_bytes().expect("serialize");
        let file = GgufFile::parse(out).expect("parse");
        let info = file
            .tensor_info("base_lm.some_other.weight")
            .expect("aux tensor must survive round-trip");
        assert_eq!(info.dtype, GgmlType::F16);
        assert_eq!(info.dimensions, vec![2, 3]);
        assert_eq!(file.tensor_bytes(info).len(), 12);
    }

    /// Pins the BF16 leg of the pass-through arm + the
    /// `bf16_passthrough` subset counter. Mirror of the qwen3-tts /
    /// vibevoice / moshi / voxtral pattern: BF16 (the upstream serving
    /// format for both VoxCPM variants) must reach the pass-through arm,
    /// emit as GGUF type 30 verbatim, and increment `bf16_passthrough`.
    #[test]
    fn bf16_aux_tensor_passes_through_verbatim() {
        // 2B route: LM embed at F32 (real vocab) + 6-element BF16 aux.
        let (builder, report) = convert(safetensors_lm_plus_aux(2048, "BF16", 2)).expect("convert");
        assert_eq!(report.variant, Some(VoxCpm2Variant::TwoB));
        assert_eq!(
            report.written, 2,
            "both LM embed (F32) and aux (BF16) reach the pass-through arm"
        );
        assert_eq!(report.skipped_non_float, 0);
        assert_eq!(
            report.bf16_passthrough, 1,
            "BF16 subset counter must record the aux tensor"
        );
        let out = builder.to_bytes().expect("serialize");
        let file = GgufFile::parse(out).expect("parse");
        let info = file
            .tensor_info("base_lm.some_other.weight")
            .expect("BF16 aux must survive pass-through");
        assert_eq!(
            info.dtype,
            GgmlType::BF16,
            "no convert-time widening — GGUF dtype must remain BF16"
        );
        assert_eq!(info.dimensions, vec![2, 3]);
        assert_eq!(
            file.tensor_bytes(info).len(),
            12,
            "BF16 payload = 6 elements × 2 bytes = 12 bytes"
        );
    }

    /// Malformed input surfaces as [`ConvertError::Parse`] (per the
    /// `SafetensorsFile::parse` error propagation contract shared with
    /// every sibling converter).
    #[test]
    fn malformed_input_returns_parse_error() {
        let err = convert(Vec::new()).expect_err("empty");
        assert!(matches!(err, ConvertError::Parse(_)), "{err:?}");

        let mut truncated = Vec::new();
        truncated.extend_from_slice(&1024u64.to_le_bytes());
        truncated.extend_from_slice(b"{}");
        let err = convert(truncated).expect_err("truncated");
        assert!(matches!(err, ConvertError::Parse(_)), "{err:?}");

        let bad_json = b"{not-json";
        let mut bad = Vec::new();
        bad.extend_from_slice(&(bad_json.len() as u64).to_le_bytes());
        bad.extend_from_slice(bad_json);
        let err = convert(bad).expect_err("bad JSON");
        assert!(matches!(err, ConvertError::Parse(_)), "{err:?}");
    }

    /// Every `vokra.voxcpm2.*` and `vokra.vae_continuous.*` key uses
    /// the documented prefix.
    #[test]
    fn every_metadata_key_carries_a_documented_prefix() {
        for key in [
            KEY_FEAT_DIM,
            KEY_PATCH_SIZE,
            KEY_RESIDUAL_LM_N_LAYER,
            KEY_RESIDUAL_LM_NO_ROPE,
            KEY_SCALAR_QUANT_LATENT_DIM,
            KEY_SCALAR_QUANT_SCALE,
            KEY_MAX_LENGTH,
            KEY_MODEL_FAMILY,
            KEY_LM_HIDDEN_DIM,
            KEY_LM_N_LAYER,
            KEY_LM_N_HEAD,
            KEY_LM_N_HEAD_KV,
            KEY_LM_KV_CHANNELS,
            KEY_LM_FFN_DIM,
            KEY_LM_VOCAB_SIZE,
            KEY_LM_MAX_POSITIONS,
            KEY_LM_ROPE_BASE,
            KEY_LM_RMS_NORM_EPS,
            KEY_LM_ROPE_SCALING_LONGROPE,
            KEY_LM_ROPE_ORIG_MAX_POS,
            KEY_LM_SCALE_EMB,
            KEY_LM_DIM_MODEL_BASE,
            KEY_LM_SCALE_DEPTH,
            KEY_LM_USE_MUP,
            KEY_ENC_HIDDEN_DIM,
            KEY_ENC_FFN_DIM,
            KEY_ENC_N_HEAD,
            KEY_ENC_N_LAYER,
            KEY_ENC_KV_CHANNELS,
            KEY_DIT_HIDDEN_DIM,
            KEY_DIT_FFN_DIM,
            KEY_DIT_N_HEAD,
            KEY_DIT_N_LAYER,
            KEY_DIT_KV_CHANNELS,
            KEY_DIT_MEAN_MODE,
            KEY_CFM_SIGMA_MIN,
            KEY_CFM_SOLVER,
            KEY_CFM_T_SCHEDULER,
            KEY_CFM_INFERENCE_CFG_RATE,
        ] {
            assert!(
                key.starts_with("vokra.voxcpm2."),
                "{key} must live under the vokra.voxcpm2.* prefix"
            );
        }
        for key in [
            KEY_VAE_SAMPLE_RATE,
            KEY_VAE_OUT_SAMPLE_RATE,
            KEY_VAE_ENCODER_DIM,
            KEY_VAE_ENCODER_RATES,
            KEY_VAE_LATENT_DIM,
            KEY_VAE_DECODER_DIM,
            KEY_VAE_DECODER_RATES,
            KEY_VAE_DEPTHWISE,
            KEY_VAE_USE_NOISE_BLOCK,
            KEY_VAE_SR_BIN_BOUNDARIES,
        ] {
            assert!(
                key.starts_with("vokra.vae_continuous."),
                "{key} must live under the vokra.vae_continuous.* prefix"
            );
        }
    }

    /// Cross-crate hparam handshake — the LM step feature width MUST
    /// equal the VAE latent, which is invariant across both variants.
    #[test]
    fn feat_dim_and_vae_latent_dim_agree() {
        assert_eq!(FEAT_DIM, VAE_LATENT_DIM);
    }
}
