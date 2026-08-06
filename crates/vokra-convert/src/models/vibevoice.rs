//! **VibeVoice-1.5B**: safetensors checkpoint → GGUF conversion
//! (SoTA plan Phase 4, 2026-07-24).
//!
//! Input: the upstream `microsoft/VibeVoice-1.5B` release —
//! `model.safetensors` (BF16). Output: a GGUF carrying every float
//! tensor plus the `vokra.vibevoice.*` / `vokra.provenance.*` /
//! `vokra.model.*` metadata chunks the native VibeVoice implementation
//! (`crates/vokra-models/src/vibevoice/`) reads.
//!
//! # What is transcribed vs. shape-driven
//!
//! - **Transcribed constants** — every hparam of the
//!   `vokra.vibevoice.*` chunk group is transcribed **verbatim** from
//!   the primary sources
//!   `huggingface.co/microsoft/VibeVoice-1.5B/raw/main/config.json` and
//!   `github.com/microsoft/VibeVoice/blob/main/vibevoice/modular/
//!   configuration_vibevoice.py` (fetched 2026-07-24 — CLAUDE.md
//!   「ハルシネーション厳禁」).
//! - **Nested config blocks** — VibeVoice splits its `config.json`
//!   into `decoder_config.*` (Qwen2 backbone),
//!   `acoustic_tokenizer_config.*` (σ-VAE mirror-symmetric
//!   encoder/decoder), `semantic_tokenizer_config.*` (encoder-only
//!   deterministic), and `diffusion_head_config.*` (4-layer AdaLN MLP
//!   with DDPM v-prediction). Every field of each is transcribed.
//! - **Handshake pins** — `diffusion_head.latent_size ==
//!   acoustic.vae_dim` (VAE handshake) and
//!   `diffusion_head.hidden_size == decoder.hidden_dim` (square
//!   cond_proj) are cross-checked at runtime by
//!   [`vokra_models::vibevoice::VibeVoiceConfig::validate_for_forward`]
//!   — the compile-time algebra at the bottom of this module's tests
//!   pins the same axes at the constant-level.
//!
//! # No side-car config
//!
//! VibeVoice-1.5B ships a real upstream `config.json`, but every field
//! is fixed for the 1.5B release and byte-parallel to the transcribed
//! constants below. A future 7B variant (the release corpus ships the
//! `Qwen2.5-7B` backbone alongside 1.5B) would demand `--config`;
//! this converter fails loudly if a tensor shape disagrees with the
//! transcribed axes at runtime bind time (FR-EX-08).
//!
//! # Tensor naming contract
//!
//! GGUF tensor names are the **upstream safetensors names verbatim**
//! (the CSM / Kokoro / CosyVoice2 / Chatterbox / Qwen3-TTS / VoxCPM
//! contract). Real-weight binding is a follow-up wave gated on the
//! upstream tensor-name manifest fetch; this converter passes every
//! F32 / F16 tensor through unchanged so a future
//! `VibeVoiceWeights::from_gguf` can walk the same names.
//!
//! # BF16 posture
//!
//! The upstream VibeVoice-1.5B release is served in **BF16**
//! (`config.json.torch_dtype = "bfloat16"`). Today's F32 / F16
//! pass-through arm hits `skipped_non_float` on BF16 tensors and the
//! converter surfaces the loud "no float tensors" note. Pre-widen
//! offline to F32 (via a small prepare script — the CSM / Kokoro /
//! VoxCPM pattern) or wait for the streaming BF16 pass-through path
//! (T29-equivalent — the Moshi / Kyutai STT pattern) to convert the
//! release build directly.
//!
//! # No ONNX (permanent)
//!
//! VibeVoice-1.5B is distributed as safetensors + a Python pipeline;
//! this converter **never** touches ONNX (FR-LD-05); the pipeline is
//! re-implemented natively in `crates/vokra-models/src/vibevoice/`
//! (whisper.cpp 型 self re-implementation, CLAUDE.md 設計判断 4).

use vokra_core::LicenseClass;
use vokra_core::gguf::{
    GgmlType, GgufArray, GgufBuilder, GgufMetadataValue, GgufValueType, chunks,
};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` for VibeVoice-1.5B GGUFs — kept in sync with the
/// runtime constant `vokra-models::vibevoice::EXPECTED_ARCH`.
/// Intentionally **distinct** from every sibling arch tag because
/// VibeVoice pairs a continuous VAE decoder with a **DDPM** diffusion
/// head, not the UnifiedCFM flow-matching sampler VoxCPM uses.
/// Silently sharing an arch tag would misroute the runtime dispatch
/// (VoxCPM → flow_sample, VibeVoice → ddpm_sample).
pub(crate) const ARCH: &str = "vibevoice";

/// `vokra.model.name` value written for the canonical VibeVoice-1.5B
/// GGUF.
pub(crate) const NAME: &str = "vibevoice-1.5b";

// --- vokra.vibevoice.* metadata keys ------------------------------------
// The runtime side lives in `crates/vokra-models/src/vibevoice/mod.rs`
// — the two crates share only `vokra-core`, so the cross-crate constant
// duplication rule the CSM / CosyVoice2 / Kokoro / Chatterbox / Qwen3-TTS
// / VoxCPM family converters use applies.

// Top-level
const KEY_MODEL_FAMILY: &str = "vokra.vibevoice.model_family";
const KEY_ACOUSTIC_VAE_DIM: &str = "vokra.vibevoice.acoustic_vae_dim";
const KEY_SEMANTIC_VAE_DIM: &str = "vokra.vibevoice.semantic_vae_dim";
const KEY_LM_FRAME_RATE_HZ: &str = "vokra.vibevoice.lm_frame_rate_hz";

// Qwen2 decoder LM axes — config.json.decoder_config.*
const KEY_DECODER_HIDDEN_DIM: &str = "vokra.vibevoice.decoder.hidden_dim";
const KEY_DECODER_N_LAYER: &str = "vokra.vibevoice.decoder.n_layer";
const KEY_DECODER_N_HEAD: &str = "vokra.vibevoice.decoder.n_head";
const KEY_DECODER_N_HEAD_KV: &str = "vokra.vibevoice.decoder.n_head_kv";
const KEY_DECODER_FFN_DIM: &str = "vokra.vibevoice.decoder.ffn_dim";
const KEY_DECODER_VOCAB_SIZE: &str = "vokra.vibevoice.decoder.vocab_size";
const KEY_DECODER_MAX_POSITIONS: &str = "vokra.vibevoice.decoder.max_position_embeddings";
const KEY_DECODER_ROPE_BASE: &str = "vokra.vibevoice.decoder.rope_base";
const KEY_DECODER_RMS_NORM_EPS: &str = "vokra.vibevoice.decoder.rms_norm_eps";
const KEY_DECODER_ATTENTION_DROPOUT: &str = "vokra.vibevoice.decoder.attention_dropout";
const KEY_DECODER_TIE_WORD_EMBEDDINGS: &str = "vokra.vibevoice.decoder.tie_word_embeddings";
const KEY_DECODER_USE_SLIDING_WINDOW: &str = "vokra.vibevoice.decoder.use_sliding_window";
const KEY_DECODER_MAX_WINDOW_LAYERS: &str = "vokra.vibevoice.decoder.max_window_layers";

// Acoustic tokenizer axes — config.json.acoustic_tokenizer_config.*
const KEY_ACOUSTIC_CHANNELS: &str = "vokra.vibevoice.acoustic.channels";
const KEY_ACOUSTIC_CAUSAL: &str = "vokra.vibevoice.acoustic.causal";
const KEY_ACOUSTIC_VAE_DIM_INNER: &str = "vokra.vibevoice.acoustic.vae_dim";
const KEY_ACOUSTIC_FIX_STD: &str = "vokra.vibevoice.acoustic.fix_std";
const KEY_ACOUSTIC_STD_DIST_TYPE: &str = "vokra.vibevoice.acoustic.std_dist_type";
const KEY_ACOUSTIC_ENCODER_N_FILTERS: &str = "vokra.vibevoice.acoustic.encoder_n_filters";
const KEY_ACOUSTIC_DECODER_N_FILTERS: &str = "vokra.vibevoice.acoustic.decoder_n_filters";
const KEY_ACOUSTIC_ENCODER_RATIOS: &str = "vokra.vibevoice.acoustic.encoder_ratios";
const KEY_ACOUSTIC_DECODER_RATIOS: &str = "vokra.vibevoice.acoustic.decoder_ratios";
const KEY_ACOUSTIC_ENCODER_DEPTHS: &str = "vokra.vibevoice.acoustic.encoder_depths";
const KEY_ACOUSTIC_LAYER_SCALE_INIT_VALUE: &str = "vokra.vibevoice.acoustic.layer_scale_init_value";
const KEY_ACOUSTIC_WEIGHT_INIT_VALUE: &str = "vokra.vibevoice.acoustic.weight_init_value";
const KEY_ACOUSTIC_LAYERNORM: &str = "vokra.vibevoice.acoustic.layernorm";
const KEY_ACOUSTIC_LAYERNORM_ELEMENTWISE_AFFINE: &str =
    "vokra.vibevoice.acoustic.layernorm_elementwise_affine";
const KEY_ACOUSTIC_LAYERNORM_EPS: &str = "vokra.vibevoice.acoustic.layernorm_eps";
const KEY_ACOUSTIC_MIXER_LAYER: &str = "vokra.vibevoice.acoustic.mixer_layer";
const KEY_ACOUSTIC_PAD_MODE: &str = "vokra.vibevoice.acoustic.pad_mode";
const KEY_ACOUSTIC_DISABLE_LAST_NORM: &str = "vokra.vibevoice.acoustic.disable_last_norm";
const KEY_ACOUSTIC_CONV_NORM: &str = "vokra.vibevoice.acoustic.conv_norm";
const KEY_ACOUSTIC_CONV_BIAS: &str = "vokra.vibevoice.acoustic.conv_bias";
const KEY_ACOUSTIC_CORPUS_NORMALIZE: &str = "vokra.vibevoice.acoustic.corpus_normalize";
const KEY_ACOUSTIC_SAMPLE_RATE_HZ: &str = "vokra.vibevoice.acoustic.sample_rate_hz";

// Semantic tokenizer axes — config.json.semantic_tokenizer_config.*
const KEY_SEMANTIC_CHANNELS: &str = "vokra.vibevoice.semantic.channels";
const KEY_SEMANTIC_CAUSAL: &str = "vokra.vibevoice.semantic.causal";
const KEY_SEMANTIC_VAE_DIM_INNER: &str = "vokra.vibevoice.semantic.vae_dim";
const KEY_SEMANTIC_FIX_STD: &str = "vokra.vibevoice.semantic.fix_std";
const KEY_SEMANTIC_STD_DIST_TYPE: &str = "vokra.vibevoice.semantic.std_dist_type";
const KEY_SEMANTIC_ENCODER_N_FILTERS: &str = "vokra.vibevoice.semantic.encoder_n_filters";
const KEY_SEMANTIC_ENCODER_RATIOS: &str = "vokra.vibevoice.semantic.encoder_ratios";
const KEY_SEMANTIC_ENCODER_DEPTHS: &str = "vokra.vibevoice.semantic.encoder_depths";
const KEY_SEMANTIC_LAYERNORM: &str = "vokra.vibevoice.semantic.layernorm";
const KEY_SEMANTIC_LAYERNORM_EPS: &str = "vokra.vibevoice.semantic.layernorm_eps";
const KEY_SEMANTIC_MIXER_LAYER: &str = "vokra.vibevoice.semantic.mixer_layer";
const KEY_SEMANTIC_CONV_BIAS: &str = "vokra.vibevoice.semantic.conv_bias";

// Diffusion head axes — config.json.diffusion_head_config.*
const KEY_DIFFUSION_HEAD_HIDDEN_SIZE: &str = "vokra.vibevoice.diffusion_head.hidden_size";
const KEY_DIFFUSION_HEAD_LAYERS: &str = "vokra.vibevoice.diffusion_head.head_layers";
const KEY_DIFFUSION_HEAD_FFN_RATIO: &str = "vokra.vibevoice.diffusion_head.head_ffn_ratio";
const KEY_DIFFUSION_HEAD_RMS_NORM_EPS: &str = "vokra.vibevoice.diffusion_head.rms_norm_eps";
const KEY_DIFFUSION_HEAD_LATENT_SIZE: &str = "vokra.vibevoice.diffusion_head.latent_size";
const KEY_DIFFUSION_HEAD_SPEECH_VAE_DIM: &str = "vokra.vibevoice.diffusion_head.speech_vae_dim";
const KEY_DIFFUSION_HEAD_PREDICTION_TYPE: &str = "vokra.vibevoice.diffusion_head.prediction_type";
const KEY_DIFFUSION_HEAD_DIFFUSION_TYPE: &str = "vokra.vibevoice.diffusion_head.diffusion_type";
const KEY_DIFFUSION_HEAD_DDPM_NUM_STEPS: &str = "vokra.vibevoice.diffusion_head.ddpm_num_steps";
const KEY_DIFFUSION_HEAD_DDPM_NUM_INFERENCE_STEPS: &str =
    "vokra.vibevoice.diffusion_head.ddpm_num_inference_steps";
const KEY_DIFFUSION_HEAD_DDPM_BETA_SCHEDULE: &str =
    "vokra.vibevoice.diffusion_head.ddpm_beta_schedule";
const KEY_DIFFUSION_HEAD_DDPM_BATCH_MUL: &str = "vokra.vibevoice.diffusion_head.ddpm_batch_mul";

// --- Transcribed constants ------------------------------------------------
// Primary sources: `huggingface.co/microsoft/VibeVoice-1.5B/raw/main/config.json`
// + `github.com/microsoft/VibeVoice/blob/main/vibevoice/modular/
// configuration_vibevoice.py` (fetched 2026-07-24 — CLAUDE.md
// 「ハルシネーション厳禁」).

/// Model family marker (`architecture = "VibeVoiceForConditionalGeneration"`).
/// Recorded so the runtime can distinguish VibeVoice from other Qwen2
/// backbone releases at telemetry time.
const MODEL_FAMILY: &str = "vibevoice";

// Acoustic + semantic tokenizer sample rate + LM frame rate.
const ACOUSTIC_SAMPLE_RATE_HZ: u32 = 24_000;
const LM_FRAME_RATE_HZ: f32 = 7.5;

// Top-level shortcuts.
const ACOUSTIC_VAE_DIM: u32 = 64;
const SEMANTIC_VAE_DIM: u32 = 128;

// Qwen2 decoder LM (config.json.decoder_config.*).
const DECODER_HIDDEN_DIM: u32 = 1536;
const DECODER_N_LAYER: u32 = 28;
const DECODER_N_HEAD: u32 = 12;
const DECODER_N_HEAD_KV: u32 = 2;
const DECODER_FFN_DIM: u32 = 8960;
const DECODER_VOCAB_SIZE: u32 = 151_936;
const DECODER_MAX_POSITIONS: u32 = 65_536;
const DECODER_ROPE_BASE: f32 = 1_000_000.0;
const DECODER_RMS_NORM_EPS: f32 = 1e-6;
const DECODER_ATTENTION_DROPOUT: f32 = 0.0;
const DECODER_TIE_WORD_EMBEDDINGS: bool = true;
const DECODER_USE_SLIDING_WINDOW: bool = false;
const DECODER_MAX_WINDOW_LAYERS: u32 = 28;

// Acoustic tokenizer (config.json.acoustic_tokenizer_config.*).
const ACOUSTIC_CHANNELS: u32 = 1;
const ACOUSTIC_CAUSAL: bool = true;
const ACOUSTIC_FIX_STD: f32 = 0.5;
const ACOUSTIC_STD_DIST_TYPE: &str = "gaussian";
const ACOUSTIC_ENCODER_N_FILTERS: u32 = 32;
const ACOUSTIC_DECODER_N_FILTERS: u32 = 32;
const ACOUSTIC_ENCODER_RATIOS: [u32; 6] = [8, 5, 5, 4, 2, 2];
const ACOUSTIC_DECODER_RATIOS: [u32; 6] = [8, 5, 5, 4, 2, 2];
const ACOUSTIC_ENCODER_DEPTHS: &str = "3-3-3-3-3-3-8";
const ACOUSTIC_LAYER_SCALE_INIT_VALUE: f32 = 1e-6;
const ACOUSTIC_WEIGHT_INIT_VALUE: f32 = 1e-2;
const ACOUSTIC_LAYERNORM: &str = "RMSNorm";
const ACOUSTIC_LAYERNORM_ELEMENTWISE_AFFINE: bool = true;
const ACOUSTIC_LAYERNORM_EPS: f32 = 1e-5;
const ACOUSTIC_MIXER_LAYER: &str = "depthwise_conv";
const ACOUSTIC_PAD_MODE: &str = "constant";
const ACOUSTIC_DISABLE_LAST_NORM: bool = true;
const ACOUSTIC_CONV_NORM: &str = "none";
const ACOUSTIC_CONV_BIAS: bool = true;
const ACOUSTIC_CORPUS_NORMALIZE: f32 = 0.0;

// Semantic tokenizer (config.json.semantic_tokenizer_config.*).
const SEMANTIC_CHANNELS: u32 = 1;
const SEMANTIC_CAUSAL: bool = true;
const SEMANTIC_FIX_STD: f32 = 0.0;
const SEMANTIC_STD_DIST_TYPE: &str = "none";
const SEMANTIC_ENCODER_N_FILTERS: u32 = 32;
const SEMANTIC_ENCODER_RATIOS: [u32; 6] = [8, 5, 5, 4, 2, 2];
const SEMANTIC_ENCODER_DEPTHS: &str = "3-3-3-3-3-3-8";
const SEMANTIC_LAYERNORM: &str = "RMSNorm";
const SEMANTIC_LAYERNORM_EPS: f32 = 1e-5;
const SEMANTIC_MIXER_LAYER: &str = "depthwise_conv";
const SEMANTIC_CONV_BIAS: bool = true;

// Diffusion head (config.json.diffusion_head_config.*).
const DIFFUSION_HEAD_HIDDEN_SIZE: u32 = 1536;
const DIFFUSION_HEAD_LAYERS: u32 = 4;
const DIFFUSION_HEAD_FFN_RATIO: f32 = 3.0;
const DIFFUSION_HEAD_RMS_NORM_EPS: f32 = 1e-5;
const DIFFUSION_HEAD_LATENT_SIZE: u32 = 64;
const DIFFUSION_HEAD_SPEECH_VAE_DIM: u32 = 64;
const DIFFUSION_HEAD_PREDICTION_TYPE: &str = "v_prediction";
const DIFFUSION_HEAD_DIFFUSION_TYPE: &str = "ddpm";
const DIFFUSION_HEAD_DDPM_NUM_STEPS: u32 = 1000;
const DIFFUSION_HEAD_DDPM_NUM_INFERENCE_STEPS: u32 = 20;
const DIFFUSION_HEAD_DDPM_BETA_SCHEDULE: &str = "cosine";
const DIFFUSION_HEAD_DDPM_BATCH_MUL: u32 = 4;

/// Outcome of a VibeVoice-1.5B conversion.
#[derive(Debug, Default)]
pub(crate) struct VibeVoiceReport {
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
    /// Operator-facing diagnostics (never fail the conversion — the
    /// runtime is the authoritative gate, FR-EX-08).
    pub(crate) notes: Vec<String>,
}

/// Converts a VibeVoice-1.5B safetensors buffer into a populated GGUF
/// builder.
///
/// Every F32 / F16 tensor passes through under its upstream name; the
/// `vokra.vibevoice.*` chunk group is written from the transcribed
/// constants above; provenance stamps mark the weight as `Permissive`
/// (MIT — end-to-end).
pub(crate) fn convert(bytes: Vec<u8>) -> Result<(GgufBuilder, VibeVoiceReport), ConvertError> {
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    write_hparams(&mut b);
    // Self-describing redistribution: the artifact carries its own licence.
    // VibeVoice-1.5B ships MIT end-to-end (LICENSE +
    // huggingface.co/microsoft/VibeVoice-1.5B model card `license: MIT`,
    // fetched 2026-07-24 — CLAUDE.md「ハルシネーション厳禁」).
    // MIT is a `Permissive` license class — same commercial verdict as
    // apache-2.0 (no runtime-side attribution obligation), just a
    // different SPDX string.
    vokra_core::stamp_provenance(
        &mut b,
        LicenseClass::Permissive,
        "mit",
        Some(NAME),
        Some("microsoft/VibeVoice-1.5B (MIT end-to-end)"),
    );

    let mut report = VibeVoiceReport::default();
    for t in st.tensors() {
        match t.dtype {
            // BF16 pass-through added 2026-07-25 (mirror of qwen3-tts +
            // moshi + voxtral): upstream VibeVoice-1.5B ships
            // `torch_dtype: bfloat16` so the release checkpoint hits this
            // arm. Emit as GGUF type 30 verbatim; runtime widens on load
            // via `decode_bf16` (exact, `bits << 16`).
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
            }
            _ => {
                report.skipped_non_float += 1;
            }
        }
    }
    if report.written == 0 {
        report.notes.push(
            "no float tensors passed through — this GGUF is metadata-only and \
             the runtime will refuse to bind any weights (FR-EX-08). The \
             upstream VibeVoice-1.5B release ships `model.safetensors` in BF16 \
             (config.json `torch_dtype: bfloat16`); the BF16 pass-through path \
             is now wired (2026-07-25), so this state is only reachable when \
             the release contains no F32 / F16 / BF16 float tensors at all."
                .into(),
        );
    }
    Ok((b, report))
}

/// Writes the `vokra.vibevoice.*` chunk group from the transcribed
/// constants above (primary sources: `config.json` and
/// `configuration_vibevoice.py`).
fn write_hparams(b: &mut GgufBuilder) {
    // Top-level.
    b.add_string(KEY_MODEL_FAMILY, MODEL_FAMILY);
    b.add_u32(KEY_ACOUSTIC_VAE_DIM, ACOUSTIC_VAE_DIM);
    b.add_u32(KEY_SEMANTIC_VAE_DIM, SEMANTIC_VAE_DIM);
    b.add_f32(KEY_LM_FRAME_RATE_HZ, LM_FRAME_RATE_HZ);

    // Qwen2 decoder LM.
    b.add_u32(KEY_DECODER_HIDDEN_DIM, DECODER_HIDDEN_DIM);
    b.add_u32(KEY_DECODER_N_LAYER, DECODER_N_LAYER);
    b.add_u32(KEY_DECODER_N_HEAD, DECODER_N_HEAD);
    b.add_u32(KEY_DECODER_N_HEAD_KV, DECODER_N_HEAD_KV);
    b.add_u32(KEY_DECODER_FFN_DIM, DECODER_FFN_DIM);
    b.add_u32(KEY_DECODER_VOCAB_SIZE, DECODER_VOCAB_SIZE);
    b.add_u32(KEY_DECODER_MAX_POSITIONS, DECODER_MAX_POSITIONS);
    b.add_f32(KEY_DECODER_ROPE_BASE, DECODER_ROPE_BASE);
    b.add_f32(KEY_DECODER_RMS_NORM_EPS, DECODER_RMS_NORM_EPS);
    b.add_f32(KEY_DECODER_ATTENTION_DROPOUT, DECODER_ATTENTION_DROPOUT);
    b.add_bool(KEY_DECODER_TIE_WORD_EMBEDDINGS, DECODER_TIE_WORD_EMBEDDINGS);
    b.add_bool(KEY_DECODER_USE_SLIDING_WINDOW, DECODER_USE_SLIDING_WINDOW);
    b.add_u32(KEY_DECODER_MAX_WINDOW_LAYERS, DECODER_MAX_WINDOW_LAYERS);

    // Acoustic tokenizer.
    b.add_u32(KEY_ACOUSTIC_CHANNELS, ACOUSTIC_CHANNELS);
    b.add_bool(KEY_ACOUSTIC_CAUSAL, ACOUSTIC_CAUSAL);
    b.add_u32(KEY_ACOUSTIC_VAE_DIM_INNER, ACOUSTIC_VAE_DIM);
    b.add_f32(KEY_ACOUSTIC_FIX_STD, ACOUSTIC_FIX_STD);
    b.add_string(KEY_ACOUSTIC_STD_DIST_TYPE, ACOUSTIC_STD_DIST_TYPE);
    b.add_u32(KEY_ACOUSTIC_ENCODER_N_FILTERS, ACOUSTIC_ENCODER_N_FILTERS);
    b.add_u32(KEY_ACOUSTIC_DECODER_N_FILTERS, ACOUSTIC_DECODER_N_FILTERS);
    b.add_metadata(
        KEY_ACOUSTIC_ENCODER_RATIOS,
        GgufMetadataValue::Array(GgufArray {
            element_type: GgufValueType::U32,
            values: ACOUSTIC_ENCODER_RATIOS
                .iter()
                .map(|&r| GgufMetadataValue::U32(r))
                .collect(),
        }),
    );
    b.add_metadata(
        KEY_ACOUSTIC_DECODER_RATIOS,
        GgufMetadataValue::Array(GgufArray {
            element_type: GgufValueType::U32,
            values: ACOUSTIC_DECODER_RATIOS
                .iter()
                .map(|&r| GgufMetadataValue::U32(r))
                .collect(),
        }),
    );
    b.add_string(KEY_ACOUSTIC_ENCODER_DEPTHS, ACOUSTIC_ENCODER_DEPTHS);
    b.add_f32(
        KEY_ACOUSTIC_LAYER_SCALE_INIT_VALUE,
        ACOUSTIC_LAYER_SCALE_INIT_VALUE,
    );
    b.add_f32(KEY_ACOUSTIC_WEIGHT_INIT_VALUE, ACOUSTIC_WEIGHT_INIT_VALUE);
    b.add_string(KEY_ACOUSTIC_LAYERNORM, ACOUSTIC_LAYERNORM);
    b.add_bool(
        KEY_ACOUSTIC_LAYERNORM_ELEMENTWISE_AFFINE,
        ACOUSTIC_LAYERNORM_ELEMENTWISE_AFFINE,
    );
    b.add_f32(KEY_ACOUSTIC_LAYERNORM_EPS, ACOUSTIC_LAYERNORM_EPS);
    b.add_string(KEY_ACOUSTIC_MIXER_LAYER, ACOUSTIC_MIXER_LAYER);
    b.add_string(KEY_ACOUSTIC_PAD_MODE, ACOUSTIC_PAD_MODE);
    b.add_bool(KEY_ACOUSTIC_DISABLE_LAST_NORM, ACOUSTIC_DISABLE_LAST_NORM);
    b.add_string(KEY_ACOUSTIC_CONV_NORM, ACOUSTIC_CONV_NORM);
    b.add_bool(KEY_ACOUSTIC_CONV_BIAS, ACOUSTIC_CONV_BIAS);
    b.add_f32(KEY_ACOUSTIC_CORPUS_NORMALIZE, ACOUSTIC_CORPUS_NORMALIZE);
    b.add_u32(KEY_ACOUSTIC_SAMPLE_RATE_HZ, ACOUSTIC_SAMPLE_RATE_HZ);

    // Semantic tokenizer.
    b.add_u32(KEY_SEMANTIC_CHANNELS, SEMANTIC_CHANNELS);
    b.add_bool(KEY_SEMANTIC_CAUSAL, SEMANTIC_CAUSAL);
    b.add_u32(KEY_SEMANTIC_VAE_DIM_INNER, SEMANTIC_VAE_DIM);
    b.add_f32(KEY_SEMANTIC_FIX_STD, SEMANTIC_FIX_STD);
    b.add_string(KEY_SEMANTIC_STD_DIST_TYPE, SEMANTIC_STD_DIST_TYPE);
    b.add_u32(KEY_SEMANTIC_ENCODER_N_FILTERS, SEMANTIC_ENCODER_N_FILTERS);
    b.add_metadata(
        KEY_SEMANTIC_ENCODER_RATIOS,
        GgufMetadataValue::Array(GgufArray {
            element_type: GgufValueType::U32,
            values: SEMANTIC_ENCODER_RATIOS
                .iter()
                .map(|&r| GgufMetadataValue::U32(r))
                .collect(),
        }),
    );
    b.add_string(KEY_SEMANTIC_ENCODER_DEPTHS, SEMANTIC_ENCODER_DEPTHS);
    b.add_string(KEY_SEMANTIC_LAYERNORM, SEMANTIC_LAYERNORM);
    b.add_f32(KEY_SEMANTIC_LAYERNORM_EPS, SEMANTIC_LAYERNORM_EPS);
    b.add_string(KEY_SEMANTIC_MIXER_LAYER, SEMANTIC_MIXER_LAYER);
    b.add_bool(KEY_SEMANTIC_CONV_BIAS, SEMANTIC_CONV_BIAS);

    // Diffusion head.
    b.add_u32(KEY_DIFFUSION_HEAD_HIDDEN_SIZE, DIFFUSION_HEAD_HIDDEN_SIZE);
    b.add_u32(KEY_DIFFUSION_HEAD_LAYERS, DIFFUSION_HEAD_LAYERS);
    b.add_f32(KEY_DIFFUSION_HEAD_FFN_RATIO, DIFFUSION_HEAD_FFN_RATIO);
    b.add_f32(KEY_DIFFUSION_HEAD_RMS_NORM_EPS, DIFFUSION_HEAD_RMS_NORM_EPS);
    b.add_u32(KEY_DIFFUSION_HEAD_LATENT_SIZE, DIFFUSION_HEAD_LATENT_SIZE);
    b.add_u32(
        KEY_DIFFUSION_HEAD_SPEECH_VAE_DIM,
        DIFFUSION_HEAD_SPEECH_VAE_DIM,
    );
    b.add_string(
        KEY_DIFFUSION_HEAD_PREDICTION_TYPE,
        DIFFUSION_HEAD_PREDICTION_TYPE,
    );
    b.add_string(
        KEY_DIFFUSION_HEAD_DIFFUSION_TYPE,
        DIFFUSION_HEAD_DIFFUSION_TYPE,
    );
    b.add_u32(
        KEY_DIFFUSION_HEAD_DDPM_NUM_STEPS,
        DIFFUSION_HEAD_DDPM_NUM_STEPS,
    );
    b.add_u32(
        KEY_DIFFUSION_HEAD_DDPM_NUM_INFERENCE_STEPS,
        DIFFUSION_HEAD_DDPM_NUM_INFERENCE_STEPS,
    );
    b.add_string(
        KEY_DIFFUSION_HEAD_DDPM_BETA_SCHEDULE,
        DIFFUSION_HEAD_DDPM_BETA_SCHEDULE,
    );
    b.add_u32(
        KEY_DIFFUSION_HEAD_DDPM_BATCH_MUL,
        DIFFUSION_HEAD_DDPM_BATCH_MUL,
    );
}

// === VibeVoice-Realtime-0.5B (streaming variant) ==========================
//
// Added 2026-08-01 for `microsoft/VibeVoice-Realtime-0.5B` publish.
// Primary source: `huggingface.co/microsoft/VibeVoice-Realtime-0.5B/raw/
// main/config.json` (fetched 2026-08-01 -- CLAUDE.md 「ハルシネーション
// 厳禁」).
//
// Diverges from the 1.5B base in four axes:
//   1. `model_type` = `vibevoice_streaming` (vs `vibevoice`).
//   2. Qwen2 backbone reshaped -- hidden 896 (vs 1536), 24 layers
//      (vs 28), 14 heads (vs 12), FFN 4864 (vs 8960),
//      max_positions 8192 (vs 65536), `tie_word_embeddings=false`
//      (vs true).
//   3. Semantic tokenizer is **absent** -- the streaming variant is
//      acoustic-tokenizer-only; the runtime dispatch must not read any
//      `vokra.vibevoice.semantic.*` key for a Realtime GGUF.
//   4. New top-level axis `tts_backbone_num_hidden_layers = 20`
//      (streaming-specific -- carried under the new key
//      `vokra.vibevoice.tts_backbone_num_hidden_layers`).
//
// Acoustic tokenizer axes + diffusion-head axes (except `hidden_size`) +
// LM frame rate + acoustic sample rate are byte-identical to 1.5B per
// primary source, so the existing 1.5B constants are reused directly.

/// Which VibeVoice release this converter emits.
///
/// The two variants share Vokra's `vokra.vibevoice.*` metadata prefix
/// but carry **distinct** `vokra.model.arch` tags -- silently sharing
/// an arch tag would misroute the runtime dispatch (Base15B ships a
/// semantic tokenizer, Realtime05B does not; Base15B binds a 1536-dim
/// Qwen2 backbone, Realtime05B binds an 896-dim one; a runtime that
/// expected the wrong config would allocate the wrong-shape KV cache
/// on the first token).
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VibeVoiceVariant {
    /// `microsoft/VibeVoice-1.5B` -- the 2026-07-24 SoTA Phase 4
    /// release. Qwen2 backbone: hidden 1536, 28 layers, 12 heads,
    /// 2 kv heads, FFN 8960, `tie_word_embeddings=true`. Semantic
    /// tokenizer present. `vokra.model.arch = "vibevoice"`.
    Base15B,
    /// `microsoft/VibeVoice-Realtime-0.5B` -- the 2026-08-01-added
    /// streaming variant (upstream `model_type =
    /// "vibevoice_streaming"`). Qwen2 backbone: hidden 896, 24
    /// layers, 14 heads, 2 kv heads, FFN 4864,
    /// `tie_word_embeddings=false`. Semantic tokenizer **absent**.
    /// Adds `tts_backbone_num_hidden_layers = 20`.
    /// `vokra.model.arch = "vibevoice_streaming"`.
    Realtime05B,
}

#[allow(dead_code)]
impl VibeVoiceVariant {
    /// `vokra.model.arch` value for this variant.
    pub const fn arch(self) -> &'static str {
        match self {
            Self::Base15B => ARCH,
            Self::Realtime05B => ARCH_STREAMING,
        }
    }

    /// `vokra.model.name` value.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Base15B => NAME,
            Self::Realtime05B => NAME_REALTIME_05B,
        }
    }

    /// Upstream HF path -- recorded in the provenance chunk group so a
    /// downstream reader can locate the source release.
    pub const fn upstream_hf(self) -> &'static str {
        match self {
            Self::Base15B => "microsoft/VibeVoice-1.5B",
            Self::Realtime05B => "microsoft/VibeVoice-Realtime-0.5B",
        }
    }

    /// Provenance `source` field text.
    const fn source_description(self) -> &'static str {
        match self {
            Self::Base15B => "microsoft/VibeVoice-1.5B (MIT end-to-end)",
            Self::Realtime05B => "microsoft/VibeVoice-Realtime-0.5B (MIT end-to-end, streaming)",
        }
    }
}

/// `vokra.model.arch` value for the Realtime streaming variant.
/// Kept **distinct** from [`ARCH`] so runtime dispatch never conflates
/// the two variants (see [`VibeVoiceVariant`] rustdoc for the topology
/// divergences).
pub(crate) const ARCH_STREAMING: &str = "vibevoice_streaming";

/// `vokra.model.name` value for the canonical Realtime-0.5B GGUF.
pub(crate) const NAME_REALTIME_05B: &str = "vibevoice-realtime-0.5b";

/// Streaming-specific top-level key
/// (`config.json.tts_backbone_num_hidden_layers`). Only written on the
/// Realtime path; the 1.5B path does not carry this axis.
const KEY_TTS_BACKBONE_NUM_HIDDEN_LAYERS: &str = "vokra.vibevoice.tts_backbone_num_hidden_layers";

// --- Realtime-0.5B transcribed constants ---------------------------------
// Primary source: `huggingface.co/microsoft/VibeVoice-Realtime-0.5B/raw/
// main/config.json` (fetched 2026-08-01 -- CLAUDE.md 「ハルシネーション
// 厳禁」).

const RT_MODEL_FAMILY: &str = "vibevoice_streaming";

// Qwen2 decoder LM -- every axis that differs from the 1.5B baseline
// is declared explicitly here. Shared axes (`n_head_kv=2`,
// `vocab_size=151_936`, `rope_theta=1_000_000`, `rms_norm_eps=1e-6`,
// `attention_dropout=0.0`, `use_sliding_window=false`) reuse the 1.5B
// constants above so a future upstream update lands atomically.
const RT_DECODER_HIDDEN_DIM: u32 = 896;
const RT_DECODER_N_LAYER: u32 = 24;
const RT_DECODER_N_HEAD: u32 = 14;
const RT_DECODER_FFN_DIM: u32 = 4864;
const RT_DECODER_MAX_POSITIONS: u32 = 8_192;
const RT_DECODER_MAX_WINDOW_LAYERS: u32 = 24;
const RT_DECODER_TIE_WORD_EMBEDDINGS: bool = false;

// Diffusion head -- only `hidden_size` differs (matches the Qwen2
// hidden so `cond_proj` is a square linear). Every other
// diffusion-head axis is byte-identical to the 1.5B baseline per
// primary source and reuses the shared constants above.
const RT_DIFFUSION_HEAD_HIDDEN_SIZE: u32 = 896;

// Streaming-only top-level axis.
const RT_TTS_BACKBONE_NUM_HIDDEN_LAYERS: u32 = 20;

// Compile-time algebra pins for the Realtime-0.5B constants -- mirror
// of the 1.5B pins in `transcribed_constants_match_primary_source`
// below, promoted to const-eval so a shape drift fails at build time.
const _: () = {
    // GQA well-formedness.
    assert!(RT_DECODER_N_HEAD % DECODER_N_HEAD_KV == 0);
    // VAE handshake reuses the shared 1.5B pins (identical values per
    // primary source).
    assert!(DIFFUSION_HEAD_LATENT_SIZE == ACOUSTIC_VAE_DIM);
    assert!(DIFFUSION_HEAD_SPEECH_VAE_DIM == ACOUSTIC_VAE_DIM);
    // Hidden-size handshake: cond_proj is a square linear on Realtime.
    assert!(RT_DIFFUSION_HEAD_HIDDEN_SIZE == RT_DECODER_HIDDEN_DIM);
    // Streaming backbone must be a proper subset of the decoder
    // layers (structural sanity: tts_backbone spans the first N
    // decoder layers).
    assert!(RT_TTS_BACKBONE_NUM_HIDDEN_LAYERS < RT_DECODER_N_LAYER);
    // Sanity: distinct SKUs must not share the exact backbone shape
    // (a drift where both variants become 896-dim / 24-layer would
    // silently collapse the round-trip test `realtime_variant_arch_
    // is_distinct_from_base` into a trivially-passing tautology).
    assert!(RT_DECODER_HIDDEN_DIM != DECODER_HIDDEN_DIM);
    assert!(RT_DECODER_N_LAYER != DECODER_N_LAYER);
};

/// Convert a `microsoft/VibeVoice-Realtime-0.5B` safetensors buffer
/// into a populated GGUF builder.
///
/// Sibling of [`convert`] -- the two paths differ only in the arch
/// tag, name, and the `write_hparams_realtime_05b` chunk-group writer.
/// The tensor pass-through loop (F32/F16/BF16 verbatim) and the
/// provenance stamp (MIT permissive, end-to-end) are otherwise
/// byte-parallel.
pub(crate) fn convert_realtime_05b(
    bytes: Vec<u8>,
) -> Result<(GgufBuilder, VibeVoiceReport), ConvertError> {
    let variant = VibeVoiceVariant::Realtime05B;
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, variant.arch());
    b.add_string(chunks::KEY_MODEL_NAME, variant.name());
    write_hparams_realtime_05b(&mut b);
    // Self-describing redistribution: MIT end-to-end
    // (`huggingface.co/microsoft/VibeVoice-Realtime-0.5B` cardData
    // `license: mit`, fetched 2026-08-01 -- CLAUDE.md 「ハルシネー
    // ション厳禁」). MIT is a `Permissive` license class (same
    // commercial verdict as apache-2.0, no runtime attribution
    // obligation), just a different SPDX string. The
    // `LicenseClass::from_id` prefix walk (`id.starts_with
    // ("vibevoice-")`) also resolves this variant to `Permissive`.
    vokra_core::stamp_provenance(
        &mut b,
        LicenseClass::Permissive,
        "mit",
        Some(variant.name()),
        Some(variant.source_description()),
    );

    let mut report = VibeVoiceReport::default();
    for t in st.tensors() {
        match t.dtype {
            // Same BF16 pass-through rule as the 1.5B path (mirror
            // of qwen3-tts + moshi + voxtral + voxcpm2 + bigvgan):
            // emit BF16 as GGUF type 30 verbatim; runtime widens on
            // load via `decode_bf16` (exact, `bits << 16`).
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
            }
            _ => {
                report.skipped_non_float += 1;
            }
        }
    }
    if report.written == 0 {
        report.notes.push(
            "no float tensors passed through -- this GGUF is \
             metadata-only and the runtime will refuse to bind any \
             weights (FR-EX-08). The upstream \
             VibeVoice-Realtime-0.5B release ships \
             `model.safetensors` in BF16 (config.json \
             `torch_dtype: bfloat16`); the BF16 pass-through path \
             is wired (2026-07-25), so this state is only reachable \
             when the release contains no F32 / F16 / BF16 float \
             tensors at all."
                .into(),
        );
    }
    Ok((b, report))
}

/// Writes the `vokra.vibevoice.*` chunk group for the Realtime-0.5B
/// variant. Reuses the shared acoustic tokenizer + diffusion-head
/// constants that are byte-identical between the two variants (per
/// primary source); overrides the decoder + `hidden_size` +
/// `tie_word_embeddings` axes and adds the streaming-only
/// `tts_backbone_num_hidden_layers` axis. Skips every
/// `vokra.vibevoice.semantic.*` key -- the streaming variant is
/// acoustic-tokenizer-only.
fn write_hparams_realtime_05b(b: &mut GgufBuilder) {
    // Top-level.
    b.add_string(KEY_MODEL_FAMILY, RT_MODEL_FAMILY);
    b.add_u32(KEY_ACOUSTIC_VAE_DIM, ACOUSTIC_VAE_DIM);
    // Semantic tokenizer is absent in the streaming variant -- do
    // NOT write KEY_SEMANTIC_VAE_DIM or any KEY_SEMANTIC_* key.
    b.add_f32(KEY_LM_FRAME_RATE_HZ, LM_FRAME_RATE_HZ);
    b.add_u32(
        KEY_TTS_BACKBONE_NUM_HIDDEN_LAYERS,
        RT_TTS_BACKBONE_NUM_HIDDEN_LAYERS,
    );

    // Qwen2 decoder LM (config.json.decoder_config.*).
    b.add_u32(KEY_DECODER_HIDDEN_DIM, RT_DECODER_HIDDEN_DIM);
    b.add_u32(KEY_DECODER_N_LAYER, RT_DECODER_N_LAYER);
    b.add_u32(KEY_DECODER_N_HEAD, RT_DECODER_N_HEAD);
    b.add_u32(KEY_DECODER_N_HEAD_KV, DECODER_N_HEAD_KV);
    b.add_u32(KEY_DECODER_FFN_DIM, RT_DECODER_FFN_DIM);
    b.add_u32(KEY_DECODER_VOCAB_SIZE, DECODER_VOCAB_SIZE);
    b.add_u32(KEY_DECODER_MAX_POSITIONS, RT_DECODER_MAX_POSITIONS);
    b.add_f32(KEY_DECODER_ROPE_BASE, DECODER_ROPE_BASE);
    b.add_f32(KEY_DECODER_RMS_NORM_EPS, DECODER_RMS_NORM_EPS);
    b.add_f32(KEY_DECODER_ATTENTION_DROPOUT, DECODER_ATTENTION_DROPOUT);
    b.add_bool(
        KEY_DECODER_TIE_WORD_EMBEDDINGS,
        RT_DECODER_TIE_WORD_EMBEDDINGS,
    );
    b.add_bool(KEY_DECODER_USE_SLIDING_WINDOW, DECODER_USE_SLIDING_WINDOW);
    b.add_u32(KEY_DECODER_MAX_WINDOW_LAYERS, RT_DECODER_MAX_WINDOW_LAYERS);

    // Acoustic tokenizer -- byte-identical to 1.5B per primary
    // source, so every axis reuses the shared constant.
    b.add_u32(KEY_ACOUSTIC_CHANNELS, ACOUSTIC_CHANNELS);
    b.add_bool(KEY_ACOUSTIC_CAUSAL, ACOUSTIC_CAUSAL);
    b.add_u32(KEY_ACOUSTIC_VAE_DIM_INNER, ACOUSTIC_VAE_DIM);
    b.add_f32(KEY_ACOUSTIC_FIX_STD, ACOUSTIC_FIX_STD);
    b.add_string(KEY_ACOUSTIC_STD_DIST_TYPE, ACOUSTIC_STD_DIST_TYPE);
    b.add_u32(KEY_ACOUSTIC_ENCODER_N_FILTERS, ACOUSTIC_ENCODER_N_FILTERS);
    b.add_u32(KEY_ACOUSTIC_DECODER_N_FILTERS, ACOUSTIC_DECODER_N_FILTERS);
    b.add_metadata(
        KEY_ACOUSTIC_ENCODER_RATIOS,
        GgufMetadataValue::Array(GgufArray {
            element_type: GgufValueType::U32,
            values: ACOUSTIC_ENCODER_RATIOS
                .iter()
                .map(|&r| GgufMetadataValue::U32(r))
                .collect(),
        }),
    );
    b.add_metadata(
        KEY_ACOUSTIC_DECODER_RATIOS,
        GgufMetadataValue::Array(GgufArray {
            element_type: GgufValueType::U32,
            values: ACOUSTIC_DECODER_RATIOS
                .iter()
                .map(|&r| GgufMetadataValue::U32(r))
                .collect(),
        }),
    );
    b.add_string(KEY_ACOUSTIC_ENCODER_DEPTHS, ACOUSTIC_ENCODER_DEPTHS);
    b.add_f32(
        KEY_ACOUSTIC_LAYER_SCALE_INIT_VALUE,
        ACOUSTIC_LAYER_SCALE_INIT_VALUE,
    );
    b.add_f32(KEY_ACOUSTIC_WEIGHT_INIT_VALUE, ACOUSTIC_WEIGHT_INIT_VALUE);
    b.add_string(KEY_ACOUSTIC_LAYERNORM, ACOUSTIC_LAYERNORM);
    b.add_bool(
        KEY_ACOUSTIC_LAYERNORM_ELEMENTWISE_AFFINE,
        ACOUSTIC_LAYERNORM_ELEMENTWISE_AFFINE,
    );
    b.add_f32(KEY_ACOUSTIC_LAYERNORM_EPS, ACOUSTIC_LAYERNORM_EPS);
    b.add_string(KEY_ACOUSTIC_MIXER_LAYER, ACOUSTIC_MIXER_LAYER);
    b.add_string(KEY_ACOUSTIC_PAD_MODE, ACOUSTIC_PAD_MODE);
    b.add_bool(KEY_ACOUSTIC_DISABLE_LAST_NORM, ACOUSTIC_DISABLE_LAST_NORM);
    b.add_string(KEY_ACOUSTIC_CONV_NORM, ACOUSTIC_CONV_NORM);
    b.add_bool(KEY_ACOUSTIC_CONV_BIAS, ACOUSTIC_CONV_BIAS);
    b.add_f32(KEY_ACOUSTIC_CORPUS_NORMALIZE, ACOUSTIC_CORPUS_NORMALIZE);
    b.add_u32(KEY_ACOUSTIC_SAMPLE_RATE_HZ, ACOUSTIC_SAMPLE_RATE_HZ);

    // Diffusion head (config.json.diffusion_head_config.*). Only
    // `hidden_size` differs; the rest reuse the shared 1.5B
    // constants.
    b.add_u32(
        KEY_DIFFUSION_HEAD_HIDDEN_SIZE,
        RT_DIFFUSION_HEAD_HIDDEN_SIZE,
    );
    b.add_u32(KEY_DIFFUSION_HEAD_LAYERS, DIFFUSION_HEAD_LAYERS);
    b.add_f32(KEY_DIFFUSION_HEAD_FFN_RATIO, DIFFUSION_HEAD_FFN_RATIO);
    b.add_f32(KEY_DIFFUSION_HEAD_RMS_NORM_EPS, DIFFUSION_HEAD_RMS_NORM_EPS);
    b.add_u32(KEY_DIFFUSION_HEAD_LATENT_SIZE, DIFFUSION_HEAD_LATENT_SIZE);
    b.add_u32(
        KEY_DIFFUSION_HEAD_SPEECH_VAE_DIM,
        DIFFUSION_HEAD_SPEECH_VAE_DIM,
    );
    b.add_string(
        KEY_DIFFUSION_HEAD_PREDICTION_TYPE,
        DIFFUSION_HEAD_PREDICTION_TYPE,
    );
    b.add_string(
        KEY_DIFFUSION_HEAD_DIFFUSION_TYPE,
        DIFFUSION_HEAD_DIFFUSION_TYPE,
    );
    b.add_u32(
        KEY_DIFFUSION_HEAD_DDPM_NUM_STEPS,
        DIFFUSION_HEAD_DDPM_NUM_STEPS,
    );
    b.add_u32(
        KEY_DIFFUSION_HEAD_DDPM_NUM_INFERENCE_STEPS,
        DIFFUSION_HEAD_DDPM_NUM_INFERENCE_STEPS,
    );
    b.add_string(
        KEY_DIFFUSION_HEAD_DDPM_BETA_SCHEDULE,
        DIFFUSION_HEAD_DDPM_BETA_SCHEDULE,
    );
    b.add_u32(
        KEY_DIFFUSION_HEAD_DDPM_BATCH_MUL,
        DIFFUSION_HEAD_DDPM_BATCH_MUL,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::{GgufFile, GgufMetadataValue};

    fn minimal_safetensors_one_f32() -> Vec<u8> {
        // Single f32 tensor so the pass-through arm fires once and the
        // report counts a non-zero write. The tensor name mirrors an
        // upstream VibeVoice scaffold name (Qwen2 embed).
        let header =
            r#"{"model.embed_tokens.weight":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}}"#;
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(&[0u8; 24]);
        out
    }

    fn minimal_safetensors_no_tensors() -> Vec<u8> {
        let header = r#"{}"#;
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out
    }

    fn minimal_safetensors_one_f16() -> Vec<u8> {
        let header =
            r#"{"model.embed_tokens.weight":{"dtype":"F16","shape":[2,3],"data_offsets":[0,12]}}"#;
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(&[0u8; 12]);
        out
    }

    /// A single BF16 tensor — the upstream VibeVoice-1.5B serving format
    /// — so this hits the pass-through's `_ =>` arm and MUST land in
    /// `skipped_non_float` (with the loud "no float tensors" note).
    fn minimal_safetensors_one_bf16() -> Vec<u8> {
        let header =
            r#"{"model.embed_tokens.weight":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,12]}}"#;
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(&[0u8; 12]);
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

    #[test]
    fn arch_string_matches_runtime_constant() {
        // The two crates only share `vokra-core`, so this constant is the
        // sole handshake with `vokra-models::vibevoice::EXPECTED_ARCH`.
        assert_eq!(ARCH, "vibevoice");
    }

    #[test]
    fn arch_is_distinct_from_every_sibling_family() {
        // VibeVoice pairs a continuous VAE decoder with a DDPM diffusion
        // head; silently sharing an arch tag with any sibling would
        // misroute the runtime dispatch (VoxCPM uses UnifiedCFM flow
        // matching, not DDPM).
        assert_ne!(ARCH, "voxcpm2");
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

    #[test]
    fn name_string_matches_hf_release() {
        assert_eq!(NAME, "vibevoice-1.5b");
    }

    /// The transcribed constants must equal the primary-source values.
    /// Changing any of these silently mis-shapes the Qwen2 backbone /
    /// VAE handshake / diffusion-head sampler.
    #[test]
    fn transcribed_constants_match_primary_source() {
        // Qwen2 decoder LM (config.json.decoder_config.*).
        assert_eq!(DECODER_HIDDEN_DIM, 1536);
        assert_eq!(DECODER_N_LAYER, 28);
        assert_eq!(DECODER_N_HEAD, 12);
        assert_eq!(DECODER_N_HEAD_KV, 2);
        assert_eq!(DECODER_FFN_DIM, 8960);
        assert_eq!(DECODER_VOCAB_SIZE, 151_936);
        assert_eq!(DECODER_MAX_POSITIONS, 65_536);
        assert!((DECODER_ROPE_BASE - 1_000_000.0).abs() < 1e-3);
        assert!((DECODER_RMS_NORM_EPS - 1e-6).abs() < 1e-9);
        assert!((DECODER_ATTENTION_DROPOUT - 0.0).abs() < 1e-9);
        // `assertions_on_constants` and `bool_assert_comparison` collide on
        // primary-source constant `bool` pins — allow the constant-assert
        // form so the transcription is stated at the bool literal level
        // (the sibling models pin the same way).
        #[allow(clippy::assertions_on_constants)]
        {
            assert!(DECODER_TIE_WORD_EMBEDDINGS);
            assert!(!DECODER_USE_SLIDING_WINDOW);
        }
        assert_eq!(DECODER_MAX_WINDOW_LAYERS, 28);

        // Acoustic tokenizer (config.json.acoustic_tokenizer_config.*).
        assert_eq!(ACOUSTIC_CHANNELS, 1);
        #[allow(clippy::assertions_on_constants)]
        {
            assert!(ACOUSTIC_CAUSAL);
            assert!(ACOUSTIC_LAYERNORM_ELEMENTWISE_AFFINE);
            assert!(ACOUSTIC_DISABLE_LAST_NORM);
            assert!(ACOUSTIC_CONV_BIAS);
        }
        assert_eq!(ACOUSTIC_VAE_DIM, 64);
        assert!((ACOUSTIC_FIX_STD - 0.5).abs() < 1e-6);
        assert_eq!(ACOUSTIC_STD_DIST_TYPE, "gaussian");
        assert_eq!(ACOUSTIC_ENCODER_N_FILTERS, 32);
        assert_eq!(ACOUSTIC_DECODER_N_FILTERS, 32);
        assert_eq!(ACOUSTIC_ENCODER_RATIOS, [8, 5, 5, 4, 2, 2]);
        assert_eq!(ACOUSTIC_DECODER_RATIOS, [8, 5, 5, 4, 2, 2]);
        assert_eq!(ACOUSTIC_ENCODER_DEPTHS, "3-3-3-3-3-3-8");
        assert!((ACOUSTIC_LAYER_SCALE_INIT_VALUE - 1e-6).abs() < 1e-9);
        assert!((ACOUSTIC_WEIGHT_INIT_VALUE - 1e-2).abs() < 1e-9);
        assert_eq!(ACOUSTIC_LAYERNORM, "RMSNorm");
        assert!((ACOUSTIC_LAYERNORM_EPS - 1e-5).abs() < 1e-9);
        assert_eq!(ACOUSTIC_MIXER_LAYER, "depthwise_conv");
        assert_eq!(ACOUSTIC_PAD_MODE, "constant");
        assert_eq!(ACOUSTIC_CONV_NORM, "none");
        assert!((ACOUSTIC_CORPUS_NORMALIZE - 0.0).abs() < 1e-9);
        assert_eq!(ACOUSTIC_SAMPLE_RATE_HZ, 24_000);

        // Semantic tokenizer (config.json.semantic_tokenizer_config.*).
        assert_eq!(SEMANTIC_CHANNELS, 1);
        #[allow(clippy::assertions_on_constants)]
        {
            assert!(SEMANTIC_CAUSAL);
            assert!(SEMANTIC_CONV_BIAS);
        }
        assert_eq!(SEMANTIC_VAE_DIM, 128);
        assert!((SEMANTIC_FIX_STD - 0.0).abs() < 1e-9);
        assert_eq!(SEMANTIC_STD_DIST_TYPE, "none");
        assert_eq!(SEMANTIC_ENCODER_N_FILTERS, 32);
        assert_eq!(SEMANTIC_ENCODER_RATIOS, [8, 5, 5, 4, 2, 2]);
        assert_eq!(SEMANTIC_ENCODER_DEPTHS, "3-3-3-3-3-3-8");
        assert_eq!(SEMANTIC_LAYERNORM, "RMSNorm");
        assert!((SEMANTIC_LAYERNORM_EPS - 1e-5).abs() < 1e-9);
        assert_eq!(SEMANTIC_MIXER_LAYER, "depthwise_conv");

        // Diffusion head (config.json.diffusion_head_config.*).
        assert_eq!(DIFFUSION_HEAD_HIDDEN_SIZE, 1536);
        assert_eq!(DIFFUSION_HEAD_LAYERS, 4);
        assert!((DIFFUSION_HEAD_FFN_RATIO - 3.0).abs() < 1e-6);
        assert!((DIFFUSION_HEAD_RMS_NORM_EPS - 1e-5).abs() < 1e-9);
        assert_eq!(DIFFUSION_HEAD_LATENT_SIZE, 64);
        assert_eq!(DIFFUSION_HEAD_SPEECH_VAE_DIM, 64);
        assert_eq!(DIFFUSION_HEAD_PREDICTION_TYPE, "v_prediction");
        assert_eq!(DIFFUSION_HEAD_DIFFUSION_TYPE, "ddpm");
        assert_eq!(DIFFUSION_HEAD_DDPM_NUM_STEPS, 1000);
        assert_eq!(DIFFUSION_HEAD_DDPM_NUM_INFERENCE_STEPS, 20);
        assert_eq!(DIFFUSION_HEAD_DDPM_BETA_SCHEDULE, "cosine");
        assert_eq!(DIFFUSION_HEAD_DDPM_BATCH_MUL, 4);

        // Top-level shortcuts.
        assert_eq!(MODEL_FAMILY, "vibevoice");
        assert!((LM_FRAME_RATE_HZ - 7.5).abs() < 1e-6);

        // Compile-time algebra: GQA + VAE handshake + hidden-size
        // handshake + tokenizer-dim distinctness pins.
        const _: () = {
            // GQA well-formedness.
            assert!(DECODER_N_HEAD % DECODER_N_HEAD_KV == 0);
            // VAE handshake: diffusion head predicts velocity in the
            // acoustic VAE latent space.
            assert!(DIFFUSION_HEAD_LATENT_SIZE == ACOUSTIC_VAE_DIM);
            assert!(DIFFUSION_HEAD_SPEECH_VAE_DIM == ACOUSTIC_VAE_DIM);
            // Hidden-size handshake: cond_proj is a square linear.
            assert!(DIFFUSION_HEAD_HIDDEN_SIZE == DECODER_HIDDEN_DIM);
            // Tokenizer dims MUST differ (canonical 64 vs 128).
            assert!(ACOUSTIC_VAE_DIM != SEMANTIC_VAE_DIM);
            // Sample-rate + LM-frame-rate consistency: acoustic hop is
            // product(encoder_ratios) = 3200; 24000 / 3200 = 7.5.
            assert!(
                ACOUSTIC_ENCODER_RATIOS[0]
                    * ACOUSTIC_ENCODER_RATIOS[1]
                    * ACOUSTIC_ENCODER_RATIOS[2]
                    * ACOUSTIC_ENCODER_RATIOS[3]
                    * ACOUSTIC_ENCODER_RATIOS[4]
                    * ACOUSTIC_ENCODER_RATIOS[5]
                    == 3200
            );
            // Positive shapes.
            assert!(ACOUSTIC_SAMPLE_RATE_HZ > 0);
            assert!(DECODER_MAX_POSITIONS > 0);
        };
    }

    /// Cross-checks the converter's transcribed acoustic-VAE axes
    /// against the shared VAE seam in `vokra-ops` (the only crate this
    /// converter depends on besides `vokra-core`). The runtime-side
    /// handshake (`ARCH` == `vokra-models::vibevoice::EXPECTED_ARCH`)
    /// is exercised through the arch-string equality tests above; the
    /// numeric tokenizer axes are pinned here against the shared VAE
    /// primitive's default validation surface so a caller can
    /// synthesize a VibeVoice-shaped `ContinuousVaeConfig` from the
    /// converter's transcribed constants without cross-referencing
    /// `vokra-models`.
    #[test]
    fn constants_produce_a_well_formed_shared_vae_config() {
        // Build a `ContinuousVaeConfig` from the converter constants
        // (mirror-symmetric, 24 kHz in/out — matches the
        // `VibeVoiceConfig::acoustic_vae_config` builder on the runtime
        // side).
        let vae = vokra_ops::vae_continuous::ContinuousVaeConfig {
            sample_rate_hz: ACOUSTIC_SAMPLE_RATE_HZ,
            out_sample_rate_hz: ACOUSTIC_SAMPLE_RATE_HZ,
            encoder_dim: ACOUSTIC_ENCODER_N_FILTERS,
            encoder_rates: ACOUSTIC_ENCODER_RATIOS.to_vec(),
            latent_dim: ACOUSTIC_VAE_DIM,
            decoder_dim: ACOUSTIC_DECODER_N_FILTERS,
            decoder_rates: ACOUSTIC_DECODER_RATIOS.to_vec(),
            depthwise: ACOUSTIC_MIXER_LAYER == "depthwise_conv",
            use_noise_block: false,
            // VibeVoice's mirror-symmetric acoustic VAE has no
            // bandwidth-adaptive decoder head (that axis is VoxCPM2-2B-only
            // so far). Keep `None` here to mirror the runtime builder in
            // `crates/vokra-models/src/vibevoice/mod.rs::acoustic_vae_config`.
            sr_bin_boundaries: None,
        };
        // The shared VAE seam accepts the transcribed axes without
        // clamping / rejecting anything — that is the numeric
        // handshake the converter needs to guarantee.
        vae.validate_for_forward().expect("well-formed VAE config");
        // Encoder hop_length = product(encoder_ratios) = 3200.
        assert_eq!(vae.hop_length(), Some(3200));
        // 24_000 / 3200 = 7.5 Hz — matches the top-level
        // `KEY_LM_FRAME_RATE_HZ` constant.
        let frame_rate = vae.frame_rate_hz().expect("finite frame rate");
        assert!(
            (frame_rate - LM_FRAME_RATE_HZ).abs() < 1e-6,
            "derived frame rate {frame_rate} != transcribed LM_FRAME_RATE_HZ {LM_FRAME_RATE_HZ}"
        );
    }

    #[test]
    fn round_trip_carries_arch_chunks_and_provenance() {
        let (builder, report) = convert(minimal_safetensors_one_f32()).expect("convert");
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
            Some(NAME)
        );
        assert_eq!(get_string(&file, KEY_MODEL_FAMILY), MODEL_FAMILY);

        // Every transcribed U32 hparam round-trips verbatim under the
        // `vokra.vibevoice.*` prefix.
        for (key, want) in [
            (KEY_ACOUSTIC_VAE_DIM, ACOUSTIC_VAE_DIM),
            (KEY_SEMANTIC_VAE_DIM, SEMANTIC_VAE_DIM),
            (KEY_DECODER_HIDDEN_DIM, DECODER_HIDDEN_DIM),
            (KEY_DECODER_N_LAYER, DECODER_N_LAYER),
            (KEY_DECODER_N_HEAD, DECODER_N_HEAD),
            (KEY_DECODER_N_HEAD_KV, DECODER_N_HEAD_KV),
            (KEY_DECODER_FFN_DIM, DECODER_FFN_DIM),
            (KEY_DECODER_VOCAB_SIZE, DECODER_VOCAB_SIZE),
            (KEY_DECODER_MAX_POSITIONS, DECODER_MAX_POSITIONS),
            (KEY_DECODER_MAX_WINDOW_LAYERS, DECODER_MAX_WINDOW_LAYERS),
            (KEY_ACOUSTIC_CHANNELS, ACOUSTIC_CHANNELS),
            (KEY_ACOUSTIC_VAE_DIM_INNER, ACOUSTIC_VAE_DIM),
            (KEY_ACOUSTIC_ENCODER_N_FILTERS, ACOUSTIC_ENCODER_N_FILTERS),
            (KEY_ACOUSTIC_DECODER_N_FILTERS, ACOUSTIC_DECODER_N_FILTERS),
            (KEY_ACOUSTIC_SAMPLE_RATE_HZ, ACOUSTIC_SAMPLE_RATE_HZ),
            (KEY_SEMANTIC_CHANNELS, SEMANTIC_CHANNELS),
            (KEY_SEMANTIC_VAE_DIM_INNER, SEMANTIC_VAE_DIM),
            (KEY_SEMANTIC_ENCODER_N_FILTERS, SEMANTIC_ENCODER_N_FILTERS),
            (KEY_DIFFUSION_HEAD_HIDDEN_SIZE, DIFFUSION_HEAD_HIDDEN_SIZE),
            (KEY_DIFFUSION_HEAD_LAYERS, DIFFUSION_HEAD_LAYERS),
            (KEY_DIFFUSION_HEAD_LATENT_SIZE, DIFFUSION_HEAD_LATENT_SIZE),
            (
                KEY_DIFFUSION_HEAD_SPEECH_VAE_DIM,
                DIFFUSION_HEAD_SPEECH_VAE_DIM,
            ),
            (
                KEY_DIFFUSION_HEAD_DDPM_NUM_STEPS,
                DIFFUSION_HEAD_DDPM_NUM_STEPS,
            ),
            (
                KEY_DIFFUSION_HEAD_DDPM_NUM_INFERENCE_STEPS,
                DIFFUSION_HEAD_DDPM_NUM_INFERENCE_STEPS,
            ),
            (
                KEY_DIFFUSION_HEAD_DDPM_BATCH_MUL,
                DIFFUSION_HEAD_DDPM_BATCH_MUL,
            ),
        ] {
            assert_eq!(get_u32(&file, key), want, "{key}");
        }

        // F32 constants round-trip too.
        assert!((get_f32(&file, KEY_DECODER_ROPE_BASE) - DECODER_ROPE_BASE).abs() < 1e-3);
        assert!((get_f32(&file, KEY_DECODER_RMS_NORM_EPS) - DECODER_RMS_NORM_EPS).abs() < 1e-9);
        assert!(
            (get_f32(&file, KEY_DECODER_ATTENTION_DROPOUT) - DECODER_ATTENTION_DROPOUT).abs()
                < 1e-9
        );
        assert!((get_f32(&file, KEY_ACOUSTIC_LAYERNORM_EPS) - ACOUSTIC_LAYERNORM_EPS).abs() < 1e-9);
        assert!((get_f32(&file, KEY_ACOUSTIC_FIX_STD) - ACOUSTIC_FIX_STD).abs() < 1e-6);
        assert!(
            (get_f32(&file, KEY_DIFFUSION_HEAD_FFN_RATIO) - DIFFUSION_HEAD_FFN_RATIO).abs() < 1e-6
        );
        assert!(
            (get_f32(&file, KEY_DIFFUSION_HEAD_RMS_NORM_EPS) - DIFFUSION_HEAD_RMS_NORM_EPS).abs()
                < 1e-9
        );
        assert!((get_f32(&file, KEY_LM_FRAME_RATE_HZ) - LM_FRAME_RATE_HZ).abs() < 1e-6);

        // Bool constants round-trip.
        assert_eq!(
            get_bool(&file, KEY_DECODER_TIE_WORD_EMBEDDINGS),
            DECODER_TIE_WORD_EMBEDDINGS
        );
        assert_eq!(
            get_bool(&file, KEY_DECODER_USE_SLIDING_WINDOW),
            DECODER_USE_SLIDING_WINDOW
        );
        assert_eq!(get_bool(&file, KEY_ACOUSTIC_CAUSAL), ACOUSTIC_CAUSAL);
        assert_eq!(
            get_bool(&file, KEY_ACOUSTIC_DISABLE_LAST_NORM),
            ACOUSTIC_DISABLE_LAST_NORM
        );
        assert_eq!(get_bool(&file, KEY_ACOUSTIC_CONV_BIAS), ACOUSTIC_CONV_BIAS);
        assert_eq!(get_bool(&file, KEY_SEMANTIC_CAUSAL), SEMANTIC_CAUSAL);
        assert_eq!(get_bool(&file, KEY_SEMANTIC_CONV_BIAS), SEMANTIC_CONV_BIAS);

        // String constants round-trip.
        assert_eq!(
            get_string(&file, KEY_ACOUSTIC_STD_DIST_TYPE),
            ACOUSTIC_STD_DIST_TYPE
        );
        assert_eq!(
            get_string(&file, KEY_ACOUSTIC_ENCODER_DEPTHS),
            ACOUSTIC_ENCODER_DEPTHS
        );
        assert_eq!(
            get_string(&file, KEY_ACOUSTIC_LAYERNORM),
            ACOUSTIC_LAYERNORM
        );
        assert_eq!(
            get_string(&file, KEY_SEMANTIC_STD_DIST_TYPE),
            SEMANTIC_STD_DIST_TYPE
        );
        assert_eq!(
            get_string(&file, KEY_DIFFUSION_HEAD_PREDICTION_TYPE),
            DIFFUSION_HEAD_PREDICTION_TYPE
        );
        assert_eq!(
            get_string(&file, KEY_DIFFUSION_HEAD_DIFFUSION_TYPE),
            DIFFUSION_HEAD_DIFFUSION_TYPE
        );
        assert_eq!(
            get_string(&file, KEY_DIFFUSION_HEAD_DDPM_BETA_SCHEDULE),
            DIFFUSION_HEAD_DDPM_BETA_SCHEDULE
        );

        // Provenance: MIT permissive (end-to-end).
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_MODEL_ID)
                .and_then(|v| v.as_str()),
            Some(NAME)
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some("mit")
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Permissive.as_str())
        );
    }

    #[test]
    fn zero_tensor_conversion_surfaces_a_loud_note() {
        let (_, report) = convert(minimal_safetensors_no_tensors()).expect("convert");
        assert_eq!(report.written, 0);
        assert!(
            report.notes.iter().any(|n| n.contains("no float tensors")),
            "zero-tensor conversion must emit a loud note: {:?}",
            report.notes
        );
    }

    /// Pins the F16 leg of the `GgmlType::F32 | GgmlType::F16` union
    /// match arm.
    #[test]
    fn f16_tensor_passes_through_verbatim() {
        let (builder, report) = convert(minimal_safetensors_one_f16()).expect("convert");
        assert_eq!(report.written, 1, "F16 must reach the pass-through arm");
        assert_eq!(
            report.skipped_non_float, 0,
            "F16 must not land in the skipped counter"
        );

        // The tensor survives the round trip under its upstream name and
        // preserves its F16 dtype.
        let out = builder.to_bytes().expect("serialize");
        let file = GgufFile::parse(out).expect("parse");
        let info = file
            .tensor_info("model.embed_tokens.weight")
            .expect("tensor present");
        assert_eq!(info.dtype, GgmlType::F16);
        assert_eq!(info.dimensions, vec![2, 3]);
        assert_eq!(file.tensor_bytes(info).len(), 12);
    }

    /// Pins the BF16 leg of the `GgmlType::F32 | GgmlType::F16 |
    /// GgmlType::BF16` union: BF16 (the upstream serving format for
    /// VibeVoice-1.5B, `torch_dtype: bfloat16`) must reach the
    /// pass-through arm, emit as GGUF type 30 verbatim, and increment
    /// `bf16_passthrough`. Mirror of qwen3-tts /
    /// `bf16_tensor_passes_through_verbatim` and moshi's `assert_eq!(
    /// info.dtype, GgmlType::BF16, "no convert-time widening")`.
    ///
    /// Rewritten 2026-07-25 from the earlier "counted as skipped" pin —
    /// the earlier pin encoded the pre-BF16-fix scaffold posture.
    /// Removing the pin outright would let a latent silent-widen slip in
    /// undetected; rewriting to the passes-through invariant keeps the
    /// regression guard.
    #[test]
    fn bf16_tensor_passes_through_verbatim() {
        let (builder, report) = convert(minimal_safetensors_one_bf16()).expect("convert");
        assert_eq!(
            report.written, 1,
            "BF16 must reach the pass-through arm and increment `written`"
        );
        assert_eq!(
            report.skipped_non_float, 0,
            "BF16 must not land in the skipped counter"
        );
        assert_eq!(
            report.bf16_passthrough, 1,
            "BF16 subset counter must record the pass-through"
        );
        // The tensor survives the round trip under its upstream name and
        // preserves its BF16 dtype (no convert-time widening — runtime
        // widens on load via `decode_bf16`).
        let out = builder.to_bytes().expect("serialize");
        let file = GgufFile::parse(out).expect("parse");
        let info = file
            .tensor_info("model.embed_tokens.weight")
            .expect("BF16 tensor must be present after pass-through");
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
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH)
        );
    }

    /// Pins `SafetensorsFile::parse(bytes)?` error propagation.
    #[test]
    fn malformed_input_returns_parse_error() {
        // Case 1: empty buffer.
        let err = convert(Vec::new()).expect_err("empty buffer must be rejected");
        assert!(
            matches!(err, ConvertError::Parse(_)),
            "expected ConvertError::Parse, got {err:?}"
        );

        // Case 2: declared header length runs off the end of the buffer.
        let mut truncated = Vec::new();
        truncated.extend_from_slice(&1024u64.to_le_bytes());
        truncated.extend_from_slice(b"{}");
        let err = convert(truncated).expect_err("truncated header must be rejected");
        assert!(matches!(err, ConvertError::Parse(_)));

        // Case 3: valid length prefix but malformed JSON body.
        let bad_json = b"{not-json";
        let mut bad = Vec::new();
        bad.extend_from_slice(&(bad_json.len() as u64).to_le_bytes());
        bad.extend_from_slice(bad_json);
        let err = convert(bad).expect_err("malformed JSON must be rejected");
        assert!(matches!(err, ConvertError::Parse(_)));
    }

    /// Every `vokra.vibevoice.*` key uses the documented prefix — a
    /// regression where a key crossed into another model's namespace
    /// (e.g. `vokra.voxcpm2.*`) would still round-trip in isolation but
    /// would misroute at the runtime dispatch layer.
    #[test]
    fn every_metadata_key_carries_a_documented_prefix() {
        for key in [
            KEY_MODEL_FAMILY,
            KEY_ACOUSTIC_VAE_DIM,
            KEY_SEMANTIC_VAE_DIM,
            KEY_LM_FRAME_RATE_HZ,
            KEY_DECODER_HIDDEN_DIM,
            KEY_DECODER_N_LAYER,
            KEY_DECODER_N_HEAD,
            KEY_DECODER_N_HEAD_KV,
            KEY_DECODER_FFN_DIM,
            KEY_DECODER_VOCAB_SIZE,
            KEY_DECODER_MAX_POSITIONS,
            KEY_DECODER_ROPE_BASE,
            KEY_DECODER_RMS_NORM_EPS,
            KEY_DECODER_ATTENTION_DROPOUT,
            KEY_DECODER_TIE_WORD_EMBEDDINGS,
            KEY_DECODER_USE_SLIDING_WINDOW,
            KEY_DECODER_MAX_WINDOW_LAYERS,
            KEY_ACOUSTIC_CHANNELS,
            KEY_ACOUSTIC_CAUSAL,
            KEY_ACOUSTIC_VAE_DIM_INNER,
            KEY_ACOUSTIC_FIX_STD,
            KEY_ACOUSTIC_STD_DIST_TYPE,
            KEY_ACOUSTIC_ENCODER_N_FILTERS,
            KEY_ACOUSTIC_DECODER_N_FILTERS,
            KEY_ACOUSTIC_ENCODER_RATIOS,
            KEY_ACOUSTIC_DECODER_RATIOS,
            KEY_ACOUSTIC_ENCODER_DEPTHS,
            KEY_ACOUSTIC_LAYER_SCALE_INIT_VALUE,
            KEY_ACOUSTIC_WEIGHT_INIT_VALUE,
            KEY_ACOUSTIC_LAYERNORM,
            KEY_ACOUSTIC_LAYERNORM_ELEMENTWISE_AFFINE,
            KEY_ACOUSTIC_LAYERNORM_EPS,
            KEY_ACOUSTIC_MIXER_LAYER,
            KEY_ACOUSTIC_PAD_MODE,
            KEY_ACOUSTIC_DISABLE_LAST_NORM,
            KEY_ACOUSTIC_CONV_NORM,
            KEY_ACOUSTIC_CONV_BIAS,
            KEY_ACOUSTIC_CORPUS_NORMALIZE,
            KEY_ACOUSTIC_SAMPLE_RATE_HZ,
            KEY_SEMANTIC_CHANNELS,
            KEY_SEMANTIC_CAUSAL,
            KEY_SEMANTIC_VAE_DIM_INNER,
            KEY_SEMANTIC_FIX_STD,
            KEY_SEMANTIC_STD_DIST_TYPE,
            KEY_SEMANTIC_ENCODER_N_FILTERS,
            KEY_SEMANTIC_ENCODER_RATIOS,
            KEY_SEMANTIC_ENCODER_DEPTHS,
            KEY_SEMANTIC_LAYERNORM,
            KEY_SEMANTIC_LAYERNORM_EPS,
            KEY_SEMANTIC_MIXER_LAYER,
            KEY_SEMANTIC_CONV_BIAS,
            KEY_DIFFUSION_HEAD_HIDDEN_SIZE,
            KEY_DIFFUSION_HEAD_LAYERS,
            KEY_DIFFUSION_HEAD_FFN_RATIO,
            KEY_DIFFUSION_HEAD_RMS_NORM_EPS,
            KEY_DIFFUSION_HEAD_LATENT_SIZE,
            KEY_DIFFUSION_HEAD_SPEECH_VAE_DIM,
            KEY_DIFFUSION_HEAD_PREDICTION_TYPE,
            KEY_DIFFUSION_HEAD_DIFFUSION_TYPE,
            KEY_DIFFUSION_HEAD_DDPM_NUM_STEPS,
            KEY_DIFFUSION_HEAD_DDPM_NUM_INFERENCE_STEPS,
            KEY_DIFFUSION_HEAD_DDPM_BETA_SCHEDULE,
            KEY_DIFFUSION_HEAD_DDPM_BATCH_MUL,
        ] {
            assert!(
                key.starts_with("vokra.vibevoice."),
                "{key} must live under the vokra.vibevoice.* prefix"
            );
        }
    }

    // === VibeVoice-Realtime-0.5B (streaming variant) tests ===================
    // Added 2026-08-01 for the `microsoft/VibeVoice-Realtime-0.5B` publish.

    #[test]
    fn realtime_variant_arch_is_distinct_from_base() {
        assert_eq!(VibeVoiceVariant::Base15B.arch(), "vibevoice");
        assert_eq!(VibeVoiceVariant::Realtime05B.arch(), "vibevoice_streaming");
        assert_ne!(
            VibeVoiceVariant::Base15B.arch(),
            VibeVoiceVariant::Realtime05B.arch(),
            "distinct arch tags avoid runtime dispatch misroute"
        );
    }

    #[test]
    fn realtime_variant_name_and_upstream_match_primary_source() {
        assert_eq!(
            VibeVoiceVariant::Realtime05B.name(),
            "vibevoice-realtime-0.5b"
        );
        assert_eq!(
            VibeVoiceVariant::Realtime05B.upstream_hf(),
            "microsoft/VibeVoice-Realtime-0.5B"
        );
        // Base variant handles must not drift under the same enum.
        assert_eq!(VibeVoiceVariant::Base15B.name(), "vibevoice-1.5b");
        assert_eq!(
            VibeVoiceVariant::Base15B.upstream_hf(),
            "microsoft/VibeVoice-1.5B"
        );
    }

    /// Realtime-0.5B transcribed constants must equal the primary-source
    /// values fetched from `huggingface.co/microsoft/VibeVoice-Realtime-
    /// 0.5B/raw/main/config.json` (2026-08-01 -- CLAUDE.md 「ハルシネー
    /// ション厳禁」). Mirror of `transcribed_constants_match_primary_
    /// source` above for the 1.5B variant.
    #[test]
    fn realtime_transcribed_constants_match_primary_source() {
        assert_eq!(RT_MODEL_FAMILY, "vibevoice_streaming");
        // Qwen2 decoder LM overrides (config.json.decoder_config.*).
        assert_eq!(RT_DECODER_HIDDEN_DIM, 896);
        assert_eq!(RT_DECODER_N_LAYER, 24);
        assert_eq!(RT_DECODER_N_HEAD, 14);
        assert_eq!(RT_DECODER_FFN_DIM, 4864);
        assert_eq!(RT_DECODER_MAX_POSITIONS, 8_192);
        assert_eq!(RT_DECODER_MAX_WINDOW_LAYERS, 24);
        // The tie_word_embeddings flag flips between the two variants --
        // 1.5B ties (true), Realtime unties (false). Getting this wrong
        // silently loses the LM head weights for a Realtime bind.
        #[allow(clippy::assertions_on_constants)]
        {
            assert!(!RT_DECODER_TIE_WORD_EMBEDDINGS);
        }
        // Diffusion-head override (config.json.diffusion_head_config.
        // hidden_size).
        assert_eq!(RT_DIFFUSION_HEAD_HIDDEN_SIZE, 896);
        // Streaming-only top-level axis.
        assert_eq!(RT_TTS_BACKBONE_NUM_HIDDEN_LAYERS, 20);
        // Shared with 1.5B -- this test does not re-pin them (their pins
        // live in `transcribed_constants_match_primary_source`) but the
        // handshake algebra depends on them, so re-assert the ones the
        // algebra reads:
        assert_eq!(DECODER_N_HEAD_KV, 2);
        assert_eq!(DIFFUSION_HEAD_LATENT_SIZE, 64);
        assert_eq!(ACOUSTIC_VAE_DIM, 64);
    }

    #[test]
    fn realtime_round_trip_carries_arch_chunks_and_provenance() {
        let (builder, report) =
            convert_realtime_05b(minimal_safetensors_one_f32()).expect("convert");
        assert_eq!(report.written, 1);
        assert_eq!(report.skipped_non_float, 0);

        let out = builder.to_bytes().expect("serialize");
        let file = GgufFile::parse(out).expect("parse");
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some("vibevoice_streaming")
        );
        assert_eq!(
            file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
            Some("vibevoice-realtime-0.5b")
        );
        assert_eq!(get_string(&file, KEY_MODEL_FAMILY), "vibevoice_streaming");

        // Overriding U32 axes round-trip verbatim.
        for (key, want) in [
            (KEY_DECODER_HIDDEN_DIM, RT_DECODER_HIDDEN_DIM),
            (KEY_DECODER_N_LAYER, RT_DECODER_N_LAYER),
            (KEY_DECODER_N_HEAD, RT_DECODER_N_HEAD),
            (KEY_DECODER_N_HEAD_KV, DECODER_N_HEAD_KV),
            (KEY_DECODER_FFN_DIM, RT_DECODER_FFN_DIM),
            (KEY_DECODER_VOCAB_SIZE, DECODER_VOCAB_SIZE),
            (KEY_DECODER_MAX_POSITIONS, RT_DECODER_MAX_POSITIONS),
            (KEY_DECODER_MAX_WINDOW_LAYERS, RT_DECODER_MAX_WINDOW_LAYERS),
            (KEY_ACOUSTIC_VAE_DIM, ACOUSTIC_VAE_DIM),
            (KEY_ACOUSTIC_VAE_DIM_INNER, ACOUSTIC_VAE_DIM),
            (KEY_ACOUSTIC_SAMPLE_RATE_HZ, ACOUSTIC_SAMPLE_RATE_HZ),
            (
                KEY_DIFFUSION_HEAD_HIDDEN_SIZE,
                RT_DIFFUSION_HEAD_HIDDEN_SIZE,
            ),
            (KEY_DIFFUSION_HEAD_LAYERS, DIFFUSION_HEAD_LAYERS),
            (KEY_DIFFUSION_HEAD_LATENT_SIZE, DIFFUSION_HEAD_LATENT_SIZE),
            (
                KEY_DIFFUSION_HEAD_SPEECH_VAE_DIM,
                DIFFUSION_HEAD_SPEECH_VAE_DIM,
            ),
            (
                KEY_DIFFUSION_HEAD_DDPM_NUM_STEPS,
                DIFFUSION_HEAD_DDPM_NUM_STEPS,
            ),
            (
                KEY_TTS_BACKBONE_NUM_HIDDEN_LAYERS,
                RT_TTS_BACKBONE_NUM_HIDDEN_LAYERS,
            ),
        ] {
            assert_eq!(get_u32(&file, key), want, "{key}");
        }

        // tie_word_embeddings flip must survive.
        assert_eq!(
            get_bool(&file, KEY_DECODER_TIE_WORD_EMBEDDINGS),
            RT_DECODER_TIE_WORD_EMBEDDINGS
        );

        // Provenance: MIT permissive (end-to-end).
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_MODEL_ID)
                .and_then(|v| v.as_str()),
            Some("vibevoice-realtime-0.5b")
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some("mit")
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Permissive.as_str())
        );
    }

    /// The streaming variant is acoustic-tokenizer-only -- no
    /// `vokra.vibevoice.semantic.*` key may be emitted on a Realtime
    /// GGUF, or the runtime will read the wrong-shape config.
    #[test]
    fn realtime_gguf_carries_no_semantic_tokenizer_keys() {
        let (builder, _) = convert_realtime_05b(minimal_safetensors_one_f32()).expect("convert");
        let out = builder.to_bytes().expect("serialize");
        let file = GgufFile::parse(out).expect("parse");
        for key in [
            KEY_SEMANTIC_CHANNELS,
            KEY_SEMANTIC_CAUSAL,
            KEY_SEMANTIC_VAE_DIM,
            KEY_SEMANTIC_VAE_DIM_INNER,
            KEY_SEMANTIC_FIX_STD,
            KEY_SEMANTIC_STD_DIST_TYPE,
            KEY_SEMANTIC_ENCODER_N_FILTERS,
            KEY_SEMANTIC_ENCODER_RATIOS,
            KEY_SEMANTIC_ENCODER_DEPTHS,
            KEY_SEMANTIC_LAYERNORM,
            KEY_SEMANTIC_LAYERNORM_EPS,
            KEY_SEMANTIC_MIXER_LAYER,
            KEY_SEMANTIC_CONV_BIAS,
        ] {
            assert!(
                file.get(key).is_none(),
                "{key}: streaming variant must NOT emit any semantic \
                 tokenizer key"
            );
        }
    }

    /// BF16 (the upstream serving format) rides the pass-through arm
    /// on the Realtime path too -- mirror of
    /// `bf16_tensor_passes_through_verbatim` for the 1.5B path.
    #[test]
    fn realtime_bf16_tensor_passes_through_verbatim() {
        let (builder, report) =
            convert_realtime_05b(minimal_safetensors_one_bf16()).expect("convert");
        assert_eq!(report.written, 1);
        assert_eq!(report.bf16_passthrough, 1);
        assert_eq!(report.skipped_non_float, 0);
        let out = builder.to_bytes().expect("serialize");
        let file = GgufFile::parse(out).expect("parse");
        let info = file
            .tensor_info("model.embed_tokens.weight")
            .expect("BF16 tensor must be present after pass-through");
        assert_eq!(
            info.dtype,
            GgmlType::BF16,
            "no convert-time widening -- GGUF dtype must remain BF16"
        );
    }

    /// Every Realtime-added `vokra.vibevoice.*` key must carry the
    /// documented prefix (matches the sibling
    /// `every_metadata_key_carries_a_documented_prefix` guard for
    /// 1.5B). Only KEY_TTS_BACKBONE_NUM_HIDDEN_LAYERS is new; the rest
    /// are shared with the 1.5B path which already pins them.
    #[test]
    fn realtime_added_key_carries_a_documented_prefix() {
        assert!(
            KEY_TTS_BACKBONE_NUM_HIDDEN_LAYERS.starts_with("vokra.vibevoice."),
            "streaming-specific key must live under vokra.vibevoice.* prefix"
        );
    }

    /// Zero-tensor Realtime input must surface the same loud note as
    /// the 1.5B path (metadata-only GGUF; runtime refuses to bind
    /// weights per FR-EX-08).
    #[test]
    fn realtime_zero_tensor_conversion_surfaces_a_loud_note() {
        let (_, report) = convert_realtime_05b(minimal_safetensors_no_tensors()).expect("convert");
        assert_eq!(report.written, 0);
        assert!(
            report.notes.iter().any(|n| n.contains("no float tensors")),
            "zero-tensor Realtime conversion must emit a loud note: {:?}",
            report.notes
        );
    }
}
