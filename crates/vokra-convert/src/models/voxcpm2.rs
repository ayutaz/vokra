//! **VoxCPM-0.5B**: safetensors checkpoint → GGUF conversion
//! (SoTA plan Phase 4, 2026-07-24).
//!
//! Input: the upstream `openbmb/VoxCPM-0.5B` release — `model.safetensors`
//! (BF16). Output: a GGUF carrying every float tensor plus the
//! `vokra.voxcpm2.*`, `vokra.vae_continuous.*`, and `vokra.model.*` /
//! `vokra.provenance.*` metadata chunks the native VoxCPM implementation
//! (`crates/vokra-models/src/voxcpm2/`) reads.
//!
//! # What is transcribed vs. shape-driven
//!
//! - **Transcribed constants** — every hparam of the
//!   `vokra.voxcpm2.*` chunk group is transcribed **verbatim** from the
//!   primary source `config.json` at
//!   `huggingface.co/openbmb/VoxCPM-0.5B/raw/main/config.json` (fetched
//!   2026-07-24 — CLAUDE.md「ハルシネーション厳禁」) plus the AudioVAE V2
//!   axes from `openbmb/VoxCPM/src/voxcpm/modules/audiovae/audio_vae_v2.py`
//!   (the release ships hparams as PyTorch defaults on
//!   `AudioVAEConfig(BaseModel)`; there is no separate `audio_vae.json`).
//! - **Nested config blocks** — VoxCPM splits its `config.json` into
//!   `lm_config.*` (MiniCPM-4 backbone), `encoder_config.*`,
//!   `dit_config.*` (with a nested `cfm_config.*`) blocks. All are
//!   transcribed in full.
//! - **VAE handshake** — `feat_dim` (top-level `config.json.feat_dim`)
//!   must equal `vae.latent_dim` (upstream `AudioVAEConfig.latent_dim`);
//!   the runtime rejects a mismatch loudly at load per FR-EX-08 via
//!   [`vokra_models::voxcpm2::VoxCpm2Config::validate_for_forward_with_vae`].
//!
//! # No side-car config
//!
//! VoxCPM-0.5B ships a real upstream `config.json`, but this converter
//! takes **no** `--config` path today because every field is fixed for
//! the 0.5B release and byte-parallel to the transcribed constants below.
//! A future variant (0.5B-CustomVoice / 1.5B family) that reshapes the
//! backbone or the VAE would demand `--config`; this converter fails
//! loudly if a tensor shape disagrees with the transcribed axes at
//! runtime bind time (FR-EX-08).
//!
//! # Tensor naming contract
//!
//! GGUF tensor names are the **upstream safetensors names verbatim**
//! (the CSM / Kokoro / CosyVoice2 / Chatterbox / Qwen3-TTS contract).
//! Real-weight binding is a follow-up wave gated on the upstream tensor-
//! name manifest fetch; this converter passes every F32 / F16 tensor
//! through unchanged so a future `VoxCpm2Weights::from_gguf` can walk
//! the same names.
//!
//! # BF16 posture
//!
//! The upstream VoxCPM-0.5B release is served in **BF16**
//! (`config.json.dtype = "bfloat16"`). Today's F32 / F16 pass-through
//! arm hits `skipped_non_float` on BF16 tensors and the converter
//! surfaces the loud "no float tensors" note. Pre-widen offline to F32
//! (via a small prepare script — the CSM / Kokoro pattern) or wait for
//! the streaming BF16 pass-through path (T29-equivalent — the Moshi /
//! Kyutai STT pattern) to convert the release build directly.
//!
//! # No ONNX (permanent)
//!
//! VoxCPM-0.5B is distributed as safetensors + a Python pipeline; this
//! converter **never** touches ONNX (FR-LD-05); the pipeline is
//! re-implemented natively in `crates/vokra-models/src/voxcpm2/`
//! (whisper.cpp 型 self re-implementation, CLAUDE.md 設計判断 4).

use vokra_core::LicenseClass;
use vokra_core::gguf::{
    GgmlType, GgufArray, GgufBuilder, GgufMetadataValue, GgufValueType, chunks,
};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` for VoxCPM-0.5B GGUFs — kept in sync with the
/// runtime constant `vokra-models::voxcpm2::EXPECTED_ARCH`.
/// Intentionally **distinct** from every sibling arch tag because
/// VoxCPM's terminal decoding hop is a continuous VAE decoder — not
/// HiFTNet / HiFT-GAN (CosyVoice2/3, Chatterbox) and not any RVQ / FSQ
/// codec (Qwen3-TTS, SNAC family, Kyutai STT, Moshi, CSM, Voxtral, Dia,
/// Zonos). Silently sharing an arch tag would mis-route the runtime
/// dispatch.
pub(crate) const ARCH: &str = "voxcpm2";

/// `vokra.model.name` value written for the canonical VoxCPM-0.5B GGUF.
pub(crate) const NAME: &str = "voxcpm-0.5b";

// --- vokra.voxcpm2.* metadata keys (kept as constants in the converter;
// the runtime side lives in `crates/vokra-models/src/voxcpm2/mod.rs` —
// the two crates share only `vokra-core`, so the cross-crate constant
// duplication rule the CSM / CosyVoice2 / Kokoro / Chatterbox / Qwen3-TTS
// family converters use applies) -----------------------------------------

// Top-level
const KEY_FEAT_DIM: &str = "vokra.voxcpm2.feat_dim";
const KEY_PATCH_SIZE: &str = "vokra.voxcpm2.patch_size";
const KEY_RESIDUAL_LM_N_LAYER: &str = "vokra.voxcpm2.residual_lm_n_layer";
const KEY_SCALAR_QUANT_LATENT_DIM: &str = "vokra.voxcpm2.scalar_quantization.latent_dim";
const KEY_SCALAR_QUANT_SCALE: &str = "vokra.voxcpm2.scalar_quantization.scale";
const KEY_MAX_LENGTH: &str = "vokra.voxcpm2.max_length";
const KEY_MODEL_FAMILY: &str = "vokra.voxcpm2.model_family";

// LM backbone axes — config.json.lm_config.*
const KEY_LM_HIDDEN_DIM: &str = "vokra.voxcpm2.lm.hidden_dim";
const KEY_LM_N_LAYER: &str = "vokra.voxcpm2.lm.n_layer";
const KEY_LM_N_HEAD: &str = "vokra.voxcpm2.lm.n_head";
const KEY_LM_N_HEAD_KV: &str = "vokra.voxcpm2.lm.n_head_kv";
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

// DiT axes — config.json.dit_config.*
const KEY_DIT_HIDDEN_DIM: &str = "vokra.voxcpm2.dit.hidden_dim";
const KEY_DIT_FFN_DIM: &str = "vokra.voxcpm2.dit.ffn_dim";
const KEY_DIT_N_HEAD: &str = "vokra.voxcpm2.dit.n_head";
const KEY_DIT_N_LAYER: &str = "vokra.voxcpm2.dit.n_layer";

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

// --- Transcribed constants (primary sources:
// `huggingface.co/openbmb/VoxCPM-0.5B/raw/main/config.json` +
// `openbmb/VoxCPM/src/voxcpm/modules/audiovae/audio_vae_v2.py` —
// `AudioVAEConfig(BaseModel)` defaults. Fetched 2026-07-24 — CLAUDE.md
// 「ハルシネーション厳禁」) ------------------------------------------

// LM backbone (config.json.lm_config.*)
const LM_HIDDEN_DIM: u32 = 1024;
const LM_N_LAYER: u32 = 24;
const LM_N_HEAD: u32 = 16;
const LM_N_HEAD_KV: u32 = 2;
const LM_FFN_DIM: u32 = 4096;
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

// Encoder (config.json.encoder_config.*)
const ENC_HIDDEN_DIM: u32 = 1024;
const ENC_FFN_DIM: u32 = 4096;
const ENC_N_HEAD: u32 = 16;
const ENC_N_LAYER: u32 = 4;

// DiT (config.json.dit_config.*)
const DIT_HIDDEN_DIM: u32 = 1024;
const DIT_FFN_DIM: u32 = 4096;
const DIT_N_HEAD: u32 = 16;
const DIT_N_LAYER: u32 = 4;

// CFM (config.json.dit_config.cfm_config.*)
const CFM_SIGMA_MIN: f32 = 1e-6;
const CFM_SOLVER: &str = "euler";
const CFM_T_SCHEDULER: &str = "log-norm";
const CFM_INFERENCE_CFG_RATE: f32 = 2.0;

// Top-level (config.json.*)
const FEAT_DIM: u32 = 64;
const PATCH_SIZE: u32 = 2;
const RESIDUAL_LM_N_LAYER: u32 = 6;
const SCALAR_QUANT_LATENT_DIM: u32 = 256;
const SCALAR_QUANT_SCALE: u32 = 9;
const MAX_LENGTH: u32 = 4096;

// AudioVAE V2 (audio_vae_v2.py `AudioVAEConfig` defaults)
const VAE_SAMPLE_RATE: u32 = 16_000;
const VAE_OUT_SAMPLE_RATE: u32 = 48_000;
const VAE_ENCODER_DIM: u32 = 128;
const VAE_ENCODER_RATES: [u32; 4] = [2, 5, 8, 8];
const VAE_LATENT_DIM: u32 = 64;
const VAE_DECODER_DIM: u32 = 2048;
const VAE_DECODER_RATES: [u32; 6] = [8, 6, 5, 2, 2, 2];
const VAE_DEPTHWISE: bool = true;
const VAE_USE_NOISE_BLOCK: bool = false;

/// Model family marker (`config.json.architecture = "voxcpm"`).
/// Distinct from the sibling Qwen family / Llama family etc. Recorded
/// so the runtime can distinguish VoxCPM from other MiniCPM-family
/// releases at telemetry time.
const MODEL_FAMILY: &str = "voxcpm";

/// Outcome of a VoxCPM-0.5B conversion.
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
    /// Operator-facing diagnostics (never fail the conversion — the
    /// runtime is the authoritative gate, FR-EX-08).
    pub(crate) notes: Vec<String>,
}

/// Converts a VoxCPM-0.5B safetensors buffer into a populated GGUF
/// builder.
///
/// Every F32 / F16 tensor passes through under its upstream name; the
/// `vokra.voxcpm2.*` + `vokra.vae_continuous.*` chunk groups are written
/// from the transcribed constants above; provenance stamps mark the
/// weight as `Permissive` (apache-2.0 — end-to-end).
pub(crate) fn convert(bytes: Vec<u8>) -> Result<(GgufBuilder, VoxCpm2Report), ConvertError> {
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    write_hparams(&mut b);
    // Self-describing redistribution: the artifact carries its own licence.
    // VoxCPM-0.5B ships apache-2.0 end-to-end (LICENSE +
    // huggingface.co/openbmb/VoxCPM-0.5B model card `license: apache-2.0`,
    // fetched 2026-07-24 — CLAUDE.md「ハルシネーション厳禁」).
    vokra_core::stamp_provenance(
        &mut b,
        LicenseClass::Permissive,
        "apache-2.0",
        Some(NAME),
        Some("openbmb/VoxCPM-0.5B (apache-2.0 end-to-end)"),
    );

    let mut report = VoxCpm2Report::default();
    for t in st.tensors() {
        match t.dtype {
            // BF16 pass-through added 2026-07-25 (mirror of qwen3-tts +
            // moshi + voxtral): upstream VoxCPM-0.5B ships
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
             upstream VoxCPM-0.5B release ships `model.safetensors` in BF16 \
             (config.json `dtype: bfloat16`); the BF16 pass-through path is \
             now wired (2026-07-25), so this state is only reachable when the \
             release contains no F32 / F16 / BF16 float tensors at all. \
             Moshi / Kyutai STT pattern) to convert the release build directly."
                .into(),
        );
    }
    Ok((b, report))
}

/// Writes the `vokra.voxcpm2.*` + `vokra.vae_continuous.*` chunk groups
/// from the transcribed constants above (primary sources: `config.json`
/// and `audio_vae_v2.py` `AudioVAEConfig` defaults).
fn write_hparams(b: &mut GgufBuilder) {
    // Top-level
    b.add_string(KEY_MODEL_FAMILY, MODEL_FAMILY);
    b.add_u32(KEY_FEAT_DIM, FEAT_DIM);
    b.add_u32(KEY_PATCH_SIZE, PATCH_SIZE);
    b.add_u32(KEY_RESIDUAL_LM_N_LAYER, RESIDUAL_LM_N_LAYER);
    b.add_u32(KEY_SCALAR_QUANT_LATENT_DIM, SCALAR_QUANT_LATENT_DIM);
    b.add_u32(KEY_SCALAR_QUANT_SCALE, SCALAR_QUANT_SCALE);
    b.add_u32(KEY_MAX_LENGTH, MAX_LENGTH);

    // LM backbone
    b.add_u32(KEY_LM_HIDDEN_DIM, LM_HIDDEN_DIM);
    b.add_u32(KEY_LM_N_LAYER, LM_N_LAYER);
    b.add_u32(KEY_LM_N_HEAD, LM_N_HEAD);
    b.add_u32(KEY_LM_N_HEAD_KV, LM_N_HEAD_KV);
    b.add_u32(KEY_LM_FFN_DIM, LM_FFN_DIM);
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
    b.add_u32(KEY_ENC_N_LAYER, ENC_N_LAYER);

    // DiT
    b.add_u32(KEY_DIT_HIDDEN_DIM, DIT_HIDDEN_DIM);
    b.add_u32(KEY_DIT_FFN_DIM, DIT_FFN_DIM);
    b.add_u32(KEY_DIT_N_HEAD, DIT_N_HEAD);
    b.add_u32(KEY_DIT_N_LAYER, DIT_N_LAYER);

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
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::{GgufFile, GgufMetadataValue};

    fn minimal_safetensors_one_f32() -> Vec<u8> {
        // Single f32 tensor so the pass-through arm fires once and the
        // report counts a non-zero write. The tensor name mirrors an
        // upstream VoxCPM scaffold name (base_lm token embedding).
        let header = r#"{"base_lm.embed_tokens.weight":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}}"#;
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
        let header = r#"{"base_lm.embed_tokens.weight":{"dtype":"F16","shape":[2,3],"data_offsets":[0,12]}}"#;
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(&[0u8; 12]);
        out
    }

    /// A single BF16 tensor — the upstream VoxCPM-0.5B serving format —
    /// so this hits the pass-through's `_ =>` arm and MUST land in
    /// `skipped_non_float` (with the loud "no float tensors" note).
    fn minimal_safetensors_one_bf16() -> Vec<u8> {
        let header = r#"{"base_lm.embed_tokens.weight":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,12]}}"#;
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

    #[test]
    fn name_string_matches_hf_release() {
        assert_eq!(NAME, "voxcpm-0.5b");
    }

    /// The transcribed constants must equal the primary-source values.
    /// Changing any of these silently mis-shapes the LM backbone / VAE
    /// handshake / CFM sampler.
    #[test]
    fn transcribed_constants_match_primary_source() {
        // LM backbone (config.json.lm_config.*).
        assert_eq!(LM_HIDDEN_DIM, 1024);
        assert_eq!(LM_N_LAYER, 24);
        assert_eq!(LM_N_HEAD, 16);
        assert_eq!(LM_N_HEAD_KV, 2);
        assert_eq!(LM_FFN_DIM, 4096);
        assert_eq!(LM_VOCAB_SIZE, 73_448);
        assert_eq!(LM_MAX_POSITIONS, 32_768);
        assert!((LM_ROPE_BASE - 10_000.0).abs() < 1e-3);
        assert!((LM_RMS_NORM_EPS - 1e-5).abs() < 1e-9);
        // `assertions_on_constants` and `bool_assert_comparison` collide on
        // primary-source constant `bool` pins — allow the constant-assert
        // form so the transcription is stated at the bool literal level
        // (the sibling models pin the same way).
        #[allow(clippy::assertions_on_constants)]
        {
            assert!(LM_ROPE_SCALING_LONGROPE);
            assert!(!LM_USE_MUP);
        }
        assert_eq!(LM_ROPE_ORIG_MAX_POS, 32_768);
        assert_eq!(LM_SCALE_EMB, 12);
        assert_eq!(LM_DIM_MODEL_BASE, 256);
        assert!((LM_SCALE_DEPTH - 1.4).abs() < 1e-5);

        // Encoder (config.json.encoder_config.*).
        assert_eq!(ENC_HIDDEN_DIM, 1024);
        assert_eq!(ENC_FFN_DIM, 4096);
        assert_eq!(ENC_N_HEAD, 16);
        assert_eq!(ENC_N_LAYER, 4);

        // DiT (config.json.dit_config.*).
        assert_eq!(DIT_HIDDEN_DIM, 1024);
        assert_eq!(DIT_FFN_DIM, 4096);
        assert_eq!(DIT_N_HEAD, 16);
        assert_eq!(DIT_N_LAYER, 4);

        // CFM sampler (config.json.dit_config.cfm_config.*).
        assert!((CFM_SIGMA_MIN - 1e-6).abs() < 1e-9);
        assert_eq!(CFM_SOLVER, "euler");
        assert_eq!(CFM_T_SCHEDULER, "log-norm");
        assert!((CFM_INFERENCE_CFG_RATE - 2.0).abs() < 1e-5);

        // Top-level (config.json.*).
        assert_eq!(FEAT_DIM, 64);
        assert_eq!(PATCH_SIZE, 2);
        assert_eq!(RESIDUAL_LM_N_LAYER, 6);
        assert_eq!(SCALAR_QUANT_LATENT_DIM, 256);
        assert_eq!(SCALAR_QUANT_SCALE, 9);
        assert_eq!(MAX_LENGTH, 4096);

        // AudioVAE V2 (audio_vae_v2.py `AudioVAEConfig`).
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

        assert_eq!(MODEL_FAMILY, "voxcpm");

        // Compile-time algebra: GQA + RoPE + VAE handshake pins.
        const _: () = {
            // GQA well-formedness.
            assert!(LM_N_HEAD % LM_N_HEAD_KV == 0);
            // VAE handshake: LM step feature width MUST equal VAE latent.
            assert!(FEAT_DIM == VAE_LATENT_DIM);
            // Positive shapes.
            assert!(FEAT_DIM > 0);
            assert!(PATCH_SIZE > 0);
            assert!(MAX_LENGTH > 0);
            assert!(RESIDUAL_LM_N_LAYER > 0);
        };
    }

    /// The VoxCPM handshake with the shared VAE seam
    /// [`vokra_ops::vae_continuous::ContinuousVaeConfig::voxcpm_0_5b`]
    /// on `latent_dim`. Drifting the two would drop or duplicate
    /// channels entering the DiT silently.
    #[test]
    fn feat_dim_matches_shared_vae_seam() {
        let vae = vokra_ops::vae_continuous::ContinuousVaeConfig::voxcpm_0_5b();
        assert_eq!(FEAT_DIM, vae.latent_dim);
        assert_eq!(VAE_LATENT_DIM, vae.latent_dim);
        assert_eq!(VAE_SAMPLE_RATE, vae.sample_rate_hz);
        assert_eq!(VAE_OUT_SAMPLE_RATE, vae.out_sample_rate_hz);
        assert_eq!(VAE_ENCODER_DIM, vae.encoder_dim);
        assert_eq!(&VAE_ENCODER_RATES[..], vae.encoder_rates.as_slice());
        assert_eq!(VAE_DECODER_DIM, vae.decoder_dim);
        assert_eq!(&VAE_DECODER_RATES[..], vae.decoder_rates.as_slice());
        assert_eq!(VAE_DEPTHWISE, vae.depthwise);
        assert_eq!(VAE_USE_NOISE_BLOCK, vae.use_noise_block);
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
        // `vokra.voxcpm2.*` or `vokra.vae_continuous.*` prefix.
        for (key, want) in [
            (KEY_FEAT_DIM, FEAT_DIM),
            (KEY_PATCH_SIZE, PATCH_SIZE),
            (KEY_RESIDUAL_LM_N_LAYER, RESIDUAL_LM_N_LAYER),
            (KEY_SCALAR_QUANT_LATENT_DIM, SCALAR_QUANT_LATENT_DIM),
            (KEY_SCALAR_QUANT_SCALE, SCALAR_QUANT_SCALE),
            (KEY_MAX_LENGTH, MAX_LENGTH),
            (KEY_LM_HIDDEN_DIM, LM_HIDDEN_DIM),
            (KEY_LM_N_LAYER, LM_N_LAYER),
            (KEY_LM_N_HEAD, LM_N_HEAD),
            (KEY_LM_N_HEAD_KV, LM_N_HEAD_KV),
            (KEY_LM_FFN_DIM, LM_FFN_DIM),
            (KEY_LM_VOCAB_SIZE, LM_VOCAB_SIZE),
            (KEY_LM_MAX_POSITIONS, LM_MAX_POSITIONS),
            (KEY_LM_ROPE_ORIG_MAX_POS, LM_ROPE_ORIG_MAX_POS),
            (KEY_LM_SCALE_EMB, LM_SCALE_EMB),
            (KEY_LM_DIM_MODEL_BASE, LM_DIM_MODEL_BASE),
            (KEY_ENC_HIDDEN_DIM, ENC_HIDDEN_DIM),
            (KEY_ENC_FFN_DIM, ENC_FFN_DIM),
            (KEY_ENC_N_HEAD, ENC_N_HEAD),
            (KEY_ENC_N_LAYER, ENC_N_LAYER),
            (KEY_DIT_HIDDEN_DIM, DIT_HIDDEN_DIM),
            (KEY_DIT_FFN_DIM, DIT_FFN_DIM),
            (KEY_DIT_N_HEAD, DIT_N_HEAD),
            (KEY_DIT_N_LAYER, DIT_N_LAYER),
            (KEY_VAE_SAMPLE_RATE, VAE_SAMPLE_RATE),
            (KEY_VAE_OUT_SAMPLE_RATE, VAE_OUT_SAMPLE_RATE),
            (KEY_VAE_ENCODER_DIM, VAE_ENCODER_DIM),
            (KEY_VAE_LATENT_DIM, VAE_LATENT_DIM),
            (KEY_VAE_DECODER_DIM, VAE_DECODER_DIM),
        ] {
            assert_eq!(get_u32(&file, key), want, "{key}");
        }

        // F32 constants round-trip too.
        assert!((get_f32(&file, KEY_LM_ROPE_BASE) - LM_ROPE_BASE).abs() < 1e-3);
        assert!((get_f32(&file, KEY_LM_RMS_NORM_EPS) - LM_RMS_NORM_EPS).abs() < 1e-9);
        assert!((get_f32(&file, KEY_LM_SCALE_DEPTH) - LM_SCALE_DEPTH).abs() < 1e-5);
        assert!((get_f32(&file, KEY_CFM_SIGMA_MIN) - CFM_SIGMA_MIN).abs() < 1e-9);
        assert!((get_f32(&file, KEY_CFM_INFERENCE_CFG_RATE) - CFM_INFERENCE_CFG_RATE).abs() < 1e-5);

        // Bool constants round-trip.
        assert_eq!(
            get_bool(&file, KEY_LM_ROPE_SCALING_LONGROPE),
            LM_ROPE_SCALING_LONGROPE
        );
        assert_eq!(get_bool(&file, KEY_LM_USE_MUP), LM_USE_MUP);
        assert_eq!(get_bool(&file, KEY_VAE_DEPTHWISE), VAE_DEPTHWISE);
        assert_eq!(
            get_bool(&file, KEY_VAE_USE_NOISE_BLOCK),
            VAE_USE_NOISE_BLOCK
        );

        // String constants round-trip.
        assert_eq!(get_string(&file, KEY_CFM_SOLVER), CFM_SOLVER);
        assert_eq!(get_string(&file, KEY_CFM_T_SCHEDULER), CFM_T_SCHEDULER);

        // Provenance: apache-2.0 permissive (end-to-end).
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_MODEL_ID)
                .and_then(|v| v.as_str()),
            Some(NAME)
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
            .tensor_info("base_lm.embed_tokens.weight")
            .expect("tensor present");
        assert_eq!(info.dtype, GgmlType::F16);
        assert_eq!(info.dimensions, vec![2, 3]);
        assert_eq!(file.tensor_bytes(info).len(), 12);
    }

    /// Pins the BF16 leg of the `GgmlType::F32 | GgmlType::F16 |
    /// GgmlType::BF16` union: BF16 (the upstream serving format for
    /// VoxCPM-0.5B, `dtype: bfloat16`) must reach the pass-through arm,
    /// emit as GGUF type 30 verbatim, and increment `bf16_passthrough`.
    /// Mirror of qwen3-tts / vibevoice / moshi.
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
            .tensor_info("base_lm.embed_tokens.weight")
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

    /// Every `vokra.voxcpm2.*` and `vokra.vae_continuous.*` key uses the
    /// documented prefix — a regression where a key crossed into another
    /// model's namespace (e.g. `vokra.cosyvoice2.*`) would still round-
    /// trip in isolation but would misroute at the runtime dispatch
    /// layer.
    #[test]
    fn every_metadata_key_carries_a_documented_prefix() {
        for key in [
            KEY_FEAT_DIM,
            KEY_PATCH_SIZE,
            KEY_RESIDUAL_LM_N_LAYER,
            KEY_SCALAR_QUANT_LATENT_DIM,
            KEY_SCALAR_QUANT_SCALE,
            KEY_MAX_LENGTH,
            KEY_MODEL_FAMILY,
            KEY_LM_HIDDEN_DIM,
            KEY_LM_N_LAYER,
            KEY_LM_N_HEAD,
            KEY_LM_N_HEAD_KV,
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
            KEY_DIT_HIDDEN_DIM,
            KEY_DIT_FFN_DIM,
            KEY_DIT_N_HEAD,
            KEY_DIT_N_LAYER,
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
        ] {
            assert!(
                key.starts_with("vokra.vae_continuous."),
                "{key} must live under the vokra.vae_continuous.* prefix"
            );
        }
    }

    /// Cross-crate hparam handshake with the runtime side.
    #[test]
    fn feat_dim_and_vae_latent_dim_agree() {
        // The runtime `VoxCpm2Config::validate_for_forward_with_vae`
        // check rejects `feat_dim != vae.latent_dim` loudly; keep the
        // converter constants in lockstep.
        assert_eq!(FEAT_DIM, VAE_LATENT_DIM);
    }
}
