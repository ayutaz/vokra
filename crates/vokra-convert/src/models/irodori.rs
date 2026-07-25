//! **Irodori-TTS-500M-v3**: safetensors checkpoint → GGUF conversion
//! (SoTA plan Phase 5 JA-TTS-1, 2026-07-24).
//!
//! Input: the upstream `Aratako/Irodori-TTS-500M-v3` release —
//! `model.safetensors` (F32 / F16 tensors verbatim). Output: a GGUF
//! carrying every float tensor plus the `vokra.irodori.*` /
//! `vokra.provenance.*` / `vokra.model.*` metadata chunks that the
//! native Irodori-TTS implementation
//! (`crates/vokra-models/src/irodori/`) reads.
//!
//! # What is transcribed vs. shape-driven
//!
//! - **Transcribed constants** — every hparam of the `vokra.irodori.*`
//!   chunk group is transcribed **verbatim** from the primary sources
//!   `github.com/Aratako/Irodori-TTS/blob/main/configs/train_500m_v3_phase1_body.yaml`
//!   and `..._phase2_duration.yaml` plus the `ModelConfig` defaults at
//!   `github.com/Aratako/Irodori-TTS/blob/main/irodori_tts/config.py`
//!   (fetched 2026-07-24 — CLAUDE.md「ハルシネーション厳禁」).
//! - **Sample rate** — the paired `Aratako/Semantic-DACVAE-Japanese-32dim`
//!   codec pins the runtime output rate at **48 kHz** (per the base
//!   model card at `huggingface.co/Aratako/Irodori-TTS-500M-v3`), not
//!   written in the training YAML.
//! - **Tokenizer** — `text_tokenizer_repo = "llm-jp/llm-jp-3-150m"` is
//!   recorded so a real-checkpoint bind can cross-check the tokenizer
//!   manifest; the tokenizer file itself rides a separate side-car (the
//!   Voxtral pattern) or is fetched by the caller at runtime.
//! - **No side-car config** — every field of `train_500m_v3_phase1_body.yaml`
//!   plus `train_500m_v3_phase2_duration.yaml` is fixed for the 500M-v3
//!   release and byte-parallel to the transcribed constants below. A
//!   future 600M VoiceDesign / 2.5B variant that reshapes the DiT or
//!   adds caption conditioning would demand `--config`; this converter
//!   fails loudly if a tensor shape disagrees with the transcribed
//!   axes at runtime bind time (FR-EX-08).
//!
//! # Tensor naming contract
//!
//! GGUF tensor names are the **upstream safetensors names verbatim**
//! (the CSM / Kokoro / CosyVoice2 / Chatterbox / Qwen3-TTS / VoxCPM /
//! VibeVoice contract). Real-weight binding is a follow-up wave gated
//! on the upstream tensor-name manifest fetch; this converter passes
//! every F32 / F16 tensor through unchanged so a future
//! `IrodoriWeights::from_gguf` can walk the same names.
//!
//! # BF16 posture
//!
//! The upstream Irodori-TTS release trains in bf16
//! (`TrainConfig.precision = "bf16"`) but the released
//! `model.safetensors` blob is typically served in F32 / F16 (the
//! `save_pretrained` default). Since 2026-07-25 the pass-through arm
//! also accepts BF16 verbatim (mirror of `qwen3-tts` / `vibevoice` /
//! `voxcpm2` — GGUF type 30 emitted unchanged; runtime widens
//! BF16 → f32 losslessly on load via
//! `vokra-core::gguf::quant::decode_bf16`, `bits << 16` — exact). No
//! pre-widen step is required for a BF16 release.
//!
//! # No ONNX (permanent)
//!
//! Irodori-TTS is distributed as safetensors + a Python pipeline
//! (`irodori_tts/inference_runtime.py`); this converter **never**
//! touches ONNX (FR-LD-05); the pipeline is re-implemented natively
//! in `crates/vokra-models/src/irodori/` (whisper.cpp 型 self
//! re-implementation, CLAUDE.md 設計判断 4).

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` for Irodori-TTS-500M-v3 GGUFs — kept in sync with
/// the runtime constant `vokra-models::irodori::EXPECTED_ARCH`.
/// Intentionally **distinct** from every sibling arch tag because
/// Irodori pairs a DACVAE-continuous DiT with a **Rectified-Flow**
/// Euler sampler; silently sharing an arch tag would misroute the
/// runtime dispatch (VibeVoice → `ddpm_sample`, VoxCPM → EpsS
/// flow_sample, Irodori → Linear / Sway flow_sample with a distinct
/// latent width of 32 vs the Phase-4 siblings' 64).
pub(crate) const ARCH: &str = "irodori-tts";

/// `vokra.model.name` value written for the canonical Irodori-TTS-500M-v3
/// GGUF.
pub(crate) const NAME: &str = "irodori-tts-500m-v3";

// --- vokra.irodori.* metadata keys --------------------------------------
// The runtime side lives in `crates/vokra-models/src/irodori/mod.rs`
// — the two crates share only `vokra-core`, so the cross-crate constant
// duplication rule the CSM / CosyVoice2 / Kokoro / Chatterbox / Qwen3-TTS
// / VoxCPM / VibeVoice family converters use applies.

// Top-level
const KEY_MODEL_FAMILY: &str = "vokra.irodori.model_family";
const KEY_SAMPLE_RATE_HZ: &str = "vokra.irodori.sample_rate_hz";
const KEY_TEXT_TOKENIZER_REPO: &str = "vokra.irodori.text_tokenizer_repo";

// DiT body — train_500m_v3_phase1_body.yaml.model.*
const KEY_DIT_LATENT_DIM: &str = "vokra.irodori.dit.latent_dim";
const KEY_DIT_LATENT_PATCH_SIZE: &str = "vokra.irodori.dit.latent_patch_size";
const KEY_DIT_MODEL_DIM: &str = "vokra.irodori.dit.model_dim";
const KEY_DIT_NUM_LAYERS: &str = "vokra.irodori.dit.num_layers";
const KEY_DIT_NUM_HEADS: &str = "vokra.irodori.dit.num_heads";
const KEY_DIT_MLP_RATIO: &str = "vokra.irodori.dit.mlp_ratio";
const KEY_DIT_TIMESTEP_EMBED_DIM: &str = "vokra.irodori.dit.timestep_embed_dim";
const KEY_DIT_ADALN_RANK: &str = "vokra.irodori.dit.adaln_rank";
const KEY_DIT_NORM_EPS: &str = "vokra.irodori.dit.norm_eps";
const KEY_DIT_DROPOUT: &str = "vokra.irodori.dit.dropout";

// Text encoder — train_500m_v3_phase1_body.yaml.model.*
const KEY_TEXT_VOCAB_SIZE: &str = "vokra.irodori.text.vocab_size";
const KEY_TEXT_DIM: &str = "vokra.irodori.text.dim";
const KEY_TEXT_N_LAYER: &str = "vokra.irodori.text.n_layer";
const KEY_TEXT_N_HEAD: &str = "vokra.irodori.text.n_head";
const KEY_TEXT_MLP_RATIO: &str = "vokra.irodori.text.mlp_ratio";
const KEY_TEXT_ADD_BOS: &str = "vokra.irodori.text.add_bos";

// Speaker (reference-latent) encoder — train_500m_v3_phase1_body.yaml.model.*
const KEY_SPEAKER_DIM: &str = "vokra.irodori.speaker.dim";
const KEY_SPEAKER_N_LAYER: &str = "vokra.irodori.speaker.n_layer";
const KEY_SPEAKER_N_HEAD: &str = "vokra.irodori.speaker.n_head";
const KEY_SPEAKER_MLP_RATIO: &str = "vokra.irodori.speaker.mlp_ratio";
const KEY_SPEAKER_PATCH_SIZE: &str = "vokra.irodori.speaker.patch_size";

// Duration predictor — train_500m_v3_phase2_duration.yaml.model.*
const KEY_DURATION_ENABLED: &str = "vokra.irodori.duration.enabled";
const KEY_DURATION_AUX_DIM: &str = "vokra.irodori.duration.aux_dim";
const KEY_DURATION_HIDDEN_DIM: &str = "vokra.irodori.duration.hidden_dim";
const KEY_DURATION_N_LAYER: &str = "vokra.irodori.duration.n_layer";
const KEY_DURATION_N_HEAD: &str = "vokra.irodori.duration.n_head";
const KEY_DURATION_DROPOUT: &str = "vokra.irodori.duration.dropout";
const KEY_DURATION_ARCHITECTURE: &str = "vokra.irodori.duration.architecture";
const KEY_DURATION_TOKEN_INIT_FRAMES: &str = "vokra.irodori.duration.token_init_frames";
const KEY_DURATION_SPEAKER_FUSION: &str = "vokra.irodori.duration.speaker_fusion";

// --- Transcribed constants ------------------------------------------------
// Primary sources: `train_500m_v3_phase1_body.yaml` +
// `train_500m_v3_phase2_duration.yaml` + `irodori_tts/config.py::ModelConfig`
// at `github.com/Aratako/Irodori-TTS` (fetched 2026-07-24 — CLAUDE.md
// 「ハルシネーション厳禁」).

/// Model family marker.
const MODEL_FAMILY: &str = "irodori-tts";

/// PCM sample rate the paired `Semantic-DACVAE-Japanese-32dim` codec
/// emits — 48 kHz per the base-model card.
const SAMPLE_RATE_HZ: u32 = 48_000;

/// Text tokenizer id the release pins.
const TEXT_TOKENIZER_REPO: &str = "llm-jp/llm-jp-3-150m";

// DiT (train_500m_v3_phase1_body.yaml.model.*).
const DIT_LATENT_DIM: u32 = 32;
const DIT_LATENT_PATCH_SIZE: u32 = 1;
const DIT_MODEL_DIM: u32 = 1_280;
const DIT_NUM_LAYERS: u32 = 12;
const DIT_NUM_HEADS: u32 = 20;
const DIT_MLP_RATIO: f32 = 2.875;
const DIT_TIMESTEP_EMBED_DIM: u32 = 512;
const DIT_ADALN_RANK: u32 = 192;
const DIT_NORM_EPS: f32 = 1e-5;
const DIT_DROPOUT: f32 = 0.0;

// Text encoder (train_500m_v3_phase1_body.yaml.model.*).
const TEXT_VOCAB_SIZE: u32 = 99_574;
const TEXT_DIM: u32 = 512;
const TEXT_N_LAYER: u32 = 10;
const TEXT_N_HEAD: u32 = 8;
const TEXT_MLP_RATIO: f32 = 2.6;
const TEXT_ADD_BOS: bool = true;

// Speaker (reference-latent) encoder.
const SPEAKER_DIM: u32 = 768;
const SPEAKER_N_LAYER: u32 = 8;
const SPEAKER_N_HEAD: u32 = 12;
const SPEAKER_MLP_RATIO: f32 = 2.6;
const SPEAKER_PATCH_SIZE: u32 = 1;

// Duration predictor (train_500m_v3_phase2_duration.yaml.model.*).
const DURATION_ENABLED: bool = true;
const DURATION_AUX_DIM: u32 = 14;
const DURATION_HIDDEN_DIM: u32 = 1_024;
const DURATION_N_LAYER: u32 = 3;
const DURATION_N_HEAD: u32 = 8;
const DURATION_DROPOUT: f32 = 0.1;
const DURATION_ARCHITECTURE: &str = "token_sum_adarn_zero_no_aux";
const DURATION_TOKEN_INIT_FRAMES: f32 = 9.0;
const DURATION_SPEAKER_FUSION: &str = "adarn_zero";

/// Outcome of an Irodori-TTS conversion.
#[derive(Debug, Default)]
pub(crate) struct IrodoriReport {
    /// Float tensors written verbatim (F32 / F16 / BF16 — all three go
    /// through the same byte-copy path since the BF16 pass-through land
    /// 2026-07-25, mirror of `qwen3-tts` / `vibevoice` / `voxcpm2`).
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

/// Converts an Irodori-TTS-500M-v3 safetensors buffer into a populated
/// GGUF builder.
///
/// Every F32 / F16 tensor passes through under its upstream name; the
/// `vokra.irodori.*` chunk group is written from the transcribed
/// constants above; provenance stamps mark the weight as `Permissive`
/// (MIT — code and weights are covered by the single MIT LICENSE at
/// `github.com/Aratako/Irodori-TTS/blob/main/LICENSE`, verified through
/// `gh api /repos/Aratako/Irodori-TTS/license` → `MIT`).
pub(crate) fn convert(bytes: Vec<u8>) -> Result<(GgufBuilder, IrodoriReport), ConvertError> {
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    write_hparams(&mut b);
    // Self-describing redistribution: the artifact carries its own licence.
    // Irodori-TTS ships MIT end-to-end (Aratako/Irodori-TTS LICENSE, verified
    // via `gh api /repos/Aratako/Irodori-TTS/license` → `MIT`, fetched
    // 2026-07-24 — CLAUDE.md「ハルシネーション厳禁」).
    vokra_core::stamp_provenance(
        &mut b,
        LicenseClass::Permissive,
        "mit",
        Some(NAME),
        Some("Aratako/Irodori-TTS-500M-v3 (MIT end-to-end)"),
    );

    let mut report = IrodoriReport::default();
    for t in st.tensors() {
        match t.dtype {
            // BF16 pass-through added 2026-07-25 (mirror of qwen3-tts +
            // vibevoice + voxcpm2): Irodori-TTS trains in bf16
            // (`TrainConfig.precision = "bf16"`) so a downstream BF16
            // release hits this arm. Emit as GGUF type 30 verbatim;
            // runtime widens on load via `decode_bf16` (exact,
            // `bits << 16`).
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
            "no float tensors passed through — this GGUF is metadata-only and the runtime will \
             refuse to bind any weights (FR-EX-08). Irodori-TTS trains in bf16 \
             (TrainConfig.precision = \"bf16\"); the BF16 pass-through path is now wired \
             (2026-07-25), so this state is only reachable when the release contains no \
             F32 / F16 / BF16 float tensors at all."
                .into(),
        );
    }
    Ok((b, report))
}

/// Writes the `vokra.irodori.*` chunk group from the transcribed
/// constants above (primary sources: `train_500m_v3_phase1_body.yaml`
/// plus `train_500m_v3_phase2_duration.yaml` plus
/// `irodori_tts/config.py::ModelConfig`).
fn write_hparams(b: &mut GgufBuilder) {
    // Top-level.
    b.add_string(KEY_MODEL_FAMILY, MODEL_FAMILY);
    b.add_u32(KEY_SAMPLE_RATE_HZ, SAMPLE_RATE_HZ);
    b.add_string(KEY_TEXT_TOKENIZER_REPO, TEXT_TOKENIZER_REPO);

    // DiT.
    b.add_u32(KEY_DIT_LATENT_DIM, DIT_LATENT_DIM);
    b.add_u32(KEY_DIT_LATENT_PATCH_SIZE, DIT_LATENT_PATCH_SIZE);
    b.add_u32(KEY_DIT_MODEL_DIM, DIT_MODEL_DIM);
    b.add_u32(KEY_DIT_NUM_LAYERS, DIT_NUM_LAYERS);
    b.add_u32(KEY_DIT_NUM_HEADS, DIT_NUM_HEADS);
    b.add_f32(KEY_DIT_MLP_RATIO, DIT_MLP_RATIO);
    b.add_u32(KEY_DIT_TIMESTEP_EMBED_DIM, DIT_TIMESTEP_EMBED_DIM);
    b.add_u32(KEY_DIT_ADALN_RANK, DIT_ADALN_RANK);
    b.add_f32(KEY_DIT_NORM_EPS, DIT_NORM_EPS);
    b.add_f32(KEY_DIT_DROPOUT, DIT_DROPOUT);

    // Text encoder.
    b.add_u32(KEY_TEXT_VOCAB_SIZE, TEXT_VOCAB_SIZE);
    b.add_u32(KEY_TEXT_DIM, TEXT_DIM);
    b.add_u32(KEY_TEXT_N_LAYER, TEXT_N_LAYER);
    b.add_u32(KEY_TEXT_N_HEAD, TEXT_N_HEAD);
    b.add_f32(KEY_TEXT_MLP_RATIO, TEXT_MLP_RATIO);
    b.add_bool(KEY_TEXT_ADD_BOS, TEXT_ADD_BOS);

    // Speaker (reference-latent) encoder.
    b.add_u32(KEY_SPEAKER_DIM, SPEAKER_DIM);
    b.add_u32(KEY_SPEAKER_N_LAYER, SPEAKER_N_LAYER);
    b.add_u32(KEY_SPEAKER_N_HEAD, SPEAKER_N_HEAD);
    b.add_f32(KEY_SPEAKER_MLP_RATIO, SPEAKER_MLP_RATIO);
    b.add_u32(KEY_SPEAKER_PATCH_SIZE, SPEAKER_PATCH_SIZE);

    // Duration predictor.
    b.add_bool(KEY_DURATION_ENABLED, DURATION_ENABLED);
    b.add_u32(KEY_DURATION_AUX_DIM, DURATION_AUX_DIM);
    b.add_u32(KEY_DURATION_HIDDEN_DIM, DURATION_HIDDEN_DIM);
    b.add_u32(KEY_DURATION_N_LAYER, DURATION_N_LAYER);
    b.add_u32(KEY_DURATION_N_HEAD, DURATION_N_HEAD);
    b.add_f32(KEY_DURATION_DROPOUT, DURATION_DROPOUT);
    b.add_string(KEY_DURATION_ARCHITECTURE, DURATION_ARCHITECTURE);
    b.add_f32(KEY_DURATION_TOKEN_INIT_FRAMES, DURATION_TOKEN_INIT_FRAMES);
    b.add_string(KEY_DURATION_SPEAKER_FUSION, DURATION_SPEAKER_FUSION);
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::{GgufFile, GgufMetadataValue};

    fn minimal_safetensors_one_f32() -> Vec<u8> {
        // Single f32 tensor so the pass-through arm fires once and the
        // report counts a non-zero write. The tensor name mirrors an
        // upstream Irodori scaffold name (RF-DiT text embed).
        let header = r#"{"text_embed.weight":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}}"#;
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
        let header = r#"{"text_embed.weight":{"dtype":"F16","shape":[2,3],"data_offsets":[0,12]}}"#;
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(&[0u8; 12]);
        out
    }

    /// A single BF16 tensor — Irodori-TTS trains in bf16 so a
    /// downstream release can ship BF16 directly. Since 2026-07-25 this
    /// must hit the `GgmlType::F32 | GgmlType::F16 | GgmlType::BF16`
    /// pass-through arm and land in `written` + `bf16_passthrough`
    /// (mirror of `qwen3-tts` / `vibevoice` / `voxcpm2`).
    fn minimal_safetensors_one_bf16() -> Vec<u8> {
        let header =
            r#"{"text_embed.weight":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,12]}}"#;
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

    // ---- Arch tag distinctness ------------------------------------------

    #[test]
    fn arch_string_matches_runtime_constant() {
        // The two crates only share `vokra-core`, so this constant is the
        // sole handshake with `vokra-models::irodori::EXPECTED_ARCH`.
        assert_eq!(ARCH, "irodori-tts");
    }

    #[test]
    fn arch_is_distinct_from_every_sibling_family() {
        // Irodori pairs a DACVAE-continuous DiT with a Rectified-Flow
        // Euler sampler; silently sharing an arch tag with a Phase-4
        // continuous-VAE sibling (VoxCPM / VibeVoice) or any earlier
        // sibling would misroute the runtime dispatch.
        assert_ne!(ARCH, "vibevoice");
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
        // Canonical release: Aratako/Irodori-TTS-500M-v3.
        assert_eq!(NAME, "irodori-tts-500m-v3");
    }

    /// The transcribed constants must equal the primary-source values.
    /// Changing any of these silently mis-shapes the RF-DiT / text
    /// encoder / speaker encoder / duration predictor.
    #[test]
    fn transcribed_constants_match_primary_source() {
        // DiT (train_500m_v3_phase1_body.yaml.model.*).
        assert_eq!(DIT_LATENT_DIM, 32);
        assert_eq!(DIT_LATENT_PATCH_SIZE, 1);
        assert_eq!(DIT_MODEL_DIM, 1_280);
        assert_eq!(DIT_NUM_LAYERS, 12);
        assert_eq!(DIT_NUM_HEADS, 20);
        assert!((DIT_MLP_RATIO - 2.875).abs() < 1e-6);
        assert_eq!(DIT_TIMESTEP_EMBED_DIM, 512);
        assert_eq!(DIT_ADALN_RANK, 192);
        assert!((DIT_NORM_EPS - 1e-5).abs() < 1e-9);
        assert!((DIT_DROPOUT - 0.0).abs() < 1e-9);

        // Text encoder.
        assert_eq!(TEXT_VOCAB_SIZE, 99_574);
        assert_eq!(TEXT_DIM, 512);
        assert_eq!(TEXT_N_LAYER, 10);
        assert_eq!(TEXT_N_HEAD, 8);
        assert!((TEXT_MLP_RATIO - 2.6).abs() < 1e-6);
        // ModelConfig default: text_add_bos=True.
        #[allow(clippy::assertions_on_constants)]
        {
            assert!(TEXT_ADD_BOS);
        }

        // Speaker encoder.
        assert_eq!(SPEAKER_DIM, 768);
        assert_eq!(SPEAKER_N_LAYER, 8);
        assert_eq!(SPEAKER_N_HEAD, 12);
        assert!((SPEAKER_MLP_RATIO - 2.6).abs() < 1e-6);
        assert_eq!(SPEAKER_PATCH_SIZE, 1);

        // Duration predictor (train_500m_v3_phase2_duration.yaml.model.*).
        #[allow(clippy::assertions_on_constants)]
        {
            assert!(DURATION_ENABLED);
        }
        assert_eq!(DURATION_AUX_DIM, 14);
        assert_eq!(DURATION_HIDDEN_DIM, 1_024);
        assert_eq!(DURATION_N_LAYER, 3);
        assert_eq!(DURATION_N_HEAD, 8);
        assert!((DURATION_DROPOUT - 0.1).abs() < 1e-6);
        assert_eq!(DURATION_ARCHITECTURE, "token_sum_adarn_zero_no_aux");
        assert!((DURATION_TOKEN_INIT_FRAMES - 9.0).abs() < 1e-6);
        assert_eq!(DURATION_SPEAKER_FUSION, "adarn_zero");

        // Top-level.
        assert_eq!(MODEL_FAMILY, "irodori-tts");
        assert_eq!(SAMPLE_RATE_HZ, 48_000);
        assert_eq!(TEXT_TOKENIZER_REPO, "llm-jp/llm-jp-3-150m");

        // Compile-time algebra: the head splits divide evenly on every
        // encoder, RoPE-even head_dim, adaln_rank ≤ model_dim, and the
        // v3-release latent width of 32 differs from the Phase-4 siblings'
        // 64 (a silent match would be a transcription error).
        const _: () = {
            // Head splits.
            assert!(DIT_MODEL_DIM % DIT_NUM_HEADS == 0);
            assert!(TEXT_DIM % TEXT_N_HEAD == 0);
            assert!(SPEAKER_DIM % SPEAKER_N_HEAD == 0);
            // Head widths per encoder (algebraic pins, NOT
            // primary-source constants — they are `dim / n_head`).
            assert!(DIT_MODEL_DIM / DIT_NUM_HEADS == 64);
            assert!(TEXT_DIM / TEXT_N_HEAD == 64);
            assert!(SPEAKER_DIM / SPEAKER_N_HEAD == 64);
            // RoPE requires even head_dim.
            assert!((DIT_MODEL_DIM / DIT_NUM_HEADS) % 2 == 0);
            assert!((TEXT_DIM / TEXT_N_HEAD) % 2 == 0);
            assert!((SPEAKER_DIM / SPEAKER_N_HEAD) % 2 == 0);
            // Low-Rank AdaLN bottleneck must fit.
            assert!(DIT_ADALN_RANK <= DIT_MODEL_DIM);
            // Latent width MUST differ from the Phase-4 continuous-VAE
            // siblings' 64 — silently matching would be a transcription
            // error (Irodori's DACVAE latent width is 32).
            assert!(DIT_LATENT_DIM == 32);
            assert!(DIT_LATENT_DIM != 64);
            // Positive sample rate.
            assert!(SAMPLE_RATE_HZ > 0);
        };
    }

    #[test]
    fn round_trip_carries_arch_chunks_and_provenance() {
        let (builder, report) = convert(minimal_safetensors_one_f32()).expect("convert");
        assert_eq!(report.written, 1);
        assert_eq!(report.skipped_non_float, 0);
        assert!(report.notes.is_empty());

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
        assert_eq!(get_u32(&file, KEY_SAMPLE_RATE_HZ), SAMPLE_RATE_HZ);
        assert_eq!(
            get_string(&file, KEY_TEXT_TOKENIZER_REPO),
            TEXT_TOKENIZER_REPO
        );

        // Every transcribed U32 hparam round-trips verbatim under the
        // `vokra.irodori.*` prefix.
        for (key, want) in [
            (KEY_DIT_LATENT_DIM, DIT_LATENT_DIM),
            (KEY_DIT_LATENT_PATCH_SIZE, DIT_LATENT_PATCH_SIZE),
            (KEY_DIT_MODEL_DIM, DIT_MODEL_DIM),
            (KEY_DIT_NUM_LAYERS, DIT_NUM_LAYERS),
            (KEY_DIT_NUM_HEADS, DIT_NUM_HEADS),
            (KEY_DIT_TIMESTEP_EMBED_DIM, DIT_TIMESTEP_EMBED_DIM),
            (KEY_DIT_ADALN_RANK, DIT_ADALN_RANK),
            (KEY_TEXT_VOCAB_SIZE, TEXT_VOCAB_SIZE),
            (KEY_TEXT_DIM, TEXT_DIM),
            (KEY_TEXT_N_LAYER, TEXT_N_LAYER),
            (KEY_TEXT_N_HEAD, TEXT_N_HEAD),
            (KEY_SPEAKER_DIM, SPEAKER_DIM),
            (KEY_SPEAKER_N_LAYER, SPEAKER_N_LAYER),
            (KEY_SPEAKER_N_HEAD, SPEAKER_N_HEAD),
            (KEY_SPEAKER_PATCH_SIZE, SPEAKER_PATCH_SIZE),
            (KEY_DURATION_AUX_DIM, DURATION_AUX_DIM),
            (KEY_DURATION_HIDDEN_DIM, DURATION_HIDDEN_DIM),
            (KEY_DURATION_N_LAYER, DURATION_N_LAYER),
            (KEY_DURATION_N_HEAD, DURATION_N_HEAD),
        ] {
            assert_eq!(get_u32(&file, key), want, "{key}");
        }

        // F32 constants round-trip.
        assert!((get_f32(&file, KEY_DIT_MLP_RATIO) - DIT_MLP_RATIO).abs() < 1e-6);
        assert!((get_f32(&file, KEY_DIT_NORM_EPS) - DIT_NORM_EPS).abs() < 1e-9);
        assert!((get_f32(&file, KEY_DIT_DROPOUT) - DIT_DROPOUT).abs() < 1e-9);
        assert!((get_f32(&file, KEY_TEXT_MLP_RATIO) - TEXT_MLP_RATIO).abs() < 1e-6);
        assert!((get_f32(&file, KEY_SPEAKER_MLP_RATIO) - SPEAKER_MLP_RATIO).abs() < 1e-6);
        assert!((get_f32(&file, KEY_DURATION_DROPOUT) - DURATION_DROPOUT).abs() < 1e-6);
        assert!(
            (get_f32(&file, KEY_DURATION_TOKEN_INIT_FRAMES) - DURATION_TOKEN_INIT_FRAMES).abs()
                < 1e-6
        );

        // Bool constants round-trip.
        assert_eq!(get_bool(&file, KEY_TEXT_ADD_BOS), TEXT_ADD_BOS);
        assert_eq!(get_bool(&file, KEY_DURATION_ENABLED), DURATION_ENABLED);

        // String constants round-trip.
        assert_eq!(
            get_string(&file, KEY_DURATION_ARCHITECTURE),
            DURATION_ARCHITECTURE
        );
        assert_eq!(
            get_string(&file, KEY_DURATION_SPEAKER_FUSION),
            DURATION_SPEAKER_FUSION
        );

        // Provenance is stamped Permissive (MIT).
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some("permissive")
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some("mit")
        );
    }

    #[test]
    fn f16_tensor_passes_through() {
        // Pins the F16 leg of the `GgmlType::F32 | GgmlType::F16` union
        // arm.
        let (_builder, report) = convert(minimal_safetensors_one_f16()).expect("convert");
        assert_eq!(report.written, 1);
        assert_eq!(report.skipped_non_float, 0);
        assert!(report.notes.is_empty());
    }

    /// Pins the BF16 leg of the `GgmlType::F32 | GgmlType::F16 |
    /// GgmlType::BF16` union: BF16 (Irodori-TTS trains in bf16 per
    /// `TrainConfig.precision = "bf16"`) must reach the pass-through
    /// arm, emit as GGUF type 30 verbatim, and increment
    /// `bf16_passthrough`. Mirror of vibevoice /
    /// `bf16_tensor_passes_through_verbatim` and moshi's `assert_eq!(
    /// info.dtype, GgmlType::BF16, "no convert-time widening")`.
    ///
    /// Rewritten 2026-07-25 from the earlier "counted as skipped" pin
    /// (`bf16_tensor_is_skipped_with_loud_note`) — the earlier pin
    /// encoded the pre-BF16-fix scaffold posture and would
    /// tautologically fail after the fix; removing it outright would
    /// let a latent silent-widen slip in undetected, so the slot is
    /// re-purposed to the passes-through invariant instead.
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
            .tensor_info("text_embed.weight")
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

    #[test]
    fn empty_safetensors_emits_loud_note() {
        // No tensors → the "no float tensors" loud note fires; the
        // hparam chunk group still round-trips (so a caller inspecting
        // the metadata-only GGUF sees the release axes).
        let (builder, report) = convert(minimal_safetensors_no_tensors()).expect("convert");
        assert_eq!(report.written, 0);
        assert_eq!(report.skipped_non_float, 0);
        assert_eq!(report.notes.len(), 1);
        let out = builder.to_bytes().expect("serialize");
        let file = GgufFile::parse(out).expect("parse");
        assert_eq!(get_u32(&file, KEY_DIT_MODEL_DIM), DIT_MODEL_DIM);
    }
}
