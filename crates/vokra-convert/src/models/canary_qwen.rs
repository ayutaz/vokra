//! NVIDIA **Canary-Qwen-2.5B**: safetensors checkpoint → GGUF conversion
//! (SoTA plan reuse bundle, 2026-07-30).
//!
//! Input: a Canary-Qwen-2.5B **prepared** safetensors (the upstream
//! release ships a `.nemo` tarball that a prepare-checkpoint script
//! flattens to safetensors + the SentencePiece tokenizer — the Canary /
//! DAC / CSM / Kokoro / DFN3 pattern). Output: a GGUF carrying every
//! F32 / F16 / BF16 tensor verbatim plus the `vokra.canary_qwen.*` /
//! `vokra.provenance.*` metadata chunks.
//!
//! # What is transcribed vs. shape-driven
//!
//! - **Transcribed constants** — encoder axes reuse the primary-source
//!   Canary-1B-v2 FastConformer defaults (32 layers × 1024 dim × 8 head
//!   × 128 mel bins); decoder axes carry canonical Qwen-family
//!   constants (GQA 16 Q ÷ 8 KV, `head_dim = 128`, `rope_base =
//!   1_000_000`, `rms_norm_eps = 1e-6`) with `0`-placeholder dims
//!   (`n_layer`, `hidden_dim`, `ffn_dim`, `vocab_size`, `n_ctx`) —
//!   the runtime `CanaryQwenConfig::validate_for_forward` rejects the
//!   `0` sentinels loudly (FR-EX-08), so a real-weight binding wave
//!   (T29-equivalent) fills them from the `.nemo` config.
//! - **Shape-driven** — every tensor's dtype + shape is preserved
//!   verbatim from the safetensors header; the converter never widens
//!   or quantises weights (the M2-08 quant policy path is whisper-only).
//!
//! # BF16 posture
//!
//! The reference release advertises BF16 weights in the `.nemo` tarball
//! (NeMo's standard save format). BF16 tensors pass through **verbatim**
//! as GGUF type 30 (`GgmlType::BF16`); the runtime widens BF16 → f32
//! losslessly at load via `crates/vokra-core/src/gguf/quant/mod.rs
//! decode_bf16` (BF16 = top 16 bits of an f32 — `bits << 16` is exact).
//! A [`CanaryQwenReport::bf16_passthrough`] subset counter records how
//! many BF16 tensors landed on the pass-through arm (FR-EX-08).
//!
//! # No ONNX (permanent)
//!
//! Canary-Qwen ships as a `.nemo` tarball / Python pipeline; the
//! pipeline is re-implemented natively in `vokra-models/src/canary_qwen/`
//! (whisper.cpp 型, CLAUDE.md 設計判断 4). This converter never touches
//! ONNX.

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` for Canary-Qwen GGUFs — kept in sync with the
/// runtime constant `vokra-models::canary_qwen::EXPECTED_ARCH`.
pub(crate) const ARCH: &str = "canary-qwen";
/// `vokra.model.name` for Canary-Qwen GGUFs (canonical model id).
pub(crate) const NAME: &str = "canary-qwen-2.5b";

/// The FR-MD-09 attribution text stamped into
/// `vokra.provenance.attribution` — wording aligned with `NOTICE` and
/// the `docs/license-audit.md` NVIDIA Canary-Qwen row.
pub(crate) const CANARY_QWEN_ATTRIBUTION_TEXT: &str = "This application uses NVIDIA Canary-Qwen-2.5B \
     (multimodal ASR + LLM: FastConformer encoder + Qwen LM decoder). \
     Model weights are licensed under CC-BY 4.0 (attribution required; \
     commercial use permitted). Copyright (c) NVIDIA. Source: \
     https://huggingface.co/nvidia/canary-qwen-2.5b";

// --- vokra.canary_qwen.* keys (kept as constants in the converter; the
// runtime module `vokra-models::canary_qwen` reads them symmetrically —
// the cross-crate string handshake pattern) -----------------------------

const KEY_SAMPLE_RATE: &str = "vokra.canary_qwen.sample_rate";

// Encoder (FastConformer — shared with Canary-1B-v2 axes)
const KEY_ENC_N_LAYER: &str = "vokra.canary_qwen.arch.encoder.n_layer";
const KEY_ENC_D_MODEL: &str = "vokra.canary_qwen.arch.encoder.d_model";
const KEY_ENC_N_HEAD: &str = "vokra.canary_qwen.arch.encoder.n_head";
const KEY_ENC_N_HEAD_KV: &str = "vokra.canary_qwen.arch.encoder.n_head_kv";
const KEY_ENC_FFN_DIM: &str = "vokra.canary_qwen.arch.encoder.ffn_dim";
const KEY_ENC_CONV_KERNEL: &str = "vokra.canary_qwen.arch.encoder.conv_kernel_size";
const KEY_ENC_IN_DIM: &str = "vokra.canary_qwen.arch.encoder.in_dim";
const KEY_ENC_SUBSAMPLING_FACTOR: &str = "vokra.canary_qwen.arch.encoder.subsampling_factor";
const KEY_ENC_MAX_POS: &str = "vokra.canary_qwen.arch.encoder.max_position_embeddings";
const KEY_ENC_ATTN_BIAS: &str = "vokra.canary_qwen.arch.encoder.attention_bias";

// Decoder (Qwen LLM — canonical Qwen-family axes)
const KEY_DEC_N_LAYER: &str = "vokra.canary_qwen.arch.decoder.n_layer";
const KEY_DEC_HIDDEN_DIM: &str = "vokra.canary_qwen.arch.decoder.hidden_dim";
const KEY_DEC_N_HEAD_Q: &str = "vokra.canary_qwen.arch.decoder.n_head_q";
const KEY_DEC_N_HEAD_KV: &str = "vokra.canary_qwen.arch.decoder.n_head_kv";
const KEY_DEC_HEAD_DIM: &str = "vokra.canary_qwen.arch.decoder.head_dim";
const KEY_DEC_FFN_DIM: &str = "vokra.canary_qwen.arch.decoder.ffn_dim";
const KEY_DEC_VOCAB_SIZE: &str = "vokra.canary_qwen.arch.decoder.vocab_size";
const KEY_DEC_N_CTX: &str = "vokra.canary_qwen.arch.decoder.n_ctx";
const KEY_DEC_ROPE_BASE: &str = "vokra.canary_qwen.arch.decoder.rope_base";
const KEY_DEC_RMS_NORM_EPS: &str = "vokra.canary_qwen.arch.decoder.rms_norm_eps";

// Cross-attention / soft-prompt bridge
const KEY_CROSS_ATTN_HIDDEN_DIM: &str = "vokra.canary_qwen.arch.cross_attn.hidden_dim";

// --- Transcribed constants ---------------------------------------------------
//
// Encoder side: primary-source Canary-1B-v2 defaults (model card +
// `fast-conformer_aed.yaml` family reference). Decoder side: canonical
// Qwen-family axes (GQA 16 Q / 8 KV, head_dim=128, rope=1_000_000,
// rms_norm_eps=1e-6) with `0`-placeholder dims (n_layer / hidden_dim /
// ffn_dim / vocab_size / n_ctx) pending `.nemo` extraction.

const CANARY_QWEN_SAMPLE_RATE: u32 = 16_000;

// Encoder
const ENC_N_LAYER: u32 = 32;
const ENC_D_MODEL: u32 = 1024;
const ENC_N_HEAD: u32 = 8;
const ENC_N_HEAD_KV: u32 = 8;
const ENC_FFN_DIM: u32 = 4096;
const ENC_CONV_KERNEL: u32 = 9;
const ENC_IN_DIM: u32 = 128;
const ENC_SUBSAMPLING_FACTOR: u32 = 8;
const ENC_MAX_POS: u32 = 5000;
const ENC_ATTN_BIAS: bool = true;

// Decoder (Qwen family)
const DEC_N_HEAD_Q: u32 = 16;
const DEC_N_HEAD_KV: u32 = 8;
const DEC_HEAD_DIM: u32 = 128;
const DEC_ROPE_BASE: f32 = 1_000_000.0;
const DEC_RMS_NORM_EPS: f32 = 1e-6;
// Placeholder dims — runtime validator rejects `0`; a real .nemo
// extraction fills them.
const DEC_N_LAYER: u32 = 0;
const DEC_HIDDEN_DIM: u32 = 0;
const DEC_FFN_DIM: u32 = 0;
const DEC_VOCAB_SIZE: u32 = 0;
const DEC_N_CTX: u32 = 0;

// Cross-attention / soft-prompt bridge — equals encoder d_model.
const CROSS_ATTN_HIDDEN_DIM: u32 = ENC_D_MODEL;

/// Outcome of a Canary-Qwen conversion.
#[derive(Debug, Default)]
pub(crate) struct CanaryQwenReport {
    /// Float tensors written verbatim (F32 / F16 / BF16 — all three go
    /// through the same byte-copy path).
    pub(crate) written: usize,
    /// Non-F32 / F16 / BF16 tensors skipped (defensive counter — the
    /// safetensors reader accepts only these three float dtypes, so any
    /// tensor reaching this counter would signal a reader change
    /// upstream).
    pub(crate) skipped_non_float: usize,
    /// Of the tensors in [`Self::written`], how many were BF16 (subset
    /// counter — pass-through arm observability, same pattern as canary /
    /// qwen3_tts / vibevoice / voxcpm2).
    pub(crate) bf16_passthrough: usize,
    /// Operator-facing diagnostics (never fail the conversion — the
    /// runtime is the authoritative gate, FR-EX-08).
    pub(crate) notes: Vec<String>,
}

/// Converts a Canary-Qwen safetensors buffer into a populated GGUF
/// builder.
///
/// Every F32 / F16 / BF16 tensor passes through under its upstream name;
/// the `vokra.canary_qwen.*` chunk group is written from the transcribed
/// constants above; provenance stamps mark the weight as
/// `AttributionRequired` (CC-BY 4.0) and the FR-MD-09 attribution
/// surface activates.
pub(crate) fn convert(bytes: Vec<u8>) -> Result<(GgufBuilder, CanaryQwenReport), ConvertError> {
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    write_hparams(&mut b);
    vokra_core::stamp_provenance(
        &mut b,
        LicenseClass::AttributionRequired,
        "CC-BY-4.0",
        Some("nvidia/canary-qwen-2.5b"),
        Some("https://huggingface.co/nvidia/canary-qwen-2.5b"),
    );
    vokra_core::stamp_attribution(&mut b, CANARY_QWEN_ATTRIBUTION_TEXT);

    let mut report = CanaryQwenReport::default();
    for t in st.tensors() {
        match t.dtype {
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
             the runtime will refuse to bind any weights (FR-EX-08). The upstream \
             Canary-Qwen-2.5B release ships a .nemo tarball whose PyTorch \
             checkpoint is typically BF16; the BF16 pass-through path is wired \
             (mirror of canary / qwen3-tts / vibevoice / voxcpm2)."
                .to_owned(),
        );
    }
    Ok((b, report))
}

/// Writes the `vokra.canary_qwen.*` chunk group from the transcribed
/// constants above. Booleans ride as u32 0/1 for GGUF portability.
fn write_hparams(b: &mut GgufBuilder) {
    b.add_u32(KEY_SAMPLE_RATE, CANARY_QWEN_SAMPLE_RATE);

    // Encoder
    b.add_u32(KEY_ENC_N_LAYER, ENC_N_LAYER);
    b.add_u32(KEY_ENC_D_MODEL, ENC_D_MODEL);
    b.add_u32(KEY_ENC_N_HEAD, ENC_N_HEAD);
    b.add_u32(KEY_ENC_N_HEAD_KV, ENC_N_HEAD_KV);
    b.add_u32(KEY_ENC_FFN_DIM, ENC_FFN_DIM);
    b.add_u32(KEY_ENC_CONV_KERNEL, ENC_CONV_KERNEL);
    b.add_u32(KEY_ENC_IN_DIM, ENC_IN_DIM);
    b.add_u32(KEY_ENC_SUBSAMPLING_FACTOR, ENC_SUBSAMPLING_FACTOR);
    b.add_u32(KEY_ENC_MAX_POS, ENC_MAX_POS);
    b.add_u32(KEY_ENC_ATTN_BIAS, u32::from(ENC_ATTN_BIAS));

    // Decoder (Qwen family)
    b.add_u32(KEY_DEC_N_LAYER, DEC_N_LAYER);
    b.add_u32(KEY_DEC_HIDDEN_DIM, DEC_HIDDEN_DIM);
    b.add_u32(KEY_DEC_N_HEAD_Q, DEC_N_HEAD_Q);
    b.add_u32(KEY_DEC_N_HEAD_KV, DEC_N_HEAD_KV);
    b.add_u32(KEY_DEC_HEAD_DIM, DEC_HEAD_DIM);
    b.add_u32(KEY_DEC_FFN_DIM, DEC_FFN_DIM);
    b.add_u32(KEY_DEC_VOCAB_SIZE, DEC_VOCAB_SIZE);
    b.add_u32(KEY_DEC_N_CTX, DEC_N_CTX);
    b.add_f32(KEY_DEC_ROPE_BASE, DEC_ROPE_BASE);
    b.add_f32(KEY_DEC_RMS_NORM_EPS, DEC_RMS_NORM_EPS);

    // Cross-attention / soft-prompt bridge
    b.add_u32(KEY_CROSS_ATTN_HIDDEN_DIM, CROSS_ATTN_HIDDEN_DIM);
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::{GgufFile, GgufMetadataValue};

    fn minimal_safetensors_one_f32() -> Vec<u8> {
        let header = r#"{"encoder.blocks.0.attn.qkv_proj.weight":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}}"#;
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(&[0u8; 24]);
        out
    }

    fn minimal_safetensors_one_bf16() -> Vec<u8> {
        let header = r#"{"decoder.model.layers.0.self_attn.q_proj.weight":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,12]}}"#;
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(&[0u8; 12]);
        out
    }

    fn minimal_safetensors_no_tensors() -> Vec<u8> {
        let header = r#"{}"#;
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out
    }

    #[test]
    fn arch_string_matches_runtime_constant() {
        // The two crates only share `vokra-core`, so this constant is
        // the sole handshake with `vokra-models::canary_qwen::EXPECTED_ARCH`.
        assert_eq!(ARCH, "canary-qwen");
        assert_ne!(ARCH, "canary", "must be distinct from base Canary arch tag");
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

        // Every transcribed U32 hparam round-trips verbatim.
        for (key, want) in [
            (KEY_SAMPLE_RATE, CANARY_QWEN_SAMPLE_RATE),
            (KEY_ENC_N_LAYER, ENC_N_LAYER),
            (KEY_ENC_D_MODEL, ENC_D_MODEL),
            (KEY_ENC_N_HEAD, ENC_N_HEAD),
            (KEY_ENC_N_HEAD_KV, ENC_N_HEAD_KV),
            (KEY_ENC_FFN_DIM, ENC_FFN_DIM),
            (KEY_ENC_CONV_KERNEL, ENC_CONV_KERNEL),
            (KEY_ENC_IN_DIM, ENC_IN_DIM),
            (KEY_ENC_SUBSAMPLING_FACTOR, ENC_SUBSAMPLING_FACTOR),
            (KEY_ENC_MAX_POS, ENC_MAX_POS),
            (KEY_ENC_ATTN_BIAS, u32::from(ENC_ATTN_BIAS)),
            (KEY_DEC_N_LAYER, DEC_N_LAYER),
            (KEY_DEC_HIDDEN_DIM, DEC_HIDDEN_DIM),
            (KEY_DEC_N_HEAD_Q, DEC_N_HEAD_Q),
            (KEY_DEC_N_HEAD_KV, DEC_N_HEAD_KV),
            (KEY_DEC_HEAD_DIM, DEC_HEAD_DIM),
            (KEY_DEC_FFN_DIM, DEC_FFN_DIM),
            (KEY_DEC_VOCAB_SIZE, DEC_VOCAB_SIZE),
            (KEY_DEC_N_CTX, DEC_N_CTX),
            (KEY_CROSS_ATTN_HIDDEN_DIM, CROSS_ATTN_HIDDEN_DIM),
        ] {
            match file.get(key) {
                Some(GgufMetadataValue::U32(v)) => assert_eq!(*v, want, "{key}"),
                other => panic!("{key}: unexpected {other:?}"),
            }
        }

        // Provenance: CC-BY 4.0 attribution-required (via `canary-` prefix walk).
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_MODEL_ID)
                .and_then(|v| v.as_str()),
            Some("nvidia/canary-qwen-2.5b")
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some("CC-BY-4.0")
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::AttributionRequired.as_str())
        );
        // Attribution text names NVIDIA + CC-BY 4.0.
        let attr = file
            .get(chunks::KEY_PROVENANCE_ATTRIBUTION)
            .and_then(|v| v.as_str())
            .expect("attribution present");
        assert!(
            attr.contains("NVIDIA") && attr.contains("CC-BY 4.0"),
            "attribution names NVIDIA + CC-BY 4.0: {attr}"
        );
    }

    /// Pins the BF16 pass-through arm: BF16 (the upstream serving format
    /// for Canary-Qwen-2.5B — the `.nemo` tarball's PyTorch checkpoint is
    /// BF16) must reach the pass-through arm, emit as GGUF type 30
    /// verbatim, and increment `bf16_passthrough`.
    #[test]
    fn bf16_tensor_passes_through_verbatim() {
        let (builder, report) = convert(minimal_safetensors_one_bf16()).expect("convert");
        assert_eq!(report.written, 1, "BF16 must reach pass-through arm");
        assert_eq!(report.skipped_non_float, 0);
        assert_eq!(report.bf16_passthrough, 1);

        let out = builder.to_bytes().expect("serialize");
        let file = GgufFile::parse(out).expect("parse");
        let info = file
            .tensor_info("decoder.model.layers.0.self_attn.q_proj.weight")
            .expect("BF16 tensor must be present after pass-through");
        assert_eq!(
            info.dtype,
            GgmlType::BF16,
            "no convert-time widening — GGUF dtype must remain BF16"
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

    #[test]
    fn malformed_input_returns_parse_error() {
        let err = convert(Vec::new()).expect_err("empty buffer must be rejected");
        assert!(
            matches!(err, ConvertError::Parse(_)),
            "expected ConvertError::Parse, got {err:?}"
        );

        let mut truncated = Vec::new();
        truncated.extend_from_slice(&1024u64.to_le_bytes());
        truncated.extend_from_slice(b"{}");
        let err = convert(truncated).expect_err("truncated header must be rejected");
        assert!(
            matches!(err, ConvertError::Parse(_)),
            "expected ConvertError::Parse, got {err:?}"
        );
    }

    /// Sanity: the encoder axes match the primary-source Canary-1B-v2
    /// values; the decoder placeholder dims are `0` (rejected by runtime
    /// validator).
    #[test]
    fn transcribed_encoder_axes_match_canary_family_defaults() {
        assert_eq!(ENC_N_LAYER, 32);
        assert_eq!(ENC_D_MODEL, 1024);
        assert_eq!(ENC_N_HEAD, 8);
        assert_eq!(ENC_N_HEAD_KV, 8);
        assert_eq!(ENC_FFN_DIM, 4096);
        assert_eq!(ENC_IN_DIM, 128);
        assert_eq!(ENC_SUBSAMPLING_FACTOR, 8);
    }

    #[test]
    fn transcribed_decoder_axes_carry_qwen_family_constants_with_zero_placeholders() {
        // Non-placeholder axes = canonical Qwen family.
        assert_eq!(DEC_N_HEAD_Q, 16);
        assert_eq!(DEC_N_HEAD_KV, 8);
        assert_eq!(DEC_HEAD_DIM, 128);
        assert!((DEC_ROPE_BASE - 1_000_000.0).abs() < 1.0);
        assert!((DEC_RMS_NORM_EPS - 1e-6).abs() < 1e-12);
        // Placeholder axes = 0 (runtime validator rejects on load).
        assert_eq!(DEC_N_LAYER, 0, "placeholder pending .nemo extraction");
        assert_eq!(DEC_HIDDEN_DIM, 0, "placeholder pending .nemo extraction");
        assert_eq!(DEC_FFN_DIM, 0, "placeholder pending .nemo extraction");
        assert_eq!(DEC_VOCAB_SIZE, 0, "placeholder pending .nemo extraction");
        assert_eq!(DEC_N_CTX, 0, "placeholder pending .nemo extraction");
    }
}
