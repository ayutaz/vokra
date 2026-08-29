//! NVIDIA **Canary-Qwen-2.5B** conversion boundary.
//!
//! Conversion is intentionally **INSPECTION_ONLY**. The public release is a
//! composite NeMo/Safetensors checkpoint whose complete snapshot, NeMo
//! source, companion Qwen tokenizer, and licensing/provenance evidence must
//! first be authenticated by the VAST inspector. Until that review lands,
//! arbitrary bytes must never produce a runtime-looking GGUF.
//!
//! # BF16 posture
//!
//! The inspector records the authenticated Safetensors BF16 header without
//! reading the 5+ GiB tensor body. No BF16 conversion is currently enabled.
//!
//! # No ONNX (permanent)
//!
//! Canary-Qwen ships as a NeMo/Python pipeline. Native implementation and
//! any future conversion remain separate follow-up work; this boundary never
//! touches or relabels an unauthenticated checkpoint.

use vokra_core::gguf::GgufBuilder;

use crate::ConvertError;

/// `vokra.model.arch` for Canary-Qwen GGUFs — kept in sync with the
/// runtime constant `vokra-models::canary_qwen::EXPECTED_ARCH`.
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "inspection-only metadata is reserved for the authenticated converter"
    )
)]
pub(crate) const ARCH: &str = "canary-qwen";
/// `vokra.model.name` for Canary-Qwen GGUFs (canonical model id).
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "inspection-only metadata is reserved for the authenticated converter"
    )
)]
pub(crate) const NAME: &str = "canary-qwen-2.5b";

/// The FR-MD-09 attribution text stamped into
/// `vokra.provenance.attribution` — wording aligned with `NOTICE` and
/// the `docs/license-audit.md` NVIDIA Canary-Qwen row.
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "inspection-only attribution is reserved for the authenticated converter"
    )
)]
pub(crate) const CANARY_QWEN_ATTRIBUTION_TEXT: &str = "This application uses NVIDIA Canary-Qwen-2.5B \
     (multimodal ASR + LLM: FastConformer encoder + Qwen LM decoder). \
     Model weights are licensed under CC-BY 4.0 (attribution required; \
     commercial use permitted). Copyright (c) NVIDIA. Source: \
     https://huggingface.co/nvidia/canary-qwen-2.5b";

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

/// Refuses conversion until the composite checkpoint and tokenizer have been
/// authenticated by the VAST inspection wave.
pub(crate) fn convert(bytes: Vec<u8>) -> Result<(GgufBuilder, CanaryQwenReport), ConvertError> {
    let _ = bytes;
    Err(ConvertError::Usage(
        "Canary-Qwen-2.5B conversion is INSPECTION_ONLY: complete authenticated HF snapshot, NeMo source, and Qwen tokenizer are not approved; no GGUF may be emitted (HF nvidia/canary-qwen-2.5b@b1469e1bba1cfe140205529c79c434ca47180960)".into(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn conversion_refuses_arbitrary_f32_input_without_artifact() {
        let error = convert(minimal_safetensors_one_f32())
            .expect_err("unreviewed Canary-Qwen input must refuse conversion");
        assert!(error.to_string().contains("INSPECTION_ONLY"));
    }

    #[test]
    fn bf16_input_refuses_without_artifact() {
        let error = convert(minimal_safetensors_one_bf16())
            .expect_err("unreviewed Canary-Qwen BF16 input must refuse conversion");
        assert!(error.to_string().contains("INSPECTION_ONLY"));
    }

    #[test]
    fn zero_tensor_input_refuses_without_artifact() {
        let error = convert(minimal_safetensors_no_tensors())
            .expect_err("unreviewed Canary-Qwen zero-tensor input must refuse conversion");
        assert!(error.to_string().contains("INSPECTION_ONLY"));
    }

    #[test]
    fn malformed_input_returns_parse_error() {
        let err = convert(Vec::new()).expect_err("empty buffer must be rejected");
        assert!(err.to_string().contains("INSPECTION_ONLY"));

        let mut truncated = Vec::new();
        truncated.extend_from_slice(&1024u64.to_le_bytes());
        truncated.extend_from_slice(b"{}");
        let err = convert(truncated).expect_err("truncated header must be rejected");
        assert!(err.to_string().contains("INSPECTION_ONLY"));
    }
}
