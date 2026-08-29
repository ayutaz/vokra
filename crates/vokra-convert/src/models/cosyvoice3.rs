//! Fail-closed inspection boundary for the CosyVoice3 composite release.
//!
//! The upstream release is a composite of a Qwen LLM, flow estimator, HiFT
//! vocoder, speech tokenizer, CampPlus, and ONNX companions. An arbitrary
//! single safetensors file cannot represent that contract, so conversion is
//! deliberately unavailable until a complete authenticated binder exists.
//! This module must not parse input, stamp provenance, or create a GGUF.

use vokra_core::gguf::GgufBuilder;

use crate::ConvertError;

/// `vokra.model.arch` retained for dispatch diagnostics.
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "inspection-only dispatch metadata is retained for the future authenticated binder"
    )
)]
pub(crate) const ARCH: &str = "cosyvoice3";
/// Canonical model name retained for dispatch diagnostics.
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "inspection-only dispatch metadata is retained for the future authenticated binder"
    )
)]
pub(crate) const NAME: &str = "fun-cosyvoice3-0.5b-2512";

/// Minimal report kept for the generic dispatcher's diagnostic formatter.
/// No successful conversion can produce this report.
#[derive(Debug, Default)]
pub(crate) struct CosyVoice3Report {
    /// Always zero because conversion is refused before parsing.
    pub(crate) written: usize,
    /// Always zero because conversion is refused before parsing.
    pub(crate) skipped_non_float: usize,
    /// No derived hparams exist at this boundary.
    pub(crate) derived: Option<DerivedHparams>,
    /// Always false because no tokenizer is embedded.
    pub(crate) tokenizer_embedded: bool,
    /// Diagnostics retained for generic dispatch compatibility.
    pub(crate) notes: Vec<String>,
}

/// Placeholder diagnostic shape retained solely for the generic formatter.
#[derive(Debug, Clone, Copy)]
pub(crate) struct DerivedHparams {
    /// Vocabulary size, unavailable before authenticated conversion.
    pub(crate) vocab_size: u32,
    /// Hidden width, unavailable before authenticated conversion.
    pub(crate) hidden_dim: u32,
    /// Layer count, unavailable before authenticated conversion.
    pub(crate) n_layer: u32,
    /// FFN width, unavailable before authenticated conversion.
    pub(crate) ffn_dim: u32,
    /// Attention head count, unavailable before authenticated conversion.
    pub(crate) n_head: u32,
    /// KV head count, unavailable before authenticated conversion.
    pub(crate) n_head_kv: u32,
    /// Context length, unavailable before authenticated conversion.
    pub(crate) n_ctx: u32,
    /// Attention bias status, unavailable before authenticated conversion.
    pub(crate) has_attn_bias: bool,
}

/// Refuses a CosyVoice3 input before reading or interpreting its bytes.
#[cfg(test)]
pub(crate) fn convert(bytes: Vec<u8>) -> Result<(GgufBuilder, CosyVoice3Report), ConvertError> {
    convert_with_config(bytes, None)
}

/// Refuses conversion regardless of an optional config sidecar.
pub(crate) fn convert_with_config(
    bytes: Vec<u8>,
    config_json: Option<&[u8]>,
) -> Result<(GgufBuilder, CosyVoice3Report), ConvertError> {
    convert_with_config_and_tokenizer::<()>(bytes, config_json, None)
}

/// Refuses conversion regardless of input, config, or tokenizer arguments.
///
/// The generic tokenizer parameter intentionally avoids coupling this refusal
/// boundary to CosyVoice2 converter internals while preserving the existing
/// call-site signature shape. Arguments are never inspected.
pub(crate) fn convert_with_config_and_tokenizer<T>(
    _bytes: Vec<u8>,
    _config_json: Option<&[u8]>,
    _tokenizer: Option<T>,
) -> Result<(GgufBuilder, CosyVoice3Report), ConvertError> {
    Err(ConvertError::Usage(
        "CosyVoice3 conversion is INSPECTION_ONLY: the authenticated Qwen/flow/HiFT/speech-tokenizer composite and full native binder are not complete; no GGUF or provenance was produced".to_owned(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn arbitrary_input_is_refused_without_builder_or_output() {
        let error = convert(b"not-a-checkpoint".to_vec()).unwrap_err();
        assert!(error.to_string().contains("INSPECTION_ONLY"));
        assert!(!Path::new("cosyvoice3.gguf").exists());
    }

    #[test]
    fn config_and_tokenizer_cannot_bypass_refusal() {
        let error = convert_with_config_and_tokenizer(
            vec![0; 8],
            Some(br#"{"hidden_size": 896}"#),
            Some((b"vocab".to_vec(), b"merges".to_vec())),
        )
        .unwrap_err();
        assert!(error.to_string().contains("INSPECTION_ONLY"));
    }

    #[test]
    fn public_constants_remain_distinct_dispatch_labels() {
        assert_eq!(ARCH, "cosyvoice3");
        assert_eq!(NAME, "fun-cosyvoice3-0.5b-2512");
    }
}
