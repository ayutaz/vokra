//! Fail-closed boundary for the CosyVoice2 composite checkpoint.
//!
//! The official release is a composite of LLM, flow, HiFT, speech-tokenizer,
//! and speaker components. A single arbitrary safetensors input therefore
//! cannot be represented by the converter's one-file GGUF API. Until a
//! complete, authenticated component manifest and binder are reviewed, every
//! public conversion entry point is inspection-only and emits no output.

use vokra_core::gguf::GgufBuilder;

use crate::ConvertError;

/// Hparams retained for the converter's operator-note compatibility surface.
/// No values are produced while conversion is inspection-only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct DerivedHparams {
    pub(crate) vocab_size: u32,
    pub(crate) hidden_dim: u32,
    pub(crate) n_layer: u32,
    pub(crate) ffn_dim: u32,
    pub(crate) n_head: u32,
    pub(crate) n_head_kv: u32,
    pub(crate) n_ctx: u32,
    pub(crate) has_attn_bias: bool,
}

/// Conversion report retained for the existing CLI dispatch surface.
#[derive(Debug, Default)]
pub(crate) struct CosyVoice2Report {
    pub(crate) written: usize,
    pub(crate) skipped_non_float: usize,
    #[allow(dead_code)]
    pub(crate) bf16_passthrough: usize,
    pub(crate) derived: Option<DerivedHparams>,
    pub(crate) tokenizer_embedded: bool,
    pub(crate) notes: Vec<String>,
}

/// Raw tokenizer sidecars retained for API compatibility with dispatch.
#[allow(dead_code)]
pub(crate) struct TokenizerFiles<'a> {
    pub(crate) vocab_json: &'a [u8],
    pub(crate) merges_txt: &'a [u8],
}

fn inspection_only() -> ConvertError {
    ConvertError::Usage(
        "CosyVoice2 conversion is INSPECTION_ONLY: the composite llm.pt + flow.pt + hift.pt + speech-tokenizer contract has no reviewed complete GGUF binder; no output was written"
            .to_owned(),
    )
}

/// Refuses conversion of an arbitrary single-file input.
pub(crate) fn convert(_bytes: Vec<u8>) -> Result<(GgufBuilder, CosyVoice2Report), ConvertError> {
    Err(inspection_only())
}

/// Refuses conversion while preserving the existing dispatch signature.
#[allow(dead_code)] // Staged until the composite binder is authenticated.
pub(crate) fn convert_with_config(
    _bytes: Vec<u8>,
    _config_json: Option<&[u8]>,
) -> Result<(GgufBuilder, CosyVoice2Report), ConvertError> {
    Err(inspection_only())
}

/// Refuses conversion while preserving tokenizer-sidecar compatibility.
pub(crate) fn convert_with_config_and_tokenizer(
    _bytes: Vec<u8>,
    _config_json: Option<&[u8]>,
    _tokenizer: Option<TokenizerFiles<'_>>,
) -> Result<(GgufBuilder, CosyVoice2Report), ConvertError> {
    Err(inspection_only())
}

#[cfg(test)]
mod inspection_tests {
    use super::*;

    #[test]
    fn all_conversion_surfaces_refuse_without_output() {
        let calls = [
            convert(Vec::new()),
            convert_with_config(Vec::new(), None),
            convert_with_config_and_tokenizer(Vec::new(), None, None),
        ];
        for result in calls {
            let error = result.expect_err("composite conversion must fail closed");
            let message = error.to_string();
            assert!(message.contains("INSPECTION_ONLY"));
            assert!(message.contains("no output"));
        }
    }

    #[test]
    fn license_or_arbitrary_input_cannot_bypass_refusal() {
        let error = convert_with_config_and_tokenizer(
            b"arbitrary bytes".to_vec(),
            Some(br#"{"license":"Apache-2.0"}"#),
            Some(TokenizerFiles {
                vocab_json: br#"{"x":0}"#,
                merges_txt: b"x y",
            }),
        )
        .expect_err("license metadata must not bypass composite gate");
        assert!(error.to_string().contains("INSPECTION_ONLY"));
    }
}
