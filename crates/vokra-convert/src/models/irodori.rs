//! Fail-closed conversion boundary for Irodori-TTS-500M-v3.
//!
//! The public release is a composite Japanese TTS checkpoint: RF-DiT,
//! reference conditioning, tokenizer, duration path, and the separate
//! Semantic-DACVAE-Japanese-32dim decoder. A lone safetensors file cannot
//! establish that contract, so conversion remains inspection-only until a
//! VAST-authenticated composite binder and parity record are reviewed.

use vokra_core::gguf::GgufBuilder;

use crate::ConvertError;

/// Conversion accounting retained for the stable dispatcher/report API.
#[derive(Debug, Default)]
pub(crate) struct IrodoriReport {
    pub(crate) written: usize,
    pub(crate) skipped_non_float: usize,
    pub(crate) notes: Vec<String>,
}

/// Refuse arbitrary single-file conversion. No GGUF builder is returned and
/// no provenance/license metadata is stamped on an unauthenticated input.
pub(crate) fn convert(_bytes: Vec<u8>) -> Result<(GgufBuilder, IrodoriReport), ConvertError> {
    Err(ConvertError::Usage(
        "irodori: INSPECTION_ONLY — conversion requires an authenticated exact composite (Irodori-TTS-500M-v3 RF-DiT checkpoint, tokenizer/reference components, duration path, and Semantic-DACVAE-Japanese-32dim codec); no GGUF is produced".to_owned(),
    ))
}

#[cfg(test)]
mod tests {
    use super::convert;

    #[test]
    fn arbitrary_input_is_refused_without_output() {
        let error = convert(vec![0; 64]).expect_err("Irodori conversion must fail closed");
        let message = error.to_string();
        assert!(message.contains("INSPECTION_ONLY"), "{message}");
        assert!(
            message.contains("Semantic-DACVAE-Japanese-32dim"),
            "{message}"
        );
    }

    #[test]
    fn empty_input_is_refused_without_output() {
        let error =
            convert(Vec::new()).expect_err("empty input must not create metadata-only GGUF");
        assert!(error.to_string().contains("no GGUF is produced"));
    }
}
