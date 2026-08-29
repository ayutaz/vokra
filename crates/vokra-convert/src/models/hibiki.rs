//! Kyutai Hibiki-2B inspection-only conversion boundary.
//!
//! Hibiki is a composite speech-to-speech translation release (LM, Mimi
//! tokenizer, SentencePiece, and streaming configuration). Until VAST
//! authenticates the complete fixed revision and source contracts, arbitrary
//! safetensors must not become a runtime-looking GGUF.

use std::path::Path;

use crate::ConvertError;

/// Compatibility report retained for existing converter dispatch.
#[derive(Debug, Default)]
pub struct HibikiReport {
    /// Number of tensors observed by a future authenticated converter.
    pub read: usize,
    /// Number of tensors written by a future authenticated converter.
    pub written: usize,
    /// Number of non-floating tensors skipped by a future authenticated converter.
    pub skipped_non_float: usize,
    /// Number of BF16 tensors preserved by a future authenticated converter.
    pub bf16_passthrough: usize,
}

/// Reject arbitrary or merged inputs until the fixed HF, Mimi, SentencePiece,
/// source, license, and tensor-manifest evidence is accepted by VAST.
pub fn convert_hibiki_file(
    _input: &Path,
    _output: &Path,
    _license: Option<&str>,
) -> Result<HibikiReport, ConvertError> {
    Err(ConvertError::Usage(
        "Hibiki-2B conversion is INSPECTION_ONLY until VAST authenticates the fixed HF tree, Mimi tokenizer, SentencePiece structure, source identities, tensor manifests, and CC-BY-4.0 contract".to_owned(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arbitrary_input_is_rejected() {
        let error = convert_hibiki_file(
            Path::new("/does/not/exist.safetensors"),
            Path::new("/tmp/hibiki.gguf"),
            None,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("INSPECTION_ONLY"), "{error}");
    }

    #[test]
    fn license_override_cannot_bypass_gate() {
        for license in [Some("cc-by-4.0"), Some("apache-2.0"), Some("mit")] {
            let error = convert_hibiki_file(
                Path::new("/does/not/exist.safetensors"),
                Path::new("/tmp/hibiki.gguf"),
                license,
            )
            .unwrap_err()
            .to_string();
            assert!(error.contains("INSPECTION_ONLY"), "{error}");
            assert!(error.contains("SentencePiece"), "{error}");
        }
    }
}
