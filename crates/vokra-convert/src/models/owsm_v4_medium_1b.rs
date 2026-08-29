//! ESPnet OWSM v4 medium 1B inspection-only conversion boundary.
//!
//! The release is a PyTorch composite (ESPnet frontend/encoder/decoder,
//! SentencePiece, and global-MVN statistics). Until VAST authenticates the
//! complete fixed tree and safe checkpoint evidence, arbitrary inputs must
//! not become runtime-looking GGUF.

use std::path::Path;

use crate::ConvertError;

/// Compatibility report retained for the existing converter dispatch.
#[derive(Debug, Default)]
pub struct OwsmV4Medium1bReport {
    /// Number of tensors observed by a future authenticated converter.
    pub read: usize,
    /// Number of tensors written by a future authenticated converter.
    pub written: usize,
    /// Number of non-floating tensors skipped by a future authenticated converter.
    pub skipped_non_float: usize,
    /// Number of BF16 tensors preserved by a future authenticated converter.
    pub bf16_passthrough: usize,
}

/// Refuse arbitrary safetensors and license relabels until the VAST
/// inspection contract is authenticated. This function never reads input or
/// creates output.
pub fn convert_owsm_v4_medium_1b_file(
    _input: &Path,
    _output: &Path,
    _license: Option<&str>,
) -> Result<OwsmV4Medium1bReport, ConvertError> {
    Err(ConvertError::Usage(
        "OWSM v4 medium 1B conversion is INSPECTION_ONLY until VAST authenticates the fixed HF tree, safe PyTorch checkpoint, exact ESPnet config, SentencePiece/BPE, MVN stats, source identities, and CC-BY-4.0 contract".to_owned(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arbitrary_input_is_rejected_without_output() {
        let output = std::env::temp_dir().join("owsm-v4-medium-1b-rejected.gguf");
        let _ = std::fs::remove_file(&output);
        let error =
            convert_owsm_v4_medium_1b_file(Path::new("/does/not/exist.safetensors"), &output, None)
                .unwrap_err()
                .to_string();
        assert!(error.contains("INSPECTION_ONLY"), "{error}");
        assert!(!output.exists());
    }

    #[test]
    fn license_override_cannot_bypass_gate() {
        for license in [Some("cc-by-4.0"), Some("apache-2.0"), Some("mit")] {
            let error = convert_owsm_v4_medium_1b_file(
                Path::new("/does/not/exist.safetensors"),
                Path::new("/tmp/owsm-v4-medium-1b.gguf"),
                license,
            )
            .unwrap_err()
            .to_string();
            assert!(error.contains("INSPECTION_ONLY"), "{error}");
        }
    }
}
