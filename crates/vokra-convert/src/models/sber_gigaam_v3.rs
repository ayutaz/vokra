//! Fail-closed conversion boundary for the GigaAM v3 RNNT release.
//!
//! The authenticated release is `ai-sage/GigaAM-v3` at a fixed revision and
//! has a Conformer + RNNT topology. Its complete tensor manifest, tokenizer
//! composition, and safe checkpoint evidence are not part of the converter
//! contract yet. Arbitrary safetensors must therefore never become a
//! runtime-looking GGUF.

use std::path::Path;

use crate::ConvertError;

/// Compatibility report retained for converter dispatch callers.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct SberGigaamV3Report {
    /// Tensors inspected by a future authenticated conversion.
    pub read: usize,
    /// Tensors written by a future authenticated conversion.
    pub written: usize,
    /// Non-floating tensors skipped by a future authenticated conversion.
    pub skipped_non_float: usize,
    /// BF16 tensors preserved by a future authenticated conversion.
    pub bf16_passthrough: usize,
}

/// Refuse every input and license override until the fixed RNNT contract is
/// authenticated by the VAST inspection wave. No input is read and no output
/// is created.
pub fn convert_sber_gigaam_v3_file(
    _input: &Path,
    _output: &Path,
    _license: Option<&str>,
) -> Result<SberGigaamV3Report, ConvertError> {
    Err(ConvertError::Usage(
        "GigaAM v3 conversion is INSPECTION_ONLY: v3 is RNNT, not the multilingual CTC topology; fixed HF/source tensor and tokenizer evidence is required before conversion".to_owned(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arbitrary_input_is_rejected_without_output() {
        let output = std::env::temp_dir().join("gigaam-v3-inspection-only.gguf");
        let _ = std::fs::remove_file(&output);
        let error = convert_sber_gigaam_v3_file(Path::new("missing"), &output, None)
            .unwrap_err()
            .to_string();
        assert!(error.contains("INSPECTION_ONLY"));
        assert!(!output.exists());
    }

    #[test]
    fn license_override_cannot_bypass_gate() {
        for license in [Some("mit"), Some("apache-2.0"), Some("cc-by-4.0")] {
            let error =
                convert_sber_gigaam_v3_file(Path::new("missing"), Path::new("out"), license)
                    .unwrap_err()
                    .to_string();
            assert!(error.contains("INSPECTION_ONLY"));
        }
    }
}
