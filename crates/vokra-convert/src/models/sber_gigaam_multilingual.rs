//! Fail-closed conversion boundary for the GigaAM Multilingual CTC release.
//!
//! This release is a distinct CTC model with a 71-class head, unlike the
//! GigaAM v3 RNNT release. Until the fixed HF tree, source identities, exact
//! config, tensor manifest, and vocabulary are authenticated, arbitrary input
//! must not be emitted as a runtime-looking GGUF.

use std::path::Path;

use crate::ConvertError;

/// Compatibility report retained for converter dispatch callers.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct SberGigaamMultilingualReport {
    /// Tensors inspected by a future authenticated conversion.
    pub read: usize,
    /// Tensors written by a future authenticated conversion.
    pub written: usize,
    /// Non-floating tensors skipped by a future authenticated conversion.
    pub skipped_non_float: usize,
    /// BF16 tensors preserved by a future authenticated conversion.
    pub bf16_passthrough: usize,
}

/// Refuse every input and license override until the fixed multilingual CTC
/// contract is authenticated by VAST. No input is read and no output is
/// created.
pub fn convert_sber_gigaam_multilingual_file(
    _input: &Path,
    _output: &Path,
    _license: Option<&str>,
) -> Result<SberGigaamMultilingualReport, ConvertError> {
    Err(ConvertError::Usage(
        "GigaAM Multilingual conversion is INSPECTION_ONLY: this is a 71-class CTC release distinct from the GigaAM v3 RNNT topology; fixed HF/source tensor and vocabulary evidence is required before conversion".to_owned(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arbitrary_input_is_rejected_without_output() {
        let output = std::env::temp_dir().join("gigaam-multilingual-inspection-only.gguf");
        let _ = std::fs::remove_file(&output);
        let error = convert_sber_gigaam_multilingual_file(Path::new("missing"), &output, None)
            .unwrap_err()
            .to_string();
        assert!(error.contains("INSPECTION_ONLY"));
        assert!(!output.exists());
    }

    #[test]
    fn license_override_cannot_bypass_gate() {
        for license in [Some("mit"), Some("apache-2.0"), Some("cc-by-4.0")] {
            let error = convert_sber_gigaam_multilingual_file(
                Path::new("missing"),
                Path::new("out"),
                license,
            )
            .unwrap_err()
            .to_string();
            assert!(error.contains("INSPECTION_ONLY"));
        }
    }
}
