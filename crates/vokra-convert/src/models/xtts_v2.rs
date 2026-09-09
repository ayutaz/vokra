//! XTTS-v2 inspection-only conversion boundary.
//!
//! The upstream release is a multi-file pickle checkpoint bundle covered by
//! Coqui's Public Model License. Until VAST authenticates the immutable HF
//! tree, safe-load inventory, exact source revision, and separate license and
//! provenance evidence, this module must not emit a runtime-looking GGUF.

use std::path::Path;

use crate::ConvertError;

/// Compatibility report retained while XTTS-v2 conversion is disabled.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct XttsV2Report {
    /// Number of tensors observed by a future authenticated converter.
    pub read: usize,
    /// Number of floating-point tensors written by a future authenticated converter.
    pub written: usize,
    /// Number of non-floating-point tensors skipped by a future authenticated converter.
    pub skipped_non_float: usize,
    /// Number of BF16 tensors preserved by a future authenticated converter.
    pub bf16_passthrough: usize,
}

/// Reject arbitrary inputs and license relabels until the VAST inspection
/// contract is reviewed. No pickle, safetensors, or GGUF work occurs here.
pub fn convert_xtts_v2_file(
    _input: &Path,
    _output: &Path,
    _license: Option<&str>,
) -> Result<XttsV2Report, ConvertError> {
    Err(ConvertError::Usage(
        "XTTS-v2 conversion is INSPECTION_ONLY until VAST authenticates the fixed HF/source identities, pickle safe-load inventory, tensor manifest, and Coqui Public Model License contract".to_owned(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arbitrary_input_is_inspection_only() {
        let error = convert_xtts_v2_file(
            Path::new("/does/not/exist.pth"),
            Path::new("/tmp/xtts-v2.gguf"),
            None,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("INSPECTION_ONLY"), "{error}");
        assert!(error.contains("safe-load"), "{error}");
    }

    #[test]
    fn permissive_license_relabel_cannot_bypass_gate() {
        let error = convert_xtts_v2_file(
            Path::new("/does/not/exist.pth"),
            Path::new("/tmp/xtts-v2.gguf"),
            Some("mit"),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("INSPECTION_ONLY"), "{error}");
        assert!(error.contains("Coqui Public Model License"), "{error}");
    }
}
