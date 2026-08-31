//! SGMSE-VoiceBank inspection-only conversion boundary.
//!
//! The public 647-tensor artifact and arbitrary safetensors are not accepted
//! as a runtime checkpoint. A VAST inspection must first authenticate the
//! fixed SpeechBrain checkpoint, safe-load container, EMA extraction, full
//! manifest, and upstream implementation contract.

use std::path::Path;

use crate::ConvertError;

/// Compatibility report retained for callers while product conversion is
/// disabled. No instance is returned until the inspection gate is cleared.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct SgmseReport {
    /// Number of tensors observed by a future authenticated converter.
    pub read: usize,
    /// Number of floating-point tensors written by a future authenticated converter.
    pub written: usize,
    /// Number of non-floating-point tensors skipped by a future authenticated converter.
    pub skipped_non_float: usize,
    /// Number of BF16 tensors preserved by a future authenticated converter.
    pub bf16_passthrough: usize,
}

/// Reject every candidate until the fixed upstream checkpoint and complete
/// tensor contract have been authenticated on VAST. In particular, this
/// prevents empty inputs and permissive license relabels from producing a
/// product-facing GGUF.
pub fn convert_sgmse_file(
    _input: &Path,
    _output: &Path,
    _license: Option<&str>,
) -> Result<SgmseReport, ConvertError> {
    Err(ConvertError::Usage(
        "SGMSE-VoiceBank conversion is AUTHENTICATED_MANIFEST_REQUIRED: VAST must authenticate the fixed checkpoint, safe-load container, EMA extraction, and complete tensor manifest before conversion".to_owned(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arbitrary_checkpoint_requires_authenticated_manifest() {
        let error = convert_sgmse_file(
            Path::new("/does/not/exist.safetensors"),
            Path::new("/tmp/sgmse-voicebank.gguf"),
            None,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("AUTHENTICATED_MANIFEST_REQUIRED"), "{error}");
        assert!(error.contains("complete tensor manifest"), "{error}");
    }

    #[test]
    fn permissive_relabel_cannot_bypass_checkpoint_gate() {
        let error = convert_sgmse_file(
            Path::new("/does/not/exist.safetensors"),
            Path::new("/tmp/sgmse-voicebank.gguf"),
            Some("mit"),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("AUTHENTICATED_MANIFEST_REQUIRED"), "{error}");
    }
}
