//! VieNeu-TTS-v3-Turbo inspection-only conversion boundary.
//!
//! The current public artifact combines a safetensors model with ONNX
//! subgraphs and an external MOSS Audio Tokenizer companion. Until a VAST
//! inspection authenticates the immutable HF/source identities, complete
//! tensor manifest, ONNX contracts, and dependency/license boundaries, this
//! module must not produce a runtime-looking GGUF.

use std::path::Path;

use crate::ConvertError;

/// Compatibility report retained for callers while product conversion is
/// disabled. No instance is returned until the inspection gate is cleared.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct VieNeuReport {
    /// Number of tensors observed by a future authenticated converter.
    pub read: usize,
    /// Number of floating-point tensors written by a future authenticated converter.
    pub written: usize,
    /// Number of non-floating-point tensors skipped by a future authenticated converter.
    pub skipped_non_float: usize,
    /// Number of BF16 tensors preserved by a future authenticated converter.
    pub bf16_passthrough: usize,
}

/// Reject arbitrary safetensors and all metadata/license relabels until VAST
/// authenticates the VieNeu model, ONNX subgraphs, tokenizer companion, and
/// source/dependency contract.
pub fn convert_vieneu_file(
    _input: &Path,
    _output: &Path,
    _license: Option<&str>,
) -> Result<VieNeuReport, ConvertError> {
    Err(ConvertError::Usage(
        "VieNeu-TTS-v3-Turbo conversion is INSPECTION_ONLY until VAST authenticates the fixed HF/source identities, tensor manifest, ONNX contracts, and MOSS tokenizer companion".to_owned(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arbitrary_input_is_inspection_only() {
        let error = convert_vieneu_file(
            Path::new("/does/not/exist.safetensors"),
            Path::new("/tmp/vieneu-v3-turbo.gguf"),
            None,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("INSPECTION_ONLY"), "{error}");
        assert!(error.contains("ONNX"), "{error}");
    }

    #[test]
    fn license_relabel_cannot_bypass_inspection_gate() {
        let error = convert_vieneu_file(
            Path::new("/does/not/exist.safetensors"),
            Path::new("/tmp/vieneu-v3-turbo.gguf"),
            Some("mit"),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("INSPECTION_ONLY"), "{error}");
    }
}
