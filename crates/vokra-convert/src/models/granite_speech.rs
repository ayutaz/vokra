//! IBM Granite Speech 4.1-2B inspection-only conversion boundary.
//!
//! The release is a multi-shard composite with an auxiliary safetensors file
//! and signed metadata. Until VAST authenticates the complete tree, source,
//! signature resources, and license boundaries, no runtime-looking GGUF may
//! be produced.

use std::path::Path;

use crate::ConvertError;

/// Supported Granite Speech checkpoint identities.  This remains a type-level
/// dispatch contract only; no variant is currently conversion-ready.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraniteSpeechVariant {
    /// IBM Granite Speech 4.1-2B composite release.
    V4_1_2B,
}

/// Compatibility report retained while conversion is fail-closed.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct GraniteSpeechReport {
    /// Number of tensors observed by a future authenticated converter.
    pub read: usize,
    /// Number of tensors written by a future authenticated converter.
    pub written: usize,
    /// Number of non-floating tensors skipped by a future authenticated converter.
    pub skipped_non_float: usize,
    /// Number of BF16 tensors preserved by a future authenticated converter.
    pub bf16_passthrough: usize,
}

/// Reject arbitrary or merged safetensors until the VAST inspection contract
/// is accepted. This function does not read checkpoint bytes or write output.
pub fn convert_granite_speech_file(
    _input: &Path,
    _output: &Path,
    variant: GraniteSpeechVariant,
    _license: Option<&str>,
) -> Result<GraniteSpeechReport, ConvertError> {
    Err(ConvertError::Usage(match variant {
        GraniteSpeechVariant::V4_1_2B => {
            "Granite Speech 4.1-2B conversion is INSPECTION_ONLY until VAST authenticates the fixed HF/source identities, sharded tensor manifest, model.sig resources, and Apache-2.0 license contract"
        }.to_owned()
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arbitrary_input_is_rejected() {
        let error = convert_granite_speech_file(
            Path::new("/does/not/exist.safetensors"),
            Path::new("/tmp/granite-speech.gguf"),
            GraniteSpeechVariant::V4_1_2B,
            None,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("INSPECTION_ONLY"), "{error}");
    }

    #[test]
    fn license_relabels_cannot_bypass_gate() {
        for license in [Some("apache-2.0"), Some("mit")] {
            let error = convert_granite_speech_file(
                Path::new("/does/not/exist.safetensors"),
                Path::new("/tmp/granite-speech.gguf"),
                GraniteSpeechVariant::V4_1_2B,
                license,
            )
            .unwrap_err()
            .to_string();
            assert!(error.contains("INSPECTION_ONLY"), "{error}");
            assert!(error.contains("model.sig"), "{error}");
        }
    }
}
