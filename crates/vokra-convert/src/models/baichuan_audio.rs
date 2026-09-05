//! Baichuan-Audio-Instruct inspection-only conversion boundary.
//!
//! The canonical model is a multi-shard, approximately 21 GB composite
//! checkpoint with bundled custom code and several upstream dependencies. No
//! arbitrary safetensors input may be relabeled or emitted as a GGUF until a
//! VAST inspection authenticates the complete composition and licenses.

use std::path::Path;

use crate::ConvertError;

/// Compatibility report retained while conversion is fail-closed.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct BaichuanAudioReport {
    /// Number of tensors observed by a future authenticated converter.
    pub read: usize,
    /// Number of tensors written by a future authenticated converter.
    pub written: usize,
    /// Number of non-floating tensors skipped by a future authenticated converter.
    pub skipped_non_float: usize,
    /// Number of BF16 tensors preserved by a future authenticated converter.
    pub bf16_passthrough: usize,
}

/// Reject arbitrary and canonical-looking inputs until the VAST evidence
/// contract is reviewed. This function never reads or writes checkpoint data.
pub fn convert_baichuan_audio_file(
    _input: &Path,
    _output: &Path,
    _license: Option<&str>,
) -> Result<BaichuanAudioReport, ConvertError> {
    Err(ConvertError::Usage(
        "Baichuan-Audio-Instruct conversion is INSPECTION_ONLY until VAST authenticates the fixed HF/source identities, five-shard tensor manifest, bundled custom code, and separate license contracts".to_owned(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arbitrary_input_is_rejected_without_output() {
        let error = convert_baichuan_audio_file(
            Path::new("/does/not/exist.safetensors"),
            Path::new("/tmp/baichuan-audio.gguf"),
            None,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("INSPECTION_ONLY"), "{error}");
    }

    #[test]
    fn canonical_and_permissive_labels_do_not_bypass_gate() {
        for license in [None, Some("apache-2.0"), Some("mit")] {
            let error = convert_baichuan_audio_file(
                Path::new("/does/not/exist.safetensors"),
                Path::new("/tmp/baichuan-audio.gguf"),
                license,
            )
            .unwrap_err()
            .to_string();
            assert!(error.contains("INSPECTION_ONLY"), "{error}");
            assert!(error.contains("five-shard"), "{error}");
        }
    }
}
