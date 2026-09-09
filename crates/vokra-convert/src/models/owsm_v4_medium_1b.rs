//! ESPnet OWSM v4 medium 1B inspection-only conversion boundary.
//!
//! The release is a PyTorch composite (ESPnet frontend/encoder/decoder,
//! SentencePiece, and global-MVN statistics). VAST has authenticated the
//! fixed source tree, checkpoint container, and the 1,172-name/shape/dtype
//! structural inventory. Per-tensor payload mapping/hashes and normalization,
//! the GGUF writer contract, frontend/decoder/tokenizer/CTC-attention
//! execution, independent real-weight parity, and dependency/dataset review
//! remain incomplete, so arbitrary inputs must not become runtime-looking
//! GGUF.

use std::path::Path;

use crate::ConvertError;

/// Fixed source identities authenticated by the VAST inspection packet.
pub const HF_REPOSITORY: &str = "espnet/owsm_v4_medium_1B";
pub const HF_REVISION: &str = "e10985c8f1d592e905c24d2ac2b2c53e3feb24dc";
pub const SOURCE_REVISION: &str = "cccc29023d43a3f504e28df7d1324bb4eb6daedd";
pub const CHECKPOINT_SHA256: &str =
    "b02d79f29a4daa31dd49ce145d9bb4cda0a1b68cdad91ae0af170ec3a4e92e09";
pub const CHECKPOINT_TENSOR_COUNT: usize = 1172;
pub const INSPECTION_MANIFEST_SHA256: &str =
    "82de20eea3cf3a247624c76cd8e108e562addda0c8582577515cf88abb3053d9";
pub const INSPECTION_LOG_SHA256: &str =
    "4df29428ea8ce381311c5e407d937b6a517750f4edcbc88b8c606cdef82dc93b";

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

/// Refuse arbitrary safetensors and license relabels while the post-inspection
/// conversion contract remains incomplete. This function never reads input or
/// creates output.
pub fn convert_owsm_v4_medium_1b_file(
    _input: &Path,
    _output: &Path,
    _license: Option<&str>,
) -> Result<OwsmV4Medium1bReport, ConvertError> {
    Err(ConvertError::Usage(format!(
        "OWSM v4 medium 1B conversion is INSPECTION_ONLY: VAST authenticated {HF_REPOSITORY}@{HF_REVISION}, ESPnet source {SOURCE_REVISION}, checkpoint sha256 {CHECKPOINT_SHA256}, {CHECKPOINT_TENSOR_COUNT} tensors in the structural inventory, inspection manifest {INSPECTION_MANIFEST_SHA256}, and log {INSPECTION_LOG_SHA256}; conversion remains blocked pending per-tensor payload mapping/hashes and normalization, the GGUF writer contract, frontend/decoder/tokenizer/CTC-attention execution, independent real-weight parity, and dependency/dataset review"
    )))
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

    #[test]
    fn reviewed_source_identity_is_pinned() {
        assert_eq!(HF_REPOSITORY, "espnet/owsm_v4_medium_1B");
        assert_eq!(HF_REVISION.len(), 40);
        assert_eq!(SOURCE_REVISION.len(), 40);
        assert_eq!(CHECKPOINT_SHA256.len(), 64);
        assert_eq!(INSPECTION_MANIFEST_SHA256.len(), 64);
        assert_eq!(INSPECTION_LOG_SHA256.len(), 64);
        assert_eq!(CHECKPOINT_TENSOR_COUNT, 1172);
    }
}
