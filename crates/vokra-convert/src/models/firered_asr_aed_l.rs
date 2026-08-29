//! Fail-closed inspection boundary for FireRedASR-AED-L.
//!
//! The public FireRed checkpoint is a PyTorch container and the native AED
//! runtime is not yet authenticated or complete. This module therefore never
//! accepts arbitrary safetensors or produces a runtime-looking GGUF.

use std::path::Path;

use crate::ConvertError;

/// Compatibility report retained by the converter dispatch API.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct FireredAsrAedLReport {
    /// Source tensors observed by a successful conversion.
    pub read: usize,
    /// Tensors written by a successful conversion.
    pub written: usize,
    /// Non-floating tensors skipped by a successful conversion.
    pub skipped_non_float: usize,
    /// BF16 tensors passed through by a successful conversion.
    pub bf16_passthrough: usize,
}

/// Canonical architecture identifier retained for alias and dispatch tests.
#[allow(dead_code)] // Retained as inspection-only dispatch metadata until native parity is authenticated.
pub const ARCH: &str = "firered_asr_aed_l";

/// Canonical model name retained for metadata consumers.
#[allow(dead_code)] // Retained as inspection-only model metadata until native parity is authenticated.
pub const NAME: &str = "firered-asr-aed-l";

/// Canonical upstream repository.
#[allow(dead_code)] // Retained as inspection-only provenance until the checkpoint is authenticated.
pub const UPSTREAM_HF: &str = "FireRedTeam/FireRedASR-AED-L";

/// Conversion is deliberately refused until inspection and native parity are
/// independently authenticated.
pub fn convert_firered_asr_aed_l_file(
    _input: &Path,
    _output: &Path,
    _license: Option<&str>,
) -> Result<FireredAsrAedLReport, ConvertError> {
    Err(ConvertError::Usage(
        "FireRedASR-AED-L conversion is INSPECTION_ONLY: authenticated checkpoint and native AED runtime are not implemented; no output was produced".to_owned(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arbitrary_inputs_and_license_overrides_are_refused_without_output() {
        let root =
            std::env::temp_dir().join(format!("vokra-firered-refusal-{}", std::process::id()));
        std::fs::create_dir_all(&root).expect("temp directory");
        let input = root.join("arbitrary.safetensors");
        let output = root.join("must-not-exist.gguf");
        std::fs::write(&input, b"arbitrary").expect("input");
        for license in [None, Some("apache-2.0"), Some("mit"), Some("")] {
            let error = convert_firered_asr_aed_l_file(&input, &output, license)
                .expect_err("inspection-only converter must refuse");
            assert!(error.to_string().contains("INSPECTION_ONLY"));
            assert!(!output.exists());
        }
        std::fs::remove_dir_all(root).expect("cleanup");
    }
}
