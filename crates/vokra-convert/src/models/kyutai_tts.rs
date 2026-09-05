//! Fail-closed inspection boundary for Kyutai TTS 1.6B EN/FR.
//!
//! The upstream release is a composite delayed-streams model whose native
//! state machine, Mimi decoder, depformer scheduling, and speaker
//! conditioning are not implemented by Vokra. Conversion is deliberately
//! unavailable until authenticated inspection and native parity evidence land.

use std::path::Path;

use crate::ConvertError;

/// Counters retained for the converter dispatch API. No successful
/// conversion can currently produce a report; all fields remain zero.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct KyutaiTtsReport {
    /// Number of source tensors observed by a successful conversion.
    pub read: usize,
    /// Number of tensors written by a successful conversion.
    pub written: usize,
    /// Number of non-floating tensors skipped by a successful conversion.
    pub skipped_non_float: usize,
    /// Number of BF16 tensors passed through by a successful conversion.
    pub bf16_passthrough: usize,
}

/// Refuse conversion until the fixed Kyutai composite release has an
/// authenticated native binder and parity evidence.
pub fn convert_kyutai_tts_file(
    _input: &Path,
    _output: &Path,
    _license: Option<&str>,
) -> Result<KyutaiTtsReport, ConvertError> {
    Err(ConvertError::Usage(
        "Kyutai TTS conversion is INSPECTION_ONLY: native composite runtime and authenticated parity are not implemented; no output was produced"
            .to_owned(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conversion_is_always_fail_closed_without_output() {
        let root = std::env::temp_dir().join(format!(
            "vokra-kyutai-tts-refusal-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        let input = root.join("arbitrary.safetensors");
        let output = root.join("should-not-exist.gguf");
        std::fs::create_dir_all(&root).expect("temporary directory");
        std::fs::write(&input, b"not an authenticated Kyutai release").expect("input");

        for license in [None, Some("cc-by-4.0"), Some("apache-2.0"), Some("")] {
            assert!(convert_kyutai_tts_file(&input, &output, license).is_err());
            assert!(!output.exists(), "refusal must not create output");
        }

        let err = convert_kyutai_tts_file(&input, &output, None).expect_err("must refuse");
        assert!(err.to_string().contains("INSPECTION_ONLY"));
        std::fs::remove_dir_all(root).expect("cleanup");
    }
}
