//! **Step-Audio-2-mini** inspection-only boundary.
//!
//! The upstream release is a composite, multi-shard speech-to-speech bundle
//! with ONNX and PyTorch token2wav companions. Its provenance, component
//! licenses, and native binding are not authenticated here, so arbitrary
//! safetensors and license relabels must never produce a GGUF.

use std::path::Path;

use crate::ConvertError;

pub(crate) const ARCH: &str = "step_audio2_mini";
pub(crate) const NAME: &str = "step-audio-2-mini";
pub const UPSTREAM_HF: &str = "stepfun-ai/Step-Audio-2-mini";
pub const UPSTREAM_HF_REVISION: &str = "e36fdd5d71e0ea22f09dd94bbab9bfc544ca1e36";
pub const OFFICIAL_SOURCE_REPOSITORY: &str = "https://github.com/stepfun-ai/Step-Audio2.git";
pub const OFFICIAL_SOURCE_REVISION: &str = "76e272b56c3917a8d7188f18bbb5a65dfc8a0845";
pub const TRANSFORMERS_TAG: &str = "v4.49.0";
pub const TRANSFORMERS_REVISION: &str = "a22a4378d97d06b7a1d9abad6e0086d30fdea199";

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct StepAudio2MiniReport {
    pub read: usize,
    pub written: usize,
    pub skipped_non_float: usize,
    pub bf16_passthrough: usize,
}

/// Refuse conversion until the composite checkpoint, token2wav companions,
/// custom code, and component provenance have an independently reviewed
/// contract. `license` is never allowed to relabel input.
pub fn convert_step_audio2_mini_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<StepAudio2MiniReport, ConvertError> {
    let _ = (input, output, license);
    Err(ConvertError::Usage(format!(
        "Step-Audio-2-mini conversion is INSPECTION_ONLY: native runtime and authenticated composite binding are not implemented; no GGUF may be emitted (HF {UPSTREAM_HF}@{UPSTREAM_HF_REVISION}; source {OFFICIAL_SOURCE_REPOSITORY}@{OFFICIAL_SOURCE_REVISION})"
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn public_conversion_is_unconditionally_inspection_only() {
        let error = convert_step_audio2_mini_file(
            Path::new("arbitrary.safetensors"),
            Path::new("should-not-exist.gguf"),
            Some("mit"),
        )
        .expect_err("unreviewed composite checkpoint must refuse conversion");
        assert!(error.to_string().contains("INSPECTION_ONLY"));
        assert_eq!(TRANSFORMERS_TAG, "v4.49.0");
        assert_eq!(
            TRANSFORMERS_REVISION,
            "a22a4378d97d06b7a1d9abad6e0086d30fdea199"
        );
    }
}
