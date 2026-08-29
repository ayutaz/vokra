//! Qwen2.5-Omni-7B inspection-only conversion boundary.
//!
//! This is a multimodal Thinker/Talker composite checkpoint. Its 5-shard
//! weights, speaker archive, custom code, and component provenance are not
//! authenticated by the converter, so arbitrary safetensors and license
//! overrides must never emit a GGUF.

use std::path::Path;

use crate::ConvertError;

#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "inspection-only dispatch metadata is retained until the composite binder is authenticated"
    )
)]
pub const ARCH: &str = "qwen2-omni";
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "inspection-only model metadata is retained until the composite binder is authenticated"
    )
)]
pub const NAME: &str = "qwen2-5-omni-7b";
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "inspection-only model metadata is retained until the composite binder is authenticated"
    )
)]
pub const CATEGORY: &str = "audio-llm";
pub const UPSTREAM_HF: &str = "Qwen/Qwen2.5-Omni-7B";
pub const UPSTREAM_HF_REVISION: &str = "ae9e1690543ffd5c0221dc27f79834d0294cba00";
pub const OFFICIAL_SOURCE_REPOSITORY: &str = "https://github.com/QwenLM/Qwen2.5-Omni.git";
pub const OFFICIAL_SOURCE_REVISION: &str = "d8a31ca56c0456b6edfcbcbf4bdbb6ae2200ef42";
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "inspection-only provenance is retained until the composite binder is authenticated"
    )
)]
pub const TRANSFORMERS_TAG: &str = "v4.52.3";
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "inspection-only provenance is retained until the composite binder is authenticated"
    )
)]
pub const TRANSFORMERS_REVISION: &str = "f4fc42216cd56ab6b68270bf80d811614d8d59e4";

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Qwen25Omni7bReport {
    pub read: usize,
    pub written: usize,
    pub skipped_non_float: usize,
    pub bf16_passthrough: usize,
}

pub fn convert_qwen2_5_omni_7b_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<Qwen25Omni7bReport, ConvertError> {
    let _ = (input, output, license);
    Err(ConvertError::Usage(format!(
        "Qwen2.5-Omni-7B conversion is INSPECTION_ONLY: native multimodal runtime and authenticated Thinker/Talker binding are not implemented; no GGUF may be emitted (HF {UPSTREAM_HF}@{UPSTREAM_HF_REVISION}; source {OFFICIAL_SOURCE_REPOSITORY}@{OFFICIAL_SOURCE_REVISION})"
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn public_conversion_refuses_arbitrary_input_and_license_override() {
        let error = convert_qwen2_5_omni_7b_file(
            Path::new("arbitrary.safetensors"),
            Path::new("must-not-exist.gguf"),
            Some("mit"),
        )
        .expect_err("unreviewed Omni checkpoint must refuse conversion");
        assert!(error.to_string().contains("INSPECTION_ONLY"));
        assert_eq!(UPSTREAM_HF_REVISION.len(), 40);
        assert_eq!(OFFICIAL_SOURCE_REVISION.len(), 40);
        assert_eq!(TRANSFORMERS_TAG, "v4.52.3");
        assert_eq!(
            TRANSFORMERS_REVISION,
            "f4fc42216cd56ab6b68270bf80d811614d8d59e4"
        );
    }
}
