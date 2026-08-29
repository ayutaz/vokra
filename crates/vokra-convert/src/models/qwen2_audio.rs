#![allow(clippy::doc_lazy_continuation)]
//! **Qwen2-Audio-7B-Instruct** (`Qwen/Qwen2-Audio-7B-Instruct`, license
//! pending audit): inspection-only boundary for the audio/text-to-text model.
//!
//! The 7B audio LLM combines a Whisper audio encoder with a Qwen2 decoder.
//! Its five-shard checkpoint is VAST-only; native runtime, conversion, and
//! parity are intentionally not implemented here.

use std::path::Path;

use crate::ConvertError;

pub const ARCH: &str = "qwen2_audio";
pub const NAME: &str = "qwen2-audio-7b-instruct";
pub const CATEGORY: &str = "audio-llm";
pub const UPSTREAM_HF: &str = "Qwen/Qwen2-Audio-7B-Instruct";
pub const UPSTREAM_HF_REVISION: &str = "0a095220c30b7b31434169c3086508ef3ea5bf0a";
pub const OFFICIAL_SOURCE_REPOSITORY: &str = "https://github.com/QwenLM/Qwen2-Audio.git";
pub const OFFICIAL_SOURCE_REVISION: &str = "595360e82b5839c1507492ec83cae5bda6d5c7d4";
pub const TRANSFORMERS_TAG: &str = "v4.45.0";
pub const TRANSFORMERS_REVISION: &str = "2ef31dec1676249d26044a8aa8abe33dbecf0d10";

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Qwen2AudioReport {
    pub read: usize,
    pub written: usize,
    pub skipped_non_float: usize,
    pub bf16_passthrough: usize,
}

pub fn convert_qwen2_audio_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<Qwen2AudioReport, ConvertError> {
    let _ = (input, output, license);
    Err(ConvertError::Usage(format!(
        "Qwen2-Audio conversion is INSPECTION_ONLY: runtime is not implemented and no GGUF may be emitted (HF {UPSTREAM_HF}@{UPSTREAM_HF_REVISION}; source {OFFICIAL_SOURCE_REPOSITORY}@{OFFICIAL_SOURCE_REVISION})"
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn public_conversion_is_explicitly_inspection_only() {
        let error = convert_qwen2_audio_file(
            Path::new("untrusted.safetensors"),
            Path::new("output.gguf"),
            Some("apache-2.0"),
        )
        .expect_err("unreviewed Qwen2-Audio must refuse conversion");
        assert!(error.to_string().contains("INSPECTION_ONLY"));
        assert_eq!(TRANSFORMERS_TAG, "v4.45.0");
        assert_eq!(
            TRANSFORMERS_REVISION,
            "2ef31dec1676249d26044a8aa8abe33dbecf0d10"
        );
    }
}
