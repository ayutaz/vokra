//! ChatTTS composite inspection-only conversion boundary.
//!
//! The upstream release contains GPT, Embed, DVAE, Decoder, Vocos, tokenizer,
//! and legacy pickle assets. Until the full release is authenticated on VAST
//! and a clean-room native binder exists, arbitrary input and license
//! relabeling are rejected before reading and no GGUF is produced.

use std::path::Path;

use crate::ConvertError;

/// Runtime architecture tag reserved for a future authenticated binder.
pub const ARCH: &str = "chattts";
/// Canonical model name.
pub const NAME: &str = "chattts";
/// Model category.
pub const CATEGORY: &str = "tts";
/// Canonical Hugging Face release.
pub const UPSTREAM_HF: &str = "2Noise/ChatTTS";
/// Weight-card SPDX identifier; not an authorization to publish or load.
pub const DEFAULT_LICENSE_SPDX: &str = "cc-by-nc-4.0";
/// GGUF category metadata key retained for API compatibility.
pub(crate) const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
/// GGUF upstream metadata key retained for API compatibility.
pub(crate) const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";

/// Legacy report type retained for source compatibility. No production call
/// can return it while ChatTTS is inspection-only.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct ChatTtsReport {
    /// Number of input tensor records, retained for source compatibility.
    pub read: usize,
    /// Number of output tensor records; always zero under inspection-only refusal.
    pub written: usize,
    /// Number of non-floating records skipped by a future preparer.
    pub skipped_non_float: usize,
    /// Number of BF16 records seen by a future preparer.
    pub bf16_passthrough: usize,
}

/// Refuses conversion before opening `input`; license overrides cannot bypass
/// the inspection-only boundary and `output` is never created.
pub fn convert_chattts_file(
    _input: &Path,
    _output: &Path,
    _license: Option<&str>,
) -> Result<ChatTtsReport, ConvertError> {
    // Keep the reserved metadata contract visible to this refusal path so
    // callers cannot mistake these labels for an authorization to convert.
    let _reserved_metadata = (
        ARCH,
        NAME,
        CATEGORY,
        UPSTREAM_HF,
        DEFAULT_LICENSE_SPDX,
        KEY_MODEL_CATEGORY,
        KEY_PROVENANCE_UPSTREAM_HF,
    );
    Err(ConvertError::Usage(
        "chattts: INSPECTION_ONLY — composite release requires authenticated VAST evidence and a native clean-room binder; no GGUF is produced".to_owned(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn public_converter_is_unconditionally_inspection_only() {
        let output = std::env::temp_dir().join(format!("chattts-output-{}", std::process::id()));
        let err = convert_chattts_file(
            Path::new("missing.safetensors"),
            &output,
            Some("apache-2.0"),
        )
        .expect_err("ChatTTS conversion must fail closed");
        assert!(err.to_string().contains("INSPECTION_ONLY"));
        assert!(!output.exists(), "refusal must not create a GGUF");
    }

    #[test]
    fn license_override_cannot_bypass_refusal() {
        let err = convert_chattts_file(
            Path::new("missing.safetensors"),
            Path::new("unused.gguf"),
            None,
        )
        .expect_err("arbitrary input must be rejected");
        assert!(err.to_string().contains("no GGUF is produced"));
    }
}
