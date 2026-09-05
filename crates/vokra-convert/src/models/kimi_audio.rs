//! Inspection-only boundary for the composite Kimi-Audio release.
//!
//! The official `moonshotai/Kimi-Audio-7B-Instruct` snapshot is a large
//! composite release (sharded language-model weights plus audio components,
//! code, and pickle checkpoints). The generic single-safetensors converter
//! cannot authenticate that layout, so this public conversion entry point is
//! deliberately fail-closed until the VAST inspection contract produces an
//! independently reviewed prepared artifact.

use std::path::Path;

use crate::ConvertError;

/// Report shape retained for the existing dispatch API. No report is
/// produced while the converter is `INSPECTION_ONLY`.
#[derive(Debug, Default)]
pub struct KimiAudioReport {
    /// Number of tensors read from an authenticated checkpoint.
    pub read: usize,
    /// Number of tensors written to a prepared artifact.
    pub written: usize,
    /// Number of non-floating tensors skipped by a prepared conversion.
    pub skipped_non_float: usize,
    /// Number of BF16 tensors preserved by a prepared conversion.
    pub bf16_passthrough: usize,
}

/// Refuse conversion until the canonical composite release has been
/// authenticated by the dedicated VAST inspector.
pub fn convert_kimi_audio_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<KimiAudioReport, ConvertError> {
    let _ = (input, output, license);
    Err(ConvertError::Usage(
        "Kimi-Audio conversion is INSPECTION_ONLY: the canonical composite release must pass the fixed-layout, safe-load, provenance, and license audit before conversion".to_owned(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn arbitrary_input_and_license_are_fail_closed_without_output() {
        let input = PathBuf::from("/definitely-not-a-kimi-checkpoint");
        for license in [None, Some("mit"), Some("apache-2.0"), Some("")] {
            let output = std::env::temp_dir().join(format!(
                "vokra-kimi-inspection-only-{}-{}.gguf",
                std::process::id(),
                license.unwrap_or("none")
            ));
            let _ = std::fs::remove_file(&output);
            let error = convert_kimi_audio_file(&input, &output, license).unwrap_err();
            assert!(error.to_string().contains("INSPECTION_ONLY"));
            assert!(!output.exists());
        }
    }
}
