//! NVIDIA Sortformer-Diar-4spk-v1 inspection-only conversion boundary.
//!
//! The public checkpoint currently has no reviewed, immutable NeMo source
//! mapping and no authenticated complete tensor contract in this repository.
//! Consequently this module intentionally performs no file I/O and emits no
//! GGUF. A future VAST wave must establish those facts before conversion is
//! enabled.

use std::path::Path;

use crate::ConvertError;

/// Compatibility report retained for callers while conversion is disabled.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct SortformerDiar4spkV1Report {
    /// Number of input tensors observed by a future authenticated converter.
    pub read: usize,
    /// Number of tensors written by a future authenticated converter.
    pub written: usize,
    /// Number of non-floating tensors skipped by a future authenticated converter.
    pub skipped_non_float: usize,
    /// Number of BF16 tensors preserved by a future authenticated converter.
    pub bf16_passthrough: usize,
}

/// Reject every candidate until VAST evidence pins the immutable HF snapshot,
/// corresponding NeMo source revision, config, complete tensor manifest, and
/// license provenance. This function deliberately does not open `input` or
/// create `output`, so arbitrary or self-authored metadata cannot become an
/// authenticated runtime artifact by reaching a latent conversion path.
pub fn convert_sortformer_diar_4spk_v1_file(
    _input: &Path,
    _output: &Path,
    _license: Option<&str>,
) -> Result<SortformerDiar4spkV1Report, ConvertError> {
    Err(ConvertError::Usage(
        "Sortformer diar 4spk v1 conversion is INSPECTION_ONLY until VAST authenticates the fixed HF checkpoint, corresponding NeMo source, config, license, and complete tensor manifest".to_owned(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arbitrary_checkpoint_conversion_is_inspection_only() {
        let error = convert_sortformer_diar_4spk_v1_file(
            Path::new("/does/not/exist.safetensors"),
            Path::new("/tmp/sortformer.gguf"),
            None,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("INSPECTION_ONLY"), "{error}");
        assert!(error.contains("complete tensor manifest"), "{error}");
    }

    #[test]
    fn permissive_or_empty_license_relabel_cannot_bypass_gate() {
        for license in [Some("apache-2.0"), Some(""), Some("cc-by-nc-4.0")] {
            let error = convert_sortformer_diar_4spk_v1_file(
                Path::new("/does/not/exist.safetensors"),
                Path::new("/tmp/sortformer.gguf"),
                license,
            )
            .unwrap_err()
            .to_string();
            assert!(error.contains("INSPECTION_ONLY"), "{error}");
        }
    }
}
