//! Fail-closed conversion boundary for Zyphra Zonos-v0.1-transformer.
//!
//! The historical implementation accepted arbitrary safetensors and stamped
//! Apache provenance. That was not a complete conversion contract: the
//! public 246-tensor artifact and its companion DAC/conditioning assets have
//! not been authenticated as a complete Zonos release in this repository.
//! Until a VAST evidence packet fixes the upstream revision, complete tensor
//! manifest, DAC identity, and conditioning composition, this module emits no
//! GGUF and accepts no license/provenance override.

use vokra_core::gguf::GgufBuilder;

use crate::ConvertError;

/// Runtime architecture label retained for shared dispatch.
#[allow(dead_code)] // Retained as inspection-only dispatch metadata until binding is authenticated.
pub(crate) const ARCH: &str = "zonos";
/// Product model name retained for shared dispatch.
#[allow(dead_code)] // Retained as inspection-only model metadata until binding is authenticated.
pub(crate) const NAME: &str = "zonos-v0.1";

/// Compatibility report retained by the dispatcher's formatting path.
#[derive(Debug, Default)]
pub(crate) struct ZonosReport {
    /// Always zero while Zonos conversion is inspection-only.
    pub(crate) written: usize,
    /// Always zero while Zonos conversion is inspection-only.
    pub(crate) skipped_non_float: usize,
    /// Reserved for a future authenticated conversion.
    pub(crate) notes: Vec<String>,
}

/// Refuses arbitrary Zonos input before parsing or writing an output.
///
/// A successful return would make the shared converter create a
/// runtime-looking GGUF. It is intentionally impossible until VAST proves
/// the pinned artifact contains the complete transformer contract and the
/// separately distributed 44.1-kHz, nine-codebook DAC/conditioning
/// composition.
pub(crate) fn convert(_bytes: Vec<u8>) -> Result<(GgufBuilder, ZonosReport), ConvertError> {
    Err(ConvertError::Usage(
        "Zonos-v0.1 conversion is INSPECTION_ONLY: VAST must authenticate the fixed upstream revision, complete 246-tensor manifest, official transformer topology, 44.1-kHz nine-codebook DAC composition, and authenticated conditioning packet before a GGUF may be emitted".to_owned(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arbitrary_input_is_rejected_without_output() {
        let output = std::env::temp_dir().join("zonos-inspection-only.gguf");
        let _ = std::fs::remove_file(&output);
        let error = convert(vec![1, 2, 3]).unwrap_err().to_string();
        assert!(error.contains("INSPECTION_ONLY"), "{error}");
        assert!(!output.exists());
    }

    #[test]
    fn constants_remain_dispatch_compatible() {
        assert_eq!(ARCH, "zonos");
        assert_eq!(NAME, "zonos-v0.1");
    }
}
