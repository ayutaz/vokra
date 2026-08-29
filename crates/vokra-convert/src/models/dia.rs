//! Fail-closed conversion boundary for nari-labs **Dia-1.6B**.
//!
//! The official release is a composite PyTorch/safetensors text-to-dialog
//! system whose delayed autoregressive decoder still requires the separate
//! `vokra/dac-44khz` codec. A single arbitrary safetensors file is therefore
//! not a product checkpoint. Until the fixed HF identity, complete 343 tensor
//! manifest, and PTH↔safetensors mapping are authenticated by VAST, this
//! module emits no GGUF and accepts no license relabel.

use vokra_core::gguf::GgufBuilder;

use crate::ConvertError;

/// Runtime architecture label retained for the existing dispatch contract.
pub(crate) const ARCH: &str = "dia";
/// Product model name retained for the existing dispatch contract.
pub(crate) const NAME: &str = "dia-1.6b";

/// Compatibility report retained by the shared dispatcher's formatting path.
#[derive(Debug, Default)]
pub(crate) struct DiaReport {
    /// Always zero while conversion is inspection-only.
    pub(crate) written: usize,
    /// Always zero while conversion is inspection-only.
    pub(crate) skipped_non_float: usize,
    /// Always zero while conversion is inspection-only.
    pub(crate) bf16_passthrough: usize,
    /// Reserved for a future authenticated converter.
    pub(crate) notes: Vec<String>,
}

/// Refuses arbitrary Dia inputs before opening or writing either path.
///
/// The API remains compatible with the shared converter, but no provenance,
/// axes, license, or output bytes are fabricated here.
pub(crate) fn convert(_bytes: Vec<u8>) -> Result<(GgufBuilder, DiaReport), ConvertError> {
    Err(ConvertError::Usage(
        "Dia-1.6B conversion is INSPECTION_ONLY: VAST must authenticate the fixed six-file HF tree, the 343-tensor safetensors contract, the safe PTH inventory and PTH↔safetensors mapping, and the separate DAC composition before a runtime GGUF can be emitted".to_owned(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arbitrary_input_is_rejected_without_output() {
        let output = std::env::temp_dir().join("dia-inspection-only.gguf");
        let _ = std::fs::remove_file(&output);
        let error = convert(vec![1, 2, 3]).unwrap_err().to_string();
        assert!(error.contains("INSPECTION_ONLY"), "{error}");
        assert!(!output.exists());
    }

    #[test]
    fn constants_remain_dispatch_compatible() {
        assert_eq!(ARCH, "dia");
        assert_eq!(NAME, "dia-1.6b");
    }
}
