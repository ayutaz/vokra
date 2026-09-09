//! Fail-closed inspection boundary for the Chatterbox composite releases.
//!
//! The historical converter accepted arbitrary T3 safetensors and stamped a
//! runtime-looking GGUF. The real pipeline also requires tokenizer, T3
//! generation, VE, S3Gen/meanflow, and conditioning components, so conversion
//! remains `INSPECTION_ONLY` until those components and independent parity are
//! authenticated together.

use vokra_core::gguf::GgufBuilder;

use crate::ConvertError;

/// Chatterbox T3 variant retained for dispatch diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ChatterboxVariant {
    /// Canonical multilingual T3 variant.
    #[default]
    Multilingual,
    /// English-only T3 variant.
    #[allow(dead_code)]
    English,
}

/// Compatibility report retained by the generic converter dispatch API.
#[derive(Debug, Default)]
pub(crate) struct ChatterboxReport {
    /// Always zero because this boundary never writes an artifact.
    pub(crate) written: usize,
    /// Always zero because no input is parsed.
    pub(crate) skipped_non_float: usize,
    /// Requested variant, for callers that log the refusal context.
    pub(crate) variant: ChatterboxVariant,
    /// Diagnostics retained for dispatch compatibility.
    pub(crate) notes: Vec<String>,
}

/// Refuses arbitrary Chatterbox input without producing GGUF or provenance.
pub(crate) fn convert(bytes: Vec<u8>) -> Result<(GgufBuilder, ChatterboxReport), ConvertError> {
    convert_variant(bytes, ChatterboxVariant::Multilingual)
}

/// Explicit variant conversion is also inspection-only.
pub(crate) fn convert_variant(
    _bytes: Vec<u8>,
    variant: ChatterboxVariant,
) -> Result<(GgufBuilder, ChatterboxReport), ConvertError> {
    Err(ConvertError::Usage(format!(
        "Chatterbox {variant:?} conversion is INSPECTION_ONLY: the authenticated T3/VE/S3Gen/conditioning composite and parity are not complete; no GGUF was produced"
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arbitrary_input_is_refused_without_builder() {
        assert!(convert(b"not-a-checkpoint".to_vec()).is_err());
    }
}
