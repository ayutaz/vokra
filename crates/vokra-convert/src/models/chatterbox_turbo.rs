//! Fail-closed inspection boundary for Chatterbox-Turbo.
//!
//! Turbo requires its GPT-2 T3 checkpoint plus tokenizer, VE, S3Gen meanflow,
//! and conditioning assets. An arbitrary single safetensors file cannot be
//! represented as a valid runtime artifact, so this converter emits none.

use vokra_core::gguf::GgufBuilder;

use crate::ConvertError;

/// Compatibility report retained by generic converter dispatch.
#[derive(Debug, Default)]
pub(crate) struct ChatterboxTurboReport {
    /// Always zero because conversion is refused.
    pub(crate) written: usize,
    /// Always zero because conversion is refused.
    pub(crate) skipped_non_float: usize,
    /// Diagnostics retained for dispatch compatibility.
    pub(crate) notes: Vec<String>,
}

/// Refuses arbitrary Turbo input without producing GGUF or provenance.
pub(crate) fn convert(
    _bytes: Vec<u8>,
) -> Result<(GgufBuilder, ChatterboxTurboReport), ConvertError> {
    Err(ConvertError::Usage(
        "Chatterbox-Turbo conversion is INSPECTION_ONLY: authenticated T3/VE/S3Gen meanflow/conditioning composite and parity are not complete; no GGUF was produced".to_owned(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arbitrary_input_is_refused_without_builder() {
        assert!(convert(b"not-a-checkpoint".to_vec()).is_err());
    }
}
