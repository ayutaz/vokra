//! CSM-1B composite inspection-only conversion boundary.
//!
//! A valid release needs the CSM model checkpoint, Mimi codec, gated Llama
//! tokenizer, config, and provenance as one authenticated composite. The
//! historical single-file conversion path stamped incomplete metadata and is
//! therefore disabled until VAST inspection and real parity are complete.

use vokra_core::gguf::GgufBuilder;

use crate::ConvertError;

/// Legacy report shape retained for the internal conversion API. No production
/// conversion returns it while CSM remains inspection-only.
#[derive(Debug, Default)]
pub(crate) struct CsmReport {
    pub(crate) written: usize,
    pub(crate) skipped_non_float: usize,
    pub(crate) tokenizer_embedded: bool,
    pub(crate) notes: Vec<String>,
}

/// Refuses the legacy single-safetensors path before parsing or constructing a
/// GGUF. This prevents incomplete CSM-core artifacts and license/provenance
/// relabeling from looking runtime-ready.
pub(crate) fn convert(
    _bytes: Vec<u8>,
    _tokenizer_bytes: Option<Vec<u8>>,
) -> Result<(GgufBuilder, CsmReport), ConvertError> {
    Err(ConvertError::Usage(
        "csm: INSPECTION_ONLY — authenticated exact CSM model + Mimi codec + tokenizer + config/provenance composite is pending VAST inspection/parity; no GGUF is produced".to_owned(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conversion_refuses_arbitrary_single_checkpoint() {
        let error = convert(vec![0; 16], None).expect_err("CSM must fail closed");
        assert!(error.to_string().contains("INSPECTION_ONLY"));
        assert!(error.to_string().contains("Mimi"));
        assert!(error.to_string().contains("tokenizer"));
    }

    #[test]
    fn tokenizer_argument_cannot_bypass_composite_gate() {
        let error = convert(vec![1, 2, 3], Some(vec![4, 5])).expect_err("must refuse");
        assert!(error.to_string().contains("no GGUF is produced"));
    }
}
