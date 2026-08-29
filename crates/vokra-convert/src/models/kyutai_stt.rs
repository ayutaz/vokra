//! Fail-closed inspection boundary for the Kyutai STT-2.6B-EN composite.
//!
//! The model, Mimi codec, and tokenizer form one unauthenticated runtime
//! boundary until a real tensor binder and independent parity evidence land.
//! Arbitrary safetensors must never become a runtime-looking GGUF.

use vokra_core::gguf::GgufBuilder;

use crate::ConvertError;

/// Compatibility report retained by the generic converter dispatch API.
#[derive(Debug, Default)]
pub(crate) struct KyutaiSttReport {
    /// Number of float tensors written by a successful conversion.
    pub(crate) written: usize,
    /// Number of non-float tensors skipped by a successful conversion.
    pub(crate) skipped_non_float: usize,
    /// Number of BF16 tensors passed through by a successful conversion.
    pub(crate) bf16_passthrough: usize,
    /// Conversion diagnostics retained for dispatch compatibility.
    pub(crate) notes: Vec<String>,
}

/// Refuse conversion until the fixed six-file release and native binder are
/// independently authenticated.
pub(crate) fn convert(_bytes: Vec<u8>) -> Result<(GgufBuilder, KyutaiSttReport), ConvertError> {
    Err(ConvertError::Usage(
        "Kyutai STT-2.6B-EN conversion is INSPECTION_ONLY: authenticated model/Mimi/tokenizer binder and parity are not implemented; no GGUF was produced".to_owned(),
    ))
}
