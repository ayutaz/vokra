//! Strict converter for `moonshine-ai/moonshine-tiny`.
//!
//! The canonical Transformers checkpoint is a 160-tensor F32 encoder-decoder:
//! raw PCM Conv1D stem, six GELU encoder layers, six SwiGLU decoder layers and
//! a tied 32,768-entry BPE embedding.  The full name/shape/dtype manifest is
//! checked before any output is written.

use std::path::Path;

use crate::ConvertError;
use crate::models::moonshine_common::{Report, VariantSpec, convert};

/// Canonical model name stamped into GGUF metadata.
pub const NAME: &str = "moonshine-tiny";
/// Canonical Hugging Face repository after the UsefulSensors redirect.
pub const UPSTREAM_HF: &str = "moonshine-ai/moonshine-tiny";
/// Pinned official checkpoint revision.
pub const REVISION: &str = "390624ed33d594443aa4aa221f5b9f283b545b5a";
/// SHA-256 of the pinned `model.safetensors`.
pub const CHECKPOINT_SHA256: &str =
    "867cd2215804859c55aa972d740bd5002be149b4e7526328c895d2408848c736";
/// SHA-256 of the pinned `tokenizer.json`.
pub const TOKENIZER_SHA256: &str =
    "6579793438bc4fbafffacf699169ff53e3769c5a0a0f5e71cdee8853e8130deb";

const SPEC: VariantSpec = VariantSpec {
    name: NAME,
    upstream_hf: UPSTREAM_HF,
    revision: REVISION,
    checkpoint_sha256: CHECKPOINT_SHA256,
    tokenizer_sha256: TOKENIZER_SHA256,
    hidden: 288,
    intermediate: 1_152,
    encoder_layers: 6,
    decoder_layers: 6,
    partial_rotary_factor: 0.9,
};

/// Strict conversion counters and tokenizer presence.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct MoonshineTinyReport {
    /// Source tensors inspected.
    pub read: usize,
    /// Tensors written to GGUF.
    pub written: usize,
    /// Always zero: non-F32 input is rejected.
    pub skipped_non_float: usize,
    /// Always zero for the pinned F32 checkpoint.
    pub bf16_passthrough: usize,
    /// Whether the decode-only tokenizer was embedded.
    pub tokenizer_embedded: bool,
}

impl From<Report> for MoonshineTinyReport {
    fn from(value: Report) -> Self {
        Self {
            read: value.read,
            written: value.written,
            skipped_non_float: value.skipped_non_float,
            bf16_passthrough: value.bf16_passthrough,
            tokenizer_embedded: value.tokenizer_embedded,
        }
    }
}

/// Converts the pinned official checkpoint.  `tokenizer` should point to the
/// matching `tokenizer.json`; omitting it preserves the library API but creates
/// a weight-only GGUF which the end-to-end runtime rejects loudly.
pub fn convert_moonshine_tiny_file_with_tokenizer(
    input: &Path,
    tokenizer: Option<&Path>,
    output: &Path,
    license: Option<&str>,
) -> Result<MoonshineTinyReport, ConvertError> {
    convert(input, tokenizer, output, license, SPEC).map(Into::into)
}

/// Backward-compatible weight-only conversion entry point.
pub fn convert_moonshine_tiny_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<MoonshineTinyReport, ConvertError> {
    convert_moonshine_tiny_file_with_tokenizer(input, None, output, license)
}
