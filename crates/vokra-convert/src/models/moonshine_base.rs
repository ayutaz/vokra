//! Strict converter for `moonshine-ai/moonshine-base`.

use std::path::Path;

use crate::ConvertError;
use crate::models::moonshine_common::{Report, VariantSpec, convert};

/// Canonical model name stamped into GGUF metadata.
pub const NAME: &str = "moonshine-base";
/// Canonical Hugging Face repository after the UsefulSensors redirect.
pub const UPSTREAM_HF: &str = "moonshine-ai/moonshine-base";
/// Pinned official checkpoint revision.
pub const REVISION: &str = "7a73d8d55ac0ba2ef3ae761593f6784b51f96dcf";
/// SHA-256 of the pinned `model.safetensors`.
pub const CHECKPOINT_SHA256: &str =
    "e020c79d0a979a7ec099f718ff1cd2f19e92aead230d69654bca5975a8e1b862";
/// SHA-256 of the pinned `tokenizer.json`.
pub const TOKENIZER_SHA256: &str =
    "6579793438bc4fbafffacf699169ff53e3769c5a0a0f5e71cdee8853e8130deb";

const SPEC: VariantSpec = VariantSpec {
    name: NAME,
    upstream_hf: UPSTREAM_HF,
    revision: REVISION,
    checkpoint_sha256: CHECKPOINT_SHA256,
    tokenizer_sha256: TOKENIZER_SHA256,
    hidden: 416,
    intermediate: 1_664,
    encoder_layers: 8,
    decoder_layers: 8,
    partial_rotary_factor: 0.62,
};

/// Strict conversion counters and tokenizer presence.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct MoonshineBaseReport {
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

impl From<Report> for MoonshineBaseReport {
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

/// Converts the pinned Base checkpoint and optionally embeds tokenizer JSON.
pub fn convert_moonshine_base_file_with_tokenizer(
    input: &Path,
    tokenizer: Option<&Path>,
    output: &Path,
    license: Option<&str>,
) -> Result<MoonshineBaseReport, ConvertError> {
    convert(input, tokenizer, output, license, SPEC).map(Into::into)
}

/// Backward-compatible weight-only conversion entry point.
pub fn convert_moonshine_base_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<MoonshineBaseReport, ConvertError> {
    convert_moonshine_base_file_with_tokenizer(input, None, output, license)
}
