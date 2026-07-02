//! # vokra-convert
//!
//! Vokra's **offline model conversion tool** (FR-TL-01, M0-03): it reads an
//! upstream checkpoint (safetensors for Whisper, ONNX for Silero VAD) and
//! writes a GGUF carrying the model's tensors plus the `vokra.*` metadata
//! chunks that Vokra's runtime understands.
//!
//! # Why this is a separate crate
//!
//! This is the *only* place ONNX / protobuf handling is allowed to live. The
//! runtime crates never load ONNX and never depend on protobuf/abseil/onnx
//! (FR-LD-05, NFR-DS-02). To keep that boundary airtight, `vokra-convert`
//! depends on nothing but `vokra-core` (for its GGUF writer): the safetensors
//! reader, the JSON parser and the ONNX protobuf decoder are all hand-written
//! here with the standard library only — no external crates — so no ONNX
//! dependency can leak toward the runtime. The dependency direction is
//! strictly one-way (`vokra-convert` -> `vokra-core`).
//!
//! # Scope (M0 minimal tool)
//!
//! Independent binary, F32/F16 tensors only. Integration into a richer
//! `vokra-cli` (FR-TL-02) is a v0.1 MVP / M1 concern.

mod json;
mod models;
mod onnx;
mod safetensors;

use std::fmt;
use std::path::Path;

/// Which model's conversion routine to run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelKind {
    /// `openai/whisper-base` safetensors checkpoint.
    WhisperBase,
    /// `snakers4/silero-vad` v5 ONNX checkpoint.
    SileroVad,
}

impl ModelKind {
    /// Parses the `--model` argument value.
    pub fn from_arg(s: &str) -> Option<Self> {
        match s {
            "whisper-base" => Some(Self::WhisperBase),
            "silero-vad" => Some(Self::SileroVad),
            _ => None,
        }
    }

    /// The canonical `--model` argument value for this kind.
    pub fn as_arg(self) -> &'static str {
        match self {
            Self::WhisperBase => "whisper-base",
            Self::SileroVad => "silero-vad",
        }
    }
}

impl fmt::Display for ModelKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_arg())
    }
}

/// Summary of a successful conversion.
#[derive(Debug)]
pub struct ConvertSummary {
    /// The model that was converted.
    pub model: ModelKind,
    /// Number of tensors written to the GGUF.
    pub tensor_count: usize,
    /// Number of metadata entries written (including `general.alignment` if
    /// injected).
    pub metadata_count: usize,
    /// Size of the output GGUF in bytes.
    pub output_bytes: u64,
    /// Human-readable notes (e.g. skipped non-float initializers).
    pub notes: Vec<String>,
}

/// Errors that can occur while converting a checkpoint.
#[derive(Debug)]
#[non_exhaustive]
pub enum ConvertError {
    /// Reading the input or writing the output failed.
    Io(std::io::Error),
    /// The input checkpoint could not be parsed (safetensors / JSON / ONNX).
    Parse(String),
    /// The GGUF could not be assembled (from `vokra-core`'s writer).
    Gguf(String),
    /// A command-line / usage problem.
    Usage(String),
}

impl fmt::Display for ConvertError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::Parse(m) => write!(f, "parse error: {m}"),
            Self::Gguf(m) => write!(f, "GGUF write error: {m}"),
            Self::Usage(m) => write!(f, "usage error: {m}"),
        }
    }
}

impl std::error::Error for ConvertError {}

impl From<std::io::Error> for ConvertError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<safetensors::SafetensorsError> for ConvertError {
    fn from(e: safetensors::SafetensorsError) -> Self {
        Self::Parse(e.to_string())
    }
}

impl From<onnx::OnnxError> for ConvertError {
    fn from(e: onnx::OnnxError) -> Self {
        Self::Parse(e.to_string())
    }
}

impl From<vokra_core::gguf::GgufError> for ConvertError {
    fn from(e: vokra_core::gguf::GgufError) -> Self {
        Self::Gguf(e.to_string())
    }
}

/// Converts `input` into a GGUF written to `output`, returning a summary.
///
/// This is the single entry point used by both the `vokra-convert` binary and
/// the integration tests.
pub fn convert_file(
    model: ModelKind,
    input: &Path,
    output: &Path,
) -> Result<ConvertSummary, ConvertError> {
    let bytes = std::fs::read(input)?;

    let (builder, notes) = match model {
        ModelKind::WhisperBase => (models::whisper::convert(bytes)?, Vec::new()),
        ModelKind::SileroVad => {
            let (builder, report) = models::silero::convert(bytes)?;
            let notes = vec![format!(
                "silero: {} float weights written, {} non-float constants skipped, {} duplicate names de-duped",
                report.written, report.skipped_non_float, report.deduped
            )];
            (builder, notes)
        }
    };

    let tensor_count = builder.tensor_count();
    let metadata_count = builder.metadata_count();
    let out_bytes = builder.to_bytes()?;
    std::fs::write(output, &out_bytes)?;

    Ok(ConvertSummary {
        model,
        tensor_count,
        metadata_count,
        output_bytes: out_bytes.len() as u64,
        notes,
    })
}
