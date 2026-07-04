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
//!
//! # Weight-license provenance (M2-13, FR-CP-05 conduit)
//!
//! A converter can stamp the produced GGUF with its **weight** license class so
//! the runtime's research-flag gate (FR-CP-03) can enforce it, by calling
//! [`vokra_core::stamp_provenance`] on the [`GgufBuilder`](vokra_core::gguf::GgufBuilder)
//! before serializing — it writes the `vokra.provenance.*` chunk. The class is
//! taken from `docs/license-audit.md` §3 (e.g. Whisper / piper-plus = permissive
//! MIT, a future F5-TTS / EnCodec voice = non-commercial). Only the `vokra.*`
//! metadata namespace is touched — no ONNX/protobuf enters the runtime
//! (NFR-DS-02). Per-model stamping in the existing `convert*` functions is a
//! deliberate follow-up (it shifts each model's metadata-key count); the conduit
//! and its round-trip through the runtime classifier are exercised in this
//! crate's tests.

mod json;
mod models;
mod onnx;
mod quantize;
mod safetensors;

use std::fmt;
use std::path::Path;

pub use quantize::QuantizeError;
use vokra_core::gguf::GgmlType;

/// Which model's conversion routine to run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelKind {
    /// `openai/whisper-base` safetensors checkpoint.
    WhisperBase,
    /// `snakers4/silero-vad` v5 ONNX checkpoint.
    SileroVad,
    /// A piper-plus (MB-iSTFT-VITS2) voice: ONNX graph + `config.json`
    /// (M0-07). Convert with [`convert_piper_plus_file`] — it needs the extra
    /// `config.json` input, so it is not a plain single-input [`convert_file`]
    /// model.
    PiperPlus,
    /// `iic/speech_campplus` (3D-Speaker CAM++) speaker-encoder ONNX checkpoint
    /// (M0-08): 80-d fbank → 192-d speaker embedding for zero-shot voice
    /// conditioning.
    CamPlus,
}

impl ModelKind {
    /// Parses the `--model` argument value.
    pub fn from_arg(s: &str) -> Option<Self> {
        match s {
            "whisper-base" => Some(Self::WhisperBase),
            "silero-vad" => Some(Self::SileroVad),
            "piper-plus" => Some(Self::PiperPlus),
            "campplus" => Some(Self::CamPlus),
            _ => None,
        }
    }

    /// The canonical `--model` argument value for this kind.
    pub fn as_arg(self) -> &'static str {
        match self {
            Self::WhisperBase => "whisper-base",
            Self::SileroVad => "silero-vad",
            Self::PiperPlus => "piper-plus",
            Self::CamPlus => "campplus",
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

impl From<QuantizeError> for ConvertError {
    fn from(e: QuantizeError) -> Self {
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
        ModelKind::WhisperBase => (models::whisper::convert(bytes, None)?, Vec::new()),
        ModelKind::SileroVad => {
            let (builder, report) = models::silero::convert(bytes)?;
            let notes = vec![format!(
                "silero: {} float weights written, {} non-float constants skipped, {} duplicate names de-duped",
                report.written, report.skipped_non_float, report.deduped
            )];
            (builder, notes)
        }
        ModelKind::PiperPlus => {
            return Err(ConvertError::Usage(
                "piper-plus needs a --config config.json; use convert_piper_plus_file".to_owned(),
            ));
        }
        ModelKind::CamPlus => {
            let (builder, report) = models::campplus::convert(&bytes)?;
            let notes = vec![format!(
                "campplus: {} weights written ({} onnx:: names recovered, {} affine-free BN params synthesized, {} unmapped, {} non-float skipped), block_config {:?}",
                report.written,
                report.renamed,
                report.synthesized,
                report.unmapped,
                report.skipped_non_float,
                report.block_config
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

/// Like [`convert_file`], but K-quantizes the model's large weight matrices to
/// `quant` (`Q4_K` / `Q5_K` / `Q6_K`) on the way out (M1-02, FR-QT-01).
///
/// Only `whisper-base` supports quantization in M1-02; other models return a
/// [`ConvertError::Usage`]. Biases, norms and non-block-aligned tensors stay in
/// full precision, and the emitted metadata is identical to the plain path —
/// only the quantized tensors' dtype and bytes differ, so the runtime loads the
/// result through the same GGUF path (dequantizing via `vokra_core::gguf::quant`).
pub fn convert_file_quantized(
    model: ModelKind,
    input: &Path,
    output: &Path,
    quant: GgmlType,
) -> Result<ConvertSummary, ConvertError> {
    let bytes = std::fs::read(input)?;

    let builder = match model {
        ModelKind::WhisperBase => models::whisper::convert(bytes, Some(quant))?,
        other => {
            return Err(ConvertError::Usage(format!(
                "quantization (--quantize) is only supported for whisper-base in M1-02, not {other}"
            )));
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
        notes: vec![format!("quantized weight matrices to {quant:?}")],
    })
}

/// Converts a piper-plus voice (`onnx` graph + `config` JSON) into a GGUF
/// written to `output`, returning a summary (M0-07-T07).
///
/// piper-plus voices are distributed as an FP16 ONNX graph plus a `config.json`
/// (phoneme table, sample rate, inference defaults), so unlike the single-input
/// [`convert_file`] models this one takes both. See
/// [`models::piper_plus`](crate) for the naming / metadata contract.
pub fn convert_piper_plus_file(
    onnx: &Path,
    config: &Path,
    output: &Path,
) -> Result<ConvertSummary, ConvertError> {
    let onnx_bytes = std::fs::read(onnx)?;
    let config_bytes = std::fs::read(config)?;
    let (builder, report) = models::piper_plus::convert(&onnx_bytes, &config_bytes)?;

    let notes = vec![format!(
        "piper-plus: {} float weights written ({} onnx:: names recovered), {} non-float skipped, {} phoneme ids over num_symbols",
        report.written, report.renamed, report.skipped_non_float, report.phoneme_ids_over_range
    )];

    let tensor_count = builder.tensor_count();
    let metadata_count = builder.metadata_count();
    let out_bytes = builder.to_bytes()?;
    std::fs::write(output, &out_bytes)?;

    Ok(ConvertSummary {
        model: ModelKind::PiperPlus,
        tensor_count,
        metadata_count,
        output_bytes: out_bytes.len() as u64,
        notes,
    })
}

#[cfg(test)]
mod compliance_conduit_tests {
    //! The minimal M2-13 conduit (FR-CP-05): a converter stamps a GGUF's weight
    //! license class via [`vokra_core::stamp_provenance`], and the runtime's
    //! research-flag gate reads it back. Exercised at the `GgufBuilder` level —
    //! exactly what the `convert*` routines assemble internally — so no existing
    //! converter output (and its metadata-count assertions) is disturbed.
    use vokra_core::gguf::{GgufBuilder, GgufFile, chunks};
    use vokra_core::{CompliancePolicy, LicenseClass, check_weight_license, resolve_license_class};

    #[test]
    fn converter_stamps_permissive_and_runtime_admits_it() {
        // What a Whisper/piper converter would do (MIT = permissive).
        let mut b = GgufBuilder::new();
        b.add_string(chunks::KEY_MODEL_ARCH, "whisper");
        vokra_core::stamp_provenance(
            &mut b,
            LicenseClass::Permissive,
            "MIT",
            Some("whisper-base"),
            Some("openai/whisper-base"),
        );
        let file = GgufFile::parse(b.to_bytes().expect("serialize")).expect("parse");
        assert_eq!(resolve_license_class(&file).class, LicenseClass::Permissive);
        assert!(check_weight_license(&file, &CompliancePolicy::strict()).is_ok());
    }

    #[test]
    fn converter_stamps_noncommercial_and_runtime_gates_it() {
        // A future F5-TTS / EnCodec converter stamping CC-BY-NC makes the
        // runtime refuse the weight without a research flag.
        let mut b = GgufBuilder::new();
        vokra_core::stamp_provenance(
            &mut b,
            LicenseClass::NonCommercial,
            "CC-BY-NC-4.0",
            Some("encodec"),
            None,
        );
        let file = GgufFile::parse(b.to_bytes().expect("serialize")).expect("parse");
        assert!(check_weight_license(&file, &CompliancePolicy::strict()).is_err());
    }
}
