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
    /// An OpenAI Whisper safetensors checkpoint (M2-06-T06). The specific size
    /// (base / small / medium / large-v3 / turbo) is **auto-detected from the
    /// checkpoint tensor shapes** (see `models::whisper` — `d_model`,
    /// `n_audio_layer`, `n_text_layer`, `n_mels` uniquely identify a size); the
    /// caller passes a single `whisper` label. The CLI keeps `whisper-base` as
    /// a backward-compatible alias for pre-M2-06 invocations, and both dispatch
    /// to the same size-detecting path.
    Whisper,
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
    /// `hexgrad/Kokoro-82M` safetensors checkpoint (M2-07 foundation): a
    /// StyleTTS 2 派生 iSTFTNet TTS model with a per-voice style-vector
    /// voicepack. Weights are bound verbatim; hparams are shape-driven with
    /// `0` placeholders on the iSTFT triple pending T02 upstream inspection.
    Kokoro,
    /// `iic/CosyVoice2-0.5B` safetensors checkpoint (M3-09 scaffold): a
    /// Text tokenizer + LLM backbone + Flow Matching CFM + Mimi codec +
    /// chunk-aware streaming TTS / S2S model (Apache 2.0 code + weight,
    /// docs/license-audit.md). Weights are bound verbatim; numeric hparams
    /// (`n_layer` / `n_head` / `hidden_dim` / `ffn_dim` / streaming chunk
    /// sizes) are `0`-placeholders pending T02 upstream inspection — the
    /// runtime rejects `0` at load per `CosyVoice2Config::from_gguf`.
    CosyVoice2,
    /// Mistral **Voxtral** safetensors checkpoint (M3-10 foundation): a
    /// Whisper-derived audio encoder plus a Mistral (GQA/RoPE/SwiGLU/RMSNorm)
    /// text decoder for ASR and S2S. The tokenizer and optional side-car
    /// hparams (RoPE base, RMSNorm ε, GQA head split, vocab size, S2S codec
    /// type) are supplied through the config-aware
    /// [`convert_voxtral_file`] path — the shape-only [`convert_file`] path
    /// writes `0` sentinels for those fields (which the runtime loader
    /// rejects at forward time per FR-EX-08).
    Voxtral,
}

impl ModelKind {
    /// Parses the `--model` argument value.
    ///
    /// `whisper` is the canonical spelling (size is auto-detected from the
    /// checkpoint shapes); `whisper-base` is kept as a backward-compatible
    /// alias for pre-M2-06 invocations — both dispatch to the same
    /// size-detecting path (M2-06-T06).
    pub fn from_arg(s: &str) -> Option<Self> {
        match s {
            // Canonical M2-06+ spelling: size auto-detected from checkpoint.
            "whisper" => Some(Self::Whisper),
            // Backward-compatible alias for pre-M2-06 invocations.
            "whisper-base" => Some(Self::Whisper),
            "silero-vad" => Some(Self::SileroVad),
            "piper-plus" => Some(Self::PiperPlus),
            "campplus" => Some(Self::CamPlus),
            "kokoro" => Some(Self::Kokoro),
            "cosyvoice2" => Some(Self::CosyVoice2),
            "voxtral" => Some(Self::Voxtral),
            _ => None,
        }
    }

    /// The canonical `--model` argument value for this kind.
    pub fn as_arg(self) -> &'static str {
        match self {
            Self::Whisper => "whisper",
            Self::SileroVad => "silero-vad",
            Self::PiperPlus => "piper-plus",
            Self::CamPlus => "campplus",
            Self::Kokoro => "kokoro",
            Self::CosyVoice2 => "cosyvoice2",
            Self::Voxtral => "voxtral",
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
    /// A [`QuantPolicy`](models::whisper) rule resolved to a K-quant target for
    /// a tensor that cannot be K-quantized (rank < 2 or element count not a
    /// whole number of `QK_K` super-blocks). Emitted instead of silently
    /// widening the tensor's dtype (FR-EX-08, M2-08 T06).
    QuantPolicyInapplicable {
        /// The offending tensor's upstream name.
        tensor: String,
        /// The scheme alias the policy resolved to (e.g. `"w4a16-q4k"`).
        scheme: &'static str,
        /// Human-readable reason (rank, element count, etc.).
        reason: String,
    },
}

impl fmt::Display for ConvertError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {e}"),
            Self::Parse(m) => write!(f, "parse error: {m}"),
            Self::Gguf(m) => write!(f, "GGUF write error: {m}"),
            Self::Usage(m) => write!(f, "usage error: {m}"),
            Self::QuantPolicyInapplicable {
                tensor,
                scheme,
                reason,
            } => write!(
                f,
                "quant policy inapplicable for tensor `{tensor}` (scheme `{scheme}`): {reason}"
            ),
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
        ModelKind::Whisper => (models::whisper::convert(bytes, None)?, Vec::new()),
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
        ModelKind::Kokoro => {
            let (builder, report) = models::kokoro::convert(bytes)?;
            // Backward-compat: the placeholder path (no --config) emits the
            // same 3-field summary M2-07 T06 shipped, plus any diagnostic
            // notes the model routine surfaced. When the caller has a
            // `config.json`, use `convert_kokoro_file` instead — it enriches
            // the summary with the phoneme-symbol count.
            let mut notes = vec![format!(
                "kokoro: {} float weights written, {} non-float skipped, style_dim {}, {} voices",
                report.written,
                report.skipped_non_float,
                report.style_dim,
                report.voices.len(),
            )];
            notes.extend(report.notes.iter().map(|n| format!("kokoro warning: {n}")));
            (builder, notes)
        }
        ModelKind::CosyVoice2 => {
            let (builder, report) = models::cosyvoice2::convert(bytes)?;
            let mut notes = vec![format!(
                "cosyvoice2: {} float weights written, {} non-float skipped \
                 (scaffold — numeric hparams are `0`-placeholders pending T02 \
                 upstream inspection; the runtime rejects the load until T04 \
                 fills them)",
                report.written, report.skipped_non_float,
            )];
            notes.extend(
                report
                    .notes
                    .iter()
                    .map(|n| format!("cosyvoice2 warning: {n}")),
            );
            (builder, notes)
        }
        ModelKind::Voxtral => {
            // Foundation path (M3-10): shape-only conversion writes `0`
            // sentinels for the RoPE / RMSNorm / GQA / vocab side-car values
            // the runtime cannot recover from tensor shapes alone. Real
            // conversions call `convert_voxtral_file` with a `VoxtralConfig`.
            let (builder, report) = models::voxtral::convert(bytes, None)?;
            let notes = vec![format!(
                "voxtral: {} float weights written, {} non-float skipped, name {}, tokenizer embedded: {} (shape-only path — pass a --config for the full hparam chunk)",
                report.written, report.skipped_non_float, report.name, report.tokenizer_embedded
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
/// Only `whisper` (all Whisper sizes) supports quantization in M1-02; other
/// models return a [`ConvertError::Usage`]. Biases, norms and non-block-aligned
/// tensors stay in full precision, and the emitted metadata is identical to
/// the plain path — only the quantized tensors' dtype and bytes differ, so the
/// runtime loads the result through the same GGUF path (dequantizing via
/// `vokra_core::gguf::quant`).
pub fn convert_file_quantized(
    model: ModelKind,
    input: &Path,
    output: &Path,
    quant: GgmlType,
) -> Result<ConvertSummary, ConvertError> {
    let bytes = std::fs::read(input)?;

    let builder = match model {
        ModelKind::Whisper => models::whisper::convert(bytes, Some(quant))?,
        other => {
            return Err(ConvertError::Usage(format!(
                "quantization (--quantize) is only supported for whisper in M1-02, not {other}"
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

/// The named quantization presets accepted by `--policy-preset` (M2-08 T06).
///
/// Presets map to a [`QuantPolicy`](models::whisper) with the shape documented
/// in `docs/design/quantization-policy.md`:
///
/// - [`PolicyPreset::VocoderSafe`] — default whole-model widen to `F16`
///   (activation-safe, matches Vocos/BigVGAN's fp16-minimum registry).
/// - [`PolicyPreset::WhisperQ4K`] — default `Q4_K` with `.bias` / `.weight_norm`
///   pinned to `F32`. Backward-compatible alias for `--quantize q4_k`.
/// - [`PolicyPreset::Fp16`] — whole-model widen to `F16` with no rules.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolicyPreset {
    /// Whole-model widen to `F16`. CLI default when `--policy-preset` is not
    /// passed.
    VocoderSafe,
    /// `Q4_K` default; `.bias` / `.weight_norm` pinned to `F32`.
    WhisperQ4K,
    /// Whole-model widen to `F16`.
    Fp16,
}

impl PolicyPreset {
    /// Parses a `--policy-preset` argument value.
    pub fn from_arg(s: &str) -> Option<Self> {
        match s {
            "vocoder_safe" => Some(Self::VocoderSafe),
            "whisper_q4_k" => Some(Self::WhisperQ4K),
            "fp16" => Some(Self::Fp16),
            _ => None,
        }
    }
}

/// Runs a whisper conversion with an explicit [`PolicyPreset`] (M2-08 T06).
///
/// This is the T06 successor to [`convert_file_quantized`]: the offline
/// converter now resolves each tensor's target dtype through a first-match
/// policy rather than a hardcoded `is_quantizable()` filter, and stamps the
/// resolved policy into `vokra.quant.*` metadata for the runtime to read back.
/// Piper / CAM++ / Silero are unchanged in T06 and reject the flag.
pub fn convert_file_with_policy(
    model: ModelKind,
    input: &Path,
    output: &Path,
    preset: PolicyPreset,
) -> Result<ConvertSummary, ConvertError> {
    let bytes = std::fs::read(input)?;

    let builder = match model {
        ModelKind::Whisper => {
            let policy = match preset {
                PolicyPreset::VocoderSafe => models::whisper::QuantPolicy::default_vocoder_safe(),
                PolicyPreset::WhisperQ4K => models::whisper::QuantPolicy::whisper_q4_k(),
                PolicyPreset::Fp16 => models::whisper::QuantPolicy::fp16(),
            };
            models::whisper::convert_with_policy(bytes, Some(policy))?
        }
        other => {
            return Err(ConvertError::Usage(format!(
                "--policy-preset is only supported for whisper in M2-08, not {other}"
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
        notes: vec![format!("applied quantization policy preset {preset:?}")],
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

/// Converts a Kokoro-82M safetensors checkpoint plus a Kokoro `config.json`
/// (misaki phoneme symbol table + voice-name list) into a GGUF written to
/// `output`, returning a summary (M2-07-T17-fixup #3).
///
/// This is the config-aware sibling of the plain [`convert_file`] path for
/// Kokoro. The safetensors bytes are converted exactly as with
/// `convert_file(ModelKind::Kokoro, …)`; the additional config JSON supplies
/// the real `vokra.kokoro.phoneme_symbols` (misaki phoneme table) and
/// `vokra.kokoro.voice_names` arrays (canonical release ships voices as
/// separate `voices/*.pt` files, so a config is authoritative for the voice
/// list). Callers who do not yet have the misaki phoneme table can still use
/// [`convert_file`] and get the `p0..p_{n_vocab-1}` placeholder for the same
/// legacy round-trip contract.
///
/// The accepted `config.json` schema is documented on
/// [`models::kokoro`](crate) — briefly: at least one of `{vocab: {symbol:id},
/// phoneme_symbols: [str], symbols: [str]}` plus at least one of `{voices:
/// [str], voice_names: [str]}` must be present; first-match wins per family.
pub fn convert_kokoro_file(
    input: &Path,
    config: &Path,
    output: &Path,
) -> Result<ConvertSummary, ConvertError> {
    let bytes = std::fs::read(input)?;
    let config_bytes = std::fs::read(config)?;
    let (builder, report) = models::kokoro::convert_with_config(bytes, Some(&config_bytes))?;

    let mut notes = vec![format!(
        "kokoro: {} float weights written, {} non-float skipped, style_dim {}, \
         {} voices, {} phoneme symbols",
        report.written,
        report.skipped_non_float,
        report.style_dim,
        report.voices.len(),
        report.phoneme_symbol_count,
    )];
    // Surface any per-tensor-vs-config mismatch diagnostics recorded by the
    // model routine. The converter never fails on these — the runtime is the
    // authoritative gate (FR-EX-08) — but the operator gets a loud warning.
    notes.extend(report.notes.iter().map(|n| format!("kokoro warning: {n}")));

    let tensor_count = builder.tensor_count();
    let metadata_count = builder.metadata_count();
    let out_bytes = builder.to_bytes()?;
    std::fs::write(output, &out_bytes)?;

    Ok(ConvertSummary {
        model: ModelKind::Kokoro,
        tensor_count,
        metadata_count,
        output_bytes: out_bytes.len() as u64,
        notes,
    })
}

/// Voxtral (Mistral) side-car hparams supplied by the caller (M3-10-T04). Same
/// shape as the module-private [`models::voxtral::VoxtralConfig`], re-exported
/// here so external callers can build one without pulling in the private
/// module.
pub use models::voxtral::VoxtralConfig;

/// Convert a Voxtral safetensors checkpoint together with a Mistral-format
/// side-car config into a Vokra GGUF (M3-10).
///
/// This is the config-aware sibling of the plain [`convert_file`] path for
/// Voxtral. The safetensors bytes are converted the same way as with
/// `convert_file(ModelKind::Voxtral, …)`; the additional [`VoxtralConfig`]
/// supplies the values shapes cannot recover (RoPE base, RMSNorm ε, GQA head
/// split, vocab size, max sequence length, S2S codec identifier) plus the
/// raw Mistral tokenizer bytes for `vokra.tokenizer.model`.
///
/// The shape-only [`convert_file`] path writes `0` sentinels for the missing
/// side-car values; the runtime loader will still reject a forward attempt
/// that needs them (FR-EX-08).
pub fn convert_voxtral_file(
    input: &Path,
    config: &VoxtralConfig,
    output: &Path,
) -> Result<ConvertSummary, ConvertError> {
    let bytes = std::fs::read(input)?;
    let (builder, report) = models::voxtral::convert(bytes, Some(config))?;

    let notes = vec![format!(
        "voxtral: {} float weights written, {} non-float skipped, name {}, tokenizer embedded: {}",
        report.written, report.skipped_non_float, report.name, report.tokenizer_embedded
    )];

    let tensor_count = builder.tensor_count();
    let metadata_count = builder.metadata_count();
    let out_bytes = builder.to_bytes()?;
    std::fs::write(output, &out_bytes)?;

    Ok(ConvertSummary {
        model: ModelKind::Voxtral,
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
