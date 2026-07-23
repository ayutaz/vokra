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

pub use quantize::{QuantizeError, quantize};
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
    /// SaruLab **UTMOS22-strong** neural MOS predictor (M5-15 T14): a
    /// wav2vec2-base SSL encoder + listener/domain conditioning + BLSTM +
    /// regression head, used by `vokra-eval` for the NFR-QL-02 5 % quality
    /// gate. Convert with [`convert_utmos_file`] — it needs the config
    /// side-car that `tools/parity/utmos_prepare_checkpoint.py` emits
    /// alongside the flattened safetensors, so it is not a plain
    /// single-input [`convert_file`] model.
    Utmos,
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
    /// Standalone Mimi (Kyutai) codec checkpoint (M4-04 T10): the moshi-native
    /// safetensors (`kyutai/moshiko-pytorch-bf16`
    /// `tokenizer-e351c8d8-checkpoint125.safetensors`, CC-BY 4.0 weights —
    /// attribution discharged by NOTICE §5). All tensors pass through; the
    /// converter additionally derives the effective (pre-projected) RVQ
    /// codebook tables the runtime decode consumes, and emits
    /// `vokra.mimi.{n_codebooks,codebook_size,d_model}` from the checkpoint
    /// shapes (ADR M4-04 §D-f/§D-k).
    Mimi,
    /// Standalone DAC (Descript Audio Codec) checkpoint (M4-04 T11): a
    /// **prepared** safetensors (from `tools/parity/dac_prepare_checkpoint.py`
    /// — the upstream release is a `.pth`) plus a JSON config side-car.
    /// Convert with [`convert_dac_file`] — the config is required, so this is
    /// not a plain single-input [`convert_file`] model. MIT weights.
    Dac,
    /// `sesame/csm-1b` safetensors checkpoint (M4-05): Sesame CSM-1B, the
    /// S2S speech-generation model (Llama-3.2-1B-flavor backbone +
    /// llama-100M-flavor depth transformer over Mimi RVQ frames; Apache 2.0
    /// code + weight, docs/license-audit.md — the HF repo is gated, T29
    /// owner hand-off). Weights are bound verbatim; flavor dims / RoPE
    /// scaling / rates are transcribed primary-source constants and the two
    /// vocab axes are `0`-placeholders the runtime rejects at load
    /// (FR-EX-08). The Llama-3.2 tokenizer blob is embedded through
    /// [`convert_csm_file`].
    Csm,
    /// `kyutai/moshiko-pytorch-bf16` safetensors checkpoint (M4-06):
    /// Moshi (Helium temporal transformer + depformer), full-duplex S2S
    /// with inner monologue. Weights are CC-BY 4.0 (`AttributionRequired`
    /// — the converter stamps the FR-MD-09 attribution text). The raw
    /// SentencePiece tokenizer embeds through [`convert_moshi_file`].
    Moshi,
    /// Rikorose/DeepFilterNet **DeepFilterNet3** denoiser checkpoint (M4-20
    /// T17): a **prepared** safetensors (from
    /// `tools/parity/dfn3_prepare_checkpoint.py` — the upstream release is a
    /// torch-pickle `.ckpt.best` inside `models/DeepFilterNet3.zip`). Every
    /// inference tensor binds verbatim under its upstream name; the
    /// published DFN3 hyper-parameters ride the `vokra.denoise.*` chunk.
    /// Dual MIT / Apache-2.0 code + weights (docs/license-audit.md).
    Denoise,
    /// nari-labs **Dia-1.6B** safetensors checkpoint (SoTA plan Phase 1-4,
    /// 2026-07-24). Text encoder (12L / 1024d / 16h × 128 head_dim / 4096
    /// FFN) + delayed-AR decoder (18L / 2048d GQA 16Q ÷ 4KV × 128 head_dim /
    /// cross-attn 16Q × 128 / 8192 FFN) over 9 DAC 44.1 kHz codebook
    /// channels with `delay_pattern=[0,8..15]`. Apache 2.0 code + weight.
    /// All hparams transcribed verbatim from `huggingface.co/nari-labs/
    /// Dia-1.6B/config.json`; every F32 / F16 tensor passes through
    /// verbatim. The upstream release ships torch `.pth`, so callers pre-
    /// flatten it to safetensors offline (the CSM / DAC pattern).
    Dia,
    /// Zyphra **Zonos-v0.1-transformer** safetensors checkpoint (SoTA plan
    /// Phase 1-5, 2026-07-24). Single-stack GQA transformer (26L / 2048d /
    /// 16Q ÷ 4KV × 128 head_dim / SwiGLU 8192 inner) with a typed prefix
    /// conditioner (espeak / speaker / Fourier / integer) over 9 DAC 44.1
    /// kHz codebook channels with `delay_pattern=[1..9]`. Apache 2.0 code
    /// plus weight. All hparams (including the 7 conditioner descriptors)
    /// transcribed verbatim from `huggingface.co/Zyphra/Zonos-v0.1-transformer/config.json`;
    /// every F32 / F16 tensor passes through verbatim. Ships safetensors
    /// directly — no `.pth` prepare step (unlike Dia).
    Zonos,
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
            "utmos" => Some(Self::Utmos),
            "piper-plus" => Some(Self::PiperPlus),
            "campplus" => Some(Self::CamPlus),
            "kokoro" => Some(Self::Kokoro),
            "cosyvoice2" => Some(Self::CosyVoice2),
            "voxtral" => Some(Self::Voxtral),
            "mimi" => Some(Self::Mimi),
            "dac" => Some(Self::Dac),
            "csm" => Some(Self::Csm),
            "moshi" => Some(Self::Moshi),
            "denoise" => Some(Self::Denoise),
            "dia" | "dia-1.6b" | "dia-1_6b" => Some(Self::Dia),
            "zonos" | "zonos-v0.1" | "zonos-v0_1" | "zonos-v0.1-transformer" => Some(Self::Zonos),
            _ => None,
        }
    }

    /// The canonical `--model` argument value for this kind.
    pub fn as_arg(self) -> &'static str {
        match self {
            Self::Whisper => "whisper",
            Self::SileroVad => "silero-vad",
            Self::Utmos => "utmos",
            Self::PiperPlus => "piper-plus",
            Self::CamPlus => "campplus",
            Self::Kokoro => "kokoro",
            Self::CosyVoice2 => "cosyvoice2",
            Self::Voxtral => "voxtral",
            Self::Mimi => "mimi",
            Self::Dac => "dac",
            Self::Csm => "csm",
            Self::Moshi => "moshi",
            Self::Denoise => "denoise",
            Self::Dia => "dia",
            Self::Zonos => "zonos",
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
    convert_file_licensed(model, input, output, None)
}

/// [`convert_file`] with an explicit weight-licence override.
///
/// Each converter stamps the licence it knows for its model. That is right when
/// the model has one canonical licence, but wrong when the *actual distribution
/// source* declares a different one — e.g. OpenAI's Whisper is MIT on GitHub,
/// yet the Hugging Face weight repos this checkpoint may have come from tag
/// `base`/`small`/`medium` as `apache-2.0`. Publishing must state the licence
/// of the artifact being redistributed, so when the two disagree the caller
/// passes the source's SPDX id here and it overrides the stamped
/// `vokra.provenance.{weight_license,license}` — keeping the GGUF the single
/// source of truth the model card is generated from (no card/artifact drift).
///
/// `license` is the raw SPDX string (e.g. `"apache-2.0"`); the class is
/// re-derived from it. `None` keeps the converter's built-in stamp.
///
/// # Errors
///
/// As [`convert_file`].
pub fn convert_file_licensed(
    model: ModelKind,
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<ConvertSummary, ConvertError> {
    // Moshi streams tensor-by-tensor (the 14 GiB full-7B checkpoint must
    // never be materialized whole — bounded-memory contract); it routes
    // through `convert_moshi_file` BEFORE the whole-file read below.
    if matches!(model, ModelKind::Moshi) {
        return convert_moshi_file(input, None, output);
    }
    let bytes = std::fs::read(input)?;

    let (mut builder, notes) = match model {
        ModelKind::Whisper => (models::whisper::convert(bytes, None)?, Vec::new()),
        ModelKind::SileroVad => {
            let (builder, report) = models::silero::convert(bytes)?;
            let notes = vec![format!(
                "silero: {} float weights written (both rates, sr8k.*/sr16k.*), {} non-float constants skipped, {} op-scope float strays skipped",
                report.written, report.skipped_non_float, report.skipped_stray
            )];
            (builder, notes)
        }
        ModelKind::PiperPlus => {
            return Err(ConvertError::Usage(
                "piper-plus needs a --config config.json; use convert_piper_plus_file".to_owned(),
            ));
        }
        ModelKind::Utmos => {
            return Err(ConvertError::Usage(
                "utmos needs a --config config.json (emitted by                  tools/parity/utmos_prepare_checkpoint.py alongside the flattened safetensors);                  use convert_utmos_file"
                    .to_owned(),
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
            // Shape-derived hparams only; the attention head split needs
            // the upstream config.json — use `convert_cosyvoice2_file`
            // with a `--config` for the full hparam chunk.
            let (builder, report) = models::cosyvoice2::convert(bytes)?;
            (builder, cosyvoice2_notes(&report))
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
        ModelKind::Mimi => {
            let (builder, report) = models::mimi::convert(bytes)?;
            let notes = vec![format!(
                "mimi: {} tensors passed through ({} non-float skipped), derived effective \
                 codebook tables [{} x {} x {}] emitted as `{}`, neural-chain adapter wrote \
                 {} structural mimi.enc.*/mimi.dec.* tensors + the vokra.mimi.* config chunk \
                 group ({})",
                report.written,
                report.skipped_non_float,
                report.n_codebooks,
                report.codebook_size,
                report.d_model,
                models::mimi::DERIVED_TABLES_TENSOR,
                report.structural_written,
                if report.structural_written > 0 {
                    "PCM encode/decode bindable"
                } else {
                    "checkpoint carries no SEANet chain — quantizer-only"
                },
            )];
            (builder, notes)
        }
        ModelKind::Dac => {
            return Err(ConvertError::Usage(
                "dac needs a --config side-car (from tools/parity/dac_prepare_checkpoint.py); \
                 use convert_dac_file"
                    .to_owned(),
            ));
        }
        ModelKind::Moshi => {
            // Handled by the streaming early-return above (bounded memory);
            // reaching this arm would mean the whole checkpoint was read.
            unreachable!("ModelKind::Moshi routes through convert_moshi_file")
        }
        ModelKind::Csm => {
            // Tokenizer-less path (M4-05-T03/T04): every float tensor
            // verbatim + the vokra.csm.* / vokra.mimi.* chunk groups. The
            // Llama-3.2 tokenizer blob (gated repo — T29) travels through
            // `convert_csm_file`.
            let (builder, report) = models::csm::convert(bytes, None)?;
            let mut notes = vec![format!(
                "csm: {} float weights written, {} non-float skipped, tokenizer \
                 embedded: {} (vocab axes are `0`-placeholders pending the T29 \
                 checkpoint; the runtime rejects the load until then)",
                report.written, report.skipped_non_float, report.tokenizer_embedded
            )];
            notes.extend(report.notes.iter().map(|n| format!("csm warning: {n}")));
            (builder, notes)
        }
        ModelKind::Denoise => {
            // M4-20 T17: prepared DFN3 safetensors → verbatim upstream-named
            // tensors + the `vokra.denoise.*` chunk. The routine hard-errors
            // on any missing / mis-shaped / unknown tensor and re-binds its
            // own output through the runtime loader before returning.
            let (builder, written) = models::denoise::convert_builder(bytes)
                .map_err(|e| ConvertError::Parse(e.to_string()))?;
            let notes = vec![format!(
                "denoise: {written} DeepFilterNet3 tensors written verbatim (dead \
                 checkpoint tensors skipped by policy: erb_fb, df_dec.df_fc_a.*), \
                 loadability re-checked via DenoiseModel::from_gguf"
            )];
            (builder, notes)
        }
        ModelKind::Dia => {
            // SoTA plan Phase 1-4: pass every F32/F16 tensor through verbatim
            // and stamp the `vokra.dia.*` chunk group from the primary-source
            // constants transcribed in `models::dia`.
            let (builder, report) = models::dia::convert(bytes)?;
            let mut notes = vec![format!(
                "dia: {} float weights written verbatim, {} non-float skipped",
                report.written, report.skipped_non_float,
            )];
            notes.extend(report.notes.iter().map(|n| format!("dia warning: {n}")));
            (builder, notes)
        }
        ModelKind::Zonos => {
            // SoTA plan Phase 1-5: pass every F32/F16 tensor through verbatim
            // and stamp the `vokra.zonos.*` chunk group (backbone hparams +
            // vocab + delay pattern + 7 typed prefix-conditioner descriptors)
            // from the primary-source constants transcribed in `models::zonos`.
            let (builder, report) = models::zonos::convert(bytes)?;
            let mut notes = vec![format!(
                "zonos: {} float weights written verbatim, {} non-float skipped",
                report.written, report.skipped_non_float,
            )];
            notes.extend(report.notes.iter().map(|n| format!("zonos warning: {n}")));
            (builder, notes)
        }
    };

    // Override the stamped licence when the caller supplies the distribution
    // source's SPDX id (add_string overwrites the key in place, so the model's
    // model_id / source / attribution stamps are preserved — only the licence
    // and its class change).
    if let Some(lic) = license {
        let class = vokra_core::LicenseClass::from_license_str(lic);
        builder.add_string(
            vokra_core::gguf::chunks::KEY_PROVENANCE_WEIGHT_LICENSE,
            class.as_str(),
        );
        builder.add_string(vokra_core::gguf::chunks::KEY_PROVENANCE_LICENSE, lic);
        // The built-in `source` string names the converter's default licence
        // (e.g. "openai/whisper (MIT)"); once the licence is overridden that
        // parenthetical would contradict it, so restate the source neutrally.
        builder.add_string(
            vokra_core::gguf::chunks::KEY_PROVENANCE_SOURCE,
            &format!("upstream distribution source (licence {lic} per source)"),
        );
    }

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
        // Voxtral has a quantization path (M5-15-T36), but it needs the
        // side-car config this signature cannot carry: without it the GGUF
        // gets `0` sentinels for RoPE base / RMSNorm eps / GQA split and the
        // runtime refuses the forward (FR-EX-08). Point the caller at the
        // config-aware entry rather than emitting an unloadable file.
        ModelKind::Voxtral => {
            return Err(ConvertError::Usage(
                "voxtral quantization needs the side-car config: use \
                 `vokra-cli convert --model voxtral --config <config.json> --quantize <kind>` \
                 (or `convert_voxtral_file_quantized`). Quantizing without it would emit a GGUF \
                 with `0` hparam sentinels that the runtime refuses to run."
                    .to_owned(),
            ));
        }
        other => {
            return Err(ConvertError::Usage(format!(
                "quantization (--quantize) is only supported for whisper and voxtral, not {other}"
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

/// Convert a **prepared** DAC safetensors checkpoint together with its JSON
/// config side-car into a Vokra GGUF (M4-04 T11).
///
/// The upstream DAC release is a torch-pickle `.pth`; run
/// `tools/parity/dac_prepare_checkpoint.py` first to flatten it into a
/// safetensors + config-JSON pair (no `.pth` parser enters the converter —
/// zero-dep, NFR-DS-02). The config supplies the shape facts the checkpoint
/// metadata carried (`n_codebooks` / `codebook_size` / `codebook_dim` /
/// `d_model` / `sample_rate` / `hop_length`); the converter cross-checks them
/// against the tensor shapes and fails explicitly on any mismatch (FR-EX-08).
///
/// All upstream tensors pass through; per-quantizer decode-ready tensors
/// (`vokra.dac.quantizer.{i}.{codebook,out_proj_weight,out_proj_bias}`, with
/// the weight norm folded offline) are emitted next to them — see
/// `models/dac.rs` module docs / ADR M4-04 §D-f.
pub fn convert_dac_file(
    input: &Path,
    config: &Path,
    output: &Path,
) -> Result<ConvertSummary, ConvertError> {
    let bytes = std::fs::read(input)?;
    let config_bytes = std::fs::read(config)?;
    let cfg = models::dac::DacConfig::parse(&config_bytes)?;
    let (builder, report) = models::dac::convert(bytes, &cfg)?;

    let notes = vec![format!(
        "dac: {} tensors passed through ({} non-float skipped), {} quantizers folded \
         (weight-norm) into vokra.dac.quantizer.* decode tensors, sample_rate {}, hop {}",
        report.written, report.skipped_non_float, cfg.n_codebooks, cfg.sample_rate, cfg.hop_length,
    )];

    let tensor_count = builder.tensor_count();
    let metadata_count = builder.metadata_count();
    let out_bytes = builder.to_bytes()?;
    std::fs::write(output, &out_bytes)?;

    Ok(ConvertSummary {
        model: ModelKind::Dac,
        tensor_count,
        metadata_count,
        output_bytes: out_bytes.len() as u64,
        notes,
    })
}

/// Convert a prepared SaruLab UTMOS22-strong checkpoint into a Vokra GGUF
/// (M5-15 T14).
///
/// `input` is the flat safetensors and `config` the JSON side-car that
/// `tools/parity/utmos_prepare_checkpoint.py` writes from the upstream
/// `.ckpt` (the Lightning checkpoint is a torch pickle, which the zero-dep
/// Rust converter deliberately does not parse — the same offline-prepare
/// split as DAC and Kokoro).
///
/// The mapping is total: every upstream tensor must be consumed, and any
/// left over is a hard error rather than a silent drop (FR-EX-08).
pub fn convert_utmos_file(
    input: &Path,
    config: &Path,
    output: &Path,
) -> Result<ConvertSummary, ConvertError> {
    let bytes = std::fs::read(input)?;
    let config_bytes = std::fs::read(config)?;
    let cfg = models::utmos::UtmosConvertConfig::parse(&config_bytes)?;
    let (builder, report) = models::utmos::convert(bytes, &cfg)?;

    let notes = vec![format!(
        "utmos: {} tensor(s) emitted from {} upstream tensor(s) (all consumed), variant \
         wav2vec2_regression.v1, {} transformer layer(s) d={}, pos_conv k={} groups={} \
         (weight-norm folded), BLSTM hidden {}, judge_id {} / domain_id {}",
        report.written,
        report.consumed,
        cfg.n_layer,
        cfg.hidden_dim,
        cfg.pos_conv_kernel,
        cfg.pos_conv_groups,
        cfg.blstm_hidden,
        cfg.judge_id,
        cfg.domain_id,
    )];

    let tensor_count = builder.tensor_count();
    let metadata_count = builder.metadata_count();
    let out_bytes = builder.to_bytes()?;
    std::fs::write(output, &out_bytes)?;

    Ok(ConvertSummary {
        model: ModelKind::Utmos,
        tensor_count,
        metadata_count,
        output_bytes: out_bytes.len() as u64,
        notes,
    })
}

/// Convert a Sesame CSM-1B safetensors checkpoint into a Vokra GGUF,
/// optionally embedding the raw `meta-llama/Llama-3.2-1B` tokenizer file
/// as `vokra.tokenizer.model` (M4-05-T03/T04/T05).
///
/// The tokenizer repo is gated (T29 owner hand-off); passing
/// `tokenizer = None` converts without the blob and the runtime text path
/// fails loudly until a tokenizer-carrying GGUF exists (FR-EX-08 — never a
/// silent byte-level fallback).
pub fn convert_csm_file(
    input: &Path,
    tokenizer: Option<&Path>,
    output: &Path,
) -> Result<ConvertSummary, ConvertError> {
    let bytes = std::fs::read(input)?;
    let tokenizer_bytes = match tokenizer {
        Some(p) => Some(std::fs::read(p)?),
        None => None,
    };
    let (builder, report) = models::csm::convert(bytes, tokenizer_bytes)?;

    let mut notes = vec![format!(
        "csm: {} float weights written, {} non-float skipped, tokenizer embedded: {}",
        report.written, report.skipped_non_float, report.tokenizer_embedded
    )];
    notes.extend(report.notes.iter().map(|n| format!("csm warning: {n}")));

    let tensor_count = builder.tensor_count();
    let metadata_count = builder.metadata_count();
    let out_bytes = builder.to_bytes()?;
    std::fs::write(output, &out_bytes)?;

    Ok(ConvertSummary {
        model: ModelKind::Csm,
        tensor_count,
        metadata_count,
        output_bytes: out_bytes.len() as u64,
        notes,
    })
}

/// Formats the operator-facing notes for a CosyVoice2 conversion (shared
/// by [`convert_file`] and [`convert_cosyvoice2_file`]).
fn cosyvoice2_notes(report: &models::cosyvoice2::CosyVoice2Report) -> Vec<String> {
    let mut notes = vec![match report.derived {
        Some(d) => format!(
            "cosyvoice2: {} float weights written, {} non-float skipped; derived \
             hparams: vocab={} hidden={} n_layer={} ffn={} n_head={} n_head_kv={} \
             n_ctx={} attn_bias={}",
            report.written,
            report.skipped_non_float,
            d.vocab_size,
            d.hidden_dim,
            d.n_layer,
            d.ffn_dim,
            d.n_head,
            d.n_head_kv,
            d.n_ctx,
            d.has_attn_bias,
        ),
        None => format!(
            "cosyvoice2: {} float weights written, {} non-float skipped (no LLM \
             backbone tensors — numeric hparams are 0-placeholders and the runtime \
             rejects the LLM bind at load)",
            report.written, report.skipped_non_float,
        ),
    }];
    notes.push(format!(
        "cosyvoice2: text tokenizer embedded: {}",
        report.tokenizer_embedded
    ));
    notes.extend(
        report
            .notes
            .iter()
            .map(|n| format!("cosyvoice2 warning: {n}")),
    );
    notes
}

/// Converts a CosyVoice2 LLM safetensors checkpoint (the upstream
/// `FunAudioLLM/CosyVoice2-0.5B` `llm.pt` exported with verbatim names)
/// into a Vokra GGUF, optionally consuming the upstream HF `config.json`
/// (Qwen2 schema) via `config`.
///
/// The config supplies the attention head split
/// (`num_attention_heads` / `num_key_value_heads`) plus `rope_theta` /
/// `rms_norm_eps` / `max_position_embeddings` — none of which are
/// derivable from tensor shapes (`q_out == hidden` leaves `head_dim`
/// free). Without it the GGUF carries the shape-derived hparams only and
/// the runtime refuses the LLM bind (loud, FR-EX-08). Config values are
/// cross-checked against the tensor shapes and any disagreement fails the
/// conversion.
pub fn convert_cosyvoice2_file(
    input: &Path,
    config: Option<&Path>,
    output: &Path,
) -> Result<ConvertSummary, ConvertError> {
    let bytes = std::fs::read(input)?;
    let config_bytes = match config {
        Some(p) => Some(std::fs::read(p)?),
        None => None,
    };
    // Qwen2 text-tokenizer side-car (T06): the upstream `vocab.json` +
    // `merges.txt` live in the same directory as `config.json`
    // (`CosyVoice-BlankEN/`). When a `--config` is given, pick them up from
    // that directory and embed both (no second CLI flag needed). A partial or
    // absent pair is a loud note in the report, not a hard error — the
    // conversion still succeeds; the runtime text path fails loudly instead.
    let tokenizer_bytes: Option<(Vec<u8>, Vec<u8>)> = config.and_then(|p| {
        let dir = p.parent().unwrap_or_else(|| Path::new("."));
        match (
            std::fs::read(dir.join("vocab.json")),
            std::fs::read(dir.join("merges.txt")),
        ) {
            (Ok(vocab), Ok(merges)) => Some((vocab, merges)),
            _ => None,
        }
    });
    let tokenizer =
        tokenizer_bytes
            .as_ref()
            .map(|(vocab, merges)| models::cosyvoice2::TokenizerFiles {
                vocab_json: vocab,
                merges_txt: merges,
            });
    let (builder, report) = models::cosyvoice2::convert_with_config_and_tokenizer(
        bytes,
        config_bytes.as_deref(),
        tokenizer,
    )?;
    let notes = cosyvoice2_notes(&report);

    let tensor_count = builder.tensor_count();
    let metadata_count = builder.metadata_count();
    let out_bytes = builder.to_bytes()?;
    std::fs::write(output, &out_bytes)?;

    Ok(ConvertSummary {
        model: ModelKind::CosyVoice2,
        tensor_count,
        metadata_count,
        output_bytes: out_bytes.len() as u64,
        notes,
    })
}

/// Convert a Moshi (`kyutai/moshiko-pytorch-bf16`) safetensors checkpoint
/// into a Vokra GGUF, optionally embedding the raw
/// `tokenizer_spm_32k_3.model` SentencePiece file as
/// `vokra.tokenizer.model` (M4-06-T22).
///
/// **Streaming / bounded memory**: the checkpoint is opened header-only
/// and every tensor payload is copied one at a time through a reused
/// buffer ([`vokra_core::gguf::GgufStreamWriter`]), so converting the
/// 14 GiB full-7B file peaks at roughly one tensor (~0.26 GiB) — the old
/// materialize-everything path peaked ≈ 97 GiB and could not run on a
/// 16 GB machine.
///
/// **BF16 passes through verbatim** (GGUF `BF16`, ggml type 30 — the
/// Voxtral converter posture): no convert-time widening; the runtime's
/// single `tensor_f32` decode path widens BF16 → f32 **exactly** at load
/// (BF16 is the top half of the f32 pattern). The `vokra.provenance.*`
/// chunks stamp the CC-BY 4.0 `AttributionRequired` class plus the
/// FR-MD-09 attribution text the runtime surfaces
/// (`Session::attribution` / C ABI / CLI banner).
pub fn convert_moshi_file(
    input: &Path,
    tokenizer: Option<&Path>,
    output: &Path,
) -> Result<ConvertSummary, ConvertError> {
    let tokenizer_bytes = match tokenizer {
        Some(p) => Some(std::fs::read(p)?),
        None => None,
    };
    let outcome = models::moshi::convert_streaming(input, output, tokenizer_bytes)?;
    let report = &outcome.report;

    let mut notes = vec![format!(
        "moshi: {} float weights written ({} BF16 passthrough — runtime widens to \
         f32 exactly at load), {} non-float skipped, tokenizer embedded: {}",
        report.written,
        report.bf16_passthrough,
        report.skipped_non_float,
        report.tokenizer_embedded
    )];
    notes.extend(report.notes.iter().map(|n| format!("moshi warning: {n}")));

    Ok(ConvertSummary {
        model: ModelKind::Moshi,
        tensor_count: outcome.tensor_count,
        metadata_count: outcome.metadata_count,
        output_bytes: outcome.output_bytes,
        notes,
    })
}

/// Voxtral (Mistral) side-car hparams supplied by the caller (M3-10-T04). Same
/// shape as the module-private [`models::voxtral::VoxtralConfig`], re-exported
/// here so external callers can build one without pulling in the private
/// module.
// M4-20 T12/T17: DeepFilterNet3 `denoise` offline GGUF path (real checkpoint
// parse from the prepared safetensors + synthetic round-trip writer).
pub use models::denoise::{convert_denoise_bytes, convert_denoise_file, convert_denoise_synthetic};
pub use models::voxtral::VoxtralConfig;

/// Voxtral audio-adapter side-car (M3-10 Wave 8). Callers supply this through
/// [`convert_voxtral_file_with_adapter_config`] (a JSON path) or by
/// constructing an [`AdapterSpec`] directly and attaching it to a
/// [`VoxtralConfig::adapter`] field.
pub use models::voxtral::AdapterSpec;

/// Parses an upstream HuggingFace-style Voxtral `config.json` into a
/// [`VoxtralConfig`] (the `vokra-cli convert --model voxtral --config` path).
/// See [`models::voxtral::parse_hf_config`] for the accepted schema; a JSON
/// with no recognized Voxtral hparams is a hard error (FR-EX-08).
pub fn parse_voxtral_hf_config(bytes: &[u8]) -> Result<VoxtralConfig, ConvertError> {
    models::voxtral::parse_hf_config(bytes)
}

/// Reads a (possibly sharded) Voxtral safetensors checkpoint into one buffer
/// per shard.
///
/// A path whose file name ends in `.index.json` (the HF
/// `model.safetensors.index.json` convention the sharded Voxtral release
/// ships) is parsed for its `weight_map`; every referenced shard file is read
/// from the index's directory, in sorted order (deterministic). Any other
/// path is read verbatim as a single-file checkpoint.
fn read_voxtral_checkpoint(input: &Path) -> Result<Vec<Vec<u8>>, ConvertError> {
    use vokra_core::json::JsonValue;

    let file_name = input.file_name().and_then(|s| s.to_str()).unwrap_or("");
    if !file_name.ends_with(".index.json") {
        return Ok(vec![std::fs::read(input)?]);
    }
    let index_bytes = std::fs::read(input)?;
    let root = vokra_core::json::parse(&index_bytes)
        .map_err(|e| ConvertError::Parse(format!("voxtral index {}: {e}", input.display())))?;
    let weight_map = root
        .get("weight_map")
        .and_then(JsonValue::as_object)
        .ok_or_else(|| {
            ConvertError::Parse(format!(
                "voxtral index {}: missing `weight_map` object",
                input.display()
            ))
        })?;
    // De-duplicate + sort the shard names (many tensors map to each shard).
    let mut shard_names = std::collections::BTreeSet::new();
    for (tensor, file) in weight_map {
        let f = file.as_str().ok_or_else(|| {
            ConvertError::Parse(format!(
                "voxtral index {}: weight_map[{tensor}] is not a file-name string",
                input.display()
            ))
        })?;
        shard_names.insert(f.to_owned());
    }
    if shard_names.is_empty() {
        return Err(ConvertError::Parse(format!(
            "voxtral index {}: empty weight_map — no shards to read",
            input.display()
        )));
    }
    let dir = input.parent().unwrap_or_else(|| Path::new("."));
    shard_names
        .into_iter()
        .map(|f| Ok(std::fs::read(dir.join(f))?))
        .collect()
}

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
    convert_voxtral_file_quantized(input, config, output, None)
}

/// [`convert_voxtral_file`] with an optional K-quant target (M5-15-T36,
/// FR-QT-01).
///
/// `quant` is `Q4_K` / `Q5_K` / `Q6_K`; `None` reproduces
/// [`convert_voxtral_file`] byte for byte. Applicability follows the same rule
/// as the Whisper converter — rank >= 2 and a whole number of 256-element
/// super-blocks — so biases, norms and 1-D tables stay full precision. The
/// upstream release is BF16, which is read through the exact
/// `SafetensorsFile::tensor_f32` widen before quantizing.
///
/// Voxtral is the **only** model besides Whisper with a quantization path:
/// [`convert_file_quantized`]'s hard error for every other model is deliberate
/// (FR-EX-08) and unchanged.
///
/// # Errors
///
/// As [`convert_voxtral_file`], plus [`ConvertError`] from the quantizer when
/// a target dtype is not a K-quant.
pub fn convert_voxtral_file_quantized(
    input: &Path,
    config: &VoxtralConfig,
    output: &Path,
    quant: Option<GgmlType>,
) -> Result<ConvertSummary, ConvertError> {
    let shards = read_voxtral_checkpoint(input)?;
    let (builder, report) = models::voxtral::convert_shards(shards, Some(config), quant)?;

    let notes = vec![format!(
        "voxtral: {} float weights written ({} BF16 passthrough — exact, {} K-quantized to {:?}, \
         {} left full precision as quant-inapplicable), {} non-float skipped, name {}, \
         tokenizer embedded: {}",
        report.written,
        report.bf16_passthrough,
        report.quantized,
        quant,
        report.quant_inapplicable,
        report.skipped_non_float,
        report.name,
        report.tokenizer_embedded
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

/// Convert a Voxtral safetensors checkpoint plus a [`VoxtralConfig`] plus a
/// caller-supplied **adapter config JSON** into a Vokra GGUF (M3-10 Wave 8).
///
/// This is the audio-conditioning-aware sibling of
/// [`convert_voxtral_file`]. In addition to the base config's tokenizer /
/// RoPE / RMSNorm / GQA / vocab side-car, this path also writes the
/// `vokra.voxtral.adapter.*` metadata chunk parsed from
/// `adapter_config`. The chunk tells the runtime
/// [`AudioAdapter::from_gguf`](../vokra_models/voxtral/adapter/struct.AudioAdapter.html)
/// loader where to find the checkpoint's adapter weight tensors (kind, tensor
/// prefix, in / out dims, activation, LayerNorm flags…). Tensor bytes
/// themselves are carried through by the shared safetensors-copy loop —
/// nothing invents upstream tensor names (FR-EX-08 / FR-LD-02 / FR-MD-02).
///
/// # Accepted schema
///
/// See [`AdapterSpec`] and the module docs on
/// [`models::voxtral::parse_adapter_config`](self) for the JSON schema.
///
/// The shape-only [`convert_file`] path and the tokenizer-only
/// [`convert_voxtral_file`] path stay adapter-less; the runtime then treats
/// the model as `AdapterKind::None` and keeps the honest Wave 7
/// LM-continuation posture.
pub fn convert_voxtral_file_with_adapter_config(
    input: &Path,
    config: &VoxtralConfig,
    adapter_config: &Path,
    output: &Path,
) -> Result<ConvertSummary, ConvertError> {
    convert_voxtral_file_with_adapter_config_quantized(input, config, adapter_config, output, None)
}

/// [`convert_voxtral_file_with_adapter_config`] with an optional K-quant
/// target (M5-15-T36). See [`convert_voxtral_file_quantized`] for the
/// applicability rule.
///
/// # Errors
///
/// As [`convert_voxtral_file_with_adapter_config`], plus quantizer errors.
pub fn convert_voxtral_file_with_adapter_config_quantized(
    input: &Path,
    config: &VoxtralConfig,
    adapter_config: &Path,
    output: &Path,
    quant: Option<GgmlType>,
) -> Result<ConvertSummary, ConvertError> {
    let adapter_bytes = std::fs::read(adapter_config)?;
    let spec = models::voxtral::parse_adapter_config(&adapter_bytes)?;
    let mut cfg = config.clone();
    cfg.adapter = Some(spec);
    let shards = read_voxtral_checkpoint(input)?;
    let (builder, report) = models::voxtral::convert_shards(shards, Some(&cfg), quant)?;

    let adapter_kind = cfg
        .adapter
        .as_ref()
        .map(|a| a.kind.as_str())
        .unwrap_or("none");
    let notes = vec![format!(
        "voxtral: {} float weights written ({} BF16 passthrough — exact, {} K-quantized to {:?}, \
         {} left full precision as quant-inapplicable), {} non-float skipped, name {}, \
         tokenizer embedded: {}, adapter kind: {}",
        report.written,
        report.bf16_passthrough,
        report.quantized,
        quant,
        report.quant_inapplicable,
        report.skipped_non_float,
        report.name,
        report.tokenizer_embedded,
        adapter_kind,
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

/// Convert a nari-labs **Dia-1.6B** safetensors checkpoint into a Vokra GGUF
/// (SoTA plan Phase 1-4, 2026-07-24).
///
/// This is the named entry point that mirrors `convert_csm_file` /
/// `convert_dac_file` / `convert_kokoro_file`. It is functionally identical
/// to `convert_file(ModelKind::Dia, input, output)` — Dia has no side-car
/// config or tokenizer to embed (the source vocab is byte-level and the
/// hparams are transcribed as constants in `models::dia`) — but the named
/// entry keeps the `convert_*_file` naming symmetry with the other
/// TTS / codec models.
///
/// The upstream Dia release ships torch `.pth`; run a prepare-checkpoint
/// script (CSM / DAC pattern) to flatten it to safetensors first.
pub fn convert_dia_file(input: &Path, output: &Path) -> Result<ConvertSummary, ConvertError> {
    convert_file(ModelKind::Dia, input, output)
}

/// Convert a Zyphra **Zonos-v0.1-transformer** safetensors checkpoint into a
/// Vokra GGUF (SoTA plan Phase 1-5, 2026-07-24).
///
/// This is the named entry point that mirrors `convert_dia_file` /
/// `convert_csm_file` / `convert_kokoro_file`. It is functionally identical
/// to `convert_file(ModelKind::Zonos, input, output)` — Zonos has no
/// side-car config or tokenizer to embed (every hparam is transcribed as
/// constants in `models::zonos`, and the eSpeak-NG phoneme conditioner keeps
/// its tokenizer state inside the tensor manifest) — but the named entry
/// keeps the `convert_*_file` naming symmetry with the other TTS / codec
/// models.
///
/// The upstream Zonos-v0.1-transformer release ships safetensors directly;
/// no `.pth` prepare step is required (unlike Dia).
pub fn convert_zonos_file(input: &Path, output: &Path) -> Result<ConvertSummary, ConvertError> {
    convert_file(ModelKind::Zonos, input, output)
}

/// Rewrite an existing GGUF's provenance metadata without re-materialising its
/// tensor payloads.
///
/// This is the low-memory publish path used when a converted artifact was
/// stamped with an incomplete provenance group (or none), and re-running the
/// full converter is impractical because the checkpoint no longer fits in this
/// host's RAM. The input is opened via [`vokra_mmap`] so tensor bytes are
/// fault-in-only, and every payload is streamed straight into a new file via
/// [`GgufStreamWriter`] — peak footprint stays at roughly one tensor plus
/// mapped-page cost, not the whole file.
///
/// `license` is the raw SPDX id (class re-derived from it); `model_id` and
/// `source` are advisory provenance strings; `attribution`, when `Some`, sets
/// the CC-BY display text a downstream must show.
///
/// # Errors
///
/// [`ConvertError`] if the input cannot be opened/parsed, a tensor payload is
/// malformed, or the output cannot be written.
#[allow(clippy::too_many_arguments)]
pub fn restamp_provenance(
    input: &Path,
    output: &Path,
    license: &str,
    model_id: &str,
    source: &str,
    attribution: Option<&str>,
) -> Result<ConvertSummary, ConvertError> {
    use vokra_core::gguf::chunks;
    use vokra_core::gguf::{GgufBuilder, GgufStreamWriter, GgufTensorDecl};

    // mmap the input so tensor payloads fault in lazily (never a whole-file
    // read) — this is what keeps the 8.7 GiB Voxtral case within memory.
    let file = vokra_mmap::open_gguf(input)
        .map_err(|e| ConvertError::Parse(format!("restamp: opening {input:?}: {e}")))?;

    // Carry every existing metadata key EXCEPT the ones we set ourselves: the
    // provenance group (replaced below) and the schema stamps (the writer
    // re-emits them universally, so passing them in would duplicate).
    let mut b = GgufBuilder::new();
    for (k, v) in file.metadata() {
        if k == chunks::KEY_PROVENANCE_WEIGHT_LICENSE
            || k == chunks::KEY_PROVENANCE_LICENSE
            || k == chunks::KEY_PROVENANCE_MODEL_ID
            || k == chunks::KEY_PROVENANCE_SOURCE
            || k == chunks::KEY_PROVENANCE_ATTRIBUTION
            || k == chunks::KEY_SCHEMA_VERSION
            || k == chunks::KEY_SCHEMA_PRODUCER
        {
            continue;
        }
        b.add_metadata(k, v.clone());
    }

    // Inject provenance (same conduit the converters use).
    let class = vokra_core::LicenseClass::from_license_str(license);
    b.add_string(chunks::KEY_PROVENANCE_WEIGHT_LICENSE, class.as_str());
    b.add_string(chunks::KEY_PROVENANCE_LICENSE, license);
    b.add_string(chunks::KEY_PROVENANCE_MODEL_ID, model_id);
    b.add_string(chunks::KEY_PROVENANCE_SOURCE, source);
    if let Some(text) = attribution {
        b.add_string(chunks::KEY_PROVENANCE_ATTRIBUTION, text);
    }

    // Declare tensors in the input's order; the stream writer wants only
    // declarations up front, then payloads one at a time.
    let decls: Vec<GgufTensorDecl> = file
        .tensors()
        .iter()
        .map(|t| GgufTensorDecl {
            name: t.name.clone(),
            dtype: t.dtype,
            dimensions: t.dimensions.clone(),
        })
        .collect();
    let tensor_count = decls.len();

    let out_file = std::fs::File::create(output)?;
    let mut w = GgufStreamWriter::begin(std::io::BufWriter::new(out_file), &b, &decls)?;
    // Copy each payload straight from the mapping — no widening, no owned copy
    // beyond the single tensor being written.
    let infos: Vec<_> = file.tensors().to_vec();
    for info in &infos {
        let bytes = file.tensor_bytes(info);
        w.write_tensor(&info.name, bytes)?;
    }
    let out_writer = w.finish()?;
    let out_file = out_writer
        .into_inner()
        .map_err(|e| ConvertError::Io(e.into_error()))?;
    out_file.sync_all().map_err(ConvertError::Io)?;
    let output_bytes = out_file.metadata().map_err(ConvertError::Io)?.len();

    Ok(ConvertSummary {
        model: ModelKind::Voxtral, // placeholder; restamp is model-agnostic
        tensor_count,
        metadata_count: b.metadata_count(),
        output_bytes,
        notes: vec![format!(
            "restamp: {tensor_count} tensors copied verbatim from {input:?}; \
             provenance set to license={license} class={} (tensors unchanged)",
            class.as_str()
        )],
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
