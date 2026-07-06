//! Kokoro-82M (StyleTTS 2 派生 iSTFTNet): safetensors checkpoint to GGUF
//! conversion (M2-07-T06/T07 foundation).
//!
//! Input: the upstream `hexgrad/Kokoro-82M` safetensors checkpoint (weights
//! only — no model code is imported, per IF-06 / FR-MD-02). Output: a GGUF
//! carrying every float tensor plus the `vokra.model.*` and `vokra.kokoro.*`
//! metadata chunks the native Kokoro implementation (a later WP) loads
//! against.
//!
//! # Tensor naming contract (M2-07 foundation)
//!
//! GGUF tensor names are the **upstream safetensors names verbatim** (same
//! contract Whisper uses). Rich Vokra-side renaming can arrive later without
//! changing the guarantees of this module.
//!
//! # No `vokra.frontend.*` chunk
//!
//! Kokoro is a TTS decoder — it has no audio front-end (mel/STFT feature
//! extractor) that the runtime controls. Its **output-side** iSTFT is stored
//! under `vokra.kokoro.istft.*` (mirroring piper's `vokra.piper.istft.*`),
//! **not** under the `vokra.frontend.*` input-side chunk.
//!
//! # iSTFTNet head, not vocos
//!
//! Kokoro's vocoder is StyleTTS 2 派生 iSTFTNet (レビュアー A 修正, CLAUDE.md
//! モデル表). The runtime decoder will lower magnitude+phase to complex re/im
//! inline and call `vokra_ops::istft` (FR-OP-01), not `vocos_head` (FR-OP-12).
//! This converter never emits a `vocos_*` metadata key.
//!
//! # Scope
//!
//! Foundation WP only: verbatim safetensors → GGUF, shape-driven hparams where
//! possible with `0` placeholders (mirroring Whisper's degenerate-shape
//! pattern) for values that need T02 upstream inspection. Voicepack layout,
//! phoneme table, and voice name list are deliberately left as empty
//! placeholders here — the follow-up ticket wires them in with an explicit
//! `--config config.json` input (same shape as piper-plus).

use vokra_core::gguf::{
    GgmlType, GgufArray, GgufBuilder, GgufMetadataValue, GgufValueType, chunks,
};

use crate::ConvertError;
use crate::safetensors::{SafeTensorInfo, SafetensorsFile};

/// `vokra.model.arch` value written for Kokoro-82M GGUFs.
pub(crate) const ARCH: &str = "kokoro-82m-istftnet";
/// `vokra.model.name` value written for the Kokoro-82M GGUF.
pub(crate) const NAME: &str = "kokoro-82m";

// --- vokra.kokoro.* metadata keys (M2-07-T06 chunk design) ------------------

/// `vokra.kokoro.sample_rate` — output PCM sample rate, Hz (`UINT32`).
const KEY_SAMPLE_RATE: &str = "vokra.kokoro.sample_rate";
/// `vokra.kokoro.style_dim` — per-voice style vector dimension (`UINT32`).
const KEY_STYLE_DIM: &str = "vokra.kokoro.style_dim";
/// `vokra.kokoro.num_voices` — voicepack voice count (`UINT32`).
const KEY_NUM_VOICES: &str = "vokra.kokoro.num_voices";
/// `vokra.kokoro.n_text_layers` — text encoder block count (`UINT32`).
const KEY_N_TEXT_LAYERS: &str = "vokra.kokoro.n_text_layers";
/// `vokra.kokoro.n_decoder_layers` — iSTFTNet decoder upsample stage count
/// (`UINT32`).
const KEY_N_DECODER_LAYERS: &str = "vokra.kokoro.n_decoder_layers";
/// `vokra.kokoro.hidden_dim` — text encoder hidden width (`UINT32`).
const KEY_HIDDEN_DIM: &str = "vokra.kokoro.hidden_dim";
/// `vokra.kokoro.istft.n_fft` — decoder iSTFT FFT size (`UINT32`).
const KEY_ISTFT_N_FFT: &str = "vokra.kokoro.istft.n_fft";
/// `vokra.kokoro.istft.hop` — decoder iSTFT hop length (`UINT32`).
const KEY_ISTFT_HOP: &str = "vokra.kokoro.istft.hop";
/// `vokra.kokoro.istft.win_length` — decoder iSTFT window length (`UINT32`).
const KEY_ISTFT_WIN_LENGTH: &str = "vokra.kokoro.istft.win_length";
/// `vokra.kokoro.phoneme_symbols` — phoneme string per id (`ARRAY<STRING>`).
const KEY_PHONEME_SYMBOLS: &str = "vokra.kokoro.phoneme_symbols";
/// `vokra.kokoro.voice_names` — voicepack entry name per row (`ARRAY<STRING>`).
const KEY_VOICE_NAMES: &str = "vokra.kokoro.voice_names";

/// Kokoro-82M output sample rate (Hz). Sourced from the hexgrad/Kokoro-82M
/// Hugging Face model card — publicly documented and not invented (constraint
/// note: hparam numbers that come from official model cards are permitted;
/// `0`-placeholder values on this module's other keys are reserved for the
/// truly TBD `istft.*` triple).
const KOKORO_SAMPLE_RATE: u32 = 24_000;

// Note on istft.n_fft / istft.hop / istft.win_length:
//
// These three values are structural to Kokoro's iSTFTNet head but their
// upstream ground-truth requires T02 checkpoint inspection (the plan
// deliberately marks the layout of `decoder.generator.conv_post.weight` — the
// tensor whose output channels encode the STFT bin count — as TBD). Rather
// than invent numbers, this foundation writes `0` for all three, matching
// Whisper's degenerate-shape pattern: a runtime consumer of this GGUF must
// reject a `0` on load (FR-EX-08 — no silent fallback). A follow-up ticket
// derives them from the actual decoder tensor shapes.

/// Outcome of a Kokoro conversion.
#[derive(Debug, Default)]
pub(crate) struct KokoroReport {
    /// Number of float weight tensors written to the GGUF.
    pub(crate) written: usize,
    /// Tensors whose dtype falls outside the F32/F16 range and were skipped.
    ///
    /// The upstream safetensors reader (`vokra_core::safetensors`) already
    /// rejects unknown dtypes at parse time (`SafetensorsError::UnsupportedDtype`),
    /// so a validly parsed buffer that reaches this converter only ever holds
    /// F32/F16 tensors. This counter is defensive/forward-compat — if the
    /// reader is later extended to admit non-float dtypes (e.g. INT8 quant),
    /// the skip path already exists and the report already reports.
    pub(crate) skipped_non_float: usize,
    /// Voice names in voicepack order (populated by a follow-up ticket that
    /// wires `--config config.json`).
    pub(crate) voices: Vec<String>,
    /// Per-voice style vector dimension (derived from `voicepack` shape[1]
    /// when the tensor is present, else `0`).
    pub(crate) style_dim: usize,
}

/// Reads dimension `axis` of tensor `name` from the checkpoint, or `0` when
/// the tensor (or that axis) is absent — a degenerate checkpoint the runtime
/// then rejects at load (FR-EX-08). Shared by [`convert`] and
/// [`write_hparams`] so every derivation reads the identical value.
fn tensor_dim(st: &SafetensorsFile, name: &str, axis: usize) -> u64 {
    st.tensors()
        .iter()
        .find(|t: &&SafeTensorInfo| t.name == name)
        .and_then(|t| t.shape.get(axis).copied())
        .unwrap_or(0)
}

/// Counts contiguous transformer blocks named `<prefix><i>.` for
/// `i = 0, 1, …`.
fn count_layers(st: &SafetensorsFile, prefix: &str) -> u32 {
    let mut n = 0u32;
    loop {
        let probe = format!("{prefix}{n}.");
        if st.tensors().iter().any(|t| t.name.starts_with(&probe)) {
            n += 1;
        } else {
            return n;
        }
    }
}

/// Derives the `vokra.kokoro.*` hparams from tensor shapes and writes them
/// into `b`.
///
/// Every value is read from a tensor shape (or a well-documented model-card
/// invariant like `sample_rate = 24_000`). Missing tensors write `0` — the
/// converter stays infallible so degenerate synthetic inputs still round-trip,
/// but a `0` on a required hparam is rejected by the runtime loader at load
/// time (FR-EX-08 — no silent fallback in the runtime).
fn write_hparams(b: &mut GgufBuilder, st: &SafetensorsFile) -> u64 {
    // Shape-driven derivations. The tensor names below are the *foundation*
    // guesses used to shape-drive hparams from typical StyleTTS 2 派生 iSTFTNet
    // exports; a follow-up ticket refines them against the actual
    // hexgrad/Kokoro-82M safetensors layout. Missing tensors yield `0`.
    //
    // - voicepack[num_voices, style_dim]           — style-vector table
    // - text_encoder.embedding.weight[n_sym, hidden] — text embedding
    // - text_encoder.layers.<i>.                    — encoder blocks
    // - decoder.generator.upsamples.<i>.            — decoder upsample stages
    let num_voices = tensor_dim(st, "voicepack", 0);
    let style_dim = tensor_dim(st, "voicepack", 1);
    let hidden_dim = tensor_dim(st, "text_encoder.embedding.weight", 1);
    let n_text_layers = count_layers(st, "text_encoder.layers.");
    let n_decoder_layers = count_layers(st, "decoder.generator.upsamples.");

    b.add_u32(KEY_SAMPLE_RATE, KOKORO_SAMPLE_RATE);
    b.add_u32(KEY_STYLE_DIM, style_dim as u32);
    b.add_u32(KEY_NUM_VOICES, num_voices as u32);
    b.add_u32(KEY_N_TEXT_LAYERS, n_text_layers);
    b.add_u32(KEY_N_DECODER_LAYERS, n_decoder_layers);
    b.add_u32(KEY_HIDDEN_DIM, hidden_dim as u32);
    // iSTFT hyper-parameters — foundation placeholder `0`s (see module note).
    b.add_u32(KEY_ISTFT_N_FFT, 0);
    b.add_u32(KEY_ISTFT_HOP, 0);
    b.add_u32(KEY_ISTFT_WIN_LENGTH, 0);
    // Empty tables — populated by a follow-up ticket that wires an explicit
    // `--config config.json` input (same shape as piper-plus).
    b.add_metadata(
        KEY_PHONEME_SYMBOLS,
        GgufMetadataValue::Array(GgufArray {
            element_type: GgufValueType::String,
            values: Vec::new(),
        }),
    );
    b.add_metadata(
        KEY_VOICE_NAMES,
        GgufMetadataValue::Array(GgufArray {
            element_type: GgufValueType::String,
            values: Vec::new(),
        }),
    );

    style_dim
}

/// Converts a Kokoro-82M safetensors buffer into a populated GGUF builder
/// plus a report of what was written vs. skipped.
///
/// Every tensor is written verbatim (bytes, dtype and shape preserved); no
/// FP16 → FP32 widening (M2-07 keeps the source dtype so the follow-up
/// quantization policy can act on the same bytes the checkpoint shipped).
pub(crate) fn convert(bytes: Vec<u8>) -> Result<(GgufBuilder, KokoroReport), ConvertError> {
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    let style_dim = write_hparams(&mut b, &st);

    let mut report = KokoroReport {
        style_dim: style_dim as usize,
        ..Default::default()
    };

    for t in st.tensors() {
        match t.dtype {
            GgmlType::F32 | GgmlType::F16 => {
                b.add_tensor(
                    &t.name,
                    t.dtype,
                    t.shape.clone(),
                    st.tensor_bytes(t).to_vec(),
                )?;
                report.written += 1;
            }
            // Defensive: the upstream safetensors reader rejects non-F32/F16
            // at parse time, so this arm is currently unreachable through a
            // validly parsed buffer. Kept so a future reader extension does
            // not silently write an unsupported dtype (FR-EX-08).
            _ => {
                report.skipped_non_float += 1;
            }
        }
    }

    Ok((b, report))
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::GgufFile;

    /// Builds a synthetic Kokoro-like safetensors buffer with a small set of
    /// F32 tensors laid out contiguously.
    ///
    /// The names track the foundation shape-driver in [`write_hparams`] so
    /// every `vokra.kokoro.*` numeric hparam derives a non-zero value from
    /// this buffer; the payloads are minimal (all-zero) since only shapes
    /// drive the assertions.
    fn synthetic_kokoro_safetensors() -> Vec<u8> {
        // (name, shape) — element count = product; F32 payload = 4 * elems.
        let entries: &[(&str, &[u64])] = &[
            // voicepack [num_voices=2, style_dim=4] → 32 bytes.
            ("voicepack", &[2, 4]),
            // text_encoder.embedding.weight [n_sym=3, hidden=8] → 96 bytes.
            ("text_encoder.embedding.weight", &[3, 8]),
            // Two encoder blocks → contiguous prefix "text_encoder.layers.<i>."
            ("text_encoder.layers.0.attn.q_proj.weight", &[1, 1]),
            ("text_encoder.layers.1.attn.q_proj.weight", &[1, 1]),
            // One decoder upsample stage.
            ("decoder.generator.upsamples.0.weight", &[1, 1]),
            // A synthesis-side tensor.
            ("decoder.generator.conv_pre.weight", &[1, 1]),
            // A prosody predictor tensor.
            ("predictor.duration.weight", &[1, 1]),
        ];

        let mut cursor = 0usize;
        let mut header_entries = Vec::new();
        for &(name, shape) in entries {
            let elems: u64 = shape.iter().product();
            let span = elems as usize * 4;
            let begin = cursor;
            let end = cursor + span;
            cursor = end;
            let dims = shape
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(",");
            header_entries.push(format!(
                r#""{name}":{{"dtype":"F32","shape":[{dims}],"data_offsets":[{begin},{end}]}}"#
            ));
        }
        let header = format!("{{{}}}", header_entries.join(","));
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(&vec![0u8; cursor]);
        out
    }

    #[test]
    fn converts_and_writes_kokoro_metadata_keys() {
        let (builder, report) = convert(synthetic_kokoro_safetensors()).expect("convert");
        let file = GgufFile::parse(builder.to_bytes().unwrap()).unwrap();

        // Model chunk present.
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH)
        );
        assert_eq!(
            file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
            Some(NAME)
        );

        // Every `vokra.kokoro.*` key from the T06 chunk design is present.
        let u = |k: &str| file.get(k).and_then(|v| v.as_u64());
        assert_eq!(u(KEY_SAMPLE_RATE), Some(u64::from(KOKORO_SAMPLE_RATE)));
        // Shape-driven derivations: `voicepack` [2, 4], embedding [3, 8], two
        // text_encoder.layers.<i>., one decoder.generator.upsamples.<i>.
        assert_eq!(u(KEY_NUM_VOICES), Some(2));
        assert_eq!(u(KEY_STYLE_DIM), Some(4));
        assert_eq!(u(KEY_HIDDEN_DIM), Some(8));
        assert_eq!(u(KEY_N_TEXT_LAYERS), Some(2));
        assert_eq!(u(KEY_N_DECODER_LAYERS), Some(1));
        // iSTFT triple: TBD-placeholder `0`s (see module note).
        assert_eq!(u(KEY_ISTFT_N_FFT), Some(0));
        assert_eq!(u(KEY_ISTFT_HOP), Some(0));
        assert_eq!(u(KEY_ISTFT_WIN_LENGTH), Some(0));
        // String-array keys present, empty (populated by follow-up config
        // wiring).
        let syms = file
            .get(KEY_PHONEME_SYMBOLS)
            .and_then(|v| v.as_array())
            .expect("phoneme_symbols present");
        assert_eq!(syms.element_type, GgufValueType::String);
        assert!(syms.values.is_empty());
        let voices = file
            .get(KEY_VOICE_NAMES)
            .and_then(|v| v.as_array())
            .expect("voice_names present");
        assert_eq!(voices.element_type, GgufValueType::String);
        assert!(voices.values.is_empty());

        // No `vokra.frontend.*` chunk (Kokoro is TTS-only, no input front-end).
        assert!(file.get(chunks::KEY_FRONTEND_N_FFT).is_none());

        // Every input tensor round-tripped verbatim.
        assert_eq!(report.written, 7);
        assert_eq!(report.skipped_non_float, 0);
        assert_eq!(report.style_dim, 4);
        // Bytes preserved for at least one representative tensor.
        let info = file.tensor_info("voicepack").expect("voicepack in gguf");
        assert_eq!(info.dtype, GgmlType::F32);
        assert_eq!(info.dimensions, vec![2, 4]);
    }

    #[test]
    fn skips_non_float_and_reports() {
        // The upstream safetensors reader admits only F32 and F16
        // (`SafetensorsFile::parse` returns `UnsupportedDtype` on anything
        // else), so a validly parsed buffer that reaches `convert()` cannot
        // hold a truly non-float tensor. Two complementary assertions still
        // verify the reporter contract:
        //
        // (1) The `skipped_non_float` counter is present on the report and
        //     correctly reports `0` for an all-F32 buffer (the counter is
        //     defensive/forward-compat — if the safetensors reader is later
        //     extended, the skip path already exists).
        // (2) A safetensors buffer whose *declared* dtype is non-float (I64
        //     here) is rejected at parse time with a `ConvertError::Parse`
        //     wrapping the reader's `UnsupportedDtype` — non-float bytes never
        //     silently reach `add_tensor`.
        let (_, report) = convert(synthetic_kokoro_safetensors()).expect("convert");
        assert_eq!(
            report.skipped_non_float, 0,
            "all-F32 buffer must report zero skipped tensors"
        );
        assert!(
            report.written > 0,
            "all-F32 buffer must report written tensors"
        );

        // Non-float buffer: an I64 tensor in a safetensors header. Parse-side
        // rejection is the runtime's non-float gate for this converter today.
        let header = r#"{"i.const":{"dtype":"I64","shape":[1],"data_offsets":[0,8]}}"#;
        let mut buf = Vec::new();
        buf.extend_from_slice(&(header.len() as u64).to_le_bytes());
        buf.extend_from_slice(header.as_bytes());
        buf.extend_from_slice(&[0u8; 8]);
        let err = convert(buf).expect_err("I64 must be rejected at parse time");
        let msg = format!("{err}");
        assert!(
            msg.contains("I64") || msg.contains("dtype"),
            "expected parse-side non-float rejection, got: {msg}"
        );
    }
}
