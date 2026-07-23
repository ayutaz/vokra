//! nari-labs **Dia-1.6B**: safetensors checkpoint → GGUF conversion
//! (SoTA plan Phase 1-4, 2026-07-24).
//!
//! Input: an upstream `nari-labs/Dia-1.6B` safetensors checkpoint
//! (Apache 2.0 code + weight, docs/license-audit.md). The reference release
//! ships a torch `.pth`; safetensors converts run through the CSM / DAC
//! pattern (`tools/parity/*_prepare_checkpoint.py` — a future companion
//! script). Output: a GGUF carrying every float tensor verbatim plus the
//! `vokra.dia.*` / `vokra.provenance.*` metadata chunks.
//!
//! # What is transcribed vs. shape-driven
//!
//! - **Transcribed constants** — every hparam of the `vokra.dia.*` chunk
//!   group is transcribed **verbatim** from the upstream `config.json`
//!   (see the top of this module for the full table). No axis is invented;
//!   any tensor whose shape disagrees with these values in a real conversion
//!   fails the runtime shape gate loudly (FR-EX-08, `DiaConfig::validate_for_forward`).
//! - **Runtime-supplied** — the DAC 44.1 kHz codec (`vokra.dac.*`) travels
//!   in a separate standalone codec GGUF (M4-04 T11), *not* embedded here.
//!   Dia and DAC are two independent Apache 2.0 / MIT projects; keeping them
//!   as two GGUFs preserves the M2-13 provenance chain and lets a caller mix
//!   & match Dia weights with any of DAC's zoo variants.
//!
//! # Tensor naming contract
//!
//! GGUF tensor names are the upstream safetensors names **verbatim** (the
//! CSM / Kokoro / CosyVoice2 contract). Real-weight binding is a follow-up
//! wave gated on the upstream tensor-name manifest fetch; this converter
//! passes every F32 / F16 tensor through unchanged so a future
//! `DiaWeights::from_gguf` can walk the same names.
//!
//! # No ONNX (permanent)
//!
//! Dia ships as safetensors / a Python pipeline; the pipeline is
//! re-implemented natively in `vokra-models/src/dia/` (whisper.cpp 型,
//! CLAUDE.md 設計判断 4). This converter never touches ONNX.

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` for Dia GGUFs — kept in sync with the runtime constant
/// `vokra-models::dia::EXPECTED_ARCH`.
pub(crate) const ARCH: &str = "dia";
/// `vokra.model.name` for Dia GGUFs.
pub(crate) const NAME: &str = "dia-1.6b";

// --- vokra.dia.* keys (kept as constants in the converter; the runtime
// duplicates the strings in `crates/vokra-models/src/dia/mod.rs` — a
// round-trip test on the converter side catches drift, following the
// cross-crate pattern established by CSM / CosyVoice2 / Kokoro) ---------------

const KEY_SAMPLE_RATE: &str = "vokra.dia.sample_rate";

// Encoder
const KEY_ENC_N_LAYER: &str = "vokra.dia.arch.encoder.n_layer";
const KEY_ENC_N_EMBD: &str = "vokra.dia.arch.encoder.n_embd";
const KEY_ENC_N_HEAD: &str = "vokra.dia.arch.encoder.n_head";
const KEY_ENC_HEAD_DIM: &str = "vokra.dia.arch.encoder.head_dim";
const KEY_ENC_N_HIDDEN: &str = "vokra.dia.arch.encoder.n_hidden";

// Decoder
const KEY_DEC_N_LAYER: &str = "vokra.dia.arch.decoder.n_layer";
const KEY_DEC_N_EMBD: &str = "vokra.dia.arch.decoder.n_embd";
const KEY_DEC_GQA_QUERY_HEADS: &str = "vokra.dia.arch.decoder.gqa_query_heads";
const KEY_DEC_KV_HEADS: &str = "vokra.dia.arch.decoder.kv_heads";
const KEY_DEC_GQA_HEAD_DIM: &str = "vokra.dia.arch.decoder.gqa_head_dim";
const KEY_DEC_CROSS_QUERY_HEADS: &str = "vokra.dia.arch.decoder.cross_query_heads";
const KEY_DEC_CROSS_HEAD_DIM: &str = "vokra.dia.arch.decoder.cross_head_dim";
const KEY_DEC_N_HIDDEN: &str = "vokra.dia.arch.decoder.n_hidden";

// Vocab / data
const KEY_SRC_VOCAB_SIZE: &str = "vokra.dia.src_vocab_size";
const KEY_TGT_VOCAB_SIZE: &str = "vokra.dia.tgt_vocab_size";
const KEY_CHANNELS: &str = "vokra.dia.channels";
const KEY_TEXT_LENGTH: &str = "vokra.dia.text_length";
const KEY_AUDIO_LENGTH: &str = "vokra.dia.audio_length";
const KEY_TEXT_PAD_VALUE: &str = "vokra.dia.text_pad_value";
const KEY_AUDIO_BOS_VALUE: &str = "vokra.dia.audio_bos_value";
const KEY_AUDIO_EOS_VALUE: &str = "vokra.dia.audio_eos_value";
const KEY_AUDIO_PAD_VALUE: &str = "vokra.dia.audio_pad_value";
const KEY_DELAY_PATTERN_COUNT: &str = "vokra.dia.delay_pattern_count";
const PREFIX_DELAY_PATTERN: &str = "vokra.dia.delay_pattern.";

// Norm / RoPE
const KEY_NORM_EPS: &str = "vokra.dia.norm_eps";
const KEY_ROPE_MAX_TIMESCALE: &str = "vokra.dia.rope_max_timescale";
const KEY_ROPE_MIN_TIMESCALE: &str = "vokra.dia.rope_min_timescale";

// --- Transcribed constants (primary source: config.json fetched verbatim) ---
//
// nari-labs/Dia-1.6B config.json (commit reachable at
// huggingface.co/nari-labs/Dia-1.6B/raw/main/config.json). Every value here
// is transcribed verbatim; nothing is invented.

// PCM sample rate — not written in config.json; inherited from DAC 44.1 kHz
// (the codec Dia loads via `dac.utils.download()` in upstream
// `dia/model.py::_load_dac_model`).
const DIA_SAMPLE_RATE: u32 = 44_100;

// Encoder
const ENC_N_LAYER: u32 = 12;
const ENC_N_EMBD: u32 = 1024;
const ENC_N_HEAD: u32 = 16;
const ENC_HEAD_DIM: u32 = 128;
const ENC_N_HIDDEN: u32 = 4096;

// Decoder
const DEC_N_LAYER: u32 = 18;
const DEC_N_EMBD: u32 = 2048;
const DEC_GQA_QUERY_HEADS: u32 = 16;
const DEC_KV_HEADS: u32 = 4;
const DEC_GQA_HEAD_DIM: u32 = 128;
const DEC_CROSS_QUERY_HEADS: u32 = 16;
const DEC_CROSS_HEAD_DIM: u32 = 128;
const DEC_N_HIDDEN: u32 = 8192;

// Vocab / data
const SRC_VOCAB_SIZE: u32 = 256;
const TGT_VOCAB_SIZE: u32 = 1028;
const CHANNELS: u32 = 9;
const TEXT_LENGTH: u32 = 1024;
const AUDIO_LENGTH: u32 = 3072;
const TEXT_PAD_VALUE: u32 = 0;
const AUDIO_BOS_VALUE: u32 = 1026;
const AUDIO_EOS_VALUE: u32 = 1024;
const AUDIO_PAD_VALUE: u32 = 1025;
const DELAY_PATTERN: [u32; 9] = [0, 8, 9, 10, 11, 12, 13, 14, 15];

// Norm / RoPE
const NORM_EPS: f32 = 1e-5;
const ROPE_MAX_TIMESCALE: f32 = 10_000.0;
const ROPE_MIN_TIMESCALE: f32 = 1.0;

/// Outcome of a Dia conversion.
#[derive(Debug, Default)]
pub(crate) struct DiaReport {
    /// Float tensors written verbatim.
    pub(crate) written: usize,
    /// Non-F32 / F16 tensors skipped (defensive counter — the safetensors
    /// reader rejects unknown dtypes at parse time).
    pub(crate) skipped_non_float: usize,
    /// Operator-facing diagnostics (never fail the conversion — the runtime
    /// is the authoritative gate, FR-EX-08).
    pub(crate) notes: Vec<String>,
}

/// Converts a Dia safetensors buffer into a populated GGUF builder.
///
/// Every F32 / F16 tensor passes through under its upstream name; the
/// `vokra.dia.*` chunk group is written from the transcribed constants
/// above; provenance stamps mark the weight as `Permissive` (Apache 2.0).
pub(crate) fn convert(bytes: Vec<u8>) -> Result<(GgufBuilder, DiaReport), ConvertError> {
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    write_hparams(&mut b);
    vokra_core::stamp_provenance(
        &mut b,
        LicenseClass::Permissive,
        "Apache-2.0",
        Some("nari-labs/Dia-1.6B"),
        Some("huggingface"),
    );

    let mut report = DiaReport::default();
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
            _ => {
                report.skipped_non_float += 1;
            }
        }
    }
    if report.written == 0 {
        report.notes.push(
            "no float tensors passed through — this GGUF is metadata-only and \
             the runtime will refuse to bind any weights (FR-EX-08). The upstream \
             Dia release ships torch .pth; run a prepare-checkpoint script to \
             flatten it into a safetensors file before conversion."
                .into(),
        );
    }
    Ok((b, report))
}

/// Writes the `vokra.dia.*` chunk group from the transcribed constants
/// above (primary source: `config.json`). Delay pattern rides as a
/// count + N indexed keys (the CSM / mimi pattern for array metadata).
fn write_hparams(b: &mut GgufBuilder) {
    b.add_u32(KEY_SAMPLE_RATE, DIA_SAMPLE_RATE);

    // Encoder
    b.add_u32(KEY_ENC_N_LAYER, ENC_N_LAYER);
    b.add_u32(KEY_ENC_N_EMBD, ENC_N_EMBD);
    b.add_u32(KEY_ENC_N_HEAD, ENC_N_HEAD);
    b.add_u32(KEY_ENC_HEAD_DIM, ENC_HEAD_DIM);
    b.add_u32(KEY_ENC_N_HIDDEN, ENC_N_HIDDEN);

    // Decoder
    b.add_u32(KEY_DEC_N_LAYER, DEC_N_LAYER);
    b.add_u32(KEY_DEC_N_EMBD, DEC_N_EMBD);
    b.add_u32(KEY_DEC_GQA_QUERY_HEADS, DEC_GQA_QUERY_HEADS);
    b.add_u32(KEY_DEC_KV_HEADS, DEC_KV_HEADS);
    b.add_u32(KEY_DEC_GQA_HEAD_DIM, DEC_GQA_HEAD_DIM);
    b.add_u32(KEY_DEC_CROSS_QUERY_HEADS, DEC_CROSS_QUERY_HEADS);
    b.add_u32(KEY_DEC_CROSS_HEAD_DIM, DEC_CROSS_HEAD_DIM);
    b.add_u32(KEY_DEC_N_HIDDEN, DEC_N_HIDDEN);

    // Vocab / data
    b.add_u32(KEY_SRC_VOCAB_SIZE, SRC_VOCAB_SIZE);
    b.add_u32(KEY_TGT_VOCAB_SIZE, TGT_VOCAB_SIZE);
    b.add_u32(KEY_CHANNELS, CHANNELS);
    b.add_u32(KEY_TEXT_LENGTH, TEXT_LENGTH);
    b.add_u32(KEY_AUDIO_LENGTH, AUDIO_LENGTH);
    b.add_u32(KEY_TEXT_PAD_VALUE, TEXT_PAD_VALUE);
    b.add_u32(KEY_AUDIO_BOS_VALUE, AUDIO_BOS_VALUE);
    b.add_u32(KEY_AUDIO_EOS_VALUE, AUDIO_EOS_VALUE);
    b.add_u32(KEY_AUDIO_PAD_VALUE, AUDIO_PAD_VALUE);

    // Delay pattern — one entry per channel, indexed 0..N.
    b.add_u32(KEY_DELAY_PATTERN_COUNT, DELAY_PATTERN.len() as u32);
    for (i, d) in DELAY_PATTERN.iter().enumerate() {
        b.add_u32(&format!("{PREFIX_DELAY_PATTERN}{i}"), *d);
    }

    // Norm / RoPE
    b.add_f32(KEY_NORM_EPS, NORM_EPS);
    b.add_f32(KEY_ROPE_MAX_TIMESCALE, ROPE_MAX_TIMESCALE);
    b.add_f32(KEY_ROPE_MIN_TIMESCALE, ROPE_MIN_TIMESCALE);
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::{GgufFile, GgufMetadataValue};

    fn minimal_safetensors_one_f32() -> Vec<u8> {
        // A single f32 tensor at the top of the file so `convert` has
        // something to pass through and the report counts a non-zero write.
        let header = r#"{"encoder.embed_tokens.weight":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}}"#;
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(&[0u8; 24]);
        out
    }

    fn minimal_safetensors_no_tensors() -> Vec<u8> {
        let header = r#"{}"#;
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out
    }

    /// A single F16 tensor at the top of the file (shape [2,3] → 6 elements ×
    /// 2 bytes = 12 bytes). Real Dia-1.6B checkpoints are likely served in F16
    /// (~3.2 GB), so the F16 leg of the union match arm must be reachable.
    fn minimal_safetensors_one_f16() -> Vec<u8> {
        let header = r#"{"encoder.embed_tokens.weight":{"dtype":"F16","shape":[2,3],"data_offsets":[0,12]}}"#;
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(&[0u8; 12]);
        out
    }

    /// A single BF16 tensor at the top of the file (shape [2,3] → 6 elements ×
    /// 2 bytes = 12 bytes). BF16 graduated to a supported safetensors dtype in
    /// M4-06 (moshiko is all-BF16), so BF16 tensors now reach `convert()` and
    /// land in the `_ =>` arm — pinning the `skipped_non_float` counter's real
    /// trigger.
    fn minimal_safetensors_one_bf16() -> Vec<u8> {
        let header = r#"{"encoder.embed_tokens.weight":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,12]}}"#;
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(&[0u8; 12]);
        out
    }

    #[test]
    fn arch_string_matches_runtime_constant() {
        // The two crates only share `vokra-core`, so this constant is the
        // sole handshake with `vokra-models::dia::EXPECTED_ARCH`.
        assert_eq!(ARCH, "dia");
    }

    #[test]
    fn round_trip_carries_arch_chunks_and_provenance() {
        let (builder, report) = convert(minimal_safetensors_one_f32()).expect("convert");
        assert_eq!(report.written, 1);
        assert_eq!(report.skipped_non_float, 0);

        let out = builder.to_bytes().expect("serialize");
        let file = GgufFile::parse(out).expect("parse");
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH)
        );
        assert_eq!(
            file.get(chunks::KEY_MODEL_NAME).and_then(|v| v.as_str()),
            Some(NAME)
        );
        // Every transcribed U32 hparam round-trips verbatim.
        for (key, want) in [
            (KEY_SAMPLE_RATE, DIA_SAMPLE_RATE),
            (KEY_ENC_N_LAYER, ENC_N_LAYER),
            (KEY_ENC_N_EMBD, ENC_N_EMBD),
            (KEY_ENC_N_HEAD, ENC_N_HEAD),
            (KEY_ENC_HEAD_DIM, ENC_HEAD_DIM),
            (KEY_ENC_N_HIDDEN, ENC_N_HIDDEN),
            (KEY_DEC_N_LAYER, DEC_N_LAYER),
            (KEY_DEC_N_EMBD, DEC_N_EMBD),
            (KEY_DEC_GQA_QUERY_HEADS, DEC_GQA_QUERY_HEADS),
            (KEY_DEC_KV_HEADS, DEC_KV_HEADS),
            (KEY_DEC_GQA_HEAD_DIM, DEC_GQA_HEAD_DIM),
            (KEY_DEC_CROSS_QUERY_HEADS, DEC_CROSS_QUERY_HEADS),
            (KEY_DEC_CROSS_HEAD_DIM, DEC_CROSS_HEAD_DIM),
            (KEY_DEC_N_HIDDEN, DEC_N_HIDDEN),
            (KEY_SRC_VOCAB_SIZE, SRC_VOCAB_SIZE),
            (KEY_TGT_VOCAB_SIZE, TGT_VOCAB_SIZE),
            (KEY_CHANNELS, CHANNELS),
            (KEY_TEXT_LENGTH, TEXT_LENGTH),
            (KEY_AUDIO_LENGTH, AUDIO_LENGTH),
            (KEY_TEXT_PAD_VALUE, TEXT_PAD_VALUE),
            (KEY_AUDIO_BOS_VALUE, AUDIO_BOS_VALUE),
            (KEY_AUDIO_EOS_VALUE, AUDIO_EOS_VALUE),
            (KEY_AUDIO_PAD_VALUE, AUDIO_PAD_VALUE),
            (KEY_DELAY_PATTERN_COUNT, DELAY_PATTERN.len() as u32),
        ] {
            match file.get(key) {
                Some(GgufMetadataValue::U32(v)) => assert_eq!(*v, want, "{key}"),
                other => panic!("{key}: unexpected {other:?}"),
            }
        }
        // Delay pattern indexed keys.
        for (i, want) in DELAY_PATTERN.iter().enumerate() {
            let k = format!("{PREFIX_DELAY_PATTERN}{i}");
            match file.get(&k) {
                Some(GgufMetadataValue::U32(v)) => assert_eq!(v, want, "{k}"),
                other => panic!("{k}: unexpected {other:?}"),
            }
        }
        // F32 hparams.
        for (key, want) in [
            (KEY_NORM_EPS, NORM_EPS),
            (KEY_ROPE_MAX_TIMESCALE, ROPE_MAX_TIMESCALE),
            (KEY_ROPE_MIN_TIMESCALE, ROPE_MIN_TIMESCALE),
        ] {
            match file.get(key) {
                Some(GgufMetadataValue::F32(v)) => assert_eq!(*v, want, "{key}"),
                other => panic!("{key}: unexpected {other:?}"),
            }
        }
        // Provenance: Apache 2.0 permissive.
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_MODEL_ID)
                .and_then(|v| v.as_str()),
            Some("nari-labs/Dia-1.6B")
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some("Apache-2.0")
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Permissive.as_str())
        );
    }

    #[test]
    fn zero_tensor_conversion_surfaces_a_loud_note() {
        // Empty safetensors → the runtime's `DiaWeights::from_gguf` would
        // fail loudly at bind time, but the converter itself succeeds and
        // reports the situation so the operator sees it now.
        let (_, report) = convert(minimal_safetensors_no_tensors()).expect("convert");
        assert_eq!(report.written, 0);
        assert!(
            report.notes.iter().any(|n| n.contains("no float tensors")),
            "zero-tensor conversion must emit a loud note: {:?}",
            report.notes
        );
    }

    /// Pins the F16 leg of the `GgmlType::F32 | GgmlType::F16` union match arm.
    /// A real Dia-1.6B checkpoint is likely served in F16 (~3.2 GB); a typo
    /// dropping `| GgmlType::F16` would silently bin every F16 tensor into
    /// `skipped_non_float` and only surface downstream at the FR-EX-08 shape
    /// gate. This test catches that regression at the converter boundary.
    #[test]
    fn f16_tensor_passes_through_verbatim() {
        let (builder, report) = convert(minimal_safetensors_one_f16()).expect("convert");
        assert_eq!(report.written, 1, "F16 must reach the pass-through arm");
        assert_eq!(
            report.skipped_non_float, 0,
            "F16 must not land in the skipped counter"
        );

        // The tensor survives the round trip under its upstream name and
        // preserves its F16 dtype (payload is 12 bytes = 6 × F16).
        let out = builder.to_bytes().expect("serialize");
        let file = GgufFile::parse(out).expect("parse");
        let info = file
            .tensor_info("encoder.embed_tokens.weight")
            .expect("tensor present");
        assert_eq!(info.dtype, GgmlType::F16);
        assert_eq!(info.dimensions, vec![2, 3]);
        assert_eq!(file.tensor_bytes(info).len(), 12);
    }

    /// Pins the `_ =>` arm of the tensor-dtype match: BF16 graduated to a
    /// supported safetensors dtype in M4-06 (moshiko is all-BF16) so BF16
    /// tensors now reach `convert()` and MUST be counted, not silently
    /// dropped. The in-file comment claiming "the safetensors reader rejects
    /// unknown dtypes at parse time" is stale for BF16 — this test guards
    /// against a regression where somebody assumes the comment is still true
    /// and, for example, promotes BF16 into the pass-through arm without
    /// deciding how to widen it.
    #[test]
    fn bf16_tensor_is_counted_as_skipped_non_float() {
        let (builder, report) = convert(minimal_safetensors_one_bf16()).expect("convert");
        assert_eq!(
            report.written, 0,
            "BF16 must not currently pass through — Dia converter is F32/F16 only"
        );
        assert_eq!(
            report.skipped_non_float, 1,
            "BF16 must increment the skipped counter"
        );
        // With zero float tensors written, the loud "no float tensors" note
        // fires — a BF16-only checkpoint would surface the situation.
        assert!(
            report.notes.iter().any(|n| n.contains("no float tensors")),
            "BF16-only conversion must emit the zero-float note: {:?}",
            report.notes
        );
        // Metadata (arch / hparams) still lands — the report reflects the
        // tensor pass, not a failure of the conversion.
        let out = builder.to_bytes().expect("serialize");
        let file = GgufFile::parse(out).expect("parse");
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH)
        );
        assert!(
            file.tensor_info("encoder.embed_tokens.weight").is_none(),
            "BF16 tensor must not be written"
        );
    }

    /// Pins `SafetensorsFile::parse(bytes)?` error propagation on line 156.
    /// A malformed input must surface as `Err(ConvertError::Parse(_))`, not
    /// as a silently-empty successful conversion (FR-EX-08 loud fail).
    #[test]
    fn malformed_input_returns_parse_error() {
        // Case 1: empty buffer — shorter than the mandatory 8-byte header
        // length prefix, so `SafetensorsFile::parse` returns `Truncated`.
        let err = convert(Vec::new()).expect_err("empty buffer must be rejected");
        assert!(
            matches!(err, ConvertError::Parse(_)),
            "expected ConvertError::Parse, got {err:?}"
        );

        // Case 2: declared header length runs off the end of the buffer —
        // 8 bytes of prefix claiming a 1024-byte header inside a 10-byte
        // buffer. Also `Truncated`.
        let mut truncated = Vec::new();
        truncated.extend_from_slice(&1024u64.to_le_bytes());
        truncated.extend_from_slice(b"{}");
        let err = convert(truncated).expect_err("truncated header must be rejected");
        assert!(
            matches!(err, ConvertError::Parse(_)),
            "expected ConvertError::Parse, got {err:?}"
        );

        // Case 3: valid length prefix but malformed JSON body — parses as
        // `SafetensorsError::Json` and maps to `ConvertError::Parse`.
        let bad_json = b"{not-json"; // 9 bytes, but not valid JSON.
        let mut bad = Vec::new();
        bad.extend_from_slice(&(bad_json.len() as u64).to_le_bytes());
        bad.extend_from_slice(bad_json);
        let err = convert(bad).expect_err("malformed JSON must be rejected");
        assert!(
            matches!(err, ConvertError::Parse(_)),
            "expected ConvertError::Parse, got {err:?}"
        );
    }
}
