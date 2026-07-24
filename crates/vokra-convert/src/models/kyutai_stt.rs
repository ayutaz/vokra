//! Kyutai **STT-2.6B-EN**: safetensors checkpoint → GGUF conversion
//! (SoTA plan Phase 2, 2026-07-24).
//!
//! Input: an upstream `kyutai/stt-2.6b-en` safetensors checkpoint
//! (Apache 2.0 code + **CC-BY 4.0** weight — `docs/license-audit.md` Kyutai
//! row). The reference release ships raw safetensors directly; no `.pth`
//! prepare step is required (unlike Dia). Output: a GGUF carrying every
//! F32 / F16 tensor verbatim plus the `vokra.kyutai_stt.*` /
//! `vokra.provenance.*` metadata chunks.
//!
//! # What is transcribed vs. shape-driven
//!
//! - **Transcribed constants** — every hparam of the `vokra.kyutai_stt.*`
//!   chunk group is transcribed **verbatim** from the upstream
//!   `config.json` (see the top of this module for the full table). No
//!   axis is invented; any tensor whose shape disagrees with these values
//!   in a real conversion fails the runtime shape gate loudly (FR-EX-08,
//!   `KyutaiSttConfig::validate_for_forward`).
//! - **Runtime-supplied** — the Mimi codec (`vokra.mimi.*`) travels in a
//!   separate standalone codec GGUF (M4-04 T10 / T11), *not* embedded
//!   here. Kyutai STT and Mimi are two boundaries (Apache 2.0 code +
//!   CC-BY 4.0 weights each); keeping them as two GGUFs preserves the
//!   M2-13 provenance chain and lets a caller pair the STT weights with
//!   any 24 kHz Mimi codec GGUF.
//!
//! # Tensor naming contract
//!
//! GGUF tensor names are the upstream safetensors names **verbatim** (the
//! CSM / Kokoro / CosyVoice2 / Dia / Zonos contract). Real-weight binding
//! is a follow-up wave gated on the upstream tensor-name manifest fetch;
//! this converter passes every F32 / F16 tensor through unchanged so a
//! future `KyutaiSttWeights::from_gguf` can walk the same names.
//!
//! # BF16 posture
//!
//! The upstream Kyutai release is **BF16** (~5.2 GB for the 2.6B). The
//! streaming-BF16 pass-through path Moshi uses is a follow-up (T29-
//! equivalent — the same owner hand-off pattern); the M2-13 provenance
//! chain and the `vokra.kyutai_stt.*` chunk group land now so a caller can
//! walk the metadata surface even against a shape-only fixture. BF16
//! tensors reach the `_ =>` arm and are counted in `skipped_non_float`
//! (Dia posture — never a silent widen or drop).
//!
//! # No ONNX (permanent)
//!
//! Kyutai STT ships as safetensors / a Python pipeline; the pipeline is
//! re-implemented natively in `vokra-models/src/kyutai_stt/` (whisper.cpp
//! 型, CLAUDE.md 設計判断 4). This converter never touches ONNX.

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` for Kyutai STT GGUFs — kept in sync with the runtime
/// constant `vokra-models::kyutai_stt::EXPECTED_ARCH`.
pub(crate) const ARCH: &str = "kyutai-stt";
/// `vokra.model.name` for Kyutai STT GGUFs.
pub(crate) const NAME: &str = "kyutai-stt-2.6b-en";

/// The FR-MD-09 attribution text stamped into
/// `vokra.provenance.attribution` — wording aligned with `NOTICE` and the
/// `docs/license-audit.md` Kyutai row (final legal sufficiency = T29-
/// equivalent owner sign-off; this converter records the attribution but
/// the owner-facing publish gate can add / edit before release).
pub(crate) const KYUTAI_STT_ATTRIBUTION_TEXT: &str = "This application uses the Kyutai STT-2.6B-EN model \
     (decoder-only English streaming ASR over Mimi audio tokens). Model \
     weights are licensed under CC-BY 4.0 (attribution required; commercial \
     use permitted). Copyright (c) Kyutai. Source: \
     https://github.com/kyutai-labs/delayed-streams-modeling / \
     https://huggingface.co/kyutai/stt-2.6b-en";

// --- vokra.kyutai_stt.* keys (kept as constants in the converter; the
// runtime duplicates the strings when it lands `KyutaiSttConfig::from_gguf`
// — the cross-crate pattern established by CSM / CosyVoice2 / Kokoro / Dia
// / Zonos) ------------------------------------------------------------------

const KEY_SAMPLE_RATE: &str = "vokra.kyutai_stt.sample_rate";

// Backbone
const KEY_BB_N_LAYER: &str = "vokra.kyutai_stt.arch.backbone.n_layer";
const KEY_BB_D_MODEL: &str = "vokra.kyutai_stt.arch.backbone.d_model";
const KEY_BB_N_HEAD: &str = "vokra.kyutai_stt.arch.backbone.n_head";
const KEY_BB_HIDDEN_SCALE: &str = "vokra.kyutai_stt.arch.backbone.hidden_scale";
const KEY_BB_FFN_HIDDEN: &str = "vokra.kyutai_stt.arch.backbone.ffn_hidden";
const KEY_BB_CONTEXT: &str = "vokra.kyutai_stt.arch.backbone.context";
const KEY_BB_ROPE_MAX_PERIOD: &str = "vokra.kyutai_stt.arch.backbone.rope_max_period";
const KEY_BB_CAUSAL: &str = "vokra.kyutai_stt.arch.backbone.causal";
const KEY_BB_RMS_NORM_EPS: &str = "vokra.kyutai_stt.arch.backbone.rms_norm_eps";

// Depformer (structurally present, unused for audio when dep_q=0)
const KEY_DEP_N_LAYER: &str = "vokra.kyutai_stt.arch.depformer.n_layer";
const KEY_DEP_D_MODEL: &str = "vokra.kyutai_stt.arch.depformer.d_model";
const KEY_DEP_N_HEAD: &str = "vokra.kyutai_stt.arch.depformer.n_head";
const KEY_DEP_MULTI_LINEAR: &str = "vokra.kyutai_stt.arch.depformer.multi_linear";
const KEY_DEP_WEIGHTS_PER_STEP: &str = "vokra.kyutai_stt.arch.depformer.weights_per_step";

// Audio input / text / streaming
const KEY_N_Q: &str = "vokra.kyutai_stt.audio.n_q";
const KEY_DEP_Q: &str = "vokra.kyutai_stt.audio.dep_q";
const KEY_AUDIO_CARD: &str = "vokra.kyutai_stt.audio.card";
const KEY_TEXT_CARD: &str = "vokra.kyutai_stt.text.card";
const KEY_TEXT_PAD_ID: &str = "vokra.kyutai_stt.text.pad_id";
const KEY_AUDIO_DELAY_SECS: &str = "vokra.kyutai_stt.stream.audio_delay_seconds";
const KEY_AUDIO_SILENCE_PREFIX_SECS: &str = "vokra.kyutai_stt.stream.audio_silence_prefix_seconds";

// Delays (indexed keys — the CSM / Moshi / Dia pattern for array metadata)
const KEY_N_DELAYS: &str = "vokra.kyutai_stt.n_delays";
const PREFIX_DELAY: &str = "vokra.kyutai_stt.delay.";

// --- Transcribed constants (primary source: config.json fetched verbatim) --
//
// `huggingface.co/kyutai/stt-2.6b-en/raw/main/config.json` (fetched
// 2026-07-24). Every value here is transcribed verbatim; nothing is
// invented.

// PCM sample rate — not written in config.json; inherited from Mimi (the
// codec `mimi_name` = `mimi-pytorch-e351c8d8@125.safetensors` at 24 kHz /
// 12.5 Hz).
const KYUTAI_STT_SAMPLE_RATE: u32 = 24_000;

// Backbone (config.json:top-level)
const BB_N_LAYER: u32 = 48; // "num_layers": 48
const BB_D_MODEL: u32 = 2048; // "dim": 2048
const BB_N_HEAD: u32 = 32; // "num_heads": 32
const BB_HIDDEN_SCALE: f32 = 4.125; // "hidden_scale": 4.125
// Derived: round(hidden_scale * d_model) = round(4.125 * 2048) = 8448.
// Stored explicitly so a `from_gguf` reader can honour the value without
// re-doing the derivation (a checkpoint whose shapes disagree with this
// value is a loud FR-EX-08 error at bind time).
const BB_FFN_HIDDEN: u32 = 8448;
const BB_CONTEXT: u32 = 375; // "context": 375
const BB_ROPE_MAX_PERIOD: f32 = 100_000.0; // "max_period": 100000.0
const BB_CAUSAL: bool = true; // "causal": true
// `norm: "rms_norm_f32"` upstream → ε = 1e-8 (Moshi transformer.py
// `create_norm_fn`; mirrored from ADR M4-06 §D2).
const BB_RMS_NORM_EPS: f32 = 1e-8;

// Depformer (config.json:top-level `depformer_*`)
const DEP_N_LAYER: u32 = 6; // "depformer_num_layers": 6
const DEP_D_MODEL: u32 = 1024; // "depformer_dim": 1024
const DEP_N_HEAD: u32 = 16; // "depformer_num_heads": 16
const DEP_MULTI_LINEAR: bool = true; // "depformer_multi_linear": true
const DEP_WEIGHTS_PER_STEP: bool = true; // "depformer_weights_per_step": true

// Audio / text / streaming
const N_Q: u32 = 32; // "n_q": 32
const DEP_Q: u32 = 0; // "dep_q": 0 (text-only prediction)
const AUDIO_CARD: u32 = 2048; // "card": 2048 (Mimi codebook size)
const TEXT_CARD: u32 = 4000; // "text_card": 4000
const TEXT_PAD_ID: u32 = 3; // "existing_text_padding_id": 3
const AUDIO_DELAY_SECS: f32 = 2.5; // "stt_config.audio_delay_seconds"
const AUDIO_SILENCE_PREFIX_SECS: f32 = 1.0; // "stt_config.audio_silence_prefix_seconds"

// Delays — 33 entries (text + 32 audio channels), all zero.
const N_DELAYS: u32 = 33;
const DELAY: u32 = 0;

/// Outcome of a Kyutai STT conversion.
#[derive(Debug, Default)]
pub(crate) struct KyutaiSttReport {
    /// Float tensors written verbatim.
    pub(crate) written: usize,
    /// Non-F32 / F16 tensors skipped (the safetensors reader accepts BF16;
    /// this converter's pass-through arm is F32 / F16 only — a real
    /// upstream all-BF16 checkpoint falls here and the loud
    /// "no float tensors" note fires; the streaming BF16 path is a
    /// follow-up wave, T29-equivalent).
    pub(crate) skipped_non_float: usize,
    /// Operator-facing diagnostics (never fail the conversion — the
    /// runtime is the authoritative gate, FR-EX-08).
    pub(crate) notes: Vec<String>,
}

/// Converts a Kyutai STT safetensors buffer into a populated GGUF builder.
///
/// Every F32 / F16 tensor passes through under its upstream name; the
/// `vokra.kyutai_stt.*` chunk group is written from the transcribed
/// constants above; provenance stamps mark the weight as
/// `AttributionRequired` (CC-BY 4.0) and the FR-MD-09 attribution surface
/// activates.
pub(crate) fn convert(bytes: Vec<u8>) -> Result<(GgufBuilder, KyutaiSttReport), ConvertError> {
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    write_hparams(&mut b);
    vokra_core::stamp_provenance(
        &mut b,
        LicenseClass::AttributionRequired,
        "CC-BY-4.0",
        Some("kyutai/stt-2.6b-en"),
        Some("https://huggingface.co/kyutai/stt-2.6b-en"),
    );
    vokra_core::stamp_attribution(&mut b, KYUTAI_STT_ATTRIBUTION_TEXT);

    let mut report = KyutaiSttReport::default();
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
             Kyutai STT-2.6B-EN release ships BF16 safetensors (~5.2 GB); the \
             streaming-BF16 pass-through path is a follow-up wave (T29-equivalent, \
             the Moshi pattern). Until then, run a prepare-checkpoint script that \
             widens BF16 → F32 or converts a F16-widened copy."
                .into(),
        );
    }
    Ok((b, report))
}

/// Writes the `vokra.kyutai_stt.*` chunk group from the transcribed
/// constants above (primary source: `config.json`). Booleans ride as u32
/// 0/1 for GGUF portability (the Zonos / CSM convention). Delays ride as
/// count + N indexed keys (the Moshi / mimi pattern for array metadata).
fn write_hparams(b: &mut GgufBuilder) {
    b.add_u32(KEY_SAMPLE_RATE, KYUTAI_STT_SAMPLE_RATE);

    // Backbone
    b.add_u32(KEY_BB_N_LAYER, BB_N_LAYER);
    b.add_u32(KEY_BB_D_MODEL, BB_D_MODEL);
    b.add_u32(KEY_BB_N_HEAD, BB_N_HEAD);
    b.add_f32(KEY_BB_HIDDEN_SCALE, BB_HIDDEN_SCALE);
    b.add_u32(KEY_BB_FFN_HIDDEN, BB_FFN_HIDDEN);
    b.add_u32(KEY_BB_CONTEXT, BB_CONTEXT);
    b.add_f32(KEY_BB_ROPE_MAX_PERIOD, BB_ROPE_MAX_PERIOD);
    b.add_u32(KEY_BB_CAUSAL, u32::from(BB_CAUSAL));
    b.add_f32(KEY_BB_RMS_NORM_EPS, BB_RMS_NORM_EPS);

    // Depformer (structurally present, unused when dep_q=0)
    b.add_u32(KEY_DEP_N_LAYER, DEP_N_LAYER);
    b.add_u32(KEY_DEP_D_MODEL, DEP_D_MODEL);
    b.add_u32(KEY_DEP_N_HEAD, DEP_N_HEAD);
    b.add_u32(KEY_DEP_MULTI_LINEAR, u32::from(DEP_MULTI_LINEAR));
    b.add_u32(KEY_DEP_WEIGHTS_PER_STEP, u32::from(DEP_WEIGHTS_PER_STEP));

    // Audio / text / streaming
    b.add_u32(KEY_N_Q, N_Q);
    b.add_u32(KEY_DEP_Q, DEP_Q);
    b.add_u32(KEY_AUDIO_CARD, AUDIO_CARD);
    b.add_u32(KEY_TEXT_CARD, TEXT_CARD);
    b.add_u32(KEY_TEXT_PAD_ID, TEXT_PAD_ID);
    b.add_f32(KEY_AUDIO_DELAY_SECS, AUDIO_DELAY_SECS);
    b.add_f32(KEY_AUDIO_SILENCE_PREFIX_SECS, AUDIO_SILENCE_PREFIX_SECS);

    // Delays — count + N indexed entries (all zero for STT).
    b.add_u32(KEY_N_DELAYS, N_DELAYS);
    for i in 0..(N_DELAYS as usize) {
        b.add_u32(&format!("{PREFIX_DELAY}{i}"), DELAY);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::{GgufFile, GgufMetadataValue};

    fn minimal_safetensors_one_f32() -> Vec<u8> {
        // A single f32 tensor at the top of the file so `convert` has
        // something to pass through and the report counts a non-zero
        // write.
        let header =
            r#"{"backbone.embed.0.weight":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}}"#;
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

    /// A single F16 tensor at the top of the file (shape [2,3] → 6
    /// elements × 2 bytes = 12 bytes).
    fn minimal_safetensors_one_f16() -> Vec<u8> {
        let header =
            r#"{"backbone.embed.0.weight":{"dtype":"F16","shape":[2,3],"data_offsets":[0,12]}}"#;
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(&[0u8; 12]);
        out
    }

    /// A single BF16 tensor at the top of the file (shape [2,3] → 6
    /// elements × 2 bytes = 12 bytes). The upstream Kyutai release is
    /// **all BF16**, so BF16 tensors reach `convert()` and MUST be
    /// counted, not silently dropped — the runtime never sees them until
    /// the streaming-BF16 follow-up lands.
    fn minimal_safetensors_one_bf16() -> Vec<u8> {
        let header =
            r#"{"backbone.embed.0.weight":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,12]}}"#;
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(&[0u8; 12]);
        out
    }

    #[test]
    fn arch_string_matches_runtime_constant() {
        // The two crates only share `vokra-core`, so this constant is the
        // sole handshake with `vokra-models::kyutai_stt::EXPECTED_ARCH`.
        assert_eq!(ARCH, "kyutai-stt");
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
            (KEY_SAMPLE_RATE, KYUTAI_STT_SAMPLE_RATE),
            (KEY_BB_N_LAYER, BB_N_LAYER),
            (KEY_BB_D_MODEL, BB_D_MODEL),
            (KEY_BB_N_HEAD, BB_N_HEAD),
            (KEY_BB_FFN_HIDDEN, BB_FFN_HIDDEN),
            (KEY_BB_CONTEXT, BB_CONTEXT),
            (KEY_BB_CAUSAL, u32::from(BB_CAUSAL)),
            (KEY_DEP_N_LAYER, DEP_N_LAYER),
            (KEY_DEP_D_MODEL, DEP_D_MODEL),
            (KEY_DEP_N_HEAD, DEP_N_HEAD),
            (KEY_DEP_MULTI_LINEAR, u32::from(DEP_MULTI_LINEAR)),
            (KEY_DEP_WEIGHTS_PER_STEP, u32::from(DEP_WEIGHTS_PER_STEP)),
            (KEY_N_Q, N_Q),
            (KEY_DEP_Q, DEP_Q),
            (KEY_AUDIO_CARD, AUDIO_CARD),
            (KEY_TEXT_CARD, TEXT_CARD),
            (KEY_TEXT_PAD_ID, TEXT_PAD_ID),
            (KEY_N_DELAYS, N_DELAYS),
        ] {
            match file.get(key) {
                Some(GgufMetadataValue::U32(v)) => assert_eq!(*v, want, "{key}"),
                other => panic!("{key}: unexpected {other:?}"),
            }
        }
        // F32 hparams.
        for (key, want) in [
            (KEY_BB_HIDDEN_SCALE, BB_HIDDEN_SCALE),
            (KEY_BB_ROPE_MAX_PERIOD, BB_ROPE_MAX_PERIOD),
            (KEY_BB_RMS_NORM_EPS, BB_RMS_NORM_EPS),
            (KEY_AUDIO_DELAY_SECS, AUDIO_DELAY_SECS),
            (KEY_AUDIO_SILENCE_PREFIX_SECS, AUDIO_SILENCE_PREFIX_SECS),
        ] {
            match file.get(key) {
                Some(GgufMetadataValue::F32(v)) => assert_eq!(*v, want, "{key}"),
                other => panic!("{key}: unexpected {other:?}"),
            }
        }
        // Delay indexed keys — all zero for STT.
        for i in 0..(N_DELAYS as usize) {
            let k = format!("{PREFIX_DELAY}{i}");
            match file.get(&k) {
                Some(GgufMetadataValue::U32(v)) => assert_eq!(*v, DELAY, "{k}"),
                other => panic!("{k}: unexpected {other:?}"),
            }
        }
        // Provenance: CC-BY 4.0 attribution-required.
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_MODEL_ID)
                .and_then(|v| v.as_str()),
            Some("kyutai/stt-2.6b-en")
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|v| v.as_str()),
            Some("CC-BY-4.0")
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::AttributionRequired.as_str())
        );
        // Attribution text is non-empty and Kyutai-named.
        let attr = file
            .get(chunks::KEY_PROVENANCE_ATTRIBUTION)
            .and_then(|v| v.as_str())
            .expect("attribution present");
        assert!(
            attr.contains("Kyutai") && attr.contains("CC-BY 4.0"),
            "attribution names Kyutai + CC-BY 4.0: {attr}"
        );
    }

    #[test]
    fn zero_tensor_conversion_surfaces_a_loud_note() {
        let (_, report) = convert(minimal_safetensors_no_tensors()).expect("convert");
        assert_eq!(report.written, 0);
        assert!(
            report.notes.iter().any(|n| n.contains("no float tensors")),
            "zero-tensor conversion must emit a loud note: {:?}",
            report.notes
        );
    }

    /// F16 tensor passes through the union match arm.
    #[test]
    fn f16_tensor_passes_through_verbatim() {
        let (builder, report) = convert(minimal_safetensors_one_f16()).expect("convert");
        assert_eq!(report.written, 1);
        assert_eq!(report.skipped_non_float, 0);

        let out = builder.to_bytes().expect("serialize");
        let file = GgufFile::parse(out).expect("parse");
        let info = file
            .tensor_info("backbone.embed.0.weight")
            .expect("tensor present");
        assert_eq!(info.dtype, GgmlType::F16);
        assert_eq!(info.dimensions, vec![2, 3]);
        assert_eq!(file.tensor_bytes(info).len(), 12);
    }

    /// The upstream Kyutai STT release is **all BF16**; BF16 tensors
    /// currently reach the `_ =>` arm and MUST be counted, not silently
    /// widened. This test guards against a regression where somebody
    /// promotes BF16 into the pass-through arm without deciding how to
    /// stream them (bounded memory — the Moshi T22 pattern).
    #[test]
    fn bf16_tensor_is_counted_as_skipped_non_float() {
        let (builder, report) = convert(minimal_safetensors_one_bf16()).expect("convert");
        assert_eq!(
            report.written, 0,
            "BF16 must not currently pass through — the streaming path is a follow-up"
        );
        assert_eq!(
            report.skipped_non_float, 1,
            "BF16 must increment the skipped counter"
        );
        // With zero float tensors written, the loud "no float tensors"
        // note fires — a BF16-only checkpoint (the real Kyutai release)
        // surfaces the situation.
        assert!(
            report.notes.iter().any(|n| n.contains("no float tensors")),
            "BF16-only conversion must emit the zero-float note: {:?}",
            report.notes
        );
        // Metadata (arch / hparams / provenance) still lands — the report
        // reflects the tensor pass, not a failure of the conversion.
        let out = builder.to_bytes().expect("serialize");
        let file = GgufFile::parse(out).expect("parse");
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH)
        );
        assert!(
            file.tensor_info("backbone.embed.0.weight").is_none(),
            "BF16 tensor must not be written"
        );
    }

    /// Pins `SafetensorsFile::parse(bytes)?` error propagation. A
    /// malformed input surfaces as `Err(ConvertError::Parse(_))`, not a
    /// silently-empty successful conversion (FR-EX-08 loud fail).
    #[test]
    fn malformed_input_returns_parse_error() {
        // Case 1: empty buffer.
        let err = convert(Vec::new()).expect_err("empty buffer must be rejected");
        assert!(
            matches!(err, ConvertError::Parse(_)),
            "expected ConvertError::Parse, got {err:?}"
        );

        // Case 2: declared header length runs off the end of the buffer.
        let mut truncated = Vec::new();
        truncated.extend_from_slice(&1024u64.to_le_bytes());
        truncated.extend_from_slice(b"{}");
        let err = convert(truncated).expect_err("truncated header must be rejected");
        assert!(
            matches!(err, ConvertError::Parse(_)),
            "expected ConvertError::Parse, got {err:?}"
        );

        // Case 3: valid length prefix but malformed JSON body.
        let bad_json = b"{not-json";
        let mut bad = Vec::new();
        bad.extend_from_slice(&(bad_json.len() as u64).to_le_bytes());
        bad.extend_from_slice(bad_json);
        let err = convert(bad).expect_err("malformed JSON must be rejected");
        assert!(
            matches!(err, ConvertError::Parse(_)),
            "expected ConvertError::Parse, got {err:?}"
        );
    }

    /// FFN hidden width is derived correctly for the 2.6B model —
    /// `round(4.125 * 2048) = 8448` — and stored in the GGUF for the
    /// runtime `from_gguf` reader to honour without re-derivation.
    #[test]
    fn ffn_hidden_derivation_matches_2_6b_shape() {
        let derived = (BB_HIDDEN_SCALE * BB_D_MODEL as f32).round() as u32;
        assert_eq!(derived, BB_FFN_HIDDEN);
        assert_eq!(BB_FFN_HIDDEN, 8448);
    }
}
