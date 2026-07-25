//! NVIDIA **Parakeet-TDT-0.6B-v3**: safetensors checkpoint → GGUF
//! conversion (SoTA plan Phase 2, 2026-07-24).
//!
//! Input: an upstream `nvidia/parakeet-tdt-0.6b-v3` safetensors
//! checkpoint. The reference HF-transformers release ships raw
//! safetensors directly (single-file or sharded); no `.pth` prepare step
//! is required (unlike Dia — matches Zonos posture). Output: a GGUF
//! carrying every F32 / F16 tensor verbatim plus the `vokra.parakeet.*`
//! / `vokra.provenance.*` metadata chunks.
//!
//! # What is transcribed vs. shape-driven
//!
//! - **Transcribed constants** — every hparam of the `vokra.parakeet.*`
//!   chunk group is transcribed **verbatim** from the upstream
//!   `config.json` (see the top of this module for the full table). No
//!   axis is invented; any tensor whose shape disagrees with these values
//!   in a real conversion fails the runtime shape gate loudly (FR-EX-08,
//!   `ParakeetConfig::validate_for_forward`).
//! - **Shape-driven** — every tensor's dtype + shape is preserved
//!   verbatim from the safetensors header; the converter never widens or
//!   quantises Parakeet weights (the M2-08 quant policy path is
//!   whisper-only — parakeet reaches the whisper-only refusal in the CLI
//!   arm, matching Dia / Zonos / Kyutai STT).
//!
//! # Tensor naming contract
//!
//! GGUF tensor names are the upstream safetensors names **verbatim** (the
//! CSM / Kokoro / CosyVoice2 / Dia / Zonos / Kyutai STT contract).
//! Real-weight binding is a follow-up wave gated on the upstream
//! tensor-name manifest fetch; this converter passes every F32 / F16
//! tensor through unchanged so a future `ParakeetWeights::from_gguf` can
//! walk the same names.
//!
//! # BF16 posture
//!
//! The reference release advertises `dtype: "float32"` in `config.json`;
//! community-converted BF16 shards (a common size-halving posture) also
//! land through this converter now. BF16 tensors pass through
//! **verbatim** as GGUF type 30 (`GgmlType::BF16`) with no convert-time
//! widening — the runtime widens BF16 → f32 losslessly at load via the
//! single choke point `crates/vokra-core/src/gguf/quant/mod.rs
//! decode_bf16` (BF16 is the top 16 bits of an f32 — `bits << 16` is
//! exact). Mirrors the qwen3-tts / vibevoice / voxcpm2 / moshi /
//! voxtral pass-through pattern (2026-07-25). The
//! [`ParakeetReport::bf16_passthrough`] observability counter records
//! how many BF16 tensors landed on the pass-through arm.
//!
//! # No ONNX (permanent)
//!
//! Parakeet ships as safetensors / a Python pipeline; the pipeline is
//! re-implemented natively in `vokra-models/src/parakeet/` (whisper.cpp
//! 型, CLAUDE.md 設計判断 4). This converter never touches ONNX.

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` for Parakeet GGUFs — kept in sync with the runtime
/// constant `vokra-models::parakeet::EXPECTED_ARCH`.
pub(crate) const ARCH: &str = "parakeet-tdt";
/// `vokra.model.name` for Parakeet GGUFs (canonical model id).
pub(crate) const NAME: &str = "parakeet-tdt-0.6b-v3";

/// The FR-MD-09 attribution text stamped into
/// `vokra.provenance.attribution` — wording aligned with `NOTICE` and
/// the `docs/license-audit.md` NVIDIA Parakeet row (final legal
/// sufficiency = T29-equivalent owner sign-off; this converter records
/// the attribution but the owner-facing publish gate can add / edit
/// before release).
pub(crate) const PARAKEET_ATTRIBUTION_TEXT: &str = "This application uses NVIDIA Parakeet-TDT-0.6B-v3 \
     (English streaming ASR — FastConformer encoder + TDT decoder). Model \
     weights are licensed under CC-BY 4.0 (attribution required; commercial \
     use permitted). Copyright (c) NVIDIA. Source: \
     https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3";

// --- vokra.parakeet.* keys (kept as constants in the converter; the
// runtime duplicates the strings when it lands `ParakeetConfig::from_gguf`
// — the cross-crate pattern established by CSM / CosyVoice2 / Kokoro /
// Dia / Zonos / Kyutai STT) ------------------------------------------------

const KEY_SAMPLE_RATE: &str = "vokra.parakeet.sample_rate";

// Encoder (FastConformer)
const KEY_ENC_N_LAYER: &str = "vokra.parakeet.arch.encoder.n_layer";
const KEY_ENC_D_MODEL: &str = "vokra.parakeet.arch.encoder.d_model";
const KEY_ENC_N_HEAD: &str = "vokra.parakeet.arch.encoder.n_head";
const KEY_ENC_N_HEAD_KV: &str = "vokra.parakeet.arch.encoder.n_head_kv";
const KEY_ENC_FFN_DIM: &str = "vokra.parakeet.arch.encoder.ffn_dim";
const KEY_ENC_CONV_KERNEL: &str = "vokra.parakeet.arch.encoder.conv_kernel_size";
const KEY_ENC_IN_DIM: &str = "vokra.parakeet.arch.encoder.in_dim";
const KEY_ENC_SUBSAMPLING_FACTOR: &str = "vokra.parakeet.arch.encoder.subsampling_factor";
const KEY_ENC_SUB_CONV_KERNEL: &str = "vokra.parakeet.arch.encoder.subsampling_conv_kernel_size";
const KEY_ENC_SUB_CONV_STRIDE: &str = "vokra.parakeet.arch.encoder.subsampling_conv_stride";
const KEY_ENC_SUB_CONV_CHANNELS: &str = "vokra.parakeet.arch.encoder.subsampling_conv_channels";
const KEY_ENC_MAX_POS: &str = "vokra.parakeet.arch.encoder.max_position_embeddings";
const KEY_ENC_ATTN_BIAS: &str = "vokra.parakeet.arch.encoder.attention_bias";
const KEY_ENC_CONV_BIAS: &str = "vokra.parakeet.arch.encoder.convolution_bias";
const KEY_ENC_SCALE_INPUT: &str = "vokra.parakeet.arch.encoder.scale_input";

// Decoder (RNN-T prediction network)
const KEY_DEC_N_LAYER: &str = "vokra.parakeet.arch.decoder.n_layer";
const KEY_DEC_D_MODEL: &str = "vokra.parakeet.arch.decoder.d_model";

// Joint / TDT head + vocab
const KEY_JOINT_VOCAB_SIZE: &str = "vokra.parakeet.joint.vocab_size";
const KEY_JOINT_BLANK_ID: &str = "vokra.parakeet.joint.blank_token_id";
const KEY_JOINT_PAD_ID: &str = "vokra.parakeet.joint.pad_token_id";
const KEY_JOINT_MAX_SYMBOLS_PER_STEP: &str = "vokra.parakeet.joint.max_symbols_per_step";
const KEY_JOINT_ACT: &str = "vokra.parakeet.joint.hidden_act";

// Duration bins (indexed keys — the CSM / Moshi / Kyutai STT pattern for
// array metadata).
const KEY_N_DURATIONS: &str = "vokra.parakeet.joint.n_durations";
const PREFIX_DURATION: &str = "vokra.parakeet.joint.duration.";

// --- Transcribed constants (primary source: config.json fetched verbatim) --
//
// `huggingface.co/nvidia/parakeet-tdt-0.6b-v3/raw/main/config.json`
// (fetched 2026-07-24). Every value here is transcribed verbatim;
// nothing is invented.

// PCM sample rate — not written in config.json; taken from the model
// card ("16 kHz mono .wav / .flac").
const PARAKEET_SAMPLE_RATE: u32 = 16_000;

// Encoder (config.json:encoder_config)
const ENC_N_LAYER: u32 = 24; // "num_hidden_layers": 24
const ENC_D_MODEL: u32 = 1024; // "hidden_size": 1024
const ENC_N_HEAD: u32 = 8; // "num_attention_heads": 8
const ENC_N_HEAD_KV: u32 = 8; // "num_key_value_heads": 8 (MHA)
const ENC_FFN_DIM: u32 = 4096; // "intermediate_size": 4096
const ENC_CONV_KERNEL: u32 = 9; // "conv_kernel_size": 9
const ENC_IN_DIM: u32 = 128; // "num_mel_bins": 128
const ENC_SUBSAMPLING_FACTOR: u32 = 8; // "subsampling_factor": 8
const ENC_SUB_CONV_KERNEL: u32 = 3; // "subsampling_conv_kernel_size": 3
const ENC_SUB_CONV_STRIDE: u32 = 2; // "subsampling_conv_stride": 2
const ENC_SUB_CONV_CHANNELS: u32 = 256; // "subsampling_conv_channels": 256
const ENC_MAX_POS: u32 = 5000; // "max_position_embeddings": 5000
const ENC_ATTN_BIAS: bool = false; // "attention_bias": false
const ENC_CONV_BIAS: bool = false; // "convolution_bias": false
const ENC_SCALE_INPUT: bool = false; // "scale_input": false

// Decoder (config.json:top-level)
const DEC_N_LAYER: u32 = 2; // "num_decoder_layers": 2
const DEC_D_MODEL: u32 = 640; // "decoder_hidden_size": 640

// Joint / TDT (config.json:top-level)
const JOINT_VOCAB_SIZE: u32 = 8193; // "vocab_size": 8193 (8192 + 1 blank)
const JOINT_BLANK_ID: u32 = 8192; // "blank_token_id": 8192
const JOINT_PAD_ID: u32 = 2; // "pad_token_id": 2
const JOINT_MAX_SYMBOLS_PER_STEP: u32 = 10; // "max_symbols_per_step": 10
const JOINT_ACT: &str = "relu"; // "hidden_act": "relu" (top-level, joint post-activation)

// TDT duration bins.
const DURATIONS: &[u32] = &[0, 1, 2, 3, 4];

/// Outcome of a Parakeet conversion.
#[derive(Debug, Default)]
pub(crate) struct ParakeetReport {
    /// Float tensors written verbatim (F32 / F16 / BF16 — the BF16 leg
    /// added 2026-07-25 to mirror qwen3-tts / vibevoice / voxcpm2 /
    /// moshi / voxtral).
    pub(crate) written: usize,
    /// Non-float tensors skipped (defensive counter — the safetensors
    /// reader accepts only `F32` / `F16` / `BF16` at parse time
    /// (`crates/vokra-core/src/safetensors.rs map_dtype`), so any
    /// tensor reaching this counter would signal a reader change
    /// upstream; kept for symmetry with the sibling converters and to
    /// surface the "no float tensors" loud note when zero writes
    /// occur).
    pub(crate) skipped_non_float: usize,
    /// Of the tensors in [`Self::written`], how many were BF16 (subset
    /// counter). Emits GGUF type 30 verbatim; runtime widens BF16 → f32
    /// losslessly via the single choke point
    /// `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16`
    /// (BF16 is the top 16 bits of an f32 — `bits << 16` is exact).
    /// Additive observability field — mirrors
    /// `Qwen3TtsReport::bf16_passthrough` /
    /// `VibeVoiceReport::bf16_passthrough` /
    /// `MoshiReport::bf16_passthrough`.
    pub(crate) bf16_passthrough: usize,
    /// Operator-facing diagnostics (never fail the conversion — the
    /// runtime is the authoritative gate, FR-EX-08).
    pub(crate) notes: Vec<String>,
}

/// Converts a Parakeet safetensors buffer into a populated GGUF builder.
///
/// Every F32 / F16 tensor passes through under its upstream name; the
/// `vokra.parakeet.*` chunk group is written from the transcribed
/// constants above; provenance stamps mark the weight as
/// `AttributionRequired` (CC-BY 4.0) and the FR-MD-09 attribution
/// surface activates.
pub(crate) fn convert(bytes: Vec<u8>) -> Result<(GgufBuilder, ParakeetReport), ConvertError> {
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    write_hparams(&mut b);
    vokra_core::stamp_provenance(
        &mut b,
        LicenseClass::AttributionRequired,
        "CC-BY-4.0",
        Some("nvidia/parakeet-tdt-0.6b-v3"),
        Some("https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3"),
    );
    vokra_core::stamp_attribution(&mut b, PARAKEET_ATTRIBUTION_TEXT);

    let mut report = ParakeetReport::default();
    // Float tensors pass through **verbatim** — no convert-time widening.
    // BF16 stays GGUF `BF16` (type 30) per the qwen3-tts / vibevoice /
    // voxcpm2 / moshi / voxtral posture (2026-07-25); the runtime
    // widens BF16 → f32 losslessly at load via the single choke point
    // `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16`.
    for t in st.tensors() {
        match t.dtype {
            GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 => {
                b.add_tensor(
                    &t.name,
                    t.dtype,
                    t.shape.clone(),
                    st.tensor_bytes(t).to_vec(),
                )?;
                report.written += 1;
                if t.dtype == GgmlType::BF16 {
                    report.bf16_passthrough += 1;
                }
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
             Parakeet-TDT-0.6B-v3 release ships F32 safetensors (per config.json \
             `dtype: \"float32\"`); the BF16 pass-through path is now wired \
             (2026-07-25 — mirror of qwen3-tts / vibevoice / voxcpm2 / moshi / \
             voxtral), so this state is only reachable when the input contains no \
             F32 / F16 / BF16 float tensors at all — check that the input path is \
             a Parakeet safetensors and not a config-only shard."
                .into(),
        );
    }
    Ok((b, report))
}

/// Writes the `vokra.parakeet.*` chunk group from the transcribed
/// constants above (primary source: `config.json`). Booleans ride as
/// u32 0/1 for GGUF portability (the Zonos / CSM / Kyutai STT
/// convention). Durations ride as count + N indexed keys (the Moshi /
/// mimi / Kyutai STT pattern for array metadata).
fn write_hparams(b: &mut GgufBuilder) {
    b.add_u32(KEY_SAMPLE_RATE, PARAKEET_SAMPLE_RATE);

    // Encoder
    b.add_u32(KEY_ENC_N_LAYER, ENC_N_LAYER);
    b.add_u32(KEY_ENC_D_MODEL, ENC_D_MODEL);
    b.add_u32(KEY_ENC_N_HEAD, ENC_N_HEAD);
    b.add_u32(KEY_ENC_N_HEAD_KV, ENC_N_HEAD_KV);
    b.add_u32(KEY_ENC_FFN_DIM, ENC_FFN_DIM);
    b.add_u32(KEY_ENC_CONV_KERNEL, ENC_CONV_KERNEL);
    b.add_u32(KEY_ENC_IN_DIM, ENC_IN_DIM);
    b.add_u32(KEY_ENC_SUBSAMPLING_FACTOR, ENC_SUBSAMPLING_FACTOR);
    b.add_u32(KEY_ENC_SUB_CONV_KERNEL, ENC_SUB_CONV_KERNEL);
    b.add_u32(KEY_ENC_SUB_CONV_STRIDE, ENC_SUB_CONV_STRIDE);
    b.add_u32(KEY_ENC_SUB_CONV_CHANNELS, ENC_SUB_CONV_CHANNELS);
    b.add_u32(KEY_ENC_MAX_POS, ENC_MAX_POS);
    b.add_u32(KEY_ENC_ATTN_BIAS, u32::from(ENC_ATTN_BIAS));
    b.add_u32(KEY_ENC_CONV_BIAS, u32::from(ENC_CONV_BIAS));
    b.add_u32(KEY_ENC_SCALE_INPUT, u32::from(ENC_SCALE_INPUT));

    // Decoder
    b.add_u32(KEY_DEC_N_LAYER, DEC_N_LAYER);
    b.add_u32(KEY_DEC_D_MODEL, DEC_D_MODEL);

    // Joint / TDT
    b.add_u32(KEY_JOINT_VOCAB_SIZE, JOINT_VOCAB_SIZE);
    b.add_u32(KEY_JOINT_BLANK_ID, JOINT_BLANK_ID);
    b.add_u32(KEY_JOINT_PAD_ID, JOINT_PAD_ID);
    b.add_u32(KEY_JOINT_MAX_SYMBOLS_PER_STEP, JOINT_MAX_SYMBOLS_PER_STEP);
    b.add_string(KEY_JOINT_ACT, JOINT_ACT);

    // Duration bins — count + N indexed entries.
    b.add_u32(KEY_N_DURATIONS, DURATIONS.len() as u32);
    for (i, d) in DURATIONS.iter().enumerate() {
        b.add_u32(&format!("{PREFIX_DURATION}{i}"), *d);
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
        let header = r#"{"encoder.blocks.0.attn.qkv_proj.weight":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}}"#;
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

    fn minimal_safetensors_one_f16() -> Vec<u8> {
        let header = r#"{"encoder.blocks.0.attn.qkv_proj.weight":{"dtype":"F16","shape":[2,3],"data_offsets":[0,12]}}"#;
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(&[0u8; 12]);
        out
    }

    #[test]
    fn arch_string_matches_runtime_constant() {
        // The two crates only share `vokra-core`, so this constant is
        // the sole handshake with `vokra-models::parakeet::EXPECTED_ARCH`.
        assert_eq!(ARCH, "parakeet-tdt");
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
            (KEY_SAMPLE_RATE, PARAKEET_SAMPLE_RATE),
            (KEY_ENC_N_LAYER, ENC_N_LAYER),
            (KEY_ENC_D_MODEL, ENC_D_MODEL),
            (KEY_ENC_N_HEAD, ENC_N_HEAD),
            (KEY_ENC_N_HEAD_KV, ENC_N_HEAD_KV),
            (KEY_ENC_FFN_DIM, ENC_FFN_DIM),
            (KEY_ENC_CONV_KERNEL, ENC_CONV_KERNEL),
            (KEY_ENC_IN_DIM, ENC_IN_DIM),
            (KEY_ENC_SUBSAMPLING_FACTOR, ENC_SUBSAMPLING_FACTOR),
            (KEY_ENC_SUB_CONV_KERNEL, ENC_SUB_CONV_KERNEL),
            (KEY_ENC_SUB_CONV_STRIDE, ENC_SUB_CONV_STRIDE),
            (KEY_ENC_SUB_CONV_CHANNELS, ENC_SUB_CONV_CHANNELS),
            (KEY_ENC_MAX_POS, ENC_MAX_POS),
            (KEY_ENC_ATTN_BIAS, u32::from(ENC_ATTN_BIAS)),
            (KEY_ENC_CONV_BIAS, u32::from(ENC_CONV_BIAS)),
            (KEY_ENC_SCALE_INPUT, u32::from(ENC_SCALE_INPUT)),
            (KEY_DEC_N_LAYER, DEC_N_LAYER),
            (KEY_DEC_D_MODEL, DEC_D_MODEL),
            (KEY_JOINT_VOCAB_SIZE, JOINT_VOCAB_SIZE),
            (KEY_JOINT_BLANK_ID, JOINT_BLANK_ID),
            (KEY_JOINT_PAD_ID, JOINT_PAD_ID),
            (KEY_JOINT_MAX_SYMBOLS_PER_STEP, JOINT_MAX_SYMBOLS_PER_STEP),
            (KEY_N_DURATIONS, DURATIONS.len() as u32),
        ] {
            match file.get(key) {
                Some(GgufMetadataValue::U32(v)) => assert_eq!(*v, want, "{key}"),
                other => panic!("{key}: unexpected {other:?}"),
            }
        }
        // Joint activation is a string.
        assert_eq!(
            file.get(KEY_JOINT_ACT).and_then(|v| v.as_str()),
            Some(JOINT_ACT)
        );
        // Duration indexed keys — [0, 1, 2, 3, 4] verbatim.
        for (i, d) in DURATIONS.iter().enumerate() {
            let k = format!("{PREFIX_DURATION}{i}");
            match file.get(&k) {
                Some(GgufMetadataValue::U32(v)) => assert_eq!(*v, *d, "{k}"),
                other => panic!("{k}: unexpected {other:?}"),
            }
        }

        // Provenance: CC-BY 4.0 attribution-required.
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_MODEL_ID)
                .and_then(|v| v.as_str()),
            Some("nvidia/parakeet-tdt-0.6b-v3")
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
        // Attribution text is non-empty and NVIDIA-named.
        let attr = file
            .get(chunks::KEY_PROVENANCE_ATTRIBUTION)
            .and_then(|v| v.as_str())
            .expect("attribution present");
        assert!(
            attr.contains("NVIDIA") && attr.contains("CC-BY 4.0"),
            "attribution names NVIDIA + CC-BY 4.0: {attr}"
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
            .tensor_info("encoder.blocks.0.attn.qkv_proj.weight")
            .expect("tensor present");
        assert_eq!(info.dtype, GgmlType::F16);
        assert_eq!(info.dimensions, vec![2, 3]);
        assert_eq!(file.tensor_bytes(info).len(), 12);
    }

    /// TDD-Red: mirror of qwen3-tts / vibevoice / voxcpm2's
    /// `bf16_tensor_passes_through_verbatim`. The upstream Parakeet
    /// 0.6B-v3 release is F32, but community-converted BF16 shards
    /// (Moshi / Voxtral posture — BF16 is the top 16 bits of an f32,
    /// runtime widens losslessly via the single choke point
    /// `vokra-core::gguf::quant::decode_bf16`) must reach the
    /// pass-through arm verbatim (GGUF type 30 = `GgmlType::BF16`, no
    /// convert-time widening). Failure of this test means either the
    /// `bf16_passthrough` observability counter is missing from
    /// `ParakeetReport` (compile-time RED) or the match arm at
    /// `parakeet.rs:194` does not yet include `GgmlType::BF16`
    /// (runtime-time RED). Non-zero BF16 bit patterns are chosen so a
    /// silent widen / downcast would break the byte-identity assert
    /// (all-zero payloads round-trip trivially through F32/F16 widen).
    #[test]
    fn bf16_tensor_passes_through_verbatim() {
        // BF16 payload with known non-zero bit patterns — a silent
        // convert-time widen to F32/F16 would zero-fill or reshape,
        // breaking the byte-identity assert below (all-zero payloads
        // would round-trip trivially through any widen path).
        let values: [f32; 6] = [1.0, -2.5, 0.15625, 3.5, -0.5, 42.0];
        let bf16: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        assert_eq!(bf16.len(), 12, "6 elements × 2 bytes BF16 payload");
        let header = r#"{"encoder.blocks.0.attn.qkv_proj.weight":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,12]}}"#;
        let mut input = Vec::new();
        input.extend_from_slice(&(header.len() as u64).to_le_bytes());
        input.extend_from_slice(header.as_bytes());
        input.extend_from_slice(&bf16);

        let (builder, report) = convert(input).expect("convert");
        assert_eq!(
            report.written, 1,
            "BF16 must reach the pass-through arm and increment `written`"
        );
        assert_eq!(
            report.skipped_non_float, 0,
            "BF16 must not land in the skipped counter after the BF16 pass-through land"
        );
        assert_eq!(
            report.bf16_passthrough, 1,
            "BF16 subset counter must record the pass-through (additive observability)"
        );

        // Round-trip through the GGUF: dtype preserved verbatim, payload
        // byte-identical to input (Moshi's `assert_eq!(info.dtype,
        // GgmlType::BF16, "no convert-time widening")` posture).
        let out = builder.to_bytes().expect("serialize");
        let file = GgufFile::parse(out).expect("parse");
        let info = file
            .tensor_info("encoder.blocks.0.attn.qkv_proj.weight")
            .expect("BF16 tensor must be present after pass-through");
        assert_eq!(
            info.dtype,
            GgmlType::BF16,
            "no convert-time widening — GGUF dtype must remain BF16 (type 30)"
        );
        assert_eq!(info.dimensions, vec![2, 3]);
        assert_eq!(
            file.tensor_bytes(info).len(),
            12,
            "BF16 payload = 6 elements × 2 bytes = 12 bytes verbatim"
        );
        assert_eq!(
            file.tensor_bytes(info),
            bf16.as_slice(),
            "BF16 payload must be byte-identical to input (no silent widen)"
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

    /// Duration bins = `[0, 1, 2, 3, 4]` exactly (5 entries) — the TDT
    /// signature. Guards against a regression where an axis is
    /// dropped / added silently.
    #[test]
    fn durations_match_primary_source_verbatim() {
        assert_eq!(DURATIONS, &[0, 1, 2, 3, 4]);
        assert_eq!(DURATIONS.len(), 5);
    }

    /// The MHA head-split (`num_attention_heads == num_key_value_heads
    /// == 8`) is verbatim — a common regression is confusing this with
    /// GQA and dividing.
    #[test]
    fn head_split_is_mha() {
        assert_eq!(ENC_N_HEAD, ENC_N_HEAD_KV);
        assert_eq!(ENC_D_MODEL % ENC_N_HEAD, 0);
        assert_eq!(ENC_D_MODEL / ENC_N_HEAD, 128); // head_dim
    }
}
