//! NVIDIA **Parakeet-CTC-1.1B**: safetensors checkpoint → GGUF
//! conversion (SoTA plan Phase 2, 2026-07-24).
//!
//! Input: an upstream `nvidia/parakeet-ctc-1.1b` safetensors checkpoint.
//! The reference HF-transformers release ships raw safetensors directly
//! (single-file or sharded); no `.pth` prepare step is required (matches
//! the Parakeet-TDT-0.6B-v3 posture). Output: a GGUF carrying every F32 /
//! F16 / BF16 tensor verbatim plus the `vokra.parakeet_ctc.*` /
//! `vokra.provenance.*` metadata chunks.
//!
//! # What is transcribed vs. shape-driven
//!
//! - **Transcribed constants** — every hparam of the `vokra.parakeet_ctc.*`
//!   chunk group is transcribed **verbatim** from the upstream
//!   `config.json` (see the top of this module for the full table). No
//!   axis is invented; any tensor whose shape disagrees with these values
//!   in a real conversion fails the runtime shape gate loudly (FR-EX-08,
//!   `ParakeetCtcConfig::validate_for_forward`).
//! - **Shape-driven** — every tensor's dtype + shape is preserved
//!   verbatim from the safetensors header; the converter never widens or
//!   quantises Parakeet-CTC weights (the M2-08 quant policy path is
//!   whisper-only — parakeet-ctc reaches the whisper-only refusal in the
//!   CLI arm, matching Parakeet-TDT / Dia / Zonos / Kyutai STT).
//!
//! # Tensor naming contract
//!
//! GGUF tensor names are the upstream safetensors names **verbatim** (the
//! CSM / Kokoro / CosyVoice2 / Dia / Zonos / Kyutai STT / Parakeet-TDT
//! contract). Real-weight binding is a follow-up wave gated on the
//! upstream tensor-name manifest fetch; this converter passes every F32 /
//! F16 / BF16 tensor through unchanged so a future
//! `ParakeetCtcWeights::from_gguf` can walk the same names.
//!
//! # BF16 posture
//!
//! The reference release advertises `dtype: "bfloat16"` in `config.json`
//! (unlike Parakeet-TDT-0.6B-v3 which is F32). Per the qwen3-tts ADR
//! (`docs/adr/qwen3-tts-bf16.md`, strategy `A_passthrough`, Accepted
//! 2026-07-25), BF16 tensors pass through **verbatim** as GGUF type 30
//! (`GgmlType::BF16`) — no convert-time widening. The runtime widens
//! BF16 → f32 losslessly at load via the single choke point
//! `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16` (BF16 = top 16
//! bits of an f32 — `bits << 16` is exact). The observability counter
//! [`ParakeetCtcReport::bf16_passthrough`] records how many BF16 tensors
//! landed on this arm — additive symmetric rewrite of the earlier
//! `bf16_tensor_is_counted_as_skipped_non_float` posture pin so a latent
//! silent-widen cannot slip in undetected.
//!
//! # No ONNX (permanent)
//!
//! Parakeet-CTC ships as safetensors / a Python pipeline; the pipeline is
//! re-implemented natively in `vokra-models/src/parakeet_ctc/`
//! (whisper.cpp 型, CLAUDE.md 設計判断 4). This converter never touches
//! ONNX.

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` for Parakeet-CTC GGUFs — kept in sync with the
/// runtime constant `vokra-models::parakeet_ctc::EXPECTED_ARCH`.
pub(crate) const ARCH: &str = "parakeet-ctc";
/// `vokra.model.name` for Parakeet-CTC GGUFs (canonical model id).
pub(crate) const NAME: &str = "parakeet-ctc-1.1b";

/// The FR-MD-09 attribution text stamped into
/// `vokra.provenance.attribution` — wording aligned with `NOTICE` and
/// the `docs/license-audit.md` NVIDIA Parakeet row (final legal
/// sufficiency = T29-equivalent owner sign-off; this converter records
/// the attribution but the owner-facing publish gate can add / edit
/// before release).
pub(crate) const PARAKEET_CTC_ATTRIBUTION_TEXT: &str = "This application uses NVIDIA Parakeet-CTC-1.1B \
     (English ASR — FastConformer encoder + CTC decoder). Model weights \
     are licensed under CC-BY 4.0 (attribution required; commercial use \
     permitted). Copyright (c) NVIDIA. Source: \
     https://huggingface.co/nvidia/parakeet-ctc-1.1b";

// --- vokra.parakeet_ctc.* keys (kept as constants in the converter; the
// runtime duplicates the strings when it lands `ParakeetCtcConfig::from_gguf`
// — the cross-crate pattern established by CSM / CosyVoice2 / Kokoro /
// Dia / Zonos / Kyutai STT / Parakeet-TDT) --------------------------------

const KEY_SAMPLE_RATE: &str = "vokra.parakeet_ctc.sample_rate";

// Encoder (FastConformer)
const KEY_ENC_N_LAYER: &str = "vokra.parakeet_ctc.arch.encoder.n_layer";
const KEY_ENC_D_MODEL: &str = "vokra.parakeet_ctc.arch.encoder.d_model";
const KEY_ENC_N_HEAD: &str = "vokra.parakeet_ctc.arch.encoder.n_head";
const KEY_ENC_N_HEAD_KV: &str = "vokra.parakeet_ctc.arch.encoder.n_head_kv";
const KEY_ENC_FFN_DIM: &str = "vokra.parakeet_ctc.arch.encoder.ffn_dim";
const KEY_ENC_CONV_KERNEL: &str = "vokra.parakeet_ctc.arch.encoder.conv_kernel_size";
const KEY_ENC_IN_DIM: &str = "vokra.parakeet_ctc.arch.encoder.in_dim";
const KEY_ENC_SUBSAMPLING_FACTOR: &str = "vokra.parakeet_ctc.arch.encoder.subsampling_factor";
const KEY_ENC_SUB_CONV_KERNEL: &str =
    "vokra.parakeet_ctc.arch.encoder.subsampling_conv_kernel_size";
const KEY_ENC_SUB_CONV_STRIDE: &str = "vokra.parakeet_ctc.arch.encoder.subsampling_conv_stride";
const KEY_ENC_SUB_CONV_CHANNELS: &str = "vokra.parakeet_ctc.arch.encoder.subsampling_conv_channels";
const KEY_ENC_MAX_POS: &str = "vokra.parakeet_ctc.arch.encoder.max_position_embeddings";
const KEY_ENC_ATTN_BIAS: &str = "vokra.parakeet_ctc.arch.encoder.attention_bias";
const KEY_ENC_CONV_BIAS: &str = "vokra.parakeet_ctc.arch.encoder.convolution_bias";
const KEY_ENC_SCALE_INPUT: &str = "vokra.parakeet_ctc.arch.encoder.scale_input";

// CTC head + vocab
const KEY_HEAD_VOCAB_SIZE: &str = "vokra.parakeet_ctc.head.vocab_size";
const KEY_HEAD_PAD_ID: &str = "vokra.parakeet_ctc.head.pad_token_id";

// --- Transcribed constants (primary source: config.json fetched verbatim) --
//
// `huggingface.co/nvidia/parakeet-ctc-1.1b/raw/main/config.json`
// (fetched 2026-07-24). Every value here is transcribed verbatim;
// nothing is invented.

// PCM sample rate — not written in config.json; taken from the model
// card ("16 kHz mono .wav / .flac").
const PARAKEET_CTC_SAMPLE_RATE: u32 = 16_000;

// Encoder (config.json:encoder_config)
const ENC_N_LAYER: u32 = 42; // "num_hidden_layers": 42
const ENC_D_MODEL: u32 = 1024; // "hidden_size": 1024
const ENC_N_HEAD: u32 = 8; // "num_attention_heads": 8
const ENC_N_HEAD_KV: u32 = 8; // "num_key_value_heads": 8 (MHA)
const ENC_FFN_DIM: u32 = 4096; // "intermediate_size": 4096
const ENC_CONV_KERNEL: u32 = 9; // "conv_kernel_size": 9
const ENC_IN_DIM: u32 = 80; // "num_mel_bins": 80 (differs from TDT-0.6B-v3 = 128)
const ENC_SUBSAMPLING_FACTOR: u32 = 8; // "subsampling_factor": 8
const ENC_SUB_CONV_KERNEL: u32 = 3; // "subsampling_conv_kernel_size": 3
const ENC_SUB_CONV_STRIDE: u32 = 2; // "subsampling_conv_stride": 2
const ENC_SUB_CONV_CHANNELS: u32 = 256; // "subsampling_conv_channels": 256
const ENC_MAX_POS: u32 = 5000; // "max_position_embeddings": 5000
const ENC_ATTN_BIAS: bool = true; // "attention_bias": true (differs from TDT-0.6B-v3 = false)
// `convolution_bias` is not listed in the CTC-1.1B encoder_config; it
// inherits the NeMo `ConformerLayer` default of false (bias-free depthwise
// / point-wise convolutions on the inference path).
const ENC_CONV_BIAS: bool = false;
const ENC_SCALE_INPUT: bool = true; // "scale_input": true (differs from TDT-0.6B-v3 = false)

// CTC head (config.json:top-level)
const HEAD_VOCAB_SIZE: u32 = 1025; // "vocab_size": 1025 (1024 SentencePiece + 1 blank)
const HEAD_PAD_ID: u32 = 1024; // "pad_token_id": 1024 (= CTC blank)

/// Outcome of a Parakeet-CTC conversion.
#[derive(Debug, Default)]
pub(crate) struct ParakeetCtcReport {
    /// Float tensors written verbatim (F32 / F16 / BF16 — all three go
    /// through the same byte-copy path since the BF16 pass-through land
    /// 2026-07-25, mirror of `qwen3-tts` / `vibevoice` / `voxcpm2` /
    /// `moshi` / `voxtral`).
    pub(crate) written: usize,
    /// Non-F32 / F16 / BF16 tensors skipped (defensive counter — the
    /// safetensors reader rejects unknown dtypes at parse time; anything
    /// that reaches this arm is a quantised dtype the runtime is not
    /// expected to consume).
    pub(crate) skipped_non_float: usize,
    /// Of the tensors in `written`, how many were BF16 (subset counter).
    /// Emits GGUF type 30 verbatim; runtime widens BF16 → f32 losslessly
    /// via the single choke point `crates/vokra-core/src/gguf/quant/mod.rs
    /// decode_bf16` (BF16 = top 16 bits of an f32 — `bits << 16` is
    /// exact). Additive observability counter — mirrors
    /// `qwen3_tts::Qwen3TtsReport::bf16_passthrough` /
    /// `vibevoice::VibeVoiceReport::bf16_passthrough` /
    /// `moshi::MoshiReport::bf16_passthrough` so a latent silent-widen
    /// cannot slip in undetected.
    pub(crate) bf16_passthrough: usize,
    /// Operator-facing diagnostics (never fail the conversion — the
    /// runtime is the authoritative gate, FR-EX-08).
    pub(crate) notes: Vec<String>,
}

/// Converts a Parakeet-CTC safetensors buffer into a populated GGUF
/// builder.
///
/// Every F32 / F16 / BF16 tensor passes through under its upstream
/// name; the `vokra.parakeet_ctc.*` chunk group is written from the
/// transcribed constants above; provenance stamps mark the weight as
/// `AttributionRequired` (CC-BY 4.0) and the FR-MD-09 attribution
/// surface activates.
pub(crate) fn convert(bytes: Vec<u8>) -> Result<(GgufBuilder, ParakeetCtcReport), ConvertError> {
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    write_hparams(&mut b);
    vokra_core::stamp_provenance(
        &mut b,
        LicenseClass::AttributionRequired,
        "CC-BY-4.0",
        Some("nvidia/parakeet-ctc-1.1b"),
        Some("https://huggingface.co/nvidia/parakeet-ctc-1.1b"),
    );
    vokra_core::stamp_attribution(&mut b, PARAKEET_CTC_ATTRIBUTION_TEXT);

    let mut report = ParakeetCtcReport::default();
    for t in st.tensors() {
        match t.dtype {
            // BF16 pass-through added 2026-07-25 (mirror of qwen3-tts +
            // vibevoice + voxcpm2 + moshi + voxtral): upstream
            // Parakeet-CTC-1.1B ships `dtype: "bfloat16"` so the release
            // checkpoint hits this arm. Emit as GGUF type 30 verbatim;
            // runtime widens on load via `decode_bf16` (exact,
            // `bits << 16`).
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
             Parakeet-CTC-1.1B release ships BF16 safetensors (per config.json \
             `dtype: \"bfloat16\"`); the BF16 pass-through path is now wired \
             (2026-07-25), so this state is only reachable when the release \
             contains no F32 / F16 / BF16 float tensors at all."
                .to_owned(),
        );
    }
    Ok((b, report))
}

/// Writes the `vokra.parakeet_ctc.*` chunk group from the transcribed
/// constants above (primary source: `config.json`). Booleans ride as
/// u32 0/1 for GGUF portability (the Zonos / CSM / Kyutai STT / Parakeet-
/// TDT convention).
fn write_hparams(b: &mut GgufBuilder) {
    b.add_u32(KEY_SAMPLE_RATE, PARAKEET_CTC_SAMPLE_RATE);

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

    // CTC head
    b.add_u32(KEY_HEAD_VOCAB_SIZE, HEAD_VOCAB_SIZE);
    b.add_u32(KEY_HEAD_PAD_ID, HEAD_PAD_ID);
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
        // the sole handshake with `vokra-models::parakeet_ctc::EXPECTED_ARCH`.
        assert_eq!(ARCH, "parakeet-ctc");
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
            (KEY_SAMPLE_RATE, PARAKEET_CTC_SAMPLE_RATE),
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
            (KEY_HEAD_VOCAB_SIZE, HEAD_VOCAB_SIZE),
            (KEY_HEAD_PAD_ID, HEAD_PAD_ID),
        ] {
            match file.get(key) {
                Some(GgufMetadataValue::U32(v)) => assert_eq!(*v, want, "{key}"),
                other => panic!("{key}: unexpected {other:?}"),
            }
        }

        // Provenance: CC-BY 4.0 attribution-required.
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_MODEL_ID)
                .and_then(|v| v.as_str()),
            Some("nvidia/parakeet-ctc-1.1b")
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

    /// Pins the BF16 leg of the `GgmlType::F32 | GgmlType::F16 |
    /// GgmlType::BF16` union: BF16 (the upstream serving format for
    /// Parakeet-CTC-1.1B, `config.json` `dtype: "bfloat16"`) must reach
    /// the pass-through arm, emit as GGUF type 30 verbatim, and
    /// increment `bf16_passthrough`. Mirror of qwen3-tts /
    /// `bf16_tensor_passes_through_verbatim` and moshi's `assert_eq!(
    /// info.dtype, GgmlType::BF16, "no convert-time widening")` at
    /// `crates/vokra-core/src/safetensors.rs:728-738`.
    ///
    /// Rewritten 2026-07-25 from the earlier "counted as skipped" pin
    /// (`bf16_tensor_is_counted_as_skipped_non_float`) — the earlier
    /// pin encoded the pre-BF16-fix scaffold posture. Removing the pin
    /// outright would let a latent silent-widen slip in undetected;
    /// rewriting to the passes-through invariant keeps the regression
    /// guard.
    ///
    /// Uses non-zero BF16 bit patterns so byte-identity is a real check
    /// (an all-zero payload would round-trip trivially through a silent
    /// F32/F16 widen too).
    #[test]
    fn bf16_tensor_passes_through_verbatim() {
        // Build a BF16 payload with known non-zero bit patterns so a
        // byte-identity assert catches any silent widen / downcast
        // attempt. BF16 = top 16 bits of an f32 — `((bits >> 16) as u16)`.
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
            "BF16 must not land in the skipped counter"
        );
        assert_eq!(
            report.bf16_passthrough, 1,
            "BF16 subset counter must record the pass-through"
        );
        // Loud-silence check for FR-EX-08: the zero-float note is a
        // false-positive here because BF16 IS a float.
        assert!(
            !report.notes.iter().any(|n| n.contains("no float tensors")),
            "BF16 pass-through must not emit the zero-float note: {:?}",
            report.notes
        );

        // Round-trip through the GGUF: dtype preserved, payload
        // byte-identical (moshi's `assert_eq!(info.dtype, GgmlType::BF16,
        // "no convert-time widening")` posture).
        let out = builder.to_bytes().expect("serialize");
        let file = GgufFile::parse(out).expect("parse");
        let info = file
            .tensor_info("encoder.blocks.0.attn.qkv_proj.weight")
            .expect("BF16 tensor must be present after pass-through");
        assert_eq!(
            info.dtype,
            GgmlType::BF16,
            "no convert-time widening — GGUF dtype must remain BF16"
        );
        assert_eq!(info.dimensions, vec![2, 3]);
        assert_eq!(
            file.tensor_bytes(info).len(),
            12,
            "BF16 payload = 6 elements × 2 bytes = 12 bytes"
        );
        assert_eq!(
            file.tensor_bytes(info),
            bf16.as_slice(),
            "BF16 payload must be byte-identical to input (no silent widen)"
        );
        // Metadata (arch chunk) still lands — the report reflects the
        // tensor pass, not a failure of the conversion.
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH)
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

    /// Guards the axes that distinguish CTC-1.1B from TDT-0.6B-v3:
    /// `num_mel_bins = 80` (not 128), `attention_bias = true` (not
    /// false), `scale_input = true` (not false), `num_hidden_layers =
    /// 42` (not 24). A regression here would silently ship a converter
    /// that emits the wrong hparam chunk group.
    ///
    /// Compares against `u32::from(bool)` so the const-bool guards do
    /// not trip clippy's `assertions_on_constants`.
    #[test]
    fn ctc_1_1b_differs_from_tdt_0_6b_v3_on_key_axes() {
        assert_eq!(ENC_N_LAYER, 42, "CTC-1.1B: 42 layers (TDT-0.6B-v3: 24)");
        assert_eq!(ENC_IN_DIM, 80, "CTC-1.1B: 80 mel bins (TDT-0.6B-v3: 128)");
        assert_eq!(
            u32::from(ENC_ATTN_BIAS),
            1,
            "CTC-1.1B: attention_bias=true (TDT-0.6B-v3: false)"
        );
        assert_eq!(
            u32::from(ENC_SCALE_INPUT),
            1,
            "CTC-1.1B: scale_input=true (TDT-0.6B-v3: false)"
        );
        assert_eq!(
            u32::from(ENC_CONV_BIAS),
            0,
            "CTC-1.1B: convolution_bias=false (inherits NeMo default)"
        );
        // Vocab is much smaller than TDT (1025 vs 8193) — CTC does not
        // need the joint / duration head.
        assert_eq!(
            HEAD_VOCAB_SIZE, 1025,
            "CTC-1.1B: vocab_size=1025 (TDT-0.6B-v3: 8193)"
        );
        assert_eq!(
            HEAD_PAD_ID, 1024,
            "CTC-1.1B: pad_token_id=1024 = CTC blank (TDT-0.6B-v3: 2 pad + 8192 blank)"
        );
    }
}
