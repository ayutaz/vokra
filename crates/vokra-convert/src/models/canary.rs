//! NVIDIA **Canary-1B-v2**: safetensors checkpoint → GGUF conversion
//! (SoTA plan Phase 2, 2026-07-24).
//!
//! Input: a Canary-1B-v2 **prepared** safetensors (the upstream release
//! ships a `.nemo` tarball — `canary-1b-v2.nemo`, 6.36 GB — that a
//! prepare-checkpoint script flattens to safetensors + the SentencePiece
//! tokenizer file; the DAC / CSM / Kokoro / DFN3 pattern). Output: a
//! GGUF carrying every F32 / F16 tensor verbatim plus the
//! `vokra.canary.*` / `vokra.provenance.*` metadata chunks.
//!
//! # What is transcribed vs. shape-driven
//!
//! - **Transcribed constants** — every hparam of the `vokra.canary.*`
//!   chunk group is transcribed **verbatim** from the primary sources
//!   (model card + the shared FastConformer-Transformer AED reference
//!   config — see the runtime module `vokra_models::canary` for the
//!   full per-field source table). No axis is invented; any tensor
//!   whose shape disagrees with these values in a real conversion fails
//!   the runtime shape gate loudly (FR-EX-08,
//!   `CanaryConfig::validate_for_forward`).
//! - **Shape-driven** — every tensor's dtype + shape is preserved
//!   verbatim from the safetensors header; the converter never widens
//!   or quantises Canary weights (the M2-08 quant policy path is
//!   whisper-only — canary reaches the whisper-only refusal in the CLI
//!   arm, matching Parakeet-CTC / Parakeet-TDT / Dia / Zonos / Kyutai
//!   STT).
//!
//! # Tensor naming contract
//!
//! GGUF tensor names are the upstream safetensors names **verbatim**
//! (the CSM / Kokoro / CosyVoice2 / Dia / Zonos / Kyutai STT /
//! Parakeet-TDT / Parakeet-CTC contract). Real-weight binding is a
//! follow-up wave gated on the `.nemo` extraction (T29-equivalent);
//! this converter passes every F32 / F16 tensor through unchanged so a
//! future `CanaryWeights::from_gguf` can walk the same names.
//!
//! # BF16 posture
//!
//! The reference release advertises BF16 weights in the `.nemo` tarball
//! (NeMo's standard save format for 1B+ FastConformer AED models). Per
//! the accepted qwen3-tts ADR (docs/adr/qwen3-tts-bf16.md, strategy
//! A_passthrough — the moshi / voxtral / vibevoice / voxcpm2 posture),
//! BF16 tensors pass through **verbatim** as GGUF type 30
//! (`GgmlType::BF16`) with no convert-time widening; the runtime widens
//! BF16 → f32 losslessly at load via the single choke point
//! `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16` (BF16 = top 16
//! bits of f32 — `bits << 16` is exact). A [`CanaryReport::bf16_passthrough`]
//! subset counter records how many BF16 tensors landed on the
//! pass-through arm so a latent silent-widen cannot slip in undetected
//! (FR-EX-08).
//!
//! # No ONNX (permanent)
//!
//! Canary ships as a `.nemo` tarball / Python pipeline; the pipeline is
//! re-implemented natively in `vokra-models/src/canary/` (whisper.cpp 型,
//! CLAUDE.md 設計判断 4). This converter never touches ONNX.

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` for Canary GGUFs — kept in sync with the runtime
/// constant `vokra-models::canary::EXPECTED_ARCH`.
pub(crate) const ARCH: &str = "canary";
/// `vokra.model.name` for Canary GGUFs (canonical model id).
pub(crate) const NAME: &str = "canary-1b-v2";

/// The FR-MD-09 attribution text stamped into
/// `vokra.provenance.attribution` — wording aligned with `NOTICE` and
/// the `docs/license-audit.md` NVIDIA Canary row (final legal
/// sufficiency = T29-equivalent owner sign-off; this converter records
/// the attribution but the owner-facing publish gate can add / edit
/// before release).
pub(crate) const CANARY_ATTRIBUTION_TEXT: &str = "This application uses NVIDIA Canary-1B-v2 \
     (multilingual multi-task ASR / AST — 25 European languages; \
     FastConformer encoder + Transformer decoder AED). Model weights \
     are licensed under CC-BY 4.0 (attribution required; commercial use \
     permitted). Copyright (c) NVIDIA. Source: \
     https://huggingface.co/nvidia/canary-1b-v2";

// --- vokra.canary.* keys (kept as constants in the converter; the
// runtime duplicates the strings when it lands `CanaryConfig::from_gguf`
// — the cross-crate pattern established by CSM / CosyVoice2 / Kokoro /
// Dia / Zonos / Kyutai STT / Parakeet-TDT / Parakeet-CTC) ------------

const KEY_SAMPLE_RATE: &str = "vokra.canary.sample_rate";

// Encoder (FastConformer)
const KEY_ENC_N_LAYER: &str = "vokra.canary.arch.encoder.n_layer";
const KEY_ENC_D_MODEL: &str = "vokra.canary.arch.encoder.d_model";
const KEY_ENC_N_HEAD: &str = "vokra.canary.arch.encoder.n_head";
const KEY_ENC_N_HEAD_KV: &str = "vokra.canary.arch.encoder.n_head_kv";
const KEY_ENC_FFN_DIM: &str = "vokra.canary.arch.encoder.ffn_dim";
const KEY_ENC_CONV_KERNEL: &str = "vokra.canary.arch.encoder.conv_kernel_size";
const KEY_ENC_IN_DIM: &str = "vokra.canary.arch.encoder.in_dim";
const KEY_ENC_SUBSAMPLING_FACTOR: &str = "vokra.canary.arch.encoder.subsampling_factor";
const KEY_ENC_SUB_CONV_KERNEL: &str = "vokra.canary.arch.encoder.subsampling_conv_kernel_size";
const KEY_ENC_SUB_CONV_STRIDE: &str = "vokra.canary.arch.encoder.subsampling_conv_stride";
const KEY_ENC_SUB_CONV_CHANNELS: &str = "vokra.canary.arch.encoder.subsampling_conv_channels";
const KEY_ENC_MAX_POS: &str = "vokra.canary.arch.encoder.max_position_embeddings";
const KEY_ENC_ATTN_BIAS: &str = "vokra.canary.arch.encoder.attention_bias";
const KEY_ENC_CONV_BIAS: &str = "vokra.canary.arch.encoder.convolution_bias";
const KEY_ENC_SCALE_INPUT: &str = "vokra.canary.arch.encoder.scale_input";

// Decoder (Transformer AED)
const KEY_DEC_N_LAYER: &str = "vokra.canary.arch.decoder.n_layer";
const KEY_DEC_D_MODEL: &str = "vokra.canary.arch.decoder.d_model";
const KEY_DEC_N_HEAD: &str = "vokra.canary.arch.decoder.n_head";
const KEY_DEC_FFN_DIM: &str = "vokra.canary.arch.decoder.ffn_dim";
const KEY_DEC_MAX_SEQ: &str = "vokra.canary.arch.decoder.max_sequence_length";
const KEY_DEC_PRE_LN: &str = "vokra.canary.arch.decoder.pre_ln";
const KEY_DEC_HIDDEN_ACT: &str = "vokra.canary.arch.decoder.hidden_act";

// Head + vocab
const KEY_HEAD_VOCAB_SIZE: &str = "vokra.canary.head.vocab_size";
const KEY_HEAD_PAD_ID: &str = "vokra.canary.head.pad_token_id";
const KEY_HEAD_BOS_ID: &str = "vokra.canary.head.bos_token_id";
const KEY_HEAD_EOS_ID: &str = "vokra.canary.head.eos_token_id";

// --- Transcribed constants ---------------------------------------------------
//
// Model card (`huggingface.co/nvidia/canary-1b-v2`, fetched 2026-07-24)
// supplies the axes it names: encoder n_layers=32, decoder n_layers=8,
// vocab_size=16384, sample_rate=16000, license=CC-BY-4.0. Every other
// axis is transcribed from the shared FastConformer-Transformer AED
// reference config
// (`github.com/NVIDIA-NeMo/Speech/blob/main/examples/asr/conf/speech_multitask/fast-conformer_aed.yaml`,
// fetched 2026-07-24) whose header explicitly names Canary variants as
// its consumers. The `.nemo` tarball's `model_config.yaml` is the
// ultimate authority; a follow-up wave (T29-equivalent) confirms every
// value against it, and the runtime shape gate catches a divergence
// loudly (FR-EX-08).

// PCM sample rate (model card).
const CANARY_SAMPLE_RATE: u32 = 16_000;

// Encoder (model card + family reference)
const ENC_N_LAYER: u32 = 32; // model card
const ENC_D_MODEL: u32 = 1024; // family default (asr_enc_hidden)
const ENC_N_HEAD: u32 = 8; // family default (encoder.n_heads)
const ENC_N_HEAD_KV: u32 = 8; // family default (MHA — no GQA)
const ENC_FFN_DIM: u32 = 4096; // family default (ff_expansion_factor=4 x 1024)
const ENC_CONV_KERNEL: u32 = 9; // family default (conv_kernel_size)
const ENC_IN_DIM: u32 = 128; // family default (preprocessor.features)
const ENC_SUBSAMPLING_FACTOR: u32 = 8; // family default
const ENC_SUB_CONV_KERNEL: u32 = 3; // family default (dw_striding stride-2 kernel-3)
const ENC_SUB_CONV_STRIDE: u32 = 2; // family default
const ENC_SUB_CONV_CHANNELS: u32 = 256; // family default
const ENC_MAX_POS: u32 = 5000; // family default (pos_emb_max_len)
const ENC_ATTN_BIAS: bool = true; // family default (untie_biases=true)
const ENC_CONV_BIAS: bool = false; // family default
const ENC_SCALE_INPUT: bool = false; // family default (xscaling=false)

// Decoder (model card + family reference)
const DEC_N_LAYER: u32 = 8; // model card
const DEC_D_MODEL: u32 = 1024; // family default (lm_dec_hidden)
const DEC_N_HEAD: u32 = 8; // family default (num_attention_heads)
const DEC_FFN_DIM: u32 = 4096; // family default (4 x lm_dec_hidden)
const DEC_MAX_SEQ: u32 = 1024; // family convention (flash variants)
const DEC_PRE_LN: bool = true; // family default (pre_ln=true)
const DEC_HIDDEN_ACT: &str = "relu"; // family default (hidden_act=relu)

// Head + vocab (model card + placeholder-sentinel token ids)
const HEAD_VOCAB_SIZE: u32 = 16_384; // model card
// pad/bos/eos ids are `0` placeholders until the `.nemo` extraction sets
// the real values (the tokenizer's pad is typically 0). The runtime
// validator rejects any id that exceeds vocab_size.
const HEAD_PAD_ID: u32 = 0;
const HEAD_BOS_ID: u32 = 0;
const HEAD_EOS_ID: u32 = 0;

/// Outcome of a Canary conversion.
#[derive(Debug, Default)]
pub(crate) struct CanaryReport {
    /// Float tensors written verbatim (F32 / F16 / BF16 — all three go
    /// through the same byte-copy path since the BF16 pass-through land
    /// 2026-07-25, mirror of `qwen3-tts` / `vibevoice` / `voxcpm2` /
    /// `moshi` / `voxtral`).
    pub(crate) written: usize,
    /// Non-F32 / F16 / BF16 tensors skipped (defensive counter — the
    /// safetensors reader accepts only `F32` / `F16` / `BF16` at parse
    /// time (`crates/vokra-core/src/safetensors.rs map_dtype`), so any
    /// tensor reaching this counter would signal a reader change
    /// upstream; kept for symmetry with the sibling converters and to
    /// surface the "no float tensors" loud note when zero writes
    /// occur).
    pub(crate) skipped_non_float: usize,
    /// Of the tensors in [`Self::written`], how many were BF16 (subset
    /// counter). Emits GGUF type 30 verbatim; runtime widens BF16 → f32
    /// losslessly via the single choke point
    /// `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16` (BF16 =
    /// top 16 bits of an f32 — `bits << 16` is exact). Additive
    /// observability counter so a latent silent-widen cannot slip in
    /// undetected. Mirrors `qwen3_tts::Qwen3TtsReport::bf16_passthrough`
    /// and `vibevoice::VibeVoiceReport::bf16_passthrough`.
    pub(crate) bf16_passthrough: usize,
    /// Operator-facing diagnostics (never fail the conversion — the
    /// runtime is the authoritative gate, FR-EX-08).
    pub(crate) notes: Vec<String>,
}

/// Converts a Canary safetensors buffer into a populated GGUF builder.
///
/// Every F32 / F16 tensor passes through under its upstream name; the
/// `vokra.canary.*` chunk group is written from the transcribed
/// constants above; provenance stamps mark the weight as
/// `AttributionRequired` (CC-BY 4.0) and the FR-MD-09 attribution
/// surface activates.
pub(crate) fn convert(bytes: Vec<u8>) -> Result<(GgufBuilder, CanaryReport), ConvertError> {
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    write_hparams(&mut b);
    vokra_core::stamp_provenance(
        &mut b,
        LicenseClass::AttributionRequired,
        "CC-BY-4.0",
        Some("nvidia/canary-1b-v2"),
        Some("https://huggingface.co/nvidia/canary-1b-v2"),
    );
    vokra_core::stamp_attribution(&mut b, CANARY_ATTRIBUTION_TEXT);

    let mut report = CanaryReport::default();
    // Float tensors pass through **verbatim** — no convert-time widening.
    // BF16 stays GGUF `BF16` (type 30) per the accepted qwen3-tts ADR
    // (docs/adr/qwen3-tts-bf16.md, strategy A_passthrough); the runtime
    // widens BF16 → f32 exactly at load via the single choke point
    // `crates/vokra-core/src/gguf/quant/mod.rs decode_bf16`. Mirrors
    // `qwen3_tts::convert` / `vibevoice::convert` / `voxcpm2::convert`
    // (2026-07-25 横展開).
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
             Canary-1B-v2 release ships a .nemo tarball whose PyTorch checkpoint \
             is typically BF16; the BF16 pass-through path is now wired \
             (2026-07-25 — mirror of qwen3-tts / vibevoice / voxcpm2), so this \
             state is only reachable when the prepared safetensors contains no \
             F32 / F16 / BF16 float tensors at all."
                .to_owned(),
        );
    }
    Ok((b, report))
}

/// Writes the `vokra.canary.*` chunk group from the transcribed
/// constants above (primary source: model card + family reference).
/// Booleans ride as u32 0/1 for GGUF portability (the Zonos / CSM /
/// Kyutai STT / Parakeet-TDT / Parakeet-CTC convention).
fn write_hparams(b: &mut GgufBuilder) {
    b.add_u32(KEY_SAMPLE_RATE, CANARY_SAMPLE_RATE);

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
    b.add_u32(KEY_DEC_N_HEAD, DEC_N_HEAD);
    b.add_u32(KEY_DEC_FFN_DIM, DEC_FFN_DIM);
    b.add_u32(KEY_DEC_MAX_SEQ, DEC_MAX_SEQ);
    b.add_u32(KEY_DEC_PRE_LN, u32::from(DEC_PRE_LN));
    b.add_string(KEY_DEC_HIDDEN_ACT, DEC_HIDDEN_ACT);

    // Head + vocab
    b.add_u32(KEY_HEAD_VOCAB_SIZE, HEAD_VOCAB_SIZE);
    b.add_u32(KEY_HEAD_PAD_ID, HEAD_PAD_ID);
    b.add_u32(KEY_HEAD_BOS_ID, HEAD_BOS_ID);
    b.add_u32(KEY_HEAD_EOS_ID, HEAD_EOS_ID);
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
        let header = r#"{"decoder.blocks.0.self_attn.qkv.weight":{"dtype":"F16","shape":[2,3],"data_offsets":[0,12]}}"#;
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(&[0u8; 12]);
        out
    }

    /// The upstream Canary-1B-v2 `.nemo` tarball's PyTorch checkpoint is
    /// BF16; today's pass-through arm handles only F32 / F16, so BF16
    /// falls to `skipped_non_float`. This test guards against a
    /// regression where somebody promotes BF16 into the pass-through arm
    /// without deciding how to stream them (bounded memory — the Moshi
    /// T22 pattern).
    fn minimal_safetensors_one_bf16() -> Vec<u8> {
        let header = r#"{"encoder.blocks.0.attn.qkv_proj.weight":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,12]}}"#;
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(&[0u8; 12]);
        out
    }

    #[test]
    fn arch_string_matches_runtime_constant() {
        // The two crates only share `vokra-core`, so this constant is
        // the sole handshake with `vokra-models::canary::EXPECTED_ARCH`.
        assert_eq!(ARCH, "canary");
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
            (KEY_SAMPLE_RATE, CANARY_SAMPLE_RATE),
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
            (KEY_DEC_N_HEAD, DEC_N_HEAD),
            (KEY_DEC_FFN_DIM, DEC_FFN_DIM),
            (KEY_DEC_MAX_SEQ, DEC_MAX_SEQ),
            (KEY_DEC_PRE_LN, u32::from(DEC_PRE_LN)),
            (KEY_HEAD_VOCAB_SIZE, HEAD_VOCAB_SIZE),
            (KEY_HEAD_PAD_ID, HEAD_PAD_ID),
            (KEY_HEAD_BOS_ID, HEAD_BOS_ID),
            (KEY_HEAD_EOS_ID, HEAD_EOS_ID),
        ] {
            match file.get(key) {
                Some(GgufMetadataValue::U32(v)) => assert_eq!(*v, want, "{key}"),
                other => panic!("{key}: unexpected {other:?}"),
            }
        }
        // Decoder hidden_act is a string.
        assert_eq!(
            file.get(KEY_DEC_HIDDEN_ACT).and_then(|v| v.as_str()),
            Some(DEC_HIDDEN_ACT)
        );

        // Provenance: CC-BY 4.0 attribution-required.
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_MODEL_ID)
                .and_then(|v| v.as_str()),
            Some("nvidia/canary-1b-v2")
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
            .tensor_info("decoder.blocks.0.self_attn.qkv.weight")
            .expect("tensor present");
        assert_eq!(info.dtype, GgmlType::F16);
        assert_eq!(info.dimensions, vec![2, 3]);
        assert_eq!(file.tensor_bytes(info).len(), 12);
    }

    /// Pins the BF16 leg of the `GgmlType::F32 | GgmlType::F16 |
    /// GgmlType::BF16` union: BF16 (the upstream serving format for
    /// Canary-1B-v2, the `.nemo` tarball's PyTorch checkpoint is BF16)
    /// must reach the pass-through arm, emit as GGUF type 30 verbatim,
    /// and increment `bf16_passthrough`. Mirror of qwen3-tts /
    /// vibevoice / voxcpm2's `bf16_tensor_passes_through_verbatim` and
    /// moshi's `assert_eq!(info.dtype, GgmlType::BF16, "no convert-time
    /// widening")`.
    ///
    /// Rewritten 2026-07-25 from the earlier "counted as skipped" pin —
    /// the earlier pin encoded the pre-BF16-fix scaffold posture.
    /// Removing the pin outright would let a latent silent-widen slip in
    /// undetected; rewriting to the passes-through invariant keeps the
    /// regression guard.
    #[test]
    fn bf16_tensor_passes_through_verbatim() {
        let (builder, report) = convert(minimal_safetensors_one_bf16()).expect("convert");
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
        // The tensor survives the round trip under its upstream name and
        // preserves its BF16 dtype (no convert-time widening — runtime
        // widens on load via `decode_bf16`, `bits << 16` — exact).
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
        // Round-trip byte equality on the tensor payload — the
        // pass-through path is a byte copy, no widening.
        assert_eq!(
            file.tensor_bytes(info),
            &[0u8; 12],
            "BF16 payload bytes must round-trip verbatim"
        );
        // Arch / provenance metadata still land alongside the tensor.
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

    /// Guards the axes that identify Canary-1B-v2 versus the earlier
    /// Canary variants (v1 with 24 encoder + 24 decoder layers vs
    /// canary-1b-flash with 32 encoder + 4 decoder layers): v2 has
    /// **32 encoder layers** and **8 decoder layers**.
    ///
    /// Compares against `u32::from(bool)` so the const-bool guards do
    /// not trip clippy's `assertions_on_constants`.
    #[test]
    fn canary_1b_v2_differs_from_v1_and_flash_on_key_axes() {
        assert_eq!(
            ENC_N_LAYER, 32,
            "canary-1b-v2: 32 encoder layers (v1: 24, flash: 32)"
        );
        assert_eq!(
            DEC_N_LAYER, 8,
            "canary-1b-v2: 8 decoder layers (v1: 24, flash: 4)"
        );
        assert_eq!(
            HEAD_VOCAB_SIZE, 16_384,
            "canary-1b-v2: vocab_size=16384 (unified multilingual SentencePiece)"
        );
        assert_eq!(
            u32::from(ENC_ATTN_BIAS),
            1,
            "canary-1b-v2: attention_bias=true (family default untie_biases=true)"
        );
        assert_eq!(
            u32::from(ENC_SCALE_INPUT),
            0,
            "canary-1b-v2: scale_input=false (family default xscaling=false)"
        );
        assert_eq!(
            u32::from(ENC_CONV_BIAS),
            0,
            "canary-1b-v2: convolution_bias=false"
        );
        assert_eq!(
            ENC_IN_DIM, 128,
            "canary-1b-v2: 128 mel bins (family default)"
        );
        assert_eq!(
            DEC_MAX_SEQ, 1024,
            "canary-1b-v2: decoder max_sequence_length=1024 (flash convention)"
        );
    }
}
