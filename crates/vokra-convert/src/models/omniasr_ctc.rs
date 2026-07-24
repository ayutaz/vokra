//! Meta **omniASR-CTC-1B**: safetensors checkpoint → GGUF conversion
//! (SoTA plan Phase 2, 2026-07-24).
//!
//! Input: a prepared safetensors flattened from the upstream
//! `facebook/omniASR-CTC-1B` fairseq2 `.pt` checkpoint (the HF release
//! ships a `.pt` + a SentencePiece tokenizer, no `config.json` and no
//! raw safetensors — a prepare-checkpoint script flattens the fairseq2
//! state dict to safetensors first, matching Dia / Kyutai STT posture).
//! Output: a GGUF carrying every F32 / F16 tensor verbatim plus the
//! `vokra.omniasr_ctc.*` / `vokra.provenance.*` metadata chunks.
//!
//! # What is transcribed vs. shape-driven
//!
//! - **Transcribed constants** — every hparam of the
//!   `vokra.omniasr_ctc.*` chunk group is transcribed **verbatim** from
//!   the upstream fairseq2 registry walk (see the top of this module
//!   for the full table and the model docstring for the walk). No axis
//!   is invented; any tensor whose shape disagrees with these values in
//!   a real conversion fails the runtime shape gate loudly (FR-EX-08,
//!   `OmniasrCtcConfig::validate_for_forward`).
//! - **Shape-driven** — every tensor's dtype + shape is preserved
//!   verbatim from the safetensors header; the converter never widens
//!   or quantises omniASR-CTC weights (the M2-08 quant policy path is
//!   whisper-only — omniasr-ctc reaches the whisper-only refusal in the
//!   CLI arm, matching Parakeet-CTC / Parakeet-TDT / Dia / Zonos /
//!   Kyutai STT).
//!
//! # Tensor naming contract
//!
//! GGUF tensor names are the upstream fairseq2 state-dict names
//! **verbatim** (the CSM / Kokoro / CosyVoice2 / Dia / Zonos / Kyutai
//! STT / Parakeet-CTC contract). Real-weight binding is a follow-up
//! wave gated on the upstream tensor-name manifest fetch; this
//! converter passes every F32 / F16 tensor through unchanged so a
//! future `OmniasrCtcWeights::from_gguf` can walk the same names.
//!
//! # BF16 posture
//!
//! The `facebook/omniASR-CTC-1B.pt` checkpoint is `torch.float32` per
//! the fairseq2 release (F32 stems + F32 transformer weights, ~3.9 GiB
//! at F32); no BF16 pass-through is required to convert the release
//! build. A downstream that pre-widens to F16 offline lands on the
//! F16 arm of this converter (also pass-through); BF16 tensors reach
//! the `_ =>` arm and increment `skipped_non_float` (never a silent
//! widen — Kyutai STT / Moshi / Parakeet-CTC posture).
//!
//! # No ONNX (permanent)
//!
//! omniASR-CTC ships as a fairseq2 `.pt` checkpoint (plus a
//! SentencePiece tokenizer); the pipeline is re-implemented natively
//! in `vokra-models/src/omniasr_ctc/` (whisper.cpp 型, CLAUDE.md 設計
//! 判断 4). This converter never touches ONNX.

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` for omniASR-CTC GGUFs — kept in sync with the
/// runtime constant `vokra-models::omniasr_ctc::EXPECTED_ARCH`.
pub(crate) const ARCH: &str = "omniasr-ctc";
/// `vokra.model.name` for omniASR-CTC GGUFs (canonical model id).
pub(crate) const NAME: &str = "omniasr-ctc-1b";

// --- vokra.omniasr_ctc.* keys (kept as constants in the converter; the
// runtime duplicates the strings when it lands `OmniasrCtcConfig::from_gguf`
// — the cross-crate pattern established by CSM / CosyVoice2 / Kokoro /
// Dia / Zonos / Kyutai STT / Parakeet-CTC) --------------------------------

const KEY_SAMPLE_RATE: &str = "vokra.omniasr_ctc.sample_rate";

// Encoder (wav2vec 2.0)
const KEY_ENC_MODEL_DIM: &str = "vokra.omniasr_ctc.arch.encoder.model_dim";
const KEY_ENC_N_LAYER: &str = "vokra.omniasr_ctc.arch.encoder.num_encoder_layers";
const KEY_ENC_N_HEAD: &str = "vokra.omniasr_ctc.arch.encoder.num_encoder_attn_heads";
const KEY_ENC_FFN_INNER: &str = "vokra.omniasr_ctc.arch.encoder.ffn_inner_dim";
const KEY_ENC_FEATURE_DIM: &str = "vokra.omniasr_ctc.arch.encoder.feature_dim";
const KEY_ENC_MAX_SEQ_LEN: &str = "vokra.omniasr_ctc.arch.encoder.max_seq_len";
const KEY_ENC_FEATURE_BIAS: &str = "vokra.omniasr_ctc.arch.encoder.feature_extractor_bias";
const KEY_ENC_FEATURE_LN_CONVS: &str =
    "vokra.omniasr_ctc.arch.encoder.feature_extractor_layer_norm_convs";
const KEY_ENC_LN_FEATURES: &str = "vokra.omniasr_ctc.arch.encoder.layer_norm_features";
const KEY_ENC_POS_KERNEL: &str = "vokra.omniasr_ctc.arch.encoder.pos_conv_kernel_size";
const KEY_ENC_POS_GROUPS: &str = "vokra.omniasr_ctc.arch.encoder.num_pos_conv_groups";
const KEY_ENC_POS_DEPTH: &str = "vokra.omniasr_ctc.arch.encoder.pos_encoder_depth";
const KEY_ENC_USE_CONFORMER: &str = "vokra.omniasr_ctc.arch.encoder.use_conformer";

// Feature extractor (7 layers — the fairseq2 wav2vec 2.0 default,
// pinned as a fixed count). Rides as `count + N × (out_dim, kernel,
// stride)` — the CSM / Dia array pattern for GGUF portability.
const KEY_ENC_FEATURE_LAYERS: &str = "vokra.omniasr_ctc.arch.encoder.feature_extractor_layer_count";
const KEY_ENC_FEATURE_OUT_PREFIX: &str =
    "vokra.omniasr_ctc.arch.encoder.feature_extractor_out_dim.";
const KEY_ENC_FEATURE_KERNEL_PREFIX: &str =
    "vokra.omniasr_ctc.arch.encoder.feature_extractor_kernel.";
const KEY_ENC_FEATURE_STRIDE_PREFIX: &str =
    "vokra.omniasr_ctc.arch.encoder.feature_extractor_stride.";

// CTC head
const KEY_HEAD_VOCAB_SIZE: &str = "vokra.omniasr_ctc.head.target_vocab_size";
const KEY_HEAD_BLANK_ID: &str = "vokra.omniasr_ctc.head.blank_id";

// --- Transcribed constants (primary source: the fairseq2 registry walk
// fetched verbatim) -------------------------------------------------------
//
// `github.com/facebookresearch/omnilingual-asr/blob/main/src/omnilingual_asr/models/wav2vec2_asr/config.py`
//   arch `1b` → `Wav2Vec2AsrConfig` "base_10h" + wav2vec 2.0 arch "1b"
// `github.com/facebookresearch/omnilingual-asr/blob/main/src/omnilingual_asr/models/wav2vec2_ssl/config.py`
//   arch `1b` → walks `large_lv60k` and overrides model_dim / n_layer / ffn
// `github.com/facebookresearch/fairseq2/blob/main/src/fairseq2/models/wav2vec2/config.py`
//   `large_lv60k` → walks `large` → overrides feature-extractor axes
// (fetched 2026-07-24). Every value here is transcribed verbatim;
// nothing is invented.

// PCM sample rate — not written in an upstream `config.json` (the HF
// release carries no config); taken from the model card + wav2vec 2.0
// convention.
const OMNIASR_CTC_SAMPLE_RATE: u32 = 16_000;

// Encoder
const ENC_MODEL_DIM: u32 = 1280; // "1b": encoder_config.model_dim = 1280
const ENC_N_LAYER: u32 = 48; // "1b": encoder_config.num_encoder_layers = 48
const ENC_N_HEAD: u32 = 16; // "large": encoder_config.num_encoder_attn_heads = 16 (1b inherits)
const ENC_FFN_INNER: u32 = 5120; // "1b": encoder_config.ffn_inner_dim = 5120
const ENC_FEATURE_DIM: u32 = 512; // base: encoder_config.feature_dim = 512 (1b inherits)
const ENC_MAX_SEQ_LEN: u32 = 4096; // base: encoder_config.max_seq_len = 4096 (1b inherits)
const ENC_FEATURE_BIAS: bool = true; // large_lv60k: feature_extractor_bias = True
const ENC_FEATURE_LN_CONVS: bool = true; // large_lv60k: feature_extractor_layer_norm_convs = True
const ENC_LN_FEATURES: bool = false; // large_lv60k: layer_norm_features = False
const ENC_POS_KERNEL: u32 = 128; // base: pos_conv_kernel_size = 128
const ENC_POS_GROUPS: u32 = 16; // base: num_pos_conv_groups = 16
const ENC_POS_DEPTH: u32 = 1; // base: pos_encoder_depth = 1
const ENC_USE_CONFORMER: bool = false; // base: use_conformer = False

// Feature extractor: fixed 7-layer stem
// `[(512,10,5), (512,3,2), (512,3,2), (512,3,2), (512,3,2),
//   (512,2,2), (512,2,2)]` — total stride 320.
const ENC_FEATURE_LAYER_COUNT: u32 = 7;
const ENC_FEATURE_OUT_DIMS: [u32; 7] = [512, 512, 512, 512, 512, 512, 512];
const ENC_FEATURE_KERNELS: [u32; 7] = [10, 3, 3, 3, 3, 2, 2];
const ENC_FEATURE_STRIDES: [u32; 7] = [5, 2, 2, 2, 2, 2, 2];

// CTC head (fairseq2 wav2vec 2.0 convention)
const HEAD_VOCAB_SIZE: u32 = 9812; // "1b_asr": target_vocab_size = 9812 (v1 tokenizer)
// Blank at index 0 — the fairseq2 wav2vec 2.0 CTC convention (torch
// `ctc_loss` called without an explicit `blank=` argument in
// `fairseq2/models/wav2vec2/asr/model.py::Wav2Vec2AsrModel.forward`
// defaults to `blank=0`).
const HEAD_BLANK_ID: u32 = 0;

/// Outcome of an omniASR-CTC conversion.
#[derive(Debug, Default)]
pub(crate) struct OmniasrCtcReport {
    /// Float tensors written verbatim.
    pub(crate) written: usize,
    /// Non-F32 / F16 tensors skipped (the safetensors reader accepts
    /// BF16 / integer dtypes; this converter's pass-through arm is
    /// F32 / F16 only — a BF16-flattened variant of the release
    /// checkpoint falls here today).
    pub(crate) skipped_non_float: usize,
    /// Operator-facing diagnostics (never fail the conversion — the
    /// runtime is the authoritative gate, FR-EX-08).
    pub(crate) notes: Vec<String>,
}

/// Converts an omniASR-CTC safetensors buffer into a populated GGUF
/// builder.
///
/// Every F32 / F16 tensor passes through under its upstream name; the
/// `vokra.omniasr_ctc.*` chunk group is written from the transcribed
/// constants above; provenance stamps mark the weight as
/// `Permissive` (Apache-2.0) — no runtime-side attribution obligation
/// (unlike Parakeet-CTC / Canary / Kyutai STT which are CC-BY 4.0).
pub(crate) fn convert(bytes: Vec<u8>) -> Result<(GgufBuilder, OmniasrCtcReport), ConvertError> {
    let st = SafetensorsFile::parse(bytes)?;

    let mut b = GgufBuilder::new();
    b.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    b.add_string(chunks::KEY_MODEL_NAME, NAME);
    write_hparams(&mut b);
    vokra_core::stamp_provenance(
        &mut b,
        LicenseClass::Permissive,
        "Apache-2.0",
        Some("facebook/omniASR-CTC-1B"),
        Some("https://huggingface.co/facebook/omniASR-CTC-1B"),
    );

    let mut report = OmniasrCtcReport::default();
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
             omniASR-CTC-1B release ships a fairseq2 `.pt` (F32); flatten it to \
             safetensors offline before conversion. If the flattening emitted BF16, \
             pre-widen offline (the streaming-BF16 pass-through path is a follow-up \
             wave — the Moshi pattern)."
                .to_owned(),
        );
    }
    Ok((b, report))
}

/// Writes the `vokra.omniasr_ctc.*` chunk group from the transcribed
/// constants above (primary source: fairseq2 registry walk).
/// Booleans ride as u32 0/1 for GGUF portability (the Zonos / CSM /
/// Kyutai STT / Parakeet-CTC convention). The 7-layer feature-extractor
/// descriptor rides as `count + 3 × N` u32 arrays (the CSM / Dia
/// array pattern for GGUF portability).
fn write_hparams(b: &mut GgufBuilder) {
    b.add_u32(KEY_SAMPLE_RATE, OMNIASR_CTC_SAMPLE_RATE);

    // Encoder
    b.add_u32(KEY_ENC_MODEL_DIM, ENC_MODEL_DIM);
    b.add_u32(KEY_ENC_N_LAYER, ENC_N_LAYER);
    b.add_u32(KEY_ENC_N_HEAD, ENC_N_HEAD);
    b.add_u32(KEY_ENC_FFN_INNER, ENC_FFN_INNER);
    b.add_u32(KEY_ENC_FEATURE_DIM, ENC_FEATURE_DIM);
    b.add_u32(KEY_ENC_MAX_SEQ_LEN, ENC_MAX_SEQ_LEN);
    b.add_u32(KEY_ENC_FEATURE_BIAS, u32::from(ENC_FEATURE_BIAS));
    b.add_u32(KEY_ENC_FEATURE_LN_CONVS, u32::from(ENC_FEATURE_LN_CONVS));
    b.add_u32(KEY_ENC_LN_FEATURES, u32::from(ENC_LN_FEATURES));
    b.add_u32(KEY_ENC_POS_KERNEL, ENC_POS_KERNEL);
    b.add_u32(KEY_ENC_POS_GROUPS, ENC_POS_GROUPS);
    b.add_u32(KEY_ENC_POS_DEPTH, ENC_POS_DEPTH);
    b.add_u32(KEY_ENC_USE_CONFORMER, u32::from(ENC_USE_CONFORMER));

    // Feature extractor stem
    b.add_u32(KEY_ENC_FEATURE_LAYERS, ENC_FEATURE_LAYER_COUNT);
    for i in 0..(ENC_FEATURE_LAYER_COUNT as usize) {
        b.add_u32(
            &format!("{KEY_ENC_FEATURE_OUT_PREFIX}{i}"),
            ENC_FEATURE_OUT_DIMS[i],
        );
        b.add_u32(
            &format!("{KEY_ENC_FEATURE_KERNEL_PREFIX}{i}"),
            ENC_FEATURE_KERNELS[i],
        );
        b.add_u32(
            &format!("{KEY_ENC_FEATURE_STRIDE_PREFIX}{i}"),
            ENC_FEATURE_STRIDES[i],
        );
    }

    // CTC head
    b.add_u32(KEY_HEAD_VOCAB_SIZE, HEAD_VOCAB_SIZE);
    b.add_u32(KEY_HEAD_BLANK_ID, HEAD_BLANK_ID);
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::{GgufFile, GgufMetadataValue};

    fn minimal_safetensors_one_f32() -> Vec<u8> {
        // A single f32 tensor at the top of the file so `convert` has
        // something to pass through and the report counts a non-zero
        // write.
        let header = r#"{"encoder.layers.0.self_attn.qkv_proj.weight":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}}"#;
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
        let header = r#"{"encoder.layers.0.self_attn.qkv_proj.weight":{"dtype":"F16","shape":[2,3],"data_offsets":[0,12]}}"#;
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(&[0u8; 12]);
        out
    }

    /// A pre-widened BF16 variant (a downstream might BF16 the F32
    /// release checkpoint offline); today's pass-through arm handles
    /// only F32 / F16, so BF16 falls to `skipped_non_float`. This test
    /// guards against a regression where somebody promotes BF16 into
    /// the pass-through arm without deciding how to stream them
    /// (bounded memory — the Moshi T22 pattern).
    fn minimal_safetensors_one_bf16() -> Vec<u8> {
        let header = r#"{"encoder.layers.0.self_attn.qkv_proj.weight":{"dtype":"BF16","shape":[2,3],"data_offsets":[0,12]}}"#;
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(&[0u8; 12]);
        out
    }

    #[test]
    fn arch_string_matches_runtime_constant() {
        // The two crates only share `vokra-core`, so this constant is
        // the sole handshake with
        // `vokra-models::omniasr_ctc::EXPECTED_ARCH`.
        assert_eq!(ARCH, "omniasr-ctc");
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
            (KEY_SAMPLE_RATE, OMNIASR_CTC_SAMPLE_RATE),
            (KEY_ENC_MODEL_DIM, ENC_MODEL_DIM),
            (KEY_ENC_N_LAYER, ENC_N_LAYER),
            (KEY_ENC_N_HEAD, ENC_N_HEAD),
            (KEY_ENC_FFN_INNER, ENC_FFN_INNER),
            (KEY_ENC_FEATURE_DIM, ENC_FEATURE_DIM),
            (KEY_ENC_MAX_SEQ_LEN, ENC_MAX_SEQ_LEN),
            (KEY_ENC_FEATURE_BIAS, u32::from(ENC_FEATURE_BIAS)),
            (KEY_ENC_FEATURE_LN_CONVS, u32::from(ENC_FEATURE_LN_CONVS)),
            (KEY_ENC_LN_FEATURES, u32::from(ENC_LN_FEATURES)),
            (KEY_ENC_POS_KERNEL, ENC_POS_KERNEL),
            (KEY_ENC_POS_GROUPS, ENC_POS_GROUPS),
            (KEY_ENC_POS_DEPTH, ENC_POS_DEPTH),
            (KEY_ENC_USE_CONFORMER, u32::from(ENC_USE_CONFORMER)),
            (KEY_ENC_FEATURE_LAYERS, ENC_FEATURE_LAYER_COUNT),
            (KEY_HEAD_VOCAB_SIZE, HEAD_VOCAB_SIZE),
            (KEY_HEAD_BLANK_ID, HEAD_BLANK_ID),
        ] {
            match file.get(key) {
                Some(GgufMetadataValue::U32(v)) => assert_eq!(*v, want, "{key}"),
                other => panic!("{key}: unexpected {other:?}"),
            }
        }

        // Per-layer feature extractor triples round-trip.
        for i in 0..(ENC_FEATURE_LAYER_COUNT as usize) {
            for (prefix, want) in [
                (KEY_ENC_FEATURE_OUT_PREFIX, ENC_FEATURE_OUT_DIMS[i]),
                (KEY_ENC_FEATURE_KERNEL_PREFIX, ENC_FEATURE_KERNELS[i]),
                (KEY_ENC_FEATURE_STRIDE_PREFIX, ENC_FEATURE_STRIDES[i]),
            ] {
                let key = format!("{prefix}{i}");
                match file.get(&key) {
                    Some(GgufMetadataValue::U32(v)) => assert_eq!(*v, want, "{key}"),
                    other => panic!("{key}: unexpected {other:?}"),
                }
            }
        }

        // Provenance: Apache-2.0 permissive.
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_MODEL_ID)
                .and_then(|v| v.as_str()),
            Some("facebook/omniASR-CTC-1B")
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
        // No attribution stamp for Apache-2.0 (unlike CC-BY 4.0
        // Parakeet-CTC / Canary / Kyutai STT).
        assert!(
            file.get(chunks::KEY_PROVENANCE_ATTRIBUTION).is_none(),
            "Apache-2.0 must not stamp attribution",
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
            .tensor_info("encoder.layers.0.self_attn.qkv_proj.weight")
            .expect("tensor present");
        assert_eq!(info.dtype, GgmlType::F16);
        assert_eq!(info.dimensions, vec![2, 3]);
        assert_eq!(file.tensor_bytes(info).len(), 12);
    }

    /// A pre-widened BF16 variant falls to the `_ =>` arm and MUST be
    /// counted, not silently widened. This test guards against a
    /// regression.
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
        assert!(
            report.notes.iter().any(|n| n.contains("no float tensors")),
            "BF16-only conversion must emit the zero-float note: {:?}",
            report.notes
        );
        // Metadata (arch / hparams / provenance) still lands — the
        // report reflects the tensor pass, not a failure of the
        // conversion.
        let out = builder.to_bytes().expect("serialize");
        let file = GgufFile::parse(out).expect("parse");
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH)
        );
        assert!(
            file.tensor_info("encoder.layers.0.self_attn.qkv_proj.weight")
                .is_none(),
            "BF16 tensor must not be written"
        );
    }

    /// Pins `SafetensorsFile::parse(bytes)?` error propagation. A
    /// malformed input surfaces as `Err(ConvertError::Parse(_))`, not
    /// a silently-empty successful conversion (FR-EX-08 loud fail).
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

    /// Guards the axes that distinguish omniASR-CTC from Parakeet-CTC:
    /// `feature_dim = 512` waveform (not 80 log-mel bins),
    /// `blank_id = 0` (not vocab tail = 1024), `num_encoder_layers =
    /// 48` (not 42 FastConformer), `use_conformer = false` (not
    /// true), Apache-2.0 (Permissive) not CC-BY 4.0. A regression here
    /// would silently ship a converter that emits the wrong hparam
    /// chunk group.
    ///
    /// Compares against `u32::from(bool)` so the const-bool guards do
    /// not trip clippy's `assertions_on_constants`.
    #[test]
    fn omniasr_ctc_differs_from_parakeet_ctc_on_key_axes() {
        assert_eq!(
            ENC_MODEL_DIM, 1280,
            "omniASR-CTC-1B: model_dim=1280 (Parakeet-CTC-1.1B: d_model=1024)"
        );
        assert_eq!(
            ENC_N_LAYER, 48,
            "omniASR-CTC-1B: 48 transformer layers (Parakeet-CTC-1.1B: 42 FastConformer)"
        );
        assert_eq!(
            ENC_FEATURE_DIM, 512,
            "omniASR-CTC-1B: waveform-feature 512 (Parakeet-CTC-1.1B: log-mel 80)"
        );
        assert_eq!(
            u32::from(ENC_USE_CONFORMER),
            0,
            "omniASR-CTC-1B: plain Transformer (Parakeet-CTC-1.1B: FastConformer)"
        );
        assert_eq!(
            u32::from(ENC_FEATURE_BIAS),
            1,
            "omniASR-CTC-1B: feature_extractor_bias=true (large_lv60k)"
        );
        assert_eq!(
            u32::from(ENC_FEATURE_LN_CONVS),
            1,
            "omniASR-CTC-1B: feature_extractor_layer_norm_convs=true (large_lv60k)"
        );
        assert_eq!(
            u32::from(ENC_LN_FEATURES),
            0,
            "omniASR-CTC-1B: layer_norm_features=false (large_lv60k)"
        );
        // Vocab and blank convention differ fundamentally from Parakeet-CTC.
        assert_eq!(
            HEAD_VOCAB_SIZE, 9812,
            "omniASR-CTC-1B: target_vocab_size=9812 v1 (Parakeet-CTC-1.1B: vocab_size=1025)"
        );
        assert_eq!(
            HEAD_BLANK_ID, 0,
            "omniASR-CTC-1B: blank_id=0 (fairseq2; Parakeet-CTC-1.1B: blank at vocab tail=1024)"
        );
    }

    /// The 7-layer stem is a pinned count with a specific stride
    /// pattern (product = 320× downsampling). Getting this wrong at
    /// conversion time would silently misalign every subsequent
    /// convolution.
    #[test]
    fn feature_extractor_stem_pattern_pins_320x_downsampling() {
        assert_eq!(ENC_FEATURE_LAYER_COUNT, 7);
        assert_eq!(ENC_FEATURE_OUT_DIMS.len(), 7);
        assert_eq!(ENC_FEATURE_KERNELS.len(), 7);
        assert_eq!(ENC_FEATURE_STRIDES.len(), 7);
        // Every out_dim = 512 (matches encoder.feature_dim).
        for d in &ENC_FEATURE_OUT_DIMS {
            assert_eq!(*d, ENC_FEATURE_DIM, "every stem layer is 512-channel");
        }
        // The first layer's (kernel, stride) = (10, 5); rest of the
        // stride pattern is 2^6 = 64. Total = 5 × 64 = 320.
        assert_eq!(ENC_FEATURE_KERNELS[0], 10);
        assert_eq!(ENC_FEATURE_STRIDES[0], 5);
        for s in &ENC_FEATURE_STRIDES[1..] {
            assert_eq!(*s, 2, "layers 1..7 all have stride 2");
        }
        let mut total: u32 = 1;
        for s in &ENC_FEATURE_STRIDES {
            total = total.saturating_mul(*s);
        }
        assert_eq!(
            total, 320,
            "wav2vec 2.0 stem produces one CTC frame per 20 ms at 16 kHz (320× downsampling)"
        );
    }
}
