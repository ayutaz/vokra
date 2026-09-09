//! Meta **omniASR-CTC-1B**: safetensors checkpoint → GGUF conversion
//! (SoTA plan Phase 2, 2026-07-24).
//!
//! Input: a prepared safetensors flattened from the upstream
//! `facebook/omniASR-CTC-1B` fairseq2 `.pt` checkpoint (the HF release
//! ships a `.pt` + a SentencePiece tokenizer, no `config.json` and no
//! raw safetensors — a prepare-checkpoint script flattens the fairseq2
//! state dict to safetensors first, matching Dia / Kyutai STT posture).
//! Output: a GGUF carrying every audited F32 tensor verbatim plus the
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
//! - **Shape/dtype-driven** — the converter checks every tensor against the
//!   audited name/shape map and requires F32 exactly; it never widens or
//!   quantises omniASR-CTC weights (the M2-08 quant policy path is
//!   whisper-only — omniASR reaches that refusal in the CLI arm).
//!
//! # Tensor naming contract
//!
//! GGUF tensor names are the upstream fairseq2 state-dict names
//! **verbatim** (the CSM / Kokoro / CosyVoice2 / Dia / Zonos / Kyutai
//! STT / Parakeet-CTC contract). The converter accepts only the exact
//! audited 807-entry F32 manifest and stamps the immutable prepared/source
//! digests consumed by `OmniasrCtcAsr::from_gguf`. The prepared safetensors
//! SHA-256 is pinned to
//! `cda8d7dd7cad2a0361b6946c42342b85ef7b0a8d672b99631dc75b4c3123dbc5`.
//!
//! # Dtype posture
//!
//! The `facebook/omniASR-CTC-1B.pt` checkpoint is `torch.float32` per
//! the fairseq2 release (F32 stems + F32 transformer weights, ~3.9 GiB
//! at F32). F16/BF16 and arbitrary same-count inputs are rejected: the
//! prepared digest and manifest are one immutable conversion contract.
//!
//! # No ONNX (permanent)
//!
//! omniASR-CTC ships as a fairseq2 `.pt` checkpoint (plus a
//! SentencePiece tokenizer); the pipeline is re-implemented natively
//! in `vokra-models/src/omniasr_ctc/` (whisper.cpp 型, CLAUDE.md 設計
//! 判断 4). This converter never touches ONNX.

use std::collections::BTreeMap;
use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use super::canary_1b_flash::{hex, sha256};
use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// `vokra.model.arch` for omniASR-CTC GGUFs — kept in sync with the
/// runtime constant `vokra-models::omniasr_ctc::EXPECTED_ARCH`.
pub(crate) const ARCH: &str = "omniasr-ctc";
/// `vokra.model.name` for omniASR-CTC GGUFs (canonical model id).
pub(crate) const NAME: &str = "omniasr-ctc-1b";

/// Immutable provenance for the audited fairseq2 extraction consumed by the
/// native runtime binder.  These values come from the authoritative VAST
/// tensor manifest; they are not inferred from a caller-supplied filename.
const UPSTREAM_REVISION: &str = "8c22e3ffdaa4aab6431b128b84b991a7d9c2515c";
const SOURCE_SHA256: &str = "e8564fa59dab7caedbcdb54ab7fb9bd6c96989f4d19add2ad81ddd969716952c";
const PREPARED_SHA256: &str = "cda8d7dd7cad2a0361b6946c42342b85ef7b0a8d672b99631dc75b4c3123dbc5";
const KEY_OMNIASR_SOURCE_SHA256: &str = "vokra.omniasr_ctc.source_sha256";
const KEY_OMNIASR_PREPARED_SHA256: &str = "vokra.omniasr_ctc.prepared_sha256";

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
    /// F32 tensors written verbatim from the audited prepared artifact.
    pub(crate) written: usize,
    /// Retained for report compatibility; strict validation rejects every
    /// non-F32 tensor, so this is always zero on a successful conversion.
    pub(crate) skipped_non_float: usize,
    /// Operator-facing diagnostics; strict validation fails before a
    /// successful conversion can require a warning, so this is empty today.
    pub(crate) notes: Vec<String>,
}

/// Converts an omniASR-CTC safetensors buffer into a populated GGUF
/// builder.
///
/// Every audited F32 tensor passes through under its upstream name; the
/// `vokra.omniasr_ctc.*` chunk group is written from the transcribed
/// constants above; provenance stamps mark the weight as
/// `Permissive` (Apache-2.0) — no runtime-side attribution obligation
/// (unlike Parakeet-CTC / Canary / Kyutai STT which are CC-BY 4.0).
pub(crate) fn convert(bytes: Vec<u8>) -> Result<(GgufBuilder, OmniasrCtcReport), ConvertError> {
    let prepared_sha256 = hex(&sha256(&bytes));
    if prepared_sha256 != PREPARED_SHA256 {
        return Err(ConvertError::Parse(format!(
            "omniasr-ctc: prepared safetensors SHA-256 {prepared_sha256}, expected pinned {PREPARED_SHA256}"
        )));
    }
    let st = SafetensorsFile::parse(bytes)?;
    validate_payload_manifest(&st)?;

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
    b.add_string("vokra.provenance.upstream_revision", UPSTREAM_REVISION);
    b.add_string(KEY_OMNIASR_SOURCE_SHA256, SOURCE_SHA256);
    b.add_string(KEY_OMNIASR_PREPARED_SHA256, PREPARED_SHA256);

    let mut report = OmniasrCtcReport::default();
    for t in st.tensors() {
        match t.dtype {
            GgmlType::F32 => {
                b.add_tensor(
                    &t.name,
                    t.dtype,
                    t.shape.clone(),
                    st.tensor_bytes(t).to_vec(),
                )?;
                report.written += 1;
            }
            _ => {
                return Err(ConvertError::Parse(format!(
                    "omniasr-ctc: tensor `{}` changed dtype after strict manifest validation",
                    t.name
                )));
            }
        }
    }
    Ok((b, report))
}

/// Validate the complete upstream state-dict contract before emitting any
/// GGUF.  This duplicates the runtime's small manifest map deliberately: the
/// converter and loader are separate crates and must each fail closed when an
/// arbitrary same-count checkpoint is supplied.
fn validate_payload_manifest(st: &SafetensorsFile) -> Result<(), ConvertError> {
    let mut expected = BTreeMap::new();
    let mut add = |name: String, shape: &[u64]| {
        expected.insert(name, shape.to_vec());
    };
    add("encoder.layer_norm.bias".to_owned(), &[1280]);
    add("encoder.layer_norm.weight".to_owned(), &[1280]);
    add("final_proj.bias".to_owned(), &[9812]);
    add("final_proj.weight".to_owned(), &[9812, 1280]);
    for i in 0..48 {
        let p = format!("encoder.layers.{i}");
        add(format!("{p}.ffn.inner_proj.bias"), &[5120]);
        add(format!("{p}.ffn.inner_proj.weight"), &[5120, 1280]);
        add(format!("{p}.ffn.output_proj.bias"), &[1280]);
        add(format!("{p}.ffn.output_proj.weight"), &[1280, 5120]);
        add(format!("{p}.ffn_layer_norm.bias"), &[1280]);
        add(format!("{p}.ffn_layer_norm.weight"), &[1280]);
        for projection in ["k", "output", "q", "v"] {
            add(format!("{p}.self_attn.{projection}_proj.bias"), &[1280]);
            add(
                format!("{p}.self_attn.{projection}_proj.weight"),
                &[1280, 1280],
            );
        }
        add(format!("{p}.self_attn_layer_norm.bias"), &[1280]);
        add(format!("{p}.self_attn_layer_norm.weight"), &[1280]);
    }
    for (i, &kernel) in [10u64, 3, 3, 3, 3, 2, 2].iter().enumerate() {
        let input = if i == 0 { 1 } else { 512 };
        let p = format!("encoder_frontend.feature_extractor.layers.{i}");
        add(format!("{p}.conv.bias"), &[512]);
        add(format!("{p}.conv.weight"), &[512, input, kernel]);
        add(format!("{p}.layer_norm.bias"), &[512]);
        add(format!("{p}.layer_norm.weight"), &[512]);
    }
    add("encoder_frontend.model_dim_proj.bias".to_owned(), &[1280]);
    add(
        "encoder_frontend.model_dim_proj.weight".to_owned(),
        &[1280, 512],
    );
    add("encoder_frontend.pos_encoder.conv.bias".to_owned(), &[1280]);
    add(
        "encoder_frontend.pos_encoder.conv.weight_g".to_owned(),
        &[1, 1, 128],
    );
    add(
        "encoder_frontend.pos_encoder.conv.weight_v".to_owned(),
        &[1280, 80, 128],
    );
    add(
        "encoder_frontend.post_extract_layer_norm.bias".to_owned(),
        &[512],
    );
    add(
        "encoder_frontend.post_extract_layer_norm.weight".to_owned(),
        &[512],
    );
    if expected.len() != 807 || st.tensors().len() != expected.len() {
        return Err(ConvertError::Parse(format!(
            "omniasr-ctc: safetensors has {} tensors, expected exact audited manifest size {}",
            st.tensors().len(),
            expected.len()
        )));
    }
    for tensor in st.tensors() {
        let Some(shape) = expected.get(&tensor.name) else {
            return Err(ConvertError::Parse(format!(
                "omniasr-ctc: unexpected tensor `{}` in audited manifest",
                tensor.name
            )));
        };
        if tensor.shape != *shape {
            return Err(ConvertError::Parse(format!(
                "omniasr-ctc: tensor `{}` shape {:?}, expected {:?}",
                tensor.name, tensor.shape, shape
            )));
        }
        if tensor.dtype != GgmlType::F32 {
            return Err(ConvertError::Parse(format!(
                "omniasr-ctc: tensor `{}` dtype {:?}, expected F32 from audited manifest",
                tensor.name, tensor.dtype
            )));
        }
    }
    Ok(())
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
    fn minimal_safetensors_one_f32() -> Vec<u8> {
        // A deliberately noncanonical fixture for digest-gate rejection.
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

    /// A deliberately noncanonical BF16 fixture for strict rejection.
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
    fn small_fixture_is_rejected_before_provenance_stamping() {
        let error = convert(minimal_safetensors_one_f32())
            .expect_err("small fixture must fail strict prepared digest gate");
        assert!(error.to_string().contains("prepared safetensors SHA-256"));
    }

    #[test]
    fn prepared_digest_bit_tamper_is_rejected_before_parsing() {
        let mut bytes = minimal_safetensors_one_f32();
        let last = bytes.last_mut().expect("fixture payload");
        *last ^= 1;
        let error = convert(bytes).expect_err("tampered prepared bytes must fail");
        assert!(error.to_string().contains("prepared safetensors SHA-256"));
    }

    #[test]
    fn zero_tensor_conversion_is_rejected_before_provenance_stamping() {
        let error = convert(minimal_safetensors_no_tensors())
            .expect_err("metadata-only fixture must fail strict prepared digest gate");
        assert!(error.to_string().contains("prepared safetensors SHA-256"));
    }

    /// F16 input is rejected by the pinned prepared-artifact contract.
    #[test]
    fn f16_tensor_is_rejected_by_strict_contract() {
        let error = convert(minimal_safetensors_one_f16())
            .expect_err("F16 fixture must fail strict digest gate");
        assert!(error.to_string().contains("prepared safetensors SHA-256"));
    }

    /// BF16 input is rejected by the pinned prepared-artifact contract.
    #[test]
    fn bf16_tensor_is_rejected_by_strict_contract() {
        let error = convert(minimal_safetensors_one_bf16())
            .expect_err("BF16 fixture must fail strict digest gate");
        assert!(error.to_string().contains("prepared safetensors SHA-256"));
    }

    /// Any byte mutation reaches the digest gate before parsing, preserving
    /// the prepared artifact identity contract.
    #[test]
    fn malformed_input_digest_mismatch_is_a_controlled_parse_error() {
        for mut input in [Vec::new(), vec![0u8; 16]] {
            if input.is_empty() {
                input.push(1);
            } else {
                input[0] ^= 1;
            }
            let error = convert(input).expect_err("unprepared input must fail");
            assert!(matches!(error, ConvertError::Parse(_)));
            assert!(error.to_string().contains("prepared safetensors SHA-256"));
        }
    }

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
