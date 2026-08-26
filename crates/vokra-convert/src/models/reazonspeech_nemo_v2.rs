//! ReazonSpeech NeMo v2 complete `.nemo` checkpoint -> executable GGUF.
//!
//! The canonical input is the F32 safetensors file extracted from the pinned
//! upstream NeMo 1.21 archive on VAST. Conversion is fail-closed: all 965
//! inference tensors, their exact names and shapes, and the exact 3,000-piece
//! SentencePiece plaintext vocabulary must be present. The multi-gigabyte
//! checkpoint is never converted on the maintainer Mac.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{
    GgmlType, GgufArray, GgufBuilder, GgufMetadataValue, GgufValueType, chunks,
};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

pub const ARCH: &str = "reazonspeech_nemo_v2";
pub const NAME: &str = "reazonspeech-nemo-v2";
pub const CATEGORY: &str = "asr";
pub const UPSTREAM_HF: &str = "reazon-research/reazonspeech-nemo-v2";
pub const UPSTREAM_REVISION: &str = "33693408be76b7cba9fd4a7546a0a8772430211b";
/// Xet object SHA-256 for `reazonspeech-nemo-v2.nemo` at the pinned revision.
pub const SOURCE_NEMO_SHA256: &str =
    "d196d43ad03466ca88beeda4bf5fafb07bab7202d4b663b8e4f12cb0a4381fae";
/// SHA-256 of the bounded tar member manifest (six members).
pub const SOURCE_TAR_MANIFEST_SHA256: &str =
    "7f5268f676ab1496ef6202bd3a031a0fce5a434c6f2bd568efa2e7f14d7c4cb1";
pub const MODEL_CONFIG_SHA256: &str =
    "88925d58533c40da62007ad39b8abd702646c7e81627dea5b15961c4ad4f9833";
pub const TOKENIZER_VOCAB_SHA256: &str =
    "989e4950cf53c0fee66f632cdd966bdd840b851a9e0e812322fd667e4b1c07bb";
pub const TENSOR_MANIFEST_SHA256: &str =
    "0663932975fb2157d11fa8ce9d7183c69c00a3d3f3f0e916aff1cab0550401ab";
pub const DEFAULT_LICENSE_SPDX: &str = "apache-2.0";

const TENSOR_COUNT: usize = 965;
const TOKENIZER_PIECES: usize = 3_000;

const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
const KEY_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";
const KEY_SOURCE_REVISION: &str = "vokra.reazonspeech_nemo_v2.source_revision";
const KEY_SOURCE_NEMO_SHA256: &str = "vokra.reazonspeech_nemo_v2.source_nemo_sha256";
const KEY_SOURCE_TAR_MANIFEST_SHA256: &str =
    "vokra.reazonspeech_nemo_v2.source_tar_manifest_sha256";
const KEY_MODEL_CONFIG_SHA256: &str = "vokra.reazonspeech_nemo_v2.model_config_sha256";
const KEY_TENSOR_MANIFEST_SHA256: &str = "vokra.reazonspeech_nemo_v2.tensor_manifest_sha256";
const KEY_TOKENIZER_VOCAB: &str = "vokra.reazonspeech_nemo_v2.tokenizer.vocab";
const KEY_TOKENIZER_VOCAB_SHA256: &str = "vokra.reazonspeech_nemo_v2.tokenizer.vocab_sha256";

const KEY_SAMPLE_RATE: &str = "vokra.reazonspeech_nemo_v2.sample_rate";
const KEY_ENC_N_LAYER: &str = "vokra.reazonspeech_nemo_v2.encoder.n_layer";
const KEY_ENC_D_MODEL: &str = "vokra.reazonspeech_nemo_v2.encoder.d_model";
const KEY_ENC_N_HEAD: &str = "vokra.reazonspeech_nemo_v2.encoder.n_head";
const KEY_ENC_FFN_DIM: &str = "vokra.reazonspeech_nemo_v2.encoder.ffn_dim";
const KEY_ENC_CONV_KERNEL: &str = "vokra.reazonspeech_nemo_v2.encoder.conv_kernel_size";
const KEY_ENC_N_MELS: &str = "vokra.reazonspeech_nemo_v2.encoder.n_mels";
const KEY_ENC_SUB_FACTOR: &str = "vokra.reazonspeech_nemo_v2.encoder.subsampling_factor";
const KEY_ENC_SUB_CHANNELS: &str = "vokra.reazonspeech_nemo_v2.encoder.subsampling_channels";
const KEY_ENC_MAX_POS: &str = "vokra.reazonspeech_nemo_v2.encoder.max_position_embeddings";
const KEY_ENC_LEFT_CONTEXT: &str = "vokra.reazonspeech_nemo_v2.encoder.left_context";
const KEY_ENC_RIGHT_CONTEXT: &str = "vokra.reazonspeech_nemo_v2.encoder.right_context";
const KEY_ENC_GLOBAL_TOKENS: &str = "vokra.reazonspeech_nemo_v2.encoder.global_tokens";
const KEY_ENC_GLOBAL_SPACING: &str = "vokra.reazonspeech_nemo_v2.encoder.global_tokens_spacing";
const KEY_DEC_N_LAYER: &str = "vokra.reazonspeech_nemo_v2.decoder.n_layer";
const KEY_DEC_D_MODEL: &str = "vokra.reazonspeech_nemo_v2.decoder.d_model";
const KEY_JOINT_VOCAB_SIZE: &str = "vokra.reazonspeech_nemo_v2.joint.vocab_size";
const KEY_JOINT_BLANK_ID: &str = "vokra.reazonspeech_nemo_v2.joint.blank_token_id";
const KEY_JOINT_MAX_SYMBOLS: &str = "vokra.reazonspeech_nemo_v2.joint.max_symbols_per_step";

const KEY_FRONTEND_N_FFT: &str = "vokra.frontend.n_fft";
const KEY_FRONTEND_HOP: &str = "vokra.frontend.hop_length";
const KEY_FRONTEND_WIN: &str = "vokra.frontend.win_length";
const KEY_FRONTEND_WINDOW: &str = "vokra.frontend.window_type";
const KEY_FRONTEND_N_MELS: &str = "vokra.frontend.n_mels";
const KEY_FRONTEND_NORMALIZE: &str = "vokra.frontend.normalize";
const KEY_FRONTEND_DITHER: &str = "vokra.frontend.dither";

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
/// Counters from one strict complete-checkpoint conversion.
pub struct ReazonspeechNemoV2Report {
    /// Input tensor descriptors read after exact manifest validation.
    pub read: usize,
    /// Exact released F32 tensors written to GGUF.
    pub written: usize,
    /// Always zero for the strict official manifest.
    pub skipped_non_float: usize,
    /// Always zero because the official NeMo state dict is F32.
    pub bf16_passthrough: usize,
}

/// The compatibility entry point fails before reading the checkpoint because
/// a runnable text-ASR artifact requires the exact tokenizer vocabulary.
pub fn convert_reazonspeech_nemo_v2_file(
    _input: &Path,
    _output: &Path,
    _license: Option<&str>,
) -> Result<ReazonspeechNemoV2Report, ConvertError> {
    Err(ConvertError::Usage(
        "reazonspeech-nemo-v2 requires the exact official tokenizer.vocab; use convert_reazonspeech_nemo_v2_file_with_tokenizer (CLI: --tokenizer tokenizer.vocab)"
            .to_owned(),
    ))
}

/// Converts the complete prepared release and embeds its exact decode-only
/// SentencePiece vocabulary. The source safetensors is approximately 2.48 GB;
/// this whole-file path is VAST-only under `AGENTS.md`.
pub fn convert_reazonspeech_nemo_v2_file_with_tokenizer(
    input: &Path,
    output: &Path,
    license: Option<&str>,
    tokenizer_vocab: &Path,
) -> Result<ReazonspeechNemoV2Report, ConvertError> {
    if let Some(value) = license.filter(|value| !value.is_empty()) {
        if !value.eq_ignore_ascii_case(DEFAULT_LICENSE_SPDX) {
            return Err(ConvertError::Usage(format!(
                "reazonspeech-nemo-v2 is pinned to the official {DEFAULT_LICENSE_SPDX} checkpoint; refusing license override {value:?}"
            )));
        }
    }

    let tokenizer = std::fs::read(tokenizer_vocab).map_err(ConvertError::Io)?;
    validate_tokenizer_vocab(&tokenizer)?;

    let bytes = std::fs::read(input).map_err(ConvertError::Io)?;
    let checkpoint = SafetensorsFile::parse(bytes)?;
    validate_checkpoint(&checkpoint)?;

    let mut builder = GgufBuilder::new();
    builder.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    builder.add_string(chunks::KEY_MODEL_NAME, NAME);
    builder.add_string(KEY_MODEL_CATEGORY, CATEGORY);
    builder.add_string(KEY_UPSTREAM_HF, UPSTREAM_HF);
    write_runtime_metadata(&mut builder, &tokenizer);
    vokra_core::stamp_provenance(
        &mut builder,
        LicenseClass::Permissive,
        DEFAULT_LICENSE_SPDX,
        Some(NAME),
        Some("https://huggingface.co/reazon-research/reazonspeech-nemo-v2"),
    );

    for tensor in checkpoint.tensors() {
        builder
            .add_tensor(
                &tensor.name,
                tensor.dtype,
                tensor.shape.clone(),
                checkpoint.tensor_bytes(tensor).to_vec(),
            )
            .map_err(|error| ConvertError::Gguf(error.to_string()))?;
    }
    let output_bytes = builder
        .to_bytes()
        .map_err(|error| ConvertError::Gguf(error.to_string()))?;
    std::fs::write(output, output_bytes).map_err(ConvertError::Io)?;

    Ok(ReazonspeechNemoV2Report {
        read: TENSOR_COUNT,
        written: TENSOR_COUNT,
        skipped_non_float: 0,
        bf16_passthrough: 0,
    })
}

fn write_runtime_metadata(builder: &mut GgufBuilder, tokenizer: &[u8]) {
    for (key, value) in [
        (KEY_SOURCE_REVISION, UPSTREAM_REVISION),
        (KEY_SOURCE_NEMO_SHA256, SOURCE_NEMO_SHA256),
        (KEY_SOURCE_TAR_MANIFEST_SHA256, SOURCE_TAR_MANIFEST_SHA256),
        (KEY_MODEL_CONFIG_SHA256, MODEL_CONFIG_SHA256),
        (KEY_TENSOR_MANIFEST_SHA256, TENSOR_MANIFEST_SHA256),
        (KEY_FRONTEND_WINDOW, "hann"),
        (KEY_FRONTEND_NORMALIZE, "per_feature"),
    ] {
        builder.add_string(key, value);
    }
    for (key, value) in [
        (KEY_SAMPLE_RATE, 16_000),
        (KEY_ENC_N_LAYER, 24),
        (KEY_ENC_D_MODEL, 1_024),
        (KEY_ENC_N_HEAD, 8),
        (KEY_ENC_FFN_DIM, 4_096),
        (KEY_ENC_CONV_KERNEL, 9),
        (KEY_ENC_N_MELS, 80),
        (KEY_ENC_SUB_FACTOR, 8),
        (KEY_ENC_SUB_CHANNELS, 256),
        (KEY_ENC_MAX_POS, 5_000),
        (KEY_ENC_LEFT_CONTEXT, 128),
        (KEY_ENC_RIGHT_CONTEXT, 128),
        (KEY_ENC_GLOBAL_TOKENS, 1),
        (KEY_ENC_GLOBAL_SPACING, 1),
        (KEY_DEC_N_LAYER, 2),
        (KEY_DEC_D_MODEL, 640),
        (KEY_JOINT_VOCAB_SIZE, 3_001),
        (KEY_JOINT_BLANK_ID, 3_000),
        (KEY_JOINT_MAX_SYMBOLS, 10),
        (KEY_FRONTEND_N_FFT, 512),
        (KEY_FRONTEND_HOP, 160),
        (KEY_FRONTEND_WIN, 400),
        (KEY_FRONTEND_N_MELS, 80),
    ] {
        builder.add_u32(key, value);
    }
    builder.add_f32(KEY_FRONTEND_DITHER, 1.0e-5);
    builder.add_metadata(
        KEY_TOKENIZER_VOCAB,
        GgufMetadataValue::Array(GgufArray {
            element_type: GgufValueType::U8,
            values: tokenizer
                .iter()
                .map(|&byte| GgufMetadataValue::U8(byte))
                .collect(),
        }),
    );
    builder.add_string(KEY_TOKENIZER_VOCAB_SHA256, TOKENIZER_VOCAB_SHA256);
}

fn validate_checkpoint(checkpoint: &SafetensorsFile) -> Result<(), ConvertError> {
    let expected = expected_manifest();
    let internal_hash =
        super::canary_1b_flash::hex(&super::canary_1b_flash::manifest_sha256(&expected));
    if expected.len() != TENSOR_COUNT || internal_hash != TENSOR_MANIFEST_SHA256 {
        return Err(ConvertError::Parse(format!(
            "ReazonSpeech-NeMo-v2 internal manifest drift: count={}, sha256={internal_hash}",
            expected.len()
        )));
    }

    let actual_names = checkpoint
        .tensors()
        .iter()
        .map(|tensor| tensor.name.as_str())
        .collect::<BTreeSet<_>>();
    let expected_names = expected.keys().map(String::as_str).collect::<BTreeSet<_>>();
    if actual_names != expected_names {
        let missing = expected_names
            .difference(&actual_names)
            .take(8)
            .copied()
            .collect::<Vec<_>>();
        let extra = actual_names
            .difference(&expected_names)
            .take(8)
            .copied()
            .collect::<Vec<_>>();
        return Err(ConvertError::Parse(format!(
            "ReazonSpeech-NeMo-v2 prepared checkpoint manifest mismatch: found {}, expected {TENSOR_COUNT}; missing={missing:?}, extra={extra:?}",
            checkpoint.tensors().len()
        )));
    }
    for tensor in checkpoint.tensors() {
        let expected_shape = &expected[&tensor.name];
        if &tensor.shape != expected_shape {
            return Err(ConvertError::Parse(format!(
                "ReazonSpeech-NeMo-v2 tensor {:?} shape {:?}, expected {:?}",
                tensor.name, tensor.shape, expected_shape
            )));
        }
        if tensor.dtype != GgmlType::F32 {
            return Err(ConvertError::Parse(format!(
                "ReazonSpeech-NeMo-v2 tensor {:?} dtype {:?}, expected the pinned F32 `.nemo` payload",
                tensor.name, tensor.dtype
            )));
        }
    }
    Ok(())
}

fn expected_manifest() -> BTreeMap<String, Vec<u64>> {
    let mut tensors = BTreeMap::new();
    let mut add = |name: String, shape: &[u64]| {
        assert!(tensors.insert(name, shape.to_vec()).is_none());
    };

    add("preprocessor.featurizer.fb".into(), &[1, 80, 257]);
    add("preprocessor.featurizer.window".into(), &[400]);
    add("encoder.pre_encode.conv.0.weight".into(), &[256, 1, 3, 3]);
    add("encoder.pre_encode.conv.0.bias".into(), &[256]);
    for index in [2, 5] {
        add(
            format!("encoder.pre_encode.conv.{index}.weight"),
            &[256, 1, 3, 3],
        );
        add(format!("encoder.pre_encode.conv.{index}.bias"), &[256]);
    }
    for index in [3, 6] {
        add(
            format!("encoder.pre_encode.conv.{index}.weight"),
            &[256, 256, 1, 1],
        );
        add(format!("encoder.pre_encode.conv.{index}.bias"), &[256]);
    }
    add("encoder.pre_encode.out.weight".into(), &[1_024, 2_560]);
    add("encoder.pre_encode.out.bias".into(), &[1_024]);

    for layer in 0..24 {
        let prefix = format!("encoder.layers.{layer}");
        for branch in ["feed_forward1", "feed_forward2"] {
            add(format!("{prefix}.{branch}.linear1.weight"), &[4_096, 1_024]);
            add(format!("{prefix}.{branch}.linear1.bias"), &[4_096]);
            add(format!("{prefix}.{branch}.linear2.weight"), &[1_024, 4_096]);
            add(format!("{prefix}.{branch}.linear2.bias"), &[1_024]);
        }
        for norm in [
            "norm_feed_forward1",
            "norm_self_att",
            "norm_conv",
            "norm_feed_forward2",
            "norm_out",
        ] {
            add(format!("{prefix}.{norm}.weight"), &[1_024]);
            add(format!("{prefix}.{norm}.bias"), &[1_024]);
        }
        for projection in ["linear_q", "linear_k", "linear_v", "linear_out"] {
            add(
                format!("{prefix}.self_attn.{projection}.weight"),
                &[1_024, 1_024],
            );
            add(format!("{prefix}.self_attn.{projection}.bias"), &[1_024]);
        }
        add(
            format!("{prefix}.self_attn.linear_pos.weight"),
            &[1_024, 1_024],
        );
        add(format!("{prefix}.self_attn.pos_bias_u"), &[8, 128]);
        add(format!("{prefix}.self_attn.pos_bias_v"), &[8, 128]);
        add(
            format!("{prefix}.conv.pointwise_conv1.weight"),
            &[2_048, 1_024, 1],
        );
        add(format!("{prefix}.conv.pointwise_conv1.bias"), &[2_048]);
        add(
            format!("{prefix}.conv.depthwise_conv.weight"),
            &[1_024, 1, 9],
        );
        add(format!("{prefix}.conv.depthwise_conv.bias"), &[1_024]);
        for name in ["weight", "bias", "running_mean", "running_var"] {
            add(format!("{prefix}.conv.batch_norm.{name}"), &[1_024]);
        }
        add(
            format!("{prefix}.conv.pointwise_conv2.weight"),
            &[1_024, 1_024, 1],
        );
        add(format!("{prefix}.conv.pointwise_conv2.bias"), &[1_024]);
    }

    add("decoder.prediction.embed.weight".into(), &[3_001, 640]);
    for layer in 0..2 {
        let prefix = "decoder.prediction.dec_rnn.lstm";
        add(format!("{prefix}.weight_ih_l{layer}"), &[2_560, 640]);
        add(format!("{prefix}.weight_hh_l{layer}"), &[2_560, 640]);
        add(format!("{prefix}.bias_ih_l{layer}"), &[2_560]);
        add(format!("{prefix}.bias_hh_l{layer}"), &[2_560]);
    }
    add("joint.enc.weight".into(), &[640, 1_024]);
    add("joint.enc.bias".into(), &[640]);
    add("joint.pred.weight".into(), &[640, 640]);
    add("joint.pred.bias".into(), &[640]);
    add("joint.joint_net.2.weight".into(), &[3_001, 640]);
    add("joint.joint_net.2.bias".into(), &[3_001]);

    tensors
}

fn validate_tokenizer_vocab(bytes: &[u8]) -> Result<(), ConvertError> {
    let actual = super::canary_1b_flash::hex(&super::canary_1b_flash::sha256(bytes));
    if actual != TOKENIZER_VOCAB_SHA256 {
        return Err(ConvertError::Parse(format!(
            "ReazonSpeech-NeMo-v2 tokenizer SHA-256 {actual}, expected {TOKENIZER_VOCAB_SHA256}"
        )));
    }
    let document = std::str::from_utf8(bytes).map_err(|error| {
        ConvertError::Parse(format!(
            "ReazonSpeech-NeMo-v2 tokenizer is not UTF-8: {error}"
        ))
    })?;
    let mut pieces = 0usize;
    for (index, line) in document.lines().enumerate() {
        let (piece, score) = line.rsplit_once('\t').ok_or_else(|| {
            ConvertError::Parse(format!(
                "ReazonSpeech-NeMo-v2 tokenizer line {} is not `piece<TAB>score`",
                index + 1
            ))
        })?;
        let score = score.parse::<f32>().map_err(|error| {
            ConvertError::Parse(format!(
                "ReazonSpeech-NeMo-v2 tokenizer line {} score: {error}",
                index + 1
            ))
        })?;
        if piece.is_empty() || !score.is_finite() {
            return Err(ConvertError::Parse(format!(
                "ReazonSpeech-NeMo-v2 tokenizer line {} is malformed",
                index + 1
            )));
        }
        if index == 0 && piece != "<unk>" {
            return Err(ConvertError::Parse(
                "ReazonSpeech-NeMo-v2 tokenizer does not begin with `<unk>`".to_owned(),
            ));
        }
        pieces += 1;
    }
    if pieces != TOKENIZER_PIECES {
        return Err(ConvertError::Parse(format!(
            "ReazonSpeech-NeMo-v2 tokenizer has {pieces} pieces, expected {TOKENIZER_PIECES}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn official_manifest_count_and_hash_are_pinned() {
        let manifest = expected_manifest();
        assert_eq!(manifest.len(), TENSOR_COUNT);
        assert_eq!(
            super::super::canary_1b_flash::hex(&super::super::canary_1b_flash::manifest_sha256(
                &manifest
            )),
            TENSOR_MANIFEST_SHA256
        );
        assert_eq!(manifest["preprocessor.featurizer.fb"], vec![1, 80, 257]);
        assert_eq!(
            manifest["encoder.layers.23.self_attn.linear_pos.weight"],
            vec![1_024, 1_024]
        );
        assert_eq!(
            manifest["decoder.prediction.dec_rnn.lstm.weight_hh_l1"],
            vec![2_560, 640]
        );
        assert_eq!(manifest["joint.joint_net.2.weight"], vec![3_001, 640]);
    }

    #[test]
    fn tokenizer_less_entry_is_explicit_error() {
        let error = convert_reazonspeech_nemo_v2_file(
            Path::new("unused.safetensors"),
            Path::new("unused.gguf"),
            None,
        )
        .expect_err("tokenizer-less conversion must fail");
        assert!(error.to_string().contains("--tokenizer"));
    }

    #[test]
    fn wrong_tokenizer_hash_is_rejected_before_structure() {
        let error = validate_tokenizer_vocab(b"<unk>\t0\n").expect_err("wrong hash");
        assert!(error.to_string().contains("SHA-256"));
    }

    #[test]
    fn runtime_metadata_axes_are_complete() {
        let mut builder = GgufBuilder::new();
        write_runtime_metadata(&mut builder, b"fixture");
        let bytes = builder.to_bytes().expect("metadata-only GGUF");
        let file = vokra_core::gguf::GgufFile::parse(bytes).expect("parse GGUF");
        assert_eq!(
            file.get(KEY_SOURCE_REVISION)
                .and_then(|value| value.as_str()),
            Some(UPSTREAM_REVISION)
        );
        assert_eq!(
            file.get(KEY_MODEL_CONFIG_SHA256)
                .and_then(|value| value.as_str()),
            Some(MODEL_CONFIG_SHA256)
        );
        assert!(file.get(KEY_TOKENIZER_VOCAB).is_some());
        assert_eq!(
            file.get(KEY_JOINT_BLANK_ID),
            Some(&GgufMetadataValue::U32(3_000))
        );
        assert_eq!(
            file.get(KEY_FRONTEND_DITHER),
            Some(&GgufMetadataValue::F32(1.0e-5))
        );
    }
}
