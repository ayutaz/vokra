//! Pinned NVIDIA Parakeet-CTC-1.1B safetensors → GGUF conversion.
//!
//! The authenticated upstream revision contains 1,652 F32 inference tensors
//! plus 42 I64 `num_batches_tracked` training counters.  The repository's
//! prepare script verifies the official checkpoint SHA-256 and removes exactly
//! those counters; this converter then requires the complete 1,652-tensor
//! inference manifest, the exact config/preprocessor contracts and the
//! official BPE + Metaspace tokenizer.  It never accepts a shape-like sibling.

use std::collections::BTreeSet;

use vokra_core::LicenseClass;
use vokra_core::gguf::{
    GgmlType, GgufArray, GgufBuilder, GgufMetadataValue, GgufValueType, chunks,
};
use vokra_core::json::{self, JsonValue};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

pub(crate) const ARCH: &str = "parakeet-ctc";
pub(crate) const NAME: &str = "parakeet-ctc-1.1b";
pub(crate) const UPSTREAM_HF: &str = "nvidia/parakeet-ctc-1.1b";
pub(crate) const UPSTREAM_REVISION: &str = "20e63a0fed6aedba145b74b826dbd41df0941730";
pub(crate) const UPSTREAM_SOURCE_REVISION: &str = "d56c55bf564ddb176759eb6ec199442682564916";
pub(crate) const CONFIG_SHA256: &str =
    "c33a8ddbf447d68d31b2f1d1e4efa061548813b7647913e67560a9b198f06ae1";
pub(crate) const PREPROCESSOR_SHA256: &str =
    "7f26808482a58d8dd187c4b87364810292b91ed7721e099bdbb05ca50da37a98";
pub(crate) const TOKENIZER_SHA256: &str =
    "f3f1dd45c3889ed2b5bf67180caf05f51d7d7e4948c20e5f24d8c24df9cc47aa";
pub(crate) const CHECKPOINT_SHA256: &str =
    "57e0bc26772f3360b7ae0c087f184364179906674d08fc8b71d48a54d4f52145";

pub(crate) const PARAKEET_CTC_ATTRIBUTION_TEXT: &str = "This application uses NVIDIA Parakeet-CTC-1.1B (English ASR — FastConformer encoder + CTC decoder). Model weights are licensed under CC-BY 4.0 (attribution required; commercial use permitted). Copyright (c) NVIDIA. Source: https://huggingface.co/nvidia/parakeet-ctc-1.1b";

const KEY_SAMPLE_RATE: &str = "vokra.parakeet_ctc.sample_rate";
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
const KEY_HEAD_VOCAB_SIZE: &str = "vokra.parakeet_ctc.head.vocab_size";
const KEY_HEAD_PAD_ID: &str = "vokra.parakeet_ctc.head.pad_token_id";
const KEY_REVISION: &str = "vokra.parakeet_ctc.revision";
const KEY_SOURCE_REVISION: &str = "vokra.parakeet_ctc.source_revision";
const KEY_CONFIG_SHA256: &str = "vokra.parakeet_ctc.config_sha256";
const KEY_PREPROCESSOR_SHA256: &str = "vokra.parakeet_ctc.preprocessor_sha256";
const KEY_TOKENIZER_SHA256: &str = "vokra.parakeet_ctc.tokenizer_sha256";
const KEY_CHECKPOINT_SHA256: &str = "vokra.parakeet_ctc.checkpoint_sha256";
const KEY_TOKENIZER_JSON: &str = "vokra.parakeet.tokenizer.json";

const SAMPLE_RATE: u32 = 16_000;
const N_LAYER: usize = 42;
const D_MODEL: usize = 1_024;
const N_HEAD: usize = 8;
const FFN_DIM: usize = 4_096;
const CONV_KERNEL: usize = 9;
const N_MELS: usize = 80;
const SUBSAMPLING_FACTOR: usize = 8;
const SUBSAMPLING_CHANNELS: usize = 256;
const SUBSAMPLING_KERNEL: usize = 3;
const SUBSAMPLING_STRIDE: usize = 2;
const MAX_POSITIONS: usize = 5_000;
const VOCAB_SIZE: usize = 1_025;
const PAD_BLANK_ID: u32 = 1_024;
const INFERENCE_TENSORS: usize = 1_652;

#[derive(Debug, Default)]
pub(crate) struct ParakeetCtcReport {
    pub(crate) written: usize,
    pub(crate) skipped_non_float: usize,
    pub(crate) bf16_passthrough: usize,
    pub(crate) notes: Vec<String>,
}

fn require_u64(value: &JsonValue, key: &str, expected: u64) -> Result<(), ConvertError> {
    let actual = value.get(key).and_then(JsonValue::as_u64);
    if actual != Some(expected) {
        return Err(ConvertError::Parse(format!(
            "Parakeet-CTC sidecar `{key}` must be {expected}, found {actual:?}"
        )));
    }
    Ok(())
}

fn require_str(value: &JsonValue, key: &str, expected: &str) -> Result<(), ConvertError> {
    let actual = value.get(key).and_then(JsonValue::as_str);
    if actual != Some(expected) {
        return Err(ConvertError::Parse(format!(
            "Parakeet-CTC sidecar `{key}` must be {expected:?}, found {actual:?}"
        )));
    }
    Ok(())
}

fn require_bool(value: &JsonValue, key: &str, expected: bool) -> Result<(), ConvertError> {
    let actual = match value.get(key) {
        Some(JsonValue::Bool(value)) => Some(*value),
        _ => None,
    };
    if actual != Some(expected) {
        return Err(ConvertError::Parse(format!(
            "Parakeet-CTC sidecar `{key}` must be {expected}, found {actual:?}"
        )));
    }
    Ok(())
}

fn validate_config(bytes: &[u8]) -> Result<(), ConvertError> {
    let root = json::parse(bytes)
        .map_err(|error| ConvertError::Parse(format!("Parakeet-CTC config.json: {error}")))?;
    require_str(&root, "model_type", "parakeet_ctc")?;
    require_u64(&root, "vocab_size", VOCAB_SIZE as u64)?;
    require_u64(&root, "pad_token_id", PAD_BLANK_ID as u64)?;
    require_bool(&root, "ctc_zero_infinity", true)?;
    require_str(&root, "ctc_loss_reduction", "mean")?;
    let architectures = root
        .get("architectures")
        .and_then(JsonValue::as_array)
        .ok_or_else(|| {
            ConvertError::Parse("Parakeet-CTC config: architectures must be an array".into())
        })?;
    if architectures.len() != 1 || architectures[0].as_str() != Some("ParakeetForCTC") {
        return Err(ConvertError::Parse(
            "Parakeet-CTC config: architectures must be [\"ParakeetForCTC\"]".into(),
        ));
    }
    let encoder = root
        .get("encoder_config")
        .ok_or_else(|| ConvertError::Parse("Parakeet-CTC config: missing encoder_config".into()))?;
    require_str(encoder, "model_type", "parakeet_encoder")?;
    for (key, expected) in [
        ("num_hidden_layers", N_LAYER as u64),
        ("hidden_size", D_MODEL as u64),
        ("num_attention_heads", N_HEAD as u64),
        ("num_key_value_heads", N_HEAD as u64),
        ("intermediate_size", FFN_DIM as u64),
        ("conv_kernel_size", CONV_KERNEL as u64),
        ("num_mel_bins", N_MELS as u64),
        ("subsampling_factor", SUBSAMPLING_FACTOR as u64),
        ("subsampling_conv_channels", SUBSAMPLING_CHANNELS as u64),
        ("subsampling_conv_kernel_size", SUBSAMPLING_KERNEL as u64),
        ("subsampling_conv_stride", SUBSAMPLING_STRIDE as u64),
        ("max_position_embeddings", MAX_POSITIONS as u64),
    ] {
        require_u64(encoder, key, expected)?;
    }
    require_str(encoder, "hidden_act", "silu")?;
    require_bool(encoder, "attention_bias", true)?;
    require_bool(encoder, "scale_input", true)?;
    // `convolution_bias` is omitted by the 2025 config and therefore uses
    // the pinned Transformers source default `True` at UPSTREAM_SOURCE_REVISION.
    Ok(())
}

fn validate_preprocessor(bytes: &[u8]) -> Result<(), ConvertError> {
    let root = json::parse(bytes).map_err(|error| {
        ConvertError::Parse(format!("Parakeet-CTC preprocessor_config.json: {error}"))
    })?;
    for (key, expected) in [
        ("feature_size", N_MELS as u64),
        ("hop_length", 160),
        ("n_fft", 512),
        ("sampling_rate", SAMPLE_RATE as u64),
        ("win_length", 400),
    ] {
        require_u64(&root, key, expected)?;
    }
    require_str(&root, "feature_extractor_type", "ParakeetFeatureExtractor")?;
    require_str(&root, "processor_class", "ParakeetProcessor")?;
    require_str(&root, "padding_side", "right")?;
    require_bool(&root, "return_attention_mask", true)?;
    match root.get("preemphasis") {
        Some(JsonValue::Float(value)) if (*value - 0.97).abs() < f64::EPSILON => {}
        other => {
            return Err(ConvertError::Parse(format!(
                "Parakeet-CTC preprocessor `preemphasis` must be 0.97, found {other:?}"
            )));
        }
    }
    Ok(())
}

fn validate_tokenizer(bytes: &[u8]) -> Result<(), ConvertError> {
    let root = json::parse(bytes)
        .map_err(|error| ConvertError::Parse(format!("Parakeet-CTC tokenizer.json: {error}")))?;
    let model = root
        .get("model")
        .ok_or_else(|| ConvertError::Parse("Parakeet-CTC tokenizer: missing model".into()))?;
    require_str(model, "type", "BPE")?;
    let vocab = model
        .get("vocab")
        .and_then(JsonValue::as_object)
        .ok_or_else(|| ConvertError::Parse("Parakeet-CTC tokenizer: missing model.vocab".into()))?;
    if vocab.len() != 1_024 {
        return Err(ConvertError::Parse(format!(
            "Parakeet-CTC tokenizer must contain 1024 BPE entries, found {}",
            vocab.len()
        )));
    }
    let decoder = root
        .get("decoder")
        .ok_or_else(|| ConvertError::Parse("Parakeet-CTC tokenizer: missing decoder".into()))?;
    require_str(decoder, "type", "Metaspace")?;
    require_str(decoder, "replacement", "▁")?;
    require_str(decoder, "prepend_scheme", "always")?;
    let added = root
        .get("added_tokens")
        .and_then(JsonValue::as_array)
        .ok_or_else(|| {
            ConvertError::Parse("Parakeet-CTC tokenizer: missing added_tokens".into())
        })?;
    let has_pad = added.iter().any(|entry| {
        entry.get("id").and_then(JsonValue::as_u64) == Some(PAD_BLANK_ID as u64)
            && entry.get("content").and_then(JsonValue::as_str) == Some("<pad>")
            && matches!(entry.get("special"), Some(JsonValue::Bool(true)))
    });
    if !has_pad {
        return Err(ConvertError::Parse(
            "Parakeet-CTC tokenizer must define special <pad> id 1024 (the CTC blank)".into(),
        ));
    }
    Ok(())
}

fn expected_manifest() -> Vec<(String, Vec<u64>)> {
    let mut manifest = Vec::with_capacity(INFERENCE_TENSORS);
    manifest.push((
        "encoder.subsampling.layers.0.weight".into(),
        vec![SUBSAMPLING_CHANNELS as u64, 1, 3, 3],
    ));
    manifest.push((
        "encoder.subsampling.layers.0.bias".into(),
        vec![SUBSAMPLING_CHANNELS as u64],
    ));
    for (depthwise, pointwise) in [(2, 3), (5, 6)] {
        manifest.push((
            format!("encoder.subsampling.layers.{depthwise}.weight"),
            vec![SUBSAMPLING_CHANNELS as u64, 1, 3, 3],
        ));
        manifest.push((
            format!("encoder.subsampling.layers.{depthwise}.bias"),
            vec![SUBSAMPLING_CHANNELS as u64],
        ));
        manifest.push((
            format!("encoder.subsampling.layers.{pointwise}.weight"),
            vec![
                SUBSAMPLING_CHANNELS as u64,
                SUBSAMPLING_CHANNELS as u64,
                1,
                1,
            ],
        ));
        manifest.push((
            format!("encoder.subsampling.layers.{pointwise}.bias"),
            vec![SUBSAMPLING_CHANNELS as u64],
        ));
    }
    manifest.push((
        "encoder.subsampling.linear.weight".into(),
        vec![D_MODEL as u64, (SUBSAMPLING_CHANNELS * 10) as u64],
    ));
    manifest.push((
        "encoder.subsampling.linear.bias".into(),
        vec![D_MODEL as u64],
    ));
    for layer in 0..N_LAYER {
        let prefix = format!("encoder.layers.{layer}");
        for branch in ["feed_forward1", "feed_forward2"] {
            for (linear, shape_w, shape_b) in [
                (
                    1,
                    vec![FFN_DIM as u64, D_MODEL as u64],
                    vec![FFN_DIM as u64],
                ),
                (
                    2,
                    vec![D_MODEL as u64, FFN_DIM as u64],
                    vec![D_MODEL as u64],
                ),
            ] {
                manifest.push((format!("{prefix}.{branch}.linear{linear}.weight"), shape_w));
                manifest.push((format!("{prefix}.{branch}.linear{linear}.bias"), shape_b));
            }
        }
        for norm in [
            "norm_feed_forward1",
            "norm_self_att",
            "norm_conv",
            "norm_feed_forward2",
            "norm_out",
        ] {
            manifest.push((format!("{prefix}.{norm}.weight"), vec![D_MODEL as u64]));
            manifest.push((format!("{prefix}.{norm}.bias"), vec![D_MODEL as u64]));
        }
        for projection in ["q_proj", "k_proj", "v_proj", "o_proj"] {
            manifest.push((
                format!("{prefix}.self_attn.{projection}.weight"),
                vec![D_MODEL as u64, D_MODEL as u64],
            ));
            manifest.push((
                format!("{prefix}.self_attn.{projection}.bias"),
                vec![D_MODEL as u64],
            ));
        }
        manifest.push((
            format!("{prefix}.self_attn.relative_k_proj.weight"),
            vec![D_MODEL as u64, D_MODEL as u64],
        ));
        for bias in ["bias_u", "bias_v"] {
            manifest.push((
                format!("{prefix}.self_attn.{bias}"),
                vec![N_HEAD as u64, (D_MODEL / N_HEAD) as u64],
            ));
        }
        manifest.push((
            format!("{prefix}.conv.pointwise_conv1.weight"),
            vec![(2 * D_MODEL) as u64, D_MODEL as u64, 1],
        ));
        manifest.push((
            format!("{prefix}.conv.pointwise_conv1.bias"),
            vec![(2 * D_MODEL) as u64],
        ));
        manifest.push((
            format!("{prefix}.conv.depthwise_conv.weight"),
            vec![D_MODEL as u64, 1, CONV_KERNEL as u64],
        ));
        manifest.push((
            format!("{prefix}.conv.depthwise_conv.bias"),
            vec![D_MODEL as u64],
        ));
        for stat in ["weight", "bias", "running_mean", "running_var"] {
            manifest.push((format!("{prefix}.conv.norm.{stat}"), vec![D_MODEL as u64]));
        }
        manifest.push((
            format!("{prefix}.conv.pointwise_conv2.weight"),
            vec![D_MODEL as u64, D_MODEL as u64, 1],
        ));
        manifest.push((
            format!("{prefix}.conv.pointwise_conv2.bias"),
            vec![D_MODEL as u64],
        ));
    }
    manifest.push((
        "ctc_head.weight".into(),
        vec![VOCAB_SIZE as u64, D_MODEL as u64, 1],
    ));
    manifest.push(("ctc_head.bias".into(), vec![VOCAB_SIZE as u64]));
    manifest
}

pub(crate) fn convert(_bytes: Vec<u8>) -> Result<(GgufBuilder, ParakeetCtcReport), ConvertError> {
    Err(ConvertError::Usage(
        "parakeet-ctc requires the exact config.json, preprocessor_config.json and tokenizer.json; use convert_parakeet_ctc_file_with_assets (CLI: --config, --preprocessor, --tokenizer)"
            .into(),
    ))
}

pub(crate) fn convert_with_assets(
    bytes: Vec<u8>,
    config: &[u8],
    preprocessor: &[u8],
    tokenizer: &[u8],
) -> Result<(GgufBuilder, ParakeetCtcReport), ConvertError> {
    validate_config(config)?;
    validate_preprocessor(preprocessor)?;
    validate_tokenizer(tokenizer)?;
    let checkpoint = SafetensorsFile::parse(bytes)?;
    let manifest = expected_manifest();
    if manifest.len() != INFERENCE_TENSORS {
        return Err(ConvertError::Parse(format!(
            "Parakeet-CTC internal manifest has {} tensors, expected {INFERENCE_TENSORS}",
            manifest.len()
        )));
    }
    let expected_names: BTreeSet<String> = manifest.iter().map(|(name, _)| name.clone()).collect();
    for (name, shape) in &manifest {
        let tensor = checkpoint.tensor_info(name).ok_or_else(|| {
            ConvertError::Parse(format!("Parakeet-CTC required tensor `{name}` is missing"))
        })?;
        if tensor.dtype != GgmlType::F32 || tensor.shape != *shape {
            return Err(ConvertError::Parse(format!(
                "Parakeet-CTC tensor `{name}` has dtype {:?}, shape {:?}; expected F32 {shape:?}",
                tensor.dtype, tensor.shape
            )));
        }
    }
    let actual_names: BTreeSet<String> = checkpoint
        .tensors()
        .iter()
        .map(|tensor| tensor.name.clone())
        .collect();
    if actual_names != expected_names {
        let missing: Vec<&String> = expected_names.difference(&actual_names).take(4).collect();
        let extra: Vec<&String> = actual_names.difference(&expected_names).take(4).collect();
        return Err(ConvertError::Parse(format!(
            "Parakeet-CTC prepared manifest mismatch: expected {}, found {}; missing={missing:?}, extra={extra:?}",
            expected_names.len(),
            actual_names.len()
        )));
    }

    let mut builder = GgufBuilder::new();
    builder.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    builder.add_string(chunks::KEY_MODEL_NAME, NAME);
    builder.add_u32(KEY_SAMPLE_RATE, SAMPLE_RATE);
    for (key, value) in [
        (KEY_ENC_N_LAYER, N_LAYER as u32),
        (KEY_ENC_D_MODEL, D_MODEL as u32),
        (KEY_ENC_N_HEAD, N_HEAD as u32),
        (KEY_ENC_N_HEAD_KV, N_HEAD as u32),
        (KEY_ENC_FFN_DIM, FFN_DIM as u32),
        (KEY_ENC_CONV_KERNEL, CONV_KERNEL as u32),
        (KEY_ENC_IN_DIM, N_MELS as u32),
        (KEY_ENC_SUBSAMPLING_FACTOR, SUBSAMPLING_FACTOR as u32),
        (KEY_ENC_SUB_CONV_KERNEL, SUBSAMPLING_KERNEL as u32),
        (KEY_ENC_SUB_CONV_STRIDE, SUBSAMPLING_STRIDE as u32),
        (KEY_ENC_SUB_CONV_CHANNELS, SUBSAMPLING_CHANNELS as u32),
        (KEY_ENC_MAX_POS, MAX_POSITIONS as u32),
        (KEY_ENC_ATTN_BIAS, 1),
        (KEY_ENC_CONV_BIAS, 1),
        (KEY_ENC_SCALE_INPUT, 1),
        (KEY_HEAD_VOCAB_SIZE, VOCAB_SIZE as u32),
        (KEY_HEAD_PAD_ID, PAD_BLANK_ID),
    ] {
        builder.add_u32(key, value);
    }
    for (key, value) in [
        (KEY_REVISION, UPSTREAM_REVISION),
        (KEY_SOURCE_REVISION, UPSTREAM_SOURCE_REVISION),
        (KEY_CONFIG_SHA256, CONFIG_SHA256),
        (KEY_PREPROCESSOR_SHA256, PREPROCESSOR_SHA256),
        (KEY_TOKENIZER_SHA256, TOKENIZER_SHA256),
        (KEY_CHECKPOINT_SHA256, CHECKPOINT_SHA256),
    ] {
        builder.add_string(key, value);
    }
    builder.add_metadata(
        KEY_TOKENIZER_JSON,
        GgufMetadataValue::Array(GgufArray {
            element_type: GgufValueType::U8,
            values: tokenizer
                .iter()
                .copied()
                .map(GgufMetadataValue::U8)
                .collect(),
        }),
    );
    vokra_core::stamp_provenance(
        &mut builder,
        LicenseClass::AttributionRequired,
        "CC-BY-4.0",
        Some(UPSTREAM_HF),
        Some("https://huggingface.co/nvidia/parakeet-ctc-1.1b"),
    );
    vokra_core::stamp_attribution(&mut builder, PARAKEET_CTC_ATTRIBUTION_TEXT);
    for tensor in checkpoint.tensors() {
        builder.add_tensor(
            &tensor.name,
            tensor.dtype,
            tensor.shape.clone(),
            checkpoint.tensor_bytes(tensor).to_vec(),
        )?;
    }
    Ok((
        builder,
        ParakeetCtcReport {
            written: INFERENCE_TENSORS,
            skipped_non_float: 0,
            bf16_passthrough: 0,
            notes: vec![
                "strict official manifest: 1652 F32 inference tensors; 42 training-only I64 BatchNorm counters removed by the pinned prepare script"
                    .into(),
            ],
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn official_manifest_has_exact_count_and_biases() {
        let manifest = expected_manifest();
        assert_eq!(manifest.len(), INFERENCE_TENSORS);
        assert!(
            manifest
                .iter()
                .any(|(name, _)| name == "encoder.layers.41.self_attn.q_proj.bias")
        );
        assert!(
            manifest
                .iter()
                .any(|(name, _)| name == "encoder.layers.41.conv.depthwise_conv.bias")
        );
        assert_eq!(manifest.last().unwrap().0, "ctc_head.bias");
    }

    #[test]
    fn legacy_conversion_refuses_missing_assets() {
        let error = convert(Vec::new()).unwrap_err();
        assert!(error.to_string().contains("preprocessor_config.json"));
    }
}
