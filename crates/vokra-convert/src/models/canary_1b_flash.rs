//! NVIDIA Canary-1B-Flash complete `.nemo` checkpoint → executable GGUF.
//!
//! The canonical input is the F32 safetensors file produced from the pinned
//! upstream `.nemo` on VAST. Conversion is fail-closed: all 1,374 inference
//! tensors, their exact names/shapes, and the exact five-tokenizer aggregate
//! vocabulary must be present. The historical 1,292-tensor encoder-only HF
//! safetensors/GGUF is rejected because it cannot run ASR/AST.
//!
//! This path reads and writes multi-gigabyte artifacts and is VAST-only under
//! `AGENTS.md`. Python/torch are preparation-time tools only; runtime loading
//! and inference remain native Rust with no ONNX or third-party runtime.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{
    GgmlType, GgufArray, GgufBuilder, GgufMetadataValue, GgufValueType, chunks,
};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

pub const ARCH: &str = "canary-1b-flash";
pub const NAME: &str = "canary-1b-flash";
pub const CATEGORY: &str = "asr";
pub const UPSTREAM_HF: &str = "nvidia/canary-1b-flash";
pub const UPSTREAM_REVISION: &str = "2b6e4d2dacb11cc1b1724de31bb48fe68c26c12e";
pub const SOURCE_NEMO_SHA256: &str =
    "3887cce1afdd425429cfc5109575a8f2cffeb07c02c503a9faff7612bd74e324";
pub const MODEL_CONFIG_SHA256: &str =
    "42d71aebc1f4b9f387a20902db71e00128b324ff5156bdac63897e1afad55ff9";
pub const DATA_PICKLE_SHA256: &str =
    "a60784f60aa5cea26d3c11d62c3ed7270e5c7bf52844d99b553656d9498a3617";
pub const TENSOR_MANIFEST_SHA256: &str =
    "f76f4c3d28147b418705c8272a81dab53425e3bd264b8a2040ffb0de03385cb6";
pub const TOKENIZER_VOCAB_SHA256: &str =
    "08cb29d15437dbd3f45c26046c2f5994b3b92c86a3aa4a6e27d253d40837db79";
pub const DEFAULT_LICENSE: &str = "cc-by-4.0";

pub const CANARY_1B_FLASH_ATTRIBUTION_TEXT: &str = "This application uses NVIDIA Canary-1B-Flash (multilingual ASR / AST for English, German, Spanish and French). Model weights are licensed under CC-BY 4.0. Copyright (c) NVIDIA. Source: https://huggingface.co/nvidia/canary-1b-flash";

const TENSOR_COUNT: usize = 1_374;
const VOCAB_SIZE: usize = 5_248;
const SPECIAL_VOCAB_SIZE: usize = 1_152;
const LANGUAGE_VOCAB_SIZE: usize = 1_024;

const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
const KEY_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";
const KEY_SOURCE_REVISION: &str = "vokra.canary_1b_flash.source_revision";
const KEY_SOURCE_NEMO_SHA256: &str = "vokra.canary_1b_flash.source_nemo_sha256";
const KEY_MODEL_CONFIG_SHA256: &str = "vokra.canary_1b_flash.model_config_sha256";
const KEY_DATA_PICKLE_SHA256: &str = "vokra.canary_1b_flash.data_pickle_sha256";
const KEY_TENSOR_MANIFEST_SHA256: &str = "vokra.canary_1b_flash.tensor_manifest_sha256";
const KEY_TOKENIZER_VOCAB: &str = "vokra.canary_1b_flash.tokenizer.vocab";
const KEY_TOKENIZER_VOCAB_SHA256: &str = "vokra.canary_1b_flash.tokenizer.vocab_sha256";

const KEY_SAMPLE_RATE: &str = "vokra.canary_1b_flash.sample_rate";
const KEY_ENC_N_LAYER: &str = "vokra.canary_1b_flash.arch.encoder.n_layer";
const KEY_ENC_D_MODEL: &str = "vokra.canary_1b_flash.arch.encoder.d_model";
const KEY_ENC_N_HEAD: &str = "vokra.canary_1b_flash.arch.encoder.n_head";
const KEY_ENC_N_HEAD_KV: &str = "vokra.canary_1b_flash.arch.encoder.n_head_kv";
const KEY_ENC_FFN_DIM: &str = "vokra.canary_1b_flash.arch.encoder.ffn_dim";
const KEY_ENC_CONV_KERNEL: &str = "vokra.canary_1b_flash.arch.encoder.conv_kernel_size";
const KEY_ENC_IN_DIM: &str = "vokra.canary_1b_flash.arch.encoder.in_dim";
const KEY_ENC_SUB_FACTOR: &str = "vokra.canary_1b_flash.arch.encoder.subsampling_factor";
const KEY_ENC_SUB_KERNEL: &str = "vokra.canary_1b_flash.arch.encoder.subsampling_conv_kernel_size";
const KEY_ENC_SUB_STRIDE: &str = "vokra.canary_1b_flash.arch.encoder.subsampling_conv_stride";
const KEY_ENC_SUB_CHANNELS: &str = "vokra.canary_1b_flash.arch.encoder.subsampling_conv_channels";
const KEY_ENC_MAX_POS: &str = "vokra.canary_1b_flash.arch.encoder.max_position_embeddings";
const KEY_ENC_ATTN_BIAS: &str = "vokra.canary_1b_flash.arch.encoder.attention_bias";
const KEY_ENC_CONV_BIAS: &str = "vokra.canary_1b_flash.arch.encoder.convolution_bias";
const KEY_ENC_SCALE_INPUT: &str = "vokra.canary_1b_flash.arch.encoder.scale_input";
const KEY_DEC_N_LAYER: &str = "vokra.canary_1b_flash.arch.decoder.n_layer";
const KEY_DEC_D_MODEL: &str = "vokra.canary_1b_flash.arch.decoder.d_model";
const KEY_DEC_N_HEAD: &str = "vokra.canary_1b_flash.arch.decoder.n_head";
const KEY_DEC_FFN_DIM: &str = "vokra.canary_1b_flash.arch.decoder.ffn_dim";
const KEY_DEC_MAX_SEQ: &str = "vokra.canary_1b_flash.arch.decoder.max_sequence_length";
const KEY_DEC_PRE_LN: &str = "vokra.canary_1b_flash.arch.decoder.pre_ln";
const KEY_DEC_HIDDEN_ACT: &str = "vokra.canary_1b_flash.arch.decoder.hidden_act";
const KEY_HEAD_VOCAB_SIZE: &str = "vokra.canary_1b_flash.head.vocab_size";
const KEY_HEAD_PAD_ID: &str = "vokra.canary_1b_flash.head.pad_token_id";
const KEY_HEAD_BOS_ID: &str = "vokra.canary_1b_flash.head.bos_token_id";
const KEY_HEAD_EOS_ID: &str = "vokra.canary_1b_flash.head.eos_token_id";

const KEY_FRONTEND_N_FFT: &str = "vokra.canary_1b_flash.frontend.n_fft";
const KEY_FRONTEND_HOP: &str = "vokra.canary_1b_flash.frontend.hop_length";
const KEY_FRONTEND_WIN: &str = "vokra.canary_1b_flash.frontend.win_length";
const KEY_FRONTEND_N_MELS: &str = "vokra.canary_1b_flash.frontend.n_mels";
const KEY_FRONTEND_PREEMPHASIS: &str = "vokra.canary_1b_flash.frontend.preemphasis";
const KEY_FRONTEND_WINDOW: &str = "vokra.canary_1b_flash.frontend.window";
const KEY_FRONTEND_WINDOW_PERIODIC: &str = "vokra.canary_1b_flash.frontend.window_periodic";
const KEY_FRONTEND_NORMALIZE: &str = "vokra.canary_1b_flash.frontend.normalize";
const KEY_FRONTEND_PAD_MODE: &str = "vokra.canary_1b_flash.frontend.pad_mode";

#[derive(Debug, Default, Clone, PartialEq, Eq)]
/// Counters from one strict complete-checkpoint conversion.
pub struct Canary1bFlashReport {
    /// Input tensor descriptors read after manifest validation.
    pub read: usize,
    /// Exact released F32 tensors written to GGUF.
    pub written: usize,
    /// Always zero for the strict released manifest.
    pub skipped_non_float: usize,
    /// Always zero because the pinned `.nemo` payload is F32.
    pub bf16_passthrough: usize,
}

/// Legacy three-path entry point. A runnable Canary artifact cannot be made
/// without the authenticated aggregate vocabulary, so this path fails before
/// touching the multi-gigabyte checkpoint.
pub fn convert_canary_1b_flash_file(
    _input: &Path,
    _output: &Path,
    _license: Option<&str>,
) -> Result<Canary1bFlashReport, ConvertError> {
    Err(ConvertError::Usage(
        "canary-1b-flash requires the exact five-tokenizer aggregate vocabulary; use convert_canary_1b_flash_file_with_tokenizer (CLI: --tokenizer canary-1b-flash.aggregate.vocab)"
            .to_owned(),
    ))
}

/// Converts the complete prepared release and embeds its exact aggregate
/// tokenizer. This multi-gigabyte path must run on VAST.
pub fn convert_canary_1b_flash_file_with_tokenizer(
    input: &Path,
    output: &Path,
    license: Option<&str>,
    tokenizer_vocab: &Path,
) -> Result<Canary1bFlashReport, ConvertError> {
    if let Some(value) = license.filter(|value| !value.is_empty()) {
        if !value.eq_ignore_ascii_case(DEFAULT_LICENSE) {
            return Err(ConvertError::Usage(format!(
                "canary-1b-flash is pinned to the official {DEFAULT_LICENSE} checkpoint; refusing license override {value:?}"
            )));
        }
    }

    let tokenizer = std::fs::read(tokenizer_vocab).map_err(ConvertError::Io)?;
    validate_tokenizer_vocab(&tokenizer)?;

    // VAST-only by repository policy: source safetensors is ~3.54 GB.
    let bytes = std::fs::read(input).map_err(ConvertError::Io)?;
    let st = SafetensorsFile::parse(bytes)?;
    validate_checkpoint(&st)?;

    let mut builder = GgufBuilder::new();
    builder.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    builder.add_string(chunks::KEY_MODEL_NAME, NAME);
    builder.add_string(KEY_MODEL_CATEGORY, CATEGORY);
    builder.add_string(KEY_UPSTREAM_HF, UPSTREAM_HF);
    write_runtime_metadata(&mut builder, &tokenizer);
    vokra_core::stamp_provenance(
        &mut builder,
        LicenseClass::AttributionRequired,
        DEFAULT_LICENSE,
        Some(NAME),
        Some("https://huggingface.co/nvidia/canary-1b-flash"),
    );
    vokra_core::stamp_attribution(&mut builder, CANARY_1B_FLASH_ATTRIBUTION_TEXT);

    for tensor in st.tensors() {
        builder
            .add_tensor(
                &tensor.name,
                tensor.dtype,
                tensor.shape.clone(),
                st.tensor_bytes(tensor).to_vec(),
            )
            .map_err(|error| ConvertError::Gguf(error.to_string()))?;
    }
    let output_bytes = builder
        .to_bytes()
        .map_err(|error| ConvertError::Gguf(error.to_string()))?;
    std::fs::write(output, output_bytes).map_err(ConvertError::Io)?;

    Ok(Canary1bFlashReport {
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
        (KEY_MODEL_CONFIG_SHA256, MODEL_CONFIG_SHA256),
        (KEY_DATA_PICKLE_SHA256, DATA_PICKLE_SHA256),
        (KEY_TENSOR_MANIFEST_SHA256, TENSOR_MANIFEST_SHA256),
        (KEY_DEC_HIDDEN_ACT, "relu"),
        (KEY_FRONTEND_WINDOW, "hann"),
        (KEY_FRONTEND_NORMALIZE, "per_feature"),
        (KEY_FRONTEND_PAD_MODE, "constant"),
    ] {
        builder.add_string(key, value);
    }
    for (key, value) in [
        (KEY_SAMPLE_RATE, 16_000),
        (KEY_ENC_N_LAYER, 32),
        (KEY_ENC_D_MODEL, 1_024),
        (KEY_ENC_N_HEAD, 8),
        (KEY_ENC_N_HEAD_KV, 8),
        (KEY_ENC_FFN_DIM, 4_096),
        (KEY_ENC_CONV_KERNEL, 9),
        (KEY_ENC_IN_DIM, 128),
        (KEY_ENC_SUB_FACTOR, 8),
        (KEY_ENC_SUB_KERNEL, 3),
        (KEY_ENC_SUB_STRIDE, 2),
        (KEY_ENC_SUB_CHANNELS, 256),
        (KEY_ENC_MAX_POS, 5_000),
        (KEY_ENC_ATTN_BIAS, 1),
        (KEY_ENC_CONV_BIAS, 1),
        (KEY_ENC_SCALE_INPUT, 0),
        (KEY_DEC_N_LAYER, 4),
        (KEY_DEC_D_MODEL, 1_024),
        (KEY_DEC_N_HEAD, 8),
        (KEY_DEC_FFN_DIM, 4_096),
        (KEY_DEC_MAX_SEQ, 1_024),
        (KEY_DEC_PRE_LN, 1),
        (KEY_HEAD_VOCAB_SIZE, 5_248),
        (KEY_HEAD_PAD_ID, 2),
        (KEY_HEAD_BOS_ID, 4),
        (KEY_HEAD_EOS_ID, 3),
        (KEY_FRONTEND_N_FFT, 512),
        (KEY_FRONTEND_HOP, 160),
        (KEY_FRONTEND_WIN, 400),
        (KEY_FRONTEND_N_MELS, 128),
        (KEY_FRONTEND_WINDOW_PERIODIC, 0),
    ] {
        builder.add_u32(key, value);
    }
    builder.add_f32(KEY_FRONTEND_PREEMPHASIS, 0.97);
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

fn validate_checkpoint(st: &SafetensorsFile) -> Result<(), ConvertError> {
    let expected = expected_manifest();
    let internal_hash = hex(&manifest_sha256(&expected));
    if expected.len() != TENSOR_COUNT || internal_hash != TENSOR_MANIFEST_SHA256 {
        return Err(ConvertError::Parse(format!(
            "Canary-1B-Flash internal manifest drift: count={}, sha256={internal_hash}",
            expected.len()
        )));
    }
    let actual_names = st
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
            "Canary-1B-Flash prepared checkpoint manifest mismatch: found {}, expected {TENSOR_COUNT}; missing={missing:?}, extra={extra:?}. The public 1,292-tensor encoder-only artifact is not executable",
            st.tensors().len()
        )));
    }
    for tensor in st.tensors() {
        let shape = &expected[&tensor.name];
        if &tensor.shape != shape {
            return Err(ConvertError::Parse(format!(
                "Canary-1B-Flash tensor {:?} shape {:?}, expected {:?}",
                tensor.name, tensor.shape, shape
            )));
        }
        if tensor.dtype != GgmlType::F32 {
            return Err(ConvertError::Parse(format!(
                "Canary-1B-Flash tensor {:?} dtype {:?}, expected the pinned F32 `.nemo` payload",
                tensor.name, tensor.dtype
            )));
        }
    }
    Ok(())
}

fn expected_manifest() -> BTreeMap<String, Vec<u64>> {
    expected_canary_aed_manifest(4, 5_248)
}

/// Shared exact FastConformer + Transformer-AED tensor-name contract.
/// Canary-1B-Flash and Canary-1B-v2 differ only by decoder depth and
/// vocabulary width; their official state-dict manifests prove every other
/// name and shape is identical.
pub(crate) fn expected_canary_aed_manifest(
    decoder_layers: usize,
    vocab_size: u64,
) -> BTreeMap<String, Vec<u64>> {
    let mut tensors = BTreeMap::new();
    let mut add = |name: String, shape: &[u64]| {
        assert!(tensors.insert(name, shape.to_vec()).is_none());
    };

    add("preprocessor.featurizer.fb".into(), &[1, 128, 257]);
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
    add("encoder.pre_encode.out.weight".into(), &[1_024, 4_096]);
    add("encoder.pre_encode.out.bias".into(), &[1_024]);

    for layer in 0..32 {
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
        for value in ["weight", "bias", "running_mean", "running_var"] {
            add(format!("{prefix}.conv.batch_norm.{value}"), &[1_024]);
        }
        add(
            format!("{prefix}.conv.pointwise_conv2.weight"),
            &[1_024, 1_024, 1],
        );
        add(format!("{prefix}.conv.pointwise_conv2.bias"), &[1_024]);
    }

    add(
        "transf_decoder._embedding.token_embedding.weight".into(),
        &[vocab_size, 1_024],
    );
    add(
        "transf_decoder._embedding.position_embedding.pos_enc".into(),
        &[1_024, 1_024],
    );
    add(
        "transf_decoder._embedding.layer_norm.weight".into(),
        &[1_024],
    );
    add("transf_decoder._embedding.layer_norm.bias".into(), &[1_024]);
    for layer in 0..decoder_layers {
        let prefix = format!("transf_decoder._decoder.layers.{layer}");
        for norm in ["layer_norm_1", "layer_norm_2", "layer_norm_3"] {
            add(format!("{prefix}.{norm}.weight"), &[1_024]);
            add(format!("{prefix}.{norm}.bias"), &[1_024]);
        }
        for sublayer in ["first_sub_layer", "second_sub_layer"] {
            for projection in ["query_net", "key_net", "value_net", "out_projection"] {
                add(
                    format!("{prefix}.{sublayer}.{projection}.weight"),
                    &[1_024, 1_024],
                );
                add(format!("{prefix}.{sublayer}.{projection}.bias"), &[1_024]);
            }
        }
        add(
            format!("{prefix}.third_sub_layer.dense_in.weight"),
            &[4_096, 1_024],
        );
        add(format!("{prefix}.third_sub_layer.dense_in.bias"), &[4_096]);
        add(
            format!("{prefix}.third_sub_layer.dense_out.weight"),
            &[1_024, 4_096],
        );
        add(format!("{prefix}.third_sub_layer.dense_out.bias"), &[1_024]);
    }
    add(
        "transf_decoder._decoder.final_layer_norm.weight".into(),
        &[1_024],
    );
    add(
        "transf_decoder._decoder.final_layer_norm.bias".into(),
        &[1_024],
    );
    add("log_softmax.mlp.layer0.weight".into(), &[vocab_size, 1_024]);
    add("log_softmax.mlp.layer0.bias".into(), &[vocab_size]);
    tensors
}

fn validate_tokenizer_vocab(bytes: &[u8]) -> Result<(), ConvertError> {
    let actual = hex(&sha256(bytes));
    if actual != TOKENIZER_VOCAB_SHA256 {
        return Err(ConvertError::Parse(format!(
            "Canary-1B-Flash aggregate tokenizer SHA-256 {actual}, expected {TOKENIZER_VOCAB_SHA256}"
        )));
    }
    let document = std::str::from_utf8(bytes).map_err(|error| {
        ConvertError::Parse(format!(
            "Canary-1B-Flash aggregate tokenizer is not UTF-8: {error}"
        ))
    })?;
    let mut pieces = Vec::with_capacity(VOCAB_SIZE);
    for (index, line) in document.lines().enumerate() {
        let (piece, score) = line.rsplit_once('\t').ok_or_else(|| {
            ConvertError::Parse(format!(
                "Canary-1B-Flash aggregate tokenizer line {} is not `piece<TAB>score`",
                index + 1
            ))
        })?;
        let score = score.parse::<f32>().map_err(|error| {
            ConvertError::Parse(format!(
                "Canary-1B-Flash aggregate tokenizer line {} score: {error}",
                index + 1
            ))
        })?;
        if piece.is_empty() || !score.is_finite() {
            return Err(ConvertError::Parse(format!(
                "Canary-1B-Flash aggregate tokenizer line {} is malformed",
                index + 1
            )));
        }
        pieces.push(piece);
    }
    if pieces.len() != VOCAB_SIZE {
        return Err(ConvertError::Parse(format!(
            "Canary-1B-Flash aggregate tokenizer has {} pieces, expected {VOCAB_SIZE}",
            pieces.len()
        )));
    }
    for offset in [
        0,
        SPECIAL_VOCAB_SIZE,
        SPECIAL_VOCAB_SIZE + LANGUAGE_VOCAB_SIZE,
        SPECIAL_VOCAB_SIZE + 2 * LANGUAGE_VOCAB_SIZE,
        SPECIAL_VOCAB_SIZE + 3 * LANGUAGE_VOCAB_SIZE,
    ] {
        if pieces[offset] != "<unk>" {
            return Err(ConvertError::Parse(format!(
                "Canary-1B-Flash aggregate tokenizer component {offset} does not begin with `<unk>`"
            )));
        }
    }
    Ok(())
}

pub(crate) fn manifest_sha256(manifest: &BTreeMap<String, Vec<u64>>) -> [u8; 32] {
    let mut canonical = Vec::new();
    for (name, dimensions) in manifest {
        canonical.extend_from_slice(name.as_bytes());
        canonical.push(0);
        canonical.extend_from_slice(&(dimensions.len() as u64).to_le_bytes());
        for dimension in dimensions {
            canonical.extend_from_slice(&dimension.to_le_bytes());
        }
    }
    sha256(&canonical)
}

pub(crate) fn hex(bytes: &[u8; 32]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(64);
    for byte in bytes {
        output.push(char::from(DIGITS[(byte >> 4) as usize]));
        output.push(char::from(DIGITS[(byte & 0x0f) as usize]));
    }
    output
}

const SHA256_K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

pub(crate) fn sha256(data: &[u8]) -> [u8; 32] {
    let mut state = [
        0x6a09e667u32,
        0xbb67ae85,
        0x3c6ef372,
        0xa54ff53a,
        0x510e527f,
        0x9b05688c,
        0x1f83d9ab,
        0x5be0cd19,
    ];
    let bit_len = (data.len() as u64) * 8;
    let mut padded = Vec::with_capacity(data.len() + 72);
    padded.extend_from_slice(data);
    padded.push(0x80);
    while padded.len() % 64 != 56 {
        padded.push(0);
    }
    padded.extend_from_slice(&bit_len.to_be_bytes());
    for block in padded.chunks_exact(64) {
        let mut words = [0u32; 64];
        for (index, word) in block.chunks_exact(4).take(16).enumerate() {
            words[index] = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
        }
        for index in 16..64 {
            let s0 = words[index - 15].rotate_right(7)
                ^ words[index - 15].rotate_right(18)
                ^ (words[index - 15] >> 3);
            let s1 = words[index - 2].rotate_right(17)
                ^ words[index - 2].rotate_right(19)
                ^ (words[index - 2] >> 10);
            words[index] = words[index - 16]
                .wrapping_add(s0)
                .wrapping_add(words[index - 7])
                .wrapping_add(s1);
        }
        let mut work = state;
        for index in 0..64 {
            let sum1 =
                work[4].rotate_right(6) ^ work[4].rotate_right(11) ^ work[4].rotate_right(25);
            let choice = (work[4] & work[5]) ^ (!work[4] & work[6]);
            let temp1 = work[7]
                .wrapping_add(sum1)
                .wrapping_add(choice)
                .wrapping_add(SHA256_K[index])
                .wrapping_add(words[index]);
            let sum0 =
                work[0].rotate_right(2) ^ work[0].rotate_right(13) ^ work[0].rotate_right(22);
            let majority = (work[0] & work[1]) ^ (work[0] & work[2]) ^ (work[1] & work[2]);
            let temp2 = sum0.wrapping_add(majority);
            work = [
                temp1.wrapping_add(temp2),
                work[0],
                work[1],
                work[2],
                work[3].wrapping_add(temp1),
                work[4],
                work[5],
                work[6],
            ];
        }
        for (value, delta) in state.iter_mut().zip(work) {
            *value = value.wrapping_add(delta);
        }
    }
    let mut output = [0u8; 32];
    for (chunk, value) in output.chunks_exact_mut(4).zip(state) {
        chunk.copy_from_slice(&value.to_be_bytes());
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn official_manifest_count_and_hash_are_pinned() {
        let manifest = expected_manifest();
        assert_eq!(manifest.len(), TENSOR_COUNT);
        assert_eq!(hex(&manifest_sha256(&manifest)), TENSOR_MANIFEST_SHA256);
        assert_eq!(manifest["preprocessor.featurizer.fb"], vec![1, 128, 257]);
        assert_eq!(
            manifest["transf_decoder._decoder.layers.3.third_sub_layer.dense_out.weight"],
            vec![1_024, 4_096]
        );
        assert_eq!(
            manifest["log_softmax.mlp.layer0.weight"],
            vec![5_248, 1_024]
        );
    }

    #[test]
    fn sha256_matches_nist_abc_vector() {
        assert_eq!(
            hex(&sha256(b"abc")),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn legacy_tokenizer_less_entry_is_explicit_error() {
        let error = convert_canary_1b_flash_file(
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
}
