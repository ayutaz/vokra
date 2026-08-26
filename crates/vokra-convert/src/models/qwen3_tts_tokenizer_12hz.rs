//! Strict decode-only conversion for `Qwen/Qwen3-TTS-Tokenizer-12Hz`.
//!
//! Qwen3-TTS main checkpoints emit sixteen codebook rows at 12.5 Hz.  Those
//! rows are not PCM and the older `qwen3_tts_codec` table fold is not a
//! waveform decoder.  The official tokenizer release contains a separate
//! 496-tensor model: 225 encoder tensors and 271 decoder tensors.  TTS needs
//! only the decoder half, so this converter authenticates the complete
//! official file and emits the exact 271 `decoder.*` tensors.
//!
//! The checkpoint is pinned to HF revision
//! `a87c50897bb00837eb857d0538b29d117541d7f6`; the current model-card-only
//! tip `7dd38ad4e9bad454aae9cd937d0cd577604fe229` does not change the weight.
//! Whole-file SHA-256 authentication deliberately rejects repacks and future
//! upstream topology changes rather than silently accepting a decoder that
//! only happens to share some tensor names.

use std::collections::BTreeMap;

use vokra_core::LicenseClass;
use vokra_core::gguf::{
    GgmlType, GgufArray, GgufBuilder, GgufMetadataValue, GgufValueType, chunks,
};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

#[cfg(test)]
use super::canary_1b_flash::manifest_sha256;
use super::canary_1b_flash::{hex, sha256};

pub(crate) const ARCH: &str = "qwen3_tts_tokenizer_12hz";
pub(crate) const NAME: &str = "qwen3-tts-tokenizer-12hz-decoder";
pub(crate) const UPSTREAM_HF: &str = "Qwen/Qwen3-TTS-Tokenizer-12Hz";
pub(crate) const UPSTREAM_REVISION: &str = "a87c50897bb00837eb857d0538b29d117541d7f6";
pub(crate) const CURRENT_REPOSITORY_REVISION: &str = "7dd38ad4e9bad454aae9cd937d0cd577604fe229";
pub(crate) const SOURCE_REVISION: &str = "022e286b98fbec7e1e916cb940cdf532cd9f488e";
pub(crate) const CHECKPOINT_SHA256: &str =
    "836b7b357f5ea43e889936a3709af68dfe3751881acefe4ecf0dbd30ba571258";
pub(crate) const CONFIG_SHA256: &str =
    "ee65bb901c876664ab8707c487157aa1a6ee57c65969b28fb5ec9dc211e68167";
pub(crate) const MODELING_SOURCE_SHA256: &str =
    "844e8dd8c0182ef9c6463c874631c22ef3c5a4fd1899dd657016164cc5379628";
pub(crate) const CONFIGURATION_SOURCE_SHA256: &str =
    "9e30c24394b00cb0366d7da3482b7436468acc1cd3da1a6fe614a1d34653a5e3";
pub(crate) const DECODER_MANIFEST_SHA256: &str =
    "501397728761b1d97763ec1817f5b36dbbf0132ba272bf8756999b8b1e7f8803";

pub(crate) const INPUT_TENSOR_COUNT: usize = 496;
pub(crate) const ENCODER_TENSOR_COUNT: usize = 225;
pub(crate) const DECODER_TENSOR_COUNT: usize = 271;

const KEY_INPUT_SAMPLE_RATE: &str = "vokra.qwen3_tts_tokenizer_12hz.input_sample_rate";
const KEY_OUTPUT_SAMPLE_RATE: &str = "vokra.qwen3_tts_tokenizer_12hz.output_sample_rate";
const KEY_DECODE_UPSAMPLE_RATE: &str = "vokra.qwen3_tts_tokenizer_12hz.decode_upsample_rate";
const KEY_NUM_QUANTIZERS: &str = "vokra.qwen3_tts_tokenizer_12hz.num_quantizers";
const KEY_NUM_SEMANTIC_QUANTIZERS: &str = "vokra.qwen3_tts_tokenizer_12hz.num_semantic_quantizers";
const KEY_CODEBOOK_SIZE: &str = "vokra.qwen3_tts_tokenizer_12hz.codebook_size";
const KEY_CONFIGURED_SEMANTIC_VOCAB_SIZE: &str =
    "vokra.qwen3_tts_tokenizer_12hz.configured_semantic_vocab_size";
const KEY_CODEBOOK_DIM: &str = "vokra.qwen3_tts_tokenizer_12hz.codebook_dim";
const KEY_QUANTIZER_DIM: &str = "vokra.qwen3_tts_tokenizer_12hz.quantizer_dim";
const KEY_LATENT_DIM: &str = "vokra.qwen3_tts_tokenizer_12hz.latent_dim";
const KEY_HIDDEN_SIZE: &str = "vokra.qwen3_tts_tokenizer_12hz.transformer.hidden_size";
const KEY_INTERMEDIATE_SIZE: &str = "vokra.qwen3_tts_tokenizer_12hz.transformer.intermediate_size";
const KEY_NUM_HIDDEN_LAYERS: &str = "vokra.qwen3_tts_tokenizer_12hz.transformer.num_hidden_layers";
const KEY_NUM_ATTENTION_HEADS: &str =
    "vokra.qwen3_tts_tokenizer_12hz.transformer.num_attention_heads";
const KEY_NUM_KEY_VALUE_HEADS: &str =
    "vokra.qwen3_tts_tokenizer_12hz.transformer.num_key_value_heads";
const KEY_HEAD_DIM: &str = "vokra.qwen3_tts_tokenizer_12hz.transformer.head_dim";
const KEY_RMS_NORM_EPS: &str = "vokra.qwen3_tts_tokenizer_12hz.transformer.rms_norm_eps";
const KEY_ROPE_THETA: &str = "vokra.qwen3_tts_tokenizer_12hz.transformer.rope_theta";
const KEY_SLIDING_WINDOW: &str = "vokra.qwen3_tts_tokenizer_12hz.transformer.sliding_window";
const KEY_LAYER_SCALE_INITIAL: &str =
    "vokra.qwen3_tts_tokenizer_12hz.transformer.layer_scale_initial";
const KEY_DECODER_DIM: &str = "vokra.qwen3_tts_tokenizer_12hz.decoder_dim";
const KEY_UPSAMPLING_RATIOS: &str = "vokra.qwen3_tts_tokenizer_12hz.upsampling_ratios";
const KEY_UPSAMPLE_RATES: &str = "vokra.qwen3_tts_tokenizer_12hz.upsample_rates";
const KEY_CHUNK_SIZE: &str = "vokra.qwen3_tts_tokenizer_12hz.chunk_size";
const KEY_LEFT_CONTEXT: &str = "vokra.qwen3_tts_tokenizer_12hz.left_context";
const KEY_CHECKPOINT_SHA256: &str = "vokra.provenance.checkpoint_sha256";
const KEY_UPSTREAM_REVISION: &str = "vokra.provenance.upstream_revision";
const KEY_REPOSITORY_REVISION: &str = "vokra.provenance.repository_revision";
const KEY_CONFIG_SHA256: &str = "vokra.qwen3_tts_tokenizer_12hz.config_sha256";
const KEY_SOURCE_REVISION: &str = "vokra.qwen3_tts_tokenizer_12hz.source_revision";
const KEY_MODELING_SOURCE_SHA256: &str = "vokra.qwen3_tts_tokenizer_12hz.modeling_source_sha256";
const KEY_CONFIGURATION_SOURCE_SHA256: &str =
    "vokra.qwen3_tts_tokenizer_12hz.configuration_source_sha256";
const KEY_MANIFEST_SHA256: &str = "vokra.qwen3_tts_tokenizer_12hz.decoder_manifest_sha256";

const INPUT_SAMPLE_RATE: u32 = 24_000;
const OUTPUT_SAMPLE_RATE: u32 = 24_000;
const DECODE_UPSAMPLE_RATE: u32 = 1_920;
const NUM_QUANTIZERS: u32 = 16;
const NUM_SEMANTIC_QUANTIZERS: u32 = 1;
// The official decoder constructor passes `config.codebook_size` to both
// RVQ branches. `semantic_codebook_size = 4096` is retained as source
// metadata but is not the size of `rvq_first` in this checkpoint.
const CODEBOOK_SIZE: u32 = 2_048;
const CONFIGURED_SEMANTIC_VOCAB_SIZE: u32 = 4_096;
const CODEBOOK_DIM: u32 = 512;
const QUANTIZER_DIM: u32 = 256;
const LATENT_DIM: u32 = 1_024;
const HIDDEN_SIZE: u32 = 512;
const INTERMEDIATE_SIZE: u32 = 1_024;
const NUM_HIDDEN_LAYERS: u32 = 8;
const NUM_ATTENTION_HEADS: u32 = 16;
const NUM_KEY_VALUE_HEADS: u32 = 16;
const HEAD_DIM: u32 = 64;
const RMS_NORM_EPS: f32 = 1e-5;
const ROPE_THETA: f32 = 10_000.0;
const SLIDING_WINDOW: u32 = 72;
const LAYER_SCALE_INITIAL: f32 = 0.01;
const DECODER_DIM: u32 = 1_536;
const UPSAMPLING_RATIOS: [u32; 2] = [2, 2];
const UPSAMPLE_RATES: [u32; 4] = [8, 5, 4, 3];
const CHUNK_SIZE: u32 = 300;
const LEFT_CONTEXT: u32 = 25;

#[derive(Debug, Default)]
pub(crate) struct Qwen3TtsTokenizer12HzReport {
    pub(crate) written: usize,
    pub(crate) stripped_encoder: usize,
}

pub(crate) fn convert(
    bytes: Vec<u8>,
) -> Result<(GgufBuilder, Qwen3TtsTokenizer12HzReport), ConvertError> {
    let actual_sha = hex(&sha256(&bytes));
    if actual_sha != CHECKPOINT_SHA256 {
        return Err(ConvertError::Parse(format!(
            "qwen3-tts-tokenizer-12hz: checkpoint SHA-256 mismatch for {UPSTREAM_HF}@{UPSTREAM_REVISION}/model.safetensors: expected {CHECKPOINT_SHA256}, found {actual_sha}"
        )));
    }

    let st = SafetensorsFile::parse(bytes)?;
    if st.tensors().len() != INPUT_TENSOR_COUNT {
        return Err(ConvertError::Parse(format!(
            "qwen3-tts-tokenizer-12hz: authenticated checkpoint must contain {INPUT_TENSOR_COUNT} tensors, found {}",
            st.tensors().len()
        )));
    }

    validate_decoder_manifest(&st)?;
    let encoder_count = st
        .tensors()
        .iter()
        .filter(|tensor| tensor.name.starts_with("encoder."))
        .count();
    if encoder_count != ENCODER_TENSOR_COUNT {
        return Err(ConvertError::Parse(format!(
            "qwen3-tts-tokenizer-12hz: authenticated checkpoint must contain {ENCODER_TENSOR_COUNT} encoder tensors, found {encoder_count}"
        )));
    }

    let mut builder = metadata_builder();
    let mut report = Qwen3TtsTokenizer12HzReport {
        written: 0,
        stripped_encoder: encoder_count,
    };
    for tensor in st
        .tensors()
        .iter()
        .filter(|tensor| tensor.name.starts_with("decoder."))
    {
        if tensor.dtype != GgmlType::F32 {
            return Err(ConvertError::Parse(format!(
                "qwen3-tts-tokenizer-12hz: decoder tensor {:?} has {:?}; the pinned official contract is F32-only",
                tensor.name, tensor.dtype
            )));
        }
        builder.add_tensor(
            &tensor.name,
            tensor.dtype,
            tensor.shape.clone(),
            st.tensor_bytes(tensor).to_vec(),
        )?;
        report.written += 1;
    }
    debug_assert_eq!(report.written, DECODER_TENSOR_COUNT);
    Ok((builder, report))
}

fn metadata_builder() -> GgufBuilder {
    let mut builder = GgufBuilder::new();
    builder.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    builder.add_string(chunks::KEY_MODEL_NAME, NAME);
    builder.add_string("vokra.model.category", "codec");
    builder.add_string(KEY_UPSTREAM_REVISION, UPSTREAM_REVISION);
    builder.add_string(KEY_REPOSITORY_REVISION, CURRENT_REPOSITORY_REVISION);
    builder.add_string(KEY_CHECKPOINT_SHA256, CHECKPOINT_SHA256);
    builder.add_string(KEY_CONFIG_SHA256, CONFIG_SHA256);
    builder.add_string(KEY_SOURCE_REVISION, SOURCE_REVISION);
    builder.add_string(KEY_MODELING_SOURCE_SHA256, MODELING_SOURCE_SHA256);
    builder.add_string(KEY_CONFIGURATION_SOURCE_SHA256, CONFIGURATION_SOURCE_SHA256);
    builder.add_string(KEY_MANIFEST_SHA256, DECODER_MANIFEST_SHA256);
    builder.add_u32(KEY_INPUT_SAMPLE_RATE, INPUT_SAMPLE_RATE);
    builder.add_u32(KEY_OUTPUT_SAMPLE_RATE, OUTPUT_SAMPLE_RATE);
    builder.add_u32(KEY_DECODE_UPSAMPLE_RATE, DECODE_UPSAMPLE_RATE);
    builder.add_u32(KEY_NUM_QUANTIZERS, NUM_QUANTIZERS);
    builder.add_u32(KEY_NUM_SEMANTIC_QUANTIZERS, NUM_SEMANTIC_QUANTIZERS);
    builder.add_u32(KEY_CODEBOOK_SIZE, CODEBOOK_SIZE);
    builder.add_u32(
        KEY_CONFIGURED_SEMANTIC_VOCAB_SIZE,
        CONFIGURED_SEMANTIC_VOCAB_SIZE,
    );
    builder.add_u32(KEY_CODEBOOK_DIM, CODEBOOK_DIM);
    builder.add_u32(KEY_QUANTIZER_DIM, QUANTIZER_DIM);
    builder.add_u32(KEY_LATENT_DIM, LATENT_DIM);
    builder.add_u32(KEY_HIDDEN_SIZE, HIDDEN_SIZE);
    builder.add_u32(KEY_INTERMEDIATE_SIZE, INTERMEDIATE_SIZE);
    builder.add_u32(KEY_NUM_HIDDEN_LAYERS, NUM_HIDDEN_LAYERS);
    builder.add_u32(KEY_NUM_ATTENTION_HEADS, NUM_ATTENTION_HEADS);
    builder.add_u32(KEY_NUM_KEY_VALUE_HEADS, NUM_KEY_VALUE_HEADS);
    builder.add_u32(KEY_HEAD_DIM, HEAD_DIM);
    builder.add_f32(KEY_RMS_NORM_EPS, RMS_NORM_EPS);
    builder.add_f32(KEY_ROPE_THETA, ROPE_THETA);
    builder.add_u32(KEY_SLIDING_WINDOW, SLIDING_WINDOW);
    builder.add_f32(KEY_LAYER_SCALE_INITIAL, LAYER_SCALE_INITIAL);
    builder.add_u32(KEY_DECODER_DIM, DECODER_DIM);
    add_u32_array(&mut builder, KEY_UPSAMPLING_RATIOS, &UPSAMPLING_RATIOS);
    add_u32_array(&mut builder, KEY_UPSAMPLE_RATES, &UPSAMPLE_RATES);
    builder.add_u32(KEY_CHUNK_SIZE, CHUNK_SIZE);
    builder.add_u32(KEY_LEFT_CONTEXT, LEFT_CONTEXT);
    vokra_core::stamp_provenance(
        &mut builder,
        LicenseClass::Permissive,
        "apache-2.0",
        Some(NAME),
        Some(&format!(
            "{UPSTREAM_HF}@{UPSTREAM_REVISION}/model.safetensors sha256:{CHECKPOINT_SHA256}"
        )),
    );
    builder
}

fn add_u32_array(builder: &mut GgufBuilder, key: &str, values: &[u32]) {
    builder.add_metadata(
        key,
        GgufMetadataValue::Array(GgufArray {
            element_type: GgufValueType::U32,
            values: values.iter().copied().map(GgufMetadataValue::U32).collect(),
        }),
    );
}

fn validate_decoder_manifest(st: &SafetensorsFile) -> Result<(), ConvertError> {
    let expected = expected_decoder_manifest();
    let actual = st
        .tensors()
        .iter()
        .filter(|tensor| tensor.name.starts_with("decoder."))
        .map(|tensor| (tensor.name.clone(), tensor.shape.clone()))
        .collect::<BTreeMap<_, _>>();
    if actual == expected {
        return Ok(());
    }
    let missing = expected
        .keys()
        .filter(|name| !actual.contains_key(*name))
        .take(8)
        .collect::<Vec<_>>();
    let extra = actual
        .keys()
        .filter(|name| !expected.contains_key(*name))
        .take(8)
        .collect::<Vec<_>>();
    let wrong_shape = expected
        .iter()
        .filter_map(|(name, expected_shape)| {
            actual
                .get(name)
                .filter(|actual_shape| *actual_shape != expected_shape)
                .map(|actual_shape| (name, actual_shape, expected_shape))
        })
        .take(8)
        .collect::<Vec<_>>();
    Err(ConvertError::Parse(format!(
        "qwen3-tts-tokenizer-12hz: decoder manifest mismatch: expected {} tensors, found {}; missing={missing:?}, extra={extra:?}, wrong_shape={wrong_shape:?}",
        expected.len(),
        actual.len()
    )))
}

pub(crate) fn expected_decoder_manifest() -> BTreeMap<String, Vec<u64>> {
    let mut out = BTreeMap::new();
    add_quantizer_manifest(&mut out);
    add_pre_transformer_manifest(&mut out);
    add_upsample_manifest(&mut out);
    add_wave_decoder_manifest(&mut out);
    debug_assert_eq!(out.len(), DECODER_TENSOR_COUNT);
    out
}

fn insert(out: &mut BTreeMap<String, Vec<u64>>, name: impl Into<String>, shape: &[u64]) {
    let old = out.insert(name.into(), shape.to_vec());
    debug_assert!(old.is_none());
}

fn add_quantizer_manifest(out: &mut BTreeMap<String, Vec<u64>>) {
    for branch in ["rvq_first", "rvq_rest"] {
        insert(
            out,
            format!("decoder.quantizer.{branch}.input_proj.weight"),
            &[QUANTIZER_DIM as u64, CODEBOOK_DIM as u64, 1],
        );
        insert(
            out,
            format!("decoder.quantizer.{branch}.output_proj.weight"),
            &[CODEBOOK_DIM as u64, QUANTIZER_DIM as u64, 1],
        );
    }
    for layer in 0..NUM_QUANTIZERS as usize {
        let (branch, layer) = if layer == 0 {
            ("rvq_first", 0)
        } else {
            ("rvq_rest", layer - 1)
        };
        let prefix = format!("decoder.quantizer.{branch}.vq.layers.{layer}._codebook");
        insert(
            out,
            format!("{prefix}.cluster_usage"),
            &[CODEBOOK_SIZE as u64],
        );
        insert(
            out,
            format!("{prefix}.embedding_sum"),
            &[CODEBOOK_SIZE as u64, QUANTIZER_DIM as u64],
        );
    }
}

fn add_pre_transformer_manifest(out: &mut BTreeMap<String, Vec<u64>>) {
    insert(
        out,
        "decoder.pre_conv.conv.weight",
        &[LATENT_DIM as u64, CODEBOOK_DIM as u64, 3],
    );
    insert(out, "decoder.pre_conv.conv.bias", &[LATENT_DIM as u64]);
    insert(
        out,
        "decoder.pre_transformer.input_proj.weight",
        &[HIDDEN_SIZE as u64, LATENT_DIM as u64],
    );
    insert(
        out,
        "decoder.pre_transformer.input_proj.bias",
        &[HIDDEN_SIZE as u64],
    );
    insert(
        out,
        "decoder.pre_transformer.output_proj.weight",
        &[LATENT_DIM as u64, HIDDEN_SIZE as u64],
    );
    insert(
        out,
        "decoder.pre_transformer.output_proj.bias",
        &[LATENT_DIM as u64],
    );
    insert(
        out,
        "decoder.pre_transformer.norm.weight",
        &[HIDDEN_SIZE as u64],
    );

    let attention_width = (NUM_ATTENTION_HEADS * HEAD_DIM) as u64;
    let key_value_width = (NUM_KEY_VALUE_HEADS * HEAD_DIM) as u64;
    for layer in 0..NUM_HIDDEN_LAYERS as usize {
        let prefix = format!("decoder.pre_transformer.layers.{layer}");
        for suffix in [
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "self_attn_layer_scale.scale",
            "mlp_layer_scale.scale",
        ] {
            insert(out, format!("{prefix}.{suffix}"), &[HIDDEN_SIZE as u64]);
        }
        insert(
            out,
            format!("{prefix}.self_attn.q_proj.weight"),
            &[attention_width, HIDDEN_SIZE as u64],
        );
        for projection in ["k_proj", "v_proj"] {
            insert(
                out,
                format!("{prefix}.self_attn.{projection}.weight"),
                &[key_value_width, HIDDEN_SIZE as u64],
            );
        }
        insert(
            out,
            format!("{prefix}.self_attn.o_proj.weight"),
            &[HIDDEN_SIZE as u64, attention_width],
        );
        for projection in ["gate_proj", "up_proj"] {
            insert(
                out,
                format!("{prefix}.mlp.{projection}.weight"),
                &[INTERMEDIATE_SIZE as u64, HIDDEN_SIZE as u64],
            );
        }
        insert(
            out,
            format!("{prefix}.mlp.down_proj.weight"),
            &[HIDDEN_SIZE as u64, INTERMEDIATE_SIZE as u64],
        );
    }
}

fn add_upsample_manifest(out: &mut BTreeMap<String, Vec<u64>>) {
    for stage in 0..UPSAMPLING_RATIOS.len() {
        let prefix = format!("decoder.upsample.{stage}");
        insert(
            out,
            format!("{prefix}.0.conv.weight"),
            &[LATENT_DIM as u64, LATENT_DIM as u64, 2],
        );
        insert(out, format!("{prefix}.0.conv.bias"), &[LATENT_DIM as u64]);
        insert(
            out,
            format!("{prefix}.1.dwconv.conv.weight"),
            &[LATENT_DIM as u64, 1, 7],
        );
        insert(
            out,
            format!("{prefix}.1.dwconv.conv.bias"),
            &[LATENT_DIM as u64],
        );
        for suffix in ["norm.weight", "norm.bias", "gamma"] {
            insert(out, format!("{prefix}.1.{suffix}"), &[LATENT_DIM as u64]);
        }
        insert(
            out,
            format!("{prefix}.1.pwconv1.weight"),
            &[4 * LATENT_DIM as u64, LATENT_DIM as u64],
        );
        insert(
            out,
            format!("{prefix}.1.pwconv1.bias"),
            &[4 * LATENT_DIM as u64],
        );
        insert(
            out,
            format!("{prefix}.1.pwconv2.weight"),
            &[LATENT_DIM as u64, 4 * LATENT_DIM as u64],
        );
        insert(
            out,
            format!("{prefix}.1.pwconv2.bias"),
            &[LATENT_DIM as u64],
        );
    }
}

fn add_wave_decoder_manifest(out: &mut BTreeMap<String, Vec<u64>>) {
    insert(
        out,
        "decoder.decoder.0.conv.weight",
        &[DECODER_DIM as u64, LATENT_DIM as u64, 7],
    );
    insert(out, "decoder.decoder.0.conv.bias", &[DECODER_DIM as u64]);
    for (stage, rate) in UPSAMPLE_RATES.iter().copied().enumerate() {
        let block = stage + 1;
        let in_dim = DECODER_DIM as u64 / (1_u64 << stage);
        let out_dim = in_dim / 2;
        let prefix = format!("decoder.decoder.{block}.block");
        for parameter in ["alpha", "beta"] {
            insert(out, format!("{prefix}.0.{parameter}"), &[in_dim]);
        }
        insert(
            out,
            format!("{prefix}.1.conv.weight"),
            &[in_dim, out_dim, (2 * rate) as u64],
        );
        insert(out, format!("{prefix}.1.conv.bias"), &[out_dim]);
        for residual in 0..3 {
            let residual = residual + 2;
            for activation in ["act1", "act2"] {
                for parameter in ["alpha", "beta"] {
                    insert(
                        out,
                        format!("{prefix}.{residual}.{activation}.{parameter}"),
                        &[out_dim],
                    );
                }
            }
            insert(
                out,
                format!("{prefix}.{residual}.conv1.conv.weight"),
                &[out_dim, out_dim, 7],
            );
            insert(
                out,
                format!("{prefix}.{residual}.conv1.conv.bias"),
                &[out_dim],
            );
            insert(
                out,
                format!("{prefix}.{residual}.conv2.conv.weight"),
                &[out_dim, out_dim, 1],
            );
            insert(
                out,
                format!("{prefix}.{residual}.conv2.conv.bias"),
                &[out_dim],
            );
        }
    }
    let output_dim = DECODER_DIM as u64 / (1_u64 << UPSAMPLE_RATES.len());
    for parameter in ["alpha", "beta"] {
        insert(out, format!("decoder.decoder.5.{parameter}"), &[output_dim]);
    }
    insert(out, "decoder.decoder.6.conv.weight", &[1, output_dim, 7]);
    insert(out, "decoder.decoder.6.conv.bias", &[1]);
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::GgufMetadataValue;

    #[test]
    fn official_decoder_manifest_count_and_hash_are_pinned() {
        let manifest = expected_decoder_manifest();
        assert_eq!(manifest.len(), DECODER_TENSOR_COUNT);
        assert_eq!(hex(&manifest_sha256(&manifest)), DECODER_MANIFEST_SHA256);
    }

    #[test]
    fn semantic_config_value_is_not_substituted_for_real_rvq_shape() {
        let manifest = expected_decoder_manifest();
        assert_eq!(CONFIGURED_SEMANTIC_VOCAB_SIZE, 4_096);
        assert_eq!(
            manifest["decoder.quantizer.rvq_first.vq.layers.0._codebook.embedding_sum"],
            vec![2_048, 256]
        );
    }

    #[test]
    fn exact_stage_shapes_match_official_header() {
        let manifest = expected_decoder_manifest();
        assert_eq!(
            manifest["decoder.pre_transformer.layers.7.self_attn.q_proj.weight"],
            vec![1_024, 512]
        );
        assert_eq!(
            manifest["decoder.upsample.1.1.pwconv1.weight"],
            vec![4_096, 1_024]
        );
        assert_eq!(
            manifest["decoder.decoder.4.block.1.conv.weight"],
            vec![192, 96, 6]
        );
        assert_eq!(manifest["decoder.decoder.6.conv.weight"], vec![1, 96, 7]);
        assert!(manifest.keys().all(|name| name.starts_with("decoder.")));
    }

    #[test]
    fn metadata_records_source_and_decode_geometry() {
        let bytes = metadata_builder().to_bytes().expect("metadata GGUF");
        let file = vokra_core::gguf::GgufFile::parse(bytes).expect("parse metadata GGUF");
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH)
                .and_then(GgufMetadataValue::as_str),
            Some(ARCH)
        );
        assert_eq!(
            file.get(KEY_CHECKPOINT_SHA256)
                .and_then(GgufMetadataValue::as_str),
            Some(CHECKPOINT_SHA256)
        );
        assert_eq!(
            file.get(KEY_DECODE_UPSAMPLE_RATE)
                .and_then(GgufMetadataValue::as_u64),
            Some(DECODE_UPSAMPLE_RATE as u64)
        );
    }

    #[test]
    fn production_converter_rejects_unpinned_bytes_before_parsing() {
        let error = convert(Vec::new()).expect_err("empty input must fail SHA gate");
        assert!(error.to_string().contains("checkpoint SHA-256 mismatch"));
    }
}
