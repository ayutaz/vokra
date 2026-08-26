//! Strict conversion contract for the canonical deepfake-audio detector.
//!
//! The public release is
//! `MelodyMachine/Deepfake-audio-detection-V2@de3cde5a29c449bb5268814e421b46bf6ebdcd72`.
//! Contrary to the historical Vokra scaffold, the checkpoint is not WavLM:
//! its pinned `config.json` declares `Wav2Vec2ForSequenceClassification` and
//! `model_type = "wav2vec2"`. This converter therefore accepts exactly the
//! 215 F32 tensors in that release and stamps the complete inference contract.

use std::collections::BTreeMap;
use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{
    GgmlType, GgufArray, GgufBuilder, GgufMetadataValue, GgufValueType, chunks,
};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

pub const ARCH: &str = "deepfake_detection";
pub const NAME: &str = "deepfake-audio-detection-v2";
pub const CATEGORY: &str = "classification";
pub const UPSTREAM_HF: &str = "MelodyMachine/Deepfake-audio-detection-V2";
pub const UPSTREAM_REVISION: &str = "de3cde5a29c449bb5268814e421b46bf6ebdcd72";
pub const CHECKPOINT_FILE: &str = "model.safetensors";
pub const CHECKPOINT_SHA256: &str =
    "997d9ce59e63151d5e444a6fa7c863986d0e56d515f67321bd705ac3b01bc38c";
pub const CONFIG_SHA256: &str = "a7ff31ca7ba4dc7fb5c4847d6dff0cb8daa1f0ec512e6ff8190664874c5b2806";
pub const PREPROCESSOR_SHA256: &str =
    "8cdfd65ff4115423185a1512bdae100e2e0cd744f5b322417429944aaafd0827";
pub const DEFAULT_LICENSE_SPDX: &str = "apache-2.0";
pub const TENSOR_COUNT: usize = 215;

pub const SAMPLE_RATE: u32 = 16_000;
pub const HIDDEN_SIZE: u32 = 768;
pub const NUM_HIDDEN_LAYERS: u32 = 12;
pub const NUM_ATTENTION_HEADS: u32 = 12;
pub const INTERMEDIATE_SIZE: u32 = 3_072;
pub const CLASSIFIER_PROJ_SIZE: u32 = 256;
pub const NUM_CLASSES: u32 = 2;
pub const LAYER_NORM_EPS: f32 = 1.0e-5;
pub const NUM_CONV_POS_EMBEDDINGS: u32 = 128;
pub const NUM_CONV_POS_EMBEDDING_GROUPS: u32 = 16;
pub const CLASS_LABELS: [&str; 2] = ["fake", "real"];

const KEY_CATEGORY: &str = "vokra.model.category";
const PREFIX: &str = "vokra.deepfake";
const CONV_DIM: [u32; 7] = [512; 7];
const CONV_KERNEL: [u32; 7] = [10, 3, 3, 3, 3, 2, 2];
const CONV_STRIDE: [u32; 7] = [5, 2, 2, 2, 2, 2, 2];

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
/// Conversion counters for the canonical Wav2Vec2 sequence classifier.
pub struct DeepfakeDetectionReport {
    pub read: usize,
    pub written: usize,
    /// Always zero after a successful strict conversion.
    pub skipped_non_float: usize,
    /// Always zero because the canonical checkpoint is F32-only.
    pub bf16_passthrough: usize,
}

/// Converts the exact canonical checkpoint into a self-describing GGUF.
pub fn convert_deepfake_detection_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<DeepfakeDetectionReport, ConvertError> {
    if let Some(value) = license
        && !value.eq_ignore_ascii_case(DEFAULT_LICENSE_SPDX)
    {
        return Err(ConvertError::Usage(format!(
            "deepfake_detection: canonical {UPSTREAM_HF}@{UPSTREAM_REVISION} has pinned Apache-2.0 weights; refusing conflicting --license {value:?}"
        )));
    }

    let safetensors = SafetensorsFile::parse(std::fs::read(input)?)?;
    validate_manifest(&safetensors)?;

    let mut builder = GgufBuilder::new();
    stamp_metadata(&mut builder);
    let mut report = DeepfakeDetectionReport {
        read: safetensors.tensors().len(),
        ..DeepfakeDetectionReport::default()
    };
    for tensor in safetensors.tensors() {
        builder.add_tensor(
            &tensor.name,
            tensor.dtype,
            tensor.shape.clone(),
            safetensors.tensor_bytes(tensor).to_vec(),
        )?;
        report.written += 1;
    }
    std::fs::write(output, builder.to_bytes()?)?;
    Ok(report)
}

fn stamp_metadata(builder: &mut GgufBuilder) {
    builder.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    builder.add_string(chunks::KEY_MODEL_NAME, NAME);
    builder.add_string(KEY_CATEGORY, CATEGORY);
    vokra_core::stamp_provenance(
        builder,
        LicenseClass::Permissive,
        DEFAULT_LICENSE_SPDX,
        Some(NAME),
        Some(&format!(
            "{UPSTREAM_HF}@{UPSTREAM_REVISION}/{CHECKPOINT_FILE} sha256:{CHECKPOINT_SHA256}"
        )),
    );
    builder.add_string("vokra.provenance.upstream_hf", UPSTREAM_HF);
    builder.add_string("vokra.provenance.upstream_revision", UPSTREAM_REVISION);
    builder.add_string("vokra.provenance.checkpoint_file", CHECKPOINT_FILE);
    builder.add_string("vokra.provenance.checkpoint_sha256", CHECKPOINT_SHA256);
    builder.add_string("vokra.provenance.config_sha256", CONFIG_SHA256);
    builder.add_string("vokra.provenance.preprocessor_sha256", PREPROCESSOR_SHA256);

    builder.add_string(
        &format!("{PREFIX}.architecture"),
        "Wav2Vec2ForSequenceClassification",
    );
    builder.add_string(&format!("{PREFIX}.model_type"), "wav2vec2");
    builder.add_u32(&format!("{PREFIX}.sample_rate"), SAMPLE_RATE);
    builder.add_bool(&format!("{PREFIX}.normalize"), true);
    builder.add_bool(&format!("{PREFIX}.return_attention_mask"), false);
    builder.add_u32(&format!("{PREFIX}.hidden_size"), HIDDEN_SIZE);
    builder.add_u32(&format!("{PREFIX}.num_hidden_layers"), NUM_HIDDEN_LAYERS);
    builder.add_u32(
        &format!("{PREFIX}.num_attention_heads"),
        NUM_ATTENTION_HEADS,
    );
    builder.add_u32(&format!("{PREFIX}.intermediate_size"), INTERMEDIATE_SIZE);
    builder.add_u32(
        &format!("{PREFIX}.classifier_proj_size"),
        CLASSIFIER_PROJ_SIZE,
    );
    builder.add_u32(&format!("{PREFIX}.num_classes"), NUM_CLASSES);
    builder.add_f32(&format!("{PREFIX}.layer_norm_eps"), LAYER_NORM_EPS);
    builder.add_string(&format!("{PREFIX}.feat_extract_norm"), "group");
    builder.add_bool(&format!("{PREFIX}.do_stable_layer_norm"), false);
    builder.add_string(&format!("{PREFIX}.hidden_act"), "gelu");
    builder.add_u32(
        &format!("{PREFIX}.num_conv_pos_embeddings"),
        NUM_CONV_POS_EMBEDDINGS,
    );
    builder.add_u32(
        &format!("{PREFIX}.num_conv_pos_embedding_groups"),
        NUM_CONV_POS_EMBEDDING_GROUPS,
    );
    builder.add_bool(&format!("{PREFIX}.use_weighted_layer_sum"), false);
    add_u32_array(builder, &format!("{PREFIX}.conv_dim"), &CONV_DIM);
    add_u32_array(builder, &format!("{PREFIX}.conv_kernel"), &CONV_KERNEL);
    add_u32_array(builder, &format!("{PREFIX}.conv_stride"), &CONV_STRIDE);
    add_string_array(builder, &format!("{PREFIX}.id2label"), &CLASS_LABELS);
}

fn validate_manifest(safetensors: &SafetensorsFile) -> Result<(), ConvertError> {
    let observed = safetensors
        .tensors()
        .iter()
        .map(|tensor| (tensor.name.clone(), (tensor.dtype, tensor.shape.clone())))
        .collect::<BTreeMap<_, _>>();
    validate_observed_manifest(&observed)
}

fn validate_observed_manifest(
    observed: &BTreeMap<String, (GgmlType, Vec<u64>)>,
) -> Result<(), ConvertError> {
    let expected = expected_manifest();
    if observed.len() != TENSOR_COUNT {
        return Err(ConvertError::Parse(format!(
            "deepfake_detection: checkpoint has {} tensors, expected exactly {TENSOR_COUNT}",
            observed.len()
        )));
    }
    for (name, (dtype, shape)) in observed {
        let wanted = expected.get(name).ok_or_else(|| {
            ConvertError::Parse(format!(
                "deepfake_detection: unexpected tensor {name:?}; refusing pass-through conversion"
            ))
        })?;
        if *dtype != GgmlType::F32 {
            return Err(ConvertError::Parse(format!(
                "deepfake_detection: tensor {name:?} has {dtype:?}, expected canonical F32"
            )));
        }
        if shape != wanted {
            return Err(ConvertError::Parse(format!(
                "deepfake_detection: tensor {name:?} shape {shape:?}, expected {wanted:?}"
            )));
        }
    }
    for name in expected.keys() {
        if !observed.contains_key(name) {
            return Err(ConvertError::Parse(format!(
                "deepfake_detection: required tensor {name:?} is missing"
            )));
        }
    }
    Ok(())
}

pub(crate) fn expected_manifest() -> BTreeMap<String, Vec<u64>> {
    let mut tensors = BTreeMap::new();
    tensors.insert("classifier.bias".to_owned(), vec![2]);
    tensors.insert("classifier.weight".to_owned(), vec![2, 256]);
    tensors.insert("projector.bias".to_owned(), vec![256]);
    tensors.insert("projector.weight".to_owned(), vec![256, 768]);

    tensors.insert("wav2vec2.encoder.layer_norm.bias".to_owned(), vec![768]);
    tensors.insert("wav2vec2.encoder.layer_norm.weight".to_owned(), vec![768]);
    for layer in 0..NUM_HIDDEN_LAYERS {
        insert_encoder_block(&mut tensors, &format!("wav2vec2.encoder.layers.{layer}"));
    }
    tensors.insert(
        "wav2vec2.encoder.pos_conv_embed.conv.bias".to_owned(),
        vec![768],
    );
    tensors.insert(
        "wav2vec2.encoder.pos_conv_embed.conv.parametrizations.weight.original0".to_owned(),
        vec![1, 1, 128],
    );
    tensors.insert(
        "wav2vec2.encoder.pos_conv_embed.conv.parametrizations.weight.original1".to_owned(),
        vec![768, 48, 128],
    );

    for (layer, &kernel) in CONV_KERNEL.iter().enumerate() {
        let input = if layer == 0 { 1 } else { 512 };
        tensors.insert(
            format!("wav2vec2.feature_extractor.conv_layers.{layer}.conv.weight"),
            vec![512, input, u64::from(kernel)],
        );
    }
    tensors.insert(
        "wav2vec2.feature_extractor.conv_layers.0.layer_norm.bias".to_owned(),
        vec![512],
    );
    tensors.insert(
        "wav2vec2.feature_extractor.conv_layers.0.layer_norm.weight".to_owned(),
        vec![512],
    );
    tensors.insert(
        "wav2vec2.feature_projection.layer_norm.bias".to_owned(),
        vec![512],
    );
    tensors.insert(
        "wav2vec2.feature_projection.layer_norm.weight".to_owned(),
        vec![512],
    );
    tensors.insert(
        "wav2vec2.feature_projection.projection.bias".to_owned(),
        vec![768],
    );
    tensors.insert(
        "wav2vec2.feature_projection.projection.weight".to_owned(),
        vec![768, 512],
    );
    tensors.insert("wav2vec2.masked_spec_embed".to_owned(), vec![768]);
    debug_assert_eq!(tensors.len(), TENSOR_COUNT);
    tensors
}

fn insert_encoder_block(tensors: &mut BTreeMap<String, Vec<u64>>, prefix: &str) {
    for (suffix, shape) in [
        ("attention.k_proj.bias", vec![768]),
        ("attention.k_proj.weight", vec![768, 768]),
        ("attention.out_proj.bias", vec![768]),
        ("attention.out_proj.weight", vec![768, 768]),
        ("attention.q_proj.bias", vec![768]),
        ("attention.q_proj.weight", vec![768, 768]),
        ("attention.v_proj.bias", vec![768]),
        ("attention.v_proj.weight", vec![768, 768]),
        ("feed_forward.intermediate_dense.bias", vec![3_072]),
        ("feed_forward.intermediate_dense.weight", vec![3_072, 768]),
        ("feed_forward.output_dense.bias", vec![768]),
        ("feed_forward.output_dense.weight", vec![768, 3_072]),
        ("final_layer_norm.bias", vec![768]),
        ("final_layer_norm.weight", vec![768]),
        ("layer_norm.bias", vec![768]),
        ("layer_norm.weight", vec![768]),
    ] {
        tensors.insert(format!("{prefix}.{suffix}"), shape);
    }
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

fn add_string_array(builder: &mut GgufBuilder, key: &str, values: &[&str]) {
    builder.add_metadata(
        key,
        GgufMetadataValue::Array(GgufArray {
            element_type: GgufValueType::String,
            values: values
                .iter()
                .map(|value| GgufMetadataValue::String((*value).to_owned()))
                .collect(),
        }),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::GgufFile;

    fn observed() -> BTreeMap<String, (GgmlType, Vec<u64>)> {
        expected_manifest()
            .into_iter()
            .map(|(name, shape)| (name, (GgmlType::F32, shape)))
            .collect()
    }

    #[test]
    fn canonical_manifest_matches_the_pinned_header() {
        let manifest = expected_manifest();
        assert_eq!(manifest.len(), 215);
        assert_eq!(manifest["projector.weight"], vec![256, 768]);
        assert_eq!(manifest["classifier.weight"], vec![2, 256]);
        assert_eq!(
            manifest["wav2vec2.encoder.pos_conv_embed.conv.parametrizations.weight.original1"],
            vec![768, 48, 128]
        );
        assert_eq!(
            manifest["wav2vec2.encoder.layers.11.feed_forward.intermediate_dense.weight"],
            vec![3_072, 768]
        );
        validate_observed_manifest(&observed()).unwrap();
    }

    #[test]
    fn metadata_pins_wav2vec2_frontend_and_class_order() {
        let mut builder = GgufBuilder::new();
        stamp_metadata(&mut builder);
        let gguf = GgufFile::parse(builder.to_bytes().unwrap()).unwrap();
        assert_eq!(
            gguf.get(chunks::KEY_MODEL_ARCH)
                .and_then(GgufMetadataValue::as_str),
            Some(ARCH)
        );
        assert_eq!(
            gguf.get("vokra.provenance.upstream_revision")
                .and_then(GgufMetadataValue::as_str),
            Some(UPSTREAM_REVISION)
        );
        assert_eq!(
            gguf.get("vokra.deepfake.architecture")
                .and_then(GgufMetadataValue::as_str),
            Some("Wav2Vec2ForSequenceClassification")
        );
        let labels = gguf
            .get("vokra.deepfake.id2label")
            .and_then(GgufMetadataValue::as_array)
            .unwrap();
        assert_eq!(labels.values[0].as_str(), Some("fake"));
        assert_eq!(labels.values[1].as_str(), Some("real"));
    }

    #[test]
    fn missing_extra_wrong_shape_and_dtype_fail_closed() {
        let mut missing = observed();
        missing.remove("classifier.bias");
        assert!(
            validate_observed_manifest(&missing)
                .unwrap_err()
                .to_string()
                .contains("214 tensors")
        );

        let mut extra = observed();
        extra.remove("classifier.bias");
        extra.insert("fabricated.weight".to_owned(), (GgmlType::F32, vec![2]));
        assert!(
            validate_observed_manifest(&extra)
                .unwrap_err()
                .to_string()
                .contains("unexpected tensor")
        );

        let mut wrong_shape = observed();
        wrong_shape.get_mut("classifier.weight").unwrap().1 = vec![2, 768];
        let error = validate_observed_manifest(&wrong_shape)
            .unwrap_err()
            .to_string();
        assert!(error.contains("classifier.weight"));
        assert!(error.contains("expected [2, 256]"));

        let mut wrong_dtype = observed();
        wrong_dtype.get_mut("classifier.bias").unwrap().0 = GgmlType::F16;
        assert!(
            validate_observed_manifest(&wrong_dtype)
                .unwrap_err()
                .to_string()
                .contains("expected canonical F32")
        );
    }

    #[test]
    fn conflicting_license_fails_before_io() {
        let missing = Path::new("does-not-exist");
        let error = convert_deepfake_detection_file(missing, missing, Some("mit"))
            .unwrap_err()
            .to_string();
        assert!(error.contains("conflicting --license"));
    }
}
