//! Strict emotion2vec+ Large checkpoint conversion.
//!
//! The canonical release is `emotion2vec/emotion2vec_plus_large` at
//! revision `6c303ba987b86b93193de93e34bb2b077a6bedc4`. Its `model.pt`
//! payload is prepared offline into one F32 safetensors file; this converter
//! requires the complete 185-tensor inference manifest before writing GGUF.

use std::collections::BTreeMap;
use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{
    GgmlType, GgufArray, GgufBuilder, GgufMetadataValue, GgufValueType, chunks,
};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

pub const ARCH: &str = "emotion2vec";
pub const NAME: &str = "emotion2vec-plus-large";
pub const CATEGORY: &str = "emotion";
pub const UPSTREAM_HF: &str = "emotion2vec/emotion2vec_plus_large";
pub const UPSTREAM_REVISION: &str = "6c303ba987b86b93193de93e34bb2b077a6bedc4";
pub const CHECKPOINT_FILE: &str = "model.pt";
pub const CHECKPOINT_SHA256: &str =
    "be501a01f26fcdc7663a062dff86af839afbaef7c4de32f5e42d7e1ad2784da4";
pub const DEFAULT_LICENSE: &str = "mit";
pub const TENSOR_COUNT: usize = 185;
pub const SAMPLE_RATE: u32 = 16_000;
pub const EMBED_DIM: u32 = 1_024;
pub const DEPTH: u32 = 8;
pub const PRENET_DEPTH: u32 = 4;
pub const NUM_HEADS: u32 = 16;
pub const MLP_DIM: u32 = 4_096;
pub const NUM_EXTRA_TOKENS: u32 = 10;
pub const NUM_CLASSES: u32 = 9;
pub const CONV_POS_DEPTH: u32 = 5;
pub const CONV_POS_KERNEL: u32 = 19;
pub const CONV_POS_GROUPS: u32 = 16;
pub const LAYER_NORM_EPS: f32 = 1.0e-5;

pub const CLASS_LABELS: [&str; 9] = [
    "生气/angry",
    "厌恶/disgusted",
    "恐惧/fearful",
    "开心/happy",
    "中立/neutral",
    "其他/other",
    "难过/sad",
    "吃惊/surprised",
    "<unk>",
];

const KEY_CATEGORY: &str = "vokra.model.category";
const PREFIX: &str = "vokra.emotion2vec";
const CONV_DIM: [u32; 7] = [512; 7];
const CONV_KERNEL: [u32; 7] = [10, 3, 3, 3, 3, 2, 2];
const CONV_STRIDE: [u32; 7] = [5, 2, 2, 2, 2, 2, 2];

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
/// Conversion counters for the canonical emotion2vec+ Large checkpoint.
pub struct Emotion2vecReport {
    /// Tensor descriptors read from the prepared safetensors header.
    pub read: usize,
    /// Canonical F32 tensors written to GGUF.
    pub written: usize,
    /// Always zero for a successful strict conversion.
    pub skipped_non_float: usize,
    /// Always zero because the canonical prepared checkpoint is F32-only.
    pub bf16_passthrough: usize,
}

/// Converts the prepared canonical F32 checkpoint.
///
/// The source weight license has already been owner-signed as MIT in
/// `docs/license-audit.md`; a conflicting override fails closed.
pub fn convert_emotion2vec_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<Emotion2vecReport, ConvertError> {
    if let Some(value) = license
        && !value.eq_ignore_ascii_case(DEFAULT_LICENSE)
    {
        return Err(ConvertError::Usage(format!(
            "emotion2vec: canonical {UPSTREAM_HF}@{UPSTREAM_REVISION} has pinned MIT weights; refusing conflicting --license {value:?}"
        )));
    }

    let st = SafetensorsFile::parse(std::fs::read(input)?)?;
    validate_manifest(&st)?;

    let mut builder = GgufBuilder::new();
    stamp_metadata(&mut builder);
    let mut report = Emotion2vecReport {
        read: st.tensors().len(),
        ..Emotion2vecReport::default()
    };
    for tensor in st.tensors() {
        builder.add_tensor(
            &tensor.name,
            tensor.dtype,
            tensor.shape.clone(),
            st.tensor_bytes(tensor).to_vec(),
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
        DEFAULT_LICENSE,
        Some(NAME),
        Some(&format!(
            "{UPSTREAM_HF}@{UPSTREAM_REVISION}/{CHECKPOINT_FILE} sha256:{CHECKPOINT_SHA256}"
        )),
    );
    builder.add_string("vokra.provenance.upstream_hf", UPSTREAM_HF);
    builder.add_string("vokra.provenance.upstream_revision", UPSTREAM_REVISION);
    builder.add_string("vokra.provenance.checkpoint_file", CHECKPOINT_FILE);
    builder.add_string("vokra.provenance.checkpoint_sha256", CHECKPOINT_SHA256);
    builder.add_u32(&format!("{PREFIX}.sample_rate"), SAMPLE_RATE);
    builder.add_u32(&format!("{PREFIX}.embed_dim"), EMBED_DIM);
    builder.add_u32(&format!("{PREFIX}.depth"), DEPTH);
    builder.add_u32(&format!("{PREFIX}.prenet_depth"), PRENET_DEPTH);
    builder.add_u32(&format!("{PREFIX}.num_heads"), NUM_HEADS);
    builder.add_u32(&format!("{PREFIX}.mlp_dim"), MLP_DIM);
    builder.add_u32(&format!("{PREFIX}.num_extra_tokens"), NUM_EXTRA_TOKENS);
    builder.add_u32(&format!("{PREFIX}.num_classes"), NUM_CLASSES);
    builder.add_u32(&format!("{PREFIX}.conv_pos_depth"), CONV_POS_DEPTH);
    builder.add_u32(&format!("{PREFIX}.conv_pos_kernel"), CONV_POS_KERNEL);
    builder.add_u32(&format!("{PREFIX}.conv_pos_groups"), CONV_POS_GROUPS);
    builder.add_f32(&format!("{PREFIX}.layer_norm_eps"), LAYER_NORM_EPS);
    builder.add_bool(&format!("{PREFIX}.normalize"), true);
    add_u32_array(builder, &format!("{PREFIX}.conv_dim"), &CONV_DIM);
    add_u32_array(builder, &format!("{PREFIX}.conv_kernel"), &CONV_KERNEL);
    add_u32_array(builder, &format!("{PREFIX}.conv_stride"), &CONV_STRIDE);
    add_string_array(builder, &format!("{PREFIX}.class_labels"), &CLASS_LABELS);
}

fn validate_manifest(st: &SafetensorsFile) -> Result<(), ConvertError> {
    let observed = st
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
            "emotion2vec: prepared checkpoint has {} tensors, expected exactly {TENSOR_COUNT}",
            observed.len()
        )));
    }
    for (name, (dtype, shape)) in observed {
        let wanted = expected.get(name).ok_or_else(|| {
            ConvertError::Parse(format!(
                "emotion2vec: unexpected tensor {name:?}; refusing pass-through conversion"
            ))
        })?;
        if *dtype != GgmlType::F32 {
            return Err(ConvertError::Parse(format!(
                "emotion2vec: tensor {name:?} has {dtype:?}, expected canonical F32"
            )));
        }
        if shape != wanted {
            return Err(ConvertError::Parse(format!(
                "emotion2vec: tensor {name:?} shape {shape:?}, expected {wanted:?}"
            )));
        }
    }
    for name in expected.keys() {
        if !observed.contains_key(name) {
            return Err(ConvertError::Parse(format!(
                "emotion2vec: required tensor {name:?} is missing"
            )));
        }
    }
    Ok(())
}

pub(crate) fn expected_manifest() -> BTreeMap<String, Vec<u64>> {
    let mut tensors = BTreeMap::new();
    for layer in 0..DEPTH {
        insert_block(&mut tensors, &format!("d2v_model.blocks.{layer}"));
    }
    let audio = "d2v_model.modality_encoders.AUDIO";
    tensors.insert(format!("{audio}.alibi_scale"), vec![16]);
    for layer in 0..PRENET_DEPTH {
        insert_block(
            &mut tensors,
            &format!("{audio}.context_encoder.blocks.{layer}"),
        );
    }
    tensors.insert(format!("{audio}.context_encoder.norm.bias"), vec![1_024]);
    tensors.insert(format!("{audio}.context_encoder.norm.weight"), vec![1_024]);
    tensors.insert(format!("{audio}.extra_tokens"), vec![1, 10, 1_024]);
    for (layer, &kernel) in CONV_KERNEL.iter().enumerate() {
        let input = if layer == 0 { 1 } else { 512 };
        tensors.insert(
            format!("{audio}.local_encoder.conv_layers.{layer}.0.weight"),
            vec![512, input, u64::from(kernel)],
        );
        tensors.insert(
            format!("{audio}.local_encoder.conv_layers.{layer}.2.1.bias"),
            vec![512],
        );
        tensors.insert(
            format!("{audio}.local_encoder.conv_layers.{layer}.2.1.weight"),
            vec![512],
        );
    }
    tensors.insert(format!("{audio}.project_features.1.bias"), vec![512]);
    tensors.insert(format!("{audio}.project_features.1.weight"), vec![512]);
    tensors.insert(format!("{audio}.project_features.2.bias"), vec![1_024]);
    tensors.insert(
        format!("{audio}.project_features.2.weight"),
        vec![1_024, 512],
    );
    for layer in 1..=CONV_POS_DEPTH {
        tensors.insert(
            format!("{audio}.relative_positional_encoder.{layer}.0.bias"),
            vec![1_024],
        );
        tensors.insert(
            format!("{audio}.relative_positional_encoder.{layer}.0.weight"),
            vec![1_024, 64, 19],
        );
    }
    tensors.insert("proj.bias".to_owned(), vec![9]);
    tensors.insert("proj.weight".to_owned(), vec![9, 1_024]);
    debug_assert_eq!(tensors.len(), TENSOR_COUNT);
    tensors
}

fn insert_block(tensors: &mut BTreeMap<String, Vec<u64>>, prefix: &str) {
    for (suffix, shape) in [
        ("attn.proj.bias", vec![1_024]),
        ("attn.proj.weight", vec![1_024, 1_024]),
        ("attn.qkv.bias", vec![3_072]),
        ("attn.qkv.weight", vec![3_072, 1_024]),
        ("mlp.fc1.bias", vec![4_096]),
        ("mlp.fc1.weight", vec![4_096, 1_024]),
        ("mlp.fc2.bias", vec![1_024]),
        ("mlp.fc2.weight", vec![1_024, 4_096]),
        ("norm1.bias", vec![1_024]),
        ("norm1.weight", vec![1_024]),
        ("norm2.bias", vec![1_024]),
        ("norm2.weight", vec![1_024]),
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
    fn canonical_manifest_is_complete() {
        let manifest = expected_manifest();
        assert_eq!(manifest.len(), 185);
        assert_eq!(
            manifest["d2v_model.blocks.7.attn.qkv.weight"],
            vec![3_072, 1_024]
        );
        assert_eq!(
            manifest["d2v_model.modality_encoders.AUDIO.relative_positional_encoder.5.0.weight"],
            vec![1_024, 64, 19]
        );
        assert_eq!(manifest["proj.weight"], vec![9, 1_024]);
        validate_observed_manifest(&observed()).unwrap();
    }

    #[test]
    fn metadata_pins_topology_and_bilingual_labels() {
        let mut builder = GgufBuilder::new();
        stamp_metadata(&mut builder);
        let gguf = GgufFile::parse(builder.to_bytes().unwrap()).unwrap();
        assert_eq!(
            gguf.get(chunks::KEY_MODEL_ARCH)
                .and_then(|value| value.as_str()),
            Some(ARCH)
        );
        assert_eq!(
            gguf.get("vokra.emotion2vec.embed_dim")
                .and_then(|value| value.as_u64()),
            Some(1_024)
        );
        let labels = gguf
            .get("vokra.emotion2vec.class_labels")
            .and_then(|value| value.as_array())
            .unwrap();
        assert_eq!(labels.values.len(), 9);
        assert_eq!(labels.values[0].as_str(), Some("生气/angry"));
    }

    #[test]
    fn missing_extra_and_wrong_shape_fail_closed() {
        let mut missing = observed();
        missing.remove("proj.bias");
        assert!(
            validate_observed_manifest(&missing)
                .unwrap_err()
                .to_string()
                .contains("184 tensors")
        );

        let mut extra = observed();
        extra.remove("proj.bias");
        extra.insert("fabricated.weight".to_owned(), (GgmlType::F32, vec![9]));
        assert!(
            validate_observed_manifest(&extra)
                .unwrap_err()
                .to_string()
                .contains("unexpected tensor")
        );

        let mut wrong = observed();
        wrong.get_mut("proj.weight").unwrap().1 = vec![8, 1_024];
        let error = validate_observed_manifest(&wrong).unwrap_err().to_string();
        assert!(error.contains("proj.weight"));
        assert!(error.contains("expected [9, 1024]"));
    }

    #[test]
    fn non_f32_and_conflicting_license_fail_closed() {
        let mut wrong = observed();
        wrong.get_mut("proj.bias").unwrap().0 = GgmlType::F16;
        assert!(
            validate_observed_manifest(&wrong)
                .unwrap_err()
                .to_string()
                .contains("expected canonical F32")
        );
        let missing = std::path::Path::new("does-not-exist");
        let error = convert_emotion2vec_file(missing, missing, Some("apache-2.0"))
            .unwrap_err()
            .to_string();
        assert!(error.contains("conflicting --license"));
    }
}
