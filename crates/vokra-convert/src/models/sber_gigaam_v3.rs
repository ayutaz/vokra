//! Strict prepared safetensors → GGUF conversion for GigaAM v3 RNNT.
//!
//! Raw pickle checkpoints are never accepted. The prepared artifact and its
//! exact sidecar must match the authenticated fixed manifest and prepared
//! artifact digest.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};
use vokra_core::json::{self, JsonValue};

use super::canary_1b_flash::{hex, sha256};
use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

pub const ARCH: &str = "sber_gigaam_v3";
pub const NAME: &str = "gigaam-v3";
pub const UPSTREAM_HF: &str = "ai-sage/GigaAM-v3";
pub const UPSTREAM_REVISION: &str = "ec1dc1f01d0d627ab2c0d3acc1e235702300d95e";
pub const UPSTREAM_SOURCE_REVISION: &str = "7447938d791c4f3e643386ee22c33777004293a5";
pub const CHECKPOINT_SHA256: &str =
    "afc6dcbae8320ea56f2cddebc0f13fbf62c9d59b6ddcad899782623c8610826a";
pub const MODELING_SHA256: &str =
    "269be43b635b1e510115baa2a843c5cbaa052e8adf0be30dc133a2ba5b5f2d86";
pub const CONFIG_SHA256: &str = "02361ba9cafd6c3ec66fcdd73494c3b562a60eb2a2d1b13f3cb04ae440d93e52";
pub const TOKENIZER_SHA256: &str =
    "828c12c991019eef952a960661f25a92d6ad279591e2ea466b4aeddf1d20a18a";
pub const PREPARED_FORMAT: &str = "vokra-gigaam-v3-prepared-v1";
pub const TENSOR_COUNT: usize = 561;
/// Set only after independent VAST review of the prepared bytes.
pub const AUTHENTICATED_PREPARED_SHA256: Option<&str> =
    Some("cee04765f031d6ee5088849ecb0e5c1db4e58ca28a345ce4d049015cd683a64e");
const KEY_PREFIX: &str = "vokra.gigaam_v3";

/// Counts emitted by strict conversion.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct SberGigaamV3Report {
    /// Number of manifest-bound tensors read.
    pub read: usize,
    /// Number of tensors copied to GGUF.
    pub written: usize,
    /// Number of non-floating tensors skipped.
    pub skipped_non_float: usize,
    /// Number of BF16 tensors preserved.
    pub bf16_passthrough: usize,
}

fn expected_manifest() -> Vec<(String, Vec<u64>, GgmlType)> {
    let mut m = vec![
        (
            "model.preprocessor.featurizer.0.spectrogram.window".into(),
            vec![320],
            GgmlType::F32,
        ),
        (
            "model.preprocessor.featurizer.0.mel_scale.fb".into(),
            vec![161, 64],
            GgmlType::F32,
        ),
        (
            "model.encoder.pre_encode.conv.0.weight".into(),
            vec![768, 64, 5],
            GgmlType::F16,
        ),
        (
            "model.encoder.pre_encode.conv.0.bias".into(),
            vec![768],
            GgmlType::F16,
        ),
        (
            "model.encoder.pre_encode.conv.2.weight".into(),
            vec![768, 768, 5],
            GgmlType::F16,
        ),
        (
            "model.encoder.pre_encode.conv.2.bias".into(),
            vec![768],
            GgmlType::F16,
        ),
    ];
    for layer in 0..16 {
        let p = format!("model.encoder.layers.{layer}");
        for n in [
            "norm_feed_forward1",
            "norm_conv",
            "norm_self_att",
            "norm_feed_forward2",
            "norm_out",
        ] {
            m.push((format!("{p}.{n}.weight"), vec![768], GgmlType::F16));
            m.push((format!("{p}.{n}.bias"), vec![768], GgmlType::F16));
        }
        for branch in ["feed_forward1", "feed_forward2"] {
            m.push((
                format!("{p}.{branch}.linear1.weight"),
                vec![3072, 768],
                GgmlType::F16,
            ));
            m.push((
                format!("{p}.{branch}.linear1.bias"),
                vec![3072],
                GgmlType::F16,
            ));
            m.push((
                format!("{p}.{branch}.linear2.weight"),
                vec![768, 3072],
                GgmlType::F16,
            ));
            m.push((
                format!("{p}.{branch}.linear2.bias"),
                vec![768],
                GgmlType::F16,
            ));
        }
        for (n, shape) in [
            ("pointwise_conv1.weight", vec![1536, 768, 1]),
            ("pointwise_conv1.bias", vec![1536]),
            ("depthwise_conv.weight", vec![768, 1, 5]),
            ("depthwise_conv.bias", vec![768]),
            ("batch_norm.weight", vec![768]),
            ("batch_norm.bias", vec![768]),
            ("pointwise_conv2.weight", vec![768, 768, 1]),
            ("pointwise_conv2.bias", vec![768]),
        ] {
            m.push((format!("{p}.conv.{n}"), shape, GgmlType::F16));
        }
        for n in ["linear_q", "linear_k", "linear_v", "linear_out"] {
            m.push((
                format!("{p}.self_attn.{n}.weight"),
                vec![768, 768],
                GgmlType::F16,
            ));
            m.push((format!("{p}.self_attn.{n}.bias"), vec![768], GgmlType::F16));
        }
    }
    m.extend([
        (
            "model.head.decoder.embed.weight".into(),
            vec![1025, 320],
            GgmlType::F32,
        ),
        (
            "model.head.decoder.lstm.weight_ih_l0".into(),
            vec![1280, 320],
            GgmlType::F32,
        ),
        (
            "model.head.decoder.lstm.weight_hh_l0".into(),
            vec![1280, 320],
            GgmlType::F32,
        ),
        (
            "model.head.decoder.lstm.bias_ih_l0".into(),
            vec![1280],
            GgmlType::F32,
        ),
        (
            "model.head.decoder.lstm.bias_hh_l0".into(),
            vec![1280],
            GgmlType::F32,
        ),
        (
            "model.head.joint.pred.weight".into(),
            vec![320, 320],
            GgmlType::F32,
        ),
        (
            "model.head.joint.pred.bias".into(),
            vec![320],
            GgmlType::F32,
        ),
        (
            "model.head.joint.enc.weight".into(),
            vec![320, 768],
            GgmlType::F32,
        ),
        ("model.head.joint.enc.bias".into(), vec![320], GgmlType::F32),
        (
            "model.head.joint.joint_net.1.weight".into(),
            vec![1025, 320],
            GgmlType::F32,
        ),
        (
            "model.head.joint.joint_net.1.bias".into(),
            vec![1025],
            GgmlType::F32,
        ),
    ]);
    m
}

fn required(root: &JsonValue, key: &str, expected: &str) -> Result<(), ConvertError> {
    if root.get(key).and_then(JsonValue::as_str) != Some(expected) {
        return Err(ConvertError::Parse(format!(
            "GigaAM v3 sidecar `{key}` mismatch"
        )));
    }
    Ok(())
}

fn exact_unique_keys(object: &[(String, JsonValue)], expected: &[&str]) -> bool {
    let keys: BTreeSet<&str> = object.iter().map(|(key, _)| key.as_str()).collect();
    keys.len() == object.len()
        && keys.len() == expected.len()
        && expected.iter().all(|key| keys.contains(key))
        && object
            .iter()
            .all(|(key, _)| expected.contains(&key.as_str()))
}

fn validate_sidecar(
    bytes: &[u8],
    manifest: &[(String, Vec<u64>, GgmlType)],
) -> Result<String, ConvertError> {
    let root =
        json::parse(bytes).map_err(|e| ConvertError::Parse(format!("GigaAM v3 sidecar: {e}")))?;
    let object = root
        .as_object()
        .ok_or_else(|| ConvertError::Parse("GigaAM v3 sidecar must be an object".into()))?;
    let allowed = [
        "format",
        "repository",
        "revision",
        "source_revision",
        "config_sha256",
        "modeling_sha256",
        "tokenizer_sha256",
        "checkpoint_sha256",
        "prepared_sha256",
        "tensor_count",
        "tensors",
    ];
    if !exact_unique_keys(object, &allowed) {
        return Err(ConvertError::Parse(
            "GigaAM v3 sidecar schema mismatch".into(),
        ));
    }
    required(&root, "format", PREPARED_FORMAT)?;
    required(&root, "repository", UPSTREAM_HF)?;
    required(&root, "revision", UPSTREAM_REVISION)?;
    required(&root, "source_revision", UPSTREAM_SOURCE_REVISION)?;
    required(&root, "config_sha256", CONFIG_SHA256)?;
    required(&root, "modeling_sha256", MODELING_SHA256)?;
    required(&root, "tokenizer_sha256", TOKENIZER_SHA256)?;
    required(&root, "checkpoint_sha256", CHECKPOINT_SHA256)?;
    let sha = root
        .get("prepared_sha256")
        .and_then(JsonValue::as_str)
        .ok_or_else(|| ConvertError::Parse("GigaAM v3 prepared_sha256 must be a string".into()))?;
    if sha.len() != 64
        || !sha
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Err(ConvertError::Parse(
            "GigaAM v3 prepared_sha256 must be SHA-256 hex".into(),
        ));
    }
    if root.get("tensor_count").and_then(JsonValue::as_u64) != Some(TENSOR_COUNT as u64) {
        return Err(ConvertError::Parse(
            "GigaAM v3 tensor count mismatch".into(),
        ));
    }
    let rows = root
        .get("tensors")
        .and_then(JsonValue::as_array)
        .ok_or_else(|| ConvertError::Parse("GigaAM v3 tensor rows missing".into()))?;
    if rows.len() != manifest.len() {
        return Err(ConvertError::Parse("GigaAM v3 tensor rows mismatch".into()));
    }
    for ((name, shape, dtype), row) in manifest.iter().zip(rows) {
        let obj = row
            .as_object()
            .ok_or_else(|| ConvertError::Parse("GigaAM v3 tensor row is not an object".into()))?;
        if !exact_unique_keys(obj, &["name", "shape", "dtype"]) {
            return Err(ConvertError::Parse(format!(
                "GigaAM v3 tensor row `{name}` schema mismatch"
            )));
        }
        let want_dtype = if *dtype == GgmlType::F16 {
            "F16"
        } else {
            "F32"
        };
        if row.get("name").and_then(JsonValue::as_str) != Some(name)
            || row.get("dtype").and_then(JsonValue::as_str) != Some(want_dtype)
        {
            return Err(ConvertError::Parse(format!(
                "GigaAM v3 tensor `{name}` identity mismatch"
            )));
        }
        let got = row
            .get("shape")
            .and_then(JsonValue::as_array)
            .ok_or_else(|| ConvertError::Parse(format!("GigaAM v3 tensor `{name}` shape missing")))?
            .iter()
            .map(JsonValue::as_u64)
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| {
                ConvertError::Parse(format!("GigaAM v3 tensor `{name}` shape invalid"))
            })?;
        if &got != shape {
            return Err(ConvertError::Parse(format!(
                "GigaAM v3 tensor `{name}` shape mismatch"
            )));
        }
    }
    Ok(sha.to_owned())
}

fn validate_checkpoint(
    file: &SafetensorsFile,
    manifest: &[(String, Vec<u64>, GgmlType)],
) -> Result<(), ConvertError> {
    // A safetensors header has no semantic ordering guarantee: for example,
    // `safetensors.torch.save_file` may serialize a manifest-ordered mapping
    // in a different order. Copy only the small descriptors so validation can
    // compare names independently of that serialization detail.
    let actual = file
        .tensors()
        .iter()
        .map(|tensor| (tensor.name.clone(), tensor.shape.clone(), tensor.dtype))
        .collect::<Vec<_>>();
    validate_manifest_metadata(&actual, manifest)
}

fn validate_manifest_metadata(
    actual: &[(String, Vec<u64>, GgmlType)],
    manifest: &[(String, Vec<u64>, GgmlType)],
) -> Result<(), ConvertError> {
    if actual.len() != manifest.len() {
        return Err(ConvertError::Parse(
            "GigaAM v3 tensor count mismatch".into(),
        ));
    }

    let expected_names: BTreeSet<&str> =
        manifest.iter().map(|(name, _, _)| name.as_str()).collect();
    if expected_names.len() != manifest.len() {
        return Err(ConvertError::Parse(
            "GigaAM v3 manifest tensor names are not unique".into(),
        ));
    }
    let actual_names: BTreeSet<&str> = actual.iter().map(|(name, _, _)| name.as_str()).collect();
    if actual_names.len() != actual.len() {
        return Err(ConvertError::Parse(
            "GigaAM v3 tensor names are not unique".into(),
        ));
    }
    if actual_names != expected_names {
        return Err(ConvertError::Parse(
            "GigaAM v3 tensor names mismatch".into(),
        ));
    }

    // Dtype and shape are bound to each name, never to the serialized order.
    for (name, expected_shape, expected_dtype) in manifest {
        let (_, actual_shape, actual_dtype) = actual
            .iter()
            .find(|(actual_name, _, _)| actual_name == name)
            .expect("validated GigaAM v3 tensor name set");
        if actual_shape != expected_shape || actual_dtype != expected_dtype {
            return Err(ConvertError::Parse(format!(
                "GigaAM v3 tensor `{name}` dtype/shape mismatch"
            )));
        }
    }
    Ok(())
}

/// Convert one strict prepared v3 artifact to GGUF.
pub fn convert_sber_gigaam_v3_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<SberGigaamV3Report, ConvertError> {
    if license.is_some_and(|v| !v.eq_ignore_ascii_case("mit")) {
        return Err(ConvertError::Usage(
            "GigaAM v3 weights are fixed MIT; license override must be `mit`".into(),
        ));
    }
    let Some(expected_sha) = AUTHENTICATED_PREPARED_SHA256 else {
        return Err(ConvertError::Usage("GigaAM v3 prepared SHA-256 is not independently authenticated; obtain VAST evidence first".into()));
    };
    let manifest = expected_manifest();
    let input_sha = hex(&sha256(&std::fs::read(input)?));
    let sidecar_sha = validate_sidecar(&std::fs::read(sidecar_path(input))?, &manifest)?;
    if input_sha != expected_sha || sidecar_sha != expected_sha {
        return Err(ConvertError::Parse(
            "GigaAM v3 prepared SHA mismatch".into(),
        ));
    }
    let checkpoint = SafetensorsFile::open(input)?;
    validate_checkpoint(&checkpoint, &manifest)?;
    let mut builder = GgufBuilder::new();
    builder
        .add_string(chunks::KEY_MODEL_ARCH, ARCH)
        .add_string(chunks::KEY_MODEL_NAME, NAME);
    for (key, value) in [
        ("sample_rate", 16000),
        ("n_mels", 64),
        ("n_fft", 320),
        ("hop_length", 160),
        ("win_length", 320),
        ("n_layers", 16),
        ("d_model", 768),
        ("n_heads", 16),
        ("ffn_dim", 3072),
        ("conv_kernel_size", 5),
        ("subsampling_kernel_size", 5),
        ("subsampling_stride", 2),
        ("subsampling_padding", 2),
        ("pred_hidden", 320),
        ("pred_rnn_layers", 1),
        ("joint_hidden", 320),
        ("vocab_size", 1025),
        ("blank_id", 1024),
    ] {
        builder.add_u32(&format!("{KEY_PREFIX}.{key}"), value);
    }
    for (key, value) in [
        ("preprocessor_center", "false"),
        ("mel_scale", "htk"),
        ("mel_norm", "None"),
        ("power", "2"),
    ] {
        builder.add_string(&format!("{KEY_PREFIX}.{key}"), value);
    }
    for (key, value) in [
        ("model_class", "rnnt"),
        ("model_name", "v3_e2e_rnnt"),
        ("topology", "RNNT"),
        ("revision", UPSTREAM_REVISION),
        ("source_revision", UPSTREAM_SOURCE_REVISION),
        ("config_sha256", CONFIG_SHA256),
        ("modeling_sha256", MODELING_SHA256),
        ("tokenizer_sha256", TOKENIZER_SHA256),
        ("checkpoint_sha256", CHECKPOINT_SHA256),
        ("prepared_sha256", expected_sha),
    ] {
        builder.add_string(&format!("{KEY_PREFIX}.{key}"), value);
    }
    vokra_core::stamp_provenance(
        &mut builder,
        LicenseClass::Permissive,
        "MIT",
        Some(UPSTREAM_HF),
        Some("https://huggingface.co/ai-sage/GigaAM-v3"),
    );
    // Emit in authenticated manifest order for deterministic GGUF bytes;
    // safetensors header order is an implementation detail, not model order.
    for (name, _, _) in &manifest {
        let tensor = checkpoint
            .tensor_info(name)
            .expect("validated GigaAM v3 tensor name set");
        builder.add_tensor(
            name,
            tensor.dtype,
            tensor.shape.clone(),
            checkpoint.tensor_bytes(tensor).to_vec(),
        )?;
    }
    std::fs::write(output, builder.to_bytes()?)?;
    Ok(SberGigaamV3Report {
        read: TENSOR_COUNT,
        written: TENSOR_COUNT,
        skipped_non_float: 0,
        bf16_passthrough: 0,
    })
}

fn sidecar_path(input: &Path) -> PathBuf {
    let mut p = input.to_owned();
    p.set_extension("manifest.json");
    p
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metadata(name: &str, shape: &[u64], dtype: GgmlType) -> (String, Vec<u64>, GgmlType) {
        (name.to_owned(), shape.to_vec(), dtype)
    }

    #[test]
    fn manifest_count_and_gate() {
        assert_eq!(expected_manifest().len(), TENSOR_COUNT);
        assert_eq!(
            AUTHENTICATED_PREPARED_SHA256,
            Some("cee04765f031d6ee5088849ecb0e5c1db4e58ca28a345ce4d049015cd683a64e")
        );
    }
    #[test]
    fn missing_input_rejected() {
        let out = std::env::temp_dir().join("gigaam-v3-rejected.gguf");
        let _ = std::fs::remove_file(&out);
        let _error = convert_sber_gigaam_v3_file(Path::new("missing"), &out, None)
            .unwrap_err()
            .to_string();
        assert_eq!(_error, "I/O error: No such file or directory (os error 2)");
        assert!(!out.exists());
    }

    #[test]
    fn sidecar_key_contract_rejects_duplicates_and_unknowns() {
        let duplicate = vec![
            ("name".to_owned(), JsonValue::Str("x".to_owned())),
            ("name".to_owned(), JsonValue::Str("y".to_owned())),
            ("shape".to_owned(), JsonValue::Array(vec![])),
            ("dtype".to_owned(), JsonValue::Str("F32".to_owned())),
        ];
        assert!(!exact_unique_keys(&duplicate, &["name", "shape", "dtype"]));
        let unknown = vec![
            ("name".to_owned(), JsonValue::Str("x".to_owned())),
            ("shape".to_owned(), JsonValue::Array(vec![])),
            ("dtype".to_owned(), JsonValue::Str("F32".to_owned())),
            ("extra".to_owned(), JsonValue::Null),
        ];
        assert!(!exact_unique_keys(&unknown, &["name", "shape", "dtype"]));
    }

    #[test]
    fn manifest_names_are_unique_and_order_bound() {
        let manifest = expected_manifest();
        let names: BTreeSet<&str> = manifest.iter().map(|(name, _, _)| name.as_str()).collect();
        assert_eq!(names.len(), manifest.len());
        assert_eq!(
            manifest.first().map(|row| row.0.as_str()),
            Some("model.preprocessor.featurizer.0.spectrogram.window")
        );
        assert_eq!(
            manifest.last().map(|row| row.0.as_str()),
            Some("model.head.joint.joint_net.1.bias")
        );
    }

    #[test]
    fn checkpoint_validation_ignores_header_order_but_rejects_metadata_drift() {
        let manifest = vec![
            metadata("first", &[2], GgmlType::F32),
            metadata("second", &[3], GgmlType::F16),
        ];
        let reordered = vec![manifest[1].clone(), manifest[0].clone()];
        assert!(validate_manifest_metadata(&reordered, &manifest).is_ok());

        let missing = vec![manifest[0].clone()];
        assert!(validate_manifest_metadata(&missing, &manifest).is_err());

        let extra = vec![
            manifest[0].clone(),
            metadata("unexpected", &[3], GgmlType::F16),
        ];
        assert!(validate_manifest_metadata(&extra, &manifest).is_err());

        let duplicate = vec![manifest[0].clone(), manifest[0].clone()];
        assert!(validate_manifest_metadata(&duplicate, &manifest).is_err());

        let wrong_shape = vec![metadata("first", &[9], GgmlType::F32), manifest[1].clone()];
        assert!(validate_manifest_metadata(&wrong_shape, &manifest).is_err());

        let wrong_dtype = vec![metadata("first", &[2], GgmlType::F16), manifest[1].clone()];
        assert!(validate_manifest_metadata(&wrong_dtype, &manifest).is_err());
    }

    #[test]
    fn parsed_reordered_checkpoint_is_accepted() {
        let header = r#"{"second":{"dtype":"F16","shape":[3],"data_offsets":[0,6]},"first":{"dtype":"F32","shape":[2],"data_offsets":[6,14]}}"#;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&[0u8; 14]);
        let checkpoint = SafetensorsFile::parse(bytes).expect("synthetic checkpoint");
        let manifest = vec![
            metadata("first", &[2], GgmlType::F32),
            metadata("second", &[3], GgmlType::F16),
        ];
        assert_eq!(checkpoint.tensors()[0].name, "second");
        assert!(validate_checkpoint(&checkpoint, &manifest).is_ok());
    }
}
