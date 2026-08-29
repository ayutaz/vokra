//! Authenticated GigaAM Multilingual CTC safetensors → GGUF conversion.
//!
//! The input is a prepared safetensors file accompanied by the fixed-revision
//! manifest emitted by the VAST preparation worker. Raw checkpoints and
//! arbitrary shape-like sidecars are rejected.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};
use vokra_core::json::{self, JsonValue};

use super::canary_1b_flash::{hex, sha256};
use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

pub const ARCH: &str = "gigaam_multilingual";
pub const NAME: &str = "sber-gigaam-multilingual";
pub const UPSTREAM_HF: &str = "ai-sage/GigaAM-Multilingual";
pub const UPSTREAM_REVISION: &str = "2f8a57144e6ec3adfd32fe0484d9ea9913305bc8";
pub const UPSTREAM_SOURCE_REVISION: &str = "7447938d791c4f3e643386ee22c33777004293a5";
pub const CHECKPOINT_SHA256: &str =
    "e1db43873ec5e296f229572e06e2470fc157ac9f8d4aacabda295630b9b91728";
pub const CONFIG_SHA256: &str = "c830232c7d51688a630a221517b52585ab5ee57e1d3c21bcbae01759351d2653";
pub const PREPARED_FORMAT: &str = "vokra-gigaam-multilingual-prepared-v1";
pub const TENSOR_COUNT: usize = 552;
/// No prepared safetensors digest has been independently authenticated yet.
/// Keep conversion impossible until a VAST-produced digest is reviewed and
/// recorded here; a self-declared sidecar digest is not an authentication.
pub const AUTHENTICATED_PREPARED_SHA256: Option<&str> = None;
const SAMPLE_RATE: u32 = 16_000;
const N_MELS: u32 = 64;
const N_FFT: u32 = 320;
const HOP_LENGTH: u32 = 160;
const WIN_LENGTH: u32 = 320;
const N_LAYER: u32 = 16;
const D_MODEL: u32 = 768;
const N_HEAD: u32 = 16;
const FFN_DIM: u32 = 3_072;
const CONV_KERNEL: u32 = 5;
const SUB_KERNEL: u32 = 5;
const SUB_STRIDE: u32 = 2;
const SUB_PADDING: u32 = 2;
const VOCAB_SIZE: u32 = 71;
const BLANK_ID: u32 = 70;
const KEY_PREFIX: &str = "vokra.gigaam_multilingual";

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
/// Counts emitted by the strict prepared-artifact conversion path.
pub struct SberGigaamMultilingualReport {
    /// Number of authenticated source tensors read from the prepared file.
    pub read: usize,
    /// Number of tensors written verbatim into the GGUF output.
    pub written: usize,
    /// Number of non-F32 tensors skipped (always zero for this strict route).
    pub skipped_non_float: usize,
    /// Number of BF16 tensors passed through unchanged (always zero here).
    pub bf16_passthrough: usize,
}

fn expected_manifest() -> Vec<(String, Vec<u64>)> {
    let mut result = vec![
        (
            "model.preprocessor.featurizer.0.spectrogram.window".into(),
            vec![320],
        ),
        (
            "model.preprocessor.featurizer.0.mel_scale.fb".into(),
            vec![161, 64],
        ),
        (
            "model.encoder.pre_encode.conv.0.weight".into(),
            vec![768, 64, 5],
        ),
        ("model.encoder.pre_encode.conv.0.bias".into(), vec![768]),
        (
            "model.encoder.pre_encode.conv.2.weight".into(),
            vec![768, 768, 5],
        ),
        ("model.encoder.pre_encode.conv.2.bias".into(), vec![768]),
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
            result.push((format!("{p}.{n}.weight"), vec![768]));
            result.push((format!("{p}.{n}.bias"), vec![768]));
        }
        for branch in ["feed_forward1", "feed_forward2"] {
            result.push((format!("{p}.{branch}.linear1.weight"), vec![3072, 768]));
            result.push((format!("{p}.{branch}.linear1.bias"), vec![3072]));
            result.push((format!("{p}.{branch}.linear2.weight"), vec![768, 3072]));
            result.push((format!("{p}.{branch}.linear2.bias"), vec![768]));
        }
        result.push((
            format!("{p}.conv.pointwise_conv1.weight"),
            vec![1536, 768, 1],
        ));
        result.push((format!("{p}.conv.pointwise_conv1.bias"), vec![1536]));
        result.push((format!("{p}.conv.depthwise_conv.weight"), vec![768, 1, 5]));
        result.push((format!("{p}.conv.depthwise_conv.bias"), vec![768]));
        result.push((format!("{p}.conv.batch_norm.weight"), vec![768]));
        result.push((format!("{p}.conv.batch_norm.bias"), vec![768]));
        result.push((
            format!("{p}.conv.pointwise_conv2.weight"),
            vec![768, 768, 1],
        ));
        result.push((format!("{p}.conv.pointwise_conv2.bias"), vec![768]));
        for n in ["linear_q", "linear_k", "linear_v", "linear_out"] {
            result.push((format!("{p}.self_attn.{n}.weight"), vec![768, 768]));
            result.push((format!("{p}.self_attn.{n}.bias"), vec![768]));
        }
    }
    result.push((
        "model.head.decoder_layers.0.weight".into(),
        vec![71, 768, 1],
    ));
    result.push(("model.head.decoder_layers.0.bias".into(), vec![71]));
    result
}

fn require_str(root: &JsonValue, key: &str, expected: &str) -> Result<(), ConvertError> {
    if root.get(key).and_then(JsonValue::as_str) != Some(expected) {
        return Err(ConvertError::Parse(format!(
            "GigaAM sidecar `{key}` must be {expected:?}"
        )));
    }
    Ok(())
}

fn require_u64(root: &JsonValue, key: &str, expected: u64) -> Result<(), ConvertError> {
    if root.get(key).and_then(JsonValue::as_u64) != Some(expected) {
        return Err(ConvertError::Parse(format!(
            "GigaAM sidecar `{key}` must be {expected}"
        )));
    }
    Ok(())
}

fn exact_object_schema(
    value: &JsonValue,
    allowed: &[&str],
    context: &str,
) -> Result<(), ConvertError> {
    let entries = value
        .as_object()
        .ok_or_else(|| ConvertError::Parse(format!("GigaAM {context} must be an object")))?;
    let allowed: BTreeSet<&str> = allowed.iter().copied().collect();
    let mut seen = BTreeSet::new();
    for (key, _) in entries {
        if !allowed.contains(key.as_str()) {
            return Err(ConvertError::Parse(format!(
                "GigaAM {context} contains unknown key `{key}`"
            )));
        }
        if !seen.insert(key.as_str()) {
            return Err(ConvertError::Parse(format!(
                "GigaAM {context} contains duplicate key `{key}`"
            )));
        }
    }
    if seen.len() != allowed.len() {
        let mut missing = Vec::new();
        for key in &allowed {
            if !seen.contains(key) {
                missing.push(*key);
            }
        }
        return Err(ConvertError::Parse(format!(
            "GigaAM {context} is missing required keys {missing:?}"
        )));
    }
    Ok(())
}

fn validate_sidecar(bytes: &[u8], manifest: &[(String, Vec<u64>)]) -> Result<String, ConvertError> {
    let root =
        json::parse(bytes).map_err(|e| ConvertError::Parse(format!("GigaAM sidecar: {e}")))?;
    exact_object_schema(
        &root,
        &[
            "format",
            "repository",
            "revision",
            "source_revision",
            "config_sha256",
            "checkpoint_sha256",
            "prepared_sha256",
            "tensor_count",
            "tensors",
        ],
        "sidecar",
    )?;
    require_str(&root, "format", PREPARED_FORMAT)?;
    require_str(&root, "repository", UPSTREAM_HF)?;
    require_str(&root, "revision", UPSTREAM_REVISION)?;
    require_str(&root, "source_revision", UPSTREAM_SOURCE_REVISION)?;
    require_str(&root, "config_sha256", CONFIG_SHA256)?;
    require_str(&root, "checkpoint_sha256", CHECKPOINT_SHA256)?;
    let prepared_sha = root
        .get("prepared_sha256")
        .and_then(JsonValue::as_str)
        .ok_or_else(|| {
            ConvertError::Parse("GigaAM sidecar prepared_sha256 must be a string".into())
        })?;
    if prepared_sha.len() != 64 || !prepared_sha.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(ConvertError::Parse(
            "GigaAM sidecar prepared_sha256 must be a 64-digit hex SHA-256".into(),
        ));
    }
    require_u64(&root, "tensor_count", TENSOR_COUNT as u64)?;
    let rows = root
        .get("tensors")
        .and_then(JsonValue::as_array)
        .ok_or_else(|| ConvertError::Parse("GigaAM sidecar tensors must be an array".into()))?;
    if rows.len() != manifest.len() {
        return Err(ConvertError::Parse(
            "GigaAM sidecar tensor manifest count mismatch".into(),
        ));
    }
    for ((expected_name, expected_shape), row) in manifest.iter().zip(rows) {
        exact_object_schema(row, &["name", "shape", "dtype"], "sidecar tensor row")?;
        if row.get("name").and_then(JsonValue::as_str) != Some(expected_name) {
            return Err(ConvertError::Parse(format!(
                "GigaAM sidecar tensor name mismatch: {expected_name}"
            )));
        }
        let shape = row
            .get("shape")
            .and_then(JsonValue::as_array)
            .ok_or_else(|| {
                ConvertError::Parse(format!(
                    "GigaAM sidecar tensor `{expected_name}` shape missing"
                ))
            })?;
        let actual: Vec<u64> = shape
            .iter()
            .map(JsonValue::as_u64)
            .collect::<Option<_>>()
            .ok_or_else(|| {
                ConvertError::Parse(format!(
                    "GigaAM sidecar tensor `{expected_name}` shape is not integers"
                ))
            })?;
        if &actual != expected_shape || row.get("dtype").and_then(JsonValue::as_str) != Some("F32")
        {
            return Err(ConvertError::Parse(format!(
                "GigaAM sidecar tensor `{expected_name}` dtype/shape mismatch"
            )));
        }
    }
    Ok(prepared_sha.to_owned())
}

fn validate_checkpoint(
    checkpoint: &SafetensorsFile,
    manifest: &[(String, Vec<u64>)],
) -> Result<(), ConvertError> {
    let expected_names: BTreeSet<&str> = manifest.iter().map(|(name, _)| name.as_str()).collect();
    let actual_names: BTreeSet<&str> = checkpoint
        .tensors()
        .iter()
        .map(|tensor| tensor.name.as_str())
        .collect();
    if expected_names != actual_names {
        return Err(ConvertError::Parse(format!(
            "GigaAM tensor manifest mismatch: expected {}, found {}",
            expected_names.len(),
            actual_names.len()
        )));
    }
    for (name, shape) in manifest {
        let tensor = checkpoint
            .tensor_info(name)
            .ok_or_else(|| ConvertError::Parse(format!("GigaAM tensor `{name}` missing")))?;
        if tensor.dtype != GgmlType::F32 || tensor.shape != *shape {
            return Err(ConvertError::Parse(format!(
                "GigaAM tensor `{name}` must be F32 {shape:?}, found {:?} {:?}",
                tensor.dtype, tensor.shape
            )));
        }
    }
    Ok(())
}

/// Convert only a prepared, manifest-bound checkpoint. The sidecar is looked
/// up as `<input stem>.manifest.json`.
pub fn convert_sber_gigaam_multilingual_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<SberGigaamMultilingualReport, ConvertError> {
    if let Some(value) = license {
        if !value.eq_ignore_ascii_case("mit") {
            return Err(ConvertError::Usage(
                "GigaAM Multilingual weights are fixed MIT; license override must be `mit`".into(),
            ));
        }
    }
    let Some(expected_prepared_sha256) = AUTHENTICATED_PREPARED_SHA256 else {
        return Err(ConvertError::Usage(
            "GigaAM prepared safetensors digest is not independently authenticated; obtain and review the VAST artifact SHA-256 before conversion".into(),
        ));
    };
    let sidecar = sidecar_path(input);
    let sidecar_bytes = std::fs::read(&sidecar)?;
    let manifest = expected_manifest();
    if manifest.len() != TENSOR_COUNT {
        return Err(ConvertError::Parse(
            "GigaAM internal manifest count drift".into(),
        ));
    }
    let sidecar_prepared_sha256 = validate_sidecar(&sidecar_bytes, &manifest)?;
    let bytes = std::fs::read(input)?;
    let actual_prepared_sha256 = hex(&sha256(&bytes));
    if sidecar_prepared_sha256 != expected_prepared_sha256
        || actual_prepared_sha256 != expected_prepared_sha256
    {
        return Err(ConvertError::Parse(format!(
            "GigaAM prepared artifact SHA-256 mismatch: sidecar={sidecar_prepared_sha256}, bytes={actual_prepared_sha256}, expected={expected_prepared_sha256}"
        )));
    }
    let checkpoint = SafetensorsFile::parse(bytes)?;
    validate_checkpoint(&checkpoint, &manifest)?;
    let mut builder = GgufBuilder::new();
    builder.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    builder.add_string(chunks::KEY_MODEL_NAME, NAME);
    for (key, value) in [
        ("sample_rate", SAMPLE_RATE),
        ("n_mels", N_MELS),
        ("n_fft", N_FFT),
        ("hop_length", HOP_LENGTH),
        ("win_length", WIN_LENGTH),
        ("n_layers", N_LAYER),
        ("d_model", D_MODEL),
        ("n_heads", N_HEAD),
        ("ffn_dim", FFN_DIM),
        ("conv_kernel_size", CONV_KERNEL),
        ("subsampling_kernel_size", SUB_KERNEL),
        ("subsampling_stride", SUB_STRIDE),
        ("subsampling_padding", SUB_PADDING),
        ("vocab_size", VOCAB_SIZE),
        ("blank_id", BLANK_ID),
    ] {
        builder.add_u32(&format!("{KEY_PREFIX}.{key}"), value);
    }
    for (key, value) in [
        ("model_class", "ctc"),
        ("model_name", "multilingual_ctc"),
        ("topology", "CTC"),
        ("source_revision", UPSTREAM_SOURCE_REVISION),
        ("config_sha256", CONFIG_SHA256),
        ("checkpoint_sha256", CHECKPOINT_SHA256),
        ("prepared_sha256", expected_prepared_sha256),
    ] {
        builder.add_string(&format!("{KEY_PREFIX}.{key}"), value);
    }
    builder.add_string("vokra.gigaam_multilingual.revision", UPSTREAM_REVISION);
    vokra_core::stamp_provenance(
        &mut builder,
        LicenseClass::Permissive,
        "MIT",
        Some(UPSTREAM_HF),
        Some("https://huggingface.co/ai-sage/GigaAM-Multilingual"),
    );
    for tensor in checkpoint.tensors() {
        builder.add_tensor(
            &tensor.name,
            tensor.dtype,
            tensor.shape.clone(),
            checkpoint.tensor_bytes(tensor).to_vec(),
        )?;
    }
    let out_bytes = builder.to_bytes()?;
    std::fs::write(output, out_bytes)?;
    Ok(SberGigaamMultilingualReport {
        read: TENSOR_COUNT,
        written: TENSOR_COUNT,
        skipped_non_float: 0,
        bf16_passthrough: 0,
    })
}

fn sidecar_path(input: &Path) -> PathBuf {
    let mut path = input.to_owned();
    path.set_extension("manifest.json");
    path
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_manifest_has_authenticated_count_and_head() {
        let manifest = expected_manifest();
        assert_eq!(manifest.len(), TENSOR_COUNT);
        assert!(
            !manifest.iter().any(|(name, _)| {
                name.ends_with("running_mean") || name.ends_with("running_var")
            })
        );
        assert_eq!(
            manifest[0],
            (
                "model.preprocessor.featurizer.0.spectrogram.window".into(),
                vec![320]
            )
        );
        assert_eq!(
            manifest.last().unwrap().0,
            "model.head.decoder_layers.0.bias"
        );
    }

    #[test]
    fn raw_or_missing_sidecar_is_rejected() {
        let output = std::env::temp_dir().join("gigaam-multilingual-rejected.gguf");
        let error =
            convert_sber_gigaam_multilingual_file(Path::new("missing.safetensors"), &output, None)
                .unwrap_err()
                .to_string();
        assert!(error.contains("prepared safetensors digest"));
        assert!(!output.exists());
    }

    #[test]
    fn conversion_stays_closed_without_independent_prepared_digest() {
        assert!(AUTHENTICATED_PREPARED_SHA256.is_none());
    }

    #[test]
    fn license_override_cannot_relabel_weights() {
        let error = convert_sber_gigaam_multilingual_file(
            Path::new("missing"),
            Path::new("out"),
            Some("apache-2.0"),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("MIT"));
    }
}
