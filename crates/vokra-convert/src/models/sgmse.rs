//! Strict prepared-safetensors → GGUF conversion for SGMSE-VoiceBank.
//!
//! The upstream release is a pickle checkpoint. Python preparation is kept
//! outside this crate; this converter accepts only the resulting
//! sgmse_voicebank.safetensors and its exact sibling manifest.
//! The prepared-byte digest is populated only from an independent VAST review.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use vokra_core::LicenseClass;
use vokra_core::gguf::{
    GgmlType, GgufArray, GgufBuilder, GgufMetadataValue, GgufValueType, chunks,
};
use vokra_core::json::{self, JsonValue};

use super::canary_1b_flash::{hex, sha256};
use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

pub const ARCH: &str = "sgmse_voicebank";
pub const NAME: &str = "sgmse-voicebank";
pub const PREPARED_FORMAT: &str = "vokra-sgmse-voicebank-prepared-v1";
pub const UPSTREAM_HF: &str = "speechbrain/sgmse-voicebank";
pub const UPSTREAM_REVISION: &str = "8f4ff7b65284c49492a43349b8106e094ac0d365";
pub const SOURCE_REPOSITORY: &str = "https://github.com/sp-uhh/sgmse.git";
pub const SOURCE_REVISION: &str = "1961cf4483e37df1bb92ccf0eb8b28bf6f44cb0e";
pub const CHECKPOINT_FILENAME: &str = "score_model_ema.ckpt";
pub const CHECKPOINT_SIZE: u64 = 262_593_305;
pub const CHECKPOINT_SHA256: &str =
    "7ca96321aca40cdca90c450d1450a5c7f343935e5b46ee34a1b575f9f774ccc3";
pub const TENSOR_COUNT: usize = 647;
/// Independently observed VAST SHA-256 of the prepared safetensors bytes.
pub const AUTHENTICATED_PREPARED_SHA256: Option<&str> =
    Some("eda3064488c670db78f7f93a77fade739e0d3941b7863e28db489a137f723045");
/// Digest of the complete, independently reviewed role/name/dtype/shape map.
pub const REVIEWED_TENSOR_MANIFEST_SHA256: &str =
    "409690f70b534771055dc4f740cc66bdb4d1b25dba5e22fd066109adce77278c";

const KEY_SOURCE_REPOSITORY: &str = "vokra.sgmse.source_repository";
const KEY_SOURCE_REVISION: &str = "vokra.sgmse.source_revision";
const KEY_CHECKPOINT_FILENAME: &str = "vokra.sgmse.checkpoint_filename";
const KEY_CHECKPOINT_SIZE: &str = "vokra.sgmse.checkpoint_size";
const KEY_CHECKPOINT_SHA256: &str = "vokra.sgmse.checkpoint_sha256";
const KEY_PREPARED_SHA256: &str = "vokra.sgmse.prepared_sha256";
const KEY_MODEL_REVISION: &str = "vokra.sgmse.model_revision";
const KEY_MANIFEST_STATUS: &str = "vokra.sgmse.manifest_status";
const KEY_TENSOR_MANIFEST_SHA256: &str = "vokra.sgmse.tensor_manifest_sha256";
const KEY_TENSOR_MANIFEST: &str = "vokra.sgmse.tensor_manifest";
const MAX_SIDECAR_BYTES: u64 = 2 * 1024 * 1024;

/// Counts emitted by the strict prepared-artifact conversion path.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct SgmseReport {
    /// Number of authenticated source tensors read.
    pub read: usize,
    /// Number of tensors written verbatim into GGUF.
    pub written: usize,
    /// Number of non-floating-point tensors skipped (always zero here).
    pub skipped_non_float: usize,
    /// Number of BF16 tensors preserved byte-for-byte.
    pub bf16_passthrough: usize,
}

/// One row in the prepared checkpoint's complete typed contract.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SgmseManifestRow {
    pub name: String,
    pub role: String,
    pub dtype_tag: u32,
    pub shape: Vec<u64>,
    pub dimensions: Vec<u64>,
}

#[derive(Debug, Clone)]
struct PreparedManifest {
    prepared_sha256: String,
    typed_manifest_sha256: String,
    rows: Vec<SgmseManifestRow>,
}

fn parse_error(message: impl Into<String>) -> ConvertError {
    ConvertError::Parse(format!("SGMSE {}", message.into()))
}

fn exact_object_schema(
    value: &JsonValue,
    allowed: &[&str],
    context: &str,
) -> Result<(), ConvertError> {
    let entries = value
        .as_object()
        .ok_or_else(|| parse_error(format!("{context} must be an object")))?;
    let allowed: BTreeSet<&str> = allowed.iter().copied().collect();
    let mut seen = BTreeSet::new();
    for (key, _) in entries {
        if !allowed.contains(key.as_str()) {
            return Err(parse_error(format!(
                "{context} contains unknown key {key:?}"
            )));
        }
        if !seen.insert(key.as_str()) {
            return Err(parse_error(format!(
                "{context} contains duplicate key {key:?}"
            )));
        }
    }
    if seen.len() != allowed.len() {
        return Err(parse_error(format!("{context} is missing required keys")));
    }
    Ok(())
}

fn required_str(root: &JsonValue, key: &str, expected: &str) -> Result<(), ConvertError> {
    if root.get(key).and_then(JsonValue::as_str) != Some(expected) {
        return Err(parse_error(format!("manifest {key:?} identity mismatch")));
    }
    Ok(())
}

fn required_u64(root: &JsonValue, key: &str, expected: u64) -> Result<(), ConvertError> {
    if root.get(key).and_then(JsonValue::as_u64) != Some(expected) {
        return Err(parse_error(format!("manifest {key:?} identity mismatch")));
    }
    Ok(())
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn required_sha256(root: &JsonValue, key: &str) -> Result<String, ConvertError> {
    let value = root
        .get(key)
        .and_then(JsonValue::as_str)
        .ok_or_else(|| parse_error(format!("manifest {key:?} must be a string")))?;
    if !valid_sha256(value) || value.bytes().any(|byte| byte.is_ascii_uppercase()) {
        return Err(parse_error(format!(
            "manifest {key:?} must be a lowercase 64-digit SHA-256"
        )));
    }
    Ok(value.to_owned())
}

fn dtype_tag(value: &str) -> Option<u32> {
    match value {
        "torch.float32" => Some(0),
        "torch.float16" => Some(1),
        "torch.bfloat16" => Some(30),
        _ => None,
    }
}

fn parse_positive_dimensions(value: &JsonValue, context: &str) -> Result<Vec<u64>, ConvertError> {
    let values = value
        .as_array()
        .ok_or_else(|| parse_error(format!("{context} must be an array")))?;
    if values.is_empty() {
        return Err(parse_error(format!("{context} must not be empty")));
    }
    values
        .iter()
        .map(|value| {
            let dimension = value
                .as_u64()
                .ok_or_else(|| parse_error(format!("{context} contains a non-positive integer")))?;
            if dimension == 0 {
                return Err(parse_error(format!("{context} contains a zero dimension")));
            }
            Ok(dimension)
        })
        .collect()
}

fn parse_row(value: &JsonValue, index: usize) -> Result<SgmseManifestRow, ConvertError> {
    exact_object_schema(
        value,
        &["name", "role", "dtype", "shape", "dimensions"],
        &format!("tensor row {index}"),
    )?;
    let name = value
        .get("name")
        .and_then(JsonValue::as_str)
        .filter(|name| {
            !name.is_empty()
                && !name.contains('|')
                && !name.chars().any(|character| character.is_control())
        })
        .ok_or_else(|| parse_error(format!("tensor row {index} has an invalid name")))?
        .to_owned();
    let role = value
        .get("role")
        .and_then(JsonValue::as_str)
        .filter(|role| valid_role(role))
        .ok_or_else(|| parse_error(format!("tensor row {index} has an invalid role")))?
        .to_owned();
    let dtype = value
        .get("dtype")
        .and_then(JsonValue::as_str)
        .and_then(dtype_tag)
        .ok_or_else(|| parse_error(format!("tensor row {index} has an unsupported dtype")))?;
    let shape = parse_positive_dimensions(
        value.get("shape").unwrap_or(&JsonValue::Null),
        &format!("tensor row {index} shape"),
    )?;
    let dimensions = parse_positive_dimensions(
        value.get("dimensions").unwrap_or(&JsonValue::Null),
        &format!("tensor row {index} dimensions"),
    )?;
    if dimensions != shape.iter().rev().copied().collect::<Vec<_>>() {
        return Err(parse_error(format!(
            "tensor row {index} dimensions are not reverse(shape)"
        )));
    }
    Ok(SgmseManifestRow {
        name,
        role,
        dtype_tag: dtype,
        shape,
        dimensions,
    })
}

/// Validate the closed typed role/name/dtype/shape declaration.
pub fn validate_typed_manifest(
    rows: &[SgmseManifestRow],
    required_roles: &[String],
) -> Result<(), String> {
    let expected: BTreeSet<_> = required_roles.iter().map(String::as_str).collect();
    if expected.is_empty() || expected.len() != required_roles.len() || rows.len() != expected.len()
    {
        return Err(
            "sgmse: typed manifest required-role set is empty, duplicate, or incomplete".to_owned(),
        );
    }
    let mut names = BTreeSet::new();
    let mut roles = BTreeSet::new();
    for row in rows {
        if row.name.is_empty()
            || row.name.contains('|')
            || row.name.chars().any(|character| character.is_control())
            || !names.insert(row.name.as_str())
            || !roles.insert(row.role.as_str())
            || !expected.contains(row.role.as_str())
            || !valid_role(&row.role)
            || !matches!(row.dtype_tag, 0 | 1 | 30)
            || row.shape.is_empty()
            || row.dimensions.is_empty()
            || row.shape.iter().any(|dimension| *dimension == 0)
            || row.dimensions != row.shape.iter().rev().copied().collect::<Vec<_>>()
        {
            return Err(
                "sgmse: typed manifest has duplicate, unknown, or unsupported row".to_owned(),
            );
        }
    }
    if roles != expected {
        return Err("sgmse: typed manifest is missing or has extra roles".to_owned());
    }
    Ok(())
}

fn valid_role(role: &str) -> bool {
    matches!(
        role,
        "fourier_frequencies"
            | "sigma_first_projection"
            | "sigma_first_bias"
            | "sigma_second_projection"
            | "sigma_second_bias"
    ) || role
        .strip_prefix("stage:")
        .and_then(|rest| {
            let mut fields = rest.split(':');
            let _index = fields.next()?.parse::<usize>().ok()?;
            let kind = fields.next()?;
            let _block = fields.next()?.parse::<usize>().ok()?;
            let module = fields.next()?;
            let slot = fields.next()?;
            if fields.next().is_some() || kind.is_empty() || slot.is_empty() {
                return None;
            }
            let valid_kind = matches!(
                kind,
                "input"
                    | "residual"
                    | "attention"
                    | "downsample"
                    | "upsample"
                    | "progressive_output"
                    | "progressive_input"
                    | "middle"
                    | "output"
            );
            let valid_module = match kind {
                "input" => module == "input_projection",
                "residual" | "middle" | "downsample" | "upsample" => matches!(
                    module,
                    "residual_norm1"
                        | "residual_conv1"
                        | "residual_time_embedding"
                        | "residual_norm2"
                        | "residual_conv2"
                        | "residual_skip"
                ),
                "attention" => matches!(
                    module,
                    "attention_norm"
                        | "attention_query"
                        | "attention_key"
                        | "attention_value"
                        | "attention_output"
                ),
                "progressive_output" => {
                    matches!(module, "progressive_output" | "progressive_output_norm")
                }
                "progressive_input" => module == "progressive_input",
                "output" => module == "output_projection",
                _ => false,
            };
            let valid_slot = if matches!(
                module,
                "residual_norm1" | "residual_norm2" | "attention_norm" | "progressive_output_norm"
            ) {
                matches!(slot, "norm_gamma" | "norm_beta")
            } else {
                matches!(slot, "weight" | "bias")
            };
            (valid_kind && valid_module && valid_slot).then_some(())
        })
        .is_some()
}

fn canonical_typed_manifest_sha256(rows: &[SgmseManifestRow]) -> [u8; 32] {
    let mut ordered: Vec<&SgmseManifestRow> = rows.iter().collect();
    ordered.sort_by(|left, right| left.name.cmp(&right.name));
    let mut bytes = Vec::new();
    for row in ordered {
        bytes.extend_from_slice(row.role.as_bytes());
        bytes.push(0);
        bytes.extend_from_slice(row.name.as_bytes());
        bytes.push(0);
        bytes.extend_from_slice(&row.dtype_tag.to_le_bytes());
        bytes.extend_from_slice(&(row.dimensions.len() as u64).to_le_bytes());
        for dimension in &row.dimensions {
            bytes.extend_from_slice(&dimension.to_le_bytes());
        }
    }
    sha256(&bytes)
}

fn validate_prepared_sidecar(bytes: &[u8]) -> Result<PreparedManifest, ConvertError> {
    let root = json::parse(bytes).map_err(|error| parse_error(format!("sidecar: {error}")))?;
    exact_object_schema(
        &root,
        &[
            "format",
            "repository",
            "model_revision",
            "source_repository",
            "source_revision",
            "checkpoint_filename",
            "checkpoint_size",
            "checkpoint_sha256",
            "prepared_sha256",
            "tensor_count",
            "typed_manifest_sha256",
            "tensor_rows",
        ],
        "sidecar",
    )?;
    required_str(&root, "format", PREPARED_FORMAT)?;
    required_str(&root, "repository", UPSTREAM_HF)?;
    required_str(&root, "model_revision", UPSTREAM_REVISION)?;
    required_str(&root, "source_repository", SOURCE_REPOSITORY)?;
    required_str(&root, "source_revision", SOURCE_REVISION)?;
    required_str(&root, "checkpoint_filename", CHECKPOINT_FILENAME)?;
    required_u64(&root, "checkpoint_size", CHECKPOINT_SIZE)?;
    required_str(&root, "checkpoint_sha256", CHECKPOINT_SHA256)?;
    let prepared_sha256 = required_sha256(&root, "prepared_sha256")?;
    let typed_manifest_sha256 = required_sha256(&root, "typed_manifest_sha256")?;
    if typed_manifest_sha256 != REVIEWED_TENSOR_MANIFEST_SHA256 {
        return Err(parse_error("sidecar typed manifest digest is not reviewed"));
    }
    if root.get("tensor_count").and_then(JsonValue::as_u64) != Some(TENSOR_COUNT as u64) {
        return Err(parse_error("sidecar tensor count mismatch"));
    }
    let values = root
        .get("tensor_rows")
        .and_then(JsonValue::as_array)
        .ok_or_else(|| parse_error("sidecar tensor_rows must be an array"))?;
    if values.len() != TENSOR_COUNT {
        return Err(parse_error("sidecar tensor_rows count mismatch"));
    }
    let rows = values
        .iter()
        .enumerate()
        .map(|(index, value)| parse_row(value, index))
        .collect::<Result<Vec<_>, _>>()?;
    if rows
        .windows(2)
        .any(|window| window[0].name >= window[1].name)
    {
        return Err(parse_error(
            "sidecar tensor_rows must be sorted by exact name",
        ));
    }
    let required_roles = rows.iter().map(|row| row.role.clone()).collect::<Vec<_>>();
    validate_typed_manifest(&rows, &required_roles).map_err(parse_error)?;
    if hex(&canonical_typed_manifest_sha256(&rows)) != typed_manifest_sha256 {
        return Err(parse_error(
            "sidecar typed manifest digest recomputation mismatch",
        ));
    }
    Ok(PreparedManifest {
        prepared_sha256,
        typed_manifest_sha256,
        rows,
    })
}

fn sidecar_path(input: &Path) -> PathBuf {
    let mut path = input.to_owned();
    path.set_extension("manifest.json");
    path
}

fn read_sidecar(input: &Path) -> Result<Vec<u8>, ConvertError> {
    let path = sidecar_path(input);
    let size = std::fs::metadata(&path)?.len();
    if size > MAX_SIDECAR_BYTES {
        return Err(parse_error(format!(
            "sidecar exceeds the {MAX_SIDECAR_BYTES}-byte limit"
        )));
    }
    Ok(std::fs::read(path)?)
}

fn validate_checkpoint(
    checkpoint: &SafetensorsFile,
    rows: &[SgmseManifestRow],
) -> Result<usize, ConvertError> {
    if checkpoint.tensors().len() != rows.len() {
        return Err(parse_error("prepared tensor count mismatch"));
    }
    let expected_names = rows
        .iter()
        .map(|row| row.name.as_str())
        .collect::<BTreeSet<_>>();
    let actual_names = checkpoint
        .tensors()
        .iter()
        .map(|tensor| tensor.name.as_str())
        .collect::<BTreeSet<_>>();
    if expected_names.len() != rows.len() || actual_names != expected_names {
        return Err(parse_error("prepared tensor name set mismatch"));
    }
    let mut bf16 = 0;
    for row in rows {
        let tensor = checkpoint
            .tensor_info(&row.name)
            .ok_or_else(|| parse_error(format!("prepared tensor {} is missing", row.name)))?;
        let expected_dtype =
            GgmlType::from_tag(row.dtype_tag).map_err(|error| parse_error(error.to_string()))?;
        if tensor.dtype != expected_dtype || tensor.shape != row.shape {
            return Err(parse_error(format!(
                "prepared tensor {} dtype/shape mismatch",
                row.name
            )));
        }
        if tensor.dtype == GgmlType::BF16 {
            bf16 += 1;
        }
    }
    Ok(bf16)
}

fn add_string_array(builder: &mut GgufBuilder, key: &str, values: Vec<String>) {
    builder.add_metadata(
        key,
        GgufMetadataValue::Array(GgufArray {
            element_type: GgufValueType::String,
            values: values.into_iter().map(GgufMetadataValue::String).collect(),
        }),
    );
}

/// Convert one authenticated prepared artifact.
pub fn convert_sgmse_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<SgmseReport, ConvertError> {
    if let Some(value) = license {
        if !value.eq_ignore_ascii_case("apache-2.0") {
            return Err(ConvertError::Usage(
                "SGMSE-VoiceBank weights are fixed Apache-2.0; license override must be apache-2.0"
                    .to_owned(),
            ));
        }
    }
    let Some(expected_prepared_sha256) = AUTHENTICATED_PREPARED_SHA256 else {
        return Err(ConvertError::Usage(
            "SGMSE-VoiceBank conversion is AUTHENTICATED_PREPARED_SHA256_REQUIRED: obtain and review the VAST prepared safetensors SHA-256 before conversion"
                .to_owned(),
        ));
    };
    let manifest = validate_prepared_sidecar(&read_sidecar(input)?)?;
    if manifest.prepared_sha256 != expected_prepared_sha256 {
        return Err(parse_error(
            "prepared SHA-256 does not match the reviewed digest",
        ));
    }
    let bytes = std::fs::read(input)?;
    let actual_prepared_sha256 = hex(&sha256(&bytes));
    if actual_prepared_sha256 != expected_prepared_sha256 {
        return Err(parse_error("prepared artifact SHA-256 mismatch"));
    }
    let checkpoint = SafetensorsFile::parse(bytes)?;
    let bf16_passthrough = validate_checkpoint(&checkpoint, &manifest.rows)?;

    let mut builder = GgufBuilder::new();
    builder
        .add_string(chunks::KEY_MODEL_ARCH, ARCH)
        .add_string(chunks::KEY_MODEL_NAME, NAME)
        .add_string(KEY_MANIFEST_STATUS, "AUTHENTICATED")
        .add_string(KEY_TENSOR_MANIFEST_SHA256, &manifest.typed_manifest_sha256)
        .add_string(KEY_SOURCE_REPOSITORY, SOURCE_REPOSITORY)
        .add_string(KEY_SOURCE_REVISION, SOURCE_REVISION)
        .add_string(KEY_MODEL_REVISION, UPSTREAM_REVISION)
        .add_string(KEY_CHECKPOINT_FILENAME, CHECKPOINT_FILENAME)
        .add_metadata(KEY_CHECKPOINT_SIZE, GgufMetadataValue::U64(CHECKPOINT_SIZE))
        .add_string(KEY_CHECKPOINT_SHA256, CHECKPOINT_SHA256)
        .add_string(KEY_PREPARED_SHA256, expected_prepared_sha256);
    add_string_array(
        &mut builder,
        KEY_TENSOR_MANIFEST,
        manifest
            .rows
            .iter()
            .map(|row| {
                format!(
                    "{}|{}|{}|{}",
                    row.role,
                    row.name,
                    row.dtype_tag,
                    row.dimensions
                        .iter()
                        .map(ToString::to_string)
                        .collect::<Vec<_>>()
                        .join(",")
                )
            })
            .collect(),
    );
    vokra_core::stamp_provenance(
        &mut builder,
        LicenseClass::Permissive,
        "Apache-2.0",
        Some(UPSTREAM_HF),
        Some("https://huggingface.co/speechbrain/sgmse-voicebank"),
    );
    for row in &manifest.rows {
        let tensor = checkpoint
            .tensor_info(&row.name)
            .expect("validated SGMSE tensor name set");
        builder.add_tensor(
            &row.name,
            tensor.dtype,
            row.dimensions.clone(),
            checkpoint.tensor_bytes(tensor).to_vec(),
        )?;
    }
    std::fs::write(output, builder.to_bytes()?)?;
    Ok(SgmseReport {
        read: TENSOR_COUNT,
        written: TENSOR_COUNT,
        skipped_non_float: 0,
        bf16_passthrough,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(name: &str, role: &str, dtype_tag: u32, shape: &[u64]) -> SgmseManifestRow {
        SgmseManifestRow {
            name: name.to_owned(),
            role: role.to_owned(),
            dtype_tag,
            shape: shape.to_vec(),
            dimensions: shape.iter().rev().copied().collect(),
        }
    }

    #[test]
    fn prepared_digest_is_independently_observed() {
        assert_eq!(
            AUTHENTICATED_PREPARED_SHA256,
            Some("eda3064488c670db78f7f93a77fade739e0d3941b7863e28db489a137f723045")
        );
    }

    #[test]
    fn missing_sidecar_is_rejected_without_output() {
        let stem = format!("sgmse-missing-sidecar-{}", std::process::id());
        let input = std::env::temp_dir().join(format!("{stem}.safetensors"));
        let output = std::env::temp_dir().join(format!("{stem}.gguf"));
        let _ = std::fs::remove_file(&input);
        let _ = std::fs::remove_file(&output);
        let error = convert_sgmse_file(&input, &output, None)
            .unwrap_err()
            .to_string();
        assert!(error.contains("I/O error"), "{error}");
        assert!(!output.exists());
    }

    #[test]
    fn license_override_cannot_relabel_weights() {
        let error = convert_sgmse_file(Path::new("missing"), Path::new("out"), Some("mit"))
            .unwrap_err()
            .to_string();
        assert!(error.contains("Apache-2.0"), "{error}");
        let error = convert_sgmse_file(Path::new("missing"), Path::new("out"), Some(""))
            .unwrap_err()
            .to_string();
        assert!(error.contains("Apache-2.0"), "{error}");
    }

    #[test]
    fn typed_manifest_rejects_drift_and_accepts_reverse_dimensions() {
        let required = vec!["fourier_frequencies".to_owned()];
        let valid = vec![row("source.frequencies", &required[0], 0, &[2, 3])];
        validate_typed_manifest(&valid, &required).unwrap();
        let mut drifted = valid.clone();
        drifted[0].dimensions = vec![2, 3];
        assert!(validate_typed_manifest(&drifted, &required).is_err());
        drifted = valid;
        drifted[0].role = "arbitrary_passthrough".to_owned();
        assert!(validate_typed_manifest(&drifted, &required).is_err());
    }

    #[test]
    fn canonical_digest_uses_non_symmetric_dimensions_and_exact_name() {
        let rows = vec![row(
            "source.nonsymmetric",
            "stage:1:residual:1:residual_conv1:weight",
            0,
            &[2, 3, 5],
        )];
        assert_eq!(
            hex(&canonical_typed_manifest_sha256(&rows)),
            "39671d48bc116445a52d6e573a9045ca5a5d080960a3993923d64c319a6c54ef"
        );
    }

    #[test]
    fn exact_schema_rejects_duplicate_and_unknown_keys() {
        let duplicate = json::parse(br#"{"a":1,"a":2}"#).unwrap();
        assert!(exact_object_schema(&duplicate, &["a"], "tiny").is_err());
        let unknown = json::parse(br#"{"a":1,"b":2}"#).unwrap();
        assert!(exact_object_schema(&unknown, &["a"], "tiny").is_err());
        let row_unknown = json::parse(
            br#"{"name":"x","role":"fourier_frequencies","dtype":"torch.float32","shape":[1],"dimensions":[1],"extra":true}"#,
        )
        .unwrap();
        assert!(parse_row(&row_unknown, 0).is_err());
        let row_duplicate = json::parse(
            br#"{"name":"x","role":"fourier_frequencies","dtype":"torch.float32","shape":[1],"dimensions":[1],"name":"y"}"#,
        )
        .unwrap();
        assert!(parse_row(&row_duplicate, 0).is_err());
    }

    #[test]
    fn row_parser_rejects_invalid_role_dtype_zero_and_nonreversed_dimensions() {
        let base = |role: &str, dtype: &str, shape: &str, dimensions: &str| {
            json::parse(
                format!(
                    "{{\"name\":\"x\",\"role\":\"{role}\",\"dtype\":\"{dtype}\",\"shape\":{shape},\"dimensions\":{dimensions}}}"
                )
                .as_bytes(),
            )
            .unwrap()
        };
        assert!(parse_row(&base("not-a-role", "torch.float32", "[1]", "[1]"), 0).is_err());
        assert!(parse_row(&base("fourier_frequencies", "torch.int64", "[1]", "[1]"), 0).is_err());
        assert!(
            parse_row(
                &base("fourier_frequencies", "torch.float32", "[0]", "[0]"),
                0
            )
            .is_err()
        );
        assert!(
            parse_row(
                &base("fourier_frequencies", "torch.float32", "[2,3]", "[2,3]"),
                0
            )
            .is_err()
        );
    }

    fn synthetic_safetensors(entries: &[(&str, &str, &[u64], usize)]) -> SafetensorsFile {
        let mut header = String::from("{");
        let mut offset = 0usize;
        for (index, (name, dtype, shape, bytes)) in entries.iter().enumerate() {
            if index != 0 {
                header.push(',');
            }
            let shape = shape
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(",");
            header.push_str(&format!(
                "\"{name}\":{{\"dtype\":\"{dtype}\",\"shape\":[{shape}],\"data_offsets\":[{offset},{}]}}",
                offset + bytes
            ));
            offset += bytes;
        }
        header.push('}');
        let mut data = Vec::with_capacity(8 + header.len() + offset);
        data.extend_from_slice(&(header.len() as u64).to_le_bytes());
        data.extend_from_slice(header.as_bytes());
        data.resize(data.len() + offset, 0);
        SafetensorsFile::parse(data).expect("synthetic safetensors")
    }

    #[test]
    fn checkpoint_validation_rejects_count_name_dtype_and_shape_drift() {
        let expected = row("source.nonsymmetric", "fourier_frequencies", 0, &[2, 3, 5]);
        let expected_rows = vec![expected.clone()];
        let duplicate = synthetic_safetensors(&[
            ("source.nonsymmetric", "F32", &[2, 3, 5], 120),
            ("source.nonsymmetric", "F32", &[2, 3, 5], 120),
        ]);
        assert!(validate_checkpoint(&duplicate, &expected_rows).is_err());
        let wrong_name = synthetic_safetensors(&[("other", "F32", &[2, 3, 5], 120)]);
        assert!(validate_checkpoint(&wrong_name, &expected_rows).is_err());
        let wrong_dtype = synthetic_safetensors(&[("source.nonsymmetric", "F16", &[2, 3, 5], 60)]);
        assert!(validate_checkpoint(&wrong_dtype, &expected_rows).is_err());
        let wrong_shape = synthetic_safetensors(&[("source.nonsymmetric", "F32", &[3, 10], 120)]);
        assert!(validate_checkpoint(&wrong_shape, &expected_rows).is_err());
    }
}
