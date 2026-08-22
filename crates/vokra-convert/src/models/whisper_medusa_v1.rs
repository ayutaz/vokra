//! `aiola/whisper-medusa-v1`: pinned Whisper-large-v2 + Medusa conversion.
//!
//! The official checkpoint is not a base/small Apache release.  It is an MIT
//! licensed, 6.25 GB F32 checkpoint whose ordinary Whisper tensors live below
//! `whisper_model.` and whose `base_head` configuration creates eleven
//! residual heads (`medusa_num_heads + 1`).  Conversion therefore reuses the
//! audited Whisper writer while removing exactly that outer wrapper and keeps
//! every `medusa_heads.*` tensor under its original name.
//!
//! The source is sharded.  `tools/parity/whisper_medusa_prepare_checkpoint.py`
//! downloads the exact revision and creates the single safetensors file this
//! converter consumes; large-artifact work belongs on VAST.

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufArray, GgufMetadataValue, GgufValueType, chunks};
use vokra_core::json::{self, JsonValue};

use crate::ConvertError;
use crate::safetensors::SafetensorsFileReader;

use super::whisper::{self, WhisperVariant};

pub const ARCH: &str = "whisper-medusa-v1";
pub const NAME: &str = "whisper-medusa-v1";
pub const CATEGORY: &str = "asr";
pub const UPSTREAM_HF: &str = "aiola/whisper-medusa-v1";
pub const DEFAULT_LICENSE: &str = "MIT";

pub const UPSTREAM_REVISION: &str = "6ea7c2f47658cfc7f9c8d1c158a9fbdb33458462";
pub const UPSTREAM_SOURCE_REVISION: &str = "19819c37ab15db6e68826e406614a2c86fbb946e";
pub const CONFIG_SHA256: &str = "16346762b14c116eeda12b48f20e2281b327a11b516f8b004ce065fcb1450186";
pub const CHECKPOINT_SHA256: &str =
    "ec634d5ece33a8d634ed2e188c7bfbde7adab4410932b8fa6c20440836a423f3";

const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
const KEY_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";
pub const KEY_REVISION: &str = "vokra.medusa.revision";
pub const KEY_SOURCE_REVISION: &str = "vokra.medusa.source_revision";
pub const KEY_CONFIG_SHA256: &str = "vokra.medusa.config_sha256";
pub const KEY_CHECKPOINT_SHA256: &str = "vokra.medusa.checkpoint_sha256";
pub const KEY_NUM_HEADS: &str = "vokra.medusa.num_heads";
pub const KEY_MODULE_COUNT: &str = "vokra.medusa.module_count";
pub const KEY_NUM_LAYERS: &str = "vokra.medusa.num_layers";
pub const KEY_HIDDEN_SIZE: &str = "vokra.medusa.hidden_size";
pub const KEY_HEADS_TYPE: &str = "vokra.medusa.heads_type";
pub const KEY_CHOICES: &str = "vokra.medusa.choices";
pub const KEY_INIT_FROM_PROJ: &str = "vokra.medusa.init_from_proj";
pub const KEY_OUTPUT_WHISPER_ORIGINAL: &str = "vokra.medusa.output_whisper_original";

const EXPECTED_NUM_HEADS: u32 = 10;
const EXPECTED_MODULE_COUNT: u32 = 11;
const EXPECTED_NUM_LAYERS: u32 = 1;
const EXPECTED_HIDDEN_SIZE: u32 = 1280;
const EXPECTED_HEADS_TYPE: &str = "base_head";
const EXPECTED_CHOICES: [u32; 11] = [1; 11];

#[derive(Debug, Clone, PartialEq, Eq)]
struct MedusaConfig {
    num_heads: u32,
    num_layers: u32,
    hidden_size: u32,
    heads_type: String,
    choices: Vec<u32>,
    init_from_proj: bool,
    output_whisper_original: bool,
    whisper_model_name: String,
}

impl MedusaConfig {
    fn parse(bytes: &[u8]) -> Result<Self, ConvertError> {
        let root = json::parse(bytes).map_err(|error| ConvertError::Parse(error.to_string()))?;
        let required_u32 = |key: &str| {
            root.get(key)
                .and_then(JsonValue::as_u64)
                .and_then(|value| u32::try_from(value).ok())
                .ok_or_else(|| {
                    ConvertError::Parse(format!("whisper-medusa config: `{key}` must be a u32"))
                })
        };
        let required_str = |key: &str| {
            root.get(key)
                .and_then(JsonValue::as_str)
                .map(str::to_owned)
                .ok_or_else(|| {
                    ConvertError::Parse(format!("whisper-medusa config: `{key}` must be a string"))
                })
        };
        let choices = root
            .get("medusa_choices")
            .and_then(JsonValue::as_array)
            .ok_or_else(|| {
                ConvertError::Parse(
                    "whisper-medusa config: `medusa_choices` must be an array".into(),
                )
            })?
            .iter()
            .enumerate()
            .map(|(index, value)| {
                value
                    .as_u64()
                    .and_then(|value| u32::try_from(value).ok())
                    .ok_or_else(|| {
                        ConvertError::Parse(format!(
                            "whisper-medusa config: medusa_choices[{index}] must be a u32"
                        ))
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let required_bool = |key: &str| match root.get(key) {
            Some(JsonValue::Bool(value)) => Ok(*value),
            _ => Err(ConvertError::Parse(format!(
                "whisper-medusa config: `{key}` must be a bool"
            ))),
        };

        Ok(Self {
            num_heads: required_u32("medusa_num_heads")?,
            num_layers: required_u32("medusa_num_layers")?,
            hidden_size: required_u32("medusa_hidden_size")?,
            heads_type: required_str("medusa_heads_type")?,
            choices,
            init_from_proj: required_bool("init_from_proj")?,
            // The official class default is false and the released config
            // omits this key.  An explicit non-bool is still rejected.
            output_whisper_original: match root.get("output_whisper_original") {
                None => false,
                Some(JsonValue::Bool(value)) => *value,
                Some(_) => {
                    return Err(ConvertError::Parse(
                        "whisper-medusa config: `output_whisper_original` must be a bool".into(),
                    ));
                }
            },
            whisper_model_name: required_str("whisper_model_name")?,
        })
    }

    fn validate_official(&self) -> Result<(), ConvertError> {
        let expected = Self {
            num_heads: EXPECTED_NUM_HEADS,
            num_layers: EXPECTED_NUM_LAYERS,
            hidden_size: EXPECTED_HIDDEN_SIZE,
            heads_type: EXPECTED_HEADS_TYPE.into(),
            choices: EXPECTED_CHOICES.to_vec(),
            init_from_proj: true,
            output_whisper_original: false,
            whisper_model_name: "openai/whisper-large-v2".into(),
        };
        if self != &expected {
            return Err(ConvertError::Parse(format!(
                "whisper-medusa config does not match the pinned v1 contract: expected {expected:?}, got {self:?}"
            )));
        }
        Ok(())
    }

    fn write_into(&self, builder: &mut vokra_core::gguf::GgufBuilder) {
        builder.add_string(KEY_REVISION, UPSTREAM_REVISION);
        builder.add_string(KEY_SOURCE_REVISION, UPSTREAM_SOURCE_REVISION);
        builder.add_string(KEY_CONFIG_SHA256, CONFIG_SHA256);
        builder.add_string(KEY_CHECKPOINT_SHA256, CHECKPOINT_SHA256);
        builder.add_u32(KEY_NUM_HEADS, self.num_heads);
        builder.add_u32(KEY_MODULE_COUNT, EXPECTED_MODULE_COUNT);
        builder.add_u32(KEY_NUM_LAYERS, self.num_layers);
        builder.add_u32(KEY_HIDDEN_SIZE, self.hidden_size);
        builder.add_string(KEY_HEADS_TYPE, &self.heads_type);
        builder.add_metadata(
            KEY_CHOICES,
            GgufMetadataValue::Array(GgufArray {
                element_type: GgufValueType::U32,
                values: self
                    .choices
                    .iter()
                    .copied()
                    .map(GgufMetadataValue::U32)
                    .collect(),
            }),
        );
        builder.add_bool(KEY_INIT_FROM_PROJ, self.init_from_proj);
        builder.add_bool(KEY_OUTPUT_WHISPER_ORIGINAL, self.output_whisper_original);
    }
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
/// Tensor accounting from one canonical conversion.
pub struct WhisperMedusaV1Report {
    /// Tensor descriptors observed in the merged safetensors header.
    pub read: usize,
    /// Floating tensors emitted by the shared Whisper writer.
    pub written: usize,
    /// Non-floating tensors skipped (zero for the official checkpoint).
    pub skipped_non_float: usize,
    /// BF16 tensors included in `written` (zero for the official F32 v1).
    pub bf16_passthrough: usize,
}

/// Converts the single-file checkpoint produced by the pinned preparation
/// script.  `config` must be the exact upstream `config.json`; topology is not
/// inferred from prose or caller defaults.
pub fn convert_whisper_medusa_v1_file(
    input: &Path,
    output: &Path,
    config: &Path,
    license: Option<&str>,
) -> Result<WhisperMedusaV1Report, ConvertError> {
    let config_bytes = std::fs::read(config).map_err(ConvertError::Io)?;
    let medusa = MedusaConfig::parse(&config_bytes)?;
    medusa.validate_official()?;

    // Header-only pass for a truthful report.  The shared Whisper converter
    // then consumes the file once; no second 6.25 GB in-memory parse is used.
    let reader = SafetensorsFileReader::open(input)?;
    let mut report = WhisperMedusaV1Report::default();
    for tensor in reader.tensors() {
        report.read += 1;
        match tensor.dtype {
            GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 => {
                report.written += 1;
                if tensor.dtype == GgmlType::BF16 {
                    report.bf16_passthrough += 1;
                }
            }
            _ => report.skipped_non_float += 1,
        }
    }

    let bytes = std::fs::read(input).map_err(ConvertError::Io)?;
    let mut builder = whisper::convert_variant(bytes, WhisperVariant::WhisperMedusaV1)?;
    builder.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    builder.add_string(chunks::KEY_MODEL_NAME, NAME);
    builder.add_string(KEY_MODEL_CATEGORY, CATEGORY);
    builder.add_string(KEY_UPSTREAM_HF, UPSTREAM_HF);
    medusa.write_into(&mut builder);

    let spdx = license
        .filter(|value| !value.is_empty())
        .unwrap_or(DEFAULT_LICENSE);
    let class = LicenseClass::from_license_str(spdx);
    vokra_core::stamp_provenance(
        &mut builder,
        class,
        spdx,
        Some(ARCH),
        Some("aiola/whisper-medusa-v1 (MIT), exact HF and source revisions pinned"),
    );

    let output_bytes = builder
        .to_bytes()
        .map_err(|error| ConvertError::Gguf(error.to_string()))?;
    std::fs::write(output, output_bytes).map_err(ConvertError::Io)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    const OFFICIAL_CONFIG: &[u8] = br#"{
        "whisper_model_name":"openai/whisper-large-v2",
        "medusa_num_heads":10,
        "medusa_num_layers":1,
        "medusa_hidden_size":1280,
        "medusa_heads_type":"base_head",
        "medusa_choices":[1,1,1,1,1,1,1,1,1,1,1],
        "init_from_proj":true
    }"#;

    #[test]
    fn official_config_contract_is_exact() {
        let config = MedusaConfig::parse(OFFICIAL_CONFIG).unwrap();
        config.validate_official().unwrap();
        assert!(!config.output_whisper_original);
        assert_eq!(config.num_heads + 1, EXPECTED_MODULE_COUNT);
    }

    #[test]
    fn mismatched_head_count_is_rejected() {
        let raw = String::from_utf8(OFFICIAL_CONFIG.to_vec())
            .unwrap()
            .replace("\"medusa_num_heads\":10", "\"medusa_num_heads\":4");
        let error = MedusaConfig::parse(raw.as_bytes())
            .unwrap()
            .validate_official()
            .unwrap_err();
        assert!(error.to_string().contains("pinned v1 contract"));
    }

    #[test]
    fn missing_init_from_proj_is_rejected() {
        let raw = String::from_utf8(OFFICIAL_CONFIG.to_vec())
            .unwrap()
            .replace("\"init_from_proj\":true", "\"unused\":true");
        assert!(MedusaConfig::parse(raw.as_bytes()).is_err());
    }
}
