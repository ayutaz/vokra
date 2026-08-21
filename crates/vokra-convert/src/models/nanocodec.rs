//! NVIDIA NeMo NanoCodec 22.05 kHz decoder-only checkpoint conversion.
//!
//! The source `.nemo` archive contains a torch-pickle checkpoint.  The
//! uv-managed `tools/parity/nanocodec/prepare_checkpoint.py` sidecar restores
//! that archive through the pinned official NeMo implementation and emits a
//! canonical F32 safetensors file plus checkpoint-derived JSON.  This module
//! consumes only those dependency-free interchange formats.

use std::collections::HashSet;

use vokra_core::LicenseClass;
use vokra_core::gguf::{
    GgmlType, GgufArray, GgufBuilder, GgufMetadataValue, GgufValueType, chunks,
};

use crate::ConvertError;
use crate::json::{self, JsonValue};
use crate::safetensors::SafetensorsFile;

pub(crate) const ARCH: &str = "nanocodec";
const NAME: &str = "NVIDIA NeMo NanoCodec 22.05 kHz";
const FORMAT_VERSION: u64 = 1;
const NVIDIA_OML: &str = "nvidia-open-model-license";
const NVIDIA_NOTICE: &str = "Licensed by NVIDIA Corporation under the NVIDIA Open Model License";
const PROFILE_06: &str = "nvidia/nemo-nano-codec-22khz-0.6kbps-12.5fps";
const PROFILE_178: &str = "nvidia/nemo-nano-codec-22khz-1.78kbps-12.5fps";
const PROFILE_189: &str = "nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps";
const REVISION_06: &str = "5c8e22ed763c14d81337fbe6ca74062f3d10f7e5";
const REVISION_178: &str = "c4ab84a92c8d36a8b5a79eaea807cfaf7f03ed86";
const REVISION_189: &str = "fc00890b604aa2de298d2641ffc6c5f6caf8c4d7";
const NEMO_SPEECH_COMMIT: &str = "4fcff72febec9395fdbd4bfa0747bfda2ecd3cef";
const NEMO_SOURCE_URL: &str = "https://github.com/NVIDIA-NeMo/Speech.git";

fn audited_revision(model_id: &str) -> Option<&'static str> {
    match model_id {
        PROFILE_06 => Some(REVISION_06),
        PROFILE_178 => Some(REVISION_178),
        PROFILE_189 => Some(REVISION_189),
        #[cfg(test)]
        "nvidia/nemo-nano-codec-22khz-test-fixture" => Some(REVISION_06),
        _ => None,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct NanoCodecConfig {
    pub(crate) source_model_id: String,
    pub(crate) source_revision: String,
    pub(crate) checkpoint_sha256: String,
    pub(crate) nemo_speech_commit: String,
    pub(crate) nemo_source_url: String,
    pub(crate) sample_rate: u32,
    pub(crate) frame_hop: u32,
    pub(crate) generator_hop: u32,
    pub(crate) n_codebooks: usize,
    pub(crate) levels_per_group: Vec<u32>,
    pub(crate) embed_dim: usize,
    pub(crate) base_channels: usize,
    pub(crate) upsample_rates: Vec<u32>,
    pub(crate) input_kernel_size: usize,
    pub(crate) output_kernel_size: usize,
    pub(crate) resblock_kernel_sizes: Vec<u32>,
    pub(crate) resblock_dilations: Vec<u32>,
}

impl NanoCodecConfig {
    pub(crate) fn parse(bytes: &[u8]) -> Result<Self, ConvertError> {
        let root = json::parse(bytes).map_err(|e| ConvertError::Parse(e.to_string()))?;
        let req_u64 = |key: &str| -> Result<u64, ConvertError> {
            root.get(key).and_then(JsonValue::as_u64).ok_or_else(|| {
                parse_error(format!(
                    "required non-negative integer field `{key}` missing"
                ))
            })
        };
        let req_usize = |key: &str| -> Result<usize, ConvertError> {
            usize::try_from(req_u64(key)?).map_err(|_| {
                parse_error(format!("field `{key}` does not fit this platform's usize"))
            })
        };
        let req_u32 = |key: &str| -> Result<u32, ConvertError> {
            u32::try_from(req_u64(key)?)
                .map_err(|_| parse_error(format!("field `{key}` exceeds u32")))
        };
        let req_string = |key: &str| -> Result<String, ConvertError> {
            let value = root
                .get(key)
                .and_then(JsonValue::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .ok_or_else(|| parse_error(format!("required non-empty string `{key}` missing")))?;
            Ok(value.to_owned())
        };
        let req_u32_array = |key: &str| -> Result<Vec<u32>, ConvertError> {
            let values = root
                .get(key)
                .and_then(JsonValue::as_array)
                .filter(|values| !values.is_empty())
                .ok_or_else(|| {
                    parse_error(format!("required non-empty integer array `{key}` missing"))
                })?;
            values
                .iter()
                .enumerate()
                .map(|(index, value)| {
                    let value = value.as_u64().ok_or_else(|| {
                        parse_error(format!("`{key}[{index}]` must be a non-negative integer"))
                    })?;
                    let value = u32::try_from(value)
                        .map_err(|_| parse_error(format!("`{key}[{index}]` exceeds u32")))?;
                    if value == 0 {
                        return Err(parse_error(format!("`{key}[{index}]` must be > 0")));
                    }
                    Ok(value)
                })
                .collect()
        };
        let req_bool = |key: &str| -> Result<bool, ConvertError> {
            match root.get(key) {
                Some(JsonValue::Bool(value)) => Ok(*value),
                _ => Err(parse_error(format!(
                    "required boolean field `{key}` missing"
                ))),
            }
        };

        let version = req_u64("format_version")?;
        if version != FORMAT_VERSION {
            return Err(parse_error(format!(
                "unsupported format_version {version}; expected {FORMAT_VERSION}"
            )));
        }
        let source_model_id = req_string("source_model_id")?;
        let expected_revision = audited_revision(&source_model_id).ok_or_else(|| {
            parse_error(format!(
                "source_model_id `{source_model_id}` is not one of the three audited NVIDIA 22 kHz NanoCodec repositories"
            ))
        })?;
        let source_revision = req_string("source_revision")?;
        if !is_lower_hex(&source_revision, 40) {
            return Err(parse_error(
                "source_revision must be a full 40-character lowercase hexadecimal commit"
                    .to_owned(),
            ));
        }
        if source_revision != expected_revision {
            return Err(parse_error(format!(
                "source_revision `{source_revision}` does not match audited revision `{expected_revision}` for `{source_model_id}`"
            )));
        }
        let checkpoint_sha256 = req_string("checkpoint_sha256")?;
        if !is_lower_hex(&checkpoint_sha256, 64) {
            return Err(parse_error(
                "checkpoint_sha256 must be 64 lowercase hexadecimal characters".to_owned(),
            ));
        }
        let nemo_speech_commit = req_string("nemo_speech_commit")?;
        if nemo_speech_commit != NEMO_SPEECH_COMMIT {
            return Err(parse_error(format!(
                "nemo_speech_commit `{nemo_speech_commit}` does not match pinned `{NEMO_SPEECH_COMMIT}`"
            )));
        }
        let nemo_source_url = req_string("nemo_source_url")?;
        if nemo_source_url != NEMO_SOURCE_URL {
            return Err(parse_error(format!(
                "nemo_source_url `{nemo_source_url}` is not the official pinned source `{NEMO_SOURCE_URL}`"
            )));
        }
        require_string(&root, "target_class", "CausalHiFiGANDecoder")?;
        require_string(&root, "activation", "HalfSnake")?;
        require_string(&root, "output_activation", "ClampActivation")?;
        require_string(&root, "pad_mode", "zeros")?;
        if !req_bool("grouped_upsample")? {
            return Err(parse_error(
                "grouped_upsample must be true for the checked NanoCodec transform".to_owned(),
            ));
        }

        let config = Self {
            source_model_id,
            source_revision,
            checkpoint_sha256,
            nemo_speech_commit,
            nemo_source_url,
            sample_rate: req_u32("sample_rate")?,
            frame_hop: req_u32("frame_hop")?,
            generator_hop: req_u32("generator_hop")?,
            n_codebooks: req_usize("n_codebooks")?,
            levels_per_group: req_u32_array("levels_per_group")?,
            embed_dim: req_usize("embed_dim")?,
            base_channels: req_usize("base_channels")?,
            upsample_rates: req_u32_array("upsample_rates")?,
            input_kernel_size: req_usize("input_kernel_size")?,
            output_kernel_size: req_usize("output_kernel_size")?,
            resblock_kernel_sizes: req_u32_array("resblock_kernel_sizes")?,
            resblock_dilations: req_u32_array("resblock_dilations")?,
        };
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<(), ConvertError> {
        if self.sample_rate == 0
            || self.frame_hop == 0
            || self.generator_hop == 0
            || self.n_codebooks == 0
            || self.embed_dim == 0
            || self.base_channels == 0
            || self.input_kernel_size == 0
            || self.output_kernel_size == 0
        {
            return Err(parse_error(
                "sample rate, frame hop, dimensions, and kernels must be > 0".to_owned(),
            ));
        }
        if self.levels_per_group.iter().any(|&level| level < 2) {
            return Err(parse_error(
                "levels_per_group entries must each be >= 2".to_owned(),
            ));
        }
        let expected_embed = self
            .n_codebooks
            .checked_mul(self.levels_per_group.len())
            .ok_or_else(|| parse_error("embed_dim product overflow".to_owned()))?;
        if self.embed_dim != expected_embed {
            return Err(parse_error(format!(
                "embed_dim {} != n_codebooks {} * levels_per_group length {}",
                self.embed_dim,
                self.n_codebooks,
                self.levels_per_group.len()
            )));
        }
        for (field, value) in [
            ("n_codebooks", self.n_codebooks),
            ("embed_dim", self.embed_dim),
            ("base_channels", self.base_channels),
            ("input_kernel_size", self.input_kernel_size),
            ("output_kernel_size", self.output_kernel_size),
        ] {
            if u32::try_from(value).is_err() {
                return Err(parse_error(format!("field `{field}` exceeds u32")));
            }
        }
        let mut channels = self.base_channels;
        let mut computed_generator_hop = 1_u32;
        for (stage, &rate) in self.upsample_rates.iter().enumerate() {
            if channels < 2 || channels % 2 != 0 {
                return Err(parse_error(format!(
                    "stage {stage} input channels {channels} cannot be halved"
                )));
            }
            let _ = usize::try_from(rate)
                .ok()
                .and_then(|rate| rate.checked_mul(2))
                .ok_or_else(|| parse_error(format!("stage {stage} upsample kernel overflow")))?;
            computed_generator_hop = computed_generator_hop
                .checked_mul(rate)
                .ok_or_else(|| parse_error("upsample-rate product exceeds u32".to_owned()))?;
            channels /= 2;
        }
        if self.generator_hop != computed_generator_hop {
            return Err(parse_error(format!(
                "generator_hop {} != upsample-rate product {}",
                self.generator_hop, computed_generator_hop
            )));
        }
        if channels == 0 {
            return Err(parse_error(
                "post-activation channel count must be > 0".to_owned(),
            ));
        }
        self.validate_audited_profile()?;
        Ok(())
    }

    fn validate_audited_profile(&self) -> Result<(), ConvertError> {
        #[cfg(test)]
        if self.source_model_id == "nvidia/nemo-nano-codec-22khz-test-fixture" {
            return Ok(());
        }

        let (n_codebooks, levels, embed_dim, frame_hop, upsample_rates) =
            match self.source_model_id.as_str() {
                PROFILE_06 => (4, &[9, 8, 8, 7][..], 16, 1764, &[7, 7, 6, 3, 2][..]),
                PROFILE_178 => (13, &[8, 7, 6, 6][..], 52, 1764, &[7, 7, 6, 3, 2][..]),
                PROFILE_189 => (8, &[8, 7, 6, 6][..], 32, 1024, &[8, 8, 4, 2, 2][..]),
                _ => {
                    return Err(parse_error(format!(
                        "source_model_id `{}` is not an audited profile",
                        self.source_model_id
                    )));
                }
            };
        if self.sample_rate != 22_050
            || self.base_channels != 864
            || self.n_codebooks != n_codebooks
            || self.levels_per_group != levels
            || self.embed_dim != embed_dim
            || self.frame_hop != frame_hop
            || self.upsample_rates != upsample_rates
        {
            return Err(parse_error(format!(
                "checkpoint geometry does not match audited profile `{}`: sample_rate={}, base_channels={}, n_codebooks={}, levels_per_group={:?}, embed_dim={}, frame_hop={}, upsample_rates={:?}",
                self.source_model_id,
                self.sample_rate,
                self.base_channels,
                self.n_codebooks,
                self.levels_per_group,
                self.embed_dim,
                self.frame_hop,
                self.upsample_rates
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Default)]
pub(crate) struct NanoCodecReport {
    pub(crate) written: usize,
}

pub(crate) fn convert(
    bytes: Vec<u8>,
    config: &NanoCodecConfig,
) -> Result<(GgufBuilder, NanoCodecReport), ConvertError> {
    config.validate()?;
    let st = SafetensorsFile::parse(bytes)?;
    let expected = expected_tensors(config)?;
    let expected_names = expected
        .iter()
        .map(|(name, _)| name.as_str())
        .collect::<HashSet<_>>();

    for tensor in st.tensors() {
        if !expected_names.contains(tensor.name.as_str()) {
            return Err(parse_error(format!(
                "unexpected prepared tensor `{}`; NanoCodec conversion is decoder-only and total",
                tensor.name
            )));
        }
    }
    for (name, shape) in &expected {
        let tensor = st
            .tensors()
            .iter()
            .find(|tensor| tensor.name == *name)
            .ok_or_else(|| parse_error(format!("required tensor `{name}` missing")))?;
        if tensor.dtype != GgmlType::F32 {
            return Err(parse_error(format!(
                "prepared tensor `{name}` must be F32, got {:?}",
                tensor.dtype
            )));
        }
        if tensor.shape != *shape {
            return Err(parse_error(format!(
                "prepared tensor `{name}` shape {:?} != checkpoint-derived {shape:?}",
                tensor.shape
            )));
        }
    }
    if st.tensors().len() != expected.len() {
        return Err(parse_error(format!(
            "prepared tensor count {} != decoder manifest {}",
            st.tensors().len(),
            expected.len()
        )));
    }

    let generator_hop = config.generator_hop;

    let mut builder = GgufBuilder::new();
    builder.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    builder.add_string(chunks::KEY_MODEL_NAME, NAME);
    builder.add_u32("vokra.nanocodec.n_codebooks", config.n_codebooks as u32);
    add_u32_array(
        &mut builder,
        "vokra.nanocodec.levels_per_group",
        &config.levels_per_group,
    );
    builder.add_u32("vokra.nanocodec.embed_dim", config.embed_dim as u32);
    builder.add_u32("vokra.nanocodec.sample_rate", config.sample_rate);
    builder.add_u32("vokra.nanocodec.frame_hop", config.frame_hop);
    builder.add_u32("vokra.nanocodec.generator_hop", generator_hop);
    builder.add_u32("vokra.nanocodec.base_channels", config.base_channels as u32);
    add_u32_array(
        &mut builder,
        "vokra.nanocodec.upsample_rates",
        &config.upsample_rates,
    );
    builder.add_u32(
        "vokra.nanocodec.input_kernel_size",
        config.input_kernel_size as u32,
    );
    builder.add_u32(
        "vokra.nanocodec.output_kernel_size",
        config.output_kernel_size as u32,
    );
    add_u32_array(
        &mut builder,
        "vokra.nanocodec.resblock_kernel_sizes",
        &config.resblock_kernel_sizes,
    );
    add_u32_array(
        &mut builder,
        "vokra.nanocodec.resblock_dilations",
        &config.resblock_dilations,
    );
    builder.add_string("vokra.nanocodec.activation", "HalfSnake");
    builder.add_string("vokra.nanocodec.output_activation", "ClampActivation");
    builder.add_string("vokra.nanocodec.pad_mode", "zeros");
    builder.add_bool("vokra.nanocodec.grouped_upsample_expanded", true);
    builder.add_string(
        "vokra.nanocodec.nemo_speech_commit",
        &config.nemo_speech_commit,
    );
    builder.add_string("vokra.nanocodec.nemo_source_url", &config.nemo_source_url);
    builder.add_string("vokra.provenance.upstream_hf", &config.source_model_id);
    builder.add_string(
        "vokra.provenance.upstream_revision",
        &config.source_revision,
    );
    builder.add_string(
        "vokra.provenance.checkpoint_sha256",
        &config.checkpoint_sha256,
    );
    let source = format!(
        "https://huggingface.co/{}/tree/{} (prepared from .nemo by the pinned official NeMo sidecar)",
        config.source_model_id, config.source_revision
    );
    vokra_core::stamp_provenance(
        &mut builder,
        LicenseClass::AttributionRequired,
        NVIDIA_OML,
        Some(&config.source_model_id),
        Some(&source),
    );
    vokra_core::stamp_attribution(&mut builder, NVIDIA_NOTICE);

    for tensor in st.tensors() {
        builder.add_tensor(
            &tensor.name,
            tensor.dtype,
            tensor.shape.clone(),
            st.tensor_bytes(tensor).to_vec(),
        )?;
    }

    Ok((
        builder,
        NanoCodecReport {
            written: st.tensors().len(),
        },
    ))
}

fn parse_error(message: String) -> ConvertError {
    ConvertError::Parse(format!("nanocodec config: {message}"))
}

fn require_string(root: &JsonValue, key: &str, expected: &str) -> Result<(), ConvertError> {
    let actual = root
        .get(key)
        .and_then(JsonValue::as_str)
        .ok_or_else(|| parse_error(format!("required string field `{key}` missing")))?;
    if actual != expected {
        return Err(parse_error(format!(
            "`{key}` must be `{expected}`, got `{actual}`"
        )));
    }
    Ok(())
}

fn is_lower_hex(value: &str, len: usize) -> bool {
    value.len() == len
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
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

fn expected_tensors(config: &NanoCodecConfig) -> Result<Vec<(String, Vec<u64>)>, ConvertError> {
    let shape = |dims: &[usize]| -> Result<Vec<u64>, ConvertError> {
        dims.iter()
            .map(|&dim| {
                u64::try_from(dim)
                    .map_err(|_| parse_error("tensor dimension does not fit u64".to_owned()))
            })
            .collect()
    };
    let mut expected = Vec::new();
    let mut push = |name: String, dims: &[usize]| -> Result<(), ConvertError> {
        expected.push((name, shape(dims)?));
        Ok(())
    };

    push(
        "nanocodec.pre_conv.weight".to_owned(),
        &[
            config.base_channels,
            config.embed_dim,
            config.input_kernel_size,
        ],
    )?;
    push(
        "nanocodec.pre_conv.bias".to_owned(),
        &[config.base_channels],
    )?;
    let mut in_channels = config.base_channels;
    for (stage, &rate) in config.upsample_rates.iter().enumerate() {
        let rate = usize::try_from(rate)
            .map_err(|_| parse_error(format!("stage {stage} rate does not fit usize")))?;
        let out_channels = in_channels / 2;
        let stage_prefix = format!("nanocodec.stage.{stage}");
        push(
            format!("{stage_prefix}.activation.alpha"),
            &[in_channels / 2],
        )?;
        push(
            format!("{stage_prefix}.activation.alpha_inv"),
            &[in_channels / 2],
        )?;
        push(
            format!("{stage_prefix}.upsample.weight"),
            &[in_channels, out_channels, 2 * rate],
        )?;
        push(format!("{stage_prefix}.upsample.bias"), &[out_channels])?;
        for (branch, &kernel) in config.resblock_kernel_sizes.iter().enumerate() {
            let kernel = usize::try_from(kernel).map_err(|_| {
                parse_error(format!(
                    "residual branch {branch} kernel does not fit usize"
                ))
            })?;
            for block in 0..config.resblock_dilations.len() {
                let block_prefix = format!("{stage_prefix}.branch.{branch}.block.{block}");
                for activation in ["input_activation", "skip_activation"] {
                    push(
                        format!("{block_prefix}.{activation}.alpha"),
                        &[out_channels / 2],
                    )?;
                    push(
                        format!("{block_prefix}.{activation}.alpha_inv"),
                        &[out_channels / 2],
                    )?;
                }
                for conv in ["input_conv", "skip_conv"] {
                    push(
                        format!("{block_prefix}.{conv}.weight"),
                        &[out_channels, out_channels, kernel],
                    )?;
                    push(format!("{block_prefix}.{conv}.bias"), &[out_channels])?;
                }
            }
        }
        in_channels = out_channels;
    }
    push(
        "nanocodec.post_activation.alpha".to_owned(),
        &[in_channels / 2],
    )?;
    push(
        "nanocodec.post_activation.alpha_inv".to_owned(),
        &[in_channels / 2],
    )?;
    push(
        "nanocodec.post_conv.weight".to_owned(),
        &[1, in_channels, config.output_kernel_size],
    )?;
    push("nanocodec.post_conv.bias".to_owned(), &[1])?;
    Ok(expected)
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::{GgufFile, GgufMetadataValue};

    const SHA256: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    fn config_json(
        model_id: &str,
        n_codebooks: usize,
        levels: &[u32],
        embed_dim: usize,
        frame_hop: u32,
        upsample_rates: &[u32],
    ) -> Vec<u8> {
        let generator_hop = upsample_rates.iter().product::<u32>();
        let levels = levels
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(",");
        let upsample_rates = upsample_rates
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(",");
        let base_channels = if upsample_rates.matches(',').count() >= 4 {
            864
        } else {
            8
        };
        let revision = audited_revision(model_id).unwrap_or(REVISION_06);
        format!(
            r#"{{
              "format_version":1,
              "source_model_id":"{model_id}",
              "source_revision":"{revision}",
              "checkpoint_sha256":"{SHA256}",
              "nemo_speech_commit":"4fcff72febec9395fdbd4bfa0747bfda2ecd3cef",
              "nemo_source_url":"https://github.com/NVIDIA-NeMo/Speech.git",
              "target_class":"CausalHiFiGANDecoder",
              "sample_rate":22050,
              "frame_hop":{frame_hop},
              "generator_hop":{generator_hop},
              "n_codebooks":{n_codebooks},
              "levels_per_group":[{levels}],
              "embed_dim":{embed_dim},
              "base_channels":{base_channels},
              "upsample_rates":[{upsample_rates}],
              "input_kernel_size":3,
              "output_kernel_size":3,
              "resblock_kernel_sizes":[3,5],
              "resblock_dilations":[1,2],
              "activation":"HalfSnake",
              "output_activation":"ClampActivation",
              "pad_mode":"zeros",
              "grouped_upsample":true
            }}"#
        )
        .into_bytes()
    }

    fn tiny_config() -> NanoCodecConfig {
        NanoCodecConfig::parse(&config_json(
            "nvidia/nemo-nano-codec-22khz-test-fixture",
            2,
            &[3, 2],
            4,
            4,
            &[2, 2],
        ))
        .expect("tiny config")
    }

    fn build_safetensors(entries: &[(String, Vec<usize>, Vec<f32>)]) -> Vec<u8> {
        let mut header = String::from("{");
        let mut data = Vec::<u8>::new();
        for (i, (name, shape, values)) in entries.iter().enumerate() {
            let start = data.len();
            for value in values {
                data.extend_from_slice(&value.to_le_bytes());
            }
            let end = data.len();
            if i > 0 {
                header.push(',');
            }
            let shape = shape
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(",");
            header.push_str(&format!(
                r#""{name}":{{"dtype":"F32","shape":[{shape}],"data_offsets":[{start},{end}]}}"#
            ));
        }
        header.push('}');
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(&data);
        out
    }

    fn tensor(
        name: impl Into<String>,
        shape: &[usize],
        seed: &mut f32,
    ) -> (String, Vec<usize>, Vec<f32>) {
        let len = shape.iter().product();
        let values = (0..len)
            .map(|_| {
                *seed += 0.01;
                *seed
            })
            .collect();
        (name.into(), shape.to_vec(), values)
    }

    fn synthetic_decoder(config: &NanoCodecConfig) -> Vec<u8> {
        let mut seed = 0.0;
        let mut entries = vec![
            tensor(
                "nanocodec.pre_conv.weight",
                &[
                    config.base_channels,
                    config.embed_dim,
                    config.input_kernel_size,
                ],
                &mut seed,
            ),
            tensor(
                "nanocodec.pre_conv.bias",
                &[config.base_channels],
                &mut seed,
            ),
        ];
        let mut in_channels = config.base_channels;
        for (stage, &rate) in config.upsample_rates.iter().enumerate() {
            let out_channels = in_channels / 2;
            entries.push(tensor(
                format!("nanocodec.stage.{stage}.activation.alpha"),
                &[in_channels / 2],
                &mut seed,
            ));
            entries.push(tensor(
                format!("nanocodec.stage.{stage}.activation.alpha_inv"),
                &[in_channels / 2],
                &mut seed,
            ));
            entries.push(tensor(
                format!("nanocodec.stage.{stage}.upsample.weight"),
                &[in_channels, out_channels, 2 * rate as usize],
                &mut seed,
            ));
            entries.push(tensor(
                format!("nanocodec.stage.{stage}.upsample.bias"),
                &[out_channels],
                &mut seed,
            ));
            for (branch, &kernel) in config.resblock_kernel_sizes.iter().enumerate() {
                for block in 0..config.resblock_dilations.len() {
                    let prefix = format!("nanocodec.stage.{stage}.branch.{branch}.block.{block}");
                    for activation in ["input_activation", "skip_activation"] {
                        entries.push(tensor(
                            format!("{prefix}.{activation}.alpha"),
                            &[out_channels / 2],
                            &mut seed,
                        ));
                        entries.push(tensor(
                            format!("{prefix}.{activation}.alpha_inv"),
                            &[out_channels / 2],
                            &mut seed,
                        ));
                    }
                    for conv in ["input_conv", "skip_conv"] {
                        entries.push(tensor(
                            format!("{prefix}.{conv}.weight"),
                            &[out_channels, out_channels, kernel as usize],
                            &mut seed,
                        ));
                        entries.push(tensor(
                            format!("{prefix}.{conv}.bias"),
                            &[out_channels],
                            &mut seed,
                        ));
                    }
                }
            }
            in_channels = out_channels;
        }
        entries.push(tensor(
            "nanocodec.post_activation.alpha",
            &[in_channels / 2],
            &mut seed,
        ));
        entries.push(tensor(
            "nanocodec.post_activation.alpha_inv",
            &[in_channels / 2],
            &mut seed,
        ));
        entries.push(tensor(
            "nanocodec.post_conv.weight",
            &[1, in_channels, config.output_kernel_size],
            &mut seed,
        ));
        entries.push(tensor("nanocodec.post_conv.bias", &[1], &mut seed));
        build_safetensors(&entries)
    }

    #[test]
    fn parses_checkpoint_derived_axes_for_every_published_profile() {
        let profiles = [
            (
                PROFILE_06,
                4,
                vec![9, 8, 8, 7],
                16,
                1764,
                vec![7, 7, 6, 3, 2],
            ),
            (
                PROFILE_178,
                13,
                vec![8, 7, 6, 6],
                52,
                1764,
                vec![7, 7, 6, 3, 2],
            ),
            (
                PROFILE_189,
                8,
                vec![8, 7, 6, 6],
                32,
                1024,
                vec![8, 8, 4, 2, 2],
            ),
        ];
        for (model_id, n_codebooks, levels, embed_dim, frame_hop, rates) in profiles {
            let cfg = NanoCodecConfig::parse(&config_json(
                model_id,
                n_codebooks,
                &levels,
                embed_dim,
                frame_hop,
                &rates,
            ))
            .unwrap_or_else(|e| panic!("{model_id}: {e}"));
            assert_eq!(cfg.n_codebooks, n_codebooks);
            assert_eq!(cfg.levels_per_group, levels);
            assert_eq!(cfg.embed_dim, embed_dim);
            assert_eq!(cfg.frame_hop, frame_hop);
            assert_eq!(cfg.upsample_rates, rates);
        }
    }

    #[test]
    fn config_rejects_missing_provenance_and_non_checkpoint_topology() {
        let mut missing_sha = config_json(PROFILE_06, 2, &[3, 2], 4, 4, &[2, 2]);
        missing_sha = String::from_utf8(missing_sha)
            .unwrap()
            .replace(SHA256, "")
            .into_bytes();
        assert!(NanoCodecConfig::parse(&missing_sha).is_err());

        let wrong_class = String::from_utf8(config_json(PROFILE_06, 2, &[3, 2], 4, 4, &[2, 2]))
            .unwrap()
            .replace("CausalHiFiGANDecoder", "HiFiGANDecoder")
            .into_bytes();
        assert!(NanoCodecConfig::parse(&wrong_class).is_err());

        let unpublished = config_json(
            "nvidia/nemo-nano-codec-22khz-0.8kbps-12.5fps",
            4,
            &[9, 8, 8, 7],
            16,
            1764,
            &[7, 7, 6, 3, 2],
        );
        assert!(
            NanoCodecConfig::parse(&unpublished).is_err(),
            "the issue-listed but unpublished 0.8 kbps repository must stay fail-closed",
        );

        let mislabeled_profile =
            config_json(PROFILE_06, 13, &[8, 7, 6, 6], 52, 1764, &[7, 7, 6, 3, 2]);
        let error = NanoCodecConfig::parse(&mislabeled_profile).unwrap_err();
        assert!(error.to_string().contains("audited profile"), "{error}");

        let wrong_nemo_source = String::from_utf8(config_json(
            PROFILE_06,
            4,
            &[9, 8, 8, 7],
            16,
            1764,
            &[7, 7, 6, 3, 2],
        ))
        .unwrap()
        .replace(
            "4fcff72febec9395fdbd4bfa0747bfda2ecd3cef",
            "0000000000000000000000000000000000000000",
        )
        .into_bytes();
        let error = NanoCodecConfig::parse(&wrong_nemo_source).unwrap_err();
        assert!(error.to_string().contains("nemo_speech_commit"), "{error}");

        let wrong_generator_hop = String::from_utf8(config_json(
            PROFILE_189,
            8,
            &[8, 7, 6, 6],
            32,
            1024,
            &[8, 8, 4, 2, 2],
        ))
        .unwrap()
        .replace("\"generator_hop\":1024", "\"generator_hop\":2048")
        .into_bytes();
        let error = NanoCodecConfig::parse(&wrong_generator_hop).unwrap_err();
        assert!(error.to_string().contains("generator_hop"), "{error}");
    }

    #[test]
    fn conversion_is_total_and_emits_decoder_contract() {
        let cfg = tiny_config();
        let (builder, report) = convert(synthetic_decoder(&cfg), &cfg).expect("convert");
        let file = GgufFile::parse(builder.to_bytes().expect("serialize")).expect("GGUF");
        assert_eq!(report.written, file.tensors().len());
        assert!(matches!(
            file.get("vokra.nanocodec.n_codebooks"),
            Some(GgufMetadataValue::U32(2))
        ));
        assert!(matches!(
            file.get("vokra.nanocodec.frame_hop"),
            Some(GgufMetadataValue::U32(4))
        ));
        assert!(matches!(
            file.get("vokra.nanocodec.nemo_speech_commit"),
            Some(GgufMetadataValue::String(value))
                if value == "4fcff72febec9395fdbd4bfa0747bfda2ecd3cef"
        ));
        assert!(matches!(
            file.get("vokra.provenance.attribution"),
            Some(GgufMetadataValue::String(text))
                if text == "Licensed by NVIDIA Corporation under the NVIDIA Open Model License"
        ));
        assert!(file.tensor_info("nanocodec.pre_conv.weight").is_some());
        assert!(
            file.tensor_info("nanocodec.stage.1.upsample.weight")
                .is_some()
        );
        assert!(file.tensor_info("nanocodec.post_conv.weight").is_some());
    }

    #[test]
    fn conversion_rejects_extra_or_misshapen_prepared_tensors() {
        let cfg = tiny_config();
        // The synthetic builder cannot append after serialization, so build a
        // single unknown tensor to pin total-consumption failure.
        let extra = build_safetensors(&[tensor("encoder.forbidden", &[1], &mut 0.0)]);
        let err = convert(extra, &cfg).unwrap_err();
        assert!(
            err.to_string().contains("required tensor") || err.to_string().contains("unexpected")
        );

        let wrong = build_safetensors(&[tensor("nanocodec.pre_conv.weight", &[1, 1, 1], &mut 0.0)]);
        assert!(convert(wrong, &cfg).is_err());
    }
}
