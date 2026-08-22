//! Strict conversion of the pinned FunASR FSMN-VAD release.
//!
//! Input must be produced by `tools/parity/fsmn_vad_prepare_checkpoint.py`.
//! That offline bridge verifies the official `.pt`, `am.mvn`, and
//! `config.yaml` hashes and combines 24 encoder weights with two reserved CMVN
//! vectors.  This converter validates that complete manifest, moves CMVN into
//! GGUF metadata, and writes only the 24 runtime weights.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{
    GgmlType, GgufArray, GgufBuilder, GgufMetadataValue, GgufValueType, chunks,
};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

pub const ARCH: &str = "fsmn-vad";
pub const NAME: &str = "fsmn-vad-zh-cn-16k-common";
pub const CATEGORY: &str = "vad";
pub const UPSTREAM_HF: &str = "funasr/fsmn-vad";
pub const UPSTREAM_MODELSCOPE: &str = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch";
pub const UPSTREAM_REVISION: &str = "df20e6b30c653645fa4ff125cacfcabd1020a669";
pub const MODEL_SHA256: &str = "b3be75be477f0780277f3bae0fe489f48718f585f3a6e45d7dd1fbb1a4255fc5";
pub const CMVN_SHA256: &str = "df189fd5f4352df84a0fd464eeab4e450a5e645665d6b38f13c832492261a739";
pub const CONFIG_SHA256: &str = "486861ca26ddb79081663b6179cb204c6bfae71c52f04aafc48a9e9d8dde1e93";
pub const DEFAULT_LICENSE: &str = "apache-2.0";

const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";
const KEY_PROVENANCE_UPSTREAM_MODELSCOPE: &str = "vokra.provenance.upstream_modelscope";
const KEY_PROVENANCE_UPSTREAM_REVISION: &str = "vokra.provenance.upstream_revision";
const KEY_CHECKPOINT_SHA256: &str = "vokra.fsmn_vad.checkpoint_sha256";
const KEY_CMVN_SHA256: &str = "vokra.fsmn_vad.cmvn_sha256";
const KEY_CONFIG_SHA256: &str = "vokra.fsmn_vad.config_sha256";

const KEY_N_BLOCKS: &str = "vokra.fsmn_vad.n_blocks";
const KEY_INPUT_DIM: &str = "vokra.fsmn_vad.input_dim";
const KEY_INPUT_AFFINE_DIM: &str = "vokra.fsmn_vad.input_affine_dim";
const KEY_LINEAR_DIM: &str = "vokra.fsmn_vad.linear_dim";
const KEY_PROJ_DIM: &str = "vokra.fsmn_vad.proj_dim";
const KEY_LORDER: &str = "vokra.fsmn_vad.lorder";
const KEY_RORDER: &str = "vokra.fsmn_vad.rorder";
const KEY_LSTRIDE: &str = "vokra.fsmn_vad.lstride";
const KEY_RSTRIDE: &str = "vokra.fsmn_vad.rstride";
const KEY_OUTPUT_AFFINE_DIM: &str = "vokra.fsmn_vad.output_affine_dim";
const KEY_OUTPUT_DIM: &str = "vokra.fsmn_vad.output_dim";
const KEY_N_MELS: &str = "vokra.fsmn_vad.n_mels";
const KEY_LFR_M: &str = "vokra.fsmn_vad.lfr_m";
const KEY_LFR_N: &str = "vokra.fsmn_vad.lfr_n";
const KEY_SAMPLE_RATE: &str = "vokra.fsmn_vad.sample_rate";
const KEY_CMVN_ADD_SHIFT: &str = "vokra.fsmn_vad.cmvn_add_shift";
const KEY_CMVN_RESCALE: &str = "vokra.fsmn_vad.cmvn_rescale";

const PREPARED_CMVN_ADD_SHIFT: &str = "__vokra__.fsmn_vad.cmvn_add_shift";
const PREPARED_CMVN_RESCALE: &str = "__vokra__.fsmn_vad.cmvn_rescale";

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
/// Counts from one strict FSMN-VAD conversion.
pub struct FsmnVadReport {
    /// Tensors read, including the two reserved CMVN vectors.
    pub read: usize,
    /// Runtime weight tensors written to GGUF.
    pub written: usize,
    /// Always zero for the strict canonical manifest.
    pub skipped_non_float: usize,
    /// Always zero because the pinned release is F32.
    pub bf16_passthrough: usize,
}

fn expected_weights() -> BTreeMap<String, Vec<u64>> {
    let mut expected = BTreeMap::from([
        (
            "encoder.in_linear1.linear.weight".to_owned(),
            vec![140, 400],
        ),
        ("encoder.in_linear1.linear.bias".to_owned(), vec![140]),
        (
            "encoder.in_linear2.linear.weight".to_owned(),
            vec![250, 140],
        ),
        ("encoder.in_linear2.linear.bias".to_owned(), vec![250]),
        (
            "encoder.out_linear1.linear.weight".to_owned(),
            vec![140, 250],
        ),
        ("encoder.out_linear1.linear.bias".to_owned(), vec![140]),
        (
            "encoder.out_linear2.linear.weight".to_owned(),
            vec![248, 140],
        ),
        ("encoder.out_linear2.linear.bias".to_owned(), vec![248]),
    ]);
    for index in 0..4 {
        let prefix = format!("encoder.fsmn.{index}");
        expected.insert(format!("{prefix}.linear.linear.weight"), vec![128, 250]);
        expected.insert(
            format!("{prefix}.fsmn_block.conv_left.weight"),
            vec![128, 1, 20, 1],
        );
        expected.insert(format!("{prefix}.affine.linear.weight"), vec![250, 128]);
        expected.insert(format!("{prefix}.affine.linear.bias"), vec![250]);
    }
    debug_assert_eq!(expected.len(), 24);
    expected
}

fn expected_tensors() -> BTreeMap<String, Vec<u64>> {
    let mut expected = expected_weights();
    expected.insert(PREPARED_CMVN_ADD_SHIFT.to_owned(), vec![400]);
    expected.insert(PREPARED_CMVN_RESCALE.to_owned(), vec![400]);
    expected
}

fn f32_array(values: &[f32]) -> GgufMetadataValue {
    GgufMetadataValue::Array(GgufArray {
        element_type: GgufValueType::F32,
        values: values.iter().copied().map(GgufMetadataValue::F32).collect(),
    })
}

fn require_prepare_metadata(bytes: &[u8]) -> Result<(), ConvertError> {
    if bytes.len() < 8 {
        return Err(ConvertError::Parse(
            "fsmn-vad safetensors is truncated".to_owned(),
        ));
    }
    let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header = bytes
        .get(8..8usize.saturating_add(header_len))
        .ok_or_else(|| {
            ConvertError::Parse("fsmn-vad safetensors header is truncated".to_owned())
        })?;
    let root =
        vokra_core::json::parse(header).map_err(|error| ConvertError::Parse(error.to_string()))?;
    let metadata = root.get("__metadata__").ok_or_else(|| {
        ConvertError::Parse(
            "fsmn-vad input lacks prepare-script metadata; regenerate with tools/parity/fsmn_vad_prepare_checkpoint.py"
                .to_owned(),
        )
    })?;
    for (key, expected) in [
        ("vokra.source_revision", UPSTREAM_REVISION),
        ("vokra.model_sha256", MODEL_SHA256),
        ("vokra.cmvn_sha256", CMVN_SHA256),
        ("vokra.config_sha256", CONFIG_SHA256),
    ] {
        let actual = metadata.get(key).and_then(|value| value.as_str());
        if actual != Some(expected) {
            return Err(ConvertError::Parse(format!(
                "fsmn-vad prepared metadata `{key}` is {actual:?}, expected `{expected}`"
            )));
        }
    }
    Ok(())
}

/// Converts a prepare-script-verified FSMN-VAD safetensors bundle to GGUF.
pub fn convert_fsmn_vad_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<FsmnVadReport, ConvertError> {
    let bytes = std::fs::read(input)?;
    require_prepare_metadata(&bytes)?;
    let safetensors = SafetensorsFile::parse(bytes)?;
    let expected = expected_tensors();
    let actual = safetensors
        .tensors()
        .iter()
        .map(|tensor| tensor.name.clone())
        .collect::<BTreeSet<_>>();
    let expected_names = expected.keys().cloned().collect::<BTreeSet<_>>();
    if actual != expected_names {
        let missing = expected_names
            .difference(&actual)
            .cloned()
            .collect::<Vec<_>>();
        let extra = actual
            .difference(&expected_names)
            .cloned()
            .collect::<Vec<_>>();
        return Err(ConvertError::Parse(format!(
            "fsmn-vad canonical manifest mismatch: missing={missing:?}, extra={extra:?}; regenerate with tools/parity/fsmn_vad_prepare_checkpoint.py"
        )));
    }
    for tensor in safetensors.tensors() {
        if tensor.dtype != GgmlType::F32 {
            return Err(ConvertError::Parse(format!(
                "fsmn-vad tensor `{}` is {:?}, expected official F32",
                tensor.name, tensor.dtype
            )));
        }
        if tensor.shape != expected[&tensor.name] {
            return Err(ConvertError::Parse(format!(
                "fsmn-vad tensor `{}` has shape {:?}, expected {:?}",
                tensor.name, tensor.shape, expected[&tensor.name]
            )));
        }
    }
    let add_shift = safetensors.tensor_f32(PREPARED_CMVN_ADD_SHIFT)?;
    let rescale = safetensors.tensor_f32(PREPARED_CMVN_RESCALE)?;
    if add_shift.iter().any(|value| !value.is_finite())
        || rescale
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err(ConvertError::Parse(
            "fsmn-vad CMVN AddShift/Rescale must be finite and Rescale positive".to_owned(),
        ));
    }

    let mut builder = GgufBuilder::new();
    builder.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    builder.add_string(chunks::KEY_MODEL_NAME, NAME);
    builder.add_string(KEY_MODEL_CATEGORY, CATEGORY);
    builder.add_string(KEY_PROVENANCE_UPSTREAM_HF, UPSTREAM_HF);
    builder.add_string(KEY_PROVENANCE_UPSTREAM_MODELSCOPE, UPSTREAM_MODELSCOPE);
    builder.add_string(KEY_PROVENANCE_UPSTREAM_REVISION, UPSTREAM_REVISION);
    builder.add_string(KEY_CHECKPOINT_SHA256, MODEL_SHA256);
    builder.add_string(KEY_CMVN_SHA256, CMVN_SHA256);
    builder.add_string(KEY_CONFIG_SHA256, CONFIG_SHA256);
    let (spdx, class) = match license {
        Some(value) if !value.is_empty() => {
            (value.to_owned(), LicenseClass::from_license_str(value))
        }
        _ => (DEFAULT_LICENSE.to_owned(), LicenseClass::Permissive),
    };
    vokra_core::stamp_provenance(
        &mut builder,
        class,
        &spdx,
        Some(NAME),
        Some("funasr/fsmn-vad weights (Apache-2.0)"),
    );
    for (key, value) in [
        (KEY_N_BLOCKS, 4),
        (KEY_INPUT_DIM, 400),
        (KEY_INPUT_AFFINE_DIM, 140),
        (KEY_LINEAR_DIM, 250),
        (KEY_PROJ_DIM, 128),
        (KEY_LORDER, 20),
        (KEY_RORDER, 0),
        (KEY_LSTRIDE, 1),
        (KEY_RSTRIDE, 0),
        (KEY_OUTPUT_AFFINE_DIM, 140),
        (KEY_OUTPUT_DIM, 248),
        (KEY_N_MELS, 80),
        (KEY_LFR_M, 5),
        (KEY_LFR_N, 1),
        (KEY_SAMPLE_RATE, 16_000),
    ] {
        builder.add_u32(key, value);
    }
    builder.add_metadata(KEY_CMVN_ADD_SHIFT, f32_array(&add_shift));
    builder.add_metadata(KEY_CMVN_RESCALE, f32_array(&rescale));

    let weights = expected_weights();
    let mut report = FsmnVadReport {
        read: safetensors.tensors().len(),
        ..FsmnVadReport::default()
    };
    for tensor in safetensors.tensors() {
        if !weights.contains_key(&tensor.name) {
            continue;
        }
        builder
            .add_tensor(
                &tensor.name,
                tensor.dtype,
                tensor.shape.clone(),
                safetensors.tensor_bytes(tensor).to_vec(),
            )
            .map_err(|error| ConvertError::Gguf(error.to_string()))?;
        report.written += 1;
    }
    let output_bytes = builder
        .to_bytes()
        .map_err(|error| ConvertError::Gguf(error.to_string()))?;
    std::fs::write(output, output_bytes)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::GgufFile;

    fn scratch(label: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "vokra-fsmn-vad-{label}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .subsec_nanos()
        ))
    }

    fn prepared(drop: Option<&str>) -> Vec<u8> {
        let mut data = Vec::new();
        let mut tensor_entries = Vec::new();
        for (name, shape) in expected_tensors() {
            if drop == Some(name.as_str()) {
                continue;
            }
            let start = data.len();
            let elements = shape.iter().product::<u64>() as usize;
            let fill: f32 = if name == PREPARED_CMVN_RESCALE {
                1.0
            } else {
                0.0
            };
            data.extend((0..elements).flat_map(|_| fill.to_le_bytes()));
            let shape = shape
                .iter()
                .map(u64::to_string)
                .collect::<Vec<_>>()
                .join(",");
            tensor_entries.push(format!(
                "\"{name}\":{{\"dtype\":\"F32\",\"shape\":[{shape}],\"data_offsets\":[{start},{}]}}",
                data.len()
            ));
        }
        let metadata = format!(
            "\"__metadata__\":{{\"format\":\"pt\",\"vokra.source_revision\":\"{UPSTREAM_REVISION}\",\"vokra.model_sha256\":\"{MODEL_SHA256}\",\"vokra.cmvn_sha256\":\"{CMVN_SHA256}\",\"vokra.config_sha256\":\"{CONFIG_SHA256}\"}}"
        );
        let header = format!("{{{metadata},{}}}", tensor_entries.join(","));
        let mut output = (header.len() as u64).to_le_bytes().to_vec();
        output.extend_from_slice(header.as_bytes());
        output.extend_from_slice(&data);
        output
    }

    #[test]
    fn canonical_bundle_writes_24_weights_and_real_cmvn() {
        let input = scratch("input");
        let output = scratch("output");
        std::fs::write(&input, prepared(None)).unwrap();
        let report = convert_fsmn_vad_file(&input, &output, None).unwrap();
        assert_eq!((report.read, report.written), (26, 24));
        let file = GgufFile::open(&output).unwrap();
        assert_eq!(
            file.get(KEY_PROVENANCE_UPSTREAM_REVISION)
                .and_then(|value| value.as_str()),
            Some(UPSTREAM_REVISION)
        );
        assert_eq!(file.tensors().len(), 24);
        assert!(file.tensor_info(PREPARED_CMVN_ADD_SHIFT).is_none());
        assert_eq!(
            file.get(KEY_CMVN_RESCALE)
                .and_then(|value| value.as_array())
                .unwrap()
                .values
                .len(),
            400
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_LICENSE)
                .and_then(|value| value.as_str()),
            Some(DEFAULT_LICENSE)
        );
        std::fs::remove_file(input).ok();
        std::fs::remove_file(output).ok();
    }

    #[test]
    fn missing_weight_fails_closed() {
        let input = scratch("missing-input");
        let output = scratch("missing-output");
        std::fs::write(&input, prepared(Some("encoder.out_linear2.linear.bias"))).unwrap();
        let error = convert_fsmn_vad_file(&input, &output, None)
            .unwrap_err()
            .to_string();
        assert!(error.contains("encoder.out_linear2.linear.bias"), "{error}");
        std::fs::remove_file(input).ok();
    }

    #[test]
    fn license_override_is_classified() {
        let input = scratch("license-input");
        let output = scratch("license-output");
        std::fs::write(&input, prepared(None)).unwrap();
        convert_fsmn_vad_file(&input, &output, Some("cc-by-4.0")).unwrap();
        let file = GgufFile::open(&output).unwrap();
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|value| value.as_str()),
            Some(LicenseClass::AttributionRequired.as_str())
        );
        std::fs::remove_file(input).ok();
        std::fs::remove_file(output).ok();
    }
}
