//! FireRedVAD official Stream-VAD safetensors → GGUF conversion.
//!
//! The accepted input is the canonical 39-tensor bundle produced by
//! `tools/parity/firered_vad_prepare_checkpoint.py` from
//! `FireRedTeam/FireRedVAD` commit
//! `c30ec49e8cc69642b0ee65362eba11b9d11c6e54`.  The converter validates the
//! complete tensor-name/shape manifest and stamps the exact Kaldi-fbank,
//! CMVN, causal DFSMN, cache, and sigmoid-head contract consumed by
//! `vokra-models::firered_vad`.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{
    GgmlType, GgufArray, GgufBuilder, GgufMetadataValue, GgufValueType, chunks,
};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

pub const ARCH: &str = "firered_vad";
pub const NAME: &str = "firered-vad-stream-v1";
pub const CATEGORY: &str = "vad";
pub const UPSTREAM_HF: &str = "FireRedTeam/FireRedVAD";
pub const UPSTREAM_URL: &str = "github.com/FireRedTeam/FireRedVAD";
pub const UPSTREAM_REVISION: &str = "c30ec49e8cc69642b0ee65362eba11b9d11c6e54";
pub const DEFAULT_LICENSE_SPDX: &str = "apache-2.0";
pub const VARIANT: &str = "stream-vad-dfsmn-v1";

const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";
const KEY_PROVENANCE_UPSTREAM_URL: &str = "vokra.provenance.upstream_url";
const KEY_PROVENANCE_UPSTREAM_REVISION: &str = "vokra.provenance.upstream_revision";
pub const KEY_VARIANT: &str = "vokra.firered_vad.variant";
pub const KEY_SAMPLE_RATE: &str = "vokra.firered_vad.sample_rate";
pub const KEY_N_MELS: &str = "vokra.firered_vad.n_mels";
pub const KEY_WINDOW_LENGTH: &str = "vokra.firered_vad.window_length";
pub const KEY_HOP_LENGTH: &str = "vokra.firered_vad.hop_length";
pub const KEY_N_BLOCKS: &str = "vokra.firered_vad.n_blocks";
pub const KEY_HIDDEN_DIM: &str = "vokra.firered_vad.hidden_dim";
pub const KEY_PROJECTION_DIM: &str = "vokra.firered_vad.projection_dim";
pub const KEY_MEMORY_ORDER: &str = "vokra.firered_vad.memory_order";
pub const KEY_MEMORY_STRIDE: &str = "vokra.firered_vad.memory_stride";
pub const KEY_N_CLASS: &str = "vokra.firered_vad.n_class";
pub const KEY_REQUIRED_TENSORS: &str = "vokra.firered_vad.required_tensors";

const SAMPLE_RATE: u32 = 16_000;
const N_MELS: u32 = 80;
const WINDOW_LENGTH: u32 = 400;
const HOP_LENGTH: u32 = 160;
const N_BLOCKS: u32 = 8;
const HIDDEN_DIM: u32 = 256;
const PROJECTION_DIM: u32 = 128;
const MEMORY_ORDER: u32 = 20;
const MEMORY_STRIDE: u32 = 1;
const N_CLASS: u32 = 1;

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct FireredVadReport {
    pub read: usize,
    pub written: usize,
    /// Retained for the shared CLI report contract; canonical input is F32.
    pub bf16_passthrough: usize,
    /// Retained for the shared CLI report contract; strict manifests skip nothing.
    pub skipped_non_float: usize,
}

fn expected_tensors() -> BTreeMap<String, Vec<u64>> {
    let mut expected = BTreeMap::from([
        ("firered_vad.cmvn.mean".to_owned(), vec![80]),
        ("firered_vad.cmvn.inverse_std".to_owned(), vec![80]),
        ("firered_vad.dfsmn.fc1.weight".to_owned(), vec![80, 256]),
        ("firered_vad.dfsmn.fc1.bias".to_owned(), vec![256]),
        ("firered_vad.dfsmn.fc2.weight".to_owned(), vec![256, 128]),
        ("firered_vad.dfsmn.fc2.bias".to_owned(), vec![128]),
        ("firered_vad.dfsmn.dnn.0.weight".to_owned(), vec![128, 256]),
        ("firered_vad.dfsmn.dnn.0.bias".to_owned(), vec![256]),
        ("firered_vad.output.weight".to_owned(), vec![256, 1]),
        ("firered_vad.output.bias".to_owned(), vec![1]),
    ]);
    for index in 0..N_BLOCKS as usize {
        expected.insert(
            format!("firered_vad.dfsmn.memory.{index}.weight"),
            vec![128, 1, 20],
        );
    }
    for index in 0..(N_BLOCKS as usize - 1) {
        expected.insert(
            format!("firered_vad.dfsmn.block.{index}.fc1.weight"),
            vec![128, 256],
        );
        expected.insert(
            format!("firered_vad.dfsmn.block.{index}.fc1.bias"),
            vec![256],
        );
        expected.insert(
            format!("firered_vad.dfsmn.block.{index}.fc2.weight"),
            vec![256, 128],
        );
    }
    debug_assert_eq!(expected.len(), 39);
    expected
}

fn string_array(values: impl IntoIterator<Item = String>) -> GgufMetadataValue {
    GgufMetadataValue::Array(GgufArray {
        element_type: GgufValueType::String,
        values: values.into_iter().map(GgufMetadataValue::String).collect(),
    })
}

pub fn convert_firered_vad_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<FireredVadReport, ConvertError> {
    let bytes = std::fs::read(input)?;
    let st = SafetensorsFile::parse(bytes)?;
    let expected = expected_tensors();
    let actual = st
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
        return Err(ConvertError::Gguf(format!(
            "firered-vad: canonical tensor manifest mismatch: missing={missing:?}, extra={extra:?}; regenerate with tools/parity/firered_vad_prepare_checkpoint.py"
        )));
    }

    for tensor in st.tensors() {
        let shape = &expected[&tensor.name];
        if tensor.dtype != GgmlType::F32 {
            return Err(ConvertError::Gguf(format!(
                "firered-vad: tensor `{}` is {:?}, expected canonical F32",
                tensor.name, tensor.dtype
            )));
        }
        if &tensor.shape != shape {
            return Err(ConvertError::Gguf(format!(
                "firered-vad: tensor `{}` has shape {:?}, expected {:?}",
                tensor.name, tensor.shape, shape
            )));
        }
    }

    let mut builder = GgufBuilder::new();
    builder.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    builder.add_string(chunks::KEY_MODEL_NAME, NAME);
    builder.add_string(KEY_MODEL_CATEGORY, CATEGORY);
    builder.add_string(KEY_PROVENANCE_UPSTREAM_HF, UPSTREAM_HF);
    builder.add_string(KEY_PROVENANCE_UPSTREAM_URL, UPSTREAM_URL);
    builder.add_string(KEY_PROVENANCE_UPSTREAM_REVISION, UPSTREAM_REVISION);
    let (spdx, class) = match license {
        Some(value) if !value.is_empty() => {
            (value.to_owned(), LicenseClass::from_license_str(value))
        }
        _ => (DEFAULT_LICENSE_SPDX.to_owned(), LicenseClass::Permissive),
    };
    vokra_core::stamp_provenance(
        &mut builder,
        class,
        &spdx,
        Some(NAME),
        Some("FireRedTeam/FireRedVAD Stream-VAD DFSMN (Apache-2.0)"),
    );
    builder.add_string(KEY_VARIANT, VARIANT);
    builder.add_u32(KEY_SAMPLE_RATE, SAMPLE_RATE);
    builder.add_u32(KEY_N_MELS, N_MELS);
    builder.add_u32(KEY_WINDOW_LENGTH, WINDOW_LENGTH);
    builder.add_u32(KEY_HOP_LENGTH, HOP_LENGTH);
    builder.add_u32(KEY_N_BLOCKS, N_BLOCKS);
    builder.add_u32(KEY_HIDDEN_DIM, HIDDEN_DIM);
    builder.add_u32(KEY_PROJECTION_DIM, PROJECTION_DIM);
    builder.add_u32(KEY_MEMORY_ORDER, MEMORY_ORDER);
    builder.add_u32(KEY_MEMORY_STRIDE, MEMORY_STRIDE);
    builder.add_u32(KEY_N_CLASS, N_CLASS);
    builder.add_metadata(KEY_REQUIRED_TENSORS, string_array(expected.keys().cloned()));

    let mut report = FireredVadReport::default();
    for tensor in st.tensors() {
        report.read += 1;
        builder
            .add_tensor(
                &tensor.name,
                tensor.dtype,
                tensor.shape.clone(),
                st.tensor_bytes(tensor).to_vec(),
            )
            .map_err(|error| ConvertError::Gguf(error.to_string()))?;
        report.written += 1;
    }
    let bytes = builder
        .to_bytes()
        .map_err(|error| ConvertError::Gguf(error.to_string()))?;
    std::fs::write(output, bytes)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::GgufFile;

    fn scratch(tag: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "vokra-firered-vad-{tag}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .subsec_nanos()
        ))
    }

    fn canonical_safetensors(drop_name: Option<&str>) -> Vec<u8> {
        let mut header = BTreeMap::new();
        let mut data = Vec::new();
        for (name, shape) in expected_tensors() {
            if drop_name == Some(name.as_str()) {
                continue;
            }
            let start = data.len();
            let elements = shape.iter().product::<u64>() as usize;
            data.resize(start + elements * 4, 0);
            header.insert(
                name,
                format!(
                    "{{\"dtype\":\"F32\",\"shape\":[{}],\"data_offsets\":[{start},{}]}}",
                    shape
                        .iter()
                        .map(u64::to_string)
                        .collect::<Vec<_>>()
                        .join(","),
                    data.len()
                ),
            );
        }
        let json = format!(
            "{{{}}}",
            header
                .into_iter()
                .map(|(name, value)| format!("{name:?}:{value}"))
                .collect::<Vec<_>>()
                .join(",")
        );
        let mut output = Vec::with_capacity(8 + json.len() + data.len());
        output.extend_from_slice(&(json.len() as u64).to_le_bytes());
        output.extend_from_slice(json.as_bytes());
        output.extend_from_slice(&data);
        output
    }

    #[test]
    fn canonical_bundle_stamps_native_contract() {
        let input = scratch("input.safetensors");
        let output = scratch("output.gguf");
        std::fs::write(&input, canonical_safetensors(None)).unwrap();
        let report = convert_firered_vad_file(&input, &output, None).unwrap();
        assert_eq!(
            report,
            FireredVadReport {
                read: 39,
                written: 39,
                bf16_passthrough: 0,
                skipped_non_float: 0,
            }
        );
        let gguf = GgufFile::open(&output).unwrap();
        assert_eq!(
            gguf.get(chunks::KEY_MODEL_ARCH)
                .and_then(|value| value.as_str()),
            Some(ARCH)
        );
        assert_eq!(
            gguf.get(KEY_VARIANT).and_then(|value| value.as_str()),
            Some(VARIANT)
        );
        assert_eq!(
            gguf.get(KEY_N_BLOCKS).and_then(|value| value.as_u64()),
            Some(8)
        );
        assert_eq!(gguf.tensors().len(), 39);
        std::fs::remove_file(input).ok();
        std::fs::remove_file(output).ok();
    }

    #[test]
    fn missing_canonical_tensor_is_refused() {
        let input = scratch("missing.safetensors");
        let output = scratch("missing.gguf");
        std::fs::write(
            &input,
            canonical_safetensors(Some("firered_vad.output.bias")),
        )
        .unwrap();
        let error = convert_firered_vad_file(&input, &output, None).unwrap_err();
        assert!(error.to_string().contains("firered_vad.output.bias"));
        std::fs::remove_file(input).ok();
        std::fs::remove_file(output).ok();
    }
}
