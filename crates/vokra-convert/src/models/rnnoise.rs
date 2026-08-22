//! Xiph RNNoise v0.2 canonical-array safetensors → GGUF converter.
//!
//! The official v0.2 release (published 2024-04-15) embeds its trained
//! network in `src/rnnoise_data.c`.  The offline prep tool parses those
//! arrays into a fixed 36-tensor safetensors manifest.  This converter is
//! deliberately strict: an opaque blob, missing array, unexpected array,
//! wrong dtype, or wrong element count is rejected before a GGUF is written.
//!
//! The default reference topology is 65 features → Conv1d(195,128) →
//! Conv1d(384,384) → 3×GRU(384) → 32 gain bands plus one VAD probability.
//! Quantized int8 matrices and sparse int indices are carried as exactly-valued
//! F32 tensors.  This avoids a private GGUF dtype while retaining Xiph's
//! layer-by-layer signed-int8 rounding semantics; the runtime binder converts
//! them back only after range and integrality checks.

use std::collections::BTreeMap;
use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

pub const ARCH: &str = "rnnoise";
pub const NAME: &str = "rnnoise-v0.2";
pub const CATEGORY: &str = "denoise";
pub const UPSTREAM_URL: &str = "https://github.com/xiph/rnnoise/releases/tag/v0.2";
pub const RELEASE_TARBALL_SHA256: &str =
    "90fce4b00b9ff24c08dbfe31b82ffd43bae383d85c5535676d28b0a2b11c0d37";
pub const DEFAULT_LICENSE: &str = "bsd-3-clause";

const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
const KEY_PROVENANCE_UPSTREAM_URL: &str = "vokra.provenance.upstream_url";
const KEY_RELEASE_SHA256: &str = "vokra.rnnoise.release_tarball_sha256";
const KEY_SAMPLE_RATE: &str = "vokra.rnnoise.sample_rate";
const KEY_FRAME_SIZE: &str = "vokra.rnnoise.frame_size";
const KEY_WINDOW_SIZE: &str = "vokra.rnnoise.window_size";
const KEY_N_BANDS: &str = "vokra.rnnoise.n_bands";
const KEY_N_FEATURES: &str = "vokra.rnnoise.n_features";
const KEY_CONV1_WIDTH: &str = "vokra.rnnoise.conv1_width";
const KEY_HIDDEN_SIZE: &str = "vokra.rnnoise.hidden_size";
const KEY_N_GRU: &str = "vokra.rnnoise.n_gru";
const KEY_QUANTIZATION: &str = "vokra.rnnoise.quantization";
const KEY_GATE_ORDER: &str = "vokra.rnnoise.gate_order";

pub const SAMPLE_RATE: u32 = 48_000;
pub const FRAME_SIZE: u32 = 480;
pub const WINDOW_SIZE: u32 = 960;
pub const N_BANDS: u32 = 32;
pub const N_FEATURES: u32 = 65;
pub const CONV1_WIDTH: u32 = 128;
pub const HIDDEN_SIZE: u32 = 384;
pub const N_GRU: u32 = 3;

/// Returns the exact canonical tensor manifest `(name, element_count)`.
pub fn tensor_manifest() -> Vec<(String, usize)> {
    let mut tensors = vec![
        ("conv1_weights_float".to_owned(), 24_960),
        ("conv1_bias".to_owned(), 128),
        ("conv2_weights_int8".to_owned(), 147_456),
        ("conv2_scale".to_owned(), 384),
        ("conv2_bias".to_owned(), 384),
    ];
    for layer in 1..=3 {
        for part in ["input", "recurrent"] {
            let prefix = format!("gru{layer}_{part}");
            tensors.push((format!("{prefix}_weights_int8"), 147_456));
            tensors.push((format!("{prefix}_weights_idx"), 4_752));
            tensors.push((format!("{prefix}_scale"), 1_152));
            tensors.push((format!("{prefix}_bias"), 1_152));
            if part == "recurrent" {
                tensors.push((format!("{prefix}_weights_diag"), 1_152));
            }
        }
    }
    tensors.extend([
        ("dense_out_weights_float".to_owned(), 12_288),
        ("dense_out_bias".to_owned(), 32),
        ("vad_dense_weights_float".to_owned(), 384),
        ("vad_dense_bias".to_owned(), 1),
    ]);
    debug_assert_eq!(tensors.len(), 36);
    tensors
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
/// Counters returned after a canonical RNNoise conversion.
pub struct RnnoiseReport {
    /// Canonical tensors read from the safetensors input.
    pub read: usize,
    /// Canonical tensors written to the GGUF output.
    pub written: usize,
    /// Always zero for a successful canonical conversion.  Retained for CLI
    /// summary compatibility with earlier converter releases.
    pub skipped_non_float: usize,
    /// Always zero: canonical arrays are F32, including exact integer
    /// containers.  Retained for CLI summary compatibility.
    pub bf16_passthrough: usize,
}

/// Validates a canonical prepared checkpoint and writes a self-describing GGUF.
pub fn convert_rnnoise_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<RnnoiseReport, ConvertError> {
    let bytes = std::fs::read(input).map_err(ConvertError::Io)?;
    let st = SafetensorsFile::parse(bytes)?;
    let expected: BTreeMap<String, usize> = tensor_manifest().into_iter().collect();

    if st.tensors().len() != expected.len() {
        return Err(ConvertError::Parse(format!(
            "rnnoise v0.2 requires exactly {} canonical arrays, found {}",
            expected.len(),
            st.tensors().len()
        )));
    }
    for tensor in st.tensors() {
        let Some(&element_count) = expected.get(&tensor.name) else {
            return Err(ConvertError::Parse(format!(
                "rnnoise v0.2 unexpected tensor `{}` (opaque blobs and aliases are refused)",
                tensor.name
            )));
        };
        if tensor.dtype != GgmlType::F32 {
            return Err(ConvertError::Parse(format!(
                "rnnoise v0.2 tensor `{}` has {:?}, expected F32 canonical container",
                tensor.name, tensor.dtype
            )));
        }
        if tensor.element_count() != element_count as u64 {
            return Err(ConvertError::Parse(format!(
                "rnnoise v0.2 tensor `{}` has {} elements, expected {element_count}",
                tensor.name,
                tensor.element_count()
            )));
        }
    }
    for name in expected.keys() {
        if st.tensor_info(name).is_none() {
            return Err(ConvertError::Parse(format!(
                "rnnoise v0.2 missing canonical tensor `{name}`"
            )));
        }
    }

    let mut builder = GgufBuilder::new();
    builder.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    builder.add_string(chunks::KEY_MODEL_NAME, NAME);
    builder.add_string(KEY_MODEL_CATEGORY, CATEGORY);
    builder.add_string(KEY_PROVENANCE_UPSTREAM_URL, UPSTREAM_URL);
    builder.add_string(KEY_RELEASE_SHA256, RELEASE_TARBALL_SHA256);
    builder.add_u32(KEY_SAMPLE_RATE, SAMPLE_RATE);
    builder.add_u32(KEY_FRAME_SIZE, FRAME_SIZE);
    builder.add_u32(KEY_WINDOW_SIZE, WINDOW_SIZE);
    builder.add_u32(KEY_N_BANDS, N_BANDS);
    builder.add_u32(KEY_N_FEATURES, N_FEATURES);
    builder.add_u32(KEY_CONV1_WIDTH, CONV1_WIDTH);
    builder.add_u32(KEY_HIDDEN_SIZE, HIDDEN_SIZE);
    builder.add_u32(KEY_N_GRU, N_GRU);
    builder.add_string(KEY_QUANTIZATION, "signed-i8-round127-f32-container");
    builder.add_string(KEY_GATE_ORDER, "zrh");
    let effective_license = license.unwrap_or(DEFAULT_LICENSE);
    let license_class = LicenseClass::from_license_str(effective_license);
    vokra_core::stamp_provenance(
        &mut builder,
        license_class,
        effective_license,
        Some(NAME),
        Some(UPSTREAM_URL),
    );

    for (name, _) in tensor_manifest() {
        let tensor = st.tensor_info(&name).expect("manifest checked above");
        builder.add_tensor(
            &name,
            GgmlType::F32,
            tensor.shape.clone(),
            st.tensor_bytes(tensor).to_vec(),
        )?;
    }
    let output_bytes = builder
        .to_bytes()
        .map_err(|error| ConvertError::Gguf(error.to_string()))?;
    std::fs::write(output, output_bytes).map_err(ConvertError::Io)?;
    Ok(RnnoiseReport {
        read: expected.len(),
        written: expected.len(),
        skipped_non_float: 0,
        bf16_passthrough: 0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::GgufFile;

    fn scratch(tag: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "vokra-rnnoise-{tag}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }

    fn canonical_safetensors() -> Vec<u8> {
        let manifest = tensor_manifest();
        let mut offset = 0usize;
        let mut entries = Vec::new();
        for (name, count) in &manifest {
            let end = offset + count * 4;
            entries.push(format!(
                "\"{name}\":{{\"dtype\":\"F32\",\"shape\":[{count}],\"data_offsets\":[{offset},{end}]}}"
            ));
            offset = end;
        }
        let header = format!("{{{}}}", entries.join(","));
        let mut bytes = Vec::with_capacity(8 + header.len() + offset);
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        bytes.resize(8 + header.len() + offset, 0);
        bytes
    }

    #[test]
    fn canonical_manifest_converts_and_stamps_real_topology() {
        let input = scratch("in.safetensors");
        let output = scratch("out.gguf");
        std::fs::write(&input, canonical_safetensors()).unwrap();
        let report = convert_rnnoise_file(&input, &output, None).unwrap();
        assert_eq!(report.written, 36);
        let gguf = GgufFile::open(&output).unwrap();
        assert_eq!(
            gguf.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH)
        );
        assert_eq!(gguf.get(KEY_N_BANDS).and_then(|v| v.as_u64()), Some(32));
        assert_eq!(gguf.get(KEY_N_FEATURES).and_then(|v| v.as_u64()), Some(65));
        assert_eq!(
            gguf.get(KEY_HIDDEN_SIZE).and_then(|v| v.as_u64()),
            Some(384)
        );
        assert!(gguf.tensor_info("gru3_recurrent_weights_diag").is_some());
        std::fs::remove_file(input).ok();
        std::fs::remove_file(output).ok();
    }

    #[test]
    fn old_opaque_blob_is_refused() {
        let header =
            r#"{"rnnoise.weights_blob_f32":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&0f32.to_le_bytes());
        let input = scratch("opaque.safetensors");
        let output = scratch("opaque.gguf");
        std::fs::write(&input, bytes).unwrap();
        let error = convert_rnnoise_file(&input, &output, None).unwrap_err();
        assert!(error.to_string().contains("exactly 36 canonical arrays"));
        assert!(!output.exists());
        std::fs::remove_file(input).ok();
    }
}
