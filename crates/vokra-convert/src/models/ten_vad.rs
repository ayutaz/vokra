//! Strict conversion for the official TEN-VAD v1.0 ONNX checkpoint.
//!
//! `tools/parity/ten_vad_prepare_checkpoint.py` is the only supported ONNX
//! bridge.  It pins `TEN-framework/ten-vad` tag `v1.0-ONNX` at commit
//! `8e96899ba05a8e8c0e883ec7417e7a144bd9dec0`, verifies the released ONNX
//! SHA-256, and rewrites all 19 float initializers to stable names.  This
//! converter then checks the complete name/shape/dtype manifest before writing
//! GGUF.  ONNX and Python remain offline-only.

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// GGUF architecture discriminator.
pub const ARCH: &str = "ten_vad";
/// Canonical model name.
pub const NAME: &str = "ten-vad-v1.0";
/// Shared runtime category.
pub const CATEGORY: &str = "vad-kws";
/// Pinned primary-source repository.
pub const UPSTREAM_URL: &str = "github.com/TEN-framework/ten-vad";
/// SPDX-style identifier for Agora's restricted TEN-VAD license.
pub const DEFAULT_LICENSE_SPDX: &str = "LicenseRef-Agora-TEN-VAD-Open-Source-License-2025";
/// SPDX expression for the LPCNet-derived frontend.
pub const FRONTEND_LICENSE_SPDX: &str = "bsd-2-clause AND bsd-3-clause";
/// Pinned upstream commit for the v1.0 ONNX release.
pub const REVISION: &str = "8e96899ba05a8e8c0e883ec7417e7a144bd9dec0";
/// SHA-256 of the pinned upstream ONNX file.
pub const ONNX_SHA256: &str = "e10b98a0cab1c98e847fbdda14cb3d45a38336d47535a3f63a0fb6c4e0f4cdf4";
/// Required PCM sample rate stamped in canonical GGUF files.
pub const SAMPLE_RATE: u32 = 16_000;
/// Streaming hop size stamped in canonical GGUF files.
pub const HOP_SIZE: u32 = 256;
/// Feature width stamped in canonical GGUF files.
pub const N_FEATURES: u32 = 41;
/// Context length stamped in canonical GGUF files.
pub const CONTEXT_FRAMES: u32 = 3;
/// Recurrent hidden width stamped in canonical GGUF files.
pub const HIDDEN_DIM: u32 = 64;
/// Recurrent layer count stamped in canonical GGUF files.
pub const N_LAYERS: u32 = 2;
/// Exact float-initializer count in the pinned graph.
pub const TENSOR_COUNT: usize = 19;

pub(crate) const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
pub(crate) const KEY_PROVENANCE_UPSTREAM_URL: &str = "vokra.provenance.upstream_url";

fn tensor_spec() -> Vec<(&'static str, Vec<usize>)> {
    vec![
        ("ten_vad.conv0.depthwise.weight", vec![1, 1, 3, 3]),
        ("ten_vad.conv0.pointwise.weight", vec![16, 1, 1, 1]),
        ("ten_vad.conv0.pointwise.bias", vec![16]),
        ("ten_vad.conv1.depthwise.weight", vec![16, 1, 1, 3]),
        ("ten_vad.conv1.pointwise.weight", vec![16, 16, 1, 1]),
        ("ten_vad.conv1.pointwise.bias", vec![16]),
        ("ten_vad.conv2.depthwise.weight", vec![16, 1, 1, 3]),
        ("ten_vad.conv2.pointwise.weight", vec![16, 16, 1, 1]),
        ("ten_vad.conv2.pointwise.bias", vec![16]),
        ("ten_vad.lstm0.weight_ih", vec![1, 256, 80]),
        ("ten_vad.lstm0.weight_hh", vec![1, 256, 64]),
        ("ten_vad.lstm0.bias", vec![1, 512]),
        ("ten_vad.lstm1.weight_ih", vec![1, 256, 64]),
        ("ten_vad.lstm1.weight_hh", vec![1, 256, 64]),
        ("ten_vad.lstm1.bias", vec![1, 512]),
        ("ten_vad.dense0.weight", vec![128, 32]),
        ("ten_vad.dense0.bias", vec![32]),
        ("ten_vad.dense1.weight", vec![32, 1]),
        ("ten_vad.dense1.bias", vec![1]),
    ]
}

/// Outcome of a strict TEN-VAD conversion.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct TenVadReport {
    /// Source tensors inspected (always 19 on success).
    pub read: usize,
    /// Canonical tensors written (always 19 on success).
    pub written: usize,
    /// Kept for report compatibility; strict conversion rejects non-F32 input.
    pub skipped_non_float: usize,
    /// Kept for report compatibility; the official release contains no BF16.
    pub bf16_passthrough: usize,
}

/// Converts the canonical sidecar output to a native GGUF.
pub fn convert_ten_vad_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<TenVadReport, ConvertError> {
    let bytes = std::fs::read(input)?;
    let st = SafetensorsFile::parse(bytes)?;
    if st.tensors().len() != TENSOR_COUNT {
        return Err(ConvertError::Parse(format!(
            "ten-vad-v1.0: source manifest has {} tensors, expected exactly {TENSOR_COUNT}; run tools/parity/ten_vad_prepare_checkpoint.py against revision {REVISION}",
            st.tensors().len()
        )));
    }

    let spec = tensor_spec();
    for (name, expected) in &spec {
        let tensor = st
            .tensors()
            .iter()
            .find(|tensor| tensor.name == *name)
            .ok_or_else(|| ConvertError::Parse(format!("ten-vad-v1.0: missing tensor `{name}`")))?;
        let expected_u64 = expected.iter().map(|&dim| dim as u64).collect::<Vec<_>>();
        if tensor.shape != expected_u64 {
            return Err(ConvertError::Parse(format!(
                "ten-vad-v1.0: tensor `{name}` has shape {:?}, expected {expected_u64:?}",
                tensor.shape
            )));
        }
        if tensor.dtype != GgmlType::F32 {
            return Err(ConvertError::Parse(format!(
                "ten-vad-v1.0: tensor `{name}` is {:?}, expected F32 from the pinned official ONNX",
                tensor.dtype
            )));
        }
    }
    for tensor in st.tensors() {
        if !spec.iter().any(|(name, _)| *name == tensor.name) {
            return Err(ConvertError::Parse(format!(
                "ten-vad-v1.0: unexpected tensor `{}`; the pinned manifest has exactly {TENSOR_COUNT} tensors",
                tensor.name
            )));
        }
    }

    let mut builder = GgufBuilder::new();
    builder.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    builder.add_string(chunks::KEY_MODEL_NAME, NAME);
    builder.add_string(KEY_MODEL_CATEGORY, CATEGORY);
    builder.add_string(KEY_PROVENANCE_UPSTREAM_URL, UPSTREAM_URL);
    builder.add_string("vokra.ten_vad.revision", REVISION);
    builder.add_string("vokra.ten_vad.onnx_sha256", ONNX_SHA256);
    builder.add_string("vokra.ten_vad.frontend_license_spdx", FRONTEND_LICENSE_SPDX);
    builder.add_u32("vokra.ten_vad.sample_rate", SAMPLE_RATE);
    builder.add_u32("vokra.ten_vad.hop_size", HOP_SIZE);
    builder.add_u32("vokra.ten_vad.n_features", N_FEATURES);
    builder.add_u32("vokra.ten_vad.context_frames", CONTEXT_FRAMES);
    builder.add_u32("vokra.ten_vad.hidden_dim", HIDDEN_DIM);
    builder.add_u32("vokra.ten_vad.n_layers", N_LAYERS);

    // The upstream file is not plain Apache-2.0: it adds non-compete,
    // application-only deployment conditions and binds derivatives to those
    // terms. A generic GGUF mirror would enable third-party applications, so
    // the default must remain non-publishable. An explicit override represents
    // a separately negotiated redistribution grant held by the caller.
    let (effective_spdx, effective_class) = match license {
        Some(value) if !value.is_empty() => (value, LicenseClass::from_license_str(value)),
        _ => (DEFAULT_LICENSE_SPDX, LicenseClass::RedistributionForbidden),
    };
    vokra_core::stamp_provenance(
        &mut builder,
        effective_class,
        effective_spdx,
        Some(NAME),
        Some(
            "TEN-framework/ten-vad v1.0-ONNX; Agora restricted deployment license with BSD-2-Clause and BSD-3-Clause LPCNet-derived frontend notices",
        ),
    );

    for (name, dims) in &spec {
        let tensor = st
            .tensors()
            .iter()
            .find(|tensor| tensor.name == *name)
            .expect("complete manifest validated above");
        builder.add_tensor(
            name,
            GgmlType::F32,
            dims.iter().map(|&dim| dim as u64).collect(),
            st.tensor_bytes(tensor).to_vec(),
        )?;
    }
    std::fs::write(output, builder.to_bytes()?)?;
    Ok(TenVadReport {
        read: TENSOR_COUNT,
        written: TENSOR_COUNT,
        skipped_non_float: 0,
        bf16_passthrough: 0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use vokra_core::gguf::GgufFile;

    fn scratch_path(tag: &str, ext: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "vokra-ten-vad-{tag}-{}-{}.{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|duration| duration.as_nanos())
                .unwrap_or(0),
            ext
        ))
    }

    struct TempFileGuard(PathBuf);
    impl Drop for TempFileGuard {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    fn canonical_safetensors() -> Vec<u8> {
        let spec = tensor_spec();
        let mut offset = 0usize;
        let mut entries = Vec::new();
        let mut payload = Vec::new();
        for (name, dims) in spec {
            let bytes = dims.iter().product::<usize>() * 4;
            entries.push(format!(
                "\"{name}\":{{\"dtype\":\"F32\",\"shape\":{:?},\"data_offsets\":[{offset},{}]}}",
                dims,
                offset + bytes
            ));
            payload.resize(payload.len() + bytes, 0);
            offset += bytes;
        }
        let header = format!("{{{}}}", entries.join(","));
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&payload);
        bytes
    }

    #[test]
    fn strict_manifest_stamps_native_contract() {
        let input = scratch_path("canonical", "safetensors");
        let output = scratch_path("canonical", "gguf");
        std::fs::write(&input, canonical_safetensors()).unwrap();
        let _input_guard = TempFileGuard(input.clone());
        let _output_guard = TempFileGuard(output.clone());
        let report = convert_ten_vad_file(&input, &output, None).unwrap();
        assert_eq!(report.read, TENSOR_COUNT);
        assert_eq!(report.written, TENSOR_COUNT);
        let file = GgufFile::open(&output).unwrap();
        assert_eq!(file.tensors().len(), TENSOR_COUNT);
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH)
                .and_then(|value| value.as_str()),
            Some(ARCH)
        );
        assert_eq!(
            file.get("vokra.ten_vad.revision")
                .and_then(|value| value.as_str()),
            Some(REVISION)
        );
        assert_eq!(
            file.get("vokra.ten_vad.context_frames")
                .and_then(|value| value.as_u64()),
            Some(CONTEXT_FRAMES as u64)
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|value| value.as_str()),
            Some(LicenseClass::RedistributionForbidden.as_str())
        );
    }

    #[test]
    fn incomplete_manifest_is_rejected() {
        let header = r#"{"ten_vad.dense1.bias":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&0.0f32.to_le_bytes());
        let input = scratch_path("incomplete", "safetensors");
        let output = scratch_path("incomplete", "gguf");
        std::fs::write(&input, bytes).unwrap();
        let _input_guard = TempFileGuard(input.clone());
        let error = convert_ten_vad_file(&input, &output, None).unwrap_err();
        assert!(error.to_string().contains("expected exactly 19"));
    }
}
