//! Strict converter for Microsoft's DNSMOS P.808 + P.835 release.
//!
//! The runtime never loads ONNX. The offline Python sidecar verifies the two
//! official ONNX files, extracts their 38 floating-point initializers, and
//! writes one prefixed safetensors bundle. This converter accepts only that
//! complete name/shape/dtype manifest; partial, renamed, reshaped, non-F32, or
//! non-finite inputs fail before an output file is written.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{
    GgmlType, GgufArray, GgufBuilder, GgufMetadataValue, GgufValueType, chunks,
};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

pub const ARCH: &str = "dnsmos";
pub const NAME: &str = "dnsmos-p808-p835";
pub const CATEGORY: &str = "eval";
pub const UPSTREAM_URL: &str = "https://github.com/microsoft/DNS-Challenge/tree/master/DNSMOS";
pub const SOURCE_REVISION: &str = "591184a9fcb2cbdec02520fed81a32bbbf9d73ff";
pub const P808_ONNX_SHA256: &str =
    "9246480c58567bc6affd4200938e77eef49468c8bc7ed3776d109c07456f6e91";
pub const P835_ONNX_SHA256: &str =
    "269fbebdb513aa23cddfbb593542ecc540284a91849ac50516870e1ac78f6edd";
pub const SOURCE_PY_SHA256: &str =
    "1ab566afe006daab32ac7073296a5d0ef99f8b82f91c7266f3ccf26113d7a28b";
pub const SOURCE_LICENSE_SHA256: &str =
    "d6239afa918961b465b07bf7411cbe34ff6685854f58553db7966f4881a0211f";
pub const PUBLIC_HF: &str = "vokra/dnsmos-p808-p835";
pub const PUBLIC_REVISION: &str = "39293917b4fccf66b149c0734140427f29f5ff84";
pub const PUBLIC_GGUF_SHA256: &str =
    "b13c264f26a83b92d27f4385332e69e426f3301d2e48de7732c2aa9355650b2d";
pub const MANIFEST_SHA256: &str =
    "d6d13fd5191d399736c8c1558d9dbbc51718a377190836a640a1992dbf404847";
pub const DEFAULT_LICENSE: &str = "mit";

pub const TENSOR_COUNT: usize = 38;
pub const SAMPLE_RATE: u32 = 16_000;
pub const INPUT_LENGTH_SAMPLES: u32 = 144_160;
pub const P808_FRAMES: u32 = 900;
pub const P808_N_FFT: u32 = 321;
pub const P808_HOP: u32 = 160;
pub const P808_N_MELS: u32 = 120;
pub const P835_FRAMES: u32 = 900;
pub const P835_WINDOW: u32 = 320;
pub const P835_HOP: u32 = 160;
pub const P835_BINS: u32 = 161;

const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
const KEY_PROVENANCE_UPSTREAM_URL: &str = "vokra.provenance.upstream_url";
pub const KEY_DNSMOS_BUNDLE: &str = "vokra.dnsmos.bundle";
pub const KEY_DNSMOS_SAMPLE_RATE: &str = "vokra.dnsmos.sample_rate";
pub const KEY_DNSMOS_P808_CKPT: &str = "vokra.dnsmos.p808.checkpoint";
pub const KEY_DNSMOS_P835_CKPT: &str = "vokra.dnsmos.p835.checkpoint";
pub const KEY_SOURCE_REVISION: &str = "vokra.dnsmos.source_revision";
pub const KEY_P808_ONNX_SHA256: &str = "vokra.dnsmos.p808.onnx_sha256";
pub const KEY_P835_ONNX_SHA256: &str = "vokra.dnsmos.p835.onnx_sha256";
pub const KEY_SOURCE_PY_SHA256: &str = "vokra.dnsmos.source_py_sha256";
pub const KEY_SOURCE_LICENSE_SHA256: &str = "vokra.dnsmos.source_license_sha256";
pub const KEY_PUBLIC_HF: &str = "vokra.dnsmos.public_hf";
pub const KEY_PUBLIC_REVISION: &str = "vokra.dnsmos.public_revision";
pub const KEY_PUBLIC_GGUF_SHA256: &str = "vokra.dnsmos.public_gguf_sha256";
pub const KEY_MANIFEST_SHA256: &str = "vokra.dnsmos.manifest_sha256";
pub const KEY_INPUT_LENGTH: &str = "vokra.dnsmos.input_length";
pub const KEY_P808_FRAMES: &str = "vokra.dnsmos.p808.frames";
pub const KEY_P808_N_FFT: &str = "vokra.dnsmos.p808.n_fft";
pub const KEY_P808_HOP: &str = "vokra.dnsmos.p808.hop";
pub const KEY_P808_N_MELS: &str = "vokra.dnsmos.p808.n_mels";
pub const KEY_P835_FRAMES: &str = "vokra.dnsmos.p835.frames";
pub const KEY_P835_WINDOW: &str = "vokra.dnsmos.p835.window";
pub const KEY_P835_HOP: &str = "vokra.dnsmos.p835.hop";
pub const KEY_P835_BINS: &str = "vokra.dnsmos.p835.bins";

#[derive(Debug, Default, Clone, PartialEq, Eq)]
/// Auditable counters for one strict DNSMOS conversion.
pub struct DnsmosReport {
    /// Input tensors inspected after parsing.
    pub read: usize,
    /// Exact F32 tensors written to GGUF.
    pub written: usize,
    /// Always zero for a successful strict conversion.
    pub skipped_non_float: usize,
    /// Always zero because the official release is F32.
    pub bf16_passthrough: usize,
    /// Always two for a successful strict conversion.
    pub bundle_variants: usize,
}

/// Converts the exact offline-prepared DNSMOS bundle into GGUF.
pub fn convert_dnsmos_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<DnsmosReport, ConvertError> {
    let spdx = require_official_license(license)?;
    let bytes = std::fs::read(input)?;
    let tensors = SafetensorsFile::parse(bytes)?;
    validate_input_manifest(&tensors)?;

    let mut builder = GgufBuilder::new();
    stamp_contract(&mut builder, spdx);
    for tensor in tensors.tensors() {
        builder.add_tensor(
            &tensor.name,
            tensor.dtype,
            tensor.shape.clone(),
            tensors.tensor_bytes(tensor).to_vec(),
        )?;
    }
    std::fs::write(output, builder.to_bytes()?)?;
    Ok(DnsmosReport {
        read: TENSOR_COUNT,
        written: TENSOR_COUNT,
        skipped_non_float: 0,
        bf16_passthrough: 0,
        bundle_variants: 2,
    })
}

fn require_official_license(license: Option<&str>) -> Result<&'static str, ConvertError> {
    let requested = license.unwrap_or(DEFAULT_LICENSE).trim();
    if !requested.eq_ignore_ascii_case(DEFAULT_LICENSE) {
        return Err(parse_error(format!(
            "license override {requested:?} conflicts with the pinned MIT DNSMOS checkpoint"
        )));
    }
    Ok(DEFAULT_LICENSE)
}

fn stamp_contract(builder: &mut GgufBuilder, spdx: &str) {
    builder.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    builder.add_string(chunks::KEY_MODEL_NAME, NAME);
    builder.add_string(KEY_MODEL_CATEGORY, CATEGORY);
    builder.add_string(KEY_PROVENANCE_UPSTREAM_URL, UPSTREAM_URL);
    builder.add_metadata(
        KEY_DNSMOS_BUNDLE,
        GgufMetadataValue::Array(GgufArray {
            element_type: GgufValueType::String,
            values: vec![
                GgufMetadataValue::String("p808".to_owned()),
                GgufMetadataValue::String("p835".to_owned()),
            ],
        }),
    );
    builder.add_u32(KEY_DNSMOS_SAMPLE_RATE, SAMPLE_RATE);
    builder.add_string(KEY_DNSMOS_P808_CKPT, "model_v8.onnx");
    builder.add_string(KEY_DNSMOS_P835_CKPT, "sig_bak_ovr.onnx");
    builder.add_string(KEY_SOURCE_REVISION, SOURCE_REVISION);
    builder.add_string(KEY_P808_ONNX_SHA256, P808_ONNX_SHA256);
    builder.add_string(KEY_P835_ONNX_SHA256, P835_ONNX_SHA256);
    builder.add_string(KEY_SOURCE_PY_SHA256, SOURCE_PY_SHA256);
    builder.add_string(KEY_SOURCE_LICENSE_SHA256, SOURCE_LICENSE_SHA256);
    builder.add_string(KEY_PUBLIC_HF, PUBLIC_HF);
    builder.add_string(KEY_PUBLIC_REVISION, PUBLIC_REVISION);
    builder.add_string(KEY_PUBLIC_GGUF_SHA256, PUBLIC_GGUF_SHA256);
    builder.add_string(KEY_MANIFEST_SHA256, MANIFEST_SHA256);
    builder.add_u32(KEY_INPUT_LENGTH, INPUT_LENGTH_SAMPLES);
    builder.add_u32(KEY_P808_FRAMES, P808_FRAMES);
    builder.add_u32(KEY_P808_N_FFT, P808_N_FFT);
    builder.add_u32(KEY_P808_HOP, P808_HOP);
    builder.add_u32(KEY_P808_N_MELS, P808_N_MELS);
    builder.add_u32(KEY_P835_FRAMES, P835_FRAMES);
    builder.add_u32(KEY_P835_WINDOW, P835_WINDOW);
    builder.add_u32(KEY_P835_HOP, P835_HOP);
    builder.add_u32(KEY_P835_BINS, P835_BINS);
    vokra_core::stamp_provenance(
        builder,
        LicenseClass::Permissive,
        spdx,
        Some(NAME),
        Some("microsoft/DNS-Challenge DNSMOS P.808 + P.835 (MIT)"),
    );
}

fn validate_input_manifest(tensors: &SafetensorsFile) -> Result<(), ConvertError> {
    let expected = expected_manifest();
    if tensors.tensors().len() != TENSOR_COUNT {
        return Err(parse_error(format!(
            "checkpoint has {} tensors, expected exactly {TENSOR_COUNT}",
            tensors.tensors().len()
        )));
    }
    let mut seen = BTreeSet::new();
    for tensor in tensors.tensors() {
        let shape = expected
            .get(&tensor.name)
            .ok_or_else(|| parse_error(format!("unexpected tensor {}", tensor.name)))?;
        if &tensor.shape != shape {
            return Err(parse_error(format!(
                "tensor {} has shape {:?}, expected {shape:?}",
                tensor.name, tensor.shape
            )));
        }
        if tensor.dtype != GgmlType::F32 {
            return Err(parse_error(format!(
                "tensor {} is {:?}, expected F32 for DNSMOS",
                tensor.name, tensor.dtype
            )));
        }
        for (index, bytes) in tensors.tensor_bytes(tensor).chunks_exact(4).enumerate() {
            let value = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
            if !value.is_finite() {
                return Err(parse_error(format!(
                    "tensor {} contains a non-finite value at element {index}",
                    tensor.name
                )));
            }
        }
        seen.insert(tensor.name.as_str());
    }
    if let Some(missing) = expected.keys().find(|name| !seen.contains(name.as_str())) {
        return Err(parse_error(format!(
            "checkpoint is missing tensor {missing}"
        )));
    }
    Ok(())
}

fn parse_error(message: impl Into<String>) -> ConvertError {
    ConvertError::Parse(format!("dnsmos: {}", message.into()))
}

fn insert(manifest: &mut BTreeMap<String, Vec<u64>>, name: &str, shape: &[u64]) {
    assert!(
        manifest.insert(name.to_owned(), shape.to_vec()).is_none(),
        "duplicate DNSMOS manifest tensor {name}"
    );
}

pub(crate) fn expected_manifest() -> BTreeMap<String, Vec<u64>> {
    let mut m = BTreeMap::new();
    for (name, shape) in [
        ("p808.conv2d_5/kernel:0", &[32, 1, 3, 3][..]),
        ("p808.conv2d_5/bias:0", &[32]),
        ("p808.conv2d_6/kernel:0", &[32, 32, 3, 3]),
        ("p808.conv2d_6/bias:0", &[32]),
        ("p808.conv2d_7/kernel:0", &[32, 32, 3, 3]),
        ("p808.conv2d_7/bias:0", &[32]),
        ("p808.conv2d_8/kernel:0", &[32, 32, 3, 3]),
        ("p808.conv2d_8/bias:0", &[32]),
        ("p808.conv2d_9/kernel:0", &[64, 32, 3, 3]),
        ("p808.conv2d_9/bias:0", &[64]),
        (
            "p808.mos_estimator_small_1/dense_3/MatMul/ReadVariableOp/resource:0",
            &[64, 64],
        ),
        (
            "p808.mos_estimator_small_1/dense_3/BiasAdd/ReadVariableOp/resource:0",
            &[64],
        ),
        (
            "p808.mos_estimator_small_1/dense_4/MatMul/ReadVariableOp/resource:0",
            &[64, 64],
        ),
        (
            "p808.mos_estimator_small_1/dense_4/BiasAdd/ReadVariableOp/resource:0",
            &[64],
        ),
        (
            "p808.mos_estimator_small_1/dense_5/MatMul/ReadVariableOp/resource:0",
            &[64, 1],
        ),
        (
            "p808.mos_estimator_small_1/dense_5/BiasAdd/ReadVariableOp/resource:0",
            &[1],
        ),
        ("p835.time2freq/stft-real/kernel:0", &[161, 320, 1]),
        ("p835.time2freq/stft-imag/kernel:0", &[161, 320, 1]),
        ("p835.conv2d/kernel:0", &[128, 1, 3, 3]),
        ("p835.conv2d/bias:0", &[128]),
        ("p835.conv2d_1/kernel:0", &[64, 128, 3, 3]),
        ("p835.conv2d_1/bias:0", &[64]),
        ("p835.conv2d_2/kernel:0", &[64, 64, 3, 3]),
        ("p835.conv2d_2/bias:0", &[64]),
        ("p835.conv2d_3/kernel:0", &[32, 64, 3, 3]),
        ("p835.conv2d_3/bias:0", &[32]),
        ("p835.conv2d_4/kernel:0", &[32, 32, 3, 3]),
        ("p835.conv2d_4/bias:0", &[32]),
        ("p835.conv2d_5/kernel:0", &[32, 32, 3, 3]),
        ("p835.conv2d_5/bias:0", &[32]),
        ("p835.conv2d_6/kernel:0", &[64, 32, 3, 3]),
        ("p835.conv2d_6/bias:0", &[64]),
        (
            "p835.mos_estimator_logpow/dense/MatMul/ReadVariableOp/resource:0",
            &[64, 128],
        ),
        (
            "p835.mos_estimator_logpow/dense/BiasAdd/ReadVariableOp/resource:0",
            &[128],
        ),
        (
            "p835.mos_estimator_logpow/dense_1/MatMul/ReadVariableOp/resource:0",
            &[128, 64],
        ),
        (
            "p835.mos_estimator_logpow/dense_1/BiasAdd/ReadVariableOp/resource:0",
            &[64],
        ),
        (
            "p835.mos_estimator_logpow/dense_3/MatMul/ReadVariableOp/resource:0",
            &[64, 3],
        ),
        (
            "p835.mos_estimator_logpow/dense_3/BiasAdd/ReadVariableOp/resource:0",
            &[3],
        ),
    ] {
        insert(&mut m, name, shape);
    }
    assert_eq!(m.len(), TENSOR_COUNT);
    m
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn scratch(tag: &str, extension: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "vokra-dnsmos-{}-{tag}-{}.{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos(),
            extension
        ))
    }

    fn safetensors(manifest: &BTreeMap<String, Vec<u64>>) -> Vec<u8> {
        let mut header = String::from("{");
        let mut payload = Vec::new();
        for (index, (name, shape)) in manifest.iter().enumerate() {
            if index > 0 {
                header.push(',');
            }
            let elements = shape.iter().product::<u64>() as usize;
            let start = payload.len();
            payload.resize(start + elements * 4, 0);
            use std::fmt::Write as _;
            write!(
                header,
                "\"{name}\":{{\"dtype\":\"F32\",\"shape\":{:?},\"data_offsets\":[{start},{}]}}",
                shape,
                payload.len()
            )
            .unwrap();
        }
        header.push('}');
        let mut out = Vec::new();
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.extend_from_slice(&payload);
        out
    }

    #[test]
    fn exact_manifest_converts_and_stamps_contract() {
        let input = scratch("exact", "safetensors");
        let output = scratch("exact", "gguf");
        std::fs::write(&input, safetensors(&expected_manifest())).unwrap();
        let report = convert_dnsmos_file(&input, &output, None).unwrap();
        assert_eq!(report.read, TENSOR_COUNT);
        assert_eq!(report.written, TENSOR_COUNT);
        assert_eq!(report.bundle_variants, 2);
        let file = vokra_core::gguf::GgufFile::open(&output).unwrap();
        assert_eq!(file.tensors().len(), TENSOR_COUNT);
        assert_eq!(
            file.get(KEY_SOURCE_REVISION)
                .and_then(|value| value.as_str()),
            Some(SOURCE_REVISION)
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
    fn missing_tensor_fails_before_output() {
        let mut manifest = expected_manifest();
        manifest.remove("p835.conv2d_6/bias:0");
        let input = scratch("missing", "safetensors");
        let output = scratch("missing", "gguf");
        std::fs::write(&input, safetensors(&manifest)).unwrap();
        let error = convert_dnsmos_file(&input, &output, None).unwrap_err();
        assert!(error.to_string().contains("expected exactly 38"));
        assert!(!output.exists());
        std::fs::remove_file(input).ok();
    }

    #[test]
    fn license_override_cannot_relabel_official_weights() {
        let input = scratch("license", "safetensors");
        let output = scratch("license", "gguf");
        std::fs::write(&input, safetensors(&expected_manifest())).unwrap();
        let error = convert_dnsmos_file(&input, &output, Some("apache-2.0")).unwrap_err();
        assert!(error.to_string().contains("conflicts"));
        assert!(!output.exists());
        std::fs::remove_file(input).ok();
    }
}
