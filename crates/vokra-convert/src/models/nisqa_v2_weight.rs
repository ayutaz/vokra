//! Strict offline conversion for the public NISQA v2 multidimensional weights.
//!
//! The trusted prepare sidecar flattens upstream `weights/nisqa.tar` to F32
//! safetensors. This converter accepts only the exact 94-tensor release, stamps
//! checkpoint-derived front-end/topology values, and keeps the upstream
//! CC-BY-NC-SA-4.0 license fail-closed. Pickle and ONNX never enter the runtime.

use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

pub const ARCH: &str = "nisqa_v2_weight";
pub const NAME: &str = "nisqa_v2_weight";
pub const CATEGORY: &str = "eval";
pub const UPSTREAM_URL: &str = "github.com/gabrielmittag/NISQA";
pub const DEFAULT_LICENSE_SPDX: &str = "cc-by-nc-sa-4.0";
pub const TENSOR_COUNT: usize = 94;

pub(crate) const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
pub(crate) const KEY_PROVENANCE_UPSTREAM_URL: &str = "vokra.provenance.upstream_url";

const SOURCE_REVISION: &str = "fe84f0f252abec382b24367d5b22498a7ce34dbb";
const SOURCE_MODEL_DEF_SHA256: &str =
    "f3ace1c00e21ae06e5d0fed9710f4e988c13685b2316a3b3ded46607fb25b71e";
const SOURCE_CONFIG_SHA256: &str =
    "afa752835c45f5d052787c024b10eab26eba980e0bde85632e674dbe557ec764";
const SOURCE_WEIGHT_LICENSE_SHA256: &str =
    "5b8e7938e1b5e0a675869ffe429cc8e7cc187d76a7c6ea1e0546c412782a43da";
const SOURCE_CHECKPOINT_SHA256: &str =
    "7ec4cf937514dd3f8860b21e66fabd8ca87a168572675ef8d979c4c4ad2e805c";
const PUBLIC_HF: &str = "vokra/nisqa-v2-weight";
const PUBLIC_REVISION: &str = "89718b026e17d3d048aa394ef8c8ddd14fee9cd8";
const PUBLIC_GGUF_SHA256: &str = "a2cacbe6f81ea2e8255eb0e2137d70d245823758e1cc4bb180c6b7cccc131e07";
const MANIFEST_SHA256: &str = "4845124c35587de7417acecac877e0f7bb131183d4aace79e47f361b7dc673f4";

const KEY_SOURCE_REVISION: &str = "vokra.nisqa.source_revision";
const KEY_SOURCE_MODEL_DEF_SHA256: &str = "vokra.nisqa.source_model_def_sha256";
const KEY_SOURCE_CONFIG_SHA256: &str = "vokra.nisqa.source_config_sha256";
const KEY_SOURCE_WEIGHT_LICENSE_SHA256: &str = "vokra.nisqa.source_weight_license_sha256";
const KEY_SOURCE_CHECKPOINT_SHA256: &str = "vokra.nisqa.source_checkpoint_sha256";
const KEY_PUBLIC_HF: &str = "vokra.nisqa.public_hf";
const KEY_PUBLIC_REVISION: &str = "vokra.nisqa.public_revision";
const KEY_PUBLIC_GGUF_SHA256: &str = "vokra.nisqa.public_gguf_sha256";
const KEY_MANIFEST_SHA256: &str = "vokra.nisqa.manifest_sha256";

const KEY_SAMPLE_RATE: &str = "vokra.nisqa.sample_rate";
const KEY_N_FFT: &str = "vokra.nisqa.n_fft";
const KEY_HOP_LENGTH_SEC: &str = "vokra.nisqa.hop_length_sec";
const KEY_WIN_LENGTH_SEC: &str = "vokra.nisqa.win_length_sec";
const KEY_N_MELS: &str = "vokra.nisqa.n_mels";
const KEY_FMAX: &str = "vokra.nisqa.fmax";
const KEY_SEG_LENGTH: &str = "vokra.nisqa.seg_length";
const KEY_SEG_HOP_LENGTH: &str = "vokra.nisqa.seg_hop_length";
const KEY_MAX_SEGMENTS: &str = "vokra.nisqa.max_segments";
const KEY_POOL_1_H: &str = "vokra.nisqa.cnn_pool_1_h";
const KEY_POOL_1_W: &str = "vokra.nisqa.cnn_pool_1_w";
const KEY_POOL_2_H: &str = "vokra.nisqa.cnn_pool_2_h";
const KEY_POOL_2_W: &str = "vokra.nisqa.cnn_pool_2_w";
const KEY_POOL_3_H: &str = "vokra.nisqa.cnn_pool_3_h";
const KEY_POOL_3_W: &str = "vokra.nisqa.cnn_pool_3_w";
const KEY_TD_SA_NHEAD: &str = "vokra.nisqa.td_sa_nhead";

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
/// Counters from one strict NISQA conversion.
pub struct NisqaV2WeightReport {
    /// Tensor entries validated on the safetensors input side.
    pub read: usize,
    /// Exact F32 tensors written to GGUF.
    pub written: usize,
    /// Retained for API compatibility; strict conversion never skips tensors.
    pub skipped_non_float: usize,
    /// Retained for API compatibility; the exact release is F32-only.
    pub bf16_passthrough: usize,
}

/// Converts the exact prepared NISQA multidimensional checkpoint to GGUF.
///
/// A `license` argument is accepted for shared CLI compatibility, but it may
/// only repeat the canonical CC-BY-NC-SA-4.0 identifier. Relicensing the
/// canonical weights is rejected before any output is written.
pub fn convert_nisqa_v2_weight_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<NisqaV2WeightReport, ConvertError> {
    if license.is_some_and(|value| !value.trim().eq_ignore_ascii_case(DEFAULT_LICENSE_SPDX)) {
        return Err(ConvertError::Parse(format!(
            "nisqa-v2-weight: the canonical weights are {DEFAULT_LICENSE_SPDX}; a conflicting --license override is not permitted"
        )));
    }
    let st = SafetensorsFile::parse(std::fs::read(input)?)?;
    validate_manifest(&st)?;

    let mut builder = GgufBuilder::new();
    builder.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    builder.add_string(chunks::KEY_MODEL_NAME, NAME);
    builder.add_string(KEY_MODEL_CATEGORY, CATEGORY);
    builder.add_string(KEY_PROVENANCE_UPSTREAM_URL, UPSTREAM_URL);
    vokra_core::stamp_provenance(
        &mut builder,
        LicenseClass::NonCommercialShareAlike,
        DEFAULT_LICENSE_SPDX,
        Some(NAME),
        Some(
            "github.com/gabrielmittag/NISQA (NISQA_DIM, weights/nisqa.tar, \
             CC-BY-NC-SA-4.0; publish requires --allow-noncommercial and \
             share-alike preservation)",
        ),
    );
    stamp_contract(&mut builder);

    for tensor in st.tensors() {
        builder.add_tensor(
            &tensor.name,
            tensor.dtype,
            tensor.shape.clone(),
            st.tensor_bytes(tensor).to_vec(),
        )?;
    }
    std::fs::write(output, builder.to_bytes()?)?;
    Ok(NisqaV2WeightReport {
        read: TENSOR_COUNT,
        written: TENSOR_COUNT,
        skipped_non_float: 0,
        bf16_passthrough: 0,
    })
}

fn validate_manifest(st: &SafetensorsFile) -> Result<(), ConvertError> {
    let expected = expected_manifest();
    if st.tensors().len() != expected.len() {
        return Err(ConvertError::Parse(format!(
            "nisqa-v2-weight: tensor count {}, expected {} for the exact NISQA_DIM release",
            st.tensors().len(),
            expected.len()
        )));
    }
    for (name, shape) in &expected {
        let tensor = st
            .tensors()
            .iter()
            .find(|tensor| tensor.name == *name)
            .ok_or_else(|| ConvertError::Parse(format!("nisqa-v2-weight: missing `{name}`")))?;
        if tensor.dtype != GgmlType::F32 {
            return Err(ConvertError::Parse(format!(
                "nisqa-v2-weight: `{name}` is {:?}, expected exact upstream F32",
                tensor.dtype
            )));
        }
        if tensor.shape != *shape {
            return Err(ConvertError::Parse(format!(
                "nisqa-v2-weight: `{name}` shape {:?}, expected {shape:?}",
                tensor.shape
            )));
        }
    }
    Ok(())
}

fn stamp_contract(builder: &mut GgufBuilder) {
    for (key, value) in [
        (KEY_SOURCE_REVISION, SOURCE_REVISION),
        (KEY_SOURCE_MODEL_DEF_SHA256, SOURCE_MODEL_DEF_SHA256),
        (KEY_SOURCE_CONFIG_SHA256, SOURCE_CONFIG_SHA256),
        (
            KEY_SOURCE_WEIGHT_LICENSE_SHA256,
            SOURCE_WEIGHT_LICENSE_SHA256,
        ),
        (KEY_SOURCE_CHECKPOINT_SHA256, SOURCE_CHECKPOINT_SHA256),
        (KEY_PUBLIC_HF, PUBLIC_HF),
        (KEY_PUBLIC_REVISION, PUBLIC_REVISION),
        (KEY_PUBLIC_GGUF_SHA256, PUBLIC_GGUF_SHA256),
        (KEY_MANIFEST_SHA256, MANIFEST_SHA256),
    ] {
        builder.add_string(key, value);
    }
    builder.add_u32(KEY_SAMPLE_RATE, 0);
    builder.add_u32(KEY_N_FFT, 4096);
    builder.add_f32(KEY_HOP_LENGTH_SEC, 0.01);
    builder.add_f32(KEY_WIN_LENGTH_SEC, 0.02);
    builder.add_u32(KEY_N_MELS, 48);
    builder.add_f32(KEY_FMAX, 20_000.0);
    builder.add_u32(KEY_SEG_LENGTH, 15);
    builder.add_u32(KEY_SEG_HOP_LENGTH, 4);
    builder.add_u32(KEY_MAX_SEGMENTS, 1300);
    builder.add_u32(KEY_POOL_1_H, 24);
    builder.add_u32(KEY_POOL_1_W, 7);
    builder.add_u32(KEY_POOL_2_H, 12);
    builder.add_u32(KEY_POOL_2_W, 5);
    builder.add_u32(KEY_POOL_3_H, 6);
    builder.add_u32(KEY_POOL_3_W, 3);
    builder.add_u32(KEY_TD_SA_NHEAD, 1);
}

fn expected_manifest() -> Vec<(String, Vec<u64>)> {
    let mut tensors = Vec::with_capacity(TENSOR_COUNT);
    let channels = [1u64, 16, 32, 64, 64, 64, 64];
    for layer in 1..=6 {
        let output = channels[layer];
        for suffix in ["bias", "running_mean", "running_var", "weight"] {
            tensors.push((format!("cnn.model.bn{layer}.{suffix}"), vec![output]));
        }
        tensors.push((format!("cnn.model.conv{layer}.bias"), vec![output]));
        tensors.push((
            format!("cnn.model.conv{layer}.weight"),
            vec![output, channels[layer - 1], 3, 3],
        ));
    }
    for head in 0..5 {
        let prefix = format!("pool_layers.{head}.model");
        for (suffix, shape) in [
            ("linear1.bias", vec![128]),
            ("linear1.weight", vec![128, 64]),
            ("linear2.bias", vec![1]),
            ("linear2.weight", vec![1, 128]),
            ("linear3.bias", vec![1]),
            ("linear3.weight", vec![1, 64]),
        ] {
            tensors.push((format!("{prefix}.{suffix}"), shape));
        }
    }
    for layer in 0..2 {
        let prefix = format!("time_dependency.model.layers.{layer}");
        for (suffix, shape) in [
            ("linear1.bias", vec![64]),
            ("linear1.weight", vec![64, 64]),
            ("linear2.bias", vec![64]),
            ("linear2.weight", vec![64, 64]),
            ("norm1.bias", vec![64]),
            ("norm1.weight", vec![64]),
            ("norm2.bias", vec![64]),
            ("norm2.weight", vec![64]),
            ("self_attn.in_proj_bias", vec![192]),
            ("self_attn.in_proj_weight", vec![192, 64]),
            ("self_attn.out_proj.bias", vec![64]),
            ("self_attn.out_proj.weight", vec![64, 64]),
        ] {
            tensors.push((format!("{prefix}.{suffix}"), shape));
        }
    }
    for (name, shape) in [
        ("time_dependency.model.linear.bias", vec![64]),
        ("time_dependency.model.linear.weight", vec![64, 384]),
        ("time_dependency.model.norm1.bias", vec![64]),
        ("time_dependency.model.norm1.weight", vec![64]),
    ] {
        tensors.push((name.to_owned(), shape));
    }
    debug_assert_eq!(tensors.len(), TENSOR_COUNT);
    tensors
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fmt::Write as _;
    use std::path::PathBuf;
    use vokra_core::gguf::GgufFile;

    fn scratch(tag: &str, extension: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "vokra-nisqa-{tag}-{}-{}.{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock")
                .as_nanos(),
            extension
        ))
    }

    struct Guard(PathBuf);
    impl Drop for Guard {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    fn write_fixture(path: &Path) {
        let manifest = expected_manifest();
        let mut header = String::from("{");
        let mut offset = 0usize;
        for (index, (name, shape)) in manifest.iter().enumerate() {
            if index != 0 {
                header.push(',');
            }
            let elements: u64 = shape.iter().product();
            let end = offset + elements as usize * 4;
            write!(
                header,
                "\"{name}\":{{\"dtype\":\"F32\",\"shape\":{shape:?},\"data_offsets\":[{offset},{end}]}}"
            )
            .expect("header");
            offset = end;
        }
        header.push('}');
        let mut bytes = Vec::with_capacity(8 + header.len() + offset);
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        bytes.resize(8 + header.len() + offset, 0);
        std::fs::write(path, bytes).expect("fixture");
    }

    #[test]
    fn exact_manifest_converts_and_stamps_checkpoint_args() {
        let input = scratch("input", "safetensors");
        let output = scratch("output", "gguf");
        let _input_guard = Guard(input.clone());
        let _output_guard = Guard(output.clone());
        write_fixture(&input);

        let report = convert_nisqa_v2_weight_file(&input, &output, None).expect("convert");
        assert_eq!(report.written, TENSOR_COUNT);
        let file = GgufFile::open(&output).expect("GGUF");
        assert_eq!(file.tensors().len(), TENSOR_COUNT);
        assert_eq!(
            file.get(KEY_N_FFT).and_then(|value| value.as_u64()),
            Some(4096)
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|value| value.as_str()),
            Some(LicenseClass::NonCommercialShareAlike.as_str())
        );
    }

    #[test]
    fn conflicting_license_override_is_rejected_before_output() {
        let input = scratch("missing", "safetensors");
        let output = scratch("not-written", "gguf");
        let error = convert_nisqa_v2_weight_file(&input, &output, Some("apache-2.0"))
            .expect_err("relicensing must fail");
        assert!(matches!(error, ConvertError::Parse(message) if message.contains("not permitted")));
        assert!(!output.exists());
    }
}
