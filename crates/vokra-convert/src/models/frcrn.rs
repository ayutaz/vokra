//! Exact Alibaba FRCRN-SE-16K checkpoint to GGUF conversion.
//!
//! The official checkpoint is a PyTorch pickle. The offline
//! `tools/parity/frcrn_prepare_checkpoint.py` sidecar extracts its inference
//! state into safetensors; this converter then accepts exactly the audited
//! 812-tensor F32 topology. Missing, extra, renamed, reshaped, or non-F32
//! tensors are errors. An arbitrary float checkpoint can therefore no longer
//! acquire the `frcrn` architecture tag.
//!
//! The topology is independently transcribed from ClearerVoice-Studio source
//! revision [`SOURCE_REVISION`] and its `FRCRN_SE_16K.yaml`. Code and weights
//! are Apache-2.0: the weight declaration is the official
//! `alibabasglab/FRCRN_SE_16K` model card, while the implementation license is
//! the ClearerVoice-Studio root `LICENSE`. The older standalone FRCRN GitHub
//! repository does not currently carry a license file and is not used as the
//! weight-license source.
//!
//! Runtime execution is native Rust. Neither this converter nor the runtime
//! consumes ONNX, protobuf, or a Python runtime.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

/// Runtime/converter architecture tag.
pub const ARCH: &str = "frcrn";
/// Canonical public model name retained for compatibility with `vokra/frcrn`.
pub const NAME: &str = "frcrn";
/// Model-zoo category.
pub const CATEGORY: &str = "denoise";
/// Official Hugging Face weight repository.
pub const UPSTREAM_HF: &str = "alibabasglab/FRCRN_SE_16K";
/// Immutable official weight revision audited on 2026-08-26.
pub const UPSTREAM_REVISION: &str = "3766e6a64b0d8cb58f08d913d617bf129f11ed53";
/// ClearerVoice-Studio source revision used for the native transcription.
pub const SOURCE_REVISION: &str = "6b3774dc79c46ae8bed2a4fa5f706f0ac8c75c61";
/// SHA-256 of official `last_best_checkpoint.pt` (161,053,751 bytes).
pub const CHECKPOINT_SHA256: &str =
    "b22256adbb91b68cf5a3db8f6657a4fb17066eecd5f069803e59c186c1cf3ebb";
/// Canonical name/shape manifest digest of the published 812-tensor GGUF.
pub const TENSOR_MANIFEST_SHA256: &str =
    "ca71dad1ae5293d3d63628b71127c0efdf004cec684e5a341ab376ce3e2851b7";
/// Pinned checkpoint byte length.
pub const CHECKPOINT_BYTES: u32 = 161_053_751;
/// Official checkpoint and source license.
pub const DEFAULT_LICENSE: &str = "apache-2.0";
/// Exact float tensor count after dropping BatchNorm counters.
pub const TENSOR_COUNT: usize = 812;

pub const SAMPLE_RATE: u32 = 16_000;
pub const WINDOW_LENGTH: u32 = 640;
pub const HOP_LENGTH: u32 = 320;
pub const FFT_LENGTH: u32 = 640;
pub const FEATURE_DIM: u32 = 321;
pub const MODEL_DEPTH: u32 = 14;
pub const CHANNELS: u32 = 128;
pub const FSMN_ORDER: u32 = 20;
pub const SE_HIDDEN: u32 = 16;
pub const UNET_COUNT: u32 = 2;

const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
const KEY_PROVENANCE_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";
pub const KEY_UPSTREAM_REVISION: &str = "vokra.frcrn.upstream_revision";
pub const KEY_SOURCE_REVISION: &str = "vokra.frcrn.source_revision";
pub const KEY_CHECKPOINT_SHA256: &str = "vokra.frcrn.checkpoint_sha256";
pub const KEY_CHECKPOINT_BYTES: &str = "vokra.frcrn.checkpoint_bytes";
pub const KEY_TENSOR_MANIFEST_SHA256: &str = "vokra.frcrn.tensor_manifest_sha256";
pub const KEY_SAMPLE_RATE: &str = "vokra.frcrn.sample_rate";
pub const KEY_WINDOW_LENGTH: &str = "vokra.frcrn.window_length";
pub const KEY_HOP_LENGTH: &str = "vokra.frcrn.hop_length";
pub const KEY_FFT_LENGTH: &str = "vokra.frcrn.fft_length";
pub const KEY_FEATURE_DIM: &str = "vokra.frcrn.feature_dim";
pub const KEY_MODEL_DEPTH: &str = "vokra.frcrn.model_depth";
pub const KEY_CHANNELS: &str = "vokra.frcrn.channels";
pub const KEY_FSMN_ORDER: &str = "vokra.frcrn.fsmn_order";
pub const KEY_SE_HIDDEN: &str = "vokra.frcrn.se_hidden";
pub const KEY_UNET_COUNT: &str = "vokra.frcrn.unet_count";
pub const KEY_WINDOW_TYPE: &str = "vokra.frcrn.window_type";
pub const KEY_COMPLEX: &str = "vokra.frcrn.complex";

const PROVENANCE_SOURCE: &str =
    "alibabasglab/FRCRN_SE_16K weights + modelscope/ClearerVoice-Studio native source (Apache-2.0)";

/// Successful conversion counters. Exact conversion always returns
/// `812 / 812 / 0 / 0`; the legacy fields remain API-compatible.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct FrcrnReport {
    /// Exact tensors read from the prepared checkpoint.
    pub read: usize,
    /// Exact F32 tensors written to GGUF.
    pub written: usize,
    /// Always zero for a successful strict conversion.
    pub skipped_non_float: usize,
    /// Always zero because the official checkpoint is F32.
    pub bf16_passthrough: usize,
}

/// Convert the exact official FRCRN-SE-16K prepared checkpoint to GGUF.
pub fn convert_frcrn_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<FrcrnReport, ConvertError> {
    let license = require_official_license(license)?;
    let bytes = std::fs::read(input).map_err(ConvertError::Io)?;
    let st = SafetensorsFile::parse(bytes)?;
    validate_input_manifest(&st)?;

    let mut builder = GgufBuilder::new();
    stamp_contract(&mut builder, license);
    for tensor in st.tensors() {
        builder
            .add_tensor(
                &tensor.name,
                GgmlType::F32,
                tensor.shape.clone(),
                st.tensor_bytes(tensor).to_vec(),
            )
            .map_err(|error| ConvertError::Gguf(error.to_string()))?;
    }
    let output_bytes = builder
        .to_bytes()
        .map_err(|error| ConvertError::Gguf(error.to_string()))?;
    std::fs::write(output, output_bytes).map_err(ConvertError::Io)?;

    Ok(FrcrnReport {
        read: TENSOR_COUNT,
        written: TENSOR_COUNT,
        skipped_non_float: 0,
        bf16_passthrough: 0,
    })
}

fn require_official_license(license: Option<&str>) -> Result<&'static str, ConvertError> {
    let requested = license.unwrap_or(DEFAULT_LICENSE).trim();
    if !requested.eq_ignore_ascii_case(DEFAULT_LICENSE) {
        return Err(parse_error(format!(
            "license override `{requested}` conflicts with the pinned official Apache-2.0 checkpoint"
        )));
    }
    Ok(DEFAULT_LICENSE)
}

fn stamp_contract(builder: &mut GgufBuilder, spdx: &str) {
    builder.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    builder.add_string(chunks::KEY_MODEL_NAME, NAME);
    builder.add_string(KEY_MODEL_CATEGORY, CATEGORY);
    builder.add_string(KEY_PROVENANCE_UPSTREAM_HF, UPSTREAM_HF);
    builder.add_string(KEY_UPSTREAM_REVISION, UPSTREAM_REVISION);
    builder.add_string(KEY_SOURCE_REVISION, SOURCE_REVISION);
    builder.add_string(KEY_CHECKPOINT_SHA256, CHECKPOINT_SHA256);
    builder.add_u32(KEY_CHECKPOINT_BYTES, CHECKPOINT_BYTES);
    builder.add_string(KEY_TENSOR_MANIFEST_SHA256, TENSOR_MANIFEST_SHA256);
    builder.add_u32(KEY_SAMPLE_RATE, SAMPLE_RATE);
    builder.add_u32(KEY_WINDOW_LENGTH, WINDOW_LENGTH);
    builder.add_u32(KEY_HOP_LENGTH, HOP_LENGTH);
    builder.add_u32(KEY_FFT_LENGTH, FFT_LENGTH);
    builder.add_u32(KEY_FEATURE_DIM, FEATURE_DIM);
    builder.add_u32(KEY_MODEL_DEPTH, MODEL_DEPTH);
    builder.add_u32(KEY_CHANNELS, CHANNELS);
    builder.add_u32(KEY_FSMN_ORDER, FSMN_ORDER);
    builder.add_u32(KEY_SE_HIDDEN, SE_HIDDEN);
    builder.add_u32(KEY_UNET_COUNT, UNET_COUNT);
    builder.add_string(KEY_WINDOW_TYPE, "hanning-sqrt-periodic");
    builder.add_bool(KEY_COMPLEX, true);
    vokra_core::stamp_provenance(
        builder,
        LicenseClass::Permissive,
        spdx,
        Some(NAME),
        Some(PROVENANCE_SOURCE),
    );
}

fn validate_input_manifest(st: &SafetensorsFile) -> Result<(), ConvertError> {
    let expected = tensor_manifest();
    if expected.len() != TENSOR_COUNT || st.tensors().len() != TENSOR_COUNT {
        return Err(parse_error(format!(
            "checkpoint has {} tensors, expected exactly {TENSOR_COUNT} from {UPSTREAM_HF}@{UPSTREAM_REVISION}",
            st.tensors().len()
        )));
    }
    let mut seen = BTreeSet::new();
    for tensor in st.tensors() {
        let shape = expected
            .get(&tensor.name)
            .ok_or_else(|| parse_error(format!("unexpected tensor `{}`", tensor.name)))?;
        if &tensor.shape != shape {
            return Err(parse_error(format!(
                "tensor `{}` has shape {:?}, expected {shape:?}",
                tensor.name, tensor.shape
            )));
        }
        if tensor.dtype != GgmlType::F32 {
            return Err(parse_error(format!(
                "tensor `{}` is {:?}, expected F32 in the pinned official checkpoint",
                tensor.name, tensor.dtype
            )));
        }
        if !seen.insert(tensor.name.as_str()) {
            return Err(parse_error(format!(
                "checkpoint repeats tensor `{}`",
                tensor.name
            )));
        }
    }
    if let Some(missing) = expected.keys().find(|name| !seen.contains(name.as_str())) {
        return Err(parse_error(format!(
            "checkpoint is missing tensor `{missing}`"
        )));
    }
    Ok(())
}

fn parse_error(message: impl Into<String>) -> ConvertError {
    ConvertError::Parse(format!("frcrn: {}", message.into()))
}

/// Exact public/official 812-tensor F32 name and shape contract.
pub fn tensor_manifest() -> BTreeMap<String, Vec<u64>> {
    let mut out = BTreeMap::new();
    insert(&mut out, "stft.weight", &[642, 1, 640]);
    insert(&mut out, "istft.weight", &[642, 1, 640]);
    insert(&mut out, "istft.window", &[1, 640, 1]);
    insert(&mut out, "istft.enframe", &[640, 1, 640]);
    add_unet(&mut out, "unet");
    add_unet(&mut out, "unet2");
    debug_assert_eq!(out.len(), TENSOR_COUNT);
    out
}

fn add_unet(out: &mut BTreeMap<String, Vec<u64>>, root: &str) {
    for layer in 0..7 {
        let in_channels = if layer == 0 { 1 } else { 128 };
        let kernel_h = if layer == 6 { 2 } else { 5 };
        let prefix = format!("{root}.encoder{layer}");
        add_complex_conv(
            out,
            &format!("{prefix}.conv"),
            "conv",
            &[128, in_channels, kernel_h, 2],
            128,
        );
        add_complex_batch_norm(out, &format!("{prefix}.bn"), 128);
    }

    let decoder_geometry: &[(u64, u64, u64)] = &[
        (128, 128, 2),
        (256, 128, 5),
        (256, 128, 5),
        (256, 128, 5),
        (256, 128, 6),
        (256, 128, 5),
        (256, 1, 5),
    ];
    for (layer, &(in_channels, out_channels, kernel_h)) in decoder_geometry.iter().enumerate() {
        let prefix = format!("{root}.decoder{layer}");
        add_complex_conv(
            out,
            &format!("{prefix}.transconv"),
            "tconv",
            &[in_channels, out_channels, kernel_h, 2],
            out_channels,
        );
        add_complex_batch_norm(out, &format!("{prefix}.bn"), out_channels);
    }

    add_central_fsmn(out, &format!("{root}.fsmn"));
    for layer in 0..7 {
        add_l1_fsmn(out, &format!("{root}.fsmn_enc{layer}"));
        add_l1_fsmn(out, &format!("{root}.fsmn_dec{layer}"));
        add_se(out, &format!("{root}.se_layer_enc{layer}"));
        if layer < 6 {
            add_se(out, &format!("{root}.se_layer_dec{layer}"));
        }
    }
    add_complex_conv(out, &format!("{root}.linear"), "conv", &[1, 1, 1, 1], 1);
}

fn add_complex_conv(
    out: &mut BTreeMap<String, Vec<u64>>,
    prefix: &str,
    stem: &str,
    weight_shape: &[u64],
    bias: u64,
) {
    for component in ["re", "im"] {
        insert(
            out,
            format!("{prefix}.{stem}_{component}.weight"),
            weight_shape,
        );
        insert(out, format!("{prefix}.{stem}_{component}.bias"), &[bias]);
    }
}

fn add_complex_batch_norm(out: &mut BTreeMap<String, Vec<u64>>, prefix: &str, channels: u64) {
    for component in ["re", "im"] {
        for field in ["weight", "bias", "running_mean", "running_var"] {
            insert(out, format!("{prefix}.bn_{component}.{field}"), &[channels]);
        }
    }
}

fn add_central_fsmn(out: &mut BTreeMap<String, Vec<u64>>, prefix: &str) {
    for component in ["re", "im"] {
        for level in ["L1", "L2"] {
            add_real_fsmn(out, &format!("{prefix}.fsmn_{component}_{level}"));
        }
    }
}

fn add_l1_fsmn(out: &mut BTreeMap<String, Vec<u64>>, prefix: &str) {
    for component in ["re", "im"] {
        add_real_fsmn(out, &format!("{prefix}.fsmn_{component}_L1"));
    }
}

fn add_real_fsmn(out: &mut BTreeMap<String, Vec<u64>>, prefix: &str) {
    insert(out, format!("{prefix}.linear.weight"), &[128, 128]);
    insert(out, format!("{prefix}.linear.bias"), &[128]);
    insert(out, format!("{prefix}.project.weight"), &[128, 128]);
    insert(out, format!("{prefix}.conv1.weight"), &[128, 1, 20, 1]);
}

fn add_se(out: &mut BTreeMap<String, Vec<u64>>, prefix: &str) {
    for component in ["r", "i"] {
        insert(out, format!("{prefix}.fc_{component}.0.weight"), &[16, 128]);
        insert(out, format!("{prefix}.fc_{component}.0.bias"), &[16]);
        insert(out, format!("{prefix}.fc_{component}.2.weight"), &[128, 16]);
        insert(out, format!("{prefix}.fc_{component}.2.bias"), &[128]);
    }
}

fn insert(out: &mut BTreeMap<String, Vec<u64>>, name: impl Into<String>, shape: &[u64]) {
    let name = name.into();
    assert!(
        out.insert(name.clone(), shape.to_vec()).is_none(),
        "duplicate FRCRN manifest entry {name}"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::GgufFile;

    fn scratch(tag: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "vokra-frcrn-{tag}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }

    fn one_tensor_safetensors(dtype: &str, name: &str, shape: &[u64]) -> Vec<u8> {
        let width = match dtype {
            "F32" => 4,
            "F16" | "BF16" => 2,
            _ => panic!("unsupported fixture dtype"),
        };
        let bytes = shape.iter().product::<u64>() as usize * width;
        let dimensions = shape
            .iter()
            .map(u64::to_string)
            .collect::<Vec<_>>()
            .join(",");
        let header = format!(
            "{{\"{name}\":{{\"dtype\":\"{dtype}\",\"shape\":[{dimensions}],\"data_offsets\":[0,{bytes}]}}}}"
        );
        let mut out = Vec::with_capacity(8 + header.len() + bytes);
        out.extend_from_slice(&(header.len() as u64).to_le_bytes());
        out.extend_from_slice(header.as_bytes());
        out.resize(8 + header.len() + bytes, 0);
        out
    }

    #[test]
    fn exact_manifest_has_public_tensor_count_and_shapes() {
        let manifest = tensor_manifest();
        assert_eq!(manifest.len(), 812);
        assert_eq!(manifest["stft.weight"], vec![642, 1, 640]);
        assert_eq!(
            manifest["unet.encoder0.conv.conv_re.weight"],
            vec![128, 1, 5, 2]
        );
        assert_eq!(
            manifest["unet2.decoder4.transconv.tconv_im.weight"],
            vec![256, 128, 6, 2]
        );
        assert_eq!(
            manifest["unet2.fsmn.fsmn_re_L2.conv1.weight"],
            vec![128, 1, 20, 1]
        );
        assert_eq!(manifest["unet.se_layer_dec5.fc_i.2.weight"], vec![128, 16]);
        assert!(!manifest.contains_key("unet.se_layer_dec6.fc_i.0.weight"));
    }

    #[test]
    fn incomplete_or_non_f32_checkpoint_is_refused_without_output() {
        for (tag, dtype) in [("incomplete", "F32"), ("bf16", "BF16")] {
            let input = scratch(&format!("{tag}.safetensors"));
            let output = scratch(&format!("{tag}.gguf"));
            std::fs::write(
                &input,
                one_tensor_safetensors(dtype, "stft.weight", &[642, 1, 640]),
            )
            .unwrap();
            let error = convert_frcrn_file(&input, &output, None).unwrap_err();
            assert!(error.to_string().contains("expected exactly 812"));
            assert!(!output.exists());
            std::fs::remove_file(input).ok();
        }
    }

    #[test]
    fn conflicting_license_override_is_refused_before_input_read() {
        let error = convert_frcrn_file(
            Path::new("does-not-exist.safetensors"),
            Path::new("does-not-exist.gguf"),
            Some("mit"),
        )
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("conflicts with the pinned official")
        );
    }

    #[test]
    fn contract_stamps_pinned_provenance_and_topology() {
        let path = scratch("contract.gguf");
        let mut builder = GgufBuilder::new();
        stamp_contract(&mut builder, DEFAULT_LICENSE);
        std::fs::write(&path, builder.to_bytes().unwrap()).unwrap();
        let file = GgufFile::open(&path).unwrap();
        assert_eq!(
            file.get(chunks::KEY_MODEL_ARCH).and_then(|v| v.as_str()),
            Some(ARCH)
        );
        assert_eq!(
            file.get(KEY_PROVENANCE_UPSTREAM_HF)
                .and_then(|v| v.as_str()),
            Some(UPSTREAM_HF)
        );
        assert_eq!(
            file.get(KEY_CHECKPOINT_SHA256).and_then(|v| v.as_str()),
            Some(CHECKPOINT_SHA256)
        );
        assert_eq!(
            file.get(KEY_CHECKPOINT_BYTES).and_then(|v| v.as_u64()),
            Some(u64::from(CHECKPOINT_BYTES))
        );
        assert_eq!(
            file.get(KEY_SAMPLE_RATE).and_then(|v| v.as_u64()),
            Some(u64::from(SAMPLE_RATE))
        );
        assert_eq!(file.get(KEY_COMPLEX).and_then(|v| v.as_bool()), Some(true));
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(|v| v.as_str()),
            Some(LicenseClass::Permissive.as_str())
        );
        std::fs::remove_file(path).ok();
    }
}
