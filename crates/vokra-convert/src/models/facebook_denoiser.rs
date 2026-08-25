//! Exact converter for Meta's causal DNS48 Facebook Denoiser checkpoint.
//!
//! The audited release is `facebookresearch/denoiser::pretrained.dns48` at
//! source revision [`SOURCE_REVISION`]. It contains exactly 48 F32 tensors:
//! five waveform encoder blocks, a two-layer unidirectional LSTM, and five
//! symmetric decoder blocks. Missing, extra, renamed, reshaped, or non-F32
//! tensors are rejected before an output file is written.
//!
//! The upstream `.th` pickle is decoded only by the offline Python sidecar;
//! neither this converter nor the runtime executes pickle or ONNX.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use vokra_core::LicenseClass;
use vokra_core::gguf::{GgmlType, GgufBuilder, chunks};

use crate::ConvertError;
use crate::safetensors::SafetensorsFile;

pub const ARCH: &str = "facebook_denoiser";
pub const NAME: &str = "facebook_denoiser";
pub const CATEGORY: &str = "enhancement";
pub const UPSTREAM_URL: &str = "github.com/facebookresearch/denoiser";
pub const SOURCE_REVISION: &str = "8afd7c166699bb3c8b2d95b6dd706f71e1075df0";
pub const CHECKPOINT_URL: &str =
    "https://dl.fbaipublicfiles.com/adiyoss/denoiser/dns48-11decc9d8e3f0998.th";
pub const CHECKPOINT_BYTES: u64 = 75_478_395;
/// The official filename embeds PyTorch's checked SHA-256 prefix. The full
/// digest is intentionally not fabricated; the VAST reference run records it.
pub const CHECKPOINT_SHA256_PREFIX: &str = "11decc9d8e3f0998";
pub const SOURCE_DEMUCS_SHA256: &str =
    "8e9c21935c647e24f31cefcc63a298cb2a1c25bc99aab44bbe63a7b5570836be";
pub const SOURCE_RESAMPLE_SHA256: &str =
    "3e8ea258036660b7d33415794fe09ee010510f4d760bdfc5d5de268d6efb40f5";
pub const SOURCE_PRETRAINED_SHA256: &str =
    "885ad1ddd6cee5d4ecf5b4bc32784ceee97dc37ae19570b7ce0f9869b360d108";
pub const SOURCE_LICENSE_SHA256: &str =
    "336255dc30193e8e15d689d9481bb05673d89055718f3a96923a7ffb99adbbaf";
pub const PUBLIC_HF: &str = "vokra/facebook-denoiser";
pub const PUBLIC_REVISION: &str = "f50187791c52af3a90e479fcbacba3f267702eaa";
pub const PUBLIC_GGUF_SHA256: &str =
    "c0b23707a2f255b5eb108c5b08b92f310fede6870106e799b195282d6a375e74";
pub const MANIFEST_SHA256: &str =
    "bd25704cddfa2acd15f57f4ebb27d6c9a3c22f08121c7335287cbf6af4602ff1";
pub const DEFAULT_LICENSE_SPDX: &str = "cc-by-nc-4.0";

pub const TENSOR_COUNT: usize = 48;
pub const SAMPLE_RATE: u32 = 16_000;
pub const HIDDEN: u32 = 48;
pub const DEPTH: u32 = 5;
pub const KERNEL_SIZE: u32 = 8;
pub const STRIDE: u32 = 4;
pub const RESAMPLE: u32 = 4;
pub const GROWTH: u32 = 2;
pub const MAX_HIDDEN: u32 = 10_000;
pub const RESAMPLE_ZEROS: u32 = 56;
pub const NORMALIZATION_FLOOR: f32 = 1.0e-3;

const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
const KEY_PROVENANCE_UPSTREAM_URL: &str = "vokra.provenance.upstream_url";
pub const KEY_SOURCE_REVISION: &str = "vokra.facebook_denoiser.source_revision";
pub const KEY_CHECKPOINT_URL: &str = "vokra.facebook_denoiser.checkpoint_url";
pub const KEY_CHECKPOINT_BYTES: &str = "vokra.facebook_denoiser.checkpoint_bytes";
pub const KEY_CHECKPOINT_SHA256_PREFIX: &str = "vokra.facebook_denoiser.checkpoint_sha256_prefix";
pub const KEY_SOURCE_DEMUCS_SHA256: &str = "vokra.facebook_denoiser.source_demucs_sha256";
pub const KEY_SOURCE_RESAMPLE_SHA256: &str = "vokra.facebook_denoiser.source_resample_sha256";
pub const KEY_SOURCE_PRETRAINED_SHA256: &str = "vokra.facebook_denoiser.source_pretrained_sha256";
pub const KEY_SOURCE_LICENSE_SHA256: &str = "vokra.facebook_denoiser.source_license_sha256";
pub const KEY_PUBLIC_HF: &str = "vokra.facebook_denoiser.public_hf";
pub const KEY_PUBLIC_REVISION: &str = "vokra.facebook_denoiser.public_revision";
pub const KEY_PUBLIC_GGUF_SHA256: &str = "vokra.facebook_denoiser.public_gguf_sha256";
pub const KEY_MANIFEST_SHA256: &str = "vokra.facebook_denoiser.manifest_sha256";
pub const KEY_SAMPLE_RATE: &str = "vokra.facebook_denoiser.sample_rate";
pub const KEY_HIDDEN: &str = "vokra.facebook_denoiser.hidden";
pub const KEY_DEPTH: &str = "vokra.facebook_denoiser.depth";
pub const KEY_KERNEL_SIZE: &str = "vokra.facebook_denoiser.kernel_size";
pub const KEY_STRIDE: &str = "vokra.facebook_denoiser.stride";
pub const KEY_RESAMPLE: &str = "vokra.facebook_denoiser.resample";
pub const KEY_GROWTH: &str = "vokra.facebook_denoiser.growth";
pub const KEY_MAX_HIDDEN: &str = "vokra.facebook_denoiser.max_hidden";
pub const KEY_RESAMPLE_ZEROS: &str = "vokra.facebook_denoiser.resample_zeros";
pub const KEY_NORMALIZATION_FLOOR: &str = "vokra.facebook_denoiser.normalization_floor";
pub const KEY_NORMALIZE: &str = "vokra.facebook_denoiser.normalize";
pub const KEY_GLU: &str = "vokra.facebook_denoiser.glu";
pub const KEY_CAUSAL: &str = "vokra.facebook_denoiser.causal";
pub const KEY_STD_CORRECTION: &str = "vokra.facebook_denoiser.std_correction";

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
/// Auditable counters for one strict DNS48 conversion.
pub struct FacebookDenoiserReport {
    /// Input tensors inspected after parsing.
    pub read: usize,
    /// Exact F32 DNS48 tensors written to GGUF.
    pub written: usize,
    /// Always zero for a successful strict conversion.
    pub skipped_non_float: usize,
    /// Always zero because DNS48 is pinned to F32.
    pub bf16_passthrough: usize,
}

/// Converts an offline-prepared DNS48 safetensors file into a strict GGUF.
pub fn convert_facebook_denoiser_file(
    input: &Path,
    output: &Path,
    license: Option<&str>,
) -> Result<FacebookDenoiserReport, ConvertError> {
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
    Ok(FacebookDenoiserReport {
        read: TENSOR_COUNT,
        written: TENSOR_COUNT,
        skipped_non_float: 0,
        bf16_passthrough: 0,
    })
}

fn require_official_license(license: Option<&str>) -> Result<&'static str, ConvertError> {
    let requested = license.unwrap_or(DEFAULT_LICENSE_SPDX).trim();
    if !requested.eq_ignore_ascii_case(DEFAULT_LICENSE_SPDX) {
        return Err(parse_error(format!(
            "license override {requested:?} conflicts with the pinned CC-BY-NC-4.0 DNS48 checkpoint"
        )));
    }
    Ok(DEFAULT_LICENSE_SPDX)
}

fn stamp_contract(builder: &mut GgufBuilder, spdx: &str) {
    builder.add_string(chunks::KEY_MODEL_ARCH, ARCH);
    builder.add_string(chunks::KEY_MODEL_NAME, NAME);
    builder.add_string(KEY_MODEL_CATEGORY, CATEGORY);
    builder.add_string(KEY_PROVENANCE_UPSTREAM_URL, UPSTREAM_URL);
    builder.add_string(KEY_SOURCE_REVISION, SOURCE_REVISION);
    builder.add_string(KEY_CHECKPOINT_URL, CHECKPOINT_URL);
    builder.add_u32(
        KEY_CHECKPOINT_BYTES,
        u32::try_from(CHECKPOINT_BYTES).expect("DNS48 checkpoint size fits u32"),
    );
    builder.add_string(KEY_CHECKPOINT_SHA256_PREFIX, CHECKPOINT_SHA256_PREFIX);
    builder.add_string(KEY_SOURCE_DEMUCS_SHA256, SOURCE_DEMUCS_SHA256);
    builder.add_string(KEY_SOURCE_RESAMPLE_SHA256, SOURCE_RESAMPLE_SHA256);
    builder.add_string(KEY_SOURCE_PRETRAINED_SHA256, SOURCE_PRETRAINED_SHA256);
    builder.add_string(KEY_SOURCE_LICENSE_SHA256, SOURCE_LICENSE_SHA256);
    builder.add_string(KEY_PUBLIC_HF, PUBLIC_HF);
    builder.add_string(KEY_PUBLIC_REVISION, PUBLIC_REVISION);
    builder.add_string(KEY_PUBLIC_GGUF_SHA256, PUBLIC_GGUF_SHA256);
    builder.add_string(KEY_MANIFEST_SHA256, MANIFEST_SHA256);
    builder.add_u32(KEY_SAMPLE_RATE, SAMPLE_RATE);
    builder.add_u32(KEY_HIDDEN, HIDDEN);
    builder.add_u32(KEY_DEPTH, DEPTH);
    builder.add_u32(KEY_KERNEL_SIZE, KERNEL_SIZE);
    builder.add_u32(KEY_STRIDE, STRIDE);
    builder.add_u32(KEY_RESAMPLE, RESAMPLE);
    builder.add_u32(KEY_GROWTH, GROWTH);
    builder.add_u32(KEY_MAX_HIDDEN, MAX_HIDDEN);
    builder.add_u32(KEY_RESAMPLE_ZEROS, RESAMPLE_ZEROS);
    builder.add_f32(KEY_NORMALIZATION_FLOOR, NORMALIZATION_FLOOR);
    builder.add_bool(KEY_NORMALIZE, true);
    builder.add_bool(KEY_GLU, true);
    builder.add_bool(KEY_CAUSAL, true);
    builder.add_u32(KEY_STD_CORRECTION, 1);
    vokra_core::stamp_provenance(
        builder,
        LicenseClass::NonCommercial,
        spdx,
        Some(NAME),
        Some(
            "facebookresearch/denoiser DNS48 causal waveform U-Net + LSTM; CC-BY-NC-4.0 research-only",
        ),
    );
}

fn validate_input_manifest(tensors: &SafetensorsFile) -> Result<(), ConvertError> {
    let expected = expected_manifest();
    if expected.len() != TENSOR_COUNT || tensors.tensors().len() != TENSOR_COUNT {
        return Err(parse_error(format!(
            "checkpoint has {} tensors, expected exactly {TENSOR_COUNT} for DNS48",
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
                "tensor {} is {:?}, expected F32 for DNS48",
                tensor.name, tensor.dtype
            )));
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
    ConvertError::Parse(format!("facebook_denoiser: {}", message.into()))
}

pub(crate) fn expected_manifest() -> BTreeMap<String, Vec<u64>> {
    let mut manifest = BTreeMap::new();
    let channels = [48u64, 96, 192, 384, 768];
    let mut input = 1u64;
    for (stage, &output) in channels.iter().enumerate() {
        insert(
            &mut manifest,
            format!("encoder.{stage}.0.weight"),
            &[output, input, 8],
        );
        insert(&mut manifest, format!("encoder.{stage}.0.bias"), &[output]);
        insert(
            &mut manifest,
            format!("encoder.{stage}.2.weight"),
            &[2 * output, output, 1],
        );
        insert(
            &mut manifest,
            format!("encoder.{stage}.2.bias"),
            &[2 * output],
        );
        input = output;
    }

    let decoder_inputs = [768u64, 384, 192, 96, 48];
    let decoder_outputs = [384u64, 192, 96, 48, 1];
    for (stage, (&input, &output)) in decoder_inputs
        .iter()
        .zip(decoder_outputs.iter())
        .enumerate()
    {
        insert(
            &mut manifest,
            format!("decoder.{stage}.0.weight"),
            &[2 * input, input, 1],
        );
        insert(
            &mut manifest,
            format!("decoder.{stage}.0.bias"),
            &[2 * input],
        );
        insert(
            &mut manifest,
            format!("decoder.{stage}.2.weight"),
            &[input, output, 8],
        );
        insert(&mut manifest, format!("decoder.{stage}.2.bias"), &[output]);
    }

    for layer in 0..2 {
        for kind in ["bias_hh", "bias_ih"] {
            insert(&mut manifest, format!("lstm.lstm.{kind}_l{layer}"), &[3072]);
        }
        for kind in ["weight_hh", "weight_ih"] {
            insert(
                &mut manifest,
                format!("lstm.lstm.{kind}_l{layer}"),
                &[3072, 768],
            );
        }
    }
    debug_assert_eq!(manifest.len(), TENSOR_COUNT);
    manifest
}

fn insert(manifest: &mut BTreeMap<String, Vec<u64>>, name: String, shape: &[u64]) {
    assert!(
        manifest.insert(name.clone(), shape.to_vec()).is_none(),
        "duplicate tensor {name}"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::{GgufFile, GgufMetadataValue};

    #[test]
    fn exact_dns48_manifest_is_complete() {
        let manifest = expected_manifest();
        assert_eq!(manifest.len(), TENSOR_COUNT);
        assert_eq!(manifest["encoder.4.2.weight"], vec![1536, 768, 1]);
        assert_eq!(manifest["decoder.0.2.weight"], vec![768, 384, 8]);
        assert_eq!(manifest["decoder.4.2.weight"], vec![48, 1, 8]);
        assert_eq!(manifest["lstm.lstm.weight_hh_l1"], vec![3072, 768]);
    }

    #[test]
    fn conflicting_license_override_is_rejected() {
        assert_eq!(
            require_official_license(Some("CC-BY-NC-4.0")).unwrap(),
            DEFAULT_LICENSE_SPDX
        );
        let error = require_official_license(Some("mit")).unwrap_err();
        assert!(error.to_string().contains("conflicts"));
    }

    #[test]
    fn additive_contract_pins_source_topology_and_public_artifact() {
        let mut builder = GgufBuilder::new();
        stamp_contract(&mut builder, DEFAULT_LICENSE_SPDX);
        let file = GgufFile::parse(builder.to_bytes().unwrap()).unwrap();
        assert_eq!(
            file.get(KEY_SOURCE_REVISION)
                .and_then(GgufMetadataValue::as_str),
            Some(SOURCE_REVISION)
        );
        assert_eq!(
            file.get(KEY_MANIFEST_SHA256)
                .and_then(GgufMetadataValue::as_str),
            Some(MANIFEST_SHA256)
        );
        assert_eq!(
            file.get(KEY_CAUSAL).and_then(GgufMetadataValue::as_bool),
            Some(true)
        );
        assert_eq!(
            file.get(chunks::KEY_PROVENANCE_WEIGHT_LICENSE)
                .and_then(GgufMetadataValue::as_str),
            Some(LicenseClass::NonCommercial.as_str())
        );
    }
}
