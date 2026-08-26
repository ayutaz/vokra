//! Native YuE-upsampler feature-to-waveform decoder for Mac CPU and Metal.
//!
//! This is the exact 81-tensor `m-a-p/YuE-upsampler` 151k release: a plain
//! Vocos ConvNeXt-1D backbone followed by the released same-padded iSTFT head.
//! The runtime accepts already-computed channel-major `[1024, frames]` YuE
//! codec features and produces mono 44.1 kHz PCM. Learned operations use the
//! selected backend; a Metal request never falls back to scalar CPU inference.

use std::sync::Arc;

use vokra_core::backend::BackendKind;
use vokra_core::gguf::{GgmlType, GgufFile, GgufMetadataValue, chunks};
use vokra_core::{LicenseClass, Result, VokraError};
use vokra_ops::{
    VocosAttrs, VocosBlockWeights, VocosIstftPadding, VocosNormWeights, VocosWeights, vocos_decode,
};

use crate::compute::{Compute, HotOp};
use crate::strict_checkpoint::{StrictCheckpoint, StrictCheckpointSpec, load_tensor};
use crate::vocos::decode_weights_with_compute;

pub const ARCH: &str = "yue_upsampler";
pub const NAME: &str = "yue-upsampler";
pub const CATEGORY: &str = "vocoder";
pub const VARIANT: &str = "upsampler";
pub const UPSTREAM_HF: &str = "m-a-p/YuE-upsampler";
pub const UPSTREAM_REVISION: &str = "c6d7494a60555672be09ca809a40be400d682a53";
pub const CHECKPOINT_FILE: &str = "decoder_151000.pth";
pub const CHECKPOINT_SHA256: &str =
    "8af97a29d3483f9d4a3755992837501bd7d6caa1a69382ed16e64039e0ea0998";
pub const CHECKPOINT_BYTES: u64 = 72_610_550;
pub const SOURCE_PACKAGE: &str = "vocos==0.1.0";
pub const SOURCE_PACKAGE_SHA256: &str =
    "0ac13eaef68596074301e912d781399b3defa4b4ca60b6bc52c8a4b9209ca235";
pub const MANIFEST_SHA256: &str =
    "c8b3f2a4de49f9d4ed1819a57e8850439b66578112de5fd94595c3e53c58956e";
pub const PUBLIC_REVISION: &str = "6eea19bd301c5214123ee69217a61a989ffe80d0";
pub const PUBLIC_GGUF_SHA256: &str =
    "17df9c667c931544cf84545266d07e3598a9528d751ca6f281fffd305f4409ff";
pub const PUBLIC_GGUF_BYTES: u64 = 72_531_456;
pub const TENSOR_COUNT: usize = 81;
pub const SAMPLE_RATE: u32 = 44_100;
pub const INPUT_CHANNELS: usize = 1_024;
pub const DIM: usize = 512;
pub const INTERMEDIATE_DIM: usize = 1_536;
pub const NUM_LAYERS: usize = 8;
pub const N_FFT: usize = 3_528;
pub const HOP_LENGTH: usize = 882;

const LABEL: &str = "yue-upsampler";
const KEY_CATEGORY: &str = "vokra.model.category";
const KEY_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";
const KEY_VARIANT: &str = "vokra.yue_bundle.variant";
const KEY_UPSTREAM_REVISION: &str = "vokra.yue_upsampler.upstream_revision";
const KEY_CHECKPOINT_FILE: &str = "vokra.yue_upsampler.checkpoint_file";
const KEY_CHECKPOINT_SHA256: &str = "vokra.yue_upsampler.checkpoint_sha256";
const KEY_CHECKPOINT_BYTES: &str = "vokra.yue_upsampler.checkpoint_bytes";
const KEY_SOURCE_PACKAGE: &str = "vokra.yue_upsampler.source_package";
const KEY_SOURCE_PACKAGE_SHA256: &str = "vokra.yue_upsampler.source_package_sha256";
const KEY_MANIFEST_SHA256: &str = "vokra.yue_upsampler.tensor_manifest_sha256";
const KEY_PUBLIC_REVISION: &str = "vokra.yue_upsampler.public_revision";
const KEY_PUBLIC_GGUF_SHA256: &str = "vokra.yue_upsampler.public_gguf_sha256";
const KEY_PUBLIC_GGUF_BYTES: &str = "vokra.yue_upsampler.public_gguf_bytes";
const KEY_SAMPLE_RATE: &str = "vokra.yue_upsampler.sample_rate";
const KEY_INPUT_CHANNELS: &str = "vokra.yue_upsampler.input_channels";
const KEY_DIM: &str = "vokra.yue_upsampler.dim";
const KEY_INTERMEDIATE_DIM: &str = "vokra.yue_upsampler.intermediate_dim";
const KEY_NUM_LAYERS: &str = "vokra.yue_upsampler.num_layers";
const KEY_N_FFT: &str = "vokra.yue_upsampler.n_fft";
const KEY_HOP_LENGTH: &str = "vokra.yue_upsampler.hop_length";
const KEY_PADDING: &str = "vokra.yue_upsampler.padding";

const ADDITIVE_KEYS: &[&str] = &[
    KEY_UPSTREAM_REVISION,
    KEY_CHECKPOINT_FILE,
    KEY_CHECKPOINT_SHA256,
    KEY_CHECKPOINT_BYTES,
    KEY_SOURCE_PACKAGE,
    KEY_SOURCE_PACKAGE_SHA256,
    KEY_MANIFEST_SHA256,
    KEY_PUBLIC_REVISION,
    KEY_PUBLIC_GGUF_SHA256,
    KEY_PUBLIC_GGUF_BYTES,
    KEY_SAMPLE_RATE,
    KEY_INPUT_CHANNELS,
    KEY_DIM,
    KEY_INTERMEDIATE_DIM,
    KEY_NUM_LAYERS,
    KEY_N_FFT,
    KEY_HOP_LENGTH,
    KEY_PADDING,
];

const SPEC: StrictCheckpointSpec = StrictCheckpointSpec {
    label: LABEL,
    arch: ARCH,
    model_name: NAME,
    model_name_alias: None,
    tensor_count: TENSOR_COUNT,
    manifest_sha256: [
        0xc8, 0xb3, 0xf2, 0xa4, 0xde, 0x49, 0xf9, 0xd4, 0xed, 0x18, 0x19, 0xa5, 0x7e, 0x88, 0x50,
        0x43, 0x9b, 0x66, 0x57, 0x81, 0x12, 0xde, 0x5f, 0xd9, 0x45, 0x95, 0xc3, 0xe5, 0x3c, 0x58,
        0x95, 0x6e,
    ],
};

/// Every trained YuE-upsampler reduction required for a full Metal forward.
pub const YUE_UPSAMPLER_HOT_OPS: &[HotOp] = &[
    HotOp::Conv1d,
    HotOp::GroupedConv1d,
    HotOp::LayerNorm,
    HotOp::Gelu,
];

pub(crate) fn attrs() -> VocosAttrs {
    VocosAttrs {
        input_channels: INPUT_CHANNELS,
        dim: DIM,
        intermediate_dim: INTERMEDIATE_DIM,
        num_layers: NUM_LAYERS,
        num_conditions: 0,
        n_fft: N_FFT,
        hop_length: HOP_LENGTH,
        padding: VocosIstftPadding::Same,
    }
}

/// Strict real-weight YuE feature decoder.
#[derive(Debug, Clone)]
pub struct YueUpsampler {
    weights: Arc<VocosWeights>,
    weight_license: LicenseClass,
    backend: BackendKind,
}

impl YueUpsampler {
    /// Binds the exact historical public artifact or a fully stamped rebuild.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        let checkpoint = StrictCheckpoint::bind(file, SPEC)?;
        require_string(
            file,
            chunks::KEY_PROVENANCE_MODEL_ID,
            checkpoint.model_name(),
        )?;
        require_string(file, KEY_CATEGORY, CATEGORY)?;
        require_string(file, KEY_VARIANT, VARIANT)?;
        require_string(file, KEY_UPSTREAM_HF, UPSTREAM_HF)?;
        require_string(file, chunks::KEY_PROVENANCE_LICENSE, "apache-2.0")?;
        require_string(
            file,
            chunks::KEY_PROVENANCE_WEIGHT_LICENSE,
            LicenseClass::Permissive.as_str(),
        )?;
        validate_f32_manifest(file)?;
        validate_additive_contract(file)?;
        let weights = load_weights(file)?;
        weights.validate(&attrs()).map_err(|error| {
            VokraError::ModelLoad(format!("{LABEL}: weight validation failed: {error}"))
        })?;
        Ok(Self {
            weights: Arc::new(weights),
            weight_license: checkpoint.weight_license(),
            backend: BackendKind::Cpu,
        })
    }

    /// Opens and strictly binds a GGUF from disk using the CPU backend.
    pub fn open(path: impl AsRef<std::path::Path>) -> Result<Self> {
        Self::from_gguf(&GgufFile::open(path)?)
    }

    /// Binds a checkpoint and preflights the complete backend op set.
    pub fn from_gguf_with_backend(file: &GgufFile, backend: BackendKind) -> Result<Self> {
        Compute::for_backend(backend, YUE_UPSAMPLER_HOT_OPS)?;
        Ok(Self::from_gguf(file)?.with_backend(backend))
    }

    /// Selects one backend for every learned operation.
    #[must_use]
    pub fn with_backend(mut self, backend: BackendKind) -> Self {
        self.backend = backend;
        self
    }

    #[must_use]
    pub const fn backend(&self) -> BackendKind {
        self.backend
    }

    #[must_use]
    pub const fn sample_rate(&self) -> u32 {
        SAMPLE_RATE
    }

    #[must_use]
    pub const fn input_channels(&self) -> usize {
        INPUT_CHANNELS
    }

    #[must_use]
    pub const fn tensor_count(&self) -> usize {
        TENSOR_COUNT
    }

    #[must_use]
    pub const fn weight_license(&self) -> LicenseClass {
        self.weight_license
    }

    /// Decodes channel-major `[1024, frames]` YuE codec features to PCM.
    pub fn decode(&self, features: &[f32], frames: usize) -> Result<Vec<f32>> {
        if let Some((index, _)) = features
            .iter()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(VokraError::InvalidArgument(format!(
                "{LABEL}: feature value {index} is not finite"
            )));
        }
        let attrs = attrs();
        if self.backend == BackendKind::Cpu {
            return vocos_decode(features, frames, None, &self.weights, &attrs);
        }
        let compute = Compute::for_backend(self.backend, YUE_UPSAMPLER_HOT_OPS)?;
        decode_weights_with_compute(features, frames, &self.weights, &attrs, &compute)
    }
}

fn load_weights(file: &GgufFile) -> Result<VocosWeights> {
    load_prefixed_weights(file, "", LABEL)
}

/// Loads the same authenticated YuE Vocos topology from a composite GGUF.
///
/// `vokra/yue-xcodec-mini` embeds the byte-identical 151k decoder under the
/// `decoder.` namespace. Keeping one loader prevents the standalone and
/// composite routes from drifting while still preserving their distinct
/// checkpoint identities.
pub(crate) fn load_prefixed_weights(
    file: &GgufFile,
    prefix: &str,
    label: &str,
) -> Result<VocosWeights> {
    let name = |suffix: &str| format!("{prefix}{suffix}");
    let norm = |prefix: &str| -> Result<VocosNormWeights> {
        Ok(VocosNormWeights {
            scale: load_tensor(file, label, &name(&format!("{prefix}.weight")), &[DIM])?,
            shift: load_tensor(file, label, &name(&format!("{prefix}.bias")), &[DIM])?,
        })
    };
    let mut blocks = Vec::with_capacity(NUM_LAYERS);
    for index in 0..NUM_LAYERS {
        let prefix = format!("backbone.convnext.{index}");
        blocks.push(VocosBlockWeights {
            depthwise_weight: load_tensor(
                file,
                label,
                &name(&format!("{prefix}.dwconv.weight")),
                &[DIM, 1, 7],
            )?,
            depthwise_bias: load_tensor(
                file,
                label,
                &name(&format!("{prefix}.dwconv.bias")),
                &[DIM],
            )?,
            norm: norm(&format!("{prefix}.norm"))?,
            pointwise1_weight: load_tensor(
                file,
                label,
                &name(&format!("{prefix}.pwconv1.weight")),
                &[INTERMEDIATE_DIM, DIM],
            )?,
            pointwise1_bias: load_tensor(
                file,
                label,
                &name(&format!("{prefix}.pwconv1.bias")),
                &[INTERMEDIATE_DIM],
            )?,
            pointwise2_weight: load_tensor(
                file,
                label,
                &name(&format!("{prefix}.pwconv2.weight")),
                &[DIM, INTERMEDIATE_DIM],
            )?,
            pointwise2_bias: load_tensor(
                file,
                label,
                &name(&format!("{prefix}.pwconv2.bias")),
                &[DIM],
            )?,
            gamma: load_tensor(file, label, &name(&format!("{prefix}.gamma")), &[DIM])?,
        });
    }
    let weights = VocosWeights {
        embed_weight: load_tensor(
            file,
            label,
            &name("backbone.embed.weight"),
            &[DIM, INPUT_CHANNELS, 7],
        )?,
        embed_bias: load_tensor(file, label, &name("backbone.embed.bias"), &[DIM])?,
        norm: norm("backbone.norm")?,
        blocks,
        final_norm_weight: load_tensor(
            file,
            label,
            &name("backbone.final_layer_norm.weight"),
            &[DIM],
        )?,
        final_norm_bias: load_tensor(file, label, &name("backbone.final_layer_norm.bias"), &[DIM])?,
        head_weight: load_tensor(file, label, &name("head.out.weight"), &[N_FFT + 2, DIM])?,
        head_bias: load_tensor(file, label, &name("head.out.bias"), &[N_FFT + 2])?,
    };
    let _window = load_tensor(file, label, &name("head.istft.window"), &[N_FFT])?;
    Ok(weights)
}

fn validate_f32_manifest(file: &GgufFile) -> Result<()> {
    if let Some(tensor) = file
        .tensors()
        .iter()
        .find(|tensor| tensor.dtype != GgmlType::F32)
    {
        return Err(VokraError::ModelLoad(format!(
            "{LABEL}: tensor `{}` is {:?}, but the pinned official manifest is F32",
            tensor.name, tensor.dtype
        )));
    }
    Ok(())
}

fn validate_additive_contract(file: &GgufFile) -> Result<()> {
    let present = ADDITIVE_KEYS
        .iter()
        .filter(|&&key| file.get(key).is_some())
        .count();
    if present == 0 {
        return Ok(());
    }
    if present != ADDITIVE_KEYS.len() {
        let missing: Vec<_> = ADDITIVE_KEYS
            .iter()
            .copied()
            .filter(|key| file.get(key).is_none())
            .collect();
        return Err(VokraError::ModelLoad(format!(
            "{LABEL}: partial `vokra.yue_upsampler.*` contract ({present}/{} keys); missing={missing:?}",
            ADDITIVE_KEYS.len()
        )));
    }
    for (key, expected) in [
        (KEY_UPSTREAM_REVISION, UPSTREAM_REVISION),
        (KEY_CHECKPOINT_FILE, CHECKPOINT_FILE),
        (KEY_CHECKPOINT_SHA256, CHECKPOINT_SHA256),
        (KEY_SOURCE_PACKAGE, SOURCE_PACKAGE),
        (KEY_SOURCE_PACKAGE_SHA256, SOURCE_PACKAGE_SHA256),
        (KEY_MANIFEST_SHA256, MANIFEST_SHA256),
        (KEY_PUBLIC_REVISION, PUBLIC_REVISION),
        (KEY_PUBLIC_GGUF_SHA256, PUBLIC_GGUF_SHA256),
        (KEY_PADDING, "same"),
    ] {
        require_string(file, key, expected)?;
    }
    for (key, expected) in [
        (KEY_CHECKPOINT_BYTES, CHECKPOINT_BYTES),
        (KEY_PUBLIC_GGUF_BYTES, PUBLIC_GGUF_BYTES),
        (KEY_SAMPLE_RATE, u64::from(SAMPLE_RATE)),
        (KEY_INPUT_CHANNELS, INPUT_CHANNELS as u64),
        (KEY_DIM, DIM as u64),
        (KEY_INTERMEDIATE_DIM, INTERMEDIATE_DIM as u64),
        (KEY_NUM_LAYERS, NUM_LAYERS as u64),
        (KEY_N_FFT, N_FFT as u64),
        (KEY_HOP_LENGTH, HOP_LENGTH as u64),
    ] {
        require_u64(file, key, expected)?;
    }
    Ok(())
}

fn required_string<'a>(file: &'a GgufFile, key: &str) -> Result<&'a str> {
    file.get(key)
        .and_then(GgufMetadataValue::as_str)
        .ok_or_else(|| VokraError::ModelLoad(format!("{LABEL}: missing/non-string `{key}`")))
}

fn require_string(file: &GgufFile, key: &str, expected: &str) -> Result<()> {
    let actual = required_string(file, key)?;
    if actual != expected {
        return Err(VokraError::ModelLoad(format!(
            "{LABEL}: `{key}`={actual:?}, expected {expected:?}"
        )));
    }
    Ok(())
}

fn require_u64(file: &GgufFile, key: &str, expected: u64) -> Result<()> {
    let actual = file.get(key).and_then(GgufMetadataValue::as_u64);
    if actual != Some(expected) {
        return Err(VokraError::ModelLoad(format!(
            "{LABEL}: `{key}`={actual:?}, expected {expected}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;

    fn read_f32(path: &Path) -> Vec<f32> {
        let bytes = std::fs::read(path)
            .unwrap_or_else(|error| panic!("read YuE fixture {}: {error}", path.display()));
        assert_eq!(bytes.len() % 4, 0, "fixture must be little-endian f32");
        bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect()
    }

    fn measure(label: &str, actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len(), "{label} length");
        assert!(!actual.is_empty(), "{label} must not be empty");
        let mut max_abs = 0.0f64;
        let mut sum_abs = 0.0f64;
        let mut sum_sq = 0.0f64;
        let mut dot = 0.0f64;
        let mut actual_sq = 0.0f64;
        let mut expected_sq = 0.0f64;
        let mut worst = 0usize;
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            let actual = f64::from(actual);
            let expected = f64::from(expected);
            let delta = (actual - expected).abs();
            if delta > max_abs {
                max_abs = delta;
                worst = index;
            }
            sum_abs += delta;
            sum_sq += delta * delta;
            dot += actual * expected;
            actual_sq += actual * actual;
            expected_sq += expected * expected;
        }
        let count = actual.len() as f64;
        let cosine = dot / (actual_sq.sqrt() * expected_sq.sqrt());
        eprintln!(
            "YUE_UPSAMPLER_MEASUREMENT label={label} samples={} max_abs={max_abs:.9e} \
             worst_index={worst} mean_abs={:.9e} rms={:.9e} cosine={cosine:.12}",
            actual.len(),
            sum_abs / count,
            (sum_sq / count).sqrt(),
        );
    }

    fn real_case() -> (GgufFile, Vec<f32>, Vec<f32>) {
        let gguf = std::env::var_os("VOKRA_YUE_UPSAMPLER_GGUF")
            .expect("VOKRA_YUE_UPSAMPLER_GGUF must point at the strict public GGUF");
        let reference = std::env::var_os("VOKRA_YUE_UPSAMPLER_REFERENCE_DIR")
            .expect("VOKRA_YUE_UPSAMPLER_REFERENCE_DIR must point at the official dump");
        let reference = std::path::PathBuf::from(reference);
        let file = GgufFile::open(gguf).expect("open real YuE-upsampler GGUF");
        let features = read_f32(&reference.join("features.f32le"));
        let expected = read_f32(&reference.join("waveform.f32le"));
        (file, features, expected)
    }

    #[test]
    fn constants_pin_the_public_release() {
        assert_eq!(TENSOR_COUNT, 81);
        assert_eq!(INPUT_CHANNELS, 1024);
        assert_eq!(N_FFT, 3528);
        assert_eq!(HOP_LENGTH, 882);
        assert_eq!(SAMPLE_RATE, 44_100);
        assert_eq!(MANIFEST_SHA256.len(), 64);
        assert_eq!(CHECKPOINT_SHA256.len(), 64);
    }

    #[test]
    fn learned_hot_ops_are_cpu_and_metal_complete() {
        Compute::for_backend(BackendKind::Cpu, YUE_UPSAMPLER_HOT_OPS)
            .expect("CPU covers every YuE-upsampler learned reduction");
        #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
        match Compute::for_backend(BackendKind::Metal, YUE_UPSAMPLER_HOT_OPS) {
            Ok(compute) => assert_eq!(compute.backend_name(), "metal"),
            Err(VokraError::BackendUnavailable(_)) => {}
            Err(error) => panic!("YuE-upsampler has a Metal coverage gap: {error}"),
        }
    }

    #[test]
    #[ignore = "requires VAST-prepared public GGUF and pinned vocos==0.1.0 fixture"]
    fn measure_real_cpu_against_official_vocos() {
        let (file, features, expected) = real_case();
        let model = YueUpsampler::from_gguf_with_backend(&file, BackendKind::Cpu)
            .expect("strict real YuE-upsampler CPU bind");
        assert_eq!(features.len() % INPUT_CHANNELS, 0);
        let frames = features.len() / INPUT_CHANNELS;
        let actual = model.decode(&features, frames).expect("real CPU forward");
        measure("cpu_vs_official", &actual, &expected);
        eprintln!(
            "YUE_UPSAMPLER_MEASUREMENT_ONLY backend=cpu numeric_bounds=UNSET verdict=MEASURED_NOT_GATED"
        );
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    #[ignore = "requires Apple Silicon, public GGUF and pinned vocos==0.1.0 fixture"]
    fn measure_real_metal_against_cpu_and_official_vocos() {
        if vokra_backend_metal::vokra_metal_probe().is_err() {
            eprintln!("skipping YuE-upsampler Metal measurement: no system Metal device");
            return;
        }
        let (file, features, expected) = real_case();
        let frames = features.len() / INPUT_CHANNELS;
        let cpu = YueUpsampler::from_gguf_with_backend(&file, BackendKind::Cpu)
            .expect("strict real CPU bind")
            .decode(&features, frames)
            .expect("real CPU forward");
        let metal = YueUpsampler::from_gguf_with_backend(&file, BackendKind::Metal)
            .expect("strict real Metal bind")
            .decode(&features, frames)
            .expect("real Metal forward");
        measure("metal_vs_cpu", &metal, &cpu);
        measure("metal_vs_official", &metal, &expected);
        eprintln!(
            "YUE_UPSAMPLER_MEASUREMENT_ONLY backend=metal numeric_bounds=UNSET verdict=MEASURED_NOT_GATED"
        );
    }
}
