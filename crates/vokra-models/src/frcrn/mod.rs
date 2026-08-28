//! Native Alibaba FRCRN-SE-16K speech enhancement for Mac CPU and Metal.
//!
//! The exact public 812-tensor F32 checkpoint contains two complex U-Nets,
//! frequency-memory blocks, squeeze/excitation paths, and fixed convolutional
//! STFT/iSTFT kernels. Every learned reduction is lowered to the selected
//! [`Compute`] backend through GEMM or grouped Conv1d. Fixed spectral
//! transforms, layout changes, BatchNorm inference, activations, overlap-add,
//! and complex arithmetic remain host DSP/glue. Backend coverage is preflighted
//! for the whole model and no Metal request can fall back to CPU inference.

mod nn;
mod weights;

use std::sync::Arc;

use vokra_core::backend::BackendKind;
use vokra_core::engines::{DenoiseEngine, DenoiseStreamHandle, SeparationEngine};
use vokra_core::gguf::{GgmlType, GgufFile, GgufMetadataValue, chunks};
use vokra_core::{LicenseClass, Result, VokraError};

use crate::compute::{Compute, HotOp};
use crate::strict_checkpoint::{StrictCheckpoint, StrictCheckpointSpec};

use self::weights::FrcrnWeights;

/// GGUF architecture tag accepted by the FRCRN binder.
pub const ARCH: &str = "frcrn";
/// Canonical `vokra.model.name` for the public release.
pub const NAME: &str = "frcrn";
/// Runtime task category stamped by the converter.
pub const CATEGORY: &str = "denoise";
/// Official Hugging Face checkpoint repository.
pub const UPSTREAM_HF: &str = "alibabasglab/FRCRN_SE_16K";
/// Provenance label carried by the audited historical public GGUF.
pub const LEGACY_UPSTREAM_HF: &str = "alibabasglab/FRCRN";
/// Pinned official Hugging Face checkpoint revision.
pub const UPSTREAM_REVISION: &str = "3766e6a64b0d8cb58f08d913d617bf129f11ed53";
/// Pinned ClearerVoice-Studio source revision used by the parity oracle.
pub const SOURCE_REVISION: &str = "6b3774dc79c46ae8bed2a4fa5f706f0ac8c75c61";
/// SHA-256 of the pinned official PyTorch checkpoint.
pub const CHECKPOINT_SHA256: &str =
    "b22256adbb91b68cf5a3db8f6657a4fb17066eecd5f069803e59c186c1cf3ebb";
/// SHA-256 of the sorted canonical tensor name/shape manifest.
pub const MANIFEST_SHA256: &str =
    "ca71dad1ae5293d3d63628b71127c0efdf004cec684e5a341ab376ce3e2851b7";
/// Exact byte length of the pinned official PyTorch checkpoint.
pub const CHECKPOINT_BYTES: u64 = 161_053_751;
/// Exact inference-tensor count after integer BatchNorm counters are removed.
pub const TENSOR_COUNT: usize = 812;
/// Fixed input and output sample rate of FRCRN-SE-16K.
pub const SAMPLE_RATE: u32 = 16_000;
pub(super) const FFT_LENGTH: usize = 640;
pub(super) const HOP_LENGTH: usize = 320;
pub(super) const FEATURE_DIM: usize = 321;
pub(super) const CHANNELS: usize = 128;
pub(super) const FSMN_ORDER: usize = 20;
pub(super) const SE_HIDDEN: usize = 16;
pub(super) const LABEL: &str = "frcrn";

const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
const KEY_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";
const KEY_UPSTREAM_REVISION: &str = "vokra.frcrn.upstream_revision";
const KEY_SOURCE_REVISION: &str = "vokra.frcrn.source_revision";
const KEY_CHECKPOINT_SHA256: &str = "vokra.frcrn.checkpoint_sha256";
const KEY_CHECKPOINT_BYTES: &str = "vokra.frcrn.checkpoint_bytes";
const KEY_MANIFEST_SHA256: &str = "vokra.frcrn.tensor_manifest_sha256";
const KEY_SAMPLE_RATE: &str = "vokra.frcrn.sample_rate";
const KEY_WINDOW_LENGTH: &str = "vokra.frcrn.window_length";
const KEY_HOP_LENGTH: &str = "vokra.frcrn.hop_length";
const KEY_FFT_LENGTH: &str = "vokra.frcrn.fft_length";
const KEY_FEATURE_DIM: &str = "vokra.frcrn.feature_dim";
const KEY_MODEL_DEPTH: &str = "vokra.frcrn.model_depth";
const KEY_CHANNELS: &str = "vokra.frcrn.channels";
const KEY_FSMN_ORDER: &str = "vokra.frcrn.fsmn_order";
const KEY_SE_HIDDEN: &str = "vokra.frcrn.se_hidden";
const KEY_UNET_COUNT: &str = "vokra.frcrn.unet_count";
const KEY_WINDOW_TYPE: &str = "vokra.frcrn.window_type";
const KEY_COMPLEX: &str = "vokra.frcrn.complex";

const ADDITIVE_KEYS: &[&str] = &[
    KEY_UPSTREAM_REVISION,
    KEY_SOURCE_REVISION,
    KEY_CHECKPOINT_SHA256,
    KEY_CHECKPOINT_BYTES,
    KEY_MANIFEST_SHA256,
    KEY_SAMPLE_RATE,
    KEY_WINDOW_LENGTH,
    KEY_HOP_LENGTH,
    KEY_FFT_LENGTH,
    KEY_FEATURE_DIM,
    KEY_MODEL_DEPTH,
    KEY_CHANNELS,
    KEY_FSMN_ORDER,
    KEY_SE_HIDDEN,
    KEY_UNET_COUNT,
    KEY_WINDOW_TYPE,
    KEY_COMPLEX,
];

const SPEC: StrictCheckpointSpec = StrictCheckpointSpec {
    label: LABEL,
    arch: ARCH,
    model_name: NAME,
    model_name_alias: None,
    tensor_count: TENSOR_COUNT,
    manifest_sha256: [
        0xca, 0x71, 0xda, 0xd1, 0xae, 0x52, 0x93, 0xd3, 0xd6, 0x36, 0x28, 0xb7, 0x11, 0x27, 0xc0,
        0xef, 0xdf, 0x00, 0x4c, 0xec, 0x68, 0x4e, 0x5a, 0x34, 0x1a, 0xb3, 0x76, 0xce, 0x3e, 0x28,
        0x51, 0xb7,
    ],
};

/// Every trained FRCRN reduction uses one of these backend kernels.
pub const FRCRN_HOT_OPS: &[HotOp] = &[HotOp::Gemm, HotOp::GroupedConv1d];

/// Strictly bound native FRCRN-SE-16K enhancement model.
#[derive(Debug, Clone)]
pub struct Frcrn {
    weights: Arc<FrcrnWeights>,
    weight_license: LicenseClass,
    backend: BackendKind,
}

impl Frcrn {
    /// Strictly binds the complete historical or newly stamped FRCRN artifact.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        let checkpoint = StrictCheckpoint::bind(file, SPEC)?;
        require_string(
            file,
            chunks::KEY_PROVENANCE_MODEL_ID,
            checkpoint.model_name(),
        )?;
        require_string(file, KEY_MODEL_CATEGORY, CATEGORY)?;
        require_string(file, chunks::KEY_PROVENANCE_LICENSE, "apache-2.0")?;
        require_string(
            file,
            chunks::KEY_PROVENANCE_WEIGHT_LICENSE,
            LicenseClass::Permissive.as_str(),
        )?;
        validate_f32_manifest(file)?;
        validate_additive_contract(file)?;
        Ok(Self {
            weights: Arc::new(FrcrnWeights::bind(file)?),
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
        Compute::for_backend(backend, FRCRN_HOT_OPS)?;
        Ok(Self::from_gguf(file)?.with_backend(backend))
    }

    /// Selects one backend for all trained reductions.
    #[must_use]
    pub fn with_backend(mut self, backend: BackendKind) -> Self {
        self.backend = backend;
        self
    }

    /// Returns the backend selected for every learned reduction.
    #[must_use]
    pub const fn backend(&self) -> BackendKind {
        self.backend
    }

    /// Returns the fixed 16 kHz waveform sample rate.
    #[must_use]
    pub const fn sample_rate(&self) -> u32 {
        SAMPLE_RATE
    }

    /// Returns the exact public inference-tensor count.
    #[must_use]
    pub const fn tensor_count(&self) -> usize {
        TENSOR_COUNT
    }

    /// Returns the weight-license class proven by the strict metadata gate.
    #[must_use]
    pub const fn weight_license(&self) -> LicenseClass {
        self.weight_license
    }

    /// Runs the exact upstream utterance-level DCCRN forward.
    pub fn enhance(&self, pcm: &[f32]) -> Result<Vec<f32>> {
        let compute = Compute::for_backend(self.backend, FRCRN_HOT_OPS)?;
        nn::enhance(&compute, &self.weights, pcm)
    }
}

impl SeparationEngine for Frcrn {
    fn separate(&self, pcm: &[f32]) -> Result<Vec<Vec<f32>>> {
        Ok(vec![self.enhance(pcm)?])
    }

    fn sample_rate(&self) -> u32 {
        SAMPLE_RATE
    }

    fn output_streams(&self) -> usize {
        1
    }

    fn backend(&self) -> BackendKind {
        self.backend
    }
}

impl DenoiseEngine for Frcrn {
    fn open_stream(&self, sample_rate: u32) -> Result<Box<dyn DenoiseStreamHandle + Send>> {
        if sample_rate != SAMPLE_RATE {
            return Err(VokraError::InvalidArgument(format!(
                "{LABEL}: expected {SAMPLE_RATE} Hz PCM, got {sample_rate} Hz; resample explicitly"
            )));
        }
        Ok(Box::new(FrcrnStream {
            model: self.clone(),
            pending: Vec::new(),
            finished: false,
        }))
    }
}

struct FrcrnStream {
    model: Frcrn,
    pending: Vec<f32>,
    finished: bool,
}

impl DenoiseStreamHandle for FrcrnStream {
    fn push_pcm(&mut self, pcm: &[f32]) -> Result<Vec<f32>> {
        if self.finished {
            return Err(VokraError::InvalidArgument(
                "frcrn: push after finalize; reset the stream first".to_owned(),
            ));
        }
        if let Some((index, _)) = pcm.iter().enumerate().find(|(_, value)| !value.is_finite()) {
            return Err(VokraError::InvalidArgument(format!(
                "frcrn: pushed PCM sample {index} is not finite"
            )));
        }
        self.pending.extend_from_slice(pcm);
        Ok(Vec::new())
    }

    fn finalize(&mut self) -> Result<Vec<f32>> {
        if self.finished {
            return Ok(Vec::new());
        }
        self.finished = true;
        if self.pending.is_empty() {
            return Ok(Vec::new());
        }
        self.model.enhance(&self.pending)
    }

    fn reset(&mut self) {
        self.pending.clear();
        self.finished = false;
    }
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
    let upstream = required_string(file, KEY_UPSTREAM_HF)?;
    if present == 0 {
        if upstream != LEGACY_UPSTREAM_HF {
            return Err(VokraError::ModelLoad(format!(
                "{LABEL}: metadata-free historical compatibility requires `{KEY_UPSTREAM_HF}={LEGACY_UPSTREAM_HF}`, got {upstream:?}"
            )));
        }
        return Ok(());
    }
    if present != ADDITIVE_KEYS.len() {
        let missing: Vec<_> = ADDITIVE_KEYS
            .iter()
            .copied()
            .filter(|key| file.get(key).is_none())
            .collect();
        return Err(VokraError::ModelLoad(format!(
            "{LABEL}: partial `vokra.frcrn.*` contract ({present}/{} keys); missing={missing:?}",
            ADDITIVE_KEYS.len()
        )));
    }
    if upstream != UPSTREAM_HF {
        return Err(VokraError::ModelLoad(format!(
            "{LABEL}: stamped contract requires `{KEY_UPSTREAM_HF}={UPSTREAM_HF}`, got {upstream:?}"
        )));
    }
    for (key, expected) in [
        (KEY_UPSTREAM_REVISION, UPSTREAM_REVISION),
        (KEY_SOURCE_REVISION, SOURCE_REVISION),
        (KEY_CHECKPOINT_SHA256, CHECKPOINT_SHA256),
        (KEY_MANIFEST_SHA256, MANIFEST_SHA256),
        (KEY_WINDOW_TYPE, "hanning-sqrt-periodic"),
    ] {
        require_string(file, key, expected)?;
    }
    for (key, expected) in [
        (KEY_CHECKPOINT_BYTES, CHECKPOINT_BYTES),
        (KEY_SAMPLE_RATE, u64::from(SAMPLE_RATE)),
        (KEY_WINDOW_LENGTH, FFT_LENGTH as u64),
        (KEY_HOP_LENGTH, HOP_LENGTH as u64),
        (KEY_FFT_LENGTH, FFT_LENGTH as u64),
        (KEY_FEATURE_DIM, FEATURE_DIM as u64),
        (KEY_MODEL_DEPTH, 14),
        (KEY_CHANNELS, CHANNELS as u64),
        (KEY_FSMN_ORDER, FSMN_ORDER as u64),
        (KEY_SE_HIDDEN, SE_HIDDEN as u64),
        (KEY_UNET_COUNT, 2),
    ] {
        require_u64(file, key, expected)?;
    }
    require_bool(file, KEY_COMPLEX, true)
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

fn require_bool(file: &GgufFile, key: &str, expected: bool) -> Result<()> {
    let actual = file.get(key).and_then(GgufMetadataValue::as_bool);
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
            .unwrap_or_else(|error| panic!("read FRCRN fixture {}: {error}", path.display()));
        assert_eq!(
            bytes.len() % 4,
            0,
            "FRCRN fixture {} is not raw little-endian f32",
            path.display()
        );
        bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte f32 chunk")))
            .collect()
    }

    fn measure(label: &str, actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len(), "{label} length");
        assert!(!actual.is_empty(), "{label} must not be empty");
        assert!(
            actual.iter().chain(expected).all(|value| value.is_finite()),
            "{label} must be finite"
        );
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
            "FRCRN_MEASUREMENT label={label} samples={} max_abs={max_abs:.9e} \
             worst_index={worst} mean_abs={:.9e} rms={:.9e} cosine={cosine:.12}",
            actual.len(),
            sum_abs / count,
            (sum_sq / count).sqrt(),
        );
    }

    fn real_case() -> (GgufFile, Vec<f32>, Vec<f32>) {
        let gguf = std::env::var_os("VOKRA_FRCRN_GGUF")
            .expect("VOKRA_FRCRN_GGUF must point at the strict public or regenerated GGUF");
        let reference = std::env::var_os("VOKRA_FRCRN_REFERENCE_DIR")
            .expect("VOKRA_FRCRN_REFERENCE_DIR must point at the independent official dump");
        let reference = std::path::PathBuf::from(reference);
        let file = GgufFile::open(gguf).expect("open real FRCRN GGUF");
        let pcm = read_f32(&reference.join("pcm.f32le"));
        let expected = read_f32(&reference.join("waveform.f32le"));
        (file, pcm, expected)
    }

    #[test]
    fn constants_pin_the_official_release() {
        assert_eq!(SAMPLE_RATE, 16_000);
        assert_eq!(FFT_LENGTH, 640);
        assert_eq!(HOP_LENGTH, 320);
        assert_eq!(FEATURE_DIM, 321);
        assert_eq!(TENSOR_COUNT, 812);
        assert_eq!(MANIFEST_SHA256.len(), 64);
        assert_eq!(CHECKPOINT_SHA256.len(), 64);
    }

    #[test]
    fn learned_hot_ops_are_cpu_and_metal_complete() {
        Compute::for_backend(BackendKind::Cpu, FRCRN_HOT_OPS)
            .expect("CPU covers every FRCRN learned reduction");
        #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
        match Compute::for_backend(BackendKind::Metal, FRCRN_HOT_OPS) {
            Ok(compute) => assert_eq!(compute.backend_name(), "metal"),
            Err(VokraError::BackendUnavailable(_)) => {}
            Err(error) => panic!("FRCRN has a Metal coverage gap: {error}"),
        }
    }

    #[test]
    #[ignore = "requires VAST-prepared GGUF and independent pinned ClearerVoice fixture"]
    fn measure_real_cpu_against_official_clearervoice() {
        let (file, pcm, expected) = real_case();
        let model = Frcrn::from_gguf_with_backend(&file, BackendKind::Cpu)
            .expect("strict real FRCRN CPU bind");
        let actual = model.enhance(&pcm).expect("real FRCRN CPU forward");
        measure("cpu_vs_official", &actual, &expected);
        eprintln!(
            "FRCRN_MEASUREMENT_ONLY backend=cpu numeric_bounds=UNSET verdict=MEASURED_NOT_GATED"
        );
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    #[test]
    #[ignore = "requires Apple Silicon, prepared GGUF and independent pinned ClearerVoice fixture"]
    fn measure_real_metal_against_cpu_and_official_clearervoice() {
        if vokra_backend_metal::vokra_metal_probe().is_err() {
            eprintln!("skipping FRCRN Metal measurement: no system Metal device");
            return;
        }
        let (file, pcm, expected) = real_case();
        let cpu = Frcrn::from_gguf_with_backend(&file, BackendKind::Cpu)
            .expect("strict real FRCRN CPU bind")
            .enhance(&pcm)
            .expect("real FRCRN CPU forward");
        let metal = Frcrn::from_gguf_with_backend(&file, BackendKind::Metal)
            .expect("strict real FRCRN Metal bind")
            .enhance(&pcm)
            .expect("real FRCRN Metal forward");
        measure("metal_vs_cpu", &metal, &cpu);
        measure("metal_vs_official", &metal, &expected);
        eprintln!(
            "FRCRN_MEASUREMENT_ONLY backend=metal numeric_bounds=UNSET verdict=MEASURED_NOT_GATED"
        );
    }
}
