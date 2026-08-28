//! Native Meta Facebook Denoiser DNS48 speech enhancement.
//!
//! This module executes the exact 48-tensor causal DNS48 release from
//! `facebookresearch/denoiser`: two official sinc-by-two input resamplers,
//! five Conv1d + GLU encoder blocks, a two-layer unidirectional LSTM, five
//! additive-skip Conv1d + GLU + ConvTranspose1d decoder blocks, and the two
//! matching output resamplers.
//!
//! CPU and Apple Metal share one graph through [`Compute`]. Every learned
//! reduction (Conv1d, LSTM GEMM/GEMV, and transposed-convolution GEMM) uses the
//! selected backend. Layout changes, activations, skip addition, normalization,
//! and the fixed sinc filters are host DSP/glue. Backend coverage is checked
//! before PCM is processed; there is no silent CPU inference fallback.

mod nn;
mod weights;

use vokra_core::backend::BackendKind;
use vokra_core::engines::{DenoiseEngine, DenoiseStreamHandle, SeparationEngine};
use vokra_core::gguf::{GgufFile, GgufMetadataValue, chunks};
use vokra_core::{LicenseClass, Result, VokraError};

use crate::compute::{Compute, HotOp};
use crate::strict_checkpoint::{StrictCheckpoint, StrictCheckpointSpec};

pub use self::weights::FbDenoiserWeights;

/// GGUF architecture tag for the dedicated DNS48 binder.
pub const ARCH: &str = "facebook_denoiser";
/// Canonical model identity stamped by the historical public converter.
pub const NAME: &str = "facebook_denoiser";
/// Model-zoo task category.
pub const CATEGORY: &str = "enhancement";
/// Audited official source repository.
pub const UPSTREAM_URL: &str = "github.com/facebookresearch/denoiser";
const SOURCE_REVISION: &str = "8afd7c166699bb3c8b2d95b6dd706f71e1075df0";
const CHECKPOINT_URL: &str =
    "https://dl.fbaipublicfiles.com/adiyoss/denoiser/dns48-11decc9d8e3f0998.th";
const CHECKPOINT_BYTES: u64 = 75_478_395;
const CHECKPOINT_SHA256_PREFIX: &str = "11decc9d8e3f0998";
const SOURCE_DEMUCS_SHA256: &str =
    "8e9c21935c647e24f31cefcc63a298cb2a1c25bc99aab44bbe63a7b5570836be";
const SOURCE_RESAMPLE_SHA256: &str =
    "3e8ea258036660b7d33415794fe09ee010510f4d760bdfc5d5de268d6efb40f5";
const SOURCE_PRETRAINED_SHA256: &str =
    "885ad1ddd6cee5d4ecf5b4bc32784ceee97dc37ae19570b7ce0f9869b360d108";
const SOURCE_LICENSE_SHA256: &str =
    "336255dc30193e8e15d689d9481bb05673d89055718f3a96923a7ffb99adbbaf";
const PUBLIC_HF: &str = "vokra/facebook-denoiser";
const PUBLIC_REVISION: &str = "f50187791c52af3a90e479fcbacba3f267702eaa";
const PUBLIC_GGUF_SHA256: &str = "c0b23707a2f255b5eb108c5b08b92f310fede6870106e799b195282d6a375e74";
const MANIFEST_SHA256: &str = "bd25704cddfa2acd15f57f4ebb27d6c9a3c22f08121c7335287cbf6af4602ff1";

/// Number of learned tensors in the immutable DNS48 release.
pub const TENSOR_COUNT: usize = 48;
/// Required input/output PCM rate for the public DNS48 model.
pub const SAMPLE_RATE: u32 = 16_000;
/// Backwards-compatible name retained from the loud-partial binder.
pub const SAMPLE_RATE_DEFAULT: u32 = SAMPLE_RATE;
const HIDDEN: usize = 48;
const DEPTH: usize = 5;
const KERNEL_SIZE: usize = 8;
const STRIDE: usize = 4;
const RESAMPLE: usize = 4;
const GROWTH: usize = 2;
const MAX_HIDDEN: usize = 10_000;
const RESAMPLE_ZEROS: usize = 56;
const NORMALIZATION_FLOOR: f32 = 1.0e-3;
const LSTM_HIDDEN: usize = 768;

const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
const KEY_PROVENANCE_UPSTREAM_URL: &str = "vokra.provenance.upstream_url";
const KEY_SOURCE_REVISION: &str = "vokra.facebook_denoiser.source_revision";
const KEY_CHECKPOINT_URL: &str = "vokra.facebook_denoiser.checkpoint_url";
const KEY_CHECKPOINT_BYTES: &str = "vokra.facebook_denoiser.checkpoint_bytes";
const KEY_CHECKPOINT_SHA256_PREFIX: &str = "vokra.facebook_denoiser.checkpoint_sha256_prefix";
const KEY_SOURCE_DEMUCS_SHA256: &str = "vokra.facebook_denoiser.source_demucs_sha256";
const KEY_SOURCE_RESAMPLE_SHA256: &str = "vokra.facebook_denoiser.source_resample_sha256";
const KEY_SOURCE_PRETRAINED_SHA256: &str = "vokra.facebook_denoiser.source_pretrained_sha256";
const KEY_SOURCE_LICENSE_SHA256: &str = "vokra.facebook_denoiser.source_license_sha256";
const KEY_PUBLIC_HF: &str = "vokra.facebook_denoiser.public_hf";
const KEY_PUBLIC_REVISION: &str = "vokra.facebook_denoiser.public_revision";
const KEY_PUBLIC_GGUF_SHA256: &str = "vokra.facebook_denoiser.public_gguf_sha256";
const KEY_MANIFEST_SHA256: &str = "vokra.facebook_denoiser.manifest_sha256";
const KEY_SAMPLE_RATE: &str = "vokra.facebook_denoiser.sample_rate";
const KEY_HIDDEN: &str = "vokra.facebook_denoiser.hidden";
const KEY_DEPTH: &str = "vokra.facebook_denoiser.depth";
const KEY_KERNEL_SIZE: &str = "vokra.facebook_denoiser.kernel_size";
const KEY_STRIDE: &str = "vokra.facebook_denoiser.stride";
const KEY_RESAMPLE: &str = "vokra.facebook_denoiser.resample";
const KEY_GROWTH: &str = "vokra.facebook_denoiser.growth";
const KEY_MAX_HIDDEN: &str = "vokra.facebook_denoiser.max_hidden";
const KEY_RESAMPLE_ZEROS: &str = "vokra.facebook_denoiser.resample_zeros";
const KEY_NORMALIZATION_FLOOR: &str = "vokra.facebook_denoiser.normalization_floor";
const KEY_NORMALIZE: &str = "vokra.facebook_denoiser.normalize";
const KEY_GLU: &str = "vokra.facebook_denoiser.glu";
const KEY_CAUSAL: &str = "vokra.facebook_denoiser.causal";
const KEY_STD_CORRECTION: &str = "vokra.facebook_denoiser.std_correction";

const SPEC: StrictCheckpointSpec = StrictCheckpointSpec {
    label: "facebook_denoiser",
    arch: ARCH,
    model_name: NAME,
    model_name_alias: None,
    tensor_count: TENSOR_COUNT,
    manifest_sha256: [
        0xbd, 0x25, 0x70, 0x4c, 0xdd, 0xfa, 0x2a, 0xcd, 0x15, 0xf5, 0x7f, 0x4e, 0xbb, 0x27, 0xd6,
        0xc9, 0xa3, 0xc2, 0x2f, 0x08, 0x12, 0x1c, 0x73, 0x35, 0x28, 0x7c, 0xbf, 0x6a, 0xf4, 0x60,
        0x2f, 0xf1,
    ],
};

/// Complete learned reduction set for CPU and Apple Metal execution.
pub const FACEBOOK_DENOISER_HOT_OPS: &[HotOp] = &[HotOp::Conv1d, HotOp::Gemm, HotOp::Gemv];

#[derive(Debug, Clone)]
/// Strict native DNS48 speech-enhancement model.
pub struct FbDenoiser {
    weights: FbDenoiserWeights,
    weight_license: LicenseClass,
    backend: BackendKind,
}

impl FbDenoiser {
    /// Binds the exact public DNS48 tensor manifest and provenance contract.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        let checkpoint = StrictCheckpoint::bind(file, SPEC)?;
        require_string(
            file,
            chunks::KEY_PROVENANCE_MODEL_ID,
            checkpoint.model_name(),
        )?;
        require_string(file, KEY_MODEL_CATEGORY, CATEGORY)?;
        require_string(file, KEY_PROVENANCE_UPSTREAM_URL, UPSTREAM_URL)?;
        require_string(file, chunks::KEY_PROVENANCE_LICENSE, "cc-by-nc-4.0")?;
        require_string(
            file,
            chunks::KEY_PROVENANCE_WEIGHT_LICENSE,
            LicenseClass::NonCommercial.as_str(),
        )?;
        validate_additive_contract(file)?;
        let weights = FbDenoiserWeights::bind(file)?;
        Ok(Self {
            weights,
            weight_license: checkpoint.weight_license(),
            backend: BackendKind::Cpu,
        })
    }

    /// Opens and strictly binds a DNS48 GGUF from disk.
    pub fn open<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        Self::from_gguf(&GgufFile::open(path)?)
    }

    /// Strictly binds a DNS48 GGUF and preflights complete backend coverage.
    pub fn from_gguf_with_backend(file: &GgufFile, backend: BackendKind) -> Result<Self> {
        Compute::for_backend(backend, FACEBOOK_DENOISER_HOT_OPS)?;
        Ok(Self::from_gguf(file)?.with_backend(backend))
    }

    /// Selects one backend for every learned operation. Unsupported or
    /// unavailable backends fail when execution starts, never via CPU fallback.
    #[must_use]
    pub fn with_backend(mut self, backend: BackendKind) -> Self {
        self.backend = backend;
        self
    }

    #[must_use]
    /// Returns the explicitly selected execution backend.
    pub const fn backend(&self) -> BackendKind {
        self.backend
    }

    #[must_use]
    /// Returns the trained PCM sample rate.
    pub const fn sample_rate(&self) -> u32 {
        SAMPLE_RATE
    }

    #[must_use]
    /// Returns the fail-closed checkpoint licence class.
    pub const fn weight_license(&self) -> LicenseClass {
        self.weight_license
    }

    #[must_use]
    /// Returns the exact DNS48 tensor count.
    pub const fn tensor_count(&self) -> usize {
        TENSOR_COUNT
    }

    /// Enhances one complete 16 kHz mono utterance with the official DNS48
    /// offline forward, preserving its correction=1 normalization and exact
    /// 112-tap resampling filters.
    pub fn denoise(&self, pcm: &[f32]) -> Result<Vec<f32>> {
        let compute = Compute::for_backend(self.backend, FACEBOOK_DENOISER_HOT_OPS)?;
        nn::denoise(&compute, &self.weights, pcm)
    }
}

impl SeparationEngine for FbDenoiser {
    fn separate(&self, pcm: &[f32]) -> Result<Vec<Vec<f32>>> {
        Ok(vec![self.denoise(pcm)?])
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

impl DenoiseEngine for FbDenoiser {
    fn open_stream(&self, sample_rate: u32) -> Result<Box<dyn DenoiseStreamHandle + Send>> {
        if sample_rate != SAMPLE_RATE {
            return Err(VokraError::InvalidArgument(format!(
                "facebook_denoiser: expected {SAMPLE_RATE} Hz PCM, got {sample_rate} Hz; resample explicitly"
            )));
        }
        Ok(Box::new(FbDenoiserStream {
            model: self.clone(),
            pending: Vec::new(),
            finished: false,
        }))
    }
}

/// The audited public entry point is utterance-level. The shared streaming
/// facade therefore buffers pushes and emits the exact one-shot result at
/// finalize instead of pretending the offline normalization is causal.
struct FbDenoiserStream {
    model: FbDenoiser,
    pending: Vec<f32>,
    finished: bool,
}

impl DenoiseStreamHandle for FbDenoiserStream {
    fn push_pcm(&mut self, pcm: &[f32]) -> Result<Vec<f32>> {
        if self.finished {
            return Err(VokraError::InvalidArgument(
                "facebook_denoiser: stream is finalized; call reset before pushing more PCM"
                    .to_owned(),
            ));
        }
        if let Some((index, _)) = pcm.iter().enumerate().find(|(_, value)| !value.is_finite()) {
            return Err(VokraError::InvalidArgument(format!(
                "facebook_denoiser: pushed PCM sample {index} is not finite"
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
        self.model.denoise(&self.pending)
    }

    fn reset(&mut self) {
        self.pending.clear();
        self.finished = false;
    }
}

fn validate_additive_contract(file: &GgufFile) -> Result<()> {
    // The historical public artifact predates these richer pins. Its exact
    // complete manifest plus generic provenance is the compatibility proof.
    // New converter output is all-or-nothing once source_revision is present.
    if file.get(KEY_SOURCE_REVISION).is_none() {
        return Ok(());
    }
    for (key, expected) in [
        (KEY_SOURCE_REVISION, SOURCE_REVISION),
        (KEY_CHECKPOINT_URL, CHECKPOINT_URL),
        (KEY_CHECKPOINT_SHA256_PREFIX, CHECKPOINT_SHA256_PREFIX),
        (KEY_SOURCE_DEMUCS_SHA256, SOURCE_DEMUCS_SHA256),
        (KEY_SOURCE_RESAMPLE_SHA256, SOURCE_RESAMPLE_SHA256),
        (KEY_SOURCE_PRETRAINED_SHA256, SOURCE_PRETRAINED_SHA256),
        (KEY_SOURCE_LICENSE_SHA256, SOURCE_LICENSE_SHA256),
        (KEY_PUBLIC_HF, PUBLIC_HF),
        (KEY_PUBLIC_REVISION, PUBLIC_REVISION),
        (KEY_PUBLIC_GGUF_SHA256, PUBLIC_GGUF_SHA256),
        (KEY_MANIFEST_SHA256, MANIFEST_SHA256),
    ] {
        require_string(file, key, expected)?;
    }
    for (key, expected) in [
        (KEY_CHECKPOINT_BYTES, CHECKPOINT_BYTES),
        (KEY_SAMPLE_RATE, u64::from(SAMPLE_RATE)),
        (KEY_HIDDEN, HIDDEN as u64),
        (KEY_DEPTH, DEPTH as u64),
        (KEY_KERNEL_SIZE, KERNEL_SIZE as u64),
        (KEY_STRIDE, STRIDE as u64),
        (KEY_RESAMPLE, RESAMPLE as u64),
        (KEY_GROWTH, GROWTH as u64),
        (KEY_MAX_HIDDEN, MAX_HIDDEN as u64),
        (KEY_RESAMPLE_ZEROS, RESAMPLE_ZEROS as u64),
        (KEY_STD_CORRECTION, 1),
    ] {
        require_u64(file, key, expected)?;
    }
    for (key, expected) in [(KEY_NORMALIZE, true), (KEY_GLU, true), (KEY_CAUSAL, true)] {
        require_bool(file, key, expected)?;
    }
    require_f32(file, KEY_NORMALIZATION_FLOOR, NORMALIZATION_FLOOR)
}

fn require_string(file: &GgufFile, key: &str, expected: &str) -> Result<()> {
    let actual = file.get(key).and_then(GgufMetadataValue::as_str);
    if actual != Some(expected) {
        return Err(VokraError::ModelLoad(format!(
            "facebook_denoiser: metadata {key}={actual:?}, expected {expected:?}"
        )));
    }
    Ok(())
}

fn require_u64(file: &GgufFile, key: &str, expected: u64) -> Result<()> {
    let actual = file.get(key).and_then(GgufMetadataValue::as_u64);
    if actual != Some(expected) {
        return Err(VokraError::ModelLoad(format!(
            "facebook_denoiser: metadata {key}={actual:?}, expected {expected}"
        )));
    }
    Ok(())
}

fn require_bool(file: &GgufFile, key: &str, expected: bool) -> Result<()> {
    let actual = file.get(key).and_then(GgufMetadataValue::as_bool);
    if actual != Some(expected) {
        return Err(VokraError::ModelLoad(format!(
            "facebook_denoiser: metadata {key}={actual:?}, expected {expected}"
        )));
    }
    Ok(())
}

fn require_f32(file: &GgufFile, key: &str, expected: f32) -> Result<()> {
    let actual = match file.get(key) {
        Some(GgufMetadataValue::F32(value)) => Some(*value),
        _ => None,
    };
    if actual.map(f32::to_bits) != Some(expected.to_bits()) {
        return Err(VokraError::ModelLoad(format!(
            "facebook_denoiser: metadata {key}={actual:?}, expected {expected}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn learned_hot_ops_are_cpu_and_metal_complete() {
        Compute::for_backend(BackendKind::Cpu, FACEBOOK_DENOISER_HOT_OPS)
            .expect("CPU covers Facebook Denoiser learned reductions");
        #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
        match Compute::for_backend(BackendKind::Metal, FACEBOOK_DENOISER_HOT_OPS) {
            Ok(compute) => assert_eq!(compute.backend_name(), "metal"),
            Err(VokraError::BackendUnavailable(_)) => {}
            Err(error) => panic!("Facebook Denoiser has a Metal coverage gap: {error}"),
        }
    }

    #[test]
    fn constants_pin_the_public_dns48_release() {
        assert_eq!(TENSOR_COUNT, 48);
        assert_eq!(SAMPLE_RATE, 16_000);
        assert_eq!(LSTM_HIDDEN, 768);
        assert_eq!(MANIFEST_SHA256.len(), 64);
        assert_eq!(CHECKPOINT_SHA256_PREFIX.len(), 16);
    }
}
