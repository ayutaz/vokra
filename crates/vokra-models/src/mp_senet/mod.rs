//! Native MP-SENet DNS speech enhancement for the audited public GGUF.
//!
//! The released checkpoint is the original-author `g_best_dns` generator
//! repackaged by `JacobLinCool/MPSENet`.  Its exact 247-tensor manifest and MIT
//! provenance are pinned here.  The package's accidental
//! `MultiheadAttention(batch_first = false)` axis interpretation is part of
//! the immutable checkpoint behaviour and is reproduced deliberately.
//!
//! CPU and Apple Metal execute the same native graph through [`Compute`].
//! STFT/iSTFT, layout changes and scalar activations are host DSP/glue; every
//! learned reduction (Conv2d, attention, GRU and normalization) uses the
//! selected backend.  An unavailable or uncovered backend fails before PCM is
//! processed, with no silent CPU inference fallback.

mod nn;
mod weights;

use vokra_core::backend::BackendKind;
use vokra_core::gguf::{GgufFile, GgufMetadataValue, chunks};
use vokra_core::{LicenseClass, Result, VokraError};

use crate::compute::{Compute, HotOp};
use crate::strict_checkpoint::{StrictCheckpoint, StrictCheckpointSpec};

use self::weights::MpSenetWeights;

/// GGUF architecture tag for MP-SENet checkpoints.
pub const ARCH: &str = "mp_senet";
/// Canonical Vokra model name.
pub const NAME: &str = "mp-senet-dns";
/// Model-zoo task category.
pub const CATEGORY: &str = "denoise";
/// Upstream Hugging Face repository containing the converted checkpoint.
pub const UPSTREAM_HF: &str = "JacobLinCool/MP-SENet-DNS";
/// Pinned upstream checkpoint revision.
pub const UPSTREAM_REVISION: &str = "8b78493f536df1aa53bd3bcbb2f620f705e8589c";
/// Repository used to authenticate the reference implementation.
pub const REFERENCE_SOURCE: &str = "JacobLinCool/MPSENet";
/// Pinned reference-implementation revision.
pub const REFERENCE_REVISION: &str = "958141ca51703c5b1e0c30362ab5b1c8b0e49957";
/// Revision used when the source checkpoint was published.
pub const PUBLICATION_REVISION: &str = "a65c76f340a0c8a885fbbf1893d5ec0ea009d718";
/// Canonical upstream source repository.
pub const OFFICIAL_SOURCE: &str = "yxlu-0102/MP-SENet";
/// Pinned canonical source revision.
pub const OFFICIAL_SOURCE_REVISION: &str = "89932cfe90d1dacb8e170e4a331d762462c21792";
/// SHA-256 of the authenticated upstream checkpoint.
pub const MODEL_SHA256: &str = "74912046c8b352d78ca4056c9624d7256ac4d7eac45ce015822a7f2282749cdc";
/// SHA-256 of the authenticated upstream configuration.
pub const CONFIG_SHA256: &str = "0c5973617000142390726f8dad98a5b6b1429b4ef1a94da25f3bc009f86a3365";
/// SHA-256 of the reference repository's model file.
pub const REFERENCE_MODEL_SHA256: &str =
    "e629e2858836489a598f9b325aa3abfc2a2360c72fc676d45c458c17efcaa7e8";
/// SHA-256 of the publication checkpoint file.
pub const PUBLICATION_MODEL_SHA256: &str =
    "63d0ddc067e87b5ebe556e60a89fa4384f5fba51fed37b6cb477abfaa19cb208";
/// SHA-256 of the authenticated reference transformer source.
pub const REFERENCE_TRANSFORMER_SHA256: &str =
    "44fb17b9a604f861304fd72517bfea73508393ca0ef00b58aaab6083c012ef0b";
/// SHA-256 of the reference repository license.
pub const REFERENCE_LICENSE_SHA256: &str =
    "df6322ce3ca3c70a0845c4a384432a9af50e7d70886d316741e2f47b5ae01f34";
/// SHA-256 of the canonical source license.
pub const OFFICIAL_LICENSE_SHA256: &str =
    "858f31052a5df6bcec94b015607bfade5a7cc6e950f7a9822aa4da3cc6f62fca";
/// Immutable revision of the public Vokra GGUF artifact.
pub const PUBLIC_REVISION: &str = "6017b7d70cf779c03f2fe061b56aa475e870d739";
/// SHA-256 of the public Vokra GGUF artifact.
pub const PUBLIC_MODEL_SHA256: &str =
    "26eec4a59c0eb8d31ea5115b3cb7d890f5b3745703ef0f0974b4e08c58e8da95";
/// SHA-256 of the canonical tensor name/shape manifest.
pub const MANIFEST_SHA256: &str =
    "84f05f3ca25e7c8f56e217d57458ea63dd7a0516cad0aeae3e6a1880c3bfd8fe";

/// Required waveform sample rate in hertz.
pub const SAMPLE_RATE: u32 = 16_000;
/// STFT transform size.
pub const N_FFT: usize = 400;
/// STFT hop length in samples.
pub const HOP_LENGTH: usize = 100;
/// STFT analysis-window length in samples.
pub const WIN_LENGTH: usize = 400;
/// Number of one-sided complex frequency bins.
pub const N_BINS: usize = N_FFT / 2 + 1;
/// Channel width of the dense encoder and decoder blocks.
pub const DENSE_CHANNELS: usize = 64;
/// Number of time-frequency separation blocks.
pub const TS_BLOCKS: usize = 4;
/// Number of attention heads in each separation block.
pub const ATTENTION_HEADS: usize = 4;
/// Hidden width of each recurrent direction.
pub const GRU_HIDDEN: usize = 128;
/// Canonical inference segment length in waveform samples.
pub const SEGMENT_SIZE: usize = 32_000;
/// Power-law magnitude compression exponent.
pub const COMPRESS_FACTOR: f32 = 0.3;
/// Phase-mask compression exponent used by the decoder.
pub const MASK_BETA: f32 = 2.0;

const KEY_MODEL_CATEGORY: &str = "vokra.model.category";
const KEY_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";
const KEY_UPSTREAM_REVISION: &str = "vokra.mp_senet.upstream_revision";
const KEY_REFERENCE_SOURCE: &str = "vokra.mp_senet.reference_source";
const KEY_REFERENCE_REVISION: &str = "vokra.mp_senet.reference_revision";
const KEY_PUBLICATION_REVISION: &str = "vokra.mp_senet.publication_revision";
const KEY_OFFICIAL_SOURCE: &str = "vokra.mp_senet.official_source";
const KEY_OFFICIAL_SOURCE_REVISION: &str = "vokra.mp_senet.official_source_revision";
const KEY_REFERENCE_MODEL_SHA256: &str = "vokra.mp_senet.reference_model_sha256";
const KEY_PUBLICATION_MODEL_SHA256: &str = "vokra.mp_senet.publication_model_sha256";
const KEY_REFERENCE_TRANSFORMER_SHA256: &str = "vokra.mp_senet.reference_transformer_sha256";
const KEY_SOURCE_LICENSE: &str = "vokra.mp_senet.source_license";
const KEY_REFERENCE_LICENSE_SHA256: &str = "vokra.mp_senet.reference_license_sha256";
const KEY_OFFICIAL_LICENSE_SHA256: &str = "vokra.mp_senet.official_license_sha256";
const KEY_MODEL_SHA256: &str = "vokra.mp_senet.model_sha256";
const KEY_CONFIG_SHA256: &str = "vokra.mp_senet.config_sha256";
const KEY_PUBLIC_REVISION: &str = "vokra.mp_senet.public_revision";
const KEY_PUBLIC_MODEL_SHA256: &str = "vokra.mp_senet.public_model_sha256";
const KEY_MANIFEST_SHA256: &str = "vokra.mp_senet.manifest_sha256";
const KEY_SAMPLE_RATE: &str = "vokra.mp_senet.sample_rate";
const KEY_N_FFT: &str = "vokra.mp_senet.n_fft";
const KEY_HOP_LENGTH: &str = "vokra.mp_senet.hop_length";
const KEY_WIN_LENGTH: &str = "vokra.mp_senet.win_length";
const KEY_COMPRESS_FACTOR: &str = "vokra.mp_senet.compress_factor";
const KEY_MASK_BETA: &str = "vokra.mp_senet.mask_beta";
const KEY_DENSE_CHANNELS: &str = "vokra.mp_senet.dense_channels";
const KEY_TS_BLOCKS: &str = "vokra.mp_senet.ts_blocks";
const KEY_ATTENTION_HEADS: &str = "vokra.mp_senet.attention_heads";
const KEY_GRU_HIDDEN: &str = "vokra.mp_senet.gru_hidden";
const KEY_SEGMENT_SIZE: &str = "vokra.mp_senet.segment_size";
const KEY_ATTENTION_BATCH_FIRST: &str = "vokra.mp_senet.attention_batch_first";
const KEY_INSTANCE_NORM_EPS: &str = "vokra.mp_senet.instance_norm_eps";
const KEY_LAYER_NORM_EPS: &str = "vokra.mp_senet.layer_norm_eps";
const KEY_STFT_CENTER: &str = "vokra.mp_senet.stft_center";
const KEY_STFT_NORMALIZED: &str = "vokra.mp_senet.stft_normalized";
const KEY_STFT_ONESIDED: &str = "vokra.mp_senet.stft_onesided";
const KEY_HANN_PERIODIC: &str = "vokra.mp_senet.hann_periodic";
const KEY_MAGNITUDE_EPS: &str = "vokra.mp_senet.magnitude_eps";
const KEY_PHASE_IMAG_EPS: &str = "vokra.mp_senet.phase_imag_eps";
const KEY_PHASE_REAL_EPS: &str = "vokra.mp_senet.phase_real_eps";

const SPEC: StrictCheckpointSpec = StrictCheckpointSpec {
    label: "mp_senet",
    arch: ARCH,
    model_name: NAME,
    model_name_alias: None,
    tensor_count: 247,
    manifest_sha256: [
        0x84, 0xf0, 0x5f, 0x3c, 0xa2, 0x5e, 0x7c, 0x8f, 0x56, 0xe2, 0x17, 0xd5, 0x74, 0x58, 0xea,
        0x63, 0xdd, 0x7a, 0x05, 0x16, 0xca, 0xd0, 0xae, 0xae, 0x3e, 0x6a, 0x18, 0x80, 0xc3, 0xbf,
        0xd8, 0xfe,
    ],
};

/// Complete learned reduction set for both CPU and Metal execution.
pub const MP_SENET_HOT_OPS: &[HotOp] = &[HotOp::Gemm, HotOp::Softmax, HotOp::LayerNorm];

#[derive(Debug, Clone)]
/// Strictly authenticated native MP-SENet speech enhancer.
pub struct MpSenet {
    weights: MpSenetWeights,
    weight_license: LicenseClass,
    backend: BackendKind,
}

impl MpSenet {
    /// Binds the exact public MP-SENet DNS tensor manifest.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        let checkpoint = StrictCheckpoint::bind(file, SPEC)?;
        require_string(
            file,
            chunks::KEY_PROVENANCE_MODEL_ID,
            checkpoint.model_name(),
        )?;
        require_string(file, KEY_MODEL_CATEGORY, CATEGORY)?;
        require_string(file, KEY_UPSTREAM_HF, UPSTREAM_HF)?;
        require_string(file, chunks::KEY_PROVENANCE_LICENSE, "mit")?;
        require_string(
            file,
            chunks::KEY_PROVENANCE_WEIGHT_LICENSE,
            LicenseClass::Permissive.as_str(),
        )?;
        validate_additive_contract(file)?;
        let weights = MpSenetWeights::bind(file)?;
        Ok(Self {
            weights,
            weight_license: checkpoint.weight_license(),
            backend: BackendKind::Cpu,
        })
    }

    /// Opens and mmap-binds an MP-SENet GGUF checkpoint.
    pub fn open(path: impl AsRef<std::path::Path>) -> Result<Self> {
        Self::from_gguf(&GgufFile::open(path)?)
    }

    /// Selects the execution backend without changing the checkpoint.
    #[must_use]
    pub fn with_backend(mut self, backend: BackendKind) -> Self {
        self.backend = backend;
        self
    }

    /// Authenticates a checkpoint and preflights the requested backend.
    pub fn from_gguf_with_backend(file: &GgufFile, backend: BackendKind) -> Result<Self> {
        Compute::for_backend(backend, MP_SENET_HOT_OPS)?;
        Ok(Self::from_gguf(file)?.with_backend(backend))
    }

    /// Returns the explicitly selected execution backend.
    #[must_use]
    pub const fn backend(&self) -> BackendKind {
        self.backend
    }

    /// Returns the required waveform sample rate in hertz.
    #[must_use]
    pub const fn sample_rate(&self) -> u32 {
        SAMPLE_RATE
    }

    /// Returns the stamped weight-license class.
    #[must_use]
    pub const fn weight_license(&self) -> LicenseClass {
        self.weight_license
    }

    /// Enhances one complete 16 kHz mono utterance using the reference
    /// package's normalization and tail-joining segmentation policy.
    pub fn enhance(&self, pcm: &[f32]) -> Result<Vec<f32>> {
        let compute = Compute::for_backend(self.backend, MP_SENET_HOT_OPS)?;
        let (normalized, norm_factor) = normalize_pcm(pcm)?;
        let mut output = Vec::new();
        for (start, end) in segment_ranges(normalized.len()) {
            let mut segment = normalized[start..end].to_vec();
            if segment.len() < WIN_LENGTH {
                segment.resize(WIN_LENGTH, 0.0);
            }
            output.extend(nn::enhance_segment(&compute, &self.weights, &segment)?);
        }
        for value in &mut output {
            *value /= norm_factor;
        }
        if output.iter().any(|value| !value.is_finite()) {
            return Err(VokraError::InvalidArgument(
                "mp_senet: enhanced waveform contains a non-finite sample".to_owned(),
            ));
        }
        Ok(output)
    }
}

impl vokra_core::engines::SeparationEngine for MpSenet {
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

fn normalize_pcm(pcm: &[f32]) -> Result<(Vec<f32>, f32)> {
    if pcm.is_empty() {
        return Err(VokraError::InvalidArgument(
            "mp_senet: input PCM must not be empty".to_owned(),
        ));
    }
    if let Some((index, _)) = pcm.iter().enumerate().find(|(_, value)| !value.is_finite()) {
        return Err(VokraError::InvalidArgument(format!(
            "mp_senet: input PCM sample {index} is not finite"
        )));
    }
    let energy = pcm.iter().map(|value| value * value).sum::<f32>();
    if !energy.is_finite() || energy <= 0.0 {
        return Err(VokraError::InvalidArgument(
            "mp_senet: input PCM must have finite, non-zero energy".to_owned(),
        ));
    }
    let factor = (pcm.len() as f32 / energy).sqrt();
    if !factor.is_finite() {
        return Err(VokraError::InvalidArgument(
            "mp_senet: input normalization factor is not finite".to_owned(),
        ));
    }
    Ok((pcm.iter().map(|value| value * factor).collect(), factor))
}

fn segment_ranges(samples: usize) -> Vec<(usize, usize)> {
    let mut ranges = Vec::new();
    let mut start = 0usize;
    while start < samples {
        let mut end = start.saturating_add(SEGMENT_SIZE).min(samples);
        let join_tail = samples - end < SEGMENT_SIZE / 2;
        if join_tail {
            end = samples;
        }
        ranges.push((start, end));
        if join_tail {
            break;
        }
        start += SEGMENT_SIZE;
    }
    ranges
}

fn validate_additive_contract(file: &GgufFile) -> Result<()> {
    // The historical public artifact predates this richer contract. Its exact
    // complete manifest is the compatibility proof. New converter output is
    // all-or-nothing once the upstream revision marker is present.
    if file.get(KEY_UPSTREAM_REVISION).is_none() {
        return Ok(());
    }
    for (key, expected) in [
        (KEY_UPSTREAM_REVISION, UPSTREAM_REVISION),
        (KEY_REFERENCE_SOURCE, REFERENCE_SOURCE),
        (KEY_REFERENCE_REVISION, REFERENCE_REVISION),
        (KEY_PUBLICATION_REVISION, PUBLICATION_REVISION),
        (KEY_OFFICIAL_SOURCE, OFFICIAL_SOURCE),
        (KEY_OFFICIAL_SOURCE_REVISION, OFFICIAL_SOURCE_REVISION),
        (KEY_REFERENCE_MODEL_SHA256, REFERENCE_MODEL_SHA256),
        (KEY_PUBLICATION_MODEL_SHA256, PUBLICATION_MODEL_SHA256),
        (
            KEY_REFERENCE_TRANSFORMER_SHA256,
            REFERENCE_TRANSFORMER_SHA256,
        ),
        (KEY_SOURCE_LICENSE, "mit"),
        (KEY_REFERENCE_LICENSE_SHA256, REFERENCE_LICENSE_SHA256),
        (KEY_OFFICIAL_LICENSE_SHA256, OFFICIAL_LICENSE_SHA256),
        (KEY_MODEL_SHA256, MODEL_SHA256),
        (KEY_CONFIG_SHA256, CONFIG_SHA256),
        (KEY_PUBLIC_REVISION, PUBLIC_REVISION),
        (KEY_PUBLIC_MODEL_SHA256, PUBLIC_MODEL_SHA256),
        (KEY_MANIFEST_SHA256, MANIFEST_SHA256),
    ] {
        require_string(file, key, expected)?;
    }
    for (key, expected) in [
        (KEY_SAMPLE_RATE, u64::from(SAMPLE_RATE)),
        (KEY_N_FFT, N_FFT as u64),
        (KEY_HOP_LENGTH, HOP_LENGTH as u64),
        (KEY_WIN_LENGTH, WIN_LENGTH as u64),
        (KEY_DENSE_CHANNELS, DENSE_CHANNELS as u64),
        (KEY_TS_BLOCKS, TS_BLOCKS as u64),
        (KEY_ATTENTION_HEADS, ATTENTION_HEADS as u64),
        (KEY_GRU_HIDDEN, GRU_HIDDEN as u64),
        (KEY_SEGMENT_SIZE, SEGMENT_SIZE as u64),
    ] {
        require_u64(file, key, expected)?;
    }
    for (key, expected) in [
        (KEY_COMPRESS_FACTOR, COMPRESS_FACTOR),
        (KEY_MASK_BETA, MASK_BETA),
        (KEY_INSTANCE_NORM_EPS, 1.0e-5),
        (KEY_LAYER_NORM_EPS, 1.0e-5),
        (KEY_MAGNITUDE_EPS, 1.0e-9),
        (KEY_PHASE_IMAG_EPS, 1.0e-10),
        (KEY_PHASE_REAL_EPS, 1.0e-5),
    ] {
        require_f32(file, key, expected)?;
    }
    for (key, expected) in [
        (KEY_ATTENTION_BATCH_FIRST, false),
        (KEY_STFT_CENTER, true),
        (KEY_STFT_NORMALIZED, false),
        (KEY_STFT_ONESIDED, true),
        (KEY_HANN_PERIODIC, true),
    ] {
        require_bool(file, key, expected)?;
    }

    require_u64(file, chunks::KEY_FRONTEND_N_FFT, N_FFT as u64)?;
    require_u64(file, chunks::KEY_FRONTEND_HOP, HOP_LENGTH as u64)?;
    require_u64(file, chunks::KEY_FRONTEND_WIN_LENGTH, WIN_LENGTH as u64)?;
    require_string(file, chunks::KEY_FRONTEND_WINDOW_TYPE, "hann")?;
    require_string(file, chunks::KEY_FRONTEND_MEL_NORM, "none")?;
    require_bool(file, chunks::KEY_FRONTEND_HTK_MODE, false)?;
    require_f32(file, chunks::KEY_FRONTEND_FMIN, 0.0)?;
    require_f32(file, chunks::KEY_FRONTEND_FMAX, SAMPLE_RATE as f32 / 2.0)?;
    require_u64(file, chunks::KEY_FRONTEND_N_MELS, 0)?;
    require_string(file, chunks::KEY_FRONTEND_PAD_MODE, "reflect")?;
    require_bool(file, chunks::KEY_FRONTEND_DC_OFFSET_REMOVAL, false)?;
    require_f32(file, chunks::KEY_FRONTEND_PRE_EMPHASIS, 0.0)?;
    require_u64(
        file,
        chunks::KEY_FRONTEND_SAMPLE_RATE,
        u64::from(SAMPLE_RATE),
    )
}

fn require_string(file: &GgufFile, key: &str, expected: &str) -> Result<()> {
    let actual = file.get(key).and_then(GgufMetadataValue::as_str);
    if actual != Some(expected) {
        return Err(VokraError::ModelLoad(format!(
            "mp_senet: metadata {key}={actual:?}, expected {expected:?}"
        )));
    }
    Ok(())
}

fn require_u64(file: &GgufFile, key: &str, expected: u64) -> Result<()> {
    let actual = file.get(key).and_then(GgufMetadataValue::as_u64);
    if actual != Some(expected) {
        return Err(VokraError::ModelLoad(format!(
            "mp_senet: metadata {key}={actual:?}, expected {expected}"
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
            "mp_senet: metadata {key}={actual:?}, expected {expected}"
        )));
    }
    Ok(())
}

fn require_bool(file: &GgufFile, key: &str, expected: bool) -> Result<()> {
    let actual = file.get(key).and_then(GgufMetadataValue::as_bool);
    if actual != Some(expected) {
        return Err(VokraError::ModelLoad(format!(
            "mp_senet: metadata {key}={actual:?}, expected {expected}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn package_tail_joining_is_preserved() {
        assert_eq!(segment_ranges(32_000), vec![(0, 32_000)]);
        assert_eq!(segment_ranges(47_999), vec![(0, 47_999)]);
        assert_eq!(segment_ranges(48_000), vec![(0, 32_000), (32_000, 48_000)]);
        assert_eq!(
            segment_ranges(80_001),
            vec![(0, 32_000), (32_000, 64_000), (64_000, 80_001)]
        );
    }

    #[test]
    fn invalid_pcm_fails_before_backend_work() {
        assert!(
            normalize_pcm(&[])
                .unwrap_err()
                .to_string()
                .contains("empty")
        );
        assert!(
            normalize_pcm(&[f32::NAN])
                .unwrap_err()
                .to_string()
                .contains("not finite")
        );
        assert!(
            normalize_pcm(&[0.0, 0.0])
                .unwrap_err()
                .to_string()
                .contains("non-zero energy")
        );
    }

    #[test]
    fn learned_hot_ops_are_cpu_and_metal_complete() {
        Compute::for_backend(BackendKind::Cpu, MP_SENET_HOT_OPS)
            .expect("CPU covers MP-SENet learned reductions");
        #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
        match Compute::for_backend(BackendKind::Metal, MP_SENET_HOT_OPS) {
            Ok(compute) => assert_eq!(compute.backend_name(), "metal"),
            Err(VokraError::BackendUnavailable(_)) => {}
            Err(error) => panic!("MP-SENet has a Metal coverage gap: {error}"),
        }
    }
}
