//! Native TIGER-DnR and TIGER-speech source separation on CPU and Metal.
//!
//! The exact public GGUF manifests are pinned before any tensor is decoded.
//! Newly converted artifacts also carry the full topology/frontend metadata;
//! the two historical public artifacts predate those additive keys and are
//! accepted only because their immutable model name and complete name/shape
//! manifest match. Learned operations use one selected Compute backend.

mod nn;
mod weights;

use vokra_core::backend::BackendKind;
use vokra_core::gguf::{GgufFile, GgufMetadataValue, GgufValueType, chunks};
use vokra_core::{LicenseClass, Result, VokraError};

use crate::compute::{Compute, HotOp};
use crate::strict_checkpoint::{StrictCheckpoint, StrictCheckpointSpec};

use self::weights::TigerWeights;

/// GGUF architecture tag shared by the authenticated TIGER releases.
pub const ARCH: &str = "tiger_separator";
/// Frequency-band feature channels used by TIGER-DnR.
pub const FEATURE_DNR: usize = 132;
/// Frequency-band feature channels used by TIGER-speech.
pub const FEATURE_SPEECH: usize = 128;
/// Internal separator channel width.
pub const INTERNAL_CHANNELS: usize = 256;
/// Number of iterative separation blocks.
pub const ITERATIONS: usize = 8;
/// Depth of the learned upsampling path.
pub const UPSAMPLING_DEPTH: usize = 5;
/// Number of attention heads in each TIGER block.
pub const ATTENTION_HEADS: usize = 4;
/// Hidden channels assigned to each attention head.
pub const ATTENTION_HIDDEN: usize = 4;

/// Immutable upstream source-code revision used for the native port.
pub const SOURCE_REVISION: &str = "9f18d4a10a7137e1ce8052cfb62215179f1287b6";
/// SHA-256 of the upstream source license at [`SOURCE_REVISION`].
pub const SOURCE_LICENSE_SHA256: &str =
    "edc64d62aa021be7612337d2ced140375f52e4fd064b2f9cf6e656913d01bfa6";

const KEY_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";
const KEY_VARIANT: &str = "vokra.tiger.variant";
const KEY_UPSTREAM_REVISION: &str = "vokra.tiger.upstream_revision";
const KEY_SOURCE_REVISION: &str = "vokra.tiger.source_revision";
const KEY_SOURCE_FILE_SHA256: &str = "vokra.tiger.source_file_sha256";
const KEY_SOURCE_LICENSE: &str = "vokra.tiger.source_license";
const KEY_SOURCE_LICENSE_SHA256: &str = "vokra.tiger.source_license_sha256";
const KEY_MODEL_SHA256: &str = "vokra.tiger.model_sha256";
const KEY_CONFIG_SHA256: &str = "vokra.tiger.config_sha256";
const KEY_PUBLIC_REVISION: &str = "vokra.tiger.public_revision";
const KEY_PUBLIC_MODEL_SHA256: &str = "vokra.tiger.public_model_sha256";
const KEY_MANIFEST_SHA256: &str = "vokra.tiger.manifest_sha256";
const KEY_SAMPLE_RATE: &str = "vokra.tiger.sample_rate";
const KEY_N_FFT: &str = "vokra.tiger.n_fft";
const KEY_HOP_LENGTH: &str = "vokra.tiger.hop_length";
const KEY_FEATURE_CHANNELS: &str = "vokra.tiger.feature_channels";
const KEY_INTERNAL_CHANNELS: &str = "vokra.tiger.internal_channels";
const KEY_NUM_BLOCKS: &str = "vokra.tiger.num_blocks";
const KEY_NUM_SOURCES: &str = "vokra.tiger.num_sources";
const KEY_UPSAMPLING_DEPTH: &str = "vokra.tiger.upsampling_depth";
const KEY_ATTENTION_HEADS: &str = "vokra.tiger.attention_heads";
const KEY_ATTENTION_HIDDEN: &str = "vokra.tiger.attention_hidden_channels";
const KEY_ATTENTION_KERNEL: &str = "vokra.tiger.attention_kernel_size";
const KEY_ATTENTION_STRIDE: &str = "vokra.tiger.attention_stride";
const KEY_BAND_WIDTHS: &str = "vokra.tiger.band_widths";
const KEY_STFT_CENTER: &str = "vokra.tiger.stft_center";
const KEY_STFT_NORMALIZED: &str = "vokra.tiger.stft_normalized";
const KEY_STFT_ONESIDED: &str = "vokra.tiger.stft_onesided";
const KEY_HANN_PERIODIC: &str = "vokra.tiger.hann_periodic";

const DNR_SPEC: StrictCheckpointSpec = StrictCheckpointSpec {
    label: "tiger-dnr",
    arch: ARCH,
    model_name: "tiger-dnr",
    model_name_alias: None,
    tensor_count: 2_304,
    manifest_sha256: [
        0xf1, 0xda, 0xf2, 0xc5, 0x10, 0xef, 0x2c, 0x27, 0x27, 0x11, 0x96, 0x3a, 0x94, 0x0e, 0x1d,
        0xad, 0x74, 0xb7, 0x95, 0xa1, 0xf0, 0x4b, 0x2b, 0x1a, 0x52, 0x4e, 0x00, 0xc6, 0x1d, 0x30,
        0x7c, 0x02,
    ],
};

const SPEECH_SPEC: StrictCheckpointSpec = StrictCheckpointSpec {
    label: "tiger-speech",
    arch: ARCH,
    model_name: "tiger-speech",
    model_name_alias: None,
    tensor_count: 838,
    manifest_sha256: [
        0xdd, 0x0f, 0x9c, 0x0f, 0x25, 0x2c, 0x9d, 0xf0, 0x49, 0x8d, 0x1e, 0x4c, 0x51, 0x6d, 0xf9,
        0xec, 0x1b, 0xf1, 0x23, 0x0b, 0x64, 0xb6, 0xfb, 0xec, 0x21, 0x47, 0x52, 0x5c, 0xb7, 0x11,
        0xee, 0x1e,
    ],
};

/// Backend operations required by the executable separator.
pub const TIGER_HOT_OPS: &[HotOp] = &[
    HotOp::Gemm,
    HotOp::Softmax,
    HotOp::LayerNorm,
    HotOp::GroupedConv1d,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Authenticated TIGER release family.
pub enum TigerVariant {
    /// Three-stream denoise, dereverberation, and target-speech separation.
    Dnr,
    /// Two-speaker speech separation.
    Speech,
}

impl TigerVariant {
    /// Returns the canonical Vokra model name.
    pub const fn model_name(self) -> &'static str {
        match self {
            Self::Dnr => "tiger-dnr",
            Self::Speech => "tiger-speech",
        }
    }

    /// Returns the short metadata tag for this variant.
    pub const fn tag(self) -> &'static str {
        match self {
            Self::Dnr => "dnr",
            Self::Speech => "speech",
        }
    }

    /// Returns the immutable upstream Hugging Face repository identifier.
    pub const fn upstream_hf(self) -> &'static str {
        match self {
            Self::Dnr => "JusperLee/TIGER-DnR",
            Self::Speech => "JusperLee/TIGER-speech",
        }
    }

    /// Returns the pinned upstream weight revision.
    pub const fn upstream_revision(self) -> &'static str {
        match self {
            Self::Dnr => "b7a59560bbca10febbcd46fb01600f868e587f57",
            Self::Speech => "f0340340b2d9bbf72074edf8c076dcab59a10ba2",
        }
    }

    /// Returns the SHA-256 of the authenticated upstream source file.
    pub const fn source_file_sha256(self) -> &'static str {
        match self {
            Self::Dnr => "89605593bdfc05669e70f2b8647514077197f9870d32b5dd745913f6e03b50e0",
            Self::Speech => "a90ec403c5c024a1c6722a5143e0bd37bb642edec0e1506787ea212a65b287fe",
        }
    }

    /// Returns the SHA-256 of the original model checkpoint.
    pub const fn model_sha256(self) -> &'static str {
        match self {
            Self::Dnr => "dd1c696e72f6adea0085ef1af640882a8260519ad666422835e387a5b4abdd2a",
            Self::Speech => "7e5fac7a9083c94b3a00c524f323188d4dd19ef09a54c29d1fec12ac114922db",
        }
    }

    /// Returns the SHA-256 of the upstream configuration file.
    pub const fn config_sha256(self) -> &'static str {
        match self {
            Self::Dnr => "ba9d2f833bf2f3a5855a35d0ccd11c786f6b92f1a482d84404bc4673edb29b54",
            Self::Speech => "1643c4e30cb97bc67024965aae13d631d44efdd304d8379cfd92143791017946",
        }
    }

    /// Returns the immutable revision of the historical public GGUF.
    pub const fn public_revision(self) -> &'static str {
        match self {
            Self::Dnr => "8c8c78888684ecc8eef6beca3434c7ec9247bb70",
            Self::Speech => "e50793924eaae3897cee01f7f7791d14c296c7ed",
        }
    }

    /// Returns the SHA-256 of the historical public GGUF artifact.
    pub const fn public_model_sha256(self) -> &'static str {
        match self {
            Self::Dnr => "8737e4993efefbfec57ed7a0924503d626d07e410f456ff5693402852784017f",
            Self::Speech => "1fc11c3476bb6938410935e4f1877dcc2fb82005bf4ec0503dc01c013c29e562",
        }
    }

    /// Returns the canonical tensor name/shape manifest SHA-256.
    pub const fn manifest_sha256(self) -> &'static str {
        match self {
            Self::Dnr => "f1daf2c510ef2c272711963a940e1dad74b795a1f04b2b1a524e00c61d307c02",
            Self::Speech => "dd0f9c0f252c9df0498d1e4c516df9ec1bf1230b64b6fbeec2147525cb711ee1",
        }
    }

    /// Returns the required waveform sample rate in hertz.
    pub const fn sample_rate(self) -> u32 {
        match self {
            Self::Dnr => 44_100,
            Self::Speech => 16_000,
        }
    }

    /// Returns the STFT transform size.
    pub const fn n_fft(self) -> usize {
        match self {
            Self::Dnr => 2_048,
            Self::Speech => 640,
        }
    }

    /// Returns the STFT hop length in samples.
    pub const fn hop_length(self) -> usize {
        match self {
            Self::Dnr => 512,
            Self::Speech => 160,
        }
    }

    /// Returns the variant-specific band-feature width.
    pub const fn feature_channels(self) -> usize {
        match self {
            Self::Dnr => FEATURE_DNR,
            Self::Speech => FEATURE_SPEECH,
        }
    }

    /// Returns the number of separated waveform streams.
    pub const fn output_streams(self) -> usize {
        match self {
            Self::Dnr => 3,
            Self::Speech => 2,
        }
    }

    /// Returns the authenticated frequency-band widths in model order.
    pub fn band_widths(self) -> Vec<usize> {
        let mut widths = Vec::new();
        match self {
            Self::Dnr => {
                widths.extend(std::iter::repeat_n(2, 20));
                widths.extend(std::iter::repeat_n(4, 10));
                widths.extend(std::iter::repeat_n(11, 8));
                widths.extend(std::iter::repeat_n(23, 8));
                widths.extend(std::iter::repeat_n(46, 8));
                widths.extend(std::iter::repeat_n(92, 2));
                widths.push(121);
            }
            Self::Speech => {
                widths.extend(std::iter::repeat_n(1, 40));
                widths.extend(std::iter::repeat_n(4, 10));
                widths.extend(std::iter::repeat_n(10, 8));
                widths.extend(std::iter::repeat_n(20, 8));
                widths.push(1);
            }
        }
        widths
    }

    const fn strict_spec(self) -> StrictCheckpointSpec {
        match self {
            Self::Dnr => DNR_SPEC,
            Self::Speech => SPEECH_SPEC,
        }
    }
}

#[derive(Debug, Clone)]
/// Strictly bound native TIGER source separator.
pub struct TigerSeparator {
    variant: TigerVariant,
    weights: TigerWeights,
    weight_license: LicenseClass,
    backend: BackendKind,
}

impl TigerSeparator {
    /// Authenticates a TIGER GGUF and selects its release variant.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        let variant = select_variant(file)?;
        let checkpoint = StrictCheckpoint::bind(file, variant.strict_spec())?;
        require_string(
            file,
            chunks::KEY_PROVENANCE_MODEL_ID,
            checkpoint.model_name(),
        )?;
        require_string(file, KEY_UPSTREAM_HF, variant.upstream_hf())?;
        require_string(file, chunks::KEY_PROVENANCE_LICENSE, "apache-2.0")?;
        require_string(
            file,
            chunks::KEY_PROVENANCE_WEIGHT_LICENSE,
            LicenseClass::Permissive.as_str(),
        )?;
        require_optional_string(file, KEY_VARIANT, variant.tag())?;
        validate_additive_contract(file, variant)?;
        let weights = TigerWeights::bind(file, variant)?;
        Ok(Self {
            variant,
            weights,
            weight_license: checkpoint.weight_license(),
            backend: BackendKind::Cpu,
        })
    }

    /// Selects the execution backend without changing the bound checkpoint.
    #[must_use]
    pub fn with_backend(mut self, backend: BackendKind) -> Self {
        self.backend = backend;
        self
    }

    /// Authenticates a checkpoint and preflights the requested backend.
    pub fn from_gguf_with_backend(file: &GgufFile, backend: BackendKind) -> Result<Self> {
        Compute::for_backend(backend, TIGER_HOT_OPS)?;
        Ok(Self::from_gguf(file)?.with_backend(backend))
    }

    /// Returns the authenticated release variant.
    #[must_use]
    pub const fn variant(&self) -> TigerVariant {
        self.variant
    }

    /// Returns the required waveform sample rate in hertz.
    #[must_use]
    pub const fn sample_rate(&self) -> u32 {
        self.variant.sample_rate()
    }

    /// Returns the number of output waveform streams.
    #[must_use]
    pub const fn output_streams(&self) -> usize {
        self.variant.output_streams()
    }

    /// Returns the explicitly selected execution backend.
    #[must_use]
    pub const fn backend(&self) -> BackendKind {
        self.backend
    }

    /// Returns the stamped weight-license class.
    #[must_use]
    pub const fn weight_license(&self) -> LicenseClass {
        self.weight_license
    }

    /// Separates a finite mono waveform into the variant's output streams.
    pub fn separate(&self, pcm: &[f32]) -> Result<Vec<Vec<f32>>> {
        if pcm.is_empty() {
            return Err(VokraError::InvalidArgument(
                "tiger: input PCM must not be empty".to_owned(),
            ));
        }
        if let Some((index, _)) = pcm.iter().enumerate().find(|(_, value)| !value.is_finite()) {
            return Err(VokraError::InvalidArgument(format!(
                "tiger: input PCM sample {index} is not finite"
            )));
        }
        let compute = Compute::for_backend(self.backend, TIGER_HOT_OPS)?;
        nn::separate(&compute, self.variant, &self.weights, pcm)
    }
}

impl vokra_core::engines::SeparationEngine for TigerSeparator {
    fn separate(&self, pcm: &[f32]) -> Result<Vec<Vec<f32>>> {
        TigerSeparator::separate(self, pcm)
    }

    fn sample_rate(&self) -> u32 {
        TigerSeparator::sample_rate(self)
    }

    fn output_streams(&self) -> usize {
        TigerSeparator::output_streams(self)
    }

    fn backend(&self) -> BackendKind {
        TigerSeparator::backend(self)
    }
}

fn select_variant(file: &GgufFile) -> Result<TigerVariant> {
    match file
        .get(chunks::KEY_MODEL_NAME)
        .and_then(GgufMetadataValue::as_str)
    {
        Some("tiger-dnr") => Ok(TigerVariant::Dnr),
        Some("tiger-speech") => Ok(TigerVariant::Speech),
        other => Err(VokraError::ModelLoad(format!(
            "tiger: unsupported model name {other:?}; expected tiger-dnr or tiger-speech"
        ))),
    }
}

fn validate_additive_contract(file: &GgufFile, variant: TigerVariant) -> Result<()> {
    if file.get(KEY_UPSTREAM_REVISION).is_none() {
        return Ok(());
    }
    require_string(file, KEY_UPSTREAM_REVISION, variant.upstream_revision())?;
    require_string(file, KEY_SOURCE_REVISION, SOURCE_REVISION)?;
    require_string(file, KEY_SOURCE_FILE_SHA256, variant.source_file_sha256())?;
    require_string(file, KEY_SOURCE_LICENSE, "mit")?;
    require_string(file, KEY_SOURCE_LICENSE_SHA256, SOURCE_LICENSE_SHA256)?;
    require_string(file, KEY_MODEL_SHA256, variant.model_sha256())?;
    require_string(file, KEY_CONFIG_SHA256, variant.config_sha256())?;
    require_string(file, KEY_PUBLIC_REVISION, variant.public_revision())?;
    require_string(file, KEY_PUBLIC_MODEL_SHA256, variant.public_model_sha256())?;
    require_string(file, KEY_MANIFEST_SHA256, variant.manifest_sha256())?;
    require_u64(file, KEY_SAMPLE_RATE, u64::from(variant.sample_rate()))?;
    require_u64(file, KEY_N_FFT, variant.n_fft() as u64)?;
    require_u64(file, KEY_HOP_LENGTH, variant.hop_length() as u64)?;
    require_u64(
        file,
        KEY_FEATURE_CHANNELS,
        variant.feature_channels() as u64,
    )?;
    require_u64(file, KEY_INTERNAL_CHANNELS, INTERNAL_CHANNELS as u64)?;
    require_u64(file, KEY_NUM_BLOCKS, ITERATIONS as u64)?;
    require_u64(file, KEY_NUM_SOURCES, variant.output_streams() as u64)?;
    require_u64(file, KEY_UPSAMPLING_DEPTH, UPSAMPLING_DEPTH as u64)?;
    require_u64(file, KEY_ATTENTION_HEADS, ATTENTION_HEADS as u64)?;
    require_u64(file, KEY_ATTENTION_HIDDEN, ATTENTION_HIDDEN as u64)?;
    require_u64(file, KEY_ATTENTION_KERNEL, 8)?;
    require_u64(file, KEY_ATTENTION_STRIDE, 1)?;
    require_u32_array(file, KEY_BAND_WIDTHS, &variant.band_widths())?;
    require_bool(file, KEY_STFT_CENTER, true)?;
    require_bool(file, KEY_STFT_NORMALIZED, false)?;
    require_bool(file, KEY_STFT_ONESIDED, true)?;
    require_bool(file, KEY_HANN_PERIODIC, true)
}

fn require_string(file: &GgufFile, key: &str, expected: &str) -> Result<()> {
    let actual = file.get(key).and_then(GgufMetadataValue::as_str);
    if actual != Some(expected) {
        return Err(VokraError::ModelLoad(format!(
            "tiger: metadata {key}={actual:?}, expected {expected:?}"
        )));
    }
    Ok(())
}

fn require_optional_string(file: &GgufFile, key: &str, expected: &str) -> Result<()> {
    match file.get(key) {
        None => Ok(()),
        Some(value) if value.as_str() == Some(expected) => Ok(()),
        Some(value) => Err(VokraError::ModelLoad(format!(
            "tiger: metadata {key}={:?}, expected {expected:?}",
            value.as_str()
        ))),
    }
}

fn require_u64(file: &GgufFile, key: &str, expected: u64) -> Result<()> {
    let actual = file.get(key).and_then(GgufMetadataValue::as_u64);
    if actual != Some(expected) {
        return Err(VokraError::ModelLoad(format!(
            "tiger: metadata {key}={actual:?}, expected {expected}"
        )));
    }
    Ok(())
}

fn require_bool(file: &GgufFile, key: &str, expected: bool) -> Result<()> {
    let actual = file.get(key).and_then(GgufMetadataValue::as_bool);
    if actual != Some(expected) {
        return Err(VokraError::ModelLoad(format!(
            "tiger: metadata {key}={actual:?}, expected {expected}"
        )));
    }
    Ok(())
}

fn require_u32_array(file: &GgufFile, key: &str, expected: &[usize]) -> Result<()> {
    let array = file
        .get(key)
        .and_then(GgufMetadataValue::as_array)
        .ok_or_else(|| VokraError::ModelLoad(format!("tiger: missing/non-array metadata {key}")))?;
    let values: Option<Vec<usize>> = if array.element_type == GgufValueType::U32 {
        array
            .values
            .iter()
            .map(|value| value.as_u64().and_then(|v| usize::try_from(v).ok()))
            .collect()
    } else {
        None
    };
    if values.as_deref() != Some(expected) {
        return Err(VokraError::ModelLoad(format!(
            "tiger: metadata {key}={values:?}, expected {expected:?}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn official_topologies_are_distinct_and_complete() {
        assert_eq!(TigerVariant::Dnr.band_widths().len(), 57);
        assert_eq!(TigerVariant::Speech.band_widths().len(), 67);
        assert_eq!(TigerVariant::Dnr.band_widths().iter().sum::<usize>(), 1_025);
        assert_eq!(
            TigerVariant::Speech.band_widths().iter().sum::<usize>(),
            321
        );
        assert_ne!(DNR_SPEC.manifest_sha256, SPEECH_SPEC.manifest_sha256);
    }

    #[test]
    fn learned_hot_ops_are_cpu_and_metal_complete() {
        Compute::for_backend(BackendKind::Cpu, TIGER_HOT_OPS)
            .expect("CPU covers TIGER learned operations");
        #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
        match Compute::for_backend(BackendKind::Metal, TIGER_HOT_OPS) {
            Ok(compute) => assert_eq!(compute.backend_name(), "metal"),
            Err(VokraError::BackendUnavailable(_)) => {}
            Err(error) => panic!("TIGER has a Metal coverage gap: {error}"),
        }
    }
}
