//! Strict native binder for OpenMOSS MOSS-Audio-Tokenizer.
//!
//! Full and Nano share an upstream Python class but not a tensor topology.
//! This module authenticates the complete public GGUF tensor manifest before
//! selecting either contract. In particular, the first public Nano GGUF was
//! accidentally stamped with Full metadata; it is accepted only behind the
//! exact 374-tensor Nano manifest and is surfaced through
//! [`MossAudioTokenizer::requires_metadata_repair`]. A same-metadata artifact
//! with any other manifest fails closed.
//!
//! Nano token-to-PCM is implemented natively with one selected [`Compute`]
//! backend for every learned reduction. Full remains an explicit loud partial
//! until its separately verified 1.77B topology lands; it never substitutes
//! Nano, another codec, or CPU inference.

mod decoder;
mod weights;

use std::path::Path;

use vokra_core::backend::BackendKind;
use vokra_core::gguf::{GgufFile, GgufMetadataValue, chunks};
use vokra_core::{LicenseClass, Result, VokraError};

use crate::compute::{Compute, HotOp};
use crate::strict_checkpoint::{StrictCheckpoint, StrictCheckpointSpec};

use self::decoder::NanoDecoder;

/// GGUF architecture emitted by the offline converter.
pub const ARCH: &str = "moss_audio_tokenizer";
/// Model-zoo category for both public variants.
pub const CATEGORY: &str = "codec";
/// Canonical Full model identity.
pub const FULL_NAME: &str = "moss-audio-tokenizer";
/// Canonical Nano model identity.
pub const NANO_NAME: &str = "moss-audio-tokenizer-nano";
/// Full upstream repository pinned by the provenance contract.
pub const FULL_UPSTREAM_HF: &str = "OpenMOSS-Team/MOSS-Audio-Tokenizer";
/// Nano upstream repository pinned by the provenance contract.
pub const NANO_UPSTREAM_HF: &str = "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano";
/// Variant metadata key shared with `vokra-convert`.
pub const KEY_VARIANT: &str = "vokra.moss_audio_tokenizer.variant";
/// Upstream repository metadata key shared with `vokra-convert`.
pub const KEY_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";

/// Learned reductions required by the native Nano decode graph.
pub const MOSS_AUDIO_TOKENIZER_NANO_HOT_OPS: &[HotOp] =
    &[HotOp::Gemm, HotOp::Softmax, HotOp::LayerNorm, HotOp::Gelu];

const FULL_TENSOR_COUNT: usize = 1_600;
const NANO_TENSOR_COUNT: usize = 374;
const FULL_SPEC: StrictCheckpointSpec = StrictCheckpointSpec {
    label: "moss_audio_tokenizer/full",
    arch: ARCH,
    model_name: FULL_NAME,
    model_name_alias: None,
    tensor_count: FULL_TENSOR_COUNT,
    manifest_sha256: [
        0xbf, 0xe6, 0x88, 0xcf, 0x82, 0x11, 0x64, 0x5d, 0x05, 0xd9, 0x3f, 0xff, 0x00, 0xcd, 0xc2,
        0x78, 0xaa, 0xdc, 0x4f, 0x66, 0xe8, 0x65, 0x4a, 0x95, 0x1d, 0xb1, 0xd3, 0x8e, 0x85, 0xa6,
        0x6f, 0xa6,
    ],
};
const NANO_SPEC: StrictCheckpointSpec = StrictCheckpointSpec {
    label: "moss_audio_tokenizer/nano",
    arch: ARCH,
    model_name: NANO_NAME,
    // Narrow compatibility for the already-public, mis-stamped Nano GGUF.
    // The complete Nano manifest is authenticated before this alias is used.
    model_name_alias: Some(FULL_NAME),
    tensor_count: NANO_TENSOR_COUNT,
    manifest_sha256: [
        0xe5, 0xfd, 0xb1, 0xf1, 0x93, 0x8f, 0xdb, 0x52, 0x37, 0xd3, 0xae, 0x8b, 0x47, 0x06, 0xf2,
        0x6c, 0x6b, 0x92, 0x6a, 0xb7, 0x9d, 0xcb, 0xff, 0xf0, 0x82, 0xcc, 0xc3, 0x38, 0x0c, 0x5d,
        0x85, 0xd9,
    ],
};

/// Audited public codec topology.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MossAudioTokenizerVariant {
    /// 24 kHz mono, 32 residual LFQ codebooks, 1,920 samples/token.
    Full,
    /// 48 kHz stereo, 16 residual LFQ codebooks, 3,840 samples/channel/token.
    Nano,
}

impl MossAudioTokenizerVariant {
    /// Output sample rate.
    pub const fn sample_rate(self) -> u32 {
        match self {
            Self::Full => 24_000,
            Self::Nano => 48_000,
        }
    }

    /// Output channel count.
    pub const fn channels(self) -> usize {
        match self {
            Self::Full => 1,
            Self::Nano => 2,
        }
    }

    /// PCM samples emitted per channel and codec frame.
    pub const fn samples_per_channel_per_frame(self) -> usize {
        match self {
            Self::Full => 1_920,
            Self::Nano => 3_840,
        }
    }

    /// Maximum number of residual codebooks in the release.
    pub const fn max_quantizers(self) -> usize {
        match self {
            Self::Full => 32,
            Self::Nano => 16,
        }
    }
}

/// Interleaved decoded PCM plus its explicit channel/timebase contract.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct MossDecodedAudio {
    /// Standard frame-interleaved PCM (`L, R, L, R, ...` for Nano).
    pub pcm: Vec<f32>,
    /// Samples per second, per channel.
    pub sample_rate: u32,
    /// Number of interleaved output channels.
    pub channels: usize,
    /// Number of samples in each channel.
    pub samples_per_channel: usize,
}

/// Strictly authenticated MOSS Audio Tokenizer checkpoint.
#[derive(Debug, Clone)]
pub struct MossAudioTokenizer {
    variant: MossAudioTokenizerVariant,
    weight_license: LicenseClass,
    backend: BackendKind,
    requires_metadata_repair: bool,
    nano_decoder: Option<NanoDecoder>,
}

impl MossAudioTokenizer {
    /// Opens and binds a public Full or Nano GGUF.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::from_gguf(&GgufFile::open(path)?)
    }

    /// Authenticates the complete tensor manifest and provenance contract.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        let tensor_count = file.tensors().len();
        let (checkpoint, variant, requires_metadata_repair) = match tensor_count {
            FULL_TENSOR_COUNT => {
                let checkpoint = StrictCheckpoint::bind(file, FULL_SPEC)?;
                validate_common_metadata(file)?;
                validate_release_metadata(
                    file,
                    FULL_NAME,
                    "full",
                    FULL_UPSTREAM_HF,
                    full_source_description(),
                    "moss_audio_tokenizer/full",
                )?;
                (checkpoint, MossAudioTokenizerVariant::Full, false)
            }
            NANO_TENSOR_COUNT => {
                let checkpoint = StrictCheckpoint::bind(file, NANO_SPEC)?;
                validate_common_metadata(file)?;
                let legacy = checkpoint.model_name() == FULL_NAME;
                if legacy {
                    validate_release_metadata(
                        file,
                        FULL_NAME,
                        "full",
                        FULL_UPSTREAM_HF,
                        full_source_description(),
                        "moss_audio_tokenizer/nano legacy metadata",
                    )?;
                } else {
                    validate_release_metadata(
                        file,
                        NANO_NAME,
                        "nano",
                        NANO_UPSTREAM_HF,
                        nano_source_description(),
                        "moss_audio_tokenizer/nano",
                    )?;
                }
                (checkpoint, MossAudioTokenizerVariant::Nano, legacy)
            }
            other => {
                return Err(VokraError::ModelLoad(format!(
                    "moss_audio_tokenizer: tensor count {other} matches neither the pinned Full ({FULL_TENSOR_COUNT}) nor Nano ({NANO_TENSOR_COUNT}) release"
                )));
            }
        };
        if checkpoint.weight_license() != LicenseClass::Permissive {
            return Err(VokraError::ModelLoad(format!(
                "moss_audio_tokenizer: weight license {:?}, expected permissive Apache-2.0",
                checkpoint.weight_license()
            )));
        }
        debug_assert_eq!(checkpoint.tensor_count(), tensor_count);
        let nano_decoder = match variant {
            MossAudioTokenizerVariant::Full => None,
            MossAudioTokenizerVariant::Nano => Some(NanoDecoder::bind(file)?),
        };
        Ok(Self {
            variant,
            weight_license: checkpoint.weight_license(),
            backend: BackendKind::Cpu,
            requires_metadata_repair,
            nano_decoder,
        })
    }

    /// Binds the artifact and preflights the Nano learned-op backend seam.
    /// Full remains loadable for discovery but its decode graph is a loud
    /// partial until the separately verified Full topology lands.
    pub fn from_gguf_with_backend(file: &GgufFile, backend: BackendKind) -> Result<Self> {
        let model = Self::from_gguf(file)?;
        if model.variant == MossAudioTokenizerVariant::Nano {
            let _ = Compute::for_backend(backend, MOSS_AUDIO_TOKENIZER_NANO_HOT_OPS)?;
        }
        Ok(model.with_backend(backend))
    }

    /// Selects one backend for the complete learned graph. Decode never
    /// silently falls back to CPU.
    #[must_use]
    pub fn with_backend(mut self, backend: BackendKind) -> Self {
        self.backend = backend;
        self
    }

    /// Authenticated variant.
    pub const fn variant(&self) -> MossAudioTokenizerVariant {
        self.variant
    }

    /// Selected execution backend.
    pub const fn backend(&self) -> BackendKind {
        self.backend
    }

    /// Artifact weight-license class.
    pub const fn weight_license(&self) -> LicenseClass {
        self.weight_license
    }

    /// Whether this is the exact public Nano manifest carrying the historical
    /// Full metadata stamp. Such an artifact is safe to route as Nano but must
    /// be replaced before the model-zoo audit can call it canonical.
    pub const fn requires_metadata_repair(&self) -> bool {
        self.requires_metadata_repair
    }

    /// Output sample rate.
    pub const fn sample_rate(&self) -> u32 {
        self.variant.sample_rate()
    }

    /// Output channel count.
    pub const fn channels(&self) -> usize {
        self.variant.channels()
    }

    /// PCM samples per channel represented by one codec frame.
    pub const fn samples_per_channel_per_frame(&self) -> usize {
        self.variant.samples_per_channel_per_frame()
    }

    /// Maximum number of residual codebooks.
    pub const fn max_quantizers(&self) -> usize {
        self.variant.max_quantizers()
    }

    /// Decodes frame-major `[frames, num_quantizers]` codes. Nano emits
    /// standard stereo-interleaved 48 kHz PCM; Full is an explicit
    /// [`VokraError::UnsupportedOp`] until its separate decoder lands.
    pub fn decode_frame_major(
        &self,
        codes: &[u32],
        frames: usize,
        num_quantizers: usize,
    ) -> Result<MossDecodedAudio> {
        let Some(decoder) = &self.nano_decoder else {
            return Err(VokraError::UnsupportedOp(
                "moss_audio_tokenizer/Full: the exact 1,600-tensor checkpoint is bound, but its 24 kHz 32-quantizer token-to-PCM graph has not passed independent real-weight parity; Nano substitution and CPU fallback are forbidden"
                    .to_owned(),
            ));
        };
        let pcm = decoder.decode_frame_major(self.backend, codes, frames, num_quantizers)?;
        let samples_per_channel = frames
            .checked_mul(MossAudioTokenizerVariant::Nano.samples_per_channel_per_frame())
            .ok_or_else(|| {
                VokraError::InvalidArgument(format!(
                    "moss_audio_tokenizer/nano: frames * samples_per_channel_per_frame overflows: {frames} * {}",
                    MossAudioTokenizerVariant::Nano.samples_per_channel_per_frame()
                ))
            })?;
        let expected = samples_per_channel
            .checked_mul(MossAudioTokenizerVariant::Nano.channels())
            .ok_or_else(|| {
                VokraError::InvalidArgument(
                    "moss_audio_tokenizer/nano: interleaved output length overflows usize"
                        .to_owned(),
                )
            })?;
        if pcm.len() != expected {
            return Err(VokraError::InvalidArgument(format!(
                "moss_audio_tokenizer/nano: decoder emitted {} interleaved values, expected {expected}",
                pcm.len()
            )));
        }
        Ok(MossDecodedAudio {
            pcm,
            sample_rate: MossAudioTokenizerVariant::Nano.sample_rate(),
            channels: MossAudioTokenizerVariant::Nano.channels(),
            samples_per_channel,
        })
    }

    /// PCM-to-token is not yet implemented for either public release.
    pub fn encode(&self, _pcm: &[f32]) -> Result<Vec<u32>> {
        Err(VokraError::UnsupportedOp(format!(
            "moss_audio_tokenizer/{:?}: native PCM-to-token encode is not implemented",
            self.variant
        )))
    }
}

fn validate_common_metadata(file: &GgufFile) -> Result<()> {
    require_string(
        file,
        "vokra.model.category",
        CATEGORY,
        "moss_audio_tokenizer",
    )?;
    require_string(
        file,
        chunks::KEY_PROVENANCE_LICENSE,
        "apache-2.0",
        "moss_audio_tokenizer",
    )?;
    require_string(
        file,
        chunks::KEY_PROVENANCE_WEIGHT_LICENSE,
        LicenseClass::Permissive.as_str(),
        "moss_audio_tokenizer",
    )
}

fn validate_release_metadata(
    file: &GgufFile,
    model_id: &str,
    variant: &str,
    upstream_hf: &str,
    source: &str,
    label: &str,
) -> Result<()> {
    require_string(file, chunks::KEY_PROVENANCE_MODEL_ID, model_id, label)?;
    require_string(file, KEY_VARIANT, variant, label)?;
    require_string(file, KEY_UPSTREAM_HF, upstream_hf, label)?;
    require_string(file, chunks::KEY_PROVENANCE_SOURCE, source, label)
}

fn require_string(file: &GgufFile, key: &str, expected: &str, label: &str) -> Result<()> {
    let actual = file.get(key).and_then(GgufMetadataValue::as_str);
    if actual != Some(expected) {
        return Err(VokraError::ModelLoad(format!(
            "{label}: metadata `{key}`={actual:?}, expected {expected:?}"
        )));
    }
    Ok(())
}

const fn full_source_description() -> &'static str {
    "OpenMOSS-Team/MOSS-Audio-Tokenizer (MOSS-Audio-Tokenizer codec Full ~1.77B params F32, arXiv:2602.10934, apache-2.0)"
}

const fn nano_source_description() -> &'static str {
    "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano (MOSS-Audio-Tokenizer codec Nano ~22M params F32 distilled per arXiv:2603.18090, apache-2.0)"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn public_variant_axes_are_not_conflated() {
        assert_eq!(MossAudioTokenizerVariant::Full.sample_rate(), 24_000);
        assert_eq!(MossAudioTokenizerVariant::Full.channels(), 1);
        assert_eq!(MossAudioTokenizerVariant::Full.max_quantizers(), 32);
        assert_eq!(MossAudioTokenizerVariant::Nano.sample_rate(), 48_000);
        assert_eq!(MossAudioTokenizerVariant::Nano.channels(), 2);
        assert_eq!(MossAudioTokenizerVariant::Nano.max_quantizers(), 16);
    }

    #[test]
    fn nano_learned_ops_are_cpu_and_metal_complete() {
        Compute::for_backend(BackendKind::Cpu, MOSS_AUDIO_TOKENIZER_NANO_HOT_OPS)
            .expect("CPU covers the MOSS Audio Tokenizer Nano learned graph");
        #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
        match Compute::for_backend(BackendKind::Metal, MOSS_AUDIO_TOKENIZER_NANO_HOT_OPS) {
            Ok(compute) => assert_eq!(compute.backend_name(), "metal"),
            Err(VokraError::BackendUnavailable(_)) => {}
            Err(error) => panic!("MOSS Audio Tokenizer Nano has a Metal coverage gap: {error}"),
        }
    }
}
