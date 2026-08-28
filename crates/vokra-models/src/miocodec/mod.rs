//! Native MioCodec 25 Hz / 44.1 kHz v2 token decoder.
//!
//! The public `vokra/miocodec-25hz-44khz-v2` GGUF preserves the official
//! 350-tensor F32 checkpoint. This module pins its complete name/shape
//! manifest and implements the released token + global-embedding to waveform
//! route: five-dimensional FSQ, local Transformer prenet, learned temporal
//! upsampling, AdaLN-Zero speaker conditioning, ResNet stacks, SnakeBeta
//! upsampling and same-padded iSTFT.
//!
//! The public checkpoint also contains the WavLM-based encoder branches. This
//! first native route deliberately exposes decode only; [`MioCodec::encode_pcm`]
//! returns an explicit unsupported-operation error rather than substituting a
//! different encoder. Voice-conversion orchestration is not part of this core
//! API: callers provide already-produced codec tokens and one 128-dimensional
//! global embedding.
//!
//! All learned operations dispatch through [`crate::compute::Compute`]. CPU is
//! the scalar oracle and Apple Metal uses the existing FSQ, GEMM, softmax,
//! LayerNorm, GroupNorm, SiLU, Conv1d and SnakeBeta kernels. ConvTranspose1d is
//! expressed exactly as zero insertion plus a reversed Conv1d on the selected
//! backend. Unsupported backends fail before execution; no CPU fallback is
//! hidden inside the model.

mod nn;
mod weights;

use vokra_core::backend::BackendKind;
use vokra_core::gguf::{GgufFile, GgufMetadataValue, chunks};
use vokra_core::{LicenseClass, Result, VokraError};

use crate::compute::{Compute, HotOp};
use crate::strict_checkpoint::{StrictCheckpoint, StrictCheckpointSpec};

use self::weights::MioCodecWeights;

/// GGUF architecture tag for MioCodec checkpoints.
pub const ARCH: &str = "miocodec";
/// Canonical Vokra model name.
pub const MODEL_NAME: &str = "miocodec-25hz-44khz-v2";
/// Output waveform sample rate in hertz.
pub const SAMPLE_RATE: u32 = 44_100;
/// Codec token rate in frames per second.
pub const TOKEN_RATE: usize = 25;
/// Inverse-STFT transform size.
pub const N_FFT: usize = 392;
/// Inverse-STFT hop length in samples.
pub const HOP_LENGTH: usize = 98;
/// Number of one-sided inverse-STFT bins.
pub const ISTFT_BINS: usize = N_FFT / 2 + 1;
/// Content-code embedding width.
pub const CONTENT_DIM: usize = 768;
/// Global conditioning embedding width.
pub const GLOBAL_DIM: usize = 128;
/// Waveform decoder hidden width.
pub const WAVE_DIM: usize = 512;
/// Number of finite-scalar-quantizer dimensions.
pub const CODE_DIM: usize = 5;
/// Number of representable FSQ code indices.
pub const CODEBOOK_SIZE: usize = 12_800;
/// Quantization levels for each FSQ dimension.
pub const FSQ_LEVELS: [u32; CODE_DIM] = [8, 8, 8, 5, 5];
/// Number of content-prenet transformer layers.
pub const PRENET_LAYERS: usize = 6;
/// Attention heads per content-prenet layer.
pub const PRENET_HEADS: usize = 12;
/// Content-prenet feed-forward width.
pub const PRENET_HIDDEN: usize = 2_048;
/// Number of adaptive waveform-decoder transformer layers.
pub const WAVE_DECODER_LAYERS: usize = 8;
/// Attention heads per waveform-decoder layer.
pub const WAVE_DECODER_HEADS: usize = 8;
/// Waveform-decoder feed-forward width.
pub const WAVE_DECODER_HIDDEN: usize = 1_536;
/// Total temporal upsampling factor before inverse STFT.
pub const UPSAMPLE_TOTAL: usize = 9;
/// Rotary-position base used by the transformer blocks.
pub const ROPE_THETA: f32 = 10_000.0;
/// Versioned standalone decode-input container magic.
pub const DECODE_INPUT_MAGIC: &[u8; 8] = b"VKRMIO01";

/// Pinned upstream checkpoint revision.
pub const UPSTREAM_REVISION: &str = "67faba34153fe74e6665991c432a7327e23c5c1c";
/// Pinned native-reference source revision.
pub const SOURCE_REVISION: &str = "77473544375d57e96cbdfd5d7d257e8f280fa8e3";
/// SHA-256 of the authenticated upstream checkpoint.
pub const MODEL_SHA256: &str = "8e319ef2231bad184f17cb73fd5a21b685c25c6c1622ef33ed9271187e81cd4a";
/// SHA-256 of the authenticated upstream configuration.
pub const CONFIG_SHA256: &str = "bfabffffaaa5709b8dc69585111ee3d53c1b0609c23d293cd1b4903eafa5bec1";

const KEY_UPSTREAM_REVISION: &str = "vokra.miocodec.upstream_revision";
const KEY_SOURCE_REVISION: &str = "vokra.miocodec.source_revision";
const KEY_MODEL_SHA256: &str = "vokra.miocodec.model_sha256";
const KEY_CONFIG_SHA256: &str = "vokra.miocodec.config_sha256";
const KEY_SAMPLE_RATE: &str = "vokra.miocodec.sample_rate";
const KEY_N_FFT: &str = "vokra.miocodec.n_fft";
const KEY_HOP_LENGTH: &str = "vokra.miocodec.hop_length";
const KEY_CONTENT_DIM: &str = "vokra.miocodec.content_dim";
const KEY_GLOBAL_DIM: &str = "vokra.miocodec.global_dim";
const KEY_WAVE_DIM: &str = "vokra.miocodec.wave_dim";
const KEY_CODE_DIM: &str = "vokra.miocodec.code_dim";
const KEY_VOCAB_SIZE: &str = "vokra.miocodec.vocab_size";
const KEY_DECODE_ONLY: &str = "vokra.miocodec.decode_only";

const SPEC: StrictCheckpointSpec = StrictCheckpointSpec {
    label: "miocodec",
    arch: ARCH,
    model_name: MODEL_NAME,
    model_name_alias: None,
    tensor_count: 350,
    manifest_sha256: [
        0xf6, 0xa0, 0xf8, 0xc7, 0x05, 0x90, 0x9e, 0xc5, 0x09, 0xe2, 0xae, 0x92, 0xac, 0xeb, 0x9f,
        0xd8, 0x83, 0xdd, 0x91, 0x8a, 0xcd, 0x27, 0x10, 0x6f, 0x9e, 0x22, 0x22, 0x71, 0x83, 0x5a,
        0xc2, 0xcf,
    ],
};

/// Complete learned-op set for the official MioCodec decode path.
pub const MIOCODEC_DECODE_HOT_OPS: &[HotOp] = &[
    HotOp::Xcodec2Fsq,
    HotOp::Gemm,
    HotOp::Softmax,
    HotOp::LayerNorm,
    HotOp::GroupNorm,
    HotOp::Silu,
    HotOp::Conv1d,
    HotOp::SnakeBeta,
];

/// Standalone MioCodec decode input.
///
/// Binary layout (little-endian): 8-byte [`DECODE_INPUT_MAGIC`], `u64`
/// target sample count, `u32` code count, zero `u32` reserved field, 128 F32
/// global-embedding values, then `code_count` U32 FSQ indices. The exact byte
/// count is checked; trailing data is rejected.
#[derive(Debug, Clone, PartialEq)]
pub struct MioCodecDecodeInput {
    /// Requested upstream waveform length before deterministic frame flooring.
    pub target_samples: usize,
    /// Global conditioning vector supplied by the upstream encoder.
    pub global_embedding: [f32; GLOBAL_DIM],
    /// FSQ code indices in temporal order.
    pub codes: Vec<u32>,
}

impl MioCodecDecodeInput {
    const FIXED_BYTES: usize = 8 + 8 + 4 + 4 + GLOBAL_DIM * 4;

    /// Parses and validates the versioned standalone decode-input container.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < Self::FIXED_BYTES || &bytes[..8] != DECODE_INPUT_MAGIC {
            return Err(VokraError::InvalidArgument(
                "miocodec: decode input is missing VKRMIO01 header".to_owned(),
            ));
        }
        let target_samples_u64 = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
        let target_samples = usize::try_from(target_samples_u64).map_err(|_| {
            VokraError::InvalidArgument(format!(
                "miocodec: target sample count {target_samples_u64} does not fit this platform"
            ))
        })?;
        let code_count = u32::from_le_bytes(bytes[16..20].try_into().unwrap()) as usize;
        let reserved = u32::from_le_bytes(bytes[20..24].try_into().unwrap());
        if reserved != 0 {
            return Err(VokraError::InvalidArgument(format!(
                "miocodec: VKRMIO01 reserved field is {reserved}, expected zero"
            )));
        }
        if code_count == 0 {
            return Err(VokraError::InvalidArgument(
                "miocodec: VKRMIO01 code count must be positive".to_owned(),
            ));
        }
        let expected = Self::FIXED_BYTES
            .checked_add(code_count.checked_mul(4).ok_or_else(|| {
                VokraError::InvalidArgument("miocodec: code byte count overflow".to_owned())
            })?)
            .ok_or_else(|| {
                VokraError::InvalidArgument("miocodec: input byte count overflow".to_owned())
            })?;
        if bytes.len() != expected {
            return Err(VokraError::InvalidArgument(format!(
                "miocodec: VKRMIO01 has {} bytes, expected {expected} for {code_count} codes",
                bytes.len()
            )));
        }

        let mut global_embedding = [0.0f32; GLOBAL_DIM];
        let mut cursor = 24;
        for (index, value) in global_embedding.iter_mut().enumerate() {
            *value = f32::from_le_bytes(bytes[cursor..cursor + 4].try_into().unwrap());
            if !value.is_finite() {
                return Err(VokraError::InvalidArgument(format!(
                    "miocodec: global_embedding[{index}] is not finite"
                )));
            }
            cursor += 4;
        }
        let mut codes = Vec::with_capacity(code_count);
        for index in 0..code_count {
            let code = u32::from_le_bytes(bytes[cursor..cursor + 4].try_into().unwrap());
            if code as usize >= CODEBOOK_SIZE {
                return Err(VokraError::InvalidArgument(format!(
                    "miocodec: codes[{index}]={code} is outside 0..{CODEBOOK_SIZE}"
                )));
            }
            codes.push(code);
            cursor += 4;
        }
        MioCodec::output_samples_for_target(target_samples)?;
        Ok(Self {
            target_samples,
            global_embedding,
            codes,
        })
    }

    /// Serializes the decode input into the canonical little-endian container.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        if self.codes.is_empty() || self.codes.len() > u32::MAX as usize {
            return Err(VokraError::InvalidArgument(format!(
                "miocodec: code count {} is outside 1..={}",
                self.codes.len(),
                u32::MAX
            )));
        }
        MioCodec::output_samples_for_target(self.target_samples)?;
        let target_samples = u64::try_from(self.target_samples).map_err(|_| {
            VokraError::InvalidArgument(format!(
                "miocodec: target sample count {} does not fit VKRMIO01 u64",
                self.target_samples
            ))
        })?;
        let capacity = Self::FIXED_BYTES
            .checked_add(self.codes.len().checked_mul(4).ok_or_else(|| {
                VokraError::InvalidArgument("miocodec: code byte count overflow".to_owned())
            })?)
            .ok_or_else(|| {
                VokraError::InvalidArgument("miocodec: input byte count overflow".to_owned())
            })?;
        let mut bytes = Vec::with_capacity(capacity);
        bytes.extend_from_slice(DECODE_INPUT_MAGIC);
        bytes.extend_from_slice(&target_samples.to_le_bytes());
        bytes.extend_from_slice(&(self.codes.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        for (index, &value) in self.global_embedding.iter().enumerate() {
            if !value.is_finite() {
                return Err(VokraError::InvalidArgument(format!(
                    "miocodec: global_embedding[{index}] is not finite"
                )));
            }
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        for (index, &code) in self.codes.iter().enumerate() {
            if code as usize >= CODEBOOK_SIZE {
                return Err(VokraError::InvalidArgument(format!(
                    "miocodec: codes[{index}]={code} is outside 0..{CODEBOOK_SIZE}"
                )));
            }
            bytes.extend_from_slice(&code.to_le_bytes());
        }
        Ok(bytes)
    }
}

/// Strict real-weight MioCodec token-to-waveform model.
#[derive(Debug, Clone)]
pub struct MioCodec {
    weights: MioCodecWeights,
    weight_license: LicenseClass,
    backend: BackendKind,
}

impl MioCodec {
    /// Binds the audited public MioCodec v2 GGUF.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        let checkpoint = StrictCheckpoint::bind(file, SPEC)?;
        require_string(
            file,
            chunks::KEY_PROVENANCE_MODEL_ID,
            checkpoint.model_name(),
        )?;
        require_string(
            file,
            "vokra.provenance.upstream_hf",
            "Aratako/MioCodec-25Hz-44.1kHz-v2",
        )?;
        require_string(file, chunks::KEY_PROVENANCE_LICENSE, "mit")?;
        require_string(
            file,
            chunks::KEY_PROVENANCE_WEIGHT_LICENSE,
            LicenseClass::Permissive.as_str(),
        )?;

        // The historical public GGUF predates these additive topology keys.
        // Its exact full manifest above is the compatibility proof. Whenever
        // a new converter emits a key, a conflicting value is an error.
        require_optional_string(file, KEY_UPSTREAM_REVISION, UPSTREAM_REVISION)?;
        require_optional_string(file, KEY_SOURCE_REVISION, SOURCE_REVISION)?;
        require_optional_string(file, KEY_MODEL_SHA256, MODEL_SHA256)?;
        require_optional_string(file, KEY_CONFIG_SHA256, CONFIG_SHA256)?;
        require_optional_u64(file, KEY_SAMPLE_RATE, u64::from(SAMPLE_RATE))?;
        require_optional_u64(file, KEY_N_FFT, N_FFT as u64)?;
        require_optional_u64(file, KEY_HOP_LENGTH, HOP_LENGTH as u64)?;
        require_optional_u64(file, KEY_CONTENT_DIM, CONTENT_DIM as u64)?;
        require_optional_u64(file, KEY_GLOBAL_DIM, GLOBAL_DIM as u64)?;
        require_optional_u64(file, KEY_WAVE_DIM, WAVE_DIM as u64)?;
        require_optional_u64(file, KEY_CODE_DIM, CODE_DIM as u64)?;
        require_optional_u64(file, KEY_VOCAB_SIZE, CODEBOOK_SIZE as u64)?;
        require_optional_bool(file, KEY_DECODE_ONLY, true)?;

        let weights = MioCodecWeights::load(file)?;
        Ok(Self {
            weights,
            weight_license: checkpoint.weight_license(),
            backend: BackendKind::Cpu,
        })
    }

    /// Opens and binds an official GGUF. CLI sessions use their mmap path;
    /// this convenience entry keeps the buffered core reader semantics.
    pub fn open(path: impl AsRef<std::path::Path>) -> Result<Self> {
        Self::from_gguf(&GgufFile::open(path)?)
    }

    /// Selects one backend for the complete decoder graph.
    #[must_use]
    pub fn with_backend(mut self, backend: BackendKind) -> Self {
        self.backend = backend;
        self
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

    /// Returns the output waveform sample rate in hertz.
    #[must_use]
    pub const fn sample_rate(&self) -> u32 {
        SAMPLE_RATE
    }

    /// Returns the actual waveform length produced for an upstream
    /// `target_audio_length`. MioCodec floors first to an iSTFT frame and then
    /// to the 9x waveform-upsample factor.
    pub fn output_samples_for_target(target_samples: usize) -> Result<usize> {
        let pre_frames = target_samples / HOP_LENGTH / UPSAMPLE_TOTAL;
        if pre_frames == 0 {
            return Err(VokraError::InvalidArgument(format!(
                "miocodec: target_samples {target_samples} is too short; need at least {}",
                HOP_LENGTH * UPSAMPLE_TOTAL
            )));
        }
        pre_frames
            .checked_mul(UPSAMPLE_TOTAL)
            .and_then(|frames| frames.checked_mul(HOP_LENGTH))
            .ok_or_else(|| {
                VokraError::InvalidArgument("miocodec: output sample count overflow".to_owned())
            })
    }

    /// Decodes one batch-free FSQ code sequence and one 128-dimensional global
    /// embedding to 44.1 kHz PCM.
    pub fn decode_codes(
        &self,
        codes: &[u32],
        global_embedding: &[f32],
        target_samples: usize,
    ) -> Result<Vec<f32>> {
        if codes.is_empty() {
            return Err(VokraError::InvalidArgument(
                "miocodec: codes must not be empty".to_owned(),
            ));
        }
        if global_embedding.len() != GLOBAL_DIM {
            return Err(VokraError::InvalidArgument(format!(
                "miocodec: global embedding length {} != {GLOBAL_DIM}",
                global_embedding.len()
            )));
        }
        if let Some((index, _)) = global_embedding
            .iter()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(VokraError::InvalidArgument(format!(
                "miocodec: global_embedding[{index}] is not finite"
            )));
        }
        let expected = Self::output_samples_for_target(target_samples)?;
        let compute = Compute::for_backend(self.backend, MIOCODEC_DECODE_HOT_OPS)?;
        let pcm = nn::decode(
            &compute,
            &self.weights,
            codes,
            global_embedding,
            target_samples,
        )?;
        if pcm.len() != expected {
            return Err(VokraError::InvalidArgument(format!(
                "miocodec: decoder emitted {} samples, expected {expected}",
                pcm.len()
            )));
        }
        Ok(pcm)
    }

    /// Decodes a validated standalone container.
    pub fn decode_input(&self, input: &MioCodecDecodeInput) -> Result<Vec<f32>> {
        self.decode_codes(&input.codes, &input.global_embedding, input.target_samples)
    }

    /// MioCodec's WavLM encoder path is not part of the first native route.
    /// This explicit error prevents a caller from assuming another codec
    /// encoder or a CPU fallback was substituted.
    pub fn encode_pcm(&self, _pcm: &[f32]) -> Result<Vec<u32>> {
        Err(VokraError::UnsupportedOp(
            "miocodec: PCM encoding is not implemented; the official token-to-waveform decoder is available, and Vokra does not substitute another encoder"
                .to_owned(),
        ))
    }
}

fn require_string(file: &GgufFile, key: &str, expected: &str) -> Result<()> {
    let actual = file.get(key).and_then(GgufMetadataValue::as_str);
    if actual != Some(expected) {
        return Err(VokraError::ModelLoad(format!(
            "miocodec: metadata `{key}`={actual:?}, expected {expected:?}"
        )));
    }
    Ok(())
}

fn require_optional_string(file: &GgufFile, key: &str, expected: &str) -> Result<()> {
    match file.get(key) {
        None => Ok(()),
        Some(value) if value.as_str() == Some(expected) => Ok(()),
        Some(value) => Err(VokraError::ModelLoad(format!(
            "miocodec: metadata `{key}`={:?}, expected {expected:?}",
            value.as_str()
        ))),
    }
}

fn require_optional_u64(file: &GgufFile, key: &str, expected: u64) -> Result<()> {
    match file.get(key) {
        None => Ok(()),
        Some(value) if value.as_u64() == Some(expected) => Ok(()),
        Some(value) => Err(VokraError::ModelLoad(format!(
            "miocodec: metadata `{key}`={:?}, expected {expected}",
            value.as_u64()
        ))),
    }
}

fn require_optional_bool(file: &GgufFile, key: &str, expected: bool) -> Result<()> {
    match file.get(key) {
        None => Ok(()),
        Some(value) if value.as_bool() == Some(expected) => Ok(()),
        Some(value) => Err(VokraError::ModelLoad(format!(
            "miocodec: metadata `{key}`={:?}, expected {expected}",
            value.as_bool()
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decoder_hot_ops_are_cpu_and_metal_complete() {
        Compute::for_backend(BackendKind::Cpu, MIOCODEC_DECODE_HOT_OPS)
            .expect("CPU covers the complete MioCodec decoder");
        #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
        match Compute::for_backend(BackendKind::Metal, MIOCODEC_DECODE_HOT_OPS) {
            Ok(compute) => assert_eq!(compute.backend_name(), "metal"),
            Err(VokraError::BackendUnavailable(_)) => {}
            Err(error) => panic!("MioCodec decode has a Metal coverage gap: {error}"),
        }
    }

    #[test]
    fn public_constants_and_manifest_are_exact() {
        assert_eq!(FSQ_LEVELS.iter().product::<u32>() as usize, CODEBOOK_SIZE);
        assert_eq!(SAMPLE_RATE as usize / TOKEN_RATE, 1_764);
        assert_eq!(N_FFT, HOP_LENGTH * 4);
        assert_eq!(SPEC.tensor_count, 350);
        assert_eq!(
            SPEC.manifest_sha256,
            [
                0xf6, 0xa0, 0xf8, 0xc7, 0x05, 0x90, 0x9e, 0xc5, 0x09, 0xe2, 0xae, 0x92, 0xac, 0xeb,
                0x9f, 0xd8, 0x83, 0xdd, 0x91, 0x8a, 0xcd, 0x27, 0x10, 0x6f, 0x9e, 0x22, 0x22, 0x71,
                0x83, 0x5a, 0xc2, 0xcf,
            ]
        );
    }

    #[test]
    fn target_length_uses_official_two_stage_floor() {
        assert_eq!(MioCodec::output_samples_for_target(44_100).unwrap(), 44_100);
        assert_eq!(MioCodec::output_samples_for_target(44_099).unwrap(), 43_218);
        assert!(MioCodec::output_samples_for_target(881).is_err());
    }

    #[test]
    fn standalone_decode_container_round_trips_and_rejects_trailing_bytes() {
        let input = MioCodecDecodeInput {
            target_samples: 44_100,
            global_embedding: core::array::from_fn(|index| index as f32 / 128.0),
            codes: vec![0, 1, 12_799],
        };
        let bytes = input.to_bytes().unwrap();
        assert_eq!(MioCodecDecodeInput::from_bytes(&bytes).unwrap(), input);
        let mut trailing = bytes;
        trailing.push(0);
        assert!(MioCodecDecodeInput::from_bytes(&trailing).is_err());
    }
}
