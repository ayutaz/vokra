//! Native runtime for the three public FocalCodec speech codecs.
//!
//! The released 12.5, 25 and 50 Hz checkpoints share one non-causal
//! `WavLM -> FocalEncoder -> binary spherical quantizer -> FocalDecoder ->
//! Vocos` topology.  Only the compressor/decompressor scale factors differ.
//! This module pins the complete 354-entry name/shape manifest for each
//! public GGUF, then executes the upstream forward directly in Rust.  No
//! PyTorch, ONNX or protobuf component is linked into the runtime.

use vokra_core::backend::BackendKind;
use vokra_core::gguf::{GgufFile, GgufMetadataValue, chunks};
use vokra_core::{LicenseClass, Result, VokraError};

use crate::compute::HotOp;
use crate::strict_checkpoint::{StrictCheckpoint, StrictCheckpointSpec};

mod focal;
mod wavlm;
mod weights;

use weights::FocalCodecWeights;

/// Converter/runtime architecture tag.
pub const ARCH: &str = "focalcodec";
/// Required category for every official release.
pub const CATEGORY: &str = "codec";
/// Variant discriminator written by the current converter.
pub const KEY_VARIANT: &str = "vokra.focalcodec.variant";
/// Provenance key used by the pass-through converter.
pub const KEY_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";

const LABEL: &str = "focalcodec";
const TENSOR_COUNT: usize = 354;
const SAMPLE_RATE: u32 = 16_000;
const CODEBOOK_SIZE: usize = 8_192;
const CODE_DIM: usize = 13;

/// Every learned operation used by the complete FocalCodec forward.
///
/// Focal transposed convolutions have `kernel == stride` (one or two) and
/// are exactly expressed as one GEMM per tap followed by a host-side layout
/// interleave.  Listing GEMM here therefore also covers that path.  Selecting
/// a backend without any listed operation fails in `Compute::for_backend`
/// before inference; there is no silent CPU fallback.
pub const FOCALCODEC_HOT_OPS: &[HotOp] = &[
    HotOp::Gemm,
    HotOp::Softmax,
    HotOp::LayerNorm,
    HotOp::Gelu,
    HotOp::Conv1d,
    HotOp::GroupedConv1d,
    HotOp::SnakeActivation,
];

const MANIFEST_50HZ: [u8; 32] = [
    0x5e, 0xf5, 0x46, 0x5e, 0xde, 0x51, 0x04, 0xa7, 0xd6, 0xf6, 0xc1, 0x60, 0xcd, 0x28, 0xbe, 0x70,
    0x62, 0x62, 0x65, 0xf6, 0xe8, 0xe7, 0x6d, 0x93, 0xb9, 0xa5, 0x86, 0xfa, 0x71, 0x9a, 0x9c, 0x53,
];
const MANIFEST_25HZ: [u8; 32] = [
    0xc5, 0xca, 0x26, 0x2f, 0x45, 0xba, 0xcc, 0x92, 0x54, 0x2b, 0x23, 0x37, 0x74, 0x25, 0x38, 0xa3,
    0x3c, 0x85, 0xaa, 0x37, 0xe8, 0x2e, 0xe6, 0x19, 0x72, 0x22, 0xbc, 0xaa, 0x24, 0x5b, 0xf1, 0x4a,
];
const MANIFEST_12_5HZ: [u8; 32] = [
    0x4d, 0x66, 0x7d, 0xf4, 0xfe, 0x47, 0x42, 0x99, 0x71, 0x36, 0xd7, 0xb6, 0xe4, 0x9f, 0x58, 0x59,
    0xe1, 0xbc, 0x07, 0xd7, 0xfb, 0xd3, 0x90, 0xb1, 0x44, 0x44, 0x6b, 0x92, 0x79, 0xbd, 0xb7, 0x01,
];

/// One of the three audited public FocalCodec checkpoints.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FocalCodecVariant {
    /// 50 tokens/s; no focal time resampling.
    Hz50,
    /// 25 tokens/s; compressor factor two, decoder factor two.
    Hz25,
    /// 12.5 tokens/s; compressor factor four, decoder factor four.
    Hz12_5,
}

impl FocalCodecVariant {
    /// Canonical converter model name.
    pub const fn model_name(self) -> &'static str {
        match self {
            Self::Hz50 => "focalcodec-50hz",
            Self::Hz25 => "focalcodec-25hz",
            Self::Hz12_5 => "focalcodec-12-5hz",
        }
    }

    /// Variant metadata value.
    pub const fn tag(self) -> &'static str {
        match self {
            Self::Hz50 => "50hz",
            Self::Hz25 => "25hz",
            Self::Hz12_5 => "12_5hz",
        }
    }

    /// Official upstream weight repository.
    pub const fn upstream_hf(self) -> &'static str {
        match self {
            Self::Hz50 => "lucadellalib/focalcodec_50hz",
            Self::Hz25 => "lucadellalib/focalcodec_25hz",
            Self::Hz12_5 => "lucadellalib/focalcodec_12_5hz",
        }
    }

    /// Token rate numerator over two (`100`, `50`, `25`).
    pub const fn token_hz_times_two(self) -> u32 {
        match self {
            Self::Hz50 => 100,
            Self::Hz25 => 50,
            Self::Hz12_5 => 25,
        }
    }

    /// PCM samples represented by one token.
    pub const fn frame_hop(self) -> usize {
        match self {
            Self::Hz50 => 320,
            Self::Hz25 => 640,
            Self::Hz12_5 => 1_280,
        }
    }

    const fn downscale_factors(self) -> [usize; 3] {
        match self {
            Self::Hz50 => [1, 1, 1],
            Self::Hz25 => [2, 1, 1],
            Self::Hz12_5 => [2, 2, 1],
        }
    }

    const fn upscale_factors(self) -> [usize; 3] {
        match self {
            Self::Hz50 => [1, 1, 1],
            Self::Hz25 => [1, 1, 2],
            Self::Hz12_5 => [1, 2, 2],
        }
    }

    const fn manifest(self) -> [u8; 32] {
        match self {
            Self::Hz50 => MANIFEST_50HZ,
            Self::Hz25 => MANIFEST_25HZ,
            Self::Hz12_5 => MANIFEST_12_5HZ,
        }
    }

    fn parse(tag: &str) -> Result<Self> {
        match tag {
            "50hz" => Ok(Self::Hz50),
            "25hz" => Ok(Self::Hz25),
            "12_5hz" => Ok(Self::Hz12_5),
            _ => Err(VokraError::ModelLoad(format!(
                "{LABEL}: unsupported `{KEY_VARIANT}`={tag:?}; expected 50hz, 25hz or 12_5hz"
            ))),
        }
    }

    const fn spec(self) -> StrictCheckpointSpec {
        StrictCheckpointSpec {
            label: LABEL,
            arch: ARCH,
            model_name: self.model_name(),
            model_name_alias: None,
            tensor_count: TENSOR_COUNT,
            manifest_sha256: self.manifest(),
        }
    }
}

/// Fully bound native FocalCodec model.
#[derive(Debug, Clone)]
pub struct FocalCodec {
    variant: FocalCodecVariant,
    weights: FocalCodecWeights,
    backend: BackendKind,
    corrected_legacy_variant: bool,
}

impl FocalCodec {
    /// Strictly binds all metadata, all 354 tensor names/shapes and payloads.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        let stamped_variant = file.get(KEY_VARIANT).and_then(GgufMetadataValue::as_str);
        let (variant, corrected_legacy_variant) = match stamped_variant {
            Some(tag) => (FocalCodecVariant::parse(tag)?, false),
            None => {
                let name = required_string(file, chunks::KEY_MODEL_NAME)?;
                let upstream = required_string(file, KEY_UPSTREAM_HF)?;
                if name == FocalCodecVariant::Hz50.model_name()
                    && upstream == FocalCodecVariant::Hz50.upstream_hf()
                {
                    (FocalCodecVariant::Hz50, true)
                } else {
                    return Err(VokraError::ModelLoad(format!(
                        "{LABEL}: missing `{KEY_VARIANT}` is accepted only for the pinned legacy focalcodec-50hz publication"
                    )));
                }
            }
        };

        let checkpoint = StrictCheckpoint::bind(file, variant.spec())?;
        require_string(file, "vokra.model.category", CATEGORY)?;
        require_string(file, KEY_UPSTREAM_HF, variant.upstream_hf())?;
        require_string(file, chunks::KEY_PROVENANCE_MODEL_ID, variant.model_name())?;
        require_string(file, chunks::KEY_PROVENANCE_LICENSE, "apache-2.0")?;
        if checkpoint.weight_license() != LicenseClass::Permissive {
            return Err(VokraError::ModelLoad(format!(
                "{LABEL}: official Apache-2.0 checkpoint must carry `permissive` weight license, got {:?}",
                checkpoint.weight_license()
            )));
        }
        debug_assert_eq!(checkpoint.tensor_count(), TENSOR_COUNT);

        let weights = weights::bind(file, variant)?;
        Ok(Self {
            variant,
            weights,
            backend: BackendKind::Cpu,
            corrected_legacy_variant,
        })
    }

    /// Selects one backend for every learned operation in the model.
    #[must_use]
    pub fn with_backend(mut self, backend: BackendKind) -> Self {
        self.backend = backend;
        self
    }

    /// Selected execution backend.
    #[must_use]
    pub const fn backend(&self) -> BackendKind {
        self.backend
    }

    /// Audited release variant.
    #[must_use]
    pub const fn variant(&self) -> FocalCodecVariant {
        self.variant
    }

    /// Model PCM sample rate.
    #[must_use]
    pub const fn sample_rate(&self) -> u32 {
        SAMPLE_RATE
    }

    /// PCM samples represented by one token.
    #[must_use]
    pub const fn frame_hop(&self) -> usize {
        self.variant.frame_hop()
    }

    /// Number of entries in the released binary spherical codebook.
    #[must_use]
    pub const fn codebook_size(&self) -> usize {
        CODEBOOK_SIZE
    }

    /// Number of binary dimensions represented by each token.
    #[must_use]
    pub const fn code_dimension(&self) -> usize {
        CODE_DIM
    }

    /// Whether the exact old 50 Hz publication without a variant tag was
    /// repaired after matching its complete manifest and identity tuple.
    #[must_use]
    pub const fn corrected_legacy_variant(&self) -> bool {
        self.corrected_legacy_variant
    }

    /// Encodes mono 16 kHz PCM into one 8,192-entry BSQ token stream.
    pub fn encode(&self, pcm: &[f32]) -> Result<Vec<u32>> {
        validate_pcm(pcm)?;
        wavlm::encode_tokens(
            pcm,
            &self.weights.wavlm,
            &self.weights.compressor,
            self.variant.downscale_factors(),
            self.backend,
        )
    }

    /// Decodes a non-empty BSQ token stream to mono 16 kHz PCM.
    pub fn decode(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        if tokens.is_empty() {
            return Err(VokraError::InvalidArgument(
                "focalcodec: token stream is empty".to_owned(),
            ));
        }
        if let Some((index, token)) = tokens
            .iter()
            .copied()
            .enumerate()
            .find(|(_, token)| *token as usize >= CODEBOOK_SIZE)
        {
            return Err(VokraError::InvalidArgument(format!(
                "focalcodec: tokens[{index}]={token} is outside 0..{CODEBOOK_SIZE}"
            )));
        }
        focal::decode_tokens(
            tokens,
            &self.weights.decompressor,
            &self.weights.vocos,
            self.variant.upscale_factors(),
            self.backend,
        )
    }

    /// Runs encode followed by decode with the same checkpoint.
    pub fn reconstruct(&self, pcm: &[f32]) -> Result<(Vec<u32>, Vec<f32>)> {
        let tokens = self.encode(pcm)?;
        let reconstructed = self.decode(&tokens)?;
        Ok((tokens, reconstructed))
    }
}

fn validate_pcm(pcm: &[f32]) -> Result<()> {
    if pcm.is_empty() {
        return Err(VokraError::InvalidArgument(
            "focalcodec: PCM input is empty".to_owned(),
        ));
    }
    if let Some((index, value)) = pcm
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(VokraError::InvalidArgument(format!(
            "focalcodec: PCM value at index {index} is non-finite ({value})"
        )));
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

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::{GgmlType, GgufBuilder};

    #[test]
    fn official_variant_axes_are_pinned() {
        assert_eq!(FocalCodecVariant::Hz50.downscale_factors(), [1, 1, 1]);
        assert_eq!(FocalCodecVariant::Hz25.downscale_factors(), [2, 1, 1]);
        assert_eq!(FocalCodecVariant::Hz12_5.downscale_factors(), [2, 2, 1]);
        assert_eq!(FocalCodecVariant::Hz50.upscale_factors(), [1, 1, 1]);
        assert_eq!(FocalCodecVariant::Hz25.upscale_factors(), [1, 1, 2]);
        assert_eq!(FocalCodecVariant::Hz12_5.upscale_factors(), [1, 2, 2]);
        assert_eq!(FocalCodecVariant::Hz12_5.frame_hop(), 1_280);
    }

    #[test]
    fn unknown_variant_fails_before_tensor_decode() {
        let mut builder = GgufBuilder::new();
        builder.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        builder.add_string(chunks::KEY_MODEL_NAME, "focalcodec-future");
        builder.add_string(KEY_VARIANT, "future");
        let file = GgufFile::parse(builder.to_bytes().unwrap()).unwrap();
        let error = FocalCodec::from_gguf(&file).unwrap_err().to_string();
        assert!(error.contains("unsupported `vokra.focalcodec.variant`"));
    }

    #[test]
    fn missing_variant_is_not_generically_defaulted() {
        let mut builder = GgufBuilder::new();
        builder.add_string(chunks::KEY_MODEL_ARCH, ARCH);
        builder.add_string(chunks::KEY_MODEL_NAME, "focalcodec-25hz");
        builder.add_string(KEY_UPSTREAM_HF, "lucadellalib/focalcodec_25hz");
        builder
            .add_tensor("probe", GgmlType::F32, vec![1], vec![0; 4])
            .unwrap();
        let file = GgufFile::parse(builder.to_bytes().unwrap()).unwrap();
        let error = FocalCodec::from_gguf(&file).unwrap_err().to_string();
        assert!(error.contains("missing `vokra.focalcodec.variant`"));
    }
}
