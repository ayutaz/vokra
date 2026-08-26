//! Native NaturalSpeech 3 FACodec V2 runtime (CPU / Metal).
//!
//! The official Amphion release is a 16 kHz factorized-VQ codec: a
//! weight-normalized convolutional encoder produces 256-channel frames at a
//! 200-sample hop, six 1,024-entry / 8-dimensional codebooks split those
//! frames into prosody (1), content (2), and detail (3), and a conditioned
//! convolutional decoder reconstructs waveform samples.  This implementation
//! binds the complete 806-tensor public Vokra manifest before decoding any
//! payload. Training-only prediction heads remain authenticated by that
//! manifest but are not executed by the official inference route.
//!
//! Learned hot operations are selected once through [`Compute`](crate::compute::Compute).
//! Metal selection covers GEMM, attention softmax/layer norm, convolution,
//! grouped anti-alias filters, SnakeBeta, factorized VQ decode, ReLU and tanh;
//! an unavailable operation is an explicit error and never a CPU fallback.

use vokra_core::backend::BackendKind;
use vokra_core::gguf::{GgufFile, GgufMetadataValue, chunks};
use vokra_core::{LicenseClass, Result, VokraError};

use crate::compute::{Compute, HotOp};
use crate::strict_checkpoint::{StrictCheckpoint, StrictCheckpointSpec};

mod nn;
mod weights;

use weights::FacodecWeights;

/// Converter/runtime architecture tag.
pub const ARCH: &str = "facodec";
/// Canonical public V2 model name.
pub const MODEL_NAME: &str = "naturalspeech3-facodec-v2";
/// Required category.
pub const CATEGORY: &str = "codec";
/// Official upstream checkpoint repository.
pub const UPSTREAM_HF: &str = "amphion/naturalspeech3_facodec";
/// Public V2 variant metadata key.
pub const KEY_VARIANT: &str = "vokra.facodec.variant";

pub(super) const LABEL: &str = "naturalspeech3-facodec-v2";
pub(super) const DIM: usize = 256;
pub(super) const CODEBOOK_SIZE: usize = 1_024;
pub(super) const CODEBOOK_DIM: usize = 8;
pub(super) const NUM_CODEBOOKS: usize = 6;
const SAMPLE_RATE: u32 = 16_000;
const FRAME_HOP: usize = 200;
const TENSOR_COUNT: usize = 806;
const PUBLIC_GGUF_REVISION: &str = "da6263e2c1a203641a5d4346a8a04d4eab4c738f";
const UPSTREAM_WEIGHT_REVISION: &str = "314afc3ea1455ba881a0e484ef9408b6cb996736";
const REFERENCE_SOURCE_REVISION: &str = "26f6883110181f1dbfe95c70a7c7dbaf4de5f42a";

const MANIFEST: [u8; 32] = [
    0xc8, 0x18, 0x98, 0x2c, 0x34, 0x66, 0x60, 0x1f, 0xcd, 0x57, 0x61, 0x3a, 0x8a, 0xc7, 0x59, 0xac,
    0xa6, 0x1c, 0x6e, 0xd3, 0x36, 0x2a, 0xcb, 0x68, 0xf8, 0x74, 0xd6, 0x31, 0x81, 0x03, 0x79, 0xfa,
];

/// Complete learned-operation set for the official V2 encode/decode graph.
pub const FACODEC_HOT_OPS: &[HotOp] = &[
    HotOp::Gemm,
    HotOp::Softmax,
    HotOp::LayerNorm,
    HotOp::Relu,
    HotOp::Tanh,
    HotOp::Conv1d,
    HotOp::GroupedConv1d,
    HotOp::SnakeBeta,
    HotOp::DacRvq,
];

/// Portable result of FACodec V2 analysis.
///
/// `codes` is frame-major `[frames, 6]` in the official order: prosody,
/// content-0, content-1, detail-0, detail-1, detail-2. The 256-value speaker
/// embedding is required by the timbre-conditioned decoder.
#[derive(Debug, Clone, PartialEq)]
pub struct FacodecEncoded {
    /// Number of 80 Hz codec frames.
    pub frames: usize,
    /// Frame-major six-codebook indices.
    pub codes: Vec<u32>,
    /// Mean-pooled official timbre encoder output.
    pub speaker_embedding: Vec<f32>,
    /// Original PCM sample count, retained by portable containers for audit
    /// and optional caller-side alignment. The decoder emits `frames * 200`.
    pub input_samples: usize,
}

impl FacodecEncoded {
    pub(super) fn validate(&self) -> Result<()> {
        if self.frames == 0 {
            return Err(VokraError::InvalidArgument(format!(
                "{LABEL}: encoded frame count is zero"
            )));
        }
        let expected = self.frames.checked_mul(NUM_CODEBOOKS).ok_or_else(|| {
            VokraError::InvalidArgument(format!("{LABEL}: encoded code count overflows usize"))
        })?;
        if self.codes.len() != expected || self.speaker_embedding.len() != DIM {
            return Err(VokraError::InvalidArgument(format!(
                "{LABEL}: encoded buffers have codes={} speaker={}, expected {expected} and {DIM}",
                self.codes.len(),
                self.speaker_embedding.len()
            )));
        }
        let decoded_extent = self.frames.checked_mul(FRAME_HOP).ok_or_else(|| {
            VokraError::InvalidArgument(format!("{LABEL}: decoded PCM extent overflows usize"))
        })?;
        if self.input_samples < decoded_extent || self.input_samples - decoded_extent >= FRAME_HOP {
            return Err(VokraError::InvalidArgument(format!(
                "{LABEL}: input_samples={} is inconsistent with {} frames at hop {FRAME_HOP} (decoded extent {decoded_extent})",
                self.input_samples, self.frames
            )));
        }
        if let Some((position, code)) = self
            .codes
            .iter()
            .copied()
            .enumerate()
            .find(|(_, code)| *code as usize >= CODEBOOK_SIZE)
        {
            return Err(VokraError::InvalidArgument(format!(
                "{LABEL}: codes[{position}]={code} is outside 0..{CODEBOOK_SIZE}"
            )));
        }
        if let Some((position, value)) = self
            .speaker_embedding
            .iter()
            .copied()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(VokraError::InvalidArgument(format!(
                "{LABEL}: speaker_embedding[{position}] is non-finite ({value})"
            )));
        }
        Ok(())
    }
}

/// Fully bound official NaturalSpeech 3 FACodec V2 model.
#[derive(Debug, Clone)]
pub struct FacodecV2 {
    weights: FacodecWeights,
    backend: BackendKind,
}

impl FacodecV2 {
    /// Strictly authenticates metadata plus all 806 tensor names/shapes and
    /// binds every tensor used by the official inference graph.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        Self::from_gguf_with_backend(file, BackendKind::Cpu)
    }

    /// Binds the public V2 artifact and preflights complete backend coverage
    /// before materialising any tensor payload.
    pub fn from_gguf_with_backend(file: &GgufFile, backend: BackendKind) -> Result<Self> {
        let _ = Compute::for_backend(backend, FACODEC_HOT_OPS)?;
        let checkpoint = StrictCheckpoint::bind(
            file,
            StrictCheckpointSpec {
                label: LABEL,
                arch: ARCH,
                model_name: MODEL_NAME,
                model_name_alias: None,
                tensor_count: TENSOR_COUNT,
                manifest_sha256: MANIFEST,
            },
        )?;
        require_string(file, "vokra.model.category", CATEGORY)?;
        require_string(file, KEY_VARIANT, "v2")?;
        require_string(file, "vokra.provenance.upstream_hf", UPSTREAM_HF)?;
        require_string(file, chunks::KEY_PROVENANCE_MODEL_ID, MODEL_NAME)?;
        require_string(file, chunks::KEY_PROVENANCE_LICENSE, "apache-2.0")?;
        require_u64(file, "vokra.facodec.sample_rate", u64::from(SAMPLE_RATE))?;
        require_u64(file, "vokra.facodec.hop_size", FRAME_HOP as u64)?;
        require_u64(file, "vokra.facodec.n_quantizers_prosody", 1)?;
        require_u64(file, "vokra.facodec.n_quantizers_content", 2)?;
        require_u64(file, "vokra.facodec.n_quantizers_detail", 3)?;
        if checkpoint.weight_license() != LicenseClass::Permissive {
            return Err(VokraError::ModelLoad(format!(
                "{LABEL}: official Apache-2.0 checkpoint must carry permissive weight license, got {:?}",
                checkpoint.weight_license()
            )));
        }
        debug_assert_eq!(checkpoint.tensor_count(), TENSOR_COUNT);
        Ok(Self {
            weights: FacodecWeights::bind(file)?,
            backend,
        })
    }

    /// Selects one backend for the entire learned graph. Coverage is checked
    /// again at execution so an unavailable Metal device remains a loud error.
    pub fn with_backend(mut self, backend: BackendKind) -> Result<Self> {
        let _ = Compute::for_backend(backend, FACODEC_HOT_OPS)?;
        self.backend = backend;
        Ok(self)
    }

    /// Selected backend.
    #[must_use]
    pub const fn backend(&self) -> BackendKind {
        self.backend
    }

    /// Model sample rate.
    #[must_use]
    pub const fn sample_rate(&self) -> u32 {
        SAMPLE_RATE
    }

    /// PCM samples represented by one codec frame.
    #[must_use]
    pub const fn frame_hop(&self) -> usize {
        FRAME_HOP
    }

    /// Number of factorized codebooks.
    #[must_use]
    pub const fn num_codebooks(&self) -> usize {
        NUM_CODEBOOKS
    }

    /// Number of entries in every factorized codebook.
    #[must_use]
    pub const fn codebook_size(&self) -> usize {
        CODEBOOK_SIZE
    }

    /// Decoder-required timbre embedding width.
    #[must_use]
    pub const fn speaker_embedding_dim(&self) -> usize {
        DIM
    }

    /// Audited Vokra publication revision whose complete manifest is pinned.
    #[must_use]
    pub const fn public_gguf_revision(&self) -> &'static str {
        PUBLIC_GGUF_REVISION
    }

    /// Official upstream weight revision used by the parity worker.
    #[must_use]
    pub const fn upstream_weight_revision(&self) -> &'static str {
        UPSTREAM_WEIGHT_REVISION
    }

    /// Fixed official Amphion source revision used by the independent parity
    /// reference. This is evidence provenance, not a claim that old GGUF
    /// metadata contained a source-revision field.
    #[must_use]
    pub const fn reference_source_revision(&self) -> &'static str {
        REFERENCE_SOURCE_REVISION
    }

    /// Encodes mono 16 kHz PCM to six factorized code streams plus the
    /// decoder's required timbre embedding.
    pub fn encode(&self, pcm: &[f32]) -> Result<FacodecEncoded> {
        let compute = Compute::for_backend(self.backend, FACODEC_HOT_OPS)?;
        nn::encode(pcm, &self.weights, &compute)
    }

    /// Decodes a validated six-codebook packet to mono 16 kHz PCM.
    pub fn decode(&self, encoded: &FacodecEncoded) -> Result<Vec<f32>> {
        let compute = Compute::for_backend(self.backend, FACODEC_HOT_OPS)?;
        nn::decode(encoded, &self.weights.decoder, &compute)
    }

    /// Runs the complete official encode/decode reconstruction path.
    pub fn reconstruct(&self, pcm: &[f32]) -> Result<(FacodecEncoded, Vec<f32>)> {
        let encoded = self.encode(pcm)?;
        let reconstructed = self.decode(&encoded)?;
        Ok((encoded, reconstructed))
    }
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
    use super::*;
    use vokra_core::gguf::GgufBuilder;

    #[test]
    fn encoded_contract_rejects_bad_shapes_ranges_and_non_finite_style() {
        let mut packet = FacodecEncoded {
            frames: 1,
            codes: vec![0; NUM_CODEBOOKS],
            speaker_embedding: vec![0.0; DIM],
            input_samples: FRAME_HOP,
        };
        packet.validate().unwrap();
        packet.codes[2] = CODEBOOK_SIZE as u32;
        assert!(
            packet
                .validate()
                .unwrap_err()
                .to_string()
                .contains("outside")
        );
        packet.codes[2] = 0;
        packet.speaker_embedding[7] = f32::NAN;
        assert!(
            packet
                .validate()
                .unwrap_err()
                .to_string()
                .contains("non-finite")
        );
        packet.speaker_embedding[7] = 0.0;
        packet.input_samples = 2 * FRAME_HOP;
        assert!(
            packet
                .validate()
                .unwrap_err()
                .to_string()
                .contains("inconsistent")
        );
    }

    #[test]
    fn wrong_arch_fails_before_payload_binding() {
        let mut builder = GgufBuilder::new();
        builder.add_string(chunks::KEY_MODEL_ARCH, "mimi");
        builder.add_string(chunks::KEY_MODEL_NAME, MODEL_NAME);
        let file = GgufFile::parse(builder.to_bytes().unwrap()).unwrap();
        let error = FacodecV2::from_gguf(&file).unwrap_err().to_string();
        assert!(error.contains("unsupported `vokra.model.arch`"));
    }

    #[test]
    fn release_revisions_and_axes_are_pinned() {
        assert_eq!(PUBLIC_GGUF_REVISION.len(), 40);
        assert_eq!(UPSTREAM_WEIGHT_REVISION.len(), 40);
        assert_eq!(REFERENCE_SOURCE_REVISION.len(), 40);
        assert_eq!(NUM_CODEBOOKS, 1 + 2 + 3);
        assert_eq!(SAMPLE_RATE as usize / FRAME_HOP, 80);
    }
}
