//! Native X-Codec2 token-to-waveform decoder.
//!
//! The audited public `vokra/xcodec2` GGUF preserves the official
//! `xcodec2==0.1.5` state-dict names. Decode is the released 65,536-way FSQ
//! projection followed by the same 1024-wide, twelve-layer Transformer/Vocos
//! topology used by distilled NeuCodec, with X-Codec2's distinct 16 kHz / 320
//! sample timebase. The official PyPI source distribution is pinned by
//! SHA-256 in the parity oracle; the public GGUF's complete 1,153-tensor
//! name/shape manifest is pinned here.
//!
//! Every learned operation routes through [`Compute`]. CPU is the scalar
//! oracle and Apple Metal uses the existing FSQ, convolution, normalization,
//! GEMM and softmax kernels. Unsupported backends fail before execution; no
//! CPU fallback is substituted. The weights are CC-BY-NC-4.0 and remain
//! subject to Vokra's explicit research-license gate.

use vokra_core::backend::BackendKind;
use vokra_core::gguf::{GgufFile, GgufMetadataValue, chunks};
use vokra_core::{LicenseClass, Result, VokraError};

use crate::compute::HotOp;
use crate::neucodec::{
    DecoderWeights, FSQ_VOCOS_DECODE_HOT_OPS, decode_fsq_vocos, load_pass_through_decoder,
};
use crate::strict_checkpoint::{StrictCheckpoint, StrictCheckpointSpec};

/// Converter/runtime architecture tag for the audited public artifact.
pub const ARCH: &str = "xcodec2";
/// X-Codec2's released waveform sample rate.
pub const SAMPLE_RATE: u32 = 16_000;
/// Samples represented by one 50 Hz code.
pub const HOP_LENGTH: usize = 320;
/// Product of the released `[4; 8]` FSQ levels.
pub const CODEBOOK_SIZE: usize = 65_536;

/// Complete learned-op set for official token-to-waveform execution.
pub const XCODEC2_DECODE_HOT_OPS: &[HotOp] = FSQ_VOCOS_DECODE_HOT_OPS;

const SPEC: StrictCheckpointSpec = StrictCheckpointSpec {
    label: "xcodec2",
    arch: ARCH,
    model_name: "xcodec2",
    model_name_alias: None,
    tensor_count: 1_153,
    manifest_sha256: [
        0xee, 0x54, 0x3e, 0x96, 0xb5, 0x15, 0x03, 0x76, 0x10, 0x13, 0x96, 0x19, 0x7b, 0xb0, 0xad,
        0xd5, 0x3d, 0xaf, 0x91, 0x3e, 0xb9, 0x91, 0xde, 0xb4, 0x2a, 0xad, 0x7b, 0xe7, 0x4e, 0xed,
        0x33, 0xf5,
    ],
};

/// Strict real-weight X-Codec2 token-to-PCM model.
#[derive(Debug, Clone)]
pub struct XCodec2 {
    weights: DecoderWeights,
    weight_license: LicenseClass,
    backend: BackendKind,
}

impl XCodec2 {
    /// Binds the audited `vokra/xcodec2` public GGUF.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        let checkpoint = StrictCheckpoint::bind(file, SPEC)?;
        require_string(
            file,
            chunks::KEY_PROVENANCE_MODEL_ID,
            checkpoint.model_name(),
        )?;
        require_string(file, "vokra.provenance.upstream_hf", "HKUSTAudio/xcodec2")?;
        require_string(file, chunks::KEY_PROVENANCE_LICENSE, "cc-by-nc-4.0")?;
        require_string(
            file,
            chunks::KEY_PROVENANCE_WEIGHT_LICENSE,
            LicenseClass::NonCommercial.as_str(),
        )?;
        let weights = load_pass_through_decoder(file, "xcodec2", HOP_LENGTH)?;
        Ok(Self {
            weights,
            weight_license: checkpoint.weight_license(),
            backend: BackendKind::Cpu,
        })
    }

    /// Opens and binds an official GGUF. The CLI session path uses mmap;
    /// this convenience entry preserves the core buffered-reader semantics.
    pub fn open(path: impl AsRef<std::path::Path>) -> Result<Self> {
        Self::from_gguf(&GgufFile::open(path)?)
    }

    /// Selects one backend for the complete decoder graph.
    #[must_use]
    pub fn with_backend(mut self, backend: BackendKind) -> Self {
        self.backend = backend;
        self
    }

    /// Selected inference backend.
    #[must_use]
    pub const fn backend(&self) -> BackendKind {
        self.backend
    }

    /// Stamped public artifact license class.
    #[must_use]
    pub const fn weight_license(&self) -> LicenseClass {
        self.weight_license
    }

    /// Output sample rate.
    #[must_use]
    pub const fn sample_rate(&self) -> u32 {
        SAMPLE_RATE
    }

    /// Samples represented by one code.
    #[must_use]
    pub const fn hop_length(&self) -> usize {
        HOP_LENGTH
    }

    /// Decodes one batch-free `[frames]` FSQ code sequence to 16 kHz PCM.
    pub fn decode_codes(&self, codes: &[u32]) -> Result<Vec<f32>> {
        decode_fsq_vocos(&self.weights, self.backend, codes, HOP_LENGTH, "xcodec2")
    }
}

fn require_string(file: &GgufFile, key: &str, expected: &str) -> Result<()> {
    let actual = file.get(key).and_then(GgufMetadataValue::as_str);
    if actual != Some(expected) {
        return Err(VokraError::ModelLoad(format!(
            "xcodec2: metadata `{key}`={actual:?}, expected {expected:?}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::Compute;

    #[test]
    fn decoder_hot_ops_are_cpu_and_metal_complete() {
        Compute::for_backend(BackendKind::Cpu, XCODEC2_DECODE_HOT_OPS)
            .expect("CPU covers the complete X-Codec2 decoder");
        #[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
        match Compute::for_backend(BackendKind::Metal, XCODEC2_DECODE_HOT_OPS) {
            Ok(compute) => assert_eq!(compute.backend_name(), "metal"),
            Err(VokraError::BackendUnavailable(_)) => {}
            Err(error) => panic!("X-Codec2 decode has a Metal coverage gap: {error}"),
        }
    }

    #[test]
    fn public_decoder_constants_are_exact() {
        assert_eq!(SAMPLE_RATE as usize / HOP_LENGTH, 50);
        assert_eq!(4usize.pow(8), CODEBOOK_SIZE);
        assert_eq!(SPEC.tensor_count, 1_153);
        assert_eq!(
            SPEC.manifest_sha256,
            [
                0xee, 0x54, 0x3e, 0x96, 0xb5, 0x15, 0x03, 0x76, 0x10, 0x13, 0x96, 0x19, 0x7b, 0xb0,
                0xad, 0xd5, 0x3d, 0xaf, 0x91, 0x3e, 0xb9, 0x91, 0xde, 0xb4, 0x2a, 0xad, 0x7b, 0xe7,
                0x4e, 0xed, 0x33, 0xf5,
            ]
        );
    }
}
