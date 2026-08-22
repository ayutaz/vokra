//! Strict official checkpoint binding for Zyphra Zonos-v0.1-transformer.

use vokra_core::gguf::GgufFile;
use vokra_core::{LicenseClass, Result, VokraError};

use crate::strict_checkpoint::{
    StrictCheckpoint, StrictCheckpointSpec, linear_rows, load_tensor, require_tensor_shape,
};

const LABEL: &str = "zonos";
const INPUT_DIM: usize = 128;
const OUTPUT_DIM: usize = 2_048;
const WEIGHT: &str = "prefix_conditioner.conditioners.1.project.weight";
const BIAS: &str = "prefix_conditioner.conditioners.1.project.bias";
const SPEC: StrictCheckpointSpec = StrictCheckpointSpec {
    label: LABEL,
    arch: "zonos",
    model_name: "zonos-v0.1",
    model_name_alias: None,
    tensor_count: 246,
    manifest_sha256: [
        0x65, 0x43, 0xaf, 0x37, 0x47, 0xd3, 0xe8, 0x5b, 0xde, 0x86, 0x2c, 0x33, 0x37, 0x74, 0x4e,
        0xea, 0x31, 0xf0, 0x10, 0x5f, 0x9d, 0xf6, 0xd8, 0x61, 0x7c, 0x1c, 0x9a, 0xfd, 0xae, 0x80,
        0x58, 0x47,
    ],
};

/// Strict handle for `vokra/zonos-v0.1-transformer`.
#[derive(Debug, Clone)]
pub struct ZonosCheckpoint {
    checkpoint: StrictCheckpoint,
}

impl ZonosCheckpoint {
    /// Validates identity and all 246 official tensor names and shapes.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        let checkpoint = StrictCheckpoint::bind(file, SPEC)?;
        require_tensor_shape(file, LABEL, WEIGHT, &[OUTPUT_DIM, INPUT_DIM])?;
        require_tensor_shape(file, LABEL, BIAS, &[OUTPUT_DIM])?;
        Ok(Self { checkpoint })
    }

    /// Decodes the real 128-to-2048 speaker projection.
    pub fn load_speaker_projection(&self, file: &GgufFile) -> Result<ZonosSpeakerProjection> {
        Ok(ZonosSpeakerProjection {
            weight: load_tensor(file, LABEL, WEIGHT, &[OUTPUT_DIM, INPUT_DIM])?,
            bias: load_tensor(file, LABEL, BIAS, &[OUTPUT_DIM])?,
        })
    }

    /// Returns the pinned model name.
    #[must_use]
    pub fn model_name(&self) -> &str {
        self.checkpoint.model_name()
    }

    /// Returns the fail-closed stamped weight-license class.
    #[must_use]
    pub const fn weight_license(&self) -> LicenseClass {
        self.checkpoint.weight_license()
    }

    /// Returns the complete manifest tensor count.
    #[must_use]
    pub const fn tensor_count(&self) -> usize {
        self.checkpoint.tensor_count()
    }

    /// End-to-end PCM stays loud until delayed-AR and DAC paths are bound.
    pub fn synthesize(&self, phoneme_ids: &[i64]) -> Result<Vec<f32>> {
        if phoneme_ids.is_empty() {
            return Err(VokraError::InvalidArgument(
                "zonos synthesize: phoneme_ids is empty".to_owned(),
            ));
        }
        Err(VokraError::NotImplemented(
            "zonos synthesize: the complete official transformer checkpoint is bound and the real speaker-conditioner projection runs natively, but all prefix conditioners, delayed nine-codebook autoregression and the separately distributed DAC decoder remain pending.",
        ))
    }
}

/// Real Zonos speaker-conditioner projection.
#[derive(Debug, Clone)]
pub struct ZonosSpeakerProjection {
    weight: Vec<f32>,
    bias: Vec<f32>,
}

impl ZonosSpeakerProjection {
    /// Applies `prefix_conditioner.conditioners.1.project`.
    pub fn forward(&self, speaker_embedding: &[f32]) -> Result<Vec<f32>> {
        linear_rows(
            "zonos speaker projection",
            speaker_embedding,
            &self.weight,
            Some(&self.bias),
            INPUT_DIM,
            OUTPUT_DIM,
        )
    }
}
