//! Strict official checkpoint binding for Microsoft VibeVoice-1.5B.

use vokra_core::gguf::GgufFile;
use vokra_core::{LicenseClass, Result, VokraError};

use crate::strict_checkpoint::{
    StrictCheckpoint, StrictCheckpointSpec, linear_rows, load_tensor, require_tensor_shape,
};

const LABEL: &str = "vibevoice";
const INPUT_DIM: usize = 64;
const OUTPUT_DIM: usize = 1_536;
const WEIGHT: &str = "model.acoustic_connector.fc1.weight";
const BIAS: &str = "model.acoustic_connector.fc1.bias";
const SPEC: StrictCheckpointSpec = StrictCheckpointSpec {
    label: LABEL,
    arch: "vibevoice",
    model_name: "vibevoice-1.5b",
    model_name_alias: None,
    tensor_count: 1_204,
    manifest_sha256: [
        0x45, 0xcb, 0x01, 0x14, 0x20, 0xfd, 0xb1, 0x14, 0xc7, 0xad, 0x61, 0xd8, 0x08, 0x88, 0x66,
        0x3b, 0xcc, 0x86, 0x1e, 0x33, 0xb7, 0x94, 0x58, 0x73, 0x83, 0x6a, 0xee, 0x24, 0x50, 0xeb,
        0x57, 0x02,
    ],
};

/// Strict handle for `vokra/vibevoice-1.5b`.
#[derive(Debug, Clone)]
pub struct VibeVoiceCheckpoint {
    checkpoint: StrictCheckpoint,
}

impl VibeVoiceCheckpoint {
    /// Validates model identity and all 1,204 official tensor names and shapes.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        let checkpoint = StrictCheckpoint::bind(file, SPEC)?;
        require_tensor_shape(file, LABEL, WEIGHT, &[OUTPUT_DIM, INPUT_DIM])?;
        require_tensor_shape(file, LABEL, BIAS, &[OUTPUT_DIM])?;
        Ok(Self { checkpoint })
    }

    /// Decodes the real first acoustic connector projection.
    pub fn load_acoustic_projection(&self, file: &GgufFile) -> Result<VibeVoiceAcousticProjection> {
        Ok(VibeVoiceAcousticProjection {
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

    /// End-to-end PCM stays loud until LM/diffusion/tokenizer paths are bound.
    pub fn synthesize(&self, text: &str) -> Result<Vec<f32>> {
        if text.is_empty() {
            return Err(VokraError::InvalidArgument(
                "vibevoice synthesize: text is empty".to_owned(),
            ));
        }
        Err(VokraError::NotImplemented(
            "vibevoice synthesize: the complete official 1.5B checkpoint is bound and the real acoustic connector projection runs natively, but full Qwen2 decoding, diffusion-head sampling and acoustic-tokenizer decode remain pending.",
        ))
    }
}

/// Real VibeVoice acoustic latent-to-decoder projection.
#[derive(Debug, Clone)]
pub struct VibeVoiceAcousticProjection {
    weight: Vec<f32>,
    bias: Vec<f32>,
}

impl VibeVoiceAcousticProjection {
    /// Applies `model.acoustic_connector.fc1`.
    pub fn forward(&self, acoustic_latent: &[f32]) -> Result<Vec<f32>> {
        linear_rows(
            "vibevoice acoustic connector fc1",
            acoustic_latent,
            &self.weight,
            Some(&self.bias),
            INPUT_DIM,
            OUTPUT_DIM,
        )
    }
}
