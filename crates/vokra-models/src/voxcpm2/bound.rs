//! Strict official checkpoint binding for openbmb VoxCPM-0.5B.

use vokra_core::gguf::GgufFile;
use vokra_core::{LicenseClass, Result, VokraError};

use crate::strict_checkpoint::{
    StrictCheckpoint, StrictCheckpointSpec, linear_rows, load_tensor, require_tensor_shape,
};

const LABEL: &str = "voxcpm2";
const DIMENSION: usize = 1_024;
const WEIGHT: &str = "stop_proj.weight";
const BIAS: &str = "stop_proj.bias";
const SPEC: StrictCheckpointSpec = StrictCheckpointSpec {
    label: LABEL,
    arch: "voxcpm2",
    // The published artifact predates the 2026-07-30 converter rename.
    model_name: "voxcpm-0.5b",
    model_name_alias: Some("voxcpm2-0.5b"),
    tensor_count: 377,
    manifest_sha256: [
        0xd3, 0x64, 0x68, 0x9d, 0x55, 0x93, 0xed, 0x88, 0x86, 0x02, 0x99, 0x07, 0xa5, 0xd1, 0x7e,
        0x76, 0x59, 0xb9, 0x4f, 0x7f, 0x31, 0x0f, 0xe9, 0x5b, 0x13, 0x3c, 0x54, 0x5b, 0x69, 0x01,
        0xc5, 0x09,
    ],
};

/// Strict handle for `vokra/voxcpm-0.5b`.
#[derive(Debug, Clone)]
pub struct VoxCpm2Checkpoint {
    checkpoint: StrictCheckpoint,
}

impl VoxCpm2Checkpoint {
    /// Validates identity and all 377 official tensor names and shapes.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        let checkpoint = StrictCheckpoint::bind(file, SPEC)?;
        require_tensor_shape(file, LABEL, WEIGHT, &[DIMENSION, DIMENSION])?;
        require_tensor_shape(file, LABEL, BIAS, &[DIMENSION])?;
        Ok(Self { checkpoint })
    }

    /// Decodes the real MiniCPM stop-state projection.
    pub fn load_stop_projection(&self, file: &GgufFile) -> Result<VoxCpm2StopProjection> {
        Ok(VoxCpm2StopProjection {
            weight: load_tensor(file, LABEL, WEIGHT, &[DIMENSION, DIMENSION])?,
            bias: load_tensor(file, LABEL, BIAS, &[DIMENSION])?,
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

    /// End-to-end PCM stays loud: this is a historical main-checkpoint
    /// diagnostic only, not a complete AudioVAE/tokenizer runtime. A
    /// source-shaped batch-one route exists internally, but this old
    /// main-only artifact cannot authorize it without the immutable complete
    /// composite manifest and independent parity evidence.
    pub fn synthesize(&self, text: &str) -> Result<Vec<f32>> {
        if text.is_empty() {
            return Err(VokraError::InvalidArgument(
                "voxcpm2 synthesize: text is empty".to_owned(),
            ));
        }
        Err(VokraError::NotImplemented(
            "voxcpm2 synthesize: historical main-checkpoint stop projection is available only as a partial diagnostic; the source-shaped batch-one route exists internally, but immutable complete-composite manifest, AudioVAE/tokenizer/provenance authentication, and independent CPU/Metal parity remain INSPECTION_ONLY blockers.",
        ))
    }
}

/// Real VoxCPM stop-state projection.
#[derive(Debug, Clone)]
pub struct VoxCpm2StopProjection {
    weight: Vec<f32>,
    bias: Vec<f32>,
}

impl VoxCpm2StopProjection {
    /// Applies the official `stop_proj` linear layer.
    pub fn forward(&self, hidden: &[f32]) -> Result<Vec<f32>> {
        linear_rows(
            "voxcpm2 stop projection",
            hidden,
            &self.weight,
            Some(&self.bias),
            DIMENSION,
            DIMENSION,
        )
    }
}
