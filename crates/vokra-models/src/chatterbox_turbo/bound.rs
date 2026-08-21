//! Strict binding for the official Chatterbox Turbo v1 T3 checkpoint.

use vokra_core::gguf::GgufFile;
use vokra_core::{LicenseClass, Result, VokraError};

use crate::strict_checkpoint::{
    StrictCheckpoint, StrictCheckpointSpec, linear_rows, load_tensor, require_tensor_shape,
};

const LABEL: &str = "chatterbox_turbo";
const INPUT_DIM: usize = 256;
const OUTPUT_DIM: usize = 1_024;
const SPEC: StrictCheckpointSpec = StrictCheckpointSpec {
    label: LABEL,
    arch: "chatterbox_turbo",
    model_name: "chatterbox-turbo-v1",
    model_name_alias: None,
    tensor_count: 299,
    manifest_sha256: [
        0xc2, 0x1c, 0xfd, 0x33, 0x6c, 0xb9, 0xb0, 0xf7, 0x01, 0x79, 0xfc, 0xf2, 0x30, 0x8e, 0xc6,
        0x6e, 0x23, 0x9a, 0x82, 0x72, 0x19, 0x34, 0x64, 0xba, 0x2a, 0x78, 0x25, 0x1d, 0x81, 0x82,
        0xb8, 0x80,
    ],
};

/// Strict handle for `vokra/chatterbox-turbo-v1`.
#[derive(Debug, Clone)]
pub struct ChatterboxTurboCheckpoint {
    checkpoint: StrictCheckpoint,
}

impl ChatterboxTurboCheckpoint {
    /// Validates identity and all 299 upstream tensor names and shapes.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        let checkpoint = StrictCheckpoint::bind(file, SPEC)?;
        require_tensor_shape(
            file,
            LABEL,
            "cond_enc.spkr_enc.weight",
            &[OUTPUT_DIM, INPUT_DIM],
        )?;
        require_tensor_shape(file, LABEL, "cond_enc.spkr_enc.bias", &[OUTPUT_DIM])?;
        Ok(Self { checkpoint })
    }

    /// Lazily decodes the real 256-to-1024 speaker conditioning projection.
    pub fn load_speaker_projection(
        &self,
        file: &GgufFile,
    ) -> Result<ChatterboxTurboSpeakerProjection> {
        Ok(ChatterboxTurboSpeakerProjection {
            weight: load_tensor(
                file,
                LABEL,
                "cond_enc.spkr_enc.weight",
                &[OUTPUT_DIM, INPUT_DIM],
            )?,
            bias: load_tensor(file, LABEL, "cond_enc.spkr_enc.bias", &[OUTPUT_DIM])?,
        })
    }

    #[must_use]
    /// Returns the pinned checkpoint variant.
    pub fn model_name(&self) -> &str {
        self.checkpoint.model_name()
    }

    #[must_use]
    /// Returns the fail-closed weight-license class stamped in the GGUF.
    pub const fn weight_license(&self) -> LicenseClass {
        self.checkpoint.weight_license()
    }

    #[must_use]
    /// Returns the number of tensors checked by the complete manifest gate.
    pub const fn tensor_count(&self) -> usize {
        self.checkpoint.tensor_count()
    }

    /// End-to-end PCM stays loud until GPT-2 sampling and S3Gen are bound.
    pub fn synthesize(&self, text: &str) -> Result<Vec<f32>> {
        if text.is_empty() {
            return Err(VokraError::InvalidArgument(
                "chatterbox_turbo synthesize: text is empty".to_owned(),
            ));
        }
        Err(VokraError::NotImplemented(
            "chatterbox_turbo synthesize: the complete official Turbo v1 checkpoint is bound and the real speaker projection runs natively, but GPT-2 speech-token sampling, one-step S3Gen and HiFTNet PCM generation remain pending.",
        ))
    }
}

/// Real Chatterbox Turbo speaker conditioning projection.
#[derive(Debug, Clone)]
pub struct ChatterboxTurboSpeakerProjection {
    weight: Vec<f32>,
    bias: Vec<f32>,
}

impl ChatterboxTurboSpeakerProjection {
    /// Applies the official `cond_enc.spkr_enc` PyTorch linear layer.
    pub fn forward(&self, speaker_embedding: &[f32]) -> Result<Vec<f32>> {
        linear_rows(
            "chatterbox_turbo speaker projection",
            speaker_embedding,
            &self.weight,
            Some(&self.bias),
            INPUT_DIM,
            OUTPUT_DIM,
        )
    }
}
