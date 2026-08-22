//! Strict binding for the official Chatterbox multilingual v3 T3 checkpoint.

use vokra_core::gguf::GgufFile;
use vokra_core::{LicenseClass, Result, VokraError};

use crate::strict_checkpoint::{
    StrictCheckpoint, StrictCheckpointSpec, linear_rows, load_tensor, require_tensor_shape,
};

const LABEL: &str = "chatterbox";
const INPUT_DIM: usize = 256;
const OUTPUT_DIM: usize = 1_024;
const SPEC: StrictCheckpointSpec = StrictCheckpointSpec {
    label: LABEL,
    arch: "chatterbox",
    model_name: "chatterbox-multilingual-v3",
    model_name_alias: None,
    tensor_count: 292,
    manifest_sha256: [
        0x4c, 0x62, 0xa9, 0x0e, 0x62, 0x41, 0x76, 0x5f, 0x74, 0x2f, 0x27, 0x91, 0x7a, 0xc0, 0x5c,
        0x08, 0xf6, 0x66, 0x23, 0xf0, 0xb1, 0x76, 0x8e, 0x48, 0xef, 0xd7, 0xf7, 0xf6, 0xbb, 0xf8,
        0x4c, 0x79,
    ],
};

/// Strict handle for `vokra/chatterbox-multilingual-v3`.
#[derive(Debug, Clone)]
pub struct ChatterboxCheckpoint {
    checkpoint: StrictCheckpoint,
}

impl ChatterboxCheckpoint {
    /// Validates identity and all 292 upstream tensor names and shapes.
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
    pub fn load_speaker_projection(&self, file: &GgufFile) -> Result<ChatterboxSpeakerProjection> {
        Ok(ChatterboxSpeakerProjection {
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

    /// End-to-end PCM stays loud until AR sampling and S3Gen are bound.
    pub fn synthesize(&self, text: &str) -> Result<Vec<f32>> {
        if text.is_empty() {
            return Err(VokraError::InvalidArgument(
                "chatterbox synthesize: text is empty".to_owned(),
            ));
        }
        Err(VokraError::NotImplemented(
            "chatterbox synthesize: the complete official v3 T3 checkpoint is bound and the real speaker projection runs natively, but text tokenization, T3 autoregressive speech-token sampling, S3Gen and HiFTNet PCM generation remain pending.",
        ))
    }
}

/// Real Chatterbox speaker conditioning projection.
#[derive(Debug, Clone)]
pub struct ChatterboxSpeakerProjection {
    weight: Vec<f32>,
    bias: Vec<f32>,
}

impl ChatterboxSpeakerProjection {
    /// Applies the official `cond_enc.spkr_enc` PyTorch linear layer.
    pub fn forward(&self, speaker_embedding: &[f32]) -> Result<Vec<f32>> {
        linear_rows(
            "chatterbox speaker projection",
            speaker_embedding,
            &self.weight,
            Some(&self.bias),
            INPUT_DIM,
            OUTPUT_DIM,
        )
    }
}
