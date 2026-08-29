//! Strict binding for the authenticated Chatterbox Nano v1 T3 slice.
//!
//! The historical public artifact contains T3-only tensors. PCM remains
//! fail-closed until the full composite pipeline is authenticated.

use vokra_core::gguf::GgufFile;
use vokra_core::{LicenseClass, Result, VokraError};

use crate::strict_checkpoint::{
    StrictCheckpoint, StrictCheckpointSpec, linear_rows, load_tensor, require_tensor_shape,
};

const LABEL: &str = "chatterbox_nano";
const INPUT_DIM: usize = 256;
const OUTPUT_DIM: usize = 768;
const SPEC: StrictCheckpointSpec = StrictCheckpointSpec {
    label: LABEL,
    arch: "chatterbox_nano",
    model_name: "chatterbox-nano-v1",
    model_name_alias: None,
    tensor_count: 155,
    manifest_sha256: [
        0xec, 0xc3, 0x3b, 0x97, 0x88, 0x7d, 0xdc, 0x77, 0xe2, 0x1d, 0x06, 0xad, 0x22, 0x5b, 0x32,
        0x3a, 0xfc, 0xd1, 0x0d, 0xaa, 0xe1, 0x5e, 0xb8, 0x2e, 0x4b, 0x3f, 0xdb, 0x25, 0x35, 0x0b,
        0x97, 0x98,
    ],
};

/// Strict handle for `vokra/chatterbox-nano-v1`.
#[derive(Debug, Clone)]
pub struct ChatterboxNanoCheckpoint {
    checkpoint: StrictCheckpoint,
}

impl ChatterboxNanoCheckpoint {
    /// Validates identity and all 155 authenticated T3 tensor names/shapes.
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

    /// Lazily decodes the real 256-to-768 speaker conditioning projection.
    pub fn load_speaker_projection(
        &self,
        file: &GgufFile,
    ) -> Result<ChatterboxNanoSpeakerProjection> {
        Ok(ChatterboxNanoSpeakerProjection {
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
    /// Returns the number of T3 tensors checked by the manifest gate.
    pub const fn tensor_count(&self) -> usize {
        self.checkpoint.tensor_count()
    }

    /// End-to-end PCM stays loud until GPT-2 sampling and S3Gen are bound.
    pub fn synthesize(&self, text: &str) -> Result<Vec<f32>> {
        if text.is_empty() {
            return Err(VokraError::InvalidArgument(
                "chatterbox_nano synthesize: text is empty".to_owned(),
            ));
        }
        Err(VokraError::NotImplemented(
            "chatterbox_nano synthesize: authenticated Nano T3-only checkpoint is bound and the real speaker projection runs natively, but the composite voice encoder/S3 tokenizer/distilled S3Gen/HiFT/watermark pipeline remains unavailable.",
        ))
    }
}

/// Real Chatterbox Nano speaker conditioning projection.
#[derive(Debug, Clone)]
pub struct ChatterboxNanoSpeakerProjection {
    weight: Vec<f32>,
    bias: Vec<f32>,
}

impl ChatterboxNanoSpeakerProjection {
    /// Applies the official `cond_enc.spkr_enc` PyTorch linear layer.
    pub fn forward(&self, speaker_embedding: &[f32]) -> Result<Vec<f32>> {
        linear_rows(
            "chatterbox_nano speaker projection",
            speaker_embedding,
            &self.weight,
            Some(&self.bias),
            INPUT_DIM,
            OUTPUT_DIM,
        )
    }
}
