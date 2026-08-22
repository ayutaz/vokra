//! Strict binding for the operator-provisioned ESPnet JSUT VITS checkpoint.
//!
//! The canonical tensor manifest is derived from the official 22.05 kHz
//! release recipe at ESPnet commit
//! `628b46282537ce532d613d6bafb75e826e8455de` (Zenodo record 5521354).
//! Vokra does not fetch or redistribute that corpus-restricted weight.  An
//! operator prepares only the `VITSGenerator` state dict with
//! `tools/parity/vits_ja_prepare_checkpoint.py`, then converts it locally.

use vokra_core::gguf::GgufFile;
use vokra_core::{LicenseClass, Result, VokraError};

use crate::strict_checkpoint::{
    StrictCheckpoint, StrictCheckpointSpec, embedding_rows, load_tensor, require_tensor_shape,
};

const LABEL: &str = "vits-ja";
const VOCAB_SIZE: usize = 43;
const HIDDEN_DIM: usize = 192;
const SPEC: StrictCheckpointSpec = StrictCheckpointSpec {
    label: LABEL,
    arch: "vits-ja",
    model_name: "espnet-jsut-vits-22khz",
    model_name_alias: None,
    tensor_count: 885,
    manifest_sha256: [
        0xb5, 0xd0, 0x39, 0xb6, 0xf6, 0xfe, 0xbf, 0xcb, 0x93, 0xf2, 0xad, 0x17, 0xf1, 0x64, 0x73,
        0x11, 0xbb, 0x0c, 0x37, 0x86, 0x9f, 0x54, 0xb5, 0xe5, 0xce, 0xac, 0x23, 0xf7, 0xb9, 0x51,
        0xb2, 0x84,
    ],
};

/// Strict handle for an operator-provisioned canonical JSUT VITS GGUF.
#[derive(Debug, Clone)]
pub struct VitsJaCheckpoint {
    checkpoint: StrictCheckpoint,
}

impl VitsJaCheckpoint {
    /// Validates identity and all 885 generator tensor names and shapes.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        let checkpoint = StrictCheckpoint::bind(file, SPEC)?;
        require_tensor_shape(
            file,
            LABEL,
            "text_encoder.emb.weight",
            &[VOCAB_SIZE, HIDDEN_DIM],
        )?;
        Ok(Self { checkpoint })
    }

    /// Lazily decodes the real 43-by-192 phoneme embedding table.
    pub fn load_text_embedding(&self, file: &GgufFile) -> Result<VitsJaTextEmbedding> {
        Ok(VitsJaTextEmbedding {
            weight: load_tensor(
                file,
                LABEL,
                "text_encoder.emb.weight",
                &[VOCAB_SIZE, HIDDEN_DIM],
            )?,
        })
    }

    /// Returns the pinned checkpoint variant.
    #[must_use]
    pub fn model_name(&self) -> &str {
        self.checkpoint.model_name()
    }

    /// Returns the fail-closed weight-license class stamped in the GGUF.
    #[must_use]
    pub const fn weight_license(&self) -> LicenseClass {
        self.checkpoint.weight_license()
    }

    /// Returns the number of tensors checked by the complete manifest gate.
    #[must_use]
    pub const fn tensor_count(&self) -> usize {
        self.checkpoint.tensor_count()
    }

    /// End-to-end PCM stays loud until the native VITS forward is complete.
    pub fn synthesize(&self, text: &str) -> Result<Vec<f32>> {
        if text.is_empty() {
            return Err(VokraError::InvalidArgument(
                "vits-ja synthesize: text is empty".to_owned(),
            ));
        }
        Err(VokraError::NotImplemented(
            "vits-ja synthesize: the complete operator-provisioned ESPnet JSUT VITS generator checkpoint is bound and the real phoneme embedding runs natively, but the Japanese frontend, Conformer text encoder, stochastic duration predictor, residual coupling flow and HiFi-GAN PCM path remain pending. Vokra never fetches or redistributes the JSUT-trained weight because the corpus terms prohibit redistribution.",
        ))
    }
}

/// Real ESPnet VITS phoneme embedding table.
#[derive(Debug, Clone)]
pub struct VitsJaTextEmbedding {
    weight: Vec<f32>,
}

impl VitsJaTextEmbedding {
    /// Looks up canonical pyopenjtalk-prosody phoneme ids.
    pub fn forward(&self, token_ids: &[u32]) -> Result<Vec<f32>> {
        embedding_rows(
            "vits-ja text embedding",
            token_ids,
            &self.weight,
            VOCAB_SIZE,
            HIDDEN_DIM,
        )
    }
}
