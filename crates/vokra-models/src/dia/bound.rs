//! Strict official checkpoint binding for nari-labs Dia-1.6B.

use vokra_core::gguf::GgufFile;
use vokra_core::{LicenseClass, Result, VokraError};

use crate::strict_checkpoint::{
    StrictCheckpoint, StrictCheckpointSpec, embedding_rows, load_tensor, require_tensor_shape,
};

const LABEL: &str = "dia";
const VOCAB: usize = 256;
const DIMENSION: usize = 1_024;
const EMBEDDING: &str = "encoder.embedding.weight";
const SPEC: StrictCheckpointSpec = StrictCheckpointSpec {
    label: LABEL,
    arch: "dia",
    model_name: "dia-1.6b",
    model_name_alias: None,
    tensor_count: 343,
    manifest_sha256: [
        0x55, 0xfc, 0xe2, 0xa3, 0x9c, 0xaf, 0xba, 0x83, 0x8b, 0xd8, 0x00, 0xf6, 0xa6, 0xae, 0xfe,
        0x63, 0xa8, 0xe3, 0xb1, 0xdd, 0x86, 0xf2, 0x72, 0x7f, 0x9a, 0x20, 0xd8, 0x7f, 0xe6, 0xd2,
        0x52, 0xf7,
    ],
};

/// Strict handle for `vokra/dia-1.6b`.
#[derive(Debug, Clone)]
pub struct DiaCheckpoint {
    checkpoint: StrictCheckpoint,
}

impl DiaCheckpoint {
    /// Validates model identity and all 343 official tensor names and shapes.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        let checkpoint = StrictCheckpoint::bind(file, SPEC)?;
        require_tensor_shape(file, LABEL, EMBEDDING, &[VOCAB, DIMENSION])?;
        Ok(Self { checkpoint })
    }

    /// Decodes the real byte-token encoder embedding.
    pub fn load_text_embedding(&self, file: &GgufFile) -> Result<DiaTextEmbedding> {
        Ok(DiaTextEmbedding {
            weight: load_tensor(file, LABEL, EMBEDDING, &[VOCAB, DIMENSION])?,
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

    /// End-to-end PCM stays loud until delayed-AR and DAC are bound.
    pub fn synthesize(&self, text: &str) -> Result<Vec<f32>> {
        if text.is_empty() {
            return Err(VokraError::InvalidArgument(
                "dia synthesize: text is empty".to_owned(),
            ));
        }
        Err(VokraError::NotImplemented(
            "dia synthesize: the complete official checkpoint is bound and the real text embedding runs natively, but encoder/decoder attention, delayed nine-codebook sampling and the separately distributed DAC decoder remain pending.",
        ))
    }
}

/// Real Dia byte-token embedding table.
#[derive(Debug, Clone)]
pub struct DiaTextEmbedding {
    weight: Vec<f32>,
}

impl DiaTextEmbedding {
    /// Looks up official Dia byte-token embeddings.
    pub fn forward(&self, token_ids: &[u32]) -> Result<Vec<f32>> {
        embedding_rows(
            "dia text embedding",
            token_ids,
            &self.weight,
            VOCAB,
            DIMENSION,
        )
    }
}
