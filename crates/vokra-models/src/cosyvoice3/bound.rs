//! Strict official checkpoint binding for Fun-CosyVoice3-0.5B-2512.

use vokra_core::gguf::GgufFile;
use vokra_core::{LicenseClass, Result, VokraError};

use crate::strict_checkpoint::{
    StrictCheckpoint, StrictCheckpointSpec, linear_rows, load_tensor, require_tensor_shape,
};

const LABEL: &str = "cosyvoice3";
const HIDDEN: usize = 896;
const SPEC: StrictCheckpointSpec = StrictCheckpointSpec {
    label: LABEL,
    arch: "cosyvoice3",
    model_name: "fun-cosyvoice3-0.5b-2512",
    model_name_alias: None,
    tensor_count: 293,
    manifest_sha256: [
        0xfb, 0x6e, 0x0c, 0x2c, 0x37, 0xf1, 0x23, 0x43, 0xbd, 0x3c, 0x7a, 0xd5, 0x2b, 0xc6, 0xa1,
        0x55, 0x1b, 0x7e, 0xd8, 0x94, 0x5c, 0x7b, 0xab, 0x0e, 0x01, 0x1e, 0x66, 0x6f, 0xcb, 0xca,
        0x77, 0x05,
    ],
};
const Q_WEIGHT: &str = "llm.model.model.layers.0.self_attn.q_proj.weight";
const Q_BIAS: &str = "llm.model.model.layers.0.self_attn.q_proj.bias";

/// Strict handle for the published Fun-CosyVoice3 LLM checkpoint.
#[derive(Debug, Clone)]
pub struct CosyVoice3Checkpoint {
    checkpoint: StrictCheckpoint,
}

impl CosyVoice3Checkpoint {
    /// Validates model identity and every official tensor name and shape.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        let checkpoint = StrictCheckpoint::bind(file, SPEC)?;
        require_tensor_shape(file, LABEL, Q_WEIGHT, &[HIDDEN, HIDDEN])?;
        require_tensor_shape(file, LABEL, Q_BIAS, &[HIDDEN])?;
        Ok(Self { checkpoint })
    }

    /// Decodes the first Qwen2 query projection from real weights.
    pub fn load_layer0_q_projection(&self, file: &GgufFile) -> Result<CosyVoice3QProjection> {
        Ok(CosyVoice3QProjection {
            weight: load_tensor(file, LABEL, Q_WEIGHT, &[HIDDEN, HIDDEN])?,
            bias: load_tensor(file, LABEL, Q_BIAS, &[HIDDEN])?,
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

    /// Returns the number of tensors covered by the complete manifest gate.
    #[must_use]
    pub const fn tensor_count(&self) -> usize {
        self.checkpoint.tensor_count()
    }

    /// End-to-end PCM stays loud until the rest of the chain is bound.
    pub fn synthesize(&self, text: &str) -> Result<Vec<f32>> {
        if text.is_empty() {
            return Err(VokraError::InvalidArgument(
                "cosyvoice3 synthesize: text is empty".to_owned(),
            ));
        }
        Err(VokraError::NotImplemented(
            "cosyvoice3 synthesize: the complete official LLM checkpoint is bound and a real Qwen2 projection runs natively, but tokenizer integration, full LLM decode, flow-matching estimator and HiFTNet PCM generation remain pending.",
        ))
    }
}

/// Real layer-0 Qwen2 query projection.
#[derive(Debug, Clone)]
pub struct CosyVoice3QProjection {
    weight: Vec<f32>,
    bias: Vec<f32>,
}

impl CosyVoice3QProjection {
    /// Applies the official biased Qwen2 query linear layer.
    pub fn forward(&self, hidden: &[f32]) -> Result<Vec<f32>> {
        linear_rows(
            "cosyvoice3 layer0 q projection",
            hidden,
            &self.weight,
            Some(&self.bias),
            HIDDEN,
            HIDDEN,
        )
    }
}
