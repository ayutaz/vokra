//! Strict partial checkpoint binding for Zyphra Zonos-v0.1-transformer.
//!
//! The public Vokra artifact is an authenticated 246-tensor main-model
//! checkpoint. It provides the real speaker projection, but does not yet
//! provide the complete delayed-AR conditioning and DAC product path.

use vokra_core::gguf::GgufFile;
use vokra_core::{LicenseClass, Result, VokraError};

use super::{
    ZonosBlockWeights, ZonosConfig, ZonosPrefixConditionerWeights, ZonosWeights,
    conditioning::ZonosPrefixConditionerParts,
};
use crate::strict_checkpoint::{
    StrictCheckpoint, StrictCheckpointSpec, linear_rows, load_tensor, require_tensor_shape,
};

const LABEL: &str = "zonos";
const INPUT_DIM: usize = 128;
const OUTPUT_DIM: usize = 2_048;
const WEIGHT: &str = "prefix_conditioner.conditioners.1.project.weight";
const BIAS: &str = "prefix_conditioner.conditioners.1.project.bias";
/// Fixed Vokra public artifact revision authenticated by the VAST gap run.
#[allow(dead_code)] // consumed when the authenticated Zonos binder is enabled
pub const PUBLIC_ARTIFACT_REVISION: &str = "b1bf5c56d470eb9097e9b04f9deca364576574ba";
/// Fixed upstream HF snapshot used by the parity/config evidence.
#[allow(dead_code)] // consumed when the authenticated Zonos binder is enabled
pub const UPSTREAM_HF_REVISION: &str = "9d8331fc49cb5ba8aad2bb56cafd809c66598f4e";
/// Content digest of the public Zonos GGUF artifact.
#[allow(dead_code)] // consumed when the authenticated Zonos binder is enabled
pub const PUBLIC_ARTIFACT_SHA256: &str =
    "12d542bd219f7f31c91b893810d85b0d810285e603029c69fbd19fd3c7da2c5c";
/// Byte size of the public Zonos GGUF artifact.
#[allow(dead_code)] // consumed when the authenticated Zonos binder is enabled
pub const PUBLIC_ARTIFACT_BYTES: u64 = 3_248_843_808;
/// Sorted `(name, dimensions)` manifest digest for all 246 tensors.
#[allow(dead_code)] // consumed when the authenticated Zonos binder is enabled
pub const PUBLIC_MANIFEST_SHA256: &str =
    "6543af3747d3e85bde862c3337744eea31f0105f9df6d8617c1c9afdae805847";
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

/// Strict handle for the authenticated public Zonos main-model checkpoint.
#[derive(Debug, Clone)]
pub struct ZonosCheckpoint {
    checkpoint: StrictCheckpoint,
}

impl ZonosCheckpoint {
    /// Validates the exact 246-tensor public manifest and speaker projection.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        let checkpoint = StrictCheckpoint::bind(file, SPEC)?;
        require_tensor_shape(file, LABEL, WEIGHT, &[OUTPUT_DIM, INPUT_DIM])?;
        require_tensor_shape(file, LABEL, BIAS, &[OUTPUT_DIM])?;
        Ok(Self { checkpoint })
    }

    /// Decodes the authenticated 128-to-2048 speaker projection.
    pub fn load_speaker_projection(&self, file: &GgufFile) -> Result<ZonosSpeakerProjection> {
        Ok(ZonosSpeakerProjection {
            weight: load_tensor(file, LABEL, WEIGHT, &[OUTPUT_DIM, INPUT_DIM])?,
            bias: load_tensor(file, LABEL, BIAS, &[OUTPUT_DIM])?,
        })
    }

    /// Loads every tensor in the authenticated 246-tensor transformer
    /// checkpoint into the typed native weight store.  The GGUF converter
    /// preserves the upstream names and PyTorch `[out, in]` shapes; linear
    /// tensors are explicitly transposed into the row-major `[in, out]` GEMM
    /// layout used by the native compute seam.
    pub fn load_weights(&self, file: &GgufFile, config: &ZonosConfig) -> Result<ZonosWeights> {
        if config.backbone.n_layer != 26
            || config.backbone.d_model != 2048
            || config.backbone.d_intermediate != 8192
            || config.num_codebooks != 9
            || config.codebook_vocab != 1026
            || config.head_vocab != 1025
        {
            return Err(VokraError::ModelLoad(
                "zonos: real checkpoint binder requires the authenticated transformer config"
                    .to_owned(),
            ));
        }
        let bb = &config.backbone;
        let mut codebook_embeddings = Vec::with_capacity(config.num_codebooks);
        let mut logit_heads = Vec::with_capacity(config.num_codebooks);
        for codebook in 0..config.num_codebooks {
            codebook_embeddings.push(load_tensor(
                file,
                LABEL,
                &format!("embeddings.{codebook}.weight"),
                &[config.codebook_vocab, bb.d_model],
            )?);
            logit_heads.push(load_gemm_weight(
                file,
                LABEL,
                &format!("heads.{codebook}.weight"),
                config.head_vocab,
                bb.d_model,
            )?);
        }
        let norm_f_w = load_tensor(file, LABEL, "backbone.norm_f.weight", &[bb.d_model])?;
        let norm_f_b = load_tensor(file, LABEL, "backbone.norm_f.bias", &[bb.d_model])?;
        let mut blocks = Vec::with_capacity(bb.n_layer);
        for layer in 0..bb.n_layer {
            let prefix = format!("backbone.layers.{layer}");
            blocks.push(ZonosBlockWeights {
                norm_1_w: load_tensor(
                    file,
                    LABEL,
                    &format!("{prefix}.norm.weight"),
                    &[bb.d_model],
                )?,
                norm_1_b: load_tensor(file, LABEL, &format!("{prefix}.norm.bias"), &[bb.d_model])?,
                qkv_proj: load_gemm_weight(
                    file,
                    LABEL,
                    &format!("{prefix}.mixer.in_proj.weight"),
                    bb.q_hidden() + 2 * bb.kv_hidden(),
                    bb.d_model,
                )?,
                o_proj: load_gemm_weight(
                    file,
                    LABEL,
                    &format!("{prefix}.mixer.out_proj.weight"),
                    bb.d_model,
                    bb.q_hidden(),
                )?,
                norm_2_w: load_tensor(
                    file,
                    LABEL,
                    &format!("{prefix}.norm2.weight"),
                    &[bb.d_model],
                )?,
                norm_2_b: load_tensor(file, LABEL, &format!("{prefix}.norm2.bias"), &[bb.d_model])?,
                mlp_fc1: load_gemm_weight(
                    file,
                    LABEL,
                    &format!("{prefix}.mlp.fc1.weight"),
                    2 * bb.d_intermediate,
                    bb.d_model,
                )?,
                mlp_fc2: load_gemm_weight(
                    file,
                    LABEL,
                    &format!("{prefix}.mlp.fc2.weight"),
                    bb.d_model,
                    bb.d_intermediate,
                )?,
            });
        }
        let prefix_conditioner =
            ZonosPrefixConditionerWeights::from_parts(ZonosPrefixConditionerParts {
                phoneme_embedder: load_tensor(
                    file,
                    LABEL,
                    "prefix_conditioner.conditioners.0.phoneme_embedder.weight",
                    &[189, bb.d_model],
                )?,
                speaker_project: load_gemm_weight(
                    file,
                    LABEL,
                    "prefix_conditioner.conditioners.1.project.weight",
                    bb.d_model,
                    128,
                )?,
                speaker_uncond: load_tensor(
                    file,
                    LABEL,
                    "prefix_conditioner.conditioners.1.uncond_vector",
                    &[bb.d_model],
                )?,
                emotion_weight: load_gemm_weight(
                    file,
                    LABEL,
                    "prefix_conditioner.conditioners.2.weight",
                    1024,
                    8,
                )?,
                emotion_uncond: load_tensor(
                    file,
                    LABEL,
                    "prefix_conditioner.conditioners.2.uncond_vector",
                    &[bb.d_model],
                )?,
                fmax_weight: load_gemm_weight(
                    file,
                    LABEL,
                    "prefix_conditioner.conditioners.3.weight",
                    1024,
                    1,
                )?,
                fmax_uncond: load_tensor(
                    file,
                    LABEL,
                    "prefix_conditioner.conditioners.3.uncond_vector",
                    &[bb.d_model],
                )?,
                pitch_std_weight: load_gemm_weight(
                    file,
                    LABEL,
                    "prefix_conditioner.conditioners.4.weight",
                    1024,
                    1,
                )?,
                pitch_std_uncond: load_tensor(
                    file,
                    LABEL,
                    "prefix_conditioner.conditioners.4.uncond_vector",
                    &[bb.d_model],
                )?,
                speaking_rate_weight: load_gemm_weight(
                    file,
                    LABEL,
                    "prefix_conditioner.conditioners.5.weight",
                    1024,
                    1,
                )?,
                speaking_rate_uncond: load_tensor(
                    file,
                    LABEL,
                    "prefix_conditioner.conditioners.5.uncond_vector",
                    &[bb.d_model],
                )?,
                language_embedder: load_tensor(
                    file,
                    LABEL,
                    "prefix_conditioner.conditioners.6.int_embedder.weight",
                    &[128, bb.d_model],
                )?,
                language_uncond: load_tensor(
                    file,
                    LABEL,
                    "prefix_conditioner.conditioners.6.uncond_vector",
                    &[bb.d_model],
                )?,
                speaker_bias: load_tensor(
                    file,
                    LABEL,
                    "prefix_conditioner.conditioners.1.project.bias",
                    &[bb.d_model],
                )?,
                project: load_gemm_weight(
                    file,
                    LABEL,
                    "prefix_conditioner.project.weight",
                    bb.d_model,
                    bb.d_model,
                )?,
                project_bias: load_tensor(
                    file,
                    LABEL,
                    "prefix_conditioner.project.bias",
                    &[bb.d_model],
                )?,
                norm_weight: load_tensor(
                    file,
                    LABEL,
                    "prefix_conditioner.norm.weight",
                    &[bb.d_model],
                )?,
                norm_bias: load_tensor(file, LABEL, "prefix_conditioner.norm.bias", &[bb.d_model])?,
            })?;
        Ok(ZonosWeights::from_bound_parts(
            vec![vec![1.0]; config.conditioners.len()],
            prefix_conditioner,
            codebook_embeddings,
            blocks,
            logit_heads,
            norm_f_w,
            norm_f_b,
        ))
    }

    /// Returns the pinned model name.
    #[must_use]
    pub fn model_name(&self) -> &str {
        self.checkpoint.model_name()
    }

    /// Returns the stamped weight-license class.
    #[must_use]
    pub const fn weight_license(&self) -> LicenseClass {
        self.checkpoint.weight_license()
    }

    /// Returns the complete authenticated main-model tensor count.
    #[must_use]
    pub const fn tensor_count(&self) -> usize {
        self.checkpoint.tensor_count()
    }

    /// The main model is bound, but end-to-end PCM remains explicitly partial.
    pub fn synthesize(&self, phoneme_ids: &[i64]) -> Result<Vec<f32>> {
        if phoneme_ids.is_empty() {
            return Err(VokraError::InvalidArgument(
                "zonos synthesize: phoneme_ids is empty".to_owned(),
            ));
        }
        Err(VokraError::NotImplemented(
            "zonos synthesize: PARTIAL_RUNTIME — the authenticated 246-tensor main model exposes the speaker projection, but prefix conditioning, delayed nine-codebook autoregression, and complete crate::dac::Dac PCM decode are not yet bound",
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

fn load_gemm_weight(
    file: &GgufFile,
    label: &str,
    name: &str,
    out_features: usize,
    in_features: usize,
) -> Result<Vec<f32>> {
    let source = load_tensor(file, label, name, &[out_features, in_features])?;
    let mut transposed = vec![0.0; source.len()];
    for output in 0..out_features {
        for input in 0..in_features {
            transposed[input * out_features + output] = source[output * in_features + input];
        }
    }
    Ok(transposed)
}
