//! Strict official checkpoint binding for nari-labs Dia-1.6B.

use vokra_core::gguf::GgufFile;
use vokra_core::{LicenseClass, Result, VokraError};

use crate::dia::{DiaConfig, DiaDecoderBlockWeights, DiaEncoderBlockWeights, DiaWeights};
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

/// Strict handle for the historical `vokra/dia-1.6b` main-model artifact.
///
/// The complete 343-tensor main-model manifest and payload are authenticated,
/// but this is still a composite-partial text-to-PCM route: the delayed-AR
/// generation evidence and the separately distributed [`crate::dac::Dac`]
/// PCM decoder must be bound independently.
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
            weight: tensor(file, EMBEDDING, &[VOCAB, DIMENSION])?,
        })
    }

    /// Loads the complete authenticated 343-tensor main-model payload.
    ///
    /// The public GGUF stores projection tensors in the source-authentic
    /// `[input, heads, head_dim]`/`[heads, head_dim, output]` layouts.  The
    /// native forward uses flattened `[input, output]` matrices, so this
    /// method performs only shape-preserving flattening and the documented
    /// interleaved `wi_fused` split.  No tensor is synthesized or silently
    /// omitted: the manifest bind runs first and every expected tensor is
    /// decoded and checked for finite values.
    pub fn load_weights(&self, file: &GgufFile, config: &DiaConfig) -> Result<DiaWeights> {
        if config != &DiaConfig::dia_1_6b() {
            return Err(VokraError::InvalidArgument(
                "dia weights: only the authenticated Dia-1.6B config is supported".to_owned(),
            ));
        }
        let encoder = &config.encoder;
        let decoder = &config.decoder;
        let text_embedding = tensor(file, EMBEDDING, &[256, 1024])?;
        let mut encoder_blocks = Vec::with_capacity(encoder.n_layer);
        for layer in 0..encoder.n_layer {
            let prefix = format!("encoder.layers.{layer}");
            let (gate_proj, up_proj) = fused(
                file,
                &format!("{prefix}.mlp.wi_fused.weight"),
                &[1024, 2, 4096],
                1024,
                4096,
            )?;
            encoder_blocks.push(DiaEncoderBlockWeights {
                norm_1: tensor(file, &format!("{prefix}.pre_sa_norm.weight"), &[1024])?,
                q_proj: tensor(
                    file,
                    &format!("{prefix}.self_attention.q_proj.weight"),
                    &[1024, 16, 128],
                )?,
                k_proj: tensor(
                    file,
                    &format!("{prefix}.self_attention.k_proj.weight"),
                    &[1024, 16, 128],
                )?,
                v_proj: tensor(
                    file,
                    &format!("{prefix}.self_attention.v_proj.weight"),
                    &[1024, 16, 128],
                )?,
                o_proj: tensor(
                    file,
                    &format!("{prefix}.self_attention.o_proj.weight"),
                    &[16, 128, 1024],
                )?,
                norm_2: tensor(file, &format!("{prefix}.post_sa_norm.weight"), &[1024])?,
                gate_proj,
                up_proj,
                down_proj: tensor(file, &format!("{prefix}.mlp.wo.weight"), &[4096, 1024])?,
            });
        }
        let mut channel_embeddings = Vec::with_capacity(config.channels);
        for channel in 0..config.channels {
            channel_embeddings.push(tensor(
                file,
                &format!("decoder.embeddings.{channel}.weight"),
                &[1028, 2048],
            )?);
        }
        let mut decoder_blocks = Vec::with_capacity(decoder.n_layer);
        for layer in 0..decoder.n_layer {
            let prefix = format!("decoder.layers.{layer}");
            let (gate_proj, up_proj) = fused(
                file,
                &format!("{prefix}.mlp.wi_fused.weight"),
                &[2048, 2, 8192],
                2048,
                8192,
            )?;
            decoder_blocks.push(DiaDecoderBlockWeights {
                sa_norm: tensor(file, &format!("{prefix}.pre_sa_norm.weight"), &[2048])?,
                sa_q_proj: tensor(
                    file,
                    &format!("{prefix}.self_attention.q_proj.weight"),
                    &[2048, 16, 128],
                )?,
                sa_k_proj: tensor(
                    file,
                    &format!("{prefix}.self_attention.k_proj.weight"),
                    &[2048, 4, 128],
                )?,
                sa_v_proj: tensor(
                    file,
                    &format!("{prefix}.self_attention.v_proj.weight"),
                    &[2048, 4, 128],
                )?,
                sa_o_proj: tensor(
                    file,
                    &format!("{prefix}.self_attention.o_proj.weight"),
                    &[16, 128, 2048],
                )?,
                xa_norm: tensor(file, &format!("{prefix}.pre_ca_norm.weight"), &[2048])?,
                xa_q_proj: tensor(
                    file,
                    &format!("{prefix}.cross_attention.q_proj.weight"),
                    &[2048, 16, 128],
                )?,
                xa_k_proj: tensor(
                    file,
                    &format!("{prefix}.cross_attention.k_proj.weight"),
                    &[1024, 16, 128],
                )?,
                xa_v_proj: tensor(
                    file,
                    &format!("{prefix}.cross_attention.v_proj.weight"),
                    &[1024, 16, 128],
                )?,
                xa_o_proj: tensor(
                    file,
                    &format!("{prefix}.cross_attention.o_proj.weight"),
                    &[16, 128, 2048],
                )?,
                ffn_norm: tensor(file, &format!("{prefix}.pre_mlp_norm.weight"), &[2048])?,
                gate_proj,
                up_proj,
                down_proj: tensor(file, &format!("{prefix}.mlp.wo.weight"), &[8192, 2048])?,
            });
        }
        let logits = tensor(file, "decoder.logits_dense.weight", &[2048, 9, 1028])?;
        let mut logit_heads = Vec::with_capacity(config.channels);
        let stride = decoder.n_embd * config.tgt_vocab_size;
        for channel in 0..config.channels {
            let mut head = vec![0.0; stride];
            for input in 0..decoder.n_embd {
                for output in 0..config.tgt_vocab_size {
                    head[input * config.tgt_vocab_size + output] = logits
                        [(input * config.channels + channel) * config.tgt_vocab_size + output];
                }
            }
            logit_heads.push(head);
        }
        Ok(DiaWeights {
            text_embedding,
            encoder_blocks,
            encoder_norm: tensor(file, "encoder.norm.weight", &[1024])?,
            channel_embeddings,
            decoder_blocks,
            decoder_norm: tensor(file, "decoder.norm.weight", &[2048])?,
            logit_heads,
            is_synthesized: false,
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

    /// Returns whether this handle still needs the separately authenticated
    /// DAC composition for text-to-PCM inference.
    #[must_use]
    pub const fn is_partial(&self) -> bool {
        true
    }

    /// End-to-end PCM stays loud until delayed-AR and DAC are bound.
    pub fn synthesize(&self, text: &str) -> Result<Vec<f32>> {
        if text.is_empty() {
            return Err(VokraError::InvalidArgument(
                "dia synthesize: text is empty".to_owned(),
            ));
        }
        Err(VokraError::NotImplemented(
            "dia synthesize: the authenticated 343-tensor main model is bound, but same-execution generation parity and the separately distributed 44.1-kHz nine-codebook crate::dac::Dac route remain pending.",
        ))
    }
}

fn tensor(file: &GgufFile, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
    let values = load_tensor(file, LABEL, name, shape)?;
    if values.iter().all(|value| value.is_finite()) {
        Ok(values)
    } else {
        Err(VokraError::ModelLoad(format!(
            "{LABEL}: tensor `{name}` contains non-finite values"
        )))
    }
}

fn fused(
    file: &GgufFile,
    name: &str,
    shape: &[usize],
    input: usize,
    output: usize,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let values = tensor(file, name, shape)?;
    let mut gate = Vec::with_capacity(input * output);
    let mut up = Vec::with_capacity(input * output);
    for row in 0..input {
        let start = row * 2 * output;
        gate.extend_from_slice(&values[start..start + output]);
        up.extend_from_slice(&values[start + output..start + 2 * output]);
    }
    Ok((gate, up))
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
