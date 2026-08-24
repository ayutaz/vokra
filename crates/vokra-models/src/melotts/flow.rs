//! MeloTTS VITS2 Transformer coupling-flow loader and runtime.

use vokra_core::backend::BackendKind;
use vokra_core::gguf::GgufFile;
use vokra_core::{Result, VokraError};

use crate::compute::{Compute, HotOp};
use crate::sbv2::flow::{FlowLayer, SbV2Flow, SbV2TransformerCouplingLayer};
use crate::sbv2::text_encoder::{LayerNorm, PositionWiseFFN, RelPositionMHA, SbV2TransformerBlock};
use crate::strict_checkpoint::load_tensor;

use super::{FILTER_CHANNELS, GIN_CHANNELS, HIDDEN_CHANNELS, INTER_CHANNELS, LABEL, N_HEADS};

const FLOW_BLOCKS: usize = 4;
const FLOW_ENCODER_LAYERS: usize = 3;
const FLOW_CONDITION_LAYER: usize = 2;
const FLOW_FFN_KERNEL: usize = 5;
const FLOW_WINDOW_SIZE: usize = 4;

/// Backend operations required by the MeloTTS latent flow.
pub const MELOTTS_FLOW_HOT_OPS: &[HotOp] =
    &[HotOp::Gemm, HotOp::Softmax, HotOp::LayerNorm, HotOp::Conv1d];

/// Loaded four-block MeloTTS VITS2 latent flow.
pub struct MeloFlowModel {
    flow: SbV2Flow,
}

impl MeloFlowModel {
    pub(super) fn from_gguf(file: &GgufFile) -> Result<Self> {
        let latent_channels = INTER_CHANNELS as usize;
        let half_latent = latent_channels / 2;
        let hidden = HIDDEN_CHANNELS as usize;
        let gin = GIN_CHANNELS as usize;
        let mut layers = Vec::with_capacity(FLOW_BLOCKS * 2);
        for block in 0..FLOW_BLOCKS {
            let upstream_index = block * 2;
            let prefix = format!("flow.flows.{upstream_index}");
            let mut encoder_stack = Vec::with_capacity(FLOW_ENCODER_LAYERS);
            for layer in 0..FLOW_ENCODER_LAYERS {
                encoder_stack.push(load_transformer_block(file, &prefix, layer)?);
            }
            layers.push(FlowLayer::Coupling(
                SbV2TransformerCouplingLayer::from_weights(
                    tensor(
                        file,
                        &format!("{prefix}.pre.weight"),
                        &[hidden, half_latent, 1],
                    )?,
                    tensor(file, &format!("{prefix}.pre.bias"), &[hidden])?,
                    tensor(
                        file,
                        &format!("{prefix}.enc.spk_emb_linear.weight"),
                        &[hidden, gin],
                    )?,
                    tensor(
                        file,
                        &format!("{prefix}.enc.spk_emb_linear.bias"),
                        &[hidden],
                    )?,
                    encoder_stack,
                    tensor(
                        file,
                        &format!("{prefix}.post.weight"),
                        &[half_latent, hidden, 1],
                    )?,
                    tensor(file, &format!("{prefix}.post.bias"), &[half_latent])?,
                    half_latent,
                    hidden,
                    gin,
                    true,
                )
                .with_conditioning_layer(FLOW_CONDITION_LAYER),
            ));
            layers.push(FlowLayer::Flip);
        }
        Ok(Self {
            flow: SbV2Flow::from_layers(layers, latent_channels),
        })
    }

    /// Maps prior latents into decoder latents through the selected backend.
    pub fn inverse(
        &self,
        prior_position_major: &[f32],
        frame_count: usize,
        global_conditioning: &[f32],
        backend: BackendKind,
    ) -> Result<Vec<f32>> {
        let compute = Compute::for_backend(backend, MELOTTS_FLOW_HOT_OPS)?;
        self.inverse_with_compute(
            &compute,
            prior_position_major,
            frame_count,
            global_conditioning,
        )
    }

    pub(crate) fn inverse_with_compute(
        &self,
        compute: &Compute,
        prior_position_major: &[f32],
        frame_count: usize,
        global_conditioning: &[f32],
    ) -> Result<Vec<f32>> {
        let latent_channels = INTER_CHANNELS as usize;
        if prior_position_major.len() != frame_count * latent_channels {
            return Err(VokraError::InvalidArgument(format!(
                "melotts flow: expected prior [{frame_count}, {latent_channels}], got {} values",
                prior_position_major.len()
            )));
        }
        if global_conditioning.len() != GIN_CHANNELS as usize {
            return Err(VokraError::InvalidArgument(format!(
                "melotts flow: expected {} speaker-conditioning values, got {}",
                GIN_CHANNELS,
                global_conditioning.len()
            )));
        }
        if frame_count == 0 {
            return Ok(Vec::new());
        }
        self.flow.inverse_with_compute(
            compute,
            prior_position_major,
            frame_count,
            global_conditioning,
        )
    }
}

fn load_transformer_block(
    file: &GgufFile,
    coupling_prefix: &str,
    layer: usize,
) -> Result<SbV2TransformerBlock> {
    let hidden = HIDDEN_CHANNELS as usize;
    let heads = N_HEADS as usize;
    let head_dim = hidden / heads;
    let filter = FILTER_CHANNELS as usize;
    let attention_prefix = format!("{coupling_prefix}.enc.attn_layers.{layer}");
    let ffn_prefix = format!("{coupling_prefix}.enc.ffn_layers.{layer}");
    Ok(SbV2TransformerBlock::new(
        RelPositionMHA::new(
            tensor(
                file,
                &format!("{attention_prefix}.conv_q.weight"),
                &[hidden, hidden, 1],
            )?,
            tensor(file, &format!("{attention_prefix}.conv_q.bias"), &[hidden])?,
            tensor(
                file,
                &format!("{attention_prefix}.conv_k.weight"),
                &[hidden, hidden, 1],
            )?,
            tensor(file, &format!("{attention_prefix}.conv_k.bias"), &[hidden])?,
            tensor(
                file,
                &format!("{attention_prefix}.conv_v.weight"),
                &[hidden, hidden, 1],
            )?,
            tensor(file, &format!("{attention_prefix}.conv_v.bias"), &[hidden])?,
            tensor(
                file,
                &format!("{attention_prefix}.conv_o.weight"),
                &[hidden, hidden, 1],
            )?,
            tensor(file, &format!("{attention_prefix}.conv_o.bias"), &[hidden])?,
            tensor(
                file,
                &format!("{attention_prefix}.emb_rel_k"),
                &[1, 2 * FLOW_WINDOW_SIZE + 1, head_dim],
            )?,
            tensor(
                file,
                &format!("{attention_prefix}.emb_rel_v"),
                &[1, 2 * FLOW_WINDOW_SIZE + 1, head_dim],
            )?,
            heads,
            head_dim,
            FLOW_WINDOW_SIZE,
        ),
        LayerNorm::new(
            tensor(
                file,
                &format!("{coupling_prefix}.enc.norm_layers_1.{layer}.gamma"),
                &[hidden],
            )?,
            tensor(
                file,
                &format!("{coupling_prefix}.enc.norm_layers_1.{layer}.beta"),
                &[hidden],
            )?,
            hidden,
        ),
        PositionWiseFFN::new(
            tensor(
                file,
                &format!("{ffn_prefix}.conv_1.weight"),
                &[filter, hidden, FLOW_FFN_KERNEL],
            )?,
            tensor(file, &format!("{ffn_prefix}.conv_1.bias"), &[filter])?,
            tensor(
                file,
                &format!("{ffn_prefix}.conv_2.weight"),
                &[hidden, filter, FLOW_FFN_KERNEL],
            )?,
            tensor(file, &format!("{ffn_prefix}.conv_2.bias"), &[hidden])?,
            hidden,
            filter,
            FLOW_FFN_KERNEL,
        ),
        LayerNorm::new(
            tensor(
                file,
                &format!("{coupling_prefix}.enc.norm_layers_2.{layer}.gamma"),
                &[hidden],
            )?,
            tensor(
                file,
                &format!("{coupling_prefix}.enc.norm_layers_2.{layer}.beta"),
                &[hidden],
            )?,
            hidden,
        ),
        hidden,
    ))
}

fn tensor(file: &GgufFile, name: &str, expected: &[usize]) -> Result<Vec<f32>> {
    load_tensor(file, LABEL, name, expected)
}
