use vokra_core::gguf::GgufFile;
use vokra_core::{Result, VokraError};

use crate::compute::Compute;
use crate::strict_checkpoint::load_tensor;

pub(super) const HIDDEN_DIM: usize = 768;
pub(super) const FFN_DIM: usize = 3_072;
pub(super) const NUM_HEADS: usize = 12;
pub(super) const HEAD_DIM: usize = 64;
pub(super) const TEXT_VOCAB_SIZE: usize = 16_384;
pub(super) const AUDIO_VOCAB_SIZE: usize = 1_024;
pub(super) const NUM_CODEBOOKS: usize = 16;
pub(super) const GLOBAL_LAYERS: usize = 12;
pub(super) const LOCAL_LAYERS: usize = 1;

const LABEL: &str = "moss_tts/nano";

#[derive(Debug, Clone)]
pub(super) struct Linear {
    input: usize,
    output: usize,
    weight_t: Vec<f32>,
    bias: Option<Vec<f32>>,
}

impl Linear {
    fn load(
        file: &GgufFile,
        weight_name: &str,
        bias_name: Option<&str>,
        input: usize,
        output: usize,
    ) -> Result<Self> {
        let weight = load_tensor(file, LABEL, weight_name, &[output, input])?;
        let bias = bias_name
            .map(|name| load_tensor(file, LABEL, name, &[output]))
            .transpose()?;
        Ok(Self {
            input,
            output,
            weight_t: transpose_out_in(&weight, output, input),
            bias,
        })
    }

    pub(super) fn forward(
        &self,
        compute: &Compute,
        input: &[f32],
        rows: usize,
    ) -> Result<Vec<f32>> {
        let expected = rows.checked_mul(self.input).ok_or_else(|| {
            VokraError::InvalidArgument(format!(
                "{LABEL}: linear input shape overflows: {rows} * {}",
                self.input
            ))
        })?;
        if input.len() != expected {
            return Err(VokraError::InvalidArgument(format!(
                "{LABEL}: linear input has {} values, expected {rows} x {} = {expected}",
                input.len(),
                self.input
            )));
        }
        let mut output = vec![0.0f32; rows * self.output];
        compute.gemm_f32(
            rows,
            self.output,
            self.input,
            input,
            &self.weight_t,
            self.bias.as_deref(),
            &mut output,
        )?;
        Ok(output)
    }
}

#[derive(Debug, Clone)]
pub(super) struct BlockWeights {
    pub(super) ln1_weight: Vec<f32>,
    pub(super) ln1_bias: Vec<f32>,
    pub(super) attention_in: Linear,
    pub(super) attention_out: Linear,
    pub(super) ln2_weight: Vec<f32>,
    pub(super) ln2_bias: Vec<f32>,
    pub(super) ffn_in: Linear,
    pub(super) ffn_out: Linear,
}

impl BlockWeights {
    fn load(file: &GgufFile, prefix: &str) -> Result<Self> {
        Ok(Self {
            ln1_weight: load_tensor(file, LABEL, &format!("{prefix}.ln_1.weight"), &[HIDDEN_DIM])?,
            ln1_bias: load_tensor(file, LABEL, &format!("{prefix}.ln_1.bias"), &[HIDDEN_DIM])?,
            attention_in: Linear::load(
                file,
                &format!("{prefix}.attn.c_attn.weight"),
                Some(&format!("{prefix}.attn.c_attn.bias")),
                HIDDEN_DIM,
                3 * HIDDEN_DIM,
            )?,
            attention_out: Linear::load(
                file,
                &format!("{prefix}.attn.c_proj.weight"),
                Some(&format!("{prefix}.attn.c_proj.bias")),
                HIDDEN_DIM,
                HIDDEN_DIM,
            )?,
            ln2_weight: load_tensor(file, LABEL, &format!("{prefix}.ln_2.weight"), &[HIDDEN_DIM])?,
            ln2_bias: load_tensor(file, LABEL, &format!("{prefix}.ln_2.bias"), &[HIDDEN_DIM])?,
            ffn_in: Linear::load(
                file,
                &format!("{prefix}.mlp.fc_in.weight"),
                Some(&format!("{prefix}.mlp.fc_in.bias")),
                HIDDEN_DIM,
                FFN_DIM,
            )?,
            ffn_out: Linear::load(
                file,
                &format!("{prefix}.mlp.fc_out.weight"),
                Some(&format!("{prefix}.mlp.fc_out.bias")),
                FFN_DIM,
                HIDDEN_DIM,
            )?,
        })
    }
}

#[derive(Debug, Clone)]
pub(super) struct TransformerWeights {
    pub(super) blocks: Vec<BlockWeights>,
    pub(super) final_norm_weight: Vec<f32>,
    pub(super) final_norm_bias: Vec<f32>,
}

impl TransformerWeights {
    fn load(file: &GgufFile, prefix: &str, layers: usize) -> Result<Self> {
        let blocks = (0..layers)
            .map(|index| BlockWeights::load(file, &format!("{prefix}.h.{index}")))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            blocks,
            final_norm_weight: load_tensor(
                file,
                LABEL,
                &format!("{prefix}.ln_f.weight"),
                &[HIDDEN_DIM],
            )?,
            final_norm_bias: load_tensor(
                file,
                LABEL,
                &format!("{prefix}.ln_f.bias"),
                &[HIDDEN_DIM],
            )?,
        })
    }
}

#[derive(Debug, Clone)]
pub(super) struct NanoWeights {
    pub(super) text_embedding: Vec<f32>,
    pub(super) audio_embeddings: Vec<Vec<f32>>,
    pub(super) global: TransformerWeights,
    pub(super) local: TransformerWeights,
    pub(super) text_head: Linear,
    pub(super) audio_heads: Vec<Linear>,
}

impl NanoWeights {
    pub(super) fn bind(file: &GgufFile) -> Result<Self> {
        let audio_embeddings = (0..NUM_CODEBOOKS)
            .map(|index| {
                load_tensor(
                    file,
                    LABEL,
                    &format!("audio_embeddings.{index}.weight"),
                    &[AUDIO_VOCAB_SIZE, HIDDEN_DIM],
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let audio_heads = (0..NUM_CODEBOOKS)
            .map(|index| {
                Linear::load(
                    file,
                    &format!("audio_lm_heads.{index}.weight"),
                    None,
                    HIDDEN_DIM,
                    AUDIO_VOCAB_SIZE,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            text_embedding: load_tensor(
                file,
                LABEL,
                "transformer.wte.weight",
                &[TEXT_VOCAB_SIZE, HIDDEN_DIM],
            )?,
            audio_embeddings,
            global: TransformerWeights::load(file, "transformer", GLOBAL_LAYERS)?,
            local: TransformerWeights::load(file, "local_transformer", LOCAL_LAYERS)?,
            text_head: Linear::load(
                file,
                "text_lm_head.weight",
                None,
                HIDDEN_DIM,
                TEXT_VOCAB_SIZE,
            )?,
            audio_heads,
        })
    }
}

fn transpose_out_in(weight: &[f32], output: usize, input: usize) -> Vec<f32> {
    let mut transposed = vec![0.0f32; weight.len()];
    for out in 0..output {
        for inner in 0..input {
            transposed[inner * output + out] = weight[out * input + inner];
        }
    }
    transposed
}
