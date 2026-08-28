//! Strict FocalCodec GGUF tensor binding.

use vokra_core::Result;
use vokra_core::gguf::GgufFile;
use vokra_ops::{VocosBlockWeights, VocosNormWeights, VocosWeights};

use crate::strict_checkpoint::load_tensor;

use super::{FocalCodecVariant, LABEL};

pub(super) const WAVLM_DIM: usize = 1_024;
pub(super) const WAVLM_FFN: usize = 4_096;
pub(super) const WAVLM_HEADS: usize = 16;
pub(super) const WAVLM_HEAD_DIM: usize = 64;
pub(super) const WAVLM_LAYERS: usize = 6;
pub(super) const FEATURE_DIM: usize = 512;
pub(super) const RELATIVE_BUCKETS: usize = 320;

#[derive(Debug, Clone)]
pub(super) struct ConvWeights {
    pub(super) weight: Vec<f32>,
    pub(super) bias: Vec<f32>,
}

#[derive(Debug, Clone)]
pub(super) struct NormWeights {
    pub(super) weight: Vec<f32>,
    pub(super) bias: Vec<f32>,
}

#[derive(Debug, Clone)]
pub(super) struct LinearWeights {
    pub(super) weight: Vec<f32>,
    pub(super) bias: Vec<f32>,
}

#[derive(Debug, Clone)]
pub(super) struct FeatureLayerWeights {
    pub(super) conv: ConvWeights,
    pub(super) norm: NormWeights,
}

#[derive(Debug, Clone)]
pub(super) struct WavLmAttentionWeights {
    pub(super) q: LinearWeights,
    pub(super) k: LinearWeights,
    pub(super) v: LinearWeights,
    pub(super) out: LinearWeights,
    pub(super) gru_weight: Vec<f32>,
    pub(super) gru_bias: Vec<f32>,
    pub(super) gru_const: Vec<f32>,
}

#[derive(Debug, Clone)]
pub(super) struct WavLmBlockWeights {
    pub(super) attention_norm: NormWeights,
    pub(super) attention: WavLmAttentionWeights,
    pub(super) feed_forward_norm: NormWeights,
    pub(super) feed_forward_in: LinearWeights,
    pub(super) feed_forward_out: LinearWeights,
}

#[derive(Debug, Clone)]
pub(super) struct WavLmWeights {
    pub(super) feature_layers: Vec<FeatureLayerWeights>,
    pub(super) input_norm: NormWeights,
    pub(super) feature_proj: LinearWeights,
    pub(super) positional_conv: ConvWeights,
    pub(super) relative_embedding: Vec<f32>,
    pub(super) blocks: Vec<WavLmBlockWeights>,
    // Present in all three official state dicts but intentionally unused by
    // the audited non-causal forward (`TransformerEncoder`: "No output norm").
    pub(super) unused_output_norm: NormWeights,
}

#[derive(Debug, Clone)]
pub(super) struct FocalModulationWeights {
    pub(super) in_proj: LinearWeights,
    pub(super) depthwise: [ConvWeights; 2],
    pub(super) context_proj: ConvWeights,
    pub(super) out_proj: LinearWeights,
}

#[derive(Debug, Clone)]
pub(super) struct FocalBlockWeights {
    pub(super) modulation_norm: NormWeights,
    pub(super) modulation: FocalModulationWeights,
    pub(super) feed_forward_norm: NormWeights,
    pub(super) feed_forward_in: LinearWeights,
    pub(super) feed_forward_out: LinearWeights,
}

#[derive(Debug, Clone)]
pub(super) struct ScaleWeights {
    pub(super) conv: ConvWeights,
    pub(super) alpha: Vec<f32>,
}

#[derive(Debug, Clone)]
pub(super) struct FocalEncoderWeights {
    pub(super) scales: Vec<ScaleWeights>,
    pub(super) blocks: Vec<FocalBlockWeights>,
    pub(super) out_proj: LinearWeights,
}

#[derive(Debug, Clone)]
pub(super) struct FocalDecoderWeights {
    pub(super) in_proj: LinearWeights,
    pub(super) blocks: Vec<FocalBlockWeights>,
    pub(super) scales: Vec<ScaleWeights>,
}

#[derive(Debug, Clone)]
pub(super) struct FocalCodecWeights {
    pub(super) wavlm: WavLmWeights,
    pub(super) compressor: FocalEncoderWeights,
    pub(super) decompressor: FocalDecoderWeights,
    pub(super) vocos: VocosWeights,
}

pub(super) fn bind(file: &GgufFile, variant: FocalCodecVariant) -> Result<FocalCodecWeights> {
    Ok(FocalCodecWeights {
        wavlm: bind_wavlm(file)?,
        compressor: bind_compressor(file, variant)?,
        decompressor: bind_decompressor(file, variant)?,
        vocos: bind_vocos(file)?,
    })
}

fn bind_wavlm(file: &GgufFile) -> Result<WavLmWeights> {
    const KERNELS: [usize; 7] = [10, 3, 3, 3, 3, 2, 2];
    let mut feature_layers = Vec::with_capacity(KERNELS.len());
    let mut input = 1;
    for (index, kernel) in KERNELS.into_iter().enumerate() {
        let prefix = format!("encoder.feature_extractor.layers.{index}");
        feature_layers.push(FeatureLayerWeights {
            conv: ConvWeights {
                weight: tensor(
                    file,
                    &format!("{prefix}.conv.weight"),
                    &[FEATURE_DIM, input, kernel],
                )?,
                bias: Vec::new(),
            },
            norm: norm(file, &format!("{prefix}.norm"), FEATURE_DIM)?,
        });
        input = FEATURE_DIM;
    }

    let mut blocks = Vec::with_capacity(WAVLM_LAYERS);
    for index in 0..WAVLM_LAYERS {
        let prefix = format!("encoder.encoder.layers.{index}");
        let attention_prefix = format!("{prefix}.attention");
        blocks.push(WavLmBlockWeights {
            attention_norm: norm(file, &format!("{prefix}.attention_norm"), WAVLM_DIM)?,
            attention: WavLmAttentionWeights {
                q: linear(
                    file,
                    &format!("{attention_prefix}.q_proj"),
                    WAVLM_DIM,
                    WAVLM_DIM,
                )?,
                k: linear(
                    file,
                    &format!("{attention_prefix}.k_proj"),
                    WAVLM_DIM,
                    WAVLM_DIM,
                )?,
                v: linear(
                    file,
                    &format!("{attention_prefix}.v_proj"),
                    WAVLM_DIM,
                    WAVLM_DIM,
                )?,
                out: linear(
                    file,
                    &format!("{attention_prefix}.out_proj"),
                    WAVLM_DIM,
                    WAVLM_DIM,
                )?,
                gru_weight: tensor(
                    file,
                    &format!("{attention_prefix}.gru_rel_pos_linear.weight"),
                    &[8, WAVLM_HEAD_DIM],
                )?,
                gru_bias: tensor(
                    file,
                    &format!("{attention_prefix}.gru_rel_pos_linear.bias"),
                    &[8],
                )?,
                gru_const: tensor(
                    file,
                    &format!("{attention_prefix}.gru_rel_pos_const"),
                    &[1, WAVLM_HEADS, 1, 1],
                )?,
            },
            feed_forward_norm: norm(file, &format!("{prefix}.feed_forward_norm"), WAVLM_DIM)?,
            feed_forward_in: linear(
                file,
                &format!("{prefix}.feed_forward.in_proj"),
                WAVLM_DIM,
                WAVLM_FFN,
            )?,
            feed_forward_out: linear(
                file,
                &format!("{prefix}.feed_forward.out_proj"),
                WAVLM_FFN,
                WAVLM_DIM,
            )?,
        });
    }

    Ok(WavLmWeights {
        feature_layers,
        input_norm: norm(file, "encoder.norm", FEATURE_DIM)?,
        feature_proj: linear(file, "encoder.feature_proj", FEATURE_DIM, WAVLM_DIM)?,
        positional_conv: ConvWeights {
            weight: tensor(
                file,
                "encoder.encoder.positional_embedding.conv.weight",
                &[WAVLM_DIM, WAVLM_DIM / WAVLM_HEADS, 128],
            )?,
            bias: tensor(
                file,
                "encoder.encoder.positional_embedding.conv.bias",
                &[WAVLM_DIM],
            )?,
        },
        relative_embedding: tensor(
            file,
            "encoder.encoder.relative_embedding.weight",
            &[RELATIVE_BUCKETS, WAVLM_HEADS],
        )?,
        blocks,
        unused_output_norm: norm(file, "encoder.encoder.norm", WAVLM_DIM)?,
    })
}

fn bind_compressor(file: &GgufFile, variant: FocalCodecVariant) -> Result<FocalEncoderWeights> {
    const INPUTS: [usize; 3] = [1_024, 1_024, 512];
    const DIMS: [usize; 3] = [1_024, 512, 256];
    let factors = variant.downscale_factors();
    let mut scales = Vec::with_capacity(3);
    let mut blocks = Vec::with_capacity(3);
    for index in 0..3 {
        let scale = format!("compressor.layers.{index}.0");
        scales.push(ScaleWeights {
            conv: ConvWeights {
                weight: tensor(
                    file,
                    &format!("{scale}.downscale.weight"),
                    &[DIMS[index], INPUTS[index], factors[index]],
                )?,
                bias: tensor(file, &format!("{scale}.downscale.bias"), &[DIMS[index]])?,
            },
            alpha: tensor(
                file,
                &format!("{scale}.activation.alpha"),
                &[DIMS[index], 1],
            )?,
        });
        blocks.push(focal_block(
            file,
            &format!("compressor.layers.{index}.1"),
            DIMS[index],
        )?);
    }
    Ok(FocalEncoderWeights {
        scales,
        blocks,
        out_proj: linear(file, "compressor.out_proj", 256, 13)?,
    })
}

fn bind_decompressor(file: &GgufFile, variant: FocalCodecVariant) -> Result<FocalDecoderWeights> {
    const DIMS: [usize; 3] = [256, 512, 1_024];
    const OUTPUTS: [usize; 3] = [512, 1_024, 1_024];
    let factors = variant.upscale_factors();
    let mut blocks = Vec::with_capacity(3);
    let mut scales = Vec::with_capacity(3);
    for index in 0..3 {
        let prefix = format!("decompressor.layers.{index}");
        blocks.push(focal_block(file, &format!("{prefix}.0"), DIMS[index])?);
        scales.push(ScaleWeights {
            conv: ConvWeights {
                // PyTorch ConvTranspose1d layout is [in, out, kernel].
                weight: tensor(
                    file,
                    &format!("{prefix}.1.upscale.weight"),
                    &[DIMS[index], OUTPUTS[index], factors[index]],
                )?,
                bias: tensor(file, &format!("{prefix}.1.upscale.bias"), &[OUTPUTS[index]])?,
            },
            alpha: tensor(
                file,
                &format!("{prefix}.1.activation.alpha"),
                &[DIMS[index], 1],
            )?,
        });
    }
    Ok(FocalDecoderWeights {
        in_proj: linear(file, "decompressor.in_proj", 13, 256)?,
        blocks,
        scales,
    })
}

fn focal_block(file: &GgufFile, prefix: &str, dim: usize) -> Result<FocalBlockWeights> {
    let modulation = format!("{prefix}.modulation");
    Ok(FocalBlockWeights {
        modulation_norm: norm(file, &format!("{prefix}.modulation_norm"), dim)?,
        modulation: FocalModulationWeights {
            in_proj: linear(file, &format!("{modulation}.in_proj"), dim, 2 * dim + 3)?,
            depthwise: [
                ConvWeights {
                    weight: tensor(
                        file,
                        &format!("{modulation}.layers.0.0.weight"),
                        &[dim, 1, 7],
                    )?,
                    bias: tensor(file, &format!("{modulation}.layers.0.0.bias"), &[dim])?,
                },
                ConvWeights {
                    weight: tensor(
                        file,
                        &format!("{modulation}.layers.1.0.weight"),
                        &[dim, 1, 9],
                    )?,
                    bias: tensor(file, &format!("{modulation}.layers.1.0.bias"), &[dim])?,
                },
            ],
            context_proj: ConvWeights {
                weight: tensor(
                    file,
                    &format!("{modulation}.context_proj.weight"),
                    &[dim, dim, 1],
                )?,
                bias: tensor(file, &format!("{modulation}.context_proj.bias"), &[dim])?,
            },
            out_proj: linear(file, &format!("{modulation}.out_proj"), dim, dim)?,
        },
        feed_forward_norm: norm(file, &format!("{prefix}.feed_forward_norm"), dim)?,
        feed_forward_in: linear(
            file,
            &format!("{prefix}.feed_forward.in_proj"),
            dim,
            4 * dim,
        )?,
        feed_forward_out: linear(
            file,
            &format!("{prefix}.feed_forward.out_proj"),
            4 * dim,
            dim,
        )?,
    })
}

fn bind_vocos(file: &GgufFile) -> Result<VocosWeights> {
    const DIM: usize = 512;
    const INPUT: usize = 1_024;
    const INTERMEDIATE: usize = 1_536;
    let mut blocks = Vec::with_capacity(8);
    for index in 0..8 {
        let prefix = format!("decoder.backbone.layers.{index}");
        blocks.push(VocosBlockWeights {
            depthwise_weight: tensor(file, &format!("{prefix}.dwconv.weight"), &[DIM, 1, 7])?,
            depthwise_bias: tensor(file, &format!("{prefix}.dwconv.bias"), &[DIM])?,
            norm: VocosNormWeights {
                scale: tensor(file, &format!("{prefix}.norm.weight"), &[DIM])?,
                shift: tensor(file, &format!("{prefix}.norm.bias"), &[DIM])?,
            },
            pointwise1_weight: tensor(
                file,
                &format!("{prefix}.pwconv1.weight"),
                &[INTERMEDIATE, DIM],
            )?,
            pointwise1_bias: tensor(file, &format!("{prefix}.pwconv1.bias"), &[INTERMEDIATE])?,
            pointwise2_weight: tensor(
                file,
                &format!("{prefix}.pwconv2.weight"),
                &[DIM, INTERMEDIATE],
            )?,
            pointwise2_bias: tensor(file, &format!("{prefix}.pwconv2.bias"), &[DIM])?,
            gamma: tensor(file, &format!("{prefix}.gamma"), &[DIM])?,
        });
    }
    Ok(VocosWeights {
        embed_weight: tensor(file, "decoder.backbone.embedding.weight", &[DIM, INPUT, 7])?,
        embed_bias: tensor(file, "decoder.backbone.embedding.bias", &[DIM])?,
        norm: VocosNormWeights {
            scale: tensor(file, "decoder.backbone.input_norm.weight", &[DIM])?,
            shift: tensor(file, "decoder.backbone.input_norm.bias", &[DIM])?,
        },
        blocks,
        final_norm_weight: tensor(file, "decoder.backbone.output_norm.weight", &[DIM])?,
        final_norm_bias: tensor(file, "decoder.backbone.output_norm.bias", &[DIM])?,
        head_weight: tensor(file, "decoder.head.proj.weight", &[1_026, DIM])?,
        head_bias: tensor(file, "decoder.head.proj.bias", &[1_026])?,
    })
}

fn linear(file: &GgufFile, prefix: &str, input: usize, output: usize) -> Result<LinearWeights> {
    Ok(LinearWeights {
        weight: tensor(file, &format!("{prefix}.weight"), &[output, input])?,
        bias: tensor(file, &format!("{prefix}.bias"), &[output])?,
    })
}

fn norm(file: &GgufFile, prefix: &str, dim: usize) -> Result<NormWeights> {
    Ok(NormWeights {
        weight: tensor(file, &format!("{prefix}.weight"), &[dim])?,
        bias: tensor(file, &format!("{prefix}.bias"), &[dim])?,
    })
}

fn tensor(file: &GgufFile, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
    load_tensor(file, LABEL, name, shape)
}
