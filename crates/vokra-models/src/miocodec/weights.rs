//! Strict decoder-weight binding for the pinned MioCodec v2 checkpoint.

use vokra_core::gguf::GgufFile;
use vokra_core::{Result, VokraError};
use vokra_ops::FsqOutProj;

use crate::strict_checkpoint::load_tensor;

use super::{
    CODE_DIM, CONTENT_DIM, GLOBAL_DIM, PRENET_HEADS, PRENET_HIDDEN, PRENET_LAYERS,
    WAVE_DECODER_HEADS, WAVE_DECODER_HIDDEN, WAVE_DECODER_LAYERS, WAVE_DIM,
};

const LABEL: &str = "miocodec";

#[derive(Debug, Clone)]
pub(super) struct Linear {
    /// Transposed PyTorch `[out, in]` weight, stored as `[in, out]` for GEMM.
    pub(super) weight_t: Vec<f32>,
    pub(super) bias: Option<Vec<f32>>,
    pub(super) input: usize,
    pub(super) output: usize,
}

#[derive(Debug, Clone)]
pub(super) struct AffineNorm {
    pub(super) weight: Vec<f32>,
    pub(super) bias: Vec<f32>,
}

#[derive(Debug, Clone)]
pub(super) struct Conv1d {
    pub(super) weight: Vec<f32>,
    pub(super) bias: Vec<f32>,
    pub(super) input: usize,
    pub(super) output: usize,
    pub(super) kernel: usize,
}

#[derive(Debug, Clone)]
pub(super) struct ConvTranspose1d {
    /// Reversed `[out, in, kernel]` kernel usable as a normal Conv1d over a
    /// zero-inserted input.
    pub(super) conv_weight: Vec<f32>,
    pub(super) bias: Vec<f32>,
    pub(super) input: usize,
    pub(super) output: usize,
    pub(super) kernel: usize,
    pub(super) stride: usize,
    pub(super) padding: usize,
}

#[derive(Debug, Clone)]
pub(super) struct ResnetBlock {
    pub(super) norm1: AffineNorm,
    pub(super) conv1: Conv1d,
    pub(super) norm2: AffineNorm,
    pub(super) conv2: Conv1d,
    pub(super) groups: usize,
}

#[derive(Debug, Clone)]
pub(super) struct Attention {
    pub(super) q: Linear,
    pub(super) k: Linear,
    pub(super) v: Linear,
    pub(super) out: Linear,
    pub(super) heads: usize,
    pub(super) window: usize,
}

#[derive(Debug, Clone)]
pub(super) struct FeedForward {
    pub(super) w1: Linear,
    pub(super) w2: Linear,
    pub(super) w3: Linear,
}

#[derive(Debug, Clone)]
pub(super) struct AffineTransformerBlock {
    pub(super) attention: Attention,
    pub(super) attention_norm: AffineNorm,
    pub(super) feed_forward: FeedForward,
    pub(super) ffn_norm: AffineNorm,
}

#[derive(Debug, Clone)]
pub(super) struct AffineTransformer {
    pub(super) layers: Vec<AffineTransformerBlock>,
    pub(super) norm: AffineNorm,
    pub(super) output_projection: Linear,
}

#[derive(Debug, Clone)]
pub(super) struct AdaTransformerBlock {
    pub(super) attention: Attention,
    pub(super) attention_condition: Linear,
    pub(super) feed_forward: FeedForward,
    pub(super) ffn_condition: Linear,
}

#[derive(Debug, Clone)]
pub(super) struct AdaTransformer {
    pub(super) layers: Vec<AdaTransformerBlock>,
    pub(super) final_condition: Linear,
}

#[derive(Debug, Clone)]
pub(super) struct UpsampleStage {
    pub(super) transpose: ConvTranspose1d,
    /// Effective (already exponentiated) SnakeBeta parameters.
    pub(super) alpha: Vec<f32>,
    pub(super) beta: Vec<f32>,
    pub(super) resnet: ResnetBlock,
}

#[derive(Debug, Clone)]
pub(super) struct WaveUpsampler {
    pub(super) stages: Vec<UpsampleStage>,
    pub(super) output_projection: Linear,
    /// Effective (already exponentiated) terminal SnakeBeta parameters.
    pub(super) output_alpha: Vec<f32>,
    pub(super) output_beta: Vec<f32>,
}

#[derive(Debug, Clone)]
pub(super) struct MioCodecWeights {
    pub(super) fsq_output: FsqOutProj,
    pub(super) prenet: AffineTransformer,
    pub(super) first_upsample: ConvTranspose1d,
    pub(super) prior: Vec<ResnetBlock>,
    pub(super) decoder: AdaTransformer,
    pub(super) post: Vec<ResnetBlock>,
    pub(super) upsampler: WaveUpsampler,
    pub(super) istft_projection: Linear,
}

impl MioCodecWeights {
    pub(super) fn load(file: &GgufFile) -> Result<Self> {
        let fsq_output = FsqOutProj::new(
            CONTENT_DIM,
            CODE_DIM,
            load_tensor(
                file,
                LABEL,
                "local_quantizer.proj_out.weight",
                &[CONTENT_DIM, CODE_DIM],
            )?,
            load_tensor(file, LABEL, "local_quantizer.proj_out.bias", &[CONTENT_DIM])?,
        )
        .map_err(|error| VokraError::ModelLoad(format!("miocodec FSQ projection: {error}")))?;

        let prenet = load_affine_transformer(
            file,
            "wave_prenet",
            PRENET_LAYERS,
            CONTENT_DIM,
            PRENET_HIDDEN,
            PRENET_HEADS,
            65,
            WAVE_DIM,
        )?;
        let first_upsample =
            load_conv_transpose(file, "wave_conv_upsample", WAVE_DIM, WAVE_DIM, 2, 2, 0)?;
        let prior = load_resnet_stack(file, "wave_prior_net", 2, WAVE_DIM, 32)?;
        let decoder = load_ada_transformer(file)?;
        let post = load_resnet_stack(file, "wave_post_net", 2, WAVE_DIM, 32)?;
        let upsampler = load_wave_upsampler(file)?;
        let istft_projection = load_linear(file, "istft_head.out", WAVE_DIM, 394, true)?;

        Ok(Self {
            fsq_output,
            prenet,
            first_upsample,
            prior,
            decoder,
            post,
            upsampler,
            istft_projection,
        })
    }
}

fn load_affine_transformer(
    file: &GgufFile,
    root: &str,
    layers: usize,
    dim: usize,
    hidden: usize,
    heads: usize,
    window: usize,
    output_dim: usize,
) -> Result<AffineTransformer> {
    let mut blocks = Vec::with_capacity(layers);
    for layer in 0..layers {
        let prefix = format!("{root}.layers.{layer}");
        blocks.push(AffineTransformerBlock {
            attention: load_attention(file, &prefix, dim, heads, window)?,
            attention_norm: load_affine_norm(file, &format!("{prefix}.attention_norm"), dim)?,
            feed_forward: load_feed_forward(file, &prefix, dim, hidden)?,
            ffn_norm: load_affine_norm(file, &format!("{prefix}.ffn_norm"), dim)?,
        });
    }
    Ok(AffineTransformer {
        layers: blocks,
        norm: load_affine_norm(file, &format!("{root}.norm"), dim)?,
        output_projection: load_linear(
            file,
            &format!("{root}.output_proj"),
            dim,
            output_dim,
            true,
        )?,
    })
}

fn load_ada_transformer(file: &GgufFile) -> Result<AdaTransformer> {
    let mut blocks = Vec::with_capacity(WAVE_DECODER_LAYERS);
    for layer in 0..WAVE_DECODER_LAYERS {
        let prefix = format!("wave_decoder.layers.{layer}");
        blocks.push(AdaTransformerBlock {
            attention: load_attention(file, &prefix, WAVE_DIM, WAVE_DECODER_HEADS, 65)?,
            attention_condition: load_linear(
                file,
                &format!("{prefix}.attention_norm.condition_proj.1"),
                GLOBAL_DIM,
                3 * WAVE_DIM,
                true,
            )?,
            feed_forward: load_feed_forward(file, &prefix, WAVE_DIM, WAVE_DECODER_HIDDEN)?,
            ffn_condition: load_linear(
                file,
                &format!("{prefix}.ffn_norm.condition_proj.1"),
                GLOBAL_DIM,
                3 * WAVE_DIM,
                true,
            )?,
        });
    }
    Ok(AdaTransformer {
        layers: blocks,
        final_condition: load_linear(
            file,
            "wave_decoder.norm.condition_proj.1",
            GLOBAL_DIM,
            2 * WAVE_DIM,
            true,
        )?,
    })
}

fn load_attention(
    file: &GgufFile,
    prefix: &str,
    dim: usize,
    heads: usize,
    window: usize,
) -> Result<Attention> {
    let attention = format!("{prefix}.attention");
    Ok(Attention {
        q: load_linear(file, &format!("{attention}.wq"), dim, dim, false)?,
        k: load_linear(file, &format!("{attention}.wk"), dim, dim, false)?,
        v: load_linear(file, &format!("{attention}.wv"), dim, dim, false)?,
        out: load_linear(file, &format!("{attention}.wo"), dim, dim, false)?,
        heads,
        window,
    })
}

fn load_feed_forward(
    file: &GgufFile,
    prefix: &str,
    dim: usize,
    hidden: usize,
) -> Result<FeedForward> {
    let root = format!("{prefix}.feed_forward");
    Ok(FeedForward {
        w1: load_linear(file, &format!("{root}.w1"), dim, hidden, false)?,
        w2: load_linear(file, &format!("{root}.w2"), hidden, dim, false)?,
        w3: load_linear(file, &format!("{root}.w3"), dim, hidden, false)?,
    })
}

fn load_affine_norm(file: &GgufFile, prefix: &str, dim: usize) -> Result<AffineNorm> {
    Ok(AffineNorm {
        weight: load_tensor(file, LABEL, &format!("{prefix}.weight"), &[dim])?,
        bias: load_tensor(file, LABEL, &format!("{prefix}.bias"), &[dim])?,
    })
}

fn load_resnet_stack(
    file: &GgufFile,
    root: &str,
    blocks: usize,
    channels: usize,
    groups: usize,
) -> Result<Vec<ResnetBlock>> {
    (0..blocks)
        .map(|block| load_resnet(file, &format!("{root}.blocks.{block}"), channels, groups))
        .collect()
}

fn load_resnet(
    file: &GgufFile,
    prefix: &str,
    channels: usize,
    groups: usize,
) -> Result<ResnetBlock> {
    Ok(ResnetBlock {
        norm1: load_affine_norm(file, &format!("{prefix}.norm1"), channels)?,
        conv1: load_conv(file, &format!("{prefix}.conv1"), channels, channels, 3)?,
        norm2: load_affine_norm(file, &format!("{prefix}.norm2"), channels)?,
        conv2: load_conv(file, &format!("{prefix}.conv2"), channels, channels, 3)?,
        groups,
    })
}

fn load_wave_upsampler(file: &GgufFile) -> Result<WaveUpsampler> {
    let mut stages = Vec::with_capacity(2);
    for (stage, input, output) in [(0, 512, 256), (1, 256, 128)] {
        let root = format!("wave_upsampler.upsample_layers.{stage}");
        let g = load_tensor(
            file,
            LABEL,
            &format!("{root}.parametrizations.weight.original0"),
            &[input, 1, 1],
        )?;
        let v = load_tensor(
            file,
            LABEL,
            &format!("{root}.parametrizations.weight.original1"),
            &[input, output, 9],
        )?;
        let effective = fold_weight_norm(&g, &v, input, output, 9)?;
        let bias = load_tensor(file, LABEL, &format!("{root}.bias"), &[output])?;
        let transpose = make_conv_transpose(effective, bias, input, output, 9, 3, 3)?;
        let snake_root = format!("wave_upsampler.snake_activations.{stage}");
        let alpha = exp_parameters(load_tensor(
            file,
            LABEL,
            &format!("{snake_root}.alpha"),
            &[output],
        )?)?;
        let beta = exp_parameters(load_tensor(
            file,
            LABEL,
            &format!("{snake_root}.beta"),
            &[output],
        )?)?;
        let resnet = load_resnet(
            file,
            &format!("wave_upsampler.resnet_blocks.{stage}"),
            output,
            32.min(output),
        )?;
        stages.push(UpsampleStage {
            transpose,
            alpha,
            beta,
            resnet,
        });
    }
    Ok(WaveUpsampler {
        stages,
        output_projection: load_linear(file, "wave_upsampler.out_proj", 128, WAVE_DIM, true)?,
        output_alpha: exp_parameters(load_tensor(
            file,
            LABEL,
            "wave_upsampler.out_snake.alpha",
            &[WAVE_DIM],
        )?)?,
        output_beta: exp_parameters(load_tensor(
            file,
            LABEL,
            "wave_upsampler.out_snake.beta",
            &[WAVE_DIM],
        )?)?,
    })
}

fn exp_parameters(values: Vec<f32>) -> Result<Vec<f32>> {
    let mut output = Vec::with_capacity(values.len());
    for value in values {
        let effective = value.exp();
        if !effective.is_finite() {
            return Err(VokraError::ModelLoad(
                "miocodec: non-finite effective SnakeBeta parameter".to_owned(),
            ));
        }
        output.push(effective);
    }
    Ok(output)
}

fn fold_weight_norm(
    g: &[f32],
    v: &[f32],
    input: usize,
    output: usize,
    kernel: usize,
) -> Result<Vec<f32>> {
    if g.len() != input || v.len() != input * output * kernel {
        return Err(VokraError::ModelLoad(format!(
            "miocodec: weight-norm shape mismatch g={} v={} expected {} and {}",
            g.len(),
            v.len(),
            input,
            input * output * kernel
        )));
    }
    let mut weight = vec![0.0f32; v.len()];
    let per_input = output * kernel;
    for channel in 0..input {
        let start = channel * per_input;
        let source = &v[start..start + per_input];
        let norm = source.iter().map(|value| value * value).sum::<f32>().sqrt();
        if !norm.is_finite() || norm == 0.0 || !g[channel].is_finite() {
            return Err(VokraError::ModelLoad(format!(
                "miocodec: invalid weight norm at input channel {channel}: g={} norm={norm}",
                g[channel]
            )));
        }
        let scale = g[channel] / norm;
        for (target, &value) in weight[start..start + per_input].iter_mut().zip(source) {
            *target = value * scale;
        }
    }
    Ok(weight)
}

fn load_conv(
    file: &GgufFile,
    prefix: &str,
    input: usize,
    output: usize,
    kernel: usize,
) -> Result<Conv1d> {
    Ok(Conv1d {
        weight: load_tensor(
            file,
            LABEL,
            &format!("{prefix}.weight"),
            &[output, input, kernel],
        )?,
        bias: load_tensor(file, LABEL, &format!("{prefix}.bias"), &[output])?,
        input,
        output,
        kernel,
    })
}

#[allow(clippy::too_many_arguments)]
fn load_conv_transpose(
    file: &GgufFile,
    prefix: &str,
    input: usize,
    output: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
) -> Result<ConvTranspose1d> {
    let weight = load_tensor(
        file,
        LABEL,
        &format!("{prefix}.weight"),
        &[input, output, kernel],
    )?;
    let bias = load_tensor(file, LABEL, &format!("{prefix}.bias"), &[output])?;
    make_conv_transpose(weight, bias, input, output, kernel, stride, padding)
}

#[allow(clippy::too_many_arguments)]
fn make_conv_transpose(
    weight: Vec<f32>,
    bias: Vec<f32>,
    input: usize,
    output: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
) -> Result<ConvTranspose1d> {
    if weight.len() != input * output * kernel || bias.len() != output {
        return Err(VokraError::ModelLoad(
            "miocodec: ConvTranspose1d weight/bias shape mismatch".to_owned(),
        ));
    }
    let mut conv_weight = vec![0.0f32; weight.len()];
    for out_channel in 0..output {
        for in_channel in 0..input {
            for tap in 0..kernel {
                let source = (in_channel * output + out_channel) * kernel + tap;
                let target = (out_channel * input + in_channel) * kernel + (kernel - 1 - tap);
                conv_weight[target] = weight[source];
            }
        }
    }
    Ok(ConvTranspose1d {
        conv_weight,
        bias,
        input,
        output,
        kernel,
        stride,
        padding,
    })
}

fn load_linear(
    file: &GgufFile,
    prefix: &str,
    input: usize,
    output: usize,
    with_bias: bool,
) -> Result<Linear> {
    let weight = load_tensor(file, LABEL, &format!("{prefix}.weight"), &[output, input])?;
    let mut weight_t = vec![0.0f32; weight.len()];
    for out_feature in 0..output {
        for in_feature in 0..input {
            weight_t[in_feature * output + out_feature] = weight[out_feature * input + in_feature];
        }
    }
    let bias = with_bias
        .then(|| load_tensor(file, LABEL, &format!("{prefix}.bias"), &[output]))
        .transpose()?;
    Ok(Linear {
        weight_t,
        bias,
        input,
        output,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weight_norm_is_per_convtranspose_input_channel() {
        let g = [2.0, 3.0];
        let v = [3.0, 4.0, 0.0, 0.0, 5.0, 12.0, 0.0, 0.0];
        let got = fold_weight_norm(&g, &v, 2, 2, 2).unwrap();
        let expected = [1.2, 1.6, 0.0, 0.0, 15.0 / 13.0, 36.0 / 13.0, 0.0, 0.0];
        assert_eq!(got.len(), expected.len());
        for (actual, expected) in got.iter().zip(expected) {
            assert!((*actual - expected).abs() <= f32::EPSILON);
        }
    }

    #[test]
    fn convtranspose_kernel_is_transposed_and_reversed() {
        let got = make_conv_transpose(vec![1.0, 2.0, 3.0, 4.0], vec![0.0], 2, 1, 2, 2, 0).unwrap();
        assert_eq!(got.conv_weight, vec![2.0, 1.0, 4.0, 3.0]);
    }
}
