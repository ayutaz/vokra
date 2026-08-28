use vokra_core::Result;
use vokra_core::gguf::GgufFile;

use crate::strict_checkpoint::load_tensor;

use super::{FEATURE_DNR, FEATURE_SPEECH, INTERNAL_CHANNELS, TigerVariant};

const LABEL: &str = "tiger";

#[derive(Debug, Clone)]
pub(super) struct Norm {
    pub(super) gamma: Vec<f32>,
    pub(super) beta: Vec<f32>,
}

#[derive(Debug, Clone)]
pub(super) struct Pointwise {
    /// Row-major [input, output], transposed once from PyTorch [output, input].
    pub(super) weight_t: Vec<f32>,
    pub(super) bias: Option<Vec<f32>>,
    pub(super) input: usize,
    pub(super) output: usize,
}

#[derive(Debug, Clone)]
pub(super) struct GroupedConv {
    /// PyTorch layout [output, input / groups, kernel].
    pub(super) weight: Vec<f32>,
    pub(super) bias: Option<Vec<f32>>,
    pub(super) input: usize,
    pub(super) output: usize,
    pub(super) kernel: usize,
    pub(super) stride: usize,
    pub(super) padding: usize,
    pub(super) groups: usize,
}

#[derive(Debug, Clone)]
pub(super) struct PointwiseNorm {
    pub(super) conv: Pointwise,
    pub(super) norm: Norm,
}

#[derive(Debug, Clone)]
pub(super) struct DepthNorm {
    pub(super) conv: GroupedConv,
    pub(super) norm: Norm,
}

#[derive(Debug, Clone)]
pub(super) struct BandInput {
    pub(super) norm: Norm,
    pub(super) projection: Pointwise,
}

#[derive(Debug, Clone)]
pub(super) struct BandMask {
    pub(super) slope: f32,
    pub(super) projection: GroupedConv,
}

#[derive(Debug, Clone)]
pub(super) struct Injection {
    pub(super) local: DepthNorm,
    pub(super) global: DepthNorm,
    pub(super) gate: DepthNorm,
}

#[derive(Debug, Clone)]
pub(super) struct Mlp {
    pub(super) first: PointwiseNorm,
    pub(super) depthwise: GroupedConv,
    pub(super) second: PointwiseNorm,
}

#[derive(Debug, Clone)]
pub(super) struct UConv {
    pub(super) projection: Pointwise,
    pub(super) projection_norm: Norm,
    pub(super) projection_slope: f32,
    pub(super) downsample: Vec<DepthNorm>,
    pub(super) scale_fusion: Vec<Injection>,
    pub(super) global_mlp: Mlp,
    pub(super) expansion: Vec<Injection>,
    pub(super) residual_projection: Pointwise,
}

#[derive(Debug, Clone)]
pub(super) struct AttentionProjection {
    pub(super) projection: Pointwise,
    pub(super) slope: f32,
    pub(super) norm: Norm,
}

#[derive(Debug, Clone)]
pub(super) struct Attention {
    pub(super) queries: Vec<AttentionProjection>,
    pub(super) keys: Vec<AttentionProjection>,
    pub(super) values: Vec<AttentionProjection>,
    pub(super) output: AttentionProjection,
}

#[derive(Debug, Clone)]
pub(super) struct Path {
    pub(super) uconv: UConv,
    pub(super) attention: Attention,
    pub(super) norm: Norm,
}

#[derive(Debug, Clone)]
pub(super) struct Separator {
    pub(super) concat: GroupedConv,
    pub(super) concat_slope: f32,
    pub(super) frequency: Path,
    pub(super) frame: Path,
}

#[derive(Debug, Clone)]
pub(super) struct CoreWeights {
    pub(super) bands: Vec<BandInput>,
    pub(super) separator: Separator,
    pub(super) masks: Vec<BandMask>,
}

#[derive(Debug, Clone)]
pub(super) enum TigerWeights {
    Dnr {
        dialog: Box<CoreWeights>,
        effect: Box<CoreWeights>,
        music: Box<CoreWeights>,
    },
    Speech(Box<CoreWeights>),
}

impl TigerWeights {
    pub(super) fn bind(file: &GgufFile, variant: TigerVariant) -> Result<Self> {
        match variant {
            TigerVariant::Dnr => Ok(Self::Dnr {
                dialog: Box::new(load_core(file, "dialog.", variant)?),
                effect: Box::new(load_core(file, "effect.", variant)?),
                music: Box::new(load_core(file, "music.", variant)?),
            }),
            TigerVariant::Speech => Ok(Self::Speech(Box::new(load_core(file, "", variant)?))),
        }
    }
}

fn load_core(file: &GgufFile, root: &str, variant: TigerVariant) -> Result<CoreWeights> {
    let channels = match variant {
        TigerVariant::Dnr => FEATURE_DNR,
        TigerVariant::Speech => FEATURE_SPEECH,
    };
    let sources = variant.output_streams();
    let widths = variant.band_widths();
    let mut bands = Vec::with_capacity(widths.len());
    let mut masks = Vec::with_capacity(widths.len());
    for (band, &width) in widths.iter().enumerate() {
        let input = 2 * width;
        bands.push(BandInput {
            norm: load_norm(file, &format!("{root}BN.{band}.0"), input)?,
            projection: load_pointwise(
                file,
                &format!("{root}BN.{band}.1"),
                input,
                channels,
                true,
                1,
            )?,
        });
        masks.push(BandMask {
            slope: load_scalar(file, &format!("{root}mask.{band}.0.weight"))?,
            projection: load_grouped(
                file,
                &format!("{root}mask.{band}.1"),
                channels,
                4 * sources * width,
                1,
                1,
                0,
                sources,
                true,
            )?,
        });
    }

    Ok(CoreWeights {
        bands,
        separator: Separator {
            concat: load_grouped_pointwise_2d(
                file,
                &format!("{root}separator.concat_block.0"),
                channels,
                channels,
                channels,
                true,
            )?,
            concat_slope: load_scalar(file, &format!("{root}separator.concat_block.1.weight"))?,
            frequency: load_path(file, &format!("{root}separator.freq_path"), channels)?,
            frame: load_path(file, &format!("{root}separator.frame_path"), channels)?,
        },
        masks,
    })
}

fn load_grouped_pointwise_2d(
    file: &GgufFile,
    prefix: &str,
    input: usize,
    output: usize,
    groups: usize,
    bias: bool,
) -> Result<GroupedConv> {
    let weight = load_tensor(
        file,
        LABEL,
        &format!("{prefix}.weight"),
        &[output, input / groups, 1, 1],
    )?;
    let bias = bias
        .then(|| load_tensor(file, LABEL, &format!("{prefix}.bias"), &[output]))
        .transpose()?;
    Ok(GroupedConv {
        weight,
        bias,
        input,
        output,
        kernel: 1,
        stride: 1,
        padding: 0,
        groups,
    })
}

fn load_path(file: &GgufFile, prefix: &str, channels: usize) -> Result<Path> {
    Ok(Path {
        uconv: load_uconv(file, &format!("{prefix}.0"), channels)?,
        attention: load_attention(file, &format!("{prefix}.1"), channels)?,
        norm: load_norm_4d(file, &format!("{prefix}.2"), channels)?,
    })
}

fn load_uconv(file: &GgufFile, prefix: &str, channels: usize) -> Result<UConv> {
    let projection = format!("{prefix}.proj_1x1");
    let mut downsample = Vec::with_capacity(5);
    let mut scale_fusion = Vec::with_capacity(5);
    let mut expansion = Vec::with_capacity(4);
    for stage in 0..5 {
        downsample.push(load_depth_norm(
            file,
            &format!("{prefix}.spp_dw.{stage}"),
            5,
            if stage == 0 { 1 } else { 2 },
            2,
            true,
        )?);
        scale_fusion.push(load_injection(
            file,
            &format!("{prefix}.loc_glo_fus.{stage}"),
            1,
        )?);
    }
    for stage in 0..4 {
        expansion.push(load_injection(
            file,
            &format!("{prefix}.last_layer.{stage}"),
            5,
        )?);
    }
    Ok(UConv {
        projection: load_pointwise(
            file,
            &format!("{projection}.conv"),
            channels,
            INTERNAL_CHANNELS,
            true,
            1,
        )?,
        projection_norm: load_norm(file, &format!("{projection}.norm"), INTERNAL_CHANNELS)?,
        projection_slope: load_scalar(file, &format!("{projection}.act.weight"))?,
        downsample,
        scale_fusion,
        global_mlp: Mlp {
            first: load_pointwise_norm(
                file,
                &format!("{prefix}.globalatt.fc1"),
                INTERNAL_CHANNELS,
            )?,
            depthwise: load_grouped(
                file,
                &format!("{prefix}.globalatt.dwconv"),
                INTERNAL_CHANNELS,
                INTERNAL_CHANNELS,
                5,
                1,
                2,
                INTERNAL_CHANNELS,
                true,
            )?,
            second: load_pointwise_norm(
                file,
                &format!("{prefix}.globalatt.fc2"),
                INTERNAL_CHANNELS,
            )?,
        },
        expansion,
        residual_projection: load_pointwise(
            file,
            &format!("{prefix}.res_conv"),
            INTERNAL_CHANNELS,
            channels,
            true,
            1,
        )?,
    })
}

fn load_injection(file: &GgufFile, prefix: &str, kernel: usize) -> Result<Injection> {
    Ok(Injection {
        local: load_depth_norm(
            file,
            &format!("{prefix}.local_embedding"),
            kernel,
            1,
            kernel / 2,
            false,
        )?,
        global: load_depth_norm(
            file,
            &format!("{prefix}.global_embedding"),
            kernel,
            1,
            kernel / 2,
            false,
        )?,
        gate: load_depth_norm(
            file,
            &format!("{prefix}.global_act"),
            kernel,
            1,
            kernel / 2,
            false,
        )?,
    })
}

fn load_attention(file: &GgufFile, prefix: &str, channels: usize) -> Result<Attention> {
    let mut queries = Vec::with_capacity(4);
    let mut keys = Vec::with_capacity(4);
    let mut values = Vec::with_capacity(4);
    for head in 0..4 {
        queries.push(load_attention_projection(
            file,
            &format!("{prefix}.Queries.{head}"),
            channels,
            4,
        )?);
        keys.push(load_attention_projection(
            file,
            &format!("{prefix}.Keys.{head}"),
            channels,
            4,
        )?);
        values.push(load_attention_projection(
            file,
            &format!("{prefix}.Values.{head}"),
            channels,
            channels / 4,
        )?);
    }
    Ok(Attention {
        queries,
        keys,
        values,
        output: load_attention_projection(
            file,
            &format!("{prefix}.attn_concat_proj"),
            channels,
            channels,
        )?,
    })
}

fn load_attention_projection(
    file: &GgufFile,
    prefix: &str,
    input: usize,
    output: usize,
) -> Result<AttentionProjection> {
    Ok(AttentionProjection {
        projection: load_pointwise(file, &format!("{prefix}.conv"), input, output, true, 2)?,
        slope: load_scalar(file, &format!("{prefix}.act.weight"))?,
        norm: load_norm_4d(file, &format!("{prefix}.norm"), output)?,
    })
}

fn load_pointwise_norm(file: &GgufFile, prefix: &str, channels: usize) -> Result<PointwiseNorm> {
    Ok(PointwiseNorm {
        conv: load_pointwise(
            file,
            &format!("{prefix}.conv"),
            channels,
            channels,
            false,
            1,
        )?,
        norm: load_norm(file, &format!("{prefix}.norm"), channels)?,
    })
}

fn load_depth_norm(
    file: &GgufFile,
    prefix: &str,
    kernel: usize,
    stride: usize,
    padding: usize,
    bias: bool,
) -> Result<DepthNorm> {
    Ok(DepthNorm {
        conv: load_grouped(
            file,
            &format!("{prefix}.conv"),
            INTERNAL_CHANNELS,
            INTERNAL_CHANNELS,
            kernel,
            stride,
            padding,
            INTERNAL_CHANNELS,
            bias,
        )?,
        norm: load_norm(file, &format!("{prefix}.norm"), INTERNAL_CHANNELS)?,
    })
}

fn load_pointwise(
    file: &GgufFile,
    prefix: &str,
    input: usize,
    output: usize,
    bias: bool,
    trailing_ones: usize,
) -> Result<Pointwise> {
    let mut shape = vec![output, input];
    shape.extend(std::iter::repeat_n(1, trailing_ones));
    let raw = load_tensor(file, LABEL, &format!("{prefix}.weight"), &shape)?;
    let mut weight_t = vec![0.0f32; raw.len()];
    for out in 0..output {
        for inner in 0..input {
            weight_t[inner * output + out] = raw[out * input + inner];
        }
    }
    let bias = bias
        .then(|| load_tensor(file, LABEL, &format!("{prefix}.bias"), &[output]))
        .transpose()?;
    Ok(Pointwise {
        weight_t,
        bias,
        input,
        output,
    })
}

#[allow(clippy::too_many_arguments)]
fn load_grouped(
    file: &GgufFile,
    prefix: &str,
    input: usize,
    output: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    groups: usize,
    bias: bool,
) -> Result<GroupedConv> {
    let weight = load_tensor(
        file,
        LABEL,
        &format!("{prefix}.weight"),
        &[output, input / groups, kernel],
    )?;
    let bias = bias
        .then(|| load_tensor(file, LABEL, &format!("{prefix}.bias"), &[output]))
        .transpose()?;
    Ok(GroupedConv {
        weight,
        bias,
        input,
        output,
        kernel,
        stride,
        padding,
        groups,
    })
}

fn load_norm(file: &GgufFile, prefix: &str, channels: usize) -> Result<Norm> {
    Ok(Norm {
        gamma: load_tensor(file, LABEL, &format!("{prefix}.weight"), &[channels])?,
        beta: load_tensor(file, LABEL, &format!("{prefix}.bias"), &[channels])?,
    })
}

fn load_norm_4d(file: &GgufFile, prefix: &str, channels: usize) -> Result<Norm> {
    Ok(Norm {
        gamma: load_tensor(
            file,
            LABEL,
            &format!("{prefix}.gamma"),
            &[1, channels, 1, 1],
        )?,
        beta: load_tensor(file, LABEL, &format!("{prefix}.beta"), &[1, channels, 1, 1])?,
    })
}

fn load_scalar(file: &GgufFile, name: &str) -> Result<f32> {
    Ok(load_tensor(file, LABEL, name, &[1])?[0])
}
