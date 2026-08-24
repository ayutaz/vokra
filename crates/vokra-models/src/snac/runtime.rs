//! Native runtime for the two public SNAC checkpoints.
//!
//! The implementation follows the upstream MIT `hubertsiuzdak/snac`
//! `Encoder`, `Decoder`, `ResidualVectorQuantize`, and `LocalMHA` modules.
//! Public Vokra GGUFs preserve the upstream state-dict names, including
//! PyTorch parametrization tensors `original0` (weight-norm magnitude) and
//! `original1` (direction). They are folded once while binding; no derived
//! tensors or replacement upload is required.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use vokra_core::gguf::{GgmlType, GgufFile, chunks};
use vokra_core::rng::GaussianSplitMix64;
use vokra_core::{BackendKind, LicenseClass, Result, VokraError};
use vokra_ops::hifigan::{HifiGanBackendOps, HifiGanConvPadding};
use vokra_ops::{CodebookTable, DacOutProj, SnacConfig as OpSnacConfig};

use crate::compute::{Compute, HotOp};
use crate::hifigan::HifiGanComputeOps;

use super::{ARCH, KEY_SNAC_VARIANT, SnacConfig, SnacVariant, VARIANT_TAG_HZ24, VARIANT_TAG_HZ44};

const CATEGORY: &str = "codec";
const KEY_CATEGORY: &str = "vokra.model.category";
const KEY_UPSTREAM_HF: &str = "vokra.provenance.upstream_hf";
const CODEBOOK_SIZE: usize = 4096;
const CODEBOOK_DIM: usize = 8;
const ATTENTION_WINDOW: usize = 32;
const ATTENTION_HEAD_DIM: usize = 64;

/// Complete learned-op set for SNAC decode on both public variants. The 24
/// kHz route does not invoke attention, but the common superset makes a bound
/// 44.1 kHz model fail at backend selection instead of half-running.
pub const SNAC_HOT_OPS: &[HotOp] = &[
    HotOp::SnacDecode,
    HotOp::Conv1d,
    HotOp::GroupedConv1d,
    HotOp::SnakeActivation,
    HotOp::Gemm,
    HotOp::Softmax,
    HotOp::LayerNorm,
];

impl SnacVariant {
    const fn model_name(self) -> &'static str {
        match self {
            Self::Hz24 => "snac-24khz",
            Self::Hz44 => "snac-44khz",
        }
    }

    const fn upstream_hf(self) -> &'static str {
        match self {
            Self::Hz24 => "hubertsiuzdak/snac_24khz",
            Self::Hz44 => "hubertsiuzdak/snac_44khz",
        }
    }

    const fn encoder_dim(self) -> usize {
        match self {
            Self::Hz24 => 48,
            Self::Hz44 => 64,
        }
    }

    const fn latent_dim(self) -> usize {
        self.encoder_dim() * 16
    }

    const fn decoder_dim(self) -> usize {
        match self {
            Self::Hz24 => 1024,
            Self::Hz44 => 1536,
        }
    }

    const fn encoder_rates(self) -> &'static [usize; 4] {
        match self {
            Self::Hz24 => &[2, 4, 8, 8],
            Self::Hz44 => &[2, 3, 8, 8],
        }
    }

    const fn decoder_rates(self) -> &'static [usize; 4] {
        match self {
            Self::Hz24 => &[8, 8, 4, 2],
            Self::Hz44 => &[8, 8, 3, 2],
        }
    }

    const fn has_attention(self) -> bool {
        matches!(self, Self::Hz44)
    }

    const fn tensor_count(self) -> usize {
        match self {
            Self::Hz24 => 269,
            Self::Hz44 => 286,
        }
    }
}

type Manifest = BTreeMap<String, Vec<usize>>;

fn add(manifest: &mut Manifest, name: impl Into<String>, shape: &[usize]) {
    let old = manifest.insert(name.into(), shape.to_vec());
    debug_assert!(old.is_none(), "duplicate SNAC manifest entry");
}

fn add_snake(manifest: &mut Manifest, name: &str, channels: usize) {
    add(manifest, name, &[1, channels, 1]);
}

fn add_wn_conv1d(
    manifest: &mut Manifest,
    prefix: &str,
    in_channels: usize,
    out_channels: usize,
    kernel: usize,
    groups: usize,
    bias: bool,
) {
    if bias {
        add(manifest, format!("{prefix}.bias"), &[out_channels]);
    }
    add(
        manifest,
        format!("{prefix}.parametrizations.weight.original0"),
        &[out_channels, 1, 1],
    );
    add(
        manifest,
        format!("{prefix}.parametrizations.weight.original1"),
        &[out_channels, in_channels / groups, kernel],
    );
}

fn add_wn_conv_transpose1d(
    manifest: &mut Manifest,
    prefix: &str,
    in_channels: usize,
    out_channels: usize,
    kernel: usize,
) {
    add(manifest, format!("{prefix}.bias"), &[out_channels]);
    add(
        manifest,
        format!("{prefix}.parametrizations.weight.original0"),
        &[in_channels, 1, 1],
    );
    add(
        manifest,
        format!("{prefix}.parametrizations.weight.original1"),
        &[in_channels, out_channels, kernel],
    );
}

fn add_residual(manifest: &mut Manifest, prefix: &str, channels: usize, depthwise: bool) {
    add_snake(manifest, &format!("{prefix}.block.0.alpha"), channels);
    add_wn_conv1d(
        manifest,
        &format!("{prefix}.block.1"),
        channels,
        channels,
        7,
        if depthwise { channels } else { 1 },
        true,
    );
    add_snake(manifest, &format!("{prefix}.block.2.alpha"), channels);
    add_wn_conv1d(
        manifest,
        &format!("{prefix}.block.3"),
        channels,
        channels,
        1,
        1,
        true,
    );
}

fn add_attention(manifest: &mut Manifest, prefix: &str, channels: usize) {
    add(manifest, format!("{prefix}.norm.bias"), &[channels]);
    add(manifest, format!("{prefix}.norm.weight"), &[channels]);
    add(
        manifest,
        format!("{prefix}.rel_pos.inv_freq"),
        &[ATTENTION_HEAD_DIM / 2],
    );
    add(
        manifest,
        format!("{prefix}.to_out.weight"),
        &[channels, channels],
    );
    add(
        manifest,
        format!("{prefix}.to_qkv.weight"),
        &[3 * channels, channels],
    );
}

pub(super) fn expected_manifest(variant: SnacVariant) -> Manifest {
    let mut manifest = Manifest::new();

    // Encoder.
    let mut channels = variant.encoder_dim();
    add_wn_conv1d(&mut manifest, "encoder.block.0", 1, channels, 7, 1, true);
    for (stage, &stride) in variant.encoder_rates().iter().enumerate() {
        let prefix = format!("encoder.block.{}.block", stage + 1);
        for residual in 0..3 {
            add_residual(
                &mut manifest,
                &format!("{prefix}.{residual}"),
                channels,
                true,
            );
        }
        add_snake(&mut manifest, &format!("{prefix}.3.alpha"), channels);
        add_wn_conv1d(
            &mut manifest,
            &format!("{prefix}.4"),
            channels,
            channels * 2,
            2 * stride,
            1,
            true,
        );
        channels *= 2;
    }
    let encoder_final = if variant.has_attention() {
        add_attention(&mut manifest, "encoder.block.5", channels);
        6
    } else {
        5
    };
    add_wn_conv1d(
        &mut manifest,
        &format!("encoder.block.{encoder_final}"),
        channels,
        channels,
        7,
        channels,
        true,
    );

    // Decoder.
    let latent = variant.latent_dim();
    let decoder_dim = variant.decoder_dim();
    add_wn_conv1d(
        &mut manifest,
        "decoder.model.0",
        latent,
        latent,
        7,
        latent,
        true,
    );
    add_wn_conv1d(
        &mut manifest,
        "decoder.model.1",
        latent,
        decoder_dim,
        1,
        1,
        true,
    );
    let decoder_start = if variant.has_attention() {
        add_attention(&mut manifest, "decoder.model.2", decoder_dim);
        3
    } else {
        2
    };
    for (stage, &stride) in variant.decoder_rates().iter().enumerate() {
        let in_channels = decoder_dim >> stage;
        let out_channels = in_channels / 2;
        let prefix = format!("decoder.model.{}.block", decoder_start + stage);
        add_snake(&mut manifest, &format!("{prefix}.0.alpha"), in_channels);
        add_wn_conv_transpose1d(
            &mut manifest,
            &format!("{prefix}.1"),
            in_channels,
            out_channels,
            2 * stride,
        );
        add_wn_conv1d(
            &mut manifest,
            &format!("{prefix}.2.linear"),
            out_channels,
            out_channels,
            1,
            1,
            false,
        );
        for residual in 0..3 {
            add_residual(
                &mut manifest,
                &format!("{prefix}.{}", residual + 3),
                out_channels,
                true,
            );
        }
    }
    let post_index = decoder_start + 4;
    add_snake(
        &mut manifest,
        &format!("decoder.model.{post_index}.alpha"),
        decoder_dim / 16,
    );
    add_wn_conv1d(
        &mut manifest,
        &format!("decoder.model.{}", post_index + 1),
        decoder_dim / 16,
        1,
        7,
        1,
        true,
    );

    // Factorized hierarchical RVQ.
    for stage in 0..SnacConfig::for_variant(variant).n_stages {
        let prefix = format!("quantizer.quantizers.{stage}");
        add(
            &mut manifest,
            format!("{prefix}.codebook.weight"),
            &[CODEBOOK_SIZE, CODEBOOK_DIM],
        );
        add_wn_conv1d(
            &mut manifest,
            &format!("{prefix}.in_proj"),
            latent,
            CODEBOOK_DIM,
            1,
            1,
            true,
        );
        add_wn_conv1d(
            &mut manifest,
            &format!("{prefix}.out_proj"),
            CODEBOOK_DIM,
            latent,
            1,
            1,
            true,
        );
    }
    manifest
}

fn require_string(file: &GgufFile, key: &str, expected: &str) -> Result<()> {
    let actual = file.get(key).and_then(|value| value.as_str());
    if actual != Some(expected) {
        return Err(VokraError::ModelLoad(format!(
            "snac: metadata `{key}` = {actual:?}, expected {expected:?}"
        )));
    }
    Ok(())
}

fn parse_variant(file: &GgufFile) -> Result<SnacVariant> {
    let tag = file
        .get(KEY_SNAC_VARIANT)
        .and_then(|value| value.as_str())
        .ok_or_else(|| {
            VokraError::ModelLoad(format!(
                "snac: required metadata `{KEY_SNAC_VARIANT}` is missing; expected \
                 `{VARIANT_TAG_HZ24}` or `{VARIANT_TAG_HZ44}`"
            ))
        })?;
    SnacVariant::from_tag(tag).ok_or_else(|| {
        VokraError::ModelLoad(format!(
            "snac: unsupported `{KEY_SNAC_VARIANT}` value `{tag}`; expected \
             `{VARIANT_TAG_HZ24}` or `{VARIANT_TAG_HZ44}`"
        ))
    })
}

fn validate_manifest(file: &GgufFile, variant: SnacVariant) -> Result<()> {
    let expected = expected_manifest(variant);
    debug_assert_eq!(expected.len(), variant.tensor_count());
    let actual_names: BTreeSet<String> = file
        .tensors()
        .iter()
        .map(|tensor| tensor.name.clone())
        .collect();
    let expected_names: BTreeSet<String> = expected.keys().cloned().collect();
    if actual_names != expected_names {
        let missing: Vec<&String> = expected_names.difference(&actual_names).take(8).collect();
        let extra: Vec<&String> = actual_names.difference(&expected_names).take(8).collect();
        return Err(VokraError::ModelLoad(format!(
            "snac: tensor manifest mismatch for {variant:?}: expected {}, found {}; \
             missing={missing:?}, extra={extra:?}",
            expected_names.len(),
            actual_names.len()
        )));
    }
    for (name, shape) in expected {
        let info = file
            .tensor_info(&name)
            .expect("tensor name set was checked");
        let actual_shape: Vec<usize> = info
            .dimensions
            .iter()
            .map(|&dimension| dimension as usize)
            .collect();
        if info.dtype != GgmlType::F32 || actual_shape != shape {
            return Err(VokraError::ModelLoad(format!(
                "snac: tensor `{name}` is {:?} {actual_shape:?}, expected F32 {shape:?}",
                info.dtype
            )));
        }
    }
    Ok(())
}

fn tensor(file: &GgufFile, name: &str) -> Result<Vec<f32>> {
    file.tensor_f32(name).map_err(|error| {
        VokraError::ModelLoad(format!("snac: tensor `{name}` decode failed: {error}"))
    })
}

fn fold_weight_norm(v: &[f32], g: &[f32], rows: usize, row_width: usize) -> Result<Vec<f32>> {
    if v.len() != rows * row_width || g.len() != rows {
        return Err(VokraError::ModelLoad(format!(
            "snac: weight-norm operands have lengths v={} g={}, expected {} and {rows}",
            v.len(),
            g.len(),
            rows * row_width
        )));
    }
    let mut weight = vec![0.0; v.len()];
    for row in 0..rows {
        let source = &v[row * row_width..(row + 1) * row_width];
        let norm = source.iter().map(|value| value * value).sum::<f32>().sqrt();
        if !norm.is_finite() || norm == 0.0 || !g[row].is_finite() {
            return Err(VokraError::ModelLoad(format!(
                "snac: invalid weight-norm row {row}: norm={norm}, g={}",
                g[row]
            )));
        }
        let scale = g[row] / norm;
        for (destination, &source) in weight[row * row_width..(row + 1) * row_width]
            .iter_mut()
            .zip(source)
        {
            *destination = source * scale;
        }
    }
    Ok(weight)
}

fn load_folded_weight(
    file: &GgufFile,
    prefix: &str,
    rows: usize,
    row_width: usize,
) -> Result<Vec<f32>> {
    let g = tensor(file, &format!("{prefix}.parametrizations.weight.original0"))?;
    let v = tensor(file, &format!("{prefix}.parametrizations.weight.original1"))?;
    fold_weight_norm(&v, &g, rows, row_width)
}

#[derive(Debug, Clone)]
struct Conv1d {
    weight: Vec<f32>,
    bias: Option<Vec<f32>>,
    in_channels: usize,
    out_channels: usize,
    kernel: usize,
    stride: usize,
    dilation: usize,
    padding: usize,
    groups: usize,
}

impl Conv1d {
    #[allow(clippy::too_many_arguments)]
    fn load(
        file: &GgufFile,
        prefix: &str,
        in_channels: usize,
        out_channels: usize,
        kernel: usize,
        stride: usize,
        dilation: usize,
        padding: usize,
        groups: usize,
        bias: bool,
    ) -> Result<Self> {
        if groups == 0
            || !in_channels.is_multiple_of(groups)
            || !out_channels.is_multiple_of(groups)
        {
            return Err(VokraError::ModelLoad(format!(
                "snac: invalid grouped convolution `{prefix}`: in={in_channels}, \
                 out={out_channels}, groups={groups}"
            )));
        }
        let row_width = (in_channels / groups) * kernel;
        Ok(Self {
            weight: load_folded_weight(file, prefix, out_channels, row_width)?,
            bias: if bias {
                Some(tensor(file, &format!("{prefix}.bias"))?)
            } else {
                None
            },
            in_channels,
            out_channels,
            kernel,
            stride,
            dilation,
            padding,
            groups,
        })
    }

    fn from_folded_1x1(
        weight: Vec<f32>,
        bias: Vec<f32>,
        in_channels: usize,
        out_channels: usize,
    ) -> Self {
        Self {
            weight,
            bias: Some(bias),
            in_channels,
            out_channels,
            kernel: 1,
            stride: 1,
            dilation: 1,
            padding: 0,
            groups: 1,
        }
    }

    fn output_time(&self, input_time: usize) -> Result<usize> {
        if input_time == 0 || self.stride == 0 || self.dilation == 0 {
            return Err(VokraError::InvalidArgument(
                "snac Conv1D requires positive input_time/stride/dilation".to_owned(),
            ));
        }
        let effective = (self.kernel - 1)
            .checked_mul(self.dilation)
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| {
                VokraError::InvalidArgument("snac Conv1D effective kernel overflow".to_owned())
            })?;
        let padded = input_time
            .checked_add(2 * self.padding)
            .ok_or_else(|| VokraError::InvalidArgument("snac Conv1D extent overflow".to_owned()))?;
        if padded < effective {
            return Err(VokraError::InvalidArgument(format!(
                "snac Conv1D padded input {padded} is shorter than effective kernel {effective}"
            )));
        }
        Ok((padded - effective) / self.stride + 1)
    }

    fn expanded_weight(&self) -> Result<(Vec<f32>, usize)> {
        if self.dilation == 1 {
            return Ok((self.weight.clone(), self.kernel));
        }
        let effective = (self.kernel - 1)
            .checked_mul(self.dilation)
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| {
                VokraError::InvalidArgument("snac Conv1D effective kernel overflow".to_owned())
            })?;
        let row_width = self.in_channels / self.groups;
        let mut expanded = vec![0.0; self.out_channels * row_width * effective];
        for output in 0..self.out_channels {
            for input in 0..row_width {
                let source = (output * row_width + input) * self.kernel;
                let destination = (output * row_width + input) * effective;
                for tap in 0..self.kernel {
                    expanded[destination + tap * self.dilation] = self.weight[source + tap];
                }
            }
        }
        Ok((expanded, effective))
    }

    fn forward(&self, compute: &Compute, input: &[f32], time: usize) -> Result<(Vec<f32>, usize)> {
        let expected = self.in_channels.checked_mul(time).ok_or_else(|| {
            VokraError::InvalidArgument("snac Conv1D input length overflow".to_owned())
        })?;
        if input.len() != expected {
            return Err(VokraError::InvalidArgument(format!(
                "snac Conv1D input length {} != channels*time {expected}",
                input.len()
            )));
        }
        let output_time = self.output_time(time)?;
        if self.groups == 1 {
            let ops = HifiGanComputeOps { compute };
            let output = ops.conv1d(
                input,
                self.in_channels,
                time,
                &self.weight,
                self.out_channels,
                self.kernel,
                self.bias.as_deref(),
                self.stride,
                self.dilation,
                self.padding,
                HifiGanConvPadding::Zero,
            )?;
            return Ok((output, output_time));
        }
        let (weight, effective_kernel) = self.expanded_weight()?;
        let mut output = vec![0.0; self.out_channels * output_time];
        compute.grouped_conv1d_f32(
            input,
            self.in_channels,
            time,
            &weight,
            self.out_channels,
            effective_kernel,
            self.bias.as_deref(),
            self.stride,
            self.padding,
            self.groups,
            &mut output,
        )?;
        Ok((output, output_time))
    }
}

#[derive(Debug, Clone)]
struct ConvTranspose1d {
    weight: Vec<f32>,
    bias: Vec<f32>,
    in_channels: usize,
    out_channels: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    output_padding: usize,
}

impl ConvTranspose1d {
    fn load(
        file: &GgufFile,
        prefix: &str,
        in_channels: usize,
        out_channels: usize,
        stride: usize,
    ) -> Result<Self> {
        let kernel = 2 * stride;
        Ok(Self {
            weight: load_folded_weight(file, prefix, in_channels, out_channels * kernel)?,
            bias: tensor(file, &format!("{prefix}.bias"))?,
            in_channels,
            out_channels,
            kernel,
            stride,
            padding: stride.div_ceil(2),
            output_padding: stride % 2,
        })
    }

    fn forward(&self, compute: &Compute, input: &[f32], time: usize) -> Result<(Vec<f32>, usize)> {
        let ops = HifiGanComputeOps { compute };
        let output = ops.conv_transpose1d_with_output_padding(
            input,
            self.in_channels,
            time,
            &self.weight,
            self.out_channels,
            self.kernel,
            Some(&self.bias),
            self.stride,
            self.padding,
            self.output_padding,
        )?;
        if !output.len().is_multiple_of(self.out_channels) {
            return Err(VokraError::InvalidArgument(
                "snac ConvTranspose1D output is not channel-aligned".to_owned(),
            ));
        }
        let output_time = output.len() / self.out_channels;
        Ok((output, output_time))
    }
}

#[derive(Debug, Clone)]
struct Snake {
    alpha: Vec<f32>,
    channels: usize,
}

impl Snake {
    fn load(file: &GgufFile, name: &str, channels: usize) -> Result<Self> {
        Ok(Self {
            alpha: tensor(file, name)?,
            channels,
        })
    }

    fn forward(&self, compute: &Compute, input: &[f32], time: usize) -> Result<Vec<f32>> {
        let mut output = vec![0.0; input.len()];
        compute.snake_activation_f32(input, &self.alpha, self.channels, time, &mut output)?;
        Ok(output)
    }
}

#[derive(Debug, Clone)]
struct ResidualUnit {
    first_snake: Snake,
    first_conv: Conv1d,
    second_snake: Snake,
    second_conv: Conv1d,
}

impl ResidualUnit {
    fn load(file: &GgufFile, prefix: &str, channels: usize, dilation: usize) -> Result<Self> {
        Ok(Self {
            first_snake: Snake::load(file, &format!("{prefix}.block.0.alpha"), channels)?,
            first_conv: Conv1d::load(
                file,
                &format!("{prefix}.block.1"),
                channels,
                channels,
                7,
                1,
                dilation,
                3 * dilation,
                channels,
                true,
            )?,
            second_snake: Snake::load(file, &format!("{prefix}.block.2.alpha"), channels)?,
            second_conv: Conv1d::load(
                file,
                &format!("{prefix}.block.3"),
                channels,
                channels,
                1,
                1,
                1,
                0,
                1,
                true,
            )?,
        })
    }

    fn forward(&self, compute: &Compute, input: Vec<f32>, time: usize) -> Result<Vec<f32>> {
        let hidden = self.first_snake.forward(compute, &input, time)?;
        let (hidden, first_time) = self.first_conv.forward(compute, &hidden, time)?;
        let hidden = self.second_snake.forward(compute, &hidden, first_time)?;
        let (mut hidden, second_time) = self.second_conv.forward(compute, &hidden, first_time)?;
        if second_time != time || hidden.len() != input.len() {
            return Err(VokraError::InvalidArgument(format!(
                "snac residual branch shape [{}, {second_time}] != skip [{}, {time}]",
                self.second_conv.out_channels, self.second_conv.in_channels
            )));
        }
        for (destination, skip) in hidden.iter_mut().zip(input) {
            *destination += skip;
        }
        Ok(hidden)
    }
}

fn transpose_matrix(input: &[f32], rows: usize, cols: usize) -> Result<Vec<f32>> {
    if input.len() != rows * cols {
        return Err(VokraError::ModelLoad(format!(
            "snac matrix transpose input length {} != {rows}*{cols}",
            input.len()
        )));
    }
    let mut output = vec![0.0; input.len()];
    for row in 0..rows {
        for col in 0..cols {
            output[col * rows + row] = input[row * cols + col];
        }
    }
    Ok(output)
}

#[derive(Debug, Clone)]
struct LocalAttention {
    channels: usize,
    heads: usize,
    norm_weight: Vec<f32>,
    norm_bias: Vec<f32>,
    inv_freq: Vec<f32>,
    qkv_transposed: Vec<f32>,
    out_transposed: Vec<f32>,
}

impl LocalAttention {
    fn load(file: &GgufFile, prefix: &str, channels: usize) -> Result<Self> {
        if !channels.is_multiple_of(ATTENTION_HEAD_DIM) {
            return Err(VokraError::ModelLoad(format!(
                "snac attention channels {channels} are not divisible by head dim \
                 {ATTENTION_HEAD_DIM}"
            )));
        }
        let qkv = tensor(file, &format!("{prefix}.to_qkv.weight"))?;
        let out = tensor(file, &format!("{prefix}.to_out.weight"))?;
        Ok(Self {
            channels,
            heads: channels / ATTENTION_HEAD_DIM,
            norm_weight: tensor(file, &format!("{prefix}.norm.weight"))?,
            norm_bias: tensor(file, &format!("{prefix}.norm.bias"))?,
            inv_freq: tensor(file, &format!("{prefix}.rel_pos.inv_freq"))?,
            qkv_transposed: transpose_matrix(&qkv, 3 * channels, channels)?,
            out_transposed: transpose_matrix(&out, channels, channels)?,
        })
    }

    fn forward(&self, compute: &Compute, input: &[f32], time: usize) -> Result<Vec<f32>> {
        if time == 0 || !time.is_multiple_of(ATTENTION_WINDOW) {
            return Err(VokraError::InvalidArgument(format!(
                "snac LocalMHA time {time} must be a positive multiple of window \
                 {ATTENTION_WINDOW}"
            )));
        }
        if input.len() != self.channels * time {
            return Err(VokraError::InvalidArgument(format!(
                "snac LocalMHA input length {} != channels*time {}",
                input.len(),
                self.channels * time
            )));
        }
        let mut time_major = vec![0.0; input.len()];
        for channel in 0..self.channels {
            for t in 0..time {
                time_major[t * self.channels + channel] = input[channel * time + t];
            }
        }
        let mut normalized = vec![0.0; time_major.len()];
        compute.layer_norm_f32(
            &time_major,
            &mut normalized,
            time,
            self.channels,
            &self.norm_weight,
            &self.norm_bias,
            1e-5,
        )?;
        let mut qkv = vec![0.0; time * 3 * self.channels];
        compute.gemm_f32(
            time,
            3 * self.channels,
            self.channels,
            &normalized,
            &self.qkv_transposed,
            None,
            &mut qkv,
        )?;
        let mut q = vec![0.0; time * self.channels];
        let mut k = vec![0.0; time * self.channels];
        let mut v = vec![0.0; time * self.channels];
        for t in 0..time {
            let source = t * 3 * self.channels;
            q[t * self.channels..(t + 1) * self.channels]
                .copy_from_slice(&qkv[source..source + self.channels]);
            k[t * self.channels..(t + 1) * self.channels]
                .copy_from_slice(&qkv[source + self.channels..source + 2 * self.channels]);
            v[t * self.channels..(t + 1) * self.channels]
                .copy_from_slice(&qkv[source + 2 * self.channels..source + 3 * self.channels]);
        }

        // Upstream SinusoidalEmbeddings sees the per-window sequence axis,
        // so positions restart at zero for each 32-frame window.
        for t in 0..time {
            let position = (t % ATTENTION_WINDOW) as f32;
            for head in 0..self.heads {
                let base = t * self.channels + head * ATTENTION_HEAD_DIM;
                let mut old_q = [0.0_f32; ATTENTION_HEAD_DIM];
                let mut old_k = [0.0_f32; ATTENTION_HEAD_DIM];
                old_q.copy_from_slice(&q[base..base + ATTENTION_HEAD_DIM]);
                old_k.copy_from_slice(&k[base..base + ATTENTION_HEAD_DIM]);
                for d in 0..ATTENTION_HEAD_DIM {
                    let half = d % (ATTENTION_HEAD_DIM / 2);
                    let angle = position * self.inv_freq[half];
                    let (sin, cos) = angle.sin_cos();
                    let rotated_q = if d < ATTENTION_HEAD_DIM / 2 {
                        -old_q[d + ATTENTION_HEAD_DIM / 2]
                    } else {
                        old_q[d - ATTENTION_HEAD_DIM / 2]
                    };
                    let rotated_k = if d < ATTENTION_HEAD_DIM / 2 {
                        -old_k[d + ATTENTION_HEAD_DIM / 2]
                    } else {
                        old_k[d - ATTENTION_HEAD_DIM / 2]
                    };
                    q[base + d] = old_q[d] * cos + rotated_q * sin;
                    k[base + d] = old_k[d] * cos + rotated_k * sin;
                }
            }
        }

        let mut merged = vec![0.0; time * self.channels];
        let scale = 1.0 / (ATTENTION_HEAD_DIM as f32).sqrt();
        for window_start in (0..time).step_by(ATTENTION_WINDOW) {
            for head in 0..self.heads {
                let mut q_matrix = vec![0.0; ATTENTION_WINDOW * ATTENTION_HEAD_DIM];
                let mut k_transposed = vec![0.0; ATTENTION_HEAD_DIM * ATTENTION_WINDOW];
                let mut v_matrix = vec![0.0; ATTENTION_WINDOW * ATTENTION_HEAD_DIM];
                for row in 0..ATTENTION_WINDOW {
                    let source = (window_start + row) * self.channels + head * ATTENTION_HEAD_DIM;
                    q_matrix[row * ATTENTION_HEAD_DIM..(row + 1) * ATTENTION_HEAD_DIM]
                        .copy_from_slice(&q[source..source + ATTENTION_HEAD_DIM]);
                    v_matrix[row * ATTENTION_HEAD_DIM..(row + 1) * ATTENTION_HEAD_DIM]
                        .copy_from_slice(&v[source..source + ATTENTION_HEAD_DIM]);
                    for d in 0..ATTENTION_HEAD_DIM {
                        k_transposed[d * ATTENTION_WINDOW + row] = k[source + d];
                    }
                }
                let mut scores = vec![0.0; ATTENTION_WINDOW * ATTENTION_WINDOW];
                compute.gemm_f32(
                    ATTENTION_WINDOW,
                    ATTENTION_WINDOW,
                    ATTENTION_HEAD_DIM,
                    &q_matrix,
                    &k_transposed,
                    None,
                    &mut scores,
                )?;
                for score in &mut scores {
                    *score *= scale;
                }
                let mut probabilities = vec![0.0; scores.len()];
                compute.softmax_f32(
                    &scores,
                    &mut probabilities,
                    ATTENTION_WINDOW,
                    ATTENTION_WINDOW,
                )?;
                let mut head_output = vec![0.0; ATTENTION_WINDOW * ATTENTION_HEAD_DIM];
                compute.gemm_f32(
                    ATTENTION_WINDOW,
                    ATTENTION_HEAD_DIM,
                    ATTENTION_WINDOW,
                    &probabilities,
                    &v_matrix,
                    None,
                    &mut head_output,
                )?;
                for row in 0..ATTENTION_WINDOW {
                    let destination =
                        (window_start + row) * self.channels + head * ATTENTION_HEAD_DIM;
                    merged[destination..destination + ATTENTION_HEAD_DIM].copy_from_slice(
                        &head_output[row * ATTENTION_HEAD_DIM..(row + 1) * ATTENTION_HEAD_DIM],
                    );
                }
            }
        }
        let mut projected = vec![0.0; merged.len()];
        compute.gemm_f32(
            time,
            self.channels,
            self.channels,
            &merged,
            &self.out_transposed,
            None,
            &mut projected,
        )?;
        let mut output = vec![0.0; input.len()];
        for channel in 0..self.channels {
            for t in 0..time {
                output[channel * time + t] =
                    projected[t * self.channels + channel] + input[channel * time + t];
            }
        }
        Ok(output)
    }
}

#[derive(Debug, Clone)]
struct EncoderBlock {
    residuals: [ResidualUnit; 3],
    snake: Snake,
    downsample: Conv1d,
}

impl EncoderBlock {
    fn load(file: &GgufFile, prefix: &str, input_channels: usize, stride: usize) -> Result<Self> {
        Ok(Self {
            residuals: [
                ResidualUnit::load(file, &format!("{prefix}.0"), input_channels, 1)?,
                ResidualUnit::load(file, &format!("{prefix}.1"), input_channels, 3)?,
                ResidualUnit::load(file, &format!("{prefix}.2"), input_channels, 9)?,
            ],
            snake: Snake::load(file, &format!("{prefix}.3.alpha"), input_channels)?,
            downsample: Conv1d::load(
                file,
                &format!("{prefix}.4"),
                input_channels,
                input_channels * 2,
                2 * stride,
                stride,
                1,
                stride.div_ceil(2),
                1,
                true,
            )?,
        })
    }

    fn forward(
        &self,
        compute: &Compute,
        mut input: Vec<f32>,
        time: usize,
    ) -> Result<(Vec<f32>, usize)> {
        for residual in &self.residuals {
            input = residual.forward(compute, input, time)?;
        }
        let input = self.snake.forward(compute, &input, time)?;
        self.downsample.forward(compute, &input, time)
    }
}

#[derive(Debug, Clone)]
struct Encoder {
    pre: Conv1d,
    blocks: Vec<EncoderBlock>,
    attention: Option<LocalAttention>,
    post: Conv1d,
    hop_length: usize,
    latent_dim: usize,
}

impl Encoder {
    fn load(file: &GgufFile, variant: SnacVariant) -> Result<Self> {
        let encoder_dim = variant.encoder_dim();
        let mut channels = encoder_dim;
        let pre = Conv1d::load(file, "encoder.block.0", 1, channels, 7, 1, 1, 3, 1, true)?;
        let mut blocks = Vec::with_capacity(4);
        for (stage, &stride) in variant.encoder_rates().iter().enumerate() {
            blocks.push(EncoderBlock::load(
                file,
                &format!("encoder.block.{}.block", stage + 1),
                channels,
                stride,
            )?);
            channels *= 2;
        }
        let (attention, post_index) = if variant.has_attention() {
            (
                Some(LocalAttention::load(file, "encoder.block.5", channels)?),
                6,
            )
        } else {
            (None, 5)
        };
        let post = Conv1d::load(
            file,
            &format!("encoder.block.{post_index}"),
            channels,
            channels,
            7,
            1,
            1,
            3,
            channels,
            true,
        )?;
        Ok(Self {
            pre,
            blocks,
            attention,
            post,
            hop_length: variant.encoder_rates().iter().product(),
            latent_dim: channels,
        })
    }

    fn forward(&self, compute: &Compute, pcm: &[f32]) -> Result<(Vec<f32>, usize)> {
        let (mut hidden, mut time) = self.pre.forward(compute, pcm, pcm.len())?;
        for block in &self.blocks {
            (hidden, time) = block.forward(compute, hidden, time)?;
        }
        if let Some(attention) = &self.attention {
            hidden = attention.forward(compute, &hidden, time)?;
        }
        let (hidden, output_time) = self.post.forward(compute, &hidden, time)?;
        if output_time != time || hidden.len() != self.latent_dim * time {
            return Err(VokraError::InvalidArgument(
                "snac encoder final depthwise convolution changed the latent extent".to_owned(),
            ));
        }
        Ok((hidden, time))
    }
}

#[derive(Debug, Clone)]
struct NoiseBlock {
    linear: Conv1d,
}

impl NoiseBlock {
    fn load(file: &GgufFile, prefix: &str, channels: usize) -> Result<Self> {
        Ok(Self {
            linear: Conv1d::load(file, prefix, channels, channels, 1, 1, 1, 0, 1, false)?,
        })
    }

    fn forward(
        &self,
        compute: &Compute,
        input: Vec<f32>,
        time: usize,
        noise: &[f32],
    ) -> Result<Vec<f32>> {
        let (projected, output_time) = self.linear.forward(compute, &input, time)?;
        if output_time != time || projected.len() != input.len() {
            return Err(VokraError::InvalidArgument(
                "snac noise projection changed the decoder extent".to_owned(),
            ));
        }
        if noise.len() != time || noise.iter().any(|value| !value.is_finite()) {
            return Err(VokraError::InvalidArgument(format!(
                "snac noise length {} must equal finite decoder extent {time}",
                noise.len()
            )));
        }
        let mut output = input;
        for channel in 0..self.linear.out_channels {
            for (t, &noise_value) in noise.iter().enumerate() {
                let index = channel * time + t;
                output[index] += noise_value * projected[index];
            }
        }
        Ok(output)
    }
}

#[derive(Debug, Clone)]
struct DecoderBlock {
    snake: Snake,
    upsample: ConvTranspose1d,
    noise: NoiseBlock,
    residuals: [ResidualUnit; 3],
}

impl DecoderBlock {
    fn load(
        file: &GgufFile,
        prefix: &str,
        input_channels: usize,
        output_channels: usize,
        stride: usize,
    ) -> Result<Self> {
        Ok(Self {
            snake: Snake::load(file, &format!("{prefix}.0.alpha"), input_channels)?,
            upsample: ConvTranspose1d::load(
                file,
                &format!("{prefix}.1"),
                input_channels,
                output_channels,
                stride,
            )?,
            noise: NoiseBlock::load(file, &format!("{prefix}.2.linear"), output_channels)?,
            residuals: [
                ResidualUnit::load(file, &format!("{prefix}.3"), output_channels, 1)?,
                ResidualUnit::load(file, &format!("{prefix}.4"), output_channels, 3)?,
                ResidualUnit::load(file, &format!("{prefix}.5"), output_channels, 9)?,
            ],
        })
    }

    fn forward<F>(
        &self,
        compute: &Compute,
        input: Vec<f32>,
        time: usize,
        next_noise: &mut F,
    ) -> Result<(Vec<f32>, usize)>
    where
        F: FnMut(usize) -> Result<Vec<f32>>,
    {
        let hidden = self.snake.forward(compute, &input, time)?;
        let (hidden, output_time) = self.upsample.forward(compute, &hidden, time)?;
        let noise = next_noise(output_time)?;
        let mut hidden = self.noise.forward(compute, hidden, output_time, &noise)?;
        for residual in &self.residuals {
            hidden = residual.forward(compute, hidden, output_time)?;
        }
        Ok((hidden, output_time))
    }
}

#[derive(Debug, Clone)]
struct Decoder {
    depthwise_pre: Conv1d,
    pointwise_pre: Conv1d,
    attention: Option<LocalAttention>,
    blocks: Vec<DecoderBlock>,
    post_snake: Snake,
    post: Conv1d,
}

impl Decoder {
    fn load(file: &GgufFile, variant: SnacVariant) -> Result<Self> {
        let latent = variant.latent_dim();
        let channels = variant.decoder_dim();
        let depthwise_pre = Conv1d::load(
            file,
            "decoder.model.0",
            latent,
            latent,
            7,
            1,
            1,
            3,
            latent,
            true,
        )?;
        let pointwise_pre = Conv1d::load(
            file,
            "decoder.model.1",
            latent,
            channels,
            1,
            1,
            1,
            0,
            1,
            true,
        )?;
        let (attention, block_start) = if variant.has_attention() {
            (
                Some(LocalAttention::load(file, "decoder.model.2", channels)?),
                3,
            )
        } else {
            (None, 2)
        };
        let mut blocks = Vec::with_capacity(4);
        for (stage, &stride) in variant.decoder_rates().iter().enumerate() {
            let input_channels = channels >> stage;
            let output_channels = input_channels / 2;
            blocks.push(DecoderBlock::load(
                file,
                &format!("decoder.model.{}.block", block_start + stage),
                input_channels,
                output_channels,
                stride,
            )?);
        }
        let post_index = block_start + 4;
        Ok(Self {
            depthwise_pre,
            pointwise_pre,
            attention,
            blocks,
            post_snake: Snake::load(
                file,
                &format!("decoder.model.{post_index}.alpha"),
                channels / 16,
            )?,
            post: Conv1d::load(
                file,
                &format!("decoder.model.{}", post_index + 1),
                channels / 16,
                1,
                7,
                1,
                1,
                3,
                1,
                true,
            )?,
        })
    }

    fn forward_with_noise<F>(
        &self,
        compute: &Compute,
        features: &[f32],
        latent_dim: usize,
        mut next_noise: F,
    ) -> Result<Vec<f32>>
    where
        F: FnMut(usize) -> Result<Vec<f32>>,
    {
        if features.is_empty() || !features.len().is_multiple_of(latent_dim) {
            return Err(VokraError::InvalidArgument(format!(
                "snac decoder feature length {} must be a positive multiple of latent dim \
                 {latent_dim}",
                features.len()
            )));
        }
        let mut time = features.len() / latent_dim;
        let (mut hidden, depthwise_time) = self.depthwise_pre.forward(compute, features, time)?;
        if depthwise_time != time {
            return Err(VokraError::InvalidArgument(
                "snac decoder pre-convolution changed time".to_owned(),
            ));
        }
        (hidden, time) = self.pointwise_pre.forward(compute, &hidden, time)?;
        if let Some(attention) = &self.attention {
            hidden = attention.forward(compute, &hidden, time)?;
        }
        for block in &self.blocks {
            (hidden, time) = block.forward(compute, hidden, time, &mut next_noise)?;
        }
        hidden = self.post_snake.forward(compute, &hidden, time)?;
        let (mut pcm, output_time) = self.post.forward(compute, &hidden, time)?;
        if output_time != time || pcm.len() != time {
            return Err(VokraError::InvalidArgument(
                "snac decoder terminal convolution changed waveform extent".to_owned(),
            ));
        }
        for sample in &mut pcm {
            *sample = sample.tanh();
        }
        Ok(pcm)
    }

    fn forward(
        &self,
        compute: &Compute,
        features: &[f32],
        latent_dim: usize,
        seed: u64,
    ) -> Result<Vec<f32>> {
        let mut rng = GaussianSplitMix64::new(seed);
        self.forward_with_noise(compute, features, latent_dim, |time| {
            Ok((0..time).map(|_| rng.next_gaussian()).collect())
        })
    }
}

#[derive(Debug, Clone)]
struct Quantizer {
    in_projs: Vec<Conv1d>,
    codebooks: Vec<CodebookTable>,
    normalized_codebooks: Vec<Vec<f32>>,
    out_projs: Vec<DacOutProj>,
    strides: Vec<usize>,
    latent_dim: usize,
}

impl Quantizer {
    fn load(file: &GgufFile, variant: SnacVariant) -> Result<Self> {
        let config = SnacConfig::for_variant(variant);
        let latent_dim = variant.latent_dim();
        let mut in_projs = Vec::with_capacity(config.n_stages);
        let mut codebooks = Vec::with_capacity(config.n_stages);
        let mut normalized_codebooks = Vec::with_capacity(config.n_stages);
        let mut out_projs = Vec::with_capacity(config.n_stages);
        for stage in 0..config.n_stages {
            let prefix = format!("quantizer.quantizers.{stage}");
            in_projs.push(Conv1d::load(
                file,
                &format!("{prefix}.in_proj"),
                latent_dim,
                CODEBOOK_DIM,
                1,
                1,
                1,
                0,
                1,
                true,
            )?);
            let table = CodebookTable::new(
                CODEBOOK_SIZE,
                CODEBOOK_DIM,
                tensor(file, &format!("{prefix}.codebook.weight"))?,
            )?;
            let mut normalized = table.data.clone();
            for row in normalized.chunks_exact_mut(CODEBOOK_DIM) {
                let norm = row.iter().map(|value| value * value).sum::<f32>().sqrt();
                let denominator = norm.max(1e-12);
                for value in row {
                    *value /= denominator;
                }
            }
            normalized_codebooks.push(normalized);
            codebooks.push(table);
            let weight = load_folded_weight(
                file,
                &format!("{prefix}.out_proj"),
                latent_dim,
                CODEBOOK_DIM,
            )?;
            let bias = tensor(file, &format!("{prefix}.out_proj.bias"))?;
            out_projs.push(DacOutProj::new(latent_dim, CODEBOOK_DIM, weight, bias)?);
        }
        Ok(Self {
            in_projs,
            codebooks,
            normalized_codebooks,
            out_projs,
            strides: config
                .active_vq_strides()
                .iter()
                .map(|&stride| stride as usize)
                .collect(),
            latent_dim,
        })
    }

    fn op_config(&self, sample_rate: u32) -> OpSnacConfig {
        let mut vq_strides = [0u32; 4];
        for (destination, &source) in vq_strides.iter_mut().zip(&self.strides) {
            *destination = source as u32;
        }
        OpSnacConfig {
            sample_rate,
            vq_strides,
            n_stages: self.strides.len(),
        }
    }

    fn decode(&self, compute: &Compute, codes: &[Vec<u32>], sample_rate: u32) -> Result<Vec<f32>> {
        compute.snac_decode_f32(
            codes,
            self.op_config(sample_rate),
            &self.codebooks,
            &self.out_projs,
        )
    }

    fn encode(&self, compute: &Compute, features: &[f32], time: usize) -> Result<Vec<Vec<u32>>> {
        if features.len() != self.latent_dim * time {
            return Err(VokraError::InvalidArgument(format!(
                "snac quantizer feature length {} != latent_dim*time {}",
                features.len(),
                self.latent_dim * time
            )));
        }
        let mut residual = features.to_vec();
        let mut codes = Vec::with_capacity(self.strides.len());
        for stage in 0..self.strides.len() {
            let stride = self.strides[stage];
            if !time.is_multiple_of(stride) {
                return Err(VokraError::InvalidArgument(format!(
                    "snac quantizer base time {time} is not divisible by stage {stage} stride \
                     {stride}"
                )));
            }
            let stage_time = time / stride;
            let pooled = if stride == 1 {
                residual.clone()
            } else {
                let mut pooled = vec![0.0; self.latent_dim * stage_time];
                for channel in 0..self.latent_dim {
                    for t in 0..stage_time {
                        let mut sum = 0.0_f32;
                        for offset in 0..stride {
                            sum += residual[channel * time + t * stride + offset];
                        }
                        pooled[channel * stage_time + t] = sum / stride as f32;
                    }
                }
                pooled
            };
            let (projected, projected_time) =
                self.in_projs[stage].forward(compute, &pooled, stage_time)?;
            if projected_time != stage_time {
                return Err(VokraError::InvalidArgument(
                    "snac quantizer in_proj changed time".to_owned(),
                ));
            }
            let mut stage_codes = Vec::with_capacity(stage_time);
            for t in 0..stage_time {
                let mut latent = [0.0_f32; CODEBOOK_DIM];
                let mut norm_sq = 0.0_f32;
                for d in 0..CODEBOOK_DIM {
                    latent[d] = projected[d * stage_time + t];
                    norm_sq += latent[d] * latent[d];
                }
                let denominator = norm_sq.sqrt().max(1e-12);
                for value in &mut latent {
                    *value /= denominator;
                }
                let latent_norm_sq = latent.iter().map(|value| value * value).sum::<f32>();
                let table = &self.normalized_codebooks[stage];
                let mut best_index = 0usize;
                let mut best_distance = f32::INFINITY;
                for index in 0..CODEBOOK_SIZE {
                    let row = &table[index * CODEBOOK_DIM..(index + 1) * CODEBOOK_DIM];
                    let mut dot = 0.0_f32;
                    let mut row_norm_sq = 0.0_f32;
                    for d in 0..CODEBOOK_DIM {
                        dot += latent[d] * row[d];
                        row_norm_sq += row[d] * row[d];
                    }
                    let distance = latent_norm_sq - 2.0 * dot + row_norm_sq;
                    if distance < best_distance {
                        best_distance = distance;
                        best_index = index;
                    }
                }
                stage_codes.push(best_index as u32);
            }
            let mut selected = vec![0.0; CODEBOOK_DIM * stage_time];
            for (t, &index) in stage_codes.iter().enumerate() {
                let row = self.codebooks[stage].row(index)?;
                for d in 0..CODEBOOK_DIM {
                    selected[d * stage_time + t] = row[d];
                }
            }
            let projection = Conv1d::from_folded_1x1(
                self.out_projs[stage].weight.clone(),
                self.out_projs[stage].bias.clone(),
                CODEBOOK_DIM,
                self.latent_dim,
            );
            let (stage_features, output_time) =
                projection.forward(compute, &selected, stage_time)?;
            if output_time != stage_time {
                return Err(VokraError::InvalidArgument(
                    "snac quantizer out_proj changed time".to_owned(),
                ));
            }
            for channel in 0..self.latent_dim {
                for t in 0..time {
                    residual[channel * time + t] -=
                        stage_features[channel * stage_time + t / stride];
                }
            }
            codes.push(stage_codes);
        }
        Ok(codes)
    }
}

/// Complete public SNAC codec. CPU supports encode and decode; Metal supports
/// the complete stochastic token-to-waveform decode. Encode on Metal returns
/// an explicit error until a GPU codebook-search kernel is available.
#[derive(Debug, Clone)]
pub struct Snac {
    config: SnacConfig,
    weight_license: LicenseClass,
    encoder: Encoder,
    quantizer: Quantizer,
    decoder: Decoder,
    backend: BackendKind,
}

impl Snac {
    /// Strictly binds one of the two revisioned public SNAC GGUF contracts.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        require_string(file, chunks::KEY_MODEL_ARCH, ARCH)?;
        let variant = parse_variant(file)?;
        require_string(file, chunks::KEY_MODEL_NAME, variant.model_name())?;
        require_string(file, KEY_CATEGORY, CATEGORY)?;
        require_string(file, chunks::KEY_PROVENANCE_MODEL_ID, variant.model_name())?;
        require_string(file, chunks::KEY_PROVENANCE_LICENSE, "mit")?;
        require_string(
            file,
            chunks::KEY_PROVENANCE_WEIGHT_LICENSE,
            LicenseClass::Permissive.as_str(),
        )?;
        require_string(file, KEY_UPSTREAM_HF, variant.upstream_hf())?;
        if file
            .get(chunks::KEY_PROVENANCE_SOURCE)
            .and_then(|value| value.as_str())
            .filter(|value| !value.trim().is_empty())
            .is_none()
        {
            return Err(VokraError::ModelLoad(format!(
                "snac: `{}` is missing or empty",
                chunks::KEY_PROVENANCE_SOURCE
            )));
        }
        validate_manifest(file, variant)?;
        Ok(Self {
            config: SnacConfig::for_variant(variant),
            weight_license: LicenseClass::Permissive,
            encoder: Encoder::load(file, variant)?,
            quantizer: Quantizer::load(file, variant)?,
            decoder: Decoder::load(file, variant)?,
            backend: BackendKind::Cpu,
        })
    }

    /// Opens and strictly binds a public SNAC GGUF.
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        Self::from_gguf(&GgufFile::open(path)?)
    }

    /// Selects one backend for every learned decode operation.
    #[must_use]
    pub fn with_backend(mut self, backend: BackendKind) -> Self {
        self.backend = backend;
        self
    }

    /// Returns the backend selected for learned decode operations.
    #[must_use]
    pub const fn backend(&self) -> BackendKind {
        self.backend
    }

    /// Returns the validated architecture configuration.
    #[must_use]
    pub const fn config(&self) -> &SnacConfig {
        &self.config
    }

    /// Returns the bound public SNAC variant.
    #[must_use]
    pub const fn variant(&self) -> SnacVariant {
        self.config.variant
    }

    /// Returns the audited weight-license class.
    #[must_use]
    pub const fn weight_license(&self) -> LicenseClass {
        self.weight_license
    }

    /// Returns the waveform sample rate expected by this model.
    #[must_use]
    pub const fn sample_rate(&self) -> u32 {
        self.config.sample_rate
    }

    /// Returns the waveform samples represented by one finest-level code.
    #[must_use]
    pub const fn hop_length(&self) -> usize {
        self.encoder.hop_length
    }

    fn validate_encode_input(&self, pcm: &[f32], sample_rate: u32) -> Result<()> {
        if sample_rate != self.sample_rate() {
            return Err(VokraError::InvalidArgument(format!(
                "snac encode sample rate {sample_rate} != model sample rate {} (no silent \
                 resampling)",
                self.sample_rate()
            )));
        }
        if pcm.is_empty() || pcm.iter().any(|sample| !sample.is_finite()) {
            return Err(VokraError::InvalidArgument(
                "snac encode requires non-empty finite mono PCM".to_owned(),
            ));
        }
        Ok(())
    }

    fn cpu_encoder_features(&self, pcm: &[f32]) -> Result<(Vec<f32>, usize)> {
        let first_stride = self.config.active_vq_strides()[0] as usize;
        let attention_multiple = if self.config.variant.has_attention() {
            ATTENTION_WINDOW
        } else {
            1
        };
        let lcm = lcm(first_stride, attention_multiple);
        let pad_to = self.hop_length().checked_mul(lcm).ok_or_else(|| {
            VokraError::InvalidArgument("snac preprocess pad extent overflow".to_owned())
        })?;
        let padded_len = pcm
            .len()
            .div_ceil(pad_to)
            .checked_mul(pad_to)
            .ok_or_else(|| {
                VokraError::InvalidArgument("snac preprocess padded length overflow".to_owned())
            })?;
        let mut padded = vec![0.0; padded_len];
        padded[..pcm.len()].copy_from_slice(pcm);
        let compute = Compute::for_backend(BackendKind::Cpu, SNAC_HOT_OPS)?;
        self.encoder.forward(&compute, &padded)
    }

    /// CPU waveform-to-hierarchical-code encode. Metal is rejected before
    /// any convolution runs because nearest-codebook search has no Metal op;
    /// there is no silent host search inside a GPU request.
    pub fn encode(&self, pcm: &[f32], sample_rate: u32) -> Result<Vec<Vec<u32>>> {
        self.validate_encode_input(pcm, sample_rate)?;
        if self.backend != BackendKind::Cpu {
            return Err(VokraError::UnsupportedOp(format!(
                "snac encode is CPU-only: backend {:?} has no hierarchical codebook-search \
                 kernel; no silent CPU fallback is performed",
                self.backend
            )));
        }
        let (features, time) = self.cpu_encoder_features(pcm)?;
        let compute = Compute::for_backend(BackendKind::Cpu, SNAC_HOT_OPS)?;
        self.quantizer.encode(&compute, &features, time)
    }

    #[cfg(test)]
    pub(crate) fn encode_features_for_parity(
        &self,
        pcm: &[f32],
        sample_rate: u32,
    ) -> Result<Vec<f32>> {
        self.validate_encode_input(pcm, sample_rate)?;
        self.cpu_encoder_features(pcm).map(|(features, _)| features)
    }

    /// Deterministic convenience decode using seed zero for upstream noise
    /// blocks. Use [`Self::decode_with_seed`] to choose a different draw.
    pub fn decode(&self, codes: &[Vec<u32>]) -> Result<Vec<f32>> {
        self.decode_with_seed(codes, 0)
    }

    /// Complete stochastic hierarchical-code-to-waveform decode.
    pub fn decode_with_seed(&self, codes: &[Vec<u32>], seed: u64) -> Result<Vec<f32>> {
        let compute = Compute::for_backend(self.backend, SNAC_HOT_OPS)?;
        let channel_major = self.decode_features_channel_major(&compute, codes)?;
        self.decoder.forward(
            &compute,
            &channel_major,
            self.config.variant.latent_dim(),
            seed,
        )
    }

    fn decode_features_channel_major(
        &self,
        compute: &Compute,
        codes: &[Vec<u32>],
    ) -> Result<Vec<f32>> {
        let time_major = self.quantizer.decode(compute, codes, self.sample_rate())?;
        if time_major.is_empty() {
            return Err(VokraError::InvalidArgument(
                "snac decode requires at least one co-aligned base frame".to_owned(),
            ));
        }
        let latent = self.config.variant.latent_dim();
        let time = time_major.len() / latent;
        let mut channel_major = vec![0.0; time_major.len()];
        for t in 0..time {
            for channel in 0..latent {
                channel_major[channel * time + t] = time_major[t * latent + channel];
            }
        }
        Ok(channel_major)
    }

    #[cfg(test)]
    pub(crate) fn decode_with_noise_for_parity(
        &self,
        codes: &[Vec<u32>],
        noises: &[Vec<f32>],
    ) -> Result<Vec<f32>> {
        let compute = Compute::for_backend(self.backend, SNAC_HOT_OPS)?;
        let channel_major = self.decode_features_channel_major(&compute, codes)?;
        let mut stage = 0usize;
        let output = self.decoder.forward_with_noise(
            &compute,
            &channel_major,
            self.config.variant.latent_dim(),
            |time| {
                let noise = noises.get(stage).ok_or_else(|| {
                    VokraError::InvalidArgument(format!(
                        "snac parity noise is missing decoder stage {stage} with extent {time}"
                    ))
                })?;
                stage += 1;
                Ok(noise.clone())
            },
        )?;
        if stage != noises.len() {
            return Err(VokraError::InvalidArgument(format!(
                "snac parity noise supplied {} stages, decoder consumed {stage}",
                noises.len()
            )));
        }
        Ok(output)
    }

    /// Runs only the hierarchical RVQ reconstruction and returns time-major
    /// `[base_frames, latent_dim]` features.
    pub fn decode_codes_to_features(&self, codes: &[Vec<u32>]) -> Result<Vec<f32>> {
        let compute = Compute::for_backend(self.backend, &[HotOp::SnacDecode])?;
        self.quantizer.decode(&compute, codes, self.sample_rate())
    }
}

const fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let remainder = a % b;
        a = b;
        b = remainder;
    }
    a
}

const fn lcm(a: usize, b: usize) -> usize {
    a / gcd(a, b) * b
}
