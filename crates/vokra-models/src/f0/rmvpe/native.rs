//! Exact native forward for the public `vokra/rmvpe` tensor contract.
//!
//! The topology is transcribed from `yxlllc/RMVPE` commit
//! `0aabafba18289ca938a73af0b0297686abf4922d` (`src/deepunet.py`,
//! `src/model.py`, `src/seq.py`, and `src/spec.py`).  The public GGUF header
//! at revision `3eb5fa8946f1074ba3959074c5cde95ec22b8c91` contains this exact
//! E2E0 state-dict layout: 623 tensors consumed by inference plus 118 optional
//! `num_batches_tracked` counters.  No tensor-name guessing or CPU fallback is
//! performed here.

use std::collections::BTreeSet;

use vokra_core::VokraError;
use vokra_core::gguf::{GgmlType, GgufFile};

use super::{
    BN_EPS, F0Frame, GGUF_KEY_BASE_HZ, GGUF_KEY_CENTS_PER_CLASS, GGUF_KEY_FMAX, GGUF_KEY_FMIN,
    GGUF_KEY_HOP, GGUF_KEY_N_CLASS, GGUF_KEY_N_FFT, GGUF_KEY_N_MELS, GGUF_KEY_SAMPLE_RATE,
    GGUF_KEY_UPSTREAM_REVISION, GGUF_KEY_WIN_LENGTH, RMVPE, RMVPE_HOT_OPS, RmvpeConfig,
    RmvpeWeights, batchnorm2d_apply, collapse_nchw_to_frames, conv2d_pad_same_with_compute,
    decode_class_to_hz, gru_cell_step_with_compute, linear_frames_with_compute, sigmoid_inplace,
};
use crate::compute::Compute;

pub(super) const UPSTREAM_REVISION: &str = "0aabafba18289ca938a73af0b0297686abf4922d";
pub(super) const PUBLIC_HF_REVISION: &str = "3eb5fa8946f1074ba3959074c5cde95ec22b8c91";

const ENCODER_LAYERS: usize = 5;
const INTERMEDIATE_LAYERS: usize = 4;
const DECODER_LAYERS: usize = 5;
const RESIDUAL_BLOCKS: usize = 4;
const REQUIRED_TENSORS: usize = 623;
const OPTIONAL_COUNTERS: usize = 118;
const GRU_INPUT: usize = 384;
const GRU_HIDDEN: usize = 256;
const N_CLASS: usize = 360;
const VOICED_THRESHOLD: f32 = 0.03;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ContractFlavor {
    /// Corrected converter metadata (`n_fft=1024`, class-0 = 31.7 Hz).
    Canonical,
    /// Historical public GGUF metadata (`n_fft=2048`, class-0 ~=32.703 Hz).
    /// Tensor data is usable, but these two metadata values are normalized to
    /// the fixed upstream source contract before inference.
    PublicLegacyMetadata,
}

fn model_load(message: impl Into<String>) -> VokraError {
    VokraError::ModelLoad(format!("rmvpe: {} (FR-EX-08)", message.into()))
}

fn allowed_float(dtype: GgmlType) -> bool {
    matches!(dtype, GgmlType::F32 | GgmlType::F16 | GgmlType::BF16)
}

fn require_tensor(
    gguf: &GgufFile,
    expected: &mut BTreeSet<String>,
    name: String,
    shape: &[usize],
) -> Result<(), VokraError> {
    let info = gguf.tensor_info(&name).ok_or_else(|| {
        model_load(format!(
            "required public-contract tensor `{name}` is missing"
        ))
    })?;
    let actual: Vec<usize> = info.dimensions.iter().map(|&v| v as usize).collect();
    if actual != shape {
        return Err(model_load(format!(
            "tensor `{name}` shape {actual:?} != fixed upstream shape {shape:?}"
        )));
    }
    if !allowed_float(info.dtype) {
        return Err(model_load(format!(
            "tensor `{name}` uses unsupported dtype {:?}; expected F32/F16/BF16",
            info.dtype
        )));
    }
    expected.insert(name);
    Ok(())
}

fn optional_counter(
    gguf: &GgufFile,
    expected: &mut BTreeSet<String>,
    name: String,
) -> Result<usize, VokraError> {
    expected.insert(name.clone());
    let Some(info) = gguf.tensor_info(&name) else {
        return Ok(0);
    };
    let actual: Vec<usize> = info.dimensions.iter().map(|&v| v as usize).collect();
    if !(actual.is_empty() || actual == [1]) {
        return Err(model_load(format!(
            "batch-norm counter `{name}` shape {actual:?} must be scalar or [1]"
        )));
    }
    if !allowed_float(info.dtype) {
        return Err(model_load(format!(
            "batch-norm counter `{name}` uses unsupported dtype {:?}",
            info.dtype
        )));
    }
    Ok(1)
}

fn require_bn(
    gguf: &GgufFile,
    expected: &mut BTreeSet<String>,
    prefix: &str,
    channels: usize,
) -> Result<usize, VokraError> {
    for suffix in ["weight", "bias", "running_mean", "running_var"] {
        require_tensor(gguf, expected, format!("{prefix}.{suffix}"), &[channels])?;
    }
    optional_counter(gguf, expected, format!("{prefix}.num_batches_tracked"))
}

fn require_conv_block(
    gguf: &GgufFile,
    expected: &mut BTreeSet<String>,
    prefix: &str,
    in_channels: usize,
    out_channels: usize,
) -> Result<usize, VokraError> {
    require_tensor(
        gguf,
        expected,
        format!("{prefix}.conv.0.weight"),
        &[out_channels, in_channels, 3, 3],
    )?;
    let mut counters = require_bn(gguf, expected, &format!("{prefix}.conv.1"), out_channels)?;
    require_tensor(
        gguf,
        expected,
        format!("{prefix}.conv.3.weight"),
        &[out_channels, out_channels, 3, 3],
    )?;
    counters += require_bn(gguf, expected, &format!("{prefix}.conv.4"), out_channels)?;
    if in_channels != out_channels {
        require_tensor(
            gguf,
            expected,
            format!("{prefix}.shortcut.weight"),
            &[out_channels, in_channels, 1, 1],
        )?;
        require_tensor(
            gguf,
            expected,
            format!("{prefix}.shortcut.bias"),
            &[out_channels],
        )?;
    }
    Ok(counters)
}

fn metadata_u64(gguf: &GgufFile, key: &str) -> Result<u64, VokraError> {
    gguf.get(key)
        .and_then(|value| value.as_u64())
        .ok_or_else(|| {
            model_load(format!(
                "required metadata `{key}` is missing or not unsigned"
            ))
        })
}

fn metadata_f64(gguf: &GgufFile, key: &str) -> Result<f64, VokraError> {
    gguf.get(key)
        .and_then(|value| value.as_f64())
        .ok_or_else(|| {
            model_load(format!(
                "required metadata `{key}` is missing or not numeric"
            ))
        })
}

fn require_close(key: &str, actual: f64, expected: f64, tolerance: f64) -> Result<(), VokraError> {
    if (actual - expected).abs() > tolerance {
        return Err(model_load(format!(
            "metadata `{key}`={actual} != fixed upstream value {expected}"
        )));
    }
    Ok(())
}

/// Validates the complete E2E0 topology from GGUF metadata and tensor
/// descriptors. A 623-tensor checkpoint (counters stripped) and the historical
/// 741-tensor public checkpoint are the only accepted manifests.
pub(super) fn validate_contract(gguf: &GgufFile) -> Result<ContractFlavor, VokraError> {
    if gguf
        .get("vokra.model.arch")
        .and_then(|value| value.as_str())
        != Some("rmvpe")
    {
        return Err(model_load("`vokra.model.arch` must be `rmvpe`"));
    }

    for (key, expected) in [
        (GGUF_KEY_HOP, 160),
        (GGUF_KEY_N_MELS, 128),
        (GGUF_KEY_WIN_LENGTH, 1024),
        (GGUF_KEY_SAMPLE_RATE, 16_000),
        (GGUF_KEY_N_CLASS, 360),
    ] {
        let actual = metadata_u64(gguf, key)?;
        if actual != expected {
            return Err(model_load(format!(
                "metadata `{key}`={actual} != fixed upstream value {expected}"
            )));
        }
    }
    require_close(
        GGUF_KEY_FMIN,
        metadata_f64(gguf, GGUF_KEY_FMIN)?,
        30.0,
        1e-6,
    )?;
    require_close(
        GGUF_KEY_FMAX,
        metadata_f64(gguf, GGUF_KEY_FMAX)?,
        1000.0,
        1e-6,
    )?;
    require_close(
        GGUF_KEY_CENTS_PER_CLASS,
        metadata_f64(gguf, GGUF_KEY_CENTS_PER_CLASS)?,
        20.0,
        1e-6,
    )?;

    let n_fft = metadata_u64(gguf, GGUF_KEY_N_FFT)?;
    let base_hz = metadata_f64(gguf, GGUF_KEY_BASE_HZ)?;
    let flavor = if n_fft == 1024 && (base_hz - 31.7).abs() <= 1e-4 {
        ContractFlavor::Canonical
    } else if n_fft == 2048 && (base_hz - 32.703_197).abs() <= 1e-4 {
        ContractFlavor::PublicLegacyMetadata
    } else {
        return Err(model_load(format!(
            "unsupported frontend/grid metadata pair: n_fft={n_fft}, base_hz={base_hz}; \
             expected canonical (1024, 31.7) or the exact historical public pair \
             (2048, 32.703197)"
        )));
    };

    let mut expected = BTreeSet::new();
    let mut counters = require_bn(gguf, &mut expected, "unet.encoder.bn", 1)?;

    let mut in_channels = 1usize;
    let mut out_channels = 16usize;
    for layer in 0..ENCODER_LAYERS {
        for block in 0..RESIDUAL_BLOCKS {
            let block_in = if block == 0 {
                in_channels
            } else {
                out_channels
            };
            counters += require_conv_block(
                gguf,
                &mut expected,
                &format!("unet.encoder.layers.{layer}.conv.{block}"),
                block_in,
                out_channels,
            )?;
        }
        in_channels = out_channels;
        out_channels *= 2;
    }

    in_channels = 256;
    out_channels = 512;
    for layer in 0..INTERMEDIATE_LAYERS {
        for block in 0..RESIDUAL_BLOCKS {
            let block_in = if block == 0 {
                in_channels
            } else {
                out_channels
            };
            counters += require_conv_block(
                gguf,
                &mut expected,
                &format!("unet.intermediate.layers.{layer}.conv.{block}"),
                block_in,
                out_channels,
            )?;
        }
        in_channels = out_channels;
    }

    in_channels = 512;
    for layer in 0..DECODER_LAYERS {
        out_channels = in_channels / 2;
        require_tensor(
            gguf,
            &mut expected,
            format!("unet.decoder.layers.{layer}.conv1.0.weight"),
            &[in_channels, out_channels, 3, 3],
        )?;
        counters += require_bn(
            gguf,
            &mut expected,
            &format!("unet.decoder.layers.{layer}.conv1.1"),
            out_channels,
        )?;
        for block in 0..RESIDUAL_BLOCKS {
            let block_in = if block == 0 {
                out_channels * 2
            } else {
                out_channels
            };
            counters += require_conv_block(
                gguf,
                &mut expected,
                &format!("unet.decoder.layers.{layer}.conv2.{block}"),
                block_in,
                out_channels,
            )?;
        }
        in_channels = out_channels;
    }

    require_tensor(gguf, &mut expected, "cnn.weight".into(), &[3, 16, 3, 3])?;
    require_tensor(gguf, &mut expected, "cnn.bias".into(), &[3])?;
    for suffix in ["l0", "l0_reverse"] {
        require_tensor(
            gguf,
            &mut expected,
            format!("fc.0.gru.weight_ih_{suffix}"),
            &[3 * GRU_HIDDEN, GRU_INPUT],
        )?;
        require_tensor(
            gguf,
            &mut expected,
            format!("fc.0.gru.weight_hh_{suffix}"),
            &[3 * GRU_HIDDEN, GRU_HIDDEN],
        )?;
        require_tensor(
            gguf,
            &mut expected,
            format!("fc.0.gru.bias_ih_{suffix}"),
            &[3 * GRU_HIDDEN],
        )?;
        require_tensor(
            gguf,
            &mut expected,
            format!("fc.0.gru.bias_hh_{suffix}"),
            &[3 * GRU_HIDDEN],
        )?;
    }
    require_tensor(
        gguf,
        &mut expected,
        "fc.1.weight".into(),
        &[N_CLASS, 2 * GRU_HIDDEN],
    )?;
    require_tensor(gguf, &mut expected, "fc.1.bias".into(), &[N_CLASS])?;

    if counters > OPTIONAL_COUNTERS {
        return Err(model_load(format!(
            "manifest contains {counters} batch-norm counters; maximum is {OPTIONAL_COUNTERS}"
        )));
    }
    let actual_count = gguf.tensors().len();
    let expected_count = REQUIRED_TENSORS + counters;
    if actual_count != expected_count {
        let unknown: Vec<&str> = gguf
            .tensors()
            .iter()
            .filter(|tensor| !expected.contains(&tensor.name))
            .take(8)
            .map(|tensor| tensor.name.as_str())
            .collect();
        return Err(model_load(format!(
            "tensor manifest count {actual_count} != {expected_count} \
             ({REQUIRED_TENSORS} inference tensors + {counters} counters); \
             first unsupported names: {unknown:?}"
        )));
    }

    let source = gguf
        .get("vokra.provenance.source")
        .and_then(|value| value.as_str());
    if source != Some("yxlllc/RMVPE") {
        return Err(model_load(format!(
            "provenance source {source:?} is not the fixed `yxlllc/RMVPE` contract at \
             {UPSTREAM_REVISION}"
        )));
    }
    let license = gguf
        .get("vokra.provenance.license")
        .and_then(|value| value.as_str());
    let weight_license = gguf
        .get("vokra.provenance.weight_license")
        .and_then(|value| value.as_str());
    let source_revision = gguf
        .get(GGUF_KEY_UPSTREAM_REVISION)
        .and_then(|value| value.as_str());
    match flavor {
        ContractFlavor::Canonical => {
            if source_revision != Some(UPSTREAM_REVISION) {
                return Err(model_load(format!(
                    "canonical artifact source revision {source_revision:?} != fixed \
                     {UPSTREAM_REVISION}"
                )));
            }
        }
        ContractFlavor::PublicLegacyMetadata => {
            if source_revision.is_some() && source_revision != Some(UPSTREAM_REVISION) {
                return Err(model_load(format!(
                    "historical artifact source revision {source_revision:?} conflicts with \
                     fixed {UPSTREAM_REVISION}"
                )));
            }
            if license != Some("unknown") || weight_license != Some("unknown") {
                return Err(model_load(format!(
                    "the historical public artifact is mis-stamped as license={license:?}, \
                     weight_license={weight_license:?}, but `yxlllc/RMVPE` has no LICENSE; \
                     use a provenance-corrected `unknown` artifact under an explicit policy \
                     (audited public revision {PUBLIC_HF_REVISION})"
                )));
            }
        }
    }
    Ok(flavor)
}

pub(super) fn normalize_config(config: &mut RmvpeConfig, flavor: ContractFlavor) {
    if flavor == ContractFlavor::PublicLegacyMetadata {
        // The public tensor payload is the upstream E2E0 checkpoint, whose
        // `src/spec.py` constructs a 1024-point STFT and whose cents constant
        // maps class zero to exactly 31.7 Hz.  Only the historical GGUF header
        // is wrong; no tensor data is changed.
        config.n_fft = 1024;
        config.base_hz = 31.7;
    }
}

fn tensor<'a>(
    weights: &'a RmvpeWeights,
    name: &str,
    shape: &[usize],
) -> Result<&'a [f32], VokraError> {
    let (actual, payload) = weights
        .tensor(name)
        .ok_or_else(|| model_load(format!("validated tensor `{name}` was not bound")))?;
    if actual != shape {
        return Err(model_load(format!(
            "bound tensor `{name}` shape {actual:?} != {shape:?}"
        )));
    }
    Ok(payload)
}

fn apply_bn(
    weights: &RmvpeWeights,
    prefix: &str,
    x: &mut [f32],
    channels: usize,
    h: usize,
    w: usize,
) -> Result<(), VokraError> {
    let shape = [channels];
    let gamma = tensor(weights, &format!("{prefix}.weight"), &shape)?;
    let beta = tensor(weights, &format!("{prefix}.bias"), &shape)?;
    let mean = tensor(weights, &format!("{prefix}.running_mean"), &shape)?;
    let var = tensor(weights, &format!("{prefix}.running_var"), &shape)?;
    batchnorm2d_apply(x, channels, h, w, gamma, beta, mean, var, BN_EPS);
    Ok(())
}

fn relu_inplace(x: &mut [f32]) {
    for value in x {
        *value = value.max(0.0);
    }
}

// The convolution shape is kept explicit to authenticate every residual block.
#[allow(clippy::too_many_arguments)]
fn conv_block_res(
    compute: &Compute,
    weights: &RmvpeWeights,
    prefix: &str,
    input: &[f32],
    in_channels: usize,
    out_channels: usize,
    h: usize,
    w: usize,
) -> Result<Vec<f32>, VokraError> {
    if input.len() != in_channels * h * w {
        return Err(model_load(format!(
            "`{prefix}` input len {} != {in_channels}*{h}*{w}",
            input.len()
        )));
    }
    let conv1 = tensor(
        weights,
        &format!("{prefix}.conv.0.weight"),
        &[out_channels, in_channels, 3, 3],
    )?;
    let mut branch = conv2d_pad_same_with_compute(
        compute,
        input,
        in_channels,
        h,
        w,
        conv1,
        out_channels,
        3,
        3,
        None,
    )?;
    apply_bn(
        weights,
        &format!("{prefix}.conv.1"),
        &mut branch,
        out_channels,
        h,
        w,
    )?;
    relu_inplace(&mut branch);

    let conv2 = tensor(
        weights,
        &format!("{prefix}.conv.3.weight"),
        &[out_channels, out_channels, 3, 3],
    )?;
    branch = conv2d_pad_same_with_compute(
        compute,
        &branch,
        out_channels,
        h,
        w,
        conv2,
        out_channels,
        3,
        3,
        None,
    )?;
    apply_bn(
        weights,
        &format!("{prefix}.conv.4"),
        &mut branch,
        out_channels,
        h,
        w,
    )?;
    relu_inplace(&mut branch);

    let residual = if in_channels == out_channels {
        input.to_vec()
    } else {
        let shortcut_w = tensor(
            weights,
            &format!("{prefix}.shortcut.weight"),
            &[out_channels, in_channels, 1, 1],
        )?;
        let shortcut_b = tensor(weights, &format!("{prefix}.shortcut.bias"), &[out_channels])?;
        conv2d_pad_same_with_compute(
            compute,
            input,
            in_channels,
            h,
            w,
            shortcut_w,
            out_channels,
            1,
            1,
            Some(shortcut_b),
        )?
    };
    for (value, skip) in branch.iter_mut().zip(residual) {
        *value += skip;
    }
    Ok(branch)
}

fn avg_pool2d_2x2(
    input: &[f32],
    channels: usize,
    h: usize,
    w: usize,
) -> Result<(Vec<f32>, usize, usize), VokraError> {
    if h < 2 || w < 2 || h % 2 != 0 || w % 2 != 0 {
        return Err(model_load(format!(
            "AvgPool2d(2) requires positive even axes, got [{channels}, {h}, {w}]"
        )));
    }
    let h_out = h / 2;
    let w_out = w / 2;
    let mut output = vec![0.0; channels * h_out * w_out];
    for channel in 0..channels {
        for oy in 0..h_out {
            for ox in 0..w_out {
                let mut sum = 0.0;
                for ky in 0..2 {
                    for kx in 0..2 {
                        sum += input[(channel * h + oy * 2 + ky) * w + ox * 2 + kx];
                    }
                }
                output[(channel * h_out + oy) * w_out + ox] = sum * 0.25;
            }
        }
    }
    Ok((output, h_out, w_out))
}

fn conv_transpose2d_3x3_stride2_pad1_output1(
    compute: &Compute,
    input: &[f32],
    in_channels: usize,
    h: usize,
    w: usize,
    weight: &[f32],
    out_channels: usize,
) -> Result<(Vec<f32>, usize, usize), VokraError> {
    let spatial = h * w;
    let projected = out_channels * 3 * 3;
    let h_out = h * 2;
    let w_out = w * 2;
    let mut input_spatial = vec![0.0; spatial * in_channels];
    for iy in 0..h {
        for ix in 0..w {
            let pixel = iy * w + ix;
            for channel in 0..in_channels {
                input_spatial[pixel * in_channels + channel] = input[(channel * h + iy) * w + ix];
            }
        }
    }
    let mut contributions = vec![0.0; spatial * projected];
    compute.gemm_f32(
        spatial,
        projected,
        in_channels,
        &input_spatial,
        weight,
        None,
        &mut contributions,
    )?;
    let mut output = vec![0.0; out_channels * h_out * w_out];
    for iy in 0..h {
        for ix in 0..w {
            let row = (iy * w + ix) * projected;
            for out_channel in 0..out_channels {
                for ky in 0..3 {
                    let oy = iy as isize * 2 + ky as isize - 1;
                    if oy < 0 || oy as usize >= h_out {
                        continue;
                    }
                    for kx in 0..3 {
                        let ox = ix as isize * 2 + kx as isize - 1;
                        if ox < 0 || ox as usize >= w_out {
                            continue;
                        }
                        let col = (out_channel * 3 + ky) * 3 + kx;
                        output[(out_channel * h_out + oy as usize) * w_out + ox as usize] +=
                            contributions[row + col];
                    }
                }
            }
        }
    }
    Ok((output, h_out, w_out))
}

fn concat_channels(
    left: &[f32],
    left_channels: usize,
    right: &[f32],
    right_channels: usize,
    h: usize,
    w: usize,
) -> Result<Vec<f32>, VokraError> {
    let spatial = h * w;
    if left.len() != left_channels * spatial || right.len() != right_channels * spatial {
        return Err(model_load(format!(
            "skip-concat shape mismatch: left={} right={} axes=[{h},{w}]",
            left.len(),
            right.len()
        )));
    }
    let mut output = Vec::with_capacity((left_channels + right_channels) * spatial);
    output.extend_from_slice(left);
    output.extend_from_slice(right);
    Ok(output)
}

fn bigru(
    compute: &Compute,
    weights: &RmvpeWeights,
    input: &[f32],
    n_frames: usize,
) -> Result<Vec<f32>, VokraError> {
    if input.len() != n_frames * GRU_INPUT {
        return Err(model_load(format!(
            "BiGRU input len {} != {n_frames}*{GRU_INPUT}",
            input.len()
        )));
    }
    let w_ih_f = tensor(
        weights,
        "fc.0.gru.weight_ih_l0",
        &[3 * GRU_HIDDEN, GRU_INPUT],
    )?;
    let w_hh_f = tensor(
        weights,
        "fc.0.gru.weight_hh_l0",
        &[3 * GRU_HIDDEN, GRU_HIDDEN],
    )?;
    let b_ih_f = tensor(weights, "fc.0.gru.bias_ih_l0", &[3 * GRU_HIDDEN])?;
    let b_hh_f = tensor(weights, "fc.0.gru.bias_hh_l0", &[3 * GRU_HIDDEN])?;
    let w_ih_r = tensor(
        weights,
        "fc.0.gru.weight_ih_l0_reverse",
        &[3 * GRU_HIDDEN, GRU_INPUT],
    )?;
    let w_hh_r = tensor(
        weights,
        "fc.0.gru.weight_hh_l0_reverse",
        &[3 * GRU_HIDDEN, GRU_HIDDEN],
    )?;
    let b_ih_r = tensor(weights, "fc.0.gru.bias_ih_l0_reverse", &[3 * GRU_HIDDEN])?;
    let b_hh_r = tensor(weights, "fc.0.gru.bias_hh_l0_reverse", &[3 * GRU_HIDDEN])?;

    let mut forward = vec![0.0; n_frames * GRU_HIDDEN];
    let mut state = vec![0.0; GRU_HIDDEN];
    for frame in 0..n_frames {
        let x = &input[frame * GRU_INPUT..(frame + 1) * GRU_INPUT];
        state = gru_cell_step_with_compute(
            compute, x, &state, w_ih_f, w_hh_f, b_ih_f, b_hh_f, GRU_HIDDEN, GRU_INPUT,
        )?;
        forward[frame * GRU_HIDDEN..(frame + 1) * GRU_HIDDEN].copy_from_slice(&state);
    }

    let mut reverse = vec![0.0; n_frames * GRU_HIDDEN];
    state.fill(0.0);
    for frame in (0..n_frames).rev() {
        let x = &input[frame * GRU_INPUT..(frame + 1) * GRU_INPUT];
        state = gru_cell_step_with_compute(
            compute, x, &state, w_ih_r, w_hh_r, b_ih_r, b_hh_r, GRU_HIDDEN, GRU_INPUT,
        )?;
        reverse[frame * GRU_HIDDEN..(frame + 1) * GRU_HIDDEN].copy_from_slice(&state);
    }

    let mut output = vec![0.0; n_frames * 2 * GRU_HIDDEN];
    for frame in 0..n_frames {
        let dst = frame * 2 * GRU_HIDDEN;
        output[dst..dst + GRU_HIDDEN]
            .copy_from_slice(&forward[frame * GRU_HIDDEN..(frame + 1) * GRU_HIDDEN]);
        output[dst + GRU_HIDDEN..dst + 2 * GRU_HIDDEN]
            .copy_from_slice(&reverse[frame * GRU_HIDDEN..(frame + 1) * GRU_HIDDEN]);
    }
    Ok(output)
}

fn probabilities_from_hidden(
    compute: &Compute,
    weights: &RmvpeWeights,
    hidden: &[f32],
    n_frames: usize,
) -> Result<Vec<f32>, VokraError> {
    let recurrent = bigru(compute, weights, hidden, n_frames)?;
    let head_w = tensor(weights, "fc.1.weight", &[N_CLASS, 2 * GRU_HIDDEN])?;
    let head_b = tensor(weights, "fc.1.bias", &[N_CLASS])?;
    let mut probabilities = linear_frames_with_compute(
        compute,
        &recurrent,
        n_frames,
        2 * GRU_HIDDEN,
        head_w,
        head_b,
        N_CLASS,
    )?;
    sigmoid_inplace(&mut probabilities);
    Ok(probabilities)
}

fn frames_from_probabilities(
    probabilities: &[f32],
    n_frames: usize,
    config: &RmvpeConfig,
) -> Vec<F0Frame> {
    let hop = config.hop as usize;
    let sample_rate = config.sample_rate as f32;
    let mut frames = Vec::with_capacity(n_frames);
    for frame in 0..n_frames {
        let row = &probabilities[frame * N_CLASS..(frame + 1) * N_CLASS];
        let (hz, voiced, confidence) = decode_class_to_hz(row, config, VOICED_THRESHOLD);
        frames.push(F0Frame {
            time_sec: (frame * hop) as f32 / sample_rate,
            hz,
            voiced,
            confidence,
        });
    }
    frames
}

pub(super) fn forward_from_hidden(
    model: &RMVPE,
    hidden: &[f32],
    n_frames: usize,
    feature_dim: usize,
    sample_rate: u32,
) -> Result<Vec<F0Frame>, VokraError> {
    if sample_rate != model.config.sample_rate {
        return Err(VokraError::InvalidArgument(format!(
            "rmvpe requires {} Hz hidden-state timing, got {sample_rate}; no silent resampling",
            model.config.sample_rate
        )));
    }
    if feature_dim != GRU_INPUT || hidden.len() != n_frames * feature_dim {
        return Err(model_load(format!(
            "hidden shape [{n_frames}, {feature_dim}] with len {} != canonical [{n_frames}, {GRU_INPUT}]",
            hidden.len()
        )));
    }
    let compute = Compute::for_backend(model.backend, RMVPE_HOT_OPS)?;
    let probabilities = probabilities_from_hidden(&compute, &model.weights, hidden, n_frames)?;
    Ok(frames_from_probabilities(
        &probabilities,
        n_frames,
        &model.config,
    ))
}

pub(super) fn forward(
    model: &RMVPE,
    pcm: &[f32],
    sample_rate: u32,
) -> Result<Vec<F0Frame>, VokraError> {
    if sample_rate != model.config.sample_rate {
        return Err(VokraError::InvalidArgument(format!(
            "rmvpe accepts {} Hz PCM, got {sample_rate}; resample explicitly before inference",
            model.config.sample_rate
        )));
    }
    let hop = model.config.hop as usize;
    let expected_frames = pcm.len() / hop;
    if expected_frames == 0 {
        return Ok(Vec::new());
    }

    // `src/inference.py` pads the waveform on the right so the centered-STFT
    // frame count is a multiple of 32 before the five AvgPool2d stages.
    let segment = hop * 32;
    let blocks = (pcm.len() + hop).div_ceil(segment);
    let padded_len = blocks * segment - hop;
    let mut padded_pcm = Vec::with_capacity(padded_len);
    padded_pcm.extend_from_slice(pcm);
    padded_pcm.resize(padded_len, 0.0);
    let mel = model.mel_spectrogram(&padded_pcm);
    let model_frames = blocks * 32;
    if mel.len() != model_frames || mel.iter().any(|row| row.len() != 128) {
        return Err(model_load(format!(
            "frontend produced [{}, ?] mel, expected [{model_frames}, 128]",
            mel.len()
        )));
    }

    let compute = Compute::for_backend(model.backend, RMVPE_HOT_OPS)?;
    let mut feature = Vec::with_capacity(model_frames * 128);
    for row in &mel {
        feature.extend_from_slice(row);
    }
    let mut channels = 1usize;
    let mut h = model_frames;
    let mut w = 128usize;
    apply_bn(
        &model.weights,
        "unet.encoder.bn",
        &mut feature,
        channels,
        h,
        w,
    )?;

    let mut skips = Vec::with_capacity(ENCODER_LAYERS);
    let mut out_channels = 16usize;
    for layer in 0..ENCODER_LAYERS {
        for block in 0..RESIDUAL_BLOCKS {
            feature = conv_block_res(
                &compute,
                &model.weights,
                &format!("unet.encoder.layers.{layer}.conv.{block}"),
                &feature,
                channels,
                out_channels,
                h,
                w,
            )?;
            channels = out_channels;
        }
        skips.push((feature.clone(), channels, h, w));
        (feature, h, w) = avg_pool2d_2x2(&feature, channels, h, w)?;
        out_channels *= 2;
    }

    out_channels = 512;
    for layer in 0..INTERMEDIATE_LAYERS {
        for block in 0..RESIDUAL_BLOCKS {
            feature = conv_block_res(
                &compute,
                &model.weights,
                &format!("unet.intermediate.layers.{layer}.conv.{block}"),
                &feature,
                channels,
                out_channels,
                h,
                w,
            )?;
            channels = out_channels;
        }
    }

    for layer in 0..DECODER_LAYERS {
        out_channels = channels / 2;
        let transposed = tensor(
            &model.weights,
            &format!("unet.decoder.layers.{layer}.conv1.0.weight"),
            &[channels, out_channels, 3, 3],
        )?;
        (feature, h, w) = conv_transpose2d_3x3_stride2_pad1_output1(
            &compute,
            &feature,
            channels,
            h,
            w,
            transposed,
            out_channels,
        )?;
        channels = out_channels;
        apply_bn(
            &model.weights,
            &format!("unet.decoder.layers.{layer}.conv1.1"),
            &mut feature,
            channels,
            h,
            w,
        )?;
        relu_inplace(&mut feature);

        let (skip, skip_channels, skip_h, skip_w) = &skips[ENCODER_LAYERS - 1 - layer];
        if (*skip_h, *skip_w, *skip_channels) != (h, w, channels) {
            return Err(model_load(format!(
                "decoder layer {layer} skip shape [{skip_channels},{skip_h},{skip_w}] != [{channels},{h},{w}]"
            )));
        }
        feature = concat_channels(&feature, channels, skip, *skip_channels, h, w)?;
        channels += *skip_channels;
        for block in 0..RESIDUAL_BLOCKS {
            feature = conv_block_res(
                &compute,
                &model.weights,
                &format!("unet.decoder.layers.{layer}.conv2.{block}"),
                &feature,
                channels,
                out_channels,
                h,
                w,
            )?;
            channels = out_channels;
        }
    }

    if (channels, h, w) != (16, model_frames, 128) {
        return Err(model_load(format!(
            "U-Net output [{channels},{h},{w}] != [16,{model_frames},128]"
        )));
    }
    let cnn_w = tensor(&model.weights, "cnn.weight", &[3, 16, 3, 3])?;
    let cnn_b = tensor(&model.weights, "cnn.bias", &[3])?;
    feature =
        conv2d_pad_same_with_compute(&compute, &feature, 16, h, w, cnn_w, 3, 3, 3, Some(cnn_b))?;
    let hidden = collapse_nchw_to_frames(&feature, 3, h, w);
    let probabilities = probabilities_from_hidden(&compute, &model.weights, &hidden, h)?;
    let mut frames = frames_from_probabilities(&probabilities, h, &model.config);
    frames.truncate(expected_frames);
    Ok(frames)
}
