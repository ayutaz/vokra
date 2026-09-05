//! Scalar INT8 kernels for microWakeWord (M5-03b Phase 2).
//!
//! # What lives here
//!
//! The four op kinds the microWakeWord (MC-MobileNet on Cortex-M55) forward
//! walks — [`conv2d_int8`], [`depthwise_conv2d_int8`], [`fully_connected_int8`],
//! [`sigmoid_int8`], and [`softmax_int8`] — plus a shared [`ConvDims`]
//! parameter struct. Every kernel is `#![no_std]`-clean (`core` + `alloc` only)
//! and uses **scalar** arithmetic — no `unsafe`, no SIMD intrinsics, no `libm`.
//! The Cortex-M55 Helium (MVE) fast path is deliberately deferred (M5-03 ADR):
//! landing a portable scalar path first pins parity vs. the offline sidecar's
//! reference dequantisation, and Helium can be added later without changing
//! the type surface.
//!
//! # Sibling to `vokra-vad-micro`
//!
//! [`vokra_vad_micro::scalar`](https://docs.rs/vokra-vad-micro) holds the same
//! shape of scalar helpers but for Silero VAD's **f32** LSTM forward (its
//! `math` sibling is a private module, so `scalar` is the public surface).
//! This module is the microWakeWord counterpart for its **int8** MC-MobileNet
//! forward. Both crates keep their numeric core out of `vokra-models` so the
//! `thumbv8m-none` cross-build (IoT Tier 3, NFR-PT-03) can name a small
//! `#![no_std] + alloc` crate directly.
//!
//! # TFLite-compatible affine quantisation
//!
//! Every kernel follows the standard TFLite signed-int8 convention (see
//! [`crate::features::quantize_int8`] for the mirror on the feature-extraction
//! side of the FFI):
//!
//! * Activations are `i8` with a per-tensor `(scale, zero_point)` pair.
//! * Weights are `i8` with **symmetric** per-tensor quantisation
//!   (`weight_zero_point ≡ 0` — the TFLite default for signed `i8` weights),
//!   so this module does not expose a weight zero-point parameter.
//! * Bias is `i32`, pre-scaled by the sidecar to `input_scale · weight_scale`
//!   with `bias_zero_point ≡ 0`. This matches
//!   `tflite::reference_ops::Conv` (Apache-2.0) verbatim.
//! * The requantisation multiplier is passed as a plain `f32`:
//!   `output_scale = input_scale · weight_scale / real_output_scale`. This is
//!   deterministic across targets (plain IEEE-754 multiply-then-round; no
//!   fixed-point shift-right whose rounding differs on Arm vs. x86 without
//!   care) and mirrors the `dequantize_int8` → `quantize_int8` round trip in
//!   [`crate::features`]. When Helium (MVE) lands in a follow-up WP the
//!   fixed-point form (`M * 2^-shift`) can drop in behind the same signature
//!   without changing callers.
//!
//! For each accumulator:
//!
//! ```text
//! acc_i32 = Σ (input_i8 - input_zero_point) · weight_i8   (bias omitted here — added below)
//! output_i8 = clamp( round((acc_i32 + bias_i32) · output_scale) + output_zero_point, -128, 127 )
//! ```
//!
//! `round` is round-half-away-from-zero — the same convention
//! [`crate::features::quantize_int8`] uses — so a round-trip through this
//! module's requantiser matches the offline sidecar bit-for-bit at the tie
//! points.
//!
//! # No-`unsafe`, no `libm`
//!
//! Workspace lints enforce `unsafe_code = "deny"` (crate has none) and
//! `deny.toml` bans `libm`. The `exp` in [`sigmoid_int8`] / [`softmax_int8`]
//! is a self-contained scalar routine private to this module (see
//! [`exp_int8`]), matching the sister-crate `vokra_vad_micro::scalar::exp`
//! rule: no library dependency, deterministic across std and no_std.

// `alloc` items that live in the prelude under `std` need explicit imports
// under `#![no_std]`. The sister `weights` / `model` modules gate the same way.
#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

use vokra_core::{Result, VokraError};

// ---------------------------------------------------------------------------
// Shared parameter struct.
// ---------------------------------------------------------------------------

/// Shape / stride / padding parameters shared by [`conv2d_int8`] and
/// [`depthwise_conv2d_int8`].
///
/// Layout conventions (TFLite-compatible):
///
/// * **Input**  is NHWC: `[1, in_h, in_w, in_c]` (batch always 1 — Vokra never
///   runs microWakeWord in a batched setting).
/// * **Weight** for [`conv2d_int8`] is OHWI: `[out_c, kh, kw, in_c]`. For
///   [`depthwise_conv2d_int8`] with the usual `depth_multiplier = 1` it is
///   `[1, kh, kw, in_c]` (i.e. `out_c` field is unused — the effective
///   `out_c` equals `in_c`).
/// * **Bias**   is `[out_c]` (int32, pre-scaled).
/// * **Output** is NHWC: `[1, out_h, out_w, out_c]` where
///   `out_h = (in_h + 2·pad_h - kh) / stride_h + 1` (and analogous for `w`).
///
/// `out_h` / `out_w` are recomputed inside the kernel from the other fields,
/// so callers never have to pass a redundant shape argument that could drift.
#[derive(Debug, Clone, Copy)]
pub struct ConvDims {
    /// Input height (spatial rows).
    pub in_h: usize,
    /// Input width (spatial cols).
    pub in_w: usize,
    /// Input channels.
    pub in_c: usize,
    /// Output channels. For [`depthwise_conv2d_int8`] this is ignored (the
    /// effective `out_c` equals `in_c` × `depth_multiplier`, and Vokra pins
    /// `depth_multiplier = 1`).
    pub out_c: usize,
    /// Kernel height.
    pub kh: usize,
    /// Kernel width.
    pub kw: usize,
    /// Vertical stride.
    pub stride_h: usize,
    /// Horizontal stride.
    pub stride_w: usize,
    /// Zero padding on top & bottom (rows).
    pub pad_h: usize,
    /// Zero padding on left & right (cols).
    pub pad_w: usize,
}

impl ConvDims {
    /// Output height, derived from `(in_h + 2·pad_h - kh) / stride_h + 1`.
    /// Returns `None` if `kh > in_h + 2·pad_h` (a degenerate config that a
    /// well-formed microWakeWord model never emits, but the sidecar is not the
    /// runtime — validate at the boundary).
    pub fn out_h(&self) -> Option<usize> {
        let padded = self.in_h.checked_add(self.pad_h.checked_mul(2)?)?;
        if self.kh > padded || self.stride_h == 0 {
            return None;
        }
        Some((padded - self.kh) / self.stride_h + 1)
    }

    /// Output width. See [`Self::out_h`].
    pub fn out_w(&self) -> Option<usize> {
        let padded = self.in_w.checked_add(self.pad_w.checked_mul(2)?)?;
        if self.kw > padded || self.stride_w == 0 {
            return None;
        }
        Some((padded - self.kw) / self.stride_w + 1)
    }
}

// ---------------------------------------------------------------------------
// Shared requantisation helper.
// ---------------------------------------------------------------------------

/// Requantises an `i32` accumulator (already includes bias) to `i8` output
/// using TFLite semantics: `out = clamp(round(acc · scale) + zero_point, -128, 127)`.
///
/// Round-half-away-from-zero — the same convention
/// [`crate::features::quantize_int8`] uses on the FFI's feature-extraction
/// side, so a round-trip through this pair is bit-identical at the tie points.
///
/// # `debug_assert` on `scale`
///
/// A `scale ≤ 0` in production would sign-flip every output. Callers get a
/// `debug_assert` in debug builds; release paths trust the sidecar's
/// quantisation metadata (validated at model-load time, follow-up WP).
#[inline]
fn requantize(
    acc_with_bias: i32,
    output_scale: f32,
    output_zero_point: i8,
    fused_relu: bool,
) -> i8 {
    let scaled = (acc_with_bias as f32) * output_scale;
    // Round-half-away-from-zero (matches `quantize_int8` in `features.rs`).
    let rounded = if scaled >= 0.0 {
        scaled + 0.5
    } else {
        scaled - 0.5
    };
    // Saturate to i32 first (guards against overflow when `scaled` is
    // absurd), then clamp to i8. `saturating_add_i32` isn't needed: the
    // `as i32` cast on an out-of-range f32 saturates by Rust semantics
    // (2021 edition; see `f32 as i32` conversion rules).
    let requant = (rounded as i32).saturating_add(output_zero_point as i32);
    let lower = if fused_relu {
        (i8::MIN as i32).max(output_zero_point as i32)
    } else {
        i8::MIN as i32
    };
    requant.clamp(lower, i8::MAX as i32) as i8
}

enum Requantization<'a> {
    Scalar { scale: f32, zero_point: i8 },
    PerChannel { scales: &'a [f32], zero_point: i8 },
}

impl<'a> Requantization<'a> {
    fn validate(&self, channels: usize, op: &str) -> Result<()> {
        match self {
            Self::Scalar { scale, .. } => {
                if !scale.is_finite() || *scale <= 0.0 {
                    return Err(VokraError::InvalidArgument(alloc::format!(
                        "{op}: output_scale must be finite and > 0"
                    )));
                }
            }
            Self::PerChannel { scales, .. } => {
                if scales.len() != channels {
                    return Err(VokraError::InvalidArgument(alloc::format!(
                        "{op}: output scales must have {channels} channels (scales={})",
                        scales.len()
                    )));
                }
                if scales
                    .iter()
                    .any(|scale| !scale.is_finite() || *scale <= 0.0)
                {
                    return Err(VokraError::InvalidArgument(alloc::format!(
                        "{op}: output scales must be finite and > 0"
                    )));
                }
            }
        }
        Ok(())
    }

    #[inline]
    fn at(&self, channel: usize) -> (f32, i8) {
        match self {
            Self::Scalar { scale, zero_point } => (*scale, *zero_point),
            Self::PerChannel { scales, zero_point } => (scales[channel], *zero_point),
        }
    }
}

// ---------------------------------------------------------------------------
// conv2d (standard) — TFLite `CONV_2D` int8 reference.
// ---------------------------------------------------------------------------

/// Standard 2-D convolution, INT8 activations & weights, INT32 bias.
///
/// Follows TFLite `CONV_2D`'s `reference_ops::Conv` (Apache-2.0) topology:
/// for each output pixel `(oy, ox, oc)`, sum
/// `(input[b, iy, ix, ic] - input_zero_point) · weight[oc, ky, kx, ic]` across
/// the receptive window and all input channels, add `bias[oc]`, then
/// requantise via [`requantize`].
///
/// Batch is fixed at 1 (Vokra never batches KWS).
///
/// # Errors
///
/// Returns [`VokraError::InvalidArgument`] if any buffer length is wrong for
/// the declared [`ConvDims`], or if the dims themselves would produce a
/// negative output extent (see [`ConvDims::out_h`]).
///
/// # Layout invariants
///
/// * `input.len() == dims.in_h · dims.in_w · dims.in_c`
/// * `weight.len() == dims.out_c · dims.kh · dims.kw · dims.in_c`
/// * `bias.len() == dims.out_c`
/// * `output.len() == out_h · out_w · dims.out_c`
///
/// The kernel does not allocate — it writes into the caller-provided
/// `output` buffer.
//
// `too_many_arguments`: 8 args (four buffers + two zero-points + output-scale
// + dims struct) is the minimum a TFLite-shape INT8 conv exposes to callers.
// The sister `conv1d` in `crates/vokra-vad-micro/src/math.rs` uses the same
// `#[allow]` for the same reason (that module is private and its `conv1d` is
// `pub(crate)`, so it has no importable path — the file is the only referent).
#[allow(clippy::too_many_arguments)]
pub fn conv2d_int8(
    input: &[i8],
    weight: &[i8],
    bias: &[i32],
    output: &mut [i8],
    input_zero_point: i8,
    output_zero_point: i8,
    output_scale: f32,
    dims: ConvDims,
) -> Result<()> {
    conv2d_int8_impl(
        input,
        weight,
        bias,
        output,
        input_zero_point,
        Requantization::Scalar {
            scale: output_scale,
            zero_point: output_zero_point,
        },
        false,
        dims,
    )
}

/// Standard convolution with per-output-channel float-scale requantisation.
///
/// `output_scales` is indexed by output channel; the output zero-point is
/// per-tensor and therefore supplied once.
/// When `fused_relu` is true, the quantised result is clamped to the TFLite
/// quantised representation of `[0, +inf)` for each channel.
#[allow(clippy::too_many_arguments)]
pub fn conv2d_int8_per_channel(
    input: &[i8],
    weight: &[i8],
    bias: &[i32],
    output: &mut [i8],
    input_zero_point: i8,
    output_scales: &[f32],
    output_zero_point: i8,
    fused_relu: bool,
    dims: ConvDims,
) -> Result<()> {
    conv2d_int8_impl(
        input,
        weight,
        bias,
        output,
        input_zero_point,
        Requantization::PerChannel {
            scales: output_scales,
            zero_point: output_zero_point,
        },
        fused_relu,
        dims,
    )
}

#[allow(clippy::too_many_arguments)]
fn conv2d_int8_impl(
    input: &[i8],
    weight: &[i8],
    bias: &[i32],
    output: &mut [i8],
    input_zero_point: i8,
    quantization: Requantization<'_>,
    fused_relu: bool,
    dims: ConvDims,
) -> Result<()> {
    let out_h = dims
        .out_h()
        .ok_or_else(|| VokraError::InvalidArgument("conv2d_int8: invalid out_h".into()))?;
    let out_w = dims
        .out_w()
        .ok_or_else(|| VokraError::InvalidArgument("conv2d_int8: invalid out_w".into()))?;

    validate_conv_buffers(
        input.len(),
        weight.len(),
        bias.len(),
        output.len(),
        dims,
        out_h,
        out_w,
        /*depthwise=*/ false,
    )?;

    quantization.validate(dims.out_c, "conv2d_int8")?;
    let input_zp = input_zero_point as i32;

    for oy in 0..out_h {
        for ox in 0..out_w {
            for oc in 0..dims.out_c {
                let mut acc: i32 = 0;
                for ky in 0..dims.kh {
                    let iy_signed = (oy * dims.stride_h) as i32 + ky as i32 - dims.pad_h as i32;
                    if iy_signed < 0 || (iy_signed as usize) >= dims.in_h {
                        continue; // padded row → contributes 0 to the sum.
                    }
                    let iy = iy_signed as usize;
                    for kx in 0..dims.kw {
                        let ix_signed = (ox * dims.stride_w) as i32 + kx as i32 - dims.pad_w as i32;
                        if ix_signed < 0 || (ix_signed as usize) >= dims.in_w {
                            continue;
                        }
                        let ix = ix_signed as usize;
                        for ic in 0..dims.in_c {
                            let in_v = input[(iy * dims.in_w + ix) * dims.in_c + ic] as i32;
                            let w_v = weight[((oc * dims.kh + ky) * dims.kw + kx) * dims.in_c + ic]
                                as i32;
                            // Standard TFLite formulation with symmetric weights
                            // (weight_zero_point = 0), so no `- 0` subtraction.
                            acc += (in_v - input_zp) * w_v;
                        }
                    }
                }
                let acc_with_bias = acc.saturating_add(bias[oc]);
                let (output_scale, output_zero_point) = quantization.at(oc);
                output[(oy * out_w + ox) * dims.out_c + oc] =
                    requantize(acc_with_bias, output_scale, output_zero_point, fused_relu);
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// depthwise_conv2d — TFLite `DEPTHWISE_CONV_2D` int8 reference.
// ---------------------------------------------------------------------------

/// Depthwise 2-D convolution, INT8 activations & weights, INT32 bias.
///
/// Pins `depth_multiplier = 1` (Vokra never uses `> 1` in the MC-MobileNet
/// forward). The effective output-channel count equals `dims.in_c`; the
/// `dims.out_c` field is ignored here and callers should set it to `in_c` for
/// clarity. Weight layout matches TFLite: `[1, kh, kw, in_c]` — the outer
/// dimension is a batch of 1 (a historical TFLite quirk).
///
/// # Errors
///
/// Same buffer-length validation as [`conv2d_int8`], except output channels
/// are derived from `in_c` not `out_c`.
//
// `too_many_arguments`: mirrors [`conv2d_int8`]'s signature (same allow, same
// TFLite-shape rationale).
#[allow(clippy::too_many_arguments)]
pub fn depthwise_conv2d_int8(
    input: &[i8],
    weight: &[i8],
    bias: &[i32],
    output: &mut [i8],
    input_zero_point: i8,
    output_zero_point: i8,
    output_scale: f32,
    dims: ConvDims,
) -> Result<()> {
    depthwise_conv2d_int8_impl(
        input,
        weight,
        bias,
        output,
        input_zero_point,
        Requantization::Scalar {
            scale: output_scale,
            zero_point: output_zero_point,
        },
        false,
        dims,
    )
}

/// Depthwise convolution with per-output-channel float-scale requantisation
/// and an optional TFLite fused RELU clamp.
#[allow(clippy::too_many_arguments)]
pub fn depthwise_conv2d_int8_per_channel(
    input: &[i8],
    weight: &[i8],
    bias: &[i32],
    output: &mut [i8],
    input_zero_point: i8,
    output_scales: &[f32],
    output_zero_point: i8,
    fused_relu: bool,
    dims: ConvDims,
) -> Result<()> {
    depthwise_conv2d_int8_impl(
        input,
        weight,
        bias,
        output,
        input_zero_point,
        Requantization::PerChannel {
            scales: output_scales,
            zero_point: output_zero_point,
        },
        fused_relu,
        dims,
    )
}

#[allow(clippy::too_many_arguments)]
fn depthwise_conv2d_int8_impl(
    input: &[i8],
    weight: &[i8],
    bias: &[i32],
    output: &mut [i8],
    input_zero_point: i8,
    quantization: Requantization<'_>,
    fused_relu: bool,
    dims: ConvDims,
) -> Result<()> {
    let out_h = dims.out_h().ok_or_else(|| {
        VokraError::InvalidArgument("depthwise_conv2d_int8: invalid out_h".into())
    })?;
    let out_w = dims.out_w().ok_or_else(|| {
        VokraError::InvalidArgument("depthwise_conv2d_int8: invalid out_w".into())
    })?;

    validate_conv_buffers(
        input.len(),
        weight.len(),
        bias.len(),
        output.len(),
        dims,
        out_h,
        out_w,
        /*depthwise=*/ true,
    )?;

    quantization.validate(dims.in_c, "depthwise_conv2d_int8")?;
    let input_zp = input_zero_point as i32;
    let effective_out_c = dims.in_c;

    for oy in 0..out_h {
        for ox in 0..out_w {
            for oc in 0..effective_out_c {
                // In depthwise conv, each output channel reads only from its
                // matching input channel (`ic == oc`).
                let ic = oc;
                let mut acc: i32 = 0;
                for ky in 0..dims.kh {
                    let iy_signed = (oy * dims.stride_h) as i32 + ky as i32 - dims.pad_h as i32;
                    if iy_signed < 0 || (iy_signed as usize) >= dims.in_h {
                        continue;
                    }
                    let iy = iy_signed as usize;
                    for kx in 0..dims.kw {
                        let ix_signed = (ox * dims.stride_w) as i32 + kx as i32 - dims.pad_w as i32;
                        if ix_signed < 0 || (ix_signed as usize) >= dims.in_w {
                            continue;
                        }
                        let ix = ix_signed as usize;
                        let in_v = input[(iy * dims.in_w + ix) * dims.in_c + ic] as i32;
                        // Weight layout `[1, kh, kw, in_c]`: outer batch dim
                        // is 1, so weight index is `((ky * kw) + kx) * in_c + ic`.
                        let w_v = weight[(ky * dims.kw + kx) * dims.in_c + ic] as i32;
                        acc += (in_v - input_zp) * w_v;
                    }
                }
                let acc_with_bias = acc.saturating_add(bias[oc]);
                let (output_scale, output_zero_point) = quantization.at(oc);
                output[(oy * out_w + ox) * effective_out_c + oc] =
                    requantize(acc_with_bias, output_scale, output_zero_point, fused_relu);
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// fully_connected — TFLite `FULLY_CONNECTED` int8 reference.
// ---------------------------------------------------------------------------

/// Fully-connected (dense) layer, INT8 activations & weights, INT32 bias.
///
/// `weight` layout is `[out_dim, in_dim]` row-major (TFLite `FULLY_CONNECTED`
/// standard). Computes:
///
/// ```text
/// acc[j] = Σ_i (input[i] - input_zero_point) · weight[j, i]
/// output[j] = requantize(acc[j] + bias[j])
/// ```
///
/// # Errors
///
/// [`VokraError::InvalidArgument`] on any length mismatch.
pub fn fully_connected_int8(
    input: &[i8],
    weight: &[i8],
    bias: &[i32],
    output: &mut [i8],
    input_zero_point: i8,
    output_zero_point: i8,
    output_scale: f32,
) -> Result<()> {
    fully_connected_int8_impl(
        input,
        weight,
        bias,
        output,
        input_zero_point,
        Requantization::Scalar {
            scale: output_scale,
            zero_point: output_zero_point,
        },
        false,
    )
}

/// Fully-connected layer with per-output-channel float-scale requantisation
/// and an optional TFLite fused RELU clamp.
#[allow(clippy::too_many_arguments)]
pub fn fully_connected_int8_per_channel(
    input: &[i8],
    weight: &[i8],
    bias: &[i32],
    output: &mut [i8],
    input_zero_point: i8,
    output_scales: &[f32],
    output_zero_point: i8,
    fused_relu: bool,
) -> Result<()> {
    fully_connected_int8_impl(
        input,
        weight,
        bias,
        output,
        input_zero_point,
        Requantization::PerChannel {
            scales: output_scales,
            zero_point: output_zero_point,
        },
        fused_relu,
    )
}

#[allow(clippy::too_many_arguments)]
fn fully_connected_int8_impl(
    input: &[i8],
    weight: &[i8],
    bias: &[i32],
    output: &mut [i8],
    input_zero_point: i8,
    quantization: Requantization<'_>,
    fused_relu: bool,
) -> Result<()> {
    let in_dim = input.len();
    let out_dim = output.len();
    let expected_weight = out_dim.checked_mul(in_dim).ok_or_else(|| {
        VokraError::InvalidArgument("fully_connected_int8: dimensions overflow".into())
    })?;
    if weight.len() != expected_weight {
        return Err(VokraError::InvalidArgument(alloc::format!(
            "fully_connected_int8: weight len {} != out_dim * in_dim ({} * {} = {})",
            weight.len(),
            out_dim,
            in_dim,
            expected_weight,
        )));
    }
    if bias.len() != out_dim {
        return Err(VokraError::InvalidArgument(alloc::format!(
            "fully_connected_int8: bias len {} != out_dim {}",
            bias.len(),
            out_dim,
        )));
    }

    quantization.validate(out_dim, "fully_connected_int8")?;

    let input_zp = input_zero_point as i32;
    for j in 0..out_dim {
        let mut acc: i32 = 0;
        let row = &weight[j * in_dim..(j + 1) * in_dim];
        for i in 0..in_dim {
            acc += ((input[i] as i32) - input_zp) * (row[i] as i32);
        }
        let acc_with_bias = acc.saturating_add(bias[j]);
        let (output_scale, output_zero_point) = quantization.at(j);
        output[j] = requantize(acc_with_bias, output_scale, output_zero_point, fused_relu);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// sigmoid — TFLite `LOGISTIC` int8 reference (LUT-backed).
// ---------------------------------------------------------------------------

/// Logistic sigmoid on INT8 activations, elementwise.
///
/// Builds a 256-entry LUT on entry (`input_i8 → output_i8`) by dequantising
/// each possible `i8` input, evaluating `1 / (1 + e^-x)` in `f32` via the
/// self-contained [`exp_int8`], then requantising to the output. Every
/// subsequent element is then a single indexed lookup — the same technique
/// TFLite's reference `LOGISTIC` kernel uses (Apache-2.0).
///
/// The LUT is a fixed 256-entry stack array, so this path performs no heap
/// allocation and keeps the signature simple.
///
/// # Errors
///
/// Returns [`VokraError::InvalidArgument`] if `input.len() != output.len()`.
pub fn sigmoid_int8(
    input: &[i8],
    output: &mut [i8],
    input_scale: f32,
    input_zero_point: i8,
    output_scale: f32,
    output_zero_point: i8,
) -> Result<()> {
    if input.len() != output.len() {
        return Err(VokraError::InvalidArgument(alloc::format!(
            "sigmoid_int8: input len {} != output len {}",
            input.len(),
            output.len(),
        )));
    }
    debug_assert!(input_scale > 0.0, "sigmoid_int8: input_scale must be > 0");
    debug_assert!(output_scale > 0.0, "sigmoid_int8: output_scale must be > 0");

    // Build a 256-entry LUT indexed by `input_i8 as u8` (i.e. -128 → 0,
    // 127 → 255). Every possible input maps to exactly one output.
    let lut = build_sigmoid_lut(
        input_scale,
        input_zero_point,
        output_scale,
        output_zero_point,
    );
    for (o, &i) in output.iter_mut().zip(input.iter()) {
        *o = lut[(i as i32 + 128) as usize];
    }
    Ok(())
}

/// Precomputes the sigmoid LUT (indexed as `input_i8 + 128 → 0..=255`).
fn build_sigmoid_lut(
    input_scale: f32,
    input_zero_point: i8,
    output_scale: f32,
    output_zero_point: i8,
) -> [i8; 256] {
    let mut lut = [0i8; 256];
    for (idx, entry) in lut.iter_mut().enumerate() {
        let input_i8 = (idx as i32) - 128; // -128 ..= 127
        let x = ((input_i8 - input_zero_point as i32) as f32) * input_scale;
        // Logistic sigmoid `1 / (1 + e^-x)`, computed via the crate-local
        // scalar `exp` (no `libm` — see module docs).
        let y = 1.0 / (1.0 + exp_int8(-x));
        // Quantise `y ∈ [0, 1]` back to i8 with the output params.
        let scaled = y / output_scale;
        let rounded = if scaled >= 0.0 {
            scaled + 0.5
        } else {
            scaled - 0.5
        };
        let requant = (rounded as i32).saturating_add(output_zero_point as i32);
        *entry = requant.clamp(i8::MIN as i32, i8::MAX as i32) as i8;
    }
    lut
}

// ---------------------------------------------------------------------------
// softmax — TFLite `SOFTMAX` int8 reference (numerically-stable).
// ---------------------------------------------------------------------------

/// Numerically-stable softmax over `input` (single row), INT8 in & out.
///
/// Standard "subtract max, exp, normalise" formulation:
///
/// 1. Dequantise every `input_i8` to `x = (i - input_zero_point) · input_scale`.
/// 2. Take `max = max(x)`; compute `e_i = exp(x_i - max)`.
/// 3. Normalise `p_i = e_i / Σ e_j`.
/// 4. Quantise `p_i` back via `(output_scale, output_zero_point)` — TFLite
///    conventionally uses `output_scale = 1/256, output_zero_point = -128`,
///    but any valid pair is accepted (the kernel doesn't hard-code them).
///
/// The `- max` shift makes every `e_i ∈ (0, 1]`, so `Σ e_j ≤ N` and the
/// `f32` accumulator never overflows for any realistic `N`. The output is
/// single-row (batching is the caller's outer loop) — mirrors the sister
/// [`vokra_vad_micro`] pattern where the LSTM math also lives at
/// single-row granularity.
///
/// # Errors
///
/// [`VokraError::InvalidArgument`] if `input.len() != output.len()` or
/// either is empty.
pub fn softmax_int8(
    input: &[i8],
    output: &mut [i8],
    input_scale: f32,
    input_zero_point: i8,
    output_scale: f32,
    output_zero_point: i8,
) -> Result<()> {
    if input.len() != output.len() {
        return Err(VokraError::InvalidArgument(alloc::format!(
            "softmax_int8: input len {} != output len {}",
            input.len(),
            output.len(),
        )));
    }
    if input.is_empty() {
        return Err(VokraError::InvalidArgument(
            "softmax_int8: empty input".into(),
        ));
    }
    debug_assert!(input_scale > 0.0, "softmax_int8: input_scale must be > 0");
    debug_assert!(output_scale > 0.0, "softmax_int8: output_scale must be > 0");

    let input_zp = input_zero_point as i32;

    // (1) dequantise.
    let mut deq: Vec<f32> = input
        .iter()
        .map(|&q| ((q as i32) - input_zp) as f32 * input_scale)
        .collect();
    // (2) subtract max, then exp.
    let mut m = f32::NEG_INFINITY;
    for &v in &deq {
        if v > m {
            m = v;
        }
    }
    let mut sum = 0.0f32;
    for v in deq.iter_mut() {
        *v = exp_int8(*v - m);
        sum += *v;
    }
    // Guard the divide: `sum` is a sum of strictly positive exponentials so
    // it can only be 0 if every input was `-inf` (impossible for a real
    // dequantised i8) — but a `debug_assert!` documents the invariant.
    debug_assert!(sum > 0.0, "softmax_int8: exp sum must be > 0");
    let inv_sum = 1.0 / sum;

    // (3, 4) normalise & requantise.
    for (o, &p) in output.iter_mut().zip(deq.iter()) {
        let prob = p * inv_sum;
        let scaled = prob / output_scale;
        let rounded = if scaled >= 0.0 {
            scaled + 0.5
        } else {
            scaled - 0.5
        };
        let requant = (rounded as i32).saturating_add(output_zero_point as i32);
        *o = requant.clamp(i8::MIN as i32, i8::MAX as i32) as i8;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Scalar `exp` (private) — mirrors `vokra_vad_micro::scalar::exp` semantics.
// ---------------------------------------------------------------------------

/// Scalar `e^x` in pure `core` arithmetic. Used by [`sigmoid_int8`] and
/// [`softmax_int8`] instead of `f32::exp` (which is `std`-gated) or `libm`
/// (banned by `deny.toml`, NFR-DS-02).
///
/// # Algorithm
///
/// Range-reduces `x = k·ln 2 + r` with `|r| ≤ ln 2 / 2 ≈ 0.347`, evaluates a
/// degree-6 Taylor polynomial for `e^r` on that reduced range, then rescales
/// via `2^k` built from the IEEE-754 exponent bit-pattern (`ldexp`-style,
/// no library call).
///
/// The polynomial's residual over `|r| ≤ 0.347` is bounded by
/// `r⁷/7! ≈ 1.4·10⁻⁶ · 0.347⁷ ≈ 8·10⁻⁹` — well inside f32 rounding.
/// Saturates cleanly at both extremes (`x ≥ ~88` → `+∞` blocked by the
/// `if x > 88.0` guard; `x ≤ -88` → `0.0`), matching `f32::exp`'s
/// saturation behaviour.
fn exp_int8(x: f32) -> f32 {
    // Saturation guards. `f32` overflow at `e^x` is ~88.72 → +inf; underflow
    // at ~-103.28 → 0. Cap at ±88 to avoid returning inf/subnormals from
    // the polynomial branch.
    if x.is_nan() {
        return f32::NAN;
    }
    if x >= 88.0 {
        return f32::INFINITY;
    }
    if x <= -88.0 {
        return 0.0;
    }

    const LN_2: f32 = core::f32::consts::LN_2;
    const INV_LN_2: f32 = 1.0 / core::f32::consts::LN_2;

    // (1) Range reduction: k = round(x / ln 2), r = x - k · ln 2.
    let y = x * INV_LN_2;
    let k = if y >= 0.0 {
        (y + 0.5) as i32
    } else {
        (y - 0.5) as i32
    };
    let r = x - (k as f32) * LN_2;

    // (2) Degree-6 Taylor for e^r on |r| ≤ ln 2 / 2. Horner-style:
    // 1 + r · (1 + r/2 · (1 + r/3 · (1 + r/4 · (1 + r/5 · (1 + r/6)))))
    let poly = 1.0
        + r * (1.0
            + r * (0.5
                + r * ((1.0 / 6.0)
                    + r * ((1.0 / 24.0) + r * ((1.0 / 120.0) + r * (1.0 / 720.0))))));

    // (3) Rescale by 2^k via IEEE-754 exponent injection: build a f32
    // with mantissa = 1 and biased exponent = (127 + k). Range-limit k
    // to keep the constructed float finite (the ±88 guards above ensure
    // |k| ≤ ~127, so the exponent stays in [0, 254] — well inside the
    // f32 normal range).
    let biased_exp = ((127i32 + k) as u32) & 0xFF;
    let pow2k = f32::from_bits(biased_exp << 23);
    poly * pow2k
}

// ---------------------------------------------------------------------------
// Shared buffer-length validator.
// ---------------------------------------------------------------------------

/// Checks that `input` / `weight` / `bias` / `output` slice lengths match
/// what [`ConvDims`] declares. Depthwise mode uses a different weight layout
/// (`[1, kh, kw, in_c]` instead of `[out_c, kh, kw, in_c]`).
//
// `too_many_arguments`: the four buffer lengths + the dims struct + the
// derived (`out_h`, `out_w`) + a mode flag. Bundling into a struct would
// duplicate `ConvDims` — kept explicit for a private helper.
#[allow(clippy::too_many_arguments)]
fn validate_conv_buffers(
    input_len: usize,
    weight_len: usize,
    bias_len: usize,
    output_len: usize,
    dims: ConvDims,
    out_h: usize,
    out_w: usize,
    depthwise: bool,
) -> Result<()> {
    let expected_input = dims
        .in_h
        .checked_mul(dims.in_w)
        .and_then(|v| v.checked_mul(dims.in_c))
        .ok_or_else(|| VokraError::InvalidArgument("conv: input dimensions overflow".into()))?;
    if input_len != expected_input {
        return Err(VokraError::InvalidArgument(alloc::format!(
            "conv: input len {input_len} != in_h * in_w * in_c = {expected_input}"
        )));
    }
    let effective_out_c = if depthwise { dims.in_c } else { dims.out_c };
    let expected_weight = if depthwise {
        dims.kh
            .checked_mul(dims.kw)
            .and_then(|v| v.checked_mul(dims.in_c))
    } else {
        dims.out_c
            .checked_mul(dims.kh)
            .and_then(|v| v.checked_mul(dims.kw))
            .and_then(|v| v.checked_mul(dims.in_c))
    }
    .ok_or_else(|| VokraError::InvalidArgument("conv: weight dimensions overflow".into()))?;
    if weight_len != expected_weight {
        return Err(VokraError::InvalidArgument(alloc::format!(
            "conv: weight len {weight_len} != expected {expected_weight}"
        )));
    }
    if bias_len != effective_out_c {
        return Err(VokraError::InvalidArgument(alloc::format!(
            "conv: bias len {bias_len} != out_c {effective_out_c}"
        )));
    }
    let expected_output = out_h
        .checked_mul(out_w)
        .and_then(|v| v.checked_mul(effective_out_c))
        .ok_or_else(|| VokraError::InvalidArgument("conv: output dimensions overflow".into()))?;
    if output_len != expected_output {
        return Err(VokraError::InvalidArgument(alloc::format!(
            "conv: output len {output_len} != out_h * out_w * out_c = {expected_output}"
        )));
    }
    Ok(())
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    // Dimensional-indexing expressions in this module are written in the
    // canonical NHWC form `(row * H + col) * C + ch` (and its degenerate
    // `row * C + col` for FC weights). Clippy's `identity_op` / `erasing_op`
    // flag the trivially-zero (`0 * K`) and one-identity (`1 * K`, `x + 0`)
    // sub-expressions, but reducing them destroys the visual anchor between
    // the index formula and the intended (row, col, ch) tuple — the intent
    // is that a reader can read `input[(1 * H + 1) * C + 0]` as "centre
    // pixel, channel 0" without doing the algebra in their head.
    #![allow(clippy::identity_op, clippy::erasing_op)]

    use super::*;

    // ---- conv2d_int8 ---------------------------------------------------

    /// A 1×1 conv with identity weight (1) and no bias is the "identity"
    /// convolution: it should reproduce the input value after requantise.
    ///
    /// Setup: 1×1×1 input, single 1×1 kernel, `output_scale = 1.0` (bit-copy),
    /// zero_points = 0. Any input `i` should come back as `i`.
    #[test]
    fn conv2d_int8_identity_kernel_reproduces_input() {
        let dims = ConvDims {
            in_h: 1,
            in_w: 1,
            in_c: 1,
            out_c: 1,
            kh: 1,
            kw: 1,
            stride_h: 1,
            stride_w: 1,
            pad_h: 0,
            pad_w: 0,
        };
        for &v in &[-40i8, -1, 0, 1, 40, 100] {
            let input = [v];
            let weight = [1i8];
            let bias = [0i32];
            let mut out = [0i8; 1];
            conv2d_int8(&input, &weight, &bias, &mut out, 0, 0, 1.0, dims).unwrap();
            assert_eq!(out[0], v, "conv2d identity failed at v={v}");
        }
    }

    /// A 3×3 kernel with stride 2 over a 5×5 input yields a 2×2 output when
    /// pad=0. Every kernel value is 0 except one, so the output is a
    /// stride-2 subsampling of a single input pixel — sanity check for
    /// spatial indexing.
    #[test]
    fn conv2d_int8_stride_2_reduces_spatial_dims() {
        let dims = ConvDims {
            in_h: 5,
            in_w: 5,
            in_c: 1,
            out_c: 1,
            kh: 3,
            kw: 3,
            stride_h: 2,
            stride_w: 2,
            pad_h: 0,
            pad_w: 0,
        };
        // input has 25 elements, weight has 9 (all zero except centre = 1
        // → picks the centre tap of each window).
        let input: Vec<i8> = (0..25).map(|i| i as i8).collect();
        let mut weight = vec![0i8; 9];
        weight[4] = 1; // centre of 3×3
        let bias = vec![0i32; 1];
        let mut out = vec![0i8; 4]; // 2×2
        conv2d_int8(&input, &weight, &bias, &mut out, 0, 0, 1.0, dims).unwrap();
        // Windows at (oy=0,ox=0) → centre tap at (1,1) → input[1*5+1] = 6
        // Windows at (oy=0,ox=1) → centre tap at (1,3) → input[1*5+3] = 8
        // Windows at (oy=1,ox=0) → centre tap at (3,1) → input[3*5+1] = 16
        // Windows at (oy=1,ox=1) → centre tap at (3,3) → input[3*5+3] = 18
        assert_eq!(out, vec![6, 8, 16, 18]);
    }

    /// Bias-only test: zero weights, non-zero bias → every output equals
    /// requantise(bias). Isolates the bias-application path from the
    /// weight-multiplication path.
    #[test]
    fn conv2d_int8_bias_only_shifts_output() {
        let dims = ConvDims {
            in_h: 2,
            in_w: 2,
            in_c: 1,
            out_c: 1,
            kh: 1,
            kw: 1,
            stride_h: 1,
            stride_w: 1,
            pad_h: 0,
            pad_w: 0,
        };
        let input = vec![10i8, 20, 30, 40];
        let weight = vec![0i8; 1]; // zero weight → accumulator = 0
        let bias = vec![5i32];
        let mut out = vec![0i8; 4];
        conv2d_int8(&input, &weight, &bias, &mut out, 0, 0, 1.0, dims).unwrap();
        // Every output = requantize(0 + 5, 1.0, 0) = 5.
        assert_eq!(out, vec![5, 5, 5, 5]);
    }

    /// Requantisation must saturate the output at the `i8` range boundaries,
    /// not wrap around. Sends an accumulator way past `127` and checks the
    /// output pins at `127`.
    #[test]
    fn conv2d_int8_saturates_output_at_i8_bounds() {
        let dims = ConvDims {
            in_h: 1,
            in_w: 1,
            in_c: 1,
            out_c: 1,
            kh: 1,
            kw: 1,
            stride_h: 1,
            stride_w: 1,
            pad_h: 0,
            pad_w: 0,
        };
        // Positive saturation: (127 · 127) · 1.0 = 16129 → clamp to 127.
        let input = [127i8];
        let weight = [127i8];
        let bias = [0i32];
        let mut out = [0i8; 1];
        conv2d_int8(&input, &weight, &bias, &mut out, 0, 0, 1.0, dims).unwrap();
        assert_eq!(out[0], 127);

        // Negative saturation: (127 · -127) · 1.0 = -16129 → clamp to -128.
        let weight = [-127i8];
        let mut out = [0i8; 1];
        conv2d_int8(&input, &weight, &bias, &mut out, 0, 0, 1.0, dims).unwrap();
        assert_eq!(out[0], -128);
    }

    /// Padding: a 1×1 input, 3×3 kernel with pad=1 → a 1×1 output whose
    /// receptive window is (top-left 3×3 with the input at the centre and
    /// zeros elsewhere). With a kernel of all-1s, the sum equals the input
    /// value.
    #[test]
    fn conv2d_int8_zero_padding_treats_padded_pixels_as_zero() {
        let dims = ConvDims {
            in_h: 1,
            in_w: 1,
            in_c: 1,
            out_c: 1,
            kh: 3,
            kw: 3,
            stride_h: 1,
            stride_w: 1,
            pad_h: 1,
            pad_w: 1,
        };
        let input = [7i8];
        let weight = vec![1i8; 9];
        let bias = [0i32];
        let mut out = [0i8; 1];
        conv2d_int8(&input, &weight, &bias, &mut out, 0, 0, 1.0, dims).unwrap();
        assert_eq!(out[0], 7);
    }

    /// Wrong buffer length is a fail-closed `InvalidArgument` — NOT a panic.
    #[test]
    fn conv2d_int8_rejects_wrong_output_length() {
        let dims = ConvDims {
            in_h: 1,
            in_w: 1,
            in_c: 1,
            out_c: 1,
            kh: 1,
            kw: 1,
            stride_h: 1,
            stride_w: 1,
            pad_h: 0,
            pad_w: 0,
        };
        let input = [1i8];
        let weight = [1i8];
        let bias = [0i32];
        let mut out = [0i8; 5]; // wrong: should be 1
        assert!(matches!(
            conv2d_int8(&input, &weight, &bias, &mut out, 0, 0, 1.0, dims),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    // ---- depthwise_conv2d_int8 -----------------------------------------

    /// Depthwise 1×1 identity: two channels, each with weight = 1, produces
    /// each input channel independently at the output. Verifies per-channel
    /// isolation (a bug where DWConv mixed channels would show up here).
    #[test]
    fn depthwise_conv2d_int8_1x1_identity_preserves_channels() {
        let dims = ConvDims {
            in_h: 1,
            in_w: 1,
            in_c: 2,
            out_c: 2, // ignored in depthwise; effective = in_c
            kh: 1,
            kw: 1,
            stride_h: 1,
            stride_w: 1,
            pad_h: 0,
            pad_w: 0,
        };
        let input = [5i8, -5];
        let weight = [1i8, 1]; // one weight per channel
        let bias = [0i32, 0];
        let mut out = [0i8; 2];
        depthwise_conv2d_int8(&input, &weight, &bias, &mut out, 0, 0, 1.0, dims).unwrap();
        assert_eq!(out, [5, -5]);
    }

    /// Depthwise with per-channel bias should shift each channel independently.
    #[test]
    fn depthwise_conv2d_int8_bias_is_per_channel() {
        let dims = ConvDims {
            in_h: 1,
            in_w: 1,
            in_c: 3,
            out_c: 3,
            kh: 1,
            kw: 1,
            stride_h: 1,
            stride_w: 1,
            pad_h: 0,
            pad_w: 0,
        };
        let input = [0i8, 0, 0];
        let weight = [0i8, 0, 0]; // zero weights: only bias contributes
        let bias = [1i32, -1, 10];
        let mut out = [0i8; 3];
        depthwise_conv2d_int8(&input, &weight, &bias, &mut out, 0, 0, 1.0, dims).unwrap();
        assert_eq!(out, [1, -1, 10]);
    }

    /// Depthwise 3×3 with a centre-tap kernel and stride 1, pad 0 over a 3×3
    /// input picks the centre pixel of each of the four windows — but with
    /// pad=0 and in_h=in_w=3, kh=kw=3, stride=1 there's exactly ONE window,
    /// so out is 1×1×in_c and equals input[centre] per channel.
    #[test]
    fn depthwise_conv2d_int8_multichannel_spatial_correctness() {
        let dims = ConvDims {
            in_h: 3,
            in_w: 3,
            in_c: 2,
            out_c: 2,
            kh: 3,
            kw: 3,
            stride_h: 1,
            stride_w: 1,
            pad_h: 0,
            pad_w: 0,
        };
        // 3×3×2 input where centre-pixel channel 0 = 11 and channel 1 = 22.
        let mut input = vec![0i8; 3 * 3 * 2];
        input[(1 * 3 + 1) * 2 + 0] = 11; // centre, ch 0
        input[(1 * 3 + 1) * 2 + 1] = 22; // centre, ch 1
                                         // Weight: 3×3×2 with only centre-tap = 1 for both channels.
        let mut weight = vec![0i8; 3 * 3 * 2];
        weight[(1 * 3 + 1) * 2 + 0] = 1;
        weight[(1 * 3 + 1) * 2 + 1] = 1;
        let bias = vec![0i32; 2];
        let mut out = vec![0i8; 2]; // 1×1×2
        depthwise_conv2d_int8(&input, &weight, &bias, &mut out, 0, 0, 1.0, dims).unwrap();
        assert_eq!(out, vec![11, 22]);
    }

    // ---- fully_connected_int8 ------------------------------------------

    /// Identity weight matrix (diagonal 1, off-diagonal 0) with no bias
    /// should reproduce the input.
    #[test]
    fn fully_connected_int8_identity_weight_reproduces_input() {
        let input = [10i8, -20, 30];
        let mut weight = vec![0i8; 3 * 3];
        weight[0 * 3] = 1; // (0,0) = 1
        weight[1 * 3 + 1] = 1; // (1,1) = 1
        weight[2 * 3 + 2] = 1; // (2,2) = 1
        let bias = vec![0i32; 3];
        let mut out = vec![0i8; 3];
        fully_connected_int8(&input, &weight, &bias, &mut out, 0, 0, 1.0).unwrap();
        assert_eq!(out, vec![10, -20, 30]);
    }

    /// Bias-only: zero weight, non-zero bias.
    #[test]
    fn fully_connected_int8_bias_only_shifts_output() {
        let input = [10i8, 20];
        let weight = vec![0i8; 3 * 2];
        let bias = vec![1i32, -1, 5];
        let mut out = vec![0i8; 3];
        fully_connected_int8(&input, &weight, &bias, &mut out, 0, 0, 1.0).unwrap();
        assert_eq!(out, vec![1, -1, 5]);
    }

    /// Wrong weight length is a fail-closed InvalidArgument.
    #[test]
    fn fully_connected_int8_rejects_wrong_weight_length() {
        let input = [1i8, 2];
        let weight = vec![1i8; 3]; // should be 3 * 2 = 6
        let bias = vec![0i32; 3];
        let mut out = vec![0i8; 3];
        assert!(matches!(
            fully_connected_int8(&input, &weight, &bias, &mut out, 0, 0, 1.0),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    // ---- sigmoid_int8 ----------------------------------------------------

    /// `sigmoid(0) = 0.5`, so an input dequantising to 0 (`input == input_zp`)
    /// should quantise back near `output_zp + 128` for TFLite's usual
    /// `output_scale = 1/256, output_zp = -128` convention (dequant 0.5 →
    /// (0.5 - 0)/scale = 128, requantised to i8 = 0).
    ///
    /// With `output_scale = 1/256, output_zp = -128`, the requantised value
    /// is `(0.5 / (1/256)) + (-128) = 128 - 128 = 0`. Off-by-one at the
    /// rounding boundary is acceptable (`atol < 2`).
    #[test]
    fn sigmoid_int8_midpoint_at_input_zero_yields_output_center() {
        let input = [0i8]; // dequantises to 0 (input_zp = 0)
        let mut out = [0i8; 1];
        sigmoid_int8(
            &input,
            &mut out,
            /*input_scale=*/ 0.1,
            /*input_zp=*/ 0,
            /*output_scale=*/ 1.0 / 256.0,
            /*output_zp=*/ -128,
        )
        .unwrap();
        // sigmoid(0) = 0.5, quantised → 0.5 / (1/256) - 128 = 0.
        assert!(
            (out[0] as i32).abs() <= 1,
            "sigmoid(0) midpoint = {}, expected ~0 (±1)",
            out[0]
        );
    }

    /// Saturation: sigmoid(large positive) → ~1.0, sigmoid(large negative) → ~0.
    #[test]
    fn sigmoid_int8_saturates_at_output_bounds() {
        // Input scale 1.0, input_zp = 0. Input 100 → dequant 100 → sigmoid ~1.
        // Output scale 1/256, output_zp -128. Quantised 1.0 → 256 - 128 = 128
        // → clamps to 127.
        let input = [100i8, -100];
        let mut out = [0i8; 2];
        sigmoid_int8(
            &input,
            &mut out,
            /*input_scale=*/ 1.0,
            /*input_zp=*/ 0,
            /*output_scale=*/ 1.0 / 256.0,
            /*output_zp=*/ -128,
        )
        .unwrap();
        // sigmoid(100) ~ 1 → i8 127 (upper clamp).
        assert_eq!(out[0], 127, "sigmoid(+100) should saturate high");
        // sigmoid(-100) ~ 0 → i8 -128 (lower clamp).
        assert_eq!(out[1], -128, "sigmoid(-100) should saturate low");
    }

    /// Dense comparison across every possible `i8` input: LUT-based INT8
    /// sigmoid should match an f32 reference within `atol < 2` LSB. The
    /// bound is the standard INT8 sigmoid tolerance (one quant step for
    /// rounding + one for scale mismatch at the extremes).
    #[test]
    fn sigmoid_int8_matches_f32_reference_across_i8_range() {
        let input_scale = 0.05f32;
        let input_zp: i8 = 0;
        let output_scale = 1.0f32 / 256.0;
        let output_zp: i8 = -128;

        let input: Vec<i8> = (i8::MIN..=i8::MAX).collect();
        let mut out = vec![0i8; input.len()];
        sigmoid_int8(
            &input,
            &mut out,
            input_scale,
            input_zp,
            output_scale,
            output_zp,
        )
        .unwrap();

        let mut worst_abs = 0i32;
        for (idx, &i) in input.iter().enumerate() {
            let x = ((i as i32) - input_zp as i32) as f32 * input_scale;
            let y_ref = 1.0f32 / (1.0f32 + exp_int8(-x));
            // Same requant path the LUT builder uses, so any drift is
            // truly a kernel bug (not a test-code artifact).
            let scaled = y_ref / output_scale;
            let rounded = if scaled >= 0.0 {
                scaled + 0.5
            } else {
                scaled - 0.5
            };
            let requant = (rounded as i32).saturating_add(output_zp as i32);
            let ref_i8 = requant.clamp(i8::MIN as i32, i8::MAX as i32) as i8;
            let diff = (out[idx] as i32 - ref_i8 as i32).abs();
            if diff > worst_abs {
                worst_abs = diff;
            }
        }
        assert!(
            worst_abs < 2,
            "sigmoid_int8 worst-case diff = {worst_abs} LSB, expected < 2"
        );
    }

    /// Wrong length is InvalidArgument.
    #[test]
    fn sigmoid_int8_rejects_length_mismatch() {
        let input = [0i8; 4];
        let mut out = [0i8; 3];
        assert!(matches!(
            sigmoid_int8(&input, &mut out, 0.1, 0, 1.0 / 256.0, -128),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    // ---- softmax_int8 ----------------------------------------------------

    /// Uniform inputs produce a uniform output distribution. With the
    /// standard TFLite output params (`1/256, -128`), each output should
    /// dequantise to `1/N` — i.e. `output_i8 + 128 ≈ 256 / N` (rounded).
    #[test]
    fn softmax_int8_uniform_input_yields_uniform_output() {
        let n = 4;
        let input = vec![10i8; n]; // every entry equal → uniform softmax
        let mut out = vec![0i8; n];
        softmax_int8(
            &input,
            &mut out,
            /*input_scale=*/ 0.1,
            /*input_zp=*/ 0,
            /*output_scale=*/ 1.0 / 256.0,
            /*output_zp=*/ -128,
        )
        .unwrap();
        // Each entry should quantise to `1/4 = 0.25` → 0.25 * 256 - 128 = -64.
        for &o in &out {
            assert!(
                (o as i32 - (-64)).abs() <= 1,
                "uniform softmax output = {o}, expected ~-64"
            );
        }
    }

    /// Peaked input: one entry much larger → softmax concentrates on it.
    /// The winning class should saturate near +127, losers near -128.
    #[test]
    fn softmax_int8_peaked_input_saturates_winner() {
        // input scale 0.1: input 100 → dequant 10, others 0.
        // exp(10) / (exp(10) + 3·exp(0)) ≈ 22026 / 22029 ≈ 0.9998
        // losers: exp(0)/22029 ≈ 4.5e-5
        let input = [100i8, 0, 0, 0];
        let mut out = [0i8; 4];
        softmax_int8(&input, &mut out, 0.1, 0, 1.0 / 256.0, -128).unwrap();
        // Winner: 0.9998 * 256 - 128 = 127.9 → clamps to 127.
        assert_eq!(out[0], 127, "winner should saturate at +127");
        // Losers: 4.5e-5 * 256 - 128 ≈ -128 (rounds to exactly -128 in i8).
        for &loser in &out[1..] {
            assert_eq!(loser, -128, "loser should saturate at -128");
        }
    }

    /// The INT8-softmax property: sum of `(output_i8 - output_zp)` ≈ 256
    /// (i.e. sum of dequantised probabilities ≈ 1.0). Off-by-a-few from
    /// rounding is expected — bound of ±5 is well inside a single LSB per
    /// entry for typical `N ≤ 16` KWS output vectors.
    #[test]
    fn softmax_int8_probability_mass_sums_to_one() {
        let input = [30i8, 10, -10, -30, 50, 5, -5, -50];
        let mut out = [0i8; 8];
        softmax_int8(
            &input,
            &mut out,
            /*input_scale=*/ 0.05,
            /*input_zp=*/ 0,
            /*output_scale=*/ 1.0 / 256.0,
            /*output_zp=*/ -128,
        )
        .unwrap();
        let output_zp = -128i32;
        let sum: i32 = out.iter().map(|&o| o as i32 - output_zp).sum();
        assert!(
            (sum - 256).abs() <= 5,
            "softmax mass sum = {sum}, expected ~256 (±5)"
        );
    }

    /// Empty input is a fail-closed error (not a divide-by-zero panic).
    #[test]
    fn softmax_int8_rejects_empty_input() {
        let input: [i8; 0] = [];
        let mut out: [i8; 0] = [];
        assert!(matches!(
            softmax_int8(&input, &mut out, 0.1, 0, 1.0 / 256.0, -128),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    /// Length mismatch is InvalidArgument.
    #[test]
    fn softmax_int8_rejects_length_mismatch() {
        let input = [0i8; 4];
        let mut out = [0i8; 3];
        assert!(matches!(
            softmax_int8(&input, &mut out, 0.1, 0, 1.0 / 256.0, -128),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    // ---- exp_int8 (private helper) -------------------------------------

    /// Anchor pins for the internal scalar `exp`. `exp(0) = 1` and
    /// `exp(1) = e ≈ 2.71828`; saturation at large ±.
    #[test]
    fn exp_int8_anchors_and_saturation() {
        assert!((exp_int8(0.0) - 1.0).abs() < 1e-6);
        assert!((exp_int8(1.0) - core::f32::consts::E).abs() < 1e-5);
        assert!((exp_int8(-1.0) - (1.0 / core::f32::consts::E)).abs() < 1e-6);
        // Saturation: exp(large positive) = +∞, exp(large negative) = 0.
        assert!(exp_int8(200.0).is_infinite());
        assert_eq!(exp_int8(-200.0), 0.0);
        // NaN passthrough (matches f32::exp).
        assert!(exp_int8(f32::NAN).is_nan());
    }

    #[test]
    fn per_channel_conv_scales_and_fused_relu_are_applied() {
        let dims = ConvDims {
            in_h: 1,
            in_w: 1,
            in_c: 2,
            out_c: 2,
            kh: 1,
            kw: 1,
            stride_h: 1,
            stride_w: 1,
            pad_h: 0,
            pad_w: 0,
        };
        let weight = [1i8, 0, 0, 1];
        let bias = [0i32, 0];
        let mut out = [0i8; 2];
        conv2d_int8_per_channel(
            &[1, -1],
            &weight,
            &bias,
            &mut out,
            0,
            &[1.0, 2.0],
            0,
            false,
            dims,
        )
        .unwrap();
        assert_eq!(out, [1, -2]);
        conv2d_int8_per_channel(
            &[1, -1],
            &weight,
            &bias,
            &mut out,
            0,
            &[1.0, 2.0],
            0,
            true,
            dims,
        )
        .unwrap();
        assert_eq!(out, [1, 0]);
        assert!(conv2d_int8_per_channel(
            &[1, -1],
            &weight,
            &bias,
            &mut out,
            0,
            &[1.0],
            0,
            false,
            dims,
        )
        .is_err());
    }

    #[test]
    fn per_channel_dense_scales_are_independent() {
        let mut out = [0i8; 2];
        fully_connected_int8_per_channel(
            &[3],
            &[1, 1],
            &[0, 0],
            &mut out,
            0,
            &[1.0, 0.5],
            0,
            false,
        )
        .unwrap();
        assert_eq!(out, [3, 2]);
        assert!(fully_connected_int8_per_channel(
            &[3],
            &[1, 1],
            &[0, 0],
            &mut out,
            0,
            &[f32::NAN, 0.5],
            0,
            false,
        )
        .is_err());
    }
}
