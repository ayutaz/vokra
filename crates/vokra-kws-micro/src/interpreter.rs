//! INT8 forward-chain interpreter for microWakeWord (M5-03b Phase 3).
//!
//! # What lives here
//!
//! The Phase 3 piece that turns Phase 1 (log-mel features) + Phase 2 (INT8
//! kernels + GGUF loader) into an actual per-frame classification: a hand-
//! executed chain of INT8 layers ([`LayerSpec`]) with pre-allocated ping-pong
//! scratch buffers ([`ChainConfig`]). The Phase 3 [`crate::KwsMicro::detect`]
//! wires a chain up to the [`crate::features`] front-end and returns
//! [`crate::KwsEvent::Wake`] when a keyword's dequantised probability crosses
//! its [`crate::KeywordDef::threshold`].
//!
//! # Why a chain, not a general op graph
//!
//! [`crate::model::Model`] stores only tensors — it deliberately does not
//! carry an op-graph struct (see the module docs there for the rationale). The
//! microWakeWord (MC-MobileNet) architecture is a fixed conv → dwconv → dense
//! chain with no dynamic branching and no residuals inside a keyword
//! classifier, which matches the whisper.cpp `whisper_encoder` and sister
//! [`vokra_vad_micro`] pattern: hand-write the topology once, keep the
//! runtime a small INT8 interpreter instead of a general graph executor.
//!
//! A [`ChainConfig`] holds:
//!
//! * The ordered list of [`LayerSpec`] variants that make up the classifier.
//! * Two pre-allocated ping-pong scratch buffers ([`Vec<i8>`]) sized to the
//!   largest intermediate stage.
//!
//! [`ChainConfig::run`] walks the chain: read from buf-A, write to buf-B,
//! `core::mem::swap` the two, repeat. The per-layer path performs no heap
//! allocation itself; the caller-visible return is a borrow into the internal
//! ping-pong buffer.
//!
//! # Design red lines
//!
//! * **Zero external deps (NFR-DS-02)** — this module uses only `core`,
//!   `alloc`, and other crate-local modules ([`crate::kernels`],
//!   [`vokra_core::VokraError`]). No `libm`, no `flatbuffers`, no
//!   third-party crates.
//! * **No `unsafe` (NFR-RL-07)** — workspace lint `unsafe_code = "deny"`;
//!   the entire chain runs in safe Rust.
//! * **Fail-closed (FR-EX-08)** — construction rejects any layer-to-layer
//!   size mismatch; runtime rejects wrong-length inputs. Silent misbehaviour
//!   would let a garbage classifier fire without warning.
//!
//! # Honest scope (Phase 3)
//!
//! The chain executor is real: with a synthetic 2-layer test chain the full
//! feature-extract → quantise → INT8 forward → softmax → threshold pipeline
//! runs end-to-end in [`crate::KwsMicro::detect`]. What is NOT yet real:
//!
//! 1. **A committed hey_jarvis chain** — the sidecar
//!    (`tools/parity/microwakeword/prepare_checkpoint.py`) now emits Q8_0
//!    source-byte carriers plus `(scale, zero_point)` metadata. A real
//!    MC-MobileNet chain now has a typed binder, but the authenticated VAST
//!    manifest and parity evidence are still required before real
//!    hey_jarvis inference is claimed.
//! 2. **Accuracy on a real `.tflite`** — needs a canned "hey jarvis" audio
//!    fixture from the owner (see the crate's honest-boundary contract in
//!    [`crate::KwsMicro::detect`]).
//! 3. **Cortex-M55 host-vs-target parity** — thumbv8m cross-build works
//!    (Phase 1 already exercises it) but real-hardware latency is owner-only.

// `alloc` items live in the prelude under `std`; under `#![no_std]` they need
// explicit imports. Same posture as sister modules (`kernels.rs`, `model.rs`).
#[cfg(not(feature = "std"))]
use alloc::{format, vec, vec::Vec};
use core::convert::TryInto;

use vokra_core::{Result, VokraError};

use crate::kernels::{
    conv2d_int8, conv2d_int8_per_channel, depthwise_conv2d_int8, depthwise_conv2d_int8_per_channel,
    fully_connected_int8, fully_connected_int8_per_channel, sigmoid_int8, softmax_int8, ConvDims,
};

/// One layer in a microWakeWord INT8 forward chain.
///
/// Each variant maps 1:1 to one call into [`crate::kernels`], plus the wiring
/// metadata (pre-quantised weights + INT8 quantisation params) needed to
/// hand-execute the chain without a general-purpose op-graph representation.
///
/// # Quantisation contract (TFLite-compatible)
///
/// * Weights are **symmetric** INT8 (`weight_zero_point ≡ 0`, so no field
///   for it) — the TFLite default for signed-int8 weights.
/// * Bias is INT32, **pre-scaled** at export time to
///   `input_scale · weight_scale` with `bias_zero_point ≡ 0` — matches
///   `tflite::reference_ops::Conv` verbatim.
/// * `output_scale` is the **requantisation multiplier**
///   `input_scale · weight_scale / real_output_scale` (a plain `f32` — the
///   deterministic form; the fixed-point `M · 2^-shift` variant can drop in
///   behind the same field when Helium/MVE lands, see [`crate::kernels`]).
#[derive(Debug, Clone)]
pub enum LayerSpec {
    /// Standard `CONV_2D`. Weight layout is `[out_c, kh, kw, in_c]`
    /// row-major (see [`ConvDims`]).
    Conv2d {
        /// Pre-quantised INT8 weight buffer.
        weight_i8: Vec<i8>,
        /// Pre-scaled INT32 bias, length `dims.out_c`.
        bias_i32: Vec<i32>,
        /// Shape / stride / padding.
        dims: ConvDims,
        /// Input activation zero-point.
        input_zero_point: i8,
        /// Output activation zero-point.
        output_zero_point: i8,
        /// Requantisation multiplier (see enum-level docs).
        output_scale: f32,
    },
    /// `DEPTHWISE_CONV_2D` with `depth_multiplier = 1`. Weight layout is
    /// `[1, kh, kw, in_c]`; effective `out_c` equals `dims.in_c` (`dims.out_c`
    /// is ignored — matches [`crate::kernels::depthwise_conv2d_int8`]).
    DepthwiseConv2d {
        /// Pre-quantised INT8 weight buffer.
        weight_i8: Vec<i8>,
        /// Pre-scaled INT32 bias, length `dims.in_c`.
        bias_i32: Vec<i32>,
        /// Shape / stride / padding.
        dims: ConvDims,
        /// Input activation zero-point.
        input_zero_point: i8,
        /// Output activation zero-point.
        output_zero_point: i8,
        /// Requantisation multiplier.
        output_scale: f32,
    },
    /// `FULLY_CONNECTED` (dense). Weight layout is `[out_dim, in_dim]`
    /// row-major.
    FullyConnected {
        /// Pre-quantised INT8 weight buffer, length `out_dim · in_dim`.
        weight_i8: Vec<i8>,
        /// Pre-scaled INT32 bias, length `out_dim`.
        bias_i32: Vec<i32>,
        /// Input vector length.
        in_dim: usize,
        /// Output vector length.
        out_dim: usize,
        /// Input activation zero-point.
        input_zero_point: i8,
        /// Output activation zero-point.
        output_zero_point: i8,
        /// Requantisation multiplier.
        output_scale: f32,
    },
    /// Standard convolution with per-output-channel quantisation and an
    /// optional TFLite fused RELU clamp.
    Conv2dPerChannel {
        /// Pre-quantised INT8 weight buffer.
        weight_i8: Vec<i8>,
        /// Pre-scaled INT32 bias, one value per output channel.
        bias_i32: Vec<i32>,
        /// Shape / stride / padding.
        dims: ConvDims,
        /// Input activation zero-point.
        input_zero_point: i8,
        /// Candidate float requantisation scales, one per output channel.
        output_scales: Vec<f32>,
        /// Per-tensor output activation zero-point.
        output_zero_point: i8,
        /// Apply the quantised TFLite RELU lower clamp.
        fused_relu: bool,
    },
    /// Depthwise convolution with per-output-channel quantisation and an
    /// optional TFLite fused RELU clamp.
    DepthwiseConv2dPerChannel {
        /// Pre-quantised INT8 weight buffer.
        weight_i8: Vec<i8>,
        /// Pre-scaled INT32 bias, one value per channel.
        bias_i32: Vec<i32>,
        /// Shape / stride / padding.
        dims: ConvDims,
        /// Input activation zero-point.
        input_zero_point: i8,
        /// Candidate float requantisation scales, one per channel.
        output_scales: Vec<f32>,
        /// Per-tensor output activation zero-point.
        output_zero_point: i8,
        /// Apply the quantised TFLite RELU lower clamp.
        fused_relu: bool,
    },
    /// Fully-connected layer with per-output-channel quantisation and an
    /// optional TFLite fused RELU clamp.
    FullyConnectedPerChannel {
        /// Pre-quantised INT8 weight buffer.
        weight_i8: Vec<i8>,
        /// Pre-scaled INT32 bias, one value per output.
        bias_i32: Vec<i32>,
        /// Input vector length.
        in_dim: usize,
        /// Output vector length.
        out_dim: usize,
        /// Input activation zero-point.
        input_zero_point: i8,
        /// Candidate float requantisation scales, one per output.
        output_scales: Vec<f32>,
        /// Per-tensor output activation zero-point.
        output_zero_point: i8,
        /// Apply the quantised TFLite RELU lower clamp.
        fused_relu: bool,
    },
    /// Elementwise `LOGISTIC` (sigmoid). Preserves buffer size.
    Sigmoid {
        /// Number of elements (input and output length).
        size: usize,
        /// Input dequantisation scale.
        input_scale: f32,
        /// Input dequantisation zero-point.
        input_zero_point: i8,
        /// Output requantisation scale.
        output_scale: f32,
        /// Output requantisation zero-point.
        output_zero_point: i8,
    },
    /// `SOFTMAX` over the whole input (single row). Preserves buffer size.
    Softmax {
        /// Number of elements (input and output length).
        size: usize,
        /// Input dequantisation scale.
        input_scale: f32,
        /// Input dequantisation zero-point.
        input_zero_point: i8,
        /// Output requantisation scale (TFLite convention: `1/256`).
        output_scale: f32,
        /// Output requantisation zero-point (TFLite convention: `-128`).
        output_zero_point: i8,
    },
}

impl LayerSpec {
    /// Number of INT8 elements this layer reads.
    pub fn input_size(&self) -> usize {
        match self {
            LayerSpec::Conv2d { dims, .. }
            | LayerSpec::DepthwiseConv2d { dims, .. }
            | LayerSpec::Conv2dPerChannel { dims, .. }
            | LayerSpec::DepthwiseConv2dPerChannel { dims, .. } => {
                dims.in_h * dims.in_w * dims.in_c
            }
            LayerSpec::FullyConnected { in_dim, .. }
            | LayerSpec::FullyConnectedPerChannel { in_dim, .. } => *in_dim,
            LayerSpec::Sigmoid { size, .. } | LayerSpec::Softmax { size, .. } => *size,
        }
    }

    /// Number of INT8 elements this layer writes.
    ///
    /// Fails with [`VokraError::InvalidArgument`] if the layer's declared
    /// dims would produce a negative-extent output (an ill-formed spec that
    /// [`ChainConfig::new`] rejects at construction — this method exists so
    /// construction can perform that check).
    pub fn output_size(&self) -> Result<usize> {
        match self {
            LayerSpec::Conv2d { dims, .. } | LayerSpec::Conv2dPerChannel { dims, .. } => {
                let oh = dims.out_h().ok_or_else(|| {
                    VokraError::InvalidArgument(format!(
                        "Conv2d: invalid dims (in_h={}, kh={}, pad_h={}, stride_h={})",
                        dims.in_h, dims.kh, dims.pad_h, dims.stride_h
                    ))
                })?;
                let ow = dims.out_w().ok_or_else(|| {
                    VokraError::InvalidArgument(format!(
                        "Conv2d: invalid dims (in_w={}, kw={}, pad_w={}, stride_w={})",
                        dims.in_w, dims.kw, dims.pad_w, dims.stride_w
                    ))
                })?;
                Ok(oh * ow * dims.out_c)
            }
            LayerSpec::DepthwiseConv2d { dims, .. }
            | LayerSpec::DepthwiseConv2dPerChannel { dims, .. } => {
                let oh = dims.out_h().ok_or_else(|| {
                    VokraError::InvalidArgument(format!(
                        "DepthwiseConv2d: invalid dims (in_h={}, kh={}, pad_h={}, stride_h={})",
                        dims.in_h, dims.kh, dims.pad_h, dims.stride_h
                    ))
                })?;
                let ow = dims.out_w().ok_or_else(|| {
                    VokraError::InvalidArgument(format!(
                        "DepthwiseConv2d: invalid dims (in_w={}, kw={}, pad_w={}, stride_w={})",
                        dims.in_w, dims.kw, dims.pad_w, dims.stride_w
                    ))
                })?;
                Ok(oh * ow * dims.in_c)
            }
            LayerSpec::FullyConnected { out_dim, .. }
            | LayerSpec::FullyConnectedPerChannel { out_dim, .. } => Ok(*out_dim),
            LayerSpec::Sigmoid { size, .. } | LayerSpec::Softmax { size, .. } => Ok(*size),
        }
    }

    /// Applies this layer: reads `input`, writes into `output`.
    ///
    /// Private helper for [`ChainConfig::run`]; the chain runner is
    /// responsible for sizing `output` to [`Self::output_size`].
    fn apply(&self, input: &[i8], output: &mut [i8]) -> Result<()> {
        match self {
            LayerSpec::Conv2d {
                weight_i8,
                bias_i32,
                dims,
                input_zero_point,
                output_zero_point,
                output_scale,
            } => conv2d_int8(
                input,
                weight_i8,
                bias_i32,
                output,
                *input_zero_point,
                *output_zero_point,
                *output_scale,
                *dims,
            ),
            LayerSpec::DepthwiseConv2d {
                weight_i8,
                bias_i32,
                dims,
                input_zero_point,
                output_zero_point,
                output_scale,
            } => depthwise_conv2d_int8(
                input,
                weight_i8,
                bias_i32,
                output,
                *input_zero_point,
                *output_zero_point,
                *output_scale,
                *dims,
            ),
            LayerSpec::FullyConnected {
                weight_i8,
                bias_i32,
                input_zero_point,
                output_zero_point,
                output_scale,
                ..
            } => fully_connected_int8(
                input,
                weight_i8,
                bias_i32,
                output,
                *input_zero_point,
                *output_zero_point,
                *output_scale,
            ),
            LayerSpec::Conv2dPerChannel {
                weight_i8,
                bias_i32,
                dims,
                input_zero_point,
                output_scales,
                output_zero_point,
                fused_relu,
            } => conv2d_int8_per_channel(
                input,
                weight_i8,
                bias_i32,
                output,
                *input_zero_point,
                output_scales,
                *output_zero_point,
                *fused_relu,
                *dims,
            ),
            LayerSpec::DepthwiseConv2dPerChannel {
                weight_i8,
                bias_i32,
                dims,
                input_zero_point,
                output_scales,
                output_zero_point,
                fused_relu,
            } => depthwise_conv2d_int8_per_channel(
                input,
                weight_i8,
                bias_i32,
                output,
                *input_zero_point,
                output_scales,
                *output_zero_point,
                *fused_relu,
                *dims,
            ),
            LayerSpec::FullyConnectedPerChannel {
                weight_i8,
                bias_i32,
                input_zero_point,
                output_scales,
                output_zero_point,
                fused_relu,
                ..
            } => fully_connected_int8_per_channel(
                input,
                weight_i8,
                bias_i32,
                output,
                *input_zero_point,
                output_scales,
                *output_zero_point,
                *fused_relu,
            ),
            LayerSpec::Sigmoid {
                input_scale,
                input_zero_point,
                output_scale,
                output_zero_point,
                ..
            } => sigmoid_int8(
                input,
                output,
                *input_scale,
                *input_zero_point,
                *output_scale,
                *output_zero_point,
            ),
            LayerSpec::Softmax {
                input_scale,
                input_zero_point,
                output_scale,
                output_zero_point,
                ..
            } => softmax_int8(
                input,
                output,
                *input_scale,
                *input_zero_point,
                *output_scale,
                *output_zero_point,
            ),
        }
    }
}

/// A validated, pre-buffered INT8 forward chain.
///
/// Construction:
///
/// 1. Validates the layer list is non-empty.
/// 2. Walks consecutive layers checking `layers[i].output_size ==
///    layers[i+1].input_size` (fail-closed on any mismatch — FR-EX-08).
/// 3. Pre-allocates two ping-pong [`Vec<i8>`] buffers sized to the maximum
///    stage encountered (input, or any layer's output).
///
/// [`Self::run`] mem-swaps the two buffers after each layer, so the per-layer
/// hot path performs no allocation of its own. (The [`crate::kernels`] sigmoid
/// / softmax kernels have their own bounded numeric scratch paths; this is not
/// a chain-executor concern.)
#[derive(Debug)]
pub struct ChainConfig {
    /// The layer chain in forward order.
    layers: Vec<LayerSpec>,
    /// Cached first-layer input size (== `layers[0].input_size()`).
    input_size: usize,
    /// Cached last-layer output size (== `layers.last().output_size()`).
    output_size: usize,
    /// Ping-pong buffer A. After each layer executes, this holds the newest
    /// output (the `run` loop swaps A and B at the end of each iteration).
    /// Sized to the max intermediate stage.
    buf_a: Vec<i8>,
    /// Ping-pong buffer B. Same size as `buf_a`; alternates as the write
    /// target during `run`.
    buf_b: Vec<i8>,
}

impl ChainConfig {
    /// Builds and validates a chain from an ordered `layers` list.
    ///
    /// # Errors
    ///
    /// Returns [`VokraError::InvalidArgument`] on:
    /// - an empty `layers` list;
    /// - any layer whose declared dims imply a non-positive output extent
    ///   (see [`LayerSpec::output_size`]);
    /// - any layer whose input size does not match the previous layer's
    ///   output size (the entire chain must be pre-composable at construction
    ///   — silent buffer-size mismatches would corrupt inference).
    pub fn new(layers: Vec<LayerSpec>) -> Result<Self> {
        if layers.is_empty() {
            return Err(VokraError::InvalidArgument(
                "ChainConfig::new: empty layer list".into(),
            ));
        }
        let input_size = layers[0].input_size();
        let mut prev_out = input_size;
        let mut max_stage = input_size;
        for (idx, layer) in layers.iter().enumerate() {
            if layer.input_size() != prev_out {
                return Err(VokraError::InvalidArgument(format!(
                    "ChainConfig: layer {idx} input size {} != previous output size {}",
                    layer.input_size(),
                    prev_out,
                )));
            }
            let out = layer.output_size()?;
            if out > max_stage {
                max_stage = out;
            }
            prev_out = out;
        }
        let output_size = prev_out;
        Ok(Self {
            layers,
            input_size,
            output_size,
            buf_a: vec![0i8; max_stage],
            buf_b: vec![0i8; max_stage],
        })
    }

    /// Runs the chain against `input` and returns a borrow into the internal
    /// ping-pong buffer holding the final output.
    ///
    /// The returned slice has length [`Self::output_size`]; it borrows the
    /// chain mutably for its entire lifetime (subsequent [`Self::run`] calls
    /// would invalidate it). Callers that need to keep the output past the
    /// next `run` must copy it (e.g. via `.to_vec()`).
    ///
    /// # Errors
    ///
    /// Returns [`VokraError::InvalidArgument`] on wrong input length (fail-
    /// closed — FR-EX-08). Kernel-level errors from
    /// [`crate::kernels`] (which should never fire here since construction
    /// has already validated every layer's buffer sizes) propagate.
    pub fn run(&mut self, input: &[i8]) -> Result<&[i8]> {
        if input.len() != self.input_size {
            return Err(VokraError::InvalidArgument(format!(
                "ChainConfig::run: input len {} != expected {}",
                input.len(),
                self.input_size,
            )));
        }
        // Seed buf_a with the input. curr_len tracks the "live" prefix of
        // buf_a (which shrinks or grows as layers change dimensionality).
        self.buf_a[..input.len()].copy_from_slice(input);
        let mut curr_len = input.len();
        for layer in &self.layers {
            // Layer construction already validated `output_size` succeeds; we
            // re-derive it here rather than caching per-layer because
            // `output_size` for pure passthrough (sigmoid/softmax) is trivial.
            let out_len = layer.output_size()?;
            layer.apply(&self.buf_a[..curr_len], &mut self.buf_b[..out_len])?;
            // Swap so buf_a now holds the newest output; buf_b becomes stale
            // scratch for the next iteration. Cheap: swaps two `Vec` headers,
            // no data copy.
            core::mem::swap(&mut self.buf_a, &mut self.buf_b);
            curr_len = out_len;
        }
        // curr_len == self.output_size by construction (validated in `new`).
        Ok(&self.buf_a[..self.output_size])
    }

    /// Expected input size (in INT8 elements).
    pub fn input_size(&self) -> usize {
        self.input_size
    }

    /// Final output size (in INT8 elements).
    pub fn output_size(&self) -> usize {
        self.output_size
    }

    /// Number of layers in the chain.
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }
}

/// Fixed authenticated hey_jarvis streaming input shape (`rows × channels`).
pub const HEY_JARVIS_INPUT_SIZE: usize = 3 * 40;
/// Number of persistent rolling activation states in the hey_jarvis graph.
pub const HEY_JARVIS_STATE_COUNT: usize = 6;
/// Quantised real zero used to initialise the authenticated rolling states.
pub const HEY_JARVIS_STATE_QUANTIZED_ZERO: i8 = -128;
const HEY_JARVIS_STATE_LENGTHS: [usize; HEY_JARVIS_STATE_COUNT] =
    [2 * 40, 4 * 30, 8 * 60, 12 * 60, 20 * 60, 4 * 60];
const HEY_JARVIS_CONCAT_LENGTHS: [usize; HEY_JARVIS_STATE_COUNT] =
    [5 * 40, 5 * 30, 9 * 60, 13 * 60, 21 * 60, 5 * 60];
const HEY_JARVIS_LAYER_COUNT: usize = 11;

/// A validated fixed topology for the authenticated hey_jarvis evidence.
///
/// The plan deliberately accepts only the eleven known operators. It is not
/// a general graph executor and has no CALL_ONCE/VAR_HANDLE interpretation.
#[derive(Debug)]
pub struct HeyJarvisStreamingPlan {
    layers: Vec<LayerSpec>,
}

impl HeyJarvisStreamingPlan {
    /// Validates and stores the fixed conv/dwconv/pointwise/dense/logistic
    /// topology. The layer list must contain exactly eleven layers.
    pub fn new(layers: Vec<LayerSpec>) -> Result<Self> {
        if layers.len() != HEY_JARVIS_LAYER_COUNT {
            return Err(VokraError::InvalidArgument(format!(
                "HeyJarvisStreamingPlan: expected {HEY_JARVIS_LAYER_COUNT} layers, got {}",
                layers.len()
            )));
        }
        validate_per_channel_conv(
            &layers[0],
            ConvDims {
                in_h: 5,
                in_w: 1,
                in_c: 40,
                out_c: 30,
                kh: 5,
                kw: 1,
                stride_h: 3,
                stride_w: 1,
                pad_h: 0,
                pad_w: 0,
            },
            false,
            -128,
            -128,
            true,
        )?;
        validate_per_channel_conv(
            &layers[1],
            ConvDims {
                in_h: 5,
                in_w: 1,
                in_c: 30,
                out_c: 30,
                kh: 5,
                kw: 1,
                stride_h: 1,
                stride_w: 1,
                pad_h: 0,
                pad_w: 0,
            },
            true,
            -128,
            -11,
            false,
        )?;
        validate_per_channel_conv(
            &layers[2],
            ConvDims {
                in_h: 1,
                in_w: 1,
                in_c: 30,
                out_c: 60,
                kh: 1,
                kw: 1,
                stride_h: 1,
                stride_w: 1,
                pad_h: 0,
                pad_w: 0,
            },
            false,
            -11,
            -128,
            true,
        )?;
        validate_per_channel_conv(
            &layers[3],
            ConvDims {
                in_h: 9,
                in_w: 1,
                in_c: 60,
                out_c: 60,
                kh: 9,
                kw: 1,
                stride_h: 1,
                stride_w: 1,
                pad_h: 0,
                pad_w: 0,
            },
            true,
            -128,
            31,
            false,
        )?;
        validate_per_channel_conv(
            &layers[4],
            ConvDims {
                in_h: 1,
                in_w: 1,
                in_c: 60,
                out_c: 60,
                kh: 1,
                kw: 1,
                stride_h: 1,
                stride_w: 1,
                pad_h: 0,
                pad_w: 0,
            },
            false,
            31,
            -128,
            true,
        )?;
        validate_per_channel_conv(
            &layers[5],
            ConvDims {
                in_h: 13,
                in_w: 1,
                in_c: 60,
                out_c: 60,
                kh: 13,
                kw: 1,
                stride_h: 1,
                stride_w: 1,
                pad_h: 0,
                pad_w: 0,
            },
            true,
            -128,
            1,
            false,
        )?;
        validate_per_channel_conv(
            &layers[6],
            ConvDims {
                in_h: 1,
                in_w: 1,
                in_c: 60,
                out_c: 60,
                kh: 1,
                kw: 1,
                stride_h: 1,
                stride_w: 1,
                pad_h: 0,
                pad_w: 0,
            },
            false,
            1,
            -128,
            true,
        )?;
        validate_per_channel_conv(
            &layers[7],
            ConvDims {
                in_h: 21,
                in_w: 1,
                in_c: 60,
                out_c: 60,
                kh: 21,
                kw: 1,
                stride_h: 1,
                stride_w: 1,
                pad_h: 0,
                pad_w: 0,
            },
            true,
            -128,
            -36,
            false,
        )?;
        validate_per_channel_conv(
            &layers[8],
            ConvDims {
                in_h: 1,
                in_w: 1,
                in_c: 60,
                out_c: 60,
                kh: 1,
                kw: 1,
                stride_h: 1,
                stride_w: 1,
                pad_h: 0,
                pad_w: 0,
            },
            false,
            -36,
            -128,
            true,
        )?;
        validate_per_channel_dense(&layers[9], 5 * 60, 1, -128, 30, false)?;
        match &layers[10] {
            LayerSpec::Sigmoid {
                size: 1,
                input_scale,
                input_zero_point: 30,
                output_scale,
                output_zero_point: -128,
            } if input_scale.is_finite()
                && *input_scale > 0.0
                && *input_scale == 0.17895515263080597
                && output_scale.is_finite()
                && *output_scale > 0.0
                && *output_scale == 0.00390625 => {}
            _ => {
                return Err(VokraError::InvalidArgument(
                    "HeyJarvisStreamingPlan: final layer must be finite Sigmoid(size=1) with input zp=30 and output zp=-128".into(),
                ));
            }
        }
        Ok(Self { layers })
    }

    /// Number of fixed operators (always eleven for a valid plan).
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }
}

fn validate_per_channel_conv(
    layer: &LayerSpec,
    expected: ConvDims,
    depthwise: bool,
    expected_input_zero_point: i8,
    expected_output_zero_point: i8,
    fused_relu: bool,
) -> Result<()> {
    let (dims, weight_len, bias_len, input_zero_point, scales, output_zero_point, actual_relu) =
        match (depthwise, layer) {
            (
                false,
                LayerSpec::Conv2dPerChannel {
                    dims,
                    weight_i8,
                    bias_i32,
                    input_zero_point,
                    output_scales,
                    output_zero_point,
                    fused_relu,
                    ..
                },
            ) => (
                *dims,
                weight_i8.len(),
                bias_i32.len(),
                *input_zero_point,
                output_scales,
                *output_zero_point,
                *fused_relu,
            ),
            (
                true,
                LayerSpec::DepthwiseConv2dPerChannel {
                    dims,
                    weight_i8,
                    bias_i32,
                    input_zero_point,
                    output_scales,
                    output_zero_point,
                    fused_relu,
                    ..
                },
            ) => (
                *dims,
                weight_i8.len(),
                bias_i32.len(),
                *input_zero_point,
                output_scales,
                *output_zero_point,
                *fused_relu,
            ),
            _ => {
                return Err(VokraError::InvalidArgument(
                    "HeyJarvisStreamingPlan: expected per-channel convolution layer".into(),
                ));
            }
        };
    if dims.in_h != expected.in_h
        || dims.in_w != expected.in_w
        || dims.in_c != expected.in_c
        || dims.out_c != expected.out_c
        || dims.kh != expected.kh
        || dims.kw != expected.kw
        || dims.stride_h != expected.stride_h
        || dims.stride_w != expected.stride_w
        || dims.pad_h != expected.pad_h
        || dims.pad_w != expected.pad_w
        || input_zero_point != expected_input_zero_point
        || output_zero_point != expected_output_zero_point
        || actual_relu != fused_relu
    {
        return Err(VokraError::InvalidArgument(
            "HeyJarvisStreamingPlan: convolution shape or fused RELU drift".into(),
        ));
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
    .ok_or_else(|| {
        VokraError::InvalidArgument("HeyJarvisStreamingPlan: weight size overflow".into())
    })?;
    if weight_len != expected_weight || bias_len != effective_out_c {
        return Err(VokraError::InvalidArgument(
            "HeyJarvisStreamingPlan: convolution buffer length mismatch".into(),
        ));
    }
    validate_quant_values(scales, effective_out_c)
}

fn validate_per_channel_dense(
    layer: &LayerSpec,
    in_dim: usize,
    out_dim: usize,
    expected_input_zero_point: i8,
    expected_output_zero_point: i8,
    fused_relu: bool,
) -> Result<()> {
    let (
        actual_in,
        actual_out,
        weight_len,
        bias_len,
        input_zero_point,
        scales,
        output_zero_point,
        actual_relu,
    ) = match layer {
        LayerSpec::FullyConnectedPerChannel {
            in_dim,
            out_dim,
            weight_i8,
            bias_i32,
            input_zero_point,
            output_scales,
            output_zero_point,
            fused_relu,
            ..
        } => (
            *in_dim,
            *out_dim,
            weight_i8.len(),
            bias_i32.len(),
            *input_zero_point,
            output_scales,
            *output_zero_point,
            *fused_relu,
        ),
        _ => {
            return Err(VokraError::InvalidArgument(
                "HeyJarvisStreamingPlan: expected per-channel dense layer".into(),
            ));
        }
    };
    let expected_weight = in_dim.checked_mul(out_dim).ok_or_else(|| {
        VokraError::InvalidArgument("HeyJarvisStreamingPlan: dense size overflow".into())
    })?;
    if actual_in != in_dim
        || actual_out != out_dim
        || weight_len != expected_weight
        || bias_len != out_dim
        || input_zero_point != expected_input_zero_point
        || output_zero_point != expected_output_zero_point
        || actual_relu != fused_relu
    {
        return Err(VokraError::InvalidArgument(
            "HeyJarvisStreamingPlan: dense shape or fused RELU drift".into(),
        ));
    }
    validate_quant_values(scales, out_dim)
}

fn validate_quant_values(scales: &[f32], channels: usize) -> Result<()> {
    if scales.len() != channels {
        return Err(VokraError::InvalidArgument(format!(
            "HeyJarvisStreamingPlan: expected {channels} output quantization scales, got {}",
            scales.len()
        )));
    }
    if scales
        .iter()
        .any(|scale| !scale.is_finite() || *scale <= 0.0)
    {
        return Err(VokraError::InvalidArgument(
            "HeyJarvisStreamingPlan: output scales must be finite and > 0".into(),
        ));
    }
    Ok(())
}

/// A preallocated executor for the fixed hey_jarvis streaming graph.
#[derive(Debug)]
pub struct HeyJarvisStreamingExecutor {
    plan: HeyJarvisStreamingPlan,
    states: [Vec<i8>; HEY_JARVIS_STATE_COUNT],
    concat: [Vec<i8>; HEY_JARVIS_STATE_COUNT],
    work_a: Vec<i8>,
    work_b: Vec<i8>,
}

impl HeyJarvisStreamingExecutor {
    /// Creates a fresh executor with all six rolling states set to quantised
    /// real zero (`-128`).
    pub fn new(plan: HeyJarvisStreamingPlan) -> Result<Self> {
        let states = HEY_JARVIS_STATE_LENGTHS.map(|len| vec![HEY_JARVIS_STATE_QUANTIZED_ZERO; len]);
        Self::from_states(plan, states.to_vec())
    }

    /// Restores explicitly supplied rolling states. Exactly six vectors with
    /// the authenticated lengths are required; this API is intentionally not
    /// a generic resource-variable mechanism.
    pub fn from_states(plan: HeyJarvisStreamingPlan, states: Vec<Vec<i8>>) -> Result<Self> {
        if states.len() != HEY_JARVIS_STATE_COUNT {
            return Err(VokraError::InvalidArgument(format!(
                "HeyJarvisStreamingExecutor: expected {HEY_JARVIS_STATE_COUNT} states, got {}",
                states.len()
            )));
        }
        for (idx, (state, &expected)) in states
            .iter()
            .zip(HEY_JARVIS_STATE_LENGTHS.iter())
            .enumerate()
        {
            if state.len() != expected {
                return Err(VokraError::InvalidArgument(format!(
                    "HeyJarvisStreamingExecutor: state {idx} length {} != expected {expected}",
                    state.len()
                )));
            }
        }
        let states: [Vec<i8>; HEY_JARVIS_STATE_COUNT] = states.try_into().map_err(|_| {
            VokraError::InvalidArgument("HeyJarvisStreamingExecutor: state count drift".into())
        })?;
        Ok(Self {
            plan,
            states,
            concat: HEY_JARVIS_CONCAT_LENGTHS.map(|len| vec![0i8; len]),
            work_a: vec![0i8; 300],
            work_b: vec![0i8; 300],
        })
    }

    /// Resets every rolling state to quantised real zero.
    pub fn reset(&mut self) {
        for state in &mut self.states {
            state.fill(HEY_JARVIS_STATE_QUANTIZED_ZERO);
        }
    }

    /// Executes one 3×40 feature window and returns the quantised probability.
    pub fn run(&mut self, input: &[i8]) -> Result<&[i8]> {
        if input.len() != HEY_JARVIS_INPUT_SIZE {
            return Err(VokraError::InvalidArgument(format!(
                "HeyJarvisStreamingExecutor::run: input len {} != expected {HEY_JARVIS_INPUT_SIZE}",
                input.len()
            )));
        }
        let mut current_len = HEY_JARVIS_INPUT_SIZE;
        let mut current_slot: Option<u8> = None;
        let mut layer_idx = 0usize;
        // States 0..=4 back the five convolutional stages. State 5 is the
        // final dense input history and is consumed below.
        for state_idx in 0..(HEY_JARVIS_STATE_COUNT - 1) {
            let state_len = HEY_JARVIS_STATE_LENGTHS[state_idx];
            let concat_len = HEY_JARVIS_CONCAT_LENGTHS[state_idx];
            self.concat[state_idx][..state_len].copy_from_slice(&self.states[state_idx]);
            match current_slot {
                None => self.concat[state_idx][state_len..concat_len].copy_from_slice(input),
                Some(0) => self.concat[state_idx][state_len..concat_len]
                    .copy_from_slice(&self.work_a[..current_len]),
                Some(1) => self.concat[state_idx][state_len..concat_len]
                    .copy_from_slice(&self.work_b[..current_len]),
                _ => unreachable!(),
            }
            self.states[state_idx]
                .copy_from_slice(&self.concat[state_idx][concat_len - state_len..concat_len]);

            let target_slot = if current_slot == Some(0) { 1 } else { 0 };
            let out_len = self.plan.layers[layer_idx].output_size()?;
            let target = if target_slot == 0 {
                &mut self.work_a[..out_len]
            } else {
                &mut self.work_b[..out_len]
            };
            self.plan.layers[layer_idx].apply(&self.concat[state_idx][..concat_len], target)?;
            current_len = out_len;
            current_slot = Some(target_slot);
            layer_idx += 1;
            if state_idx > 0 {
                // Every later state backs one depthwise convolution. Its
                // following pointwise convolution has no additional history.
                let op_out = self.plan.layers[layer_idx].output_size()?;
                let op_target_slot = if current_slot == Some(0) { 1 } else { 0 };
                if current_slot == Some(0) {
                    self.plan.layers[layer_idx]
                        .apply(&self.work_a[..current_len], &mut self.work_b[..op_out])?;
                } else {
                    self.plan.layers[layer_idx]
                        .apply(&self.work_b[..current_len], &mut self.work_a[..op_out])?;
                }
                current_len = op_out;
                current_slot = Some(op_target_slot);
                layer_idx += 1;
            }
        }
        // The six rolling stages consume layers 0..=8. Layer 9 consumes the
        // final 5×60 concat; layer 10 applies sigmoid to the dense scalar.
        let final_state_idx = HEY_JARVIS_STATE_COUNT - 1;
        let state_len = HEY_JARVIS_STATE_LENGTHS[final_state_idx];
        let concat_len = HEY_JARVIS_CONCAT_LENGTHS[final_state_idx];
        self.concat[final_state_idx][..state_len].copy_from_slice(&self.states[final_state_idx]);
        let current = if current_slot == Some(0) {
            &self.work_a[..current_len]
        } else {
            &self.work_b[..current_len]
        };
        self.concat[final_state_idx][state_len..concat_len].copy_from_slice(current);
        self.states[final_state_idx]
            .copy_from_slice(&self.concat[final_state_idx][concat_len - state_len..concat_len]);
        self.plan.layers[9].apply(
            &self.concat[final_state_idx][..concat_len],
            &mut self.work_a[..1],
        )?;
        self.plan.layers[10].apply(&self.work_a[..1], &mut self.work_b[..1])?;
        Ok(&self.work_b[..1])
    }

    /// Number of rolling states (always six).
    pub fn state_count(&self) -> usize {
        self.states.len()
    }

    /// Returns a read-only rolling-state snapshot for diagnostics and a
    /// checkpointing caller. Invalid indices are rejected explicitly.
    pub fn state(&self, index: usize) -> Result<&[i8]> {
        self.states.get(index).map(Vec::as_slice).ok_or_else(|| {
            VokraError::InvalidArgument(format!(
                "HeyJarvisStreamingExecutor: state index {index} out of range"
            ))
        })
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// A 1×1 identity conv (`out_c == in_c`, weights are the identity matrix
    /// under the `[out_c, 1, 1, in_c]` layout): output equals input. Reused
    /// across chain-composition tests as a "pass-through" primitive whose
    /// correctness is proven by [`single_identity_conv_reproduces_input`].
    fn identity_conv(in_c: usize) -> LayerSpec {
        let dims = ConvDims {
            in_h: 1,
            in_w: 1,
            in_c,
            out_c: in_c,
            kh: 1,
            kw: 1,
            stride_h: 1,
            stride_w: 1,
            pad_h: 0,
            pad_w: 0,
        };
        // `[out_c, kh, kw, in_c] = [in_c, 1, 1, in_c]` — flattened,
        // weight[oc * in_c + ic] with `oc == ic` == 1, else 0.
        let mut weight = vec![0i8; in_c * in_c];
        for oc in 0..in_c {
            weight[oc * in_c + oc] = 1;
        }
        LayerSpec::Conv2d {
            weight_i8: weight,
            bias_i32: vec![0i32; in_c],
            dims,
            input_zero_point: 0,
            output_zero_point: 0,
            output_scale: 1.0,
        }
    }

    #[test]
    fn single_identity_conv_reproduces_input() {
        let mut chain = ChainConfig::new(vec![identity_conv(3)]).unwrap();
        let out = chain.run(&[5i8, -10, 20]).unwrap().to_vec();
        assert_eq!(out, vec![5, -10, 20]);
        // Introspection surface is correct.
        assert_eq!(chain.input_size(), 3);
        assert_eq!(chain.output_size(), 3);
        assert_eq!(chain.layer_count(), 1);
    }

    #[test]
    fn two_layer_chain_composes() {
        // Layer 1: identity conv (in=3, out=3) → pass-through.
        // Layer 2: fully-connected (in=3, out=2) with all-ones weight, no
        // bias → sums the three inputs to both outputs.
        let fc = LayerSpec::FullyConnected {
            weight_i8: vec![1i8, 1, 1, 1, 1, 1],
            bias_i32: vec![0i32, 0],
            in_dim: 3,
            out_dim: 2,
            input_zero_point: 0,
            output_zero_point: 0,
            output_scale: 1.0,
        };
        let mut chain = ChainConfig::new(vec![identity_conv(3), fc]).unwrap();
        let out = chain.run(&[1i8, 2, 3]).unwrap().to_vec();
        // Layer 1 → [1, 2, 3]; Layer 2 → [1+2+3, 1+2+3] = [6, 6].
        assert_eq!(out, vec![6, 6]);
    }

    #[test]
    fn chain_rejects_size_mismatch_between_layers() {
        // First layer outputs 3 (identity_conv on in_c=3); second expects 5.
        let bad_fc = LayerSpec::FullyConnected {
            weight_i8: vec![0i8; 5 * 2],
            bias_i32: vec![0i32; 2],
            in_dim: 5,
            out_dim: 2,
            input_zero_point: 0,
            output_zero_point: 0,
            output_scale: 1.0,
        };
        let err = ChainConfig::new(vec![identity_conv(3), bad_fc]).unwrap_err();
        match err {
            VokraError::InvalidArgument(m) => {
                assert!(m.contains("layer 1"), "message names offending index: {m}");
                assert!(m.contains("5"), "message mentions expected size: {m}");
                assert!(m.contains("3"), "message mentions actual size: {m}");
            }
            other => panic!("expected InvalidArgument for size mismatch, got {other:?}"),
        }
    }

    #[test]
    fn chain_rejects_wrong_input_length() {
        let mut chain = ChainConfig::new(vec![identity_conv(3)]).unwrap();
        let err = chain.run(&[1i8, 2]).unwrap_err(); // 2 elements, expected 3
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn chain_rejects_empty_layer_list() {
        assert!(matches!(
            ChainConfig::new(vec![]),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn chain_reused_across_calls_preserves_correctness() {
        // Repeated `run` calls must not leak state between them: buf_a/buf_b
        // are ping-ponged, and their contents from a previous call could
        // shadow into the next if the write-length bookkeeping were wrong.
        let mut chain = ChainConfig::new(vec![identity_conv(4)]).unwrap();
        assert_eq!(
            chain.run(&[1i8, 2, 3, 4]).unwrap().to_vec(),
            vec![1, 2, 3, 4]
        );
        assert_eq!(
            chain.run(&[5i8, 6, 7, 8]).unwrap().to_vec(),
            vec![5, 6, 7, 8]
        );
        // Distinct-sign pattern verifies no wrap-around bookkeeping bug in
        // the ping-pong seeding.
        assert_eq!(
            chain.run(&[-1i8, -2, -3, -4]).unwrap().to_vec(),
            vec![-1, -2, -3, -4]
        );
    }

    #[test]
    fn chain_with_softmax_final_produces_normalised_i8() {
        // FC(3 → 2, picks logits from input[0] & input[1] via one-hot weights)
        // then Softmax(2 classes). Equal input logits → uniform softmax →
        // each class ≈ 0.5 → quantised i8 near output_zp midpoint.
        let fc = LayerSpec::FullyConnected {
            weight_i8: vec![1i8, 0, 0, 0, 1, 0],
            bias_i32: vec![0i32, 0],
            in_dim: 3,
            out_dim: 2,
            input_zero_point: 0,
            output_zero_point: 0,
            output_scale: 1.0,
        };
        let softmax = LayerSpec::Softmax {
            size: 2,
            input_scale: 0.1,
            input_zero_point: 0,
            output_scale: 1.0 / 256.0,
            output_zero_point: -128,
        };
        let mut chain = ChainConfig::new(vec![fc, softmax]).unwrap();
        // input[0] and input[1] equal → equal logits at FC output.
        let out = chain.run(&[10i8, 10, 0]).unwrap().to_vec();
        assert_eq!(out.len(), 2);
        for &o in &out {
            // Uniform softmax → 0.5 → 0.5 · 256 - 128 = 0 (±1 rounding).
            assert!(
                (o as i32).abs() <= 1,
                "uniform softmax entry = {o}, expected ~0 (±1)"
            );
        }
    }

    #[test]
    fn sigmoid_layer_preserves_size_and_monotonicity() {
        let sigmoid = LayerSpec::Sigmoid {
            size: 4,
            input_scale: 0.05,
            input_zero_point: 0,
            output_scale: 1.0 / 256.0,
            output_zero_point: -128,
        };
        let mut chain = ChainConfig::new(vec![sigmoid]).unwrap();
        // Use the actual i8 range extremes. With `input_scale = 0.05`, the
        // dequantised range is [-128 · 0.05, 127 · 0.05] = [-6.4, 6.35],
        // which is enough for sigmoid to reach ~0 / ~1 (sigmoid(±6) is
        // 0.0025 / 0.9975 — within one LSB of the ±128 output extremes at
        // `1/256` scale).
        let out = chain.run(&[i8::MIN, -50, 50, i8::MAX]).unwrap().to_vec();
        assert_eq!(out.len(), 4);
        // sigmoid is monotonically increasing on any monotonic input.
        for w in out.windows(2) {
            assert!(w[1] >= w[0], "sigmoid non-monotonic: {out:?}");
        }
        // Near-saturation at the range extremes. `atol ≤ 2 LSB` matches the
        // sister `sigmoid_int8_matches_f32_reference_across_i8_range` test in
        // `kernels.rs`, which pins the same INT8-sigmoid tolerance globally.
        assert!(
            (out[0] as i32 - (-128)).abs() <= 2,
            "sigmoid(i8::MIN) = {} should be within 2 LSB of -128",
            out[0]
        );
        assert!(
            (out[3] as i32 - 127).abs() <= 2,
            "sigmoid(i8::MAX) = {} should be within 2 LSB of +127",
            out[3]
        );
    }

    #[test]
    fn dwconv_dense_softmax_chain_end_to_end() {
        // Real-shape sanity: 1×1×4 → dwconv (1×1, in_c=4, per-channel identity)
        // → FC(4 → 3) → Softmax(3). This is the smallest chain that exercises
        // every kernel except Sigmoid, and verifies the ping-pong scratch
        // sizes cover the widest intermediate stage.
        let dwconv = LayerSpec::DepthwiseConv2d {
            weight_i8: vec![1i8, 1, 1, 1], // per-channel identity
            bias_i32: vec![0i32; 4],
            dims: ConvDims {
                in_h: 1,
                in_w: 1,
                in_c: 4,
                out_c: 4,
                kh: 1,
                kw: 1,
                stride_h: 1,
                stride_w: 1,
                pad_h: 0,
                pad_w: 0,
            },
            input_zero_point: 0,
            output_zero_point: 0,
            output_scale: 1.0,
        };
        // FC picks each input channel to a distinct output (permutation +
        // amplification): output[0] = in[0] · 2, output[1] = in[1] · 2,
        // output[2] = in[2] · 2. Amplification ensures dequant is not
        // degenerate.
        let fc = LayerSpec::FullyConnected {
            weight_i8: vec![
                2i8, 0, 0, 0, //
                0, 2, 0, 0, //
                0, 0, 2, 0, //
            ],
            bias_i32: vec![0i32; 3],
            in_dim: 4,
            out_dim: 3,
            input_zero_point: 0,
            output_zero_point: 0,
            output_scale: 1.0,
        };
        let softmax = LayerSpec::Softmax {
            size: 3,
            input_scale: 0.05,
            input_zero_point: 0,
            output_scale: 1.0 / 256.0,
            output_zero_point: -128,
        };
        let mut chain = ChainConfig::new(vec![dwconv, fc, softmax]).unwrap();
        // Peak at input[2]: dwconv → [1, 2, 40, 3]; FC → [2, 4, 80]; softmax
        // dominated by class 2.
        let out = chain.run(&[1i8, 2, 40, 3]).unwrap().to_vec();
        assert_eq!(out.len(), 3);
        // Class 2 must win; probability should be near 1 (saturates at 127).
        assert!(
            out[2] > out[0] && out[2] > out[1],
            "class 2 must be the winner: {out:?}"
        );
    }

    fn synthetic_hey_jarvis_layers() -> Vec<LayerSpec> {
        let q = |n: usize| vec![1.0f32; n];
        let conv = |dims: ConvDims,
                    weight_len: usize,
                    out_c: usize,
                    input_zp: i8,
                    output_zp: i8,
                    fused_relu: bool| {
            let scales = q(out_c);
            LayerSpec::Conv2dPerChannel {
                weight_i8: vec![0; weight_len],
                bias_i32: vec![0; out_c],
                dims,
                input_zero_point: input_zp,
                output_scales: scales,
                output_zero_point: output_zp,
                fused_relu,
            }
        };
        let dw = |h: usize, c: usize, input_zp: i8, output_zp: i8| {
            let scales = q(c);
            LayerSpec::DepthwiseConv2dPerChannel {
                weight_i8: vec![0; h * c],
                bias_i32: vec![0; c],
                dims: ConvDims {
                    in_h: h,
                    in_w: 1,
                    in_c: c,
                    out_c: c,
                    kh: h,
                    kw: 1,
                    stride_h: 1,
                    stride_w: 1,
                    pad_h: 0,
                    pad_w: 0,
                },
                input_zero_point: input_zp,
                output_scales: scales,
                output_zero_point: output_zp,
                fused_relu: false,
            }
        };
        let pw = |in_c: usize, out_c: usize, input_zp: i8| {
            let scales = q(out_c);
            LayerSpec::Conv2dPerChannel {
                weight_i8: vec![0; in_c * out_c],
                bias_i32: vec![0; out_c],
                dims: ConvDims {
                    in_h: 1,
                    in_w: 1,
                    in_c,
                    out_c,
                    kh: 1,
                    kw: 1,
                    stride_h: 1,
                    stride_w: 1,
                    pad_h: 0,
                    pad_w: 0,
                },
                input_zero_point: input_zp,
                output_scales: scales,
                output_zero_point: -128,
                fused_relu: true,
            }
        };
        let scales = q(1);
        vec![
            conv(
                ConvDims {
                    in_h: 5,
                    in_w: 1,
                    in_c: 40,
                    out_c: 30,
                    kh: 5,
                    kw: 1,
                    stride_h: 3,
                    stride_w: 1,
                    pad_h: 0,
                    pad_w: 0,
                },
                5 * 40 * 30,
                30,
                -128,
                -128,
                true,
            ),
            dw(5, 30, -128, -11),
            pw(30, 60, -11),
            dw(9, 60, -128, 31),
            pw(60, 60, 31),
            dw(13, 60, -128, 1),
            pw(60, 60, 1),
            dw(21, 60, -128, -36),
            pw(60, 60, -36),
            LayerSpec::FullyConnectedPerChannel {
                weight_i8: vec![0; 300],
                bias_i32: vec![0],
                in_dim: 300,
                out_dim: 1,
                input_zero_point: -128,
                output_scales: scales,
                output_zero_point: 30,
                fused_relu: false,
            },
            LayerSpec::Sigmoid {
                size: 1,
                input_scale: 0.17895515263080597,
                input_zero_point: 30,
                output_scale: 0.00390625,
                output_zero_point: -128,
            },
        ]
    }

    fn synthetic_hey_jarvis_plan() -> HeyJarvisStreamingPlan {
        HeyJarvisStreamingPlan::new(synthetic_hey_jarvis_layers()).unwrap()
    }

    #[test]
    fn hey_jarvis_streaming_initializes_updates_and_resets_states() {
        let plan = synthetic_hey_jarvis_plan();
        let mut executor = HeyJarvisStreamingExecutor::new(plan).unwrap();
        assert_eq!(executor.state_count(), HEY_JARVIS_STATE_COUNT);
        assert!(executor
            .state(0)
            .unwrap()
            .iter()
            .all(|&v| v == HEY_JARVIS_STATE_QUANTIZED_ZERO));
        let input: Vec<i8> = (0..HEY_JARVIS_INPUT_SIZE).map(|v| v as i8).collect();
        assert_eq!(executor.run(&input).unwrap().len(), 1);
        assert_eq!(executor.state(0).unwrap(), &input[40..]);
        assert!(executor.state(99).is_err());
        executor.reset();
        assert!(executor
            .state(0)
            .unwrap()
            .iter()
            .all(|&v| v == HEY_JARVIS_STATE_QUANTIZED_ZERO));
    }

    #[test]
    fn hey_jarvis_streaming_rejects_wrong_state_count_and_input() {
        let plan = synthetic_hey_jarvis_plan();
        assert!(HeyJarvisStreamingExecutor::from_states(plan, vec![]).is_err());
        let mut executor = HeyJarvisStreamingExecutor::new(synthetic_hey_jarvis_plan()).unwrap();
        assert!(executor.run(&[0; HEY_JARVIS_INPUT_SIZE - 1]).is_err());
        assert_eq!(executor.run(&[0; HEY_JARVIS_INPUT_SIZE]).unwrap(), &[0]);
        assert_eq!(executor.run(&[0; HEY_JARVIS_INPUT_SIZE]).unwrap(), &[0]);
    }

    #[test]
    fn hey_jarvis_plan_rejects_affine_and_fused_contract_tampering() {
        let mut layers = synthetic_hey_jarvis_layers();
        if let LayerSpec::Sigmoid { input_scale, .. } = &mut layers[10] {
            *input_scale = 0.1;
        }
        assert!(HeyJarvisStreamingPlan::new(layers).is_err());

        let mut layers = synthetic_hey_jarvis_layers();
        if let LayerSpec::Sigmoid {
            input_zero_point, ..
        } = &mut layers[10]
        {
            *input_zero_point = -128;
        }
        assert!(HeyJarvisStreamingPlan::new(layers).is_err());

        let mut layers = synthetic_hey_jarvis_layers();
        if let LayerSpec::Sigmoid { output_scale, .. } = &mut layers[10] {
            *output_scale = 1.0 / 128.0;
        }
        assert!(HeyJarvisStreamingPlan::new(layers).is_err());

        let mut layers = synthetic_hey_jarvis_layers();
        if let LayerSpec::Sigmoid {
            output_zero_point, ..
        } = &mut layers[10]
        {
            *output_zero_point = 0;
        }
        assert!(HeyJarvisStreamingPlan::new(layers).is_err());

        let mut layers = synthetic_hey_jarvis_layers();
        if let LayerSpec::DepthwiseConv2dPerChannel {
            output_zero_point, ..
        } = &mut layers[1]
        {
            *output_zero_point = -128;
        }
        assert!(HeyJarvisStreamingPlan::new(layers).is_err());

        let mut layers = synthetic_hey_jarvis_layers();
        if let LayerSpec::DepthwiseConv2dPerChannel { fused_relu, .. } = &mut layers[1] {
            *fused_relu = true;
        }
        assert!(HeyJarvisStreamingPlan::new(layers).is_err());
    }
}
