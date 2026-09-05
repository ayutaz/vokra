//! SBV2 stochastic duration predictor (SDP): the real Style-Bert-VITS2 v2
//! shape — a `pre` 1×1 conv, a module-level 3-layer depth-wise-separable
//! `DDSConv`, a speaker-conditioning 1×1 `cond` conv, a `proj` 1×1 conv, an
//! `ElementwiseAffine` slot 0 and four `ConvFlow`s at slots 1/3/5/7 whose
//! inverse walks a piecewise-rational-quadratic spline. **286 total
//! tensors**: 144 production side (`sdp.*`) walked at inference time and 142
//! training-side (`sdp.post_*`) the converter skips (Blocker 2c).
//!
//! (Clean-room comment: see `mod.rs`. The DDS-net + spline-coupling
//! construction follows the VITS paper arXiv:2106.06103 and the vendored MIT
//! reference at `tools/parity/vendor/vits/{modules,transforms,sdp}.py`
//! (pinned commit `2e561ba58618d021b5b8323d3765880f7e0ecfdb`); no
//! SBV2/BV2/AGPL source was referenced. The rewrite from the pre-Blocker-2c
//! scalar-affine placeholder — which shipped as `SbV2CouplingLayer`, kept in
//! `sbv2/mod.rs`'s `synthetic_for_test_e2e` factory only through end-of-file
//! constant `FIXED_DURATION` and reachable through no other path — was
//! forced by the real safetensors' 286-tensor layout: the placeholder had
//! no way to load a real checkpoint at all, so Blocker 2c replaced the
//! whole primitive stack rather than layer atop it.)
//!
//! # Layout conventions
//!
//! Two different layouts appear here:
//!
//! - **Row-major, position-major `[T, D]`** — the layout every other `sbv2`
//!   module (`text_encoder.rs`, `flow.rs`, `style.rs`) uses. Row `p` is
//!   `buf[p * D .. (p + 1) * D]`. `SbV2SDP::sample`'s `hidden` argument uses
//!   this.
//! - **Channel-major `[C, T]`** — the layout the DDS / ConvFlow / spline
//!   primitives internally use, matching PyTorch/VITS `nn.Conv1d`'s native
//!   `[N=batch, C, T]` (with N=1 elided) — channel `c` at time `t` is
//!   `buf[c * T + t]`. Every primitive below (`DDSConv`, `ElementwiseAffine`,
//!   `ConvFlow`, `SbV2SDP::body`) speaks this convention.
//!
//! `SbV2SDP::sample` transposes `hidden` from row-major `[T, D]` into
//! channel-major `[D, T]` at the entry to `body`, and returns a flat
//! `Vec<i32>` of length `text_seq_len` (no layout concern).
//!
//! # Empty-flows / all-zero-weight identity path (backward compat)
//!
//! [`SbV2SDP::empty`] returns an SDP with `flows = vec![]`, an
//! `ElementwiseAffine` with `m = 0, logs = 0` (identity), and every conv
//! weight/bias zero. Combined with `noise_scale_w == 0.0`, that path
//! deterministically returns all-`1`s (`exp(0).ceil().max(1) == 1`)
//! regardless of `hidden`, `g`, or RNG state — mirrors the pre-Blocker-2c
//! scaffold's empty-`flow_layers` behaviour so
//! `SbV2Model::synthetic_for_test` (`sbv2/mod.rs`) continues to compile and
//! pass without any parameter changes on the callers' side.

use vokra_core::Result;
use vokra_core::rng::NormalSource;

use crate::compute::Compute;

// -----------------------------------------------------------------------------
// SDP-wide constants (from upstream `models.py::StochasticDurationPredictor`
// + `modules.py::ConvFlow` defaults + `transforms.py`; mirror piper_plus's
// `piper_plus::config::{DP_KERNEL, DP_CONV_LAYERS, RQS_NUM_BINS,
// RQS_TAIL_BOUND, LAYER_NORM_EPS}` verbatim — SBV2 v2 base uses the same
// values).
// -----------------------------------------------------------------------------

/// DDS / ConvFlow kernel size (`kernel_size = 3` in
/// `StochasticDurationPredictor.__init__`'s only construction call).
pub(crate) const DP_KERNEL: usize = 3;
/// DDS layer depth (`n_layers = 3` in the same `__init__`).
pub(crate) const DP_CONV_LAYERS: usize = 3;
/// Piecewise-rational-quadratic-spline bins per ConvFlow (upstream
/// `ConvFlow`'s `num_bins = 10` default).
pub(crate) const RQS_NUM_BINS: usize = 10;
/// Spline tail bound (upstream `ConvFlow`'s `tail_bound = 5.0` default).
pub(crate) const RQS_TAIL_BOUND: f32 = 5.0;
/// LayerNorm epsilon (`nn.LayerNorm` default; VITS inherits it — see
/// `modules.py::LayerNorm.__init__(eps=1e-5)`).
pub(crate) const LAYER_NORM_EPS: f32 = 1e-5;

/// Minimum spline bin width (`transforms.py::piecewise_rational_quadratic_transform`'s
/// `min_bin_width = 1e-3` default).
const MIN_BIN_WIDTH: f32 = 1e-3;
/// Minimum spline bin height (same file, `min_bin_height = 1e-3`).
const MIN_BIN_HEIGHT: f32 = 1e-3;
/// Minimum spline derivative (same file, `min_derivative = 1e-3`).
const MIN_DERIVATIVE: f32 = 1e-3;

// -----------------------------------------------------------------------------
// Numeric helpers — kept private to this module to preserve `sbv2`'s Compute-
// free policy (see `flow.rs`'s module doc for the identical rationale
// applied to `SbV2AffineCouplingLayer`).
// -----------------------------------------------------------------------------

/// Softplus `ln(1 + eˣ)` with the large-`x` guard PyTorch uses (mirror of
/// `piper_plus::nn::softplus`).
fn softplus(x: f32) -> f32 {
    // WP-12 (2026-08-10): exp + log through vokra_math for cross-plat
    // determinism within Vokra (SDP softplus, per-block feed-forward).
    if x > 20.0 {
        x
    } else {
        vokra_math::log(1.0 + vokra_math::exp(x))
    }
}

/// Exact (erf-based) GELU, matching PyTorch `F.gelu` default (mirror of
/// `piper_plus::nn::gelu`).
fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + erf(x * std::f32::consts::FRAC_1_SQRT_2))
}

/// Error function (Abramowitz & Stegun 7.1.26; ~1e-7 max error — well
/// inside the FP32 parity bound). Mirror of `piper_plus::nn::erf`.
#[allow(clippy::excessive_precision)] // A&S reference coefficients kept verbatim
fn erf(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    // WP-12 (2026-08-10): erf exp through vokra_math for cross-plat
    // determinism within Vokra (SDP GELU derivation, per-hidden-dim).
    let y = 1.0
        - (((((1.061_405_43 * t - 1.453_152_027) * t) + 1.421_413_741) * t - 0.284_496_736) * t
            + 0.254_829_592)
            * t
            * vokra_math::exp(-x * x);
    sign * y
}

/// LayerNorm over the channel axis of a channel-major `[C, T]` buffer
/// (VITS `modules.py::LayerNorm.forward` normalises the channel vector at
/// each time step, then affine-transforms with `gamma`/`beta`). Mirror of
/// `piper_plus::nn::layer_norm_channels`.
fn layer_norm_channels(
    x: &[f32],
    channels: usize,
    time: usize,
    gamma: &[f32],
    beta: &[f32],
) -> Vec<f32> {
    let mut out = vec![0.0_f32; x.len()];
    for t in 0..time {
        let mut mean = 0.0_f32;
        for c in 0..channels {
            mean += x[c * time + t];
        }
        mean /= channels as f32;
        let mut var = 0.0_f32;
        for c in 0..channels {
            let d = x[c * time + t] - mean;
            var += d * d;
        }
        var /= channels as f32;
        // WP-12 (2026-08-10): LayerNorm sqrt through vokra_math for cross-plat
        // determinism within Vokra (SDP LayerNorm, per-time per-block).
        let inv = 1.0 / vokra_math::sqrt(var + LAYER_NORM_EPS);
        for c in 0..channels {
            out[c * time + t] = (x[c * time + t] - mean) * inv * gamma[c] + beta[c];
        }
    }
    out
}

/// General 1D convolution over a channel-major `[in_ch, in_len]` signal,
/// producing `[out_ch, out_len]` channel-major. `weight` is
/// `[out_ch, in_ch/groups, kernel]` row-major (PyTorch `nn.Conv1d`'s native
/// layout). Same-padding controlled by the caller via `pad`. `groups` is
/// depth-wise-separable's only knob (`groups == in_ch == out_ch` = depthwise
/// per-channel conv). Written scalar rather than reusing
/// `piper_plus::nn::conv1d` because that function threads a `Compute` handle
/// through — the `sbv2` module deliberately stays Compute-free (see
/// `flow.rs`'s module doc for the identical rationale).
#[allow(clippy::too_many_arguments)]
fn conv1d_scalar(
    x: &[f32],
    in_ch: usize,
    in_len: usize,
    weight: &[f32],
    out_ch: usize,
    kernel: usize,
    bias: Option<&[f32]>,
    pad: usize,
    dilation: usize,
    groups: usize,
) -> Vec<f32> {
    let eff = dilation * (kernel - 1) + 1;
    let out_len = in_len + 2 * pad - eff + 1; // stride=1
    debug_assert!(out_len > 0, "conv1d_scalar: degenerate output length");
    debug_assert_eq!(in_ch % groups, 0, "in_ch must be divisible by groups");
    debug_assert_eq!(out_ch % groups, 0, "out_ch must be divisible by groups");
    let in_g = in_ch / groups;
    let out_g = out_ch / groups;
    let mut out = vec![0.0_f32; out_ch * out_len];

    for g in 0..groups {
        for oc_local in 0..out_g {
            let oc = g * out_g + oc_local;
            let wrow_base = oc * (in_g * kernel);
            let b = bias.map_or(0.0, |bs| bs[oc]);
            for ot in 0..out_len {
                let mut acc = b;
                for ic_local in 0..in_g {
                    let ic = g * in_g + ic_local;
                    for kk in 0..kernel {
                        // Signed index into x[ic, ...]; skip if in padding.
                        let it = ot as isize + (kk * dilation) as isize - pad as isize;
                        if it < 0 || (it as usize) >= in_len {
                            continue;
                        }
                        let it = it as usize;
                        let w = weight[wrow_base + ic_local * kernel + kk];
                        acc += w * x[ic * in_len + it];
                    }
                }
                out[oc * out_len + ot] = acc;
            }
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn conv1d_with_compute(
    compute: &Compute,
    input: &[f32],
    in_channels: usize,
    input_len: usize,
    weight: &[f32],
    out_channels: usize,
    kernel: usize,
    bias: Option<&[f32]>,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> Result<Vec<f32>> {
    let effective_kernel = dilation * (kernel - 1) + 1;
    let expanded;
    let effective_weight = if dilation == 1 {
        weight
    } else {
        let in_per_group = in_channels / groups;
        let mut values = vec![0.0; out_channels * in_per_group * effective_kernel];
        for output in 0..out_channels {
            for input in 0..in_per_group {
                for tap in 0..kernel {
                    values[(output * in_per_group + input) * effective_kernel + tap * dilation] =
                        weight[(output * in_per_group + input) * kernel + tap];
                }
            }
        }
        expanded = values;
        &expanded
    };
    let output_len = input_len + 2 * padding - effective_kernel + 1;
    let mut output = vec![0.0; out_channels * output_len];
    if groups == 1 {
        compute.conv1d_f32(
            input,
            in_channels,
            input_len,
            effective_weight,
            out_channels,
            effective_kernel,
            bias,
            1,
            padding,
            &mut output,
        )?;
    } else {
        compute.grouped_conv1d_f32(
            input,
            in_channels,
            input_len,
            effective_weight,
            out_channels,
            effective_kernel,
            bias,
            1,
            padding,
            groups,
            &mut output,
        )?;
    }
    Ok(output)
}

fn layer_norm_channels_with_compute(
    compute: &Compute,
    input: &[f32],
    channels: usize,
    time: usize,
    gamma: &[f32],
    beta: &[f32],
) -> Result<Vec<f32>> {
    let position_major = transpose_channel_major(input, channels, time);
    let mut normalized = vec![0.0; input.len()];
    compute.layer_norm_f32(
        &position_major,
        &mut normalized,
        time,
        channels,
        gamma,
        beta,
        LAYER_NORM_EPS,
    )?;
    Ok(transpose_position_major(&normalized, time, channels))
}

fn transpose_channel_major(input: &[f32], channels: usize, time: usize) -> Vec<f32> {
    let mut output = vec![0.0; input.len()];
    for channel in 0..channels {
        for position in 0..time {
            output[position * channels + channel] = input[channel * time + position];
        }
    }
    output
}

fn transpose_position_major(input: &[f32], time: usize, channels: usize) -> Vec<f32> {
    let mut output = vec![0.0; input.len()];
    for position in 0..time {
        for channel in 0..channels {
            output[channel * time + position] = input[position * channels + channel];
        }
    }
    output
}

/// Channel flip of a `[2, T]` latent (`torch.flip(x, [1])` in the VITS SDP's
/// upstream inference loop — swaps channel 0 and channel 1 pointwise across
/// all `T` time steps). Mirror of `piper_plus::duration::flip2`.
fn flip2(x: &mut [f32], time: usize) {
    for t in 0..time {
        x.swap(t, time + t);
    }
}

// -----------------------------------------------------------------------------
// Primitives — public so `tests/sbv2_sdp.rs` can build them from clean-room
// weights; `pub(crate)` where a caller from outside the crate has no
// legitimate use.
// -----------------------------------------------------------------------------

/// LayerNorm affine parameters, `[channels]`-shaped each.
#[derive(Clone, Debug)]
pub struct SdpLayerNorm {
    /// Per-channel affine scale (`nn.LayerNorm.weight`).
    pub gamma: Vec<f32>,
    /// Per-channel affine bias (`nn.LayerNorm.bias`).
    pub beta: Vec<f32>,
}

impl SdpLayerNorm {
    /// Zero-weight LayerNorm (`gamma = 0, beta = 0`): output = 0 regardless
    /// of input — the identity for the residual path in a
    /// [`DDSConv`] built via [`DDSConv::zero`].
    fn zero(channels: usize) -> Self {
        Self {
            gamma: vec![0.0; channels],
            beta: vec![0.0; channels],
        }
    }
}

/// Dilated depth-separable conv stack (VITS `modules.py::DDSConv`):
/// per-layer `convs_sep` (depthwise, `groups = channels`, `kernel = 3`,
/// dilation `kernel^i` for layer `i`), a channel-wise LayerNorm, GELU, a
/// point-wise `convs_1x1` (kernel 1), another LayerNorm, GELU, then a
/// residual add — for `n_layers = 3` layers total.
///
/// Real SBV2 v2 base uses `channels = 192`. Tests use small values (4-8) —
/// [`DDSConv::from_weights`] takes whatever `channels` the caller supplies.
#[derive(Clone, Debug)]
pub struct DDSConv {
    channels: usize,
    n_layers: usize,
    kernel: usize,
    // Per-layer weights, each list of length `n_layers`.
    convs_sep_w: Vec<Vec<f32>>, // depthwise, each `[channels, 1, kernel]`
    convs_sep_b: Vec<Vec<f32>>, // each `[channels]`
    convs_1x1_w: Vec<Vec<f32>>, // pointwise, each `[channels, channels, 1]`
    convs_1x1_b: Vec<Vec<f32>>, // each `[channels]`
    norms_1: Vec<SdpLayerNorm>, // each `channels`-wide
    norms_2: Vec<SdpLayerNorm>, // each `channels`-wide
}

impl DDSConv {
    /// Builds a DDS-net from `n_layers`-long lists of per-layer weights,
    /// each list shape-checked in debug builds. Public so the parity dumper
    /// (`tools/parity/sbv2_dump_reference.py`) and unit tests can construct
    /// an instance from real GGUF-loaded weights or hand-picked test
    /// values.
    ///
    /// # Panics
    ///
    /// Panics (via `debug_assert!`, hot inner-loop constructor per this
    /// crate's established convention — see
    /// [`StyleVectorInjector::from_projections`](super::style::StyleVectorInjector::from_projections)'s
    /// panic docs) if any list length disagrees with `n_layers`, or any
    /// weight/bias tensor's length disagrees with its documented shape.
    #[allow(clippy::too_many_arguments)]
    pub fn from_weights(
        channels: usize,
        n_layers: usize,
        kernel: usize,
        convs_sep_w: Vec<Vec<f32>>,
        convs_sep_b: Vec<Vec<f32>>,
        convs_1x1_w: Vec<Vec<f32>>,
        convs_1x1_b: Vec<Vec<f32>>,
        norms_1: Vec<SdpLayerNorm>,
        norms_2: Vec<SdpLayerNorm>,
    ) -> Self {
        debug_assert_eq!(convs_sep_w.len(), n_layers);
        debug_assert_eq!(convs_sep_b.len(), n_layers);
        debug_assert_eq!(convs_1x1_w.len(), n_layers);
        debug_assert_eq!(convs_1x1_b.len(), n_layers);
        debug_assert_eq!(norms_1.len(), n_layers);
        debug_assert_eq!(norms_2.len(), n_layers);
        for l in 0..n_layers {
            debug_assert_eq!(
                convs_sep_w[l].len(),
                channels * kernel,
                "convs_sep.{l}.weight shape mismatch"
            );
            debug_assert_eq!(convs_sep_b[l].len(), channels, "convs_sep.{l}.bias shape");
            debug_assert_eq!(
                convs_1x1_w[l].len(),
                channels * channels,
                "convs_1x1.{l}.weight shape"
            );
            debug_assert_eq!(convs_1x1_b[l].len(), channels, "convs_1x1.{l}.bias shape");
            debug_assert_eq!(norms_1[l].gamma.len(), channels, "norms_1.{l}.gamma shape");
            debug_assert_eq!(norms_1[l].beta.len(), channels, "norms_1.{l}.beta shape");
            debug_assert_eq!(norms_2[l].gamma.len(), channels, "norms_2.{l}.gamma shape");
            debug_assert_eq!(norms_2[l].beta.len(), channels, "norms_2.{l}.beta shape");
        }
        Self {
            channels,
            n_layers,
            kernel,
            convs_sep_w,
            convs_sep_b,
            convs_1x1_w,
            convs_1x1_b,
            norms_1,
            norms_2,
        }
    }

    /// Zero-weight DDS: every layer's convs, bias, and LayerNorm affine
    /// params are zero — so each layer's `y = norm(conv(x))` returns zero,
    /// GELU of zero is zero, the residual `x += y` is a no-op, and the
    /// whole stack is the identity. Used by [`SbV2SDP::empty`] for the
    /// synthetic-test scaffold and by the RED-phase primitive tests
    /// (`tests/sbv2_sdp.rs`).
    pub fn zero(channels: usize, n_layers: usize, kernel: usize) -> Self {
        let sep_w = vec![vec![0.0_f32; channels * kernel]; n_layers];
        let sep_b = vec![vec![0.0_f32; channels]; n_layers];
        let one_w = vec![vec![0.0_f32; channels * channels]; n_layers];
        let one_b = vec![vec![0.0_f32; channels]; n_layers];
        let n1 = (0..n_layers)
            .map(|_| SdpLayerNorm::zero(channels))
            .collect();
        let n2 = (0..n_layers)
            .map(|_| SdpLayerNorm::zero(channels))
            .collect();
        Self::from_weights(
            channels, n_layers, kernel, sep_w, sep_b, one_w, one_b, n1, n2,
        )
    }

    /// Runs the DDS-net forward on a channel-major `[channels, time]`
    /// buffer, returning a `[channels, time]` buffer of the same shape.
    /// `g`, when `Some`, is the ConvFlow-conditioning input (also
    /// `[channels, time]`), pointwise-added to `x` **before** the first
    /// layer — this is how upstream `modules.py::DDSConv.forward` folds the
    /// SDP body's `[dp_filter, T]` conditioner into the flow's own
    /// hidden state.
    ///
    /// # Panics
    ///
    /// Panics (via `debug_assert!`) if `x.len() != channels * time` or if
    /// `g.is_some()` and `g.len() != channels * time`.
    pub fn forward(&self, x: &[f32], time: usize, g: Option<&[f32]>) -> Vec<f32> {
        debug_assert_eq!(x.len(), self.channels * time, "x shape");
        if let Some(g) = g {
            debug_assert_eq!(g.len(), self.channels * time, "g shape");
        }
        let c = self.channels;
        let mut buf = if let Some(g) = g {
            x.iter().zip(g).map(|(a, b)| a + b).collect::<Vec<f32>>()
        } else {
            x.to_vec()
        };
        for l in 0..self.n_layers {
            let dilation = self.kernel.pow(l as u32); // 1, 3, 9 for kernel=3, n_layers=3
            let pad = dilation * (self.kernel - 1) / 2;

            // Depthwise sep conv (groups = channels).
            let y = conv1d_scalar(
                &buf,
                c,
                time,
                &self.convs_sep_w[l],
                c,
                self.kernel,
                Some(&self.convs_sep_b[l]),
                pad,
                dilation,
                c, // groups = channels
            );
            let mut y =
                layer_norm_channels(&y, c, time, &self.norms_1[l].gamma, &self.norms_1[l].beta);
            for v in &mut y {
                *v = gelu(*v);
            }

            // Point-wise 1x1 conv (groups = 1).
            let y2 = conv1d_scalar(
                &y,
                c,
                time,
                &self.convs_1x1_w[l],
                c,
                1,
                Some(&self.convs_1x1_b[l]),
                0, // pad
                1, // dilation
                1, // groups
            );
            let mut y2 =
                layer_norm_channels(&y2, c, time, &self.norms_2[l].gamma, &self.norms_2[l].beta);
            for v in &mut y2 {
                *v = gelu(*v);
            }
            // Residual add (in place).
            for (b, s) in buf.iter_mut().zip(&y2) {
                *b += s;
            }
        }
        buf
    }

    /// Backend-dispatched DDS forward used by MeloTTS. Dilated depthwise
    /// kernels are expanded with zero taps and sent through the existing
    /// grouped-convolution backend, so no learned convolution falls back to
    /// host scalar execution.
    pub fn forward_with_compute(
        &self,
        compute: &Compute,
        input: &[f32],
        time: usize,
        conditioning: Option<&[f32]>,
    ) -> Result<Vec<f32>> {
        debug_assert_eq!(input.len(), self.channels * time);
        let mut buffer = if let Some(conditioning) = conditioning {
            debug_assert_eq!(conditioning.len(), input.len());
            input
                .iter()
                .zip(conditioning)
                .map(|(left, right)| left + right)
                .collect()
        } else {
            input.to_vec()
        };
        for layer in 0..self.n_layers {
            let dilation = self.kernel.pow(layer as u32);
            let padding = dilation * (self.kernel - 1) / 2;
            let separated = conv1d_with_compute(
                compute,
                &buffer,
                self.channels,
                time,
                &self.convs_sep_w[layer],
                self.channels,
                self.kernel,
                Some(&self.convs_sep_b[layer]),
                padding,
                dilation,
                self.channels,
            )?;
            let normalized = layer_norm_channels_with_compute(
                compute,
                &separated,
                self.channels,
                time,
                &self.norms_1[layer].gamma,
                &self.norms_1[layer].beta,
            )?;
            let mut activated = vec![0.0; normalized.len()];
            compute.gelu_f32(&normalized, &mut activated)?;

            let pointwise = conv1d_with_compute(
                compute,
                &activated,
                self.channels,
                time,
                &self.convs_1x1_w[layer],
                self.channels,
                1,
                Some(&self.convs_1x1_b[layer]),
                0,
                1,
                1,
            )?;
            let normalized = layer_norm_channels_with_compute(
                compute,
                &pointwise,
                self.channels,
                time,
                &self.norms_2[layer].gamma,
                &self.norms_2[layer].beta,
            )?;
            compute.gelu_f32(&normalized, &mut activated)?;
            for (value, residual) in buffer.iter_mut().zip(&activated) {
                *value += residual;
            }
        }
        Ok(buffer)
    }
}

/// Element-wise affine flow (`modules.py::ElementwiseAffine`): a fixed
/// 2-channel-wide, time-independent affine transform `x = m + exp(logs) * z`
/// (forward) or `z = (x - m) * exp(-logs)` (reverse). SBV2 v2's SDP has
/// exactly one such layer at flow slot 0 (`sdp.flows.0.{m,logs}`, each
/// `[2, 1]`).
#[derive(Clone, Debug)]
pub struct ElementwiseAffine {
    /// `[2]`, one value per channel (upstream stores as `[2, 1]`; the `1`
    /// dimension is a broadcast axis over time and collapses at load).
    m: Vec<f32>,
    /// `[2]`, one value per channel.
    logs: Vec<f32>,
}

impl ElementwiseAffine {
    /// Builds an `ElementwiseAffine` from `[2]`-length `m` and `logs`
    /// vectors. Panics in debug if either length is not 2.
    pub fn from_weights(m: Vec<f32>, logs: Vec<f32>) -> Self {
        debug_assert_eq!(m.len(), 2, "m must be [channels=2]");
        debug_assert_eq!(logs.len(), 2, "logs must be [channels=2]");
        Self { m, logs }
    }

    /// Zero-weight identity (`m = 0, logs = 0`): reverse output = input.
    pub fn identity() -> Self {
        Self::from_weights(vec![0.0, 0.0], vec![0.0, 0.0])
    }

    /// Inverse pass: `z = (x - m) * exp(-logs)`, applied per channel with
    /// broadcast across the `time` axis. Returns a fresh `[2, time]`
    /// channel-major buffer.
    ///
    /// # Panics
    ///
    /// Panics (via `debug_assert!`) if `x.len() != 2 * time`.
    pub fn reverse(&self, x: &[f32], time: usize) -> Vec<f32> {
        debug_assert_eq!(x.len(), 2 * time, "x must be [2, time]");
        let mut out = vec![0.0_f32; 2 * time];
        for c in 0..2 {
            let m = self.m[c];
            // WP-12 (2026-08-10): ElementwiseAffine inverse exp through
            // vokra_math for cross-plat determinism within Vokra (SDP flow).
            let inv = vokra_math::exp(-self.logs[c]);
            for t in 0..time {
                out[c * time + t] = (x[c * time + t] - m) * inv;
            }
        }
        out
    }
}

/// Piecewise-rational-quadratic-spline coupling flow
/// (`modules.py::ConvFlow`) — the workhorse of VITS SDP's inference-time
/// reverse pass. Splits a `[2, T]` latent into halves (channel 0 =
/// `x0` passes through unchanged, channel 1 = `x1` is the transformed
/// half), runs `x0` (plus the SDP body's `[dp_filter, T]` conditioning)
/// through a `pre` 1×1 conv → [`DDSConv`] → `proj` 1×1 conv to predict the
/// spline parameters (`[num_bins * 3 - 1, T]`, split into unnormalised
/// widths / heights / derivatives), then inverts the spline per time step
/// to compute the new `x1`.
///
/// Real SBV2 v2 base has 4 `ConvFlow`s, at `sdp.flows.{1,3,5,7}` (indices
/// 2/4/6/8 are `Flip` layers with no parameters — see [`SbV2SDP::sample`]
/// for how they are handled at inference time). Each ConvFlow's
/// `dp_filter == 192` (matches the SDP-wide `d_hidden`).
#[derive(Clone, Debug)]
pub struct ConvFlow {
    pre_w: Vec<f32>,  // `[dp_filter, 1, 1]` = `[dp_filter]`
    pre_b: Vec<f32>,  // `[dp_filter]`
    convs: DDSConv,   // channels = dp_filter
    proj_w: Vec<f32>, // `[num_bins * 3 - 1, dp_filter, 1]`
    proj_b: Vec<f32>, // `[num_bins * 3 - 1]`
    dp_filter: usize,
}

impl ConvFlow {
    /// Builds a `ConvFlow` from real safetensors-loaded weights (shapes
    /// documented on each field). Panics in debug if any shape disagrees.
    pub fn from_weights(
        pre_w: Vec<f32>,
        pre_b: Vec<f32>,
        convs: DDSConv,
        proj_w: Vec<f32>,
        proj_b: Vec<f32>,
        dp_filter: usize,
    ) -> Self {
        let out = RQS_NUM_BINS * 3 - 1;
        debug_assert_eq!(pre_w.len(), dp_filter, "pre.weight [dp_filter, 1, 1]");
        debug_assert_eq!(pre_b.len(), dp_filter, "pre.bias [dp_filter]");
        debug_assert_eq!(
            convs.channels, dp_filter,
            "convs channels must be dp_filter"
        );
        debug_assert_eq!(
            proj_w.len(),
            out * dp_filter,
            "proj.weight [num_bins*3-1={out}, dp_filter, 1]"
        );
        debug_assert_eq!(proj_b.len(), out, "proj.bias [num_bins*3-1={out}]");
        Self {
            pre_w,
            pre_b,
            convs,
            proj_w,
            proj_b,
            dp_filter,
        }
    }

    /// Reverse pass over a `[2, time]` channel-major latent, conditioned on
    /// the SDP body's `g` `[dp_filter, time]` channel-major output. Returns
    /// a fresh `[2, time]` buffer. Channel 0 is preserved bit-exact;
    /// channel 1 is transformed by the per-time RQS inverse.
    ///
    /// # Panics
    ///
    /// Panics (via `debug_assert!`) if `x.len() != 2 * time` or `g.len() !=
    /// dp_filter * time`.
    pub fn reverse(&self, x: &[f32], time: usize, g: &[f32]) -> Vec<f32> {
        debug_assert_eq!(x.len(), 2 * time, "x must be [2, time]");
        debug_assert_eq!(
            g.len(),
            self.dp_filter * time,
            "g must be [dp_filter, time]"
        );

        // 1. pre: `x0` (channel 0, `[1, T]`) → `h` `[dp_filter, T]`.
        let x0 = &x[..time];
        let h = conv1d_scalar(
            x0,
            1, // in_ch
            time,
            &self.pre_w,
            self.dp_filter,
            1, // kernel
            Some(&self.pre_b),
            0, // pad
            1, // dilation
            1, // groups
        );

        // 2. DDS(h, g).
        let h = self.convs.forward(&h, time, Some(g));

        // 3. proj: `h` → `params` `[num_bins*3-1, T]`.
        let params = conv1d_scalar(
            &h,
            self.dp_filter,
            time,
            &self.proj_w,
            RQS_NUM_BINS * 3 - 1,
            1, // kernel
            Some(&self.proj_b),
            0, // pad
            1, // dilation
            1, // groups
        );

        // 4. Per-time RQS inverse on channel 1.
        // WP-12 (2026-08-10): ConvFlow scale sqrt through vokra_math for
        // cross-plat determinism within Vokra (SDP RQS parameter scaling).
        let scale = vokra_math::sqrt(self.dp_filter as f32);
        let mut result = x.to_vec();
        for t in 0..time {
            let mut w = [0.0_f32; RQS_NUM_BINS];
            let mut hh = [0.0_f32; RQS_NUM_BINS];
            let mut d = [0.0_f32; RQS_NUM_BINS - 1];
            // params rows: [0..num_bins) = widths, [num_bins..2*num_bins) =
            // heights, [2*num_bins..3*num_bins-1) = derivatives.
            for b in 0..RQS_NUM_BINS {
                w[b] = params[b * time + t] / scale;
                hh[b] = params[(RQS_NUM_BINS + b) * time + t] / scale;
            }
            for b in 0..(RQS_NUM_BINS - 1) {
                d[b] = params[(2 * RQS_NUM_BINS + b) * time + t];
            }
            let x1 = x[time + t];
            result[time + t] = unconstrained_rqs_inverse(x1, &w, &hh, &d);
        }
        result
    }

    /// Backend-dispatched reverse coupling. The rational-quadratic spline is
    /// scalar control math; every learned convolution and DDS normalization
    /// uses `compute`.
    pub fn reverse_with_compute(
        &self,
        compute: &Compute,
        input: &[f32],
        time: usize,
        conditioning: &[f32],
    ) -> Result<Vec<f32>> {
        debug_assert_eq!(input.len(), 2 * time);
        debug_assert_eq!(conditioning.len(), self.dp_filter * time);
        let hidden = conv1d_with_compute(
            compute,
            &input[..time],
            1,
            time,
            &self.pre_w,
            self.dp_filter,
            1,
            Some(&self.pre_b),
            0,
            1,
            1,
        )?;
        let hidden = self
            .convs
            .forward_with_compute(compute, &hidden, time, Some(conditioning))?;
        let parameters = conv1d_with_compute(
            compute,
            &hidden,
            self.dp_filter,
            time,
            &self.proj_w,
            RQS_NUM_BINS * 3 - 1,
            1,
            Some(&self.proj_b),
            0,
            1,
            1,
        )?;
        let scale = vokra_math::sqrt(self.dp_filter as f32);
        let mut result = input.to_vec();
        for position in 0..time {
            let mut widths = [0.0; RQS_NUM_BINS];
            let mut heights = [0.0; RQS_NUM_BINS];
            let mut derivatives = [0.0; RQS_NUM_BINS - 1];
            for bin in 0..RQS_NUM_BINS {
                widths[bin] = parameters[bin * time + position] / scale;
                heights[bin] = parameters[(RQS_NUM_BINS + bin) * time + position] / scale;
            }
            for bin in 0..RQS_NUM_BINS - 1 {
                derivatives[bin] = parameters[(2 * RQS_NUM_BINS + bin) * time + position];
            }
            result[time + position] =
                unconstrained_rqs_inverse(input[time + position], &widths, &heights, &derivatives);
        }
        Ok(result)
    }
}

// -----------------------------------------------------------------------------
// Rational-quadratic-spline inverse (mirror of `piper_plus::duration::rqs_*`
// and upstream `transforms.py`).
// -----------------------------------------------------------------------------

fn unconstrained_rqs_inverse(
    input: f32,
    unnorm_w: &[f32; RQS_NUM_BINS],
    unnorm_h: &[f32; RQS_NUM_BINS],
    unnorm_d: &[f32; RQS_NUM_BINS - 1],
) -> f32 {
    let tb = RQS_TAIL_BOUND;
    if input < -tb || input > tb {
        return input;
    }
    // Pad derivatives with the linear-tail constant on both ends.
    // WP-12 (2026-08-10): RQS tail constant exp/log through vokra_math for
    // cross-plat determinism within Vokra (SDP RQS boundary derivative pad).
    let constant = vokra_math::log(vokra_math::exp(1.0 - MIN_DERIVATIVE) - 1.0);
    let mut derivs = [0.0_f32; RQS_NUM_BINS + 1];
    derivs[0] = constant;
    derivs[RQS_NUM_BINS] = constant;
    derivs[1..RQS_NUM_BINS].copy_from_slice(unnorm_d);
    rqs_inverse(input, unnorm_w, unnorm_h, &derivs, -tb, tb, -tb, tb)
}

#[allow(clippy::too_many_arguments)]
fn rqs_inverse(
    input: f32,
    unnorm_w: &[f32; RQS_NUM_BINS],
    unnorm_h: &[f32; RQS_NUM_BINS],
    derivatives_unnorm: &[f32; RQS_NUM_BINS + 1],
    left: f32,
    right: f32,
    bottom: f32,
    top: f32,
) -> f32 {
    let n = RQS_NUM_BINS;
    let nf = n as f32;

    let widths_sm = softmax(unnorm_w);
    let mut widths = [0.0_f32; RQS_NUM_BINS];
    for b in 0..n {
        widths[b] = MIN_BIN_WIDTH + (1.0 - MIN_BIN_WIDTH * nf) * widths_sm[b];
    }
    let cumwidths = cumulative(&widths, left, right - left);
    let widths = diffs(&cumwidths);

    let mut derivatives = [0.0_f32; RQS_NUM_BINS + 1];
    for b in 0..=n {
        derivatives[b] = MIN_DERIVATIVE + softplus(derivatives_unnorm[b]);
    }

    let heights_sm = softmax(unnorm_h);
    let mut heights = [0.0_f32; RQS_NUM_BINS];
    for b in 0..n {
        heights[b] = MIN_BIN_HEIGHT + (1.0 - MIN_BIN_HEIGHT * nf) * heights_sm[b];
    }
    let cumheights = cumulative(&heights, bottom, top - bottom);
    let heights = diffs(&cumheights);

    let bin = searchsorted(&cumheights, input);
    let input_cumwidths = cumwidths[bin];
    let input_bin_widths = widths[bin];
    let input_cumheights = cumheights[bin];
    let input_delta = heights[bin] / widths[bin];
    let input_derivatives = derivatives[bin];
    let input_derivatives_plus_one = derivatives[bin + 1];
    let input_heights = heights[bin];

    let dy = input - input_cumheights;
    let a = dy * (input_derivatives + input_derivatives_plus_one - 2.0 * input_delta)
        + input_heights * (input_delta - input_derivatives);
    let b = input_heights * input_derivatives
        - dy * (input_derivatives + input_derivatives_plus_one - 2.0 * input_delta);
    let c = -input_delta * dy;
    let discriminant = (b * b - 4.0 * a * c).max(0.0);
    // WP-12 (2026-08-10): quadratic root sqrt through vokra_math for
    // cross-plat determinism within Vokra (SDP RQS inverse solver).
    let root = 2.0 * c / (-b - vokra_math::sqrt(discriminant));
    root * input_bin_widths + input_cumwidths
}

fn searchsorted(bin_locations: &[f32; RQS_NUM_BINS + 1], input: f32) -> usize {
    let eps = 1e-6;
    let mut count = 0_usize;
    for (i, &loc) in bin_locations.iter().enumerate() {
        let loc = if i == RQS_NUM_BINS { loc + eps } else { loc };
        if input >= loc {
            count += 1;
        }
    }
    count.saturating_sub(1).min(RQS_NUM_BINS - 1)
}

fn cumulative(bins: &[f32; RQS_NUM_BINS], base: f32, span: f32) -> [f32; RQS_NUM_BINS + 1] {
    let mut cum = [0.0_f32; RQS_NUM_BINS + 1];
    let mut acc = 0.0_f32;
    for b in 0..RQS_NUM_BINS {
        acc += bins[b];
        cum[b + 1] = acc;
    }
    for c in cum.iter_mut() {
        *c = span * *c + base;
    }
    cum[0] = base;
    cum[RQS_NUM_BINS] = base + span;
    cum
}

fn diffs(cum: &[f32; RQS_NUM_BINS + 1]) -> [f32; RQS_NUM_BINS] {
    let mut d = [0.0_f32; RQS_NUM_BINS];
    for b in 0..RQS_NUM_BINS {
        d[b] = cum[b + 1] - cum[b];
    }
    d
}

fn softmax(x: &[f32; RQS_NUM_BINS]) -> [f32; RQS_NUM_BINS] {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut out = [0.0_f32; RQS_NUM_BINS];
    let mut sum = 0.0_f32;
    for b in 0..RQS_NUM_BINS {
        // WP-12 (2026-08-10): RQS bin softmax exp through vokra_math for
        // cross-plat determinism within Vokra (SDP RQS width/height/slope).
        out[b] = vokra_math::exp(x[b] - max);
        sum += out[b];
    }
    for v in &mut out {
        *v /= sum;
    }
    out
}

// -----------------------------------------------------------------------------
// SbV2SDP — the full stochastic duration predictor
// -----------------------------------------------------------------------------

/// SBV2 (Style-Bert-VITS2 v2) stochastic duration predictor. Samples one
/// non-negative integer duration per phoneme position by:
///
/// 1. Running `hidden` (row-major `[T, d_hidden]`) through a `pre` 1×1
///    conv → speaker-conditioning add (`cond(g)`) → 3-layer [`DDSConv`] →
///    `proj` 1×1 conv, yielding the SDP body `[d_hidden, T]` channel-major.
/// 2. Sampling a 2-channel Gaussian latent `z ~ N(0, 1) * noise_scale_w`
///    (deterministic all-zero when `noise_scale_w == 0.0`).
/// 3. Walking the reverse flow — mirror of upstream
///    `StochasticDurationPredictor.forward(reverse=True)`'s
///    `flows = list(reversed(self.flows)); flows = flows[:-2] +
///    [flows[-1]]` slice: iterate `self.flows[1..]` in reverse (dropping
///    the "useless vflow" at `self.flows[0]` = upstream `sdp.flows.1`),
///    calling `flip(z); z = flow.reverse(z, cond=body)` for each; then a
///    final `flip(z)`; then the ElementwiseAffine reverse. Skipping the
///    first ConvFlow matches upstream — the layer receives no gradients
///    during training and applying it at inference produces divergent
///    log-durations. See [`sample`](Self::sample)'s in-body comment for
///    the full trace.
/// 4. Returning `duration[p] = exp(z[0, p]).ceil().max(1)` as `i32`
///    (VITS-family log-duration convention — mirrors
///    `piper_plus::duration::DurationPredictor`'s caller-side
///    `logw.exp() * length_scale).ceil().max(1)`).
///
/// # Empty / synthetic-test path
///
/// [`SbV2SDP::empty`] returns an SDP with `flows = vec![]` and every conv
/// weight/bias zero (identity `ElementwiseAffine`, zero-weight `DDSConv`).
/// Combined with `noise_scale_w == 0.0`, that path returns all-`1`s — the
/// same behaviour the pre-Blocker-2c `SbV2SDP::from_weights(Vec::new(), ..)`
/// scaffold provided (`SbV2Model::synthetic_for_test`'s dependent).
pub struct SbV2SDP {
    d_hidden: usize,
    /// Global-conditioning width (speaker embedding). Equal to `d_speaker`
    /// in `SbV2Model::from_gguf`; in the real SBV2 v2 base it is 512.
    gin: usize,
    // Body:
    pre_w: Vec<f32>,  // `[d_hidden, d_hidden, 1]` — 1x1 conv
    pre_b: Vec<f32>,  // `[d_hidden]`
    convs: DDSConv,   // channels = d_hidden
    cond_w: Vec<f32>, // `[d_hidden, gin, 1]` — 1x1 conv (speaker cond)
    cond_b: Vec<f32>, // `[d_hidden]`
    proj_w: Vec<f32>, // `[d_hidden, d_hidden, 1]`
    proj_b: Vec<f32>, // `[d_hidden]`
    // Flow stack:
    ea: ElementwiseAffine,
    /// ConvFlows stored in **forward-index order** — i.e. `flows[0]` maps
    /// to upstream `sdp.flows.1` (the "useless vflow" skipped at inference),
    /// `flows[1]` to `sdp.flows.3`, `flows[2]` to `sdp.flows.5`, `flows[3]`
    /// to `sdp.flows.7`. See [`sample`](Self::sample) for the reverse-order
    /// walk that mirrors upstream's `list(reversed(...))[:-2] + [flows[-1]]`
    /// slice. The `Vec` is loader-order to keep the converter's dense re-
    /// index `sbv2.sdp.flow.{0..n}` load path a straight `for i in 0..n`
    /// with no additional `rev()` at load time — reversing at `sample` time
    /// keeps the load path a simple `push`.
    flows: Vec<ConvFlow>,
}

impl SbV2SDP {
    /// Assembles an SDP from real GGUF-loaded weights. Every shape is
    /// documented on the struct's field list. Public so the from_gguf
    /// loader (`sbv2/mod.rs`) can call it.
    #[allow(clippy::too_many_arguments)]
    pub fn from_weights(
        d_hidden: usize,
        gin: usize,
        pre_w: Vec<f32>,
        pre_b: Vec<f32>,
        convs: DDSConv,
        cond_w: Vec<f32>,
        cond_b: Vec<f32>,
        proj_w: Vec<f32>,
        proj_b: Vec<f32>,
        ea: ElementwiseAffine,
        flows: Vec<ConvFlow>,
    ) -> Self {
        debug_assert_eq!(pre_w.len(), d_hidden * d_hidden, "pre.weight shape");
        debug_assert_eq!(pre_b.len(), d_hidden, "pre.bias shape");
        debug_assert_eq!(cond_w.len(), d_hidden * gin, "cond.weight shape");
        debug_assert_eq!(cond_b.len(), d_hidden, "cond.bias shape");
        debug_assert_eq!(proj_w.len(), d_hidden * d_hidden, "proj.weight shape");
        debug_assert_eq!(proj_b.len(), d_hidden, "proj.bias shape");
        debug_assert_eq!(convs.channels, d_hidden, "convs channels");
        for flow in &flows {
            debug_assert_eq!(
                flow.dp_filter, d_hidden,
                "every ConvFlow's dp_filter must equal the SDP's d_hidden"
            );
        }
        Self {
            d_hidden,
            gin,
            pre_w,
            pre_b,
            convs,
            cond_w,
            cond_b,
            proj_w,
            proj_b,
            ea,
            flows,
        }
    }

    /// Zero-weight identity SDP: an [`ElementwiseAffine::identity`], a
    /// zero-weight [`DDSConv::zero`], every conv weight/bias zero, and
    /// `flows = vec![]`. Combined with `noise_scale_w == 0.0`,
    /// [`sample`](Self::sample) returns all-`1`s (`exp(0).ceil().max(1) ==
    /// 1`) regardless of `hidden`/`g`/RNG state — the documented
    /// no-op path `SbV2Model::synthetic_for_test` (`sbv2/mod.rs`) relies
    /// on.
    pub fn empty(d_hidden: usize, gin: usize) -> Self {
        Self::from_weights(
            d_hidden,
            gin,
            vec![0.0; d_hidden * d_hidden],
            vec![0.0; d_hidden],
            DDSConv::zero(d_hidden, DP_CONV_LAYERS, DP_KERNEL),
            vec![0.0; d_hidden * gin],
            vec![0.0; d_hidden],
            vec![0.0; d_hidden * d_hidden],
            vec![0.0; d_hidden],
            ElementwiseAffine::identity(),
            Vec::new(),
        )
    }

    /// SDP `d_hidden` (== `dp_filter` in the safetensors, == `d_model` in
    /// [`SbV2Model`](super::SbV2Model)'s single-hidden-width scaffold).
    pub fn d_hidden(&self) -> usize {
        self.d_hidden
    }

    /// Speaker-embedding width (`gin` in upstream models.py; `d_speaker` in
    /// `SbV2Model::from_gguf`'s metadata).
    pub fn gin(&self) -> usize {
        self.gin
    }

    /// Number of stored ConvFlows (real SBV2 v2 base has 4; empty SDP has
    /// 0). ElementwiseAffine slot 0 is not counted here — it is a separate
    /// mandatory field, not a variable-length flow list entry.
    pub fn n_conv_flows(&self) -> usize {
        self.flows.len()
    }

    /// SDP body: transposes `hidden` from row-major `[T, d_hidden]` into
    /// channel-major `[d_hidden, T]`, then runs pre → +cond(g) → DDS →
    /// proj, returning `[d_hidden, T]` channel-major.
    pub(crate) fn body(
        &self,
        hidden_row_major: &[f32],
        text_seq_len: usize,
        g: &[f32],
    ) -> Vec<f32> {
        debug_assert_eq!(hidden_row_major.len(), text_seq_len * self.d_hidden);
        debug_assert_eq!(g.len(), self.gin);
        let d = self.d_hidden;

        // Transpose [T, D] row-major -> [D, T] channel-major.
        let mut x_dp = vec![0.0_f32; d * text_seq_len];
        for t in 0..text_seq_len {
            for c in 0..d {
                x_dp[c * text_seq_len + t] = hidden_row_major[t * d + c];
            }
        }

        // pre.
        let x = conv1d_scalar(
            &x_dp,
            d,
            text_seq_len,
            &self.pre_w,
            d,
            1, // kernel
            Some(&self.pre_b),
            0, // pad
            1, // dilation
            1, // groups
        );

        // cond(g): treat g as a [gin, 1] "1-step" input, produce cg [d, 1],
        // then broadcast-add across every time step. Equivalent to running
        // conv1d over [gin, T] where every column is `g`.
        let cg = conv1d_scalar(
            g,
            self.gin,
            1, // in_len = 1 (broadcast axis)
            &self.cond_w,
            d,
            1, // kernel
            Some(&self.cond_b),
            0, // pad
            1, // dilation
            1, // groups
        );
        debug_assert_eq!(cg.len(), d);
        let mut x = x;
        for c in 0..d {
            let v = cg[c];
            for t in 0..text_seq_len {
                x[c * text_seq_len + t] += v;
            }
        }

        // DDS (no additional conditioning at the body level — g was already
        // folded above; upstream `models.py::SDP.forward` matches this
        // shape).
        let x = self.convs.forward(&x, text_seq_len, None);

        // proj.
        conv1d_scalar(
            &x,
            d,
            text_seq_len,
            &self.proj_w,
            d,
            1, // kernel
            Some(&self.proj_b),
            0, // pad
            1, // dilation
            1, // groups
        )
    }

    /// Backend-dispatched SDP conditioner used by native GPU-capable models.
    pub fn body_with_compute(
        &self,
        compute: &Compute,
        hidden_row_major: &[f32],
        text_seq_len: usize,
        global_conditioning: &[f32],
    ) -> Result<Vec<f32>> {
        debug_assert_eq!(hidden_row_major.len(), text_seq_len * self.d_hidden);
        debug_assert_eq!(global_conditioning.len(), self.gin);
        let hidden_channel_major =
            transpose_position_major(hidden_row_major, text_seq_len, self.d_hidden);
        let mut hidden = conv1d_with_compute(
            compute,
            &hidden_channel_major,
            self.d_hidden,
            text_seq_len,
            &self.pre_w,
            self.d_hidden,
            1,
            Some(&self.pre_b),
            0,
            1,
            1,
        )?;
        let conditioning = conv1d_with_compute(
            compute,
            global_conditioning,
            self.gin,
            1,
            &self.cond_w,
            self.d_hidden,
            1,
            Some(&self.cond_b),
            0,
            1,
            1,
        )?;
        for channel in 0..self.d_hidden {
            for position in 0..text_seq_len {
                hidden[channel * text_seq_len + position] += conditioning[channel];
            }
        }
        let hidden = self
            .convs
            .forward_with_compute(compute, &hidden, text_seq_len, None)?;
        conv1d_with_compute(
            compute,
            &hidden,
            self.d_hidden,
            text_seq_len,
            &self.proj_w,
            self.d_hidden,
            1,
            Some(&self.proj_b),
            0,
            1,
            1,
        )
    }

    /// Samples one duration per phoneme position. See the struct-level doc
    /// for the exact 4-step algorithm.
    ///
    /// `hidden` is row-major `[text_seq_len, d_hidden]`. `g` is
    /// `[gin]` — the raw speaker embedding, not broadcast-added into
    /// `hidden` (this differs architecturally from the pre-Blocker-2c
    /// scaffold; see this module's top-level doc). `noise_scale_w` scales
    /// the Gaussian latent; `noise_scale_w == 0.0` makes every draw
    /// exactly `0.0`.
    ///
    /// # Panics
    ///
    /// Panics (via `debug_assert!`) if `hidden.len() != text_seq_len *
    /// self.d_hidden` or `g.len() != self.gin`.
    pub fn sample<R: NormalSource>(
        &self,
        hidden: &[f32],
        text_seq_len: usize,
        g: &[f32],
        rng: &mut R,
        noise_scale_w: f32,
    ) -> Vec<i32> {
        debug_assert_eq!(hidden.len(), text_seq_len * self.d_hidden);
        debug_assert_eq!(g.len(), self.gin);

        if text_seq_len == 0 {
            return Vec::new();
        }

        // 1. Body conditioner.
        let body = self.body(hidden, text_seq_len, g);

        // 2. Latent `z` [2, T].
        let mut z = vec![0.0_f32; 2 * text_seq_len];
        if noise_scale_w != 0.0 {
            // Per-sample streaming fill via `NormalSource::next_normal`
            // — see `SbV2SynthRequest::rng_mode` for the two impls
            // this can be (`GaussianSplitMix64` for legacy synthetic
            // tests, `TorchRandnStream` for torch parity).
            //
            // # Why per-sample streaming (not `rng.fill(&mut z)`)
            //
            // The `fill` trait method, when overridden by
            // `TorchRandnStream`, dispatches to torch's `normal_fill`
            // fast path (`normal_fill_16_scalar` in 16-wide blocks).
            // That path is correct — but on x86_64 CI hosts, torch
            // itself dispatches to `normal_fill_AVX2` which uses
            // `avx_mathfun`'s vectorized `log256_ps` / `sincos256_ps`
            // approximations (~1 ULP off from libm scalar). SBV2's
            // downstream flow inverse amplifies that residual
            // chaotically, giving CI parity mismatches that are not
            // bugs but genuine AVX2-vs-scalar micro-differences.
            //
            // The Python SBV2 reference dumper
            // (`tools/parity/sbv2_dump_reference.py`) works around
            // this by constructing a NON-contiguous tensor for the
            // SDP noise, which forces torch's `normal_kernel` to take
            // the `else` branch (line 246) —
            // `at::normal_distribution<double>` streaming with pair
            // caching. Vokra's `TorchRandnStream::next_normal` is a
            // bit-exact port of that same streaming path (verified
            // by `crates/vokra-core/tests/rng_torch_randn_cpu_parity.rs`),
            // so a per-sample loop here matches torch's dumper output
            // byte-for-byte on every CPU host regardless of AVX2
            // support. See that dumper file's inline note for the
            // full rationale.
            //
            // The `*v *= noise_scale_w` step matches Python's
            // `z * noise_scale_w` — a post-fill f32 multiply, not
            // baked into the RNG (torch.randn's default is std=1).
            for v in &mut z {
                *v = rng.next_normal();
            }
            for v in &mut z {
                *v *= noise_scale_w;
            }
        }

        // 3. Reverse flow — matches upstream
        // `models.py::StochasticDurationPredictor.forward(reverse=True)`:
        //
        //     flows = list(reversed(self.flows))
        //     flows = flows[:-2] + [flows[-1]]  # remove a useless vflow
        //     for flow in flows:
        //         z = flow(z, x_mask, g=x, reverse=reverse)
        //
        // Upstream `self.flows` is `[EA, CF, Flip, CF, Flip, CF, Flip, CF, Flip]`
        // (9 items — 1 ElementwiseAffine + 4 (ConvFlow, Flip) pairs). Reversed
        // then `[:-2] + [-1]` drops the SECOND-TO-LAST item of the reversed
        // list, which is the FIRST ConvFlow in forward order (upstream
        // `sdp.flows.1`). At inference the "useless vflow" is skipped because
        // it never received gradients (upstream only trains through the flow
        // path when computing NLL in the non-reverse branch, and index-1 is
        // the layer that ends up shielded by the interleaved Flips).
        //
        // Layout: the converter densifies upstream `sdp.flows.{1, 3, 5, 7}`
        // into `sbv2.sdp.flow.{0, 1, 2, 3}`, and the loader pushes them in
        // that order, so `self.flows[0]` == upstream `sdp.flows.1` == the
        // useless vflow that must be skipped, and `self.flows[3]` == upstream
        // `sdp.flows.7` == the first ConvFlow applied at inference. We walk
        // `flows[1..]` in reverse (giving `flows[3], flows[2], flows[1]`),
        // each preceded by a Flip, then one final Flip, then EA reverse.
        // Sequence for n_flows=4: `Flip, flows[3], Flip, flows[2], Flip,
        // flows[1], Flip, EA` = 4 Flips + 3 ConvFlows + 1 EA = 8 items,
        // matching upstream's combined list length exactly.
        //
        // n_flows == 0 (empty SDP for `SbV2SDP::empty`) skips the whole
        // Flip/ConvFlow chain and jumps straight to EA — matches upstream
        // when `n_flows=0` reduces the 9-item list to `[EA]` and the
        // `[:-2] + [-1]` slice degenerates to `[EA]`.
        if let Some((_useless_vflow, rest)) = self.flows.split_first() {
            for flow in rest.iter().rev() {
                flip2(&mut z, text_seq_len);
                z = flow.reverse(&z, text_seq_len, &body);
            }
            flip2(&mut z, text_seq_len);
        }
        let z = self.ea.reverse(&z, text_seq_len);

        // 4. logw = z[0, :]; duration = ceil(exp(logw)).max(1).
        //
        // WP-12 (2026-08-10) CRITICAL PATH: this exp gates the integer
        // duration quantization. Wave-1 investigation flagged this as the
        // one site where cross-plat 1-ULP scatter could shift an integer
        // by 1, which propagates as a whole-frame duration shift downstream
        // (mel_seq_len changes → decoder output length changes → NO atol
        // can absorb the shape divergence). Routing through vokra_math::exp
        // ensures cross-plat deterministic input to ceil() — the integer
        // output is now guaranteed identical across Linux/macOS/Windows
        // for any given logw.
        z.iter()
            .take(text_seq_len) // channel 0 spans the first text_seq_len entries
            .map(|&logw| vokra_math::exp(logw).ceil().max(1.0) as i32)
            .collect()
    }

    /// Backend-dispatched duration sampling. Random-number generation and
    /// spline control math remain deterministic host glue; all learned
    /// convolutions, normalizations and GELUs use `compute`.
    pub fn sample_with_compute<R: NormalSource>(
        &self,
        compute: &Compute,
        hidden: &[f32],
        text_seq_len: usize,
        global_conditioning: &[f32],
        rng: &mut R,
        noise_scale_w: f32,
    ) -> Result<Vec<i32>> {
        Ok(self
            .sample_log_duration_with_compute(
                compute,
                hidden,
                text_seq_len,
                global_conditioning,
                rng,
                noise_scale_w,
            )?
            .into_iter()
            .map(|log_duration| vokra_math::exp(log_duration).ceil().max(1.0) as i32)
            .collect())
    }

    /// Returns the stochastic predictor's raw log-duration samples before
    /// exponentiation. MeloTTS blends these with its deterministic predictor
    /// in log space through `sdp_ratio`.
    pub fn sample_log_duration_with_compute<R: NormalSource>(
        &self,
        compute: &Compute,
        hidden: &[f32],
        text_seq_len: usize,
        global_conditioning: &[f32],
        rng: &mut R,
        noise_scale_w: f32,
    ) -> Result<Vec<f32>> {
        debug_assert_eq!(hidden.len(), text_seq_len * self.d_hidden);
        debug_assert_eq!(global_conditioning.len(), self.gin);
        if text_seq_len == 0 {
            return Ok(Vec::new());
        }
        let body = self.body_with_compute(compute, hidden, text_seq_len, global_conditioning)?;
        let mut latent = vec![0.0; 2 * text_seq_len];
        if noise_scale_w != 0.0 {
            for value in &mut latent {
                *value = rng.next_normal() * noise_scale_w;
            }
        }
        if let Some((_unused, rest)) = self.flows.split_first() {
            for flow in rest.iter().rev() {
                flip2(&mut latent, text_seq_len);
                latent = flow.reverse_with_compute(compute, &latent, text_seq_len, &body)?;
            }
            flip2(&mut latent, text_seq_len);
        }
        let latent = self.ea.reverse(&latent, text_seq_len);
        Ok(latent[..text_seq_len].to_vec())
    }
}

// -----------------------------------------------------------------------------
// length_regulate — unchanged from the pre-Blocker-2c module; kept public.
// -----------------------------------------------------------------------------

/// Expands per-phoneme hidden states into a mel-frame timeline: phoneme
/// `i`'s `[d_model]` row is repeated `durations[i]` times, and every
/// repeated row is appended in phoneme order — the standard VITS-family
/// length-regulator step that consumes [`SbV2SDP::sample`]'s output.
/// Row-major/position-major layout throughout (see this module's doc's
/// "Layout conventions" section).
///
/// Non-positive durations (`0` or negative) contribute **no** output rows
/// for that phoneme, rather than being cast to `usize`: a negative `i32`
/// duration cast directly to `usize` would wrap to an enormous repeat
/// count (a silent out-of-memory/panic hazard, not a legitimate request),
/// so this function skips any `durations[i] <= 0` instead of casting it.
///
/// # Panics
///
/// Panics in debug builds if `hidden.len() != durations.len() * d_model`.
pub fn length_regulate(hidden: &[f32], durations: &[i32], d_model: usize) -> Vec<f32> {
    debug_assert_eq!(
        hidden.len(),
        durations.len() * d_model,
        "hidden must be [durations.len(), d_model]"
    );
    let total_frames: usize = durations
        .iter()
        .filter(|&&dur| dur > 0)
        .map(|&dur| dur as usize)
        .sum();
    let mut out = Vec::with_capacity(total_frames * d_model);
    for (i, &dur) in durations.iter().enumerate() {
        if dur > 0 {
            let row = &hidden[i * d_model..(i + 1) * d_model];
            for _ in 0..dur {
                out.extend_from_slice(row);
            }
        }
    }
    out
}

// -----------------------------------------------------------------------------
// Internal unit tests — round-trip sanity for the primitives, exercised
// beyond what `tests/sbv2_sdp.rs` covers externally (this crate's
// `pub(crate)` helpers cannot be reached from the external test target).
// -----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    // `GaussianSplitMix64` is the concrete NormalSource these tests happened
    // to use before the Step 8 generic refactor. Kept here so the tests
    // continue to exercise the exact same numeric stream byte-for-byte
    // (NFR-PT-01 cross-build non-interference — this refactor changes
    // nothing observable to the tests).
    use vokra_core::rng::GaussianSplitMix64;

    #[test]
    fn softplus_matches_reference_at_typical_inputs() {
        // Reference values from Python's `math.log(1 + math.exp(x))`:
        //  f(0.0) = ln 2                → checked via `f32::consts::LN_2`
        //  f(1.0) = ln(1 + e) ≈ 1.3132617
        //  f(-1.0) ≈ 0.3132617
        for (x, want) in &[
            (0.0_f32, std::f32::consts::LN_2),
            (1.0, 1.313_261_7),
            (-1.0, 0.313_261_7),
        ] {
            let got = softplus(*x);
            assert!(
                (got - want).abs() < 1e-5,
                "softplus({x}) got {got}, want {want}"
            );
        }
        // Large-x guard: >20 short-circuits to x itself (no overflow).
        assert_eq!(softplus(30.0), 30.0);
    }

    #[test]
    fn erf_symmetric_within_tolerance() {
        // `erf` is odd — `erf(-x) = -erf(x)`.
        for x in [0.1_f32, 0.5, 1.0, 2.0, 3.0] {
            assert!(
                (erf(-x) + erf(x)).abs() < 1e-6,
                "erf({x}) not antisymmetric"
            );
        }
        // Reference: erf(1) ≈ 0.8427007929.
        assert!((erf(1.0) - 0.8427008).abs() < 1e-5);
    }

    #[test]
    fn conv1d_scalar_matches_hand_computed_1x1() {
        // Simplest case: 1×1 pointwise conv over [2, 3] channel-major input,
        // producing [2, 3] channel-major output. Weight [2, 2, 1] row-major
        // = [[w00, w01], [w10, w11]].
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [2, 3]
        let w = vec![1.0, 0.5, 0.0, 2.0]; // out=2, in=2, kernel=1
        let b = vec![0.5, -0.5];
        let out = conv1d_scalar(&x, 2, 3, &w, 2, 1, Some(&b), 0, 1, 1);
        // out[0, t] = 1.0 * x[0,t] + 0.5 * x[1,t] + 0.5
        // out[1, t] = 0.0 * x[0,t] + 2.0 * x[1,t] - 0.5
        // t=0: out[0]=1+2+0.5=3.5, out[1]=0+8-0.5=7.5
        assert!((out[0] - 3.5).abs() < 1e-6);
        assert!((out[3] - 7.5).abs() < 1e-6);
        // t=2: out[0]=3+3+0.5=6.5, out[1]=0+12-0.5=11.5
        assert!((out[2] - 6.5).abs() < 1e-6);
        assert!((out[5] - 11.5).abs() < 1e-6);
    }

    #[test]
    fn conv1d_scalar_depthwise_isolates_channels() {
        // groups = channels = 2, kernel=3, so each output channel sees only
        // its own input channel — a fingerprint depthwise conv should NOT
        // mix channel 0 and channel 1.
        let x = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]; // [2, 3]: chan 0 = [1,0,0], chan 1 = [0,1,0]
        // Weight [2, 1, 3]: out=2, in_g=1, k=3. Channel 0 kernel = [1,2,3];
        // channel 1 kernel = [4,5,6].
        let w = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        // Same-padding for kernel=3, dilation=1: pad=1.
        let out = conv1d_scalar(&x, 2, 3, &w, 2, 3, None, 1, 1, 2);
        // Channel 0 output: convolve [1,0,0] with [1,2,3], pad=1
        //   t=0: 0*1 + 1*2 + 0*3 = 2
        //   t=1: 1*1 + 0*2 + 0*3 = 1
        //   t=2: 0*1 + 0*2 + 0*3 = 0
        assert_eq!(&out[..3], &[2.0, 1.0, 0.0]);
        // Channel 1 output: convolve [0,1,0] with [4,5,6], pad=1
        //   t=0: 0*4 + 0*5 + 1*6 = 6
        //   t=1: 0*4 + 1*5 + 0*6 = 5
        //   t=2: 1*4 + 0*5 + 0*6 = 4
        assert_eq!(&out[3..], &[6.0, 5.0, 4.0]);
    }

    #[test]
    fn dilated_depthwise_compute_matches_scalar_reference() {
        let channels = 2;
        let time = 7;
        let kernel = 3;
        let dilation = 3;
        let padding = dilation * (kernel - 1) / 2;
        let input: Vec<f32> = (0..channels * time)
            .map(|index| index as f32 * 0.07 - 0.3)
            .collect();
        let weight = vec![0.2, -0.4, 0.7, -0.1, 0.5, 0.3];
        let bias = vec![0.05, -0.2];
        let reference = conv1d_scalar(
            &input,
            channels,
            time,
            &weight,
            channels,
            kernel,
            Some(&bias),
            padding,
            dilation,
            channels,
        );
        let actual = conv1d_with_compute(
            &Compute::cpu(),
            &input,
            channels,
            time,
            &weight,
            channels,
            kernel,
            Some(&bias),
            padding,
            dilation,
            channels,
        )
        .unwrap();
        for (index, (actual, reference)) in actual.iter().zip(&reference).enumerate() {
            assert!(
                (actual - reference).abs() <= 1e-6,
                "dilated depthwise mismatch at {index}: actual={actual}, reference={reference}"
            );
        }
    }

    #[test]
    fn dds_compute_matches_scalar_with_conditioning_and_asymmetric_weights() {
        let channels = 2;
        let time = 5;
        let dds = DDSConv::from_weights(
            channels,
            1,
            3,
            vec![vec![0.2, -0.4, 0.7, -0.3, 0.5, 0.9]],
            vec![vec![0.11, -0.23]],
            vec![vec![0.6, -0.8, 0.25, 0.45]],
            vec![vec![-0.07, 0.19]],
            vec![SdpLayerNorm {
                gamma: vec![1.2, -0.6],
                beta: vec![0.13, -0.17],
            }],
            vec![SdpLayerNorm {
                gamma: vec![0.8, 1.4],
                beta: vec![-0.05, 0.09],
            }],
        );
        let input: Vec<f32> = (0..channels * time)
            .map(|i| (i as f32 * 0.31) - 0.9)
            .collect();
        let conditioning: Vec<f32> = (0..channels * time)
            .map(|i| 0.4 - i as f32 * 0.13)
            .collect();
        let expected = dds.forward(&input, time, Some(&conditioning));
        let actual = dds
            .forward_with_compute(&Compute::cpu(), &input, time, Some(&conditioning))
            .expect("CPU Compute DDS");
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= 2e-6,
                "DDS mismatch at {index}: {actual} != {expected}"
            );
        }
    }

    #[test]
    fn element_wise_affine_reverse_matches_hand_computed() {
        // m = [1.0, -2.0], logs = [ln 2, 0.0].
        // reverse: z0 = (x0 - 1.0) * exp(-ln 2) = (x0 - 1.0) / 2
        //          z1 = (x1 - (-2.0)) * exp(0)  = x1 + 2.0
        let ea = ElementwiseAffine::from_weights(vec![1.0, -2.0], vec![2.0_f32.ln(), 0.0]);
        let time = 2;
        let x = vec![
            // channel 0
            3.0, 5.0, // channel 1
            0.5, -1.0,
        ];
        let out = ea.reverse(&x, time);
        assert!((out[0] - 1.0).abs() < 1e-6, "got {}", out[0]); // (3-1)/2 = 1
        assert!((out[1] - 2.0).abs() < 1e-6, "got {}", out[1]); // (5-1)/2 = 2
        assert!((out[2] - 2.5).abs() < 1e-6, "got {}", out[2]); // 0.5 + 2 = 2.5
        assert!((out[3] - 1.0).abs() < 1e-6, "got {}", out[3]); // -1 + 2 = 1
    }

    #[test]
    fn dds_conv_zero_weights_is_identity() {
        // Every layer's y is zero (zero convs → zero LayerNorm affine
        // → zero everywhere), so the residual `x += y` is a no-op and
        // the whole DDS is the identity.
        let channels = 3;
        let time = 4;
        let dds = DDSConv::zero(channels, 3, 3);
        let x: Vec<f32> = (0..channels * time).map(|i| i as f32 * 0.1).collect();
        let out = dds.forward(&x, time, None);
        for (a, b) in out.iter().zip(&x) {
            assert!((a - b).abs() < 1e-5, "zero-weight DDS must be identity");
        }
    }

    #[test]
    fn sdp_empty_returns_ones_deterministic() {
        // Empty SDP + `noise_scale_w = 0.0` = every duration is 1.
        let sdp = SbV2SDP::empty(4, 4);
        let hidden = vec![0.5_f32; 3 * 4];
        let g = vec![0.7_f32; 4];
        let mut rng = GaussianSplitMix64::new(0);
        let out = sdp.sample(&hidden, 3, &g, &mut rng, 0.0);
        assert_eq!(out, vec![1_i32; 3]);
    }

    #[test]
    fn sdp_compute_path_matches_scalar_empty_reference() {
        let sdp = SbV2SDP::empty(4, 4);
        let hidden = vec![0.5; 3 * 4];
        let global = vec![0.7; 4];
        let mut reference_rng = GaussianSplitMix64::new(17);
        let mut compute_rng = GaussianSplitMix64::new(17);
        let reference = sdp.sample(&hidden, 3, &global, &mut reference_rng, 0.6);
        let actual = sdp
            .sample_with_compute(&Compute::cpu(), &hidden, 3, &global, &mut compute_rng, 0.6)
            .unwrap();
        assert_eq!(actual, reference);
    }

    #[test]
    fn sdp_empty_with_biased_ea_produces_fixed_duration() {
        // Empty flows + noise_scale_w = 0 leaves z = [0, 0], the final
        // flip is a no-op (both halves are zero), and EA reverse with
        // m = [-ln(40), 0], logs = [0, 0] maps z[0]=0 -> (0 - (-ln 40)) *
        // exp(0) = ln(40); z[1]=0 -> 0. So every duration is exp(ln 40)
        // ceil-ed to 40.
        let d_hidden = 4;
        let gin = 4;
        let ea = ElementwiseAffine::from_weights(vec![-40.0_f32.ln(), 0.0], vec![0.0, 0.0]);
        let sdp = SbV2SDP::from_weights(
            d_hidden,
            gin,
            vec![0.0; d_hidden * d_hidden],
            vec![0.0; d_hidden],
            DDSConv::zero(d_hidden, DP_CONV_LAYERS, DP_KERNEL),
            vec![0.0; d_hidden * gin],
            vec![0.0; d_hidden],
            vec![0.0; d_hidden * d_hidden],
            vec![0.0; d_hidden],
            ea,
            Vec::new(),
        );
        let hidden = vec![0.0_f32; 3 * d_hidden];
        let g = vec![0.0_f32; gin];
        let mut rng = GaussianSplitMix64::new(0);
        let out = sdp.sample(&hidden, 3, &g, &mut rng, 0.0);
        assert_eq!(out, vec![40_i32; 3]);
    }

    #[test]
    fn flip2_involution() {
        let mut x = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // [2, 3]
        let orig = x.clone();
        flip2(&mut x, 3);
        assert_eq!(x, vec![4.0, 5.0, 6.0, 1.0, 2.0, 3.0]);
        flip2(&mut x, 3);
        assert_eq!(x, orig, "flip2 is its own inverse");
    }

    #[test]
    fn rqs_inverse_identity_outside_tails() {
        let w = [0.0_f32; RQS_NUM_BINS];
        let h = [0.0_f32; RQS_NUM_BINS];
        let d = [0.0_f32; RQS_NUM_BINS - 1];
        assert_eq!(unconstrained_rqs_inverse(7.0, &w, &h, &d), 7.0);
        assert_eq!(unconstrained_rqs_inverse(-6.0, &w, &h, &d), -6.0);
    }

    #[test]
    fn sample_deterministic_for_fixed_seed() {
        // With noise_scale_w != 0, `sample` reads the RNG; same seed →
        // same output.
        let sdp = SbV2SDP::empty(2, 2);
        let hidden = vec![0.0_f32; 3 * 2];
        let g = vec![0.0_f32; 2];
        let out1 = {
            let mut rng = GaussianSplitMix64::new(7);
            sdp.sample(&hidden, 3, &g, &mut rng, 0.6)
        };
        let out2 = {
            let mut rng = GaussianSplitMix64::new(7);
            sdp.sample(&hidden, 3, &g, &mut rng, 0.6)
        };
        assert_eq!(out1, out2, "same seed must give same durations");
    }
}
