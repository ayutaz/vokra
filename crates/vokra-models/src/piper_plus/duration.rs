//! Stochastic duration predictor (M0-07-T14): phoneme features → log-durations.
//!
//! Follows piper-plus `vits/models.py::StochasticDurationPredictor` (reverse /
//! inference path) and `vits/modules.py::{DDSConv, ConvFlow, ElementwiseAffine}`
//! plus the rational-quadratic-spline transform in `vits/transforms.py`. With
//! the stochastic noise disabled (`noise_w = 0`, the parity determinism knob),
//! the reverse flow is deterministic: a zero latent is pushed back through
//! `[Flip, ConvFlow7, Flip, ConvFlow5, Flip, ConvFlow3, Flip,
//! ElementwiseAffine0]` (the "useless" `ConvFlow1` is dropped in inference —
//! `docs/piper-plus-integration.md` §4/§9), and `logw = z0`.

use super::config::{DP_CONV_LAYERS, DP_KERNEL, Dims, RQS_NUM_BINS, RQS_TAIL_BOUND};
use super::nn;
use super::weights::TensorStore;
use vokra_core::Result;

const MIN_BIN_WIDTH: f32 = 1e-3;
const MIN_BIN_HEIGHT: f32 = 1e-3;
const MIN_DERIVATIVE: f32 = 1e-3;

/// LayerNorm affine params.
struct Norm {
    gamma: Vec<f32>,
    beta: Vec<f32>,
}

/// Dilated depth-separable conv stack (`DDSConv`): depthwise sep conv →
/// LayerNorm → GELU → 1×1 conv → LayerNorm → GELU, residual, ×`n_layers`.
struct DdsConv {
    convs_sep: Vec<(Vec<f32>, Vec<f32>)>, // depthwise [C,1,k]
    convs_1x1: Vec<(Vec<f32>, Vec<f32>)>, // [C,C,1]
    norms_1: Vec<Norm>,
    norms_2: Vec<Norm>,
    channels: usize,
}

impl DdsConv {
    fn load(store: &TensorStore, prefix: &str, channels: usize) -> Result<Self> {
        let mut convs_sep = Vec::new();
        let mut convs_1x1 = Vec::new();
        let mut norms_1 = Vec::new();
        let mut norms_2 = Vec::new();
        for i in 0..DP_CONV_LAYERS {
            convs_sep.push((
                store.tensor_shaped(
                    &format!("{prefix}.convs_sep.{i}.weight"),
                    &[channels, 1, DP_KERNEL],
                )?,
                store.tensor_shaped(&format!("{prefix}.convs_sep.{i}.bias"), &[channels])?,
            ));
            convs_1x1.push((
                store.tensor_shaped(
                    &format!("{prefix}.convs_1x1.{i}.weight"),
                    &[channels, channels, 1],
                )?,
                store.tensor_shaped(&format!("{prefix}.convs_1x1.{i}.bias"), &[channels])?,
            ));
            norms_1.push(load_norm(
                store,
                &format!("{prefix}.norms_1.{i}"),
                channels,
            )?);
            norms_2.push(load_norm(
                store,
                &format!("{prefix}.norms_2.{i}"),
                channels,
            )?);
        }
        Ok(Self {
            convs_sep,
            convs_1x1,
            norms_1,
            norms_2,
            channels,
        })
    }

    /// `g` (when `Some`) is added to the input first (the ConvFlow conditioning).
    fn forward(&self, x: &[f32], t: usize, g: Option<&[f32]>) -> Vec<f32> {
        let c = self.channels;
        let mut x = match g {
            Some(g) => x.iter().zip(g).map(|(a, b)| a + b).collect::<Vec<_>>(),
            None => x.to_vec(),
        };
        for i in 0..DP_CONV_LAYERS {
            let dilation = DP_KERNEL.pow(i as u32); // 1, 3, 9
            let pad = dilation * (DP_KERNEL - 1) / 2;
            let (sw, sb) = &self.convs_sep[i];
            // Depthwise (groups = channels).
            let (mut y, _) = nn::conv1d(&x, c, t, sw, c, DP_KERNEL, Some(sb), 1, pad, dilation, c);
            y = nn::layer_norm_channels(
                &y,
                c,
                t,
                &self.norms_1[i].gamma,
                &self.norms_1[i].beta,
                super::config::LAYER_NORM_EPS,
            );
            for v in &mut y {
                *v = nn::gelu(*v);
            }
            let (cw, cb) = &self.convs_1x1[i];
            let (mut y2, _) = nn::conv1d(&y, c, t, cw, c, 1, Some(cb), 1, 0, 1, 1);
            y2 = nn::layer_norm_channels(
                &y2,
                c,
                t,
                &self.norms_2[i].gamma,
                &self.norms_2[i].beta,
                super::config::LAYER_NORM_EPS,
            );
            for v in &mut y2 {
                *v = nn::gelu(*v);
            }
            for (xv, yv) in x.iter_mut().zip(&y2) {
                *xv += yv;
            }
        }
        x
    }
}

/// A spline coupling flow (`ConvFlow`) over the 2-channel duration latent.
struct ConvFlow {
    pre: (Vec<f32>, Vec<f32>),  // [dp_filter, 1, 1]
    convs: DdsConv,             // dp_filter channels
    proj: (Vec<f32>, Vec<f32>), // [num_bins*3-1, dp_filter, 1]
    dp_filter: usize,
}

impl ConvFlow {
    fn load(store: &TensorStore, idx: usize, dp_filter: usize) -> Result<Self> {
        let p = format!("dp.flows.{idx}");
        let out = RQS_NUM_BINS * 3 - 1;
        Ok(Self {
            pre: (
                store.tensor_shaped(&format!("{p}.pre.weight"), &[dp_filter, 1, 1])?,
                store.tensor_shaped(&format!("{p}.pre.bias"), &[dp_filter])?,
            ),
            convs: DdsConv::load(store, &format!("{p}.convs"), dp_filter)?,
            proj: (
                store.tensor_shaped(&format!("{p}.proj.weight"), &[out, dp_filter, 1])?,
                store.tensor_shaped(&format!("{p}.proj.bias"), &[out])?,
            ),
            dp_filter,
        })
    }

    /// Reverse pass: transform `x1` (channel 1) by the per-time spline the
    /// conditioner predicts; keep `x0` (channel 0). `x` is `[2, T]`.
    fn reverse(&self, x: &[f32], t: usize, g: &[f32]) -> Vec<f32> {
        let dp_filter = self.dp_filter;
        let x0 = &x[..t];
        let (pw, pb) = &self.pre;
        let (h, _) = nn::conv1d(x0, 1, t, pw, dp_filter, 1, Some(pb), 1, 0, 1, 1);
        let h = self.convs.forward(&h, t, Some(g));
        let (jw, jb) = &self.proj;
        let out = RQS_NUM_BINS * 3 - 1;
        let (params, _) = nn::conv1d(&h, dp_filter, t, jw, out, 1, Some(jb), 1, 0, 1, 1);

        let scale = (dp_filter as f32).sqrt();
        let mut result = x.to_vec();
        for ti in 0..t {
            // params[:, ti]: [0:10] widths, [10:20] heights, [20:29] derivatives.
            let mut w = [0.0f32; RQS_NUM_BINS];
            let mut hh = [0.0f32; RQS_NUM_BINS];
            let mut d = [0.0f32; RQS_NUM_BINS - 1];
            for b in 0..RQS_NUM_BINS {
                w[b] = params[b * t + ti] / scale;
                hh[b] = params[(RQS_NUM_BINS + b) * t + ti] / scale;
            }
            for b in 0..RQS_NUM_BINS - 1 {
                d[b] = params[(2 * RQS_NUM_BINS + b) * t + ti];
            }
            let x1 = x[t + ti]; // channel 1
            result[t + ti] = unconstrained_rqs_inverse(x1, &w, &hh, &d);
        }
        result
    }
}

/// The stochastic duration predictor.
pub(super) struct DurationPredictor {
    pre: (Vec<f32>, Vec<f32>),  // [dp_filter, dp_filter, 1]
    cond: (Vec<f32>, Vec<f32>), // [dp_filter, gin, 1]
    convs: DdsConv,
    proj: (Vec<f32>, Vec<f32>), // [dp_filter, dp_filter, 1]
    ea_m: Vec<f32>,             // [2, 1]
    ea_logs: Vec<f32>,          // [2, 1]
    flows: Vec<ConvFlow>,       // dp.flows.7, .5, .3 (reverse order)
    gin: usize,
    dp_filter: usize,
}

impl DurationPredictor {
    pub(super) fn load(store: &TensorStore, dims: &Dims) -> Result<Self> {
        let gin = dims.gin;
        let dp_filter = dims.dp_filter;
        Ok(Self {
            pre: (
                store.tensor_shaped("dp.pre.weight", &[dp_filter, dp_filter, 1])?,
                store.tensor_shaped("dp.pre.bias", &[dp_filter])?,
            ),
            cond: (
                store.tensor_shaped("dp.cond.weight", &[dp_filter, gin, 1])?,
                store.tensor_shaped("dp.cond.bias", &[dp_filter])?,
            ),
            convs: DdsConv::load(store, "dp.convs", dp_filter)?,
            proj: (
                store.tensor_shaped("dp.proj.weight", &[dp_filter, dp_filter, 1])?,
                store.tensor_shaped("dp.proj.bias", &[dp_filter])?,
            ),
            ea_m: store.tensor_shaped("dp.flows.0.m", &[2, 1])?,
            ea_logs: store.tensor_shaped("dp.flows.0.logs", &[2, 1])?,
            // Inference reverse order: ConvFlow 7, 5, 3 (ConvFlow 1 is dropped).
            flows: vec![
                ConvFlow::load(store, 7, dp_filter)?,
                ConvFlow::load(store, 5, dp_filter)?,
                ConvFlow::load(store, 3, dp_filter)?,
            ],
            gin,
            dp_filter,
        })
    }

    /// Computes `logw` `[T]` from the phoneme features `x_dp` `[DP_FILTER, T]`
    /// under global conditioning `g` `[GIN]`.
    ///
    /// `noise_scale_w` scales the (Gaussian) latent; the deterministic parity
    /// path passes 0, making the initial latent all zeros.
    /// The SDP body: `pre → + cond(g) → DDSConv → proj` `[DP_FILTER, T]`
    /// (the conditioning the flows read).
    pub(super) fn body(&self, x_dp: &[f32], t: usize, g: &[f32]) -> Vec<f32> {
        let dp_filter = self.dp_filter;
        let (pw, pb) = &self.pre;
        let (mut x, _) = nn::conv1d(x_dp, dp_filter, t, pw, dp_filter, 1, Some(pb), 1, 0, 1, 1);
        let cg = cond_project(&self.cond, g, self.gin);
        for c in 0..dp_filter {
            for ti in 0..t {
                x[c * t + ti] += cg[c];
            }
        }
        x = self.convs.forward(&x, t, None);
        let (jw, jb) = &self.proj;
        let (body, _) = nn::conv1d(&x, dp_filter, t, jw, dp_filter, 1, Some(jb), 1, 0, 1, 1);
        body
    }

    pub(super) fn logw(&self, x_dp: &[f32], t: usize, g: &[f32], noise_scale_w: f32) -> Vec<f32> {
        let body = self.body(x_dp, t, g);

        // Reverse flow from a (zeroed, for noise_w=0) latent.
        let mut z = vec![0.0f32; 2 * t];
        if noise_scale_w != 0.0 {
            // Non-deterministic path not exercised in M0 parity; kept explicit.
            // A real RNG would fill z ~ N(0,1) * noise_scale_w here.
        }
        for flow in &self.flows {
            flip2(&mut z, t);
            z = flow.reverse(&z, t, &body);
        }
        flip2(&mut z, t);
        // ElementwiseAffine reverse: x = (x - m) * exp(-logs). The exported
        // `dp.flows.0.logs` buffer (from `onnx::Exp_*`) is already the folded
        // `-logs` that feeds the graph's `Exp`, so `exp(-logs) = exp(buffer)`.
        for c in 0..2 {
            let m = self.ea_m[c];
            let inv = self.ea_logs[c].exp();
            for ti in 0..t {
                z[c * t + ti] = (z[c * t + ti] - m) * inv;
            }
        }
        // logw = z0 (channel 0).
        z[..t].to_vec()
    }
}

fn load_norm(store: &TensorStore, prefix: &str, channels: usize) -> Result<Norm> {
    Ok(Norm {
        gamma: store.tensor_shaped(&format!("{prefix}.gamma"), &[channels])?,
        beta: store.tensor_shaped(&format!("{prefix}.beta"), &[channels])?,
    })
}

#[allow(clippy::needless_range_loop)] // channel-major matrix indexing
fn cond_project(layer: &(Vec<f32>, Vec<f32>), g: &[f32], gin: usize) -> Vec<f32> {
    let (w, b) = layer;
    let out_ch = b.len();
    let mut out = b.clone();
    for c in 0..out_ch {
        let wrow = c * gin;
        let mut acc = out[c];
        for i in 0..gin {
            acc += w[wrow + i] * g[i];
        }
        out[c] = acc;
    }
    out
}

/// Channel flip of a `[2, T]` latent (`torch.flip(x, [1])`).
fn flip2(x: &mut [f32], t: usize) {
    for ti in 0..t {
        x.swap(ti, t + ti);
    }
}

// --- Rational quadratic spline (transforms.py) ------------------------------

/// Unconstrained (linear-tails) rational-quadratic-spline inverse for one
/// scalar. Outside `[-tail_bound, tail_bound]` the map is the identity.
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
    let constant = ((1.0 - MIN_DERIVATIVE).exp() - 1.0).ln();
    let mut derivs = [0.0f32; RQS_NUM_BINS + 1];
    derivs[0] = constant;
    derivs[RQS_NUM_BINS] = constant;
    derivs[1..RQS_NUM_BINS].copy_from_slice(unnorm_d);
    rqs_inverse(input, unnorm_w, unnorm_h, &derivs, -tb, tb, -tb, tb)
}

/// Rational-quadratic-spline inverse for one scalar over the box
/// `[left,right]×[bottom,top]`.
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

    // widths → cumwidths (in the x-box).
    let widths_sm = softmax(unnorm_w);
    let mut widths = [0.0f32; RQS_NUM_BINS];
    for b in 0..n {
        widths[b] = MIN_BIN_WIDTH + (1.0 - MIN_BIN_WIDTH * nf) * widths_sm[b];
    }
    let cumwidths = cumulative(&widths, left, right - left);
    let widths = diffs(&cumwidths);

    // derivatives = min + softplus(unnorm).
    let mut derivatives = [0.0f32; RQS_NUM_BINS + 1];
    for b in 0..=n {
        derivatives[b] = MIN_DERIVATIVE + nn::softplus(derivatives_unnorm[b]);
    }

    // heights → cumheights (in the y-box).
    let heights_sm = softmax(unnorm_h);
    let mut heights = [0.0f32; RQS_NUM_BINS];
    for b in 0..n {
        heights[b] = MIN_BIN_HEIGHT + (1.0 - MIN_BIN_HEIGHT * nf) * heights_sm[b];
    }
    let cumheights = cumulative(&heights, bottom, top - bottom);
    let heights = diffs(&cumheights);

    // Inverse: locate the bin by the y (height) coordinate.
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
    let root = 2.0 * c / (-b - discriminant.sqrt());
    root * input_bin_widths + input_cumwidths
}

/// `searchsorted`: index of the bin whose lower edge is `<= input`
/// (transforms.py adds `eps` to the last edge; clamps to a valid bin).
fn searchsorted(bin_locations: &[f32; RQS_NUM_BINS + 1], input: f32) -> usize {
    let eps = 1e-6;
    let mut count = 0usize;
    for (i, &loc) in bin_locations.iter().enumerate() {
        let loc = if i == RQS_NUM_BINS { loc + eps } else { loc };
        if input >= loc {
            count += 1;
        }
    }
    count.saturating_sub(1).min(RQS_NUM_BINS - 1)
}

/// Cumulative edges: `cumsum` padded with a leading 0, scaled into `[base,
/// base+span]` with the first/last edges pinned exactly.
fn cumulative(bins: &[f32; RQS_NUM_BINS], base: f32, span: f32) -> [f32; RQS_NUM_BINS + 1] {
    let mut cum = [0.0f32; RQS_NUM_BINS + 1];
    let mut acc = 0.0f32;
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
    let mut d = [0.0f32; RQS_NUM_BINS];
    for b in 0..RQS_NUM_BINS {
        d[b] = cum[b + 1] - cum[b];
    }
    d
}

fn softmax(x: &[f32; RQS_NUM_BINS]) -> [f32; RQS_NUM_BINS] {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut out = [0.0f32; RQS_NUM_BINS];
    let mut sum = 0.0f32;
    for b in 0..RQS_NUM_BINS {
        out[b] = (x[b] - max).exp();
        sum += out[b];
    }
    for v in &mut out {
        *v /= sum;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::rng::Xorshift64Star;

    #[test]
    fn rqs_identity_outside_tails() {
        let w = [0.0; RQS_NUM_BINS];
        let h = [0.0; RQS_NUM_BINS];
        let d = [0.0; RQS_NUM_BINS - 1];
        // |input| > tail_bound → identity.
        assert_eq!(unconstrained_rqs_inverse(7.0, &w, &h, &d), 7.0);
        assert_eq!(unconstrained_rqs_inverse(-6.0, &w, &h, &d), -6.0);
    }

    #[test]
    fn rqs_inverse_maps_into_x_box() {
        // With uniform params the spline is ~identity; inverse of 0 is near 0.
        let w = [0.1; RQS_NUM_BINS];
        let h = [0.1; RQS_NUM_BINS];
        let d = [0.0; RQS_NUM_BINS - 1];
        let out = unconstrained_rqs_inverse(0.0, &w, &h, &d);
        assert!(out.abs() < RQS_TAIL_BOUND, "inverse stays in box: {out}");
    }

    #[test]
    fn searchsorted_picks_bin() {
        let edges = [-5.0, -3.0, -1.0, 0.0, 1.0, 2.0, 3.0, 3.5, 4.0, 4.5, 5.0];
        assert_eq!(searchsorted(&edges, -4.0), 0);
        assert_eq!(searchsorted(&edges, 0.5), 3);
        assert_eq!(searchsorted(&edges, 4.9), 9);
    }

    #[test]
    fn rqs_inverse_maps_box_corners_and_is_monotonic() {
        // Arbitrary but fixed (non-degenerate) spline params, drawn from the
        // shared reproducible xorshift64* (uniform in [-1, 1)).
        let mut rng = Xorshift64Star::new(0x1234_5678_9abc_def0);
        let mut w = [0.0f32; RQS_NUM_BINS];
        let mut h = [0.0f32; RQS_NUM_BINS];
        for b in 0..RQS_NUM_BINS {
            w[b] = rng.next_signed_f32();
            h[b] = rng.next_signed_f32();
        }
        let d = [0.1f32; RQS_NUM_BINS - 1];
        let tb = RQS_TAIL_BOUND;

        // Analytic invariant #1: the box corners map onto themselves exactly,
        // independent of the params. At input=-tb the bin-0 quadratic has dy=0
        // ⇒ root 0 ⇒ left edge; at input=+tb the last bin has dy=height ⇒ root 1
        // ⇒ right edge. (Derived, not referenced.)
        assert!((unconstrained_rqs_inverse(-tb, &w, &h, &d) + tb).abs() < 1e-4);
        assert!((unconstrained_rqs_inverse(tb, &w, &h, &d) - tb).abs() < 1e-4);

        // Analytic invariant #2: the inverse of a strictly-increasing spline is
        // strictly increasing.
        let mut prev = f32::NEG_INFINITY;
        for &inp in &[-4.5f32, -3.0, -1.0, 0.0, 2.0, 4.5] {
            let out = unconstrained_rqs_inverse(inp, &w, &h, &d);
            assert!(out > prev, "monotonic: f({inp}) = {out} <= {prev}");
            prev = out;
        }
    }
}
