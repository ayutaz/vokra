//! Normalizing flow (M0-07-T16/T17): `ResidualCouplingBlock` reverse pass.
//!
//! Follows piper-plus `vits/models.py::ResidualCouplingBlock` and
//! `vits/modules.py::{ResidualCouplingLayer, WN, Flip}`. Mean-only coupling
//! layers interleaved with channel flips; inference runs them in reverse, so a
//! coupling is simply `x1 ← x1 − m` (`logs = 0`). The coupling conditioner is a
//! WaveNet-style gated dilated-conv stack (`WN`) whose weights are the
//! `onnx::Conv_*` tensors the converter recovered to clean names (M0-07-T07).
//!
//! Sizes (`hidden` / `gin`, coupling count, WN layer count and the WN dilation
//! base) are threaded from the shape-/architecture-derived [`Dims`]; the global
//! conditioning `g` (`spk_proj(speaker) + emb_lang`, composed once by
//! [`super::conditioning`]) is passed in. The zero-shot v7 flow uses WN
//! `dilation_rate = 2` (per-layer dilations 1,2,4,8); the legacy single-speaker
//! voice used 1 — see [`Dims::flow_wn_dilation_rate`].

use super::config::{Dims, FLOW_WN_KERNEL};
use super::nn;
use super::weights::TensorStore;
use vokra_core::Result;

/// A WaveNet conditioner (`WN`): `n_layers` gated dilated convs (dilation
/// `dilation_rate^i` at layer `i`) with a shared conditioning projection.
struct Wn {
    in_layers: Vec<(Vec<f32>, Vec<f32>)>, // [2*hidden, hidden, kernel]
    res_skip_layers: Vec<(Vec<f32>, Vec<f32>)>, // last is [hidden,...], rest [2*hidden,...]
    cond_layer: (Vec<f32>, Vec<f32>),     // [2*hidden*n_layers, gin, 1]
    hidden: usize,
    n_layers: usize,
    dilation_rate: usize,
}

impl Wn {
    fn forward(&self, x: &[f32], t: usize, g_cond: &[f32]) -> Vec<f32> {
        let hidden = self.hidden;
        let two_h = 2 * hidden;
        let mut h = x.to_vec();
        let mut output = vec![0.0f32; hidden * t];
        for i in 0..self.n_layers {
            // Layer i uses dilation dilation_rate^i (v7: 1,2,4,8), same padding.
            let dilation = self.dilation_rate.pow(i as u32);
            let pad = dilation * (FLOW_WN_KERNEL - 1) / 2;
            let (iw, ib) = &self.in_layers[i];
            let (x_in, _) = nn::conv1d(
                &h,
                hidden,
                t,
                iw,
                two_h,
                FLOW_WN_KERNEL,
                Some(ib),
                1,
                pad,
                dilation,
                1,
            );
            // Gated activation: tanh(a+g)·sigmoid(b+g) over the two halves, with
            // the conditioning slice for this layer broadcast over time.
            let goff = i * two_h;
            let mut acts = vec![0.0f32; hidden * t];
            for c in 0..hidden {
                let gt = g_cond[goff + c];
                let gs = g_cond[goff + hidden + c];
                for ti in 0..t {
                    let a = x_in[c * t + ti] + gt;
                    let b = x_in[(hidden + c) * t + ti] + gs;
                    acts[c * t + ti] = a.tanh() * nn::sigmoid(b);
                }
            }
            let (rw, rb) = &self.res_skip_layers[i];
            let out_ch = if i < self.n_layers - 1 { two_h } else { hidden };
            let (res_skip, _) = nn::conv1d(&acts, hidden, t, rw, out_ch, 1, Some(rb), 1, 0, 1, 1);
            if i < self.n_layers - 1 {
                for c in 0..hidden {
                    for ti in 0..t {
                        h[c * t + ti] += res_skip[c * t + ti]; // residual (first hidden)
                        output[c * t + ti] += res_skip[(hidden + c) * t + ti]; // skip
                    }
                }
            } else {
                for idx in 0..hidden * t {
                    output[idx] += res_skip[idx];
                }
            }
        }
        output
    }
}

/// One mean-only residual coupling layer.
struct Coupling {
    pre: (Vec<f32>, Vec<f32>),  // [hidden, half, 1]
    post: (Vec<f32>, Vec<f32>), // [half, hidden, 1] (mean-only)
    enc: Wn,
    hidden: usize,
    half: usize,
}

impl Coupling {
    /// Reverse pass: `x1 ← x1 − m(x0, g)`, keeping `x0`.
    fn reverse(&self, x: &[f32], t: usize, g_cond: &[f32]) -> Vec<f32> {
        let (hidden, half) = (self.hidden, self.half);
        // Channel split: x0 = x[..half], x1 = x[half..].
        let x0 = &x[..half * t];
        let (pw, pb) = &self.pre;
        let (h, _) = nn::conv1d(x0, half, t, pw, hidden, 1, Some(pb), 1, 0, 1, 1);
        let h = self.enc.forward(&h, t, g_cond);
        let (sw, sb) = &self.post;
        let (m, _) = nn::conv1d(&h, hidden, t, sw, half, 1, Some(sb), 1, 0, 1, 1);
        // x1 - m; x0 unchanged.
        let mut out = x.to_vec();
        for c in 0..half {
            for ti in 0..t {
                out[(half + c) * t + ti] -= m[c * t + ti];
            }
        }
        out
    }
}

/// The residual-coupling flow block.
pub(super) struct Flow {
    couplings: Vec<Coupling>,
    hidden: usize,
    gin: usize,
    n_flows: usize,
}

impl Flow {
    pub(super) fn load(store: &TensorStore, dims: &Dims) -> Result<Self> {
        let hidden = dims.hidden;
        let gin = dims.gin;
        let half = hidden / 2;
        let two_h = 2 * hidden;
        let n_flows = dims.flow_n_flows;
        let wn_layers = dims.flow_wn_layers;
        let dilation_rate = dims.flow_wn_dilation_rate;
        let mut couplings = Vec::with_capacity(n_flows);
        // Coupling layers live at even flow indices (odds are `Flip`).
        for k in 0..n_flows {
            let idx = 2 * k;
            let p = format!("flow.flows.{idx}");
            let e = format!("{p}.enc");
            let mut in_layers = Vec::with_capacity(wn_layers);
            let mut res_skip_layers = Vec::with_capacity(wn_layers);
            for l in 0..wn_layers {
                in_layers.push((
                    store.tensor_shaped(
                        &format!("{e}.in_layers.{l}.weight"),
                        &[two_h, hidden, FLOW_WN_KERNEL],
                    )?,
                    store.tensor_shaped(&format!("{e}.in_layers.{l}.bias"), &[two_h])?,
                ));
                let rs_out = if l < wn_layers - 1 { two_h } else { hidden };
                res_skip_layers.push((
                    store.tensor_shaped(
                        &format!("{e}.res_skip_layers.{l}.weight"),
                        &[rs_out, hidden, 1],
                    )?,
                    store.tensor_shaped(&format!("{e}.res_skip_layers.{l}.bias"), &[rs_out])?,
                ));
            }
            couplings.push(Coupling {
                pre: (
                    store.tensor_shaped(&format!("{p}.pre.weight"), &[hidden, half, 1])?,
                    store.tensor_shaped(&format!("{p}.pre.bias"), &[hidden])?,
                ),
                post: (
                    store.tensor_shaped(&format!("{p}.post.weight"), &[half, hidden, 1])?,
                    store.tensor_shaped(&format!("{p}.post.bias"), &[half])?,
                ),
                enc: Wn {
                    in_layers,
                    res_skip_layers,
                    cond_layer: (
                        store.tensor_shaped(
                            &format!("{e}.cond_layer.weight"),
                            &[two_h * wn_layers, gin, 1],
                        )?,
                        store
                            .tensor_shaped(&format!("{e}.cond_layer.bias"), &[two_h * wn_layers])?,
                    ),
                    hidden,
                    n_layers: wn_layers,
                    dilation_rate,
                },
                hidden,
                half,
            });
        }
        Ok(Self {
            couplings,
            hidden,
            gin,
            n_flows,
        })
    }

    /// Reverse flow: `z_p` `[hidden, T]` → `z` `[hidden, T]` under `g` `[gin]`.
    ///
    /// Runs `[Flip, Coupling_{2(N-1)}, …, Flip, Coupling_0]` (the `flows` list
    /// reversed), with a flip before every coupling.
    pub(super) fn reverse(&self, z_p: &[f32], t: usize, g: &[f32]) -> Vec<f32> {
        let mut x = z_p.to_vec();
        // Precompute each coupling's conditioning projection cond_layer(g).
        let cond: Vec<Vec<f32>> = self
            .couplings
            .iter()
            .map(|c| cond_project(&c.enc.cond_layer, g, self.gin))
            .collect();
        for k in (0..self.n_flows).rev() {
            flip_channels(&mut x, self.hidden, t);
            x = self.couplings[k].reverse(&x, t, &cond[k]);
        }
        x
    }
}

/// `cond_layer(g)` = `Conv1d(gin, out, 1)` applied to `g[gin]` → `[out]`.
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

/// `torch.flip(x, [1])`: reverse the channel axis of `[hidden, T]` in place.
fn flip_channels(x: &mut [f32], hidden: usize, t: usize) {
    for c in 0..hidden / 2 {
        let other = hidden - 1 - c;
        for ti in 0..t {
            x.swap(c * t + ti, other * t + ti);
        }
    }
}
