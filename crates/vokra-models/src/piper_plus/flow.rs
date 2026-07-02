//! Normalizing flow (M0-07-T16/T17): `ResidualCouplingBlock` reverse pass.
//!
//! Follows piper-plus `vits/models.py::ResidualCouplingBlock` and
//! `vits/modules.py::{ResidualCouplingLayer, WN, Flip}`. Four **mean-only**
//! coupling layers interleaved with channel flips; inference runs them in
//! reverse, so a coupling is simply `x1 ← x1 − m` (`logs = 0`). The coupling
//! conditioner is a WaveNet-style gated dilated-conv stack (`WN`) whose weights
//! are the `onnx::Conv_*` tensors the converter recovered to clean names
//! (M0-07-T07). This voice exports every WN layer at dilation 1.

use super::config::{FLOW_N_FLOWS, FLOW_WN_DILATION, FLOW_WN_KERNEL, FLOW_WN_LAYERS, GIN, HIDDEN};
use super::nn;
use super::weights::TensorStore;
use vokra_core::Result;

const HALF: usize = HIDDEN / 2; // 96

/// A WaveNet conditioner (`WN`): `n_layers` gated dilated convs with a shared
/// conditioning projection.
struct Wn {
    in_layers: Vec<(Vec<f32>, Vec<f32>)>, // [2*HIDDEN, HIDDEN, kernel]
    res_skip_layers: Vec<(Vec<f32>, Vec<f32>)>, // last is [HIDDEN,...], rest [2*HIDDEN,...]
    cond_layer: (Vec<f32>, Vec<f32>),     // [2*HIDDEN*n_layers, GIN, 1]
}

impl Wn {
    fn forward(&self, x: &[f32], t: usize, g_cond: &[f32]) -> Vec<f32> {
        let two_h = 2 * HIDDEN;
        let mut h = x.to_vec();
        let mut output = vec![0.0f32; HIDDEN * t];
        let pad = FLOW_WN_DILATION * (FLOW_WN_KERNEL - 1) / 2;
        for i in 0..FLOW_WN_LAYERS {
            let (iw, ib) = &self.in_layers[i];
            let (x_in, _) = nn::conv1d(
                &h,
                HIDDEN,
                t,
                iw,
                two_h,
                FLOW_WN_KERNEL,
                Some(ib),
                1,
                pad,
                FLOW_WN_DILATION,
                1,
            );
            // Gated activation: tanh(a+g)·sigmoid(b+g) over the two halves, with
            // the conditioning slice for this layer broadcast over time.
            let goff = i * two_h;
            let mut acts = vec![0.0f32; HIDDEN * t];
            for c in 0..HIDDEN {
                let gt = g_cond[goff + c];
                let gs = g_cond[goff + HIDDEN + c];
                for ti in 0..t {
                    let a = x_in[c * t + ti] + gt;
                    let b = x_in[(HIDDEN + c) * t + ti] + gs;
                    acts[c * t + ti] = a.tanh() * sigmoid(b);
                }
            }
            let (rw, rb) = &self.res_skip_layers[i];
            let out_ch = if i < FLOW_WN_LAYERS - 1 {
                two_h
            } else {
                HIDDEN
            };
            let (res_skip, _) = nn::conv1d(&acts, HIDDEN, t, rw, out_ch, 1, Some(rb), 1, 0, 1, 1);
            if i < FLOW_WN_LAYERS - 1 {
                for c in 0..HIDDEN {
                    for ti in 0..t {
                        h[c * t + ti] += res_skip[c * t + ti]; // residual (first HIDDEN)
                        output[c * t + ti] += res_skip[(HIDDEN + c) * t + ti]; // skip
                    }
                }
            } else {
                for idx in 0..HIDDEN * t {
                    output[idx] += res_skip[idx];
                }
            }
        }
        output
    }
}

/// One mean-only residual coupling layer.
struct Coupling {
    pre: (Vec<f32>, Vec<f32>),  // [HIDDEN, HALF, 1]
    post: (Vec<f32>, Vec<f32>), // [HALF, HIDDEN, 1] (mean-only)
    enc: Wn,
}

impl Coupling {
    /// Reverse pass: `x1 ← x1 − m(x0, g)`, keeping `x0`.
    fn reverse(&self, x: &[f32], t: usize, g_cond: &[f32]) -> Vec<f32> {
        // Channel split: x0 = x[..HALF], x1 = x[HALF..].
        let x0 = &x[..HALF * t];
        let (pw, pb) = &self.pre;
        let (h, _) = nn::conv1d(x0, HALF, t, pw, HIDDEN, 1, Some(pb), 1, 0, 1, 1);
        let h = self.enc.forward(&h, t, g_cond);
        let (sw, sb) = &self.post;
        let (m, _) = nn::conv1d(&h, HIDDEN, t, sw, HALF, 1, Some(sb), 1, 0, 1, 1);
        // x1 - m; x0 unchanged.
        let mut out = x.to_vec();
        for c in 0..HALF {
            for ti in 0..t {
                out[(HALF + c) * t + ti] -= m[c * t + ti];
            }
        }
        out
    }
}

/// The residual-coupling flow block.
pub(super) struct Flow {
    couplings: Vec<Coupling>,
}

impl Flow {
    pub(super) fn load(store: &TensorStore) -> Result<Self> {
        let two_h = 2 * HIDDEN;
        let mut couplings = Vec::with_capacity(FLOW_N_FLOWS);
        // Coupling layers live at flow indices 0, 2, 4, 6 (odds are Flip).
        for k in 0..FLOW_N_FLOWS {
            let idx = 2 * k;
            let p = format!("flow.flows.{idx}");
            let e = format!("{p}.enc");
            let mut in_layers = Vec::with_capacity(FLOW_WN_LAYERS);
            let mut res_skip_layers = Vec::with_capacity(FLOW_WN_LAYERS);
            for l in 0..FLOW_WN_LAYERS {
                in_layers.push((
                    store.tensor_shaped(
                        &format!("{e}.in_layers.{l}.weight"),
                        &[two_h, HIDDEN, FLOW_WN_KERNEL],
                    )?,
                    store.tensor_shaped(&format!("{e}.in_layers.{l}.bias"), &[two_h])?,
                ));
                let rs_out = if l < FLOW_WN_LAYERS - 1 {
                    two_h
                } else {
                    HIDDEN
                };
                res_skip_layers.push((
                    store.tensor_shaped(
                        &format!("{e}.res_skip_layers.{l}.weight"),
                        &[rs_out, HIDDEN, 1],
                    )?,
                    store.tensor_shaped(&format!("{e}.res_skip_layers.{l}.bias"), &[rs_out])?,
                ));
            }
            couplings.push(Coupling {
                pre: (
                    store.tensor_shaped(&format!("{p}.pre.weight"), &[HIDDEN, HALF, 1])?,
                    store.tensor_shaped(&format!("{p}.pre.bias"), &[HIDDEN])?,
                ),
                post: (
                    store.tensor_shaped(&format!("{p}.post.weight"), &[HALF, HIDDEN, 1])?,
                    store.tensor_shaped(&format!("{p}.post.bias"), &[HALF])?,
                ),
                enc: Wn {
                    in_layers,
                    res_skip_layers,
                    cond_layer: (
                        store.tensor_shaped(
                            &format!("{e}.cond_layer.weight"),
                            &[two_h * FLOW_WN_LAYERS, GIN, 1],
                        )?,
                        store.tensor_shaped(
                            &format!("{e}.cond_layer.bias"),
                            &[two_h * FLOW_WN_LAYERS],
                        )?,
                    ),
                },
            });
        }
        Ok(Self { couplings })
    }

    /// Reverse flow: `z_p` `[HIDDEN, T]` → `z` `[HIDDEN, T]` under `g` `[GIN]`.
    ///
    /// Runs `[Flip, Coupling6, Flip, Coupling4, Flip, Coupling2, Flip,
    /// Coupling0]` (the `flows` list reversed).
    pub(super) fn reverse(&self, z_p: &[f32], t: usize, g: &[f32]) -> Vec<f32> {
        let mut x = z_p.to_vec();
        // Precompute each coupling's conditioning projection cond_layer(g).
        let cond: Vec<Vec<f32>> = self
            .couplings
            .iter()
            .map(|c| cond_project(&c.enc.cond_layer, g))
            .collect();
        // reversed(flows): flip, then coupling k for k = N-1..0, with a flip
        // before every coupling.
        for k in (0..FLOW_N_FLOWS).rev() {
            flip_channels(&mut x, t);
            x = self.couplings[k].reverse(&x, t, &cond[k]);
        }
        x
    }
}

/// `cond_layer(g)` = `Conv1d(GIN, out, 1)` applied to `g[GIN]` → `[out]`.
#[allow(clippy::needless_range_loop)] // channel-major matrix indexing
fn cond_project(layer: &(Vec<f32>, Vec<f32>), g: &[f32]) -> Vec<f32> {
    let (w, b) = layer;
    let out_ch = b.len();
    let mut out = b.clone();
    for c in 0..out_ch {
        let wrow = c * GIN;
        let mut acc = out[c];
        for i in 0..GIN {
            acc += w[wrow + i] * g[i];
        }
        out[c] = acc;
    }
    out
}

/// `torch.flip(x, [1])`: reverse the channel axis of `[HIDDEN, T]` in place.
fn flip_channels(x: &mut [f32], t: usize) {
    for c in 0..HIDDEN / 2 {
        let other = HIDDEN - 1 - c;
        for ti in 0..t {
            x.swap(c * t + ti, other * t + ti);
        }
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
