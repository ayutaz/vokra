//! Forward-pass building blocks shared by the encoder and decoder.
//!
//! These are thin wrappers over the M0-08 `vokra-backend-cpu` compute kernels
//! (`gemm_f32`, `layer_norm_f32`, `gelu_f32`, `softmax_f32`), plus a plain-Rust
//! multi-head attention assembled from them. Shapes are row-major throughout; a
//! shape mismatch surfaces as the kernel's [`VokraError::InvalidArgument`]
//! (NFR-RL-07).
//!
//! # `*_into`: caller-owned buffers, zero hot-path malloc (FR-EX-05, M1-04)
//!
//! Each block op has an **`_into`** form that writes into a caller-owned buffer
//! (a `super::scratch` field, or the nested [`AttnScratch`]): after the first
//! sizing it does no heap allocation, so the autoregressive decode loop is
//! malloc-free (proven by the capacity-stability oracle in the `super::decoder`
//! tests). The two **allocating** wrappers kept here — `project_kv` and
//! `attention_from_kv` — forward to the `_into` form with a fresh `Vec`; they
//! serve the unit tests and the one-shot cross-attention K/V precomputation
//! (computed once per audio window, off the hot path). Both forms run the
//! **identical** kernel calls in the identical order, so results are bit-for-bit
//! equal — reusing a buffer never perturbs the accumulation order.

use vokra_backend_cpu::kernels::LAYER_NORM_DEFAULT_EPS;
use vokra_core::{Result, VokraError};

use super::scratch::{AttnScratch, resize_zeroed};
use super::weights::{Attention, LayerNorm, Linear};
use crate::compute::Compute;

// ---- `_into` forms (no allocation after the buffers are sized) --------------
// ZERO-ALLOC-BEGIN — the functions below must not allocate on the hot path;
// guarded by scripts/check-hot-path-allocs.sh (no vec![], Vec::with_capacity,
// .to_vec(), .collect()). They only `resize` caller-owned scratch, which never
// reallocates within the reserve (see super::scratch).

/// Affine layer norm `[rows, d] → [rows, d]` into `out` (sized here), using the
/// PyTorch/Whisper default epsilon (`1e-5`).
pub(crate) fn layer_norm_into(
    compute: &Compute,
    out: &mut Vec<f32>,
    x: &[f32],
    rows: usize,
    ln: &LayerNorm,
) -> Result<()> {
    let d = ln.gamma.len();
    resize_zeroed(out, rows * d);
    compute.layer_norm_f32(x, out, rows, d, &ln.gamma, &ln.beta, LAYER_NORM_DEFAULT_EPS)
}

/// `nn.Linear` into `out` (sized here):
/// `out[t, o] = bias[o] + sum_i x[t, i] * w_t[i, o]`.
pub(crate) fn linear_into(
    compute: &Compute,
    out: &mut Vec<f32>,
    x: &[f32],
    t: usize,
    lin: &Linear,
) -> Result<()> {
    resize_zeroed(out, t * lin.out_features);
    compute.gemm_f32(
        t,
        lin.out_features,
        lin.in_features,
        x,
        &lin.w_t,
        lin.bias.as_deref(),
        out,
    )
}

/// In-place residual add `a += b` (kills the temporary the old `add_into`
/// needed). Element-wise independent adds, so this is bit-identical to the
/// out-of-place `add_f32` kernel it replaces.
pub(crate) fn add_assign(a: &mut [f32], b: &[f32]) -> Result<()> {
    if a.len() != b.len() {
        return Err(VokraError::InvalidArgument(format!(
            "add_assign: length mismatch {} != {}",
            a.len(),
            b.len()
        )));
    }
    for (dst, &src) in a.iter_mut().zip(b) {
        *dst += src;
    }
    Ok(())
}

/// The MLP sub-block `fc2(gelu(fc1(x)))` into `out` (sized here), using `mlp_h`
/// / `mlp_a` (both sized here) as the `[t, ffn_dim]` intermediates. `x` is
/// `[t, d]`, `out` is `[t, d]`.
#[allow(clippy::too_many_arguments)] // MLP block operands + the backend dispatcher
pub(crate) fn mlp_into(
    compute: &Compute,
    mlp_h: &mut Vec<f32>,
    mlp_a: &mut Vec<f32>,
    out: &mut Vec<f32>,
    x: &[f32],
    t: usize,
    fc1: &Linear,
    fc2: &Linear,
) -> Result<()> {
    linear_into(compute, mlp_h, x, t, fc1)?;
    resize_zeroed(mlp_a, mlp_h.len());
    compute.gelu_f32(mlp_h, mlp_a)?;
    linear_into(compute, out, mlp_a, t, fc2)
}

/// Projects the key/value inputs of an attention block into `k_out` / `v_out`
/// (both sized here), each `[t_kv, d]`.
pub(crate) fn project_kv_into(
    compute: &Compute,
    k_out: &mut Vec<f32>,
    v_out: &mut Vec<f32>,
    xkv: &[f32],
    t_kv: usize,
    attn: &Attention,
) -> Result<()> {
    linear_into(compute, k_out, xkv, t_kv, &attn.k)?;
    linear_into(compute, v_out, xkv, t_kv, &attn.v)
}

/// Multi-head attention from **pre-projected** keys/values, into `out` (sized
/// here) using `scratch` for every intermediate.
///
/// - `xq` is `[t_q, d]`; `k` / `v` are `[t_kv, d]` (already `k_proj`/`v_proj`);
/// - `q_pos_offset` is the absolute position of `xq[0]` (the keys span absolute
///   positions `0..t_kv`); used only when `causal` so a decode step masks
///   future keys correctly;
/// - the query is scaled by `head_dim^-0.5` (Whisper applies the scale to the
///   query, not the scores).
///
/// Writes the block output `[t_q, d]` (after the output projection) into `out`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn attention_from_kv_into(
    compute: &Compute,
    scratch: &mut AttnScratch,
    xq: &[f32],
    t_q: usize,
    k: &[f32],
    v: &[f32],
    t_kv: usize,
    q_lin: &Linear,
    out_lin: &Linear,
    n_head: usize,
    causal: bool,
    q_pos_offset: usize,
    out: &mut Vec<f32>,
) -> Result<()> {
    let d = q_lin.out_features;
    let hd = d / n_head;
    let scale = (hd as f32).powf(-0.5);

    scratch.ensure(t_q, t_kv, d, n_head);

    // Scaled query projection into scratch.q (bias applied by the GEMM).
    compute.gemm_f32(
        t_q,
        d,
        q_lin.in_features,
        xq,
        &q_lin.w_t,
        q_lin.bias.as_deref(),
        &mut scratch.q,
    )?;
    for val in &mut scratch.q {
        *val *= scale;
    }

    for h in 0..n_head {
        let c0 = h * hd;
        // Gather this head's q [t_q, hd] and v [t_kv, hd]; k transposed [hd, t_kv].
        for i in 0..t_q {
            scratch.qh[i * hd..i * hd + hd]
                .copy_from_slice(&scratch.q[i * d + c0..i * d + c0 + hd]);
        }
        for j in 0..t_kv {
            scratch.vh[j * hd..j * hd + hd].copy_from_slice(&v[j * d + c0..j * d + c0 + hd]);
            for c in 0..hd {
                scratch.kh_t[c * t_kv + j] = k[j * d + c0 + c];
            }
        }
        // scores [t_q, t_kv] = qh [t_q, hd] @ kh_t [hd, t_kv].
        compute.gemm_f32(
            t_q,
            t_kv,
            hd,
            &scratch.qh,
            &scratch.kh_t,
            None,
            &mut scratch.scores,
        )?;
        // Causal mask: query i (abs pos q_pos_offset + i) may not attend key j
        // when j > q_pos_offset + i.
        if causal {
            for i in 0..t_q {
                let last = q_pos_offset + i;
                for j in (last + 1)..t_kv {
                    scratch.scores[i * t_kv + j] = f32::NEG_INFINITY;
                }
            }
        }
        compute.softmax_f32(&scratch.scores, &mut scratch.probs, t_q, t_kv)?;
        // ctx_h [t_q, hd] = probs [t_q, t_kv] @ vh [t_kv, hd].
        compute.gemm_f32(
            t_q,
            hd,
            t_kv,
            &scratch.probs,
            &scratch.vh,
            None,
            &mut scratch.ctx_h,
        )?;
        for i in 0..t_q {
            scratch.context[i * d + c0..i * d + c0 + hd]
                .copy_from_slice(&scratch.ctx_h[i * hd..i * hd + hd]);
        }
    }

    // Output projection: out [t_q, d] = context [t_q, d] @ out_lin.
    linear_into(compute, out, &scratch.context, t_q, out_lin)
}
// ZERO-ALLOC-END

// ---- allocating wrappers (unit tests + one-shot cross-K/V precompute) -------

/// Allocating [`project_kv_into`]: returns `(k, v)` each `[t_kv, d]`. Used by
/// the decoder's one-shot cross-attention K/V precomputation and the tests.
pub(crate) fn project_kv(
    compute: &Compute,
    xkv: &[f32],
    t_kv: usize,
    attn: &Attention,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let mut k = Vec::new();
    let mut v = Vec::new();
    project_kv_into(compute, &mut k, &mut v, xkv, t_kv, attn)?;
    Ok((k, v))
}

/// Allocating [`attention_from_kv_into`]: returns the block output `[t_q, d]`.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn attention_from_kv(
    compute: &Compute,
    xq: &[f32],
    t_q: usize,
    k: &[f32],
    v: &[f32],
    t_kv: usize,
    q_lin: &Linear,
    out_lin: &Linear,
    n_head: usize,
    causal: bool,
    q_pos_offset: usize,
) -> Result<Vec<f32>> {
    let mut scratch = AttnScratch::with_reserve(t_q, t_kv, q_lin.out_features, n_head);
    let mut out = Vec::new();
    attention_from_kv_into(
        compute,
        &mut scratch,
        xq,
        t_q,
        k,
        v,
        t_kv,
        q_lin,
        out_lin,
        n_head,
        causal,
        q_pos_offset,
        &mut out,
    )?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::whisper::weights::{Attention, Linear};

    fn ident_linear(d: usize, bias: bool) -> Linear {
        // w_t = identity [d, d]; optional zero bias.
        let mut w_t = vec![0.0f32; d * d];
        for i in 0..d {
            w_t[i * d + i] = 1.0;
        }
        Linear {
            w_t,
            in_features: d,
            out_features: d,
            bias: bias.then(|| vec![0.0f32; d]),
        }
    }

    #[test]
    fn single_head_attention_is_softmax_weighted_values() {
        // d=2, 1 head. Identity q/k/out projections, so scores = scale * q·k.
        let d = 2;
        let attn = Attention {
            q: ident_linear(d, true),
            k: ident_linear(d, false),
            v: ident_linear(d, true),
            out: ident_linear(d, true),
        };
        // Two key/value positions with distinct values.
        let xkv = vec![1.0, 0.0, 0.0, 1.0]; // k=v=[[1,0],[0,1]]
        let (k, v) = project_kv(&Compute::cpu(), &xkv, 2, &attn).unwrap();
        // One query aligned with key 0.
        let xq = vec![10.0, 0.0];
        let out = attention_from_kv(
            &Compute::cpu(),
            &xq,
            1,
            &k,
            &v,
            2,
            &attn.q,
            &attn.out,
            1,
            false,
            0,
        )
        .unwrap();
        // Large positive score on key 0 → context ≈ value 0 = [1,0].
        assert!((out[0] - 1.0).abs() < 1e-3, "{out:?}");
        assert!(out[1].abs() < 1e-3, "{out:?}");
    }

    #[test]
    fn causal_mask_blocks_future_keys() {
        let d = 2;
        let attn = Attention {
            q: ident_linear(d, true),
            k: ident_linear(d, false),
            v: ident_linear(d, true),
            out: ident_linear(d, true),
        };
        // Keys/values at 3 positions; query at position 0 must only see key 0.
        let xkv = vec![5.0, 0.0, 0.0, 5.0, 0.0, -5.0]; // values [5,0],[0,5],[0,-5]
        let (k, v) = project_kv(&Compute::cpu(), &xkv, 3, &attn).unwrap();
        let xq = vec![0.0, 0.0]; // uniform scores pre-mask
        let out = attention_from_kv(
            &Compute::cpu(),
            &xq,
            1,
            &k,
            &v,
            3,
            &attn.q,
            &attn.out,
            1,
            true,
            0,
        )
        .unwrap();
        // Only key 0 visible → context == value 0 == [5,0].
        assert!((out[0] - 5.0).abs() < 1e-4, "{out:?}");
        assert!(out[1].abs() < 1e-4, "{out:?}");
    }
}
