//! Forward-pass building blocks shared by the encoder and decoder.
//!
//! These are thin, correctness-first wrappers over the M0-08
//! `vokra-backend-cpu` compute kernels (`gemm_f32`, `layer_norm_f32`,
//! `gelu_f32`, `add_f32`, `softmax_f32`), plus a plain-Rust multi-head
//! attention assembled from them. Shapes are row-major throughout; a shape
//! mismatch surfaces as the kernel's [`VokraError::InvalidArgument`]
//! (NFR-RL-07). Static-arena buffer reuse (FR-EX-05) is M1 — M0 allocates per
//! call, mirroring the existing `conv1d` im2col path.

use vokra_backend_cpu::kernels::{
    LAYER_NORM_DEFAULT_EPS, add_f32, gelu_f32, gemm_f32, layer_norm_f32, softmax_f32,
};
use vokra_core::Result;

use super::weights::{Attention, LayerNorm, Linear};

/// Affine layer norm over the innermost axis: `[rows, d] → [rows, d]`.
///
/// Uses the PyTorch/Whisper default epsilon (`1e-5`).
pub(crate) fn layer_norm(x: &[f32], rows: usize, ln: &LayerNorm) -> Result<Vec<f32>> {
    let d = ln.gamma.len();
    let mut out = vec![0.0f32; rows * d];
    layer_norm_f32(
        x,
        &mut out,
        rows,
        d,
        &ln.gamma,
        &ln.beta,
        LAYER_NORM_DEFAULT_EPS,
    )?;
    Ok(out)
}

/// Applies an `nn.Linear`: `y[t, o] = bias[o] + sum_i x[t, i] * w_t[i, o]`,
/// with `x` shaped `[t, in_features]` and the result `[t, out_features]`.
pub(crate) fn linear(x: &[f32], t: usize, lin: &Linear) -> Result<Vec<f32>> {
    let mut out = vec![0.0f32; t * lin.out_features];
    gemm_f32(
        t,
        lin.out_features,
        lin.in_features,
        x,
        &lin.w_t,
        lin.bias.as_deref(),
        &mut out,
    )?;
    Ok(out)
}

/// Exact (erf) GELU, matching Whisper's `nn.GELU()`.
pub(crate) fn gelu(x: &[f32]) -> Result<Vec<f32>> {
    let mut out = vec![0.0f32; x.len()];
    gelu_f32(x, &mut out)?;
    Ok(out)
}

/// Element-wise `a += b` (residual add).
pub(crate) fn add_into(a: &mut [f32], b: &[f32]) -> Result<()> {
    // add_f32 writes to a separate output; do it in place via a temp view.
    let mut out = vec![0.0f32; a.len()];
    add_f32(a, b, &mut out)?;
    a.copy_from_slice(&out);
    Ok(())
}

/// The MLP sub-block: `fc2(gelu(fc1(x)))`, `x` shaped `[t, d]`.
pub(crate) fn mlp(x: &[f32], t: usize, fc1: &Linear, fc2: &Linear) -> Result<Vec<f32>> {
    let h = linear(x, t, fc1)?;
    let a = gelu(&h)?;
    linear(&a, t, fc2)
}

/// Projects the key/value inputs of an attention block: returns
/// `(k, v)` each `[t_kv, d]`.
pub(crate) fn project_kv(
    xkv: &[f32],
    t_kv: usize,
    attn: &Attention,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let k = linear(xkv, t_kv, &attn.k)?;
    let v = linear(xkv, t_kv, &attn.v)?;
    Ok((k, v))
}

/// Multi-head attention from **pre-projected** keys/values.
///
/// - `xq` is `[t_q, d]`; `k` / `v` are `[t_kv, d]` (already `k_proj`/`v_proj`);
/// - `q_pos_offset` is the absolute position of `xq[0]` (the keys span absolute
///   positions `0..t_kv`); used only when `causal` so a decode step masks
///   future keys correctly;
/// - the query is scaled by `head_dim^-0.5` (Whisper applies the scale to the
///   query, not the scores).
///
/// Returns the block output `[t_q, d]` after the output projection.
#[allow(clippy::too_many_arguments)]
pub(crate) fn attention_from_kv(
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
    let d = q_lin.out_features;
    let hd = d / n_head;
    let scale = (hd as f32).powf(-0.5);

    // Scaled query projection.
    let mut q = linear(xq, t_q, q_lin)?;
    for v in &mut q {
        *v *= scale;
    }

    // Per-head attention, writing each head's output into the [t_q, d] concat.
    let mut context = vec![0.0f32; t_q * d];
    let mut scores = vec![0.0f32; t_q * t_kv];
    let mut probs = vec![0.0f32; t_q * t_kv];
    // Head slice buffers.
    let mut qh = vec![0.0f32; t_q * hd];
    let mut kh_t = vec![0.0f32; hd * t_kv]; // k head transposed to [hd, t_kv]
    let mut vh = vec![0.0f32; t_kv * hd];
    let mut ctx_h = vec![0.0f32; t_q * hd];

    for h in 0..n_head {
        let c0 = h * hd;
        // Gather this head's q [t_q, hd] and v [t_kv, hd]; k transposed [hd, t_kv].
        for i in 0..t_q {
            qh[i * hd..i * hd + hd].copy_from_slice(&q[i * d + c0..i * d + c0 + hd]);
        }
        for j in 0..t_kv {
            vh[j * hd..j * hd + hd].copy_from_slice(&v[j * d + c0..j * d + c0 + hd]);
            for c in 0..hd {
                kh_t[c * t_kv + j] = k[j * d + c0 + c];
            }
        }
        // scores [t_q, t_kv] = qh [t_q, hd] @ kh_t [hd, t_kv].
        gemm_f32(t_q, t_kv, hd, &qh, &kh_t, None, &mut scores)?;
        // Causal mask: query i (abs pos q_pos_offset + i) may not attend key j
        // when j > q_pos_offset + i.
        if causal {
            for i in 0..t_q {
                let last = q_pos_offset + i;
                for j in (last + 1)..t_kv {
                    scores[i * t_kv + j] = f32::NEG_INFINITY;
                }
            }
        }
        softmax_f32(&scores, &mut probs, t_q, t_kv)?;
        // ctx_h [t_q, hd] = probs [t_q, t_kv] @ vh [t_kv, hd].
        gemm_f32(t_q, hd, t_kv, &probs, &vh, None, &mut ctx_h)?;
        for i in 0..t_q {
            context[i * d + c0..i * d + c0 + hd].copy_from_slice(&ctx_h[i * hd..i * hd + hd]);
        }
    }

    linear(&context, t_q, out_lin)
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
        let (k, v) = project_kv(&xkv, 2, &attn).unwrap();
        // One query aligned with key 0.
        let xq = vec![10.0, 0.0];
        let out = attention_from_kv(&xq, 1, &k, &v, 2, &attn.q, &attn.out, 1, false, 0).unwrap();
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
        let (k, v) = project_kv(&xkv, 3, &attn).unwrap();
        let xq = vec![0.0, 0.0]; // uniform scores pre-mask
        let out = attention_from_kv(&xq, 1, &k, &v, 3, &attn.q, &attn.out, 1, true, 0).unwrap();
        // Only key 0 visible → context == value 0 == [5,0].
        assert!((out[0] - 5.0).abs() < 1e-4, "{out:?}");
        assert!(out[1].abs() < 1e-4, "{out:?}");
    }
}
