//! Whisper audio encoder forward pass.
//!
//! Structure (openai/whisper `AudioEncoder`, HF `WhisperEncoder`):
//!
//! 1. `conv1`: Conv1d(`n_mels → d`, kernel 3, stride 1, pad 1) + GELU;
//! 2. `conv2`: Conv1d(`d → d`, kernel 3, stride 2, pad 1) + GELU (halves the
//!    time axis: 3000 → 1500 = `n_audio_ctx`);
//! 3. transpose `[d, 1500] → [1500, d]` and add the sinusoidal positional
//!    embedding;
//! 4. `n_audio_layer` pre-norm self-attention blocks (bidirectional — no mask);
//! 5. a final LayerNorm.
//!
//! All matmul / conv / norm / activation work is done by the M0-08
//! `vokra-backend-cpu` kernels via [`super::nn`]; this file only wires shapes
//! and residuals.

use vokra_backend_cpu::kernels::LAYER_NORM_DEFAULT_EPS;
use vokra_core::{PrenormLayer, Result, VokraError};

use super::config::WhisperConfig;
use super::nn::{add_assign, attention_from_kv_into, layer_norm_into, mlp_into, project_kv_into};
use super::scratch::{BlockScratch, EncoderScratch};
use super::weights::{EncoderLayer, EncoderWeights};
use crate::compute::Compute;

/// Encoder hidden states `[n_ctx, d_model]` (row-major).
#[derive(Debug, Clone)]
pub struct EncoderOutput {
    /// Row-major `[n_ctx, d_model]` hidden states.
    pub hidden: Vec<f32>,
    /// Number of audio context positions (1500 for base).
    pub n_ctx: usize,
    /// Hidden width.
    pub d_model: usize,
}

/// Runs the encoder on `[n_mels, n_frames]` log-mel features, dispatching the
/// conv / GEMM / softmax / layer-norm / GELU hot path through `compute` (the
/// backend seam; the CPU dispatcher reproduces the pre-seam kernel calls
/// bit-for-bit — M2-01 Phase 3).
pub(crate) fn encode(
    compute: &Compute,
    cfg: &WhisperConfig,
    w: &EncoderWeights,
    log_mel: &[f32],
    n_frames: usize,
) -> Result<EncoderOutput> {
    let d = cfg.d_model;
    if log_mel.len() != cfg.n_mels * n_frames {
        return Err(VokraError::InvalidArgument(format!(
            "whisper encoder: log-mel len {} != n_mels*n_frames {}",
            log_mel.len(),
            cfg.n_mels * n_frames
        )));
    }

    // conv1: [n_mels, n_frames] -> [d, n_frames] (stride 1, pad 1), then GELU.
    let len1 = conv_out_len(n_frames, 3, 1, 1);
    let mut c1 = vec![0.0f32; d * len1];
    compute.conv1d_f32(
        log_mel,
        cfg.n_mels,
        n_frames,
        &w.conv1_w,
        d,
        3,
        Some(&w.conv1_b),
        1,
        1,
        &mut c1,
    )?;
    gelu_inplace(compute, &mut c1)?;

    // conv2: [d, len1] -> [d, len2] (stride 2, pad 1), then GELU.
    let len2 = conv_out_len(len1, 3, 2, 1);
    let mut c2 = vec![0.0f32; d * len2];
    compute.conv1d_f32(
        &c1,
        d,
        len1,
        &w.conv2_w,
        d,
        3,
        Some(&w.conv2_b),
        2,
        1,
        &mut c2,
    )?;
    gelu_inplace(compute, &mut c2)?;

    if len2 != cfg.n_audio_ctx {
        return Err(VokraError::InvalidArgument(format!(
            "whisper encoder: conv output length {len2} != n_audio_ctx {}",
            cfg.n_audio_ctx
        )));
    }
    let t = len2;

    // Transpose [d, t] -> [t, d] and add positional embedding.
    let mut hidden = vec![0.0f32; t * d];
    for c in 0..d {
        for i in 0..t {
            hidden[i * d + c] = c2[c * t + i] + w.pos_emb[i * d + c];
        }
    }

    // Phase-5-follow-on fused path: on a GPU backend run the WHOLE pre-norm block
    // stack (+ final LayerNorm) device-resident in ONE submission
    // (`Compute::encode_prenorm_encoder`), keeping the hidden state and every
    // intermediate on the GPU across all blocks — bit-identical to the per-op
    // `encoder_block` loop below but collapsing its `6·N + 1` command-buffer syncs
    // to one. Gated on `prenorm_stack_is_fused()` so the CPU always takes the
    // untouched per-op loop (no silent fall back, FR-EX-08). The `Vec<PrenormLayer>`
    // + `normed` allocations sit here in `encode()`, OUTSIDE the ZERO-ALLOC
    // `encoder_block` region, so the hot-path alloc guard stays green.
    if compute.prenorm_stack_is_fused() {
        let layers: Vec<PrenormLayer<'_>> =
            w.layers.iter().map(prenorm_view).collect::<Result<_>>()?;
        let mut normed = vec![0.0f32; t * d];
        compute.encode_prenorm_encoder(
            t,
            d,
            cfg.ffn_dim,
            cfg.n_audio_head,
            LAYER_NORM_DEFAULT_EPS,
            &hidden,
            &layers,
            &w.ln_post.gamma,
            &w.ln_post.beta,
            &mut normed,
        )?;
        return Ok(EncoderOutput {
            hidden: normed,
            n_ctx: t,
            d_model: d,
        });
    }

    // Pre-norm self-attention blocks. One `EncoderScratch` is reserved to the
    // audio context length and reused across every block — the encoder is
    // bidirectional (`t_q == t_kv == t`), so nothing grows between blocks and
    // the per-block transients allocate only on the first block.
    let mut scratch = EncoderScratch::with_reserve(t, d, cfg.ffn_dim, cfg.n_audio_head);
    for layer in &w.layers {
        encoder_block(
            compute,
            &mut scratch.block,
            &mut hidden,
            t,
            d,
            cfg.ffn_dim,
            cfg.n_audio_head,
            layer,
        )?;
    }

    // Final LayerNorm into the returned buffer.
    let mut normed = Vec::new();
    layer_norm_into(compute, &mut normed, &hidden, t, &w.ln_post)?;

    Ok(EncoderOutput {
        hidden: normed,
        n_ctx: t,
        d_model: d,
    })
}

/// One encoder block: `h += self_attn(ln(h))`, then `h += mlp(ln(h))`, using the
/// reused `scratch` for every intermediate (no per-block allocation).
///
/// `pub(crate)`: the Voxtral audio tower (`crate::voxtral::audio_encoder`) is
/// the same pre-norm block (HF `modeling_voxtral.py` `VoxtralEncoderLayer`
/// forward == HF `WhisperEncoderLayer` forward), so it drives this function
/// verbatim — one audited block implementation for both models.
// ZERO-ALLOC-BEGIN — per-block forward into reused scratch; guarded by
// scripts/check-hot-path-allocs.sh (no vec![], Vec::with_capacity, .to_vec(),
// .collect()). The conv stem in `encode` above allocates once per utterance and
// is intentionally outside this region.
#[allow(clippy::too_many_arguments)] // block shape (d, ff, n_head) + scratch + h
pub(crate) fn encoder_block(
    compute: &Compute,
    scratch: &mut BlockScratch,
    h: &mut [f32],
    t: usize,
    d: usize,
    ff: usize,
    n_head: usize,
    layer: &EncoderLayer,
) -> Result<()> {
    scratch.ensure_residual(t, d, ff);

    layer_norm_into(compute, &mut scratch.ln, h, t, &layer.attn_ln)?;
    project_kv_into(
        compute,
        &mut scratch.k,
        &mut scratch.v,
        &scratch.ln,
        t,
        &layer.attn,
    )?;
    attention_from_kv_into(
        compute,
        &mut scratch.attn,
        &scratch.ln,
        t,
        &scratch.k,
        &scratch.v,
        t,
        &layer.attn.q,
        &layer.attn.out,
        n_head,
        false,
        0,
        &mut scratch.block_out,
    )?;
    add_assign(h, &scratch.block_out)?;

    layer_norm_into(compute, &mut scratch.ln, h, t, &layer.mlp_ln)?;
    mlp_into(
        compute,
        &mut scratch.mlp_h,
        &mut scratch.mlp_a,
        &mut scratch.block_out,
        &scratch.ln,
        t,
        &layer.fc1,
        &layer.fc2,
    )?;
    add_assign(h, &scratch.block_out)?;
    Ok(())
}
// ZERO-ALLOC-END

/// Borrows one [`EncoderLayer`]'s weights as a backend-agnostic
/// [`PrenormLayer`] slice view for the fused device-resident encoder
/// ([`Compute::encode_prenorm_encoder`]). Whisper's `k_proj` has no bias
/// (`k_bias: None`); every other projection carries one. Called once per block in
/// `encode()`, off the ZERO-ALLOC hot region. `pub(crate)` so the Voxtral
/// audio tower's fused-stack path reuses the identical view.
/// # Errors
///
/// [`VokraError::UnsupportedOp`] when any projection kept its K-quant
/// super-blocks (M5-15): this view exists to hand a device f32 weight
/// pointers, and the fused K-quant path is CPU-only. Failing here is the
/// FR-EX-08-correct outcome — the alternative is uploading a weight that is
/// not there.
pub(crate) fn prenorm_view(l: &EncoderLayer) -> Result<PrenormLayer<'_>> {
    Ok(PrenormLayer {
        attn_ln_gamma: &l.attn_ln.gamma,
        attn_ln_beta: &l.attn_ln.beta,
        q_w: l.attn.q.dense_w_t()?,
        q_bias: l.attn.q.bias.as_deref(),
        k_w: l.attn.k.dense_w_t()?,
        k_bias: l.attn.k.bias.as_deref(),
        v_w: l.attn.v.dense_w_t()?,
        v_bias: l.attn.v.bias.as_deref(),
        out_w: l.attn.out.dense_w_t()?,
        out_bias: l.attn.out.bias.as_deref(),
        mlp_ln_gamma: &l.mlp_ln.gamma,
        mlp_ln_beta: &l.mlp_ln.beta,
        fc1_w: l.fc1.dense_w_t()?,
        fc1_bias: l.fc1.bias.as_deref(),
        fc2_w: l.fc2.dense_w_t()?,
        fc2_bias: l.fc2.bias.as_deref(),
    })
}

/// Conv1d output length for the given kernel / stride / padding.
fn conv_out_len(in_len: usize, kernel: usize, stride: usize, pad: usize) -> usize {
    (in_len + 2 * pad - kernel) / stride + 1
}

fn gelu_inplace(compute: &Compute, x: &mut [f32]) -> Result<()> {
    let mut out = vec![0.0f32; x.len()];
    compute.gelu_f32(x, &mut out)?;
    x.copy_from_slice(&out);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conv_out_len_matches_whisper_stem() {
        assert_eq!(conv_out_len(3000, 3, 1, 1), 3000); // conv1
        assert_eq!(conv_out_len(3000, 3, 2, 1), 1500); // conv2
    }
}
