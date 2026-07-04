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

use vokra_core::{Result, VokraError};

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
// ZERO-ALLOC-BEGIN — per-block forward into reused scratch; guarded by
// scripts/check-hot-path-allocs.sh (no vec![], Vec::with_capacity, .to_vec(),
// .collect()). The conv stem in `encode` above allocates once per utterance and
// is intentionally outside this region.
#[allow(clippy::too_many_arguments)] // block shape (d, ff, n_head) + scratch + h
fn encoder_block(
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
