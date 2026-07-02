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

use vokra_backend_cpu::kernels::{conv1d_f32, gelu_f32};
use vokra_core::{Result, VokraError};

use super::config::WhisperConfig;
use super::nn::{add_into, attention_from_kv, layer_norm, mlp, project_kv};
use super::weights::{EncoderLayer, EncoderWeights};

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

/// Runs the encoder on `[n_mels, n_frames]` log-mel features.
pub(crate) fn encode(
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
    conv1d_f32(
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
    gelu_inplace(&mut c1)?;

    // conv2: [d, len1] -> [d, len2] (stride 2, pad 1), then GELU.
    let len2 = conv_out_len(len1, 3, 2, 1);
    let mut c2 = vec![0.0f32; d * len2];
    conv1d_f32(
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
    gelu_inplace(&mut c2)?;

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

    // Pre-norm self-attention blocks.
    for layer in &w.layers {
        encoder_block(&mut hidden, t, cfg.n_audio_head, layer)?;
    }

    // Final LayerNorm.
    let hidden = layer_norm(&hidden, t, &w.ln_post)?;

    Ok(EncoderOutput {
        hidden,
        n_ctx: t,
        d_model: d,
    })
}

/// One encoder block: `h += self_attn(ln(h))`, then `h += mlp(ln(h))`.
fn encoder_block(h: &mut [f32], t: usize, n_head: usize, layer: &EncoderLayer) -> Result<()> {
    let normed = layer_norm(h, t, &layer.attn_ln)?;
    let (k, v) = project_kv(&normed, t, &layer.attn)?;
    let attn = attention_from_kv(
        &normed,
        t,
        &k,
        &v,
        t,
        &layer.attn.q,
        &layer.attn.out,
        n_head,
        false,
        0,
    )?;
    add_into(h, &attn)?;

    let normed = layer_norm(h, t, &layer.mlp_ln)?;
    let ff = mlp(&normed, t, &layer.fc1, &layer.fc2)?;
    add_into(h, &ff)?;
    Ok(())
}

/// Conv1d output length for the given kernel / stride / padding.
fn conv_out_len(in_len: usize, kernel: usize, stride: usize, pad: usize) -> usize {
    (in_len + 2 * pad - kernel) / stride + 1
}

fn gelu_inplace(x: &mut [f32]) -> Result<()> {
    let mut out = vec![0.0f32; x.len()];
    gelu_f32(x, &mut out)?;
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
