//! Public factory helpers for building tiny synthesized Voxtral models
//! for integration tests (M3-10 beam search + n-best decode).
//!
//! Integration tests live in `crates/vokra-models/tests/` and can only see
//! `pub` items — the internal weight structs (`Linear`, `GqaAttention`,
//! `SwiGluFfn`, `DecoderBlock`) are `pub(crate)` so they don't leak into
//! the public API. This module exposes narrowly-scoped factories that
//! construct a small deterministic model against those internals, so
//! integration tests can drive the beam search end-to-end without a real
//! upstream Voxtral checkpoint.
//!
//! The helpers are `#[doc(hidden)]` — they are NOT a stable public
//! surface. External callers (Unity / Godot / vokra-server) never
//! construct Voxtral models this way; they load from a GGUF via
//! [`super::VoxtralModel::from_gguf`].

use super::config::{AudioEncoderConfig, TextDecoderConfig};
use super::text_decoder::{DecoderBlock, GqaAttention, Linear, SwiGluFfn};
use super::{AudioEncoder, TextDecoder, VoxtralConfig};

/// A small VoxtralConfig with `n_layer = 1`, `n_head_q = 2, n_head_kv = 1`
/// (GQA 2:1 split), `hidden_dim = 4`, `vocab_size = 8`, `n_ctx = 16`.
///
/// The audio side is `n_layer = 1`, `hidden_dim = 4`, `n_mels = 2`,
/// `n_ctx = 8`. Enough headroom for beam decodes of up to 8 new tokens.
#[doc(hidden)]
#[must_use]
pub fn tiny_config() -> VoxtralConfig {
    VoxtralConfig {
        audio: AudioEncoderConfig {
            n_layer: 1,
            n_head: 2,
            hidden_dim: 4,
            n_ctx: 8,
            n_mels: 2,
            ffn_dim: 8,
        },
        text: TextDecoderConfig {
            n_layer: 1,
            n_head_q: 2,
            n_head_kv: 1,
            hidden_dim: 4,
            ffn_dim: 8,
            vocab_size: 8,
            n_ctx: 16,
            rope_base: 10_000.0,
            rms_norm_eps: 1e-5,
        },
        cross_attn_hidden_dim: 4,
        mode: "asr".to_owned(),
        s2s_codec_type: "none".to_owned(),
    }
}

/// An audio encoder shaped to [`tiny_config`] with non-zero learned
/// positional embeddings — the beam-search integration tests need
/// non-zero encoder output so the adapter-conditioned path meaningfully
/// diverges from the LM-prior path.
#[doc(hidden)]
#[must_use]
pub fn tiny_encoder(cfg: &VoxtralConfig) -> AudioEncoder {
    let mut ae = AudioEncoder {
        conv1_w: vec![0.0; cfg.audio.hidden_dim * cfg.audio.n_mels * 3],
        conv1_b: vec![0.0; cfg.audio.hidden_dim],
        conv2_w: vec![0.0; cfg.audio.hidden_dim * cfg.audio.hidden_dim * 3],
        conv2_b: vec![0.0; cfg.audio.hidden_dim],
        pos_emb: vec![0.0; cfg.audio.n_ctx * cfg.audio.hidden_dim],
        has_learned_pos_emb: true,
    };
    for (i, v) in ae.pos_emb.iter_mut().enumerate() {
        *v = ((i as i32 % 3) - 1) as f32 * 0.1;
    }
    ae
}

/// A text decoder shaped to [`tiny_config`] with the deterministic weight
/// initialization pattern shared by the unit tests in `text_decoder_session.rs`.
#[doc(hidden)]
#[must_use]
pub fn tiny_decoder(cfg: &VoxtralConfig) -> TextDecoder {
    let d = cfg.text.hidden_dim;
    let ffn = cfg.text.ffn_dim;
    let vocab = cfg.text.vocab_size;
    let head_dim = d / cfg.text.n_head_q;
    let kv_hidden = cfg.text.n_head_kv * head_dim;

    let mut token_emb = vec![0.0f32; vocab * d];
    for (i, v) in token_emb.iter_mut().enumerate() {
        *v = ((i as i32 % 7) - 3) as f32 * 0.05;
    }
    fn linear(rows: usize, cols: usize, base: f32) -> Linear {
        let mut w_t = vec![0.0f32; rows * cols];
        for (i, v) in w_t.iter_mut().enumerate() {
            *v = base + 0.01 * ((i as i32 % 5) - 2) as f32;
        }
        Linear {
            w_t,
            in_features: rows,
            out_features: cols,
        }
    }
    let blocks = (0..cfg.text.n_layer)
        .map(|_| DecoderBlock {
            attn_norm_gamma: vec![1.0f32; d],
            attn: GqaAttention {
                q: linear(d, d, 0.10),
                k: linear(d, kv_hidden, -0.07),
                v: linear(d, kv_hidden, 0.05),
                o: linear(d, d, -0.04),
            },
            ffn_norm_gamma: vec![1.0f32; d],
            ffn: SwiGluFfn {
                gate: linear(d, ffn, 0.06),
                up: linear(d, ffn, -0.02),
                down: linear(ffn, d, 0.03),
            },
        })
        .collect();
    TextDecoder {
        token_emb,
        blocks,
        final_norm_gamma: vec![1.0f32; d],
        prefix: "",
    }
}
