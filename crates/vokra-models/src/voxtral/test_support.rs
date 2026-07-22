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
use super::{
    AudioAdapter, AudioEncoder, TextDecoder, VoxtralConfig, VoxtralModel, VoxtralTokenizer,
};
use crate::whisper::weights::{
    Attention as WwAttention, EncoderLayer, LayerNorm as WwLayerNorm, Linear as WwLinear,
    LinearWeight as WwLinearWeight,
};
use vokra_core::gguf::{GgmlType, GgufBuilder, GgufFile};

/// A small VoxtralConfig with `n_layer = 1`, `n_head_q = 2, n_head_kv = 1`
/// (GQA 2:1 split), `hidden_dim = 4`, `vocab_size = 8`, `n_ctx = 2048`.
///
/// The audio side is `n_layer = 1`, `hidden_dim = 4`, `n_mels = 2`,
/// **`n_ctx = 1500`** — the real Voxtral positional length. The full-stack
/// audio encoder mirrors upstream's strict input contract (the mel window
/// must be exactly `2 * n_ctx` frames — see
/// [`super::audio_encoder::forward`]), and the `VoxtralAsr::transcribe`
/// front-end always produces the 30 s / 3000-frame Whisper window, so any
/// fixture that reaches the PCM → mel path must carry the real 1500-position
/// geometry. The text side's `n_ctx = 2048` leaves headroom for the
/// 1500-row adapter soft-prefix + BOS + generated tokens.
#[doc(hidden)]
#[must_use]
pub fn tiny_config() -> VoxtralConfig {
    VoxtralConfig {
        audio: AudioEncoderConfig {
            n_layer: 1,
            n_head: 2,
            hidden_dim: 4,
            n_ctx: 1500,
            n_mels: 2,
            ffn_dim: 8,
        },
        text: TextDecoderConfig {
            n_layer: 1,
            n_head_q: 2,
            n_head_kv: 1,
            head_dim: 0,
            hidden_dim: 4,
            ffn_dim: 8,
            vocab_size: 8,
            n_ctx: 2048,
            rope_base: 10_000.0,
            rms_norm_eps: 1e-5,
        },
        cross_attn_hidden_dim: 4,
        mode: "asr".to_owned(),
        s2s_codec_type: "none".to_owned(),
    }
}

/// An identity-affine LayerNorm (γ=1, β=0) of width `d` — the neutral
/// final-LN fixture the synthetic encoders use.
#[must_use]
pub(crate) fn identity_ln(d: usize) -> WwLayerNorm {
    WwLayerNorm {
        gamma: vec![1.0; d],
        beta: vec![0.0; d],
    }
}

/// `n_layer` pass-through transformer blocks shaped to `cfg`: zero
/// projection weights (with zero biases where upstream has them, bias-less
/// `k_proj`) and **identity LayerNorms** (γ=1, β=0). Each block therefore
/// adds exactly `0` to the residual stream — the encoder output is the
/// final LayerNorm of the conv+pos hidden, keeping the pre-full-stack
/// fixture semantics (deterministic, non-zero when the pos table is
/// non-zero) while exercising the real block loop.
#[doc(hidden)]
#[must_use]
pub(crate) fn passthrough_layers(cfg: &VoxtralConfig) -> Vec<EncoderLayer> {
    let d = cfg.audio.hidden_dim;
    let ff = cfg.audio.ffn_dim;
    let zero_linear = |rows: usize, cols: usize, bias: bool| WwLinear {
        w: WwLinearWeight::Dense(vec![0.0; rows * cols]),
        in_features: rows,
        out_features: cols,
        bias: bias.then(|| vec![0.0; cols]),
    };
    let identity_ln = || WwLayerNorm {
        gamma: vec![1.0; d],
        beta: vec![0.0; d],
    };
    (0..cfg.audio.n_layer)
        .map(|_| EncoderLayer {
            attn_ln: identity_ln(),
            attn: WwAttention {
                q: zero_linear(d, d, true),
                k: zero_linear(d, d, false),
                v: zero_linear(d, d, true),
                out: zero_linear(d, d, true),
            },
            mlp_ln: identity_ln(),
            fc1: zero_linear(d, ff, true),
            fc2: zero_linear(ff, d, true),
        })
        .collect()
}

/// An audio encoder shaped to [`tiny_config`] with non-zero learned
/// positional embeddings — the beam-search integration tests need
/// non-zero encoder output so the adapter-conditioned path meaningfully
/// diverges from the LM-prior path. The transformer blocks are
/// [`passthrough_layers`] (zero-weight, identity-LN) and the final
/// LayerNorm is identity-affine (γ=1, β=0), so the output is the
/// normalized conv+pos hidden — deterministic and non-zero.
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
        layers: passthrough_layers(cfg),
        ln_post: identity_ln(cfg.audio.hidden_dim),
    };
    for (i, v) in ae.pos_emb.iter_mut().enumerate() {
        *v = ((i as i32 % 3) - 1) as f32 * 0.1;
    }
    ae
}

/// A text decoder shaped to the given config with the deterministic weight
/// initialization pattern shared by the unit tests in `text_decoder_session.rs`.
///
/// Projection shapes come off the config's [`TextDecoderConfig::q_hidden`] /
/// [`kv_hidden`](TextDecoderConfig::kv_hidden) helpers, so the same factory
/// serves both the head_dim-tied [`tiny_config`] (`q_hidden == hidden_dim`)
/// and the decoupled [`gqa_config`] (`q_hidden != hidden_dim` — the real
/// Voxtral-mini shape class).
#[doc(hidden)]
#[must_use]
pub fn tiny_decoder(cfg: &VoxtralConfig) -> TextDecoder {
    let d = cfg.text.hidden_dim;
    let ffn = cfg.text.ffn_dim;
    let vocab = cfg.text.vocab_size;
    let q_hidden = cfg.text.q_hidden();
    let kv_hidden = cfg.text.kv_hidden();

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
                q: linear(d, q_hidden, 0.10),
                k: linear(d, kv_hidden, -0.07),
                v: linear(d, kv_hidden, 0.05),
                o: linear(q_hidden, d, -0.04),
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
        lm_head: None,
        blocks,
        final_norm_gamma: vec![1.0f32; d],
        prefix: "",
        mapped: None,
    }
}

/// A [`VoxtralConfig`] whose text decoder has an explicit `head_dim`
/// **decoupled** from `hidden_dim / n_head_q` — the real Voxtral-mini shape
/// class (`hidden_dim = 3072`, 32 × 128 = 4096-wide Q), scaled down:
/// `hidden_dim = 6`, `n_head_q = 2`, `n_head_kv = 1`, `head_dim = 4` →
/// `q_hidden = 8 ≠ 6`, `kv_hidden = 4`. Pair with [`tiny_decoder`] (which
/// reads the projection shapes off the config).
#[doc(hidden)]
#[must_use]
pub fn gqa_config() -> VoxtralConfig {
    let mut cfg = tiny_config();
    cfg.text.hidden_dim = 6;
    cfg.text.n_head_q = 2;
    cfg.text.n_head_kv = 1;
    cfg.text.head_dim = 4;
    // Keep the audio/cross-attn widths in lock-step with the text residual
    // width so the adapter-less ASR wiring stays shape-consistent.
    cfg.audio.hidden_dim = 6;
    cfg.cross_attn_hidden_dim = 6;
    cfg
}

/// A full [`VoxtralModel`] wired from [`tiny_config`] + [`tiny_encoder`] +
/// [`tiny_decoder`] with an inactive ([`AudioAdapter::none`]) adapter.
///
/// Load-bearing for the M3-10 GPU integration tests that need to drive
/// [`super::VoxtralAsr`] end-to-end without a real upstream checkpoint.
/// The Wave 10 `voxtral_gpu_session.rs` beam × Metal tests use this.
#[doc(hidden)]
#[must_use]
pub fn tiny_voxtral_model() -> VoxtralModel {
    let cfg = tiny_config();
    let audio = tiny_encoder(&cfg);
    let text = tiny_decoder(&cfg);
    VoxtralModel {
        config: cfg,
        audio,
        text,
        audio_adapter: AudioAdapter::none(),
    }
}

/// A full [`VoxtralModel`] identical to [`tiny_voxtral_model`] but wired
/// with an identity-weight `Linear` [`AudioAdapter`] so the integration
/// tests can exercise the Wave 8 soft-prefix conditioning path.
///
/// The adapter is constructed from a synthetic GGUF blob (same shape the
/// converter would emit for a real upstream `audio_adapter.weight`
/// tensor) so the load path is the real one — not a private constructor.
///
/// # Panics
///
/// Panics if the synthetic GGUF cannot be parsed / the adapter cannot be
/// loaded — this is a test-only helper and both would indicate a broken
/// fixture / regression in [`AudioAdapter::from_gguf`].
#[doc(hidden)]
#[must_use]
pub fn tiny_voxtral_model_with_linear_adapter() -> VoxtralModel {
    let cfg = tiny_config();
    let audio = tiny_encoder(&cfg);
    let text = tiny_decoder(&cfg);
    let d = cfg.text.hidden_dim;

    // Identity linear adapter — same shape the `voxtral_beam_search.rs`
    // integration tests build (kept in sync intentionally so the two
    // suites cover the same adapter routing).
    let mut b = GgufBuilder::new();
    b.add_string("vokra.voxtral.adapter.kind", "linear");
    b.add_string("vokra.voxtral.adapter.tensor_prefix", "audio_adapter.");
    b.add_u32("vokra.voxtral.adapter.in_dim", d as u32);
    b.add_u32("vokra.voxtral.adapter.out_dim", d as u32);
    b.add_bool("vokra.voxtral.adapter.has_bias", false);
    b.add_bool("vokra.voxtral.adapter.has_layernorm", false);
    let mut w = vec![0.0f32; d * d];
    for i in 0..d {
        w[i * d + i] = 1.0;
    }
    b.add_tensor(
        "audio_adapter.weight",
        GgmlType::F32,
        vec![d as u64, d as u64],
        w.iter().flat_map(|v| v.to_le_bytes()).collect(),
    )
    .expect("build audio_adapter.weight");
    let bytes = b.to_bytes().expect("serialize adapter GGUF");
    let file = GgufFile::parse(bytes).expect("parse adapter GGUF");
    let audio_adapter = AudioAdapter::from_gguf(&file).expect("load LinearAdapter");
    assert!(
        audio_adapter.is_active(),
        "identity linear adapter must be active"
    );

    VoxtralModel {
        config: cfg,
        audio,
        text,
        audio_adapter,
    }
}

/// A compact-vocab tokenizer covering ids `0..vocab_size` with
/// `id -> "t{id} "` renderings. `eos` is the id `<eos>` maps to (does
/// not need to be inside the vocab; the beam / greedy stops on that
/// exact id).
///
/// # Panics
///
/// Panics if the synthetic tokenizer blob cannot be parsed by
/// [`VoxtralTokenizer::from_bytes`] — indicates a test-only bug or
/// a regression in the parser.
#[doc(hidden)]
#[must_use]
pub fn tiny_tokenizer(vocab_size: usize, eos: u32) -> VoxtralTokenizer {
    // Compact-vocab dump format: u32 count + records.
    let mut blob = (vocab_size as u32).to_le_bytes().to_vec();
    for id in 0..vocab_size {
        let s = format!("t{id} ");
        let bytes = s.as_bytes();
        blob.push(0u8); // not special
        blob.extend_from_slice(&(bytes.len() as u16).to_le_bytes());
        blob.extend_from_slice(bytes);
    }
    VoxtralTokenizer::from_bytes(blob, eos).expect("build tiny_tokenizer")
}
