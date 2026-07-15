//! Moshi (Helium + Mimi) — native full-duplex S2S with inner monologue
//! (M4-06, FR-MD-09).
//!
//! # What Moshi is (ADR M4-06 §D2, upstream-verified)
//!
//! Moshi (Kyutai; code Apache 2.0, weights **CC-BY 4.0** —
//! `AttributionRequired`, the attribution surface is FR-MD-09-mandatory)
//! is a **full-duplex** speech dialog model: a single temporal transformer
//! (**Helium**, 7B) consumes, every 80 ms Mimi frame, the sum of 17 token
//! channels — 1 text channel (the **inner monologue**, which the model
//! itself generates), 8 own-audio channels, and 8 user-audio channels —
//! and a small per-frame **depformer** autoregresses the 8 own-audio
//! codebooks. Unlike CSM (M4-05) the caller supplies **no reply text**:
//! Moshi derives its reply and its transcript simultaneously.
//!
//! # Architecture (whisper.cpp-style native re-implementation)
//!
//! Every value is transcribed from `kyutai-labs/moshi` (`loaders.py`
//! `_lm_kwargs`, `lm.py`, `transformer.py`, `gating.py`, `rope.py` — ADR
//! M4-06 §D2) and cross-checked against the real
//! `kyutai/moshiko-pytorch-bf16` tensor manifest
//! (`tests/parity/moshi/moshiko_tensor_manifest.json`, 355 tensors) —
//! never invented:
//!
//! - [`MoshiBackbone`] — pre-norm `rms_norm_f32` (ε=1e-8) blocks, MHA +
//!   interleaved-pair RoPE (max_period 10 000), SiLU **gating** FFN
//!   (`hidden = 2·ffn/3` upstream arithmetic — the config carries the
//!   hidden width directly), **sliding-window causal attention**
//!   (`delta < context`), M3-03 paged KV (`BlockSize::Two`);
//! - [`MoshiDepthTransformer`] — 6 layers at d=1024 with **per-step
//!   weights** (one in/out projection + gating set per codebook step),
//!   no positional embedding, per-frame KV reset;
//! - [`MoshiModel`] — the LMGen-transcribed step: delay-ring token
//!   plumbing, 17-channel summed embedding, text sample → depformer →
//!   own codes, warmup `None` while `offset <= max_delay`;
//! - [`MoshiDuplexSession`] — mic → AEC (M4-03) → Mimi encode → step →
//!   Mimi decode → speaker, with barge-in ([`DuplexInterruptHandle`]
//!   semantics) and the far-end reference queue fed at pull time.
//!
//! The PCM ends ride the **shared Mimi neural chain** (`crate::mimi`,
//! landed by M4-05 — ADR M4-06 §D1-(b): this WP consumes it unchanged;
//! the Moshi-specific delta is `quantizer.n_q = 8`, written by the
//! converter).
//!
//! # Honest loading state (FR-EX-08)
//!
//! Real-checkpoint weight binding is gated on the T29 owner hand-off.
//! Until then `from_gguf` weight paths return
//! [`vokra_core::VokraError::NotImplemented`] and deterministic
//! synthesized fixtures (SplitMix64 + Xavier) drive shape / stability /
//! property tests — never a silent zero-fill.

pub mod backbone;
pub mod config;

pub use backbone::{
    MOSHI_FROM_GGUF_DEFAULT_SEED, MOSHI_ZERO_TOKEN, MoshiBackbone, MoshiBackboneState,
    MoshiBackboneWeights,
};
pub use config::{
    DEFAULT_MOSHI_RMS_NORM_EPS, DEFAULT_MOSHI_ROPE_MAX_PERIOD, MoshiConfig, MoshiTransformerConfig,
};

/// `vokra.model.arch` a Moshi GGUF must carry. Written by
/// `vokra-convert::models::moshi::ARCH`; the compliance registry
/// (`vokra_core::compliance`) maps `moshi` to
/// [`vokra_core::LicenseClass::AttributionRequired`] (CC-BY 4.0 — the
/// M2-13 gate passes commercially *and* the FR-MD-09 attribution surface
/// activates).
pub const EXPECTED_ARCH: &str = "moshi";
