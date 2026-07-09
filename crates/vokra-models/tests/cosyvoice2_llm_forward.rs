//! CosyVoice2 LLM backbone — integration tests (M3-09-T09 synthesized
//! fixture).
//!
//! These tests exercise the **Mistral-style forward path** end-to-end
//! against a seed-deterministic weight fixture built by
//! [`vokra_models::cosyvoice2::LlmWeights::synthesized`]. The goal is to
//! verify:
//!
//! - **Shape + finiteness** across the layer stack (no NaN / Inf from the
//!   pre-norm block sequence).
//! - **Consistency** between the bulk `forward` and the incremental
//!   `step` (KV cache carries across the split).
//! - **Determinism** at fixed seed (property test).
//! - **KV cache growth semantics** (positions +1 per step).
//! - **Greedy decode early-stopping** (`eos` and `max_new_tokens`).
//! - **Reset** clears state without touching the weight store.
//!
//! Real HF-checkpoint parity is deferred (T02 tensor manifest is an
//! owner-side deliverable), so no real safetensors download is exercised
//! here — every fixture is built by `SplitMix64` deterministically. When
//! the owner drops in `iic/CosyVoice2-0.5B`, the flip-the-switch harness
//! at `vokra_models::cosyvoice2::llm::parity::assert_vs_hf_reference`
//! becomes the additional gate.
//!
//! # No fabricated pass (FR-EX-08)
//!
//! Every assertion checks a **numeric or structural** property against the
//! synthesized fixture; nothing is a bare `assert!(true)`. If a future
//! refactor breaks the forward, at least one of these tests fires.

#![allow(clippy::items_after_statements)]

use vokra_models::cosyvoice2::llm::{LlmBackbone, LlmBackboneConfig, LlmBackboneStep, parity};

/// Canonical CosyVoice2 LLM config for the integration harness. Small
/// enough to run fast; large enough that the Mistral block stack, GQA
/// head split, and RoPE stride cover their code paths. Values are
/// arbitrary but fixed so the tests are byte-reproducible.
fn make_config() -> LlmBackboneConfig {
    LlmBackboneConfig {
        vocab_size: 32,
        hidden_dim: 16,
        n_layer: 3, // Layer stack ≥ 2 so the residual chain compounds.
        n_head_q: 4,
        n_head_kv: 2, // n_kv_groups = 2 → GQA broadcast tested.
        ffn_dim: 32,
        rope_base: 10_000.0,
        rms_norm_eps: 1e-5,
        n_ctx: 32,
    }
}

/// Fixed seed for the integration harness. Documented so callers can
/// reproduce the fixtures.
const SEED: u64 = 42;

// -----------------------------------------------------------------------------
// Forward: shape + finiteness
// -----------------------------------------------------------------------------

#[test]
fn synthesized_forward_produces_finite_logits_across_layer_stack() {
    let cfg = make_config();
    let backbone = LlmBackbone::synthesized(cfg.clone(), SEED).expect("build");
    // 5-token prefix exercises causal mask + all 3 layers.
    let logits = backbone.forward(&[0, 1, 2, 3, 4], 0).expect("forward runs");
    assert_eq!(logits.len(), 5 * cfg.vocab_size);
    for (i, &l) in logits.iter().enumerate() {
        assert!(
            l.is_finite(),
            "logit at position {} = {} is not finite (NaN / Inf leaked from the layer stack)",
            i,
            l
        );
    }
}

#[test]
fn synthesized_forward_finite_across_multiple_prefix_lengths() {
    let cfg = make_config();
    let backbone = LlmBackbone::synthesized(cfg.clone(), SEED).expect("build");
    for prefix_len in [1usize, 2, 3, 5, 8, 16] {
        let tokens: Vec<u32> = (0..prefix_len as u32).collect();
        let logits = backbone
            .forward(&tokens, 0)
            .unwrap_or_else(|e| panic!("prefix_len {prefix_len}: {e}"));
        assert_eq!(logits.len(), prefix_len * cfg.vocab_size);
        for &l in &logits {
            assert!(l.is_finite());
        }
    }
}

#[test]
fn synthesized_forward_logits_are_bounded() {
    // Property: Xavier init keeps activations bounded; the logits should
    // stay within [-1e3, +1e3] for this small configuration.
    let cfg = make_config();
    let backbone = LlmBackbone::synthesized(cfg, SEED).expect("build");
    let logits = backbone.forward(&[0, 1, 2, 3], 0).expect("forward");
    for &l in &logits {
        assert!(
            l.abs() < 1e3,
            "logit magnitude {l} exceeds sanity bound (potential overflow)"
        );
    }
}

// -----------------------------------------------------------------------------
// Determinism
// -----------------------------------------------------------------------------

#[test]
fn synthesized_forward_is_deterministic_same_seed() {
    let cfg = make_config();
    let a = LlmBackbone::synthesized(cfg.clone(), SEED).unwrap();
    let b = LlmBackbone::synthesized(cfg, SEED).unwrap();
    let out_a = a.forward(&[1, 2, 3], 0).unwrap();
    let out_b = b.forward(&[1, 2, 3], 0).unwrap();
    assert_eq!(
        out_a, out_b,
        "same seed + same input → identical logits (reproducibility)"
    );
}

#[test]
fn synthesized_forward_differs_across_seeds() {
    let cfg = make_config();
    let a = LlmBackbone::synthesized(cfg.clone(), SEED).unwrap();
    let b = LlmBackbone::synthesized(cfg, SEED + 1).unwrap();
    let out_a = a.forward(&[1, 2, 3], 0).unwrap();
    let out_b = b.forward(&[1, 2, 3], 0).unwrap();
    assert_ne!(
        out_a, out_b,
        "different seeds should produce distinct logits (probabilistic — collision odds ~0 for f32)"
    );
}

// -----------------------------------------------------------------------------
// Step: KV cache growth
// -----------------------------------------------------------------------------

#[test]
fn step_grows_kv_cache_by_one_position_per_call() {
    let cfg = make_config();
    let backbone = LlmBackbone::synthesized(cfg.clone(), SEED).unwrap();
    let mut state = LlmBackboneStep::new();
    assert!(
        state.kv_cache.is_none(),
        "cache lazy-allocated on first step"
    );
    assert_eq!(state.seq_len, 0);

    for expected_positions in 1..=5 {
        let _logits = backbone
            .step(&mut state, expected_positions - 1)
            .unwrap_or_else(|e| panic!("step {expected_positions}: {e}"));
        assert_eq!(state.seq_len, expected_positions as usize);
        let cache_positions = state.kv_cache.as_ref().unwrap().positions();
        assert_eq!(
            cache_positions, expected_positions as usize,
            "kv cache positions must grow +1 per step"
        );
    }
}

#[test]
fn step_shape_is_vocab_size() {
    let cfg = make_config();
    let backbone = LlmBackbone::synthesized(cfg.clone(), SEED).unwrap();
    let mut state = LlmBackboneStep::new();
    let logits = backbone.step(&mut state, 0).unwrap();
    assert_eq!(logits.len(), cfg.vocab_size);
    for &l in &logits {
        assert!(l.is_finite());
    }
}

// -----------------------------------------------------------------------------
// Forward ↔ step consistency (the M3-09-T09 property test)
// -----------------------------------------------------------------------------

#[test]
fn forward_bulk_matches_step_by_step_last_row() {
    // The core parity test: bulk `forward` and per-token `step` must
    // produce numerically-identical logits at the last position (up to
    // tight atol for GEMM associativity drift).
    let cfg = make_config();
    let backbone = LlmBackbone::synthesized(cfg, SEED).unwrap();
    parity::forward_matches_step_by_step(&backbone, &[0, 1, 2, 3, 4, 5], 1e-3)
        .expect("forward vs step-by-step consistency");
}

#[test]
fn forward_matches_step_by_step_across_seeds() {
    let cfg = make_config();
    for seed in [1u64, 42, 100, 12345, 999999] {
        let backbone = LlmBackbone::synthesized(cfg.clone(), seed).unwrap();
        parity::forward_matches_step_by_step(&backbone, &[0, 1, 2, 3], 1e-3)
            .unwrap_or_else(|e| panic!("seed {seed}: {e}"));
    }
}

// -----------------------------------------------------------------------------
// Greedy decode: early stop + max_new
// -----------------------------------------------------------------------------

#[test]
fn greedy_decode_early_stops_on_eos() {
    let cfg = make_config();
    let backbone = LlmBackbone::synthesized(cfg, SEED).unwrap();
    // Learn the first token the backbone would sample from prefix [0, 1].
    let dry = backbone.greedy_decode(&[0, 1], u32::MAX, 1).unwrap();
    assert_eq!(dry.len(), 1);
    let first_tok = dry[0];
    // Now set eos = first_tok; decode should terminate after committing it.
    let out = backbone.greedy_decode(&[0, 1], first_tok, 10).unwrap();
    assert_eq!(out.len(), 1, "eos must terminate the loop immediately");
    assert_eq!(out[0], first_tok, "eos IS included in the output");
}

#[test]
fn greedy_decode_respects_max_new_tokens() {
    let cfg = make_config();
    let backbone = LlmBackbone::synthesized(cfg, SEED).unwrap();
    // Pick eos = u32::MAX (never sampled — outside vocab range).
    let out = backbone.greedy_decode(&[0, 1], u32::MAX, 4).unwrap();
    assert_eq!(out.len(), 4);
}

#[test]
fn greedy_decode_is_deterministic() {
    let cfg = make_config();
    let backbone = LlmBackbone::synthesized(cfg, SEED).unwrap();
    let a = backbone.greedy_decode(&[0, 1, 2], u32::MAX, 5).unwrap();
    let b = backbone.greedy_decode(&[0, 1, 2], u32::MAX, 5).unwrap();
    assert_eq!(
        a, b,
        "greedy sampling with fixed seed + prefix is deterministic"
    );
}

// -----------------------------------------------------------------------------
// Reset
// -----------------------------------------------------------------------------

#[test]
fn step_reset_clears_state_and_reuses_capacity() {
    let cfg = make_config();
    let backbone = LlmBackbone::synthesized(cfg.clone(), SEED).unwrap();
    let mut state = LlmBackboneStep::new();
    for i in 0..4 {
        let _ = backbone.step(&mut state, i).unwrap();
    }
    assert_eq!(state.seq_len, 4);
    // Reset should clear seq_len + rewind the KV cache.
    state.reset();
    assert_eq!(state.seq_len, 0);
    let positions_after_reset = state.kv_cache.as_ref().unwrap().positions();
    assert_eq!(positions_after_reset, 0);
    // A fresh step against the reset state runs cleanly.
    let logits = backbone.step(&mut state, 0).unwrap();
    assert_eq!(logits.len(), cfg.vocab_size);
    for &l in &logits {
        assert!(l.is_finite());
    }
    assert_eq!(state.seq_len, 1);
}

// -----------------------------------------------------------------------------
// Error surfaces (no silent fallback, FR-EX-08)
// -----------------------------------------------------------------------------

#[test]
fn forward_rejects_token_out_of_range() {
    let cfg = make_config();
    let backbone = LlmBackbone::synthesized(cfg.clone(), SEED).unwrap();
    let err = backbone
        .forward(&[cfg.vocab_size as u32], 0)
        .expect_err("out-of-range token must fail");
    assert!(format!("{err:?}").contains("InvalidArgument"));
}

#[test]
fn forward_rejects_position_past_n_ctx() {
    let cfg = make_config();
    let backbone = LlmBackbone::synthesized(cfg.clone(), SEED).unwrap();
    // n_ctx = 32; a bulk forward of 33 tokens at offset 0 fails.
    let tokens: Vec<u32> = (0..(cfg.n_ctx as u32 + 1)).collect();
    let err = backbone.forward(&tokens, 0).expect_err("past n_ctx");
    assert!(format!("{err:?}").contains("InvalidArgument"));
}

#[test]
fn step_refuses_past_n_ctx() {
    let cfg = make_config();
    let backbone = LlmBackbone::synthesized(cfg.clone(), SEED).unwrap();
    let mut state = LlmBackboneStep::new();
    // Fill exactly to n_ctx.
    for i in 0..cfg.n_ctx {
        let _ = backbone
            .step(&mut state, (i as u32) % (cfg.vocab_size as u32))
            .unwrap();
    }
    let err = backbone.step(&mut state, 0).expect_err("past n_ctx");
    assert!(format!("{err:?}").contains("InvalidArgument"));
}

// -----------------------------------------------------------------------------
// Real HF-checkpoint parity (owner-side flip-the-switch)
// -----------------------------------------------------------------------------

#[test]
fn parity_hf_reference_is_not_wired_today() {
    // Honest signal: the real-checkpoint gate returns NotImplemented until
    // the T02 tensor manifest lands. Owners flipping this test on to
    // green must first drop the real GGUF into the workspace and swap
    // the harness body.
    let cfg = make_config();
    let err = parity::assert_vs_hf_reference(&cfg, &[])
        .expect_err("HF real-checkpoint parity is owner-side (T02)");
    assert!(format!("{err:?}").contains("NotImplemented"));
}
