//! #10 — CosyVoice2 LLM backbone Metal real-GPU parity (the M4-05 CSM
//! `csm_gpu_session.rs` posture applied to the M3-09 Qwen2-style LLM
//! backbone).
//!
//! - **Off-feature band**: routing the backbone to Metal is an explicit
//!   [`vokra_core::VokraError::BackendUnavailable`] at forward time — never a
//!   silent CPU fall back (FR-EX-08).
//! - **Metal band** (`--features metal`, Apple): real-GPU parity on THIS M1
//!   iMac. `LlmBackbone::synthesized` needs no real weights; the GQA / RoPE /
//!   SwiGLU / RMSNorm blocks dispatch GEMM / GEMV / Softmax (`LLM_HOT_OPS`,
//!   all Metal-covered) through `Compute::Metal`. The `[t, vocab]` logits stay
//!   within `atol ≤ 5e-4` of the CPU path. Clean skip when no device.
//!
//! The atol bound is honest: a **negative control** shows a changed input
//! token moves the logits far past 5e-4. Real-weight parity (T02 checkpoint)
//! stays owner.

use vokra_models::cosyvoice2::LlmBackboneConfig;

/// A tiny GQA-well-formed config (the `llm.rs` unit `test_config`): 8-wide,
/// 2 layers, 2 query heads over 1 KV head, n_ctx 8.
fn tiny_config() -> LlmBackboneConfig {
    LlmBackboneConfig {
        vocab_size: 16,
        hidden_dim: 8,
        n_layer: 2,
        n_head_q: 2,
        n_head_kv: 1,
        ffn_dim: 16,
        rope_base: 10_000.0,
        rms_norm_eps: 1e-5,
        n_ctx: 8,
    }
}

#[cfg(not(all(feature = "metal", any(target_os = "macos", target_os = "ios"))))]
mod off_feature {
    use super::tiny_config;
    use vokra_core::{BackendKind, VokraError};
    use vokra_models::cosyvoice2::LlmBackbone;

    #[test]
    fn routing_the_backbone_to_metal_is_an_explicit_error() {
        let backbone = LlmBackbone::synthesized(tiny_config(), 42)
            .unwrap()
            .with_backend(BackendKind::Metal);
        let err = backbone.forward(&[1, 2, 3], 0).unwrap_err();
        assert!(
            matches!(err, VokraError::BackendUnavailable(_)),
            "off-feature Metal must be BackendUnavailable (no silent CPU fall back), got {err:?}"
        );
    }
}

#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
mod metal_band {
    use super::tiny_config;
    use vokra_core::{BackendKind, Result, VokraError};
    use vokra_models::cosyvoice2::LlmBackbone;

    const ATOL: f32 = 5e-4;

    fn max_delta(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    fn parity() -> Result<bool> {
        let cfg = tiny_config();
        let cpu = LlmBackbone::synthesized(cfg.clone(), 42)?;
        let gpu = LlmBackbone::synthesized(cfg, 42)?.with_backend(BackendKind::Metal);
        let tokens = [1u32, 2, 3, 4];

        let cpu_logits = cpu.forward(&tokens, 0)?;
        let gpu_logits = match gpu.forward(&tokens, 0) {
            Ok(l) => l,
            Err(VokraError::BackendUnavailable(_)) => return Ok(false),
            Err(e) => return Err(e),
        };
        assert_eq!(cpu_logits.len(), gpu_logits.len());
        let d = max_delta(&cpu_logits, &gpu_logits);
        assert!(
            d <= ATOL,
            "cosyvoice2 LLM Metal vs CPU max |Δ| = {d} > {ATOL}"
        );

        // Negative control: changing one input token moves the logits far past
        // ATOL, so the CPU-vs-Metal agreement is a real match.
        let cpu_logits2 = cpu.forward(&[1, 2, 3, 7], 0)?;
        let control = max_delta(&cpu_logits, &cpu_logits2);
        assert!(
            control > ATOL,
            "negative control: a changed input token moved the logits only {control} ≤ {ATOL} — \
             the atol bound would be vacuous"
        );
        Ok(true)
    }

    #[test]
    fn cosyvoice2_llm_metal_parity_or_clean_skip() {
        let ran = parity().expect("cosyvoice2 parity driver");
        if !ran {
            println!("skip: no Metal device on this host (clean skip, never fabricated)");
        }
    }
}
