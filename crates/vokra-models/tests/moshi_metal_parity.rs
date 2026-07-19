//! #10 — Moshi temporal backbone Metal real-GPU parity (the M4-05 CSM
//! `csm_gpu_session.rs` posture applied to the Moshi LM backbone).
//!
//! - **Off-feature band**: routing the backbone to Metal is an explicit
//!   [`vokra_core::VokraError::BackendUnavailable`] at forward time — never a
//!   silent CPU fall back (FR-EX-08).
//! - **Metal band** (`--features metal`, Apple): real-GPU parity on THIS M1
//!   iMac. `MoshiBackbone::synthesized` needs no real weights; the pre-norm
//!   MHA stack (GEMM / GEMV / Softmax = `MOSHI_HOT_OPS`, all Metal-covered)
//!   routes through `Compute::Metal`. The out_norm-applied hidden state stays
//!   within `atol ≤ 5e-4` of the CPU path. Clean skip when no device.
//!
//! The atol bound is honest: a **negative control** shows a one-token input
//! change moves the hidden state far past 5e-4, so the CPU-vs-Metal bound is
//! discriminating, not vacuous. Real-weight numeric parity stays owner
//! (`tests/parity_moshi.rs` flip-the-switch).

#[cfg(not(all(feature = "metal", any(target_os = "macos", target_os = "ios"))))]
mod off_feature {
    use vokra_core::{BackendKind, VokraError};
    use vokra_models::moshi::{MoshiBackbone, MoshiBackboneState, MoshiConfig};

    #[test]
    fn routing_the_backbone_to_metal_is_an_explicit_error() {
        let cfg = MoshiConfig::tiny_for_tests();
        let b = MoshiBackbone::synthesized(cfg.clone(), 11)
            .unwrap()
            .with_backend(BackendKind::Metal);
        let mut v = Vec::with_capacity(cfg.n_channels());
        v.push(1u32 % cfg.text_card as u32);
        for k in 0..cfg.n_q_in {
            v.push((1 + 3 * k) as u32 % cfg.audio_card as u32);
        }
        let mut state = MoshiBackboneState::new(&cfg).unwrap();
        let err = b.forward(&[v], &mut state).unwrap_err();
        assert!(
            matches!(err, VokraError::BackendUnavailable(_)),
            "off-feature Metal must be BackendUnavailable (no silent CPU fall back), got {err:?}"
        );
    }
}

#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
mod metal_band {
    use vokra_core::{BackendKind, Result, VokraError};
    use vokra_models::moshi::{MoshiBackbone, MoshiBackboneState, MoshiConfig};

    const ATOL: f32 = 5e-4;

    /// A valid step-token row (real ids on every channel).
    fn step_tokens(cfg: &MoshiConfig, seed: u32) -> Vec<u32> {
        let mut v = Vec::with_capacity(cfg.n_channels());
        v.push((seed as usize % cfg.text_card) as u32);
        for k in 0..cfg.n_q_in {
            v.push(((seed as usize + 3 * k + 1) % cfg.audio_card) as u32);
        }
        v
    }

    fn max_delta(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    fn forward_cpu(b: &MoshiBackbone, cfg: &MoshiConfig, steps: &[Vec<u32>]) -> Result<Vec<f32>> {
        let mut state = MoshiBackboneState::new(cfg)?;
        b.forward(steps, &mut state)
    }

    fn parity() -> Result<bool> {
        let cfg = MoshiConfig::tiny_for_tests();
        let cpu = MoshiBackbone::synthesized(cfg.clone(), 11)?;
        let gpu = MoshiBackbone::synthesized(cfg.clone(), 11)?.with_backend(BackendKind::Metal);
        let steps: Vec<Vec<u32>> = (0..3).map(|i| step_tokens(&cfg, i)).collect();

        let cpu_h = forward_cpu(&cpu, &cfg, &steps)?;
        let mut gpu_state = MoshiBackboneState::new(&cfg)?;
        let gpu_h = match gpu.forward(&steps, &mut gpu_state) {
            Ok(h) => h,
            Err(VokraError::BackendUnavailable(_)) => return Ok(false),
            Err(e) => return Err(e),
        };
        assert_eq!(cpu_h.len(), gpu_h.len());
        let d = max_delta(&cpu_h, &gpu_h);
        assert!(
            d <= ATOL,
            "moshi backbone Metal vs CPU max |Δ| = {d} > {ATOL}"
        );

        // Negative control: changing one token moves the hidden state far past
        // ATOL, so the CPU-vs-Metal agreement is a real match, not a vacuous
        // floor.
        let mut other = steps.clone();
        other[0] = step_tokens(&cfg, 5);
        let cpu_h2 = forward_cpu(&cpu, &cfg, &other)?;
        let control = max_delta(&cpu_h, &cpu_h2);
        assert!(
            control > ATOL,
            "negative control: a changed input token moved the hidden state only {control} ≤ \
             {ATOL} — the atol bound would be vacuous"
        );
        Ok(true)
    }

    #[test]
    fn moshi_backbone_metal_parity_or_clean_skip() {
        let ran = parity().expect("moshi parity driver");
        if !ran {
            println!("skip: no Metal device on this host (clean skip, never fabricated)");
        }
    }
}
