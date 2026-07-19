//! #9 — Mimi neural chain (encoder + neural decoder) Metal real-GPU parity
//! (the M4-05 CSM `csm_gpu_session.rs` posture applied to the shared Mimi
//! codec ends).
//!
//! - **Off-feature band** (compiled when `metal` is off / non-Apple): routing
//!   the neural decoder to Metal is an explicit
//!   [`vokra_core::VokraError::BackendUnavailable`] at decode time — never a
//!   silent CPU fall back (FR-EX-08).
//! - **Metal band** (`--features metal`, Apple): real-GPU parity on THIS M1
//!   iMac (CLAUDE.md dev environment). The synthesized Mimi chain needs no
//!   real weights (`MimiEncoder::synthesized` / `MimiNeuralDecoder::
//!   synthesized`); the neural math (SEANet convolutions as im2col-GEMM +
//!   the bottleneck transformer's LayerNorm/GEMM/Softmax/GELU) routes through
//!   `Compute::Metal` — all of `MIMI_HOT_OPS`
//!   (Gemm/Gemv/Softmax/LayerNorm/Gelu) are Metal-covered. Decoder PCM stays
//!   within `atol ≤ 5e-4` of the CPU path; encoder RVQ codes are identical.
//!   Skips cleanly (printed reason) when no Metal device is present.
//!
//! The atol bound is honest, not tuned to force green: a **negative control**
//! shows the same pipeline moves far more than 5e-4 under a small input
//! perturbation, so the CPU-vs-Metal bound is discriminating rather than
//! vacuous. Real-weight numeric parity stays an owner task (gated tokenizer).

#[cfg(not(all(feature = "metal", any(target_os = "macos", target_os = "ios"))))]
mod off_feature {
    use vokra_core::{BackendKind, VokraError};
    use vokra_models::mimi::{MimiNeuralConfig, MimiNeuralDecoder};

    #[test]
    fn routing_the_neural_decoder_to_metal_is_an_explicit_error() {
        let cfg = MimiNeuralConfig::tiny_for_tests();
        let dec = MimiNeuralDecoder::synthesized(&cfg, 9, true)
            .unwrap()
            .with_backend(BackendKind::Metal);
        let feats: Vec<f32> = (0..3 * dec.expected_feature_dim())
            .map(|i| (i as f32 * 0.31).sin() * 0.4)
            .collect();
        let err = dec.decode_all(&feats).unwrap_err();
        assert!(
            matches!(err, VokraError::BackendUnavailable(_)),
            "off-feature Metal must be BackendUnavailable (no silent CPU fall back), got {err:?}"
        );
    }
}

#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
mod metal_band {
    use vokra_core::{BackendKind, Result, VokraError};
    use vokra_models::mimi::{MimiEncoder, MimiNeuralConfig, MimiNeuralDecoder};

    const ATOL: f32 = 5e-4;

    fn features(n_frames: usize, width: usize) -> Vec<f32> {
        (0..n_frames * width)
            .map(|i| (i as f32 * 0.31).sin() * 0.4)
            .collect()
    }

    fn max_delta(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    /// Decoder features → PCM parity. Returns `Ok(false)` (clean skip) when no
    /// Metal device is present at construction (FR-EX-08 explicit error,
    /// distinguished from a wrong result which panics).
    fn decoder_parity() -> Result<bool> {
        let cfg = MimiNeuralConfig::tiny_for_tests();
        let cpu = MimiNeuralDecoder::synthesized(&cfg, 9, true)?;
        let gpu = MimiNeuralDecoder::synthesized(&cfg, 9, true)?.with_backend(BackendKind::Metal);
        let feats = features(3, cpu.expected_feature_dim());
        let cpu_pcm = cpu.decode_all(&feats)?;
        let gpu_pcm = match gpu.decode_all(&feats) {
            Ok(p) => p,
            Err(VokraError::BackendUnavailable(_)) => return Ok(false),
            Err(e) => return Err(e),
        };
        assert_eq!(cpu_pcm.len(), gpu_pcm.len());
        let d = max_delta(&cpu_pcm, &gpu_pcm);
        assert!(
            d <= ATOL,
            "mimi decoder Metal vs CPU max |Δ| = {d} > {ATOL}"
        );

        // Negative control: the bound is discriminating, not vacuous — a small
        // input perturbation moves the CPU output well past ATOL, so the
        // CPU-vs-Metal ≤ ATOL agreement above is a real match, not a floor
        // that any two outputs would satisfy.
        let mut perturbed = feats.clone();
        perturbed[0] += 0.05;
        let cpu_pcm2 = cpu.decode_all(&perturbed)?;
        let control = max_delta(&cpu_pcm, &cpu_pcm2);
        assert!(
            control > ATOL,
            "negative control: a 0.05 feature perturbation moved PCM only {control} ≤ {ATOL} — \
             the atol bound would be vacuous; test cannot honestly claim parity"
        );
        Ok(true)
    }

    /// Encoder PCM → RVQ codes identity (argmin over an FP-parity latent — the
    /// CSM greedy-code identity precedent).
    fn encoder_code_identity() -> Result<bool> {
        let cfg = MimiNeuralConfig::tiny_for_tests();
        let cpu = MimiEncoder::synthesized(&cfg, 5)?;
        let gpu = MimiEncoder::synthesized(&cfg, 5)?.with_backend(BackendKind::Metal);
        let pcm = features(4, cpu.frame_hop()?);
        let cpu_codes = cpu.encode_all(&pcm)?;
        let gpu_codes = match gpu.encode_all(&pcm) {
            Ok(c) => c,
            Err(VokraError::BackendUnavailable(_)) => return Ok(false),
            Err(e) => return Err(e),
        };
        assert_eq!(
            cpu_codes, gpu_codes,
            "mimi encoder Metal codes must match CPU (argmin over the FP-parity latent)"
        );
        Ok(true)
    }

    #[test]
    fn mimi_decoder_metal_parity_or_clean_skip() {
        let ran = decoder_parity().expect("decoder parity driver");
        if !ran {
            println!("skip: no Metal device on this host (clean skip, never fabricated)");
        }
    }

    #[test]
    fn mimi_encoder_metal_code_identity_or_clean_skip() {
        let ran = encoder_code_identity().expect("encoder parity driver");
        if !ran {
            println!("skip: no Metal device on this host (clean skip, never fabricated)");
        }
    }
}
