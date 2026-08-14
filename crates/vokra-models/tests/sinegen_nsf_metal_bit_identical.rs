//! Vocoder Metal wave common vocoder primitive — SineGen deterministic
//! forward MSL kernel real-GPU parity (`vokra_sinegen_deterministic_f32`
//! implemented in `crates/vokra-backend-metal/src/context.rs`, wired
//! through `Compute::sinegen_deterministic_f32` in
//! `crates/vokra-models/src/compute.rs`).
//!
//! - **Off-feature band** (compiled when `metal` is off / non-Apple):
//!   `HotOp::SinegenDeterministic` is not Metal-covered off the feature, so
//!   `Compute::for_backend(Metal, [SinegenDeterministic])` is an explicit
//!   [`vokra_core::VokraError::BackendUnavailable`] — never a silent CPU
//!   substitute (FR-EX-08).
//! - **Metal band** (`--features metal`, Apple): real-GPU parity on THIS M1
//!   iMac (CLAUDE.md dev environment). Skips cleanly (printed reason) when no
//!   Metal device is present.
//!
//! # atol bound and negative control
//!
//! SineGen carries a per-harmonic sequential cumsum (`cs += f0[j] *
//! harmonic_gain`), then applies `sin(2π · (cs mod 1))` and multiplies by
//! the voiced/unvoiced mask. The GPU kernel walks the same per-harmonic
//! reduction (one thread per harmonic; the sequential cumsum makes any
//! finer parallelisation redundant), so only the transcendental `sin` gap
//! between MSL fast-math and Rust `f32::sin` is a bit-level source of
//! divergence. The atol bound is `≤ 5e-4` — same envelope as the sibling
//! snake / codec-family parity tests.
//!
//! A **negative control** shows the bound is discriminating rather than
//! vacuous: shifting the sample rate by 10% moves the CPU output well past
//! 5e-4 (the frequency of every harmonic changes proportionally), so a
//! "CPU vs Metal ≤ 5e-4" agreement is a real match, not a floor.
//!
//! # Deterministic-only
//!
//! The MSL kernel matches the `NsfEntropy::Deterministic` path of
//! [`vokra_ops::nsf::SineGen::forward`] — zero per-harmonic phase, zero
//! noise. The seeded variant (per-harmonic phase draw + Gaussian noise)
//! carries host-side RNG state and would need a separate seam.

#[cfg(not(all(feature = "metal", any(target_os = "macos", target_os = "ios"))))]
mod off_feature {
    use vokra_core::{BackendKind, VokraError};
    use vokra_models::compute::{Compute, HotOp};

    /// Off the Metal build the SinegenDeterministic coverage arm is a
    /// `BackendUnavailable` at the `for_backend` entry — never a silent CPU
    /// substitute (FR-EX-08).
    #[test]
    fn for_backend_metal_sinegen_deterministic_off_feature_is_backend_unavailable() {
        let Err(err) = Compute::for_backend(BackendKind::Metal, &[HotOp::SinegenDeterministic])
        else {
            panic!("off-feature Metal must fail explicitly, not silently CPU-substitute");
        };
        assert!(
            matches!(err, VokraError::BackendUnavailable(_)),
            "expected BackendUnavailable off the metal feature, got {err:?}",
        );
    }
}

#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
mod metal_band {
    use vokra_core::{BackendKind, VokraError};
    use vokra_models::compute::{Compute, HotOp};
    use vokra_ops::sinegen_deterministic_f32;

    /// Metal / CPU parity bound for the SineGen deterministic FP32 op. Same
    /// 5e-4 budget as the sibling snake_activation / codec-family parity
    /// tests. The only bit-level source of divergence is `sin`'s
    /// transcendental gap between MSL fast-math and Rust `f32::sin`, well
    /// inside the bound for finite inputs.
    const ATOL: f32 = 5e-4;

    fn max_delta(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f32, f32::max)
    }

    /// Build a Metal `Compute` covering SinegenDeterministic. Returns
    /// `None` when no Metal device is present (clean skip).
    fn metal_compute() -> Option<Compute> {
        match Compute::for_backend(BackendKind::Metal, &[HotOp::SinegenDeterministic]) {
            Ok(c) => Some(c),
            Err(VokraError::BackendUnavailable(_)) => None,
            Err(e) => panic!("unexpected Metal for_backend error: {e}"),
        }
    }

    /// Constant 100 Hz F0, fundamental-only (`H = 0`). Short enough to
    /// reason about, exercises the sequential cumsum for a single harmonic.
    #[test]
    fn constant_100hz_fundamental_metal_matches_cpu_or_clean_skip() {
        let Some(compute_metal) = metal_compute() else {
            println!("skip: no Metal device on this host (clean skip, never fabricated)");
            return;
        };
        assert_eq!(compute_metal.backend_name(), "metal");
        let compute_cpu = Compute::cpu();

        let samp_rate = 22_050u32;
        let harmonic_num = 0u32;
        let sine_amp = 0.1f32;
        let voiced_threshold = 0.0f32;
        let t = 64usize;
        let f0 = vec![100.0f32; t];
        let h1 = (harmonic_num + 1) as usize;

        let mut cpu_out = vec![0.0f32; t * h1];
        let mut metal_out = vec![0.0f32; t * h1];
        compute_cpu
            .sinegen_deterministic_f32(
                &f0,
                samp_rate,
                harmonic_num,
                sine_amp,
                voiced_threshold,
                &mut cpu_out,
            )
            .expect("CPU arm must succeed");
        compute_metal
            .sinegen_deterministic_f32(
                &f0,
                samp_rate,
                harmonic_num,
                sine_amp,
                voiced_threshold,
                &mut metal_out,
            )
            .expect("Metal arm must succeed");
        assert_eq!(cpu_out.len(), metal_out.len(), "output shapes must match");

        let d = max_delta(&cpu_out, &metal_out);
        println!("constant-100Hz-fund (T=64, H+1=1) SineGen Metal vs CPU max |Δ| = {d:e}");
        assert!(
            d <= ATOL,
            "sinegen_deterministic_f32 constant-100Hz-fund Metal vs CPU max |Δ| = {d} > {ATOL}"
        );

        // Cross-check: the CPU arm is bit-identical to the vokra_ops free
        // function (sanity gate on the CPU arm rather than a Metal claim).
        let mut reference = vec![0.0f32; t * h1];
        sinegen_deterministic_f32(
            &f0,
            samp_rate,
            harmonic_num,
            sine_amp,
            voiced_threshold,
            &mut reference,
        )
        .unwrap();
        for (i, (&c, &r)) in cpu_out.iter().zip(reference.iter()).enumerate() {
            assert_eq!(
                c.to_bits(),
                r.to_bits(),
                "index {i}: Compute::cpu()::sinegen_deterministic_f32 must be bit-identical to \
                 vokra_ops::sinegen_deterministic_f32 (got {c} vs {r})",
            );
        }
    }

    /// HiFTNet-typical shape: T = 256, harmonic_num = 4 → H+1 = 5 channels.
    /// F0 is a slow modulator around 150 Hz (typical speech pitch) to
    /// exercise the cumsum-modulo-1 unwrap path over multiple sinusoid
    /// periods. Also asserts the negative-control bound so ATOL is not
    /// vacuous.
    #[test]
    fn hiftnet_shape_metal_matches_cpu_with_negative_control_or_clean_skip() {
        let Some(compute_metal) = metal_compute() else {
            println!("skip: no Metal device on this host (clean skip, never fabricated)");
            return;
        };
        let compute_cpu = Compute::cpu();

        let samp_rate = 22_050u32;
        let harmonic_num = 4u32;
        let sine_amp = 0.1f32;
        let voiced_threshold = 0.0f32;
        let t = 256usize;
        let h1 = (harmonic_num + 1) as usize;
        // Slow pitch modulator around 150 Hz.
        let f0: Vec<f32> = (0..t)
            .map(|j| 150.0 + (j as f32 * 0.05).sin() * 30.0)
            .collect();

        let mut cpu_out = vec![0.0f32; t * h1];
        let mut metal_out = vec![0.0f32; t * h1];
        compute_cpu
            .sinegen_deterministic_f32(
                &f0,
                samp_rate,
                harmonic_num,
                sine_amp,
                voiced_threshold,
                &mut cpu_out,
            )
            .expect("CPU arm must succeed");
        compute_metal
            .sinegen_deterministic_f32(
                &f0,
                samp_rate,
                harmonic_num,
                sine_amp,
                voiced_threshold,
                &mut metal_out,
            )
            .expect("Metal arm must succeed");
        assert_eq!(cpu_out.len(), metal_out.len());

        let d = max_delta(&cpu_out, &metal_out);
        println!("hiftnet (T=256, H+1=5) SineGen Metal vs CPU max |Δ| = {d:e}");
        assert!(
            d <= ATOL,
            "hiftnet sinegen_deterministic_f32 Metal vs CPU max |Δ| = {d} > {ATOL}"
        );

        // Negative control: shift the sample rate by 10%. Every harmonic's
        // frequency scales inversely with samp_rate (harmonic_gain =
        // (i+1) / samp_rate), so a 10% shift shifts every sinusoid past
        // ATOL by t = 256. This proves the ≤ ATOL agreement above is a
        // real match, not a floor any two outputs would satisfy.
        let shifted_samp_rate = 22_050 * 11 / 10;
        let mut cpu_out_shifted = vec![0.0f32; t * h1];
        compute_cpu
            .sinegen_deterministic_f32(
                &f0,
                shifted_samp_rate,
                harmonic_num,
                sine_amp,
                voiced_threshold,
                &mut cpu_out_shifted,
            )
            .expect("CPU shifted arm must succeed");
        let control = max_delta(&cpu_out, &cpu_out_shifted);
        println!("negative control (Δsamp_rate = +10%) max |Δ| = {control:e}");
        assert!(
            control > ATOL,
            "negative control: a 10% samp_rate shift moved CPU output only {control} ≤ {ATOL} \
             — the atol bound would be vacuous; test cannot honestly claim parity"
        );
    }

    /// Voiced-threshold path: half the F0 sequence is below threshold
    /// (uv=0 → output=0), half is above (uv=1 → sinusoid). Exercises the
    /// branch inside the kernel and ensures the mask propagates identically
    /// to the CPU.
    #[test]
    fn voiced_threshold_masks_low_f0_or_clean_skip() {
        let Some(compute_metal) = metal_compute() else {
            println!("skip: no Metal device on this host (clean skip, never fabricated)");
            return;
        };
        let compute_cpu = Compute::cpu();

        let samp_rate = 22_050u32;
        let harmonic_num = 1u32; // H+1 = 2
        let sine_amp = 0.1f32;
        let voiced_threshold = 50.0f32;
        let t = 32usize;
        let h1 = (harmonic_num + 1) as usize;
        // First half below threshold (10 Hz), second half above (100 Hz).
        let mut f0 = vec![10.0f32; t / 2];
        f0.extend(vec![100.0f32; t / 2]);

        let mut cpu_out = vec![0.0f32; t * h1];
        let mut metal_out = vec![0.0f32; t * h1];
        compute_cpu
            .sinegen_deterministic_f32(
                &f0,
                samp_rate,
                harmonic_num,
                sine_amp,
                voiced_threshold,
                &mut cpu_out,
            )
            .expect("CPU arm must succeed");
        compute_metal
            .sinegen_deterministic_f32(
                &f0,
                samp_rate,
                harmonic_num,
                sine_amp,
                voiced_threshold,
                &mut metal_out,
            )
            .expect("Metal arm must succeed");

        let d = max_delta(&cpu_out, &metal_out);
        println!(
            "voiced-threshold (T=32, H+1=2, threshold=50) SineGen Metal vs CPU max |Δ| = {d:e}"
        );
        assert!(
            d <= ATOL,
            "voiced-threshold sinegen_deterministic_f32 Metal vs CPU max |Δ| = {d} > {ATOL}"
        );
        // Below-threshold slots must be all zero on both arms.
        for (i, (&c, &m)) in cpu_out[..h1 * (t / 2)]
            .iter()
            .zip(metal_out.iter())
            .enumerate()
        {
            assert_eq!(c, 0.0, "cpu below-threshold slot {i} must be 0, got {c}");
            assert_eq!(m, 0.0, "metal below-threshold slot {i} must be 0, got {m}");
        }
    }

    /// FR-EX-08 host-side validation: empty f0 is `InvalidArgument`.
    #[test]
    fn empty_f0_is_invalid_argument_or_clean_skip() {
        let Some(compute_metal) = metal_compute() else {
            println!("skip: no Metal device on this host (clean skip, never fabricated)");
            return;
        };
        let mut out: Vec<f32> = Vec::new();
        let err = compute_metal
            .sinegen_deterministic_f32(&[], 22_050, 0, 0.1, 0.0, &mut out)
            .expect_err("empty f0 must be an explicit InvalidArgument (FR-EX-08)");
        assert!(
            matches!(err, VokraError::InvalidArgument(_)),
            "expected InvalidArgument for empty f0, got {err:?}",
        );
    }

    /// FR-EX-08 host-side validation: samp_rate = 0 is `InvalidArgument`.
    #[test]
    fn zero_samp_rate_is_invalid_argument_or_clean_skip() {
        let Some(compute_metal) = metal_compute() else {
            println!("skip: no Metal device on this host (clean skip, never fabricated)");
            return;
        };
        let mut out = vec![0.0f32; 4];
        let err = compute_metal
            .sinegen_deterministic_f32(&vec![100.0f32; 4], 0, 0, 0.1, 0.0, &mut out)
            .expect_err("samp_rate=0 must be an explicit InvalidArgument (FR-EX-08)");
        assert!(
            matches!(err, VokraError::InvalidArgument(_)),
            "expected InvalidArgument for samp_rate=0, got {err:?}",
        );
    }

    /// FR-EX-08 host-side validation: wrong out length is `InvalidArgument`.
    #[test]
    fn wrong_out_length_is_invalid_argument_or_clean_skip() {
        let Some(compute_metal) = metal_compute() else {
            println!("skip: no Metal device on this host (clean skip, never fabricated)");
            return;
        };
        let f0 = vec![100.0f32; 4];
        let harmonic_num = 2u32; // H+1 = 3 → expected out.len() = 12
        let mut out = vec![0.0f32; 4 * 3 - 1];
        let err = compute_metal
            .sinegen_deterministic_f32(&f0, 22_050, harmonic_num, 0.1, 0.0, &mut out)
            .expect_err("wrong out length must be an explicit InvalidArgument (FR-EX-08)");
        assert!(
            matches!(err, VokraError::InvalidArgument(_)),
            "expected InvalidArgument for wrong out length, got {err:?}",
        );
    }
}
