//! Vocoder Metal wave common vocoder primitive — SnakeBeta activation MSL
//! kernel real-GPU parity (`vokra_snake_beta_f32` implemented in
//! `crates/vokra-backend-metal/src/context.rs`, wired through
//! `Compute::snake_beta_f32` in `crates/vokra-models/src/compute.rs`).
//!
//! - **Off-feature band** (compiled when `metal` is off / non-Apple):
//!   `HotOp::SnakeBeta` is not Metal-covered off the feature, so
//!   `Compute::for_backend(Metal, [SnakeBeta])` is an explicit
//!   [`vokra_core::VokraError::BackendUnavailable`] — never a silent CPU
//!   substitute (FR-EX-08).
//! - **Metal band** (`--features metal`, Apple): real-GPU parity on THIS M1
//!   iMac (CLAUDE.md dev environment). Skips cleanly (printed reason) when no
//!   Metal device is present.
//!
//! # atol bound and negative control
//!
//! SnakeBeta is trivially element-wise (no reduction, no gather) — the only
//! bit-level source of divergence is the transcendental gap between MSL's
//! intrinsic `sin` (compiled with fast-math defaults) and Rust's `f32::sin`.
//! In practice max |Δ| is at or below the ULP scale of `sin` for finite
//! inputs, and we assert the honest atol ≤ 5e-4 bound — the same envelope
//! used by the sibling snake_activation / codec-family parity tests.
//!
//! A **negative control** shows the bound is discriminating rather than
//! vacuous: perturbing one β entry by 0.5 moves the CPU output well past
//! 5e-4 (the perturbation cascades through the per-channel row), so a "CPU
//! vs Metal ≤ 5e-4" agreement is a real match, not a floor.
//!
//! # Not `alpha_logscale`, not plain Snake
//!
//! The MSL kernel matches the two-vector SnakeBeta closed form (`y = x +
//! (1/(β+ε))·sin(α·x)²`) — the same as
//! [`vokra_ops::bigvgan_generator::SnakeBeta::forward_in_place`] under
//! `alpha_logscale = false`. The `alpha_logscale = true` variant is
//! expected to have `exp(alpha_raw)` / `exp(beta_raw)` applied on the host
//! before dispatch (the CPU free function has the same contract). Plain
//! Snake (single-vector variant) is a distinct op with its own kernel
//! (`vokra_snake_activation_f32`).

#[cfg(not(all(feature = "metal", any(target_os = "macos", target_os = "ios"))))]
mod off_feature {
    use vokra_core::{BackendKind, VokraError};
    use vokra_models::compute::{Compute, HotOp};

    /// Off the Metal build the SnakeBeta coverage arm is a
    /// `BackendUnavailable` at the `for_backend` entry — never a silent CPU
    /// substitute (FR-EX-08).
    #[test]
    fn for_backend_metal_snake_beta_off_feature_is_backend_unavailable() {
        let Err(err) = Compute::for_backend(BackendKind::Metal, &[HotOp::SnakeBeta]) else {
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
    use vokra_ops::snake_beta_f32;

    /// Metal / CPU parity bound for the snake_beta FP32 element-wise op.
    /// Same 5e-4 budget as the sibling snake_activation / codec-family
    /// parity tests. The only bit-level source of divergence is `sin`'s
    /// transcendental gap between MSL fast-math and Rust `f32::sin`, which
    /// is well inside the bound for finite inputs.
    const ATOL: f32 = 5e-4;

    fn max_delta(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f32, f32::max)
    }

    /// Deterministic pseudo-random f32 (SplitMix64 → 24-bit mantissa) — no
    /// stdlib RNG dep, deterministic across runs and across hosts. Values
    /// stay in `[-scale, scale]`.
    fn splitmix_f32(state: &mut u64, scale: f32) -> f32 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        let bits = (z >> 40) as u32; // 24 bits
        let u = (bits as f32) / (1u32 << 24) as f32; // [0, 1)
        (u * 2.0 - 1.0) * scale
    }

    /// Build a Metal `Compute` covering SnakeBeta. Returns `None` when no
    /// Metal device is present (clean skip, mirrors the sibling parity
    /// tests' pattern).
    fn metal_compute() -> Option<Compute> {
        match Compute::for_backend(BackendKind::Metal, &[HotOp::SnakeBeta]) {
            Ok(c) => Some(c),
            Err(VokraError::BackendUnavailable(_)) => None,
            Err(e) => panic!("unexpected Metal for_backend error: {e}"),
        }
    }

    /// Tiny shape (4 channels × 7 time). Small enough to reason about, big
    /// enough to exercise the ragged tail of the 16×16 threadgroup dispatch
    /// on both axes (4 < 16 and 7 < 16).
    #[test]
    fn tiny_shape_metal_matches_cpu_within_atol_or_clean_skip() {
        let Some(compute_metal) = metal_compute() else {
            println!("skip: no Metal device on this host (clean skip, never fabricated)");
            return;
        };
        assert_eq!(compute_metal.backend_name(), "metal");
        let compute_cpu = Compute::cpu();

        let channels = 4;
        let time = 7;
        let alpha: Vec<f32> = (0..channels).map(|c| 0.3 + c as f32 * 0.17).collect();
        let beta: Vec<f32> = (0..channels).map(|c| 0.5 + c as f32 * 0.11).collect();
        let x: Vec<f32> = (0..channels * time)
            .map(|i| ((i as f32) * 0.11).sin() * 1.7)
            .collect();

        let mut cpu_out = vec![0.0f32; channels * time];
        let mut metal_out = vec![0.0f32; channels * time];
        compute_cpu
            .snake_beta_f32(&x, &alpha, &beta, channels, time, &mut cpu_out)
            .expect("CPU arm must succeed");
        compute_metal
            .snake_beta_f32(&x, &alpha, &beta, channels, time, &mut metal_out)
            .expect("Metal arm must succeed post Vocoder wave");
        assert_eq!(cpu_out.len(), metal_out.len(), "output shapes must match");

        let d = max_delta(&cpu_out, &metal_out);
        println!("tiny (4 ch × 7 t) SnakeBeta Metal vs CPU max |Δ| = {d:e}");
        assert!(
            d <= ATOL,
            "snake_beta_f32 tiny Metal vs CPU max |Δ| = {d} > {ATOL}"
        );

        // Cross-check: the CPU arm is bit-identical to the vokra_ops free
        // function (this is a sanity gate on the CPU arm rather than a
        // Metal claim).
        let mut reference = vec![0.0f32; channels * time];
        snake_beta_f32(&x, &alpha, &beta, channels, time, &mut reference).unwrap();
        for (i, (&c, &r)) in cpu_out.iter().zip(reference.iter()).enumerate() {
            assert_eq!(
                c.to_bits(),
                r.to_bits(),
                "index {i}: Compute::cpu()::snake_beta_f32 must be bit-identical to \
                 vokra_ops::snake_beta_f32 (got {c} vs {r})",
            );
        }
    }

    /// BigVGAN-AMP-ish shape (256 channels × 32 time). Squarely inside a
    /// 16×16 threadgroup grid on both axes; canonical vocoder chunk size.
    /// Also asserts the negative-control bound holds so ATOL is not vacuous.
    #[test]
    fn bigvgan_amp_shape_metal_matches_cpu_with_negative_control_or_clean_skip() {
        let Some(compute_metal) = metal_compute() else {
            println!("skip: no Metal device on this host (clean skip, never fabricated)");
            return;
        };
        let compute_cpu = Compute::cpu();

        let channels = 256;
        let time = 32;
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        let alpha: Vec<f32> = (0..channels)
            .map(|_| 0.2 + (splitmix_f32(&mut state, 1.0).abs() + 0.1) * 0.65)
            .collect();
        let beta: Vec<f32> = (0..channels)
            .map(|_| 0.5 + (splitmix_f32(&mut state, 1.0).abs() + 0.1) * 0.85)
            .collect();
        let x: Vec<f32> = (0..channels * time)
            .map(|_| splitmix_f32(&mut state, 1.5))
            .collect();

        let mut cpu_out = vec![0.0f32; channels * time];
        let mut metal_out = vec![0.0f32; channels * time];
        compute_cpu
            .snake_beta_f32(&x, &alpha, &beta, channels, time, &mut cpu_out)
            .expect("CPU arm must succeed");
        compute_metal
            .snake_beta_f32(&x, &alpha, &beta, channels, time, &mut metal_out)
            .expect("Metal arm must succeed");
        assert_eq!(cpu_out.len(), metal_out.len());

        let d = max_delta(&cpu_out, &metal_out);
        println!("bigvgan-amp (256 ch × 32 t) SnakeBeta Metal vs CPU max |Δ| = {d:e}");
        assert!(
            d <= ATOL,
            "bigvgan-amp snake_beta_f32 Metal vs CPU max |Δ| = {d} > {ATOL}"
        );

        // Negative control: perturb one β entry by 0.5. The perturbation
        // cascades through the entire row (all `time` outputs on that
        // channel), so a well-behaved kernel must move the CPU output past
        // ATOL. This proves the ≤ ATOL agreement above is a real match,
        // not a floor any two outputs would satisfy.
        let perturb_channel = 42_usize.min(channels - 1);
        let mut beta_perturbed = beta.clone();
        beta_perturbed[perturb_channel] += 0.5;
        let mut cpu_out_perturbed = vec![0.0f32; channels * time];
        compute_cpu
            .snake_beta_f32(
                &x,
                &alpha,
                &beta_perturbed,
                channels,
                time,
                &mut cpu_out_perturbed,
            )
            .expect("CPU perturbed arm must succeed");
        let control = max_delta(&cpu_out, &cpu_out_perturbed);
        println!("negative control (Δβ = 0.5 on channel {perturb_channel}) max |Δ| = {control:e}");
        assert!(
            control > ATOL,
            "negative control: a 0.5 β perturbation moved CPU output only {control} ≤ {ATOL} \
             — the atol bound would be vacuous; test cannot honestly claim parity"
        );
    }

    /// Ragged tail on both axes (channels=5, time=17) — neither divides
    /// 16, so the 16×16 threadgroup dispatch must correctly bounds-check
    /// every out-of-range thread. A broken guard would produce garbage on
    /// the out-of-range slots or touch adjacent memory — either shows up
    /// as a large max |Δ|.
    #[test]
    fn ragged_tail_shape_metal_matches_cpu_or_clean_skip() {
        let Some(compute_metal) = metal_compute() else {
            println!("skip: no Metal device on this host (clean skip, never fabricated)");
            return;
        };
        let compute_cpu = Compute::cpu();

        let channels = 5;
        let time = 17;
        let alpha: Vec<f32> = (0..channels).map(|c| 0.4 + c as f32 * 0.13).collect();
        let beta: Vec<f32> = (0..channels).map(|c| 0.7 + c as f32 * 0.09).collect();
        let x: Vec<f32> = (0..channels * time)
            .map(|i| ((i as f32) * 0.19).sin() * 1.3)
            .collect();

        let mut cpu_out = vec![0.0f32; channels * time];
        let mut metal_out = vec![0.0f32; channels * time];
        compute_cpu
            .snake_beta_f32(&x, &alpha, &beta, channels, time, &mut cpu_out)
            .expect("CPU arm must succeed");
        compute_metal
            .snake_beta_f32(&x, &alpha, &beta, channels, time, &mut metal_out)
            .expect("Metal arm must succeed");

        let d = max_delta(&cpu_out, &metal_out);
        println!("ragged (5 ch × 17 t) SnakeBeta Metal vs CPU max |Δ| = {d:e}");
        assert!(
            d <= ATOL,
            "ragged-tail snake_beta_f32 Metal vs CPU max |Δ| = {d} > {ATOL}"
        );
    }

    /// FR-EX-08 host-side validation: wrong alpha length is
    /// `InvalidArgument` (never a silent GPU OOB or clamp).
    #[test]
    fn wrong_alpha_length_is_invalid_argument_or_clean_skip() {
        let Some(compute_metal) = metal_compute() else {
            println!("skip: no Metal device on this host (clean skip, never fabricated)");
            return;
        };
        let channels = 4;
        let time = 3;
        let alpha = vec![1.0f32; channels + 1]; // one too many
        let beta = vec![1.0f32; channels];
        let x = vec![0.0f32; channels * time];
        let mut out = vec![0.0f32; channels * time];
        let err = compute_metal
            .snake_beta_f32(&x, &alpha, &beta, channels, time, &mut out)
            .expect_err("wrong alpha length must be an explicit InvalidArgument (FR-EX-08)");
        assert!(
            matches!(err, VokraError::InvalidArgument(_)),
            "expected InvalidArgument for wrong alpha length, got {err:?}",
        );
    }

    /// FR-EX-08 host-side validation: wrong beta length is `InvalidArgument`.
    #[test]
    fn wrong_beta_length_is_invalid_argument_or_clean_skip() {
        let Some(compute_metal) = metal_compute() else {
            println!("skip: no Metal device on this host (clean skip, never fabricated)");
            return;
        };
        let channels = 4;
        let time = 3;
        let alpha = vec![1.0f32; channels];
        let beta = vec![1.0f32; channels + 1]; // one too many
        let x = vec![0.0f32; channels * time];
        let mut out = vec![0.0f32; channels * time];
        let err = compute_metal
            .snake_beta_f32(&x, &alpha, &beta, channels, time, &mut out)
            .expect_err("wrong beta length must be an explicit InvalidArgument (FR-EX-08)");
        assert!(
            matches!(err, VokraError::InvalidArgument(_)),
            "expected InvalidArgument for wrong beta length, got {err:?}",
        );
    }

    /// Empty `channels = 0` or `time = 0` is a no-op on both arms (no
    /// dispatch, no allocation panic). Mirrors the CPU op's contract.
    #[test]
    fn empty_shape_returns_ok_no_op_or_clean_skip() {
        let Some(compute_metal) = metal_compute() else {
            println!("skip: no Metal device on this host (clean skip, never fabricated)");
            return;
        };
        let mut out: Vec<f32> = Vec::new();
        compute_metal
            .snake_beta_f32(&[], &[], &[], 0, 5, &mut out)
            .expect("channels=0 must be a no-op");
        assert!(out.is_empty());
        let alpha = vec![1.0f32, 2.0, 3.0];
        let beta = vec![1.0f32, 2.0, 3.0];
        let mut out: Vec<f32> = Vec::new();
        compute_metal
            .snake_beta_f32(&[], &alpha, &beta, 3, 0, &mut out)
            .expect("time=0 must be a no-op");
        assert!(out.is_empty());
    }
}
