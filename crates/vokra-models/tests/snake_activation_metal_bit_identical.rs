//! Vocoder Metal wave WF2 — Snake activation Metal MSL kernel real-GPU parity
//! (`vokra_snake_activation_f32` implemented in
//! `crates/vokra-backend-metal/src/context.rs`, wired through
//! `Compute::snake_activation_f32` in `crates/vokra-models/src/compute.rs`).
//!
//! - **Off-feature band** (compiled when `metal` is off / non-Apple):
//!   `HotOp::SnakeActivation` is not Metal-covered off the feature, so
//!   `Compute::for_backend(Metal, [SnakeActivation])` is an explicit
//!   [`vokra_core::VokraError::BackendUnavailable`] — never a silent CPU
//!   substitute (FR-EX-08).
//! - **Metal band** (`--features metal`, Apple): real-GPU parity on THIS M1
//!   iMac (CLAUDE.md dev environment). Skips cleanly (printed reason) when no
//!   Metal device is present.
//!
//! # atol bound and negative control
//!
//! Snake is trivially element-wise (no reduction, no gather) — the only
//! bit-level source of divergence is the transcendental gap between MSL's
//! intrinsic `sin` (compiled with fast-math defaults) and Rust's `f32::sin`.
//! In practice max |Δ| is at or below the ULP scale of `sin` for finite
//! inputs, and we assert the honest atol ≤ 5e-4 bound — the same envelope
//! used by the sibling codec-family parity tests
//! (`mimi_rvq_metal_bit_identical.rs::ATOL`,
//! `dac_rvq_decode_metal_bit_identical.rs::ATOL`,
//! `fsq_codec_decode_metal_bit_identical.rs::ATOL`).
//!
//! A **negative control** shows the bound is discriminating rather than
//! vacuous: perturbing one α entry by 0.1 moves the CPU output well past
//! 5e-4 (the perturbation cascades through the per-channel row), so a "CPU
//! vs Metal ≤ 5e-4" agreement is a real match, not a floor.
//!
//! # Not `alpha_logscale`, not `SnakeBeta`
//!
//! The MSL kernel matches the plain-Snake closed form (`y = x +
//! (1/(α+ε))·sin(α·x)²`) — the same as
//! [`vokra_ops::hiftnet::Snake::forward_in_place`] under
//! `alpha_logscale = false` and the private
//! `kokoro::nn::snake_activation` helper in vokra-models. The
//! `alpha_logscale = true` variant is expected to have `exp(alpha_raw)`
//! applied on the host before dispatch (the CPU free function has the same
//! contract). `SnakeBeta` (two-vector variant) is a distinct op and lands
//! its own kernel separately.
//!
//! # Real-weight parity
//!
//! Synthetic ramp / random alphas exercise the shader shape end-to-end on
//! tiny (4 × 7), Kokoro-decoder-ish (256 × 32) and BigVGAN-AMP-ish (512 × 96)
//! canonical shapes. Real Kokoro / BigVGAN weights live inside vokra-models
//! private paths and their CPU-only decoders are what consumes this op today;
//! per-op parity here pins the Metal↔CPU numeric bound, and the
//! decoder-level parity tests exercise the same code path indirectly.

#[cfg(not(all(feature = "metal", any(target_os = "macos", target_os = "ios"))))]
mod off_feature {
    use vokra_core::{BackendKind, VokraError};
    use vokra_models::compute::{Compute, HotOp};

    /// Off the Metal build the SnakeActivation coverage arm is a
    /// `BackendUnavailable` at the `for_backend` entry — never a silent CPU
    /// substitute (FR-EX-08).
    #[test]
    fn for_backend_metal_snake_activation_off_feature_is_backend_unavailable() {
        let Err(err) = Compute::for_backend(BackendKind::Metal, &[HotOp::SnakeActivation]) else {
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
    use vokra_ops::snake_activation_f32;

    /// Metal / CPU parity bound for the snake_activation FP32 element-wise
    /// op. Same 5e-4 budget as the sibling codec-family parity tests
    /// (`mimi_rvq_metal_bit_identical.rs::ATOL`,
    /// `dac_rvq_decode_metal_bit_identical.rs::ATOL`,
    /// `fsq_codec_decode_metal_bit_identical.rs::ATOL`) — the M4-05 CSM /
    /// Moshi Metal parity envelope. The only bit-level source of divergence
    /// here is `sin`'s transcendental gap between MSL fast-math and Rust
    /// `f32::sin`, which is well inside the bound for finite inputs.
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
        // Take 24 bits and map to [-1, 1), then scale.
        let bits = (z >> 40) as u32; // 24 bits
        let u = (bits as f32) / (1u32 << 24) as f32; // [0, 1)
        (u * 2.0 - 1.0) * scale
    }

    /// Build a Metal `Compute` covering SnakeActivation. Returns `None` when
    /// no Metal device is present (clean skip, mirrors the sibling codec
    /// parity tests' pattern).
    fn metal_compute() -> Option<Compute> {
        match Compute::for_backend(BackendKind::Metal, &[HotOp::SnakeActivation]) {
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
        // Deterministic α: spread from 0.3 to ~0.81 (finite, non-trivial).
        let alpha: Vec<f32> = (0..channels).map(|c| 0.3 + c as f32 * 0.17).collect();
        // Deterministic ramp × sin (mildly non-linear input; hits multiple
        // sin() periods so the transcendental gap has room to matter).
        let x: Vec<f32> = (0..channels * time)
            .map(|i| ((i as f32) * 0.11).sin() * 1.7)
            .collect();

        let mut cpu_out = vec![0.0f32; channels * time];
        let mut metal_out = vec![0.0f32; channels * time];
        compute_cpu
            .snake_activation_f32(&x, &alpha, channels, time, &mut cpu_out)
            .expect("CPU arm must succeed");
        compute_metal
            .snake_activation_f32(&x, &alpha, channels, time, &mut metal_out)
            .expect("Metal arm must succeed post Vocoder wave WF2");
        assert_eq!(cpu_out.len(), metal_out.len(), "output shapes must match");

        let d = max_delta(&cpu_out, &metal_out);
        println!("tiny (4 ch × 7 t) Metal vs CPU max |Δ| = {d:e}");
        assert!(
            d <= ATOL,
            "snake_activation_f32 tiny Metal vs CPU max |Δ| = {d} > {ATOL}"
        );

        // Cross-check: the CPU arm is bit-identical to the vokra_ops free
        // function (this is a sanity gate on the CPU arm rather than a
        // Metal claim).
        let mut reference = vec![0.0f32; channels * time];
        snake_activation_f32(&x, &alpha, channels, time, &mut reference).unwrap();
        for (i, (&c, &r)) in cpu_out.iter().zip(reference.iter()).enumerate() {
            assert_eq!(
                c.to_bits(),
                r.to_bits(),
                "index {i}: Compute::cpu()::snake_activation_f32 must be bit-identical to \
                 vokra_ops::snake_activation_f32 (got {c} vs {r})",
            );
        }
    }

    /// Kokoro-decoder-ish shape (256 channels × 32 time). Squarely inside a
    /// 16×16 threadgroup grid on both axes, exercises typical vocoder
    /// per-chunk sizes. Also asserts the negative-control bound holds so
    /// the ATOL is not vacuous.
    #[test]
    fn kokoro_shape_metal_matches_cpu_with_negative_control_or_clean_skip() {
        let Some(compute_metal) = metal_compute() else {
            println!("skip: no Metal device on this host (clean skip, never fabricated)");
            return;
        };
        let compute_cpu = Compute::cpu();

        let channels = 256;
        let time = 32;
        // Deterministic α spread ∈ [0.2, ~1.5], all finite.
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        let alpha: Vec<f32> = (0..channels)
            .map(|_| 0.2 + (splitmix_f32(&mut state, 1.0).abs() + 0.1) * 0.65)
            .collect();
        // Deterministic x ∈ [-1.5, 1.5], covering several sin periods per
        // channel to exercise the transcendental gap end-to-end.
        let x: Vec<f32> = (0..channels * time)
            .map(|_| splitmix_f32(&mut state, 1.5))
            .collect();

        let mut cpu_out = vec![0.0f32; channels * time];
        let mut metal_out = vec![0.0f32; channels * time];
        compute_cpu
            .snake_activation_f32(&x, &alpha, channels, time, &mut cpu_out)
            .expect("CPU arm must succeed");
        compute_metal
            .snake_activation_f32(&x, &alpha, channels, time, &mut metal_out)
            .expect("Metal arm must succeed");
        assert_eq!(cpu_out.len(), metal_out.len());

        let d = max_delta(&cpu_out, &metal_out);
        println!("kokoro (256 ch × 32 t) Metal vs CPU max |Δ| = {d:e}");
        assert!(
            d <= ATOL,
            "kokoro snake_activation_f32 Metal vs CPU max |Δ| = {d} > {ATOL}"
        );

        // Negative control: perturb one α entry by 0.1. The perturbation
        // cascades through the entire row (all `time` outputs on that
        // channel), so a well-behaved kernel must move the CPU output past
        // ATOL. This proves the ≤ ATOL agreement above is a real match,
        // not a floor any two outputs would satisfy.
        let perturb_channel = 42_usize.min(channels - 1);
        let mut alpha_perturbed = alpha.clone();
        alpha_perturbed[perturb_channel] += 0.1;
        let mut cpu_out_perturbed = vec![0.0f32; channels * time];
        compute_cpu
            .snake_activation_f32(&x, &alpha_perturbed, channels, time, &mut cpu_out_perturbed)
            .expect("CPU perturbed arm must succeed");
        let control = max_delta(&cpu_out, &cpu_out_perturbed);
        println!("negative control (Δα = 0.1 on channel {perturb_channel}) max |Δ| = {control:e}");
        assert!(
            control > ATOL,
            "negative control: a 0.1 α perturbation moved CPU output only {control} ≤ {ATOL} \
             — the atol bound would be vacuous; test cannot honestly claim parity"
        );
    }

    /// BigVGAN-AMP-ish shape (512 channels × 96 time). Multiple threadgroup
    /// tiles in both dimensions (512/16 = 32, 96/16 = 6 — no ragged tail on
    /// either axis) so the launch geometry is fully exercised.
    #[test]
    fn bigvgan_shape_metal_matches_cpu_or_clean_skip() {
        let Some(compute_metal) = metal_compute() else {
            println!("skip: no Metal device on this host (clean skip, never fabricated)");
            return;
        };
        let compute_cpu = Compute::cpu();

        let channels = 512;
        let time = 96;
        let mut state: u64 = 0xC0FF_EE00_1337_0042;
        let alpha: Vec<f32> = (0..channels)
            .map(|_| 0.5 + splitmix_f32(&mut state, 0.4).abs())
            .collect();
        let x: Vec<f32> = (0..channels * time)
            .map(|_| splitmix_f32(&mut state, 2.0))
            .collect();

        let mut cpu_out = vec![0.0f32; channels * time];
        let mut metal_out = vec![0.0f32; channels * time];
        compute_cpu
            .snake_activation_f32(&x, &alpha, channels, time, &mut cpu_out)
            .expect("CPU arm must succeed");
        compute_metal
            .snake_activation_f32(&x, &alpha, channels, time, &mut metal_out)
            .expect("Metal arm must succeed");

        let d = max_delta(&cpu_out, &metal_out);
        println!("bigvgan (512 ch × 96 t) Metal vs CPU max |Δ| = {d:e}");
        assert!(
            d <= ATOL,
            "bigvgan snake_activation_f32 Metal vs CPU max |Δ| = {d} > {ATOL}"
        );
    }

    /// Ragged tail on both axes (channels=5, time=17) — neither divides
    /// 16, so the 16×16 threadgroup dispatch must correctly bounds-check
    /// every out-of-range thread. If the `c >= d.channels || t >= d.time`
    /// guard is broken the read would either produce garbage output on the
    /// out-of-range slots or (with unified memory) touch adjacent memory —
    /// either case would show up as a large max |Δ|.
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
        let x: Vec<f32> = (0..channels * time)
            .map(|i| ((i as f32) * 0.19).sin() * 1.3)
            .collect();

        let mut cpu_out = vec![0.0f32; channels * time];
        let mut metal_out = vec![0.0f32; channels * time];
        compute_cpu
            .snake_activation_f32(&x, &alpha, channels, time, &mut cpu_out)
            .expect("CPU arm must succeed");
        compute_metal
            .snake_activation_f32(&x, &alpha, channels, time, &mut metal_out)
            .expect("Metal arm must succeed");

        let d = max_delta(&cpu_out, &metal_out);
        println!("ragged (5 ch × 17 t) Metal vs CPU max |Δ| = {d:e}");
        assert!(
            d <= ATOL,
            "ragged-tail snake_activation_f32 Metal vs CPU max |Δ| = {d} > {ATOL}"
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
        let x = vec![0.0f32; channels * time];
        let mut out = vec![0.0f32; channels * time];
        let err = compute_metal
            .snake_activation_f32(&x, &alpha, channels, time, &mut out)
            .expect_err("wrong alpha length must be an explicit InvalidArgument (FR-EX-08)");
        assert!(
            matches!(err, VokraError::InvalidArgument(_)),
            "expected InvalidArgument for wrong alpha length, got {err:?}",
        );
    }

    /// FR-EX-08 host-side validation: wrong x length is `InvalidArgument`.
    #[test]
    fn wrong_x_length_is_invalid_argument_or_clean_skip() {
        let Some(compute_metal) = metal_compute() else {
            println!("skip: no Metal device on this host (clean skip, never fabricated)");
            return;
        };
        let channels = 4;
        let time = 3;
        let alpha = vec![1.0f32; channels];
        let x = vec![0.0f32; channels * time - 1]; // one short
        let mut out = vec![0.0f32; channels * time];
        let err = compute_metal
            .snake_activation_f32(&x, &alpha, channels, time, &mut out)
            .expect_err("wrong x length must be an explicit InvalidArgument (FR-EX-08)");
        assert!(
            matches!(err, VokraError::InvalidArgument(_)),
            "expected InvalidArgument for wrong x length, got {err:?}",
        );
    }

    /// FR-EX-08 host-side validation: wrong out length is `InvalidArgument`.
    #[test]
    fn wrong_out_length_is_invalid_argument_or_clean_skip() {
        let Some(compute_metal) = metal_compute() else {
            println!("skip: no Metal device on this host (clean skip, never fabricated)");
            return;
        };
        let channels = 4;
        let time = 3;
        let alpha = vec![1.0f32; channels];
        let x = vec![0.0f32; channels * time];
        let mut out = vec![0.0f32; channels * time + 1]; // one too many
        let err = compute_metal
            .snake_activation_f32(&x, &alpha, channels, time, &mut out)
            .expect_err("wrong out length must be an explicit InvalidArgument (FR-EX-08)");
        assert!(
            matches!(err, VokraError::InvalidArgument(_)),
            "expected InvalidArgument for wrong out length, got {err:?}",
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
        // channels = 0
        let mut out: Vec<f32> = Vec::new();
        compute_metal
            .snake_activation_f32(&[], &[], 0, 5, &mut out)
            .expect("channels=0 must be a no-op");
        assert!(out.is_empty());
        // time = 0
        let alpha = vec![1.0f32, 2.0, 3.0];
        let mut out: Vec<f32> = Vec::new();
        compute_metal
            .snake_activation_f32(&[], &alpha, 3, 0, &mut out)
            .expect("time=0 must be a no-op");
        assert!(out.is_empty());
    }
}
