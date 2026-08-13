//! Vocoder Metal wave common vocoder primitive — polyphase anti-aliased
//! upsample MSL kernel real-GPU parity
//! (`vokra_anti_aliased_upsample_f32` implemented in
//! `crates/vokra-backend-metal/src/context.rs`, wired through
//! `Compute::anti_aliased_upsample_f32` in
//! `crates/vokra-models/src/compute.rs`).
//!
//! - **Off-feature band** (compiled when `metal` is off / non-Apple):
//!   `HotOp::AntiAliasedUpsample` is not Metal-covered off the feature, so
//!   `Compute::for_backend(Metal, [AntiAliasedUpsample])` is an explicit
//!   [`vokra_core::VokraError::BackendUnavailable`] — never a silent CPU
//!   substitute (FR-EX-08).
//! - **Metal band** (`--features metal`, Apple): real-GPU parity on THIS M1
//!   iMac (CLAUDE.md dev environment). Skips cleanly (printed reason) when no
//!   Metal device is present.
//!
//! # atol bound and negative control
//!
//! Anti-aliased upsample is a pure FIR reduction (`Σ_j kernel[j*ratio + r] *
//! x[t - j]`) — no transcendentals, no branching gathers, no floating-point
//! division. The only bit-level source of divergence between the CPU and
//! GPU is the compiler's freedom to fuse the multiply-add to `fma`: MSL
//! fast-math is on by default and may emit `fma`, while the CPU op is a
//! Rust `for` loop with `acc += x * k` that rustc does NOT fuse to `fmadd`
//! without explicit `.mul_add`. For a typical Kaiser kernel of ≤ 64 taps
//! the FMA-vs-non-FMA divergence stays well inside `atol ≤ 1e-4`.
//!
//! A **negative control** shows the bound is discriminating rather than
//! vacuous: perturbing one kernel tap by 0.1 moves the CPU output well
//! past 1e-4 (the perturbation cascades through every output position that
//! uses that tap), so a "CPU vs Metal ≤ 1e-4" agreement is a real match,
//! not a floor.
//!
//! # Kaiser design lives on the host
//!
//! The `cutoff`, `filter_kernel`, `periodicity` attributes on the audit
//! request are Kaiser-window filter design metadata; the op consumes the
//! already-designed taps. This test uses simple hand-crafted kernels
//! (impulse, box, small triangular) that stress the polyphase index math
//! without needing a Kaiser designer.

#[cfg(not(all(feature = "metal", any(target_os = "macos", target_os = "ios"))))]
mod off_feature {
    use vokra_core::{BackendKind, VokraError};
    use vokra_models::compute::{Compute, HotOp};

    /// Off the Metal build the AntiAliasedUpsample coverage arm is a
    /// `BackendUnavailable` at the `for_backend` entry — never a silent CPU
    /// substitute (FR-EX-08).
    #[test]
    fn for_backend_metal_anti_aliased_upsample_off_feature_is_backend_unavailable() {
        let err = Compute::for_backend(BackendKind::Metal, &[HotOp::AntiAliasedUpsample])
            .expect_err("off-feature Metal must fail explicitly, not silently CPU-substitute");
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
    use vokra_ops::anti_aliased_upsample_f32;

    /// Metal / CPU parity bound for the FIR reduction. Tighter than the
    /// sibling snake / codec-family bound (5e-4) because there is no
    /// transcendental gap — only the FMA-vs-non-FMA freedom on the
    /// multiply-add. For ≤ 64 taps the divergence stays well below 1e-4.
    const ATOL: f32 = 1e-4;

    fn max_delta(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f32, f32::max)
    }

    /// Deterministic pseudo-random f32 (SplitMix64 → 24-bit mantissa).
    /// Values stay in `[-scale, scale]`.
    fn splitmix_f32(state: &mut u64, scale: f32) -> f32 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        let bits = (z >> 40) as u32;
        let u = (bits as f32) / (1u32 << 24) as f32;
        (u * 2.0 - 1.0) * scale
    }

    /// Build a Metal `Compute` covering AntiAliasedUpsample. Returns `None`
    /// when no Metal device is present (clean skip).
    fn metal_compute() -> Option<Compute> {
        match Compute::for_backend(BackendKind::Metal, &[HotOp::AntiAliasedUpsample]) {
            Ok(c) => Some(c),
            Err(VokraError::BackendUnavailable(_)) => None,
            Err(e) => panic!("unexpected Metal for_backend error: {e}"),
        }
    }

    /// Small ratio-2 case with a length-4 causal kernel. Hand-computed
    /// output in the CPU test (`ratio_two_length_four_kernel_hand_computed`);
    /// here we confirm Metal reproduces the CPU numerics within ATOL.
    #[test]
    fn ratio_two_length_four_metal_matches_cpu_or_clean_skip() {
        let Some(compute_metal) = metal_compute() else {
            println!("skip: no Metal device on this host (clean skip, never fabricated)");
            return;
        };
        assert_eq!(compute_metal.backend_name(), "metal");
        let compute_cpu = Compute::cpu();

        let x = vec![1.0f32, 2.0, 3.0];
        let kernel = vec![0.5f32, 0.25, 0.1, 0.05];
        let ratio = 2usize;
        let channels = 1usize;
        let time_in = 3usize;
        let out_len = channels * time_in * ratio;

        let mut cpu_out = vec![0.0f32; out_len];
        let mut metal_out = vec![0.0f32; out_len];
        compute_cpu
            .anti_aliased_upsample_f32(&x, &kernel, ratio, channels, time_in, &mut cpu_out)
            .expect("CPU arm must succeed");
        compute_metal
            .anti_aliased_upsample_f32(&x, &kernel, ratio, channels, time_in, &mut metal_out)
            .expect("Metal arm must succeed");
        assert_eq!(cpu_out.len(), metal_out.len(), "output shapes must match");

        let d = max_delta(&cpu_out, &metal_out);
        println!("ratio-2 len-4 kernel Metal vs CPU max |Δ| = {d:e}");
        assert!(
            d <= ATOL,
            "ratio-2 len-4 kernel Metal vs CPU max |Δ| = {d} > {ATOL}"
        );

        // Cross-check: the CPU arm is bit-identical to the vokra_ops free
        // function (sanity gate on the CPU arm rather than a Metal claim).
        let mut reference = vec![0.0f32; out_len];
        anti_aliased_upsample_f32(&x, &kernel, ratio, channels, time_in, &mut reference).unwrap();
        for (i, (&c, &r)) in cpu_out.iter().zip(reference.iter()).enumerate() {
            assert_eq!(
                c.to_bits(),
                r.to_bits(),
                "index {i}: Compute::cpu()::anti_aliased_upsample_f32 must be bit-identical to \
                 vokra_ops::anti_aliased_upsample_f32 (got {c} vs {r})",
            );
        }
    }

    /// Typical BigVGAN AMP-scale upsample: 128 channels × 32 time_in with
    /// ratio = 4 and a 24-tap kernel. Squarely inside a 16×16 threadgroup
    /// grid; exercises the reduction shape at production sizes and asserts
    /// the negative-control bound so ATOL is not vacuous.
    #[test]
    fn bigvgan_scale_metal_matches_cpu_with_negative_control_or_clean_skip() {
        let Some(compute_metal) = metal_compute() else {
            println!("skip: no Metal device on this host (clean skip, never fabricated)");
            return;
        };
        let compute_cpu = Compute::cpu();

        let channels = 128usize;
        let time_in = 32usize;
        let ratio = 4usize;
        let taps = 24usize;
        let out_len = channels * time_in * ratio;

        // Deterministic FIR taps: small triangular window (peak in the
        // middle, decays either side). Real Kaiser taps would be similar
        // shape but with sinc oscillations; the polyphase math is the same.
        let kernel: Vec<f32> = (0..taps)
            .map(|j| {
                let mid = taps as f32 / 2.0;
                let d = 1.0 - (j as f32 - mid).abs() / mid;
                d.max(0.0) * 0.5
            })
            .collect();
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        let x: Vec<f32> = (0..channels * time_in)
            .map(|_| splitmix_f32(&mut state, 1.5))
            .collect();

        let mut cpu_out = vec![0.0f32; out_len];
        let mut metal_out = vec![0.0f32; out_len];
        compute_cpu
            .anti_aliased_upsample_f32(&x, &kernel, ratio, channels, time_in, &mut cpu_out)
            .expect("CPU arm must succeed");
        compute_metal
            .anti_aliased_upsample_f32(&x, &kernel, ratio, channels, time_in, &mut metal_out)
            .expect("Metal arm must succeed");
        assert_eq!(cpu_out.len(), metal_out.len());

        let d = max_delta(&cpu_out, &metal_out);
        println!("bigvgan-scale (128 ch × 32 in × 4 ratio × 24 taps) Metal vs CPU max |Δ| = {d:e}");
        assert!(
            d <= ATOL,
            "bigvgan-scale anti_aliased_upsample_f32 Metal vs CPU max |Δ| = {d} > {ATOL}"
        );

        // Negative control: perturb one tap by 0.1. The perturbation
        // cascades through every output position that uses that tap
        // (roughly 1 / ratio of the outputs on every channel), which is
        // large enough to push CPU output past ATOL by many orders of
        // magnitude. This proves the ≤ ATOL agreement above is a real
        // match, not a floor any two outputs would satisfy.
        let mut kernel_perturbed = kernel.clone();
        kernel_perturbed[taps / 2] += 0.1; // peak tap
        let mut cpu_out_perturbed = vec![0.0f32; out_len];
        compute_cpu
            .anti_aliased_upsample_f32(
                &x,
                &kernel_perturbed,
                ratio,
                channels,
                time_in,
                &mut cpu_out_perturbed,
            )
            .expect("CPU perturbed arm must succeed");
        let control = max_delta(&cpu_out, &cpu_out_perturbed);
        println!("negative control (Δkernel = 0.1 on peak tap) max |Δ| = {control:e}");
        assert!(
            control > ATOL,
            "negative control: a 0.1 tap perturbation moved CPU output only {control} ≤ {ATOL} \
             — the atol bound would be vacuous; test cannot honestly claim parity"
        );
    }

    /// Ragged tail on both output axes (channels = 5, time_out = 17 with
    /// ratio = 2 → time_in = 8..9 wait no; construct ratio=2, time_in=9 →
    /// time_out=18 which we round to a tight 17-wide out but we can't have
    /// time_out not divisible by ratio — use time_in=9, ratio=2 →
    /// time_out=18, still divides 16 unevenly). Kernel length 5 (ragged
    /// polyphase branch — one branch has 3 taps, one has 2). Broken guard
    /// would surface as a large max |Δ|.
    #[test]
    fn ragged_tail_metal_matches_cpu_or_clean_skip() {
        let Some(compute_metal) = metal_compute() else {
            println!("skip: no Metal device on this host (clean skip, never fabricated)");
            return;
        };
        let compute_cpu = Compute::cpu();

        let channels = 5usize;
        let time_in = 9usize;
        let ratio = 2usize;
        let kernel = vec![0.3f32, 0.2, 0.15, 0.1, 0.05];
        let time_out = time_in * ratio;
        let out_len = channels * time_out;
        let mut state: u64 = 0xC0FF_EE00_1337_0042;
        let x: Vec<f32> = (0..channels * time_in)
            .map(|_| splitmix_f32(&mut state, 2.0))
            .collect();

        let mut cpu_out = vec![0.0f32; out_len];
        let mut metal_out = vec![0.0f32; out_len];
        compute_cpu
            .anti_aliased_upsample_f32(&x, &kernel, ratio, channels, time_in, &mut cpu_out)
            .expect("CPU arm must succeed");
        compute_metal
            .anti_aliased_upsample_f32(&x, &kernel, ratio, channels, time_in, &mut metal_out)
            .expect("Metal arm must succeed");

        let d = max_delta(&cpu_out, &metal_out);
        println!("ragged (5 ch × 9 in × 2 ratio × 5 taps) Metal vs CPU max |Δ| = {d:e}");
        assert!(
            d <= ATOL,
            "ragged-tail anti_aliased_upsample_f32 Metal vs CPU max |Δ| = {d} > {ATOL}"
        );
    }

    /// FR-EX-08 host-side validation: ratio = 0 is `InvalidArgument`.
    #[test]
    fn zero_ratio_is_invalid_argument_or_clean_skip() {
        let Some(compute_metal) = metal_compute() else {
            println!("skip: no Metal device on this host (clean skip, never fabricated)");
            return;
        };
        let mut out = vec![0.0f32; 4];
        let err = compute_metal
            .anti_aliased_upsample_f32(&[0.0f32; 4], &[1.0], 0, 1, 4, &mut out)
            .expect_err("ratio=0 must be an explicit InvalidArgument (FR-EX-08)");
        assert!(
            matches!(err, VokraError::InvalidArgument(_)),
            "expected InvalidArgument for ratio=0, got {err:?}",
        );
    }

    /// FR-EX-08 host-side validation: empty kernel is `InvalidArgument`.
    #[test]
    fn empty_kernel_is_invalid_argument_or_clean_skip() {
        let Some(compute_metal) = metal_compute() else {
            println!("skip: no Metal device on this host (clean skip, never fabricated)");
            return;
        };
        let mut out = vec![0.0f32; 4];
        let err = compute_metal
            .anti_aliased_upsample_f32(&[0.0f32; 4], &[], 1, 1, 4, &mut out)
            .expect_err("empty kernel must be an explicit InvalidArgument (FR-EX-08)");
        assert!(
            matches!(err, VokraError::InvalidArgument(_)),
            "expected InvalidArgument for empty kernel, got {err:?}",
        );
    }

    /// FR-EX-08 host-side validation: wrong x length is `InvalidArgument`.
    #[test]
    fn wrong_x_length_is_invalid_argument_or_clean_skip() {
        let Some(compute_metal) = metal_compute() else {
            println!("skip: no Metal device on this host (clean skip, never fabricated)");
            return;
        };
        let x = vec![0.0f32; 3]; // channels=2, time_in=2 → expected 4
        let mut out = vec![0.0f32; 8];
        let err = compute_metal
            .anti_aliased_upsample_f32(&x, &[1.0], 2, 2, 2, &mut out)
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
        let x = vec![0.0f32; 4]; // channels=2, time_in=2
        let mut out = vec![0.0f32; 7]; // ratio=2 → expected 8
        let err = compute_metal
            .anti_aliased_upsample_f32(&x, &[1.0], 2, 2, 2, &mut out)
            .expect_err("wrong out length must be an explicit InvalidArgument (FR-EX-08)");
        assert!(
            matches!(err, VokraError::InvalidArgument(_)),
            "expected InvalidArgument for wrong out length, got {err:?}",
        );
    }

    /// Empty `channels = 0` or `time_in = 0` is a no-op on both arms.
    #[test]
    fn empty_shape_returns_ok_no_op_or_clean_skip() {
        let Some(compute_metal) = metal_compute() else {
            println!("skip: no Metal device on this host (clean skip, never fabricated)");
            return;
        };
        let mut out: Vec<f32> = Vec::new();
        compute_metal
            .anti_aliased_upsample_f32(&[], &[1.0], 2, 0, 5, &mut out)
            .expect("channels=0 must be a no-op");
        assert!(out.is_empty());
        let mut out: Vec<f32> = Vec::new();
        compute_metal
            .anti_aliased_upsample_f32(&[], &[1.0], 2, 3, 0, &mut out)
            .expect("time_in=0 must be a no-op");
        assert!(out.is_empty());
    }
}
