//! Vocoder Metal wave WF5 — Denoise spectral-gate primitive Metal MSL kernel
//! real-GPU parity (`vokra_denoise_apply_mask_f32` implemented in
//! `crates/vokra-backend-metal/src/context.rs`, wired through
//! `Compute::denoise_apply_mask_f32` in `crates/vokra-models/src/compute.rs`).
//!
//! - **Off-feature band** (compiled when `metal` is off / non-Apple):
//!   `HotOp::DenoiseApplyMask` is not Metal-covered off the feature, so
//!   `Compute::for_backend(Metal, [DenoiseApplyMask])` is an explicit
//!   [`vokra_core::VokraError::BackendUnavailable`] — never a silent CPU
//!   substitute (FR-EX-08).
//! - **Metal band** (`--features metal`, Apple): real-GPU parity on THIS M1
//!   iMac (CLAUDE.md dev environment). Skips cleanly (printed reason) when no
//!   Metal device is present.
//!
//! # atol bound and negative control
//!
//! The primitive is `out_re = spec_re · gain`, `out_im = spec_im · gain` —
//! trivially per-element, no reduction, no transcendental, and no FMA
//! opportunity in a single `re * g`. IEEE-754 correctly-rounded FP32
//! multiplication is the same on CPU and GPU for every finite input, so
//! CPU and GPU are expected to be **bit-for-bit identical** (max |Δ| = 0).
//!
//! We assert the sibling `atol ≤ 5e-4` codec-family bound rather than
//! `to_bits()` equality because:
//! - the codec-family precedent is uniform (`mimi_rvq_metal_bit_identical`,
//!   `dac_rvq_decode_metal_bit_identical`, `fsq_codec_decode_metal_bit_identical`,
//!   `snake_activation_metal_bit_identical`, `snac_decode_metal_bit_identical`
//!   all use it), and
//! - a future MSL fast-math opt-in (e.g., `-ffast-math`) or a driver update
//!   that reassociates something surprising would move the outputs off
//!   bit-identical but keep them well within 5e-4; assuming permanent
//!   bit-identical would create a brittle test.
//!
//! The actual measured max |Δ| is logged (0 in practice — any future drift
//! is immediately visible from the logs). A **negative control** shows the
//! bound is discriminating rather than vacuous: perturbing one gain entry
//! by 0.1 moves the CPU output well past 5e-4 (the perturbation scales
//! `re[i]` by 0.1 for that position, which is O(0.1) for typical
//! spectrogram magnitudes).
//!
//! # Not the whole DenoiseModel
//!
//! The MSL kernel implements only the mask-apply primitive. The full DFN3
//! network (STFT → ERB features → DfNet → mask + deep-filter → iSTFT)
//! lives in [`vokra_ops::denoise::DenoiseModel::enhance`] and still uses
//! its fused inline output-stage loop (denoise.rs L1852-1870) on the CPU-
//! only path it has always taken. Per-op parity here pins the primitive's
//! Metal↔CPU numeric bound so a downstream mask denoiser (GTCRN / RNNoise)
//! that emits per-position gains directly can dispatch through this seam.
//!
//! # Real-weight parity
//!
//! Synthetic ramp / random spectrograms exercise the shader shape end-to-
//! end on tiny (4 frames × 7 bins), DFN3-24 kHz-ish (32 frames × 481 bins)
//! and GTCRN-16 kHz-ish (48 frames × 257 bins) canonical shapes. The full
//! DFN3 real-checkpoint parity harness (`tests/parity_denoise_dfn3.rs`,
//! env-gated on the real GGUF) exercises the whole DenoiseModel and would
//! catch any regression in the primitive too, but the per-op parity here
//! pins the Metal↔CPU numeric bound directly.

#[cfg(not(all(feature = "metal", any(target_os = "macos", target_os = "ios"))))]
mod off_feature {
    use vokra_core::{BackendKind, VokraError};
    use vokra_models::compute::{Compute, HotOp};

    /// Off the Metal build the DenoiseApplyMask coverage arm is a
    /// `BackendUnavailable` at the `for_backend` entry — never a silent CPU
    /// substitute (FR-EX-08).
    #[test]
    fn for_backend_metal_denoise_apply_mask_off_feature_is_backend_unavailable() {
        let Err(err) = Compute::for_backend(BackendKind::Metal, &[HotOp::DenoiseApplyMask]) else {
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
    use vokra_ops::denoise_apply_mask_f32;

    /// Metal / CPU parity bound for the denoise spectral-gate FP32
    /// element-wise op. Same 5e-4 budget as the sibling codec-family parity
    /// tests (`mimi_rvq_metal_bit_identical.rs::ATOL`,
    /// `dac_rvq_decode_metal_bit_identical.rs::ATOL`,
    /// `fsq_codec_decode_metal_bit_identical.rs::ATOL`,
    /// `snake_activation_metal_bit_identical.rs::ATOL`,
    /// `snac_decode_metal_bit_identical.rs::ATOL`) — the M4-05 CSM /
    /// Moshi Metal parity envelope. The primitive is pure IEEE-754
    /// correctly-rounded FP32 multiply with no reduction / transcendental
    /// / FMA opportunity, so in practice max |Δ| = 0 (the test logs the
    /// measured value so any future drift is immediately visible).
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

    /// Build a Metal `Compute` covering DenoiseApplyMask. Returns `None`
    /// when no Metal device is present (clean skip, mirrors the sibling
    /// codec parity tests' pattern).
    fn metal_compute() -> Option<Compute> {
        match Compute::for_backend(BackendKind::Metal, &[HotOp::DenoiseApplyMask]) {
            Ok(c) => Some(c),
            Err(VokraError::BackendUnavailable(_)) => None,
            Err(e) => panic!("unexpected Metal for_backend error: {e}"),
        }
    }

    /// Tiny shape (4 frames × 7 bins). Small enough to reason about, big
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

        let n_frames = 4;
        let n_bins = 7;
        let n = n_frames * n_bins;
        // Deterministic ramp × sin: mildly non-linear values covering
        // multiple periods.
        let spec_re: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.13).sin() * 2.1).collect();
        let spec_im: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.19).cos() * 1.7).collect();
        // Per-position gain in [0.1, 1.0]: covers a realistic denoiser mask
        // range (0 = fully suppress, 1 = pass through).
        let gain: Vec<f32> = (0..n)
            .map(|i| 0.1 + ((i as f32) * 0.07).sin().abs() * 0.9)
            .collect();

        let mut cpu_out_re = vec![0.0f32; n];
        let mut cpu_out_im = vec![0.0f32; n];
        let mut metal_out_re = vec![0.0f32; n];
        let mut metal_out_im = vec![0.0f32; n];
        compute_cpu
            .denoise_apply_mask_f32(
                &spec_re,
                &spec_im,
                &gain,
                n_frames,
                n_bins,
                &mut cpu_out_re,
                &mut cpu_out_im,
            )
            .expect("CPU arm must succeed");
        compute_metal
            .denoise_apply_mask_f32(
                &spec_re,
                &spec_im,
                &gain,
                n_frames,
                n_bins,
                &mut metal_out_re,
                &mut metal_out_im,
            )
            .expect("Metal arm must succeed post Vocoder wave WF5");
        assert_eq!(cpu_out_re.len(), metal_out_re.len(), "re shapes must match");
        assert_eq!(cpu_out_im.len(), metal_out_im.len(), "im shapes must match");

        let d_re = max_delta(&cpu_out_re, &metal_out_re);
        let d_im = max_delta(&cpu_out_im, &metal_out_im);
        println!("tiny (4 fr × 7 bins) Metal vs CPU max |Δ|: re = {d_re:e}, im = {d_im:e}");
        assert!(
            d_re <= ATOL,
            "denoise_apply_mask_f32 tiny Metal vs CPU max |Δ|.re = {d_re} > {ATOL}"
        );
        assert!(
            d_im <= ATOL,
            "denoise_apply_mask_f32 tiny Metal vs CPU max |Δ|.im = {d_im} > {ATOL}"
        );

        // Cross-check: the CPU arm is bit-identical to the vokra_ops free
        // function (this is a sanity gate on the CPU arm rather than a
        // Metal claim).
        let mut ref_re = vec![0.0f32; n];
        let mut ref_im = vec![0.0f32; n];
        denoise_apply_mask_f32(
            &spec_re,
            &spec_im,
            &gain,
            n_frames,
            n_bins,
            &mut ref_re,
            &mut ref_im,
        )
        .unwrap();
        for i in 0..n {
            assert_eq!(
                cpu_out_re[i].to_bits(),
                ref_re[i].to_bits(),
                "re index {i}: Compute::cpu()::denoise_apply_mask_f32 must be bit-identical to \
                 vokra_ops::denoise_apply_mask_f32 (got {} vs {})",
                cpu_out_re[i],
                ref_re[i]
            );
            assert_eq!(
                cpu_out_im[i].to_bits(),
                ref_im[i].to_bits(),
                "im index {i}: Compute::cpu()::denoise_apply_mask_f32 must be bit-identical to \
                 vokra_ops::denoise_apply_mask_f32 (got {} vs {})",
                cpu_out_im[i],
                ref_im[i]
            );
        }
    }

    /// DFN3-24 kHz-ish canonical shape: 32 frames × 481 bins
    /// (`n_fft = 960` → `n_fft/2 + 1 = 481`). Fully exercises the 16×16
    /// threadgroup grid — 481/16 = 31 groups on the fast axis (with a
    /// ragged tail of 1 col), 32/16 = 2 groups on the frame axis (no
    /// ragged tail). Also asserts the negative-control bound holds so
    /// the ATOL is not vacuous.
    #[test]
    fn dfn3_shape_metal_matches_cpu_with_negative_control_or_clean_skip() {
        let Some(compute_metal) = metal_compute() else {
            println!("skip: no Metal device on this host (clean skip, never fabricated)");
            return;
        };
        let compute_cpu = Compute::cpu();

        let n_frames = 32;
        let n_bins = 481;
        let n = n_frames * n_bins;
        // Deterministic spectrogram values in [-2, 2].
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        let spec_re: Vec<f32> = (0..n).map(|_| splitmix_f32(&mut state, 2.0)).collect();
        let spec_im: Vec<f32> = (0..n).map(|_| splitmix_f32(&mut state, 2.0)).collect();
        // Deterministic gain in [0.0, 1.0] (typical mask range).
        let gain: Vec<f32> = (0..n)
            .map(|_| 0.5 * (splitmix_f32(&mut state, 1.0) + 1.0))
            .collect();

        let mut cpu_out_re = vec![0.0f32; n];
        let mut cpu_out_im = vec![0.0f32; n];
        let mut metal_out_re = vec![0.0f32; n];
        let mut metal_out_im = vec![0.0f32; n];
        compute_cpu
            .denoise_apply_mask_f32(
                &spec_re,
                &spec_im,
                &gain,
                n_frames,
                n_bins,
                &mut cpu_out_re,
                &mut cpu_out_im,
            )
            .expect("CPU arm must succeed");
        compute_metal
            .denoise_apply_mask_f32(
                &spec_re,
                &spec_im,
                &gain,
                n_frames,
                n_bins,
                &mut metal_out_re,
                &mut metal_out_im,
            )
            .expect("Metal arm must succeed");

        let d_re = max_delta(&cpu_out_re, &metal_out_re);
        let d_im = max_delta(&cpu_out_im, &metal_out_im);
        println!("dfn3 (32 fr × 481 bins) Metal vs CPU max |Δ|: re = {d_re:e}, im = {d_im:e}");
        assert!(
            d_re <= ATOL,
            "dfn3 denoise_apply_mask_f32 Metal vs CPU max |Δ|.re = {d_re} > {ATOL}"
        );
        assert!(
            d_im <= ATOL,
            "dfn3 denoise_apply_mask_f32 Metal vs CPU max |Δ|.im = {d_im} > {ATOL}"
        );

        // Negative control: perturb one gain entry by 0.1 at a position
        // where |re| ≥ 0.2 so the perturbation cascades to |Δ| ≥ 0.02,
        // well past ATOL = 5e-4. The pertubation scales that one position's
        // re/im by an extra 0.1×; the rest of the output is unchanged.
        // We deliberately pick a position with a large |re| so the
        // discriminator is unambiguous.
        let perturb_idx = spec_re
            .iter()
            .enumerate()
            .find(|&(_, &v)| v.abs() >= 0.5)
            .map(|(i, _)| i)
            .expect("large-magnitude position exists in random data");
        let mut gain_perturbed = gain.clone();
        gain_perturbed[perturb_idx] += 0.1;
        let mut cpu_out_re_perturbed = vec![0.0f32; n];
        let mut cpu_out_im_perturbed = vec![0.0f32; n];
        compute_cpu
            .denoise_apply_mask_f32(
                &spec_re,
                &spec_im,
                &gain_perturbed,
                n_frames,
                n_bins,
                &mut cpu_out_re_perturbed,
                &mut cpu_out_im_perturbed,
            )
            .expect("CPU perturbed arm must succeed");
        let control_re = max_delta(&cpu_out_re, &cpu_out_re_perturbed);
        let control_im = max_delta(&cpu_out_im, &cpu_out_im_perturbed);
        println!(
            "negative control (Δgain = 0.1 at idx {perturb_idx}) max |Δ|: re = {control_re:e}, \
             im = {control_im:e}"
        );
        // Either the re or the im control must exceed ATOL — both branches
        // are separately valid discriminators (one of them will be by far
        // the strongest because both use the same gain).
        assert!(
            control_re > ATOL || control_im > ATOL,
            "negative control: a 0.1 gain perturbation moved CPU output only re={control_re}, \
             im={control_im} — both ≤ {ATOL}; the atol bound would be vacuous"
        );
    }

    /// GTCRN-16 kHz-ish shape (48 frames × 257 bins, `n_fft = 512` →
    /// `n_fft/2 + 1 = 257`). Multiple threadgroup tiles in both dimensions
    /// (257/16 = 17 groups × ragged tail, 48/16 = 3 groups) so both the
    /// bin-axis and the frame-axis boundary conditions are exercised.
    #[test]
    fn gtcrn_shape_metal_matches_cpu_or_clean_skip() {
        let Some(compute_metal) = metal_compute() else {
            println!("skip: no Metal device on this host (clean skip, never fabricated)");
            return;
        };
        let compute_cpu = Compute::cpu();

        let n_frames = 48;
        let n_bins = 257;
        let n = n_frames * n_bins;
        let mut state: u64 = 0xC0FF_EE00_1337_0042;
        let spec_re: Vec<f32> = (0..n).map(|_| splitmix_f32(&mut state, 3.0)).collect();
        let spec_im: Vec<f32> = (0..n).map(|_| splitmix_f32(&mut state, 3.0)).collect();
        // Mask range [0, 1] with a sharper distribution (heavy on the ends,
        // simulating a strongly-decided per-position mask).
        let gain: Vec<f32> = (0..n)
            .map(|_| {
                let u = splitmix_f32(&mut state, 1.0);
                if u > 0.0 { u.abs() } else { 0.0 }
            })
            .collect();

        let mut cpu_out_re = vec![0.0f32; n];
        let mut cpu_out_im = vec![0.0f32; n];
        let mut metal_out_re = vec![0.0f32; n];
        let mut metal_out_im = vec![0.0f32; n];
        compute_cpu
            .denoise_apply_mask_f32(
                &spec_re,
                &spec_im,
                &gain,
                n_frames,
                n_bins,
                &mut cpu_out_re,
                &mut cpu_out_im,
            )
            .expect("CPU arm must succeed");
        compute_metal
            .denoise_apply_mask_f32(
                &spec_re,
                &spec_im,
                &gain,
                n_frames,
                n_bins,
                &mut metal_out_re,
                &mut metal_out_im,
            )
            .expect("Metal arm must succeed");

        let d_re = max_delta(&cpu_out_re, &metal_out_re);
        let d_im = max_delta(&cpu_out_im, &metal_out_im);
        println!("gtcrn (48 fr × 257 bins) Metal vs CPU max |Δ|: re = {d_re:e}, im = {d_im:e}");
        assert!(
            d_re <= ATOL,
            "gtcrn denoise_apply_mask_f32 Metal vs CPU max |Δ|.re = {d_re} > {ATOL}"
        );
        assert!(
            d_im <= ATOL,
            "gtcrn denoise_apply_mask_f32 Metal vs CPU max |Δ|.im = {d_im} > {ATOL}"
        );
    }

    /// Ragged tail on both axes (n_frames=5, n_bins=17) — neither divides
    /// 16, so the 16×16 threadgroup dispatch must correctly bounds-check
    /// every out-of-range thread. If the `f >= d.n_bins || t >= d.n_frames`
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

        let n_frames = 5;
        let n_bins = 17;
        let n = n_frames * n_bins;
        let spec_re: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.19).sin() * 1.3).collect();
        let spec_im: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.29).cos() * 1.1).collect();
        let gain: Vec<f32> = (0..n)
            .map(|i| 0.2 + ((i as f32) * 0.11).sin().abs() * 0.7)
            .collect();

        let mut cpu_out_re = vec![0.0f32; n];
        let mut cpu_out_im = vec![0.0f32; n];
        let mut metal_out_re = vec![0.0f32; n];
        let mut metal_out_im = vec![0.0f32; n];
        compute_cpu
            .denoise_apply_mask_f32(
                &spec_re,
                &spec_im,
                &gain,
                n_frames,
                n_bins,
                &mut cpu_out_re,
                &mut cpu_out_im,
            )
            .expect("CPU arm must succeed");
        compute_metal
            .denoise_apply_mask_f32(
                &spec_re,
                &spec_im,
                &gain,
                n_frames,
                n_bins,
                &mut metal_out_re,
                &mut metal_out_im,
            )
            .expect("Metal arm must succeed");

        let d_re = max_delta(&cpu_out_re, &metal_out_re);
        let d_im = max_delta(&cpu_out_im, &metal_out_im);
        println!("ragged (5 fr × 17 bins) Metal vs CPU max |Δ|: re = {d_re:e}, im = {d_im:e}");
        assert!(
            d_re <= ATOL,
            "ragged denoise_apply_mask_f32 Metal vs CPU max |Δ|.re = {d_re} > {ATOL}"
        );
        assert!(
            d_im <= ATOL,
            "ragged denoise_apply_mask_f32 Metal vs CPU max |Δ|.im = {d_im} > {ATOL}"
        );
    }

    /// FR-EX-08 host-side validation: wrong spec_re length is
    /// `InvalidArgument` (never a silent GPU OOB or clamp).
    #[test]
    fn wrong_spec_re_length_is_invalid_argument_or_clean_skip() {
        let Some(compute_metal) = metal_compute() else {
            println!("skip: no Metal device on this host (clean skip, never fabricated)");
            return;
        };
        let n_frames = 3;
        let n_bins = 4;
        let n = n_frames * n_bins;
        let spec_re = vec![0.0f32; n - 1]; // one short
        let spec_im = vec![0.0f32; n];
        let gain = vec![1.0f32; n];
        let mut out_re = vec![0.0f32; n];
        let mut out_im = vec![0.0f32; n];
        let err = compute_metal
            .denoise_apply_mask_f32(
                &spec_re,
                &spec_im,
                &gain,
                n_frames,
                n_bins,
                &mut out_re,
                &mut out_im,
            )
            .expect_err("wrong spec_re length must be an explicit InvalidArgument (FR-EX-08)");
        assert!(
            matches!(err, VokraError::InvalidArgument(_)),
            "expected InvalidArgument for wrong spec_re length, got {err:?}",
        );
    }

    /// FR-EX-08 host-side validation: wrong spec_im length is
    /// `InvalidArgument`.
    #[test]
    fn wrong_spec_im_length_is_invalid_argument_or_clean_skip() {
        let Some(compute_metal) = metal_compute() else {
            println!("skip: no Metal device on this host (clean skip, never fabricated)");
            return;
        };
        let n_frames = 3;
        let n_bins = 4;
        let n = n_frames * n_bins;
        let spec_re = vec![0.0f32; n];
        let spec_im = vec![0.0f32; n + 1]; // one too many
        let gain = vec![1.0f32; n];
        let mut out_re = vec![0.0f32; n];
        let mut out_im = vec![0.0f32; n];
        let err = compute_metal
            .denoise_apply_mask_f32(
                &spec_re,
                &spec_im,
                &gain,
                n_frames,
                n_bins,
                &mut out_re,
                &mut out_im,
            )
            .expect_err("wrong spec_im length must be an explicit InvalidArgument (FR-EX-08)");
        assert!(
            matches!(err, VokraError::InvalidArgument(_)),
            "expected InvalidArgument for wrong spec_im length, got {err:?}",
        );
    }

    /// FR-EX-08 host-side validation: wrong gain length is `InvalidArgument`.
    #[test]
    fn wrong_gain_length_is_invalid_argument_or_clean_skip() {
        let Some(compute_metal) = metal_compute() else {
            println!("skip: no Metal device on this host (clean skip, never fabricated)");
            return;
        };
        let n_frames = 3;
        let n_bins = 4;
        let n = n_frames * n_bins;
        let spec_re = vec![0.0f32; n];
        let spec_im = vec![0.0f32; n];
        let gain = vec![1.0f32; n - 1]; // one short
        let mut out_re = vec![0.0f32; n];
        let mut out_im = vec![0.0f32; n];
        let err = compute_metal
            .denoise_apply_mask_f32(
                &spec_re,
                &spec_im,
                &gain,
                n_frames,
                n_bins,
                &mut out_re,
                &mut out_im,
            )
            .expect_err("wrong gain length must be an explicit InvalidArgument (FR-EX-08)");
        assert!(
            matches!(err, VokraError::InvalidArgument(_)),
            "expected InvalidArgument for wrong gain length, got {err:?}",
        );
    }

    /// Empty `n_frames = 0` or `n_bins = 0` is a no-op on both arms (no
    /// dispatch, no allocation panic). Mirrors the CPU op's contract.
    #[test]
    fn empty_shape_returns_ok_no_op_or_clean_skip() {
        let Some(compute_metal) = metal_compute() else {
            println!("skip: no Metal device on this host (clean skip, never fabricated)");
            return;
        };
        // n_frames = 0
        let mut out_re: Vec<f32> = Vec::new();
        let mut out_im: Vec<f32> = Vec::new();
        compute_metal
            .denoise_apply_mask_f32(&[], &[], &[], 0, 5, &mut out_re, &mut out_im)
            .expect("n_frames=0 must be a no-op");
        assert!(out_re.is_empty());
        assert!(out_im.is_empty());
        // n_bins = 0
        compute_metal
            .denoise_apply_mask_f32(&[], &[], &[], 3, 0, &mut out_re, &mut out_im)
            .expect("n_bins=0 must be a no-op");
        assert!(out_re.is_empty());
        assert!(out_im.is_empty());
    }
}
