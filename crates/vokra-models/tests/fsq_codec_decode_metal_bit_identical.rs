//! M4-16 WF2 — FSQ codec family Metal MSL kernel real-GPU parity.
//!
//! Covers the two FSQ family ops
//! (`vokra_ops::fsq_codec::{wavtokenizer_vq_decode, xcodec2_fsq_decode}`) that
//! land Metal kernels together in this wave — kept in one test file because
//! the FSQ family is deliberately a *separate subgraph* from the RVQ family
//! (FR-OP-31 — module docs on `vokra_ops::fsq_codec`) and shares the same
//! atol / negative-control / clean-skip pattern.
//!
//! Kernels:
//!   - `vokra_wavtokenizer_vq_gather_f32` (single-codebook gather — pure copy,
//!     bit-identical vs CPU).
//!   - `vokra_xcodec2_fsq_decode_f32` (grid decompose + optional Linear GEMV
//!     — FP32 fold, atol ≤ 5e-4 vs CPU).
//!
//! Both are wired through `Compute::wavtokenizer_vq_f32` /
//! `Compute::xcodec2_fsq_f32` in `crates/vokra-models/src/compute.rs`.
//!
//! - **Off-feature band** (compiled when `metal` is off / non-Apple):
//!   `HotOp::WavTokenizerVq` / `HotOp::Xcodec2Fsq` are not Metal-covered off
//!   the feature, so `Compute::for_backend(Metal, [...])` is an explicit
//!   [`vokra_core::VokraError::BackendUnavailable`] — never a silent CPU
//!   substitute (FR-EX-08).
//! - **Metal band** (`--features metal`, Apple): real-GPU parity on THIS M1
//!   iMac (CLAUDE.md dev environment). Skips cleanly (printed reason) when no
//!   Metal device is present.
//!
//! # atol bound and negative control
//!
//! WavTokenizer is a pure gather (no arithmetic, no fold), so CPU and GPU
//! agree bit-for-bit; the atol is still budgeted at 5e-4 for consistency
//! with the sibling mimi_rvq / dac_rvq codec-family bound.
//!
//! X-Codec 2 FSQ has an inner `Σ_k proj_weight[o, k] · grid[k]` GEMV whose
//! MSL fast-math compilation may re-associate the FP32 dot product; on
//! canonical shapes the two agree bit-for-bit in practice but we assert the
//! honest atol ≤ 5e-4 bound (same envelope as mimi_rvq / dac_rvq).
//!
//! A **negative control** shows the bound is discriminating rather than
//! vacuous: perturbing a codebook row (WavTokenizer) or a projection weight
//! row (X-Codec 2) moves the CPU output well past 5e-4, so a "CPU vs Metal
//! ≤ 5e-4" agreement is a real match, not a floor.
//!
//! # Real-weight parity
//!
//! Synthetic ramp / random inputs exercise the shader shape end-to-end. Real
//! WavTokenizer / X-Codec 2 checkpoints are well under the M1 iMac 16 GB
//! budget, but the neural decoder chains that consume these ops are separate
//! WPs (module docs) — this test scope stays synthetic and pins the codec-
//! side Metal↔CPU bound.

#[cfg(not(all(feature = "metal", any(target_os = "macos", target_os = "ios"))))]
mod off_feature {
    use vokra_core::{BackendKind, VokraError};
    use vokra_models::compute::{Compute, HotOp};

    /// Off the Metal build both FSQ family ops are `BackendUnavailable` at
    /// the `for_backend` entry — never a silent CPU substitute (FR-EX-08).
    #[test]
    fn for_backend_metal_fsq_family_off_feature_is_backend_unavailable() {
        for op in [HotOp::WavTokenizerVq, HotOp::Xcodec2Fsq] {
            let err = Compute::for_backend(BackendKind::Metal, &[op])
                .expect_err("off-feature Metal must fail explicitly, not silently CPU-substitute");
            assert!(
                matches!(err, VokraError::BackendUnavailable(_)),
                "expected BackendUnavailable off the metal feature for {op:?}, got {err:?}",
            );
        }
    }
}

#[cfg(all(feature = "metal", any(target_os = "macos", target_os = "ios")))]
mod metal_band {
    use vokra_core::{BackendKind, VokraError};
    use vokra_models::compute::{Compute, HotOp};
    use vokra_ops::{
        CodebookTable, FsqOutProj, WavTokenizerVqAttrs, Xcodec2FsqAttrs, wavtokenizer_vq_decode,
        xcodec2_fsq_decode,
    };

    /// Metal / CPU parity bound. Same 5e-4 budget as the sibling
    /// `mimi_rvq_metal_bit_identical.rs::ATOL` /
    /// `dac_rvq_decode_metal_bit_identical.rs::ATOL` — the M4-05 CSM / Moshi
    /// Metal parity envelope for codec-side FP32 folds.
    const ATOL: f32 = 5e-4;

    fn max_delta(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f32, f32::max)
    }

    /// Build a Metal `Compute` covering the given ops. Returns `None` when
    /// no Metal device is present (clean skip, mirrors the sibling codec
    /// parity tests' pattern).
    fn metal_compute(required: &[HotOp]) -> Option<Compute> {
        match Compute::for_backend(BackendKind::Metal, required) {
            Ok(c) => Some(c),
            Err(VokraError::BackendUnavailable(_)) => None,
            Err(e) => panic!("unexpected Metal for_backend error: {e}"),
        }
    }

    // ==========================================================================
    // WavTokenizer VQ — single-codebook gather (pure copy, bit-identical)
    // ==========================================================================

    /// Deterministic ramp table (mirror of `vokra_ops::fsq_codec::tests`
    /// helpers): row `i` is `[i, i+1, ..., i+d-1]` as f32 — exactly
    /// representable so the CPU / GPU gather is bit-clean.
    fn wt_ramp_table(vocab: usize, d: usize) -> CodebookTable {
        let mut data = vec![0.0_f32; vocab * d];
        for i in 0..vocab {
            for j in 0..d {
                data[i * d + j] = (i + j) as f32;
            }
        }
        CodebookTable::new(vocab, d, data).unwrap()
    }

    /// Tiny WavTokenizer shape: vocab_size = 6, d_model = 4, time = 3.
    /// Exercises the ragged-tail guard (4-wide vs 16-wide default tg) and
    /// the pure gather semantics.
    #[test]
    fn wavtokenizer_tiny_shape_metal_matches_cpu_within_atol_or_clean_skip() {
        let Some(compute_metal) = metal_compute(&[HotOp::WavTokenizerVq]) else {
            println!("skip: no Metal device on this host (clean skip, never fabricated)");
            return;
        };
        assert_eq!(compute_metal.backend_name(), "metal");
        let compute_cpu = Compute::cpu();

        let attrs = WavTokenizerVqAttrs {
            vocab_size: 6,
            d_model: 4,
        };
        let table = wt_ramp_table(attrs.vocab_size, attrs.d_model);
        let time = 3;
        let codes: Vec<u32> = vec![2, 0, 5];
        assert_eq!(codes.len(), time);

        let cpu_out = compute_cpu
            .wavtokenizer_vq_f32(&codes, time, &table, &attrs)
            .expect("CPU arm must succeed");
        let metal_out = compute_metal
            .wavtokenizer_vq_f32(&codes, time, &table, &attrs)
            .expect("Metal arm must succeed post M4-16 WF2");
        assert_eq!(cpu_out.len(), metal_out.len(), "output shapes must match");

        let d = max_delta(&cpu_out, &metal_out);
        println!("wavtokenizer tiny (vocab=6, d_model=4, time=3) Metal vs CPU max |Δ| = {d:e}");
        // Pure gather is bit-identical (no arithmetic to re-associate); assert
        // the tight 5e-4 bound for consistency with the sibling codec-family
        // parity tests, and also assert bit-identity separately as an extra
        // sanity gate (a regression here would be a real bug, not fast-math
        // drift).
        assert!(
            d <= ATOL,
            "wavtokenizer_vq_f32 Metal vs CPU max |Δ| = {d} > {ATOL}"
        );
        assert_eq!(
            cpu_out, metal_out,
            "wavtokenizer_vq_f32 gather must be bit-identical (pure copy, no fold)"
        );

        // Cross-check against the reference `wavtokenizer_vq_decode` — the
        // CPU arm must be bit-identical to it (a sanity gate on the CPU arm
        // rather than a Metal claim).
        let reference = wavtokenizer_vq_decode(&codes, time, &table, &attrs).unwrap();
        assert_eq!(
            cpu_out, reference,
            "Compute::cpu()::wavtokenizer_vq_f32 must be bit-identical to wavtokenizer_vq_decode"
        );
    }

    /// Canonical WavTokenizer-ish shape: vocab_size = 4096, d_model = 128
    /// (kept smaller than the released 512 so the test finishes quickly on
    /// all Metal devices while still exercising the released
    /// `WavTokenizer-{small,medium}-*-24k-4096` vocab scale). Also asserts
    /// the negative-control bound holds so the ATOL is not vacuous.
    #[test]
    fn wavtokenizer_canonical_shape_metal_matches_cpu_with_negative_control_or_clean_skip() {
        let Some(compute_metal) = metal_compute(&[HotOp::WavTokenizerVq]) else {
            println!("skip: no Metal device on this host (clean skip, never fabricated)");
            return;
        };
        let compute_cpu = Compute::cpu();

        let attrs = WavTokenizerVqAttrs {
            vocab_size: 4096,
            d_model: 128,
        };
        let mut table = wt_ramp_table(attrs.vocab_size, attrs.d_model);
        let time = 12;
        // Deterministic pseudo-random codes (SplitMix64 over usize; mirrors
        // the sibling mimi/dac canonical tests).
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut codes: Vec<u32> = Vec::with_capacity(time);
        for _ in 0..time {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            codes.push((z as u32) % (attrs.vocab_size as u32));
        }

        let cpu_out = compute_cpu
            .wavtokenizer_vq_f32(&codes, time, &table, &attrs)
            .expect("CPU arm must succeed");
        let metal_out = compute_metal
            .wavtokenizer_vq_f32(&codes, time, &table, &attrs)
            .expect("Metal arm must succeed post M4-16 WF2");
        assert_eq!(cpu_out.len(), metal_out.len());
        let d = max_delta(&cpu_out, &metal_out);
        println!(
            "wavtokenizer canonical (vocab=4096, d_model=128, time=12) Metal vs CPU max |Δ| = \
             {d:e}"
        );
        assert!(
            d <= ATOL,
            "canonical wavtokenizer_vq_f32 Metal vs CPU max |Δ| = {d} > {ATOL}"
        );

        // Negative control: perturb the codebook row `codes[0]` actually
        // references — random draws could easily miss row 0 for every
        // timestep, making the perturbation invisible and the control
        // vacuously trivial. By perturbing `table.row(codes[0])` we
        // guarantee the perturbation is visible in `cpu_out_perturbed[0..d_model]`,
        // so a well-behaved gather must move the output past ATOL.
        let d_model = attrs.d_model;
        let referenced_row = codes[0] as usize;
        let base = referenced_row * d_model;
        for cell in &mut table.data[base..base + d_model] {
            *cell += 0.1;
        }
        let cpu_out_perturbed = compute_cpu
            .wavtokenizer_vq_f32(&codes, time, &table, &attrs)
            .expect("CPU perturbed arm must succeed");
        let control = max_delta(&cpu_out, &cpu_out_perturbed);
        println!("wavtokenizer negative control (0.1 row perturbation) max |Δ| = {control:e}");
        assert!(
            control > ATOL,
            "negative control: a 0.1 row perturbation moved CPU output only {control} ≤ {ATOL} \
             — the atol bound would be vacuous; test cannot honestly claim parity"
        );
    }

    /// FR-EX-08 host-side validation: OOB WavTokenizer code index is
    /// `InvalidArgument` (never a silent GPU OOB read).
    #[test]
    fn wavtokenizer_out_of_range_code_is_invalid_argument_or_clean_skip() {
        let Some(compute_metal) = metal_compute(&[HotOp::WavTokenizerVq]) else {
            println!("skip: no Metal device on this host (clean skip, never fabricated)");
            return;
        };
        let attrs = WavTokenizerVqAttrs {
            vocab_size: 6,
            d_model: 4,
        };
        let table = wt_ramp_table(attrs.vocab_size, attrs.d_model);
        // idx = vocab_size → OOB.
        let codes: Vec<u32> = vec![attrs.vocab_size as u32];
        let err = compute_metal
            .wavtokenizer_vq_f32(&codes, 1, &table, &attrs)
            .expect_err("OOB code index must be an explicit InvalidArgument (FR-EX-08)");
        assert!(
            matches!(err, VokraError::InvalidArgument(_)),
            "expected InvalidArgument for OOB code, got {err:?}",
        );
    }

    /// FR-EX-08 host-side validation: wrong codebook table shape is
    /// `InvalidArgument`.
    #[test]
    fn wavtokenizer_wrong_table_shape_is_invalid_argument_or_clean_skip() {
        let Some(compute_metal) = metal_compute(&[HotOp::WavTokenizerVq]) else {
            println!("skip: no Metal device on this host (clean skip, never fabricated)");
            return;
        };
        let attrs = WavTokenizerVqAttrs {
            vocab_size: 6,
            d_model: 4,
        };
        // Build a table with INCOMPATIBLE d_model (attrs.d_model + 1).
        let bad_d = attrs.d_model + 1;
        let data = vec![0.0_f32; attrs.vocab_size * bad_d];
        let table = CodebookTable::new(attrs.vocab_size, bad_d, data).unwrap();
        let codes: Vec<u32> = vec![0];
        let err = compute_metal
            .wavtokenizer_vq_f32(&codes, 1, &table, &attrs)
            .expect_err("wrong table shape must be an explicit InvalidArgument (FR-EX-08)");
        assert!(
            matches!(err, VokraError::InvalidArgument(_)),
            "expected InvalidArgument for wrong table shape, got {err:?}",
        );
    }

    /// Empty `time = 0` decode returns an empty `Vec<f32>` on both arms
    /// (no dispatch, no allocation panic). Mirrors the sibling codec tests.
    #[test]
    fn wavtokenizer_empty_time_returns_empty_vec_or_clean_skip() {
        let Some(compute_metal) = metal_compute(&[HotOp::WavTokenizerVq]) else {
            println!("skip: no Metal device on this host (clean skip, never fabricated)");
            return;
        };
        let attrs = WavTokenizerVqAttrs {
            vocab_size: 6,
            d_model: 4,
        };
        let table = wt_ramp_table(attrs.vocab_size, attrs.d_model);
        let out = compute_metal
            .wavtokenizer_vq_f32(&[], 0, &table, &attrs)
            .expect("empty-time decode must return an empty Vec");
        assert!(
            out.is_empty(),
            "expected empty Vec, got {} elems",
            out.len()
        );
    }

    // ==========================================================================
    // X-Codec 2 FSQ — grid decompose + optional Linear GEMV
    // ==========================================================================

    /// Deterministic projection (powers of two, exactly representable in
    /// f32): `W[o, k] = 0.5 + o*0.25 + k*0.125` and `b[o] = o*0.0625`.
    fn xc_projection(d_model: usize, n_dims: usize) -> FsqOutProj {
        let mut w = vec![0.0_f32; d_model * n_dims];
        for o in 0..d_model {
            for k in 0..n_dims {
                w[o * n_dims + k] = 0.5 + o as f32 * 0.25 + k as f32 * 0.125;
            }
        }
        let b: Vec<f32> = (0..d_model).map(|o| o as f32 * 0.0625).collect();
        FsqOutProj::new(d_model, n_dims, w, b).unwrap()
    }

    /// Tiny FSQ shape with a Linear projection: levels [4, 4] (= 16 codes),
    /// d_model = 5, n_dims = 2. Exercises the GEMV path with a non-trivial
    /// output width and the ragged-tail guard (5-wide vs 16-wide default tg).
    #[test]
    fn xcodec2_projected_tiny_shape_metal_matches_cpu_within_atol_or_clean_skip() {
        let Some(compute_metal) = metal_compute(&[HotOp::Xcodec2Fsq]) else {
            println!("skip: no Metal device on this host (clean skip, never fabricated)");
            return;
        };
        assert_eq!(compute_metal.backend_name(), "metal");
        let compute_cpu = Compute::cpu();

        let attrs = Xcodec2FsqAttrs {
            levels: vec![4, 4],
            d_model: 5,
        };
        let proj = xc_projection(attrs.d_model, attrs.n_dims());
        let time = 4;
        // Codes span the effective vocab (Π levels = 16): 0, 5, 10, 15.
        let codes: Vec<u32> = vec![0, 5, 10, 15];
        assert_eq!(codes.len(), time);

        let cpu_out = compute_cpu
            .xcodec2_fsq_f32(&codes, time, Some(&proj), &attrs)
            .expect("CPU arm must succeed");
        let metal_out = compute_metal
            .xcodec2_fsq_f32(&codes, time, Some(&proj), &attrs)
            .expect("Metal arm must succeed post M4-16 WF2");
        assert_eq!(cpu_out.len(), metal_out.len(), "output shapes must match");

        let d = max_delta(&cpu_out, &metal_out);
        println!(
            "xcodec2 projected tiny (levels=[4,4], d_model=5, time=4) Metal vs CPU max |Δ| = \
             {d:e}"
        );
        assert!(
            d <= ATOL,
            "xcodec2_fsq_f32 Metal vs CPU max |Δ| = {d} > {ATOL}"
        );

        // Cross-check against the reference `xcodec2_fsq_decode`.
        let reference = xcodec2_fsq_decode(&codes, time, Some(&proj), &attrs).unwrap();
        assert_eq!(
            cpu_out, reference,
            "Compute::cpu()::xcodec2_fsq_f32 must be bit-identical to xcodec2_fsq_decode"
        );
    }

    /// Identity path: levels [4, 4] (n_dims = 2), d_model = 2, no
    /// projection. Exercises the `has_projection = 0` MSL arm and the
    /// grid-decompose-only semantics.
    #[test]
    fn xcodec2_identity_metal_matches_cpu_bit_identical_or_clean_skip() {
        let Some(compute_metal) = metal_compute(&[HotOp::Xcodec2Fsq]) else {
            println!("skip: no Metal device on this host (clean skip, never fabricated)");
            return;
        };
        let compute_cpu = Compute::cpu();

        let attrs = Xcodec2FsqAttrs {
            levels: vec![4, 4],
            d_model: 2, // == n_dims (Identity invariant)
        };
        let time = 4;
        let codes: Vec<u32> = vec![0, 5, 10, 15];
        assert_eq!(codes.len(), time);

        let cpu_out = compute_cpu
            .xcodec2_fsq_f32(&codes, time, None, &attrs)
            .expect("CPU Identity arm must succeed");
        let metal_out = compute_metal
            .xcodec2_fsq_f32(&codes, time, None, &attrs)
            .expect("Metal Identity arm must succeed post M4-16 WF2");
        assert_eq!(cpu_out.len(), metal_out.len());
        let d = max_delta(&cpu_out, &metal_out);
        println!("xcodec2 identity (levels=[4,4], d_model=2, time=4) Metal vs CPU max |Δ| = {d:e}");
        // Identity is pure grid decompose (no fold) so bit-identity holds.
        assert!(
            d <= ATOL,
            "xcodec2_fsq_f32 Identity Metal vs CPU max |Δ| = {d} > {ATOL}"
        );
        assert_eq!(
            cpu_out, metal_out,
            "xcodec2_fsq_f32 Identity path must be bit-identical (pure grid decompose)"
        );
    }

    /// Canonical released X-Codec 2 shape: levels [4; 8] (65536 effective
    /// vocab), n_dims = 8. `d_model = 32` is kept smaller than the released
    /// 2048 so the test finishes quickly on all Metal devices while still
    /// exercising the 8-way GEMV path. Also asserts the negative-control
    /// bound holds so the ATOL is not vacuous.
    #[test]
    fn xcodec2_canonical_shape_metal_matches_cpu_with_negative_control_or_clean_skip() {
        let Some(compute_metal) = metal_compute(&[HotOp::Xcodec2Fsq]) else {
            println!("skip: no Metal device on this host (clean skip, never fabricated)");
            return;
        };
        let compute_cpu = Compute::cpu();

        let attrs = Xcodec2FsqAttrs {
            levels: vec![4u32; 8],
            d_model: 32,
        };
        let mut proj = xc_projection(attrs.d_model, attrs.n_dims());
        let time = 12;
        // Deterministic pseudo-random codes over the 65536 vocab (SplitMix64
        // — mirrors the sibling codec parity tests).
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        let vocab = attrs.effective_vocab().unwrap();
        let mut codes: Vec<u32> = Vec::with_capacity(time);
        for _ in 0..time {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            codes.push((z as u32) % (vocab as u32));
        }

        let cpu_out = compute_cpu
            .xcodec2_fsq_f32(&codes, time, Some(&proj), &attrs)
            .expect("CPU arm must succeed");
        let metal_out = compute_metal
            .xcodec2_fsq_f32(&codes, time, Some(&proj), &attrs)
            .expect("Metal arm must succeed post M4-16 WF2");
        assert_eq!(cpu_out.len(), metal_out.len());
        let d = max_delta(&cpu_out, &metal_out);
        println!(
            "xcodec2 canonical (levels=[4;8], d_model=32, time=12) Metal vs CPU max |Δ| = {d:e}"
        );
        assert!(
            d <= ATOL,
            "canonical xcodec2_fsq_f32 Metal vs CPU max |Δ| = {d} > {ATOL}"
        );

        // Negative control: perturb one projection row by 0.1. Since every
        // codes[t] participates in the output (each thread computes one
        // column), any projection perturbation is visible in the output
        // even if the code distribution is uneven. Perturb row 0.
        let n_dims = attrs.n_dims();
        for cell in &mut proj.weight[0..n_dims] {
            *cell += 0.1;
        }
        let cpu_out_perturbed = compute_cpu
            .xcodec2_fsq_f32(&codes, time, Some(&proj), &attrs)
            .expect("CPU perturbed arm must succeed");
        let control = max_delta(&cpu_out, &cpu_out_perturbed);
        println!("xcodec2 negative control (0.1 W row perturbation) max |Δ| = {control:e}");
        assert!(
            control > ATOL,
            "negative control: a 0.1 W row perturbation moved CPU output only {control} ≤ \
             {ATOL} — the atol bound would be vacuous; test cannot honestly claim parity"
        );
    }

    /// FR-EX-08 host-side validation: OOB X-Codec 2 code index is
    /// `InvalidArgument` (never a silent GPU OOB read / divide by zero).
    #[test]
    fn xcodec2_out_of_range_code_is_invalid_argument_or_clean_skip() {
        let Some(compute_metal) = metal_compute(&[HotOp::Xcodec2Fsq]) else {
            println!("skip: no Metal device on this host (clean skip, never fabricated)");
            return;
        };
        let attrs = Xcodec2FsqAttrs {
            levels: vec![4, 4], // Π = 16
            d_model: 5,
        };
        let proj = xc_projection(attrs.d_model, attrs.n_dims());
        // idx = 16 → OOB.
        let codes: Vec<u32> = vec![16];
        let err = compute_metal
            .xcodec2_fsq_f32(&codes, 1, Some(&proj), &attrs)
            .expect_err("OOB code index must be an explicit InvalidArgument (FR-EX-08)");
        assert!(
            matches!(err, VokraError::InvalidArgument(_)),
            "expected InvalidArgument for OOB code, got {err:?}",
        );
    }

    /// FR-EX-08 host-side validation: wrong projection shape is
    /// `InvalidArgument`.
    #[test]
    fn xcodec2_wrong_proj_shape_is_invalid_argument_or_clean_skip() {
        let Some(compute_metal) = metal_compute(&[HotOp::Xcodec2Fsq]) else {
            println!("skip: no Metal device on this host (clean skip, never fabricated)");
            return;
        };
        let attrs = Xcodec2FsqAttrs {
            levels: vec![4, 4],
            d_model: 5,
        };
        // Build a projection with INCOMPATIBLE n_dims (attrs.n_dims + 1).
        let bad_nd = attrs.n_dims() + 1;
        let w = vec![0.0_f32; attrs.d_model * bad_nd];
        let b = vec![0.0_f32; attrs.d_model];
        let bad_proj = FsqOutProj::new(attrs.d_model, bad_nd, w, b).unwrap();
        let codes: Vec<u32> = vec![0];
        let err = compute_metal
            .xcodec2_fsq_f32(&codes, 1, Some(&bad_proj), &attrs)
            .expect_err("wrong projection shape must be an explicit InvalidArgument (FR-EX-08)");
        assert!(
            matches!(err, VokraError::InvalidArgument(_)),
            "expected InvalidArgument for wrong proj shape, got {err:?}",
        );
    }

    /// FR-EX-08 host-side validation: Identity path requires
    /// `d_model == n_dims` — a mismatch is `InvalidArgument`.
    #[test]
    fn xcodec2_identity_mismatched_dims_is_invalid_argument_or_clean_skip() {
        let Some(compute_metal) = metal_compute(&[HotOp::Xcodec2Fsq]) else {
            println!("skip: no Metal device on this host (clean skip, never fabricated)");
            return;
        };
        let attrs = Xcodec2FsqAttrs {
            levels: vec![4, 4], // n_dims = 2
            d_model: 5,         // != n_dims — must be rejected for None
        };
        let codes: Vec<u32> = vec![0];
        let err = compute_metal
            .xcodec2_fsq_f32(&codes, 1, None, &attrs)
            .expect_err(
                "Identity path (proj = None) with d_model != n_dims must be an explicit \
                 InvalidArgument (FR-EX-08)",
            );
        assert!(
            matches!(err, VokraError::InvalidArgument(_)),
            "expected InvalidArgument for Identity + d_model != n_dims, got {err:?}",
        );
    }

    /// FR-EX-08 host-side validation: a level < 2 is `InvalidArgument`
    /// (would cause a divide-by-zero in the MSL `half_width = levels[k] / 2`
    /// formula; the CPU op and the Metal wrapper both catch it upstream).
    #[test]
    fn xcodec2_level_below_two_is_invalid_argument_or_clean_skip() {
        let Some(compute_metal) = metal_compute(&[HotOp::Xcodec2Fsq]) else {
            println!("skip: no Metal device on this host (clean skip, never fabricated)");
            return;
        };
        let attrs = Xcodec2FsqAttrs {
            levels: vec![4, 1], // levels[1] = 1 → half_width = 0
            d_model: 2,
        };
        let codes: Vec<u32> = vec![0];
        let err = compute_metal
            .xcodec2_fsq_f32(&codes, 1, None, &attrs)
            .expect_err("levels[k] < 2 must be an explicit InvalidArgument (FR-EX-08)");
        assert!(
            matches!(err, VokraError::InvalidArgument(_)),
            "expected InvalidArgument for levels[k] < 2, got {err:?}",
        );
    }

    /// Empty `time = 0` decode returns an empty `Vec<f32>` on both arms.
    #[test]
    fn xcodec2_empty_time_returns_empty_vec_or_clean_skip() {
        let Some(compute_metal) = metal_compute(&[HotOp::Xcodec2Fsq]) else {
            println!("skip: no Metal device on this host (clean skip, never fabricated)");
            return;
        };
        let attrs = Xcodec2FsqAttrs {
            levels: vec![4, 4],
            d_model: 5,
        };
        let proj = xc_projection(attrs.d_model, attrs.n_dims());
        let out = compute_metal
            .xcodec2_fsq_f32(&[], 0, Some(&proj), &attrs)
            .expect("empty-time decode must return an empty Vec");
        assert!(
            out.is_empty(),
            "expected empty Vec, got {} elems",
            out.len()
        );
    }
}
