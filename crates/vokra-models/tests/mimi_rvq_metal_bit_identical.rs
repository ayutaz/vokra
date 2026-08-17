//! M3-06 T14 — Mimi RVQ decode Metal MSL kernel real-GPU parity
//! (`vokra_mimi_rvq_gather_fold_f32` implemented in
//! `crates/vokra-backend-metal/src/context.rs`, wired through
//! `Compute::mimi_rvq_f32` in `crates/vokra-models/src/compute.rs`).
//!
//! - **Off-feature band** (compiled when `metal` is off / non-Apple):
//!   `HotOp::MimiRvq` is not Metal-covered off the feature, so
//!   `Compute::for_backend(Metal, [MimiRvq])` is an explicit
//!   [`vokra_core::VokraError::BackendUnavailable`] — never a silent CPU
//!   substitute (FR-EX-08).
//! - **Metal band** (`--features metal`, Apple): real-GPU parity on THIS M1
//!   iMac (CLAUDE.md dev environment). Skips cleanly (printed reason) when no
//!   Metal device is present.
//!
//! # atol bound and negative control
//!
//! The MSL fold uses fast-math-compiled FP32 arithmetic which may re-associate;
//! the CPU fold in `vokra_ops::mimi_rvq::rvq_fold_core` is a strictly
//! left-to-right FP32 loop. On canonical shapes (n_codebooks = 8) the two
//! agree bit-for-bit in practice, but we assert the honest atol ≤ 5e-4 bound
//! (mirrors `mimi_metal_parity.rs::ATOL`, which is the M4-05 CSM / Moshi
//! Metal parity envelope for a codec-side FP32 fold).
//!
//! A **negative control** shows the bound is discriminating rather than
//! vacuous: perturbing one codebook row by 0.1 moves the CPU output well past
//! 5e-4, so a "CPU vs Metal ≤ 5e-4" agreement is a real match, not a floor.
//!
//! # Real-weight parity
//!
//! Synthetic ramp / random codebooks exercise the shader shape end-to-end
//! (`n_codebooks × codebook_size × d_model` gather + fold + read-back). Real
//! Mimi codebook tables (from the Kyutai HF checkpoint under CC-BY 4.0
//! attribution) are ≤ 2 GB and fit inside the M1 iMac's 16 GB budget, so a
//! real-weight parity run is technically local-safe — but the tokenizer
//! side is `gated: true` on HF and the neural chain already asserts real
//! parity in `real_mimi_roundtrip.rs`. This test scope stays synthetic.

#[cfg(not(all(feature = "metal", any(target_os = "macos", target_os = "ios"))))]
mod off_feature {
    use vokra_core::{BackendKind, VokraError};
    use vokra_models::compute::{Compute, HotOp};

    /// Off the Metal build the MimiRvq coverage arm is a `BackendUnavailable`
    /// at the `for_backend` entry — never a silent CPU substitute (FR-EX-08).
    #[test]
    fn for_backend_metal_mimi_rvq_off_feature_is_backend_unavailable() {
        let Err(err) = Compute::for_backend(BackendKind::Metal, &[HotOp::MimiRvq]) else {
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
    use vokra_ops::{CodebookTable, MimiRvqAttrs, mimi_rvq_decode};

    /// Metal / CPU parity bound for the mimi_rvq FP32 gather + fold. The MSL
    /// kernel adds the same FP32 codebook rows in the same left-to-right
    /// order as the CPU fold, so we typically see |Δ| = 0. The bound is
    /// budgeted at 5e-4 to match the M4-05 CSM / Moshi Metal parity envelope
    /// for codec-side FP32 folds (accommodates fast-math re-association).
    const ATOL: f32 = 5e-4;

    /// A tiny 3 × 4 × 5 test shape — small enough to reason about, big
    /// enough to exercise ragged-tail guards in the 2-D dispatch (16×16
    /// threadgroups vs a 5-wide inner dim + 3-tall outer dim).
    fn tiny_attrs() -> MimiRvqAttrs {
        MimiRvqAttrs {
            n_codebooks: 3,
            codebook_size: 4,
            d_model: 5,
        }
    }

    /// Deterministic ramp: table[cb] row i has values `[i+d + cb*100 for d in 0..d_model]`.
    fn make_ramp_tables(attrs: MimiRvqAttrs) -> Vec<CodebookTable> {
        let mut out = Vec::with_capacity(attrs.n_codebooks);
        for cb in 0..attrs.n_codebooks {
            let mut data = vec![0.0_f32; attrs.codebook_size * attrs.d_model];
            for i in 0..attrs.codebook_size {
                for d in 0..attrs.d_model {
                    data[i * attrs.d_model + d] = (i + d) as f32 + (cb as f32) * 100.0;
                }
            }
            out.push(CodebookTable::new(attrs.codebook_size, attrs.d_model, data).unwrap());
        }
        out
    }

    fn max_delta(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f32, f32::max)
    }

    /// Build a Metal `Compute` covering MimiRvq. Returns `None` when no Metal
    /// device is present (clean skip, mirrors the mimi_metal_parity pattern).
    fn metal_compute() -> Option<Compute> {
        match Compute::for_backend(BackendKind::Metal, &[HotOp::MimiRvq]) {
            Ok(c) => Some(c),
            Err(VokraError::BackendUnavailable(_)) => None,
            Err(e) => panic!("unexpected Metal for_backend error: {e}"),
        }
    }

    /// Tiny shape: (3 codebooks × 4 entries × 5 features, 3 timesteps). This
    /// exercises both the ragged-tail guard and the FP32 fold accumulation.
    #[test]
    fn tiny_shape_metal_matches_cpu_within_atol_or_clean_skip() {
        let Some(compute_metal) = metal_compute() else {
            println!("skip: no Metal device on this host (clean skip, never fabricated)");
            return;
        };
        assert_eq!(compute_metal.backend_name(), "metal");
        let compute_cpu = Compute::cpu();

        let attrs = tiny_attrs();
        let tables = make_ramp_tables(attrs);
        let time = 3;
        let codes: Vec<u32> = vec![0, 1, 2, 3, 2, 1, 1, 0, 3];
        assert_eq!(codes.len(), time * attrs.n_codebooks);

        let cpu_out = compute_cpu
            .mimi_rvq_f32(&codes, time, &tables, &attrs)
            .expect("CPU arm must succeed");
        let metal_out = compute_metal
            .mimi_rvq_f32(&codes, time, &tables, &attrs)
            .expect("Metal arm must succeed post M3-06 T14");
        assert_eq!(cpu_out.len(), metal_out.len(), "output shapes must match");

        let d = max_delta(&cpu_out, &metal_out);
        println!("tiny (3x4x5, time=3) Metal vs CPU max |Δ| = {d:e}");
        assert!(
            d <= ATOL,
            "mimi_rvq_f32 Metal vs CPU max |Δ| = {d} > {ATOL}"
        );

        // Also cross-check against the reference `mimi_rvq_decode` — the
        // CPU arm must be bit-identical to it (this is a sanity gate on the
        // CPU arm rather than a Metal claim).
        let reference = mimi_rvq_decode(&codes, time, &tables, &attrs).unwrap();
        assert_eq!(
            cpu_out, reference,
            "Compute::cpu()::mimi_rvq_f32 must be bit-identical to mimi_rvq_decode"
        );
    }

    /// Canonical Mimi-ish shape: n_codebooks = 8, codebook_size = 32 (kept
    /// small so the test finishes quickly on all Metal devices while still
    /// exercising 8-way folds and larger d_model), d_model = 64. Also
    /// asserts the negative-control bound holds so the ATOL is not vacuous.
    #[test]
    fn canonical_shape_metal_matches_cpu_with_negative_control_or_clean_skip() {
        let Some(compute_metal) = metal_compute() else {
            println!("skip: no Metal device on this host (clean skip, never fabricated)");
            return;
        };
        let compute_cpu = Compute::cpu();

        let attrs = MimiRvqAttrs {
            n_codebooks: 8,
            codebook_size: 32,
            d_model: 64,
        };
        let mut tables = make_ramp_tables(attrs);
        let time = 12;
        // Deterministic pseudo-random codes (avoid a stdlib RNG dep — SplitMix64
        // over usize is enough for a spread).
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut codes: Vec<u32> = Vec::with_capacity(time * attrs.n_codebooks);
        for _ in 0..(time * attrs.n_codebooks) {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            codes.push((z as u32) % (attrs.codebook_size as u32));
        }

        let cpu_out = compute_cpu
            .mimi_rvq_f32(&codes, time, &tables, &attrs)
            .expect("CPU arm must succeed");
        let metal_out = compute_metal
            .mimi_rvq_f32(&codes, time, &tables, &attrs)
            .expect("Metal arm must succeed post M3-06 T14");
        assert_eq!(cpu_out.len(), metal_out.len());
        let d = max_delta(&cpu_out, &metal_out);
        println!("canonical (8x32x64, time=12) Metal vs CPU max |Δ| = {d:e}");
        assert!(
            d <= ATOL,
            "canonical mimi_rvq_f32 Metal vs CPU max |Δ| = {d} > {ATOL}"
        );

        // Negative control: perturb one codebook row by 0.1. Choose the row
        // that `codes[0]` (timestep 0, codebook 0) actually references — a
        // random-index draw could easily skip row 0 for every timestep,
        // which would leave the perturbation invisible and make the control
        // vacuously trivial. By perturbing `tables[0].row(codes[0])` we
        // guarantee the perturbation is visible in
        // `cpu_out_perturbed[0..d_model]`, so a well-behaved fold must move
        // the output past ATOL. This proves the CPU-vs-Metal ≤ ATOL
        // agreement above is a real match, not a floor any two outputs
        // would satisfy.
        let d_model = attrs.d_model;
        let referenced_row = codes[0] as usize;
        let cb0_data = &mut tables[0].data;
        let base = referenced_row * d_model;
        for cell in &mut cb0_data[base..base + d_model] {
            *cell += 0.1;
        }
        let cpu_out_perturbed = compute_cpu
            .mimi_rvq_f32(&codes, time, &tables, &attrs)
            .expect("CPU perturbed arm must succeed");
        let control = max_delta(&cpu_out, &cpu_out_perturbed);
        println!("negative control (0.1 codebook perturbation) max |Δ| = {control:e}");
        assert!(
            control > ATOL,
            "negative control: a 0.1 codebook perturbation moved CPU output only {control} ≤ {ATOL} \
             — the atol bound would be vacuous; test cannot honestly claim parity"
        );
    }

    /// FR-EX-08 host-side validation: OOB code index is `InvalidArgument`
    /// (never a silent GPU OOB read).
    #[test]
    fn out_of_range_code_is_invalid_argument_or_clean_skip() {
        let Some(compute_metal) = metal_compute() else {
            println!("skip: no Metal device on this host (clean skip, never fabricated)");
            return;
        };
        let attrs = tiny_attrs();
        let tables = make_ramp_tables(attrs);
        // idx = codebook_size → OOB. The kernel would silently read past the
        // codebook row without host-side validation; FR-EX-08 catches it
        // upstream of the dispatch.
        let mut codes: Vec<u32> = vec![0; attrs.n_codebooks];
        codes[1] = attrs.codebook_size as u32;
        let err = compute_metal
            .mimi_rvq_f32(&codes, 1, &tables, &attrs)
            .expect_err("OOB code index must be an explicit InvalidArgument (FR-EX-08)");
        assert!(
            matches!(err, VokraError::InvalidArgument(_)),
            "expected InvalidArgument for OOB code, got {err:?}",
        );
    }

    /// FR-EX-08 host-side validation: wrong codebook table count is
    /// `InvalidArgument`.
    #[test]
    fn wrong_table_count_is_invalid_argument_or_clean_skip() {
        let Some(compute_metal) = metal_compute() else {
            println!("skip: no Metal device on this host (clean skip, never fabricated)");
            return;
        };
        let attrs = tiny_attrs();
        let mut tables = make_ramp_tables(attrs);
        tables.pop(); // n_codebooks - 1 tables now
        let codes: Vec<u32> = vec![0; attrs.n_codebooks];
        let err = compute_metal
            .mimi_rvq_f32(&codes, 1, &tables, &attrs)
            .expect_err("wrong table count must be an explicit InvalidArgument (FR-EX-08)");
        assert!(
            matches!(err, VokraError::InvalidArgument(_)),
            "expected InvalidArgument for wrong table count, got {err:?}",
        );
    }

    /// FR-EX-08 host-side validation: wrong per-table shape is
    /// `InvalidArgument`.
    #[test]
    fn wrong_table_shape_is_invalid_argument_or_clean_skip() {
        let Some(compute_metal) = metal_compute() else {
            println!("skip: no Metal device on this host (clean skip, never fabricated)");
            return;
        };
        let attrs = tiny_attrs();
        // Build tables with an INCOMPATIBLE inner shape — d_model = attrs.d_model + 1.
        let mut tables = Vec::with_capacity(attrs.n_codebooks);
        for _ in 0..attrs.n_codebooks {
            let data = vec![0.0_f32; attrs.codebook_size * (attrs.d_model + 1)];
            tables.push(CodebookTable::new(attrs.codebook_size, attrs.d_model + 1, data).unwrap());
        }
        let codes: Vec<u32> = vec![0; attrs.n_codebooks];
        let err = compute_metal
            .mimi_rvq_f32(&codes, 1, &tables, &attrs)
            .expect_err("wrong table shape must be an explicit InvalidArgument (FR-EX-08)");
        assert!(
            matches!(err, VokraError::InvalidArgument(_)),
            "expected InvalidArgument for wrong table shape, got {err:?}",
        );
    }

    /// Empty `time = 0` decode returns an empty `Vec<f32>` on both arms
    /// (no dispatch, no allocation panic).
    #[test]
    fn empty_time_returns_empty_vec_or_clean_skip() {
        let Some(compute_metal) = metal_compute() else {
            println!("skip: no Metal device on this host (clean skip, never fabricated)");
            return;
        };
        let attrs = tiny_attrs();
        let tables = make_ramp_tables(attrs);
        let out = compute_metal
            .mimi_rvq_f32(&[], 0, &tables, &attrs)
            .expect("empty-time decode must return an empty Vec");
        assert!(
            out.is_empty(),
            "expected empty Vec, got {} elems",
            out.len()
        );
    }
}
