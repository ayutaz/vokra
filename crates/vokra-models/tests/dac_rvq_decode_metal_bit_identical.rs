//! M4-04 WF2 — DAC RVQ decode Metal MSL kernel real-GPU parity
//! (`vokra_dac_rvq_gather_project_fold_f32` implemented in
//! `crates/vokra-backend-metal/src/context.rs`, wired through
//! `Compute::dac_rvq_f32` in `crates/vokra-models/src/compute.rs`).
//!
//! - **Off-feature band** (compiled when `metal` is off / non-Apple):
//!   `HotOp::DacRvq` is not Metal-covered off the feature, so
//!   `Compute::for_backend(Metal, [DacRvq])` is an explicit
//!   [`vokra_core::VokraError::BackendUnavailable`] — never a silent CPU
//!   substitute (FR-EX-08).
//! - **Metal band** (`--features metal`, Apple): real-GPU parity on THIS M1
//!   iMac (CLAUDE.md dev environment). Skips cleanly (printed reason) when no
//!   Metal device is present.
//!
//! # atol bound and negative control
//!
//! The MSL fold does an FP32 gather + per-quantizer factorized GEMV + bias
//! and per-quantizer residual sum. Fast-math compilation may re-associate the
//! inner `Σ_c W[o, c] · low[c]` dot product; the CPU fold in
//! `vokra_ops::dac_rvq::dac_rvq_decode` is a strictly left-to-right FP32 loop.
//! On canonical shapes the two agree bit-for-bit in practice, but we assert the
//! honest atol ≤ 5e-4 bound — the same FP32 GEMV-scale bound used by the
//! sibling Mimi kernel parity test (`mimi_rvq_metal_bit_identical.rs::ATOL`),
//! which itself mirrors the M4-05 CSM / Moshi Metal parity envelope.
//!
//! A **negative control** shows the bound is discriminating rather than
//! vacuous: perturbing one codebook row that `codes[0]` actually references
//! moves the CPU output well past 5e-4, so a "CPU vs Metal ≤ 5e-4" agreement
//! is a real match, not a floor.
//!
//! # Real-weight parity
//!
//! Synthetic ramp / random codebooks exercise the shader shape end-to-end
//! (`n_codebooks × codebook_size × codebook_dim`,
//! `n_codebooks × d_model × codebook_dim` weights,
//! `n_codebooks × d_model` biases). Real DAC codebook / projection tables
//! (from the descriptinc/descript-audio-codec MIT-licensed `weights_24khz.pth`
//! checkpoint) are well under the M1 iMac 16 GB budget so a real-weight parity
//! run is technically local-safe; the neural feature→PCM decoder chain that
//! consumes this op is a separate WP (ADR M4-04 §D-g), so this test scope
//! stays synthetic and pins the codec-side FP32 fold's Metal↔CPU bound.

#[cfg(not(all(feature = "metal", any(target_os = "macos", target_os = "ios"))))]
mod off_feature {
    use vokra_core::{BackendKind, VokraError};
    use vokra_models::compute::{Compute, HotOp};

    /// Off the Metal build the DacRvq coverage arm is a `BackendUnavailable`
    /// at the `for_backend` entry — never a silent CPU substitute (FR-EX-08).
    #[test]
    fn for_backend_metal_dac_rvq_off_feature_is_backend_unavailable() {
        let Err(err) = Compute::for_backend(BackendKind::Metal, &[HotOp::DacRvq]) else {
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
    use vokra_ops::{CodebookTable, DacOutProj, DacRvqAttrs, dac_rvq_decode};

    /// Metal / CPU parity bound for the dac_rvq factorized FP32 fold. Same
    /// budget as the sibling `mimi_rvq_metal_bit_identical.rs::ATOL` (5e-4)
    /// which mirrors the M4-05 CSM / Moshi Metal parity envelope for codec-
    /// side FP32 folds (accommodates fast-math re-association of the inner
    /// `Σ_c W[o, c] · low[c]` dot product; the outer `Σ_cb` residual sum
    /// order is preserved between CPU and MSL).
    const ATOL: f32 = 5e-4;

    /// A tiny 3 × 4 × 2 → 5 test shape — small enough to reason about, big
    /// enough to exercise ragged-tail guards in the 2-D dispatch (16×16
    /// threadgroups vs a 5-wide inner dim + 3-tall outer dim) and to have a
    /// non-trivial factorized inner dim (codebook_dim=2).
    fn tiny_attrs() -> DacRvqAttrs {
        DacRvqAttrs {
            n_codebooks: 3,
            codebook_size: 4,
            codebook_dim: 2,
            d_model: 5,
        }
    }

    /// Deterministic low-dim ramp codebooks: row `i` of codebook `cb` has
    /// values `[(i+d) + cb*10 for d in 0..codebook_dim]` (mirror of
    /// `dac_rvq.rs::make_low_tables` helper style — exactly representable in
    /// f32 so the CPU hand fold is bit-clean).
    fn make_low_tables(attrs: DacRvqAttrs) -> Vec<CodebookTable> {
        let mut tables = Vec::with_capacity(attrs.n_codebooks);
        for cb in 0..attrs.n_codebooks {
            let mut data = vec![0.0_f32; attrs.codebook_size * attrs.codebook_dim];
            for i in 0..attrs.codebook_size {
                for d in 0..attrs.codebook_dim {
                    data[i * attrs.codebook_dim + d] = (i + d) as f32 + (cb as f32) * 10.0;
                }
            }
            tables.push(CodebookTable::new(attrs.codebook_size, attrs.codebook_dim, data).unwrap());
        }
        tables
    }

    /// Deterministic projections (powers of two, exactly representable in
    /// f32): `W_cb[o, c] = 0.5 + o*0.25 + c*0.125 + cb` and
    /// `b_cb[o] = o*0.0625 - cb*0.5`.
    fn make_projs(attrs: DacRvqAttrs) -> Vec<DacOutProj> {
        let mut projs = Vec::with_capacity(attrs.n_codebooks);
        for cb in 0..attrs.n_codebooks {
            let mut w = vec![0.0_f32; attrs.d_model * attrs.codebook_dim];
            for o in 0..attrs.d_model {
                for c in 0..attrs.codebook_dim {
                    w[o * attrs.codebook_dim + c] =
                        0.5 + o as f32 * 0.25 + c as f32 * 0.125 + cb as f32;
                }
            }
            let b: Vec<f32> = (0..attrs.d_model)
                .map(|o| o as f32 * 0.0625 - cb as f32 * 0.5)
                .collect();
            projs.push(DacOutProj::new(attrs.d_model, attrs.codebook_dim, w, b).unwrap());
        }
        projs
    }

    fn max_delta(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f32, f32::max)
    }

    /// Build a Metal `Compute` covering DacRvq. Returns `None` when no Metal
    /// device is present (clean skip, mirrors the mimi_rvq_metal_parity
    /// pattern).
    fn metal_compute() -> Option<Compute> {
        match Compute::for_backend(BackendKind::Metal, &[HotOp::DacRvq]) {
            Ok(c) => Some(c),
            Err(VokraError::BackendUnavailable(_)) => None,
            Err(e) => panic!("unexpected Metal for_backend error: {e}"),
        }
    }

    /// Tiny shape: 3 codebooks × 4 entries × codebook_dim 2 → d_model 5, time
    /// 3. Exercises the 2D dispatch ragged tail and the factorized-projection
    /// fold semantics.
    #[test]
    fn tiny_shape_metal_matches_cpu_within_atol_or_clean_skip() {
        let Some(compute_metal) = metal_compute() else {
            println!("skip: no Metal device on this host (clean skip, never fabricated)");
            return;
        };
        assert_eq!(compute_metal.backend_name(), "metal");
        let compute_cpu = Compute::cpu();

        let attrs = tiny_attrs();
        let tables = make_low_tables(attrs);
        let projs = make_projs(attrs);
        let time = 3;
        // Same pattern as vokra-ops dac_rvq::tests: 3 timesteps × 3 codebooks
        // = 9 indices spanning the codebook_size range.
        let codes: Vec<u32> = vec![0, 1, 2, 3, 2, 1, 1, 0, 3];
        assert_eq!(codes.len(), time * attrs.n_codebooks);

        let cpu_out = compute_cpu
            .dac_rvq_f32(&codes, time, &tables, &projs, &attrs)
            .expect("CPU arm must succeed");
        let metal_out = compute_metal
            .dac_rvq_f32(&codes, time, &tables, &projs, &attrs)
            .expect("Metal arm must succeed post M4-04 WF2");
        assert_eq!(cpu_out.len(), metal_out.len(), "output shapes must match");

        let d = max_delta(&cpu_out, &metal_out);
        println!("tiny (3x4x2->5, time=3) Metal vs CPU max |Δ| = {d:e}");
        assert!(d <= ATOL, "dac_rvq_f32 Metal vs CPU max |Δ| = {d} > {ATOL}");

        // Also cross-check against the reference `dac_rvq_decode` — the CPU
        // arm must be bit-identical to it (a sanity gate on the CPU arm
        // rather than a Metal claim).
        let reference = dac_rvq_decode(&codes, time, &tables, &projs, &attrs).unwrap();
        assert_eq!(
            cpu_out, reference,
            "Compute::cpu()::dac_rvq_f32 must be bit-identical to dac_rvq_decode"
        );
    }

    /// Canonical DAC-ish shape: n_codebooks = 8 (kept smaller than the 24 kHz
    /// variant's 32 so the test finishes quickly on every Metal device while
    /// still exercising an 8-way outer fold), codebook_size = 32,
    /// codebook_dim = 8 (the canonical DAC factorized inner dim), d_model =
    /// 64. Also asserts the negative-control bound holds so the ATOL is not
    /// vacuous.
    #[test]
    fn canonical_shape_metal_matches_cpu_with_negative_control_or_clean_skip() {
        let Some(compute_metal) = metal_compute() else {
            println!("skip: no Metal device on this host (clean skip, never fabricated)");
            return;
        };
        let compute_cpu = Compute::cpu();

        let attrs = DacRvqAttrs {
            n_codebooks: 8,
            codebook_size: 32,
            codebook_dim: 8,
            d_model: 64,
        };
        let mut tables = make_low_tables(attrs);
        let projs = make_projs(attrs);
        let time = 12;
        // Deterministic pseudo-random codes (avoid a stdlib RNG dep —
        // SplitMix64 over usize is enough for spread; mirrors the sibling
        // canonical-shape Mimi test).
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
            .dac_rvq_f32(&codes, time, &tables, &projs, &attrs)
            .expect("CPU arm must succeed");
        let metal_out = compute_metal
            .dac_rvq_f32(&codes, time, &tables, &projs, &attrs)
            .expect("Metal arm must succeed post M4-04 WF2");
        assert_eq!(cpu_out.len(), metal_out.len());
        let d = max_delta(&cpu_out, &metal_out);
        println!("canonical (8x32x8->64, time=12) Metal vs CPU max |Δ| = {d:e}");
        assert!(
            d <= ATOL,
            "canonical dac_rvq_f32 Metal vs CPU max |Δ| = {d} > {ATOL}"
        );

        // Negative control: perturb one codebook row by 0.1. Perturb the row
        // that `codes[0]` (timestep 0, codebook 0) actually references — a
        // random-index draw could easily skip row 0 for every timestep,
        // making the perturbation invisible and the control vacuous. Because
        // DAC then projects `low_row · W_cb + b` before summing, the
        // perturbation is amplified by the W_cb row norms, so a well-behaved
        // fold must move the output well past ATOL. This proves the CPU-vs-
        // Metal ≤ ATOL agreement above is a real match, not a floor any two
        // outputs would satisfy.
        let referenced_row = codes[0] as usize;
        let codebook_dim = attrs.codebook_dim;
        let cb0_data = &mut tables[0].data;
        let base = referenced_row * codebook_dim;
        for cell in &mut cb0_data[base..base + codebook_dim] {
            *cell += 0.1;
        }
        let cpu_out_perturbed = compute_cpu
            .dac_rvq_f32(&codes, time, &tables, &projs, &attrs)
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
        let tables = make_low_tables(attrs);
        let projs = make_projs(attrs);
        // idx = codebook_size → OOB. The kernel would silently read past the
        // codebook row without host-side validation; FR-EX-08 catches it
        // upstream of the dispatch.
        let mut codes: Vec<u32> = vec![0; attrs.n_codebooks];
        codes[1] = attrs.codebook_size as u32;
        let err = compute_metal
            .dac_rvq_f32(&codes, 1, &tables, &projs, &attrs)
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
        let mut tables = make_low_tables(attrs);
        tables.pop(); // n_codebooks - 1 tables now
        let projs = make_projs(attrs);
        let codes: Vec<u32> = vec![0; attrs.n_codebooks];
        let err = compute_metal
            .dac_rvq_f32(&codes, 1, &tables, &projs, &attrs)
            .expect_err("wrong table count must be an explicit InvalidArgument (FR-EX-08)");
        assert!(
            matches!(err, VokraError::InvalidArgument(_)),
            "expected InvalidArgument for wrong table count, got {err:?}",
        );
    }

    /// FR-EX-08 host-side validation: wrong per-table row width is
    /// `InvalidArgument` (DAC-specific — the low-dim table's row width must
    /// be `codebook_dim`, not `d_model`; a plain "table row = d_model"
    /// mistake — the mimi_rvq shape — is a shape error here).
    #[test]
    fn wrong_table_shape_is_invalid_argument_or_clean_skip() {
        let Some(compute_metal) = metal_compute() else {
            println!("skip: no Metal device on this host (clean skip, never fabricated)");
            return;
        };
        let attrs = tiny_attrs();
        let projs = make_projs(attrs);
        // Build tables with INCOMPATIBLE row width — d_model (5) instead of
        // codebook_dim (2). This is the exact "wrong-shape" error DAC's
        // factorized design catches: for a mimi-style plain-table op the
        // row width would be d_model, so the shape check must reject it here.
        let mut tables = Vec::with_capacity(attrs.n_codebooks);
        for _ in 0..attrs.n_codebooks {
            let data = vec![0.0_f32; attrs.codebook_size * attrs.d_model];
            tables.push(CodebookTable::new(attrs.codebook_size, attrs.d_model, data).unwrap());
        }
        let codes: Vec<u32> = vec![0; attrs.n_codebooks];
        let err = compute_metal
            .dac_rvq_f32(&codes, 1, &tables, &projs, &attrs)
            .expect_err(
                "wrong table row width (d_model instead of codebook_dim) must be an explicit \
                 InvalidArgument (FR-EX-08)",
            );
        assert!(
            matches!(err, VokraError::InvalidArgument(_)),
            "expected InvalidArgument for wrong table shape, got {err:?}",
        );
    }

    /// FR-EX-08 host-side validation: wrong out_projs count is
    /// `InvalidArgument`. New coverage vs the mimi kernel test — DAC's
    /// per-quantizer projection introduces a second per-codebook operand
    /// array with its own count check.
    #[test]
    fn wrong_proj_count_is_invalid_argument_or_clean_skip() {
        let Some(compute_metal) = metal_compute() else {
            println!("skip: no Metal device on this host (clean skip, never fabricated)");
            return;
        };
        let attrs = tiny_attrs();
        let tables = make_low_tables(attrs);
        let mut projs = make_projs(attrs);
        projs.pop(); // n_codebooks - 1 projections now
        let codes: Vec<u32> = vec![0; attrs.n_codebooks];
        let err = compute_metal
            .dac_rvq_f32(&codes, 1, &tables, &projs, &attrs)
            .expect_err("wrong out_projs count must be an explicit InvalidArgument (FR-EX-08)");
        assert!(
            matches!(err, VokraError::InvalidArgument(_)),
            "expected InvalidArgument for wrong out_projs count, got {err:?}",
        );
    }

    /// FR-EX-08 host-side validation: wrong per-projection shape is
    /// `InvalidArgument`. New coverage vs the mimi kernel test — DAC's
    /// projection introduces an [d_model, codebook_dim] weight and [d_model]
    /// bias whose shapes must match attrs.
    #[test]
    fn wrong_proj_shape_is_invalid_argument_or_clean_skip() {
        let Some(compute_metal) = metal_compute() else {
            println!("skip: no Metal device on this host (clean skip, never fabricated)");
            return;
        };
        let attrs = tiny_attrs();
        let tables = make_low_tables(attrs);
        // Build projections with INCOMPATIBLE d_model (attrs.d_model + 1).
        let mut projs = Vec::with_capacity(attrs.n_codebooks);
        let bad_d = attrs.d_model + 1;
        for _ in 0..attrs.n_codebooks {
            let w = vec![0.0_f32; bad_d * attrs.codebook_dim];
            let b = vec![0.0_f32; bad_d];
            projs.push(DacOutProj::new(bad_d, attrs.codebook_dim, w, b).unwrap());
        }
        let codes: Vec<u32> = vec![0; attrs.n_codebooks];
        let err = compute_metal
            .dac_rvq_f32(&codes, 1, &tables, &projs, &attrs)
            .expect_err("wrong out_projs shape must be an explicit InvalidArgument (FR-EX-08)");
        assert!(
            matches!(err, VokraError::InvalidArgument(_)),
            "expected InvalidArgument for wrong out_projs shape, got {err:?}",
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
        let tables = make_low_tables(attrs);
        let projs = make_projs(attrs);
        let out = compute_metal
            .dac_rvq_f32(&[], 0, &tables, &projs, &attrs)
            .expect("empty-time decode must return an empty Vec");
        assert!(
            out.is_empty(),
            "expected empty Vec, got {} elems",
            out.len()
        );
    }
}
