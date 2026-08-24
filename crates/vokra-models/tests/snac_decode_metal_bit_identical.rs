//! Vocoder Metal wave WF5 — SNAC 3-stage hierarchical RVQ decode Metal MSL
//! kernel real-GPU parity (`vokra_snac_decode_f32` implemented in
//! `crates/vokra-backend-metal/src/context.rs`, wired through
//! `Compute::snac_decode_f32` in `crates/vokra-models/src/compute.rs`).
//!
//! - **Off-feature band** (compiled when `metal` is off / non-Apple):
//!   `HotOp::SnacDecode` is not Metal-covered off the feature, so
//!   `Compute::for_backend(Metal, [SnacDecode])` is an explicit
//!   [`vokra_core::VokraError::BackendUnavailable`] — never a silent CPU
//!   substitute (FR-EX-08).
//! - **Metal band** (`--features metal`, Apple): real-GPU parity on THIS M1
//!   iMac (CLAUDE.md dev environment). Skips cleanly (printed reason) when no
//!   Metal device is present.
//!
//! # atol bound and negative control
//!
//! The MSL fold does a per-stage FP32 gather + factorized GEMV + bias +
//! temporal-upsample reindex + outer stage residual sum. Fast-math
//! compilation may re-associate the inner `Σ_c W[o, c] · low[c]` GEMV; the
//! CPU fold in `vokra_ops::snac_decode::SnacDecoder::decode` is a strictly
//! left-to-right FP32 loop. On canonical shapes the two agree bit-for-bit in
//! practice, but we assert the honest atol ≤ 5e-4 bound — the same FP32
//! GEMV-scale bound used by the sibling Mimi / DAC / FSQ / snake_activation
//! kernel parity tests, which itself mirrors the M4-05 CSM / Moshi Metal
//! parity envelope.
//!
//! A **negative control** shows the bound is discriminating rather than
//! vacuous: perturbing one codebook row that stage 0's first code actually
//! references moves the CPU output well past 5e-4 (the DAC-style per-stage
//! projection then amplifies the perturbation by the W_s row norms), so a
//! "CPU vs Metal ≤ 5e-4" agreement is a real match, not a floor.
//!
//! # Real-weight parity
//!
//! Synthetic ramp / random codebooks exercise the shader shape end-to-end
//! (3 stages × `codebook_size × codebook_dim` factorized codebooks,
//! 3 stages × `d_model × codebook_dim` weights, 3 stages × `d_model`
//! biases, per-stage codes concatenated). Real SNAC 24 kHz codebook /
//! projection tables (from the hubertsiuzdak/snac MIT+Apache-2.0
//! `snac_24khz.pth` checkpoint) are well under the M1 iMac 16 GB budget so
//! a real-weight parity run is technically local-safe; the neural
//! feature→PCM decoder chain that consumes this op is a separate WP, so
//! this test scope stays synthetic and pins the codec-side FP32 fold's
//! Metal↔CPU bound.

#[cfg(not(all(feature = "metal", any(target_os = "macos", target_os = "ios"))))]
mod off_feature {
    use vokra_core::{BackendKind, VokraError};
    use vokra_models::compute::{Compute, HotOp};

    /// Off the Metal build the SnacDecode coverage arm is a
    /// `BackendUnavailable` at the `for_backend` entry — never a silent CPU
    /// substitute (FR-EX-08).
    #[test]
    fn for_backend_metal_snac_decode_off_feature_is_backend_unavailable() {
        let Err(err) = Compute::for_backend(BackendKind::Metal, &[HotOp::SnacDecode]) else {
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
    use vokra_ops::{CodebookTable, DacOutProj, SnacConfig, SnacDecoder, SnacWeights};

    /// Metal / CPU parity bound for the SNAC 3-stage factorized FP32 fold.
    /// Same budget as the sibling `dac_rvq_decode_metal_bit_identical.rs
    /// ::ATOL` (5e-4) — the M4-05 CSM / Moshi Metal parity envelope for
    /// codec-side FP32 folds (accommodates fast-math re-association of the
    /// inner `Σ_c W[o, c] · low[c]` dot product; the outer per-stage
    /// residual sum order is preserved between CPU and MSL; the temporal
    /// upsample `t_stage = t_out / stride` is exact integer division so
    /// adds no numeric slack).
    const ATOL: f32 = 5e-4;

    const CB_SIZE: usize = 4;
    const CB_DIM: usize = 2;
    const D_MODEL: usize = 5;

    /// Deterministic low-dim ramp codebook for stage `s`: row `i` has values
    /// `[(i + d) + s * 10 for d in 0..CB_DIM]`. Distinct per stage so the
    /// residual sum picks up cross-stage differences (mirror of the vokra-ops
    /// `snac_decode::tests` helper style — exactly representable in f32 so
    /// the CPU hand fold is bit-clean).
    fn make_codebook(stage: usize) -> CodebookTable {
        let mut data = vec![0.0_f32; CB_SIZE * CB_DIM];
        for i in 0..CB_SIZE {
            for d in 0..CB_DIM {
                data[i * CB_DIM + d] = (i + d) as f32 + (stage as f32) * 10.0;
            }
        }
        CodebookTable::new(CB_SIZE, CB_DIM, data).unwrap()
    }

    /// Deterministic per-stage projection: exactly representable (powers-of-
    /// two + integer coefficients) so hand folds stay bit-clean.
    /// `W_s[o, c] = 0.5 + o*0.25 + c*0.125 + s`, `b_s[o] = o*0.0625 - s*0.5`.
    fn make_proj(stage: usize) -> DacOutProj {
        let mut w = vec![0.0_f32; D_MODEL * CB_DIM];
        for o in 0..D_MODEL {
            for c in 0..CB_DIM {
                w[o * CB_DIM + c] = 0.5 + o as f32 * 0.25 + c as f32 * 0.125 + stage as f32;
            }
        }
        let b: Vec<f32> = (0..D_MODEL)
            .map(|o| o as f32 * 0.0625 - stage as f32 * 0.5)
            .collect();
        DacOutProj::new(D_MODEL, CB_DIM, w, b).unwrap()
    }

    fn make_codebooks_tiny() -> [CodebookTable; 3] {
        [make_codebook(0), make_codebook(1), make_codebook(2)]
    }

    fn make_projs_tiny() -> [DacOutProj; 3] {
        [make_proj(0), make_proj(1), make_proj(2)]
    }

    /// SNAC 24 kHz canonical vq_strides `[4, 2, 1]`. Exercises the temporal-
    /// upsample logic: stage 0 broadcasts 1 code to 4 timesteps, stage 1
    /// broadcasts 1 code to 2 timesteps, stage 2 runs at full rate. With
    /// `codes[0].len()=1 / codes[1].len()=2 / codes[2].len()=4`, every
    /// stage expands to T=4 (the "co-aligned base frames" invariant).
    fn tiny_config() -> SnacConfig {
        SnacConfig {
            sample_rate: 24_000,
            vq_strides: [4, 2, 1, 0],
            n_stages: 3,
        }
    }

    fn max_delta(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f32, f32::max)
    }

    /// Build a Metal `Compute` covering SnacDecode. Returns `None` when no
    /// Metal device is present (clean skip, mirrors the sibling metal_parity
    /// pattern).
    fn metal_compute() -> Option<Compute> {
        match Compute::for_backend(BackendKind::Metal, &[HotOp::SnacDecode]) {
            Ok(c) => Some(c),
            Err(VokraError::BackendUnavailable(_)) => None,
            Err(e) => panic!("unexpected Metal for_backend error: {e}"),
        }
    }

    /// Tiny shape with the canonical SNAC 24 kHz strides `[4, 2, 1]` and
    /// 3 codebooks × 4 entries × codebook_dim 2 → d_model 5, giving
    /// `t_expanded = 4`. Exercises both the 2D dispatch ragged tail (5-wide
    /// inner dim + 4-tall outer dim vs 16×16 threadgroups) AND the SNAC-
    /// specific per-stage `t_stage = t_out / stride` reindex.
    #[test]
    fn tiny_shape_metal_matches_cpu_within_atol_or_clean_skip() {
        let Some(compute_metal) = metal_compute() else {
            println!("skip: no Metal device on this host (clean skip, never fabricated)");
            return;
        };
        assert_eq!(compute_metal.backend_name(), "metal");
        let compute_cpu = Compute::cpu();

        let cfg = tiny_config();
        let codebooks = make_codebooks_tiny();
        let out_projs = make_projs_tiny();
        // Co-aligned base frames: 1*4 == 2*2 == 4*1 == 4.
        let codes: [Vec<u32>; 3] = [vec![1], vec![2, 3], vec![0, 1, 2, 3]];

        let cpu_out = compute_cpu
            .snac_decode_f32(&codes, cfg, &codebooks, &out_projs)
            .expect("CPU arm must succeed");
        let metal_out = compute_metal
            .snac_decode_f32(&codes, cfg, &codebooks, &out_projs)
            .expect("Metal arm must succeed post Vocoder Metal wave WF5");
        assert_eq!(cpu_out.len(), metal_out.len(), "output shapes must match");
        assert_eq!(
            cpu_out.len(),
            4 * D_MODEL,
            "canonical strides [4,2,1] with codes lens [1,2,4] must give t_expanded=4"
        );

        let d = max_delta(&cpu_out, &metal_out);
        println!("tiny (3 stages [4,2,1], t_expanded=4, d_model=5) Metal vs CPU max |Δ| = {d:e}");
        assert!(
            d <= ATOL,
            "snac_decode_f32 Metal vs CPU max |Δ| = {d} > {ATOL}"
        );

        // Also cross-check against the reference `SnacDecoder::decode` — the
        // CPU arm must be bit-identical to it (a sanity gate on the CPU arm
        // rather than a Metal claim).
        let ref_weights = SnacWeights {
            codebooks: vec![
                codebooks[0].clone(),
                codebooks[1].clone(),
                codebooks[2].clone(),
            ],
            out_projs: vec![
                out_projs[0].clone(),
                out_projs[1].clone(),
                out_projs[2].clone(),
            ],
        };
        let reference = SnacDecoder::new(cfg, ref_weights)
            .unwrap()
            .decode(&codes)
            .unwrap();
        assert_eq!(
            cpu_out, reference,
            "Compute::cpu()::snac_decode_f32 must be bit-identical to SnacDecoder::decode"
        );
    }

    /// The published 44.1 kHz topology has four stages. This is a separate
    /// ABI/parity pin because a three-stage-only shader can pass every 24 kHz
    /// test while silently omitting the finest 44.1 kHz code stream.
    #[test]
    fn four_stage_44khz_metal_matches_cpu_within_atol_or_clean_skip() {
        let Some(compute_metal) = metal_compute() else {
            println!("skip: no Metal device on this host (clean skip, never fabricated)");
            return;
        };
        let compute_cpu = Compute::cpu();
        let cfg = SnacConfig::snac_44khz();
        let codebooks = vec![
            make_codebook(0),
            make_codebook(1),
            make_codebook(2),
            make_codebook(3),
        ];
        let out_projs = vec![make_proj(0), make_proj(1), make_proj(2), make_proj(3)];
        // 1*8 == 2*4 == 4*2 == 8*1.
        let codes = vec![
            vec![1],
            vec![2, 3],
            vec![0, 1, 2, 3],
            vec![3, 2, 1, 0, 1, 2, 3, 0],
        ];

        let cpu_out = compute_cpu
            .snac_decode_f32(&codes, cfg, &codebooks, &out_projs)
            .expect("four-stage CPU arm must succeed");
        let metal_out = compute_metal
            .snac_decode_f32(&codes, cfg, &codebooks, &out_projs)
            .expect("four-stage Metal arm must succeed");
        assert_eq!(cpu_out.len(), 8 * D_MODEL);
        assert_eq!(cpu_out.len(), metal_out.len());
        let d = max_delta(&cpu_out, &metal_out);
        println!("four-stage [8,4,2,1] Metal vs CPU max |Δ| = {d:e}");
        assert!(d <= ATOL, "four-stage Metal vs CPU max |Δ| = {d} > {ATOL}");
    }

    /// Strides `[1, 1, 1]` collapse SNAC to a standard 3-stage factorized
    /// RVQ (DAC-shape). Uses longer T=8 to exercise multiple base timesteps
    /// AND confirms the temporal upsample degenerates to identity when
    /// every stride is 1 (bit-identical to a `dac_rvq_decode` fold with
    /// n_codebooks=3).
    #[test]
    fn stride_1_collapses_to_standard_rvq_within_atol_or_clean_skip() {
        let Some(compute_metal) = metal_compute() else {
            println!("skip: no Metal device on this host (clean skip, never fabricated)");
            return;
        };
        let compute_cpu = Compute::cpu();

        let cfg = SnacConfig {
            sample_rate: 24_000,
            vq_strides: [1, 1, 1, 0],
            n_stages: 3,
        };
        let codebooks = make_codebooks_tiny();
        let out_projs = make_projs_tiny();
        let t = 8usize;
        // Deterministic pseudo-random codes across the 3 stages (SplitMix64
        // for a spread — mirrors the sibling canonical-shape parity tests).
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut mk_stage = |len: usize| -> Vec<u32> {
            let mut v = Vec::with_capacity(len);
            for _ in 0..len {
                state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
                let mut z = state;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                z ^= z >> 31;
                v.push((z as u32) % (CB_SIZE as u32));
            }
            v
        };
        let codes: [Vec<u32>; 3] = [mk_stage(t), mk_stage(t), mk_stage(t)];

        let cpu_out = compute_cpu
            .snac_decode_f32(&codes, cfg, &codebooks, &out_projs)
            .expect("CPU arm must succeed");
        let metal_out = compute_metal
            .snac_decode_f32(&codes, cfg, &codebooks, &out_projs)
            .expect("Metal arm must succeed post Vocoder Metal wave WF5");
        assert_eq!(cpu_out.len(), t * D_MODEL);
        let d = max_delta(&cpu_out, &metal_out);
        println!("strides [1,1,1] (t={t}, d_model=5) Metal vs CPU max |Δ| = {d:e}");
        assert!(
            d <= ATOL,
            "snac_decode_f32 strides [1,1,1] Metal vs CPU max |Δ| = {d} > {ATOL}"
        );
    }

    /// Canonical-ish shape: 3 stages × 32 entries × codebook_dim 8 →
    /// d_model 64, with strides `[4, 2, 1]` and codes lens `[3, 6, 12]`
    /// giving `t_expanded = 12`. Kept smaller than the released SNAC 24 kHz
    /// shape (codebook_size=4096, d_model=768) so the test finishes quickly
    /// on every Metal device while still exercising a non-trivial
    /// factorized inner dim (codebook_dim=8, the canonical DAC/SNAC value)
    /// and the multi-scale upsample. Also asserts the negative-control
    /// bound holds so the ATOL is not vacuous.
    #[test]
    fn canonical_shape_metal_matches_cpu_with_negative_control_or_clean_skip() {
        let Some(compute_metal) = metal_compute() else {
            println!("skip: no Metal device on this host (clean skip, never fabricated)");
            return;
        };
        let compute_cpu = Compute::cpu();

        // Custom shape for this test (bigger than the tiny helpers).
        const BIG_CB_SIZE: usize = 32;
        const BIG_CB_DIM: usize = 8;
        const BIG_D_MODEL: usize = 64;

        let make_cb = |s: usize| -> CodebookTable {
            let mut data = vec![0.0_f32; BIG_CB_SIZE * BIG_CB_DIM];
            for i in 0..BIG_CB_SIZE {
                for d in 0..BIG_CB_DIM {
                    data[i * BIG_CB_DIM + d] = (i + d) as f32 + (s as f32) * 10.0;
                }
            }
            CodebookTable::new(BIG_CB_SIZE, BIG_CB_DIM, data).unwrap()
        };
        let make_p = |s: usize| -> DacOutProj {
            let mut w = vec![0.0_f32; BIG_D_MODEL * BIG_CB_DIM];
            for o in 0..BIG_D_MODEL {
                for c in 0..BIG_CB_DIM {
                    w[o * BIG_CB_DIM + c] = 0.5 + o as f32 * 0.25 + c as f32 * 0.125 + s as f32;
                }
            }
            let b: Vec<f32> = (0..BIG_D_MODEL)
                .map(|o| o as f32 * 0.0625 - s as f32 * 0.5)
                .collect();
            DacOutProj::new(BIG_D_MODEL, BIG_CB_DIM, w, b).unwrap()
        };
        let cfg = SnacConfig {
            sample_rate: 24_000,
            vq_strides: [4, 2, 1, 0],
            n_stages: 3,
        };
        let mut codebooks: [CodebookTable; 3] = [make_cb(0), make_cb(1), make_cb(2)];
        let out_projs: [DacOutProj; 3] = [make_p(0), make_p(1), make_p(2)];
        // Co-aligned frames: 3*4 == 6*2 == 12*1 == 12.
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut mk_stage = |len: usize| -> Vec<u32> {
            let mut v = Vec::with_capacity(len);
            for _ in 0..len {
                state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
                let mut z = state;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                z ^= z >> 31;
                v.push((z as u32) % (BIG_CB_SIZE as u32));
            }
            v
        };
        let codes: [Vec<u32>; 3] = [mk_stage(3), mk_stage(6), mk_stage(12)];

        let cpu_out = compute_cpu
            .snac_decode_f32(&codes, cfg, &codebooks, &out_projs)
            .expect("CPU arm must succeed");
        let metal_out = compute_metal
            .snac_decode_f32(&codes, cfg, &codebooks, &out_projs)
            .expect("Metal arm must succeed post Vocoder Metal wave WF5");
        assert_eq!(cpu_out.len(), 12 * BIG_D_MODEL);
        let d = max_delta(&cpu_out, &metal_out);
        println!(
            "canonical (3 stages [4,2,1], cb=32,cb_dim=8,d=64, t=12) Metal vs CPU max |Δ| = {d:e}"
        );
        assert!(
            d <= ATOL,
            "canonical snac_decode_f32 Metal vs CPU max |Δ| = {d} > {ATOL}"
        );

        // Negative control: perturb one codebook row that stage 0's first
        // code (`codes[0][0]`) actually references. A random-index draw could
        // easily skip row 0 for every timestep, making the perturbation
        // invisible and the control vacuous. Because SNAC then projects
        // `low_row · W_0 + b_0` before summing AND broadcasts to `stride_0`
        // output timesteps, the perturbation is amplified by both the W_0
        // row norms and the stride expansion, so a well-behaved fold must
        // move the output well past ATOL. This proves the CPU-vs-Metal ≤
        // ATOL agreement above is a real match, not a floor any two outputs
        // would satisfy.
        let referenced_row = codes[0][0] as usize;
        let base = referenced_row * BIG_CB_DIM;
        for cell in &mut codebooks[0].data[base..base + BIG_CB_DIM] {
            *cell += 0.1;
        }
        let cpu_out_perturbed = compute_cpu
            .snac_decode_f32(&codes, cfg, &codebooks, &out_projs)
            .expect("CPU perturbed arm must succeed");
        let control = max_delta(&cpu_out, &cpu_out_perturbed);
        println!("negative control (0.1 codebook perturbation) max |Δ| = {control:e}");
        assert!(
            control > ATOL,
            "negative control: a 0.1 codebook perturbation moved CPU output only {control} ≤ \
             {ATOL} — the atol bound would be vacuous; test cannot honestly claim parity"
        );
    }

    /// FR-EX-08 host-side validation: OOB code index is `InvalidArgument`
    /// (never a silent GPU OOB read). This is the most important guard for
    /// SNAC: unlike Mimi / DAC where the RVQ family shares a `[time,
    /// n_codebooks]` code layout, SNAC's `[Vec<u32>; 3]` shape means an
    /// invalid index in any of the three stage vectors must be caught with
    /// its stage index in the error message.
    #[test]
    fn out_of_range_code_is_invalid_argument_or_clean_skip() {
        let Some(compute_metal) = metal_compute() else {
            println!("skip: no Metal device on this host (clean skip, never fabricated)");
            return;
        };
        let cfg = tiny_config();
        let codebooks = make_codebooks_tiny();
        let out_projs = make_projs_tiny();
        // Set stage 1's second code to codebook_size (== OOB by 1). The
        // kernel would silently read past the codebook row without host-side
        // validation; FR-EX-08 catches it upstream of the dispatch.
        let codes: [Vec<u32>; 3] = [
            vec![0],
            vec![0, CB_SIZE as u32], // stride 2 → 2 codes, second is OOB
            vec![0, 0, 0, 0],
        ];
        let err = compute_metal
            .snac_decode_f32(&codes, cfg, &codebooks, &out_projs)
            .expect_err("OOB code index must be an explicit InvalidArgument (FR-EX-08)");
        assert!(
            matches!(err, VokraError::InvalidArgument(_)),
            "expected InvalidArgument for OOB code, got {err:?}",
        );
    }

    /// FR-EX-08 host-side validation: cross-stage T mis-alignment is
    /// `InvalidArgument`. SNAC's multi-scale RVQ requires
    /// `codes[i].len() * strides[i]` to equal the same T for every stage
    /// (upstream `ResidualVectorQuantize.from_codes` broadcasts every
    /// stage's projection to the same output timeline). This test pins the
    /// host-side check that mirrors `SnacDecoder::check_and_measure`.
    #[test]
    fn cross_stage_t_mismatch_is_invalid_argument_or_clean_skip() {
        let Some(compute_metal) = metal_compute() else {
            println!("skip: no Metal device on this host (clean skip, never fabricated)");
            return;
        };
        let cfg = tiny_config(); // strides [4, 2, 1]
        let codebooks = make_codebooks_tiny();
        let out_projs = make_projs_tiny();
        // codes[0].len() * 4 = 4 (expected T=4), codes[1].len() * 2 = 6
        // (mismatch), codes[2].len() * 1 = 4 (matches stage 0). Stage 1's
        // T=6 breaks the co-aligned base frames invariant.
        let codes: [Vec<u32>; 3] = [vec![0], vec![0, 0, 0], vec![0, 0, 0, 0]];
        let err = compute_metal
            .snac_decode_f32(&codes, cfg, &codebooks, &out_projs)
            .expect_err(
                "cross-stage T mis-alignment must be an explicit InvalidArgument (FR-EX-08)",
            );
        assert!(
            matches!(err, VokraError::InvalidArgument(_)),
            "expected InvalidArgument for cross-stage T mismatch, got {err:?}",
        );
    }

    /// FR-EX-08 host-side validation: zero stride is `InvalidArgument`.
    /// Upstream `SnacDecoder::new` rejects `stride = 0` because it would
    /// divide the base frame rate by zero; the Metal arm mirrors that
    /// check before touching the kernel dispatch.
    #[test]
    fn zero_stride_is_invalid_argument_or_clean_skip() {
        let Some(compute_metal) = metal_compute() else {
            println!("skip: no Metal device on this host (clean skip, never fabricated)");
            return;
        };
        let mut cfg = tiny_config();
        cfg.vq_strides[1] = 0; // stage 1 stride = 0 → divide-by-zero risk.
        let codebooks = make_codebooks_tiny();
        let out_projs = make_projs_tiny();
        let codes: [Vec<u32>; 3] = [vec![0], vec![0, 0], vec![0, 0, 0, 0]];
        let err = compute_metal
            .snac_decode_f32(&codes, cfg, &codebooks, &out_projs)
            .expect_err("zero stride must be an explicit InvalidArgument (FR-EX-08)");
        assert!(
            matches!(err, VokraError::InvalidArgument(_)),
            "expected InvalidArgument for zero stride, got {err:?}",
        );
    }

    /// FR-EX-08 host-side validation: wrong per-stage codebook shape is
    /// `InvalidArgument`. All three stages must share `codebook_size` and
    /// `codebook_dim` (upstream `SnacDecoder::new` invariant — new to SNAC
    /// vs the plain Mimi/DAC parity tests where the same shape is checked
    /// per `n_codebooks` count).
    #[test]
    fn wrong_codebook_shape_is_invalid_argument_or_clean_skip() {
        let Some(compute_metal) = metal_compute() else {
            println!("skip: no Metal device on this host (clean skip, never fabricated)");
            return;
        };
        let cfg = tiny_config();
        // Stage 1 has a different codebook_size than stages 0 / 2.
        let bigger_cb =
            CodebookTable::new(CB_SIZE + 1, CB_DIM, vec![0.0; (CB_SIZE + 1) * CB_DIM]).unwrap();
        let codebooks: [CodebookTable; 3] = [make_codebook(0), bigger_cb, make_codebook(2)];
        let out_projs = make_projs_tiny();
        let codes: [Vec<u32>; 3] = [vec![0], vec![0, 0], vec![0, 0, 0, 0]];
        let err = compute_metal
            .snac_decode_f32(&codes, cfg, &codebooks, &out_projs)
            .expect_err(
                "mismatched per-stage codebook shape must be an explicit InvalidArgument \
                 (FR-EX-08)",
            );
        assert!(
            matches!(err, VokraError::InvalidArgument(_)),
            "expected InvalidArgument for wrong codebook shape, got {err:?}",
        );
    }

    /// FR-EX-08 host-side validation: wrong per-stage projection shape is
    /// `InvalidArgument`. All three stages' projections must share
    /// `d_model` and `codebook_dim` (upstream `SnacDecoder::new` invariant).
    #[test]
    fn wrong_proj_shape_is_invalid_argument_or_clean_skip() {
        let Some(compute_metal) = metal_compute() else {
            println!("skip: no Metal device on this host (clean skip, never fabricated)");
            return;
        };
        let cfg = tiny_config();
        let codebooks = make_codebooks_tiny();
        // Build a mis-sized projection for stage 2 (d_model + 1).
        let bad_d = D_MODEL + 1;
        let bad_proj = DacOutProj::new(
            bad_d,
            CB_DIM,
            vec![0.0_f32; bad_d * CB_DIM],
            vec![0.0_f32; bad_d],
        )
        .unwrap();
        let out_projs: [DacOutProj; 3] = [make_proj(0), make_proj(1), bad_proj];
        let codes: [Vec<u32>; 3] = [vec![0], vec![0, 0], vec![0, 0, 0, 0]];
        let err = compute_metal
            .snac_decode_f32(&codes, cfg, &codebooks, &out_projs)
            .expect_err(
                "mismatched per-stage projection shape must be an explicit InvalidArgument \
                 (FR-EX-08)",
            );
        assert!(
            matches!(err, VokraError::InvalidArgument(_)),
            "expected InvalidArgument for wrong projection shape, got {err:?}",
        );
    }

    /// All three code vectors empty → returns an empty `Vec<f32>` on both
    /// arms (no dispatch, no allocation panic). Mirrors
    /// `SnacDecoder::decode`'s `decode_all_empty_returns_empty` test.
    #[test]
    fn empty_all_stages_returns_empty_vec_or_clean_skip() {
        let Some(compute_metal) = metal_compute() else {
            println!("skip: no Metal device on this host (clean skip, never fabricated)");
            return;
        };
        let cfg = tiny_config();
        let codebooks = make_codebooks_tiny();
        let out_projs = make_projs_tiny();
        let codes: [Vec<u32>; 3] = [vec![], vec![], vec![]];
        let out = compute_metal
            .snac_decode_f32(&codes, cfg, &codebooks, &out_projs)
            .expect("empty-all-stages decode must return an empty Vec");
        assert!(
            out.is_empty(),
            "expected empty Vec, got {} elems",
            out.len()
        );
    }
}
