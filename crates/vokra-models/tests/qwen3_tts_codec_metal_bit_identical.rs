//! Vocoder Metal wave WF5 — Qwen3-TTS-Codec RVQ decode Metal MSL kernel
//! real-GPU parity (`vokra_qwen3_tts_codec_decode_f32` implemented in
//! `crates/vokra-backend-metal/src/context.rs`, wired through
//! `Compute::qwen3_tts_codec_f32` in `crates/vokra-models/src/compute.rs`).
//!
//! - **Off-feature band** (compiled when `metal` is off / non-Apple):
//!   `HotOp::Qwen3TtsCodec` is not Metal-covered off the feature, so
//!   `Compute::for_backend(Metal, [Qwen3TtsCodec])` is an explicit
//!   [`vokra_core::VokraError::BackendUnavailable`] — never a silent CPU
//!   substitute (FR-EX-08).
//! - **Metal band** (`--features metal`, Apple): real-GPU parity on THIS M1
//!   iMac (CLAUDE.md dev environment). Skips cleanly (printed reason) when no
//!   Metal device is present.
//!
//! # atol bound and negative control
//!
//! The MSL fold uses fast-math-compiled FP32 arithmetic which may re-associate;
//! the CPU fold in `vokra_ops::qwen3_tts_codec::qwen3_tts_codec_decode` is a
//! strictly left-to-right FP32 loop. On canonical shapes the two typically
//! agree bit-for-bit, but we assert the honest atol ≤ 5e-4 bound (mirrors
//! `mimi_rvq_metal_bit_identical.rs::ATOL`, the codec-family FP32 GEMV-scale
//! envelope).
//!
//! A **negative control** shows the bound is discriminating rather than
//! vacuous: perturbing one codebook row by 0.1 moves the CPU output well past
//! 5e-4, so a "CPU vs Metal ≤ 5e-4" agreement is a real match, not a floor.
//!
//! # Real-weight parity
//!
//! Synthetic ramp codebooks exercise the shader shape end-to-end (semantic +
//! acoustic gather + FP32 fold + read-back). Real Qwen3-TTS-Codec codebook
//! tables (from the Apache-2.0 Qwen3-TTS-12Hz upstream) can be added by a
//! later real-weight parity slice; this test scope stays synthetic (the
//! primary source is verified via `qwen3_tts_codec::tests` in `vokra-ops`).

#[cfg(not(all(feature = "metal", any(target_os = "macos", target_os = "ios"))))]
mod off_feature {
    use vokra_core::{BackendKind, VokraError};
    use vokra_models::compute::{Compute, HotOp};

    /// Off the Metal build the Qwen3TtsCodec coverage arm is a
    /// `BackendUnavailable` at the `for_backend` entry — never a silent CPU
    /// substitute (FR-EX-08).
    #[test]
    fn for_backend_metal_qwen3_tts_codec_off_feature_is_backend_unavailable() {
        let Err(err) = Compute::for_backend(BackendKind::Metal, &[HotOp::Qwen3TtsCodec]) else {
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
    use vokra_ops::{CodebookTable, Qwen3TtsCodecConfig, qwen3_tts_codec_decode};

    /// Metal / CPU parity bound for the Qwen3-TTS-Codec FP32 gather + fold. The
    /// MSL kernel adds the same FP32 codebook rows in the same left-to-right
    /// order as the CPU fold, so we typically see |Δ| = 0. The bound is
    /// budgeted at 5e-4 to match the M4-05 CSM / Moshi Metal parity envelope
    /// for codec-side FP32 folds (accommodates fast-math re-association).
    const ATOL: f32 = 5e-4;

    fn max_delta(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f32, f32::max)
    }

    /// Build a Metal `Compute` covering Qwen3TtsCodec. Returns `None` when no
    /// Metal device is present (clean skip, mirrors the mimi_rvq pattern).
    fn metal_compute() -> Option<Compute> {
        match Compute::for_backend(BackendKind::Metal, &[HotOp::Qwen3TtsCodec]) {
            Ok(c) => Some(c),
            Err(VokraError::BackendUnavailable(_)) => None,
            Err(e) => panic!("unexpected Metal for_backend error: {e}"),
        }
    }

    /// Tiny config: 3 quantizers (1 semantic + 2 acoustic), semantic vocab 5,
    /// acoustic vocab 4, feature width 3. Small enough to hand-fold and
    /// distinct enough to catch a silent swap of semantic vs acoustic (which
    /// would either silently clamp the semantic index at 3 or panic on
    /// acoustic vocab 5 — both would be immediately visible in the parity
    /// output).
    fn tiny_config() -> Qwen3TtsCodecConfig {
        Qwen3TtsCodecConfig {
            num_quantizers: 3,
            num_semantic_quantizers: 1,
            codebook_size: 4,
            semantic_codebook_size: 5,
            codebook_dim: 3,
            sample_rate: 24_000,
            downsample_rate: 1_920,
        }
    }

    /// Deterministic tables: quantizer `q`, row `i` cell `d` =
    /// `q*100 + i*10 + d`. Semantic quantizer (q=0) has 5 rows; acoustic
    /// quantizers have 4 rows.
    fn make_tiny_tables(c: &Qwen3TtsCodecConfig) -> Vec<CodebookTable> {
        let mut tables = Vec::with_capacity(c.num_quantizers);
        for q in 0..c.num_quantizers {
            let vocab = c.quantizer_vocab_size(q).unwrap();
            let mut data = vec![0.0_f32; vocab * c.codebook_dim];
            for i in 0..vocab {
                for d in 0..c.codebook_dim {
                    data[i * c.codebook_dim + d] =
                        (q as f32) * 100.0 + (i as f32) * 10.0 + (d as f32);
                }
            }
            tables.push(CodebookTable::new(vocab, c.codebook_dim, data).unwrap());
        }
        tables
    }

    /// Tiny hybrid shape: (1 semantic + 2 acoustic × 5/4 vocab × 3 features,
    /// 4 timesteps). Exercises both the ragged-tail guard and the semantic /
    /// acoustic branch selection inside the fold.
    #[test]
    fn tiny_hybrid_shape_metal_matches_cpu_within_atol_or_clean_skip() {
        let Some(compute_metal) = metal_compute() else {
            println!("skip: no Metal device on this host (clean skip, never fabricated)");
            return;
        };
        assert_eq!(compute_metal.backend_name(), "metal");
        let compute_cpu = Compute::cpu();

        let c = tiny_config();
        let tables = make_tiny_tables(&c);
        // Codes: 4 timesteps. Semantic idx 4 is legal for semantic vocab 5 but
        // would overflow acoustic vocab 4 — this pins that per-quantizer vocab
        // is honored, not a silent shared-vocab clamp.
        let codes: Vec<Vec<u32>> = vec![
            vec![0, 1, 4, 2], // semantic quantizer, vocab 5 (max 4)
            vec![3, 0, 2, 1], // acoustic quantizer #0, vocab 4 (max 3)
            vec![1, 2, 3, 0], // acoustic quantizer #1, vocab 4 (max 3)
        ];

        let cpu_out = compute_cpu
            .qwen3_tts_codec_f32(&codes, &tables, &c)
            .expect("CPU arm must succeed");
        let metal_out = compute_metal
            .qwen3_tts_codec_f32(&codes, &tables, &c)
            .expect("Metal arm must succeed post Vocoder wave WF5");
        assert_eq!(cpu_out.len(), metal_out.len(), "output shapes must match");
        assert_eq!(cpu_out.len(), 4 * c.codebook_dim);

        let d = max_delta(&cpu_out, &metal_out);
        println!("tiny hybrid (1sem+2ac × 5/4 × 3, time=4) Metal vs CPU max |Δ| = {d:e}");
        assert!(
            d <= ATOL,
            "qwen3_tts_codec_f32 Metal vs CPU max |Δ| = {d} > {ATOL}"
        );

        // Also cross-check against the reference `qwen3_tts_codec_decode` —
        // the CPU arm must be bit-identical to it (this is a sanity gate on
        // the CPU arm rather than a Metal claim).
        let reference = qwen3_tts_codec_decode(&codes, &tables, &c).unwrap();
        assert_eq!(
            cpu_out, reference,
            "Compute::cpu()::qwen3_tts_codec_f32 must be bit-identical to qwen3_tts_codec_decode"
        );
    }

    /// Canonical-shaped shape: mirrors the Qwen3-TTS-12Hz split ratio (1
    /// semantic + N-1 acoustic) but with smaller vocabs so the test finishes
    /// quickly on every Metal device while still exercising 8-way folds and
    /// larger `codebook_dim`. Also asserts the negative-control bound holds
    /// so the ATOL is not vacuous.
    #[test]
    fn canonical_shape_metal_matches_cpu_with_negative_control_or_clean_skip() {
        let Some(compute_metal) = metal_compute() else {
            println!("skip: no Metal device on this host (clean skip, never fabricated)");
            return;
        };
        let compute_cpu = Compute::cpu();

        // 8 quantizers (1 semantic + 7 acoustic), semantic vocab 32, acoustic
        // vocab 16, codebook_dim 64 — non-trivial per-quantizer stride
        // difference (`semantic_codebook_size != codebook_size`) so a shared-
        // stride kernel bug would be immediately visible.
        let c = Qwen3TtsCodecConfig {
            num_quantizers: 8,
            num_semantic_quantizers: 1,
            codebook_size: 16,
            semantic_codebook_size: 32,
            codebook_dim: 64,
            sample_rate: 24_000,
            downsample_rate: 1_920,
        };
        let mut tables: Vec<CodebookTable> = Vec::with_capacity(c.num_quantizers);
        for q in 0..c.num_quantizers {
            let vocab = c.quantizer_vocab_size(q).unwrap();
            let mut data = vec![0.0_f32; vocab * c.codebook_dim];
            for i in 0..vocab {
                for d in 0..c.codebook_dim {
                    data[i * c.codebook_dim + d] =
                        (q as f32) * 100.0 + (i as f32) * 10.0 + (d as f32) * 0.1;
                }
            }
            tables.push(CodebookTable::new(vocab, c.codebook_dim, data).unwrap());
        }
        let time = 12;
        // Deterministic pseudo-random codes with per-quantizer vocab modulo
        // (no stdlib RNG dep — SplitMix64 over usize is enough for a spread).
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut codes: Vec<Vec<u32>> = Vec::with_capacity(c.num_quantizers);
        for q in 0..c.num_quantizers {
            let vocab = c.quantizer_vocab_size(q).unwrap() as u32;
            let mut stream = Vec::with_capacity(time);
            for _ in 0..time {
                state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
                let mut z = state;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                z ^= z >> 31;
                stream.push((z as u32) % vocab);
            }
            codes.push(stream);
        }

        let cpu_out = compute_cpu
            .qwen3_tts_codec_f32(&codes, &tables, &c)
            .expect("CPU arm must succeed");
        let metal_out = compute_metal
            .qwen3_tts_codec_f32(&codes, &tables, &c)
            .expect("Metal arm must succeed post Vocoder wave WF5");
        assert_eq!(cpu_out.len(), metal_out.len());
        let d = max_delta(&cpu_out, &metal_out);
        println!("canonical (1sem+7ac × 32/16 × 64, time=12) Metal vs CPU max |Δ| = {d:e}");
        assert!(
            d <= ATOL,
            "canonical qwen3_tts_codec_f32 Metal vs CPU max |Δ| = {d} > {ATOL}"
        );

        // Negative control: perturb ONE codebook row by 0.1 — specifically the
        // row that `codes[0][0]` (timestep 0, semantic quantizer) actually
        // references, so the perturbation is guaranteed to affect
        // `cpu_out_perturbed[0..codebook_dim]`. A random-index draw could
        // easily skip row 0 for every timestep, which would leave the
        // perturbation invisible and make the control vacuously trivial. By
        // perturbing `tables[0].row(codes[0][0])` we guarantee visibility, so
        // a well-behaved fold must move the output past ATOL — the CPU-vs-
        // Metal ≤ ATOL agreement above is a real match, not a floor any two
        // outputs would satisfy.
        let referenced_row = codes[0][0] as usize;
        let semantic_data = &mut tables[0].data;
        let base = referenced_row * c.codebook_dim;
        for cell in &mut semantic_data[base..base + c.codebook_dim] {
            *cell += 0.1;
        }
        let cpu_out_perturbed = compute_cpu
            .qwen3_tts_codec_f32(&codes, &tables, &c)
            .expect("CPU perturbed arm must succeed");
        let control = max_delta(&cpu_out, &cpu_out_perturbed);
        println!("negative control (0.1 semantic-codebook perturbation) max |Δ| = {control:e}");
        assert!(
            control > ATOL,
            "negative control: a 0.1 codebook perturbation moved CPU output only {control} ≤ \
             {ATOL} — the atol bound would be vacuous; test cannot honestly claim parity"
        );
    }

    /// Semantic-only edge case (num_semantic_quantizers == num_quantizers).
    /// The acoustic side is empty; the Metal `new_buffer_from_slice` pads the
    /// empty slice to a 4-byte placeholder and the kernel never reads it
    /// (the loop's acoustic branch is dead when `num_semantic_quantizers ==
    /// num_quantizers`).
    #[test]
    fn semantic_only_shape_metal_matches_cpu_or_clean_skip() {
        let Some(compute_metal) = metal_compute() else {
            println!("skip: no Metal device on this host (clean skip, never fabricated)");
            return;
        };
        let compute_cpu = Compute::cpu();

        // 2 quantizers, both semantic. The acoustic buffer will be empty on
        // the Metal side — this pins that the 4-byte placeholder path is
        // safe.
        let c = Qwen3TtsCodecConfig {
            num_quantizers: 2,
            num_semantic_quantizers: 2,
            codebook_size: 4,
            semantic_codebook_size: 4,
            codebook_dim: 3,
            sample_rate: 24_000,
            downsample_rate: 1_920,
        };
        let mut tables: Vec<CodebookTable> = Vec::with_capacity(c.num_quantizers);
        for q in 0..c.num_quantizers {
            let mut data = vec![0.0_f32; c.semantic_codebook_size * c.codebook_dim];
            for i in 0..c.semantic_codebook_size {
                for d in 0..c.codebook_dim {
                    data[i * c.codebook_dim + d] =
                        (q as f32) * 10.0 + (i as f32) + (d as f32) * 0.01;
                }
            }
            tables
                .push(CodebookTable::new(c.semantic_codebook_size, c.codebook_dim, data).unwrap());
        }
        let codes: Vec<Vec<u32>> = vec![vec![0, 1, 2, 3], vec![3, 2, 1, 0]];

        let cpu_out = compute_cpu
            .qwen3_tts_codec_f32(&codes, &tables, &c)
            .expect("CPU arm must succeed");
        let metal_out = compute_metal
            .qwen3_tts_codec_f32(&codes, &tables, &c)
            .expect("Metal arm (empty acoustic side) must succeed");
        let d = max_delta(&cpu_out, &metal_out);
        println!("semantic-only (2sem+0ac × 4 × 3, time=4) Metal vs CPU max |Δ| = {d:e}");
        assert!(
            d <= ATOL,
            "semantic-only qwen3_tts_codec_f32 Metal vs CPU max |Δ| = {d} > {ATOL}"
        );
    }

    /// Acoustic-only edge case (num_semantic_quantizers == 0). Mirror of the
    /// semantic-only path: the semantic-side buffer is empty; the Metal
    /// `new_buffer_from_slice` pads it to a 4-byte placeholder and the
    /// kernel's semantic branch is dead.
    #[test]
    fn acoustic_only_shape_metal_matches_cpu_or_clean_skip() {
        let Some(compute_metal) = metal_compute() else {
            println!("skip: no Metal device on this host (clean skip, never fabricated)");
            return;
        };
        let compute_cpu = Compute::cpu();

        let c = Qwen3TtsCodecConfig {
            num_quantizers: 2,
            num_semantic_quantizers: 0,
            codebook_size: 4,
            semantic_codebook_size: 4,
            codebook_dim: 3,
            sample_rate: 24_000,
            downsample_rate: 1_920,
        };
        let mut tables: Vec<CodebookTable> = Vec::with_capacity(c.num_quantizers);
        for q in 0..c.num_quantizers {
            let mut data = vec![0.0_f32; c.codebook_size * c.codebook_dim];
            for i in 0..c.codebook_size {
                for d in 0..c.codebook_dim {
                    data[i * c.codebook_dim + d] = (q as f32) + (i as f32) * 0.5 + (d as f32);
                }
            }
            tables.push(CodebookTable::new(c.codebook_size, c.codebook_dim, data).unwrap());
        }
        let codes: Vec<Vec<u32>> = vec![vec![0, 1, 2, 3], vec![3, 2, 1, 0]];

        let cpu_out = compute_cpu
            .qwen3_tts_codec_f32(&codes, &tables, &c)
            .expect("CPU arm must succeed");
        let metal_out = compute_metal
            .qwen3_tts_codec_f32(&codes, &tables, &c)
            .expect("Metal arm (empty semantic side) must succeed");
        let d = max_delta(&cpu_out, &metal_out);
        println!("acoustic-only (0sem+2ac × 4 × 3, time=4) Metal vs CPU max |Δ| = {d:e}");
        assert!(
            d <= ATOL,
            "acoustic-only qwen3_tts_codec_f32 Metal vs CPU max |Δ| = {d} > {ATOL}"
        );
    }

    /// FR-EX-08 host-side validation: out-of-range **semantic** index is
    /// `InvalidArgument` (never a silent GPU OOB read). Semantic vocab is 5;
    /// index 5 is illegal.
    #[test]
    fn out_of_range_semantic_index_is_invalid_argument_or_clean_skip() {
        let Some(compute_metal) = metal_compute() else {
            println!("skip: no Metal device on this host (clean skip, never fabricated)");
            return;
        };
        let c = tiny_config();
        let tables = make_tiny_tables(&c);
        // Semantic vocab is 5 (indices 0..=4); index 5 must fail explicitly.
        let codes: Vec<Vec<u32>> = vec![vec![5], vec![0], vec![0]];
        let err = compute_metal
            .qwen3_tts_codec_f32(&codes, &tables, &c)
            .expect_err("OOB semantic index must be an explicit InvalidArgument (FR-EX-08)");
        assert!(
            matches!(err, VokraError::InvalidArgument(_)),
            "expected InvalidArgument for OOB semantic code, got {err:?}",
        );
    }

    /// FR-EX-08 host-side validation: out-of-range **acoustic** index is
    /// `InvalidArgument`. Acoustic vocab is 4; index 4 is illegal for the
    /// acoustic quantizers even though index 4 IS legal for the semantic
    /// vocab of 5 — this pins that per-quantizer vocab is enforced, not a
    /// global maximum.
    #[test]
    fn out_of_range_acoustic_index_is_invalid_argument_or_clean_skip() {
        let Some(compute_metal) = metal_compute() else {
            println!("skip: no Metal device on this host (clean skip, never fabricated)");
            return;
        };
        let c = tiny_config();
        let tables = make_tiny_tables(&c);
        // Semantic index 4 is legal (vocab 5); acoustic index 4 is illegal
        // (vocab 4). The op must reject the acoustic 4.
        let codes: Vec<Vec<u32>> = vec![vec![4], vec![4], vec![0]];
        let err = compute_metal
            .qwen3_tts_codec_f32(&codes, &tables, &c)
            .expect_err("OOB acoustic index must be an explicit InvalidArgument (FR-EX-08)");
        assert!(
            matches!(err, VokraError::InvalidArgument(_)),
            "expected InvalidArgument for OOB acoustic code, got {err:?}",
        );
    }

    /// FR-EX-08 host-side validation: wrong number of code streams is
    /// `InvalidArgument` (never silently truncated to shortest).
    #[test]
    fn wrong_stream_count_is_invalid_argument_or_clean_skip() {
        let Some(compute_metal) = metal_compute() else {
            println!("skip: no Metal device on this host (clean skip, never fabricated)");
            return;
        };
        let c = tiny_config();
        let tables = make_tiny_tables(&c);
        // Only 2 streams provided; config expects 3.
        let codes: Vec<Vec<u32>> = vec![vec![0], vec![1]];
        let err = compute_metal
            .qwen3_tts_codec_f32(&codes, &tables, &c)
            .expect_err("wrong stream count must be an explicit InvalidArgument (FR-EX-08)");
        assert!(
            matches!(err, VokraError::InvalidArgument(_)),
            "expected InvalidArgument for wrong stream count, got {err:?}",
        );
    }

    /// FR-EX-08 host-side validation: mismatched inner-length is
    /// `InvalidArgument` (never silently shortest-truncated).
    #[test]
    fn mismatched_inner_length_is_invalid_argument_or_clean_skip() {
        let Some(compute_metal) = metal_compute() else {
            println!("skip: no Metal device on this host (clean skip, never fabricated)");
            return;
        };
        let c = tiny_config();
        let tables = make_tiny_tables(&c);
        // Streams 0 and 1 are length 3; stream 2 is length 2 → mismatch.
        let codes: Vec<Vec<u32>> = vec![vec![0, 1, 2], vec![1, 2, 0], vec![0, 1]];
        let err = compute_metal
            .qwen3_tts_codec_f32(&codes, &tables, &c)
            .expect_err("mismatched inner-length must be an explicit InvalidArgument (FR-EX-08)");
        assert!(
            matches!(err, VokraError::InvalidArgument(_)),
            "expected InvalidArgument for mismatched inner-length, got {err:?}",
        );
    }

    /// FR-EX-08 host-side validation: wrong per-table shape is
    /// `InvalidArgument` (semantic entry sized as acoustic).
    #[test]
    fn semantic_table_with_acoustic_vocab_is_invalid_argument_or_clean_skip() {
        let Some(compute_metal) = metal_compute() else {
            println!("skip: no Metal device on this host (clean skip, never fabricated)");
            return;
        };
        let c = tiny_config();
        // Semantic slot (q=0) built with acoustic vocab (4) instead of
        // semantic vocab (5) — WRONG.
        let mut tables: Vec<CodebookTable> = Vec::with_capacity(c.num_quantizers);
        tables.push(
            CodebookTable::new(
                c.codebook_size,
                c.codebook_dim,
                vec![0.0; c.codebook_size * c.codebook_dim],
            )
            .unwrap(),
        );
        for _ in 1..c.num_quantizers {
            tables.push(
                CodebookTable::new(
                    c.codebook_size,
                    c.codebook_dim,
                    vec![0.0; c.codebook_size * c.codebook_dim],
                )
                .unwrap(),
            );
        }
        let codes: Vec<Vec<u32>> = vec![vec![0], vec![0], vec![0]];
        let err = compute_metal
            .qwen3_tts_codec_f32(&codes, &tables, &c)
            .expect_err(
                "semantic-with-acoustic vocab must be an explicit InvalidArgument (FR-EX-08)",
            );
        assert!(
            matches!(err, VokraError::InvalidArgument(_)),
            "expected InvalidArgument for wrong semantic vocab, got {err:?}",
        );
    }

    /// Empty `time = 0` decode returns an empty `Vec<f32>` on the Metal arm
    /// (no dispatch, no allocation panic — mirrors the CPU op's zero-length
    /// window semantics).
    #[test]
    fn empty_time_returns_empty_vec_or_clean_skip() {
        let Some(compute_metal) = metal_compute() else {
            println!("skip: no Metal device on this host (clean skip, never fabricated)");
            return;
        };
        let c = tiny_config();
        let tables = make_tiny_tables(&c);
        // Every quantizer has a zero-length stream (time == 0).
        let codes: Vec<Vec<u32>> = (0..c.num_quantizers).map(|_| Vec::new()).collect();
        let out = compute_metal
            .qwen3_tts_codec_f32(&codes, &tables, &c)
            .expect("empty-time decode must return an empty Vec");
        assert!(
            out.is_empty(),
            "expected empty Vec, got {} elems",
            out.len()
        );
    }
}
