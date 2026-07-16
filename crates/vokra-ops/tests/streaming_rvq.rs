//! M4-04 T07/T08 — CSM/Moshi multi-stream streaming completion tests for the
//! RVQ codec family (`mimi_rvq` + `dac_rvq`).
//!
//! These are the **consumer-contract** tests M4-05 (Sesame CSM-1B) and M4-06
//! (Moshi full-duplex) build against. Verified patterns (rustdoc'd on the
//! ops as the streaming contract):
//!
//! 1. **Incremental decode** — repeated `decode_paged` with an advancing
//!    `time_start` on the same stream; every timestep reads back equal to a
//!    one-shot direct decode.
//! 2. **Position clock** — `advance()` is a commit-only counter
//!    (`positions()` bookkeeping); it never affects reads or appends, and
//!    there is **no sliding-window eviction** in `PagedKvCache` (ADR M4-04
//!    §D-i; page reclamation is `release_layer` / `reset` only).
//! 3. **Reclaim-then-reuse hygiene** — after `release_layer` / `reset`,
//!    reads observe the empty state and re-appended data only (released
//!    pages are zeroed → no stale bleed-through).
//! 4. **Per-stream independent cursors** — full-duplex streams write at
//!    unrelated time offsets without aliasing (n_stream ≥ 2 generic form;
//!    the actual stream count is fixed by M4-06).
//! 5. **Chunk-window summed read** (`mimi_rvq_read_summed_range`) —
//!    bit-identical to the per-`t` `read_summed` loop, page-boundary
//!    crossing included; unwritten gaps read as zero rows.

use vokra_core::VokraError;
use vokra_core::cache::paged::{BlockSize, PagedKvCache};
use vokra_ops::{
    CodebookTable, DacOutProj, DacRvqAttrs, MimiRvqAttrs, dac_paged_dims, dac_rvq_decode,
    dac_rvq_decode_paged, dac_rvq_read_summed, dac_rvq_read_summed_range, mimi_paged_dims,
    mimi_rvq_decode, mimi_rvq_decode_paged, mimi_rvq_read_summed, mimi_rvq_read_summed_range,
};

// ---------------------------------------------------------------------------
// Shared fixtures
// ---------------------------------------------------------------------------

fn mimi_attrs() -> MimiRvqAttrs {
    MimiRvqAttrs {
        n_codebooks: 3,
        codebook_size: 5,
        d_model: 4,
    }
}

fn mimi_tables(attrs: MimiRvqAttrs) -> Vec<CodebookTable> {
    let mut tables = Vec::with_capacity(attrs.n_codebooks);
    for cb in 0..attrs.n_codebooks {
        let mut data = vec![0.0_f32; attrs.codebook_size * attrs.d_model];
        for i in 0..attrs.codebook_size {
            for d in 0..attrs.d_model {
                data[i * attrs.d_model + d] = (i * 7 + d) as f32 + (cb as f32) * 100.0;
            }
        }
        tables.push(CodebookTable::new(attrs.codebook_size, attrs.d_model, data).unwrap());
    }
    tables
}

fn dac_attrs() -> DacRvqAttrs {
    DacRvqAttrs {
        n_codebooks: 2,
        codebook_size: 4,
        codebook_dim: 3,
        d_model: 5,
    }
}

fn dac_tables(attrs: DacRvqAttrs) -> Vec<CodebookTable> {
    let mut tables = Vec::with_capacity(attrs.n_codebooks);
    for cb in 0..attrs.n_codebooks {
        let mut data = vec![0.0_f32; attrs.codebook_size * attrs.codebook_dim];
        for i in 0..attrs.codebook_size {
            for d in 0..attrs.codebook_dim {
                data[i * attrs.codebook_dim + d] = (i + 2 * d) as f32 - (cb as f32) * 3.0;
            }
        }
        tables.push(CodebookTable::new(attrs.codebook_size, attrs.codebook_dim, data).unwrap());
    }
    tables
}

fn dac_projs(attrs: DacRvqAttrs) -> Vec<DacOutProj> {
    let mut projs = Vec::with_capacity(attrs.n_codebooks);
    for cb in 0..attrs.n_codebooks {
        let mut w = vec![0.0_f32; attrs.d_model * attrs.codebook_dim];
        for o in 0..attrs.d_model {
            for c in 0..attrs.codebook_dim {
                w[o * attrs.codebook_dim + c] =
                    0.25 * (o as f32 + 1.0) - 0.5 * c as f32 + cb as f32;
            }
        }
        let b: Vec<f32> = (0..attrs.d_model).map(|o| o as f32 * 0.125).collect();
        projs.push(DacOutProj::new(attrs.d_model, attrs.codebook_dim, w, b).unwrap());
    }
    projs
}

fn ramp_codes(time: usize, n_codebooks: usize, codebook_size: usize, salt: u32) -> Vec<u32> {
    (0..(time * n_codebooks) as u32)
        .map(|i| (i.wrapping_mul(31).wrapping_add(salt)) % codebook_size as u32)
        .collect()
}

// ---------------------------------------------------------------------------
// (1) Incremental decode — time_start advances chunk by chunk
// ---------------------------------------------------------------------------

#[test]
fn mimi_incremental_time_start_chunks_match_one_shot_decode() {
    let attrs = mimi_attrs();
    let tables = mimi_tables(attrs);
    let total_time = 11; // deliberately not a block multiple
    let codes = ramp_codes(total_time, attrs.n_codebooks, attrs.codebook_size, 3);

    for &bs in &[BlockSize::Two, BlockSize::Four] {
        let dims = mimi_paged_dims(&attrs, 1, total_time);
        let mut cache = PagedKvCache::<f32>::pre_allocate(dims, bs).unwrap();

        // Stream the codes in uneven chunks (1, 3, 2, 5) like a live decoder.
        let mut t0 = 0usize;
        for &chunk in &[1usize, 3, 2, 5] {
            let chunk_codes = &codes[t0 * attrs.n_codebooks..(t0 + chunk) * attrs.n_codebooks];
            mimi_rvq_decode_paged(chunk_codes, chunk, &tables, &attrs, 0, &mut cache, t0).unwrap();
            cache.advance(chunk); // commit the position clock per chunk
            t0 += chunk;
        }
        assert_eq!(t0, total_time);
        assert_eq!(
            cache.positions(),
            total_time,
            "position clock == committed steps"
        );

        let direct = mimi_rvq_decode(&codes, total_time, &tables, &attrs).unwrap();
        for t in 0..total_time {
            let got = mimi_rvq_read_summed(&cache, &attrs, 0, t).unwrap();
            assert_eq!(
                got,
                &direct[t * attrs.d_model..(t + 1) * attrs.d_model],
                "incremental vs one-shot mismatch at t={t} (block={bs:?})"
            );
        }
    }
}

#[test]
fn dac_incremental_time_start_chunks_match_one_shot_decode() {
    let attrs = dac_attrs();
    let tables = dac_tables(attrs);
    let projs = dac_projs(attrs);
    let total_time = 9;
    let codes = ramp_codes(total_time, attrs.n_codebooks, attrs.codebook_size, 7);

    let dims = dac_paged_dims(&attrs, 1, total_time);
    let mut cache = PagedKvCache::<f32>::pre_allocate(dims, BlockSize::Four).unwrap();

    let mut t0 = 0usize;
    for &chunk in &[2usize, 4, 3] {
        let chunk_codes = &codes[t0 * attrs.n_codebooks..(t0 + chunk) * attrs.n_codebooks];
        dac_rvq_decode_paged(
            chunk_codes,
            chunk,
            &tables,
            &projs,
            &attrs,
            0,
            &mut cache,
            t0,
        )
        .unwrap();
        cache.advance(chunk);
        t0 += chunk;
    }
    assert_eq!(cache.positions(), total_time);

    let direct = dac_rvq_decode(&codes, total_time, &tables, &projs, &attrs).unwrap();
    for t in 0..total_time {
        let got = dac_rvq_read_summed(&cache, &attrs, 0, t).unwrap();
        assert_eq!(
            got,
            &direct[t * attrs.d_model..(t + 1) * attrs.d_model],
            "dac incremental vs one-shot mismatch at t={t}"
        );
    }
}

// ---------------------------------------------------------------------------
// (2) Position clock: advance() commits only — it never mutates state
// ---------------------------------------------------------------------------

#[test]
fn advance_is_a_pure_position_clock_commit() {
    // ADR M4-04 §D-i: advance() = `self.pos += n` and nothing else. Reads
    // before/after advance are identical; appends after advance land where
    // the caller says (time_start), not where the clock points.
    let attrs = mimi_attrs();
    let tables = mimi_tables(attrs);
    let dims = mimi_paged_dims(&attrs, 1, 8);
    let mut cache = PagedKvCache::<f32>::pre_allocate(dims, BlockSize::Two).unwrap();

    let codes = ramp_codes(2, attrs.n_codebooks, attrs.codebook_size, 1);
    mimi_rvq_decode_paged(&codes, 2, &tables, &attrs, 0, &mut cache, 0).unwrap();

    let before: Vec<Vec<f32>> = (0..2)
        .map(|t| mimi_rvq_read_summed(&cache, &attrs, 0, t).unwrap())
        .collect();
    assert_eq!(cache.positions(), 0, "no advance yet");

    cache.advance(2);
    assert_eq!(cache.positions(), 2);

    let after: Vec<Vec<f32>> = (0..2)
        .map(|t| mimi_rvq_read_summed(&cache, &attrs, 0, t).unwrap())
        .collect();
    assert_eq!(before, after, "advance() must not perturb stored rows");

    // Appending beyond the clock is caller-directed — t=5 while pos=2 works,
    // because there is no eviction window tied to the clock.
    let one = ramp_codes(1, attrs.n_codebooks, attrs.codebook_size, 9);
    mimi_rvq_decode_paged(&one, 1, &tables, &attrs, 0, &mut cache, 5).unwrap();
    let direct = mimi_rvq_decode(&one, 1, &tables, &attrs).unwrap();
    assert_eq!(
        mimi_rvq_read_summed(&cache, &attrs, 0, 5).unwrap(),
        direct,
        "append at t=5 with pos=2 must land at t=5"
    );
}

// ---------------------------------------------------------------------------
// (3) Reclaim-then-reuse hygiene (release_layer / reset)
// ---------------------------------------------------------------------------

#[test]
fn release_layer_then_reuse_shows_no_stale_data() {
    let attrs = mimi_attrs();
    let tables = mimi_tables(attrs);
    let dims = mimi_paged_dims(&attrs, 1, 4);
    let mut cache = PagedKvCache::<f32>::pre_allocate(dims, BlockSize::Two).unwrap();

    // Round 1: all codebooks written with non-zero rows.
    let round1 = vec![1u32; attrs.n_codebooks * 2];
    mimi_rvq_decode_paged(&round1, 2, &tables, &attrs, 0, &mut cache, 0).unwrap();
    assert!(cache.pages_in_use() > 0);

    // Reclaim layer 0 (the only layer): every read now sees the empty state.
    cache.release_layer(0).unwrap();
    assert_eq!(cache.pages_in_use(), 0);
    assert_eq!(
        cache.read_step(0, 0, 0, 0),
        None,
        "released block is unbound"
    );
    let zeroed = mimi_rvq_read_summed(&cache, &attrs, 0, 0).unwrap();
    assert_eq!(
        zeroed,
        vec![0.0; attrs.d_model],
        "read-side zero after release"
    );

    // Round 2: write only codebook 0 (partial write) at t=0 via a 1-codebook
    // attrs view — the released page must have been zeroed, so codebooks 1..
    // contribute nothing (no round-1 bleed-through).
    let narrow = MimiRvqAttrs {
        n_codebooks: 1,
        ..attrs
    };
    let round2 = vec![3u32];
    mimi_rvq_decode_paged(&round2, 1, &tables[..1], &narrow, 0, &mut cache, 0).unwrap();

    let got = mimi_rvq_read_summed(&cache, &attrs, 0, 0).unwrap();
    let want = tables[0].row(3).unwrap();
    assert_eq!(got, want, "only the re-written codebook may contribute");
}

#[test]
fn reset_then_reuse_shows_no_stale_data() {
    let attrs = mimi_attrs();
    let tables = mimi_tables(attrs);
    let dims = mimi_paged_dims(&attrs, 2, 4);
    let mut cache = PagedKvCache::<f32>::pre_allocate(dims, BlockSize::Two).unwrap();

    let round1 = vec![2u32; attrs.n_codebooks];
    mimi_rvq_decode_paged(&round1, 1, &tables, &attrs, 0, &mut cache, 0).unwrap();
    mimi_rvq_decode_paged(&round1, 1, &tables, &attrs, 1, &mut cache, 2).unwrap();
    cache.advance(1);

    cache.reset();
    assert_eq!(cache.positions(), 0, "reset rewinds the position clock");
    assert_eq!(cache.pages_in_use(), 0);
    for (s, t) in [(0usize, 0usize), (1, 2)] {
        assert_eq!(
            mimi_rvq_read_summed(&cache, &attrs, s, t).unwrap(),
            vec![0.0; attrs.d_model],
            "post-reset read must be zero (s={s}, t={t})"
        );
    }

    // Fresh decode after reset matches direct decode (arena reuse is clean).
    let codes = ramp_codes(3, attrs.n_codebooks, attrs.codebook_size, 5);
    mimi_rvq_decode_paged(&codes, 3, &tables, &attrs, 1, &mut cache, 0).unwrap();
    let direct = mimi_rvq_decode(&codes, 3, &tables, &attrs).unwrap();
    for t in 0..3 {
        assert_eq!(
            mimi_rvq_read_summed(&cache, &attrs, 1, t).unwrap(),
            &direct[t * attrs.d_model..(t + 1) * attrs.d_model]
        );
    }
}

// ---------------------------------------------------------------------------
// (4) Per-stream independent cursors (full-duplex generic form)
// ---------------------------------------------------------------------------

#[test]
fn per_stream_cursors_at_unrelated_time_offsets_do_not_alias() {
    // Full-duplex shape: stream A far ahead (t≈100), stream B near the start
    // (t≈10). n_stream = 2 is the generic form; M4-06 fixes the real count.
    let attrs = mimi_attrs();
    let tables = mimi_tables(attrs);
    let max_time = 128;
    let dims = mimi_paged_dims(&attrs, 2, max_time);
    let mut cache = PagedKvCache::<f32>::pre_allocate(dims, BlockSize::Two).unwrap();

    let codes_a = ramp_codes(4, attrs.n_codebooks, attrs.codebook_size, 11);
    let codes_b = ramp_codes(4, attrs.n_codebooks, attrs.codebook_size, 22);
    mimi_rvq_decode_paged(&codes_a, 4, &tables, &attrs, 0, &mut cache, 100).unwrap();
    mimi_rvq_decode_paged(&codes_b, 4, &tables, &attrs, 1, &mut cache, 10).unwrap();

    let direct_a = mimi_rvq_decode(&codes_a, 4, &tables, &attrs).unwrap();
    let direct_b = mimi_rvq_decode(&codes_b, 4, &tables, &attrs).unwrap();
    for i in 0..4 {
        assert_eq!(
            mimi_rvq_read_summed(&cache, &attrs, 0, 100 + i).unwrap(),
            &direct_a[i * attrs.d_model..(i + 1) * attrs.d_model],
            "stream A row {i}"
        );
        assert_eq!(
            mimi_rvq_read_summed(&cache, &attrs, 1, 10 + i).unwrap(),
            &direct_b[i * attrs.d_model..(i + 1) * attrs.d_model],
            "stream B row {i}"
        );
        // Cross-reads: the *other* stream's slots at these times were never
        // written → zero rows.
        assert_eq!(
            mimi_rvq_read_summed(&cache, &attrs, 1, 100 + i).unwrap(),
            vec![0.0; attrs.d_model],
            "stream B must not see stream A's t=100 band"
        );
        assert_eq!(
            mimi_rvq_read_summed(&cache, &attrs, 0, 10 + i).unwrap(),
            vec![0.0; attrs.d_model],
            "stream A must not see stream B's t=10 band"
        );
    }

    // Out-of-range stream stays an explicit error (FR-EX-08 surface).
    assert!(matches!(
        mimi_rvq_decode_paged(&codes_a, 4, &tables, &attrs, 2, &mut cache, 0),
        Err(VokraError::InvalidArgument(_))
    ));
}

// ---------------------------------------------------------------------------
// (5) T08 — chunk-window summed read
// ---------------------------------------------------------------------------

#[test]
fn read_summed_range_is_bit_identical_to_per_t_loop() {
    let attrs = mimi_attrs();
    let tables = mimi_tables(attrs);
    let total_time = 10;
    let codes = ramp_codes(total_time, attrs.n_codebooks, attrs.codebook_size, 13);

    for &bs in &[BlockSize::Two, BlockSize::Four] {
        let dims = mimi_paged_dims(&attrs, 1, total_time);
        let mut cache = PagedKvCache::<f32>::pre_allocate(dims, bs).unwrap();
        mimi_rvq_decode_paged(&codes, total_time, &tables, &attrs, 0, &mut cache, 0).unwrap();

        for (t0, t1) in [(0usize, 10usize), (1, 5), (3, 4), (0, 1), (9, 10), (2, 9)] {
            let chunk = mimi_rvq_read_summed_range(&cache, &attrs, 0, t0..t1).unwrap();
            assert_eq!(chunk.len(), (t1 - t0) * attrs.d_model);
            for t in t0..t1 {
                let per_t = mimi_rvq_read_summed(&cache, &attrs, 0, t).unwrap();
                assert_eq!(
                    &chunk[(t - t0) * attrs.d_model..(t - t0 + 1) * attrs.d_model],
                    per_t.as_slice(),
                    "range [{t0},{t1}) vs per-t at t={t} (block={bs:?})"
                );
            }
        }
    }
}

#[test]
fn read_summed_range_crosses_page_boundaries() {
    // block=2 → pages are [0,1], [2,3], [4,5]; the window [1,5) touches
    // three pages with partial coverage on both ends (the spec's example).
    let attrs = mimi_attrs();
    let tables = mimi_tables(attrs);
    let codes = ramp_codes(6, attrs.n_codebooks, attrs.codebook_size, 17);
    let dims = mimi_paged_dims(&attrs, 1, 6);
    let mut cache = PagedKvCache::<f32>::pre_allocate(dims, BlockSize::Two).unwrap();
    mimi_rvq_decode_paged(&codes, 6, &tables, &attrs, 0, &mut cache, 0).unwrap();

    let direct = mimi_rvq_decode(&codes, 6, &tables, &attrs).unwrap();
    let chunk = mimi_rvq_read_summed_range(&cache, &attrs, 0, 1..5).unwrap();
    assert_eq!(chunk, direct[attrs.d_model..5 * attrs.d_model].to_vec());
}

#[test]
fn read_summed_range_gap_semantics_zero_rows() {
    // Write t=0..2 and t=6..8 only; the window [0,8) must return zeros for
    // the unbound middle blocks (never-written gap).
    let attrs = mimi_attrs();
    let tables = mimi_tables(attrs);
    let dims = mimi_paged_dims(&attrs, 1, 8);
    let mut cache = PagedKvCache::<f32>::pre_allocate(dims, BlockSize::Two).unwrap();

    let head = ramp_codes(2, attrs.n_codebooks, attrs.codebook_size, 19);
    let tail = ramp_codes(2, attrs.n_codebooks, attrs.codebook_size, 23);
    mimi_rvq_decode_paged(&head, 2, &tables, &attrs, 0, &mut cache, 0).unwrap();
    mimi_rvq_decode_paged(&tail, 2, &tables, &attrs, 0, &mut cache, 6).unwrap();

    let chunk = mimi_rvq_read_summed_range(&cache, &attrs, 0, 0..8).unwrap();
    let head_direct = mimi_rvq_decode(&head, 2, &tables, &attrs).unwrap();
    let tail_direct = mimi_rvq_decode(&tail, 2, &tables, &attrs).unwrap();

    assert_eq!(&chunk[..2 * attrs.d_model], head_direct.as_slice());
    assert_eq!(
        &chunk[2 * attrs.d_model..6 * attrs.d_model],
        vec![0.0; 4 * attrs.d_model].as_slice(),
        "gap rows must be zero"
    );
    assert_eq!(&chunk[6 * attrs.d_model..], tail_direct.as_slice());
}

#[test]
fn read_summed_range_multi_stream_and_empty_window() {
    let attrs = mimi_attrs();
    let tables = mimi_tables(attrs);
    let dims = mimi_paged_dims(&attrs, 2, 4);
    let mut cache = PagedKvCache::<f32>::pre_allocate(dims, BlockSize::Two).unwrap();

    let c0 = vec![0u32; attrs.n_codebooks * 2];
    let c1 = vec![4u32; attrs.n_codebooks * 2];
    mimi_rvq_decode_paged(&c0, 2, &tables, &attrs, 0, &mut cache, 0).unwrap();
    mimi_rvq_decode_paged(&c1, 2, &tables, &attrs, 1, &mut cache, 0).unwrap();

    let r0 = mimi_rvq_read_summed_range(&cache, &attrs, 0, 0..2).unwrap();
    let r1 = mimi_rvq_read_summed_range(&cache, &attrs, 1, 0..2).unwrap();
    assert_ne!(r0, r1, "streams must not alias in the range read");
    assert_eq!(r0, mimi_rvq_decode(&c0, 2, &tables, &attrs).unwrap());
    assert_eq!(r1, mimi_rvq_decode(&c1, 2, &tables, &attrs).unwrap());

    // Empty window is Ok(vec![]), not an error.
    assert_eq!(
        mimi_rvq_read_summed_range(&cache, &attrs, 0, 1..1).unwrap(),
        Vec::<f32>::new()
    );
}

#[test]
fn read_summed_range_rejects_bad_windows() {
    let attrs = mimi_attrs();
    let dims = mimi_paged_dims(&attrs, 1, 4);
    let cache = PagedKvCache::<f32>::pre_allocate(dims, BlockSize::Two).unwrap();

    // t1 > max_time.
    assert!(matches!(
        mimi_rvq_read_summed_range(&cache, &attrs, 0, 0..5),
        Err(VokraError::InvalidArgument(_))
    ));
    // t0 > t1 (Range<usize> allows constructing it; the op must reject).
    #[allow(clippy::reversed_empty_ranges)]
    let reversed = 3..1;
    assert!(matches!(
        mimi_rvq_read_summed_range(&cache, &attrs, 0, reversed),
        Err(VokraError::InvalidArgument(_))
    ));
    // Out-of-range stream.
    assert!(matches!(
        mimi_rvq_read_summed_range(&cache, &attrs, 9, 0..1),
        Err(VokraError::InvalidArgument(_))
    ));
}

// ---------------------------------------------------------------------------
// (5-DAC) T08 — chunk-window summed read, DAC-symmetric mirror
//
// `dac_rvq_read_summed_range` is the RVQ-family symmetric partner of
// `mimi_rvq_read_summed_range` — same shape-generic core (`read_summed_range_core`,
// mimi_rvq.rs), just the factorized DAC attrs. These mirror the mimi (5)
// section so the CSM / Moshi (M4-05/06) chunk read mouth is verified on the
// factorized codec too.
// ---------------------------------------------------------------------------

#[test]
fn dac_read_summed_range_is_bit_identical_to_per_t_loop() {
    let attrs = dac_attrs();
    let tables = dac_tables(attrs);
    let projs = dac_projs(attrs);
    let total_time = 10;
    let codes = ramp_codes(total_time, attrs.n_codebooks, attrs.codebook_size, 29);

    for &bs in &[BlockSize::Two, BlockSize::Four] {
        let dims = dac_paged_dims(&attrs, 1, total_time);
        let mut cache = PagedKvCache::<f32>::pre_allocate(dims, bs).unwrap();
        dac_rvq_decode_paged(
            &codes, total_time, &tables, &projs, &attrs, 0, &mut cache, 0,
        )
        .unwrap();

        for (t0, t1) in [(0usize, 10usize), (1, 5), (3, 4), (0, 1), (9, 10), (2, 9)] {
            let chunk = dac_rvq_read_summed_range(&cache, &attrs, 0, t0..t1).unwrap();
            assert_eq!(chunk.len(), (t1 - t0) * attrs.d_model);
            for t in t0..t1 {
                let per_t = dac_rvq_read_summed(&cache, &attrs, 0, t).unwrap();
                assert_eq!(
                    &chunk[(t - t0) * attrs.d_model..(t - t0 + 1) * attrs.d_model],
                    per_t.as_slice(),
                    "dac range [{t0},{t1}) vs per-t at t={t} (block={bs:?})"
                );
            }
        }
    }
}

#[test]
fn dac_read_summed_range_matches_full_decode_over_window() {
    // The direct oracle: a chunk read over [t0,t1) equals the corresponding
    // rows of a one-shot `dac_rvq_decode` (windowed read == full decode over
    // that window — the T08 contract, factorized edition).
    let attrs = dac_attrs();
    let tables = dac_tables(attrs);
    let projs = dac_projs(attrs);
    let total_time = 8;
    let codes = ramp_codes(total_time, attrs.n_codebooks, attrs.codebook_size, 31);

    let dims = dac_paged_dims(&attrs, 1, total_time);
    let mut cache = PagedKvCache::<f32>::pre_allocate(dims, BlockSize::Four).unwrap();
    dac_rvq_decode_paged(
        &codes, total_time, &tables, &projs, &attrs, 0, &mut cache, 0,
    )
    .unwrap();

    let direct = dac_rvq_decode(&codes, total_time, &tables, &projs, &attrs).unwrap();
    // A window that starts and ends mid-page (block=4 → pages [0..4], [4..8]).
    let chunk = dac_rvq_read_summed_range(&cache, &attrs, 0, 1..6).unwrap();
    assert_eq!(chunk, direct[attrs.d_model..6 * attrs.d_model].to_vec());
}

#[test]
fn dac_read_summed_range_gap_semantics_zero_rows() {
    // Write t=0..2 and t=6..8 only; the window [0,8) returns zeros for the
    // unbound middle blocks (never-written gap).
    let attrs = dac_attrs();
    let tables = dac_tables(attrs);
    let projs = dac_projs(attrs);
    let dims = dac_paged_dims(&attrs, 1, 8);
    let mut cache = PagedKvCache::<f32>::pre_allocate(dims, BlockSize::Two).unwrap();

    let head = ramp_codes(2, attrs.n_codebooks, attrs.codebook_size, 37);
    let tail = ramp_codes(2, attrs.n_codebooks, attrs.codebook_size, 41);
    dac_rvq_decode_paged(&head, 2, &tables, &projs, &attrs, 0, &mut cache, 0).unwrap();
    dac_rvq_decode_paged(&tail, 2, &tables, &projs, &attrs, 0, &mut cache, 6).unwrap();

    let chunk = dac_rvq_read_summed_range(&cache, &attrs, 0, 0..8).unwrap();
    let head_direct = dac_rvq_decode(&head, 2, &tables, &projs, &attrs).unwrap();
    let tail_direct = dac_rvq_decode(&tail, 2, &tables, &projs, &attrs).unwrap();

    assert_eq!(&chunk[..2 * attrs.d_model], head_direct.as_slice());
    assert_eq!(
        &chunk[2 * attrs.d_model..6 * attrs.d_model],
        vec![0.0; 4 * attrs.d_model].as_slice(),
        "gap rows must be zero"
    );
    assert_eq!(&chunk[6 * attrs.d_model..], tail_direct.as_slice());
}

#[test]
fn dac_read_summed_range_multi_stream_and_empty_window() {
    let attrs = dac_attrs();
    let tables = dac_tables(attrs);
    let projs = dac_projs(attrs);
    let dims = dac_paged_dims(&attrs, 2, 4);
    let mut cache = PagedKvCache::<f32>::pre_allocate(dims, BlockSize::Four).unwrap();

    // Distinct constant codes per stream (0 vs 3, both < codebook_size=4) so
    // the two streams genuinely differ — a ramp_codes salt pair could collide
    // mod codebook_size.
    let c0 = vec![0u32; attrs.n_codebooks * 2];
    let c1 = vec![3u32; attrs.n_codebooks * 2];
    dac_rvq_decode_paged(&c0, 2, &tables, &projs, &attrs, 0, &mut cache, 0).unwrap();
    dac_rvq_decode_paged(&c1, 2, &tables, &projs, &attrs, 1, &mut cache, 0).unwrap();

    let r0 = dac_rvq_read_summed_range(&cache, &attrs, 0, 0..2).unwrap();
    let r1 = dac_rvq_read_summed_range(&cache, &attrs, 1, 0..2).unwrap();
    assert_ne!(r0, r1, "streams must not alias in the range read");
    assert_eq!(r0, dac_rvq_decode(&c0, 2, &tables, &projs, &attrs).unwrap());
    assert_eq!(r1, dac_rvq_decode(&c1, 2, &tables, &projs, &attrs).unwrap());

    // Empty window is Ok(vec![]), not an error.
    assert_eq!(
        dac_rvq_read_summed_range(&cache, &attrs, 0, 1..1).unwrap(),
        Vec::<f32>::new()
    );
}

#[test]
fn dac_read_summed_range_rejects_bad_windows() {
    let attrs = dac_attrs();
    let dims = dac_paged_dims(&attrs, 1, 4);
    let cache = PagedKvCache::<f32>::pre_allocate(dims, BlockSize::Four).unwrap();

    // t1 > max_time.
    assert!(matches!(
        dac_rvq_read_summed_range(&cache, &attrs, 0, 0..5),
        Err(VokraError::InvalidArgument(_))
    ));
    // t0 > t1.
    #[allow(clippy::reversed_empty_ranges)]
    let reversed = 3..1;
    assert!(matches!(
        dac_rvq_read_summed_range(&cache, &attrs, 0, reversed),
        Err(VokraError::InvalidArgument(_))
    ));
    // Out-of-range stream.
    assert!(matches!(
        dac_rvq_read_summed_range(&cache, &attrs, 9, 0..1),
        Err(VokraError::InvalidArgument(_))
    ));
}
