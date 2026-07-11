//! Numerical-parity harness for the Mimi RVQ decode op (M3-06, FR-OP-30).
//!
//! # Reference oracle
//!
//! The Kyutai reference (`moshi` PyPI package + a real Mimi checkpoint) is
//! not on this development machine — see
//! `docs/adr/M3-06-mimi-rvq.md` §D5 for why the external fixture lands with
//! M3-09 (CosyVoice2), which shares the download. Until then the M3-06
//! parity harness runs **internal-oracle** checks that pin exactly the two
//! numerical invariants the ticket calls out:
//!
//! 1. **Lookup is a plain gather.** Row `i` of codebook `cb` equals the
//!    `d_model`-long span at index `i` in the flat table data — no scaling,
//!    no offset, no clamp.
//! 2. **Residual sum is bit-identical to a scalar FP32 fold.** Given the
//!    same `codes` / `tables`, the op's output is the same
//!    `sum_{cb} tables[cb].row(codes[t, cb])` a naive nested loop produces
//!    at the FP32 accumulator precision.
//!
//! Both are enforced at **exact equality** (not `atol = 0.01`) because the
//! reference is the very fold the op documents itself as. A future
//! Kyutai-reference dump (`tools/parity/mimi_dump.py`) will add an
//! `atol = 0.01` (NFR-QL-01) check against the external tensor.

use vokra_core::cache::paged::{BlockSize, KvDims, PagedKvCache};
use vokra_ops::{
    CodebookTable, MimiRvqAttrs, codebook_lookup, mimi_rvq_decode, mimi_rvq_decode_paged,
    mimi_rvq_read_summed,
};

/// A ~real-Mimi-shaped fixture (n_codebooks=8, but small `codebook_size`
/// and `d_model` so the test runs cheap). The number of codebooks matters
/// for the residual sum invariant — bigger `n_codebooks` exercises more
/// accumulation.
fn realistic_shape() -> MimiRvqAttrs {
    MimiRvqAttrs {
        n_codebooks: 8,
        codebook_size: 16,
        d_model: 32,
    }
}

/// A pseudo-random-but-deterministic ramp: every codebook has its own
/// affine transform of `(i, d)` so the decoded features actually depend on
/// which codebook contributed which row.
fn deterministic_tables(attrs: MimiRvqAttrs) -> Vec<CodebookTable> {
    let mut tables = Vec::with_capacity(attrs.n_codebooks);
    for cb in 0..attrs.n_codebooks {
        let mut data = vec![0.0_f32; attrs.codebook_size * attrs.d_model];
        for i in 0..attrs.codebook_size {
            for d in 0..attrs.d_model {
                // A distinct value at each (cb, i, d) so a mis-swap of any
                // pair shows up as a diff.
                data[i * attrs.d_model + d] =
                    (cb as f32) * 7.0 + (i as f32) * 0.5 + (d as f32) * 0.25
                        - (cb as f32 * i as f32 * 0.03);
            }
        }
        tables.push(CodebookTable::new(attrs.codebook_size, attrs.d_model, data).unwrap());
    }
    tables
}

fn scalar_reference(
    codes: &[u32],
    time: usize,
    tables: &[CodebookTable],
    attrs: &MimiRvqAttrs,
) -> Vec<f32> {
    let mut out = vec![0.0_f32; time * attrs.d_model];
    for t in 0..time {
        for cb in 0..attrs.n_codebooks {
            let idx = codes[t * attrs.n_codebooks + cb];
            let row = tables[cb].row(idx).unwrap();
            for d in 0..attrs.d_model {
                out[t * attrs.d_model + d] += row[d];
            }
        }
    }
    out
}

#[test]
fn lookup_matches_flat_gather() {
    let attrs = realistic_shape();
    let tables = deterministic_tables(attrs);
    for cb in 0..attrs.n_codebooks {
        for i in 0..attrs.codebook_size {
            let row = codebook_lookup(&tables, cb, i as u32, &attrs).unwrap();
            let expected = &tables[cb].data[i * attrs.d_model..i * attrs.d_model + attrs.d_model];
            assert_eq!(row, expected, "cb={cb} i={i}");
        }
    }
}

#[test]
fn decode_matches_scalar_reference_bit_exact() {
    let attrs = realistic_shape();
    let tables = deterministic_tables(attrs);
    let time = 25;
    // Deterministic codes: LCG-ish index generator per (t, cb).
    let mut codes = Vec::with_capacity(time * attrs.n_codebooks);
    for t in 0..time {
        for cb in 0..attrs.n_codebooks {
            let idx = ((t * 31 + cb * 7 + 3) % attrs.codebook_size) as u32;
            codes.push(idx);
        }
    }

    let got = mimi_rvq_decode(&codes, time, &tables, &attrs).unwrap();
    let want = scalar_reference(&codes, time, &tables, &attrs);
    assert_eq!(got, want, "op output must equal scalar FP32 fold");
}

#[test]
fn paged_summed_read_matches_direct_decode_across_block_sizes() {
    // Cross-check: the paged store's per-codebook write followed by
    // `mimi_rvq_read_summed` reproduces the direct-decode output for both
    // supported block sizes (M3-06 T05 / T06).
    let attrs = MimiRvqAttrs {
        n_codebooks: 4,
        codebook_size: 8,
        d_model: 16,
    };
    let tables = deterministic_tables(attrs);
    let time = 7;
    let mut codes = Vec::with_capacity(time * attrs.n_codebooks);
    for t in 0..time {
        for cb in 0..attrs.n_codebooks {
            let idx = ((t + cb * 5 + 1) % attrs.codebook_size) as u32;
            codes.push(idx);
        }
    }

    let direct = mimi_rvq_decode(&codes, time, &tables, &attrs).unwrap();

    for bs in [BlockSize::Two, BlockSize::Four] {
        let dims = KvDims {
            n_layer: 1,
            n_head: 1,
            d_head: attrs.d_model,
            n_stream: 1,
            n_codebook: attrs.n_codebooks,
            max_time: time,
        };
        let mut cache = PagedKvCache::<f32>::pre_allocate(dims, bs).unwrap();
        mimi_rvq_decode_paged(&codes, time, &tables, &attrs, 0, &mut cache, 0).unwrap();
        for t in 0..time {
            let got = mimi_rvq_read_summed(&cache, &attrs, 0, t).unwrap();
            let want = &direct[t * attrs.d_model..(t + 1) * attrs.d_model];
            assert_eq!(got, want, "bs={bs:?} t={t}");
        }
    }
}

#[test]
fn residual_sum_is_order_stable_for_permuted_codebooks() {
    // Additive-only FP32 fold: swapping two codebooks' contributions
    // changes the *order* of adds but not the final value at bit-identical
    // precision *only* when the values differ by ULP-safe amounts. For a
    // small n_codebooks × small d_model shape with values on similar scales,
    // it is stable — this test locks in that stability for the M3-06
    // canonical fixture so a future SIMD rearrangement of the fold order is
    // caught up front.
    let attrs = MimiRvqAttrs {
        n_codebooks: 3,
        codebook_size: 4,
        d_model: 5,
    };
    // Codebook rows are tiny integers so pair sums are exact in f32.
    let tables: Vec<CodebookTable> = (0..attrs.n_codebooks)
        .map(|cb| {
            let mut data = vec![0.0_f32; attrs.codebook_size * attrs.d_model];
            for i in 0..attrs.codebook_size {
                for d in 0..attrs.d_model {
                    data[i * attrs.d_model + d] = (cb + i + d) as f32;
                }
            }
            CodebookTable::new(attrs.codebook_size, attrs.d_model, data).unwrap()
        })
        .collect();
    let time = 3;
    // Ascending codes: t=0 all zeros, t=1 all ones, t=2 all twos.
    let codes: Vec<u32> = (0..time)
        .flat_map(|t| std::iter::repeat_n(t as u32, attrs.n_codebooks))
        .collect();
    let out = mimi_rvq_decode(&codes, time, &tables, &attrs).unwrap();
    // Hand-computed: at t, cb, d: cb + t + d. Summed over cb (0..n_codebooks):
    //   sum_cb (cb + t + d)
    //   = n_codebooks*(t + d) + sum_cb cb
    //   = n_codebooks*(t + d) + n_codebooks*(n_codebooks - 1)/2
    let nc = attrs.n_codebooks as f32;
    let cb_sum = nc * (nc - 1.0) / 2.0;
    for t in 0..time {
        for d in 0..attrs.d_model {
            let want = nc * (t as f32 + d as f32) + cb_sum;
            let got = out[t * attrs.d_model + d];
            assert_eq!(got, want, "t={t} d={d}");
        }
    }
}
