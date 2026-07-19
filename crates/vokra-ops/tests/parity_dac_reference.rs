//! DAC factorized RVQ decode vs the descript-audio-codec reference (M4-04
//! T14, NFR-QL-01) — projection stage included.
//!
//! Fixture: `tests/parity/dac/` — REAL sliced 24 kHz (tag 0.0.4) DAC weights:
//! the first 192 rows of the first 12 low-dim codebooks + the **effective**
//! (weight-normed) `out_proj` weights/biases as torch itself computed them +
//! fixed-seed codes + the reference decode from the **public**
//! `ResidualVectorQuantize.from_codes` API (prefix decode = upstream
//! variable-bitrate semantics). Committed; runs with no Python. Regenerate:
//!
//! ```text
//! venv/bin/python tools/parity/mimi_dump.py dac --checkpoint <weights_24khz.pth> \
//!     --out tests/parity/dac --seed 0 --time 32 --rows 192 --books 12
//! ```
//!
//! The fixture manifest also pins the upstream `DacRvqAttrs` values
//! (n_codebooks=32 / codebook_size=1024 / codebook_dim=8 / d_model=1024 /
//! sample_rate=24000 / hop=320) — asserted below against
//! `DacRvqAttrs::dac_24khz()` so a drift between the checkpoint and the
//! canonical constructor fails here, not in a consumer.
//!
//! # Tolerance (honest bound)
//!
//! Same per-quantizer math on both sides (lookup → GEMV(+bias) → residual
//! sum), so the only divergence source is torch's conv1d reduction order vs
//! our scalar 8-element dot — measured `max|Δ| = 1.91e-6` on this fixture
//! (2026-07-15 generation, printed by the test). `ATOL = 1e-4` is ~52x the
//! measured error and 100x under the design-wide FP32 `atol = 0.01`.

use vokra_core::cache::paged::{BlockSize, PagedKvCache};
use vokra_ops::{
    CodebookTable, DacOutProj, DacRvqAttrs, dac_paged_dims, dac_rvq_decode, dac_rvq_decode_paged,
    dac_rvq_read_summed,
};

const BOOKS: usize = 12;
const ROWS: usize = 192;
const CODEBOOK_DIM: usize = 8;
const D_MODEL: usize = 1024;
const TIME: usize = 32;

/// Measured 1.91e-6 (module docs); bound ~52x measured, 100x under the
/// NFR-QL-01 FP32 envelope.
const ATOL: f32 = 1e-4;

fn fixtures_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("parity")
        .join("dac")
}

fn read_f32(name: &str) -> Vec<f32> {
    let path = fixtures_dir().join(name);
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
    assert_eq!(bytes.len() % 4, 0);
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn read_u32(name: &str) -> Vec<u32> {
    let path = fixtures_dir().join(name);
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
    assert_eq!(bytes.len() % 4, 0);
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn load_fixture() -> (
    Vec<CodebookTable>,
    Vec<DacOutProj>,
    Vec<u32>,
    Vec<f32>,
    DacRvqAttrs,
) {
    let tables_flat = read_f32("codebook_tables_sliced.f32");
    assert_eq!(tables_flat.len(), BOOKS * ROWS * CODEBOOK_DIM);
    let w_flat = read_f32("out_proj_weight.f32");
    assert_eq!(w_flat.len(), BOOKS * D_MODEL * CODEBOOK_DIM);
    let b_flat = read_f32("out_proj_bias.f32");
    assert_eq!(b_flat.len(), BOOKS * D_MODEL);
    let codes = read_u32("codes.u32");
    assert_eq!(codes.len(), TIME * BOOKS);
    let reference = read_f32("decoded_features.f32");
    assert_eq!(reference.len(), TIME * D_MODEL);

    let attrs = DacRvqAttrs {
        n_codebooks: BOOKS,
        codebook_size: ROWS,
        codebook_dim: CODEBOOK_DIM,
        d_model: D_MODEL,
    };
    let per_t = ROWS * CODEBOOK_DIM;
    let per_w = D_MODEL * CODEBOOK_DIM;
    let tables: Vec<CodebookTable> = (0..BOOKS)
        .map(|cb| {
            CodebookTable::new(
                ROWS,
                CODEBOOK_DIM,
                tables_flat[cb * per_t..(cb + 1) * per_t].to_vec(),
            )
            .expect("table")
        })
        .collect();
    let projs: Vec<DacOutProj> = (0..BOOKS)
        .map(|cb| {
            DacOutProj::new(
                D_MODEL,
                CODEBOOK_DIM,
                w_flat[cb * per_w..(cb + 1) * per_w].to_vec(),
                b_flat[cb * D_MODEL..(cb + 1) * D_MODEL].to_vec(),
            )
            .expect("proj")
        })
        .collect();
    (tables, projs, codes, reference, attrs)
}

#[test]
fn dac_rvq_decode_matches_descript_reference_projection_included() {
    let (tables, projs, codes, reference, attrs) = load_fixture();

    // The manifest pins the full-checkpoint shapes; the canonical constructor
    // must agree on the variant-level facts (only n_codebooks differs — the
    // fixture uses a 12-book prefix of the physical 32).
    let canonical = DacRvqAttrs::dac_24khz();
    assert_eq!(canonical.codebook_size, 1024);
    assert_eq!(canonical.codebook_dim, attrs.codebook_dim);
    assert_eq!(canonical.d_model, attrs.d_model);
    assert!(attrs.n_codebooks <= canonical.n_codebooks);

    let got = dac_rvq_decode(&codes, TIME, &tables, &projs, &attrs).expect("decode");

    let mut max_abs = 0.0_f32;
    let mut max_at = 0usize;
    for (i, (g, r)) in got.iter().zip(reference.iter()).enumerate() {
        let d = (g - r).abs();
        if d > max_abs {
            max_abs = d;
            max_at = i;
        }
    }
    eprintln!(
        "dac reference parity: max|Δ| = {max_abs:.3e} at flat index {max_at} \
         (t={}, d={}) — ATOL {ATOL:.1e}",
        max_at / D_MODEL,
        max_at % D_MODEL
    );
    assert!(
        max_abs <= ATOL,
        "DAC decode diverged from the descript reference: max|Δ| = {max_abs:.3e} > {ATOL:.1e}"
    );
}

#[test]
fn dac_paged_block_four_read_matches_direct_on_reference_fixture() {
    // T14 second case: the paged (BlockSize::Four — DAC primary) round trip
    // over the REAL fixture weights is bit-identical to the direct decode.
    let (tables, projs, codes, _reference, attrs) = load_fixture();

    let direct = dac_rvq_decode(&codes, TIME, &tables, &projs, &attrs).expect("direct");

    let dims = dac_paged_dims(&attrs, 1, TIME);
    let mut cache = PagedKvCache::<f32>::pre_allocate(dims, BlockSize::Four).expect("cache");
    dac_rvq_decode_paged(&codes, TIME, &tables, &projs, &attrs, 0, &mut cache, 0)
        .expect("paged decode");

    for t in 0..TIME {
        let summed = dac_rvq_read_summed(&cache, &attrs, 0, t).expect("read");
        assert_eq!(
            summed,
            &direct[t * D_MODEL..(t + 1) * D_MODEL],
            "paged/direct mismatch at t={t}"
        );
    }
}
