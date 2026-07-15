//! Mimi RVQ decode vs the Kyutai reference implementation (M4-04 T13,
//! NFR-QL-01) — resolves the M3-06 deferral ("Kyutai dump lands later"; see
//! ADR M3-06 §D5 / ADR M4-04 §D-j).
//!
//! Fixture: `tests/parity/mimi/` — REAL sliced Mimi weights (first 48 rows of
//! each of the 8 consumer-prefix codebooks, pre-projected by the upstream
//! `output_proj` modules themselves) + fixed-seed codes + the reference
//! decode from the **public** `moshi` `SplitResidualVectorQuantizer.decode`
//! API. Committed, so this runs in plain `cargo test` with no Python
//! (kaldi_fbank precedent). Regenerate:
//!
//! ```text
//! venv/bin/python tools/parity/mimi_dump.py mimi --checkpoint <mimi.safetensors> \
//!     --out tests/parity/mimi --seed 0 --time 32 --rows 48
//! ```
//!
//! # Tolerance (honest bound)
//!
//! The reference projects **after** the per-split residual sum; the Vokra op
//! sums **pre-projected** rows. For Mimi's bias-free 1×1 conv the two are
//! mathematically equal but FP32-reassociation-different: the acoustic split
//! folds 7 embeddings (values O(1)) before/after a 256→512 GEMV, so the
//! expected wobble is a few ULP of the partial sums — measured
//! `max|Δ| = 1.24e-5` on this fixture (2026-07-15 generation, printed by the
//! test itself). `ATOL = 1e-4` is ~8x the measured error and 100x under the
//! design-wide FP32
//! `atol = 0.01` (NFR-QL-01) — comfortably honest in both directions.

use vokra_ops::{CodebookTable, MimiRvqAttrs, mimi_rvq_decode};

const N_CODEBOOKS: usize = 8;
const ROWS: usize = 48;
const D_MODEL: usize = 512;
const TIME: usize = 32;

/// Measured 1.24e-5 (see module docs); bound ~8x measured, 100x under the
/// NFR-QL-01 FP32 envelope.
const ATOL: f32 = 1e-4;

fn fixtures_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("parity")
        .join("mimi")
}

fn read_f32(name: &str) -> Vec<f32> {
    let path = fixtures_dir().join(name);
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
    assert_eq!(bytes.len() % 4, 0, "{name}: not a whole number of f32");
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn read_u32(name: &str) -> Vec<u32> {
    let path = fixtures_dir().join(name);
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
    assert_eq!(bytes.len() % 4, 0, "{name}: not a whole number of u32");
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[test]
fn mimi_rvq_decode_matches_kyutai_reference() {
    let tables_flat = read_f32("codebook_tables_sliced.f32");
    assert_eq!(tables_flat.len(), N_CODEBOOKS * ROWS * D_MODEL);
    let codes = read_u32("codes.u32");
    assert_eq!(codes.len(), TIME * N_CODEBOOKS);
    let reference = read_f32("decoded_features.f32");
    assert_eq!(reference.len(), TIME * D_MODEL);

    let attrs = MimiRvqAttrs {
        n_codebooks: N_CODEBOOKS,
        codebook_size: ROWS,
        d_model: D_MODEL,
    };
    let per = ROWS * D_MODEL;
    let tables: Vec<CodebookTable> = (0..N_CODEBOOKS)
        .map(|cb| {
            CodebookTable::new(
                ROWS,
                D_MODEL,
                tables_flat[cb * per..(cb + 1) * per].to_vec(),
            )
            .expect("table")
        })
        .collect();

    let got = mimi_rvq_decode(&codes, TIME, &tables, &attrs).expect("decode");

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
        "mimi reference parity: max|Δ| = {max_abs:.3e} at flat index {max_at} \
         (t={}, d={}) — ATOL {ATOL:.1e}",
        max_at / D_MODEL,
        max_at % D_MODEL
    );
    assert!(
        max_abs <= ATOL,
        "Mimi decode diverged from the Kyutai reference: max|Δ| = {max_abs:.3e} > {ATOL:.1e}"
    );
}
