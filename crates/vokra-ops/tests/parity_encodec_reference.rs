//! EnCodec RVQ decode vs the MIT `encodec` reference **code** on
//! **fixed-seed synthetic codebooks** (M4-04 T15, NFR-QL-01 / FR-OP-32).
//!
//! **No pretrained EnCodec weight was downloaded or used anywhere in the
//! fixture pipeline** — the weights are CC-BY-NC 4.0 and permanently
//! zoo-excluded (FR-OP-32 permanent constraint); the dump script builds
//! `EncodecModel.encodec_model_24khz(pretrained=False)` (canonical 24 kHz
//! shapes read from the constructed model: n_q=32, bins=1024, dimension=128)
//! and seed-randomizes the codebooks before decoding through the public
//! `ResidualVectorQuantizer.decode` API. Fixture filenames are
//! `.f32`/`.u32`/`.txt` only, so `scripts/compliance/check-encodec-exclusion.sh`
//! (which matches weight extensions `.safetensors/.gguf/.pth/.bin`) stays
//! inert on them — asserted below. Regenerate:
//!
//! ```text
//! venv/bin/python tools/parity/mimi_dump.py encodec --out tests/parity/encodec \
//!     --seed 0 --time 32 --rows 128 --books 8
//! ```
//!
//! # Tolerance (honest bound)
//!
//! EnCodec's decode is a plain per-layer gather + elementwise residual sum in
//! codebook order — the same fold order as `encodec_rvq_decode` — so the two
//! sides are expected to agree to the last bit; measured `max|Δ| = 0.0e0`
//! (bit-identical) on this fixture (2026-07-15 generation, printed by the
//! test). `ATOL = 1e-5` guards against a future torch changing its
//! elementwise-add order while staying 1000x under the design-wide FP32
//! `atol = 0.01`.

use vokra_ops::{CodebookTable, EncodecRvqAttrs, encodec_rvq_decode};

const BOOKS: usize = 8;
const ROWS: usize = 128;
const D_MODEL: usize = 128;
const TIME: usize = 32;

/// Measured 0.0 (bit-identical; module docs). Non-zero head-room retained for
/// torch elementwise-order drift on regeneration.
const ATOL: f32 = 1e-5;

fn fixtures_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("parity")
        .join("encodec")
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

#[test]
fn encodec_rvq_decode_matches_mit_reference_code_on_synthetic_codebooks() {
    let tables_flat = read_f32("codebook_tables_sliced.f32");
    assert_eq!(tables_flat.len(), BOOKS * ROWS * D_MODEL);
    let codes = read_u32("codes.u32");
    assert_eq!(codes.len(), TIME * BOOKS);
    let reference = read_f32("decoded_features.f32");
    assert_eq!(reference.len(), TIME * D_MODEL);

    let attrs = EncodecRvqAttrs {
        n_codebooks: BOOKS,
        codebook_size: ROWS,
        d_model: D_MODEL,
    };
    let per = ROWS * D_MODEL;
    let tables: Vec<CodebookTable> = (0..BOOKS)
        .map(|cb| {
            CodebookTable::new(
                ROWS,
                D_MODEL,
                tables_flat[cb * per..(cb + 1) * per].to_vec(),
            )
            .expect("table")
        })
        .collect();

    let got = encodec_rvq_decode(&codes, TIME, &tables, &attrs).expect("decode");

    let mut max_abs = 0.0_f32;
    for (g, r) in got.iter().zip(reference.iter()) {
        max_abs = max_abs.max((g - r).abs());
    }
    eprintln!("encodec synthetic parity: max|Δ| = {max_abs:.3e} — ATOL {ATOL:.1e}");
    assert!(
        max_abs <= ATOL,
        "EnCodec decode diverged from the MIT reference code: max|Δ| = {max_abs:.3e} > {ATOL:.1e}"
    );
}

#[test]
fn fixture_filenames_stay_inert_for_the_encodec_exclusion_gate() {
    // FR-OP-32 integration point: the distribution gate
    // (`scripts/compliance/check-encodec-exclusion.sh`) flags files whose
    // *filename* contains "encodec" AND ends in a weight extension. This
    // fixture directory is literally named `encodec/`, so pin that none of
    // its files ever carries a weight extension — a `.pth`/`.safetensors`/
    // `.gguf`/`.bin` appearing here would trip the release gate (correctly!)
    // and would also mean someone put a real weight file where only synthetic
    // `.f32`/`.u32`/`.txt` fixtures belong.
    let dir = fixtures_dir();
    let mut n = 0usize;
    for entry in std::fs::read_dir(&dir).unwrap_or_else(|e| panic!("read_dir {dir:?}: {e}")) {
        let name = entry.expect("entry").file_name();
        let name = name.to_string_lossy();
        for banned in [".safetensors", ".gguf", ".pth", ".bin"] {
            assert!(
                !name.to_ascii_lowercase().ends_with(banned),
                "weight-extension file `{name}` in tests/parity/encodec/ — FR-OP-32 forbids \
                 EnCodec weight artefacts anywhere in the shipping tree"
            );
        }
        n += 1;
    }
    assert!(n >= 4, "fixture directory unexpectedly sparse ({n} files)");
}
