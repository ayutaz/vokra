//! Real-checkpoint codec parity harness (M4-04 T16 — consumed by
//! `.github/workflows/parity-rvq-real.yml`; owner-dispatched, never a
//! required check).
//!
//! **Env-gated** (repo convention for real-GGUF parity): each test returns
//! early with a loud skip note unless its env vars point at artefacts the
//! workflow (or a local operator) prepared:
//!
//! - `VOKRA_MIMI_GGUF` + `VOKRA_MIMI_REF_DIR` — a `vokra-cli convert --model
//!   mimi` output GGUF + a `mimi_dump.py mimi` reference dir (full-table run:
//!   `--rows <codebook_size> --books <n>`);
//! - `VOKRA_DAC_GGUF` + `VOKRA_DAC_REF_DIR` — same for DAC.
//!
//! What it proves beyond the committed sliced fixtures (T13/T14): the
//! **converter runs end-to-end on the real checkpoint** and the GGUF-loaded
//! tables/projections reproduce the upstream reference decode over the FULL
//! tables (per-tensor max |Δ| table printed for the step summary;
//! `atol = 0.01` FP32, NFR-QL-01 — one tensor over budget fails the test,
//! fabricated pass forbidden).

use vokra_core::gguf::GgufFile;
use vokra_models::codec::{DacCodecGguf, MimiCodecGguf};
use vokra_ops::{CodebookTable, MimiRvqAttrs, dac_rvq_decode, mimi_rvq_decode};

const ATOL: f32 = 0.01;

fn read_f32(path: &std::path::Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
    assert_eq!(bytes.len() % 4, 0, "{path:?}: not a whole number of f32");
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn read_u32(path: &std::path::Path) -> Vec<u32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
    assert_eq!(bytes.len() % 4, 0, "{path:?}: not a whole number of u32");
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "length mismatch");
    a.iter()
        .zip(b.iter())
        .fold(0.0_f32, |m, (x, y)| m.max((x - y).abs()))
}

/// One row of the per-tensor verdict table. Printed as
/// `PARITY <name> max|Δ|=<v> atol=<a> <PASS|FAIL>` so the workflow can grep
/// the lines into `$GITHUB_STEP_SUMMARY`.
fn report(name: &str, delta: f32, atol: f32) -> bool {
    let pass = delta <= atol;
    eprintln!(
        "PARITY {name} max|Δ|={delta:.3e} atol={atol:.1e} {}",
        if pass { "PASS" } else { "FAIL" }
    );
    pass
}

#[test]
fn real_mimi_gguf_full_table_parity() {
    let (Ok(gguf_path), Ok(ref_dir)) = (
        std::env::var("VOKRA_MIMI_GGUF"),
        std::env::var("VOKRA_MIMI_REF_DIR"),
    ) else {
        eprintln!(
            "real_mimi_gguf_full_table_parity: SKIP — set VOKRA_MIMI_GGUF + \
             VOKRA_MIMI_REF_DIR to run (parity-rvq-real.yml owner workflow)"
        );
        return;
    };
    let ref_dir = std::path::PathBuf::from(ref_dir);

    let file = GgufFile::open(&gguf_path).expect("open mimi GGUF");
    let codec = MimiCodecGguf::from_gguf(&file).expect("bind mimi GGUF");

    // Reference dump shapes: infer books from codes width, rows from table
    // length (the workflow runs mimi_dump.py with full rows).
    let ref_tables = read_f32(&ref_dir.join("codebook_tables_sliced.f32"));
    let codes = read_u32(&ref_dir.join("codes.u32"));
    let reference = read_f32(&ref_dir.join("decoded_features.f32"));

    let d_model = codec.attrs.d_model;
    let time = reference.len() / d_model;
    assert_eq!(reference.len() % d_model, 0);
    let books = codes.len() / time;
    assert_eq!(codes.len() % time, 0);
    assert!(
        books <= codec.attrs.n_codebooks,
        "reference uses {books} books but the GGUF carries {}",
        codec.attrs.n_codebooks
    );
    let rows = ref_tables.len() / (books * d_model);
    assert_eq!(ref_tables.len(), books * rows * d_model);
    assert!(
        rows <= codec.attrs.codebook_size,
        "reference rows {rows} > GGUF codebook_size {}",
        codec.attrs.codebook_size
    );

    let mut all_pass = true;

    // Per-tensor: GGUF derived tables vs the upstream-computed effective
    // tables (prefix rows of the prefix books).
    for cb in 0..books {
        let ref_t = &ref_tables[cb * rows * d_model..(cb + 1) * rows * d_model];
        let gguf_t = &codec.tables[cb].data[..rows * d_model];
        let delta = max_abs_diff(gguf_t, ref_t);
        all_pass &= report(&format!("mimi.codebook_tables[{cb}]"), delta, ATOL);
    }

    // End-to-end decode over the GGUF-loaded tables (prefix attrs).
    let attrs = MimiRvqAttrs {
        n_codebooks: books,
        codebook_size: codec.attrs.codebook_size,
        d_model,
    };
    let tables: Vec<CodebookTable> = codec.tables[..books].to_vec();
    let got = mimi_rvq_decode(&codes, time, &tables, &attrs).expect("decode");
    all_pass &= report(
        "mimi.decoded_features",
        max_abs_diff(&got, &reference),
        ATOL,
    );

    assert!(all_pass, "at least one Mimi tensor exceeded atol={ATOL}");
}

#[test]
fn real_dac_gguf_full_table_parity() {
    let (Ok(gguf_path), Ok(ref_dir)) = (
        std::env::var("VOKRA_DAC_GGUF"),
        std::env::var("VOKRA_DAC_REF_DIR"),
    ) else {
        eprintln!(
            "real_dac_gguf_full_table_parity: SKIP — set VOKRA_DAC_GGUF + \
             VOKRA_DAC_REF_DIR to run (parity-rvq-real.yml owner workflow)"
        );
        return;
    };
    let ref_dir = std::path::PathBuf::from(ref_dir);

    let file = GgufFile::open(&gguf_path).expect("open dac GGUF");
    let codec = DacCodecGguf::from_gguf(&file).expect("bind dac GGUF");

    let ref_tables = read_f32(&ref_dir.join("codebook_tables_sliced.f32"));
    let ref_w = read_f32(&ref_dir.join("out_proj_weight.f32"));
    let ref_b = read_f32(&ref_dir.join("out_proj_bias.f32"));
    let codes = read_u32(&ref_dir.join("codes.u32"));
    let reference = read_f32(&ref_dir.join("decoded_features.f32"));

    let d_model = codec.attrs.d_model;
    let dim = codec.attrs.codebook_dim;
    let time = reference.len() / d_model;
    let books = codes.len() / time;
    assert!(books <= codec.attrs.n_codebooks);
    let rows = ref_tables.len() / (books * dim);
    assert_eq!(ref_tables.len(), books * rows * dim);
    assert_eq!(ref_w.len(), books * d_model * dim);
    assert_eq!(ref_b.len(), books * d_model);

    let mut all_pass = true;
    for cb in 0..books {
        let delta = max_abs_diff(
            &codec.tables[cb].data[..rows * dim],
            &ref_tables[cb * rows * dim..(cb + 1) * rows * dim],
        );
        all_pass &= report(&format!("dac.codebook[{cb}]"), delta, ATOL);

        // The converter's offline weight-norm fold vs torch's own effective
        // weight — the tensor where a fold-formula bug would show first.
        let delta = max_abs_diff(
            &codec.out_projs[cb].weight,
            &ref_w[cb * d_model * dim..(cb + 1) * d_model * dim],
        );
        all_pass &= report(&format!("dac.out_proj_weight[{cb}]"), delta, ATOL);

        let delta = max_abs_diff(
            &codec.out_projs[cb].bias,
            &ref_b[cb * d_model..(cb + 1) * d_model],
        );
        all_pass &= report(&format!("dac.out_proj_bias[{cb}]"), delta, ATOL);
    }

    let attrs = vokra_ops::DacRvqAttrs {
        n_codebooks: books,
        codebook_size: codec.attrs.codebook_size,
        codebook_dim: dim,
        d_model,
    };
    let got = dac_rvq_decode(
        &codes,
        time,
        &codec.tables[..books],
        &codec.out_projs[..books],
        &attrs,
    )
    .expect("decode");
    all_pass &= report("dac.decoded_features", max_abs_diff(&got, &reference), ATOL);

    assert!(all_pass, "at least one DAC tensor exceeded atol={ATOL}");
}
