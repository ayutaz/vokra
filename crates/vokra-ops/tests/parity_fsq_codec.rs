//! FSQ codec family decode vs pinned upstream references (M4-16 T10/T11 +
//! NanoCodec Group-FSQ #45, NFR-QL-01 / FR-OP-31).
//!
//! **No pretrained WavTokenizer / X-Codec 2 weight is downloaded or used in
//! the fixture pipeline** — the committed fixtures are synthetic (seeded),
//! the *reference code paths* are the upstream ones (M4-04 EnCodec T15
//! precedent; real-checkpoint parity is the flip-the-switch harness of the
//! future model-integration WPs, owner-side):
//!
//! - `wavtokenizer/` — reference = `torch.nn.functional.embedding(codes,
//!   table)`, which is the **verbatim call** WavTokenizer's decode makes
//!   (`encoder/quantization/core_vq.py`:
//!   `EuclideanCodebook.dequantize = F.embedding(embed_ind, self.embed)`;
//!   `n_q = 1`, `project_out = Identity` — ADR M4-16 §D-c). The upstream
//!   repo is not pip-packaged, so the fixture drives the exact torch API its
//!   decode reduces to; the sliced synthetic table (rows 256 × d 64 vs the
//!   released 4096 × 512) is a complete op-semantics fixture because the
//!   decode is a pure gather (mimi_dump.py slicing rationale).
//! - `xcodec2/` — reference = **`vector-quantize-pytorch==1.17.8`** (the
//!   exact `xcodec2==0.1.5` pin) `ResidualFSQ(dim, levels=[4;8],
//!   num_quantizers=1).get_output_from_indices(...)` — the same module +
//!   public API `modeling_xcodec2.py::decode_code` calls, with a seeded
//!   random `project_out` (d_model sliced to 64; the **levels tuple is the
//!   real [4; 8]** and the full 65536 code range is exercised).
//!
//! Regenerate (see `tools/parity/fsq_dump.py`; venv with the pins above):
//!
//! ```text
//! venv/bin/python tools/parity/fsq_dump.py wavtokenizer \
//!     --out tests/parity/fsq/wavtokenizer --seed 0 --time 32 --rows 256 --d-model 64
//! venv/bin/python tools/parity/fsq_dump.py xcodec2 \
//!     --out tests/parity/fsq/xcodec2 --seed 0 --time 32 --d-model 64
//! ```
//!
//! # Tolerance (honest bound)
//!
//! - `wavtokenizer_vq`: both sides are a pure row gather — expected and
//!   **measured bit-identical** (`max|Δ| = 0.0e0`, printed by the test).
//!   `ATOL = 1e-6` retains headroom for a future torch changing its
//!   embedding copy path (it cannot round, so any non-zero delta means a
//!   real bug).
//! - `xcodec2_fsq`: grid values are exact dyadic rationals (`(l−half)/half`
//!   with power-of-two halves for L=4), and the measured gap on this
//!   fixture is `max|Δ| = 0.0e0` (**bit-identical**, 2026-07-15 generation,
//!   printed by the test) — torch's Linear on the 8-wide input folds in the
//!   same order as the Rust sequential FP32 loop at this shape.
//!   `ATOL = 1e-5` retains headroom for a future torch/BLAS build blocking
//!   the GEMV differently on regeneration, while staying 1000x under the
//!   design-wide FP32 `atol = 0.01` (NFR-QL-01).

use vokra_ops::{
    CodebookTable, FsqOutProj, WavTokenizerVqAttrs, Xcodec2FsqAttrs, group_fsq_decode,
    group_fsq_decode_into, wavtokenizer_vq_decode, xcodec2_fsq_decode,
};

fn fixtures_dir(sub: &str) -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("parity")
        .join("fsq")
        .join(sub)
}

fn read_f32(sub: &str, name: &str) -> Vec<f32> {
    let path = fixtures_dir(sub).join(name);
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
    assert_eq!(bytes.len() % 4, 0);
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn read_u32(sub: &str, name: &str) -> Vec<u32> {
    let path = fixtures_dir(sub).join(name);
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
    assert_eq!(bytes.len() % 4, 0);
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn read_text(sub: &str, name: &str) -> String {
    let path = fixtures_dir(sub).join(name);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path:?}: {e}"))
}

fn max_abs_delta(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len());
    got.iter()
        .zip(want.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max)
}

// ---------------------------------------------------------------------------
// T10 — wavtokenizer_vq vs torch F.embedding (the verbatim upstream gather)
// ---------------------------------------------------------------------------

const WT_ROWS: usize = 256;
const WT_D_MODEL: usize = 64;
const WT_TIME: usize = 32;
/// Pure gather — measured 0.0 (bit-identical); any non-zero delta is a bug.
const WT_ATOL: f32 = 1e-6;

#[test]
fn wavtokenizer_vq_decode_matches_torch_embedding_reference_on_synthetic_table() {
    let table_flat = read_f32("wavtokenizer", "codebook_table_sliced.f32");
    assert_eq!(table_flat.len(), WT_ROWS * WT_D_MODEL);
    let codes = read_u32("wavtokenizer", "codes.u32");
    assert_eq!(codes.len(), WT_TIME);
    let reference = read_f32("wavtokenizer", "decoded_features.f32");
    assert_eq!(reference.len(), WT_TIME * WT_D_MODEL);

    let attrs = WavTokenizerVqAttrs {
        vocab_size: WT_ROWS,
        d_model: WT_D_MODEL,
    };
    let table = CodebookTable::new(WT_ROWS, WT_D_MODEL, table_flat).expect("table");
    let got = wavtokenizer_vq_decode(&codes, WT_TIME, &table, &attrs).expect("decode");

    let delta = max_abs_delta(&got, &reference);
    println!("wavtokenizer_vq parity max|Δ| = {delta:.3e} (ATOL = {WT_ATOL:.1e})");
    assert!(
        delta <= WT_ATOL,
        "wavtokenizer_vq parity FAIL: max|Δ| = {delta:.3e} > ATOL {WT_ATOL:.1e} — a pure \
         gather must be bit-identical; investigate before touching the tolerance \
         (fabricated pass 禁止)",
    );
}

// ---------------------------------------------------------------------------
// T11 — xcodec2_fsq vs vector-quantize-pytorch 1.17.8 (the xcodec2 pin)
// ---------------------------------------------------------------------------

const FSQ_N_DIMS: usize = 8;
const FSQ_D_MODEL: usize = 64;
const FSQ_TIME: usize = 32;
/// Measured 0.0 (bit-identical; module docs). Non-zero headroom retained for
/// torch/BLAS GEMV-blocking drift on regeneration; 1000x under the
/// design-wide atol 0.01.
const FSQ_ATOL: f32 = 1e-5;

#[test]
fn xcodec2_fsq_decode_matches_vq_pytorch_1_17_8_reference_on_synthetic_projection() {
    // The levels tuple is pinned by the manifest generator to the real
    // X-Codec 2 [4; 8] (ADR M4-16 §D-c); the committed file makes the pin
    // machine-checkable here so a drifted regeneration cannot silently
    // change the attrs under the test.
    let levels_raw = read_u32("xcodec2", "levels.u32");
    assert_eq!(
        levels_raw,
        vec![4u32; FSQ_N_DIMS],
        "fixture levels tuple must be the released X-Codec 2 [4; 8]",
    );

    let weight = read_f32("xcodec2", "out_proj_weight.f32");
    assert_eq!(weight.len(), FSQ_D_MODEL * FSQ_N_DIMS);
    let bias = read_f32("xcodec2", "out_proj_bias.f32");
    assert_eq!(bias.len(), FSQ_D_MODEL);
    let codes = read_u32("xcodec2", "codes.u32");
    assert_eq!(codes.len(), FSQ_TIME);
    let reference = read_f32("xcodec2", "decoded_features.f32");
    assert_eq!(reference.len(), FSQ_TIME * FSQ_D_MODEL);

    // Codes must exercise the full 65536 range (the FR-OP-31 "65k+ vocab"
    // claim is about the effective vocab = Π levels — assert the fixture
    // actually reaches the top quartile so the mixed-radix decompose of the
    // high dims is exercised, not just dim 0).
    assert!(
        codes.iter().any(|&c| c >= 49_152),
        "fixture codes must reach the top quartile of the 65536 vocab",
    );

    let attrs = Xcodec2FsqAttrs {
        levels: levels_raw,
        d_model: FSQ_D_MODEL,
    };
    assert_eq!(attrs.effective_vocab().expect("vocab"), 65_536);
    let proj = FsqOutProj::new(FSQ_D_MODEL, FSQ_N_DIMS, weight, bias).expect("proj");
    let got = xcodec2_fsq_decode(&codes, FSQ_TIME, Some(&proj), &attrs).expect("decode");

    let delta = max_abs_delta(&got, &reference);
    println!("xcodec2_fsq parity max|Δ| = {delta:.3e} (ATOL = {FSQ_ATOL:.1e})");
    assert!(
        delta <= FSQ_ATOL,
        "xcodec2_fsq parity FAIL: max|Δ| = {delta:.3e} > ATOL {FSQ_ATOL:.1e} — measured \
         bit-identical (0.0) at generation; a larger gap means the grid decompose or the \
         GEMV drifted from the vector-quantize-pytorch 1.17.8 pin (fabricated pass 禁止)",
    );
}

// ---------------------------------------------------------------------------
// #45 — grouped FSQ vs the quantizer restored from NVIDIA's real checkpoint
// ---------------------------------------------------------------------------

const NANOCODEC_GROUPS: usize = 4;
const NANOCODEC_DIMS_PER_GROUP: usize = 4;
const NANOCODEC_TIME: usize = 16;
const NANOCODEC_ATOL: f32 = 1e-6;

#[test]
fn group_fsq_matches_nemo_real_checkpoint_quantizer_reference() {
    let levels = read_u32("nanocodec", "levels_per_group.u32");
    assert_eq!(
        levels,
        vec![9, 8, 8, 7],
        "fixture must pin the official 0.6 kbps checkpoint's levels",
    );
    let codes = read_u32("nanocodec", "codes_time_group.u32");
    assert_eq!(codes.len(), NANOCODEC_TIME * NANOCODEC_GROUPS);
    assert!(
        codes.contains(&4031),
        "fixture must exercise the top valid code in the real 4032-entry codebook",
    );
    let reference = read_f32("nanocodec", "decoded_features.f32");
    assert_eq!(
        reference.len(),
        NANOCODEC_TIME * NANOCODEC_GROUPS * NANOCODEC_DIMS_PER_GROUP,
    );

    let got = group_fsq_decode(&codes, NANOCODEC_TIME, &levels).expect("group FSQ decode");
    let mut into = vec![f32::NAN; reference.len()];
    group_fsq_decode_into(&codes, NANOCODEC_TIME, &levels, &mut into)
        .expect("group FSQ decode into");
    assert_eq!(
        got, into,
        "allocating and caller-owned APIs must be identical"
    );

    let delta = max_abs_delta(&got, &reference);
    println!("NanoCodec Group-FSQ parity max|Δ| = {delta:.3e} (ATOL = {NANOCODEC_ATOL:.1e})");
    assert!(
        delta <= NANOCODEC_ATOL,
        "NanoCodec Group-FSQ parity FAIL: max|Δ| = {delta:.3e} > ATOL \
         {NANOCODEC_ATOL:.1e}; inspect the worst group/dimension before changing the bound",
    );

    let manifest = read_text("nanocodec", "manifest.txt");
    for required in [
        "nvidia/nemo-nano-codec-22khz-0.6kbps-12.5fps",
        "repo commit: 5c8e22ed763c14d81337fbe6ca74062f3d10f7e5",
        "checkpoint sha256: bd5883099d0c74ceda760b6b7a1600b86da4d8a02531c9c282679951dcb08870",
        "NVIDIA-NeMo/Speech commit: 4fcff72febec9395fdbd4bfa0747bfda2ecd3cef",
        "oracle class: nemo.collections.tts.modules.audio_codec_modules.GroupFiniteScalarQuantizer",
        "fixture kind: real checkpoint-loaded quantizer; deterministic synthetic indices",
    ] {
        assert!(
            manifest.contains(required),
            "manifest missing provenance pin: {required}"
        );
    }
}

// ---------------------------------------------------------------------------
// Fixture hygiene — the compliance scanner must stay inert on these files
// (mirror of the M4-04 T15 assertion; fixtures are .f32/.u32/.txt only).
// ---------------------------------------------------------------------------

#[test]
fn fsq_fixture_dir_contains_no_weight_extension_files() {
    for sub in ["wavtokenizer", "xcodec2", "nanocodec"] {
        let dir = fixtures_dir(sub);
        for entry in std::fs::read_dir(&dir).unwrap_or_else(|e| panic!("read_dir {dir:?}: {e}")) {
            let name = entry.expect("entry").file_name();
            let name = name.to_string_lossy();
            for bad in [".safetensors", ".gguf", ".pth", ".bin"] {
                assert!(
                    !name.ends_with(bad),
                    "fixture {name} in {sub}/ must not use a weight extension ({bad})",
                );
            }
        }
    }
}
