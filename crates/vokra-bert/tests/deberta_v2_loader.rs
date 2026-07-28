//! DeBERTa v2 GGUF loader test. Clean-room per arXiv:2006.03654 +
//! HF transformers deberta_v2 (Apache-2.0).
//!
//! # NOT REFERENCED
//!
//! - github.com/litagin02/Style-Bert-VITS2 (AGPL-3.0)

use std::path::{Path, PathBuf};

use vokra_bert::deberta_v2::DebertaV2Encoder;
use vokra_core::gguf::GgufFile;

/// Repo-root-relative real-fixture directory for the DeBERTa v2/v3 GGUF
/// fixtures shared with the SBV2 v2 loader/parity tests
/// (`tests/fixtures/sbv2/`, gated by the committed `*.gguf.sha256`
/// sidecars). `CARGO_MANIFEST_DIR` is `<repo>/crates/vokra-bert` — `cargo
/// test` sets a test binary's working directory to the crate root, not the
/// invocation directory, so every repo-root fixture path in this workspace
/// is built this way (`parity_sbv2_real.rs`, `parity_whisper.rs`,
/// `parity_kokoro.rs`, `parity_voxtral.rs`, `parity_csm.rs`,
/// `parity_moshi.rs`) rather than as a bare relative literal.
fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("fixtures")
        .join("sbv2")
}

#[test]
#[ignore = "requires real deberta-v2 GGUF fixture, gated by tests/fixtures/sbv2/*.gguf.sha256"]
fn load_real_deberta_v2_ja() {
    let path = fixtures_dir().join("deberta-v2-large-japanese-char-wwm.gguf");
    let g = GgufFile::open(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let enc = DebertaV2Encoder::from_gguf(&g)
        .unwrap_or_else(|e| panic!("DebertaV2Encoder::from_gguf: {e}"));
    let out = enc.forward(&[2, 100, 200, 3]); // <s> ... </s>
    assert!(!out.is_empty());
}
