//! Blocker 5 verify (2026-08-06) — real DeBERTa v2 JA + v3 EN GGUFs
//! carry a loadable `vokra.bert.tokenizer.*` chunk group and their ids
//! match the fixtures verified at scout time (v2:
//! `/tmp/sbv2-fixtures/deberta-v2-ja/vocab.txt` +
//! `special_tokens_map.json`; v3: `spm.model` via
//! `tools/parity/extract_spm_metadata.py`).
//!
//! Gated on the same `tests/fixtures/sbv2/*.gguf` fixture family as
//! `sbv2_gguf_loader::from_gguf_loads_real_sbv2_weights`; skips clean
//! when the fixtures are not present (FR-EX-08 does not fire from an
//! absent fixture — it fires from a broken schema).

use std::path::{Path, PathBuf};

use vokra_bert::tokenizer::SbertTokenizer;
use vokra_core::gguf::GgufFile;

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("fixtures")
        .join("sbv2")
}

#[test]
#[ignore = "Blocker 5 real fixture: tests/fixtures/sbv2/{deberta-v2-large-japanese-char-wwm,deberta-v3-large}.gguf"]
fn tokenizer_from_gguf_smoke() {
    let dir = fixtures_dir();
    let ja_path = dir.join("deberta-v2-large-japanese-char-wwm.gguf");
    let en_path = dir.join("deberta-v3-large.gguf");
    let ja = GgufFile::open(&ja_path).unwrap_or_else(|e| panic!("{}: {e}", ja_path.display()));
    let en = GgufFile::open(&en_path).unwrap_or_else(|e| panic!("{}: {e}", en_path.display()));
    let ja_tok = SbertTokenizer::from_gguf(&ja, "vokra.bert.tokenizer")
        .unwrap_or_else(|e| panic!("v2 SbertTokenizer::from_gguf: {e}"));
    let en_tok = SbertTokenizer::from_gguf(&en, "vokra.bert.tokenizer")
        .unwrap_or_else(|e| panic!("v3 SbertTokenizer::from_gguf: {e}"));
    // v2 JA (BertJapaneseTokenizer, char-level): unk=3, bos=1, eos=2 —
    // verified against /tmp/sbv2-fixtures/deberta-v2-ja/vocab.txt +
    // special_tokens_map.json at scout time.
    assert_eq!(ja_tok.unk_id(), 3, "v2 JA unk_id");
    assert_eq!(ja_tok.bos_id(), 1, "v2 JA bos_id (= [CLS])");
    assert_eq!(ja_tok.eos_id(), 2, "v2 JA eos_id (= [SEP])");
    // v3 EN (SentencePiece Unigram): unk=3, bos=1, eos=2 — same
    // ordering as v2's char-vocab (both derived from BERT-family
    // conventions), verified against spm.model via
    // tools/parity/extract_spm_metadata.py at scout time.
    assert_eq!(en_tok.unk_id(), 3, "v3 EN unk_id");
    assert_eq!(en_tok.bos_id(), 1, "v3 EN bos_id (= [CLS])");
    assert_eq!(en_tok.eos_id(), 2, "v3 EN eos_id (= [SEP])");
    // Both tokenizers encode without panicking. Byte-level `unk`
    // fallback is exercised at will inside viterbi's inner loop.
    let ja_ids = ja_tok.encode("こんにちは");
    let en_ids = en_tok.encode("hello world");
    assert!(!ja_ids.is_empty(), "JA encoder produced no tokens");
    assert!(!en_ids.is_empty(), "EN encoder produced no tokens");
}
