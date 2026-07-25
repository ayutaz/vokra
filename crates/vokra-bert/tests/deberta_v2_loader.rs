//! DeBERTa v2 GGUF loader test. Clean-room per arXiv:2006.03654 +
//! HF transformers deberta_v2 (Apache-2.0).
//!
//! # NOT REFERENCED
//!
//! - github.com/litagin02/Style-Bert-VITS2 (AGPL-3.0)

use vokra_bert::deberta_v2::DebertaV2Encoder;
use vokra_core::gguf::GgufFile;

#[test]
#[ignore = "requires real deberta-v2 GGUF fixture, gated by tests/fixtures/sbv2/*.gguf.sha256"]
fn load_real_deberta_v2_ja() {
    let path = "tests/fixtures/sbv2/deberta-v2-large-japanese-char-wwm.gguf";
    let g = GgufFile::open(path).expect("open");
    let enc = DebertaV2Encoder::from_gguf(&g).expect("load");
    let out = enc.forward(&[2, 100, 200, 3]); // <s> ... </s>
    assert!(!out.is_empty());
}
