//! DeBERTa v3 synthetic-weight structure test.
//! Clean-room per arXiv:2111.09543 + HF transformers deberta_v3 (Apache-2.0).
//!
//! # NOT REFERENCED
//!
//! - github.com/litagin02/Style-Bert-VITS2 (AGPL-3.0)

use vokra_bert::deberta_v3::DebertaV3Encoder;

#[test]
fn deberta_v3_forward_shape() {
    let enc = DebertaV3Encoder::synthetic_for_test(2, 8, 2, 16, 512);
    let out = enc.forward(&[1, 2, 3, 4]);
    assert_eq!(out.len(), 4 * 8);
}
