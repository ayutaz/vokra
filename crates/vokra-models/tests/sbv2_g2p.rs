//! SBV2 module scaffold + G2P wrapper tests (Task 14, synthetic mapping).
//!
//! Exercises `SbV2Phonemizer::synthetic_for_test()` — a deterministic
//! JA/EN char-level mapping used only to prove the module wiring before the
//! real piper-plus G2P bridge lands (Task 15).

use vokra_models::sbv2::{Language, SbV2Phonemizer};

#[test]
fn ja_phonemize_produces_ids() {
    let p = SbV2Phonemizer::synthetic_for_test();
    let r = p.phonemize("こんにちは", Language::JA).expect("phonemize");
    assert!(!r.phoneme_ids.is_empty());
    assert_eq!(r.tones.len(), r.phoneme_ids.len());
    assert_eq!(r.word_boundaries.len(), r.phoneme_ids.len());
    assert_eq!(r.bert_input_text, "こんにちは");
}

#[test]
fn en_phonemize_zero_tones() {
    let p = SbV2Phonemizer::synthetic_for_test();
    let r = p.phonemize("hello world", Language::EN).expect("phonemize");
    assert!(r.tones.iter().all(|&t| t == 0), "EN tones must be all zero");
}
