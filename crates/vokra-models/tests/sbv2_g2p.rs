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

/// Regression: EN word-boundary flags must align to each word start,
/// not drift for inputs with 2+ spaces (bug found in Task 14 review;
/// original truncation logic dropped tail elements to reconcile a
/// per-space phantom push instead of tracking boundaries at correct positions).
#[test]
fn en_phonemize_multiword_word_boundaries_aligned() {
    let p = SbV2Phonemizer::synthetic_for_test();
    let r = p.phonemize("ab cd ef", Language::EN).expect("phonemize");
    // 6 non-space chars → 6 phoneme ids
    assert_eq!(r.phoneme_ids.len(), 6);
    assert_eq!(r.word_boundaries.len(), 6);
    // a, c, e each start a word
    assert_eq!(
        r.word_boundaries,
        vec![true, false, true, false, true, false]
    );
}
