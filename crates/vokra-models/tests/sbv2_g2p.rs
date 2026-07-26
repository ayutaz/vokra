//! SBV2 module scaffold + G2P wrapper tests (Task 14 synthetic mapping +
//! Task 15 real piper-plus G2P routing).
//!
//! The first three tests exercise `SbV2Phonemizer::synthetic_for_test()` — a
//! deterministic JA/EN char-level mapping that proves the module wiring
//! without depending on a real G2P instance or a real SBV2 phoneme table.
//! The fourth test (`wired_with_passthrough_phonemizer`, Task 15) exercises
//! `SbV2Phonemizer::from_piper_g2p` — the real piper-plus `Phonemizer` trait
//! boundary — via `PassthroughPhonemizer`, proving the real-G2P routing path
//! is actually reached (not merely that it compiles).

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

/// Task 15: `from_piper_g2p` routes phonemize calls through the real
/// piper-plus `Phonemizer` trait boundary (here: `PassthroughPhonemizer`)
/// instead of the synthetic char mapping the tests above exercise.
///
/// `PassthroughPhonemizer` never guesses linguistics — it only parses and
/// frames already-computed phoneme content — so this is a *routing* test,
/// not a G2P-quality test: input `"3 4"` is `PassthroughPhonemizer`'s raw
/// id-sequence form (`crates/vokra-piper-plus/src/phonemizer.rs`), parsed to
/// `[3, 4]` and framed by the table to `[BOS=1, 3, PAD=0, 4, PAD=0, EOS=2]`
/// (6 ids). With an *empty* `en_mapping`, every one of those 6 piper ids
/// falls back to `sbv2_default_phoneme_id` (0) — the documented mapping
/// fallback, not a silent no-op (FR-EX-08); the empty map is this test's
/// deliberate input, matching the brief's "空 mapping = 全 phoneme が
/// default id にfallback" note.
#[test]
fn wired_with_passthrough_phonemizer() {
    use vokra_piper_plus::{PassthroughPhonemizer, PhonemeTable};

    // Minimal 5-symbol voice table: PAD=`_`(0), BOS=`^`(1), EOS=`$`(2), plus
    // two arbitrary phoneme symbols (mirrors vokra-piper-plus's own
    // `phonemizer.rs` test fixture). Sufficient because
    // PassthroughPhonemizer's raw-id-sequence form doesn't validate ids
    // against the symbol table at all (only its bracket-literal `[[symbol]]`
    // form does).
    let symbols = vec![
        "_".to_owned(),
        "^".to_owned(),
        "$".to_owned(),
        "a".to_owned(),
        "i".to_owned(),
    ];
    let table = PhonemeTable::from_symbols(&symbols).expect("valid table");
    let ja = Box::new(PassthroughPhonemizer::new(table.clone()));
    let en = Box::new(PassthroughPhonemizer::new(table));
    let p = SbV2Phonemizer::from_piper_g2p(
        ja,
        en,
        std::collections::HashMap::new(), // ja mapping (empty = all default)
        std::collections::HashMap::new(), // en mapping (empty = all default)
    );

    let r = p.phonemize("3 4", Language::EN).expect("ok");

    // Routing assertions (not tautological): the real Phonemizer ran (6
    // framed ids came back -- BOS/3/PAD/4/PAD/EOS -- not e.g. some count
    // derived from a char map), every id fell back to the default SBV2 id
    // because en_mapping is empty, and ids/tones/word_boundaries agree in
    // length.
    assert_eq!(
        r.phoneme_ids,
        vec![0u16; 6],
        "empty en_mapping -> every piper id falls back to the default"
    );
    assert_eq!(r.tones, vec![0u8; 6], "EN tones are always zero");
    assert_eq!(
        r.word_boundaries,
        vec![true, false, false, false, false, false],
        "conservative rule: only the first emitted phoneme starts a word"
    );
    assert_eq!(r.bert_input_text, "3 4");
}
