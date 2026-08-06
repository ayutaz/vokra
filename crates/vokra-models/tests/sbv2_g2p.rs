//! SBV2 module scaffold + G2P wrapper tests (Task 14 synthetic mapping +
//! Task 15 real piper-plus G2P routing + Task 7 fixture-driven parity
//! bypass).
//!
//! The first three tests exercise `SbV2Phonemizer::synthetic_for_test()` — a
//! deterministic JA/EN char-level mapping that proves the module wiring
//! without depending on a real G2P instance or a real SBV2 phoneme table.
//! The fourth test (`wired_with_passthrough_phonemizer`, Task 15) exercises
//! `SbV2Phonemizer::from_piper_g2p` — the real piper-plus `Phonemizer` trait
//! boundary — via `PassthroughPhonemizer`, proving the real-G2P routing path
//! is actually reached (not merely that it compiles).
//!
//! The `from_fixture_*` tests (Task 7) exercise
//! `SbV2Phonemizer::from_fixture` — the pre-computed `(language, text)`
//! lookup that lets `SbV2Model::from_gguf_with_phonemizer`-loaded models
//! reproduce a Python reference dumper's exact G2P output for a fixed set
//! of test sentences without needing a real 8-language piper-plus G2P
//! in-workspace. They cover: (a) a match returns the pre-computed result
//! verbatim, (b) three distinct miss cases all fail loudly (FR-EX-08:
//! unknown text / wrong language / a text that WOULD match the synthetic
//! char mapping but isn't in the fixture — proving the fixture path never
//! falls through).
//!
//! # M6 refactor note (2026-08-06)
//!
//! `word_boundaries` on [`PhonemizeResult`] is retained even though
//! [`SbV2TextEncoder::forward`](vokra_models::sbv2::SbV2TextEncoder::forward)
//! no longer consumes it (the SBV2 v2 real checkpoint has a
//! `language_embed [3, d_model]` table instead of the design-doc-guessed
//! `wb_embed [2, d_model]` table — see `SbV2TextEncoder`'s design
//! correction). The G2P layer still emits per-phoneme word-boundary flags
//! because they are honest linguistic output of the G2P stage (a future
//! BERT-tokenize helper or a downstream consumer may still want them),
//! and dropping them would silently change the
//! [`PhonemizeFixture`]-driven parity fixture format (`word_boundaries.bin`
//! is already documented in `tests/parity_sbv2_real.rs`'s manifest
//! schema). The tests below therefore still exercise the word-boundary
//! output shape.

use vokra_core::VokraError;
use vokra_models::sbv2::{Language, PhonemizeFixture, PhonemizeResult, SbV2Phonemizer};

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

/// M6 refactor (2026-08-06): `Language::ZH` selects the SBV2 v2 real
/// checkpoint's `enc_p.language_emb.weight` row 2, but no in-crate ZH G2P
/// is wired — [`SbV2Phonemizer::phonemize`] must therefore return a loud
/// [`VokraError::NotImplemented`] on the char-mapping / real-piper paths,
/// never a silent JA fallback (FR-EX-08). The fixture path is unaffected
/// (a caller with pre-computed ZH phoneme ids can still hit
/// `language_id = 2` code paths via [`PhonemizeFixture`]).
#[test]
fn zh_phonemize_fails_loudly_without_fixture() {
    let p = SbV2Phonemizer::synthetic_for_test();
    match p.phonemize("你好", Language::ZH) {
        Ok(res) => panic!(
            "ZH without a fixture must fail loudly (FR-EX-08), not silently fall back to \
             JA/EN char mapping; got {res:?}"
        ),
        Err(VokraError::NotImplemented(msg)) => {
            assert!(
                msg.contains("ZH"),
                "error must name the offending language, got: {msg}"
            );
        }
        Err(other) => panic!("expected VokraError::NotImplemented, got {other:?}"),
    }
}

/// M6 refactor: `Language::language_id` pins the tentative row-ordering
/// convention (`JA = 0, EN = 1, ZH = 2`) that
/// [`SbV2TextEncoder::forward`](vokra_models::sbv2::SbV2TextEncoder::forward)
/// consumes. Pinning it here (a plain enum-value equality check) catches
/// an accidental permutation that would otherwise only manifest as a
/// parity mismatch on a real checkpoint.
#[test]
fn language_id_row_ordering_is_stable() {
    assert_eq!(Language::JA.language_id(), 0);
    assert_eq!(Language::EN.language_id(), 1);
    assert_eq!(Language::ZH.language_id(), 2);
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

// ---------------------------------------------------------------------------
// Task 7: SbV2Phonemizer::from_fixture — pre-computed (language, text) lookup
// ---------------------------------------------------------------------------

/// Small helper for the fixture tests below: builds a distinctive,
/// deterministic [`PhonemizeResult`] so a passing test can only be
/// explained by an actual fixture hit (not a coincidence with either the
/// synthetic char mapping's or piper-plus's output).
///
/// The distinctive ids (7000/7001/...) sit far outside both other paths'
/// output ranges: `synthetic_for_test`'s JA/EN maps only reach ids in the
/// `100-226` band (see `g2p.rs`), and `PassthroughPhonemizer` framed by a
/// 5-symbol table only reaches ids `0..=4` — so a `phoneme_ids[0] == 7000`
/// assertion below is a load-bearing routing check, not a tautology.
fn distinctive_result(bert_text: &str) -> PhonemizeResult {
    PhonemizeResult {
        phoneme_ids: vec![7000, 7001, 7002, 7003, 7004],
        tones: vec![1, 2, 0, 2, 1],
        word_boundaries: vec![true, false, true, false, true],
        bert_input_text: bert_text.to_string(),
    }
}

/// Task 7: a `(language, text)` pair present in the fixture returns the
/// stored [`PhonemizeResult`] verbatim (Vec-for-Vec equal, not a
/// close-enough shape check) — proving the fixture path actually runs.
#[test]
fn from_fixture_returns_precomputed_result_for_matching_text() {
    let mut fixture = PhonemizeFixture::new();
    let stored = distinctive_result("こんにちは");
    fixture.insert(Language::JA, "こんにちは", stored.clone());

    let phonemizer = SbV2Phonemizer::from_fixture(fixture);
    let got = phonemizer
        .phonemize("こんにちは", Language::JA)
        .expect("fixture hit must succeed");

    assert_eq!(got.phoneme_ids, stored.phoneme_ids);
    assert_eq!(got.tones, stored.tones);
    assert_eq!(got.word_boundaries, stored.word_boundaries);
    assert_eq!(got.bert_input_text, stored.bert_input_text);
}

/// Task 7: a `(language, text)` pair absent from the fixture returns
/// [`VokraError::InvalidArgument`] (FR-EX-08 loud refusal), never a silent
/// success with wrong/plausible-looking ids.
#[test]
fn from_fixture_errors_loudly_on_unknown_text() {
    let mut fixture = PhonemizeFixture::new();
    fixture.insert(Language::JA, "こんにちは", distinctive_result("こんにちは"));

    let phonemizer = SbV2Phonemizer::from_fixture(fixture);
    match phonemizer.phonemize("さようなら", Language::JA) {
        Ok(res) => {
            panic!("unknown text must fail loudly (FR-EX-08), not silently succeed with {res:?}")
        }
        Err(VokraError::InvalidArgument(msg)) => {
            assert!(
                msg.contains("no fixture entry"),
                "error must name the fixture-miss condition, got: {msg}"
            );
            assert!(
                msg.contains("さようなら"),
                "error must include the requested text for actionability, got: {msg}"
            );
        }
        Err(other) => panic!("expected VokraError::InvalidArgument, got {other:?}"),
    }
}

/// Task 7: a fixture entry stored under one language does not silently
/// satisfy a lookup under the other language — even for identical `text`.
/// Proves `(language, text)` really is a compound key.
#[test]
fn from_fixture_errors_loudly_on_wrong_language() {
    let mut fixture = PhonemizeFixture::new();
    fixture.insert(Language::JA, "hello", distinctive_result("hello"));

    let phonemizer = SbV2Phonemizer::from_fixture(fixture);
    match phonemizer.phonemize("hello", Language::EN) {
        Ok(res) => panic!(
            "same text but wrong language must fail loudly (FR-EX-08), not silently succeed \
             with {res:?}"
        ),
        Err(VokraError::InvalidArgument(msg)) => {
            assert!(
                msg.contains("EN"),
                "error must name the requested language, got: {msg}"
            );
        }
        Err(other) => panic!("expected VokraError::InvalidArgument, got {other:?}"),
    }
}

/// Task 7: the fixture path never falls through to the synthetic
/// char-mapping (or piper-plus) paths on a miss — even for input text that
/// WOULD have produced a valid result via `synthetic_for_test`'s hiragana
/// table (i.e. "こんにちは", the trailing sequence of that table). Proves
/// the priority-order dispatch documented on
/// [`SbV2Phonemizer::phonemize`].
#[test]
fn from_fixture_never_falls_through_to_synthetic_char_mapping() {
    // Deliberately empty fixture: no entries at all. If the fixture path
    // were to silently fall through to the synthetic char mapping on a
    // miss, "こんにちは" would produce non-empty phoneme_ids (the
    // synthetic table literally spells "...わをんこんにちは" in its final
    // slots — see `g2p.rs`'s `synthetic_for_test`).
    let fixture = PhonemizeFixture::new();
    assert!(fixture.is_empty(), "test setup: fixture must start empty");

    let phonemizer = SbV2Phonemizer::from_fixture(fixture);
    match phonemizer.phonemize("こんにちは", Language::JA) {
        Ok(res) => panic!(
            "empty fixture must not fall through to synthetic_for_test's char mapping \
             (FR-EX-08); instead returned {res:?}"
        ),
        Err(VokraError::InvalidArgument(_)) => { /* expected */ }
        Err(other) => panic!("expected VokraError::InvalidArgument, got {other:?}"),
    }
}

/// Task 7: `PhonemizeFixture::insert` returns the replaced value when the
/// same `(language, text)` key is overwritten — this proves the fixture is
/// actually keyed on `(language, text)` and not on something narrower
/// (e.g. only `text`). Also proves the second insert's value is what the
/// phonemizer subsequently returns.
#[test]
fn fixture_insert_overwrites_and_returns_prior_value() {
    let mut fixture = PhonemizeFixture::new();
    assert_eq!(fixture.len(), 0);
    let first = PhonemizeResult {
        phoneme_ids: vec![1, 2, 3],
        tones: vec![0, 0, 0],
        word_boundaries: vec![true, false, false],
        bert_input_text: "test".to_string(),
    };
    assert!(
        fixture
            .insert(Language::JA, "test", first.clone())
            .is_none()
    );
    assert_eq!(fixture.len(), 1);

    let second = distinctive_result("test");
    let replaced = fixture
        .insert(Language::JA, "test", second.clone())
        .expect("second insert with same key must return the first");
    assert_eq!(replaced.phoneme_ids, first.phoneme_ids);
    assert_eq!(fixture.len(), 1);

    let phonemizer = SbV2Phonemizer::from_fixture(fixture);
    let got = phonemizer
        .phonemize("test", Language::JA)
        .expect("fixture hit must succeed");
    assert_eq!(
        got.phoneme_ids, second.phoneme_ids,
        "phonemize must reflect the latest inserted value"
    );
}
