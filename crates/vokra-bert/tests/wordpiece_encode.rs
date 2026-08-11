//! WordPiece tokenizer unit tests (WP-17 / TDD RED first).
//!
//! Clean-room per Devlin et al. 2018 (arXiv:1810.04805) + public WordPiece
//! (Wu et al. 2016, arXiv:1609.08144). No AGPL Style-Bert-VITS2 sources
//! consulted (see `crates/vokra-bert/src/wordpiece.rs` module doc).
//!
//! Scope: this WP does NOT wire the tokenizer into any live BERT model —
//! WP-19 handles wiring. These tests exercise `BertWordpieceTokenizer`
//! standalone against a synthetic 10-entry vocab.

use vokra_bert::wordpiece::{BertWordpieceTokenizer, OovPolicy};
use vokra_core::gguf::{GgufArray, GgufBuilder, GgufFile, GgufMetadataValue, GgufValueType};

/// Standard BERT special-token ids used by `hfl/chinese-roberta-wwm-ext-large`
/// (and every uncased-base BERT layout): `[PAD]=0`, `[UNK]=100`, `[CLS]=101`,
/// `[SEP]=102`. The synthetic vocab in these tests uses positions matching
/// these ids so the loader/tokenizer contract can be verified without
/// depending on a real 21128-entry vocab.
const PAD_ID: u32 = 0;
const UNK_ID: u32 = 1;
const CLS_ID: u32 = 2;
const SEP_ID: u32 = 3;

/// Ten-entry synthetic vocab exercising every path:
/// - specials at ids 0..=3
/// - full-word English piece "hello"
/// - split English pair "wor" + "##ld" (WordPiece continuation)
/// - Chinese char pieces "你" / "好"
/// - ASCII punctuation "!"
///
/// | id | token   | purpose                                    |
/// |----|---------|--------------------------------------------|
/// | 0  | [PAD]   | pad (never emitted by encode)              |
/// | 1  | [UNK]   | OOV fallback                               |
/// | 2  | [CLS]   | prepended when add_special_tokens=true     |
/// | 3  | [SEP]   | appended when add_special_tokens=true      |
/// | 4  | hello   | English full-word                          |
/// | 5  | wor     | English WordPiece start                    |
/// | 6  | ##ld    | English WordPiece continuation             |
/// | 7  | 你      | Chinese char (single codepoint)            |
/// | 8  | 好      | Chinese char (single codepoint)            |
/// | 9  | !       | ASCII punctuation                          |
fn synthetic_vocab() -> Vec<String> {
    vec![
        "[PAD]".to_string(),
        "[UNK]".to_string(),
        "[CLS]".to_string(),
        "[SEP]".to_string(),
        "hello".to_string(),
        "wor".to_string(),
        "##ld".to_string(),
        "你".to_string(),
        "好".to_string(),
        "!".to_string(),
    ]
}

fn build() -> BertWordpieceTokenizer {
    BertWordpieceTokenizer::from_vocab(synthetic_vocab(), UNK_ID, CLS_ID, SEP_ID, PAD_ID)
        .expect("valid synthetic vocab")
}

// -----------------------------------------------------------------------------
// Construction validation
// -----------------------------------------------------------------------------

#[test]
fn from_vocab_rejects_special_id_out_of_range() {
    let err = BertWordpieceTokenizer::from_vocab(synthetic_vocab(), 999, CLS_ID, SEP_ID, PAD_ID)
        .expect_err("unk_id 999 exceeds vocab size 10");
    let msg = format!("{err}");
    assert!(
        msg.contains("unk_id") || msg.contains("out of range"),
        "loud: {msg}"
    );
}

#[test]
fn from_vocab_rejects_duplicate_entries() {
    let mut v = synthetic_vocab();
    v[5] = "hello".to_string(); // duplicate of id 4
    let err = BertWordpieceTokenizer::from_vocab(v, UNK_ID, CLS_ID, SEP_ID, PAD_ID)
        .expect_err("duplicate 'hello' at ids 4 and 5 must be rejected");
    let msg = format!("{err}");
    assert!(
        msg.contains("duplicate") || msg.contains("hello"),
        "loud: {msg}"
    );
}

#[test]
fn from_vocab_rejects_empty_vocab() {
    let err = BertWordpieceTokenizer::from_vocab(vec![], UNK_ID, CLS_ID, SEP_ID, PAD_ID)
        .expect_err("empty vocab must be rejected");
    let msg = format!("{err}");
    assert!(
        msg.contains("empty") || msg.contains("vocab"),
        "loud: {msg}"
    );
}

// -----------------------------------------------------------------------------
// English WordPiece behavior (subword split + full-word match)
// -----------------------------------------------------------------------------

#[test]
fn encode_english_wordpiece_greedy_longest_match() {
    let tok = build();
    let ids = tok.encode("hello world", false).expect("encode");
    // "hello" -> [4], "world" -> [5, 6] because vocab has "wor" + "##ld"
    // but no "world" full-word entry.
    assert_eq!(ids, vec![4, 5, 6]);
}

#[test]
fn encode_english_add_special_tokens_wraps_with_cls_sep() {
    let tok = build();
    let ids = tok.encode("hello", true).expect("encode");
    assert_eq!(ids, vec![CLS_ID, 4, SEP_ID]);
}

#[test]
fn encode_english_lowercases_by_default() {
    let tok = build();
    // "HELLO" must lowercase to "hello" and hit id 4.
    let ids = tok.encode("HELLO", false).expect("encode");
    assert_eq!(ids, vec![4]);
}

#[test]
fn encode_english_lower_case_can_be_disabled() {
    let tok = build().with_lower_case(false);
    // With lower_case off, "HELLO" != "hello" so wordpiece must fall to UNK.
    let ids = tok.encode("HELLO", false).expect("encode");
    assert_eq!(ids, vec![UNK_ID]);
}

#[test]
fn encode_english_splits_on_punctuation() {
    let tok = build();
    // "hello!" -> ["hello", "!"] -> [4, 9]. Punctuation always splits
    // regardless of case.
    let ids = tok.encode("hello!", false).expect("encode");
    assert_eq!(ids, vec![4, 9]);
}

// -----------------------------------------------------------------------------
// Chinese char-level behavior (each CJK codepoint becomes its own token)
// -----------------------------------------------------------------------------

#[test]
fn encode_chinese_char_level_ni_hao_with_special_tokens() {
    let tok = build();
    // "你好" (2 CJK chars) -> [cls, 你_id, 好_id, sep] per the task spec.
    let ids = tok.encode("你好", true).expect("encode");
    assert_eq!(ids, vec![CLS_ID, 7, 8, SEP_ID]);
}

#[test]
fn encode_chinese_char_level_without_specials() {
    let tok = build();
    let ids = tok.encode("你好", false).expect("encode");
    assert_eq!(ids, vec![7, 8]);
}

#[test]
fn encode_chinese_char_that_is_oov_becomes_unk() {
    let tok = build();
    // "龘" (U+9F98, CJK) is not in the synthetic vocab; must map to UNK.
    let ids = tok.encode("龘", false).expect("encode");
    assert_eq!(ids, vec![UNK_ID]);
}

// -----------------------------------------------------------------------------
// OOV policy (default Unk vs. explicit Error)
// -----------------------------------------------------------------------------

#[test]
fn encode_oov_english_word_defaults_to_unk() {
    let tok = build();
    // "xyz" has no piece "xyz", "xy", "x", "##z", ... in vocab.
    let ids = tok.encode("xyz", false).expect("encode");
    assert_eq!(ids, vec![UNK_ID]);
}

#[test]
fn encode_oov_with_error_policy_returns_err() {
    let tok = build().with_oov_policy(OovPolicy::Error);
    let err = tok
        .encode("xyz", false)
        .expect_err("OovPolicy::Error must fail loudly (FR-EX-08)");
    let msg = format!("{err}");
    assert!(
        msg.contains("xyz") || msg.contains("OOV") || msg.contains("segment"),
        "loud: {msg}"
    );
}

// -----------------------------------------------------------------------------
// Edge cases
// -----------------------------------------------------------------------------

#[test]
fn encode_empty_string_with_specials_is_just_cls_sep() {
    let tok = build();
    let ids = tok.encode("", true).expect("encode");
    assert_eq!(ids, vec![CLS_ID, SEP_ID]);
}

#[test]
fn encode_empty_string_without_specials_is_empty() {
    let tok = build();
    let ids = tok.encode("", false).expect("encode");
    assert_eq!(ids, Vec::<u32>::new());
}

#[test]
fn encode_whitespace_only_produces_no_tokens() {
    let tok = build();
    let ids = tok.encode("   \t\n", false).expect("encode");
    assert_eq!(ids, Vec::<u32>::new());
}

// -----------------------------------------------------------------------------
// Accessors
// -----------------------------------------------------------------------------

#[test]
fn accessors_report_configured_special_ids() {
    let tok = build();
    assert_eq!(tok.unk_id(), UNK_ID);
    assert_eq!(tok.cls_id(), CLS_ID);
    assert_eq!(tok.sep_id(), SEP_ID);
    assert_eq!(tok.pad_id(), PAD_ID);
    assert_eq!(tok.vocab_size(), 10);
}

// -----------------------------------------------------------------------------
// GGUF loader (mirrors the SentencePiece tokenizer's `from_gguf` contract)
// -----------------------------------------------------------------------------

fn build_gguf_bytes(prefix: &str) -> Vec<u8> {
    let mut b = GgufBuilder::new();
    // vocab (STRING array, id = index)
    let vocab_values: Vec<GgufMetadataValue> = synthetic_vocab()
        .into_iter()
        .map(GgufMetadataValue::String)
        .collect();
    b.add_metadata(
        &format!("{prefix}.vocab"),
        GgufMetadataValue::Array(GgufArray {
            element_type: GgufValueType::String,
            values: vocab_values,
        }),
    );
    b.add_u32(&format!("{prefix}.unk_id"), UNK_ID);
    b.add_u32(&format!("{prefix}.cls_id"), CLS_ID);
    b.add_u32(&format!("{prefix}.sep_id"), SEP_ID);
    b.add_u32(&format!("{prefix}.pad_id"), PAD_ID);
    b.add_bool(&format!("{prefix}.do_lower_case"), true);
    b.to_bytes().expect("build GGUF")
}

#[test]
fn from_gguf_round_trip_matches_from_vocab() {
    let prefix = "vokra.bert_base.wordpiece";
    let bytes = build_gguf_bytes(prefix);
    let gguf = GgufFile::parse(bytes).expect("parse GGUF");
    let tok = BertWordpieceTokenizer::from_gguf(&gguf, prefix).expect("load from GGUF");

    // Same encoding contract as `from_vocab`-built instance.
    let ids = tok.encode("hello world", false).expect("encode");
    assert_eq!(ids, vec![4, 5, 6]);
    let ids_zh = tok.encode("你好", true).expect("encode");
    assert_eq!(ids_zh, vec![CLS_ID, 7, 8, SEP_ID]);

    assert_eq!(tok.unk_id(), UNK_ID);
    assert_eq!(tok.cls_id(), CLS_ID);
    assert_eq!(tok.sep_id(), SEP_ID);
    assert_eq!(tok.pad_id(), PAD_ID);
    assert_eq!(tok.vocab_size(), 10);
}

#[test]
fn from_gguf_missing_vocab_key_is_loud() {
    let mut b = GgufBuilder::new();
    b.add_u32("vokra.bert_base.wordpiece.unk_id", UNK_ID);
    let bytes = b.to_bytes().expect("build GGUF");
    let gguf = GgufFile::parse(bytes).expect("parse GGUF");
    let err = BertWordpieceTokenizer::from_gguf(&gguf, "vokra.bert_base.wordpiece")
        .expect_err("missing .vocab must be loud (FR-EX-08)");
    let msg = format!("{err}");
    assert!(msg.contains("vocab"), "loud: {msg}");
}

#[test]
fn from_gguf_respects_do_lower_case_false() {
    let prefix = "vokra.bert_base.wordpiece";
    let mut b = GgufBuilder::new();
    let vocab_values: Vec<GgufMetadataValue> = synthetic_vocab()
        .into_iter()
        .map(GgufMetadataValue::String)
        .collect();
    b.add_metadata(
        &format!("{prefix}.vocab"),
        GgufMetadataValue::Array(GgufArray {
            element_type: GgufValueType::String,
            values: vocab_values,
        }),
    );
    b.add_u32(&format!("{prefix}.unk_id"), UNK_ID);
    b.add_u32(&format!("{prefix}.cls_id"), CLS_ID);
    b.add_u32(&format!("{prefix}.sep_id"), SEP_ID);
    b.add_u32(&format!("{prefix}.pad_id"), PAD_ID);
    b.add_bool(&format!("{prefix}.do_lower_case"), false);
    let bytes = b.to_bytes().expect("build GGUF");
    let gguf = GgufFile::parse(bytes).expect("parse GGUF");
    let tok = BertWordpieceTokenizer::from_gguf(&gguf, prefix).expect("load from GGUF");

    // do_lower_case=false: "HELLO" fails to match "hello" (id 4).
    let ids = tok.encode("HELLO", false).expect("encode");
    assert_eq!(ids, vec![UNK_ID]);
}
