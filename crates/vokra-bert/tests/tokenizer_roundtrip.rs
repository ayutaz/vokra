//! SentencePiece BPE tokenizer round-trip test.
//! Clean-room impl per Kudo & Richardson 2018.

use vokra_bert::tokenizer::SbertTokenizer;

/// A synthetic tiny vocab: ids 0..=6 map to characters,
/// letting us assert encode is deterministic without needing a real .model file.
fn synthetic_tokenizer() -> SbertTokenizer {
    SbertTokenizer::from_pieces_for_test(vec![
        ("<pad>".to_string(), 0.0),
        ("<unk>".to_string(), 0.0),
        ("<s>".to_string(), 0.0),
        ("</s>".to_string(), 0.0),
        ("▁h".to_string(), -1.0),
        ("ello".to_string(), -1.0),
        ("▁world".to_string(), -1.5),
    ])
}

#[test]
fn encode_greedy_deterministic() {
    let tok = synthetic_tokenizer();
    let ids = tok.encode("hello world");
    // Greedy BPE with SentencePiece "▁" word start marker:
    // "hello world" -> "▁hello▁world" -> ["▁h", "ello", "▁world"] -> [4, 5, 6]
    assert_eq!(ids, vec![4, 5, 6]);
}

#[test]
fn encode_unknown_falls_back_to_unk() {
    let tok = synthetic_tokenizer();
    let ids = tok.encode("xyz");
    // No piece for "xyz" -> <unk> (id 1)
    assert!(ids.contains(&1));
}

#[test]
fn decode_round_trip() {
    let tok = synthetic_tokenizer();
    let ids = vec![4u32, 5, 6];
    let text = tok.decode(&ids);
    // ▁h + ello + ▁world -> "▁hello▁world" -> "hello world" (▁ → space, leading strip)
    assert_eq!(text, "hello world");
}

#[test]
fn decode_skips_special_tokens() {
    let tok = synthetic_tokenizer();
    let ids = vec![2u32, 4, 5, 3]; // <s> ▁h ello </s>
    let text = tok.decode(&ids);
    assert_eq!(text, "hello");
}

/// Char-level tokenizer for HF BertJapaneseTokenizer with
/// subword_tokenizer_type="character" (e.g. ku-nlp/deberta-v2-large-japanese-char-wwm).
/// Ids: 0=[PAD], 1=[CLS], 2=[SEP], 3=[UNK], 4=テ, 5=ス, 6=ト
fn synthetic_charsplit_tokenizer() -> SbertTokenizer {
    SbertTokenizer::from_pieces_for_test_charsplit(
        vec![
            ("[PAD]".to_string(), 0.0),
            ("[CLS]".to_string(), 0.0),
            ("[SEP]".to_string(), 0.0),
            ("[UNK]".to_string(), 0.0),
            ("テ".to_string(), 0.0),
            ("ス".to_string(), 0.0),
            ("ト".to_string(), 0.0),
        ],
        /*unk_id*/ 3,
        /*cls_id*/ 1,
        /*sep_id*/ 2,
    )
}

/// WP-shape-fix (2026-08-09): char-level HF BERT tokenizers split by char
/// (one code point per token) — NOT SentencePiece Viterbi (which prepends
/// `▁` and greedy-matches multi-char pieces). Reproduces the shape mismatch
/// that showed as `bert_hidden_ja` Rust=4096 vs Python=5120 in the parity
/// CI (see task #7): Python HF tokenizer returned 3 chars + CLS + SEP = 5
/// tokens; Rust returned 4 (Viterbi + WORD_START artifact).
#[test]
fn charsplit_encode_matches_hf_char_tokenizer_shape() {
    let tok = synthetic_charsplit_tokenizer();
    let ids = tok.encode_with_special_tokens("テスト");
    // Expect: [CLS=1, テ=4, ス=5, ト=6, SEP=2] = 5 tokens
    assert_eq!(ids, vec![1, 4, 5, 6, 2]);
}

/// Char-level unknown char falls back to [UNK].
#[test]
fn charsplit_encode_unknown_char_falls_back_to_unk() {
    let tok = synthetic_charsplit_tokenizer();
    let ids = tok.encode_with_special_tokens("テX");
    // "X" is not in vocab → [UNK=3]. Expect: [CLS, テ, UNK, SEP]
    assert_eq!(ids, vec![1, 4, 3, 2]);
}

/// SentencePiece path preserves the pre-WP-shape-fix behaviour (no
/// silent regression on SentencePiece consumers).
#[test]
fn sentencepiece_encode_with_special_tokens_wraps_existing_encode() {
    let tok = synthetic_tokenizer();
    let ids = tok.encode_with_special_tokens("hello world");
    // Existing SentencePiece Viterbi: [4, 5, 6], with CLS/SEP wrap: [2, 4, 5, 6, 3]
    assert_eq!(ids, vec![2, 4, 5, 6, 3]);
}
