//! SentencePiece BPE tokenizer round-trip test.
//! Clean-room impl per Kudo & Richardson 2018.

use vokra_bert::tokenizer::SbertTokenizer;
use vokra_core::gguf::{value::GgufArray, GgufBuilder, GgufFile, GgufMetadataValue, GgufValueType};

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

// ============================================================================
// SPM (SentencePiece) scheme verification tests — Task 7 synthetic roundtrip
// ============================================================================

/// Helper: build a GGUF with SPM-style tokenizer metadata (pieces, scores, control ids).
/// This mirrors what the deberta-v3 converter writes to GGUF.
fn build_spm_test_gguf(
    pieces: &[&str],
    scores: &[f32],
    unk_id: u32,
    bos_id: u32,
    eos_id: u32,
) -> GgufFile {
    let mut builder = GgufBuilder::new();

    // Stamp the scheme as "unigram" (SentencePiece Unigram variant)
    builder.add_string("vokra.bert.tokenizer.scheme", "unigram");

    // Pieces array
    builder.add_metadata(
        "vokra.bert.tokenizer.pieces",
        GgufMetadataValue::Array(GgufArray {
            element_type: GgufValueType::String,
            values: pieces
                .iter()
                .map(|p| GgufMetadataValue::String((*p).to_owned()))
                .collect(),
        }),
    );

    // Scores array
    builder.add_metadata(
        "vokra.bert.tokenizer.scores",
        GgufMetadataValue::Array(GgufArray {
            element_type: GgufValueType::F32,
            values: scores.iter().map(|s| GgufMetadataValue::F32(*s)).collect(),
        }),
    );

    // Control ids
    builder.add_u32("vokra.bert.tokenizer.unk_id", unk_id);
    builder.add_u32("vokra.bert.tokenizer.bos_id", bos_id);
    builder.add_u32("vokra.bert.tokenizer.eos_id", eos_id);

    // Tokenizer kind — stamp it as SentencePiece (not char-split)
    builder.add_string("vokra.bert.tokenizer.kind", "sentencepiece-unigram");

    let bytes = builder.to_bytes().expect("build GGUF");
    GgufFile::parse(bytes).expect("parse GGUF")
}

/// Task 7: Synthetic SPM-scheme GGUF roundtrip verification.
/// Creates a minimal SentencePiece vocab (5 pieces), writes it to GGUF
/// with scheme="sentencepiece-unigram", and verifies from_gguf loads it
/// correctly with the expected control ids and kind dispatch.
#[test]
fn spm_scheme_synthetic_vocab_roundtrip_via_gguf() {
    // Build a tiny vocab: <unk>, <s>, </s>, plus two dummy pieces
    let pieces = vec!["<unk>", "<s>", "</s>", "▁hello", "▁world"];
    let scores = vec![0.0_f32, 0.0, 0.0, -1.0, -1.5];
    let unk_id = 0u32;
    let bos_id = 1u32;
    let eos_id = 2u32;

    let gguf = build_spm_test_gguf(&pieces, &scores, unk_id, bos_id, eos_id);

    // Load via from_gguf with the expected metadata prefix
    let tok = SbertTokenizer::from_gguf(&gguf, "vokra.bert.tokenizer")
        .expect("from_gguf should load SPM scheme GGUF");

    // Verify control ids
    assert_eq!(tok.unk_id(), unk_id, "unk_id should match metadata");
    assert_eq!(tok.bos_id(), bos_id, "bos_id should match metadata");
    assert_eq!(tok.eos_id(), eos_id, "eos_id should match metadata");

    // Encode a test string to verify the pieces are accessible
    // "hello world" -> prepend ▁ -> "▁hello world" -> match ▁hello + ▁world (or parts)
    let ids = tok.encode("hello world");
    // We expect pieces 3 and 4 (▁hello, ▁world) to be matched.
    // The viterbi will match ▁hello=3 + ▁world=4 (greedy longest-match)
    assert_eq!(ids, vec![3, 4], "encode should match expected pieces");

    // Decode back
    let decoded = tok.decode(&ids);
    assert_eq!(decoded, "hello world", "decode should round-trip");
}

/// Verify that missing scheme key defaults to Unigram (backward compat)
/// but still loads SentencePiece pieces correctly.
#[test]
fn spm_scheme_absent_defaults_to_unigram() {
    let mut builder = GgufBuilder::new();

    let pieces = vec!["<unk>", "<s>", "</s>", "▁hi"];
    let scores = vec![0.0_f32, 0.0, 0.0, -1.0];
    let unk_id = 0u32;
    let bos_id = 1u32;
    let eos_id = 2u32;

    // Deliberately omit the .scheme key to test backward compatibility
    // (no `builder.add_string("vokra.bert.tokenizer.scheme", ...)`)

    builder.add_metadata(
        "vokra.bert.tokenizer.pieces",
        GgufMetadataValue::Array(GgufArray {
            element_type: GgufValueType::String,
            values: pieces
                .iter()
                .map(|p| GgufMetadataValue::String((*p).to_owned()))
                .collect(),
        }),
    );

    builder.add_metadata(
        "vokra.bert.tokenizer.scores",
        GgufMetadataValue::Array(GgufArray {
            element_type: GgufValueType::F32,
            values: scores.iter().map(|s| GgufMetadataValue::F32(*s)).collect(),
        }),
    );

    builder.add_u32("vokra.bert.tokenizer.unk_id", unk_id);
    builder.add_u32("vokra.bert.tokenizer.bos_id", bos_id);
    builder.add_u32("vokra.bert.tokenizer.eos_id", eos_id);

    let bytes = builder.to_bytes().expect("build GGUF");
    let gguf = GgufFile::parse(bytes).expect("parse GGUF");

    let tok = SbertTokenizer::from_gguf(&gguf, "vokra.bert.tokenizer")
        .expect("from_gguf should load legacy GGUF without .scheme key");

    // Should still work as Unigram (default)
    let ids = tok.encode("hi");
    // ▁hi matches piece 3 directly
    assert_eq!(ids, vec![3]);
}
