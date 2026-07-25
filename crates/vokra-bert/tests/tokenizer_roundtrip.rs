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
