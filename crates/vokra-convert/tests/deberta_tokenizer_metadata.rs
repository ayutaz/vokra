//! End-to-end tests for Blocker 5 — DeBERTa v2 (WordPiece) and v3
//! (SentencePiece Unigram) converters discover the sibling tokenizer
//! file next to the safetensors input, emit `vokra.bert.tokenizer.*`
//! metadata (scheme + pieces + scores + special-token ids), and the
//! `vokra_bert::tokenizer::SbertTokenizer::from_gguf` loader consumes
//! the metadata via the scheme dispatch.
//!
//! # Fixture strategy
//!
//! Everything is hand-crafted inside the test — no committed fixture
//! files. WordPiece `vocab.txt` is a plain-text UTF-8 buffer (one piece
//! per line); the SentencePiece `spm.model` is a raw proto3 byte buffer
//! hand-encoded through the same primitive the parser reads back, so
//! the fixture is self-checking (any parser bug flips a round-trip
//! test rather than silently drifting away from an opaque binary).
//!
//! # References (permissive only)
//!
//! - Protocol Buffers 3 wire format spec (Google, Apache-2.0 spec)
//! - SentencePiece `sentencepiece_model.proto` field numbers
//!   (Apache-2.0 spec)
//!
//! # NOT REFERENCED
//!
//! - github.com/google/sentencepiece C++ / Python parser source
//! - github.com/litagin02/Style-Bert-VITS2 (AGPL-3.0)

//
//! # Layering note
//!
//! `vokra-convert` deliberately does **not** depend on `vokra-bert` (the
//! runtime crate) — `vokra-bert` reads GGUFs the converter produces, so
//! the edge points converter → GGUF → bert, not both directions. These
//! tests therefore only verify the **metadata stamping** side: that
//! every key `SbertTokenizer::from_gguf` reads (as of Blocker 5 Wave 2)
//! is present with the right value and type. The runtime dispatch is
//! pinned by `crates/vokra-bert/src/tokenizer.rs::scheme_dispatch_tests`.

use std::path::{Path, PathBuf};

use vokra_convert::{convert_deberta_v2_file, convert_deberta_v3_file};
use vokra_core::gguf::{GgufFile, GgufMetadataValue};

/// Build a minimal `.safetensors` buffer with just a `word_embeddings`
/// tensor — enough for `infer_vocab_and_d_model` to run.
fn safetensors_embed_only(vocab_size: u64, d_model: u64) -> Vec<u8> {
    let elems = (vocab_size * d_model) as usize;
    let payload: Vec<u8> = (0..elems * 4).map(|i| (i % 251) as u8).collect();
    let header = format!(
        r#"{{"deberta.embeddings.word_embeddings.weight":{{"dtype":"F32","shape":[{vocab_size},{d_model}],"data_offsets":[0,{}]}}}}"#,
        payload.len()
    );
    let mut out = Vec::new();
    out.extend_from_slice(&(header.len() as u64).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(&payload);
    out
}

/// Hand-encode one proto3 varint into the buffer.
fn encode_varint(mut value: u64, out: &mut Vec<u8>) {
    loop {
        let byte = (value & 0x7F) as u8;
        value >>= 7;
        if value == 0 {
            out.push(byte);
            return;
        }
        out.push(byte | 0x80);
    }
}

/// Hand-encode a proto3 tag `(field_number, wire_type)`.
fn encode_tag(field_number: u32, wire_type: u8, out: &mut Vec<u8>) {
    encode_varint(((field_number as u64) << 3) | u64::from(wire_type), out);
}

/// Hand-encode one `ModelProto.pieces` entry — a length-delimited
/// nested `SentencePiece` message with piece / score / type fields.
fn encode_piece(out: &mut Vec<u8>, piece: &str, score: f32, piece_type: u64) {
    let mut inner = Vec::new();
    // piece = field 1, length-delimited.
    encode_tag(1, 2, &mut inner);
    encode_varint(piece.len() as u64, &mut inner);
    inner.extend_from_slice(piece.as_bytes());
    // score = field 2, fixed32.
    encode_tag(2, 5, &mut inner);
    inner.extend_from_slice(&score.to_le_bytes());
    // type = field 3, varint.
    encode_tag(3, 0, &mut inner);
    encode_varint(piece_type, &mut inner);

    encode_tag(1, 2, out);
    encode_varint(inner.len() as u64, out);
    out.extend_from_slice(&inner);
}

/// Isolated temp directory (checkpoint sibling files must live next to
/// each other for the discovery path to fire).
struct TempDir(PathBuf);

impl TempDir {
    fn new(label: &str) -> Self {
        let mut d = std::env::temp_dir();
        d.push(format!(
            "vokra-deberta-tokenizer-{label}-{}",
            std::process::id()
        ));
        // Recreate to guarantee isolation across serial test runs.
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).expect("create temp dir");
        Self(d)
    }
    fn path(&self) -> &Path {
        &self.0
    }
    fn write(&self, name: &str, bytes: &[u8]) -> PathBuf {
        let p = self.0.join(name);
        std::fs::write(&p, bytes).expect("write fixture file");
        p
    }
    fn join(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

// -------------------------------------------------------------------
// DeBERTa v2 — WordPiece (`vocab.txt`) discovery + metadata round-trip
// -------------------------------------------------------------------

#[test]
fn deberta_v2_reads_wordpiece_vocab_and_stamps_metadata() {
    let dir = TempDir::new("v2-wordpiece");
    // Minimal BERT vocab with the four required sentinels + a couple
    // of real word starts + a continuation, plus a Japanese char.
    let vocab = "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nhello\n##world\nplay\n\u{732B}\n";
    dir.write("vocab.txt", vocab.as_bytes());
    let input = dir.write("model.safetensors", &safetensors_embed_only(9, 4));
    let output = dir.join("out.gguf");

    convert_deberta_v2_file(&input, &output, None, None).expect("convert with tokenizer");
    let bytes = std::fs::read(&output).expect("read gguf");
    let g = GgufFile::parse(bytes).expect("parse gguf");

    // Scheme = wordpiece.
    assert_eq!(
        g.get("vokra.bert.tokenizer.scheme")
            .and_then(|v| v.as_str()),
        Some("wordpiece")
    );
    // Pieces array shape matches the vocab.
    let pieces_arr = g
        .get("vokra.bert.tokenizer.pieces")
        .and_then(|v| v.as_array())
        .expect(".pieces present");
    assert_eq!(pieces_arr.values.len(), 9);
    assert_eq!(pieces_arr.values[0].as_str(), Some("[PAD]"));
    assert_eq!(pieces_arr.values[5].as_str(), Some("hello"));
    assert_eq!(pieces_arr.values[6].as_str(), Some("##world"));
    assert_eq!(pieces_arr.values[8].as_str(), Some("\u{732B}"));
    // Scores: same length, all zero for WordPiece.
    let scores_arr = g
        .get("vokra.bert.tokenizer.scores")
        .and_then(|v| v.as_array())
        .expect(".scores present");
    assert_eq!(scores_arr.values.len(), 9);
    for v in &scores_arr.values {
        assert!(matches!(v, GgufMetadataValue::F32(f) if *f == 0.0));
    }
    // Special-token IDs discovered from the vocab.
    assert_eq!(
        g.get("vokra.bert.tokenizer.unk_id")
            .and_then(|v| v.as_u64()),
        Some(1)
    );
    assert_eq!(
        g.get("vokra.bert.tokenizer.bos_id")
            .and_then(|v| v.as_u64()),
        Some(2),
        "[CLS] is at id 2"
    );
    assert_eq!(
        g.get("vokra.bert.tokenizer.eos_id")
            .and_then(|v| v.as_u64()),
        Some(3),
        "[SEP] is at id 3"
    );

    // The runtime SbertTokenizer::from_gguf dispatch is pinned by
    // vokra-bert::tokenizer::scheme_dispatch_tests — the converter's
    // job here is only to stamp every key the loader reads. Pinning
    // both sides in one crate would violate the converter → GGUF →
    // bert layering.
}

#[test]
fn deberta_v2_missing_vocab_txt_leaves_metadata_unwritten() {
    let dir = TempDir::new("v2-no-vocab");
    // No vocab.txt in dir — backward-compat path.
    let input = dir.write("model.safetensors", &safetensors_embed_only(9, 4));
    let output = dir.join("out.gguf");

    convert_deberta_v2_file(&input, &output, None, None).expect("convert without tokenizer");
    let g = GgufFile::parse(std::fs::read(&output).unwrap()).expect("parse");

    // Metadata group must not be stamped (backward compat with pre-
    // Blocker-5 GGUFs — the runtime SbertTokenizer::from_gguf then
    // loud-errors on the missing `.pieces` key, which is the correct
    // outcome per FR-EX-08).
    assert!(g.get("vokra.bert.tokenizer.scheme").is_none());
    assert!(g.get("vokra.bert.tokenizer.pieces").is_none());
    assert!(g.get("vokra.bert.tokenizer.scores").is_none());
}

#[test]
fn deberta_v2_vocab_missing_sentinel_is_loud_error() {
    let dir = TempDir::new("v2-no-sentinel");
    // Vocab without [CLS] — must loud-error, never silently produce
    // a broken tokenizer.
    let vocab = "[PAD]\n[UNK]\n[SEP]\nhello\n";
    dir.write("vocab.txt", vocab.as_bytes());
    let input = dir.write("model.safetensors", &safetensors_embed_only(9, 4));
    let output = dir.join("out.gguf");
    let err =
        convert_deberta_v2_file(&input, &output, None, None).expect_err("missing [CLS] must fail");
    let msg = format!("{err}");
    assert!(msg.contains("[CLS]"), "error must name the sentinel: {msg}");
}

#[test]
fn deberta_v2_vocab_non_utf8_is_loud_error() {
    let dir = TempDir::new("v2-non-utf8");
    // Invalid UTF-8 (lone 0xFF bytes) in the vocab.txt.
    dir.write(
        "vocab.txt",
        &[b'[', b'P', b'A', b'D', b']', b'\n', 0xFF, 0xFE],
    );
    let input = dir.write("model.safetensors", &safetensors_embed_only(9, 4));
    let output = dir.join("out.gguf");
    let err = convert_deberta_v2_file(&input, &output, None, None)
        .expect_err("non-UTF-8 vocab must fail");
    let msg = format!("{err}");
    assert!(
        msg.contains("UTF-8"),
        "error must name the UTF-8 problem: {msg}"
    );
}

// -------------------------------------------------------------------
// DeBERTa v3 — SentencePiece Unigram (`spm.model`) discovery + round-trip
// -------------------------------------------------------------------

/// Hand-craft a minimal SentencePiece `spm.model` byte buffer with 5
/// pieces: `<unk>`(Unknown=2), `<s>`(Control=3), `</s>`(Control=3),
/// `▁hello`(Normal=1), `▁world`(Normal=1). Each has a distinct score
/// so a round-trip assertion can pin the byte-for-byte parse.
fn hand_crafted_spm_model() -> Vec<u8> {
    let mut out = Vec::new();
    encode_piece(&mut out, "<unk>", 0.0, 2); // Unknown
    encode_piece(&mut out, "<s>", 0.0, 3); // Control
    encode_piece(&mut out, "</s>", 0.0, 3); // Control
    encode_piece(&mut out, "\u{2581}hello", -1.5, 1); // Normal
    encode_piece(&mut out, "\u{2581}world", -2.25, 1); // Normal
    out
}

#[test]
fn deberta_v3_reads_spm_model_and_stamps_metadata() {
    let dir = TempDir::new("v3-spm");
    dir.write("spm.model", &hand_crafted_spm_model());
    let input = dir.write("model.safetensors", &safetensors_embed_only(5, 4));
    let output = dir.join("out.gguf");

    convert_deberta_v3_file(&input, &output, None, None).expect("convert with tokenizer");
    let g = GgufFile::parse(std::fs::read(&output).unwrap()).expect("parse");

    // Scheme = unigram.
    assert_eq!(
        g.get("vokra.bert.tokenizer.scheme")
            .and_then(|v| v.as_str()),
        Some("unigram")
    );
    // Pieces preserved in on-disk order, including the U+2581 marker.
    let pieces_arr = g
        .get("vokra.bert.tokenizer.pieces")
        .and_then(|v| v.as_array())
        .expect(".pieces present");
    assert_eq!(pieces_arr.values.len(), 5);
    assert_eq!(pieces_arr.values[0].as_str(), Some("<unk>"));
    assert_eq!(pieces_arr.values[3].as_str(), Some("\u{2581}hello"));
    assert_eq!(pieces_arr.values[4].as_str(), Some("\u{2581}world"));
    // Scores round-trip byte-exact.
    let scores_arr = g
        .get("vokra.bert.tokenizer.scores")
        .and_then(|v| v.as_array())
        .expect(".scores present");
    let scores: Vec<f32> = scores_arr
        .values
        .iter()
        .map(|v| match v {
            GgufMetadataValue::F32(f) => *f,
            _ => panic!("non-F32 score"),
        })
        .collect();
    assert_eq!(scores.len(), 5);
    assert!((scores[3] - (-1.5)).abs() < f32::EPSILON);
    assert!((scores[4] - (-2.25)).abs() < f32::EPSILON);
    // Special-token IDs discovered — <unk>=0 (Unknown piece type),
    // <s>=1, </s>=2.
    assert_eq!(
        g.get("vokra.bert.tokenizer.unk_id")
            .and_then(|v| v.as_u64()),
        Some(0)
    );
    assert_eq!(
        g.get("vokra.bert.tokenizer.bos_id")
            .and_then(|v| v.as_u64()),
        Some(1)
    );
    assert_eq!(
        g.get("vokra.bert.tokenizer.eos_id")
            .and_then(|v| v.as_u64()),
        Some(2)
    );

    // Runtime dispatch (Unigram viterbi over the stamped pieces) is
    // pinned by vokra-bert::tokenizer::scheme_dispatch_tests.
}

#[test]
fn deberta_v3_missing_spm_model_leaves_metadata_unwritten() {
    let dir = TempDir::new("v3-no-spm");
    let input = dir.write("model.safetensors", &safetensors_embed_only(5, 4));
    let output = dir.join("out.gguf");

    convert_deberta_v3_file(&input, &output, None, None).expect("convert without tokenizer");
    let g = GgufFile::parse(std::fs::read(&output).unwrap()).expect("parse");

    assert!(g.get("vokra.bert.tokenizer.scheme").is_none());
    assert!(g.get("vokra.bert.tokenizer.pieces").is_none());
}

#[test]
fn deberta_v3_malformed_spm_model_is_loud_error() {
    let dir = TempDir::new("v3-malformed");
    // Truncated length-delimited field: tag+length claim 100 bytes,
    // only 3 follow — the proto3 parser must catch this.
    let mut bad = Vec::new();
    encode_tag(1, 2, &mut bad);
    encode_varint(100, &mut bad);
    bad.extend_from_slice(&[1, 2, 3]);
    dir.write("spm.model", &bad);
    let input = dir.write("model.safetensors", &safetensors_embed_only(5, 4));
    let output = dir.join("out.gguf");
    let err = convert_deberta_v3_file(&input, &output, None, None)
        .expect_err("truncated spm.model must fail");
    let msg = format!("{err}");
    assert!(msg.contains("spm.model"), "error must name the file: {msg}");
}

#[test]
fn deberta_v3_nested_tokenizer_dir_discovery_works() {
    // Some HF releases nest the tokenizer under `<parent>/tokenizer/`.
    // Both v2 (`vocab.txt`) and v3 (`spm.model`) discovery walks look
    // there as a second search location — verify the v3 nested path
    // is honored.
    let dir = TempDir::new("v3-nested");
    let tok_dir = dir.path().join("tokenizer");
    std::fs::create_dir_all(&tok_dir).unwrap();
    std::fs::write(tok_dir.join("spm.model"), hand_crafted_spm_model()).unwrap();
    let input = dir.write("model.safetensors", &safetensors_embed_only(5, 4));
    let output = dir.join("out.gguf");

    convert_deberta_v3_file(&input, &output, None, None).expect("nested spm.model is found");
    let g = GgufFile::parse(std::fs::read(&output).unwrap()).expect("parse");
    assert_eq!(
        g.get("vokra.bert.tokenizer.scheme")
            .and_then(|v| v.as_str()),
        Some("unigram")
    );
}
