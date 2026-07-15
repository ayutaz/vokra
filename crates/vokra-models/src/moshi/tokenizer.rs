//! Moshi inner-monologue text tokenizer — **decode-only** SentencePiece
//! (M4-06-T14).
//!
//! # Why decode-only
//!
//! The Moshi text channel is always **self-generated**: the model's own
//! previous text token is its next text input (lm.py `_step` writes the
//! sampled token back into the ring) and user text never enters the
//! model. The runtime therefore only needs `id → piece` for displaying
//! the inner monologue; there is no encode path to implement (contrast
//! CSM, whose caller-supplied `reply_text` needs encoding — M4-05).
//!
//! # Display rule (run_inference.py, transcribed)
//!
//! A token is displayed iff it is neither `text_pad_id` (3 upstream) nor
//! `text_end_pad_id` (0), as `id_to_piece(id).replace("▁", " ")` —
//! [`decode_monologue`] mirrors that exactly (including printing byte /
//! control pieces literally, as the upstream terminal printer does).
//!
//! # Zero-dep SPM parsing (ADR M4-06 §D1-(e))
//!
//! The upstream `tokenizer_spm_32k_3.model` is a SentencePiece
//! `ModelProto`; the GGUF embeds its raw bytes verbatim
//! (`vokra.tokenizer.model` U8 array — M2-06 precedent, byte-exact
//! upstream artifact). [`GgufMoshiTokenizer`] extracts the piece table
//! with a **minimal protobuf wire-format walker** (`pieces` = repeated
//! field 1; each `SentencePiece` = {piece: string = 1, score: float = 2,
//! type: enum = 3}) — pure-Rust binary parsing in the same spirit as the
//! native GGUF / safetensors readers, no protobuf crate (NFR-DS-02).
//! Anything unparseable is a loud [`VokraError::ModelLoad`] (FR-EX-08).

use vokra_core::gguf::{GgufFile, GgufMetadataValue};
use vokra_core::{Result, VokraError};

use super::config::MoshiConfig;

/// The GGUF key carrying the raw tokenizer blob (M2-06 Whisper / M3-10
/// Voxtral / M4-05 CSM precedent — one key for every embedded tokenizer).
pub const KEY_TOKENIZER_MODEL: &str = "vokra.tokenizer.model";

/// The word-boundary marker SentencePiece prefixes pieces with (U+2581,
/// "LOWER ONE EIGHTH BLOCK") — replaced by a space for display.
pub const SPM_SPACE: char = '\u{2581}';

/// Decode-only text tokenizer face (module docs).
pub trait MoshiTextTokenizer: Send + Sync {
    /// Table size (must equal the config's `text_card`).
    fn vocab_size(&self) -> usize;

    /// The display piece for `id`, with the SentencePiece `▁` marker
    /// already replaced by a space (upstream `id_to_piece(id).replace`).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on an out-of-range id.
    fn piece(&self, id: u32) -> Result<String>;
}

/// Decodes an inner-monologue token stream for display: pad /
/// end-of-padding ids are skipped, every other id contributes its piece
/// (module docs "Display rule").
///
/// # Errors
///
/// Propagates [`MoshiTextTokenizer::piece`] errors.
pub fn decode_monologue(
    tokenizer: &dyn MoshiTextTokenizer,
    config: &MoshiConfig,
    ids: &[u32],
) -> Result<String> {
    let mut out = String::new();
    for &id in ids {
        if id == config.text_pad_id || id == config.text_end_pad_id {
            continue;
        }
        out.push_str(&tokenizer.piece(id)?);
    }
    Ok(out)
}

/// A deterministic fixture tokenizer for synthesized-weight tests: id `i`
/// decodes to `" t{i}"` (a `▁`-prefixed piece shape, so the space
/// semantics of the display rule are exercised).
#[derive(Debug, Clone)]
pub struct FixtureMoshiTokenizer {
    vocab_size: usize,
}

impl FixtureMoshiTokenizer {
    /// A fixture table of `vocab_size` pieces.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on a zero vocab.
    pub fn new(vocab_size: usize) -> Result<Self> {
        if vocab_size == 0 {
            return Err(VokraError::InvalidArgument(
                "moshi fixture tokenizer: vocab_size must be > 0".into(),
            ));
        }
        Ok(Self { vocab_size })
    }
}

impl MoshiTextTokenizer for FixtureMoshiTokenizer {
    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn piece(&self, id: u32) -> Result<String> {
        if id as usize >= self.vocab_size {
            return Err(VokraError::InvalidArgument(format!(
                "moshi fixture tokenizer: id {id} >= vocab {}",
                self.vocab_size
            )));
        }
        Ok(format!(" t{id}"))
    }
}

/// The GGUF-embedded SentencePiece tokenizer (decode-only — module docs).
pub struct GgufMoshiTokenizer {
    /// Display pieces, `▁` already replaced (index = id).
    pieces: Vec<String>,
}

impl std::fmt::Debug for GgufMoshiTokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GgufMoshiTokenizer")
            .field("vocab_size", &self.pieces.len())
            .finish()
    }
}

impl GgufMoshiTokenizer {
    /// Extracts the piece table from the GGUF's raw SPM blob and checks
    /// it against the config's `text_card`.
    ///
    /// # Errors
    ///
    /// [`VokraError::ModelLoad`] when the blob is absent (the converter
    /// ran without the tokenizer file — T29 hand-off pending), fails to
    /// parse as a `ModelProto`, or its piece count differs from
    /// `expected_vocab`.
    pub fn from_gguf(file: &GgufFile, expected_vocab: usize) -> Result<Self> {
        let blob = match file.get(KEY_TOKENIZER_MODEL) {
            Some(GgufMetadataValue::Array(arr)) => {
                let mut bytes = Vec::with_capacity(arr.values.len());
                for v in &arr.values {
                    match v {
                        GgufMetadataValue::U8(b) => bytes.push(*b),
                        other => {
                            return Err(VokraError::ModelLoad(format!(
                                "moshi tokenizer: `{KEY_TOKENIZER_MODEL}` element is not \
                                 U8 (got {:?})",
                                other.value_type()
                            )));
                        }
                    }
                }
                bytes
            }
            None => {
                return Err(VokraError::ModelLoad(format!(
                    "moshi tokenizer: `{KEY_TOKENIZER_MODEL}` is absent — convert with \
                     --tokenizer <tokenizer_spm_32k_3.model> (kyutai repo file; T29 \
                     owner hand-off pins the blob)"
                )));
            }
            Some(other) => {
                return Err(VokraError::ModelLoad(format!(
                    "moshi tokenizer: `{KEY_TOKENIZER_MODEL}` is not an array (got {:?})",
                    other.value_type()
                )));
            }
        };
        Self::from_spm_bytes(&blob, expected_vocab)
    }

    /// Parses a raw SentencePiece `ModelProto` (module docs — minimal
    /// wire-format walker over `pieces` only).
    ///
    /// # Errors
    ///
    /// [`VokraError::ModelLoad`] on malformed wire data or a piece-count
    /// mismatch (never a silent partial table — FR-EX-08).
    pub fn from_spm_bytes(blob: &[u8], expected_vocab: usize) -> Result<Self> {
        let mut pieces = Vec::new();
        let mut cursor = 0usize;
        while cursor < blob.len() {
            let (tag, next) = read_varint(blob, cursor)?;
            cursor = next;
            let field = (tag >> 3) as u32;
            let wire = (tag & 0x7) as u8;
            if field == 1 && wire == 2 {
                // repeated SentencePiece pieces = 1
                let (len, next) = read_varint(blob, cursor)?;
                cursor = next;
                let end = cursor
                    .checked_add(len as usize)
                    .filter(|&e| e <= blob.len())
                    .ok_or_else(|| truncated("pieces entry"))?;
                pieces.push(parse_piece(&blob[cursor..end])?);
                cursor = end;
            } else {
                cursor = skip_field(blob, cursor, wire)?;
            }
        }
        if pieces.len() != expected_vocab {
            return Err(VokraError::ModelLoad(format!(
                "moshi tokenizer: SPM model carries {} pieces but the config's \
                 text_card is {expected_vocab} — wrong tokenizer file?",
                pieces.len()
            )));
        }
        Ok(Self { pieces })
    }
}

impl MoshiTextTokenizer for GgufMoshiTokenizer {
    fn vocab_size(&self) -> usize {
        self.pieces.len()
    }

    fn piece(&self, id: u32) -> Result<String> {
        self.pieces.get(id as usize).cloned().ok_or_else(|| {
            VokraError::InvalidArgument(format!(
                "moshi tokenizer: id {id} >= vocab {}",
                self.pieces.len()
            ))
        })
    }
}

/// Parses one `SentencePiece` message, returning its display piece
/// (field 1 = piece string; fields 2 (score) / 3 (type) are skipped —
/// the display rule prints every piece literally, upstream-faithful).
fn parse_piece(msg: &[u8]) -> Result<String> {
    let mut piece: Option<String> = None;
    let mut cursor = 0usize;
    while cursor < msg.len() {
        let (tag, next) = read_varint(msg, cursor)?;
        cursor = next;
        let field = (tag >> 3) as u32;
        let wire = (tag & 0x7) as u8;
        if field == 1 && wire == 2 {
            let (len, next) = read_varint(msg, cursor)?;
            cursor = next;
            let end = cursor
                .checked_add(len as usize)
                .filter(|&e| e <= msg.len())
                .ok_or_else(|| truncated("piece string"))?;
            let s = std::str::from_utf8(&msg[cursor..end]).map_err(|_| {
                VokraError::ModelLoad("moshi tokenizer: SPM piece is not valid UTF-8".into())
            })?;
            piece = Some(s.replace(SPM_SPACE, " "));
            cursor = end;
        } else {
            cursor = skip_field(msg, cursor, wire)?;
        }
    }
    piece.ok_or_else(|| {
        VokraError::ModelLoad("moshi tokenizer: SentencePiece entry without a piece string".into())
    })
}

/// Reads a base-128 varint at `cursor`.
fn read_varint(buf: &[u8], mut cursor: usize) -> Result<(u64, usize)> {
    let mut value = 0u64;
    let mut shift = 0u32;
    loop {
        let byte = *buf.get(cursor).ok_or_else(|| truncated("varint"))?;
        cursor += 1;
        if shift >= 64 {
            return Err(VokraError::ModelLoad(
                "moshi tokenizer: varint longer than 64 bits".into(),
            ));
        }
        value |= u64::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            return Ok((value, cursor));
        }
        shift += 7;
    }
}

/// Skips a field of the given wire type (0 varint / 1 fixed64 / 2
/// length-delimited / 5 fixed32; groups are unsupported → loud error).
fn skip_field(buf: &[u8], cursor: usize, wire: u8) -> Result<usize> {
    match wire {
        0 => Ok(read_varint(buf, cursor)?.1),
        1 => cursor
            .checked_add(8)
            .filter(|&e| e <= buf.len())
            .ok_or_else(|| truncated("fixed64")),
        2 => {
            let (len, next) = read_varint(buf, cursor)?;
            next.checked_add(len as usize)
                .filter(|&e| e <= buf.len())
                .ok_or_else(|| truncated("length-delimited field"))
        }
        5 => cursor
            .checked_add(4)
            .filter(|&e| e <= buf.len())
            .ok_or_else(|| truncated("fixed32")),
        other => Err(VokraError::ModelLoad(format!(
            "moshi tokenizer: unsupported protobuf wire type {other} (group fields \
             are not part of sentencepiece_model.proto)"
        ))),
    }
}

fn truncated(what: &str) -> VokraError {
    VokraError::ModelLoad(format!("moshi tokenizer: truncated SPM blob ({what})"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hand-encodes a minimal SentencePiece `ModelProto` (test-side wire
    /// writer — the runtime only ever reads).
    fn spm_blob(pieces: &[(&str, f32, u64)]) -> Vec<u8> {
        fn varint(mut v: u64, out: &mut Vec<u8>) {
            loop {
                let mut b = (v & 0x7f) as u8;
                v >>= 7;
                if v != 0 {
                    b |= 0x80;
                }
                out.push(b);
                if v == 0 {
                    break;
                }
            }
        }
        let mut blob = Vec::new();
        for (piece, score, ptype) in pieces {
            let mut msg = Vec::new();
            // field 1 (piece), wire 2
            msg.push(0x0a);
            varint(piece.len() as u64, &mut msg);
            msg.extend_from_slice(piece.as_bytes());
            // field 2 (score), wire 5 (float)
            msg.push(0x15);
            msg.extend_from_slice(&score.to_le_bytes());
            // field 3 (type), wire 0 (enum varint)
            msg.push(0x18);
            varint(*ptype, &mut msg);
            // top-level field 1 (pieces), wire 2
            blob.push(0x0a);
            varint(msg.len() as u64, &mut blob);
            blob.extend_from_slice(&msg);
        }
        // A trailing unrelated field (e.g. trainer_spec = 2, wire 2) the
        // walker must skip.
        blob.push(0x12);
        blob.push(0x02);
        blob.extend_from_slice(&[0x08, 0x01]);
        blob
    }

    fn table() -> Vec<(&'static str, f32, u64)> {
        vec![
            ("<pad0>", 0.0, 3), // id 0 = end_pad (CONTROL)
            ("<s>", 0.0, 3),    // id 1
            ("</s>", 0.0, 3),   // id 2
            ("<pad>", 0.0, 3),  // id 3 = pad
            ("\u{2581}hello", -1.0, 1),
            ("\u{2581}world", -1.5, 1),
            ("!", -2.0, 1),
            ("<0x0A>", -3.0, 6), // BYTE piece — displayed literally
        ]
    }

    #[test]
    fn spm_walker_extracts_pieces_and_replaces_the_space_marker() {
        let blob = spm_blob(&table());
        let tok = GgufMoshiTokenizer::from_spm_bytes(&blob, 8).expect("parse");
        assert_eq!(tok.vocab_size(), 8);
        assert_eq!(tok.piece(4).unwrap(), " hello");
        assert_eq!(tok.piece(5).unwrap(), " world");
        assert_eq!(tok.piece(6).unwrap(), "!");
        assert_eq!(
            tok.piece(7).unwrap(),
            "<0x0A>",
            "byte pieces print literally"
        );
        assert!(tok.piece(8).is_err(), "out of range is loud");
    }

    #[test]
    fn piece_count_mismatch_is_a_loud_model_load_error() {
        let blob = spm_blob(&table());
        let err = GgufMoshiTokenizer::from_spm_bytes(&blob, 9).unwrap_err();
        assert!(matches!(err, VokraError::ModelLoad(_)));
        assert!(err.to_string().contains("8 pieces"), "counts named: {err}");
    }

    #[test]
    fn truncated_blob_is_loud_never_a_partial_table() {
        let blob = spm_blob(&table());
        let err = GgufMoshiTokenizer::from_spm_bytes(&blob[..blob.len() - 3], 8).unwrap_err();
        assert!(matches!(err, VokraError::ModelLoad(_)));
    }

    #[test]
    fn monologue_decode_skips_pad_ids_and_concats_pieces() {
        // Display rule transcription: skip {pad_id, end_pad_id}, join the
        // rest (run_inference.py `not in [0, 3]`).
        let blob = spm_blob(&table());
        let tok = GgufMoshiTokenizer::from_spm_bytes(&blob, 8).unwrap();
        let mut cfg = MoshiConfig::tiny_for_tests();
        cfg.text_card = 8;
        cfg.text_pad_id = 3;
        cfg.text_end_pad_id = 0;
        let ids = [3, 4, 0, 5, 3, 3, 6, 0];
        let text = decode_monologue(&tok, &cfg, &ids).unwrap();
        assert_eq!(text, " hello world!");
    }

    #[test]
    fn gguf_blob_round_trip_and_absence_are_honest() {
        use vokra_core::gguf::{GgufArray, GgufBuilder, GgufValueType};
        let blob = spm_blob(&table());
        let mut b = GgufBuilder::new();
        b.add_string("vokra.model.arch", "moshi");
        b.add_metadata(
            KEY_TOKENIZER_MODEL,
            GgufMetadataValue::Array(GgufArray {
                element_type: GgufValueType::U8,
                values: blob.iter().map(|&x| GgufMetadataValue::U8(x)).collect(),
            }),
        );
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let tok = GgufMoshiTokenizer::from_gguf(&file, 8).expect("blob round-trips");
        assert_eq!(tok.piece(4).unwrap(), " hello");

        let mut b = GgufBuilder::new();
        b.add_string("vokra.model.arch", "moshi");
        let file = GgufFile::parse(b.to_bytes().unwrap()).unwrap();
        let err = GgufMoshiTokenizer::from_gguf(&file, 8).unwrap_err();
        assert!(err.to_string().contains("--tokenizer"), "actionable: {err}");
    }

    #[test]
    fn fixture_tokenizer_matches_the_trait_contract() {
        let f = FixtureMoshiTokenizer::new(13).unwrap();
        assert_eq!(f.vocab_size(), 13);
        assert_eq!(f.piece(4).unwrap(), " t4");
        assert!(f.piece(13).is_err());
        let cfg = MoshiConfig::tiny_for_tests();
        // pad (3) and end_pad (0) vanish from the display.
        let text = decode_monologue(&f, &cfg, &[0, 3, 1, 2]).unwrap();
        assert_eq!(text, " t1 t2");
    }
}
