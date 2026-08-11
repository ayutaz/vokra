//! Hand-rolled proto3 parser for a **SentencePiece** `ModelProto` blob
//! (`spm.model`), extracting only what a Vokra converter needs to stamp the
//! `vokra.bert.tokenizer.pieces / .scores / .unk_id / .bos_id / .eos_id`
//! metadata group. Everything else in the model — `trainer_spec`,
//! `normalizer_spec`, `self_test_data`, `denormalizer_spec` — is skipped
//! by proto3 unknown-field rules.
//!
//! # Why hand-rolled
//!
//! `vokra-convert` is the isolation crate for offline conversion; adding
//! the `protobuf` crate here would drag it through `vokra-core`'s
//! zero-dep boundary (NFR-DS-02 / FR-LD-05: the root `Cargo.lock` must
//! stay `vokra-*`-only). The SentencePiece `ModelProto` we care about
//! uses only three of proto3's wire types (varint / fixed32 /
//! length-delimited), and only three fields inside `SentencePiece` and
//! one repeated-message field inside `ModelProto`, so a full-featured
//! runtime library is not needed.
//!
//! # References (permissive only — SPEC ONLY, NO CODE COPIED)
//!
//! - Protocol Buffers 3 wire format spec (Google, Apache 2.0 spec — the
//!   varint / fixed32 / length-delimited encodings are a wire-level
//!   description independent of any implementation).
//! - SentencePiece `sentencepiece_model.proto` field-number definitions
//!   (Apache 2.0). The `.proto` file itself is a schema declaration, not
//!   code; the byte layout it describes is what this parser recognizes.
//! - Kudo & Richardson 2018 (arXiv:1808.06226) for the meaning of the
//!   `piece / score / type` triple.
//!
//! # NOT REFERENCED
//!
//! - github.com/google/sentencepiece C++ / Python parser source
//! - github.com/litagin02/Style-Bert-VITS2 (AGPL-3.0)
//! - github.com/fishaudio/Bert-VITS2 (AGPL-3.0)
//!
//! # Wire format we support
//!
//! - Field tag = `(field_number << 3) | wire_type` as a varint.
//! - `wire_type = 0`: varint payload (used for `type` enum in
//!   `SentencePiece`).
//! - `wire_type = 5`: fixed 32-bit little-endian payload (used for
//!   `score` `float`).
//! - `wire_type = 2`: length-delimited (varint length + N raw bytes;
//!   used for the outer `pieces` repeated message and the inner `piece`
//!   `string`).
//! - Any other wire type (fixed 64 = 1, group start/end = 3/4) is
//!   accepted at the unknown-field-skip level for forward compatibility.
//!
//! Sub-field numbers in `sentencepiece_model.proto`:
//! - Outer `ModelProto.pieces` = field 1 (`repeated SentencePiece`).
//! - Inner `SentencePiece.piece` = field 1 (`string`).
//! - Inner `SentencePiece.score` = field 2 (`float`).
//! - Inner `SentencePiece.type` = field 3 (`enum PieceType`).
//!
//! Any other field number at either level is skipped losslessly.

use std::fmt;

/// One entry of the SentencePiece vocabulary — a subword string and the
/// log-probability the SentencePiece Unigram search will consult.
///
/// `piece_type` distinguishes the four SentencePiece categories: `Normal`
/// (regular subword), `Unknown` (the `<unk>` sentinel), `Control` (the
/// `<s>` / `</s>` sentinels), and `UserDefined` (byte-fallback,
/// user-injected specials, and byte pieces like `<0x00>`).
#[derive(Debug, Clone, PartialEq)]
pub struct SentencePiece {
    /// UTF-8 bytes of the subword (SentencePiece uses U+2581 `▁` as the
    /// word-start marker; the bytes are preserved exactly).
    pub piece: String,
    /// SentencePiece Unigram log-probability. `0.0` for `Control` /
    /// `Unknown` sentinels (upstream convention).
    pub score: f32,
    /// SentencePiece piece type. Encoded as a proto3 enum: 1=Normal,
    /// 2=Unknown, 3=Control, 4=UserDefined, 5=Byte, 6=Unused.
    pub piece_type: PieceType,
}

/// SentencePiece `PieceType` enum values (from
/// `sentencepiece_model.proto` §`enum Type`). Kept as a plain enum with an
/// explicit numeric mapping so the wire format is decoded losslessly and
/// an unknown value from a newer schema does not silently become a valid
/// variant (FR-EX-08).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PieceType {
    /// Not seen — the SentencePiece default is `NORMAL = 1`, so a value
    /// of `0` on the wire signals "field was omitted". Kept as an
    /// explicit variant rather than folded into `Normal` so tests can
    /// pin the distinction.
    Unspecified,
    /// A regular subword.
    Normal,
    /// The `<unk>` sentinel.
    Unknown,
    /// A control token (`<s>` / `</s>` / `<pad>` / …).
    Control,
    /// A user-defined token — treated by the Unigram search as an
    /// atomic subword regardless of score.
    UserDefined,
    /// A raw byte piece (`<0x00>` .. `<0xFF>`) used by SentencePiece's
    /// byte-fallback strategy.
    Byte,
    /// Reserved by SentencePiece; kept round-trippable.
    Unused,
    /// Any wire value the SentencePiece schema does not know about — a
    /// forward-compatibility escape hatch, carrying the raw varint value
    /// so a caller can decide whether to error or fall back.
    Other(u32),
}

/// Minimal SentencePiece `ModelProto` view: the `pieces` array only.
///
/// Every other top-level field (`trainer_spec`, `normalizer_spec`,
/// `self_test_data`, `denormalizer_spec`) is skipped losslessly. Adding
/// them here later is additive.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelProto {
    /// The vocabulary. Piece index = ID in every SentencePiece consumer;
    /// the on-disk order is preserved.
    pub pieces: Vec<SentencePiece>,
}

/// A parse error, tagged with the byte offset where it was detected so a
/// hand-crafted fixture that goes wrong can be pinpointed by the test.
#[derive(Debug, Clone, PartialEq)]
pub enum SpmProtoError {
    /// The varint reader ran off the end of the buffer.
    UnexpectedEof {
        /// Byte offset where the read attempt started.
        at: usize,
        /// What the reader was trying to do at the failure site.
        context: &'static str,
    },
    /// A varint exceeded the 10-byte / 64-bit ceiling — malformed input.
    VarintTooLong {
        /// Byte offset where the offending varint started.
        at: usize,
    },
    /// A length-delimited field claims more bytes than the buffer
    /// contains after the length prefix.
    LengthOverflow {
        /// Byte offset of the length prefix.
        at: usize,
        /// Declared length in bytes.
        declared: u64,
        /// Bytes actually remaining in the buffer past the length
        /// prefix.
        remaining: usize,
    },
    /// A wire type outside `{0, 1, 2, 5}` was seen. Wire types `3` and
    /// `4` (start_group / end_group) were removed in proto3, so their
    /// appearance means the input is not a proto3 message.
    UnsupportedWireType {
        /// Byte offset of the offending tag.
        at: usize,
        /// The unsupported wire type value.
        wire_type: u8,
    },
    /// A `piece` field inside `SentencePiece` was not valid UTF-8.
    InvalidUtf8 {
        /// Byte offset into the outer buffer where the piece payload
        /// ends.
        at: usize,
    },
}

impl fmt::Display for SpmProtoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnexpectedEof { at, context } => {
                write!(f, "spm_proto: unexpected EOF at byte {at} while {context}")
            }
            Self::VarintTooLong { at } => {
                write!(f, "spm_proto: varint at byte {at} exceeds 10-byte limit")
            }
            Self::LengthOverflow {
                at,
                declared,
                remaining,
            } => write!(
                f,
                "spm_proto: length-delimited field at byte {at} declares {declared} bytes but \
                 only {remaining} bytes remain in the buffer"
            ),
            Self::UnsupportedWireType { at, wire_type } => write!(
                f,
                "spm_proto: unsupported proto wire type {wire_type} at byte {at} (proto3 admits \
                 only 0/1/2/5)"
            ),
            Self::InvalidUtf8 { at } => write!(
                f,
                "spm_proto: SentencePiece.piece at byte {at} is not valid UTF-8"
            ),
        }
    }
}

impl std::error::Error for SpmProtoError {}

/// Parse a SentencePiece `ModelProto` from a raw `spm.model` byte buffer.
///
/// Only `ModelProto.pieces` (field 1) and its inner `piece` / `score` /
/// `type` (fields 1 / 2 / 3) are extracted — every other field is
/// skipped losslessly. Returns [`SpmProtoError`] for malformed varints,
/// truncated length-delimited fields, non-proto3 wire types, or
/// non-UTF-8 piece strings.
///
/// # Errors
///
/// See [`SpmProtoError`] for the full set.
pub fn parse_model(bytes: &[u8]) -> Result<ModelProto, SpmProtoError> {
    let mut cursor = Cursor::new(bytes);
    let mut pieces = Vec::new();
    while !cursor.is_empty() {
        let start = cursor.pos();
        let (field_number, wire_type) = cursor.read_tag()?;
        match (field_number, wire_type) {
            (1, 2) => {
                // ModelProto.pieces — a repeated nested message.
                let piece_bytes = cursor.read_length_delimited()?;
                pieces.push(parse_sentence_piece(piece_bytes, start)?);
            }
            _ => cursor.skip_field(wire_type)?,
        }
    }
    Ok(ModelProto { pieces })
}

/// Parse one `SentencePiece` nested message.
///
/// `start_offset` is only used to attribute the error position back to the
/// outer buffer; the byte slice passed in is the length-delimited payload
/// itself.
fn parse_sentence_piece(bytes: &[u8], start_offset: usize) -> Result<SentencePiece, SpmProtoError> {
    let mut cursor = Cursor::new(bytes);
    let mut piece: Option<String> = None;
    let mut score: f32 = 0.0;
    let mut piece_type: PieceType = PieceType::Unspecified;
    while !cursor.is_empty() {
        let (field_number, wire_type) = cursor.read_tag()?;
        match (field_number, wire_type) {
            (1, 2) => {
                let raw = cursor.read_length_delimited()?;
                piece = Some(std::str::from_utf8(raw).map(str::to_owned).map_err(|_| {
                    SpmProtoError::InvalidUtf8 {
                        at: start_offset + cursor.pos(),
                    }
                })?);
            }
            (2, 5) => {
                score = f32::from_le_bytes(cursor.read_fixed32()?);
            }
            (3, 0) => {
                piece_type = decode_piece_type(cursor.read_varint()?);
            }
            _ => cursor.skip_field(wire_type)?,
        }
    }
    Ok(SentencePiece {
        piece: piece.unwrap_or_default(),
        score,
        piece_type,
    })
}

fn decode_piece_type(raw: u64) -> PieceType {
    match raw {
        // 0 = default / unset (SentencePiece treats absent field as NORMAL,
        // but we preserve the "field was omitted" signal).
        0 => PieceType::Unspecified,
        1 => PieceType::Normal,
        2 => PieceType::Unknown,
        3 => PieceType::Control,
        4 => PieceType::UserDefined,
        5 => PieceType::Byte,
        6 => PieceType::Unused,
        other => PieceType::Other(other as u32),
    }
}

/// Byte cursor over a proto3 message body — extracted so the outer and
/// nested parsers can reuse the same primitives (varint, tag, fixed32,
/// length-delimited, skip-unknown).
struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn is_empty(&self) -> bool {
        self.pos >= self.bytes.len()
    }

    fn pos(&self) -> usize {
        self.pos
    }

    /// Read one proto3 varint. Wire format: 7 low bits of each byte are
    /// the payload; the high bit is set on every byte except the last.
    /// Capped at 10 bytes (64 bits + 1 continuation-bit worth of slack).
    fn read_varint(&mut self) -> Result<u64, SpmProtoError> {
        let start = self.pos;
        let mut result: u64 = 0;
        let mut shift: u32 = 0;
        for i in 0..10 {
            if self.pos >= self.bytes.len() {
                return Err(SpmProtoError::UnexpectedEof {
                    at: self.pos,
                    context: "reading varint",
                });
            }
            let byte = self.bytes[self.pos];
            self.pos += 1;
            // Lower 7 bits are payload; shift into result. For the 10th
            // byte, only bit 0 is meaningful (bits 1-6 would overflow a
            // u64) but proto3 encoders never set them, and we preserve
            // the full byte to match every existing SentencePiece
            // parser's behavior of "accept what fits, drop overflow"
            // rather than error on the 64-bit boundary.
            result |= u64::from(byte & 0x7F) << shift;
            if byte & 0x80 == 0 {
                return Ok(result);
            }
            shift += 7;
            let _ = i;
        }
        Err(SpmProtoError::VarintTooLong { at: start })
    }

    /// Read a field tag — a varint decomposed as `(field_number,
    /// wire_type)` where the low 3 bits are the wire type and the rest
    /// is the field number.
    fn read_tag(&mut self) -> Result<(u32, u8), SpmProtoError> {
        let raw = self.read_varint()?;
        let wire_type = (raw & 0x7) as u8;
        let field_number = (raw >> 3) as u32;
        Ok((field_number, wire_type))
    }

    /// Read exactly 4 little-endian bytes for a `fixed32` payload.
    fn read_fixed32(&mut self) -> Result<[u8; 4], SpmProtoError> {
        if self.pos + 4 > self.bytes.len() {
            return Err(SpmProtoError::UnexpectedEof {
                at: self.pos,
                context: "reading fixed32",
            });
        }
        let mut out = [0u8; 4];
        out.copy_from_slice(&self.bytes[self.pos..self.pos + 4]);
        self.pos += 4;
        Ok(out)
    }

    /// Read exactly 8 little-endian bytes for a `fixed64` payload — used
    /// by [`skip_field`] to discard `wire_type = 1` fields losslessly.
    fn read_fixed64(&mut self) -> Result<(), SpmProtoError> {
        if self.pos + 8 > self.bytes.len() {
            return Err(SpmProtoError::UnexpectedEof {
                at: self.pos,
                context: "reading fixed64",
            });
        }
        self.pos += 8;
        Ok(())
    }

    /// Read a length-delimited field: a varint length followed by N raw
    /// bytes. Returns the raw payload slice.
    fn read_length_delimited(&mut self) -> Result<&'a [u8], SpmProtoError> {
        let start = self.pos;
        let len = self.read_varint()? as usize;
        let remaining = self.bytes.len() - self.pos;
        if len > remaining {
            return Err(SpmProtoError::LengthOverflow {
                at: start,
                declared: len as u64,
                remaining,
            });
        }
        let out = &self.bytes[self.pos..self.pos + len];
        self.pos += len;
        Ok(out)
    }

    /// Skip an unknown field, respecting proto3 unknown-field forward
    /// compatibility rules.
    fn skip_field(&mut self, wire_type: u8) -> Result<(), SpmProtoError> {
        match wire_type {
            0 => {
                let _ = self.read_varint()?;
                Ok(())
            }
            1 => self.read_fixed64(),
            2 => {
                let _ = self.read_length_delimited()?;
                Ok(())
            }
            5 => {
                let _ = self.read_fixed32()?;
                Ok(())
            }
            other => Err(SpmProtoError::UnsupportedWireType {
                at: self.pos.saturating_sub(1),
                wire_type: other,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hand-encode one proto3 varint into a byte vector.
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

    /// Hand-encode a proto3 field tag `(field_number, wire_type)`.
    fn encode_tag(field_number: u32, wire_type: u8, out: &mut Vec<u8>) {
        encode_varint(((field_number as u64) << 3) | u64::from(wire_type), out);
    }

    /// Hand-encode one `SentencePiece` sub-message as a
    /// length-delimited payload.
    fn encode_sentence_piece(out: &mut Vec<u8>, piece: &str, score: f32, type_value: Option<u64>) {
        let mut inner = Vec::new();
        // piece = field 1, length-delimited.
        encode_tag(1, 2, &mut inner);
        encode_varint(piece.len() as u64, &mut inner);
        inner.extend_from_slice(piece.as_bytes());
        // score = field 2, fixed32.
        encode_tag(2, 5, &mut inner);
        inner.extend_from_slice(&score.to_le_bytes());
        // type = field 3, varint — but only when caller specifies it,
        // so the "missing type" case round-trips as Unspecified.
        if let Some(t) = type_value {
            encode_tag(3, 0, &mut inner);
            encode_varint(t, &mut inner);
        }

        // Now wrap as ModelProto.pieces (field 1, length-delimited).
        encode_tag(1, 2, out);
        encode_varint(inner.len() as u64, out);
        out.extend_from_slice(&inner);
    }

    #[test]
    fn parse_empty_buffer_yields_empty_pieces() {
        let model = parse_model(&[]).expect("empty proto is valid");
        assert_eq!(model.pieces.len(), 0);
    }

    #[test]
    fn parse_single_piece_round_trip() {
        let mut buf = Vec::new();
        encode_sentence_piece(&mut buf, "hello", 1.25, Some(1));

        let model = parse_model(&buf).expect("valid single piece");
        assert_eq!(model.pieces.len(), 1);
        assert_eq!(model.pieces[0].piece, "hello");
        assert!((model.pieces[0].score - 1.25).abs() < f32::EPSILON);
        assert_eq!(model.pieces[0].piece_type, PieceType::Normal);
    }

    #[test]
    fn parse_multiple_pieces_preserves_order_and_types() {
        let mut buf = Vec::new();
        encode_sentence_piece(&mut buf, "<unk>", 0.0, Some(2)); // Unknown
        encode_sentence_piece(&mut buf, "<s>", 0.0, Some(3)); // Control
        encode_sentence_piece(&mut buf, "</s>", 0.0, Some(3)); // Control
        encode_sentence_piece(&mut buf, "\u{2581}", -1.5, Some(1)); // Normal, word-start marker
        encode_sentence_piece(&mut buf, "he", -2.0, Some(1));
        encode_sentence_piece(&mut buf, "<0x00>", 0.0, Some(5)); // Byte

        let model = parse_model(&buf).expect("valid model");
        assert_eq!(model.pieces.len(), 6);
        assert_eq!(model.pieces[0].piece, "<unk>");
        assert_eq!(model.pieces[0].piece_type, PieceType::Unknown);
        assert_eq!(model.pieces[1].piece_type, PieceType::Control);
        assert_eq!(model.pieces[3].piece, "\u{2581}"); // U+2581 must survive UTF-8 round-trip
        assert_eq!(model.pieces[5].piece_type, PieceType::Byte);
    }

    #[test]
    fn missing_type_field_yields_unspecified() {
        let mut buf = Vec::new();
        encode_sentence_piece(&mut buf, "x", 0.5, None); // no type field
        let model = parse_model(&buf).expect("valid");
        assert_eq!(model.pieces[0].piece_type, PieceType::Unspecified);
    }

    #[test]
    fn unknown_top_level_field_is_skipped() {
        // Simulate a ModelProto with an unrecognized top-level varint
        // (field 99, wire type 0) BEFORE the pieces list, then a valid
        // piece — the parser must skip the unknown and still recover
        // the piece.
        let mut buf = Vec::new();
        encode_tag(99, 0, &mut buf);
        encode_varint(12345, &mut buf);
        encode_sentence_piece(&mut buf, "kept", -1.0, Some(1));
        let model = parse_model(&buf).expect("parse over unknown field");
        assert_eq!(model.pieces.len(), 1);
        assert_eq!(model.pieces[0].piece, "kept");
    }

    #[test]
    fn unknown_length_delimited_top_level_field_is_skipped() {
        // Insert a foreign length-delimited field (field 2, wire type 2)
        // — this exercises the length-varint + payload skip path that
        // any real SentencePiece model triggers because
        // `ModelProto.trainer_spec` is a nested message at field 2.
        let mut buf = Vec::new();
        encode_tag(2, 2, &mut buf);
        let junk = b"totally unknown message payload";
        encode_varint(junk.len() as u64, &mut buf);
        buf.extend_from_slice(junk);
        encode_sentence_piece(&mut buf, "survived", 0.75, Some(1));
        let model = parse_model(&buf).expect("parse over unknown length-delimited");
        assert_eq!(model.pieces.len(), 1);
        assert_eq!(model.pieces[0].piece, "survived");
    }

    #[test]
    fn unknown_fixed32_top_level_field_is_skipped() {
        let mut buf = Vec::new();
        encode_tag(42, 5, &mut buf);
        buf.extend_from_slice(&f32::to_le_bytes(2.5));
        encode_sentence_piece(&mut buf, "x", 0.0, Some(1));
        let model = parse_model(&buf).expect("parse over unknown fixed32");
        assert_eq!(model.pieces[0].piece, "x");
    }

    #[test]
    fn unknown_fixed64_top_level_field_is_skipped() {
        let mut buf = Vec::new();
        encode_tag(7, 1, &mut buf);
        buf.extend_from_slice(&[0u8; 8]);
        encode_sentence_piece(&mut buf, "x", 0.0, Some(1));
        let model = parse_model(&buf).expect("parse over unknown fixed64");
        assert_eq!(model.pieces[0].piece, "x");
    }

    #[test]
    fn unknown_wire_type_is_loud_error() {
        // wire_type = 3 (start_group) — deprecated in proto3, must not
        // silently pass.
        let mut buf = Vec::new();
        encode_tag(1, 3, &mut buf);
        let err = parse_model(&buf).expect_err("wire type 3 rejected");
        assert!(matches!(
            err,
            SpmProtoError::UnsupportedWireType { wire_type: 3, .. }
        ));
    }

    #[test]
    fn truncated_length_delimited_is_loud_error() {
        let mut buf = Vec::new();
        encode_tag(1, 2, &mut buf);
        encode_varint(100, &mut buf); // claims 100 bytes but only 3 follow
        buf.extend_from_slice(&[1, 2, 3]);
        let err = parse_model(&buf).expect_err("truncated payload rejected");
        assert!(matches!(err, SpmProtoError::LengthOverflow { .. }));
    }

    #[test]
    fn overlong_varint_is_loud_error() {
        // 11 continuation bytes = varint too long.
        let buf = vec![0x80; 11];
        let err = parse_model(&buf).expect_err("11-byte varint rejected");
        assert!(matches!(err, SpmProtoError::VarintTooLong { at: 0 }));
    }

    #[test]
    fn invalid_utf8_piece_is_loud_error() {
        // Piece with bytes 0xFF 0xFF — not valid UTF-8.
        let mut buf = Vec::new();
        let mut inner = Vec::new();
        encode_tag(1, 2, &mut inner);
        encode_varint(2, &mut inner);
        inner.extend_from_slice(&[0xFF, 0xFF]);
        encode_tag(1, 2, &mut buf);
        encode_varint(inner.len() as u64, &mut buf);
        buf.extend_from_slice(&inner);
        let err = parse_model(&buf).expect_err("non-UTF-8 piece rejected");
        assert!(matches!(err, SpmProtoError::InvalidUtf8 { .. }));
    }

    #[test]
    fn empty_piece_string_survives() {
        // An empty `piece` field is legal (a proto3 length-delimited
        // string with length 0 encodes as tag + 0x00 + no bytes).
        let mut buf = Vec::new();
        encode_sentence_piece(&mut buf, "", 0.0, Some(1));
        let model = parse_model(&buf).expect("empty piece is valid");
        assert_eq!(model.pieces[0].piece, "");
    }

    #[test]
    fn large_varint_field_number_round_trips() {
        // Field number > 15 forces a 2-byte tag varint; verifies the
        // tag decoder does not truncate the field-number bits when
        // reading the varint back.
        let mut buf = Vec::new();
        encode_tag(31, 2, &mut buf); // 31 << 3 = 248 -> continuation
        let inner = b"skip me";
        encode_varint(inner.len() as u64, &mut buf);
        buf.extend_from_slice(inner);
        encode_sentence_piece(&mut buf, "kept", 0.0, Some(1));
        let model = parse_model(&buf).expect("valid over 2-byte tag");
        assert_eq!(model.pieces[0].piece, "kept");
    }

    #[test]
    fn unknown_piece_type_value_is_preserved_as_other() {
        // Enum value 42 is not defined by SentencePiece today — the
        // parser must preserve it losslessly as `PieceType::Other(42)`
        // rather than silently mapping to `Normal` (forward-compat).
        let mut buf = Vec::new();
        encode_sentence_piece(&mut buf, "future", 0.0, Some(42));
        let model = parse_model(&buf).expect("valid");
        assert_eq!(model.pieces[0].piece_type, PieceType::Other(42));
    }
}
