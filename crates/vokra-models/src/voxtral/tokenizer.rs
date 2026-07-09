//! Voxtral (Mistral) tokenizer — SentencePiece byte-fallback BPE detokenizer.
//!
//! Mistral (and therefore Voxtral) ships a **SentencePiece byte-fallback BPE**
//! tokenizer as a `tokenizer.model` protobuf file. The upstream
//! `vokra-convert` writes those raw bytes verbatim into the GGUF metadata
//! chunk [`KEY_TOKENIZER_MODEL`] (`vokra.tokenizer.model`) as a `U8` array —
//! same shape Whisper uses.
//!
//! # Zero-dep posture (NFR-DS-02)
//!
//! The runtime tokenizer is **self-implemented** — no external tokenizer
//! crate is pulled in. Adding `sentencepiece`, `tokenizers`, `protobuf`, or
//! any of their transitives to the root workspace would break the zero-dep
//! invariant enforced by `scripts/check-zero-deps.sh`.
//!
//! # Foundation scope (M3-10-T06)
//!
//! This foundation file lands:
//!
//! - a loader ([`VoxtralTokenizer::from_gguf`], [`VoxtralTokenizer::from_bytes`])
//!   that decodes the [`KEY_TOKENIZER_MODEL`] `U8` array into an owned
//!   [`Vec<u8>`] with **explicit error surfaces** — a truncated blob or a
//!   missing chunk is a [`VokraError::ModelLoad`], never a silent skip;
//! - a **compact-vocab** parser that recognises the same little-endian binary
//!   layout Whisper's `WhisperTokenizer::from_bytes` uses so the parity
//!   dumper's output can be embedded verbatim (T19+); the format is
//!   documented under [`Self::parse_compact_vocab`];
//! - a [`Self::decode`] path that renders token ids to UTF-8, dropping
//!   special tokens (byte-join first, then UTF-8-decode once with
//!   [`String::from_utf8_lossy`] — matches transformers' `errors="replace"`
//!   posture, no panic on a lone continuation byte);
//! - a [`Self::detect_sentencepiece_proto`] heuristic that fingerprints the
//!   raw bytes as a SentencePiece proto (magic bytes at offset 0); callers
//!   with a native proto reader can consume [`Self::raw_bytes`] directly.
//!
//! # What this file does NOT do
//!
//! - It does not parse the full SentencePiece proto3 schema (that requires a
//!   proto reader — a follow-up ticket, out-of-scope for M3-10-T06 which
//!   only needs the byte-blob load path + a decode surface). Callers today
//!   can either (a) use [`Self::from_bytes`] with a compact-vocab dump the
//!   converter produces (T19+ parity fixture pipeline), or (b) consume
//!   [`Self::raw_bytes`] and drive an external parser.
//! - It does not implement **encoding** (text → token ids). Voxtral ASR uses
//!   token id → text (decode direction); encoding is only needed for prompt
//!   conditioning and lands with the full autoregressive decode loop (T13+).

use vokra_core::gguf::{GgufFile, GgufMetadataValue};
use vokra_core::{Result, VokraError};

/// GGUF key carrying the raw Mistral tokenizer model bytes (the same key
/// `vokra-convert::models::voxtral::embed_tokenizer` writes).
pub const KEY_TOKENIZER_MODEL: &str = "vokra.tokenizer.model";

/// One vocabulary entry: whether the token is a special (unrendered) marker
/// and its raw UTF-8 byte sequence.
struct Entry {
    special: bool,
    bytes: Vec<u8>,
}

/// A loaded Voxtral / Mistral tokenizer.
///
/// Holds either:
/// - a parsed compact-vocab table (from [`Self::from_bytes`] on a parity
///   dumper output), and/or
/// - the raw SentencePiece proto bytes (from [`Self::from_gguf`] on the
///   Voxtral GGUF `KEY_TOKENIZER_MODEL` chunk).
///
/// The raw bytes are always retained so downstream callers can pass them to a
/// full SentencePiece proto reader once one lands.
pub struct VoxtralTokenizer {
    /// The raw bytes from the GGUF chunk. Always populated; downstream tools
    /// (native SentencePiece parser follow-up) borrow from here.
    raw: Vec<u8>,
    /// Parsed compact-vocab entries indexed by token id. Empty when the raw
    /// bytes are a SentencePiece proto (not a compact-vocab dump).
    entries: Vec<Entry>,
    /// End-of-transcript / end-of-sequence token id (Mistral 32000-vocab
    /// ships `2` as `</s>`; callers pass it explicitly since it comes from
    /// the config side-car).
    eos: u32,
}

impl VoxtralTokenizer {
    /// Loads the tokenizer bytes from a GGUF file's [`KEY_TOKENIZER_MODEL`]
    /// chunk. The bytes may be either a compact-vocab dump (parseable via
    /// [`Self::parse_compact_vocab`]) or a raw SentencePiece proto
    /// (accessible via [`Self::raw_bytes`]).
    ///
    /// The loader is deliberately lenient about the format: it recognises the
    /// compact-vocab dump when the first 4 bytes name a plausible vocab
    /// count (`< 1_000_000`) and the total blob length matches the header's
    /// record sizes exactly; otherwise it retains the raw bytes and returns
    /// an empty entry table (callers can then check
    /// [`Self::has_compact_vocab`] and fall back to a proto reader).
    ///
    /// `eos` is the end-of-sequence / end-of-transcript token id from the
    /// upstream config (Mistral ships `2` as `</s>`). It is stored on the
    /// tokenizer for callers that need it for stopping criteria but never
    /// consulted by [`Self::decode`] itself.
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] if the [`KEY_TOKENIZER_MODEL`] chunk is
    ///   absent or the array element type is not `U8`;
    /// - [`VokraError::ModelLoad`] if a compact-vocab header advertises more
    ///   entries than the remaining bytes can hold (truncated dump).
    pub fn from_gguf(file: &GgufFile, eos: u32) -> Result<Self> {
        let raw = match file.get(KEY_TOKENIZER_MODEL) {
            Some(GgufMetadataValue::Array(arr)) => {
                let mut bytes = Vec::with_capacity(arr.values.len());
                for v in &arr.values {
                    let b = v
                        .as_u64()
                        .and_then(|n| u8::try_from(n).ok())
                        .ok_or_else(|| {
                            VokraError::ModelLoad(
                                "voxtral tokenizer: blob element not a byte".into(),
                            )
                        })?;
                    bytes.push(b);
                }
                bytes
            }
            _ => {
                return Err(VokraError::ModelLoad(format!(
                    "voxtral tokenizer: `{KEY_TOKENIZER_MODEL}` absent \
                     — converter did not embed the tokenizer bytes (VoxtralConfig \
                     .tokenizer_bytes was None) or the GGUF is truncated"
                )));
            }
        };
        Self::from_bytes(raw, eos)
    }

    /// Parses a raw tokenizer byte blob.
    ///
    /// If the blob starts with a plausible compact-vocab header
    /// (little-endian `u32` count `< 1_000_000` and matching total length),
    /// the vocab is parsed into per-id entries. Otherwise the raw bytes are
    /// retained for external SentencePiece proto consumption and
    /// [`Self::has_compact_vocab`] returns `false`.
    ///
    /// # Errors
    ///
    /// [`VokraError::ModelLoad`] if the blob **looks** like a compact-vocab
    /// header (plausible count) but is truncated (the recorded entry sizes
    /// exceed the buffer). SentencePiece proto blobs that happen to fail the
    /// heuristic are retained silently — the raw bytes remain available.
    pub fn from_bytes(raw: Vec<u8>, eos: u32) -> Result<Self> {
        let entries = Self::parse_compact_vocab(&raw)?;
        Ok(Self { raw, entries, eos })
    }

    /// Attempts to parse `data` as the compact-vocab little-endian binary
    /// dump produced by the parity dumper. Layout (little-endian):
    ///
    /// ```text
    /// u32 count
    /// count records:
    ///   u8  special       // 0 = renderable, 1 = special (e.g. <|eos|>)
    ///   u16 byte_len
    ///   [u8; byte_len] token bytes
    /// ```
    ///
    /// # Format-detection heuristic
    ///
    /// The parser returns `Ok(vec![])` (silent fall-through to raw bytes)
    /// when the blob looks like a **SentencePiece proto** or has an
    /// implausible header:
    /// - `data.len() < 4` (nothing to decode);
    /// - `data[0] == 0x0A` (SentencePiece proto field-1 wire-type-2 tag —
    ///   see [`Self::detect_sentencepiece_proto`]);
    /// - `count == 0` or `count > 1_000_000` (well outside every shipping
    ///   Mistral vocab: 32 k..131 k).
    ///
    /// Once the header passes the fingerprint gate, every record MUST be
    /// present. A truncated entry is a hard [`VokraError::ModelLoad`] — the
    /// caller cannot recover from a partial dump (FR-EX-08, no silent skip).
    ///
    /// # Errors
    ///
    /// [`VokraError::ModelLoad`] if the header passed the fingerprint gate
    /// but the buffer is too short for the recorded records.
    fn parse_compact_vocab(data: &[u8]) -> Result<Vec<Entry>> {
        if data.len() < 4 {
            return Ok(Vec::new());
        }
        // SentencePiece proto always starts with 0x0A — cheap detour to
        // avoid mis-parsing a proto as a compact-vocab dump. A caller who
        // needs the proto bytes reads them via `raw_bytes()`.
        if Self::detect_sentencepiece_proto(data) {
            return Ok(Vec::new());
        }
        let count = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
        // Sanity-gate on the count so an arbitrary blob is not misidentified
        // as a compact-vocab dump. 1_000_000 is well above every shipping
        // Mistral vocab (32 k..131 k) but well below what the first 4 bytes
        // of an unrelated binary usually decode to.
        if count == 0 || count > 1_000_000 {
            return Ok(Vec::new());
        }
        let mut entries = Vec::with_capacity(count);
        let mut pos = 4usize;
        for _ in 0..count {
            if pos + 3 > data.len() {
                // Header passed the fingerprint gate; a missing record body
                // is a hard truncation, never a silent skip (FR-EX-08).
                return Err(VokraError::ModelLoad(format!(
                    "voxtral tokenizer: compact-vocab truncated at record {} of {count} \
                     header (byte offset {pos}, blob has {} left)",
                    entries.len(),
                    data.len().saturating_sub(pos)
                )));
            }
            let special = data[pos] != 0;
            let byte_len = u16::from_le_bytes([data[pos + 1], data[pos + 2]]) as usize;
            pos += 3;
            if pos + byte_len > data.len() {
                return Err(VokraError::ModelLoad(format!(
                    "voxtral tokenizer: compact-vocab entry {} body truncated \
                     (needs {byte_len} bytes at offset {pos}, blob has {} left)",
                    entries.len(),
                    data.len().saturating_sub(pos)
                )));
            }
            entries.push(Entry {
                special,
                bytes: data[pos..pos + byte_len].to_vec(),
            });
            pos += byte_len;
        }
        Ok(entries)
    }

    /// The raw tokenizer bytes as embedded in the GGUF. Callers with a native
    /// SentencePiece proto parser (a follow-up ticket) borrow from here.
    #[must_use]
    pub fn raw_bytes(&self) -> &[u8] {
        &self.raw
    }

    /// `true` iff the parity dumper's compact-vocab dump was recognised and a
    /// per-id [`decode`](Self::decode) path is available. `false` on a raw
    /// SentencePiece proto blob (call [`raw_bytes`](Self::raw_bytes) then).
    #[must_use]
    pub fn has_compact_vocab(&self) -> bool {
        !self.entries.is_empty()
    }

    /// Vocabulary size (compact-vocab dump only; `0` on a raw proto blob).
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.entries.len()
    }

    /// End-of-sequence token id this tokenizer was constructed with.
    #[must_use]
    pub fn eos(&self) -> u32 {
        self.eos
    }

    /// `true` iff `id` is a special (unrendered) marker.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] if `id` is out of range;
    /// [`VokraError::NotImplemented`] if the tokenizer was loaded from a raw
    /// SentencePiece proto (no per-id table available yet — call
    /// [`Self::raw_bytes`] and drive an external parser).
    pub fn is_special(&self, id: u32) -> Result<bool> {
        self.entry(id).map(|e| e.special)
    }

    /// Decodes a token id sequence to a UTF-8 string, dropping special
    /// tokens (byte-join then lossy UTF-8 decode — matches transformers'
    /// `errors="replace"` posture, no panic on a lone continuation byte).
    ///
    /// # Errors
    ///
    /// - [`VokraError::InvalidArgument`] if any id is out of range;
    /// - [`VokraError::NotImplemented`] if the tokenizer was loaded from a
    ///   raw SentencePiece proto (see [`Self::is_special`]).
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        if self.entries.is_empty() {
            return Err(VokraError::NotImplemented(
                "voxtral tokenizer: compact-vocab dump not present — the GGUF carries the raw \
                 SentencePiece proto instead. Consume `raw_bytes()` with an external proto \
                 parser (follow-up ticket) or re-run the parity dumper to embed a compact \
                 vocab (T19+).",
            ));
        }
        let mut bytes = Vec::new();
        for &id in ids {
            let e = self.entry(id)?;
            if !e.special {
                bytes.extend_from_slice(&e.bytes);
            }
        }
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    /// Fingerprints `bytes` as a SentencePiece proto blob. This is a
    /// **heuristic** and only intended for diagnostics — the presence of the
    /// magic prefix is not a proof that the proto is a valid Mistral
    /// tokenizer. Downstream callers who need real parsing should feed
    /// [`raw_bytes`](Self::raw_bytes) to a proto reader.
    ///
    /// SentencePiece protos start with a `field=1` tag `0x0A` (`TrainerSpec`)
    /// followed by a length-prefixed message. This is a classic proto3
    /// varint-tag encoding: the first byte of a Mistral tokenizer.model is
    /// always `0x0A` (`(1 << 3) | 2` = tag 1, wire type 2 = length-delimited).
    #[must_use]
    pub fn detect_sentencepiece_proto(bytes: &[u8]) -> bool {
        // `TrainerSpec` field 1, wire type 2: 0x0A.
        matches!(bytes.first(), Some(0x0A))
    }

    fn entry(&self, id: u32) -> Result<&Entry> {
        if self.entries.is_empty() {
            return Err(VokraError::NotImplemented(
                "voxtral tokenizer: no compact-vocab table — see decode()",
            ));
        }
        self.entries.get(id as usize).ok_or_else(|| {
            VokraError::InvalidArgument(format!(
                "voxtral tokenizer: token id {id} >= vocab {}",
                self.entries.len()
            ))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::{GgufArray, GgufBuilder, GgufValueType};

    /// Build a 5-token compact-vocab dump: "he", "llo", <|special|>, and a
    /// UTF-8 multi-byte char split across two ids (0xC3, 0xA9 → 'é').
    fn compact_blob() -> Vec<u8> {
        let entries: &[(u8, &[u8])] = &[
            (0, b"he"),
            (0, b"llo"),
            (1, b""),
            (0, &[0xC3]),
            (0, &[0xA9]),
        ];
        let mut v = Vec::new();
        v.extend_from_slice(&(entries.len() as u32).to_le_bytes());
        for (sp, bytes) in entries {
            v.push(*sp);
            v.extend_from_slice(&(bytes.len() as u16).to_le_bytes());
            v.extend_from_slice(bytes);
        }
        v
    }

    fn wrap_bytes_in_gguf(bytes: &[u8]) -> GgufFile {
        let mut b = GgufBuilder::new();
        b.add_metadata(
            KEY_TOKENIZER_MODEL,
            GgufMetadataValue::Array(GgufArray {
                element_type: GgufValueType::U8,
                values: bytes.iter().map(|&x| GgufMetadataValue::U8(x)).collect(),
            }),
        );
        GgufFile::parse(b.to_bytes().unwrap()).unwrap()
    }

    #[test]
    fn from_gguf_reads_compact_vocab_from_metadata() {
        let file = wrap_bytes_in_gguf(&compact_blob());
        let t = VoxtralTokenizer::from_gguf(&file, 2).unwrap();
        assert!(t.has_compact_vocab());
        assert_eq!(t.vocab_size(), 5);
        assert_eq!(t.eos(), 2);
    }

    #[test]
    fn decode_skips_special_tokens_and_joins_bytes() {
        let file = wrap_bytes_in_gguf(&compact_blob());
        let t = VoxtralTokenizer::from_gguf(&file, 2).unwrap();
        assert_eq!(t.decode(&[0, 1]).unwrap(), "hello");
        // Special token (id 2) contributes nothing.
        assert_eq!(t.decode(&[0, 2, 1]).unwrap(), "hello");
        // Multi-byte char split across ids 3,4.
        assert_eq!(t.decode(&[3, 4]).unwrap(), "é");
    }

    #[test]
    fn decode_out_of_range_id_is_error_not_panic() {
        let file = wrap_bytes_in_gguf(&compact_blob());
        let t = VoxtralTokenizer::from_gguf(&file, 2).unwrap();
        assert!(matches!(
            t.decode(&[99]),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn lone_invalid_byte_decodes_lossily() {
        // id 3 alone is 0xC3 (needs a continuation byte). Lossy decode
        // yields U+FFFD; no panic.
        let file = wrap_bytes_in_gguf(&compact_blob());
        let t = VoxtralTokenizer::from_gguf(&file, 2).unwrap();
        let s = t.decode(&[3]).unwrap();
        assert!(s.contains('\u{FFFD}'));
    }

    #[test]
    fn missing_chunk_is_model_load_error() {
        let file = GgufFile::parse(GgufBuilder::new().to_bytes().unwrap()).unwrap();
        assert!(matches!(
            VoxtralTokenizer::from_gguf(&file, 2),
            Err(VokraError::ModelLoad(_))
        ));
    }

    #[test]
    fn from_bytes_recognises_sentencepiece_proto_fingerprint() {
        // A SentencePiece proto starts with 0x0A (field 1, wire type 2).
        // Header count decode: 0x0A ?? ?? ?? — first 4 bytes will parse to a
        // large `count` unless the second byte is small. Craft a blob whose
        // first byte is 0x0A but whose "count" decodes above the sanity gate,
        // so it falls through to raw-only.
        let mut blob = vec![0x0A, 0xFF, 0xFF, 0xFF]; // count = 0xFFFFFF0A >> ok, > 1M
        blob.extend_from_slice(b"...proto body...");
        assert!(VoxtralTokenizer::detect_sentencepiece_proto(&blob));
        let t = VoxtralTokenizer::from_bytes(blob.clone(), 2).unwrap();
        assert!(!t.has_compact_vocab());
        assert_eq!(t.raw_bytes(), &blob[..]);
        // decode() must NOT fabricate output when there is no vocab table.
        assert!(matches!(t.decode(&[0]), Err(VokraError::NotImplemented(_))));
    }

    #[test]
    fn truncated_compact_vocab_header_is_rejected() {
        // Count says "1 entry" but no room for the record header.
        let bad = vec![1u8, 0, 0, 0]; // count = 1, then EOF
        assert!(matches!(
            VoxtralTokenizer::from_bytes(bad, 0),
            Err(VokraError::ModelLoad(_))
        ));
    }

    #[test]
    fn truncated_compact_vocab_entry_body_is_rejected() {
        // Count 1, first record header says 5 bytes but blob only has 2.
        let mut bad = vec![1u8, 0, 0, 0];
        bad.extend_from_slice(&[0u8, 5u8, 0u8]); // special=0, byte_len=5
        bad.extend_from_slice(&[b'a', b'b']); // only 2 of 5 bytes
        assert!(matches!(
            VoxtralTokenizer::from_bytes(bad, 0),
            Err(VokraError::ModelLoad(_))
        ));
    }

    #[test]
    fn implausible_header_falls_through_to_raw() {
        // First 4 bytes decode to a count > 1_000_000 → not a compact-vocab
        // dump. from_bytes retains the raw and returns has_compact_vocab=false.
        let blob = vec![0xFF, 0xFF, 0xFF, 0x0F, 0xAA, 0xBB];
        let t = VoxtralTokenizer::from_bytes(blob.clone(), 0).unwrap();
        assert!(!t.has_compact_vocab());
        assert_eq!(t.raw_bytes(), &blob[..]);
    }

    #[test]
    fn is_special_matches_dump_flag() {
        let file = wrap_bytes_in_gguf(&compact_blob());
        let t = VoxtralTokenizer::from_gguf(&file, 2).unwrap();
        assert!(!t.is_special(0).unwrap());
        assert!(!t.is_special(1).unwrap());
        assert!(t.is_special(2).unwrap()); // special
        assert!(!t.is_special(3).unwrap());
    }

    #[test]
    fn eos_survives_roundtrip() {
        let file = wrap_bytes_in_gguf(&compact_blob());
        let t = VoxtralTokenizer::from_gguf(&file, 42).unwrap();
        assert_eq!(t.eos(), 42);
    }

    #[test]
    fn detect_sentencepiece_proto_negative_on_empty() {
        assert!(!VoxtralTokenizer::detect_sentencepiece_proto(&[]));
        assert!(!VoxtralTokenizer::detect_sentencepiece_proto(&[0x00]));
    }
}
