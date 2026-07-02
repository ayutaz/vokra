//! Whisper detokenizer (token id → UTF-8 text).
//!
//! Whisper uses a GPT-2 byte-level BPE vocabulary. **Decoding** (the only
//! direction M0 needs) is: map each token id to its raw byte string, drop the
//! special `<|…|>` tokens, concatenate the bytes and UTF-8-decode. A single
//! token's bytes may not be valid UTF-8 on their own (BPE can split a
//! multi-byte character); only the concatenation is, so bytes are joined first
//! and decoded once (lossily, matching `transformers`' `errors="replace"`).
//!
//! # Where the vocabulary comes from (M0)
//!
//! The vocabulary is **not** in the safetensors checkpoint (it lives in the
//! tokenizer files), so the M0-03 safetensors-only converter cannot embed it in
//! the GGUF. M0 therefore loads it from a compact binary produced by the parity
//! dump script ([`WhisperTokenizer::from_bytes`]); the demo takes it as an
//! optional sidecar. The **designed contract** is a `vokra.tokenizer.model`
//! GGUF blob in exactly this format ([`WhisperTokenizer::from_gguf`]); wiring a
//! tokenizer-aware converter is a follow-up.
//!
//! # Binary format (`from_bytes`)
//!
//! Little-endian: `u32 count`, then `count` records of
//! `{ u8 special; u16 byte_len; [u8; byte_len] }` indexed by token id.

use vokra_core::gguf::{GgufFile, GgufMetadataValue};
use vokra_core::{Result, VokraError};

/// GGUF key carrying the tokenizer binary blob (designed contract; not written
/// by the M0 converter).
pub const KEY_TOKENIZER_MODEL: &str = "vokra.tokenizer.model";

struct Entry {
    special: bool,
    bytes: Vec<u8>,
}

/// A loaded Whisper detokenizer.
pub struct WhisperTokenizer {
    entries: Vec<Entry>,
    eot: u32,
}

impl WhisperTokenizer {
    /// Parses the tokenizer binary (see the module docs for the format).
    ///
    /// `eot` is the end-of-transcript id (from [`WhisperConfig`], used only for
    /// diagnostics / callers that want to strip it).
    ///
    /// [`WhisperConfig`]: super::config::WhisperConfig
    ///
    /// # Errors
    ///
    /// [`VokraError::ModelLoad`] if the buffer is truncated or malformed.
    pub fn from_bytes(data: &[u8], eot: u32) -> Result<Self> {
        let mut r = Cursor::new(data);
        let count = r.u32()? as usize;
        let mut entries = Vec::with_capacity(count);
        for _ in 0..count {
            let special = r.u8()? != 0;
            let len = r.u16()? as usize;
            let bytes = r.take(len)?.to_vec();
            entries.push(Entry { special, bytes });
        }
        Ok(Self { entries, eot })
    }

    /// Loads the tokenizer from a `vokra.tokenizer.model` GGUF blob.
    ///
    /// The M0 converter does not write this key, so this returns
    /// [`VokraError::ModelLoad`] on the current Whisper GGUFs; use
    /// [`from_bytes`](Self::from_bytes) with the parity fixture / sidecar until
    /// a tokenizer-aware converter lands.
    pub fn from_gguf(file: &GgufFile, eot: u32) -> Result<Self> {
        match file.get(KEY_TOKENIZER_MODEL) {
            Some(GgufMetadataValue::Array(arr)) => {
                let mut bytes = Vec::with_capacity(arr.values.len());
                for v in &arr.values {
                    let b = v
                        .as_u64()
                        .and_then(|n| u8::try_from(n).ok())
                        .ok_or_else(|| {
                            VokraError::ModelLoad(
                                "whisper tokenizer: blob element not a byte".into(),
                            )
                        })?;
                    bytes.push(b);
                }
                Self::from_bytes(&bytes, eot)
            }
            _ => Err(VokraError::ModelLoad(format!(
                "whisper tokenizer: `{KEY_TOKENIZER_MODEL}` absent (M0 converter does not \
                 embed the tokenizer; load it from the parity fixture / sidecar)"
            ))),
        }
    }

    /// Vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.entries.len()
    }

    /// The end-of-transcript token id this tokenizer was built with.
    pub fn eot(&self) -> u32 {
        self.eot
    }

    /// Returns `true` if `id` is a special `<|…|>` token (not rendered as text).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] if `id` is out of range.
    pub fn is_special(&self, id: u32) -> Result<bool> {
        self.entry(id).map(|e| e.special)
    }

    /// Decodes a token id sequence to text, skipping special tokens.
    ///
    /// Byte joins then UTF-8-decodes once (lossily), so invalid byte sequences
    /// yield U+FFFD rather than a panic (NFR-RL-07). An out-of-range id is an
    /// explicit [`VokraError::InvalidArgument`].
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        let mut bytes = Vec::new();
        for &id in ids {
            let e = self.entry(id)?;
            if !e.special {
                bytes.extend_from_slice(&e.bytes);
            }
        }
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    fn entry(&self, id: u32) -> Result<&Entry> {
        self.entries.get(id as usize).ok_or_else(|| {
            VokraError::InvalidArgument(format!(
                "whisper tokenizer: token id {id} >= vocab {}",
                self.entries.len()
            ))
        })
    }
}

/// A tiny bounds-checked little-endian cursor for the tokenizer blob.
struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.pos + n > self.data.len() {
            return Err(VokraError::ModelLoad(
                "whisper tokenizer: truncated blob".into(),
            ));
        }
        let s = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16> {
        let b = self.take(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }
    fn u32(&mut self) -> Result<u32> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a 4-token blob: "he", "llo", "<|special|>", and a byte that with
    /// its neighbour forms a 2-byte UTF-8 char split across tokens.
    fn blob() -> Vec<u8> {
        let mut v = Vec::new();
        let entries: &[(u8, &[u8])] = &[
            (0, b"he"),
            (0, b"llo"),
            (1, b""),     // special
            (0, &[0xC3]), // first byte of 'é' (U+00E9 = C3 A9)
            (0, &[0xA9]), // second byte of 'é'
        ];
        v.extend_from_slice(&(entries.len() as u32).to_le_bytes());
        for (sp, bytes) in entries {
            v.push(*sp);
            v.extend_from_slice(&(bytes.len() as u16).to_le_bytes());
            v.extend_from_slice(bytes);
        }
        v
    }

    #[test]
    fn decodes_and_skips_special_tokens() {
        let t = WhisperTokenizer::from_bytes(&blob(), 2).unwrap();
        assert_eq!(t.vocab_size(), 5);
        assert_eq!(t.decode(&[0, 1]).unwrap(), "hello");
        // Special token (id 2) contributes nothing.
        assert_eq!(t.decode(&[0, 2, 1]).unwrap(), "hello");
    }

    #[test]
    fn multibyte_char_split_across_tokens_is_joined() {
        let t = WhisperTokenizer::from_bytes(&blob(), 2).unwrap();
        // ids 3,4 = 0xC3,0xA9 → 'é'.
        assert_eq!(t.decode(&[3, 4]).unwrap(), "é");
    }

    #[test]
    fn out_of_range_id_is_an_error_not_a_panic() {
        let t = WhisperTokenizer::from_bytes(&blob(), 2).unwrap();
        assert!(matches!(
            t.decode(&[99]),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn lone_invalid_byte_decodes_lossily_without_panic() {
        // id 3 alone (0xC3) is not valid UTF-8 → U+FFFD, but no panic / error.
        let t = WhisperTokenizer::from_bytes(&blob(), 2).unwrap();
        let s = t.decode(&[3]).unwrap();
        assert!(s.contains('\u{FFFD}'));
    }

    #[test]
    fn truncated_blob_is_rejected() {
        assert!(matches!(
            WhisperTokenizer::from_bytes(&[1, 0, 0, 0], 0), // count=1, no record
            Err(VokraError::ModelLoad(_))
        ));
    }
}
