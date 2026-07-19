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

/// Python `string.punctuation` (the 32 ASCII punctuation characters), the exact
/// set openai-whisper `tokenizer.py::split_tokens_on_spaces` treats as a
/// word boundary (`subword.strip() in string.punctuation`). Transcribed verbatim
/// from CPython's `string` module (CLAUDE.md ハルシネーション厳禁); do not
/// reorder or "clean up" — membership is a substring test that must match
/// Python's `in` operator character-for-character.
const ASCII_PUNCTUATION: &str = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";

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

    /// Per-word token counts for `tokens`, grouping subword tokens into words
    /// (M4-20, FR-OP-40 `word_timestamps`). This is the tokenizer-specific
    /// grouping [`vokra_core::decode::words_from_alignment`] needs to turn the
    /// per-token cross-attention timings into one timing per **word**.
    ///
    /// Transcribed from openai-whisper `tokenizer.py::split_to_word_tokens`
    /// (the space-delimited path — `split_tokens_on_unicode` then
    /// `split_tokens_on_spaces`, CLAUDE.md ハルシネーション厳禁): first regroup
    /// the tokens so each subword decodes to valid unicode (a multi-byte char
    /// split across BPE tokens is merged), then merge subwords into words,
    /// starting a **new word** on a leading space, a punctuation subword, a
    /// special first token, or the very first subword; otherwise the subword
    /// joins the running word. The returned counts sum to `tokens.len()`, so
    /// they can be handed straight to `words_from_alignment` alongside the
    /// per-token times.
    ///
    /// # Boundary vs openai
    ///
    /// * The decode used here is [`decode`](Self::decode) (skips specials),
    ///   not openai's `decode_with_timestamps` (renders timestamp tokens as
    ///   `<|x.xx|>`); a special first token is still caught by the
    ///   `is_special` check, so the word boundaries agree for text tokens.
    ///   Exact per-token numeric parity vs openai stays owner (needs a real
    ///   checkpoint + vocab).
    /// * The CJK "no spaces" path (openai splits zh/ja/th/lo/my/yue on unicode
    ///   only) needs a language tag the M0 tokenizer does not carry; that
    ///   refinement is a follow-up. The space path is the default.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] if any token id is out of range (via
    /// [`decode`](Self::decode) / [`is_special`](Self::is_special)).
    pub fn word_token_lens(&self, tokens: &[u32]) -> Result<Vec<usize>> {
        // Stage 1: unicode subword grouping (openai split_tokens_on_unicode).
        let subword_lens = self.split_lens_on_unicode(tokens)?;

        // Stage 2: merge subwords into words (openai split_tokens_on_spaces).
        let mut words: Vec<usize> = Vec::with_capacity(subword_lens.len());
        let mut cursor = 0usize;
        for &sub_len in &subword_lens {
            let sub_tokens = &tokens[cursor..cursor + sub_len];
            let subword = self.decode(sub_tokens)?;
            // `sub_len >= 1` for every group, so `sub_tokens[0]` is valid.
            let special = self.is_special(sub_tokens[0])?;
            let with_space = subword.starts_with(' ');
            // Python `subword.strip() in string.punctuation` is a substring
            // test; `str::contains(&str)` matches it (incl. the empty-string
            // edge that Python's `in` also treats as present).
            let punctuation = ASCII_PUNCTUATION.contains(subword.trim());
            if special || with_space || punctuation || words.is_empty() {
                words.push(sub_len);
            } else {
                *words.last_mut().expect("words non-empty in else branch") += sub_len;
            }
            cursor += sub_len;
        }
        Ok(words)
    }

    /// Per-subword token counts, regrouping `tokens` so each group decodes to
    /// valid unicode (openai-whisper `tokenizer.py::split_tokens_on_unicode`,
    /// transcribed — CLAUDE.md ハルシネーション厳禁). A byte token that is the
    /// lone first half of a multi-byte character decodes to U+FFFD on its own;
    /// it is held back until the group decodes cleanly, unless the replacement
    /// character is *genuine* at that position in the full decode (i.e. the
    /// input really carried an undecodable byte there — don't wait forever).
    ///
    /// The counts sum to `tokens.len()`: a trailing group that never decodes
    /// cleanly (an incomplete character at the very end) is still emitted, so
    /// the caller's `sum(lens) == tokens.len()` invariant holds. openai drops
    /// such a dangling remainder, but keeping it accounted for is required by
    /// [`words_from_alignment`](vokra_core::decode::words_from_alignment) and
    /// only differs on malformed tails a well-formed hypothesis never produces.
    fn split_lens_on_unicode(&self, tokens: &[u32]) -> Result<Vec<usize>> {
        const REPLACEMENT: char = '\u{FFFD}';
        // Full decode (as a scalar sequence) to distinguish a genuine U+FFFD
        // from one produced by a split multi-byte char.
        let decoded_full: Vec<char> = self.decode(tokens)?.chars().collect();

        let mut lens: Vec<usize> = Vec::new();
        let mut group_start = 0usize; // first token index of the current group
        let mut unicode_offset = 0usize; // scalars emitted by earlier groups
        for end in 1..=tokens.len() {
            let decoded: Vec<char> = self.decode(&tokens[group_start..end])?.chars().collect();
            let emit = match decoded.iter().position(|&c| c == REPLACEMENT) {
                None => true,
                Some(pos) => decoded_full.get(unicode_offset + pos).copied() == Some(REPLACEMENT),
            };
            if emit {
                lens.push(end - group_start);
                unicode_offset += decoded.len();
                group_start = end;
            }
        }
        // Defensive: account for a trailing group that never decoded cleanly
        // (keeps `sum(lens) == tokens.len()` for the caller). A well-formed
        // hypothesis ends on a complete character and never hits this.
        if group_start < tokens.len() {
            lens.push(tokens.len() - group_start);
        }
        Ok(lens)
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

    // ---- M4-20: subword -> word grouping (word_token_lens) ----------------

    /// Builds a tokenizer from a synthetic `(special, bytes)` vocabulary — the
    /// "synthetic token->string map" the M4-20 split-rule tests drive.
    fn tok(entries: &[(u8, &[u8])]) -> WhisperTokenizer {
        let mut v = Vec::new();
        v.extend_from_slice(&(entries.len() as u32).to_le_bytes());
        for (sp, bytes) in entries {
            v.push(*sp);
            v.extend_from_slice(&(bytes.len() as u16).to_le_bytes());
            v.extend_from_slice(bytes);
        }
        WhisperTokenizer::from_bytes(&v, 0).unwrap()
    }

    #[test]
    fn word_token_lens_merges_leading_space_and_punctuation() {
        // openai split_tokens_on_spaces: id0 "hel" starts word 0, id1 "lo"
        // (no space/punct/special) joins it -> "hello"; id2 "," is punctuation
        // -> new word; id3 " world" has a leading space -> new word.
        // Tokens "hel","lo",","," world"  ->  word_token_lens [2, 1, 1].
        let t = tok(&[(0, b"hel"), (0, b"lo"), (0, b","), (0, b" world")]);
        assert_eq!(t.word_token_lens(&[0, 1, 2, 3]).unwrap(), vec![2, 1, 1]);
    }

    #[test]
    fn word_token_lens_groups_multibyte_char_split_across_tokens() {
        // 'é' = U+00E9 = bytes C3 A9. id0 " a" (leading space) is word 0;
        // id1 = [0x20,0xC3] (space + first half of é) decodes to " \u{FFFD}"
        // and is held back until id2 = [0xA9] completes it -> subword " é"
        // (one unicode group of 2 tokens), which starts a new word (leading
        // space). So [" a"], [" é"]  ->  word_token_lens [1, 2].
        let t = tok(&[(0, b" a"), (0, &[0x20, 0xC3]), (0, &[0xA9])]);
        // Sanity: the split really merges the two byte tokens into one subword.
        assert_eq!(t.split_lens_on_unicode(&[0, 1, 2]).unwrap(), vec![1, 2]);
        assert_eq!(t.word_token_lens(&[0, 1, 2]).unwrap(), vec![1, 2]);
    }

    #[test]
    fn word_token_lens_special_token_starts_a_new_word() {
        // A special first token forces a new word (openai `special` rule).
        // "hi", <special>, " yo"  ->  three words [1, 1, 1] (the special is
        // its own single-token word; " yo" starts a new one on its space).
        let t = tok(&[(0, b"hi"), (1, b""), (0, b" yo")]);
        assert_eq!(t.word_token_lens(&[0, 1, 2]).unwrap(), vec![1, 1, 1]);
    }

    #[test]
    fn word_token_lens_sums_to_token_count() {
        // The invariant words_from_alignment relies on: the per-word counts
        // partition all the input tokens.
        let t = tok(&[(0, b" the"), (0, b" quick"), (0, b"est"), (0, b"!")]);
        let lens = t.word_token_lens(&[0, 1, 2, 3]).unwrap();
        assert_eq!(lens.iter().sum::<usize>(), 4);
        // " the" | " quick"+"est" | "!"  ->  [1, 2, 1].
        assert_eq!(lens, vec![1, 2, 1]);
    }

    #[test]
    fn word_token_lens_empty_is_empty() {
        let t = tok(&[(0, b"a")]);
        assert_eq!(t.word_token_lens(&[]).unwrap(), Vec::<usize>::new());
    }
}
