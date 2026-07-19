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
//! - It does not implement **general text encoding** (arbitrary text → token
//!   ids): that would require the full tekken regex pre-tokenizer (Unicode
//!   property classes) on top of the BPE merge. What it DOES implement (P2
//!   cc-05/07 follow-up) is the narrow encode surface the runtime
//!   transcription prompt needs:
//!   - [`Self::token_id_of_special`] — exact-name lookup of a special token
//!     (the compact vocab stores special-token names verbatim, e.g.
//!     `[INST]`, `[BEGIN_AUDIO]`, `[TRANSCRIBE]`);
//!   - [`Self::encode_piece`] — tiktoken-style byte-pair encoding of ONE
//!     pre-split regex piece (whole-piece shortcut, then lowest-rank-first
//!     adjacent merges; the compact vocab's id order IS the tekken rank
//!     order, so ids double as merge ranks);
//!   - [`Self::transcription_prompt`] — the trained Voxtral
//!     transcription-request wrapper built from the two primitives above.

use std::collections::HashMap;
use std::sync::OnceLock;

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
    /// Lazily-built bytes → id map over the **non-special** entries, used by
    /// [`Self::encode_piece`]. On duplicate byte sequences the LOWEST id
    /// wins: compact-vocab id order is the tekken rank order, and BPE merge
    /// priority is by rank, so the lowest id is the token the reference
    /// encoder would produce. Built on first encode use (decode-only callers
    /// never pay for it).
    encode_map: OnceLock<HashMap<Vec<u8>, u32>>,
}

/// The runtime-constructed Voxtral transcription-request prompt (the trained
/// layout `VoxtralProcessor.apply_transcription_request` drives upstream):
///
/// ```text
/// [<s>] [INST] [BEGIN_AUDIO]  {audio soft-prefix rows}  [/INST] ("lang:xx")? [TRANSCRIBE]
/// └────────── pre_audio ────┘                          └────────── post_audio ─────────┘
/// ```
///
/// The audio run between the two segments is fed as **embedding rows** (the
/// adapter's soft prefix) via
/// [`TextDecoderSession::step_into_with_embed_prefix`](super::TextDecoderSession::step_into_with_embed_prefix)
/// — the `masked_scatter` semantics of upstream
/// `VoxtralForConditionalGeneration.forward` (audio embeds replace the
/// `[AUDIO]` placeholder positions in order), so the placeholder ids
/// themselves never enter the runtime session.
///
/// Built by [`VoxtralTokenizer::transcription_prompt`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TranscriptionPrompt {
    /// Token ids before the audio run: `[<s>, [INST], [BEGIN_AUDIO]]`.
    pub pre_audio: Vec<u32>,
    /// Token ids after the audio run: `[[/INST], ("lang:xx" pieces)?, [TRANSCRIBE]]`.
    pub post_audio: Vec<u32>,
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
        Ok(Self {
            raw,
            entries,
            eos,
            encode_map: OnceLock::new(),
        })
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

    // -----------------------------------------------------------------
    // Narrow encode surface (P2 cc-05/07 follow-up: runtime transcription
    // prompt). NOT a general text encoder — see the module doc.
    // -----------------------------------------------------------------

    /// Looks up a **special** token id by its literal name (e.g. `"[INST]"`,
    /// `"[BEGIN_AUDIO]"`, `"[TRANSCRIBE]"`, `"<s>"`). The tekken compact
    /// vocab stores special-token names verbatim with the special flag set,
    /// so no id is ever guessed — a vocab that does not carry the name is an
    /// explicit error (FR-EX-08), never a hard-coded fallback constant.
    ///
    /// Only entries flagged special are matched: a *text* token that happens
    /// to render as `"[INST]"` (user-typed bracket text) is NOT a control
    /// token and must not be returned here.
    ///
    /// # Errors
    ///
    /// - [`VokraError::NotImplemented`] if the tokenizer holds a raw
    ///   SentencePiece / tekken proto instead of a compact vocab;
    /// - [`VokraError::ModelLoad`] if the compact vocab has no special entry
    ///   named `name`.
    pub fn token_id_of_special(&self, name: &str) -> Result<u32> {
        if self.entries.is_empty() {
            return Err(VokraError::NotImplemented(
                "voxtral tokenizer: no compact-vocab table — special-token lookup needs the \
                 compact vocab embedded in the GGUF (see decode())",
            ));
        }
        let needle = name.as_bytes();
        self.entries
            .iter()
            .position(|e| e.special && e.bytes == needle)
            .map(|i| i as u32)
            .ok_or_else(|| {
                VokraError::ModelLoad(format!(
                    "voxtral tokenizer: special token `{name}` is not present in the embedded \
                     compact vocab ({} entries) — the GGUF's `{KEY_TOKENIZER_MODEL}` chunk does \
                     not carry the tekken special-token table this prompt layout needs. \
                     Re-embed the full compact vocab (special names included) at convert time.",
                    self.entries.len()
                ))
            })
    }

    /// The bytes → id encode map (non-special entries only), built once on
    /// first use. Lowest id wins on duplicates — see the field docstring.
    fn encode_map(&self) -> &HashMap<Vec<u8>, u32> {
        self.encode_map.get_or_init(|| {
            let mut m: HashMap<Vec<u8>, u32> = HashMap::with_capacity(self.entries.len());
            for (i, e) in self.entries.iter().enumerate() {
                if !e.special {
                    m.entry(e.bytes.clone()).or_insert(i as u32);
                }
            }
            m
        })
    }

    /// Encodes ONE pre-split regex piece to token ids with tiktoken-style
    /// byte-pair encoding:
    ///
    /// 1. whole-piece shortcut: if `piece` is a vocab entry, return its id;
    /// 2. otherwise start from the individual bytes and repeatedly merge the
    ///    adjacent pair whose concatenation has the LOWEST id (= lowest
    ///    tekken rank; leftmost wins ties) until no adjacent pair merges.
    ///
    /// This is exact for a single piece because a tiktoken-family vocab is
    /// its own merge table (a merge `(A, B) → AB` is legal iff `AB` is in
    /// the vocab, with priority = rank of `AB`) and the compact vocab
    /// preserves id order = rank order. What this function does NOT do is
    /// the tekken regex pre-tokenization — the caller must supply a piece
    /// the upstream regex would emit as one unit (see
    /// [`Self::transcription_prompt`] for the one place the runtime does
    /// that, with the split reasoning documented).
    ///
    /// # Errors
    ///
    /// - [`VokraError::NotImplemented`] without a compact vocab;
    /// - [`VokraError::InvalidArgument`] on an empty piece or a byte with no
    ///   single-byte vocab entry (the piece cannot be encoded — surfaced,
    ///   never skipped, FR-EX-08).
    pub fn encode_piece(&self, piece: &[u8]) -> Result<Vec<u32>> {
        if self.entries.is_empty() {
            return Err(VokraError::NotImplemented(
                "voxtral tokenizer: no compact-vocab table — encode_piece needs the compact \
                 vocab embedded in the GGUF (see decode())",
            ));
        }
        if piece.is_empty() {
            return Err(VokraError::InvalidArgument(
                "voxtral tokenizer: encode_piece called with an empty piece".into(),
            ));
        }
        let map = self.encode_map();
        // Whole-piece shortcut (tiktoken `_encode_ordinary` fast path).
        if let Some(&id) = map.get(piece) {
            return Ok(vec![id]);
        }
        // Byte-pair merge over `(start, end)` ranges into `piece`.
        let mut parts: Vec<(usize, usize)> = (0..piece.len()).map(|i| (i, i + 1)).collect();
        for &(s, e) in &parts {
            if !map.contains_key(&piece[s..e]) {
                return Err(VokraError::InvalidArgument(format!(
                    "voxtral tokenizer: byte 0x{:02X} has no single-byte vocab entry — the \
                     piece {:?} cannot be byte-pair encoded against this vocab",
                    piece[s],
                    String::from_utf8_lossy(piece),
                )));
            }
        }
        loop {
            // Find the adjacent pair whose merged bytes have the lowest id
            // (strict `<` keeps the leftmost on ties — tiktoken semantics).
            let mut best: Option<(usize, u32)> = None;
            for i in 0..parts.len().saturating_sub(1) {
                let cand = &piece[parts[i].0..parts[i + 1].1];
                if let Some(&id) = map.get(cand) {
                    if best.is_none_or(|(_, b)| id < b) {
                        best = Some((i, id));
                    }
                }
            }
            let Some((i, _)) = best else { break };
            parts[i].1 = parts[i + 1].1;
            parts.remove(i + 1);
        }
        // Invariant: every part's bytes are in `map` — the initial
        // single-byte parts were checked above, and a merge only happens
        // when the merged candidate itself was found in `map`. So this
        // index cannot panic.
        Ok(parts.iter().map(|&(s, e)| map[&piece[s..e]]).collect())
    }

    /// Builds the trained Voxtral **transcription-request prompt** at
    /// runtime from the embedded compact vocab — no offline
    /// `mistral_common` dump involved.
    ///
    /// Layout (mistral_common `InstructTokenizerV7.encode_transcription`,
    /// `tokens/tokenizers/instruct.py`):
    ///
    /// ```text
    /// [<s>] [INST] [BEGIN_AUDIO] {audio} [/INST] ("lang:{code}")? [TRANSCRIBE]
    /// ```
    ///
    /// with the `lang:{code}` segment present iff `language` is `Some`
    /// (upstream: `if request.language is not None: tokens +=
    /// tokenizer.encode(f"lang:{request.language}", bos=False, eos=False)`).
    ///
    /// # Language-code tokenization (honest scope)
    ///
    /// Upstream runs the full tekken regex over `"lang:{code}"`. For a code
    /// of pure lowercase ASCII letters the split is provably
    /// `["lang", ":{code}"]`: the pattern's first alternative
    /// `[^\r\n\p{L}\p{N}]?[\p{Lu}…]*[\p{Ll}…]+` matches `lang` (letters
    /// run), then at `:` matches the optional leading non-letter plus the
    /// lowercase run `:{code}` (leftmost-first alternation). Each piece is
    /// then byte-pair encoded — exactly what [`Self::encode_piece`] does.
    /// Codes containing anything but lowercase ASCII letters (digits,
    /// hyphens, uppercase — e.g. `zh-CN`) would split differently under the
    /// full regex, so they are rejected with an explicit error rather than
    /// encoded unfaithfully (FR-EX-08). ISO 639-1/-3 codes (`en`, `fr`,
    /// `ja`, `deu`, …) all fit the accepted alphabet.
    ///
    /// Verified against the offline `mistral_common` dump: `"en"` produces
    /// `post_audio = [[/INST], "lang", ":", "en", [TRANSCRIBE]]` =
    /// `[4, 9909, 1058, 1262, 34]` on the shipping tekken vocab (see the
    /// env-gated bit-check test `voxtral_transcription_prompt.rs`).
    ///
    /// # Errors
    ///
    /// - [`VokraError::NotImplemented`] without a compact vocab;
    /// - [`VokraError::ModelLoad`] if a required special token
    ///   (`<s>` / `[INST]` / `[BEGIN_AUDIO]` / `[/INST]` / `[TRANSCRIBE]`)
    ///   is absent from the vocab;
    /// - [`VokraError::InvalidArgument`] on a language code outside
    ///   `[a-z]{1,8}`.
    pub fn transcription_prompt(&self, language: Option<&str>) -> Result<TranscriptionPrompt> {
        let bos = self.token_id_of_special("<s>")?;
        let inst = self.token_id_of_special("[INST]")?;
        let begin_audio = self.token_id_of_special("[BEGIN_AUDIO]")?;
        let inst_end = self.token_id_of_special("[/INST]")?;
        let transcribe = self.token_id_of_special("[TRANSCRIBE]")?;

        let mut post_audio = vec![inst_end];
        if let Some(code) = language {
            if code.is_empty() || code.len() > 8 || !code.bytes().all(|b| b.is_ascii_lowercase()) {
                return Err(VokraError::InvalidArgument(format!(
                    "voxtral tokenizer: language code `{code}` is outside the runtime-encodable \
                     alphabet [a-z]{{1,8}}. The runtime mirrors the tekken regex split only for \
                     pure lowercase-ASCII codes (ISO 639-1/-3); other codes would need the full \
                     tekken pre-tokenizer to encode faithfully — pass a lowercase two/three-letter \
                     code, or None to omit the `lang:` segment (upstream semantics)."
                )));
            }
            post_audio.extend(self.encode_piece(b"lang")?);
            // ":{code}" is ONE regex piece (optional leading non-letter +
            // lowercase run) — see the docstring's split reasoning.
            let mut colon_code = Vec::with_capacity(1 + code.len());
            colon_code.push(b':');
            colon_code.extend_from_slice(code.as_bytes());
            post_audio.extend(self.encode_piece(&colon_code)?);
        }
        post_audio.push(transcribe);

        Ok(TranscriptionPrompt {
            pre_audio: vec![bos, inst, begin_audio],
            post_audio,
        })
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
        bad.extend_from_slice(b"ab"); // only 2 of 5 bytes
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

    // -----------------------------------------------------------------
    // Encode surface (P2 cc-05/07 follow-up): special-name lookup, piece
    // BPE, transcription prompt.
    // -----------------------------------------------------------------

    /// Builds a compact-vocab blob from `(special, bytes)` entries.
    fn blob_from(entries: &[(u8, &[u8])]) -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(&(entries.len() as u32).to_le_bytes());
        for (sp, bytes) in entries {
            v.push(*sp);
            v.extend_from_slice(&(bytes.len() as u16).to_le_bytes());
            v.extend_from_slice(bytes);
        }
        v
    }

    /// A tekken-shaped miniature vocab: specials at the REAL shipping ids
    /// (1 = `<s>`, 2 = `</s>`, 3 = `[INST]`, 4 = `[/INST]`, 24 = `[AUDIO]`,
    /// 25 = `[BEGIN_AUDIO]`, 34 = `[TRANSCRIBE]`), text tokens for the
    /// `lang:en` pieces plus the single bytes BPE needs. Filler entries are
    /// unique unmergeable text tokens.
    fn tekken_mini() -> VoxtralTokenizer {
        let mut entries: Vec<(u8, Vec<u8>)> = (0..64u32)
            .map(|i| (0u8, format!("<f{i}>").into_bytes()))
            .collect();
        let mut set = |id: usize, sp: u8, bytes: &[u8]| {
            entries[id] = (sp, bytes.to_vec());
        };
        set(0, 1, b"<unk>");
        set(1, 1, b"<s>");
        set(2, 1, b"</s>");
        set(3, 1, b"[INST]");
        set(4, 1, b"[/INST]");
        set(24, 1, b"[AUDIO]");
        set(25, 1, b"[BEGIN_AUDIO]");
        set(34, 1, b"[TRANSCRIBE]");
        set(40, 0, b"lang");
        set(41, 0, b":");
        set(42, 0, b"en");
        set(50, 0, b"e");
        set(51, 0, b"n");
        // Single bytes for the "fr" BPE-path test: no "fr" merged token, so
        // the encode must fall back to the byte pair.
        set(52, 0, b"f");
        set(53, 0, b"r");
        let flat: Vec<(u8, &[u8])> = entries.iter().map(|(sp, b)| (*sp, b.as_slice())).collect();
        VoxtralTokenizer::from_bytes(blob_from(&flat), 2).unwrap()
    }

    #[test]
    fn special_lookup_finds_shipping_names_at_their_ids() {
        let t = tekken_mini();
        assert_eq!(t.token_id_of_special("<s>").unwrap(), 1);
        assert_eq!(t.token_id_of_special("[INST]").unwrap(), 3);
        assert_eq!(t.token_id_of_special("[/INST]").unwrap(), 4);
        assert_eq!(t.token_id_of_special("[AUDIO]").unwrap(), 24);
        assert_eq!(t.token_id_of_special("[BEGIN_AUDIO]").unwrap(), 25);
        assert_eq!(t.token_id_of_special("[TRANSCRIBE]").unwrap(), 34);
    }

    #[test]
    fn special_lookup_missing_name_is_model_load_error() {
        let t = tekken_mini();
        let err = t.token_id_of_special("[NOT_A_TOKEN]").unwrap_err();
        assert!(matches!(err, VokraError::ModelLoad(_)), "{err:?}");
    }

    #[test]
    fn special_lookup_ignores_text_entries_with_special_looking_bytes() {
        // A TEXT token whose bytes read "[INST]" must not be returned by the
        // special lookup (user-typed bracket text is not a control token).
        let entries: &[(u8, &[u8])] = &[(0, b"[INST]"), (1, b"[INST]")];
        let t = VoxtralTokenizer::from_bytes(blob_from(entries), 0).unwrap();
        assert_eq!(t.token_id_of_special("[INST]").unwrap(), 1);
    }

    #[test]
    fn special_lookup_without_compact_vocab_is_not_implemented() {
        let blob = vec![0x0A, 0xFF, 0xFF, 0xFF]; // proto fingerprint
        let t = VoxtralTokenizer::from_bytes(blob, 2).unwrap();
        assert!(matches!(
            t.token_id_of_special("<s>"),
            Err(VokraError::NotImplemented(_))
        ));
    }

    #[test]
    fn encode_piece_whole_piece_shortcut() {
        let t = tekken_mini();
        assert_eq!(t.encode_piece(b"lang").unwrap(), vec![40]);
        assert_eq!(t.encode_piece(b":").unwrap(), vec![41]);
        assert_eq!(t.encode_piece(b"en").unwrap(), vec![42]);
    }

    #[test]
    fn encode_piece_bpe_merges_lowest_rank_first() {
        // ":en": no whole-piece entry, bytes [':', 'e', 'n']. Candidate
        // merges: ('e','n') → "en" (id 42). ':'+"en" → ":en" absent. Result
        // must be [":" = 41, "en" = 42] — the shipping tekken shape.
        let t = tekken_mini();
        assert_eq!(t.encode_piece(b":en").unwrap(), vec![41, 42]);
        // ":fr": "fr" has no merged entry → stays as single bytes.
        assert_eq!(t.encode_piece(b":fr").unwrap(), vec![41, 52, 53]);
    }

    #[test]
    fn encode_piece_merge_priority_is_by_id_leftmost_on_tie() {
        // Vocab: bytes a(id 3), b(id 4), c(id 5); merges "ab"(id 1),
        // "bc"(id 2), "abc" absent. For "abc": lowest-id merge is "ab"
        // (1 < 2) even though "bc" also matches → [ab, c] = [1, 5]. A
        // naive left-to-right greedy would also pick "ab" here, so pin the
        // priority with the mirrored vocab too: "ab"(id 2), "bc"(id 1) →
        // must produce [a, bc] = [3, 1].
        let v1: &[(u8, &[u8])] = &[
            (0, b"<pad>"),
            (0, b"ab"),
            (0, b"bc"),
            (0, b"a"),
            (0, b"b"),
            (0, b"c"),
        ];
        let t1 = VoxtralTokenizer::from_bytes(blob_from(v1), 0).unwrap();
        assert_eq!(t1.encode_piece(b"abc").unwrap(), vec![1, 5]);
        let v2: &[(u8, &[u8])] = &[
            (0, b"<pad>"),
            (0, b"bc"),
            (0, b"ab"),
            (0, b"a"),
            (0, b"b"),
            (0, b"c"),
        ];
        let t2 = VoxtralTokenizer::from_bytes(blob_from(v2), 0).unwrap();
        assert_eq!(t2.encode_piece(b"abc").unwrap(), vec![3, 1]);
    }

    #[test]
    fn encode_piece_missing_single_byte_is_invalid_argument() {
        let t = tekken_mini();
        // 'z' has no single-byte entry in the mini vocab and "zz" is not a
        // whole-piece entry.
        let err = t.encode_piece(b"zz").unwrap_err();
        assert!(matches!(err, VokraError::InvalidArgument(_)), "{err:?}");
    }

    #[test]
    fn encode_piece_empty_and_no_vocab_are_errors() {
        let t = tekken_mini();
        assert!(matches!(
            t.encode_piece(b""),
            Err(VokraError::InvalidArgument(_))
        ));
        let proto = VoxtralTokenizer::from_bytes(vec![0x0A, 0xFF, 0xFF, 0xFF], 2).unwrap();
        assert!(matches!(
            proto.encode_piece(b"lang"),
            Err(VokraError::NotImplemented(_))
        ));
    }

    #[test]
    fn encode_piece_duplicate_bytes_prefers_lowest_id() {
        // Two text entries with identical bytes: the encode map must keep
        // the lowest id (rank order = merge preference).
        let entries: &[(u8, &[u8])] = &[(0, b"x"), (0, b"dup"), (0, b"dup")];
        let t = VoxtralTokenizer::from_bytes(blob_from(entries), 0).unwrap();
        assert_eq!(t.encode_piece(b"dup").unwrap(), vec![1]);
    }

    #[test]
    fn transcription_prompt_matches_shipping_layout_for_en() {
        // On the tekken-shaped mini vocab the prompt must reproduce the
        // exact structure the offline mistral_common dump pinned:
        // pre = [<s>, [INST], [BEGIN_AUDIO]], post = [[/INST], "lang", ":",
        // "en", [TRANSCRIBE]]. (The REAL-vocab bit-check against the dump
        // ids lives in the env-gated voxtral_transcription_prompt.rs.)
        let t = tekken_mini();
        let p = t.transcription_prompt(Some("en")).unwrap();
        assert_eq!(p.pre_audio, vec![1, 3, 25]);
        assert_eq!(p.post_audio, vec![4, 40, 41, 42, 34]);
    }

    #[test]
    fn transcription_prompt_language_none_omits_lang_segment() {
        // Upstream: `if request.language is not None` — None skips the
        // whole "lang:{code}" run.
        let t = tekken_mini();
        let p = t.transcription_prompt(None).unwrap();
        assert_eq!(p.pre_audio, vec![1, 3, 25]);
        assert_eq!(p.post_audio, vec![4, 34]);
    }

    #[test]
    fn transcription_prompt_rejects_non_lowercase_ascii_codes() {
        let t = tekken_mini();
        for bad in ["EN", "zh-CN", "e n", "", "abcdefghi", "ja1"] {
            let err = t.transcription_prompt(Some(bad)).unwrap_err();
            assert!(
                matches!(err, VokraError::InvalidArgument(_)),
                "code {bad:?}: {err:?}"
            );
        }
    }

    #[test]
    fn transcription_prompt_missing_special_is_model_load_error() {
        // A vocab without [TRANSCRIBE] must fail loudly — the prompt cannot
        // be fabricated from partial specials.
        let entries: &[(u8, &[u8])] = &[
            (1, b"<unk>"),
            (1, b"<s>"),
            (1, b"</s>"),
            (1, b"[INST]"),
            (1, b"[/INST]"),
            (1, b"[BEGIN_AUDIO]"),
        ];
        let t = VoxtralTokenizer::from_bytes(blob_from(entries), 2).unwrap();
        let err = t.transcription_prompt(None).unwrap_err();
        assert!(matches!(err, VokraError::ModelLoad(_)), "{err:?}");
    }
}
