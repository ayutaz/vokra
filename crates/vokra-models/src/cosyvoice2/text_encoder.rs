//! CosyVoice2 text encoder + LLM backbone — stub (M3-09-T07 / T08).
//!
//! The real text encoder + LLM backbone (embedding lookup → transformer
//! blocks → hidden states consumed by the Flow Matching CFM) is implemented
//! against the upstream safetensors manifest in the follow-on session
//! (T07 embedding / positional / stem; T08 transformer blocks + GEMM hot
//! path). This scaffold intentionally lands the **type + trait surface**
//! only, so a caller who wires an engine against this module receives an
//! explicit [`VokraError::NotImplemented`] on any forward attempt rather
//! than a silent zero-fill fallback (FR-EX-08).
//!
//! # Numeric parity strategy (follow-on)
//!
//! The follow-on sessions will:
//!
//! 1. Read the upstream CosyVoice2 safetensors on a build machine (T02
//!    upstream inspection, still open — this scaffold does not invent
//!    tensor names) and record each `tensor_name → shape/dtype` in
//!    `docs/adr/M3-09-cosyvoice2.md` §T02.
//! 2. Bind those tensors verbatim through
//!    [`vokra_core::gguf::GgufFile::get_tensor`] in a new
//!    `weights::TensorStore` (mirrors `piper_plus::weights::TensorStore`).
//! 3. Route the GEMM hot path through [`crate::compute::Compute::gemm_f32`]
//!    so the Metal / CUDA seams (T19/T20) offload without a second
//!    kernel path.

use std::collections::HashMap;

use vokra_core::gguf::{GgufFile, GgufMetadataValue};
use vokra_core::{Result, VokraError};

use super::config::CosyVoice2Config;

/// Text encoder + LLM backbone — scaffold handle.
///
/// The struct itself holds no numeric state yet; the follow-on session adds
/// the tensor store + block cache. The public shape (`encode`) is stable
/// so callers can compile against the final surface today.
pub struct TextEncoderStub {
    /// Copy of the caller-provided config so shape validation can proceed
    /// without a live GGUF handle — the follow-on session replaces this
    /// with a `TensorStore` reference.
    #[allow(dead_code)] // consumed by T07/T08 numeric implementation
    config: CosyVoice2Config,
}

impl TextEncoderStub {
    /// Builds a stub bound to `config`. Never fails today; the follow-on
    /// implementation may return [`VokraError::InvalidArgument`] on a
    /// shape mismatch between `config` and the loaded weight tensors.
    #[must_use]
    pub fn new(config: CosyVoice2Config) -> Self {
        Self { config }
    }

    /// Encodes `token_ids` through the text embedding + LLM backbone and
    /// returns the per-token hidden features (`[t, hidden_dim]` row-major).
    ///
    /// # Errors
    ///
    /// This scaffold returns [`VokraError::NotImplemented`] unconditionally
    /// — the real forward path lands with T07/T08. The `token_ids` param
    /// is documented so callers can build the plumbing (tokenizer +
    /// batching) against the final signature today.
    pub fn encode(&self, token_ids: &[u32]) -> Result<Vec<f32>> {
        // Reference the arg so the intent is documented in-source.
        let _ = token_ids.len();
        Err(VokraError::NotImplemented(
            "CosyVoice2 text encoder + LLM backbone forward is not implemented in this \
             scaffold; T07 embedding / T08 transformer blocks / T09 unit test land the \
             numeric path against the upstream safetensors manifest",
        ))
    }
}

// ===========================================================================
// CosyVoice2 text tokenizer — Qwen2 byte-level BPE (M3-09-T06)
// ===========================================================================
//
// CosyVoice2-0.5B's LLM backbone is a Qwen2, whose tokenizer is a GPT-2-family
// **byte-level BPE**: each input byte is first mapped to a printable Unicode
// "byte-char" (`bytes_to_unicode`), the byte-char stream is split into pieces
// by a pretokenizer, and every piece is byte-pair-merged in `merges.txt` rank
// order into vocabulary tokens (`vocab.json`). Decoding is the exact inverse.
//
// The upstream `vocab.json` + `merges.txt` are embedded verbatim in the GGUF
// as raw U8 arrays (the M2-06 Whisper / M3-10 Voxtral / M4-05 CSM zero-dep
// embed pattern) under `vokra.cosyvoice2.tokenizer.vocab` / `.merges`, so the
// runtime is self-contained — no ONNX, no `tokenizers` crate (NFR-DS-02).
//
// # What is exact vs. owner-verified (honest scope)
//
// - The **byte↔unicode map** is the canonical GPT-2 bijection, verified
//   against the shipping Qwen2 `vocab.json` (byte 0x20 → 'Ġ' id 220, byte
//   0x0A → 'Ċ' id 198, and all 256 byte-chars present).
// - The **merge-rank BPE** is the canonical GPT-2 algorithm (lowest-rank
//   adjacent pair first, merge all its occurrences, repeat) — exact for a
//   given (vocab, merges) pair; covered by hand-derived id tests.
// - **`decode(encode(s)) == s`** round-trips for any UTF-8 input (every byte
//   has a single-byte vocab entry, so encode never drops a byte).
// - The **pretokenizer** implements the Qwen2 regex alternation for the
//   practical subset (case-insensitive contractions, `\p{L}` runs with the
//   optional leading char, per-digit `\p{N}`, punctuation runs, the
//   `\s+(?!\S)` trailing-space rule, newline runs). It uses
//   `char::is_alphabetic` / `is_numeric` / `is_whitespace` for `\p{L}` /
//   `\p{N}` / `\s`, which agree for ASCII and common scripts but not every
//   Unicode category. **Byte-exact id parity vs. the HuggingFace Qwen2
//   tokenizer over arbitrary Unicode is therefore owner-verified** (no HF
//   import is available here — the same honest boundary
//   `voxtral/tokenizer.rs` draws). This module guarantees the round-trip and
//   the hand-derived ids, not a blanket HF-parity claim.

/// GGUF key: the raw upstream Qwen2 `vocab.json` bytes (U8 array).
pub const KEY_TOKENIZER_VOCAB: &str = "vokra.cosyvoice2.tokenizer.vocab";
/// GGUF key: the raw upstream Qwen2 `merges.txt` bytes (U8 array).
pub const KEY_TOKENIZER_MERGES: &str = "vokra.cosyvoice2.tokenizer.merges";

/// A loaded CosyVoice2 (Qwen2) byte-level BPE tokenizer.
#[derive(Debug, Clone)]
pub struct CosyVoice2Tokenizer {
    /// token unicode-string → id (`vocab.json`).
    encoder: HashMap<String, u32>,
    /// id → token unicode-string (decode direction; ids may be sparse).
    decoder: HashMap<u32, String>,
    /// merge pair `(left, right)` → rank (0 = highest priority; `merges.txt`
    /// line order).
    bpe_ranks: HashMap<(String, String), usize>,
    /// byte value → byte-char (GPT-2 `bytes_to_unicode`).
    byte_encoder: [char; 256],
    /// byte-char → byte value (the inverse of `byte_encoder`).
    byte_decoder: HashMap<char, u8>,
}

impl CosyVoice2Tokenizer {
    /// Loads the tokenizer from the two U8 GGUF chunks
    /// ([`KEY_TOKENIZER_VOCAB`] + [`KEY_TOKENIZER_MERGES`]).
    ///
    /// # Errors
    ///
    /// [`VokraError::ModelLoad`] when either chunk is absent, is not a U8
    /// array, or the payload does not parse as `vocab.json` / `merges.txt`.
    pub fn from_gguf(file: &GgufFile) -> Result<Self> {
        let vocab = read_u8_array(file, KEY_TOKENIZER_VOCAB)?;
        let merges = read_u8_array(file, KEY_TOKENIZER_MERGES)?;
        Self::from_parts(&vocab, &merges)
    }

    /// Builds the tokenizer from the raw `vocab.json` + `merges.txt` bytes.
    ///
    /// # Errors
    ///
    /// [`VokraError::ModelLoad`] on malformed vocab / merges — never a silent
    /// partial table (FR-EX-08).
    pub fn from_parts(vocab_json: &[u8], merges_txt: &[u8]) -> Result<Self> {
        let root = vokra_core::json::parse(vocab_json).map_err(|e| {
            VokraError::ModelLoad(format!(
                "cosyvoice2 tokenizer: vocab.json is not valid JSON: {e}"
            ))
        })?;
        let obj = root.as_object().ok_or_else(|| {
            VokraError::ModelLoad("cosyvoice2 tokenizer: vocab.json is not a JSON object".into())
        })?;
        let mut encoder = HashMap::with_capacity(obj.len());
        let mut decoder = HashMap::with_capacity(obj.len());
        for (tok, v) in obj {
            let id = v
                .as_u64()
                .and_then(|n| u32::try_from(n).ok())
                .ok_or_else(|| {
                    VokraError::ModelLoad(format!(
                        "cosyvoice2 tokenizer: vocab.json id for {tok:?} is not a u32"
                    ))
                })?;
            encoder.insert(tok.clone(), id);
            decoder.insert(id, tok.clone());
        }
        if encoder.is_empty() {
            return Err(VokraError::ModelLoad(
                "cosyvoice2 tokenizer: vocab.json is empty".into(),
            ));
        }
        let merges_text = std::str::from_utf8(merges_txt).map_err(|e| {
            VokraError::ModelLoad(format!(
                "cosyvoice2 tokenizer: merges.txt is not UTF-8: {e}"
            ))
        })?;
        let mut bpe_ranks = HashMap::new();
        let mut rank = 0usize;
        for line in merges_text.lines() {
            // A blank line or a leading `#version:` header carries no merge
            // (HF sometimes writes the latter; the shipping Qwen2 file does
            // not). Skip without consuming a rank so the rank == line order.
            if line.is_empty() || line.starts_with("#version") {
                continue;
            }
            let mut it = line.splitn(2, ' ');
            match (it.next(), it.next()) {
                (Some(a), Some(b)) if !a.is_empty() && !b.is_empty() => {
                    bpe_ranks.insert((a.to_owned(), b.to_owned()), rank);
                    rank += 1;
                }
                _ => {
                    return Err(VokraError::ModelLoad(format!(
                        "cosyvoice2 tokenizer: merges.txt entry {rank} is not a space-separated \
                         `LEFT RIGHT` pair: {line:?}"
                    )));
                }
            }
        }
        let byte_encoder = build_byte_encoder();
        let byte_decoder = byte_encoder
            .iter()
            .enumerate()
            .map(|(b, &c)| (c, u8::try_from(b).expect("byte index < 256")))
            .collect();
        Ok(Self {
            encoder,
            decoder,
            bpe_ranks,
            byte_encoder,
            byte_decoder,
        })
    }

    /// Number of base BPE tokens in `vocab.json`.
    ///
    /// This excludes the special / added tokens (`<|endoftext|>`,
    /// `<|im_start|>`, `<|im_end|>`), which live in `tokenizer_config.json`,
    /// not `vocab.json`, and are added by the chat / prompt layer (an LLM
    /// forward concern, T07/T08), not by this base tokenizer.
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.encoder.len()
    }

    /// Encodes `text` to Qwen2 byte-level BPE ids.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] if a byte has no single-byte vocab
    /// entry, or a merged token is absent from `vocab.json` (vocab/merges
    /// inconsistent) — surfaced, never silently dropped (FR-EX-08).
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let mut ids = Vec::new();
        for piece in pre_tokenize(text) {
            let word: Vec<String> = piece
                .as_bytes()
                .iter()
                .map(|&b| self.byte_encoder[b as usize].to_string())
                .collect();
            for tok in self.bpe(word) {
                match self.encoder.get(&tok) {
                    Some(&id) => ids.push(id),
                    None => {
                        return Err(VokraError::InvalidArgument(format!(
                            "cosyvoice2 tokenizer: BPE produced token {tok:?} absent from \
                             vocab.json — vocab/merges are inconsistent (FR-EX-08: not dropped)"
                        )));
                    }
                }
            }
        }
        Ok(ids)
    }

    /// Decodes byte-level BPE ids back to text — the exact inverse of
    /// [`encode`](Self::encode) for in-vocabulary ids (lossy UTF-8 on
    /// deliberately malformed byte sequences, matching HF `errors="replace"`).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on an id with no `vocab.json` entry
    /// (a special / added-token id, or out of range).
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        let mut bytes = Vec::new();
        for &id in ids {
            let tok = self.decoder.get(&id).ok_or_else(|| {
                VokraError::InvalidArgument(format!(
                    "cosyvoice2 tokenizer: id {id} has no vocab.json entry (special / added \
                     tokens live in tokenizer_config.json, not the base BPE vocab)"
                ))
            })?;
            for ch in tok.chars() {
                let b = self.byte_decoder.get(&ch).ok_or_else(|| {
                    VokraError::ModelLoad(format!(
                        "cosyvoice2 tokenizer: vocab token {tok:?} contains char {ch:?} outside \
                         the byte-level alphabet — not a GPT-2 byte-level BPE vocab"
                    ))
                })?;
                bytes.push(*b);
            }
        }
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    /// Canonical GPT-2 byte-pair merge over one pretokenized piece:
    /// repeatedly merge the adjacent pair with the lowest `merges.txt` rank
    /// (merging **all** of that pair's occurrences left-to-right) until no
    /// adjacent pair has a rank.
    fn bpe(&self, mut word: Vec<String>) -> Vec<String> {
        if word.len() < 2 {
            return word;
        }
        loop {
            // Lowest-rank adjacent pair. Ranks are unique per pair, so the
            // strict `<` (keep earliest on the impossible tie) is exact.
            let mut best: Option<(usize, usize)> = None; // (rank, index)
            for i in 0..word.len() - 1 {
                if let Some(&rank) = self.bpe_ranks.get(&(word[i].clone(), word[i + 1].clone())) {
                    if best.is_none_or(|(r, _)| rank < r) {
                        best = Some((rank, i));
                    }
                }
            }
            let Some((_, idx)) = best else { break };
            let first = word[idx].clone();
            let second = word[idx + 1].clone();
            let mut merged = Vec::with_capacity(word.len());
            let mut i = 0;
            while i < word.len() {
                if i + 1 < word.len() && word[i] == first && word[i + 1] == second {
                    merged.push(format!("{first}{second}"));
                    i += 2;
                } else {
                    merged.push(word[i].clone());
                    i += 1;
                }
            }
            word = merged;
            if word.len() == 1 {
                break;
            }
        }
        word
    }
}

/// Reads a `U8` GGUF metadata array into a byte buffer (mirrors the CSM /
/// Voxtral tokenizer readers). An absent key or a non-U8 element is a loud
/// [`VokraError::ModelLoad`] (FR-EX-08).
fn read_u8_array(file: &GgufFile, key: &str) -> Result<Vec<u8>> {
    match file.get(key) {
        Some(GgufMetadataValue::Array(arr)) => {
            let mut bytes = Vec::with_capacity(arr.values.len());
            for v in &arr.values {
                match v {
                    GgufMetadataValue::U8(x) => bytes.push(*x),
                    other => {
                        return Err(VokraError::ModelLoad(format!(
                            "cosyvoice2 tokenizer: `{key}` carries a non-U8 element ({:?})",
                            other.value_type()
                        )));
                    }
                }
            }
            Ok(bytes)
        }
        Some(other) => Err(VokraError::ModelLoad(format!(
            "cosyvoice2 tokenizer: `{key}` is not a U8 array (got {:?})",
            other.value_type()
        ))),
        None => Err(VokraError::ModelLoad(format!(
            "cosyvoice2 tokenizer: `{key}` missing — re-convert with `vokra-cli convert \
             --model cosyvoice2 --config <CosyVoice-BlankEN/config.json>` so the Qwen2 \
             vocab.json + merges.txt are embedded"
        ))),
    }
}

/// The GPT-2 `bytes_to_unicode` bijection: 256 byte values → printable
/// "byte-chars". The printable Latin-1 ranges map to themselves; every other
/// byte maps to U+0100.. in ascending byte order. Transcribed from
/// openai/gpt-2 `encoder.py` (CLAUDE.md ハルシネーション厳禁) and verified
/// against the shipping Qwen2 `vocab.json` (0x20 → 'Ġ', 0x0A → 'Ċ').
fn build_byte_encoder() -> [char; 256] {
    let mut is_direct = [false; 256];
    for b in b'!'..=b'~' {
        is_direct[b as usize] = true;
    }
    for b in 0xA1u8..=0xAC {
        is_direct[b as usize] = true;
    }
    for b in 0xAEu8..=0xFF {
        is_direct[b as usize] = true;
    }
    let mut enc = ['\0'; 256];
    let mut n: u32 = 0;
    for b in 0u32..256 {
        if is_direct[b as usize] {
            enc[b as usize] = char::from_u32(b).expect("Latin-1 scalar is valid");
        } else {
            enc[b as usize] = char::from_u32(256 + n).expect("U+0100.. scalar is valid");
            n += 1;
        }
    }
    enc
}

/// Splits `text` into Qwen2-pretokenizer pieces (ordered regex alternation,
/// first alternative that matches wins, greedy within each). See the module
/// header for the honest scope: `\p{L}` / `\p{N}` / `\s` are approximated by
/// `char::is_alphabetic` / `is_numeric` / `is_whitespace`.
fn pre_tokenize(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut pieces = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        // Every non-empty position matches at least `\s+` or one class; the
        // `.max(i + 1)` is a defensive progress guarantee that keeps `encode`
        // total and round-trip-safe on any unforeseen scalar.
        let end = match_alternation(&chars, i).max(i + 1);
        pieces.push(chars[i..end].iter().collect());
        i = end;
    }
    pieces
}

/// Tries the pretokenizer alternatives in order and returns the end index of
/// the first that matches at `i` (or `i` if none does — see [`pre_tokenize`]).
fn match_alternation(chars: &[char], i: usize) -> usize {
    m_contraction(chars, i)
        .or_else(|| m_letters(chars, i))
        .or_else(|| m_number(chars, i))
        .or_else(|| m_others(chars, i))
        .or_else(|| m_newlines(chars, i))
        .or_else(|| m_ws_trailing(chars, i))
        .or_else(|| m_ws(chars, i))
        .unwrap_or(i)
}

fn tok_is_l(c: char) -> bool {
    c.is_alphabetic()
}
fn tok_is_n(c: char) -> bool {
    c.is_numeric()
}
fn tok_is_ws(c: char) -> bool {
    c.is_whitespace()
}
fn tok_is_nl(c: char) -> bool {
    c == '\r' || c == '\n'
}
fn tok_is_other(c: char) -> bool {
    !tok_is_ws(c) && !tok_is_l(c) && !tok_is_n(c)
}

/// `(?i:'s|'t|'re|'ve|'m|'ll|'d)`
fn m_contraction(chars: &[char], i: usize) -> Option<usize> {
    if chars[i] != '\'' {
        return None;
    }
    // None of these suffixes is a prefix of another, so the order is
    // immaterial; the ASCII case-insensitive compare is exactly the Qwen2
    // `(?i:...)` for these letters.
    const SUFFIXES: [&str; 7] = ["s", "t", "re", "ve", "m", "ll", "d"];
    let rest = &chars[i + 1..];
    for suf in SUFFIXES {
        let sc: Vec<char> = suf.chars().collect();
        if rest.len() >= sc.len()
            && rest[..sc.len()]
                .iter()
                .zip(&sc)
                .all(|(a, b)| a.eq_ignore_ascii_case(b))
        {
            return Some(i + 1 + sc.len());
        }
    }
    None
}

/// `[^\r\n\p{L}\p{N}]?\p{L}+` — an optional single leading non-letter /
/// non-digit / non-newline char (only if a letter follows), then a letter run.
fn m_letters(chars: &[char], i: usize) -> Option<usize> {
    let n = chars.len();
    let mut j = i;
    if !tok_is_nl(chars[j]) && !tok_is_l(chars[j]) && !tok_is_n(chars[j]) {
        // Optional leading char: consume it only if a letter follows (else
        // this alternative fails and control passes to `m_number` / etc.).
        if j + 1 < n && tok_is_l(chars[j + 1]) {
            j += 1;
        } else {
            return None;
        }
    }
    if j >= n || !tok_is_l(chars[j]) {
        return None;
    }
    while j < n && tok_is_l(chars[j]) {
        j += 1;
    }
    Some(j)
}

/// `\p{N}` — Qwen2 emits one digit per token.
fn m_number(chars: &[char], i: usize) -> Option<usize> {
    if tok_is_n(chars[i]) {
        Some(i + 1)
    } else {
        None
    }
}

/// ` ?[^\s\p{L}\p{N}]+[\r\n]*` — an optional single leading space, a run of
/// non-whitespace / non-letter / non-digit chars, then trailing newlines.
fn m_others(chars: &[char], i: usize) -> Option<usize> {
    let n = chars.len();
    let mut j = i;
    if chars[j] == ' ' && j + 1 < n && tok_is_other(chars[j + 1]) {
        j += 1;
    }
    if j >= n || !tok_is_other(chars[j]) {
        return None;
    }
    while j < n && tok_is_other(chars[j]) {
        j += 1;
    }
    while j < n && tok_is_nl(chars[j]) {
        j += 1;
    }
    Some(j)
}

/// `\s*[\r\n]+` — a whitespace run up to and including its last newline.
fn m_newlines(chars: &[char], i: usize) -> Option<usize> {
    if !tok_is_ws(chars[i]) {
        return None;
    }
    let n = chars.len();
    let mut e = i;
    while e < n && tok_is_ws(chars[e]) {
        e += 1;
    }
    let mut last_nl = None;
    for (k, &c) in chars.iter().enumerate().take(e).skip(i) {
        if tok_is_nl(c) {
            last_nl = Some(k);
        }
    }
    last_nl.map(|k| k + 1)
}

/// `\s+(?!\S)` — a whitespace run, giving back its last char if a non-ws
/// char follows (so that char can attach to the next token).
fn m_ws_trailing(chars: &[char], i: usize) -> Option<usize> {
    if !tok_is_ws(chars[i]) {
        return None;
    }
    let n = chars.len();
    let mut e = i;
    while e < n && tok_is_ws(chars[e]) {
        e += 1;
    }
    if e == n {
        Some(e) // reaches end-of-string → matches the whole run
    } else if e - i >= 2 {
        Some(e - 1) // followed by non-ws → give back the last ws char
    } else {
        None // single ws followed by non-ws → falls through to `m_ws`
    }
}

/// `\s+` — the remaining whitespace fallthrough.
fn m_ws(chars: &[char], i: usize) -> Option<usize> {
    if !tok_is_ws(chars[i]) {
        return None;
    }
    let n = chars.len();
    let mut e = i;
    while e < n && tok_is_ws(chars[e]) {
        e += 1;
    }
    Some(e)
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::gguf::chunks::KEY_MODEL_ARCH;
    use vokra_core::gguf::{GgufBuilder, GgufFile};

    fn stub_config() -> CosyVoice2Config {
        let mut b = GgufBuilder::new();
        b.add_string(KEY_MODEL_ARCH, "cosyvoice2");
        b.add_u32(super::super::config::KEY_SAMPLE_RATE, 24_000);
        b.add_u32(super::super::config::KEY_VOCAB_SIZE, 32);
        b.add_u32(super::super::config::KEY_HIDDEN_DIM, 16);
        b.add_u32(super::super::config::KEY_N_LAYER, 2);
        b.add_u32(super::super::config::KEY_N_HEAD, 2);
        b.add_u32(super::super::config::KEY_FFN_DIM, 32);
        b.add_u32(super::super::config::KEY_FLOW_NFE, 4);
        b.add_string(super::super::config::KEY_FLOW_SCHEDULE, "linear");
        b.add_u32(super::super::config::KEY_MIMI_N_CODEBOOKS, 4);
        b.add_u32(super::super::config::KEY_MIMI_CODEBOOK_SIZE, 16);
        b.add_u32(super::super::config::KEY_MIMI_D_MODEL, 8);
        b.add_u32(super::super::config::KEY_STREAMING_CHUNK_SIZE, 4);
        b.add_u32(super::super::config::KEY_STREAMING_CHUNK_HOP, 4);
        let bytes = b.to_bytes().expect("serialize");
        let file = GgufFile::parse(bytes).expect("parse");
        CosyVoice2Config::from_gguf(&file).expect("read")
    }

    #[test]
    fn encode_returns_not_implemented_never_silent() {
        // No silent zero-fill fallback (FR-EX-08). The stub returns an
        // explicit NotImplemented on any call, so a caller who wires
        // against this scaffold today learns immediately that the
        // numeric path is not yet available.
        let enc = TextEncoderStub::new(stub_config());
        let err = enc
            .encode(&[1, 2, 3])
            .expect_err("scaffold must not produce features");
        assert!(matches!(err, VokraError::NotImplemented(_)));
    }

    #[test]
    fn encode_stub_accepts_empty_token_sequence() {
        // Degenerate but well-defined: an empty token sequence must still
        // produce the same NotImplemented error today (never an empty
        // Vec that could be misread as "encoded successfully").
        let enc = TextEncoderStub::new(stub_config());
        let err = enc
            .encode(&[])
            .expect_err("scaffold must not produce features");
        assert!(matches!(err, VokraError::NotImplemented(_)));
    }
}

#[cfg(test)]
mod tokenizer_tests {
    use super::{
        CosyVoice2Tokenizer, build_byte_encoder, m_letters, m_number, m_others, m_ws_trailing,
        pre_tokenize,
    };
    use vokra_core::VokraError;

    /// A tiny hand-controlled byte-level vocab + merge table, so the exact
    /// output ids are derivable by executing the canonical GPT-2 BPE by hand.
    ///
    /// vocab: a=0 b=1 c=2 Ġ=3 ab=4 abc=5 Ġa=6 (Ġ is byte 0x20's byte-char).
    /// merges (rank order): (a,b) (ab,c) (Ġ,a).
    fn tiny() -> CosyVoice2Tokenizer {
        let vocab = r#"{"a":0,"b":1,"c":2,"Ġ":3,"ab":4,"abc":5,"Ġa":6}"#;
        let merges = "a b\nab c\nĠ a\n";
        CosyVoice2Tokenizer::from_parts(vocab.as_bytes(), merges.as_bytes())
            .expect("tiny tokenizer")
    }

    #[test]
    fn byte_encoder_matches_gpt2_reference() {
        let enc = build_byte_encoder();
        // Spot-checks verified against the shipping Qwen2 vocab.json.
        assert_eq!(enc[0x20], 'Ġ'); // space
        assert_eq!(enc[0x0A], 'Ċ'); // newline
        assert_eq!(enc[b'!' as usize], '!');
        assert_eq!(enc[b'a' as usize], 'a');
        assert_eq!(enc[0xC3], 'Ã'); // Latin-1 direct range
        // Bijection: all 256 byte-chars are distinct.
        let set: std::collections::HashSet<char> = enc.iter().copied().collect();
        assert_eq!(set.len(), 256, "byte→char map must be injective");
    }

    #[test]
    fn synthetic_exact_ids_hand_derived() {
        let t = tiny();
        // "abc": [a,b,c] --(a,b)r0--> [ab,c] --(ab,c)r1--> [abc] = id 5.
        assert_eq!(t.encode("abc").unwrap(), vec![5]);
        // "ab": [a,b] --(a,b)r0--> [ab] = id 4.
        assert_eq!(t.encode("ab").unwrap(), vec![4]);
        // "cab": [c,a,b] --(a,b)r0--> [c,ab]; no (c,ab) merge = ids [2,4].
        assert_eq!(t.encode("cab").unwrap(), vec![2, 4]);
        // " a": leading space attaches → piece " " + "a" → [Ġ,a] --(Ġ,a)r2--> [Ġa] = id 6.
        assert_eq!(t.encode(" a").unwrap(), vec![6]);
    }

    #[test]
    fn roundtrip_on_tiny_vocab() {
        let t = tiny();
        for s in ["abc", "ab", "cab", " a"] {
            assert_eq!(
                t.decode(&t.encode(s).unwrap()).unwrap(),
                s,
                "round-trip {s:?}"
            );
        }
    }

    #[test]
    fn roundtrip_arbitrary_utf8_full_byte_vocab() {
        // A full 256-byte-char vocab (every byte encodable) plus one merge so
        // the merge+decode path is exercised. Round-trip must be the identity
        // for any UTF-8 input.
        let (vocab, merges) = full_byte_vocab(&[("a", "b")]);
        let t = CosyVoice2Tokenizer::from_parts(&vocab, &merges).expect("full-byte tokenizer");
        for s in [
            "",
            "a",
            "ab",
            "hello world",
            "don't stop",
            "café 世界!",
            "line1\nline2",
            "  leading and trailing  ",
            "123 abc 45",
            "🎵♪ mixed",
        ] {
            let ids = t.encode(s).unwrap_or_else(|e| panic!("encode {s:?}: {e}"));
            assert_eq!(
                t.decode(&ids).unwrap(),
                s,
                "round-trip failed for {s:?} -> {ids:?}"
            );
        }
        // The (a,b) merge actually fires: "ab" collapses to one id (256).
        assert_eq!(t.encode("ab").unwrap(), vec![256]);
    }

    #[test]
    fn pretokenizer_splits_match_qwen2_common_cases() {
        // Space attaches to the following word (byte-level BPE convention).
        assert_eq!(pre_tokenize("hello world"), vec!["hello", " world"]);
        // Qwen2 splits digits one per token.
        assert_eq!(pre_tokenize("abc123"), vec!["abc", "1", "2", "3"]);
        // Contraction is its own piece.
        assert_eq!(pre_tokenize("don't"), vec!["don", "'t"]);
        // N spaces before a word → (N-1) as one piece + 1 attached.
        assert_eq!(pre_tokenize("a  b"), vec!["a", " ", " b"]);
        // A leading non-letter attaches to the word ([^\r\n\p{L}\p{N}]? rule).
        assert_eq!(pre_tokenize("(hi)"), vec!["(hi", ")"]);
        // Trailing whitespace to end-of-string is one piece.
        assert_eq!(pre_tokenize("a  "), vec!["a", "  "]);
        // Consecutive newlines are one piece (\s*[\r\n]+ rule).
        assert_eq!(pre_tokenize("a\n\nb"), vec!["a", "\n\n", "b"]);
    }

    #[test]
    fn pretokenizer_partitions_every_byte_char() {
        // Whatever the alternatives do, the pieces must concatenate back to
        // the input exactly (this is what makes the round-trip total).
        for s in ["", "a", "  ", "\n\t\r", "a1!b 2\n", "'CAPS", "!!! ???"] {
            let joined: String = pre_tokenize(s).concat();
            assert_eq!(joined, s, "pretokenize must partition {s:?} losslessly");
        }
    }

    #[test]
    fn decode_out_of_range_id_is_loud() {
        let t = tiny();
        let err = t.decode(&[999]).expect_err("out-of-range id must error");
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn encode_byte_without_vocab_entry_is_loud() {
        // The tiny vocab has no entry for 'd' (byte-char 'd'); encoding it
        // must fail loudly, never silently drop the byte (FR-EX-08).
        let t = tiny();
        let err = t.encode("d").expect_err("missing byte-char must error");
        assert!(matches!(err, VokraError::InvalidArgument(_)));
    }

    #[test]
    fn merges_version_header_is_skipped() {
        let vocab = r#"{"a":0,"b":1,"ab":2}"#;
        let merges = "#version: 0.2\na b\n";
        let t = CosyVoice2Tokenizer::from_parts(vocab.as_bytes(), merges.as_bytes())
            .expect("header must be skipped, not fatal");
        // The (a,b) merge is rank 0 despite the header line preceding it.
        assert_eq!(t.encode("ab").unwrap(), vec![2]);
    }

    #[test]
    fn empty_vocab_is_loud() {
        let err = CosyVoice2Tokenizer::from_parts(b"{}", b"").expect_err("empty vocab must error");
        assert!(matches!(err, VokraError::ModelLoad(_)));
    }

    // Individual pretokenizer-alternative sanity (guards against silent
    // drift when the alternatives are edited).
    #[test]
    fn pretokenizer_alternative_units() {
        let cs: Vec<char> = " a1! ".chars().collect();
        assert_eq!(m_letters(&cs, 0), Some(2)); // " a" (space + letter)
        assert_eq!(m_number(&cs, 2), Some(3)); // "1"
        assert_eq!(m_others(&cs, 3), Some(4)); // "!"
        let ws: Vec<char> = "  x".chars().collect();
        assert_eq!(m_ws_trailing(&ws, 0), Some(1)); // give back the last space
    }

    /// Builds a `(vocab.json, merges.txt)` pair covering all 256 byte-chars
    /// (ids 0..256) plus the supplied merges (ids 256..). JSON string keys are
    /// escaped for `"` and `\` only (the byte-chars needing it).
    fn full_byte_vocab(merges: &[(&str, &str)]) -> (Vec<u8>, Vec<u8>) {
        let enc = build_byte_encoder();
        let mut json = String::from("{");
        for (id, &c) in enc.iter().enumerate() {
            if id > 0 {
                json.push(',');
            }
            push_json_key(&mut json, &c.to_string());
            json.push(':');
            json.push_str(&id.to_string());
        }
        let mut merges_txt = String::new();
        for (next, (a, b)) in (256u32..).zip(merges.iter()) {
            json.push(',');
            push_json_key(&mut json, &format!("{a}{b}"));
            json.push(':');
            json.push_str(&next.to_string());
            merges_txt.push_str(a);
            merges_txt.push(' ');
            merges_txt.push_str(b);
            merges_txt.push('\n');
        }
        json.push('}');
        (json.into_bytes(), merges_txt.into_bytes())
    }

    fn push_json_key(out: &mut String, key: &str) {
        out.push('"');
        for ch in key.chars() {
            if ch == '"' || ch == '\\' {
                out.push('\\');
            }
            out.push(ch);
        }
        out.push('"');
    }
}
