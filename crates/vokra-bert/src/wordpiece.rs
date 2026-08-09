//! WordPiece tokenizer — clean-room per Devlin et al. 2018
//! ([arXiv:1810.04805](https://arxiv.org/abs/1810.04805)) and the original
//! WordPiece algorithm ([Wu et al. 2016 §4.1](https://arxiv.org/abs/1609.08144)).
//!
//! # Scope
//!
//! Aimed at BERT-family models that use a WordPiece vocabulary — specifically
//! `hfl/chinese-roberta-wwm-ext-large` (Apache-2.0, `model_type: bert`,
//! `vocab_size: 21128`, `hidden_size: 1024`, 24 layers, 16 heads) which is
//! wired into SBV2's `language_id = 2` (ZH) BERT slot. Reused by any future
//! `BertForMaskedLM`-style model that ships a WordPiece vocab and standard
//! `[PAD]/[UNK]/[CLS]/[SEP]` layout.
//!
//! **Does not** replace [`crate::tokenizer::SbertTokenizer`], which is a
//! SentencePiece Viterbi tokenizer for a different family of models.
//!
//! # Algorithm summary
//!
//! Two stages, per the BERT paper:
//!
//! 1. **`BasicTokenizer`** — clean control chars, split every CJK codepoint
//!    into its own whitespace-delimited token (`tokenize_chinese_chars`),
//!    whitespace-split, optionally lowercase (`do_lower_case`), split each
//!    resulting token on ASCII punctuation.
//! 2. **`WordPiece`** — for each token, greedy longest-prefix match against
//!    the vocab. The first prefix is looked up as-is; every subsequent prefix
//!    is prefixed with `"##"` (the WordPiece continuation marker). A token
//!    that cannot be segmented becomes a single `[UNK]` (or an error under
//!    [`OovPolicy::Error`]).
//!
//! # Deliberate simplifications (documented gaps)
//!
//! - **No Unicode NFD normalization / accent stripping.** HuggingFace
//!   `BertTokenizer` performs NFD followed by removal of Mn (nonspacing mark)
//!   codepoints when `do_lower_case=True`, so `café` decomposes to `cafe`.
//!   This crate has no `unicode-normalization` dependency (zero-dep
//!   NFR-DS-02), so this pass is skipped. For `hfl/chinese-roberta-wwm-ext-large`
//!   whose vocab is dominated by CJK codepoints that have no combining marks
//!   this is a lossless simplification; a follow-up would revisit if we wire
//!   a Latin-heavy checkpoint.
//! - **Punctuation = ASCII P/S ranges only.** BERT's Python uses Unicode
//!   `P*` categories; we cover the same ASCII rows plus the ranges the paper
//!   calls out as CJK punctuation (`0x3000`-`0x303F`, `0xFF00`-`0xFFEF`),
//!   which is what a Chinese-language model actually sees.
//!
//! # Not wired into any BERT model here
//!
//! WP-17 delivers this module standalone. WP-19 handles wiring into a live
//! `BertBaseEncoder` (a new encoder that WordPiece pairs with — the existing
//! [`crate::deberta_v2`] / [`crate::deberta_v3`] are DeBERTa and cannot host
//! a WordPiece tokenizer, because DeBERTa uses SentencePiece).
//!
//! # References (all permissive)
//!
//! - Devlin et al. 2018 (arXiv:1810.04805) — BERT paper
//! - Wu et al. 2016 (arXiv:1609.08144 §4.1) — WordPiece algorithm
//! - google-research/bert (Apache-2.0) — reference tokenizer
//! - HuggingFace transformers `BertTokenizer` (Apache-2.0)
//!
//! # NOT REFERENCED
//!
//! - github.com/litagin02/Style-Bert-VITS2 (AGPL-3.0)
//! - github.com/fishaudio/Bert-VITS2 (AGPL-3.0)

use std::collections::HashMap;

use vokra_core::gguf::GgufFile;
use vokra_core::VokraError;

/// WordPiece continuation marker (Wu et al. 2016 §4.1). Every non-first
/// subword of a segmented token carries this prefix.
const CONTINUATION_PREFIX: &str = "##";

/// Max codepoint count per input word before falling back to `[UNK]`. The
/// BERT paper cap. Guards against pathological inputs (e.g. a 10 KB
/// no-space blob).
const MAX_CHARS_PER_WORD: usize = 100;

/// Behavior when a token cannot be segmented into any vocab entry.
///
/// # WP-14 precedent
///
/// A sibling OOV-policy enum was landed in another loader (see the SBV2
/// converter task graph); this variant follows the same pattern so callers
/// wanting FR-EX-08 fail-fast can opt in without disturbing the standard
/// HuggingFace-compatible default.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OovPolicy {
    /// Replace the unsegmentable token with `[UNK]` (standard BERT
    /// behavior — matches HuggingFace `BertTokenizer`).
    #[default]
    Unk,
    /// Return [`VokraError::InvalidArgument`] on any unsegmentable token
    /// (loud, no silent fallback — FR-EX-08 posture).
    Error,
}

/// WordPiece tokenizer for a BERT-family model.
///
/// Constructed from an ordered vocabulary (index = token id) plus the four
/// standard special-token ids (`[PAD]/[UNK]/[CLS]/[SEP]`). Optional builder
/// methods flip `do_lower_case` and switch [`OovPolicy`].
#[derive(Debug, Clone)]
pub struct BertWordpieceTokenizer {
    vocab: HashMap<String, u32>,
    ids_to_tokens: Vec<String>,
    unk_id: u32,
    cls_id: u32,
    sep_id: u32,
    pad_id: u32,
    do_lower_case: bool,
    oov_policy: OovPolicy,
}

impl BertWordpieceTokenizer {
    /// Build from an ordered vocab (`vocab[i]` is the token string for id `i`).
    ///
    /// All four special-token ids must be within `vocab.len()` and no token
    /// string may appear twice (duplicates would leave one id unreachable —
    /// [`VokraError::ModelLoad`] instead of a silent shadow).
    ///
    /// Defaults: `do_lower_case = true`, `oov_policy = OovPolicy::Unk` (both
    /// matching HuggingFace `BertTokenizer` uncased-base defaults).
    pub fn from_vocab(
        vocab: Vec<String>,
        unk_id: u32,
        cls_id: u32,
        sep_id: u32,
        pad_id: u32,
    ) -> Result<Self, VokraError> {
        if vocab.is_empty() {
            return Err(VokraError::ModelLoad(
                "BertWordpieceTokenizer: empty vocab".to_string(),
            ));
        }
        let vocab_size = vocab.len();
        for id in [unk_id, cls_id, sep_id, pad_id] {
            if (id as usize) >= vocab_size {
                return Err(VokraError::ModelLoad(format!(
                    "BertWordpieceTokenizer: special id {id} out of range (vocab_size {vocab_size})"
                )));
            }
        }
        let mut map: HashMap<String, u32> = HashMap::with_capacity(vocab_size);
        for (i, tok) in vocab.iter().enumerate() {
            if map.insert(tok.clone(), i as u32).is_some() {
                return Err(VokraError::ModelLoad(format!(
                    "BertWordpieceTokenizer: duplicate token '{tok}' in vocab"
                )));
            }
        }
        Ok(Self {
            vocab: map,
            ids_to_tokens: vocab,
            unk_id,
            cls_id,
            sep_id,
            pad_id,
            do_lower_case: true,
            oov_policy: OovPolicy::default(),
        })
    }

    /// Enable or disable the initial-lowercasing pass (default `true`).
    #[must_use]
    pub fn with_lower_case(mut self, do_lower_case: bool) -> Self {
        self.do_lower_case = do_lower_case;
        self
    }

    /// Override the OOV policy (default [`OovPolicy::Unk`]).
    #[must_use]
    pub fn with_oov_policy(mut self, oov_policy: OovPolicy) -> Self {
        self.oov_policy = oov_policy;
        self
    }

    /// Vocab size (id space is `0..vocab_size()`).
    pub fn vocab_size(&self) -> usize {
        self.ids_to_tokens.len()
    }

    /// `[UNK]` token id, as configured by [`from_vocab`](Self::from_vocab).
    pub fn unk_id(&self) -> u32 {
        self.unk_id
    }

    /// `[CLS]` token id.
    pub fn cls_id(&self) -> u32 {
        self.cls_id
    }

    /// `[SEP]` token id.
    pub fn sep_id(&self) -> u32 {
        self.sep_id
    }

    /// `[PAD]` token id.
    pub fn pad_id(&self) -> u32 {
        self.pad_id
    }

    /// Encode `text` into a sequence of token ids.
    ///
    /// If `add_special_tokens` is true, the result is `[CLS] ...ids... [SEP]`.
    ///
    /// # Errors
    ///
    /// Returns [`VokraError::InvalidArgument`] only if `oov_policy` is
    /// [`OovPolicy::Error`] and any input token cannot be segmented.
    /// The default [`OovPolicy::Unk`] path is infallible.
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>, VokraError> {
        let tokens = self.tokenize(text)?;
        let capacity = tokens.len() + if add_special_tokens { 2 } else { 0 };
        let mut ids: Vec<u32> = Vec::with_capacity(capacity);
        if add_special_tokens {
            ids.push(self.cls_id);
        }
        for tok in &tokens {
            let id = *self.vocab.get(tok).unwrap_or(&self.unk_id);
            ids.push(id);
        }
        if add_special_tokens {
            ids.push(self.sep_id);
        }
        Ok(ids)
    }

    /// Run just the tokenization pipeline (basic + WordPiece) without
    /// mapping to ids. Exposed for testing and for callers that want the
    /// intermediate string tokens.
    pub fn tokenize(&self, text: &str) -> Result<Vec<String>, VokraError> {
        let cleaned = clean_text(text);
        let with_cjk_split = insert_cjk_whitespace(&cleaned);
        let mut split_tokens: Vec<String> = Vec::new();
        for ws_tok in whitespace_split(&with_cjk_split) {
            let cased = if self.do_lower_case {
                ws_tok.to_lowercase()
            } else {
                ws_tok.to_string()
            };
            for piece in split_on_punc(&cased) {
                split_tokens.push(piece);
            }
        }
        let mut output = Vec::new();
        for tok in &split_tokens {
            for sub in self.wordpiece_segment(tok)? {
                output.push(sub);
            }
        }
        Ok(output)
    }

    /// WordPiece per-word segmentation (Wu et al. 2016 §4.1).
    ///
    /// Greedy longest-prefix; continuation prefixes are `"##"`. Non-first
    /// slice indexes over `chars()`, so multibyte codepoints (Chinese, kana,
    /// emoji) are counted as one unit each — matching HuggingFace's Python
    /// which likewise indexes over Python `str` (which is codepoint-based).
    fn wordpiece_segment(&self, word: &str) -> Result<Vec<String>, VokraError> {
        let chars: Vec<char> = word.chars().collect();
        if chars.is_empty() {
            return Ok(Vec::new());
        }
        if chars.len() > MAX_CHARS_PER_WORD {
            return self.oov_result(word);
        }

        let mut sub_tokens: Vec<String> = Vec::new();
        let mut start = 0usize;
        while start < chars.len() {
            let mut end = chars.len();
            let mut matched: Option<(String, usize)> = None; // (substr, end_char_index)
            while start < end {
                let slice: String = chars[start..end].iter().collect();
                let candidate = if start == 0 {
                    slice
                } else {
                    format!("{CONTINUATION_PREFIX}{slice}")
                };
                if self.vocab.contains_key(&candidate) {
                    matched = Some((candidate, end));
                    break;
                }
                end -= 1;
            }
            match matched {
                Some((cand, new_start)) => {
                    sub_tokens.push(cand);
                    start = new_start;
                }
                None => {
                    // Cannot segment this word at all — the standard BERT
                    // behavior is to replace the *whole* word with UNK, not
                    // to emit partial ##-fragments.
                    return self.oov_result(word);
                }
            }
        }
        Ok(sub_tokens)
    }

    fn oov_result(&self, word: &str) -> Result<Vec<String>, VokraError> {
        match self.oov_policy {
            OovPolicy::Unk => {
                let unk_tok = self.ids_to_tokens[self.unk_id as usize].clone();
                Ok(vec![unk_tok])
            }
            OovPolicy::Error => Err(VokraError::InvalidArgument(format!(
                "BertWordpieceTokenizer: cannot segment '{word}' (OovPolicy::Error)"
            ))),
        }
    }

    /// Load from GGUF metadata written by a converter.
    ///
    /// # Metadata keys (all under caller-supplied `prefix`)
    ///
    /// | Key                       | Type            | Required | Notes                          |
    /// |---------------------------|-----------------|----------|--------------------------------|
    /// | `{prefix}.vocab`          | `ARRAY<STRING>` | yes      | id = index                     |
    /// | `{prefix}.unk_id`         | `U32`           | no       | default `100` (BERT standard)  |
    /// | `{prefix}.cls_id`         | `U32`           | no       | default `101`                  |
    /// | `{prefix}.sep_id`         | `U32`           | no       | default `102`                  |
    /// | `{prefix}.pad_id`         | `U32`           | no       | default `0`                    |
    /// | `{prefix}.do_lower_case`  | `BOOL`          | no       | default `true`                 |
    ///
    /// A future WordPiece converter should emit these keys under e.g.
    /// `vokra.bert_base.wordpiece` (for a standalone `BertBaseEncoder`) or
    /// `vokra.sbv2.bert_zh.wordpiece` (for SBV2's ZH slot). The choice of
    /// prefix is left to the converter; this loader only reads what the
    /// caller names.
    ///
    /// Mirrors [`crate::tokenizer::SbertTokenizer::from_gguf`]'s contract
    /// (same missing-key error style, same default-fallback pattern).
    pub fn from_gguf(gguf: &GgufFile, prefix: &str) -> Result<Self, VokraError> {
        let vocab_key = format!("{prefix}.vocab");
        let vocab: Vec<String> = {
            let val = gguf.get(&vocab_key).ok_or_else(|| {
                VokraError::ModelLoad(format!("missing GGUF metadata key: {vocab_key}"))
            })?;
            let arr = val.as_array().ok_or_else(|| {
                VokraError::ModelLoad(format!("GGUF metadata key {vocab_key} is not an array"))
            })?;
            arr.values
                .iter()
                .map(|v| {
                    v.as_str().map(|s| s.to_owned()).ok_or_else(|| {
                        VokraError::ModelLoad(format!(
                            "element in {vocab_key} array is not a string"
                        ))
                    })
                })
                .collect::<Result<Vec<_>, _>>()?
        };

        // Standard uncased-base BERT layout defaults; safe if the converter
        // does not emit them.
        let unk_id = gguf
            .get(&format!("{prefix}.unk_id"))
            .and_then(|v| v.as_u64())
            .map(|v| v as u32)
            .unwrap_or(100);
        let cls_id = gguf
            .get(&format!("{prefix}.cls_id"))
            .and_then(|v| v.as_u64())
            .map(|v| v as u32)
            .unwrap_or(101);
        let sep_id = gguf
            .get(&format!("{prefix}.sep_id"))
            .and_then(|v| v.as_u64())
            .map(|v| v as u32)
            .unwrap_or(102);
        let pad_id = gguf
            .get(&format!("{prefix}.pad_id"))
            .and_then(|v| v.as_u64())
            .map(|v| v as u32)
            .unwrap_or(0);
        let do_lower_case = gguf
            .get(&format!("{prefix}.do_lower_case"))
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

        let mut tok = Self::from_vocab(vocab, unk_id, cls_id, sep_id, pad_id)?;
        tok.do_lower_case = do_lower_case;
        Ok(tok)
    }
}

// -----------------------------------------------------------------------------
// BasicTokenizer helpers (public-BERT algorithm)
// -----------------------------------------------------------------------------

/// Strip control chars and replace unusual whitespace with ASCII space.
///
/// BERT paper "BasicTokenizer._clean_text": drop `\0`, `0xFFFD`, and Unicode
/// controls (categories starting with `C` except whitespace); replace any
/// whitespace char with a single ASCII space. We approximate the Unicode
/// category test with an explicit low/high control-code range check plus
/// the four ASCII whitespace codes — enough for the corpora BERT targets.
fn clean_text(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        let cp = c as u32;
        if cp == 0 || cp == 0xFFFD || is_control(c) {
            continue;
        }
        if is_whitespace(c) {
            out.push(' ');
        } else {
            out.push(c);
        }
    }
    out
}

/// Wrap every CJK codepoint in ASCII spaces, so downstream whitespace
/// tokenization emits it as a standalone token — the mechanism by which
/// Chinese input becomes one-char-per-token.
fn insert_cjk_whitespace(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 4);
    for c in text.chars() {
        if is_cjk_char(c as u32) {
            out.push(' ');
            out.push(c);
            out.push(' ');
        } else {
            out.push(c);
        }
    }
    out
}

/// Iterator over whitespace-delimited runs — the BERT
/// `whitespace_tokenize` step. Uses Rust `str::split_whitespace` which
/// tracks Unicode whitespace (matches the paper's intent).
fn whitespace_split(text: &str) -> impl Iterator<Item = &str> {
    text.split_whitespace()
}

/// Split a single already-cased, whitespace-free string on punctuation.
///
/// Runs of non-punctuation chars stay glued; each punctuation codepoint
/// becomes its own single-char string. Matches the BERT `_run_split_on_punc`
/// semantics: `"don't"` → `["don", "'", "t"]`.
fn split_on_punc(text: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut buf = String::new();
    for c in text.chars() {
        if is_punctuation_char(c) {
            if !buf.is_empty() {
                out.push(std::mem::take(&mut buf));
            }
            out.push(c.to_string());
        } else {
            buf.push(c);
        }
    }
    if !buf.is_empty() {
        out.push(buf);
    }
    out
}

/// BERT `_is_whitespace`: ASCII space/tab/CR/LF.
fn is_whitespace(c: char) -> bool {
    matches!(c, ' ' | '\t' | '\n' | '\r')
}

/// BERT `_is_control`: Unicode C* except whitespace. Approximated with the
/// ASCII/Latin-1 control blocks (documented gap for exotic controls beyond
/// U+009F — a follow-up if we start ingesting deliberately-adversarial input).
fn is_control(c: char) -> bool {
    if is_whitespace(c) {
        return false;
    }
    let cp = c as u32;
    cp <= 0x1F || (0x7F..=0x9F).contains(&cp)
}

/// BERT `_is_punctuation`: ASCII P/S rows plus CJK punctuation blocks
/// (`0x3000`-`0x303F` general, `0xFF00`-`0xFFEF` fullwidth).
///
/// The paper's Python uses `unicodedata.category(c).startswith("P")` which
/// covers more codepoints than these ranges; the documented gap is Latin
/// extended punctuation not used by our target `hfl/chinese-roberta-wwm-ext-large`.
fn is_punctuation_char(c: char) -> bool {
    let cp = c as u32;
    if (0x21..=0x2F).contains(&cp)
        || (0x3A..=0x40).contains(&cp)
        || (0x5B..=0x60).contains(&cp)
        || (0x7B..=0x7E).contains(&cp)
    {
        return true;
    }
    if (0x3000..=0x303F).contains(&cp) || (0xFF00..=0xFFEF).contains(&cp) {
        return true;
    }
    false
}

/// BERT `_is_chinese_char`: the eight CJK Unicode ranges the paper lists.
/// (Includes CJK Unified, Extensions A–E, Compatibility, and their
/// supplementary planes.)
fn is_cjk_char(cp: u32) -> bool {
    (0x4E00..=0x9FFF).contains(&cp)
        || (0x3400..=0x4DBF).contains(&cp)
        || (0x20000..=0x2A6DF).contains(&cp)
        || (0x2A700..=0x2B73F).contains(&cp)
        || (0x2B740..=0x2B81F).contains(&cp)
        || (0x2B820..=0x2CEAF).contains(&cp)
        || (0xF900..=0xFAFF).contains(&cp)
        || (0x2F800..=0x2FA1F).contains(&cp)
}

// -----------------------------------------------------------------------------
// Unit tests (helper predicates + tiny end-to-end)
// -----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cjk_range_covers_ni_hao() {
        assert!(is_cjk_char('你' as u32));
        assert!(is_cjk_char('好' as u32));
        assert!(is_cjk_char('龘' as u32));
        assert!(!is_cjk_char('a' as u32));
        assert!(!is_cjk_char('1' as u32));
    }

    #[test]
    fn punctuation_matches_ascii_and_cjk_ranges() {
        assert!(is_punctuation_char('!'));
        assert!(is_punctuation_char('?'));
        assert!(is_punctuation_char('.'));
        assert!(is_punctuation_char(','));
        assert!(is_punctuation_char('。')); // 0x3002 CJK ideographic full stop
        assert!(is_punctuation_char('！')); // 0xFF01 fullwidth exclamation
        assert!(!is_punctuation_char('a'));
        assert!(!is_punctuation_char('你'));
    }

    #[test]
    fn whitespace_and_control_predicates() {
        assert!(is_whitespace(' '));
        assert!(is_whitespace('\t'));
        assert!(!is_whitespace('a'));
        assert!(is_control('\x01'));
        assert!(is_control('\x7F'));
        assert!(!is_control('\n')); // whitespace, not control
        assert!(!is_control('a'));
    }

    #[test]
    fn tokenize_intermediate_pipeline_visible_via_tokenize_only() {
        // 4-entry vocab: [UNK], hello, wor, ##ld
        let vocab = vec![
            "[UNK]".to_string(),
            "hello".to_string(),
            "wor".to_string(),
            "##ld".to_string(),
        ];
        let t = BertWordpieceTokenizer::from_vocab(vocab, 0, 0, 0, 0)
            .expect("valid vocab (special ids reused for test brevity)");
        let toks = t.tokenize("Hello world").expect("tokenize");
        assert_eq!(
            toks,
            vec!["hello".to_string(), "wor".to_string(), "##ld".to_string(),]
        );
    }
}
