//! SentencePiece BPE tokenizer — clean-room per Kudo & Richardson 2018
//! (arXiv:1808.06226). No AGPL sources consulted.
//!
//! # Algorithm
//!
//! SentencePiece stores (piece, log_prob) pairs. `encode` performs the
//! standard viterbi search over the input: at each position, choose the
//! piece that maximizes total log_prob (equiv. to shortest-cost path in
//! the "trellis" of prefixes ending at that position).
//!
//! # References (permissive only)
//!
//! - Kudo & Richardson 2018 (arXiv:1808.06226)
//! - google/sentencepiece (Apache-2.0)
//!
//! # NOT REFERENCED
//!
//! - github.com/litagin02/Style-Bert-VITS2 (AGPL-3.0)

use vokra_core::gguf::GgufFile;
use vokra_core::VokraError;

use crate::wordpiece::BertWordpieceTokenizer;

const WORD_START: char = '▁'; // U+2581 LOWER ONE EIGHTH BLOCK — SentencePiece word boundary

/// Runtime tokenizer algorithm. HF `BertJapaneseTokenizer` with
/// `subword_tokenizer_type="character"` (used by
/// `ku-nlp/deberta-v2-large-japanese-char-wwm`) splits by Unicode code point;
/// SentencePiece Unigram tokenizers (used by `microsoft/deberta-v3-large`)
/// run a Viterbi search with a `▁` word-start marker. The converter stamps
/// the discriminator into `<prefix>.kind`
/// (`vokra_convert::models::deberta_v2::{KIND_BERT_CHARSPLIT,
/// KIND_SENTENCEPIECE_UNIGRAM}`) so `from_gguf` can pick the correct branch.
/// Runtime choice is intentional over converter-time flattening: the tensor
/// vocabulary is the same shape either way, only encode's semantics differ.
///
/// # Why char-split was silent-wrong pre-2026-08-09 (task #7 root cause)
///
/// Before this enum, `encode()` always ran SentencePiece Viterbi + prepended
/// `▁`. Fed a char-level vocab (which has entries like `"テ"` but not `"▁テ"`),
/// it turned "テスト" into 4 tokens (`[unk, テ, ス, ト]` — 1 spurious for the
/// `▁` prefix), not the 3 chars HF `BertJapaneseTokenizer(add_special_tokens=True)`
/// wraps into `[CLS, テ, ス, ト, SEP]` = 5 tokens. The 4 vs 5 length mismatch
/// then propagated into `bert_hidden_ja` as a `[4, 1024]` vs `[5, 1024]`
/// shape divergence in the parity harness.
///
/// # Relationship to [`TokenizerScheme`]
///
/// `TokenizerKind` is the sub-choice **within** the Unigram scheme (SentencePiece
/// Viterbi vs Bert char-split, both semantically Unigram-family), read from
/// `<prefix>.kind` metadata. [`TokenizerScheme`] is the top-level algorithm-family
/// selector (Unigram vs WordPiece), read from `<prefix>.scheme` metadata. When
/// `scheme = wordpiece`, the [`TokenizerKind`] field is ignored at encode time —
/// the WordPiece path is taken.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenizerKind {
    /// HF `BertJapaneseTokenizer(subword_tokenizer_type="character")` — one
    /// Unicode code point per token, unknown chars go to `unk_id`.
    BertCharSplit,
    /// SentencePiece Unigram Viterbi (existing behaviour).
    SentencePiece,
}

/// Which sub-word algorithm-family a loaded [`SbertTokenizer`] runs.
///
/// The scheme is stamped by the converter into `<prefix>.scheme`
/// (STRING) — DeBERTa v3's `spm.model` gives
/// [`TokenizerScheme::Unigram`], DeBERTa v2's `vocab.txt` gives
/// [`TokenizerScheme::Wordpiece`]. When the metadata key is absent the
/// loader defaults to [`TokenizerScheme::Unigram`] for backward
/// compatibility with GGUFs that predate the schema.
///
/// See [`TokenizerKind`] for the sub-choice within Unigram scheme.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenizerScheme {
    /// SentencePiece Unigram (viterbi over log-probabilities).
    Unigram,
    /// WordPiece (greedy longest-match with `##` continuation).
    Wordpiece,
}

impl TokenizerScheme {
    fn from_str(s: &str) -> Result<Self, VokraError> {
        match s {
            "unigram" => Ok(Self::Unigram),
            "wordpiece" => Ok(Self::Wordpiece),
            other => Err(VokraError::ModelLoad(format!(
                "SbertTokenizer::from_gguf: unknown tokenizer scheme `{other}` (expected \
                 `unigram` or `wordpiece`)"
            ))),
        }
    }
}

/// SBV2 / DeBERTa BERT-family tokenizer.
///
/// Dispatches at load time on the `<prefix>.scheme` metadata key:
/// SentencePiece Unigram (viterbi over `(piece, log_prob)` pairs) for
/// DeBERTa v3, WordPiece (greedy longest-match with `##`
/// continuation) for DeBERTa v2. Within the Unigram scheme, the
/// [`TokenizerKind`] sub-selector further chooses between SentencePiece
/// Viterbi (`microsoft/deberta-v3-large`) and HF BERT char-split
/// (`ku-nlp/deberta-v2-large-japanese-char-wwm`). The public
/// [`Self::encode`] / [`Self::decode`] entry points route to the
/// underlying implementation transparently.
#[derive(Debug, Clone)]
pub struct SbertTokenizer {
    pieces: Vec<(String, f32)>, // (piece, log_prob) — populated for Unigram
    unk_id: u32,
    bos_id: u32,
    eos_id: u32,
    kind: TokenizerKind,
    // For fast lookup: map piece -> id (used in Task 4)
    #[allow(dead_code)]
    piece_to_id: std::collections::HashMap<String, u32>,
    /// WordPiece variant only — populated when `scheme = wordpiece`;
    /// otherwise `None` and the Unigram viterbi over `pieces` is used.
    wordpiece: Option<BertWordpieceTokenizer>,
}

impl SbertTokenizer {
    /// Test constructor — build from a pieces vec (id = index).
    /// Requires: pieces[0]=<pad>, pieces[1]=<unk>, pieces[2]=<s>, pieces[3]=</s>.
    #[doc(hidden)]
    pub fn from_pieces_for_test(pieces: Vec<(String, f32)>) -> Self {
        assert!(pieces.len() >= 4);
        assert_eq!(pieces[1].0, "<unk>");
        let piece_to_id = pieces
            .iter()
            .enumerate()
            .map(|(i, (p, _))| (p.clone(), i as u32))
            .collect();
        Self {
            pieces,
            unk_id: 1,
            bos_id: 2,
            eos_id: 3,
            kind: TokenizerKind::SentencePiece,
            piece_to_id,
            wordpiece: None,
        }
    }

    /// Test constructor for char-level HF BERT tokenizers
    /// (`BertJapaneseTokenizer(subword_tokenizer_type="character")`). Explicit
    /// unk/cls/sep ids because the char vocab convention (`[PAD]=0, [CLS]=1,
    /// [SEP]=2, [UNK]=3`) does NOT match SentencePiece defaults
    /// (`<pad>=0, <unk>=1, <s>=2, </s>=3`) — silently reusing SentencePiece
    /// ids would put `[UNK]` where `[SEP]` belongs (task #7 root cause
    /// pattern).
    #[doc(hidden)]
    pub fn from_pieces_for_test_charsplit(
        pieces: Vec<(String, f32)>,
        unk_id: u32,
        cls_id: u32,
        sep_id: u32,
    ) -> Self {
        assert!(pieces.len() >= 4);
        let piece_to_id = pieces
            .iter()
            .enumerate()
            .map(|(i, (p, _))| (p.clone(), i as u32))
            .collect();
        Self {
            pieces,
            unk_id,
            bos_id: cls_id,
            eos_id: sep_id,
            kind: TokenizerKind::BertCharSplit,
            piece_to_id,
            wordpiece: None,
        }
    }

    /// Test constructor — build a WordPiece-backed tokenizer directly
    /// from a plain piece list (id = index), bypassing the GGUF loader.
    #[doc(hidden)]
    pub fn from_wordpiece_for_test(
        pieces: Vec<String>,
        unk_id: u32,
        bos_id: u32,
        eos_id: u32,
    ) -> Self {
        let piece_to_id = pieces
            .iter()
            .enumerate()
            .map(|(i, p)| (p.clone(), i as u32))
            .collect();
        let wp = BertWordpieceTokenizer::from_pieces(pieces.clone(), unk_id, bos_id, eos_id);
        Self {
            pieces: pieces.into_iter().map(|p| (p, 0.0)).collect(),
            unk_id,
            bos_id,
            eos_id,
            // kind is ignored when wordpiece = Some(...); pick SentencePiece as a
            // safe default so no debug traversal accidentally treats a wordpiece
            // tokenizer as a char-splitter.
            kind: TokenizerKind::SentencePiece,
            piece_to_id,
            wordpiece: Some(wp),
        }
    }

    /// Char-level encode for HF `BertJapaneseTokenizer(subword_tokenizer_type="character")`:
    /// one Unicode code point per token, out-of-vocab chars go to `unk_id`.
    /// Consumers that need HF's `add_special_tokens=True` wrap should call
    /// [`encode_with_special_tokens`](Self::encode_with_special_tokens) instead.
    fn encode_charsplit(&self, text: &str) -> Vec<u32> {
        text.chars()
            .map(|c| {
                let key = c.to_string();
                self.piece_to_id.get(&key).copied().unwrap_or(self.unk_id)
            })
            .collect()
    }

    /// HF `tokenizer(text, add_special_tokens=True)` equivalent — wraps the
    /// active tokenizer's output with `bos_id` (=CLS for BERT-family char /
    /// `<s>` for SentencePiece) at the front and `eos_id` (=SEP / `</s>`) at
    /// the tail. Consumers reading `bert_hidden_*` reference dumps from HF
    /// MUST call this (not [`encode`](Self::encode)) so the token count
    /// matches; see [`TokenizerKind::BertCharSplit`]'s doc for the task #7
    /// regression that motivated this API.
    ///
    /// For WordPiece-backed tokenizers this simply wraps
    /// [`Self::encode`] (WordPiece has no charsplit sub-choice — the
    /// path is already correct).
    pub fn encode_with_special_tokens(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        ids.push(self.bos_id);
        if self.wordpiece.is_some() {
            ids.extend(self.encode(text));
        } else {
            ids.extend(match self.kind {
                TokenizerKind::BertCharSplit => self.encode_charsplit(text),
                TokenizerKind::SentencePiece => self.encode(text),
            });
        }
        ids.push(self.eos_id);
        ids
    }

    /// Encode a UTF-8 string into piece ids.
    ///
    /// Routes to the WordPiece greedy longest-match walk when the
    /// tokenizer was loaded with `scheme = wordpiece`; otherwise runs
    /// the SentencePiece Unigram viterbi (default). Neither path emits
    /// BOS / EOS sentinels — the caller supplies them.
    ///
    /// NOTE: for char-level HF `BertJapaneseTokenizer` consumers the Viterbi
    /// path silently produces wrong-shape output (WORD_START token is not in
    /// the char vocab so it maps to UNK, adding a spurious token). Use
    /// [`encode_with_special_tokens`](Self::encode_with_special_tokens) —
    /// which respects [`TokenizerKind`] — for those consumers instead.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        if let Some(wp) = &self.wordpiece {
            return wp.encode_no_special(text);
        }
        self.encode_unigram(text)
    }

    /// SentencePiece Unigram viterbi over `(piece, log_prob)` pairs.
    /// Prepends `▁` for word starts (SentencePiece "add_dummy_prefix"
    /// default).
    fn encode_unigram(&self, text: &str) -> Vec<u32> {
        // Prepare input: replace ASCII space with `▁`, prepend `▁` to first token.
        let mut prepared = String::with_capacity(text.len() + 4);
        prepared.push(WORD_START);
        for c in text.chars() {
            if c == ' ' {
                prepared.push(WORD_START);
            } else {
                prepared.push(c);
            }
        }
        // Viterbi: dp[i] = (best log_prob to reach byte i, back_piece_len_bytes, back_piece_id).
        let bytes = prepared.as_bytes();
        let n = bytes.len();
        let mut dp: Vec<(f32, usize, u32)> = vec![(f32::NEG_INFINITY, 0, self.unk_id); n + 1];
        dp[0] = (0.0, 0, self.unk_id);
        for i in 0..n {
            if dp[i].0 == f32::NEG_INFINITY {
                continue;
            }
            // Try all pieces starting at i.
            for (id, (piece, log_prob)) in self.pieces.iter().enumerate() {
                let pb = piece.as_bytes();
                if i + pb.len() <= n && &bytes[i..i + pb.len()] == pb {
                    let cand = dp[i].0 + log_prob;
                    if cand > dp[i + pb.len()].0 {
                        dp[i + pb.len()] = (cand, pb.len(), id as u32);
                    }
                }
            }
            // Byte-level unk fallback (1 byte cost = log(1e-4)).
            if dp[i + 1].0 == f32::NEG_INFINITY {
                let cand = dp[i].0 + (-9.21_f32); // ln(1e-4)
                dp[i + 1] = (cand, 1, self.unk_id);
            }
        }
        // Backtrack.
        let mut ids = Vec::new();
        let mut pos = n;
        while pos > 0 {
            let (_, len, id) = dp[pos];
            ids.push(id);
            pos -= len;
        }
        ids.reverse();
        ids
    }

    pub fn unk_id(&self) -> u32 {
        self.unk_id
    }
    pub fn bos_id(&self) -> u32 {
        self.bos_id
    }
    pub fn eos_id(&self) -> u32 {
        self.eos_id
    }

    /// Decode piece ids back to a UTF-8 string.
    ///
    /// Routes to the WordPiece decode (`##` continuation stitching)
    /// when the tokenizer was loaded with `scheme = wordpiece`;
    /// otherwise runs the Unigram decode (`▁` → space, special tokens
    /// filtered).
    pub fn decode(&self, ids: &[u32]) -> String {
        if let Some(wp) = &self.wordpiece {
            return wp.decode(ids);
        }
        let mut out = String::new();
        for &id in ids {
            // Skip special tokens: pad (always 0) + unk/bos/eos
            if id == self.unk_id || id == self.bos_id || id == self.eos_id || id == 0 {
                continue;
            }
            if let Some((piece, _)) = self.pieces.get(id as usize) {
                for c in piece.chars() {
                    if c == WORD_START {
                        out.push(' ');
                    } else {
                        out.push(c);
                    }
                }
            }
        }
        // Strip leading space that comes from the mandatory word-start `▁`.
        out.trim_start().to_string()
    }

    /// Load from GGUF metadata written by
    /// `vokra-convert::models::deberta_v2::convert_deberta_v2_file` (WordPiece)
    /// or
    /// `vokra-convert::models::deberta_v3::convert_deberta_v3_file` (SentencePiece
    /// Unigram).
    ///
    /// Metadata keys:
    /// - `<prefix>.scheme` = STRING (`"unigram"` or `"wordpiece"`) —
    ///   optional; defaults to `unigram` for backward compatibility
    ///   with pre-Blocker-5 GGUFs
    /// - `<prefix>.pieces` = STRING array (piece list, id = index)
    /// - `<prefix>.scores` = F32 array (same length; all-zero for
    ///   WordPiece which ignores scores)
    /// - `<prefix>.unk_id` / `.bos_id` / `.eos_id` = U32
    pub fn from_gguf(gguf: &GgufFile, prefix: &str) -> Result<Self, VokraError> {
        let scheme_key = format!("{prefix}.scheme");
        let pieces_key = format!("{prefix}.pieces");
        let scores_key = format!("{prefix}.scores");
        let unk_key = format!("{prefix}.unk_id");
        let bos_key = format!("{prefix}.bos_id");
        let eos_key = format!("{prefix}.eos_id");
        let kind_key = format!("{prefix}.kind");

        // Resolve scheme first (default = Unigram for legacy GGUFs).
        let scheme = match gguf.get(&scheme_key).and_then(|v| v.as_str()) {
            Some(s) => TokenizerScheme::from_str(s)?,
            None => TokenizerScheme::Unigram,
        };

        // Extract pieces array
        let pieces: Vec<String> = {
            let val = gguf.get(&pieces_key).ok_or_else(|| {
                VokraError::ModelLoad(format!("missing GGUF metadata key: {pieces_key}"))
            })?;
            let arr = val.as_array().ok_or_else(|| {
                VokraError::ModelLoad(format!("GGUF metadata key {pieces_key} is not an array"))
            })?;
            arr.values
                .iter()
                .map(|v| {
                    v.as_str().map(|s| s.to_owned()).ok_or_else(|| {
                        VokraError::ModelLoad(format!(
                            "element in {pieces_key} array is not a string"
                        ))
                    })
                })
                .collect::<Result<Vec<_>, _>>()?
        };

        // Extract scores array
        let scores: Vec<f32> = {
            let val = gguf.get(&scores_key).ok_or_else(|| {
                VokraError::ModelLoad(format!("missing GGUF metadata key: {scores_key}"))
            })?;
            let arr = val.as_array().ok_or_else(|| {
                VokraError::ModelLoad(format!("GGUF metadata key {scores_key} is not an array"))
            })?;
            arr.values
                .iter()
                .map(|v| {
                    v.as_f64().map(|f| f as f32).ok_or_else(|| {
                        VokraError::ModelLoad(format!(
                            "element in {scores_key} array is not a float"
                        ))
                    })
                })
                .collect::<Result<Vec<_>, _>>()?
        };

        // Verify length consistency
        if pieces.len() != scores.len() {
            return Err(VokraError::ModelLoad(format!(
                "{prefix}: pieces ({}) vs scores ({}) length mismatch",
                pieces.len(),
                scores.len()
            )));
        }

        // Extract special token ids with sensible defaults
        let unk_id = gguf
            .get(&unk_key)
            .and_then(|v| v.as_u64())
            .map(|v| v as u32)
            .unwrap_or(1);
        let bos_id = gguf
            .get(&bos_key)
            .and_then(|v| v.as_u64())
            .map(|v| v as u32)
            .unwrap_or(2);
        let eos_id = gguf
            .get(&eos_key)
            .and_then(|v| v.as_u64())
            .map(|v| v as u32)
            .unwrap_or(3);

        // Kind discriminator stamped by the converter (see
        // `vokra_convert::models::deberta_v2::{KIND_BERT_CHARSPLIT,
        // KIND_SENTENCEPIECE_UNIGRAM}`). Default = SentencePiece for
        // backward compat with GGUFs written before the kind stamp existed.
        let kind = match gguf.get(&kind_key).and_then(|v| v.as_str()) {
            Some("bert-charsplit") => TokenizerKind::BertCharSplit,
            Some("sentencepiece-unigram") | None => TokenizerKind::SentencePiece,
            Some(other) => {
                return Err(VokraError::ModelLoad(format!(
                    "{kind_key}: unknown tokenizer kind {other:?} — expected \
                     \"bert-charsplit\" or \"sentencepiece-unigram\""
                )));
            }
        };

        let piece_to_id = pieces
            .iter()
            .enumerate()
            .map(|(i, p)| (p.clone(), i as u32))
            .collect();

        let wordpiece = match scheme {
            TokenizerScheme::Unigram => None,
            TokenizerScheme::Wordpiece => Some(BertWordpieceTokenizer::from_pieces(
                pieces.clone(),
                unk_id,
                bos_id,
                eos_id,
            )),
        };

        Ok(Self {
            pieces: pieces.into_iter().zip(scores).collect(),
            unk_id,
            bos_id,
            eos_id,
            kind,
            piece_to_id,
            wordpiece,
        })
    }
}

#[cfg(test)]
mod scheme_dispatch_tests {
    use super::*;
    use vokra_core::gguf::value::GgufValueType;
    use vokra_core::gguf::{value::GgufArray, GgufBuilder, GgufFile, GgufMetadataValue};

    /// Build a GGUF with the specified tokenizer metadata written under
    /// `vokra.bert.tokenizer.*`, parse it back and return the file.
    fn build_gguf(
        scheme: Option<&str>,
        pieces: &[&str],
        scores: &[f32],
        unk_id: u32,
        bos_id: u32,
        eos_id: u32,
    ) -> GgufFile {
        let mut b = GgufBuilder::new();
        if let Some(s) = scheme {
            b.add_string("vokra.bert.tokenizer.scheme", s);
        }
        b.add_metadata(
            "vokra.bert.tokenizer.pieces",
            GgufMetadataValue::Array(GgufArray {
                element_type: GgufValueType::String,
                values: pieces
                    .iter()
                    .map(|p| GgufMetadataValue::String((*p).to_owned()))
                    .collect(),
            }),
        );
        b.add_metadata(
            "vokra.bert.tokenizer.scores",
            GgufMetadataValue::Array(GgufArray {
                element_type: GgufValueType::F32,
                values: scores.iter().map(|s| GgufMetadataValue::F32(*s)).collect(),
            }),
        );
        b.add_u32("vokra.bert.tokenizer.unk_id", unk_id);
        b.add_u32("vokra.bert.tokenizer.bos_id", bos_id);
        b.add_u32("vokra.bert.tokenizer.eos_id", eos_id);
        let bytes = b.to_bytes().expect("serialize");
        GgufFile::parse(bytes).expect("parse")
    }

    #[test]
    fn absent_scheme_defaults_to_unigram_for_backward_compat() {
        // Legacy GGUFs (pre-Blocker 5) do not stamp `.scheme` — must
        // still load as SentencePiece Unigram. Uses ASCII pieces so
        // the Unigram viterbi over `▁hello` will succeed.
        let pieces = ["<pad>", "<unk>", "<s>", "</s>", "\u{2581}hello"];
        let scores = [0.0, 0.0, 0.0, 0.0, -1.0];
        let gguf = build_gguf(None, &pieces, &scores, 1, 2, 3);
        let tok = SbertTokenizer::from_gguf(&gguf, "vokra.bert.tokenizer")
            .expect("legacy GGUF loads as Unigram");
        // `▁hello` piece matches directly.
        let ids = tok.encode("hello");
        assert_eq!(ids, vec![4]);
        assert_eq!(tok.decode(&ids), "hello");
    }

    #[test]
    fn explicit_unigram_scheme_matches_default() {
        let pieces = ["<pad>", "<unk>", "<s>", "</s>", "\u{2581}hi"];
        let scores = [0.0, 0.0, 0.0, 0.0, -1.0];
        let g_impl = build_gguf(Some("unigram"), &pieces, &scores, 1, 2, 3);
        let g_default = build_gguf(None, &pieces, &scores, 1, 2, 3);
        let t_impl = SbertTokenizer::from_gguf(&g_impl, "vokra.bert.tokenizer").unwrap();
        let t_default = SbertTokenizer::from_gguf(&g_default, "vokra.bert.tokenizer").unwrap();
        assert_eq!(t_impl.encode("hi"), t_default.encode("hi"));
    }

    #[test]
    fn wordpiece_scheme_dispatches_to_greedy_walk() {
        // Minimal WordPiece vocab: [PAD]=0 [UNK]=1 [CLS]=2 [SEP]=3
        // hello=4 ##world=5. encode("hello") should produce [4],
        // encode("helloworld") should produce [4, 5].
        let pieces = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "hello", "##world"];
        let scores = [0.0f32; 6];
        let gguf = build_gguf(Some("wordpiece"), &pieces, &scores, 1, 2, 3);
        let tok = SbertTokenizer::from_gguf(&gguf, "vokra.bert.tokenizer")
            .expect("wordpiece scheme loads");
        assert_eq!(tok.encode("hello"), vec![4]);
        assert_eq!(tok.encode("helloworld"), vec![4, 5]);
        // Decode stitches continuation onto previous piece.
        assert_eq!(tok.decode(&[4, 5]), "helloworld");
    }

    #[test]
    fn unknown_scheme_is_loud_error() {
        let pieces = ["<pad>", "<unk>", "<s>", "</s>"];
        let scores = [0.0f32; 4];
        let gguf = build_gguf(Some("bpe"), &pieces, &scores, 1, 2, 3);
        let err = SbertTokenizer::from_gguf(&gguf, "vokra.bert.tokenizer")
            .expect_err("unknown scheme rejected");
        let msg = format!("{err}");
        assert!(msg.contains("bpe"), "error must name the scheme: {msg}");
    }

    #[test]
    fn wordpiece_pieces_score_length_mismatch_still_errors() {
        // Length mismatch check must fire regardless of scheme.
        let pieces = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"];
        let scores = [0.0f32; 3]; // one short
        let gguf = build_gguf(Some("wordpiece"), &pieces, &scores, 1, 2, 3);
        let err = SbertTokenizer::from_gguf(&gguf, "vokra.bert.tokenizer")
            .expect_err("length mismatch rejected");
        let msg = format!("{err}");
        assert!(
            msg.contains("length mismatch"),
            "error must be a length mismatch: {msg}"
        );
    }

    #[test]
    fn missing_scheme_key_does_not_crash_on_wordpiece_pieces() {
        // A caller who forgets to stamp `.scheme` on a WordPiece vocab
        // still gets a loaded tokenizer — but it will run Unigram
        // viterbi on WordPiece pieces, which is user-error and produces
        // whatever the viterbi thinks is best. The loader must not
        // panic; correctness is the converter's responsibility (which
        // always stamps `.scheme` per Blocker 5).
        let pieces = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "hello"];
        let scores = [0.0f32; 5];
        let gguf = build_gguf(None, &pieces, &scores, 1, 2, 3);
        let tok =
            SbertTokenizer::from_gguf(&gguf, "vokra.bert.tokenizer").expect("load does not panic");
        // Just prove encode does not panic — no strong assertion on
        // the exact ids (the Unigram viterbi over a WordPiece vocab
        // is inherently a mismatch).
        let _ = tok.encode("hello");
    }
}
