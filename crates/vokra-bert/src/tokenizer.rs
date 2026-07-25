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

const WORD_START: char = '▁'; // U+2581 LOWER ONE EIGHTH BLOCK — SentencePiece word boundary

#[derive(Debug, Clone)]
pub struct SbertTokenizer {
    pieces: Vec<(String, f32)>, // (piece, log_prob)
    unk_id: u32,
    bos_id: u32,
    eos_id: u32,
    // For fast lookup: map piece -> id (used in Task 4)
    #[allow(dead_code)]
    piece_to_id: std::collections::HashMap<String, u32>,
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
            piece_to_id,
        }
    }

    /// Encode a UTF-8 string into piece ids using viterbi.
    /// Prepends `▁` for word starts (SentencePiece "add_dummy_prefix" default).
    pub fn encode(&self, text: &str) -> Vec<u32> {
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

    /// Decode piece ids back to a UTF-8 string. Skips special tokens
    /// (`<pad>` / `<unk>` / `<s>` / `</s>`), converts `▁` back to space.
    pub fn decode(&self, ids: &[u32]) -> String {
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

    /// Load from GGUF metadata written by `vokra-bert::converter::convert_tokenizer`.
    /// Metadata keys:
    /// - `<prefix>.pieces` = STRING array
    /// - `<prefix>.scores` = F32 array (same length)
    /// - `<prefix>.unk_id` / `.bos_id` / `.eos_id` = U32
    pub fn from_gguf(gguf: &GgufFile, prefix: &str) -> Result<Self, VokraError> {
        let pieces_key = format!("{prefix}.pieces");
        let scores_key = format!("{prefix}.scores");
        let unk_key = format!("{prefix}.unk_id");
        let bos_key = format!("{prefix}.bos_id");
        let eos_key = format!("{prefix}.eos_id");

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

        let piece_to_id = pieces
            .iter()
            .enumerate()
            .map(|(i, p)| (p.clone(), i as u32))
            .collect();

        Ok(Self {
            pieces: pieces.into_iter().zip(scores).collect(),
            unk_id,
            bos_id,
            eos_id,
            piece_to_id,
        })
    }
}
