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

use vokra_core::VokraError;

const WORD_START: char = '▁'; // U+2581 LOWER ONE EIGHTH BLOCK — SentencePiece word boundary

#[derive(Debug, Clone)]
pub struct SbertTokenizer {
    pieces: Vec<(String, f32)>, // (piece, log_prob)
    unk_id: u32,
    bos_id: u32,
    eos_id: u32,
    // For fast lookup: map piece -> id
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
}
