//! # vokra-bert
//!
//! BERT-family encoder + tokenizer for Vokra. Clean-room implementation
//! (Apache-2.0). Independent crate so future Vokra models beyond SBV2
//! (Bert-VITS2, Voxtral 系) can reuse the same encoder.
//!
//! # References (all permissive)
//!
//! - DeBERTa v2 paper: arXiv:2006.03654
//! - DeBERTa v3 paper: arXiv:2111.09543
//! - microsoft/DeBERTa (MIT): reference implementation
//! - HuggingFace transformers `deberta_v2` / `deberta_v3` (Apache-2.0)
//! - SentencePiece: Kudo & Richardson 2018
//!
//! # NOT REFERENCED
//!
//! - github.com/litagin02/Style-Bert-VITS2 (AGPL-3.0)
//! - github.com/fishaudio/Bert-VITS2 (AGPL-3.0)
//! - Any AGPL derivative of the above.

#![deny(unsafe_code)]

/// Uniform BERT encoder trait — implemented by DeBERTa v2, DeBERTa v3,
/// and plain BERT ([`bert_base::BertBaseEncoder`]).
///
/// `forward(ids)` returns hidden states as a flat `[seq_len × d_model]` Vec.
/// Callers know `seq_len = ids.len()`, so `d_model()` is exposed for the
/// row stride.
pub trait BertEncoder {
    fn forward(&self, ids: &[u32]) -> Vec<f32>;
    fn d_model(&self) -> usize;
}

impl BertEncoder for deberta_v2::DebertaV2Encoder {
    fn forward(&self, ids: &[u32]) -> Vec<f32> {
        deberta_v2::DebertaV2Encoder::forward(self, ids)
    }
    fn d_model(&self) -> usize {
        self.get_d_model()
    }
}

impl BertEncoder for deberta_v3::DebertaV3Encoder {
    fn forward(&self, ids: &[u32]) -> Vec<f32> {
        deberta_v3::DebertaV3Encoder::forward(self, ids)
    }
    fn d_model(&self) -> usize {
        self.get_d_model()
    }
}

/// Plain BERT encoder implements the uniform trait too — `forward(ids)`
/// runs with `token_type_ids = None` (single-segment). Consumers that
/// need multi-segment input call `bert_base::BertBaseEncoder::forward`
/// directly (WP-19 SBV2 wiring will do that).
impl BertEncoder for bert_base::BertBaseEncoder {
    fn forward(&self, ids: &[u32]) -> Vec<f32> {
        bert_base::BertBaseEncoder::forward(self, ids, None)
    }
    fn d_model(&self) -> usize {
        // Fully-qualified to avoid method-name shadowing between the
        // inherent `BertBaseEncoder::d_model` and this trait method.
        bert_base::BertBaseEncoder::d_model(self)
    }
}

pub mod bert_base;
pub mod deberta_v2;
pub mod deberta_v3;
pub mod tokenizer;
pub mod wordpiece;

#[cfg(test)]
mod tests {
    #[test]
    fn crate_compiles_and_trait_object_safe() {
        // Trait must exist and be nameable.
        let _: Option<Box<dyn super::BertEncoder>> = None;
    }
}
