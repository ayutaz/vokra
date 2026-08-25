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

pub mod backend;
pub mod bert_base;
pub mod deberta_v2;
pub mod deberta_v3;
pub mod tokenizer;
pub mod wordpiece;

#[cfg(test)]
mod tests {
    use super::backend::TestBackend;

    fn assert_close(expected: &[f32], actual: &[f32], bound: f32) {
        assert_eq!(actual.len(), expected.len());
        let max_abs = actual
            .iter()
            .zip(expected)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0, f32::max);
        assert!(
            max_abs <= bound,
            "backend seam max|delta|={max_abs:.9e} exceeds {bound:.9e}"
        );
    }

    #[test]
    fn crate_compiles_and_trait_object_safe() {
        // Trait must exist and be nameable.
        let _: Option<Box<dyn super::BertEncoder>> = None;
    }

    #[test]
    fn plain_bert_backend_seam_matches_scalar_forward() {
        let config = super::bert_base::BertConfig {
            vocab_size: 16,
            hidden_size: 8,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            intermediate_size: 32,
            max_position_embeddings: 16,
            type_vocab_size: 2,
            layer_norm_eps: 1e-12,
        };
        let encoder = super::bert_base::BertBaseEncoder::synthetic_for_test(&config);
        let ids = [1, 2, 3, 4];
        let expected = encoder.forward(&ids, None);
        let actual = encoder
            .forward_with_backend(&TestBackend, &ids, None)
            .expect("backend forward");
        assert_close(&expected, &actual, 1e-5);
    }

    #[test]
    fn deberta_v2_backend_seam_matches_scalar_forward() {
        let encoder = super::deberta_v2::DebertaV2Encoder::synthetic_for_test(2, 8, 2, 16, 8);
        let ids = [1, 2, 3, 4];
        let expected = encoder.forward(&ids);
        let actual = encoder
            .forward_with_backend(&TestBackend, &ids)
            .expect("backend forward");
        assert_close(&expected, &actual, 1e-5);
    }

    #[test]
    fn deberta_v3_backend_seam_matches_scalar_forward() {
        let encoder = super::deberta_v3::DebertaV3Encoder::synthetic_for_test(2, 8, 2, 16, 8);
        let ids = [1, 2, 3, 4];
        let expected = encoder.forward(&ids);
        let actual = encoder
            .forward_with_backend(&TestBackend, &ids)
            .expect("backend forward");
        assert_close(&expected, &actual, 1e-5);
    }
}
