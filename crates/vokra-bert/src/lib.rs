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

/// Placeholder trait — filled in by later tasks (Task 12).
pub trait BertEncoder {}

pub mod deberta_v2;
pub mod tokenizer;

#[cfg(test)]
mod tests {
    #[test]
    fn crate_compiles_and_trait_object_safe() {
        // Trait must exist and be nameable.
        let _: Option<Box<dyn super::BertEncoder>> = None;
    }
}
