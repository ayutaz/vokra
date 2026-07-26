//! # SBV2 (Style-Bert-VITS2 v2) native TTS.
//!
//! Clean-room Apache-2.0 implementation of Style-Bert-VITS2 v2 inference,
//! per the design doc `docs/superpowers/specs/2026-07-26-sbv2-v2-design.md`.
//!
//! # References (permissive only)
//!
//! - VITS paper: arXiv:2106.06103 (Kim et al. 2021)
//! - jaywalnut310/vits (MIT): VITS core reference
//! - VITS2 paper: arXiv:2307.16430
//! - p0p4k/vits2_pytorch (MIT): VITS2 code reference
//! - DeBERTa v2 paper: arXiv:2006.03654
//! - DeBERTa v3 paper: arXiv:2111.09543
//! - HF transformers deberta_v2/v3 (Apache-2.0): BERT reference
//!
//! # NOT REFERENCED
//!
//! - github.com/litagin02/Style-Bert-VITS2 (AGPL-3.0)
//! - github.com/fishaudio/Bert-VITS2 (AGPL-3.0)
//! - Any AGPL derivative of the above.

pub mod g2p;
pub mod style;
pub mod text_encoder;
// Later tasks add: mod duration; mod flow; mod decoder;
//                  mod converter; mod parity;

pub use g2p::{Language, PhonemizeResult, SbV2Phonemizer};
pub use style::StyleVectorInjector;
pub use text_encoder::{BertBridge, SbV2TextEncoder};
