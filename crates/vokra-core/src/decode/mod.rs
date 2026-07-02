//! Host-side decoding / search algorithms (model-independent).
//!
//! Per FR-OP-40, search operators such as `beam_search` are **host-side
//! runtime functions**, not ops embedded in the model graph (which would
//! break execution-provider compatibility, the "contrib op" anti-pattern).
//! They live here in `vokra-core` (the IR / execution-engine crate) so any
//! model can drive them through a model-independent scorer abstraction.
//!
//! # Scope
//!
//! - **M0-06** adds [`beam_search`] (FR-OP-40: beam width, length
//!   normalization, early stopping, n-best, word-level timestamps) with a
//!   [`BeamScorer`] trait so Whisper (and later models) plug in without this
//!   module knowing anything model-specific.
//! - `ctc_decode` / `rnnt_decode` / `wfst_decode` (FR-OP-41..43) are later
//!   milestones.

pub mod beam_search;

pub use beam_search::{BeamHypothesis, BeamScorer, BeamSearchConfig, beam_search};
