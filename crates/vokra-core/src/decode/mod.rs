//! Host-side decoding / search algorithms (model-independent).
//!
//! Per FR-OP-40, search operators such as `beam_search` are **host-side
//! runtime functions**, not ops embedded in the model graph (which would
//! break execution-provider compatibility, the "contrib op" anti-pattern).
//! They live here in `vokra-core` (the IR / execution-engine crate) so any
//! model can drive them through a model-independent scorer abstraction.
//!
//! # The model ↔ decoder contract
//!
//! Every decoder here asks a model exactly one thing — "given this token
//! prefix, what comes next?" — through one of two sibling traits:
//!
//! - [`LogitsSource`] returns the **raw logits** for the next token. It is the
//!   lower-level primitive, consumed by the [`Sampler`] (temperature / top-k /
//!   top-p / repetition penalty) and by [`sample_sequence`].
//! - [`BeamScorer`] returns normalized **log-probabilities**, consumed by
//!   [`beam_search`]. A concrete scorer is a thin adapter over a
//!   [`LogitsSource`] (a `log_softmax` of its logits).
//!
//! # Scope
//!
//! - **M0-06** adds [`beam_search`] (FR-OP-40: beam width, length
//!   normalization, early stopping, n-best, word-level timestamps).
//! - **M1-04** adds [`LogitsSource`], the [`Sampler`] / [`sample_sequence`]
//!   stochastic decoder, and the classifier-free-guidance combiner
//!   ([`apply_cfg`], [`CfgMode`], FR-EX-10).
//! - `ctc_decode` / `rnnt_decode` / `wfst_decode` (FR-OP-41..43) are later
//!   milestones.

pub mod beam_search;
pub mod cfg;
pub mod sampler;

use crate::error::Result;

pub use beam_search::{BeamHypothesis, BeamScorer, BeamSearchConfig, beam_search};
pub use cfg::{CfgMode, apply_cfg, apply_cfg_inplace};
pub use sampler::{Sampler, SamplerConfig, argmax, sample_sequence};

/// Raw next-token logits for a model, the low-level model ↔ decoder primitive.
///
/// A [`LogitsSource`] answers "given this full token prefix, what are the
/// unnormalized logits over the vocabulary for the next token?". It is
/// deliberately minimal and model-independent: the [`Sampler`] and
/// [`sample_sequence`] drive any model — Whisper today, others later — without
/// knowing anything about attention, KV caches or audio. A model may recompute
/// or use an internal cache; that does not change this interface.
pub trait LogitsSource {
    /// Returns the raw logits over the whole vocabulary for the token following
    /// `tokens` (the full sequence so far, including any forced prefix), length
    /// [`vocab_size`](Self::vocab_size).
    fn logits(&mut self, tokens: &[u32]) -> Result<Vec<f32>>;

    /// Vocabulary size (the length of every [`logits`](Self::logits) result).
    fn vocab_size(&self) -> usize;
}
