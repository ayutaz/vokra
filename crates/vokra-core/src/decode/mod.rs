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
//! - **M1-08f** adds [`DecodeStepper`], the cache-based incremental decode
//!   stepper (the FR-ST-01 cache pattern) over the same [`LogitsSource`].
//! - `ctc_decode` / `rnnt_decode` / `wfst_decode` (FR-OP-41..43) are later
//!   milestones.

pub mod beam_search;
pub mod cfg;
pub mod sampler;
pub mod stepper;
// M4-20 (a): host-side Whisper cross-attention DTW word-timestamp alignment
// (FR-OP-40 `word_timestamps`). Model-independent — the Whisper decoder only
// supplies the cross-attention weights (ADR M4-20 §D-2/§D-3).
pub mod word_timing;

use crate::error::Result;

pub use beam_search::{BeamHypothesis, BeamScorer, BeamSearchConfig, beam_search};
pub use cfg::{CfgMode, apply_cfg, apply_cfg_inplace};
pub use sampler::{Sampler, SamplerConfig, argmax, sample_sequence};
pub use stepper::{DecodeStepper, TOKEN_FLAG_EOT};
pub use word_timing::{
    APPEND_PUNCTUATIONS, AlignmentParams, CrossAttention, PREPEND_PUNCTUATIONS, WordTiming,
    merge_punctuations, token_alignment, words_from_alignment,
};

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

    /// Batched [`logits`](Self::logits): one logits vector per prefix, in the
    /// **same order** as `prefixes` (M5-14-BACKLOG-T07).
    ///
    /// The default implementation loops [`logits`](Self::logits) in order, so
    /// it is byte-for-byte identical to calling `logits` once per prefix — a
    /// model that has no batched forward keeps its exact behaviour. A model
    /// with a batched decoder step (e.g. Whisper folding the `beam_width`
    /// per-beam projections into one m = `beam_width` GEMM) overrides this; the
    /// override **must** return the same bits as the per-prefix loop (the
    /// packed-GEMM parity invariant makes an m = N GEMM bit-identical to N
    /// m = 1 GEMMs row-for-row), which its parity oracle pins.
    fn logits_batch(&mut self, prefixes: &[&[u32]]) -> Result<Vec<Vec<f32>>> {
        prefixes.iter().map(|p| self.logits(p)).collect()
    }
}
