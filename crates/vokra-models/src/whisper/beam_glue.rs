//! Adapters binding the Whisper decoder to the model-independent decoding
//! traits ([`LogitsSource`] / [`BeamScorer`], M0-06-T23 + M1-04).
//!
//! [`WhisperLogitsSource`] answers the raw-logits question — "logits of the next
//! token given this prefix" — by running the decoder over the full prefix. It is
//! the primitive both the [`Sampler`](vokra_core::decode::Sampler) and beam
//! search consume; [`WhisperBeamScorer`] is a **thin adapter** layering the
//! `log_softmax` that [`beam_search`](vokra_core::decode::beam_search) wants on
//! top of it.
//!
//! # Per-beam KV cache (M0: recompute)
//!
//! Each query hands the full token sequence, so the source **recomputes** from a
//! reset cache every call: correctness-first, no cache aliasing between beams.
//! Efficient per-beam cache reuse / reordering (FR-EX-02 / M1-04) is a later
//! optimization behind the same interface.

use std::sync::Arc;

use vokra_core::Result;
use vokra_core::decode::{BeamScorer, LogitsSource};

use super::WhisperModel;
use super::decoder::DecoderState;
use super::encoder::EncoderOutput;

/// [`LogitsSource`] over a Whisper decoder bound to one encoder output.
///
/// Owns its [`DecoderState`] (which owns the model via an [`Arc`]), so the
/// source carries no lifetime and can drive greedy, sampled or beam decoding.
pub struct WhisperLogitsSource {
    state: DecoderState,
    vocab: usize,
}

impl WhisperLogitsSource {
    /// Builds a source for `encoder`'s audio (precomputes cross-attention K/V).
    pub(crate) fn new(model: Arc<WhisperModel>, encoder: &EncoderOutput) -> Result<Self> {
        let vocab = model.config().n_vocab;
        let state = model.decoder(encoder)?;
        Ok(Self { state, vocab })
    }
}

impl LogitsSource for WhisperLogitsSource {
    fn logits(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        self.state.reset();
        self.state.step_last(tokens)
    }

    fn vocab_size(&self) -> usize {
        self.vocab
    }
}

/// [`BeamScorer`] over a Whisper decoder: a thin `log_softmax` adapter on top of
/// [`WhisperLogitsSource`].
pub struct WhisperBeamScorer {
    source: WhisperLogitsSource,
}

impl WhisperBeamScorer {
    /// Builds a scorer for `encoder`'s audio (precomputes cross-attention K/V).
    pub(crate) fn new(model: Arc<WhisperModel>, encoder: &EncoderOutput) -> Result<Self> {
        Ok(Self {
            source: WhisperLogitsSource::new(model, encoder)?,
        })
    }
}

impl BeamScorer for WhisperBeamScorer {
    fn logprobs(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        Ok(log_softmax(&self.source.logits(tokens)?))
    }

    fn vocab_size(&self) -> usize {
        self.source.vocab_size()
    }
}

/// Numerically stable `log_softmax` over a logits vector.
fn log_softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f64;
    for &l in logits {
        sum += ((l - max) as f64).exp();
    }
    let log_sum = max as f64 + sum.ln();
    logits
        .iter()
        .map(|&l| (l as f64 - log_sum) as f32)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use vokra_core::decode::{SamplerConfig, sample_sequence};

    use crate::whisper::decoder::test_support::{tiny_encoder, tiny_model};
    use crate::whisper::greedy::greedy_decode;

    #[test]
    fn log_softmax_sums_to_one_in_prob_space() {
        let lp = log_softmax(&[1.0, 2.0, 3.0, -1.0]);
        let s: f64 = lp.iter().map(|&x| (x as f64).exp()).sum();
        assert!((s - 1.0).abs() < 1e-6, "sum {s}");
        // Monotonic: larger logit → larger log-prob.
        assert!(lp[2] > lp[1] && lp[1] > lp[0] && lp[0] > lp[3]);
    }

    /// Temperature-0 sampling through the [`WhisperLogitsSource`] must reproduce
    /// the incremental greedy decoder token-for-token. This is the CI-runnable
    /// oracle for the sampled-transcribe wiring: the recompute-per-step source
    /// (reset + `step_last` on the full prefix) and the incremental greedy loop
    /// agree because reset+replay is bit-identical to incremental decoding.
    #[test]
    fn greedy_sampling_over_logits_source_matches_greedy_decode() {
        let model = tiny_model(2);
        let enc = tiny_encoder(model.config().d_model, 4);
        let start = model.config().decoder_start_ids.clone();
        let eot = model.config().eot;

        let mut st = model.decoder(&enc).unwrap();
        let greedy = greedy_decode(&mut st, &start, eot, 6).unwrap();

        let mut src = WhisperLogitsSource::new(Arc::clone(&model), &enc).unwrap();
        let sampled = sample_sequence(&mut src, &start, eot, &SamplerConfig::greedy(), 6).unwrap();

        assert_eq!(
            greedy, sampled,
            "temperature-0 sampling must equal greedy decode"
        );
    }
}
