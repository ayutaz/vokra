//! Adapter binding the Whisper decoder to the model-independent
//! [`BeamScorer`](vokra_core::decode::BeamScorer) (M0-06-T23).
//!
//! [`WhisperBeamScorer`] answers the search's only question — "log-probs of the
//! next token given this prefix" — by running the decoder over the full prefix
//! and `log_softmax`-ing the final logits.
//!
//! # Per-beam KV cache (M0: recompute)
//!
//! The search hands each beam its full token sequence, so this scorer
//! **recomputes** from a reset cache every call: correctness-first, no cache
//! aliasing between beams. Efficient per-beam cache reuse / reordering
//! (FR-EX-02 / M1-04) is a later optimization behind the same interface.

use vokra_core::Result;
use vokra_core::decode::BeamScorer;

use super::config::WhisperConfig;
use super::decoder::DecoderState;
use super::encoder::EncoderOutput;
use super::weights::DecoderWeights;

/// [`BeamScorer`] over a Whisper decoder bound to one encoder output.
pub struct WhisperBeamScorer<'a> {
    state: DecoderState<'a>,
    vocab: usize,
}

impl<'a> WhisperBeamScorer<'a> {
    /// Builds a scorer for `encoder`'s audio (precomputes cross-attention K/V).
    pub(crate) fn new(
        cfg: &'a WhisperConfig,
        w: &'a DecoderWeights,
        encoder: &EncoderOutput,
    ) -> Result<Self> {
        Ok(Self {
            state: DecoderState::new(cfg, w, encoder)?,
            vocab: cfg.n_vocab,
        })
    }
}

impl BeamScorer for WhisperBeamScorer<'_> {
    fn logprobs(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        self.state.reset();
        let logits = self.state.step_last(tokens)?;
        Ok(log_softmax(&logits))
    }

    fn vocab_size(&self) -> usize {
        self.vocab
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

    #[test]
    fn log_softmax_sums_to_one_in_prob_space() {
        let lp = log_softmax(&[1.0, 2.0, 3.0, -1.0]);
        let s: f64 = lp.iter().map(|&x| (x as f64).exp()).sum();
        assert!((s - 1.0).abs() < 1e-6, "sum {s}");
        // Monotonic: larger logit → larger log-prob.
        assert!(lp[2] > lp[1] && lp[1] > lp[0] && lp[0] > lp[3]);
    }
}
