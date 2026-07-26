//! SBV2 stochastic duration predictor (SDP): flow-based per-phoneme
//! duration sampling with additive JP pitch-accent tone conditioning.
//! (Clean-room comment: see `mod.rs` — the flow-based SDP structure below
//! follows the generic affine-coupling normalizing-flow construction of
//! RealNVP (arXiv:1605.08803, Dinh, Sohl-Dickstein & Bengio 2016); no
//! SBV2/BV2 source referenced.)
//!
//! `CouplingLayer` reuse decision (Task 19): **(B) new local
//! [`SbV2CouplingLayer`]**, not a `piper_plus` reuse — `piper_plus`'s only
//! flow-coupling type, `duration::ConvFlow`
//! (`crates/vokra-models/src/piper_plus/duration.rs:138`), has **no
//! visibility modifier at all** (module-private to `piper_plus::duration`,
//! not even `pub(super)` like its owner `DurationPredictor`), so it is
//! fully unreachable from `sbv2`. Its `reverse` method also takes a
//! `&Compute` handle, a `[2, T]`-shaped whole-sequence buffer (the classic
//! RealNVP-style 2-channel coupling: channel 0 fixed, channel 1
//! transformed by a rational-quadratic spline whose params a `DDSConv`
//! stack predicts from channel 0 + conditioning), and owns nine per-layer
//! weight tensors loaded from a `TensorStore` — architecturally heavier
//! than this task needs and not reachable regardless. See
//! `text_encoder.rs`'s module doc for the identical reasoning applied to
//! `TransformerBlock` (Task 17).
//!
//! [`SbV2CouplingLayer`] here is a **scalar** affine coupling (not
//! 2-channel spline): each phoneme position's duration latent is a single
//! scalar `x` (there is no "other half" of a per-position pair to split
//! off, the way piper's `[2, T]` buffer has channel 0), so the "coupling"
//! is between `x` and the external conditioning vector `cond`
//! (tone+hidden, see below) rather than between two slices of the same
//! buffer: `inverse(x, cond) = (x - shift(cond)) *
//! exp(-log_scale(cond))`, the exact inverse of the canonical
//! affine-coupling forward `y = x * exp(log_scale(cond)) + shift(cond)`.
//! `log_scale`/`shift` are each one affine (bias + linear) readout of
//! `cond`. Only `inverse` is implemented — [`SbV2SDP::sample`] below only
//! ever walks the reverse/inference direction, matching
//! `piper_plus::duration::ConvFlow`, which likewise implements only
//! `reverse`, never a forward.
//!
//! # Tone conditioning (JP pitch accent)
//!
//! `tone_proj` is a `[n_tones, d_hidden]` row-major **embedding table**
//! (interpretation (b) of the task brief, matching
//! [`SbV2TextEncoder`](super::text_encoder::SbV2TextEncoder)'s
//! `tone_embed` embedding-table convention rather than a `[d_hidden,
//! n_tones]` one-hot-projection linear layer):
//! `tone_proj[tone as usize * d_hidden .. (tone as usize + 1) * d_hidden]`
//! is tone `tone`'s `[d_hidden]` additive delta. `tone_bias` (`[d_hidden]`)
//! is a single global bias added at every position regardless of tone. Per
//! phoneme position `p`: `cond[p, d] = hidden[p, d] + tone_proj[tones[p],
//! d] + tone_bias[d]`.
//!
//! # Layout convention
//!
//! `hidden` is flat, row-major, position-major `[text_seq_len, d_hidden]`
//! (see `text_encoder.rs`'s identical convention) — position `p`'s
//! `d_hidden`-wide row is `hidden[p * d_hidden .. (p + 1) * d_hidden]`.

use vokra_core::rng::GaussianSplitMix64;

/// A single scalar affine flow-coupling layer — see the module doc's
/// `CouplingLayer` reuse decision for why this is a fresh, minimal type
/// rather than a `piper_plus` reuse, and for the exact inverse formula.
pub struct SbV2CouplingLayer {
    /// Row-major `[2, d_hidden]`: row 0 projects `cond` to `log_scale`, row
    /// 1 projects `cond` to `shift` (both before adding `proj_bias`).
    proj_weight: Vec<f32>,
    /// `[2]`: `[log_scale_bias, shift_bias]`, added after the
    /// `proj_weight` dot product.
    proj_bias: Vec<f32>,
    /// Conditioning-vector width this layer expects (`cond.len()`).
    d_hidden: usize,
}

impl SbV2CouplingLayer {
    /// Builds a coupling layer from a pre-trained `[2, d_hidden]`
    /// `(log_scale, shift)` projection. Crate-internal: no caller
    /// constructs a non-empty `flow_layers` stack yet — the future
    /// `converter` module (see `mod.rs`'s roadmap comment) loads real
    /// GGUF weights and will call this.
    ///
    /// # Panics
    ///
    /// Panics (via `debug_assert!`, hot inner-loop constructor per this
    /// crate's established convention — see
    /// [`StyleVectorInjector::from_projections`](super::style::StyleVectorInjector::from_projections)'s
    /// panic docs) if `proj_weight.len() != 2 * d_hidden` or
    /// `proj_bias.len() != 2`.
    #[allow(dead_code)] // constructed by the future converter once real GGUF-loaded SDP weights are wired
    pub(crate) fn new(proj_weight: Vec<f32>, proj_bias: Vec<f32>, d_hidden: usize) -> Self {
        debug_assert_eq!(
            proj_weight.len(),
            2 * d_hidden,
            "proj_weight must be [2, d_hidden]"
        );
        debug_assert_eq!(proj_bias.len(), 2, "proj_bias must be [2]");
        Self {
            proj_weight,
            proj_bias,
            d_hidden,
        }
    }

    /// Inverse affine-coupling transform: `(x - shift(cond)) *
    /// exp(-log_scale(cond))` — see the module doc for the full
    /// derivation.
    ///
    /// # Panics
    ///
    /// Panics (via `debug_assert!`) if `cond.len() != self.d_hidden`.
    fn inverse(&self, x: f32, cond: &[f32]) -> f32 {
        debug_assert_eq!(cond.len(), self.d_hidden, "cond must be [d_hidden]");
        let d = self.d_hidden;
        let log_scale = self.proj_bias[0] + dot(&self.proj_weight[..d], cond);
        let shift = self.proj_bias[1] + dot(&self.proj_weight[d..2 * d], cond);
        (x - shift) * (-log_scale).exp()
    }
}

/// Dot product of two equal-length slices.
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// SBV2's stochastic duration predictor: samples one non-negative integer
/// duration per phoneme position by walking a stack of
/// [`SbV2CouplingLayer`]s in reverse (inference direction) from Gaussian
/// noise, conditioned on the text hidden state additively combined with a
/// per-position JP pitch-accent tone (see the module doc's "Tone
/// conditioning" section for the exact formula).
pub struct SbV2SDP {
    /// Flow-coupling stack, applied in **reverse** order (last layer
    /// first) — the standard normalizing-flow inference/sampling
    /// direction. An empty stack is a legitimate, exercised no-op
    /// configuration (see this crate's `tests/sbv2_duration.rs`): the
    /// sampled latent `z` passes through unchanged, so `x = z` (not a
    /// silent fallback — it is the documented behavior of a documented
    /// empty configuration, matching
    /// [`SbV2TextEncoder`](super::text_encoder::SbV2TextEncoder)'s empty
    /// `transformer_layers` precedent).
    flow_layers: Vec<SbV2CouplingLayer>,
    /// Tone embedding table, row-major `[n_tones, d_hidden]`
    /// (interpretation (b) — see the module doc's "Tone conditioning"
    /// section).
    tone_proj: Vec<f32>,
    /// Global additive bias, `[d_hidden]`, added at every position
    /// regardless of tone.
    tone_bias: Vec<f32>,
    /// Hidden (text encoder) dimension shared by `hidden`, `tone_proj`'s
    /// rows, `tone_bias`, and every [`SbV2CouplingLayer`]'s conditioning
    /// input.
    d_hidden: usize,
    /// Pitch-accent tone count (`tone_proj.len() == n_tones * d_hidden`).
    n_tones: usize,
}

impl SbV2SDP {
    /// Builds a predictor from a pre-trained flow-coupling stack and tone
    /// conditioning tables.
    ///
    /// # Panics
    ///
    /// Panics (via `debug_assert!`) if `tone_proj.len() != n_tones *
    /// d_hidden` or `tone_bias.len() != d_hidden`.
    pub fn from_weights(
        flow_layers: Vec<SbV2CouplingLayer>,
        tone_proj: Vec<f32>,
        tone_bias: Vec<f32>,
        d_hidden: usize,
        n_tones: usize,
    ) -> Self {
        debug_assert_eq!(
            tone_proj.len(),
            n_tones * d_hidden,
            "tone_proj must be [n_tones, d_hidden]"
        );
        debug_assert_eq!(tone_bias.len(), d_hidden, "tone_bias must be [d_hidden]");
        Self {
            flow_layers,
            tone_proj,
            tone_bias,
            d_hidden,
            n_tones,
        }
    }

    /// Samples one non-negative integer duration per phoneme position.
    ///
    /// For each position `p` in `0..text_seq_len`: draws `z =
    /// rng.next_gaussian() * noise_scale_w`, sets `x = z`, walks
    /// `flow_layers` in reverse applying [`SbV2CouplingLayer::inverse`]
    /// (conditioned on `hidden[p] + tone_proj[tones[p]] + tone_bias`, see
    /// the module doc), then returns `duration = ceil(exp(x)).max(1)` —
    /// the standard VITS-family log-duration convention (mirrors
    /// `piper_plus::duration::DurationPredictor`'s flow structure, whose
    /// caller likewise computes `logw.exp() * length_scale).ceil().max(1)`
    /// from the flow's output).
    ///
    /// `noise_scale_w` scales the Gaussian prior; `noise_scale_w == 0.0`
    /// makes every draw exactly `0.0` regardless of `rng`'s state, so an
    /// empty `flow_layers` stack combined with `noise_scale_w == 0.0`
    /// deterministically returns all-`1`s (`exp(0).ceil().max(1) == 1`).
    ///
    /// # Panics
    ///
    /// Panics (via `debug_assert!`) if `hidden.len() != text_seq_len *
    /// self.d_hidden`, if `tones.len() != text_seq_len`, or if any
    /// `tones` entry is `>= self.n_tones`.
    pub fn sample(
        &self,
        hidden: &[f32],
        tones: &[u8],
        text_seq_len: usize,
        rng: &mut GaussianSplitMix64,
        noise_scale_w: f32,
    ) -> Vec<i32> {
        debug_assert_eq!(
            hidden.len(),
            text_seq_len * self.d_hidden,
            "hidden must be [text_seq_len, d_hidden]"
        );
        debug_assert_eq!(
            tones.len(),
            text_seq_len,
            "tones must have text_seq_len entries"
        );
        debug_assert!(
            tones.iter().all(|&t| (t as usize) < self.n_tones),
            "tone out of range"
        );

        let d = self.d_hidden;
        let mut cond = vec![0.0_f32; d];
        let mut out = Vec::with_capacity(text_seq_len);
        for p in 0..text_seq_len {
            let hidden_row = &hidden[p * d..(p + 1) * d];
            let tone = tones[p] as usize;
            let tone_row = &self.tone_proj[tone * d..(tone + 1) * d];
            for i in 0..d {
                cond[i] = hidden_row[i] + tone_row[i] + self.tone_bias[i];
            }

            let mut x = rng.next_gaussian() * noise_scale_w;
            for layer in self.flow_layers.iter().rev() {
                x = layer.inverse(x, &cond);
            }
            let duration = x.exp().ceil().max(1.0) as i32;
            out.push(duration);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_weight_coupling_layer_inverse_is_identity() {
        let layer = SbV2CouplingLayer::new(vec![0.0; 8], vec![0.0; 2], 4);
        let cond = [0.3, -1.2, 5.0, 0.0];
        for &x in &[-2.5_f32, 0.0, 1.0, 7.25] {
            assert_eq!(
                layer.inverse(x, &cond),
                x,
                "zero-weight coupling layer must be the identity"
            );
        }
    }

    #[test]
    fn coupling_layer_inverse_matches_hand_computed_affine() {
        // d_hidden = 2. log_scale row = [1.0, 0.0], shift row = [0.0, 1.0],
        // bias = [0.0, 0.0]. cond = [2.0, 3.0].
        // log_scale = 1.0*2.0 + 0.0*3.0 = 2.0
        // shift     = 0.0*2.0 + 1.0*3.0 = 3.0
        // inverse(x=5.0) = (5.0 - 3.0) * exp(-2.0) = 2.0 * exp(-2.0)
        let layer = SbV2CouplingLayer::new(vec![1.0, 0.0, 0.0, 1.0], vec![0.0, 0.0], 2);
        let got = layer.inverse(5.0, &[2.0, 3.0]);
        let expected = 2.0_f32 * (-2.0_f32).exp();
        assert!(
            (got - expected).abs() < 1e-6,
            "got {got}, expected {expected}"
        );
    }

    #[test]
    fn sample_chains_through_nonempty_flow_stack() {
        // Two identity (zero-weight) coupling layers in a row must still
        // compose to the identity, exercising the reverse-order loop over
        // a *non-empty* `flow_layers` (the external integration tests in
        // `tests/sbv2_duration.rs` only cover the empty-stack no-op path).
        let d_hidden = 3;
        let n_tones = 2;
        let flow_layers = vec![
            SbV2CouplingLayer::new(vec![0.0; 2 * d_hidden], vec![0.0; 2], d_hidden),
            SbV2CouplingLayer::new(vec![0.0; 2 * d_hidden], vec![0.0; 2], d_hidden),
        ];
        let sdp = SbV2SDP::from_weights(
            flow_layers,
            vec![0.0; n_tones * d_hidden],
            vec![0.0; d_hidden],
            d_hidden,
            n_tones,
        );
        let text_seq_len = 4;
        let hidden = vec![0.0_f32; text_seq_len * d_hidden];
        let tones = vec![0u8; text_seq_len];
        let mut rng = GaussianSplitMix64::new(1);
        // noise_scale_w = 0.0 -> z = 0 at every position regardless of rng
        // state; identity flow layers leave x = 0 -> duration = 1.
        let out = sdp.sample(&hidden, &tones, text_seq_len, &mut rng, 0.0);
        assert_eq!(out, vec![1; text_seq_len]);
    }
}
