//! openWakeWord classifier MLP primitive (SoTA plan KWS binder — 2026-08-05).
//!
//! openWakeWord (`dscripka/openWakeWord`, Apache-2.0 code) is a small
//! keyword-spotting family where each wake-word is a **tiny per-wake-word
//! MLP classifier** over a **shared 96-d speech embedding** produced from
//! a precomputed melspectrogram by the frozen Google `speech_embedding`
//! TFLite. The embedding is a 76-frame rolling window (~775 ms at 16 kHz)
//! collapsed to a single 96-d vector; the per-wake-word classifier is
//! (in the reference release) a `Linear(96 → hidden) → ReLU → Linear(hidden → 1)
//! → Sigmoid` that emits a wake-word probability in `[0, 1]`.
//!
//! # Scope of this module
//!
//! This module hosts the **numeric MLP forward** — synthesizable-weight
//! testable and independently unit-verifiable — for the classifier half
//! of the openWakeWord pipeline. The embedding extractor (the frozen
//! Google `speech_embedding` net) is intentionally NOT implemented here:
//! it is a loud-partial follow-up wave gated on the owner-provisioned
//! bundle (mirror of `crate::denoise::denoise` / `crate::f0::rmvpe`
//! loud-partial pattern; see `crates/vokra-models/src/kws/openwakeword`
//! for the runtime session and the `VOKRA_OPENWAKEWORD_REAL_GGUF` env
//! gate).
//!
//! # FR-EX-08 loud-fail contract
//!
//! Every shape mismatch — wrong embedding width, empty hidden layer, out
//! bias length ≠ 1, weight length ≠ `out × in` — is a hard error
//! ([`vokra_core::VokraError::InvalidArgument`]) naming the offending
//! dimension. No silent zero-pad, no silent truncation, no silent
//! sigmoid-of-zero on missing weights.

use vokra_core::{Result, VokraError};

/// One per-wake-word MLP classifier weight bundle (`Linear` → ReLU →
/// `Linear` → Sigmoid, where the final Sigmoid is applied by the
/// [`openwakeword_classifier_forward`] caller).
///
/// Every field is required and must be self-consistent: the runtime
/// binder ([`vokra_models::kws::openwakeword`]) validates the shapes at
/// GGUF load time via [`Self::validate`], and
/// [`openwakeword_classifier_forward`] re-validates at forward time so a
/// hand-built bundle in a downstream crate cannot silently misforward.
#[derive(Debug, Clone, PartialEq)]
pub struct OpenwakewordClassifierWeights {
    /// Embedding dimensionality (`96` in the reference release).
    pub embedding_dim: usize,
    /// Hidden layer width (per-wake-word; upstream defaults vary from
    /// `128` to `256`).
    pub hidden_dim: usize,
    /// First linear layer weight, row-major `[hidden_dim, embedding_dim]`.
    pub linear1_weight: Vec<f32>,
    /// First linear layer bias, `[hidden_dim]`.
    pub linear1_bias: Vec<f32>,
    /// Output linear layer weight, row-major `[1, hidden_dim]` (each
    /// wake-word is a binary classifier).
    pub linear2_weight: Vec<f32>,
    /// Output linear layer bias, `[1]`.
    pub linear2_bias: Vec<f32>,
}

impl OpenwakewordClassifierWeights {
    /// Validates the shape contract loudly (FR-EX-08).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on any dimension mismatch, empty
    /// hidden layer, or an out layer that is not a single-class binary
    /// classifier.
    pub fn validate(&self) -> Result<()> {
        if self.embedding_dim == 0 {
            return Err(VokraError::InvalidArgument(
                "openwakeword classifier: embedding_dim must be > 0 (got 0)".to_owned(),
            ));
        }
        if self.hidden_dim == 0 {
            return Err(VokraError::InvalidArgument(
                "openwakeword classifier: hidden_dim must be > 0 (got 0)".to_owned(),
            ));
        }
        let l1_expected = self.hidden_dim * self.embedding_dim;
        if self.linear1_weight.len() != l1_expected {
            return Err(VokraError::InvalidArgument(format!(
                "openwakeword classifier: linear1_weight has {} elements, expected {} \
                 (hidden_dim={} * embedding_dim={})",
                self.linear1_weight.len(),
                l1_expected,
                self.hidden_dim,
                self.embedding_dim
            )));
        }
        if self.linear1_bias.len() != self.hidden_dim {
            return Err(VokraError::InvalidArgument(format!(
                "openwakeword classifier: linear1_bias has {} elements, expected hidden_dim={}",
                self.linear1_bias.len(),
                self.hidden_dim
            )));
        }
        let l2_expected = self.hidden_dim;
        if self.linear2_weight.len() != l2_expected {
            return Err(VokraError::InvalidArgument(format!(
                "openwakeword classifier: linear2_weight has {} elements, expected {} \
                 (1 output class * hidden_dim={})",
                self.linear2_weight.len(),
                l2_expected,
                self.hidden_dim
            )));
        }
        if self.linear2_bias.len() != 1 {
            return Err(VokraError::InvalidArgument(format!(
                "openwakeword classifier: linear2_bias has {} elements, expected 1 (single \
                 binary output class per wake-word)",
                self.linear2_bias.len()
            )));
        }
        Ok(())
    }
}

/// Runs one classifier forward pass on a single embedding vector,
/// returning the sigmoid probability in `[0, 1]`.
///
/// Pipeline: `y = sigmoid(linear2_bias + linear2_weight ⋅ relu(linear1_bias
/// + linear1_weight ⋅ embedding))`.
///
/// # Errors
///
/// - [`VokraError::InvalidArgument`] if the embedding length does not
///   match `weights.embedding_dim`, or if [`OpenwakewordClassifierWeights::validate`]
///   rejects the bundle.
pub fn openwakeword_classifier_forward(
    weights: &OpenwakewordClassifierWeights,
    embedding: &[f32],
) -> Result<f32> {
    weights.validate()?;
    if embedding.len() != weights.embedding_dim {
        return Err(VokraError::InvalidArgument(format!(
            "openwakeword classifier: embedding has {} elements, expected embedding_dim={}",
            embedding.len(),
            weights.embedding_dim
        )));
    }

    // Layer 1: hidden = ReLU(linear1_bias + linear1_weight @ embedding).
    let hidden_dim = weights.hidden_dim;
    let embedding_dim = weights.embedding_dim;
    let mut hidden = vec![0.0f32; hidden_dim];
    for (h, cell) in hidden.iter_mut().enumerate() {
        let row = &weights.linear1_weight[h * embedding_dim..(h + 1) * embedding_dim];
        let mut acc = weights.linear1_bias[h];
        for (w, x) in row.iter().zip(embedding.iter()) {
            acc += w * x;
        }
        // ReLU.
        *cell = if acc > 0.0 { acc } else { 0.0 };
    }

    // Layer 2: logit = linear2_bias + linear2_weight @ hidden.
    let mut logit = weights.linear2_bias[0];
    for (w, h) in weights.linear2_weight.iter().zip(hidden.iter()) {
        logit += w * h;
    }

    // Sigmoid — numerically-stable form via tanh (avoids overflow in
    // `exp(-x)` for large-magnitude logits).
    Ok(0.5 * (0.5 * logit).tanh() + 0.5)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A canonical tiny bundle used across the shape / forward tests
    /// (embedding_dim=4, hidden_dim=3). All weights below produce a
    /// mathematically-tractable forward the tests can assert on directly.
    fn tiny_bundle() -> OpenwakewordClassifierWeights {
        OpenwakewordClassifierWeights {
            embedding_dim: 4,
            hidden_dim: 3,
            // Identity-ish layer 1: row 0 sums (positive inputs); rows
            // 1/2 subtract them (so ReLU zeros them for positive input).
            #[rustfmt::skip]
            linear1_weight: vec![
                 1.0,  1.0,  1.0,  1.0, // row 0: acc = sum(x)
                -1.0, -1.0, -1.0, -1.0, // row 1: acc = -sum(x) → 0 after ReLU
                -1.0, -1.0, -1.0, -1.0, // row 2: same
            ],
            linear1_bias: vec![0.0, 0.0, 0.0],
            // Layer 2 picks up row 0 only.
            linear2_weight: vec![1.0, 0.0, 0.0],
            linear2_bias: vec![0.0],
        }
    }

    #[test]
    fn forward_returns_sigmoid_of_positive_sum() {
        let w = tiny_bundle();
        // x = [1, 1, 1, 1] → sum = 4 → hidden[0] = 4, hidden[1..] = 0 →
        // logit = 4 → probability = sigmoid(4) ≈ 0.9820.
        let p = openwakeword_classifier_forward(&w, &[1.0, 1.0, 1.0, 1.0]).unwrap();
        let expected = 1.0f32 / (1.0 + (-4.0f32).exp());
        assert!(
            (p - expected).abs() < 1e-6,
            "expected sigmoid(4) = {expected}, got {p}"
        );
    }

    #[test]
    fn forward_probability_stays_in_unit_interval() {
        let w = tiny_bundle();
        // Extreme positive and negative inputs to exercise numeric
        // stability of the sigmoid.
        for magnitude in [-1000.0f32, -10.0, 0.0, 10.0, 1000.0] {
            let x = vec![magnitude; w.embedding_dim];
            let p = openwakeword_classifier_forward(&w, &x).unwrap();
            assert!(
                p.is_finite(),
                "probability must be finite (magnitude={magnitude}, got {p})"
            );
            assert!(
                (0.0..=1.0).contains(&p),
                "sigmoid must live in [0, 1] (magnitude={magnitude}, got {p})"
            );
        }
    }

    #[test]
    fn forward_rejects_wrong_embedding_length_loudly() {
        let w = tiny_bundle();
        let err = openwakeword_classifier_forward(&w, &[1.0, 1.0])
            .expect_err("embedding_dim mismatch must be a loud error (FR-EX-08)");
        let msg = err.to_string();
        assert!(
            msg.contains("embedding"),
            "error message must mention embedding: {msg}"
        );
        assert!(
            msg.contains("2"),
            "error message must mention actual length 2: {msg}"
        );
        assert!(
            msg.contains("4"),
            "error message must mention expected 4: {msg}"
        );
    }

    #[test]
    fn validate_rejects_zero_embedding_dim() {
        let mut w = tiny_bundle();
        w.embedding_dim = 0;
        assert!(matches!(w.validate(), Err(VokraError::InvalidArgument(_))));
    }

    #[test]
    fn validate_rejects_zero_hidden_dim() {
        let mut w = tiny_bundle();
        w.hidden_dim = 0;
        assert!(matches!(w.validate(), Err(VokraError::InvalidArgument(_))));
    }

    #[test]
    fn validate_rejects_wrong_linear1_shape() {
        let mut w = tiny_bundle();
        w.linear1_weight.pop();
        let err = w
            .validate()
            .expect_err("shape mismatch must be a loud error");
        assert!(err.to_string().contains("linear1_weight"));
    }

    #[test]
    fn validate_rejects_multi_class_out_layer() {
        // openwakeword's per-wake-word classifier is binary (single output
        // class). A `[2]` out bias is an architectural mismatch that must
        // be refused loudly rather than silently returning the first row.
        let mut w = tiny_bundle();
        w.linear2_bias = vec![0.0, 0.0];
        let err = w
            .validate()
            .expect_err("multi-class out layer must be a loud error");
        assert!(err.to_string().contains("linear2_bias"));
    }

    #[test]
    fn forward_negative_sum_relu_masks_to_bias_only() {
        let w = tiny_bundle();
        // x = [-1, -1, -1, -1] → sum = -4 → hidden[0] = 0 after ReLU →
        // logit = 0 → probability = sigmoid(0) = 0.5.
        let p = openwakeword_classifier_forward(&w, &[-1.0, -1.0, -1.0, -1.0]).unwrap();
        assert!(
            (p - 0.5).abs() < 1e-6,
            "ReLU-masked forward must return sigmoid(0) = 0.5, got {p}"
        );
    }
}
