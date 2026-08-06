//! Terminal classifier + Softmax for `PyanNet`.
//!
//! # Primary source
//!
//! `PyanNet.build()` (PyanNet.py:158-160):
//!
//! ```python
//! self.classifier = nn.Linear(in_features, self.dimension)
//! ```
//!
//! where `in_features = linear.hidden_size` (128 for pyannote-3.0
//! defaults) and `self.dimension = num_powerset_classes` (7).
//!
//! `PyanNet.forward()` (PyanNet.py:233-235):
//!
//! ```python
//! outputs = self.classifier(outputs)
//! return self.activation(outputs)
//! ```
//!
//! For the powerset multi-class problem `SpeakerDiarization.default_
//! activation()` returns `nn.LogSoftmax(dim=-1)` — but the runtime
//! consumer (the powerset argmax decoder) is invariant to the
//! monotonic `log`, so Vokra returns per-frame **probabilities**
//! (numerically stable softmax) directly. This lets the caller compare
//! magnitudes without an implicit `exp()`; the argmax is identical to
//! the argmax over `LogSoftmax` output by construction (`log` is
//! monotonic).
//!
//! # Zero-dep invariant (NFR-DS-02)
//!
//! Pure Rust — matmul + numerically stable softmax (max-subtract,
//! then exp + normalise). No BLAS, no crates.io.

use vokra_core::{Result, VokraError};

use super::PyanNetWeights;

/// Terminal `Linear(in → num_classes)` + numerically stable Softmax.
#[derive(Debug)]
pub(crate) struct Classifier {
    in_dim: usize,
    num_classes: usize,
    weight: Vec<f32>, // row-major [num_classes, in_dim]
    bias: Vec<f32>,   // [num_classes]
}

impl Classifier {
    /// Binds the classifier from the `classifier.*` prefix.
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] with FR-EX-08 if a tensor is
    ///   missing or mis-shaped.
    pub fn from_weights(w: &PyanNetWeights, in_dim: usize, num_classes: usize) -> Result<Self> {
        let weight = bind_tensor(w, "classifier.weight", &[num_classes, in_dim])?;
        let bias = bind_tensor(w, "classifier.bias", &[num_classes])?;
        Ok(Self {
            in_dim,
            num_classes,
            weight,
            bias,
        })
    }

    /// Forward on `[seq_len · in_dim]` row-major → `[seq_len ·
    /// num_classes]` row-major softmax probabilities that sum to ~1
    /// per row.
    ///
    /// Numerically stable softmax: `p_i = exp(x_i - max_j x_j) /
    /// sum_k exp(x_k - max_j x_j)`. The max-subtract avoids overflow
    /// when the pre-activation logits are large positive.
    pub fn forward(&self, input: &[f32], seq_len: usize) -> Vec<f32> {
        debug_assert_eq!(input.len(), seq_len * self.in_dim);
        let mut out = vec![0.0f32; seq_len * self.num_classes];
        for t in 0..seq_len {
            let x = &input[t * self.in_dim..(t + 1) * self.in_dim];
            let y = &mut out[t * self.num_classes..(t + 1) * self.num_classes];
            // 1) Compute logits `y = W·x + b`. We use `enumerate` +
            // `iter_mut` (instead of `for i in 0..num_classes`) so the
            // `needless_range_loop` clippy lint is happy while keeping
            // both the class index (for row slicing) and the mutable
            // slot.
            for (i, y_i) in y.iter_mut().enumerate() {
                let row = &self.weight[i * self.in_dim..(i + 1) * self.in_dim];
                let mut acc = self.bias[i];
                for j in 0..self.in_dim {
                    acc += row[j] * x[j];
                }
                *y_i = acc;
            }
            // 2) Numerically stable softmax.
            let mut max_logit = f32::NEG_INFINITY;
            for &v in y.iter() {
                if v > max_logit {
                    max_logit = v;
                }
            }
            let mut sum = 0.0f32;
            for v in y.iter_mut() {
                *v = (*v - max_logit).exp();
                sum += *v;
            }
            if sum > 0.0 {
                let inv = 1.0 / sum;
                for v in y.iter_mut() {
                    *v *= inv;
                }
            }
        }
        out
    }
}

fn bind_tensor(w: &PyanNetWeights, name: &str, expect_shape: &[usize]) -> Result<Vec<f32>> {
    let (dims, payload) = w.tensor(name).ok_or_else(|| {
        VokraError::ModelLoad(format!(
            "pyannote-segmentation Classifier: required tensor `{name}` is missing from the \
             GGUF (FR-EX-08). Expected shape {expect_shape:?}."
        ))
    })?;
    if dims != expect_shape {
        return Err(VokraError::ModelLoad(format!(
            "pyannote-segmentation Classifier: tensor `{name}` has shape {dims:?}, expected \
             {expect_shape:?} (FR-EX-08)"
        )));
    }
    let expect_elems: usize = expect_shape.iter().product();
    if payload.len() != expect_elems {
        return Err(VokraError::ModelLoad(format!(
            "pyannote-segmentation Classifier: tensor `{name}` element count {} != expected \
             {expect_elems} (FR-EX-08)",
            payload.len()
        )));
    }
    Ok(payload.to_vec())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifier_softmax_rows_sum_to_one() {
        // 3-class classifier with identity-ish weights so probs are
        // interpretable.
        let cls = Classifier {
            in_dim: 3,
            num_classes: 3,
            weight: vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            bias: vec![0.0, 0.0, 0.0],
        };
        let input = vec![2.0, 1.0, 0.0, -1.0, -2.0, -3.0];
        let out = cls.forward(&input, 2);
        assert_eq!(out.len(), 6);
        // Every row should sum to ~1.
        for t in 0..2 {
            let sum: f32 = out[t * 3..(t + 1) * 3].iter().sum();
            assert!((sum - 1.0).abs() < 1e-5, "row {t} sum={sum}, expected 1.0");
        }
        // Frame 0: input=[2,1,0], identity weight, so class 0 should
        // win.
        let (max_idx, _) =
            out[0..3]
                .iter()
                .enumerate()
                .fold((0, f32::NEG_INFINITY), |(mi, mv), (i, &v)| {
                    if v > mv { (i, v) } else { (mi, mv) }
                });
        assert_eq!(max_idx, 0);
    }

    #[test]
    fn classifier_softmax_all_positive_finite_values() {
        let cls = Classifier {
            in_dim: 2,
            num_classes: 4,
            weight: vec![1.0, 2.0, -1.0, -2.0, 0.5, -0.5, 3.0, -3.0],
            bias: vec![0.0, 0.0, 0.0, 0.0],
        };
        let input = vec![1.0, 1.0, -1.0, -1.0];
        let out = cls.forward(&input, 2);
        for &v in &out {
            assert!((0.0..=1.0).contains(&v), "softmax prob out of range: {v}");
            assert!(v.is_finite(), "non-finite: {v}");
        }
    }

    #[test]
    fn classifier_softmax_handles_large_positive_logits_without_overflow() {
        // Without max-subtract, exp(1000) overflows to +inf. The
        // numerically stable path must produce a finite probability
        // distribution.
        let cls = Classifier {
            in_dim: 1,
            num_classes: 2,
            weight: vec![1000.0, 999.0],
            bias: vec![0.0, 0.0],
        };
        let input = vec![1.0];
        let out = cls.forward(&input, 1);
        assert_eq!(out.len(), 2);
        let sum: f32 = out.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum={sum} not 1.0 (overflow?)");
        // Class 0 should dominate (logit 1000 > 999).
        assert!(out[0] > out[1]);
    }
}
