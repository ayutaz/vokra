//! Linear feed-forward stack for `PyanNet`.
//!
//! # Primary source
//!
//! `PyanNet.build()` (PyanNet.py:128-139) constructs
//!
//! ```python
//! self.linear = nn.ModuleList(
//!     [nn.Linear(in_features, out_features)
//!      for in_features, out_features in pairwise(
//!          [2*hidden_size] + [linear.hidden_size] * linear.num_layers
//!      )]
//! )
//! ```
//!
//! For `hidden_size=128` (BiLSTM) and `linear.hidden_size=128,
//! num_layers=2` this yields two `nn.Linear` layers:
//!
//! ```text
//! linear.0.weight (128, 256)  linear.0.bias (128,)
//! linear.1.weight (128, 128)  linear.1.bias (128,)
//! ```
//!
//! `PyanNet.forward()` (PyanNet.py:229-232) applies
//! `F.leaky_relu(linear[k](x))` after each layer with the default
//! slope `0.01`.
//!
//! # Zero-dep invariant (NFR-DS-02)
//!
//! Pure Rust — matmul via a scalar accumulator (LLVM auto-vectorises),
//! `leaky_relu` via a branch on the sign. No BLAS, no crates.io.

use vokra_core::{Result, VokraError};

use super::PyanNetWeights;

/// LeakyReLU slope used between and after every Linear layer
/// (PyTorch `F.leaky_relu` default slope 0.01 — the upstream
/// `PyanNet.forward()` call passes no explicit slope).
pub const LEAKY_RELU_SLOPE: f32 = 0.01;

/// A stack of `nn.Linear(in_dim → hidden_dim)` / (`nn.Linear(hidden_dim
/// → hidden_dim)`) layers with a `F.leaky_relu(0.01)` after each layer.
#[derive(Debug)]
pub(crate) struct LinearStack {
    layers: Vec<LinearLayer>,
}

#[derive(Debug)]
struct LinearLayer {
    in_dim: usize,
    out_dim: usize,
    weight: Vec<f32>, // row-major [out_dim, in_dim]
    bias: Vec<f32>,   // [out_dim]
}

impl LinearStack {
    /// Binds `num_layers` linear layers from the `linear.*` prefix of
    /// [`PyanNetWeights`]. The first layer's input dim is
    /// `input_dim` (usually `2 * lstm_hidden_size` = 256), every
    /// subsequent layer has input dim `hidden_dim` (usually 128).
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] with FR-EX-08 if any tensor is
    ///   missing, mis-shaped, or has the wrong element count.
    pub fn from_weights(
        w: &PyanNetWeights,
        input_dim: usize,
        hidden_dim: usize,
        num_layers: usize,
    ) -> Result<Self> {
        if num_layers == 0 {
            return Err(VokraError::ModelLoad(
                "pyannote-segmentation Linear stack: num_layers must be >= 1 (FR-EX-08)"
                    .to_string(),
            ));
        }
        let mut layers = Vec::with_capacity(num_layers);
        for k in 0..num_layers {
            let in_dim = if k == 0 { input_dim } else { hidden_dim };
            let out_dim = hidden_dim;
            let weight = bind_tensor(w, &format!("linear.{k}.weight"), &[out_dim, in_dim])?;
            let bias = bind_tensor(w, &format!("linear.{k}.bias"), &[out_dim])?;
            layers.push(LinearLayer {
                in_dim,
                out_dim,
                weight,
                bias,
            });
        }
        Ok(Self { layers })
    }

    /// Forward on `[seq_len · input_dim]` row-major → `[seq_len ·
    /// last_hidden_dim]` row-major. Every layer applies
    /// `y = W·x + b`, then in-place `F.leaky_relu(y, 0.01)`.
    ///
    /// This is a plain sequential matmul (batch-of-timesteps loop
    /// hoisted outside for cache friendliness). The row-major
    /// `[out_dim, in_dim]` weight layout matches PyTorch's
    /// `nn.Linear.weight` shape.
    pub fn forward(&self, input: &[f32], seq_len: usize) -> Vec<f32> {
        let mut buf = input.to_vec();
        for layer in &self.layers {
            debug_assert_eq!(buf.len(), seq_len * layer.in_dim);
            let mut out = vec![0.0f32; seq_len * layer.out_dim];
            for t in 0..seq_len {
                let x = &buf[t * layer.in_dim..(t + 1) * layer.in_dim];
                let y = &mut out[t * layer.out_dim..(t + 1) * layer.out_dim];
                // `enumerate` + `iter_mut` keeps both the row index
                // (for weight row slicing) and the mutable output slot,
                // silencing `needless_range_loop`.
                for (i, y_i) in y.iter_mut().enumerate() {
                    let row = &layer.weight[i * layer.in_dim..(i + 1) * layer.in_dim];
                    let mut acc = layer.bias[i];
                    for j in 0..layer.in_dim {
                        acc += row[j] * x[j];
                    }
                    // In-place LeakyReLU(0.01).
                    *y_i = if acc < 0.0 {
                        acc * LEAKY_RELU_SLOPE
                    } else {
                        acc
                    };
                }
            }
            buf = out;
        }
        buf
    }
}

/// Same shape-checking pattern the sibling BiLSTM binder uses.
fn bind_tensor(w: &PyanNetWeights, name: &str, expect_shape: &[usize]) -> Result<Vec<f32>> {
    let (dims, payload) = w.tensor(name).ok_or_else(|| {
        VokraError::ModelLoad(format!(
            "pyannote-segmentation Linear stack: required tensor `{name}` is missing from the \
             GGUF (FR-EX-08). Expected shape {expect_shape:?}."
        ))
    })?;
    if dims != expect_shape {
        return Err(VokraError::ModelLoad(format!(
            "pyannote-segmentation Linear stack: tensor `{name}` has shape {dims:?}, expected \
             {expect_shape:?} (FR-EX-08)"
        )));
    }
    let expect_elems: usize = expect_shape.iter().product();
    if payload.len() != expect_elems {
        return Err(VokraError::ModelLoad(format!(
            "pyannote-segmentation Linear stack: tensor `{name}` element count {} != expected \
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
    fn linear_layer_zero_weight_returns_bias_broadcast_then_leaky() {
        // Zero weights + non-zero biases give `y = b` uniformly.
        let layer = LinearLayer {
            in_dim: 3,
            out_dim: 2,
            weight: vec![0.0f32; 6],
            bias: vec![1.0, -1.0],
        };
        let stack = LinearStack {
            layers: vec![layer],
        };
        let input = vec![10.0f32; 5 * 3];
        let out = stack.forward(&input, 5);
        assert_eq!(out.len(), 5 * 2);
        for t in 0..5 {
            // bias[0] = 1.0 -> passes through leaky
            assert_eq!(out[t * 2], 1.0);
            // bias[1] = -1.0 -> multiplied by 0.01
            assert!((out[t * 2 + 1] - (-0.01)).abs() < 1e-6);
        }
    }

    #[test]
    fn linear_stack_forward_produces_expected_shape() {
        // 2-layer stack, matching PyanNet defaults but scaled down.
        let layer0 = LinearLayer {
            in_dim: 4,
            out_dim: 3,
            weight: vec![0.1f32; 12],
            bias: vec![0.0f32; 3],
        };
        let layer1 = LinearLayer {
            in_dim: 3,
            out_dim: 3,
            weight: vec![0.1f32; 9],
            bias: vec![0.0f32; 3],
        };
        let stack = LinearStack {
            layers: vec![layer0, layer1],
        };
        let input = vec![1.0f32; 6 * 4];
        let out = stack.forward(&input, 6);
        assert_eq!(out.len(), 6 * 3);
        for &v in &out {
            assert!(v.is_finite(), "output contains non-finite: {v}");
        }
    }

    #[test]
    fn linear_stack_from_weights_rejects_missing_tensor_loudly() {
        // Empty weights manifest — from_weights should fail loudly at
        // the first missing tensor lookup.
        use vokra_core::gguf::{GgmlType, GgufBuilder, GgufFile};

        let mut b = GgufBuilder::new();
        // Arch stamp — `PyanNetWeights::from_gguf` gates on it before any
        // tensor scan, and this fixture must reach `LinearStack::
        // from_weights`'s missing-tensor error (FR-EX-08).
        b.add_string(
            vokra_core::gguf::chunks::KEY_MODEL_ARCH,
            crate::pyannote::EXPECTED_ARCH,
        );
        b.add_tensor(
            "sincnet.conv1d.0.weight",
            GgmlType::F32,
            vec![2, 2],
            vec![0u8; 16],
        )
        .unwrap();
        b.add_tensor(
            "lstm.weight_ih_l0",
            GgmlType::F32,
            vec![2, 2],
            vec![0u8; 16],
        )
        .unwrap();
        b.add_tensor(
            "classifier.weight",
            GgmlType::F32,
            vec![2, 2],
            vec![0u8; 16],
        )
        .unwrap();
        // Deliberately omit `linear.0.*` — the binder should refuse.
        let bytes = b.to_bytes().unwrap();
        let mut path = std::env::temp_dir();
        path.push(format!(
            "vokra-pyannote-linear-refuse-{}.gguf",
            std::process::id()
        ));
        std::fs::write(&path, &bytes).unwrap();
        let g = GgufFile::open(&path).unwrap();
        let w = super::super::PyanNetWeights::from_gguf(&g).unwrap();
        let err = LinearStack::from_weights(&w, 4, 3, 2).unwrap_err();
        match err {
            VokraError::ModelLoad(msg) => {
                assert!(
                    msg.contains("linear.0.weight") && msg.contains("FR-EX-08"),
                    "missing-tensor error must name the tensor + FR-EX-08: {msg}"
                );
            }
            other => panic!("expected ModelLoad, got {other:?}"),
        }
        std::fs::remove_file(&path).ok();
    }
}
