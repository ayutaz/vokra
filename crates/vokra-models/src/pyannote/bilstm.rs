//! Monolithic multi-layer bidirectional LSTM stack for `PyanNet`.
//!
//! # Primary source
//!
//! PyanNet.py declares a monolithic bidirectional `nn.LSTM`. Its class default
//! is two layers, while the immutable segmentation-3.0 model config overrides
//! it to four:
//!
//! ```python
//! self.lstm = nn.LSTM(60, hidden_size=128, num_layers=4,
//!                     bidirectional=True, batch_first=True, dropout=0.0)
//! ```
//!
//! The released `state_dict` independently proves layers `l0..l3` in both
//! directions. Its layout is the standard PyTorch one:
//!
//! ```text
//! lstm.weight_ih_l0        (4·H, I)          # forward,  layer 0
//! lstm.weight_hh_l0        (4·H, H)
//! lstm.bias_ih_l0          (4·H,)
//! lstm.bias_hh_l0          (4·H,)
//! lstm.weight_ih_l0_reverse(4·H, I)          # backward, layer 0
//! lstm.weight_hh_l0_reverse(4·H, H)
//! lstm.bias_ih_l0_reverse  (4·H,)
//! lstm.bias_hh_l0_reverse  (4·H,)
//! lstm.weight_ih_lN        (4·H, 2·H)        # forward,  layers N=1..3
//! lstm.weight_hh_lN        (4·H, H)
//! lstm.bias_ih_lN          (4·H,)
//! lstm.bias_hh_lN          (4·H,)
//! lstm.weight_ih_lN_reverse(4·H, 2·H)        # backward, layers N=1..3
//! lstm.weight_hh_lN_reverse(4·H, H)
//! lstm.bias_ih_lN_reverse  (4·H,)
//! lstm.bias_hh_lN_reverse  (4·H,)
//! ```
//!
//! Gate order in each row-major weight: `i | f | g | o` at
//! `[0..H, H..2H, 2H..3H, 3H..4H]`. This matches the layout
//! documented in Vokra's sibling [`crate::kokoro::nn::BiLstm1d`] (which
//! we pattern-match here for numerical parity with PyTorch cuDNN).
//!
//! # Design rationale (numeric parity)
//!
//! The scalar sequential reduction ordering — `acc = b_ih + b_hh; for
//! each j { acc += w[j] · v[j]; }` — matches Kokoro's `BiLstm1d`
//! (`crate::kokoro::nn`). Kokoro documented that this order lands ~10 %
//! closer to PyTorch's `_thnn_fused_lstm_cell` output than a horizontal
//! SIMD-tree reduction (M2-07 T17-fixup #2 comment). This is a
//! **duplicate implementation, not a re-export**, because the parent
//! `kokoro::nn` module is `mod nn;` (private to kokoro), and cross-
//! module coupling would violate the intentional module-independence
//! Kokoro's docstring cites. The math is identical; a future audit
//! could de-duplicate by lifting `BiLstm1d` to a `pub(crate)` sibling.
//!
//! # Zero-dep invariant (NFR-DS-02)
//!
//! Pure Rust — `sigmoid` via `1.0 / (1.0 + (-x).exp())`, `tanh` via
//! `f32::tanh`, no BLAS, no crates.io addition.

use vokra_core::{Result, VokraError};

use crate::compute::Compute;

use super::PyanNetWeights;

// ---------------------------------------------------------------------------
// Per-layer BiLSTM primitive
// ---------------------------------------------------------------------------

/// One bidirectional layer of a monolithic `nn.LSTM` stack.
///
/// Row-major weight layout matches PyTorch: `w_ih` is `[4·H, I]` with
/// the `i|f|g|o` gate stack on the outer axis. `input_dim` is the
/// per-timestep feature width the layer consumes; `hidden_dim` is the
/// per-direction cell state width.
#[derive(Debug)]
pub(crate) struct BiLstmLayer {
    input_dim: usize,
    hidden_dim: usize,
    w_ih: [Vec<f32>; 2],
    w_hh: [Vec<f32>; 2],
    b_ih: [Vec<f32>; 2],
    b_hh: [Vec<f32>; 2],
}

impl BiLstmLayer {
    /// Builds a layer from validated weight vectors.
    ///
    /// Every buffer length is cross-checked against `(input_dim,
    /// hidden_dim)`; a length mismatch is a loud
    /// [`VokraError::ModelLoad`] (FR-EX-08) naming the offending
    /// state_dict path.
    #[allow(clippy::too_many_arguments)]
    fn new(
        input_dim: usize,
        hidden_dim: usize,
        w_ih_fwd: Vec<f32>,
        w_hh_fwd: Vec<f32>,
        b_ih_fwd: Vec<f32>,
        b_hh_fwd: Vec<f32>,
        w_ih_rev: Vec<f32>,
        w_hh_rev: Vec<f32>,
        b_ih_rev: Vec<f32>,
        b_hh_rev: Vec<f32>,
        layer_idx: usize,
    ) -> Result<Self> {
        if input_dim == 0 || hidden_dim == 0 {
            return Err(VokraError::ModelLoad(format!(
                "pyannote-segmentation BiLSTM layer {layer_idx}: input_dim ({input_dim}) \
                 and hidden_dim ({hidden_dim}) must be > 0 (FR-EX-08)"
            )));
        }
        let g = 4 * hidden_dim;
        let want_ih = g * input_dim;
        let want_hh = g * hidden_dim;
        check_len("weight_ih_l0", &w_ih_fwd, want_ih, layer_idx, "forward")?;
        check_len("weight_hh_l0", &w_hh_fwd, want_hh, layer_idx, "forward")?;
        check_len("bias_ih_l0", &b_ih_fwd, g, layer_idx, "forward")?;
        check_len("bias_hh_l0", &b_hh_fwd, g, layer_idx, "forward")?;
        check_len(
            "weight_ih_l0_reverse",
            &w_ih_rev,
            want_ih,
            layer_idx,
            "reverse",
        )?;
        check_len(
            "weight_hh_l0_reverse",
            &w_hh_rev,
            want_hh,
            layer_idx,
            "reverse",
        )?;
        check_len("bias_ih_l0_reverse", &b_ih_rev, g, layer_idx, "reverse")?;
        check_len("bias_hh_l0_reverse", &b_hh_rev, g, layer_idx, "reverse")?;
        Ok(Self {
            input_dim,
            hidden_dim,
            w_ih: [w_ih_fwd, w_ih_rev],
            w_hh: [w_hh_fwd, w_hh_rev],
            b_ih: [b_ih_fwd, b_ih_rev],
            b_hh: [b_hh_fwd, b_hh_rev],
        })
    }

    /// Bidirectional forward on `[seq_len · input_dim]` row-major
    /// input; returns `[seq_len · (2·hidden_dim)]` row-major output.
    /// Forward direction's `h_t` occupies columns `[0, hidden_dim)`;
    /// reverse direction's `h_t` occupies `[hidden_dim, 2·hidden_dim)`
    /// — matching `nn.LSTM(batch_first=True, bidirectional=True)`.
    fn forward(&self, input: &[f32], seq_len: usize) -> Vec<f32> {
        debug_assert_eq!(input.len(), seq_len * self.input_dim);
        let h = self.hidden_dim;
        let mut output = vec![0.0f32; seq_len * 2 * h];
        if seq_len == 0 {
            return output;
        }
        // Forward direction.
        let mut hs = vec![0.0f32; h];
        let mut cs = vec![0.0f32; h];
        let mut gates = vec![0.0f32; 4 * h];
        for t in 0..seq_len {
            let x_t = &input[t * self.input_dim..(t + 1) * self.input_dim];
            self.step(0, x_t, &mut hs, &mut cs, &mut gates);
            output[t * 2 * h..t * 2 * h + h].copy_from_slice(&hs);
        }
        // Reverse direction.
        hs.fill(0.0);
        cs.fill(0.0);
        for t in (0..seq_len).rev() {
            let x_t = &input[t * self.input_dim..(t + 1) * self.input_dim];
            self.step(1, x_t, &mut hs, &mut cs, &mut gates);
            output[t * 2 * h + h..(t + 1) * 2 * h].copy_from_slice(&hs);
        }
        output
    }

    fn forward_with_compute(
        &self,
        input: &[f32],
        seq_len: usize,
        compute: &Compute,
    ) -> Result<Vec<f32>> {
        debug_assert_eq!(input.len(), seq_len * self.input_dim);
        let hidden = self.hidden_dim;
        let mut output = vec![0.0f32; seq_len * 2 * hidden];
        if seq_len == 0 {
            return Ok(output);
        }
        let mut state = vec![0.0f32; hidden];
        let mut cell = vec![0.0f32; hidden];
        let mut gates = vec![0.0f32; 4 * hidden];
        for time in 0..seq_len {
            let input = &input[time * self.input_dim..(time + 1) * self.input_dim];
            self.step_with_compute(0, input, &mut state, &mut cell, &mut gates, compute)?;
            output[time * 2 * hidden..time * 2 * hidden + hidden].copy_from_slice(&state);
        }
        state.fill(0.0);
        cell.fill(0.0);
        for time in (0..seq_len).rev() {
            let input = &input[time * self.input_dim..(time + 1) * self.input_dim];
            self.step_with_compute(1, input, &mut state, &mut cell, &mut gates, compute)?;
            output[time * 2 * hidden + hidden..(time + 1) * 2 * hidden].copy_from_slice(&state);
        }
        Ok(output)
    }

    /// One `nn.LSTMCell` step for a given direction; mutates `h` / `c`
    /// in place. Formula (PyTorch, no peephole):
    ///
    /// ```text
    /// i = σ(W_ii·x + b_ii + W_hi·h + b_hi)
    /// f = σ(W_if·x + b_if + W_hf·h + b_hf)
    /// g = tanh(W_ig·x + b_ig + W_hg·h + b_hg)
    /// o = σ(W_io·x + b_io + W_ho·h + b_ho)
    /// c' = f·c + i·g
    /// h' = o·tanh(c')
    /// ```
    fn step(&self, dir: usize, x: &[f32], h: &mut [f32], c: &mut [f32], gates: &mut [f32]) {
        let hd = self.hidden_dim;
        let idim = self.input_dim;
        let w_ih = &self.w_ih[dir];
        let w_hh = &self.w_hh[dir];
        let b_ih = &self.b_ih[dir];
        let b_hh = &self.b_hh[dir];
        for i in 0..(4 * hd) {
            let ih_row = &w_ih[i * idim..(i + 1) * idim];
            let hh_row = &w_hh[i * hd..(i + 1) * hd];
            let mut acc = b_ih[i] + b_hh[i];
            for j in 0..idim {
                acc += ih_row[j] * x[j];
            }
            for j in 0..hd {
                acc += hh_row[j] * h[j];
            }
            gates[i] = acc;
        }
        for j in 0..hd {
            let ig = sigmoid(gates[j]);
            let fg = sigmoid(gates[hd + j]);
            let gg = gates[2 * hd + j].tanh();
            let og = sigmoid(gates[3 * hd + j]);
            let new_c = fg * c[j] + ig * gg;
            let new_h = og * new_c.tanh();
            c[j] = new_c;
            h[j] = new_h;
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn step_with_compute(
        &self,
        direction: usize,
        input: &[f32],
        state: &mut [f32],
        cell: &mut [f32],
        gates: &mut [f32],
        compute: &Compute,
    ) -> Result<()> {
        let hidden = self.hidden_dim;
        let rows = 4 * hidden;
        compute.gemv_f32(
            rows,
            self.input_dim,
            &self.w_ih[direction],
            input,
            Some(&self.b_ih[direction]),
            gates,
        )?;
        let mut recurrent = vec![0.0f32; rows];
        compute.gemv_f32(
            rows,
            hidden,
            &self.w_hh[direction],
            state,
            Some(&self.b_hh[direction]),
            &mut recurrent,
        )?;
        for (gate, recurrent) in gates.iter_mut().zip(recurrent) {
            *gate += recurrent;
        }
        for index in 0..hidden {
            let input_gate = sigmoid(gates[index]);
            let forget_gate = sigmoid(gates[hidden + index]);
            let candidate = gates[2 * hidden + index].tanh();
            let output_gate = sigmoid(gates[3 * hidden + index]);
            let next_cell = forget_gate * cell[index] + input_gate * candidate;
            cell[index] = next_cell;
            state[index] = output_gate * next_cell.tanh();
        }
        Ok(())
    }
}

fn check_len(
    tensor: &str,
    buf: &[f32],
    want: usize,
    layer_idx: usize,
    direction: &str,
) -> Result<()> {
    if buf.len() != want {
        return Err(VokraError::ModelLoad(format!(
            "pyannote-segmentation BiLSTM layer {layer_idx} {direction}: tensor `{tensor}` \
             length {}, expected {want} (FR-EX-08)",
            buf.len()
        )));
    }
    Ok(())
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// ---------------------------------------------------------------------------
// Monolithic multi-layer BiLSTM stack
// ---------------------------------------------------------------------------

/// Released four-layer monolithic bidirectional LSTM stack, matching
/// segmentation-3.0's `nn.LSTM(60, hidden_size=128, num_layers=4, bidirectional=True,
/// batch_first=True, dropout=0.0)`.
///
/// Layer indexing follows PyTorch's `_lN` suffix (l0 = first layer,
/// l1..l3 = subsequent layers). Every layer after `l0` consumes
/// `2·hidden_dim` because the preceding output concatenates forward and
/// reverse hidden states.
#[derive(Debug)]
pub(crate) struct MonoLithicBiLstmStack {
    layers: Vec<BiLstmLayer>,
}

impl MonoLithicBiLstmStack {
    /// Binds the stack from a [`PyanNetWeights`] manifest by walking
    /// each layer's 8 named tensors (`weight_ih_l<k>` / `weight_hh_l<k>`
    /// / `bias_ih_l<k>` / `bias_hh_l<k>` and each `_reverse` variant).
    ///
    /// # Errors
    ///
    /// - [`VokraError::ModelLoad`] with FR-EX-08 if any tensor is
    ///   missing or mis-shaped. The message names the offending
    ///   state_dict path.
    pub fn from_weights(
        w: &PyanNetWeights,
        input_dim: usize,
        hidden_dim: usize,
        num_layers: usize,
    ) -> Result<Self> {
        if num_layers == 0 {
            return Err(VokraError::ModelLoad(
                "pyannote-segmentation BiLSTM: num_layers must be >= 1 (FR-EX-08)".to_string(),
            ));
        }
        let mut layers = Vec::with_capacity(num_layers);
        for k in 0..num_layers {
            let in_dim = if k == 0 { input_dim } else { 2 * hidden_dim };
            let g = 4 * hidden_dim;
            let want_ih = g * in_dim;
            let want_hh = g * hidden_dim;
            let w_ih_fwd = bind_tensor(w, &format!("lstm.weight_ih_l{k}"), &[g, in_dim], want_ih)?;
            let w_hh_fwd = bind_tensor(
                w,
                &format!("lstm.weight_hh_l{k}"),
                &[g, hidden_dim],
                want_hh,
            )?;
            let b_ih_fwd = bind_tensor(w, &format!("lstm.bias_ih_l{k}"), &[g], g)?;
            let b_hh_fwd = bind_tensor(w, &format!("lstm.bias_hh_l{k}"), &[g], g)?;
            let w_ih_rev = bind_tensor(
                w,
                &format!("lstm.weight_ih_l{k}_reverse"),
                &[g, in_dim],
                want_ih,
            )?;
            let w_hh_rev = bind_tensor(
                w,
                &format!("lstm.weight_hh_l{k}_reverse"),
                &[g, hidden_dim],
                want_hh,
            )?;
            let b_ih_rev = bind_tensor(w, &format!("lstm.bias_ih_l{k}_reverse"), &[g], g)?;
            let b_hh_rev = bind_tensor(w, &format!("lstm.bias_hh_l{k}_reverse"), &[g], g)?;
            let layer = BiLstmLayer::new(
                in_dim, hidden_dim, w_ih_fwd, w_hh_fwd, b_ih_fwd, b_hh_fwd, w_ih_rev, w_hh_rev,
                b_ih_rev, b_hh_rev, k,
            )?;
            layers.push(layer);
        }
        Ok(Self { layers })
    }

    /// Sequential forward through every layer. `input` is
    /// `[seq_len · input_dim]` row-major (batch_first=True); output is
    /// `[seq_len · (2·hidden_dim)]` row-major.
    pub fn forward(&self, input: &[f32], seq_len: usize) -> Vec<f32> {
        let mut buf = input.to_vec();
        for layer in &self.layers {
            buf = layer.forward(&buf, seq_len);
        }
        buf
    }

    /// Backend-dispatched sibling of [`Self::forward`]. Every learned input
    /// and recurrent projection uses GEMV on the selected backend.
    pub fn forward_with_compute(
        &self,
        input: &[f32],
        seq_len: usize,
        compute: &Compute,
    ) -> Result<Vec<f32>> {
        let mut buffer = input.to_vec();
        for layer in &self.layers {
            buffer = layer.forward_with_compute(&buffer, seq_len, compute)?;
        }
        Ok(buffer)
    }
}

/// Looks up a tensor from [`PyanNetWeights`] by name and verifies its
/// shape and element count. Mirrors the pattern the SincNet binder
/// uses.
fn bind_tensor(
    w: &PyanNetWeights,
    name: &str,
    expect_shape: &[usize],
    expect_elems: usize,
) -> Result<Vec<f32>> {
    let (dims, payload) = w.tensor(name).ok_or_else(|| {
        VokraError::ModelLoad(format!(
            "pyannote-segmentation BiLSTM: required tensor `{name}` is missing from the GGUF \
             (FR-EX-08). Expected shape {expect_shape:?}."
        ))
    })?;
    if dims != expect_shape {
        return Err(VokraError::ModelLoad(format!(
            "pyannote-segmentation BiLSTM: tensor `{name}` has shape {dims:?}, expected \
             {expect_shape:?} (FR-EX-08)"
        )));
    }
    if payload.len() != expect_elems {
        return Err(VokraError::ModelLoad(format!(
            "pyannote-segmentation BiLSTM: tensor `{name}` element count {} != expected \
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
    fn bilstm_layer_zero_input_returns_zero_output() {
        // With zero inputs and zero initial state, the gates reduce to
        // `[b_ih + b_hh]` — a pure bias term. Setting biases to zero
        // gives `gates = 0`, so `sigmoid(0) = 0.5`, `tanh(0) = 0`,
        // therefore `c' = 0.5·0 + 0.5·0 = 0` and `h' = 0.5·tanh(0) = 0`.
        let input_dim = 2;
        let hidden_dim = 3;
        let g = 4 * hidden_dim;
        let layer = BiLstmLayer::new(
            input_dim,
            hidden_dim,
            vec![0.0f32; g * input_dim],
            vec![0.0f32; g * hidden_dim],
            vec![0.0f32; g],
            vec![0.0f32; g],
            vec![0.0f32; g * input_dim],
            vec![0.0f32; g * hidden_dim],
            vec![0.0f32; g],
            vec![0.0f32; g],
            0,
        )
        .unwrap();
        let input = vec![0.0f32; 5 * input_dim];
        let out = layer.forward(&input, 5);
        assert_eq!(out.len(), 5 * 2 * hidden_dim);
        for &v in &out {
            assert_eq!(v, 0.0, "zero-input + zero-weight BiLSTM must be all zeros");
        }
    }

    #[test]
    fn bilstm_layer_rejects_wrong_weight_size_loudly() {
        let g = 4 * 3;
        let err = BiLstmLayer::new(
            2,
            3,
            vec![0.0f32; g * 2 - 1], // wrong: one short
            vec![0.0f32; g * 3],
            vec![0.0f32; g],
            vec![0.0f32; g],
            vec![0.0f32; g * 2],
            vec![0.0f32; g * 3],
            vec![0.0f32; g],
            vec![0.0f32; g],
            0,
        )
        .unwrap_err();
        match err {
            VokraError::ModelLoad(msg) => {
                assert!(msg.contains("FR-EX-08") && msg.contains("weight_ih_l0"));
            }
            other => panic!("expected ModelLoad, got {other:?}"),
        }
    }

    #[test]
    fn bilstm_layer_produces_finite_output_on_nontrivial_weights() {
        // Small deterministic weights and input; assert every output is
        // finite (no NaN / inf leaked from sigmoid or tanh).
        let input_dim = 2;
        let hidden_dim = 2;
        let g = 4 * hidden_dim;
        let w_ih: Vec<f32> = (0..g * input_dim).map(|i| (i as f32) * 0.01).collect();
        let w_hh: Vec<f32> = (0..g * hidden_dim).map(|i| (i as f32) * 0.02).collect();
        let b_ih: Vec<f32> = vec![0.01; g];
        let b_hh: Vec<f32> = vec![-0.01; g];
        let layer = BiLstmLayer::new(
            input_dim,
            hidden_dim,
            w_ih.clone(),
            w_hh.clone(),
            b_ih.clone(),
            b_hh.clone(),
            w_ih,
            w_hh,
            b_ih,
            b_hh,
            0,
        )
        .unwrap();
        let input: Vec<f32> = (0..4 * input_dim)
            .map(|i| ((i as f32) - 4.0) * 0.1)
            .collect();
        let out = layer.forward(&input, 4);
        let dispatched = layer
            .forward_with_compute(&input, 4, &Compute::cpu())
            .expect("Compute BiLSTM");
        let max_abs = out
            .iter()
            .zip(&dispatched)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_abs <= 1e-5, "BiLSTM Compute max_abs={max_abs}");
        for &v in &out {
            assert!(v.is_finite(), "output contains non-finite: {v}");
        }
    }

    #[test]
    fn bilstm_layer_forward_produces_expected_shape() {
        let g = 4 * 3;
        let layer = BiLstmLayer::new(
            2,
            3,
            vec![0.1f32; g * 2],
            vec![0.1f32; g * 3],
            vec![0.0f32; g],
            vec![0.0f32; g],
            vec![0.1f32; g * 2],
            vec![0.1f32; g * 3],
            vec![0.0f32; g],
            vec![0.0f32; g],
            0,
        )
        .unwrap();
        let input = vec![0.5f32; 10 * 2];
        let out = layer.forward(&input, 10);
        // seq_len · 2·hidden_dim = 10 · 6 = 60.
        assert_eq!(out.len(), 60);
    }

    #[test]
    fn sigmoid_matches_reference_points() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!((sigmoid(10.0) - 1.0).abs() < 1e-3);
        assert!((sigmoid(-10.0) - 0.0).abs() < 1e-3);
    }
}
