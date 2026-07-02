//! LSTM cell + output head (M0-05-T07).
//!
//! One `LSTM(128, 128)` cell whose `h`/`c` state is **carried across frames**
//! (the streaming state, hidden behind the handle — FR-LD-06), followed by the
//! decoder head `ReLU -> Conv1d(128, 1, k=1) -> Sigmoid`.
//!
//! Gate order is PyTorch **`ifgo`** (input, forget, cell, output) — the stored
//! `decoder.rnn.*` weights are the original PyTorch parameters, and this order
//! was confirmed by matching the onnxruntime oracle to ~5e-8 (see the crate
//! SPEC). Both `bias_ih` and `bias_hh` are applied (as PyTorch does).

use super::encoder::EncoderOut;
use super::math::{matvec, sigmoid};
use super::weights::{HIDDEN, RateWeights};

/// Streaming LSTM state (`h` and `c`), zero-initialised for a fresh utterance.
#[derive(Clone)]
pub(super) struct LstmState {
    pub(super) h: Vec<f32>,
    pub(super) c: Vec<f32>,
}

impl LstmState {
    /// A zeroed state (the initial state Silero's ONNX interface feeds).
    pub(super) fn zeros() -> Self {
        Self {
            h: vec![0.0; HIDDEN],
            c: vec![0.0; HIDDEN],
        }
    }
}

/// Advances the LSTM over every frame of the encoder output, mutating `state`,
/// and returns the final hidden vector `[128]`.
pub(super) fn lstm_forward(w: &RateWeights, enc: &EncoderOut, state: &mut LstmState) -> Vec<f32> {
    let t_len = enc.frames;
    let mut x = vec![0.0f32; enc.channels];
    for t in 0..t_len {
        // Gather timestep t (encoder output is channel-major [C, T]).
        for (c, xc) in x.iter_mut().enumerate() {
            *xc = enc.data[c * t_len + t];
        }
        let gih = matvec(&w.lstm_wih, 4 * HIDDEN, HIDDEN, &x);
        let ghh = matvec(&w.lstm_whh, 4 * HIDDEN, HIDDEN, &state.h);
        for j in 0..HIDDEN {
            // ifgo gate slices.
            let i = sigmoid(gih[j] + w.lstm_bih[j] + ghh[j] + w.lstm_bhh[j]);
            let f = sigmoid(
                gih[HIDDEN + j] + w.lstm_bih[HIDDEN + j] + ghh[HIDDEN + j] + w.lstm_bhh[HIDDEN + j],
            );
            let g = (gih[2 * HIDDEN + j]
                + w.lstm_bih[2 * HIDDEN + j]
                + ghh[2 * HIDDEN + j]
                + w.lstm_bhh[2 * HIDDEN + j])
                .tanh();
            let o = sigmoid(
                gih[3 * HIDDEN + j]
                    + w.lstm_bih[3 * HIDDEN + j]
                    + ghh[3 * HIDDEN + j]
                    + w.lstm_bhh[3 * HIDDEN + j],
            );
            let c_new = f * state.c[j] + i * g;
            state.c[j] = c_new;
            state.h[j] = o * c_new.tanh();
        }
    }
    state.h.clone()
}

/// Decoder head: `ReLU -> Conv1d(128, 1, k=1) -> Sigmoid`, producing the frame's
/// speech probability. (The head runs on the single collapsed time frame, so
/// the ONNX `ReduceMean` over time is a no-op here.)
pub(super) fn head_probability(w: &RateWeights, hidden: &[f32]) -> f32 {
    debug_assert_eq!(hidden.len(), HIDDEN);
    debug_assert_eq!(w.head.c_out, 1);
    debug_assert_eq!(w.head.c_in, HIDDEN);
    // k = 1, so the head weight [1, 128, 1] is a plain [128] dot product.
    let mut acc = w.head.bias[0];
    for (&wc, &hc) in w.head.weight.iter().zip(hidden) {
        acc += wc * hc.max(0.0);
    }
    sigmoid(acc)
}

#[cfg(test)]
mod tests {
    use super::super::SampleRate;
    use super::*;

    #[test]
    fn zero_state_zero_input_gives_half() {
        // With zero weights, zero input and zero state: gates biased at 0 ->
        // i=f=o=0.5, g=0 -> c stays 0, h stays 0; head acc = 0 -> sigmoid = 0.5.
        let w = RateWeights::zeros_for_test(SampleRate::Hz16000);
        let enc = EncoderOut {
            data: vec![0.0; 128],
            channels: 128,
            frames: 1,
        };
        let mut st = LstmState::zeros();
        let h = lstm_forward(&w, &enc, &mut st);
        assert!(h.iter().all(|v| *v == 0.0));
        assert!((head_probability(&w, &h) - 0.5).abs() < 1e-6);
    }
}
