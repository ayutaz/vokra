//! Encoder conv stack (M0-05-T06): four `Conv1d`+ReLU layers that reduce the
//! magnitude spectrogram `[bins, 3]` to a single 128-channel feature frame.
//!
//! Kernel 3, pad 1 throughout; strides `1, 2, 2, 1` (verified against
//! `silero_vad.onnx`). Channel path: `bins -> 128 -> 64 -> 64 -> 128`. For the
//! canonical frame the time length collapses `3 -> 3 -> 2 -> 1 -> 1`, so the
//! output is `[128, 1]`.

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use crate::math::{conv1d_wt, relu_in_place};
use crate::pseudo_stft::Magnitude;
use crate::weights::RateWeights;

/// Encoder output: `channels` × `frames`, row-major (channel-major).
///
/// Exposed for the `vokra-models::silero_vad::parity` stage harness (T06).
pub struct EncoderOut {
    /// Feature values, row-major `[channels, frames]` (channel-major).
    pub data: Vec<f32>,
    /// Feature channel count (128 for the canonical Silero encoder tail).
    pub channels: usize,
    /// Number of time frames (1 after the canonical collapse).
    pub frames: usize,
}

/// Strides for the four encoder convolutions.
const STRIDES: [usize; 4] = [1, 2, 2, 1];

/// Runs the encoder conv stack on the magnitude spectrogram.
pub fn encode(w: &RateWeights, mag: &Magnitude) -> EncoderOut {
    let mut data = mag.data.clone();
    let mut c_in = mag.bins;
    let mut len = mag.frames;
    for (layer, stride) in w.encoder.iter().zip(STRIDES) {
        debug_assert_eq!(layer.c_in, c_in);
        // M5-14 Wave-2 (T21): the transposed-weight formulation of the same
        // conv — bit-identical per element (see `math::conv1d_wt`).
        let out = conv1d_wt(
            &data,
            c_in,
            len,
            &layer.weight_t,
            Some(&layer.bias),
            layer.c_out,
            layer.k,
            stride,
            1, // pad = 1
        );
        len = (len + 2 - layer.k) / stride + 1;
        c_in = layer.c_out;
        data = out;
        relu_in_place(&mut data);
    }
    EncoderOut {
        data,
        channels: c_in,
        frames: len,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SampleRate;
    use crate::pseudo_stft::pseudo_stft;
    use crate::weights::RateWeights;

    #[test]
    fn encoder_collapses_to_128_by_1() {
        for rate in [SampleRate::Hz8000, SampleRate::Hz16000] {
            let w = RateWeights::zeros_for_test(rate);
            let mag = pseudo_stft(rate, &w, &vec_zeros(rate.frame_len()));
            let enc = encode(&w, &mag);
            assert_eq!(enc.channels, 128);
            assert_eq!(enc.frames, 1);
            assert_eq!(enc.data.len(), 128);
        }
    }

    /// A no_std-safe zero vector helper (avoids `vec!` macro imports in the
    /// test module across the std/no_std split — tests run under std anyway).
    fn vec_zeros(n: usize) -> Vec<f32> {
        let mut v = Vec::with_capacity(n);
        v.resize(n, 0.0);
        v
    }
}
