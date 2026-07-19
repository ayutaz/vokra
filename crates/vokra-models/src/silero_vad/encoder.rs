//! Encoder conv stack (M0-05-T06): four `Conv1d`+ReLU layers that reduce the
//! magnitude spectrogram `[bins, 3]` to a single 128-channel feature frame.
//!
//! Kernel 3, pad 1 throughout; strides `1, 2, 2, 1` (verified against
//! `silero_vad.onnx`). Channel path: `bins -> 128 -> 64 -> 64 -> 128`. For the
//! canonical frame the time length collapses `3 -> 3 -> 2 -> 1 -> 1`, so the
//! output is `[128, 1]`.

use super::math::{conv1d_wt, relu_in_place};
use super::pseudo_stft::Magnitude;
use super::weights::RateWeights;

/// Encoder output: `channels` × `frames`, row-major (channel-major).
pub(super) struct EncoderOut {
    pub(super) data: Vec<f32>,
    pub(super) channels: usize,
    pub(super) frames: usize,
}

/// Strides for the four encoder convolutions.
const STRIDES: [usize; 4] = [1, 2, 2, 1];

/// Runs the encoder conv stack on the magnitude spectrogram.
pub(super) fn encode(w: &RateWeights, mag: &Magnitude) -> EncoderOut {
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
    use super::super::SampleRate;
    use super::super::pseudo_stft::pseudo_stft;
    use super::super::weights::RateWeights;
    use super::*;

    #[test]
    fn encoder_collapses_to_128_by_1() {
        for rate in [SampleRate::Hz8000, SampleRate::Hz16000] {
            let w = RateWeights::zeros_for_test(rate);
            let mag = pseudo_stft(rate, &w, &vec![0.0; rate.frame_len()]);
            let enc = encode(&w, &mag);
            assert_eq!(enc.channels, 128);
            assert_eq!(enc.frames, 1);
            assert_eq!(enc.data.len(), 128);
        }
    }
}
