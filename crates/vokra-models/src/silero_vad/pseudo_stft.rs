//! The learned pseudo-STFT front-end (M0-05-T04 / T05).
//!
//! **NFR-QL-05 / FR-LD-06 red line**: this is a *learned* `Conv1d(1, 2*bins, k)`
//! (the `stft.forward_basis_buffer` weight), **not** a DSP STFT. It is reproduced
//! op-for-op from the upstream model — reflection pad on the right, strided
//! convolution, then `magnitude = sqrt(real^2 + imag^2)` over the two halves of
//! the conv output channels. Substituting `fft(window(x))` (the `vokra-ops`
//! `stft` op) would corrupt the trained weights, so this module never calls it.
//!
//! Layout verified against `silero_vad.onnx`: for a frame of `f` samples the
//! right reflection pad is `k/4` (`pad`), the conv has stride `k/2`, and the
//! `2*bins` output channels split as `real = ch[0..bins]`, `imag =
//! ch[bins..2*bins]`.

use super::SampleRate;
use super::math::{conv1d_wt, reflect_pad_right};
use super::weights::RateWeights;

/// Output of the pseudo-STFT: magnitude spectrogram, `bins` rows × `frames`
/// columns, stored row-major (bin-major, frame fastest).
pub(super) struct Magnitude {
    pub(super) data: Vec<f32>,
    pub(super) bins: usize,
    pub(super) frames: usize,
}

/// The raw pseudo-STFT conv output `[2*bins, frames]` (channel-major), before
/// the magnitude step. The `real`/`imag` halves live in channels `[0, bins)`
/// and `[bins, 2*bins)`.
pub(super) fn stft_conv(rate: SampleRate, w: &RateWeights, frame: &[f32]) -> (Vec<f32>, usize) {
    let padded = reflect_pad_right(frame, rate.pad());
    // Conv1d(1, 2*bins, k=n_fft, stride=n_fft/2): a single input channel.
    // M5-14 Wave-2 (T21): transposed-weight formulation, bit-identical per
    // element to the original scalar conv (see `math::conv1d_wt`).
    let conv = conv1d_wt(
        &padded,
        1,
        padded.len(),
        &w.stft.weight_t,
        None,
        w.stft.c_out, // 2*bins
        w.stft.k,
        rate.stft_stride(),
        0,
    );
    let frames = conv.len() / w.stft.c_out;
    (conv, frames)
}

/// Runs the pseudo-STFT on one graph input — a bare fixed frame (512 @ 16 kHz
/// / 256 @ 8 kHz → 3 STFT frames) or a context-prefixed one (576 / 288 → 4;
/// the official interface) — and returns the magnitude spectrogram. Length is
/// dynamic exactly as in the ONNX graph.
pub(super) fn pseudo_stft(rate: SampleRate, w: &RateWeights, frame: &[f32]) -> Magnitude {
    let bins = rate.bins();
    let (conv, frames) = stft_conv(rate, w, frame);
    // real = channels [0, bins), imag = channels [bins, 2*bins); both [bins, frames].
    let mut data = vec![0.0f32; bins * frames];
    for b in 0..bins {
        for t in 0..frames {
            let re = conv[b * frames + t];
            let im = conv[(bins + b) * frames + t];
            data[b * frames + t] = (re * re + im * im).sqrt();
        }
    }
    Magnitude { data, bins, frames }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn magnitude_shape_is_bins_by_three() {
        // A frame of zeros still produces the fixed [bins, 3] shape (the pad
        // does not change the canonical frame count).
        for rate in [SampleRate::Hz8000, SampleRate::Hz16000] {
            // Build a trivial rate-weight with a zero stft basis of the right
            // shape so we exercise only the shape/pad plumbing here.
            let bins = rate.bins();
            let k = rate.n_fft();
            let w = super::super::weights::RateWeights::zeros_for_test(rate);
            let mag = pseudo_stft(rate, &w, &vec![0.0; rate.frame_len()]);
            assert_eq!(mag.bins, bins);
            assert_eq!(mag.frames, 3, "canonical frame count for k={k}");
            assert!(mag.data.iter().all(|v| *v == 0.0));
        }
    }
}
