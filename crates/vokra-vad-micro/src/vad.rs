//! Sample-rate geometry + the single-input forward (`run_frame`).
//!
//! This is the length-driven pipeline of the 1:1 Silero subgraph, exactly as in
//! the ONNX graph (dynamic time axis): pseudo-STFT → magnitude → encoder → LSTM
//! → head. It is shared verbatim by the std `vokra-models` wrapper and the
//! no_std thumbv8m build (M5-03), so both are bit-identical.

#[cfg(not(feature = "std"))]
use alloc::format;

use vokra_core::{Result, VokraError};

use crate::encoder;
use crate::lstm::{LstmState, head_probability, lstm_forward};
use crate::pseudo_stft::pseudo_stft;
use crate::weights::RateWeights;

/// A sample rate Silero v5 supports (the single model handles both).
///
/// Each rate implies a fixed frame length and pseudo-STFT geometry; the two
/// rates are backed by *different* weight sets in the GGUF (see the SPEC).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SampleRate {
    /// 8 kHz — 256-sample frames, 128-point pseudo-STFT, 65 magnitude bins.
    Hz8000,
    /// 16 kHz — 512-sample frames, 256-point pseudo-STFT, 129 magnitude bins.
    Hz16000,
}

impl SampleRate {
    /// Maps a raw rate in Hz to a [`SampleRate`], rejecting anything but the two
    /// rates Silero supports (resampling is out of M0 scope).
    pub fn from_hz(hz: u32) -> Result<Self> {
        match hz {
            8000 => Ok(Self::Hz8000),
            16000 => Ok(Self::Hz16000),
            other => Err(VokraError::InvalidArgument(format!(
                "Silero VAD accepts only 8000 or 16000 Hz, got {other}"
            ))),
        }
    }

    /// The rate in Hz.
    pub fn hz(self) -> u32 {
        match self {
            Self::Hz8000 => 8000,
            Self::Hz16000 => 16000,
        }
    }

    /// Fixed frame length in samples (256 @ 8 kHz, 512 @ 16 kHz).
    pub fn frame_len(self) -> usize {
        match self {
            Self::Hz8000 => 256,
            Self::Hz16000 => 512,
        }
    }

    /// Rolling audio-context length of the **official** interface (64 @ 16 kHz,
    /// 32 @ 8 kHz = `frame_len / 8`): the trailing samples of the previous
    /// frame that the upstream python wrapper (`utils_vad.py OnnxWrapper`)
    /// prepends to every frame, so the graph sees `[1, 576]` / `[1, 288]`.
    /// Zeros before the first frame; reset together with the LSTM state.
    /// Without this context the probabilities collapse on real speech (the
    /// 2026-07-16 real-weight eval P1).
    pub fn context_len(self) -> usize {
        match self {
            Self::Hz8000 => 32,
            Self::Hz16000 => 64,
        }
    }

    /// Right-side reflection pad width (`n_fft / 4`).
    pub(crate) fn pad(self) -> usize {
        self.n_fft() / 4
    }

    /// Pseudo-STFT kernel length / FFT size (128 @ 8 kHz, 256 @ 16 kHz).
    pub(crate) fn n_fft(self) -> usize {
        match self {
            Self::Hz8000 => 128,
            Self::Hz16000 => 256,
        }
    }

    /// Pseudo-STFT stride (`n_fft / 2`).
    pub(crate) fn stft_stride(self) -> usize {
        self.n_fft() / 2
    }

    /// Magnitude bins (`n_fft / 2 + 1`): 65 @ 8 kHz, 129 @ 16 kHz.
    pub(crate) fn bins(self) -> usize {
        self.n_fft() / 2 + 1
    }

    /// GGUF tensor-name prefix for this rate in the corrected both-rate scheme.
    pub(crate) fn gguf_prefix(self) -> &'static str {
        match self {
            Self::Hz8000 => "sr8k",
            Self::Hz16000 => "sr16k",
        }
    }
}

/// Runs one graph input through the whole subgraph, advancing `state`, and
/// returns its speech probability.
///
/// `frame` is either a bare fixed frame ([`SampleRate::frame_len`], the raw
/// 1:1 ONNX interface) or a context-prefixed one (`context_len + frame_len`,
/// the official interface). The pipeline is length-driven exactly like the
/// ONNX graph (dynamic time axis): the pseudo-STFT yields 3 or 4 frames and
/// the encoder collapses either to a single time step.
pub fn run_frame(rate: SampleRate, w: &RateWeights, frame: &[f32], state: &mut LstmState) -> f32 {
    let mag = pseudo_stft(rate, w, frame);
    let enc = encoder::encode(w, &mag);
    let hidden = lstm_forward(w, &enc, state);
    head_probability(w, &hidden)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_rate_geometry() {
        assert_eq!(SampleRate::Hz16000.bins(), 129);
        assert_eq!(SampleRate::Hz8000.bins(), 65);
        assert_eq!(SampleRate::Hz16000.pad(), 64);
        assert_eq!(SampleRate::Hz8000.pad(), 32);
        assert_eq!(SampleRate::Hz16000.n_fft(), 256);
        assert_eq!(SampleRate::Hz8000.n_fft(), 128);
        assert_eq!(SampleRate::Hz16000.stft_stride(), 128);
        assert_eq!(SampleRate::Hz8000.stft_stride(), 64);
        // Official-wrapper rolling context (upstream utils_vad.py OnnxWrapper).
        assert_eq!(SampleRate::Hz16000.context_len(), 64);
        assert_eq!(SampleRate::Hz8000.context_len(), 32);
        assert_eq!(SampleRate::Hz16000.frame_len(), 512);
        assert_eq!(SampleRate::Hz8000.frame_len(), 256);
        assert_eq!(SampleRate::Hz16000.gguf_prefix(), "sr16k");
        assert_eq!(SampleRate::Hz8000.gguf_prefix(), "sr8k");
        assert!(SampleRate::from_hz(44100).is_err());
        assert_eq!(SampleRate::from_hz(8000).unwrap(), SampleRate::Hz8000);
        assert_eq!(SampleRate::from_hz(16000).unwrap(), SampleRate::Hz16000);
    }
}
