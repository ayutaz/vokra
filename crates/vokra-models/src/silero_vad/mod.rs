//! Silero VAD v5 as a 1:1-preserved dedicated subgraph (M0-05).
//!
//! # Design red lines (permanent)
//!
//! - **1:1 preservation (FR-LD-06 / FR-OP-50)**: Silero VAD is kept as a
//!   dedicated subgraph, *not* lowered to generic audio-dialect ops, and it is
//!   *not* an audio-dialect op itself. Its internal recurrent state (LSTM
//!   `h`/`c`) and the learned pseudo-STFT are hidden behind the stream handle
//!   ([`VadStream`](stream::VadStream), via [`vokra_core::engines::VadEngine`]).
//! - **No librosa/FFT STFT approximation (NFR-QL-05)**: the pseudo-STFT is a
//!   *learned* `Conv1d(1, 2*bins, k)` ([`pseudo_stft`]). Replacing it with a
//!   standard `stft` op (FR-OP-01) would corrupt the trained weights, so this
//!   module never calls the `vokra-ops` `stft` path — every op runs through the
//!   module-private [`math`] helpers.
//!
//! # Architecture (source: `docs/_research/03-speech-specialized-runtimes.md`
//! §3.1, cross-checked against the upstream `silero_vad.onnx`)
//!
//! ```text
//! frame [512 @16k / 256 @8k]
//!  -> reflect-pad right by n_fft/4
//!  -> Conv1d(1, 2*bins, k=n_fft, stride=n_fft/2)      (learned pseudo-STFT)
//!  -> magnitude = sqrt(real^2 + imag^2)   [bins, 3]   (bins = 129 @16k / 65 @8k)
//!  -> encoder: Conv1d(+ReLU) x4, strides 1,2,2,1      [128, 1]
//!  -> LSTM(128,128)  (h/c carried across frames)      [128]
//!  -> ReLU -> Conv1d(128,1,k=1) -> Sigmoid            -> probability
//! ```
//!
//! Silero v5 is really **two** independently-trained networks with this same
//! topology, one per sample rate, chosen in the ONNX by `If(sr == 16000)`. The
//! per-layer GGUF tensor map, the two-branch weight gap, the exact pad/gate
//! findings and the parity methodology are recorded in
//! `crates/vokra-models/src/silero_vad/SPEC.md`.
//!
//! # Layout (M0-05)
//!
//! - [`weights`] — GGUF binding for one/both sample-rate weight sets (T03);
//! - [`pseudo_stft`] — reflection pad + learned conv + magnitude (T04/T05);
//! - [`encoder`] — the four Conv1d+ReLU layers (T06);
//! - [`lstm`] — the LSTM cell (state carry) + decoder head (T07);
//! - [`stream`] — the [`VadStream`](stream::VadStream) handle (T08).

mod encoder;
mod lstm;
mod math;
mod pseudo_stft;
mod stream;
mod weights;

pub mod wav;

#[cfg(test)]
mod parity;

use std::sync::Arc;

use vokra_core::engines::{VadEngine, VadStreamHandle};
use vokra_core::gguf::GgufFile;
use vokra_core::{Result, VokraError};

use lstm::{LstmState, head_probability, lstm_forward};
use pseudo_stft::pseudo_stft;
use weights::{RateWeights, SileroWeights};

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

    /// Right-side reflection pad width (`n_fft / 4`).
    fn pad(self) -> usize {
        self.n_fft() / 4
    }

    /// Pseudo-STFT kernel length / FFT size (128 @ 8 kHz, 256 @ 16 kHz).
    fn n_fft(self) -> usize {
        match self {
            Self::Hz8000 => 128,
            Self::Hz16000 => 256,
        }
    }

    /// Pseudo-STFT stride (`n_fft / 2`).
    fn stft_stride(self) -> usize {
        self.n_fft() / 2
    }

    /// Magnitude bins (`n_fft / 2 + 1`): 65 @ 8 kHz, 129 @ 16 kHz.
    fn bins(self) -> usize {
        self.n_fft() / 2 + 1
    }

    /// GGUF tensor-name prefix for this rate in the corrected both-rate scheme.
    fn gguf_prefix(self) -> &'static str {
        match self {
            Self::Hz8000 => "sr8k",
            Self::Hz16000 => "sr16k",
        }
    }
}

/// Runs one fixed-size frame through the whole subgraph, advancing `state`, and
/// returns the frame's speech probability. Private to the subgraph; reached by
/// [`SileroVadV5::forward_chunk`] and the [`stream`] handle.
fn run_frame(rate: SampleRate, w: &RateWeights, frame: &[f32], state: &mut LstmState) -> f32 {
    let mag = pseudo_stft(rate, w, frame);
    let enc = encoder::encode(w, &mag);
    let hidden = lstm_forward(w, &enc, state);
    head_probability(w, &hidden)
}

/// Silero VAD v5 — a 1:1-preserved subgraph model (M0-05).
///
/// Load with [`from_gguf`](Self::from_gguf) / [`open`](Self::open), then obtain
/// a stateful stream through the [`VadEngine`] trait ([`open_stream`]). The
/// model itself is immutable and shareable; all mutable recurrent state lives in
/// the stream handle (FR-LD-06).
///
/// [`open_stream`]: VadEngine::open_stream
pub struct SileroVadV5 {
    weights: Arc<SileroWeights>,
}

impl SileroVadV5 {
    /// Binds the model from a parsed GGUF (FR-LD-01).
    ///
    /// Accepts the corrected both-rate GGUF (`sr8k.*` / `sr16k.*`) or the legacy
    /// single-rate one; see [`weights`]. Returns [`VokraError::ModelLoad`] if no
    /// Silero weights are present or a tensor has the wrong shape/dtype.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        Ok(Self {
            weights: Arc::new(SileroWeights::from_gguf(gguf)?),
        })
    }

    /// Opens and binds the model from a GGUF file on disk.
    pub fn open(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let gguf = GgufFile::open(path)?;
        Self::from_gguf(&gguf)
    }

    /// Returns whether the loaded GGUF carries weights for `rate`.
    pub fn supports(&self, rate: SampleRate) -> bool {
        self.weights.rate(rate).is_some()
    }

    /// Runs a single fixed-size frame from a **fresh zero state** and returns its
    /// speech probability (the T07 single-chunk entry point).
    ///
    /// `frame` must be exactly [`SampleRate::frame_len`] samples. Errors if the
    /// model lacks weights for `rate` or the frame length is wrong.
    pub fn forward_chunk(&self, rate: SampleRate, frame: &[f32]) -> Result<f32> {
        let w = self.weights.rate(rate).ok_or_else(|| {
            VokraError::InvalidArgument(format!("model has no weights for {} Hz", rate.hz()))
        })?;
        if frame.len() != rate.frame_len() {
            return Err(VokraError::InvalidArgument(format!(
                "frame must be {} samples for {} Hz, got {}",
                rate.frame_len(),
                rate.hz(),
                frame.len()
            )));
        }
        let mut state = LstmState::zeros();
        Ok(run_frame(rate, w, frame, &mut state))
    }
}

impl VadEngine for SileroVadV5 {
    fn open_stream(&self) -> Box<dyn VadStreamHandle + Send> {
        Box::new(stream::VadStream::new(Arc::clone(&self.weights)))
    }
}

/// Absolute path to the committed parity fixture GGUF (both rates), for tests.
#[cfg(test)]
pub(crate) fn test_gguf_path() -> std::path::PathBuf {
    parity_dir().join("silero-vad-v5.gguf")
}

/// Absolute path to the `tests/parity/silero_vad` fixture directory.
#[cfg(test)]
pub(crate) fn parity_dir() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/parity/silero_vad")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_both_rates_from_fixture() {
        let m = SileroVadV5::open(test_gguf_path()).expect("load fixture gguf");
        assert!(m.supports(SampleRate::Hz8000));
        assert!(m.supports(SampleRate::Hz16000));
    }

    #[test]
    fn from_gguf_reports_missing_tensor() {
        // An empty GGUF has no Silero weights -> explicit ModelLoad error.
        let bytes = vokra_core::gguf::GgufBuilder::new().to_bytes().unwrap();
        let gguf = GgufFile::parse(bytes).unwrap();
        assert!(matches!(
            SileroVadV5::from_gguf(&gguf),
            Err(VokraError::ModelLoad(_))
        ));
    }

    #[test]
    fn forward_chunk_rejects_wrong_frame_len() {
        let m = SileroVadV5::open(test_gguf_path()).unwrap();
        assert!(m.forward_chunk(SampleRate::Hz16000, &[0.0; 400]).is_err());
    }

    #[test]
    fn sample_rate_geometry() {
        assert_eq!(SampleRate::Hz16000.bins(), 129);
        assert_eq!(SampleRate::Hz8000.bins(), 65);
        assert_eq!(SampleRate::Hz16000.pad(), 64);
        assert_eq!(SampleRate::Hz8000.pad(), 32);
        assert!(SampleRate::from_hz(44100).is_err());
    }
}
