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
//! input [ctx + frame = 576 @16k / 288 @8k]  (official; raw graph interface = bare 512 / 256)
//!  -> reflect-pad right by n_fft/4
//!  -> Conv1d(1, 2*bins, k=n_fft, stride=n_fft/2)      (learned pseudo-STFT)
//!  -> magnitude = sqrt(real^2 + imag^2)   [bins, 4]   (bins = 129 @16k / 65 @8k; 3 on raw input)
//!  -> encoder: Conv1d(+ReLU) x4, strides 1,2,2,1      [128, 1]
//!  -> LSTM(128,128)  (h/c carried across frames)      [128]
//!  -> ReLU -> Conv1d(128,1,k=1) -> Sigmoid            -> probability
//! ```
//!
//! The graph's time axis is dynamic; **official usage** (the upstream python
//! wrapper, and this module's default stream) prepends a rolling
//! [`SampleRate::context_len`] audio context — the previous frame's tail,
//! zeros at start — to every fixed frame. Feeding bare frames is numerically
//! valid but collapses on real speech (2026-07-16 real-weight eval P1).
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

    /// Rolling audio-context length of the **official** interface (64 @ 16 kHz,
    /// 32 @ 8 kHz = `frame_len / 8`): the trailing samples of the previous
    /// frame that the upstream python wrapper (`utils_vad.py OnnxWrapper`)
    /// prepends to every frame, so the graph sees `[1, 576]` / `[1, 288]`.
    /// Zeros before the first frame; reset together with the LSTM state.
    /// Without this context the probabilities collapse on real speech (the
    /// 2026-07-16 real-weight eval P1) — see [`stream`].
    pub fn context_len(self) -> usize {
        match self {
            Self::Hz8000 => 32,
            Self::Hz16000 => 64,
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

/// Runs one graph input through the whole subgraph, advancing `state`, and
/// returns its speech probability. Private to the subgraph; reached by
/// [`SileroVadV5::forward_chunk`] and the [`stream`] handle.
///
/// `frame` is either a bare fixed frame ([`SampleRate::frame_len`], the raw
/// 1:1 ONNX interface) or a context-prefixed one (`context_len + frame_len`,
/// the official interface). The pipeline is length-driven exactly like the
/// ONNX graph (dynamic time axis): the pseudo-STFT yields 3 or 4 frames and
/// the encoder collapses either to a single time step.
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
    /// Follows the official interface: a zero rolling context of
    /// [`SampleRate::context_len`] samples is prepended (exactly the first
    /// frame of a fresh [`VadEngine::open_stream`] stream, and of the upstream
    /// python wrapper after `reset_states`).
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
        let mut buf = vec![0.0f32; rate.context_len() + frame.len()];
        buf[rate.context_len()..].copy_from_slice(frame);
        let mut state = LstmState::zeros();
        Ok(run_frame(rate, w, &buf, &mut state))
    }

    /// Opens a stream over the **raw** 1:1 ONNX frame interface: bare
    /// [`SampleRate::frame_len`] frames, no rolling audio context (only the
    /// LSTM `h`/`c` crosses frames). This is the interface the bare-frame
    /// parity fixtures (`probs_16k.txt` / `probs_8k.txt`) are generated on;
    /// it is **not** how the model is used upstream and it cannot detect real
    /// speech (2026-07-16 eval P1) — test-gated, for parity only, so no
    /// production path can reach the collapsed semantics.
    #[cfg(test)]
    pub(crate) fn open_raw_stream(&self) -> Box<dyn VadStreamHandle + Send> {
        Box::new(stream::VadStream::new_raw(Arc::clone(&self.weights)))
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
        // Official-wrapper rolling context (upstream utils_vad.py OnnxWrapper).
        assert_eq!(SampleRate::Hz16000.context_len(), 64);
        assert_eq!(SampleRate::Hz8000.context_len(), 32);
        assert!(SampleRate::from_hz(44100).is_err());
    }

    /// The single-chunk entry point follows the official semantics: it must be
    /// bit-identical to the first frame of a fresh official stream.
    #[test]
    fn forward_chunk_matches_official_stream_first_frame() {
        use vokra_core::engines::VadEngine;

        let model = SileroVadV5::open(test_gguf_path()).unwrap();
        let wav = wav::read_wav_f32(parity_dir().join("test_16k.wav")).unwrap();
        let frame = &wav.samples[..SampleRate::Hz16000.frame_len()];

        let single = model.forward_chunk(SampleRate::Hz16000, frame).unwrap();
        let mut stream = model.open_stream();
        let first = stream.push_pcm(frame, 16_000).unwrap();
        assert_eq!(first, vec![single]);
    }
}
