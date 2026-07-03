//! `VadStream` — the handle that hides all recurrent state (M0-05-T08).
//!
//! Implements [`vokra_core::engines::VadStreamHandle`]. Per research 03 §3.1 and
//! FR-LD-06, the caller only pushes PCM and reads probabilities; the LSTM
//! `h`/`c`, the fixed-frame buffering and the pseudo-STFT are all private here —
//! none is exposed as a public field. A stream is single-rate: the sample rate
//! is fixed by the first `push_pcm` and a later change is rejected.
//!
//! Framing matches the reference: non-overlapping fixed frames of 512 samples
//! @ 16 kHz / 256 @ 8 kHz, LSTM state carried across frames, a trailing partial
//! frame buffered until the next push. No audio context crosses frames (the
//! reflection pad is internal to each frame) — only `h`/`c` carry over.
//!
//! `stream.poll(events)` + lock-free ring buffer (FR-ST-02) and generalised
//! streaming state (FR-ST-05) are M1; M0 provides this minimal synchronous
//! handle only (see the ticket scope note).

use std::sync::Arc;

use vokra_core::engines::VadStreamHandle;
use vokra_core::{Result, VokraError};

use super::lstm::LstmState;
use super::weights::SileroWeights;
use super::{SampleRate, run_frame};

/// A stateful, single-rate VAD stream over a shared model.
pub(super) struct VadStream {
    weights: Arc<SileroWeights>,
    /// Fixed once the first sample arrives; `None` until then.
    rate: Option<SampleRate>,
    state: LstmState,
    /// Samples not yet forming a complete frame.
    pending: Vec<f32>,
}

impl VadStream {
    /// Creates a fresh stream (zeroed state) over the model's weights.
    pub(super) fn new(weights: Arc<SileroWeights>) -> Self {
        Self {
            weights,
            rate: None,
            state: LstmState::zeros(),
            pending: Vec::new(),
        }
    }
}

impl VadStreamHandle for VadStream {
    fn push_pcm(&mut self, pcm: &[f32], sample_rate: u32) -> Result<Vec<f32>> {
        let rate = SampleRate::from_hz(sample_rate)?;
        match self.rate {
            None => {
                // The model must actually carry this rate's weights.
                if self.weights.rate(rate).is_none() {
                    return Err(VokraError::InvalidArgument(format!(
                        "model has no weights for {} Hz (this GGUF carries only the other rate)",
                        rate.hz()
                    )));
                }
                self.rate = Some(rate);
            }
            Some(fixed) if fixed != rate => {
                return Err(VokraError::InvalidArgument(format!(
                    "stream is fixed at {} Hz; cannot switch to {} Hz mid-stream (open a new stream)",
                    fixed.hz(),
                    rate.hz()
                )));
            }
            Some(_) => {}
        }

        let w = self
            .weights
            .rate(rate)
            .expect("presence checked when the rate was fixed");
        let frame_len = rate.frame_len();
        self.pending.extend_from_slice(pcm);

        let mut probs = Vec::new();
        let mut consumed = 0;
        while self.pending.len() - consumed >= frame_len {
            let frame = &self.pending[consumed..consumed + frame_len];
            probs.push(run_frame(rate, w, frame, &mut self.state));
            consumed += frame_len;
        }
        if consumed > 0 {
            self.pending.drain(0..consumed);
        }
        Ok(probs)
    }

    fn reset(&mut self) {
        self.state = LstmState::zeros();
        self.pending.clear();
        self.rate = None;
    }
}

#[cfg(test)]
mod tests {
    use vokra_core::engines::{VadEngine, VadStreamHandle};
    use vokra_core::rng::Xorshift64Star;

    use crate::silero_vad::SileroVadV5;
    use crate::silero_vad::wav::read_wav_f32;
    use crate::silero_vad::{parity_dir, test_gguf_path};

    fn stream() -> Box<dyn VadStreamHandle + Send> {
        SileroVadV5::open(test_gguf_path())
            .expect("load fixture gguf")
            .open_stream()
    }

    /// Property: with LSTM state carried across non-overlapping fixed frames,
    /// the per-frame probabilities must not depend on how the *same* PCM is
    /// split across `push_pcm` calls. A whole-utterance push and the identical
    /// samples re-split into irregular chunks must yield bit-identical output.
    /// Oracle is internal self-consistency; no external reference.
    fn assert_chunk_invariant(wav_name: &str, hz: u32) {
        let model = SileroVadV5::open(test_gguf_path()).expect("load fixture gguf");
        let wav = read_wav_f32(parity_dir().join(wav_name)).expect("read fixture wav");
        assert_eq!(wav.sample_rate, hz, "fixture rate");

        // (a) one whole-utterance push.
        let mut whole = model.open_stream();
        let probs_whole = whole.push_pcm(&wav.samples, hz).unwrap();
        assert!(!probs_whole.is_empty(), "fixture yields at least one frame");

        // (b) the same samples re-split into irregular chunks (sizes 1..=700,
        // straddling the 256/512-sample frame boundary and single-sample pushes).
        let mut split = model.open_stream();
        let mut probs_split = Vec::new();
        let mut rng = Xorshift64Star::new(0x5EED_51E1);
        let mut i = 0;
        while i < wav.samples.len() {
            let remaining = wav.samples.len() - i;
            let len = (1 + (rng.next_u64() % 700) as usize).min(remaining);
            probs_split.extend(split.push_pcm(&wav.samples[i..i + len], hz).unwrap());
            i += len;
        }

        assert_eq!(
            probs_whole, probs_split,
            "{hz} Hz: framing must be chunk-invariant"
        );
    }

    #[test]
    fn rejects_unsupported_sample_rate() {
        let mut s = stream();
        assert!(s.push_pcm(&[0.0; 512], 44100).is_err());
    }

    #[test]
    fn buffers_partial_frames_until_complete() {
        let mut s = stream();
        // 300 samples < 512: no frame yet.
        assert!(s.push_pcm(&vec![0.0; 300], 16000).unwrap().is_empty());
        // +300 = 600 >= 512: exactly one frame emitted, 88 buffered.
        assert_eq!(s.push_pcm(&vec![0.0; 300], 16000).unwrap().len(), 1);
    }

    #[test]
    fn rejects_rate_switch_mid_stream() {
        let mut s = stream();
        s.push_pcm(&[0.0; 512], 16000).unwrap();
        assert!(s.push_pcm(&[0.0; 256], 8000).is_err());
    }

    #[test]
    fn reset_reproduces_first_run() {
        let mut s = stream();
        let a = s.push_pcm(&vec![0.1; 512 * 3], 16000).unwrap();
        s.reset();
        let b = s.push_pcm(&vec![0.1; 512 * 3], 16000).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn push_pcm_is_chunk_invariant_16k() {
        assert_chunk_invariant("test_16k.wav", 16000);
    }

    #[test]
    fn push_pcm_is_chunk_invariant_8k() {
        assert_chunk_invariant("test_8k.wav", 8000);
    }
}
