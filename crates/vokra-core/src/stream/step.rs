//! The [`StreamStep`] trait and the three first-class stepping patterns
//! (FR-ST-01), plus the VAD adapter that bridges the M0 [`VadStreamHandle`].
//!
//! FR-ST-01 names three ways to drive a streaming model; they unify under one
//! trait:
//!
//! - **frame-by-frame** — [`step_frame`](StreamStep::step_frame) consumes exactly
//!   [`frame_len`](StreamStep::frame_len) samples and emits the events for that
//!   one frame;
//! - **chunk-based** — [`step_chunk`](StreamStep::step_chunk) consumes an
//!   arbitrary number of samples, buffering internally, and emits zero or more
//!   events;
//! - **cache-based** — a stepper whose hidden state is a KV cache; the ASR
//!   incremental decoder ([`DecodeStepper`](crate::decode::DecodeStepper),
//!   M1-08f) is that pattern for tokens rather than audio-in.
//!
//! All recurrent state (LSTM `h`/`c`, KV cache, iSTFT tail) stays hidden inside
//! the stepper (FR-ST-05): a `StreamStep` only ever takes `&[f32]` and emits
//! [`StreamEvent`]s — never a tensor name.

use crate::engines::VadStreamHandle;
use crate::error::{Result, VokraError};

use super::event::{EventSink, StreamEvent};

/// A stateful streaming inference step.
///
/// `Send` (not `Sync`): a stepper is moved onto one worker/audio thread and
/// driven there; the cross-thread hand-off of *events* is the ring, not shared
/// mutable access to the stepper. Implementors provide
/// [`step_chunk`](Self::step_chunk) and [`reset`](Self::reset); the default
/// [`step_frame`](Self::step_frame) validates the frame length against
/// [`frame_len`](Self::frame_len) and forwards to `step_chunk`.
pub trait StreamStep: Send {
    /// Feeds an arbitrary-length `samples` chunk, emitting every event that
    /// completes into `sink`. Trailing partial state is buffered internally.
    fn step_chunk(&mut self, samples: &[f32], sink: &mut dyn EventSink) -> Result<()>;

    /// Feeds exactly one frame. The default checks `frame.len()` against
    /// [`frame_len`](Self::frame_len) (when the stepper declares a fixed frame)
    /// and then delegates to [`step_chunk`](Self::step_chunk).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] if the stepper declares a fixed
    /// [`frame_len`](Self::frame_len) and `frame.len()` differs from it.
    fn step_frame(&mut self, frame: &[f32], sink: &mut dyn EventSink) -> Result<()> {
        if let Some(len) = self.frame_len() {
            if frame.len() != len {
                return Err(VokraError::InvalidArgument(format!(
                    "step_frame: expected exactly {len} samples, got {}",
                    frame.len()
                )));
            }
        }
        self.step_chunk(frame, sink)
    }

    /// Clears all hidden state, returning the stepper to its initial state so a
    /// fresh run reproduces the first run bit-for-bit.
    fn reset(&mut self);

    /// The fixed frame length for [`step_frame`](Self::step_frame), or `None`
    /// for a stepper that buffers/frames internally (e.g. VAD, which accepts any
    /// chunk length and frames the PCM itself).
    fn frame_len(&self) -> Option<usize> {
        None
    }
}

/// Adapts an M0 [`VadStreamHandle`] into a [`StreamStep`], emitting one
/// [`StreamEvent::SpeechProb`] per completed frame.
///
/// This reuses the existing native Silero VAD stream (M0-05) unchanged: the
/// handle keeps hiding its LSTM `h`/`c` and framing, and this adapter only
/// tags each returned probability with a monotonic frame index. It captures the
/// sample rate at construction so the rate-free [`StreamStep`] surface stays
/// uniform across models.
pub(crate) struct VadStreamStep {
    handle: Box<dyn VadStreamHandle + Send>,
    sample_rate: u32,
    next_frame: u32,
}

impl VadStreamStep {
    /// Wraps `handle`, pinning the stream sample rate.
    pub(crate) fn new(handle: Box<dyn VadStreamHandle + Send>, sample_rate: u32) -> Self {
        Self {
            handle,
            sample_rate,
            next_frame: 0,
        }
    }
}

impl StreamStep for VadStreamStep {
    fn step_chunk(&mut self, samples: &[f32], sink: &mut dyn EventSink) -> Result<()> {
        let probs = self.handle.push_pcm(samples, self.sample_rate)?;
        for prob in probs {
            let frame_index = self.next_frame;
            // The frame completed regardless of whether the sink accepts it, so
            // the index advances monotonically even under ring backpressure
            // (a dropped event never reuses an index).
            self.next_frame = self.next_frame.wrapping_add(1);
            let _accepted = sink.emit(StreamEvent::SpeechProb { frame_index, prob });
        }
        Ok(())
    }

    fn reset(&mut self) {
        self.handle.reset();
        self.next_frame = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fake VAD handle emitting a canned probability per 4-sample block, so the
    /// adapter is exercised without a real Silero model or a `vokra-models`
    /// dependency (an internal oracle).
    struct FakeVad {
        buffered: usize,
    }
    impl VadStreamHandle for FakeVad {
        fn push_pcm(&mut self, pcm: &[f32], _rate: u32) -> Result<Vec<f32>> {
            self.buffered += pcm.len();
            let mut out = Vec::new();
            while self.buffered >= 4 {
                self.buffered -= 4;
                out.push(0.25);
            }
            Ok(out)
        }
        fn reset(&mut self) {
            self.buffered = 0;
        }
    }

    fn adapter() -> VadStreamStep {
        VadStreamStep::new(Box::new(FakeVad { buffered: 0 }), 16_000)
    }

    #[test]
    fn emits_one_speech_prob_per_completed_frame_with_monotonic_indices() {
        let mut step = adapter();
        let mut sink: Vec<StreamEvent> = Vec::new();
        step.step_chunk(&[0.0; 10], &mut sink).unwrap(); // 10 samples -> 2 frames, 2 buffered
        step.step_chunk(&[0.0; 6], &mut sink).unwrap(); //  +6 = 8 -> 2 frames
        let indices: Vec<u32> = sink
            .iter()
            .map(|e| match e {
                StreamEvent::SpeechProb { frame_index, .. } => *frame_index,
                other => panic!("unexpected {other:?}"),
            })
            .collect();
        assert_eq!(
            indices,
            vec![0, 1, 2, 3],
            "frame indices are monotonic 0..n"
        );
    }

    #[test]
    fn reset_restarts_frame_indexing() {
        let mut step = adapter();
        let mut sink: Vec<StreamEvent> = Vec::new();
        step.step_chunk(&[0.0; 8], &mut sink).unwrap();
        step.reset();
        sink.clear();
        step.step_chunk(&[0.0; 4], &mut sink).unwrap();
        assert_eq!(
            sink,
            vec![StreamEvent::SpeechProb {
                frame_index: 0,
                prob: 0.25
            }],
            "reset rewinds the frame index and internal buffer"
        );
    }

    #[test]
    fn vad_declares_no_fixed_frame_length() {
        // VAD frames internally, so it accepts any chunk length via step_frame.
        assert_eq!(adapter().frame_len(), None);
    }
}
