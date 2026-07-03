//! Typed streaming events and the [`EventSink`] the steppers emit into.
//!
//! [`StreamEvent`] is the model-independent, tensor-name-free output of a
//! streaming step (FR-ST-05): callers only ever see `&[f32]` in and
//! `StreamEvent` out. Each event round-trips through the ring as a 16-byte
//! [`RawEvent`] via [`StreamEvent::to_raw`] / [`StreamEvent::from_raw`].
//!
//! An [`EventSink`] is the write end a [`StreamStep`](super::StreamStep) emits
//! into — a plain [`Vec`] for ergonomic single-thread use, or a
//! [`RingProducer`] for the lock-free cross-thread path.

use super::ring::{RawEvent, RingProducer};

/// Event tag stored in the high 32 bits of [`RawEvent::w0`].
const TAG_SPEECH_PROB: u32 = 1;
const TAG_TOKEN: u32 = 2;

/// One typed streaming event.
///
/// `#[non_exhaustive]`: more event kinds land with later models (KWS hits,
/// diarization turns, ...), so downstream matches must keep a wildcard arm.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum StreamEvent {
    /// A VAD speech-probability for one completed frame (Silero VAD = M0-05).
    SpeechProb {
        /// Monotonic frame index within the stream (0-based).
        frame_index: u32,
        /// Speech probability in `[0, 1]` for that frame.
        prob: f32,
    },
    /// One decoded ASR token (the cache-based decode stepper, M1-08f).
    Token {
        /// Vocabulary id of the emitted token.
        id: u32,
        /// Bitflags for the token (bit 0 = end-of-transcript; see
        /// [`crate::decode::TOKEN_FLAG_EOT`]).
        flags: u32,
    },
}

impl StreamEvent {
    /// Encodes the event into its 16-byte ring payload.
    #[must_use]
    pub fn to_raw(self) -> RawEvent {
        match self {
            StreamEvent::SpeechProb { frame_index, prob } => RawEvent {
                w0: ((TAG_SPEECH_PROB as u64) << 32) | u64::from(frame_index),
                w1: u64::from(prob.to_bits()),
            },
            StreamEvent::Token { id, flags } => RawEvent {
                w0: ((TAG_TOKEN as u64) << 32) | u64::from(id),
                w1: u64::from(flags),
            },
        }
    }

    /// Decodes a ring payload back into a [`StreamEvent`], or `None` for an
    /// unknown tag (a defensively-handled corrupt / future encoding).
    #[must_use]
    pub fn from_raw(raw: RawEvent) -> Option<StreamEvent> {
        let tag = (raw.w0 >> 32) as u32;
        let a = (raw.w0 & 0xFFFF_FFFF) as u32;
        let b = (raw.w1 & 0xFFFF_FFFF) as u32;
        match tag {
            TAG_SPEECH_PROB => Some(StreamEvent::SpeechProb {
                frame_index: a,
                prob: f32::from_bits(b),
            }),
            TAG_TOKEN => Some(StreamEvent::Token { id: a, flags: b }),
            _ => None,
        }
    }
}

/// The write end a [`StreamStep`](super::StreamStep) emits events into.
///
/// [`emit`](EventSink::emit) returns `false` when the event could not be
/// accepted (a full ring), which the caller may treat as backpressure. Two
/// impls ship: [`Vec<StreamEvent>`] (single-thread, never full) and
/// [`RingProducer`] (lock-free cross-thread transport).
pub trait EventSink {
    /// Accepts one event, returning `false` on backpressure (e.g. a full ring).
    fn emit(&mut self, ev: StreamEvent) -> bool;
}

impl EventSink for Vec<StreamEvent> {
    fn emit(&mut self, ev: StreamEvent) -> bool {
        self.push(ev);
        true
    }
}

impl EventSink for RingProducer {
    fn emit(&mut self, ev: StreamEvent) -> bool {
        self.try_push(ev.to_raw()).is_ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn speech_prob_round_trips_bit_exact() {
        for &(frame_index, prob) in &[(0u32, 0.0f32), (5, 0.5), (123_456, 0.999_999), (7, 1.0)] {
            let ev = StreamEvent::SpeechProb { frame_index, prob };
            let back = StreamEvent::from_raw(ev.to_raw()).expect("known tag");
            match back {
                StreamEvent::SpeechProb {
                    frame_index: fi,
                    prob: p,
                } => {
                    assert_eq!(fi, frame_index);
                    // Bit-exact: to_bits/from_bits round-trips a finite f32 exactly.
                    assert_eq!(p.to_bits(), prob.to_bits());
                }
                other => panic!("wrong variant: {other:?}"),
            }
        }
    }

    #[test]
    fn token_round_trips() {
        let ev = StreamEvent::Token {
            id: 42,
            flags: 0b11,
        };
        assert_eq!(StreamEvent::from_raw(ev.to_raw()), Some(ev));
    }

    #[test]
    fn unknown_tag_decodes_to_none() {
        // Tag 9 is not a defined event kind.
        let raw = RawEvent {
            w0: (9u64 << 32) | 3,
            w1: 0,
        };
        assert_eq!(StreamEvent::from_raw(raw), None);
    }

    #[test]
    fn vec_sink_collects_in_order() {
        let mut sink: Vec<StreamEvent> = Vec::new();
        assert!(sink.emit(StreamEvent::SpeechProb {
            frame_index: 0,
            prob: 0.1
        }));
        assert!(sink.emit(StreamEvent::SpeechProb {
            frame_index: 1,
            prob: 0.2
        }));
        assert_eq!(sink.len(), 2);
    }

    #[test]
    fn ring_sink_reports_backpressure_when_full() {
        use crate::stream::channel;
        let (mut tx, _rx) = channel(2); // cap 2
        assert!(tx.emit(StreamEvent::Token { id: 0, flags: 0 }));
        assert!(tx.emit(StreamEvent::Token { id: 1, flags: 0 }));
        // Third emit hits a full ring ⇒ false (backpressure signalled).
        assert!(!tx.emit(StreamEvent::Token { id: 2, flags: 0 }));
    }
}
