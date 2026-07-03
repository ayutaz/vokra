//! Stream handles: session/stream management (M0-02) + the M1-08 streaming API.
//!
//! A [`Session`] opens independent [`Stream`] handles, each with a
//! session-unique id, released on [`Drop`]. On top of that M0 lifecycle, M1-08
//! adds the streaming inference API:
//!
//! - a lock-free SPSC [`channel`] ([`ring`], FR-ST-02);
//! - the three [`StreamStep`] patterns (frame / chunk / cache, FR-ST-01);
//! - [`Stream::step_chunk`] / [`step_frame`](Stream::step_frame) for ergonomic
//!   single-thread stepping, and [`Stream::push`] + [`Stream::take_poller`] →
//!   [`EventPoller::poll`] for the cross-thread producer/consumer split;
//! - `Send` guarantees proven at compile time (see the assertion block below)
//!   and [`Session`] `Clone` (an atomic `Arc` bump — the mechanism behind the
//!   C-ABI atomic ref count, FR-API-03).
//!
//! All streaming state (RNN `h`/`c`, KV cache, iSTFT tail) is owned by the
//! stepper inside the stream handle, so callers never manage tensor names
//! (FR-ST-05): a stream takes `&[f32]` in and yields [`StreamEvent`]s out.

mod event;
mod ring;
mod step;

pub use event::{EventSink, StreamEvent};
pub use ring::{RawEvent, RingConsumer, RingFull, RingProducer, channel};
pub use step::StreamStep;

use std::sync::Arc;
use std::sync::atomic::Ordering;

use crate::error::{Result, VokraError};
use crate::session::{Session, SessionInner};
use step::VadStreamStep;

/// Default per-stream ring capacity (events). A power of two comfortably above
/// the frame count of a multi-second VAD utterance, so a single push→poll cycle
/// never overflows; sustained overrun surfaces as a reduced [`Stream::push`]
/// count (reject-on-full backpressure), never silent corruption.
const DEFAULT_RING_CAPACITY: usize = 4096;

/// Per-stream state shell (FR-ST-05).
///
/// The concrete recurrent state now lives inside the stepper
/// ([`StreamStep`]); this type is retained as the stable public handle for
/// per-stream metadata and stays `#[non_exhaustive]`.
#[derive(Debug, Default)]
#[non_exhaustive]
pub struct StreamState {}

/// Handle to one streaming inference context of a [`Session`].
///
/// Created with [`Session::open_stream`] (a bare lifecycle stream) or
/// [`Session::open_vad_stepper`] / [`Session::open_step_stream`] (a stepping
/// stream). Carries a session-unique [`id`](Stream::id) and releases its slot on
/// [`Drop`].
///
/// `Send` (move it onto the audio-callback thread and call [`push`](Stream::push));
/// not `Sync` — it owns a `&mut` stepper and the ring producer.
pub struct Stream {
    session: Arc<SessionInner>,
    id: u64,
    state: StreamState,
    /// The stepping engine; `None` for a bare lifecycle-only stream.
    step: Option<Box<dyn StreamStep + Send>>,
    /// Producer half of the per-stream event ring.
    producer: RingProducer,
    /// Consumer half; taken once by [`take_poller`](Stream::take_poller) for the
    /// cross-thread split, otherwise drained in place by
    /// [`poll_into`](Stream::poll_into) for same-thread use.
    consumer: Option<RingConsumer>,
}

impl std::fmt::Debug for Stream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Stream")
            .field("id", &self.id)
            .field("has_stepper", &self.step.is_some())
            .field("poller_taken", &self.consumer.is_none())
            .finish()
    }
}

/// The consuming half of a [`Stream`], handed out by
/// [`Stream::take_poller`] for the cross-thread producer/consumer split.
///
/// `Send`: move it to the main thread and [`poll`](EventPoller::poll) without
/// ever blocking on the audio thread (FR-ST-02, NFR-RL-08).
pub struct EventPoller {
    consumer: RingConsumer,
}

/// Drains decoded [`StreamEvent`]s from `consumer` into `out`, skipping any
/// undecodable payload (defensive; never happens for events this crate emits).
fn drain_decoded(consumer: &mut RingConsumer, out: &mut [StreamEvent]) -> usize {
    let mut n = 0;
    while n < out.len() {
        match consumer.pop() {
            Some(raw) => {
                if let Some(ev) = StreamEvent::from_raw(raw) {
                    out[n] = ev;
                    n += 1;
                }
            }
            None => break,
        }
    }
    n
}

/// Pops the next decoded event from `consumer`, or `None` when empty.
fn pop_decoded(consumer: &mut RingConsumer) -> Option<StreamEvent> {
    loop {
        match consumer.pop() {
            Some(raw) => {
                if let Some(ev) = StreamEvent::from_raw(raw) {
                    return Some(ev);
                }
            }
            None => return None,
        }
    }
}

/// An [`EventSink`] over a `&mut RingProducer` that counts accepted events, so
/// [`Stream::push`] can report how many events were enqueued (backpressure).
struct CountingProducer<'a> {
    producer: &'a mut RingProducer,
    count: usize,
}

impl EventSink for CountingProducer<'_> {
    fn emit(&mut self, ev: StreamEvent) -> bool {
        if self.producer.emit(ev) {
            self.count += 1;
            true
        } else {
            false
        }
    }
}

impl Session {
    /// Allocates a session-unique stream over an optional stepper.
    fn new_stream(&self, step: Option<Box<dyn StreamStep + Send>>) -> Stream {
        // Relaxed: plain counters; id uniqueness comes from the atomic RMW.
        let id = self.inner.next_stream_id.fetch_add(1, Ordering::Relaxed);
        self.inner.active_streams.fetch_add(1, Ordering::Relaxed);
        let (producer, consumer) = channel(DEFAULT_RING_CAPACITY);
        Stream {
            session: Arc::clone(&self.inner),
            id,
            state: StreamState {},
            step,
            producer,
            consumer: Some(consumer),
        }
    }

    /// Opens a bare independent stream (M0 lifecycle: id + active-stream
    /// bookkeeping, no stepper).
    ///
    /// Stepping calls ([`Stream::step_chunk`] / [`push`](Stream::push)) on a bare
    /// stream return [`VokraError::NotImplemented`]; attach a stepper with
    /// [`open_vad_stepper`](Self::open_vad_stepper) or
    /// [`open_step_stream`](Self::open_step_stream).
    ///
    /// ```no_run
    /// let session = vokra_core::Session::from_file("voice.gguf").build()?;
    /// let a = session.open_stream()?;
    /// let b = session.open_stream()?;
    /// assert_ne!(a.id(), b.id());
    /// assert_eq!(session.active_stream_count(), 2);
    /// # Ok::<(), vokra_core::VokraError>(())
    /// ```
    pub fn open_stream(&self) -> Result<Stream> {
        Ok(self.new_stream(None))
    }

    /// Opens a VAD stepping stream at `sample_rate` (Silero VAD = M0-05).
    ///
    /// Returns [`VokraError::NotImplemented`] if no VAD engine is attached
    /// (mirrors [`open_vad_stream`](Self::open_vad_stream)).
    pub fn open_vad_stepper(&self, sample_rate: u32) -> Result<Stream> {
        let handle = self.open_vad_stream()?;
        Ok(self.new_stream(Some(Box::new(VadStreamStep::new(handle, sample_rate)))))
    }

    /// Opens a stepping stream over a caller-supplied [`StreamStep`].
    ///
    /// The general hook a native model uses to expose its own stepper (e.g. a
    /// Whisper incremental-decode stepper wrapping
    /// [`DecodeStepper`](crate::decode::DecodeStepper), M1-08f) without
    /// `vokra-core` knowing the model.
    pub fn open_step_stream(&self, step: Box<dyn StreamStep + Send>) -> Result<Stream> {
        Ok(self.new_stream(Some(step)))
    }

    /// Number of currently open streams on this session.
    pub fn active_stream_count(&self) -> u64 {
        self.inner.active_streams.load(Ordering::Relaxed)
    }
}

impl Stream {
    /// Identifier of this stream, unique within its originating session.
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Per-stream state handle (see [`StreamState`]).
    pub fn state(&self) -> &StreamState {
        &self.state
    }

    /// The stepper's fixed frame length, or `None` (no stepper, or a stepper
    /// that frames internally).
    pub fn frame_len(&self) -> Option<usize> {
        self.step.as_ref().and_then(|s| s.frame_len())
    }

    fn stepper(&mut self) -> Result<&mut (dyn StreamStep + Send)> {
        match self.step.as_deref_mut() {
            Some(s) => Ok(s),
            None => Err(VokraError::NotImplemented(
                "stream has no stepper; open one via open_vad_stepper / open_step_stream",
            )),
        }
    }

    /// Feeds an arbitrary-length `samples` chunk and returns the events it
    /// produced (chunk-based pattern, FR-ST-01). Ergonomic single-thread path;
    /// does not touch the ring.
    ///
    /// # Errors
    ///
    /// [`VokraError::NotImplemented`] on a bare stream, or any error from the
    /// stepper.
    pub fn step_chunk(&mut self, samples: &[f32]) -> Result<Vec<StreamEvent>> {
        let mut out: Vec<StreamEvent> = Vec::new();
        self.stepper()?.step_chunk(samples, &mut out)?;
        Ok(out)
    }

    /// Feeds exactly one frame and returns its events (frame-by-frame pattern,
    /// FR-ST-01). Rejects a wrong-length frame when the stepper declares a fixed
    /// [`frame_len`](Self::frame_len).
    ///
    /// # Errors
    ///
    /// [`VokraError::NotImplemented`] on a bare stream,
    /// [`VokraError::InvalidArgument`] on a wrong-length frame, or any stepper
    /// error.
    pub fn step_frame(&mut self, frame: &[f32]) -> Result<Vec<StreamEvent>> {
        let mut out: Vec<StreamEvent> = Vec::new();
        self.stepper()?.step_frame(frame, &mut out)?;
        Ok(out)
    }

    /// Feeds a chunk, emitting its events straight into the lock-free ring, and
    /// returns the number enqueued (FR-ST-02). Move the `Stream` onto the audio
    /// thread and call this; a persistently full ring surfaces as a returned
    /// count below the number of events the step produced (reject-on-full
    /// backpressure). Hot-path allocation-free (FR-EX-05 / NFR-RL-08).
    ///
    /// # Errors
    ///
    /// [`VokraError::NotImplemented`] on a bare stream, or any stepper error.
    pub fn push(&mut self, samples: &[f32]) -> Result<usize> {
        // Split the borrows so the stepper and the producer are disjoint.
        let step = match self.step.as_deref_mut() {
            Some(s) => s,
            None => {
                return Err(VokraError::NotImplemented(
                    "stream has no stepper; open one via open_vad_stepper / open_step_stream",
                ));
            }
        };
        let mut sink = CountingProducer {
            producer: &mut self.producer,
            count: 0,
        };
        step.step_chunk(samples, &mut sink)?;
        Ok(sink.count)
    }

    /// Drains up to `out.len()` events from this stream's own ring (same-thread
    /// use). Returns 0 once [`take_poller`](Self::take_poller) has moved the
    /// consumer out. Non-blocking.
    pub fn poll_into(&mut self, out: &mut [StreamEvent]) -> usize {
        match self.consumer.as_mut() {
            Some(c) => drain_decoded(c, out),
            None => 0,
        }
    }

    /// Pops one event from this stream's own ring, or `None` (empty / poller
    /// taken). Non-blocking.
    pub fn poll_one(&mut self) -> Option<StreamEvent> {
        pop_decoded(self.consumer.as_mut()?)
    }

    /// Hands out the consumer as a `Send` [`EventPoller`] exactly once (the
    /// `Option` enforces the single-consumer discipline). After this, the
    /// `Stream` is the producer half — move it to the audio thread and call
    /// [`push`](Self::push); move the poller to the main thread and
    /// [`poll`](EventPoller::poll).
    pub fn take_poller(&mut self) -> Option<EventPoller> {
        self.consumer
            .take()
            .map(|consumer| EventPoller { consumer })
    }

    /// Resets the stepper to its initial state and discards any ring backlog, so
    /// a fresh run reproduces the first run bit-for-bit. No-op stepper part on a
    /// bare stream.
    pub fn reset(&mut self) {
        if let Some(step) = self.step.as_mut() {
            step.reset();
        }
        if let Some(c) = self.consumer.as_mut() {
            while c.pop().is_some() {}
        }
    }
}

impl EventPoller {
    /// Drains up to `out.len()` events, returning the count written.
    /// Wait-free and non-blocking (returns 0 promptly on an empty ring).
    pub fn poll(&mut self, out: &mut [StreamEvent]) -> usize {
        drain_decoded(&mut self.consumer, out)
    }

    /// Pops one event, or `None` if the ring is currently empty. Non-blocking.
    pub fn poll_one(&mut self) -> Option<StreamEvent> {
        pop_decoded(&mut self.consumer)
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        self.session.active_streams.fetch_sub(1, Ordering::Relaxed);
    }
}

// ---------------------------------------------------------------------------
// Compile-time Send / Sync verification (FR-API-03, the M1-08 "Send/Sync
// 制約のコンパイル時検証" deliverable). This block fails to build if any type
// loses its required thread-safety bound — that failure *is* the test.
// ---------------------------------------------------------------------------
const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    const fn assert_send<T: Send>() {}

    // Session is immutable + shareable across threads (FR-API-03).
    assert_send_sync::<Session>();
    // Stream is Send (moved to a worker) but intentionally not Sync.
    assert_send::<Stream>();
    // The cross-thread event path is Send on both ends.
    assert_send::<EventPoller>();
    assert_send::<RingProducer>();
    assert_send::<RingConsumer>();
    // The wire payloads are trivially thread-safe PODs.
    assert_send_sync::<RawEvent>();
    assert_send_sync::<StreamEvent>();
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::tests::TempModelFile;
    use std::sync::Barrier;

    fn session(tag: &str) -> (TempModelFile, Session) {
        let file = TempModelFile::new(tag);
        let session = Session::from_file(&file.0).build().expect("session builds");
        (file, session)
    }

    /// A deterministic synthetic stepper: each input sample completes one frame,
    /// emitting a `SpeechProb` whose `frame_index` is a monotonic counter and
    /// whose `prob` mirrors the index. An internal oracle (no model).
    struct RampStep {
        next: u32,
    }
    impl RampStep {
        fn new() -> Self {
            Self { next: 0 }
        }
    }
    impl StreamStep for RampStep {
        fn step_chunk(&mut self, samples: &[f32], sink: &mut dyn EventSink) -> Result<()> {
            for _ in samples {
                let frame_index = self.next;
                self.next += 1;
                let _ = sink.emit(StreamEvent::SpeechProb {
                    frame_index,
                    prob: frame_index as f32,
                });
            }
            Ok(())
        }
        fn reset(&mut self) {
            self.next = 0;
        }
    }

    /// A fixed-frame stepper (frame_len == 4) that emits one token per frame; it
    /// exercises the frame-by-frame validation path.
    struct FixedFrameStep;
    impl StreamStep for FixedFrameStep {
        fn step_chunk(&mut self, samples: &[f32], sink: &mut dyn EventSink) -> Result<()> {
            let _ = sink.emit(StreamEvent::Token {
                id: samples.len() as u32,
                flags: 0,
            });
            Ok(())
        }
        fn reset(&mut self) {}
        fn frame_len(&self) -> Option<usize> {
            Some(4)
        }
    }

    // ---- M0 lifecycle (carried over unchanged) -----------------------------

    #[test]
    fn open_and_drop_lifecycle() {
        let (_file, session) = session("stream-lifecycle");
        assert_eq!(session.active_stream_count(), 0);
        {
            let stream = session.open_stream().expect("stream opens");
            let _ = stream.state();
            assert_eq!(session.active_stream_count(), 1);
        }
        assert_eq!(session.active_stream_count(), 0);
    }

    #[test]
    fn multiple_streams_have_unique_ids_and_release_independently() {
        let (_file, session) = session("stream-multi");
        let s0 = session.open_stream().expect("s0");
        let s1 = session.open_stream().expect("s1");
        let s2 = session.open_stream().expect("s2");
        assert_eq!(session.active_stream_count(), 3);
        assert_ne!(s0.id(), s1.id());
        assert_ne!(s1.id(), s2.id());
        assert_ne!(s0.id(), s2.id());

        drop(s1);
        assert_eq!(session.active_stream_count(), 2);
        drop(s0);
        assert_eq!(session.active_stream_count(), 1);
        drop(s2);
        assert_eq!(session.active_stream_count(), 0);
    }

    #[test]
    fn stream_id_is_not_reused_after_drop() {
        let (_file, session) = session("stream-id-reuse");
        let s0 = session.open_stream().expect("s0");
        assert_eq!(s0.id(), 0);
        drop(s0);
        assert_eq!(session.active_stream_count(), 0);
        let s1 = session.open_stream().expect("s1");
        assert_eq!(s1.id(), 1, "ids are monotonic, never reused");
    }

    // ---- M1-08 stepping ----------------------------------------------------

    #[test]
    fn bare_stream_rejects_stepping() {
        let (_file, session) = session("bare-step");
        let mut s = session.open_stream().expect("bare stream");
        assert!(matches!(
            s.step_chunk(&[0.0; 4]),
            Err(VokraError::NotImplemented(_))
        ));
        assert!(matches!(
            s.push(&[0.0; 4]),
            Err(VokraError::NotImplemented(_))
        ));
    }

    #[test]
    fn step_chunk_returns_events_ergonomically() {
        let (_file, session) = session("step-chunk");
        let mut s = session
            .open_step_stream(Box::new(RampStep::new()))
            .expect("stepping stream");
        let evs = s.step_chunk(&[0.0; 3]).expect("step");
        assert_eq!(evs.len(), 3);
        assert_eq!(
            evs[0],
            StreamEvent::SpeechProb {
                frame_index: 0,
                prob: 0.0
            }
        );
    }

    #[test]
    fn step_frame_validates_length_and_frame_vs_chunk_agree() {
        let (_file, session) = session("step-frame");
        let mut s = session
            .open_step_stream(Box::new(FixedFrameStep))
            .expect("stepping stream");
        assert_eq!(s.frame_len(), Some(4));
        // Wrong length is rejected.
        assert!(matches!(
            s.step_frame(&[0.0; 3]),
            Err(VokraError::InvalidArgument(_))
        ));
        // A correct frame via step_frame equals the same via step_chunk.
        let via_frame = s.step_frame(&[0.0; 4]).expect("frame");
        let via_chunk = s.step_chunk(&[0.0; 4]).expect("chunk");
        assert_eq!(via_frame, via_chunk);
    }

    #[test]
    fn reset_reproduces_the_first_run() {
        let (_file, session) = session("step-reset");
        let mut s = session
            .open_step_stream(Box::new(RampStep::new()))
            .expect("stepping stream");
        let first = s.step_chunk(&[0.0; 5]).expect("first run");
        s.reset();
        let second = s.step_chunk(&[0.0; 5]).expect("second run");
        assert_eq!(first, second, "reset rewinds to a bit-identical first run");
    }

    #[test]
    fn push_then_same_thread_poll_round_trips_via_the_ring() {
        let (_file, session) = session("push-poll");
        let mut s = session
            .open_step_stream(Box::new(RampStep::new()))
            .expect("stepping stream");
        let enqueued = s.push(&[0.0; 6]).expect("push");
        assert_eq!(enqueued, 6, "all six events fit the ring");
        let mut buf = [StreamEvent::Token { id: 0, flags: 0 }; 8];
        let n = s.poll_into(&mut buf);
        assert_eq!(n, 6);
        for (i, e) in buf[..n].iter().enumerate() {
            assert_eq!(
                *e,
                StreamEvent::SpeechProb {
                    frame_index: i as u32,
                    prob: i as f32
                }
            );
        }
        // Nothing left.
        assert_eq!(s.poll_into(&mut buf), 0);
    }

    #[test]
    fn take_poller_is_single_shot() {
        let (_file, session) = session("take-poller");
        let mut s = session
            .open_step_stream(Box::new(RampStep::new()))
            .expect("stepping stream");
        assert!(s.take_poller().is_some());
        assert!(s.take_poller().is_none(), "consumer handed out only once");
        // Same-thread poll now returns 0 (consumer moved out).
        s.push(&[0.0; 2]).expect("push");
        let mut buf = [StreamEvent::Token { id: 0, flags: 0 }; 4];
        assert_eq!(s.poll_into(&mut buf), 0);
    }

    /// THE stream-level concurrency oracle: the producer `Stream` is moved onto
    /// a spawned thread pushing a contiguous ramp in irregular chunks; the main
    /// thread polls the `EventPoller` until every event arrives, asserting the
    /// drained frame-index sequence is exactly the monotonic `0..N`. The `push`
    /// count is asserted equal to the chunk length on every call, so any ring
    /// overflow (a dropped event) would fail deterministically. `N` stays below
    /// the ring capacity so a lagging consumer never overflows. `Barrier` forces
    /// overlap.
    #[test]
    fn cross_thread_push_poll_is_lossless_and_ordered() {
        let (_file, session) = session("concurrency");
        let mut stream = session
            .open_step_stream(Box::new(RampStep::new()))
            .expect("stepping stream");
        let mut poller = stream.take_poller().expect("poller");

        let n: usize = 3000; // < DEFAULT_RING_CAPACITY (4096)
        let barrier = Arc::new(Barrier::new(2));
        let b2 = Arc::clone(&barrier);

        let producer = std::thread::spawn(move || {
            b2.wait();
            let chunks = [1usize, 7, 3, 50, 2, 128, 11];
            let mut pushed = 0;
            let mut ci = 0;
            while pushed < n {
                let take = chunks[ci % chunks.len()].min(n - pushed);
                ci += 1;
                let dummy = vec![0.0f32; take];
                let enq = stream.push(&dummy).expect("push");
                assert_eq!(enq, take, "no event dropped (ring never overflowed)");
                pushed += take;
            }
            stream // keep the producer alive until join
        });

        barrier.wait();
        let mut got: Vec<u32> = Vec::with_capacity(n);
        let mut buf = [StreamEvent::Token { id: 0, flags: 0 }; 64];
        while got.len() < n {
            let m = poller.poll(&mut buf);
            for e in &buf[..m] {
                match e {
                    StreamEvent::SpeechProb { frame_index, .. } => got.push(*frame_index),
                    other => panic!("unexpected event {other:?}"),
                }
            }
        }
        let _stream = producer.join().expect("producer joins");

        assert_eq!(got.len(), n);
        for (i, &f) in got.iter().enumerate() {
            assert_eq!(
                f as usize, i,
                "frame index {i} arrived in order, no gap/dup"
            );
        }
    }

    #[test]
    fn stream_moves_to_a_thread_and_reports_events() {
        // A minimal Send smoke test: move a stepping Stream into a thread, run
        // it there, and read events back through the joined handle.
        let (_file, session) = session("stream-send");
        let stream = session
            .open_step_stream(Box::new(RampStep::new()))
            .expect("stepping stream");
        let count = std::thread::spawn(move || {
            let mut s = stream;
            s.step_chunk(&[0.0; 4]).expect("step on worker").len()
        })
        .join()
        .expect("worker joins");
        assert_eq!(count, 4);
    }
}
