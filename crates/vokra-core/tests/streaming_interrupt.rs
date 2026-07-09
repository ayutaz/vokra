//! M3-14 T06 — barge-in integration test.
//!
//! Anchors the WP completion criterion (docs/tickets/m3/M3-14-barge-in.md,
//! milestones.md §7.2 M3-14): `stream.interrupt()` on a mid-generation stream
//! flushes the output *immediately* and the next `push` is accepted in a
//! clean state, whether the interrupt was raised on the owning thread or from
//! a cross-thread [`InterruptHandle`].
//!
//! The oracle is *internal* — the four semantics from the ticket
//! (a)–(d) are verified against a synthetic [`StreamStep`] that carries a
//! deterministic monotonic frame counter, so a stepper reset is observable
//! (`frame_index` restarts at 0) without needing a real Whisper/piper
//! checkpoint. TTS / ASR *real-model* interrupt (piper-plus native / Kokoro /
//! Whisper) is exercised at the model layer and gated on GGUF fixtures under
//! `tests/parity/*`; this file is the model-independent core.
//!
//! Zero-dep NFR-DS-02 preserved: `std` + `vokra-core` only, no fixture files.

use std::sync::Arc;
use std::sync::Barrier;

use vokra_core::{BackendKind, EventSink, Result as VokraResult, Session, StreamEvent, StreamStep};

/// Deterministic synthetic stepper: one input sample completes one frame,
/// emitting a [`StreamEvent::SpeechProb`] with a monotonic frame counter.
/// The counter is the observable that pins the (c) stepper-state-reset
/// semantic — after `interrupt` the next frame must be index 0.
struct Ramp {
    next: u32,
}

impl Ramp {
    fn new() -> Self {
        Self { next: 0 }
    }
}

impl StreamStep for Ramp {
    fn step_chunk(&mut self, samples: &[f32], sink: &mut dyn EventSink) -> VokraResult<()> {
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

/// Materialises an in-memory GGUF fixture so `Session::from_file` succeeds
/// without disturbing any committed parity fixtures. The Session backend is
/// unused: the stepper here is synthetic and does not touch a real backend.
struct TempModelFile(std::path::PathBuf);

impl TempModelFile {
    fn new(tag: &str) -> Self {
        // Minimum-valid GGUF: magic, version 3, tensor_count=0, kv_count=0.
        // Sanctioned by the `vokra-core::gguf` reader for a container-only
        // load (no tensors, no metadata). Kept identical to the pattern
        // `session::tests::TempModelFile` uses so the two paths agree.
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF"); // magic
        bytes.extend_from_slice(&3u32.to_le_bytes()); // version
        bytes.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
        bytes.extend_from_slice(&0u64.to_le_bytes()); // kv_count
        let path = std::env::temp_dir().join(format!(
            "vokra-m3-14-integration-{tag}-{}.gguf",
            std::process::id()
        ));
        std::fs::write(&path, &bytes).expect("write temp gguf");
        Self(path)
    }
}

impl Drop for TempModelFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// Builds a stepping Stream over a synthetic Ramp stepper, so the integration
/// tests are hermetic and hexadecimally reproducible (no models loaded).
fn session_and_stream(tag: &str) -> (TempModelFile, vokra_core::Stream) {
    let file = TempModelFile::new(tag);
    let session: Session = Session::from_file(&file.0)
        .with_backend(BackendKind::Cpu)
        .expect("session builds");
    let stream = session
        .open_step_stream(Box::new(Ramp::new()))
        .expect("stepping stream opens");
    (file, stream)
}

/// (i) Same-thread `stream.interrupt()` mid-generation: partial output is
/// discarded, ring is drained, stepper is reset — the direct check for
/// milestones.md §7.2 M3-14 completion criterion.
#[test]
fn interrupt_mid_generation_flushes_output_and_restarts_cleanly() {
    let (_file, mut s) = session_and_stream("tts-mid-generation");

    // Push 20 samples to simulate a long-form generation, do not poll — the
    // 20 events sit on the ring waiting to be consumed.
    assert_eq!(s.push(&[0.0; 20]).expect("prime push"), 20);

    // Interrupt: expected to (a) discard partial output, (b) drain ring,
    // (c) reset stepper, (d) clear flag.
    s.interrupt().expect("interrupt succeeds");
    assert!(!s.is_interrupt_pending(), "flag cleared by interrupt()");

    // (b) ring is empty right after interrupt.
    let mut buf = [StreamEvent::Token { id: 0, flags: 0 }; 32];
    assert_eq!(
        s.poll_into(&mut buf),
        0,
        "no stale events after interrupt (b)"
    );

    // (c/d) next push starts a clean run — frame index restarts at 0.
    assert_eq!(s.push(&[0.0; 5]).expect("post-interrupt push"), 5);
    let n = s.poll_into(&mut buf);
    assert_eq!(n, 5);
    for (i, ev) in buf[..n].iter().enumerate() {
        assert_eq!(
            *ev,
            StreamEvent::SpeechProb {
                frame_index: i as u32,
                prob: i as f32
            },
            "post-interrupt frame {i} is bit-exact with a fresh run"
        );
    }
}

/// (ii) Cross-thread barge-in via `InterruptHandle`: a spawned worker holding
/// only the handle (no `Stream` access) raises the flag; the owning thread
/// picks it up on its next `push`. Verifies the lock-free contract.
#[test]
fn cross_thread_handle_interrupt_is_seen_on_next_push() {
    let (_file, mut s) = session_and_stream("cross-thread-handle");
    let handle = s.interrupt_handle();

    // Prime the ring on the owning thread — these events must NOT survive.
    assert_eq!(s.push(&[0.0; 12]).expect("prime push"), 12);

    let barrier = Arc::new(Barrier::new(2));
    let b2 = Arc::clone(&barrier);
    let worker = std::thread::spawn(move || {
        b2.wait();
        // The worker only holds the handle; it cannot touch the Stream.
        handle.interrupt();
    });
    barrier.wait();
    worker.join().expect("worker joins");

    // The handle's Release-store is now visible via the Stream's Acquire-load
    // in `check_interrupt`.
    assert!(s.is_interrupt_pending(), "audio thread sees the request");

    // Next push handles the interrupt and processes 3 new samples.
    assert_eq!(s.push(&[0.0; 3]).expect("post-signal push"), 3);
    assert!(!s.is_interrupt_pending(), "handler cleared the flag");

    let mut buf = [StreamEvent::Token { id: 0, flags: 0 }; 32];
    let n = s.poll_into(&mut buf);
    assert_eq!(n, 3, "12 primed events were flushed by the interrupt");
    assert_eq!(
        buf[0],
        StreamEvent::SpeechProb {
            frame_index: 0,
            prob: 0.0
        },
        "post-interrupt stepper started fresh"
    );
}

/// (iii) The repeat oracle: `interrupt → push → poll` ten times yields the
/// same events every cycle, so barge-in leaves no state behind between
/// cycles.
#[test]
fn ten_interrupt_cycles_leave_no_state_leak() {
    let (_file, mut s) = session_and_stream("cycles");
    let mut buf = [StreamEvent::Token { id: 0, flags: 0 }; 16];

    // Baseline: what a fresh push emits.
    assert_eq!(s.push(&[0.0; 4]).expect("baseline"), 4);
    let n = s.poll_into(&mut buf);
    let baseline: Vec<StreamEvent> = buf[..n].to_vec();
    s.interrupt().expect("interrupt");

    for cycle in 0..10 {
        assert_eq!(
            s.push(&[0.0; 4]).expect("cycle push"),
            4,
            "cycle {cycle} push count"
        );
        let n = s.poll_into(&mut buf);
        let got: Vec<StreamEvent> = buf[..n].to_vec();
        assert_eq!(got, baseline, "cycle {cycle} equals baseline");
        s.interrupt().expect("cycle interrupt");
    }
}

/// (iv) Combined poller-side drain: when the consumer half was moved to an
/// `EventPoller`, the `Stream::interrupt` cannot touch the ring — the poller
/// must call `EventPoller::drain_all` to complete the (b) semantics.
#[test]
fn interrupt_with_poller_detached_uses_poller_drain_all() {
    let (_file, mut s) = session_and_stream("with-poller");
    let mut poller = s.take_poller().expect("poller taken");

    // Prime 15 events, then interrupt.
    assert_eq!(s.push(&[0.0; 15]).expect("prime"), 15);
    s.interrupt().expect("interrupt with detached poller");
    assert!(!s.is_interrupt_pending(), "flag cleared regardless");

    // Ring still has the 15 stale events — the Stream side cannot reach them.
    let discarded = poller.drain_all();
    assert_eq!(discarded, 15, "poller.drain_all discards every stale event");

    // Post-interrupt push starts a fresh run (stepper reset).
    assert_eq!(s.push(&[0.0; 6]).expect("post-interrupt"), 6);
    let mut buf = [StreamEvent::Token { id: 0, flags: 0 }; 32];
    let n = poller.poll(&mut buf);
    assert_eq!(n, 6);
    assert_eq!(
        buf[0],
        StreamEvent::SpeechProb {
            frame_index: 0,
            prob: 0.0
        },
        "stepper was still reset by the interrupt (c)"
    );
}
