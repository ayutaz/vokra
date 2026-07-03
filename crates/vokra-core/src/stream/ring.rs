//! Lock-free single-producer / single-consumer (SPSC) event ring (FR-ST-02).
//!
//! This is the native reimplementation of the `rtrb` / `ringbuf` external crate
//! mandated by NFR-DS-02 (zero external dependencies): a bounded, wait-free ring
//! whose whole payload lives in [`AtomicU64`] cells, so there is **no data race
//! and no `unsafe`** — it compiles under `vokra-core`'s workspace
//! `unsafe_code = "deny"`.
//!
//! # Design
//!
//! - Capacity is rounded up to a power of two so index wrapping is a mask
//!   (`idx = cursor & mask`).
//! - The producer cursor (`tail`) and consumer cursor (`head`) each sit in a
//!   64-byte-aligned [`CachePad`] to avoid false sharing on the hot path.
//! - Each slot is a [`Cell`] of two `AtomicU64` words — a [`RawEvent`] is a
//!   16-byte POD that covers every M1 streaming event (see
//!   [`super::event::StreamEvent`]).
//! - [`channel`] pre-allocates the whole buffer, so `try_push` / `pop` never
//!   allocate (FR-EX-05, NFR-RL-08 hot-path malloc-free).
//!
//! # Concurrency contract
//!
//! Exactly one [`RingProducer`] and one [`RingConsumer`] are handed out by
//! [`channel`]; neither is [`Clone`], so the SPSC discipline is enforced by the
//! type system. Both are [`Send`] (move the producer to the audio-callback
//! thread, the consumer to the polling thread). The Release/Acquire pairing on
//! the two cursors publishes each slot write before the matching read and frees
//! a slot before it is reused — the textbook Lamport/Vyukov SPSC ordering.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// A 16-byte plain-old-data event payload carried through the ring.
///
/// Two `u64` words are enough to encode every M1 [`StreamEvent`](super::event::StreamEvent)
/// (a tagged pair of a `u32` index/id and a `u32`/`f32` value); the codec lives
/// in [`super::event`]. `RawEvent` is `Copy`, so a push/pop moves 16 bytes with
/// no allocation and no `Drop`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RawEvent {
    /// First payload word (tag + primary field, per the event codec).
    pub w0: u64,
    /// Second payload word (value field, per the event codec).
    pub w1: u64,
}

/// A cursor padded to a cache line so the producer's `tail` and the consumer's
/// `head` never share one, eliminating false sharing under contention.
#[repr(align(64))]
struct CachePad(AtomicUsize);

/// One ring slot: a `RawEvent`'s two words held as atomics so the read on the
/// consumer thread and the write on the producer thread are never a data race.
struct Cell {
    w0: AtomicU64,
    w1: AtomicU64,
}

/// The shared ring storage behind the producer/consumer [`Arc`]s.
struct Ring {
    /// `cap` slots, `cap` a power of two.
    buf: Box<[Cell]>,
    /// `cap - 1`, the index mask.
    mask: usize,
    /// Consumer cursor (monotonically increasing, wrapping `usize`).
    head: CachePad,
    /// Producer cursor (monotonically increasing, wrapping `usize`).
    tail: CachePad,
}

/// The producing half of a [`channel`]: the only handle allowed to
/// [`try_push`](RingProducer::try_push). `Send`, not `Clone` (single producer).
pub struct RingProducer {
    ring: Arc<Ring>,
}

/// The consuming half of a [`channel`]: the only handle allowed to
/// [`pop`](RingConsumer::pop). `Send`, not `Clone` (single consumer).
pub struct RingConsumer {
    ring: Arc<Ring>,
}

/// Returned by [`RingProducer::try_push`] when the ring is full; carries the
/// event back to the caller so **no event is silently dropped** at the ring
/// level (reject-on-full backpressure, see the `decisions_to_flag`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RingFull(pub RawEvent);

/// Creates a bounded SPSC ring, returning its single producer and consumer.
///
/// `capacity` is rounded up to the next power of two (minimum 1). The whole
/// buffer is allocated here, so the hot path never allocates.
///
/// ```
/// use vokra_core::stream::{channel, RawEvent};
///
/// let (mut tx, mut rx) = channel(4);
/// tx.try_push(RawEvent { w0: 7, w1: 0 }).unwrap();
/// assert_eq!(rx.pop(), Some(RawEvent { w0: 7, w1: 0 }));
/// assert_eq!(rx.pop(), None);
/// ```
#[must_use]
pub fn channel(capacity: usize) -> (RingProducer, RingConsumer) {
    let cap = capacity.max(1).next_power_of_two();
    let buf: Box<[Cell]> = (0..cap)
        .map(|_| Cell {
            w0: AtomicU64::new(0),
            w1: AtomicU64::new(0),
        })
        .collect();
    let ring = Arc::new(Ring {
        buf,
        mask: cap - 1,
        head: CachePad(AtomicUsize::new(0)),
        tail: CachePad(AtomicUsize::new(0)),
    });
    (
        RingProducer {
            ring: Arc::clone(&ring),
        },
        RingConsumer { ring },
    )
}

impl RingProducer {
    /// Number of slots in the ring (the rounded-up power of two).
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.ring.buf.len()
    }

    /// Enqueues one event, or hands it back in [`RingFull`] if the ring is full.
    ///
    /// Wait-free and allocation-free. The producer is the sole writer of `tail`
    /// (Relaxed load), reads the consumer's `head` with Acquire to test
    /// fullness, writes the slot, then publishes it with a Release store of the
    /// bumped `tail`.
    pub fn try_push(&mut self, ev: RawEvent) -> Result<(), RingFull> {
        let tail = self.ring.tail.0.load(Ordering::Relaxed);
        let head = self.ring.head.0.load(Ordering::Acquire);
        if tail.wrapping_sub(head) == self.ring.buf.len() {
            return Err(RingFull(ev));
        }
        let cell = &self.ring.buf[tail & self.ring.mask];
        cell.w0.store(ev.w0, Ordering::Relaxed);
        cell.w1.store(ev.w1, Ordering::Relaxed);
        self.ring
            .tail
            .0
            .store(tail.wrapping_add(1), Ordering::Release);
        Ok(())
    }
}

impl RingConsumer {
    /// Number of slots in the ring (the rounded-up power of two).
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.ring.buf.len()
    }

    /// Dequeues the oldest event, or `None` if the ring is empty.
    ///
    /// Wait-free and allocation-free. The consumer is the sole writer of `head`
    /// (Relaxed load), reads the producer's `tail` with Acquire to test
    /// emptiness, reads the slot, then frees it with a Release store of the
    /// bumped `head`.
    pub fn pop(&mut self) -> Option<RawEvent> {
        let head = self.ring.head.0.load(Ordering::Relaxed);
        let tail = self.ring.tail.0.load(Ordering::Acquire);
        if head == tail {
            return None;
        }
        let cell = &self.ring.buf[head & self.ring.mask];
        let w0 = cell.w0.load(Ordering::Relaxed);
        let w1 = cell.w1.load(Ordering::Relaxed);
        self.ring
            .head
            .0
            .store(head.wrapping_add(1), Ordering::Release);
        Some(RawEvent { w0, w1 })
    }

    /// Drains up to `out.len()` events into `out`, returning the count written.
    /// Non-blocking: stops early at the first empty state.
    pub fn drain_into(&mut self, out: &mut [RawEvent]) -> usize {
        let mut n = 0;
        while n < out.len() {
            match self.pop() {
                Some(ev) => {
                    out[n] = ev;
                    n += 1;
                }
                None => break,
            }
        }
        n
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Barrier;

    fn ev(i: u64) -> RawEvent {
        RawEvent { w0: i, w1: !i }
    }

    #[test]
    fn capacity_is_rounded_up_to_power_of_two() {
        let (tx, rx) = channel(5);
        assert_eq!(tx.capacity(), 8);
        assert_eq!(rx.capacity(), 8);
        // A zero request still yields a usable one-slot ring.
        let (tx0, _rx0) = channel(0);
        assert_eq!(tx0.capacity(), 1);
    }

    #[test]
    fn single_thread_fifo_is_ordered_and_deterministic() {
        let (mut tx, mut rx) = channel(16);
        for i in 0..10 {
            tx.try_push(ev(i)).expect("space available");
        }
        for i in 0..10 {
            assert_eq!(rx.pop(), Some(ev(i)), "FIFO order, no reorder");
        }
        assert_eq!(rx.pop(), None, "drained to empty");
    }

    #[test]
    fn full_returns_the_bounced_event_and_empty_returns_none() {
        let (mut tx, mut rx) = channel(4); // cap 4
        for i in 0..4 {
            tx.try_push(ev(i)).expect("fits");
        }
        // The 5th push bounces the exact event back — no data loss.
        let bounced = tx.try_push(ev(99)).unwrap_err();
        assert_eq!(bounced, RingFull(ev(99)));
        // Draining frees slots; then a push succeeds again.
        assert_eq!(rx.pop(), Some(ev(0)));
        tx.try_push(ev(99)).expect("slot freed");
        assert_eq!(rx.pop(), Some(ev(1)));
    }

    #[test]
    fn wraps_past_the_mask_boundary() {
        // Repeatedly fill/drain so the cursors advance well past `cap`, forcing
        // the index to wrap through the mask many times.
        let (mut tx, mut rx) = channel(4);
        let mut expect = 0u64;
        let mut next = 0u64;
        for _round in 0..1000 {
            while tx.try_push(ev(next)).is_ok() {
                next += 1;
            }
            while let Some(got) = rx.pop() {
                assert_eq!(got, ev(expect));
                expect += 1;
            }
        }
        assert_eq!(expect, next, "every pushed event was popped exactly once");
    }

    #[test]
    fn drain_into_reports_count_and_clamps() {
        let (mut tx, mut rx) = channel(16);
        for i in 0..6 {
            tx.try_push(ev(i)).unwrap();
        }
        let mut buf = [RawEvent { w0: 0, w1: 0 }; 4];
        assert_eq!(rx.drain_into(&mut buf), 4, "clamped to out.len()");
        assert_eq!(rx.drain_into(&mut buf), 2, "remaining two");
        assert_eq!(rx.drain_into(&mut buf), 0, "empty");
    }

    /// THE deterministic concurrency oracle: a spawned producer pushes the
    /// contiguous counter `0..N` (retrying on a full ring — a structural,
    /// lock-free spin, never a blocking wait), the consumer pops them, and we
    /// assert the popped `w0` values are exactly the monotonic `0..N` with no
    /// gap / dup / reorder. That invariant holds for *every* thread
    /// interleaving, so the test is deterministic despite the threads. A tiny
    /// ring plus a [`Barrier`] forces real contention and many wraps.
    #[test]
    fn spsc_concurrent_transfer_is_lossless_and_ordered() {
        let (mut tx, mut rx) = channel(64);
        let n = 100_000u64;
        let barrier = Arc::new(Barrier::new(2));
        let b2 = Arc::clone(&barrier);

        let producer = std::thread::spawn(move || {
            b2.wait();
            for i in 0..n {
                let e = ev(i);
                // Reject-on-full ⇒ retry until the consumer frees a slot.
                while tx.try_push(e).is_err() {
                    std::hint::spin_loop();
                }
            }
        });

        barrier.wait();
        let mut got = 0u64;
        while got < n {
            match rx.pop() {
                Some(e) => {
                    assert_eq!(e.w0, got, "no gap / dup / reorder at position {got}");
                    assert_eq!(e.w1, !got, "payload word travels intact");
                    got += 1;
                }
                None => std::hint::spin_loop(),
            }
        }
        producer.join().expect("producer thread joins");
        assert_eq!(got, n);
    }

    #[test]
    fn producer_and_consumer_are_send() {
        const fn assert_send<T: Send>() {}
        assert_send::<RingProducer>();
        assert_send::<RingConsumer>();
        assert_send::<RawEvent>();
    }
}
