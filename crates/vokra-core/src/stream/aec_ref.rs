//! Time-tagged far-end (playback) reference-signal queue for the `aec` op
//! (M4-03, FR-OP-60): the "参照信号の時間タグ付き queue を runtime レベルで
//! 管理" half of the AEC work package. The adaptive filter itself lives in
//! `vokra-ops::aec` (the crate edge runs `vokra-ops → vokra-core`, so the
//! queue sits here and the filter consumes it — ADR M4-03 §D-(c)).
//!
//! # What it carries
//!
//! Far-end PCM — the audio *this* device is playing back (TTS / S2S output)
//! — tagged with its **playback sample position** (`u64`, monotonically
//! increasing; sample-clock, not wall-clock — ADR M4-03 §D-(d)). The AEC
//! consumer asks for the far-end window that is time-aligned with a mic
//! frame (`[pos, pos + frame)`) and gets a zero-filled gap wherever nothing
//! was played, with the gap made visible in [`AecRefWindowStatus`] (never a
//! silent fill — FR-EX-08).
//!
//! # Concurrency contract (FR-ST-02 discipline, inherited from [`super::ring`])
//!
//! Exactly one [`AecRefWriter`] and one [`AecRefReader`] are handed out by
//! [`aec_ref_queue`]; neither is `Clone`, so the SPSC discipline is enforced
//! by the type system. Both are `Send`: move the writer to the playback
//! callback thread, the reader to the inference thread. Cursors publish with
//! the same Release/Acquire pairing as the event ring.
//!
//! The whole buffer is allocated once in [`aec_ref_queue`]; `push` / `window`
//! never allocate (FR-EX-05, NFR-RL-08) and never block. A full ring rejects
//! (partial-accepts) instead of overwriting (reject-on-full backpressure,
//! never silent corruption).
//!
//! # Why not reuse the [`super::ring`] `RawEvent` ring
//!
//! The M1-08 event ring carries 16-byte POD events; a PCM block transfer
//! with per-sample time tags is type-level a different animal. What *is*
//! reused is the discipline: pre-allocation, reject-on-full, Release/Acquire
//! cursor pairing, SPSC by construction (spec M4-03-T02). Each slot here is
//! `{pos: AtomicU64, bits: AtomicU32}` — the f32 sample travels as its bit
//! pattern in an `AtomicU32` cell, which keeps this file `unsafe`-free under
//! the workspace `unsafe_code = "deny"` (NFR-RL-07: vokra-core stays at
//! unsafe 0).
//!
//! # Time-tag semantics
//!
//! - `push(pcm, playback_pos)` requires `playback_pos` ≥ the end of the
//!   previous push (backward / overlapping tags are an explicit
//!   [`VokraError::InvalidArgument`] — FR-EX-08). A *forward* gap is legal
//!   and means "nothing was played in between"; it costs no queue space and
//!   reads back as zeros.
//! - `window(pos, out)` consumes the queue up to `pos + out.len()` and
//!   reports coverage: [`Complete`](AecRefWindowStatus::Complete) /
//!   [`Partial`](AecRefWindowStatus::Partial) (gap zero-filled, `missing`
//!   counted) / [`Empty`](AecRefWindowStatus::Empty). Regions the reader has
//!   moved past are released (fixed memory over any run length); re-reading
//!   an already-consumed region reads back as missing.
//! - Clock drift / asynchronous-clock compensation is out of scope for
//!   M4-03 (a documented follow-up — ADR M4-03 §D-(d)); a constant playback
//!   offset is absorbed by the AEC filter tail (`filter_length`).

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};

use crate::error::{Result, VokraError};

/// One queue slot: an f32 PCM sample (as its bit pattern) plus the absolute
/// playback sample position it belongs to. Both fields are atomics so the
/// producer-side write and the consumer-side read are never a data race
/// (and the file needs no `unsafe`).
struct Slot {
    /// Absolute playback sample position of this sample.
    pos: AtomicU64,
    /// The f32 sample as `f32::to_bits`.
    bits: AtomicU32,
}

/// A cursor padded to a cache line so the producer's `tail` and the
/// consumer's `head` never share one (same layout trick as `ring.rs`).
#[repr(align(64))]
struct CachePad(AtomicUsize);

/// Shared storage behind the writer/reader [`Arc`]s.
struct RefRing {
    /// `cap` slots, `cap` a power of two.
    buf: Box<[Slot]>,
    /// `cap - 1`, the index mask.
    mask: usize,
    /// Consumer cursor (monotonically increasing, wrapping `usize`).
    head: CachePad,
    /// Producer cursor (monotonically increasing, wrapping `usize`).
    tail: CachePad,
    /// Sample rate of the PCM timeline; checked against `AecAttrs` by the
    /// `vokra-ops` AEC consumer (rate mismatch = explicit error, FR-EX-08).
    sample_rate: u32,
}

/// The producing half of an [`aec_ref_queue`]: the only handle allowed to
/// [`push`](AecRefWriter::push). `Send`, not `Clone` (single producer —
/// the playback-callback thread).
pub struct AecRefWriter {
    ring: Arc<RefRing>,
    /// The playback position one past the end of the previous push; the
    /// monotonicity gate (writer-local: only this half touches it).
    expected_next: Option<u64>,
}

/// The consuming half of an [`aec_ref_queue`]: the only handle allowed to
/// [`window`](AecRefReader::window). `Send`, not `Clone` (single consumer —
/// the inference thread).
pub struct AecRefReader {
    ring: Arc<RefRing>,
}

/// Coverage report of one [`AecRefReader::window`] call (FR-EX-08: a gap is
/// zero-filled *and* reported, never silently).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AecRefWindowStatus {
    /// Every sample of the window was covered by pushed far-end data.
    Complete,
    /// `missing` samples had no pushed data and were zero-filled.
    Partial {
        /// Number of window samples that had no far-end data.
        missing: usize,
    },
    /// No sample of the window was covered (nothing was played in this
    /// region, or the region was already consumed by an earlier window).
    Empty,
}

/// Creates a bounded SPSC far-end reference queue, returning its single
/// writer and reader.
///
/// `capacity_samples` is rounded up to the next power of two (minimum 1).
/// The whole buffer is allocated here, so the `push` / `window` hot paths
/// never allocate. `sample_rate` is carried as queue metadata for the AEC
/// consumer's rate cross-check and must be non-zero.
///
/// ```
/// use vokra_core::stream::{AecRefWindowStatus, aec_ref_queue};
///
/// let (mut tx, mut rx) = aec_ref_queue(1024, 16_000)?;
/// tx.push(&[0.25, -0.5], 100)?;
/// let mut win = [0.0f32; 2];
/// assert_eq!(rx.window(100, &mut win), AecRefWindowStatus::Complete);
/// assert_eq!(win, [0.25, -0.5]);
/// # Ok::<(), vokra_core::VokraError>(())
/// ```
///
/// # Errors
///
/// [`VokraError::InvalidArgument`] if `sample_rate == 0`.
pub fn aec_ref_queue(
    capacity_samples: usize,
    sample_rate: u32,
) -> Result<(AecRefWriter, AecRefReader)> {
    if sample_rate == 0 {
        return Err(VokraError::InvalidArgument(
            "aec_ref_queue: sample_rate must be non-zero".into(),
        ));
    }
    let cap = capacity_samples.max(1).next_power_of_two();
    let buf: Box<[Slot]> = (0..cap)
        .map(|_| Slot {
            pos: AtomicU64::new(0),
            bits: AtomicU32::new(0),
        })
        .collect();
    let ring = Arc::new(RefRing {
        buf,
        mask: cap - 1,
        head: CachePad(AtomicUsize::new(0)),
        tail: CachePad(AtomicUsize::new(0)),
        sample_rate,
    });
    Ok((
        AecRefWriter {
            ring: Arc::clone(&ring),
            expected_next: None,
        },
        AecRefReader { ring },
    ))
}

impl AecRefWriter {
    /// Number of slots in the ring (the rounded-up power of two).
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.ring.buf.len()
    }

    /// Sample rate of the queue's PCM timeline.
    #[must_use]
    pub fn sample_rate(&self) -> u32 {
        self.ring.sample_rate
    }

    /// Pushes a far-end chunk whose first sample plays at absolute position
    /// `playback_pos`, returning how many samples were accepted.
    ///
    /// Wait-free and allocation-free. Backpressure is reject-on-full: when
    /// the ring cannot take the whole chunk, the *prefix* that fits is
    /// accepted and its length returned (possibly 0). Retry the remainder as
    /// `push(&pcm[accepted..], playback_pos + accepted as u64)` after the
    /// reader has consumed a window.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] if `playback_pos` runs backward into
    /// (or overlaps) an already-pushed region — the time tag must be
    /// monotonic (FR-EX-08; a *forward* gap is legal and reads as zeros).
    pub fn push(&mut self, pcm: &[f32], playback_pos: u64) -> Result<usize> {
        if let Some(expected) = self.expected_next {
            if playback_pos < expected {
                return Err(VokraError::InvalidArgument(format!(
                    "aec_ref_queue: playback_pos {playback_pos} runs backward into the \
                     already-pushed region ending at {expected} (time tags must be monotonic)"
                )));
            }
        }
        if pcm.is_empty() {
            return Ok(0);
        }
        let tail = self.ring.tail.0.load(Ordering::Relaxed);
        let head = self.ring.head.0.load(Ordering::Acquire);
        let free = self.ring.buf.len() - tail.wrapping_sub(head);
        let n = pcm.len().min(free);
        for (i, &sample) in pcm[..n].iter().enumerate() {
            let slot = &self.ring.buf[tail.wrapping_add(i) & self.ring.mask];
            slot.pos.store(playback_pos + i as u64, Ordering::Relaxed);
            slot.bits.store(sample.to_bits(), Ordering::Relaxed);
        }
        self.ring
            .tail
            .0
            .store(tail.wrapping_add(n), Ordering::Release);
        self.expected_next = Some(playback_pos + n as u64);
        Ok(n)
    }
}

impl AecRefReader {
    /// Number of slots in the ring (the rounded-up power of two).
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.ring.buf.len()
    }

    /// Sample rate of the queue's PCM timeline (the AEC consumer checks this
    /// against its own `AecAttrs.sample_rate`; a mismatch is an explicit
    /// error on the `vokra-ops` side — FR-EX-08).
    #[must_use]
    pub fn sample_rate(&self) -> u32 {
        self.ring.sample_rate
    }

    /// Assembles the far-end window time-aligned with a mic frame
    /// `[pos, pos + out.len())`, zero-filling any gap, and reports coverage.
    ///
    /// Wait-free and allocation-free. Queue data at positions `< pos + len`
    /// is *consumed* (its slots are released, so the queue runs in fixed
    /// memory forever); data at positions `≥ pos + len` stays for future
    /// windows. Calling with a `pos` that goes backward re-reads a released
    /// region and therefore reports it as missing — mic positions are
    /// expected to be monotonic, mirroring the writer side.
    ///
    /// A zero-length `out` is trivially [`AecRefWindowStatus::Complete`].
    pub fn window(&mut self, pos: u64, out: &mut [f32]) -> AecRefWindowStatus {
        out.fill(0.0);
        if out.is_empty() {
            return AecRefWindowStatus::Complete;
        }
        let end = pos + out.len() as u64;
        let mut head = self.ring.head.0.load(Ordering::Relaxed);
        let tail = self.ring.tail.0.load(Ordering::Acquire);
        let mut covered = 0usize;
        while head != tail {
            let slot = &self.ring.buf[head & self.ring.mask];
            let p = slot.pos.load(Ordering::Relaxed);
            if p >= end {
                break;
            }
            if p >= pos {
                out[(p - pos) as usize] = f32::from_bits(slot.bits.load(Ordering::Relaxed));
                covered += 1;
            }
            // Positions below `pos` are a released (already-passed) region:
            // the slot is consumed without contributing to the window.
            head = head.wrapping_add(1);
        }
        self.ring.head.0.store(head, Ordering::Release);
        if covered == 0 {
            AecRefWindowStatus::Empty
        } else if covered == out.len() {
            AecRefWindowStatus::Complete
        } else {
            AecRefWindowStatus::Partial {
                missing: out.len() - covered,
            }
        }
    }
}

// Compile-time Send verification, same style as the `super` stream module's
// assertion block: the failure IS the test. Writer and reader are Send (move
// them to the playback / inference threads) and deliberately NOT Clone (SPSC
// by type).
const _: () = {
    const fn assert_send<T: Send>() {}
    assert_send::<AecRefWriter>();
    assert_send::<AecRefReader>();
};

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Barrier;

    #[test]
    fn zero_sample_rate_is_rejected() {
        assert!(matches!(
            aec_ref_queue(64, 0),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn capacity_is_rounded_up_to_power_of_two() {
        let (tx, rx) = aec_ref_queue(5, 16_000).unwrap();
        assert_eq!(tx.capacity(), 8);
        assert_eq!(rx.capacity(), 8);
        let (tx0, _rx0) = aec_ref_queue(0, 16_000).unwrap();
        assert_eq!(tx0.capacity(), 1);
    }

    #[test]
    fn sample_rate_is_visible_on_both_halves() {
        let (tx, rx) = aec_ref_queue(8, 24_000).unwrap();
        assert_eq!(tx.sample_rate(), 24_000);
        assert_eq!(rx.sample_rate(), 24_000);
    }

    /// (1) aligned push → window is a bit-exact restore.
    #[test]
    fn aligned_push_then_window_is_bit_exact() {
        let (mut tx, mut rx) = aec_ref_queue(64, 16_000).unwrap();
        let pcm: Vec<f32> = (0..16).map(|i| (i as f32 - 7.5) * 0.125).collect();
        assert_eq!(tx.push(&pcm, 1000).unwrap(), 16);
        let mut win = [0.0f32; 16];
        assert_eq!(rx.window(1000, &mut win), AecRefWindowStatus::Complete);
        for (i, (&got, &want)) in win.iter().zip(pcm.iter()).enumerate() {
            assert_eq!(got.to_bits(), want.to_bits(), "sample {i} bit-exact");
        }
    }

    /// (2) a window spanning two pushed chunks.
    #[test]
    fn window_spans_chunk_boundary() {
        let (mut tx, mut rx) = aec_ref_queue(64, 16_000).unwrap();
        let a: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let b: Vec<f32> = (8..16).map(|i| i as f32).collect();
        assert_eq!(tx.push(&a, 0).unwrap(), 8);
        assert_eq!(tx.push(&b, 8).unwrap(), 8);
        let mut win = [0.0f32; 12];
        // [2, 14) crosses the chunk boundary at 8.
        assert_eq!(rx.window(2, &mut win), AecRefWindowStatus::Complete);
        for (i, &got) in win.iter().enumerate() {
            assert_eq!(got, (i + 2) as f32);
        }
    }

    /// (3) gap zero-fill + Partial / Empty statuses.
    #[test]
    fn gap_is_zero_filled_and_reported() {
        let (mut tx, mut rx) = aec_ref_queue(64, 16_000).unwrap();
        assert_eq!(tx.push(&[1.0; 4], 0).unwrap(), 4);
        // Forward gap [4, 8) — legal, costs no space.
        assert_eq!(tx.push(&[2.0; 4], 8).unwrap(), 4);
        let mut win = [9.9f32; 12];
        assert_eq!(
            rx.window(0, &mut win),
            AecRefWindowStatus::Partial { missing: 4 }
        );
        assert_eq!(&win[0..4], &[1.0; 4]);
        assert_eq!(&win[4..8], &[0.0; 4], "gap zero-filled");
        assert_eq!(&win[8..12], &[2.0; 4]);
    }

    #[test]
    fn untouched_region_is_empty() {
        let (mut tx, mut rx) = aec_ref_queue(64, 16_000).unwrap();
        assert_eq!(tx.push(&[1.0; 4], 100).unwrap(), 4);
        let mut win = [5.0f32; 4];
        // Entirely before the pushed region: nothing covers it.
        assert_eq!(rx.window(0, &mut win), AecRefWindowStatus::Empty);
        assert_eq!(win, [0.0; 4], "empty window still zero-fills");
        // The pushed region is still there afterwards.
        let mut win2 = [0.0f32; 4];
        assert_eq!(rx.window(100, &mut win2), AecRefWindowStatus::Complete);
    }

    /// Window overlapping only the tail of the pushed data.
    #[test]
    fn window_past_data_end_is_partial() {
        let (mut tx, mut rx) = aec_ref_queue(64, 16_000).unwrap();
        assert_eq!(tx.push(&[3.0; 8], 0).unwrap(), 8);
        let mut win = [0.0f32; 8];
        assert_eq!(
            rx.window(4, &mut win),
            AecRefWindowStatus::Partial { missing: 4 }
        );
        assert_eq!(&win[0..4], &[3.0; 4]);
        assert_eq!(&win[4..8], &[0.0; 4]);
    }

    /// (4) backward / overlapping time tags are explicit errors.
    #[test]
    fn backward_or_duplicate_tags_are_explicit_errors() {
        let (mut tx, _rx) = aec_ref_queue(64, 16_000).unwrap();
        assert_eq!(tx.push(&[1.0; 8], 100).unwrap(), 8);
        // Strictly backward.
        assert!(matches!(
            tx.push(&[9.0; 2], 50),
            Err(VokraError::InvalidArgument(_))
        ));
        // Overlapping the already-pushed [100, 108) region.
        assert!(matches!(
            tx.push(&[9.0; 2], 107),
            Err(VokraError::InvalidArgument(_))
        ));
        // Exactly contiguous is fine.
        assert_eq!(tx.push(&[2.0; 2], 108).unwrap(), 2);
    }

    #[test]
    fn empty_push_is_a_noop_but_still_gated() {
        let (mut tx, _rx) = aec_ref_queue(8, 16_000).unwrap();
        assert_eq!(tx.push(&[1.0; 4], 10).unwrap(), 4);
        // Empty push at a legal position: no-op.
        assert_eq!(tx.push(&[], 14).unwrap(), 0);
        // Empty push into the past is still a tag violation.
        assert!(matches!(
            tx.push(&[], 5),
            Err(VokraError::InvalidArgument(_))
        ));
        // The no-op did not advance the timeline: 14 is still legal.
        assert_eq!(tx.push(&[2.0; 2], 14).unwrap(), 2);
    }

    /// (5) reject-on-full backpressure: prefix accepted, count returned.
    #[test]
    fn reject_on_full_accepts_prefix_and_reports_count() {
        let (mut tx, mut rx) = aec_ref_queue(8, 16_000).unwrap(); // cap 8
        let pcm: Vec<f32> = (0..12).map(|i| i as f32).collect();
        // Only the first 8 fit.
        assert_eq!(tx.push(&pcm, 0).unwrap(), 8);
        // Ring is full: zero accepted, no corruption.
        assert_eq!(tx.push(&pcm[8..], 8).unwrap(), 0);
        // Drain, then the retry (at the advanced position) succeeds.
        let mut win = [0.0f32; 8];
        assert_eq!(rx.window(0, &mut win), AecRefWindowStatus::Complete);
        for (i, &v) in win.iter().enumerate() {
            assert_eq!(v, i as f32);
        }
        assert_eq!(tx.push(&pcm[8..], 8).unwrap(), 4);
        let mut win2 = [0.0f32; 4];
        assert_eq!(rx.window(8, &mut win2), AecRefWindowStatus::Complete);
        assert_eq!(&win2, &[8.0, 9.0, 10.0, 11.0]);
    }

    /// Consumed regions are released: the queue runs in fixed memory over an
    /// arbitrarily long run (many wraps of the cursor mask).
    #[test]
    fn long_run_wraps_in_fixed_memory() {
        let (mut tx, mut rx) = aec_ref_queue(16, 16_000).unwrap();
        let mut pos = 0u64;
        let mut win = [0.0f32; 16];
        for round in 0..1000u64 {
            let chunk: Vec<f32> = (0..16).map(|i| (round * 16 + i) as f32).collect();
            assert_eq!(tx.push(&chunk, pos).unwrap(), 16, "round {round} accepts");
            assert_eq!(
                rx.window(pos, &mut win),
                AecRefWindowStatus::Complete,
                "round {round} complete"
            );
            assert_eq!(win[0], (round * 16) as f32);
            assert_eq!(win[15], (round * 16 + 15) as f32);
            pos += 16;
        }
    }

    /// Data beyond the requested window stays queued for the next window.
    #[test]
    fn future_data_survives_earlier_window() {
        let (mut tx, mut rx) = aec_ref_queue(64, 16_000).unwrap();
        assert_eq!(tx.push(&[1.0; 8], 0).unwrap(), 8);
        assert_eq!(tx.push(&[2.0; 8], 8).unwrap(), 8);
        let mut w1 = [0.0f32; 8];
        assert_eq!(rx.window(0, &mut w1), AecRefWindowStatus::Complete);
        assert_eq!(w1, [1.0; 8]);
        let mut w2 = [0.0f32; 8];
        assert_eq!(rx.window(8, &mut w2), AecRefWindowStatus::Complete);
        assert_eq!(w2, [2.0; 8]);
    }

    /// Re-reading a consumed region is Empty (release semantics, documented).
    #[test]
    fn rereading_a_consumed_region_is_empty() {
        let (mut tx, mut rx) = aec_ref_queue(64, 16_000).unwrap();
        assert_eq!(tx.push(&[7.0; 8], 0).unwrap(), 8);
        let mut win = [0.0f32; 8];
        assert_eq!(rx.window(0, &mut win), AecRefWindowStatus::Complete);
        assert_eq!(rx.window(0, &mut win), AecRefWindowStatus::Empty);
    }

    #[test]
    fn zero_length_window_is_complete() {
        let (_tx, mut rx) = aec_ref_queue(8, 16_000).unwrap();
        let mut empty: [f32; 0] = [];
        assert_eq!(rx.window(0, &mut empty), AecRefWindowStatus::Complete);
    }

    /// (6) THE two-thread oracle (ring.rs test flavor): a playback thread
    /// pushes the contiguous ramp `0..N` (value == position) with
    /// backpressure retries; the inference thread windows sequential frames.
    /// Every window must come back Complete with `out[i] == pos + i`, for
    /// every interleaving — deterministic despite the threads.
    #[test]
    fn spsc_concurrent_push_window_is_lossless_and_aligned() {
        let (mut tx, mut rx) = aec_ref_queue(64, 16_000).unwrap();
        let n: u64 = 50_000;
        let frame = 25usize; // deliberately not a divisor of the capacity
        let barrier = Arc::new(Barrier::new(2));
        let b2 = Arc::clone(&barrier);

        let producer = std::thread::spawn(move || {
            b2.wait();
            let mut pos = 0u64;
            let mut chunk = [0.0f32; 40];
            while pos < n {
                let take = chunk.len().min((n - pos) as usize);
                for (i, c) in chunk[..take].iter_mut().enumerate() {
                    *c = (pos + i as u64) as f32;
                }
                let mut off = 0usize;
                while off < take {
                    let accepted = tx
                        .push(&chunk[off..take], pos + off as u64)
                        .expect("monotonic push");
                    off += accepted;
                    if accepted == 0 {
                        std::hint::spin_loop();
                    }
                }
                pos += take as u64;
            }
        });

        barrier.wait();
        // The producer publishes samples in strictly ascending position
        // order, so at any instant the published part of [pos, pos + take)
        // is a contiguous PREFIX. A window therefore comes back as:
        //   - Complete       → all `take` samples covered;
        //   - Partial{miss}  → exactly the prefix [pos, pos + take - miss)
        //                      covered (and now consumed);
        //   - Empty          → nothing consumed, safe to retry.
        // Asserting the covered prefix and advancing by its length checks
        // every one of the N samples exactly once, in order — a
        // deterministic oracle for every thread interleaving.
        let mut pos = 0u64;
        let mut win = vec![0.0f32; frame];
        while pos < n {
            let take = frame.min((n - pos) as usize);
            let covered = match rx.window(pos, &mut win[..take]) {
                AecRefWindowStatus::Complete => take,
                AecRefWindowStatus::Partial { missing } => take - missing,
                AecRefWindowStatus::Empty => {
                    std::hint::spin_loop();
                    continue;
                }
            };
            for (i, &v) in win[..covered].iter().enumerate() {
                assert_eq!(
                    v,
                    (pos + i as u64) as f32,
                    "no gap / dup / misalignment at {}",
                    pos + i as u64
                );
            }
            pos += covered as u64;
        }
        producer.join().expect("producer joins");
    }

    #[test]
    fn writer_and_reader_are_send() {
        const fn assert_send<T: Send>() {}
        assert_send::<AecRefWriter>();
        assert_send::<AecRefReader>();
        assert_send::<AecRefWindowStatus>();
    }
}
