//! Sliding-window ring KV cache (M4-06 #16): bounded-memory key/value history
//! for a full-duplex streaming decoder.
//!
//! The Moshi backbone applies a **sliding-window causal mask** — position `q`
//! attends key `k` iff `0 <= q - k < context` (`moshi/backbone.rs`,
//! `transformer.py attn_bias = (delta >= 0) & (delta < context)`). Keys older
//! than `context` contribute `exp(-inf) = 0` to every future softmax, so they
//! are dead weight the moment they leave the window. The M3-03 [`PagedKvCache`]
//! still keeps them resident (memory `O(max_ctx)`), which is fine for a bounded
//! prompt but unbounded-in-practice for a long streaming session.
//!
//! [`RingKvCache`] stores only the last `capacity` positions per layer in a
//! circular buffer keyed by `t % capacity`, so a session of any length uses
//! `O(n_layer · capacity · row_width)` memory. With `capacity = context` the
//! retained window is **exactly** the set of keys a streaming (single new
//! position per step) decoder can still attend, so reading the ring over that
//! window is numerically identical to reading the full paged history and
//! letting the mask zero the rest (proven against [`PagedKvCache`] in the tests
//! and in `crates/vokra-models/src/moshi/backbone.rs`).
//!
//! # Streaming only — the ring is not a drop-in for bulk forward
//!
//! A single forward over `t` positions reads every key from the oldest row's
//! window start up to the newest position — up to `context + t - 1` distinct
//! positions when the clock is past the first window. A `capacity = context`
//! ring cannot hold that for `t > 1`, so the ring is for the **streaming**
//! (`t = 1` per step) path; a bulk step whose read span exceeds `capacity`
//! surfaces a loud [`VokraError`] rather than silently returning evicted keys
//! (FR-EX-08). This is why the shared Moshi forward keeps the paged cache for
//! bulk priming and only routes streaming sessions through the ring.
//!
//! Safe Rust, no external dependencies (NFR-DS-02).

use std::ops::Range;

use super::paged::{KvDims, KvElement};
use crate::error::{Result, VokraError};

/// A fixed-capacity circular KV cache. `k` / `v` storage and the per-slot
/// position table are allocated once at construction and never grow, so a
/// stream of unbounded length stays within `O(n_layer · capacity · row_width)`
/// memory.
///
/// Logical addressing mirrors [`PagedKvCache`](super::paged::PagedKvCache): a
/// row is `(layer, t, stream, codebook)` with a per-slot payload of
/// `n_head · d_head` elements; the physical slot is `t % capacity`. A slot
/// holds position `t` until position `t + capacity` overwrites it — the
/// eviction is implicit and exact.
pub struct RingKvCache<T: KvElement> {
    n_layer: usize,
    n_stream: usize,
    n_codebook: usize,
    /// `n_head · d_head` — the per-(stream, codebook) payload width.
    per_slot: usize,
    /// `n_stream · n_codebook · per_slot` — a full ring slot's width.
    row_width: usize,
    /// The sliding-window length in positions.
    capacity: usize,
    /// `[n_layer · capacity · row_width]` key store (row-major by
    /// `(layer, slot, stream, codebook, head·d_head)`).
    k: Vec<T>,
    /// Value store, same shape as `k`.
    v: Vec<T>,
    /// Absolute position currently occupying each `(layer, slot)`, or `None`
    /// when the slot has never been written since the last [`Self::reset`].
    /// This is what makes a read of an evicted position return `None` instead
    /// of the stale value a bare `t % capacity` index would surface.
    slot_pos: Vec<Option<usize>>,
    /// Highest absolute position appended since the last reset (observability +
    /// window bookkeeping).
    highest: Option<usize>,
    /// Committed-position clock, advanced by [`Self::advance`] (API parity with
    /// [`PagedKvCache`](super::paged::PagedKvCache)).
    pos: usize,
}

impl<T: KvElement> RingKvCache<T> {
    /// Pre-allocates a ring sized to `capacity` positions with the shape axes of
    /// `dims` (`max_time` is only used as an upper sanity bound). All storage is
    /// allocated here; the hot path never allocates (FR-EX-05).
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] if any axis is zero, if `capacity` is
    /// zero, or if `capacity > dims.max_time` (a window larger than the hard
    /// bound is a caller bug).
    pub fn pre_allocate(dims: KvDims, capacity: usize) -> Result<Self> {
        if dims.n_layer == 0
            || dims.n_head == 0
            || dims.d_head == 0
            || dims.n_stream == 0
            || dims.n_codebook == 0
        {
            return Err(VokraError::InvalidArgument(format!(
                "RingKvCache: every shape axis must be >= 1 (got n_layer={}, n_head={}, \
                 d_head={}, n_stream={}, n_codebook={})",
                dims.n_layer, dims.n_head, dims.d_head, dims.n_stream, dims.n_codebook
            )));
        }
        if capacity == 0 {
            return Err(VokraError::InvalidArgument(
                "RingKvCache: capacity must be >= 1".to_owned(),
            ));
        }
        if capacity > dims.max_time {
            return Err(VokraError::InvalidArgument(format!(
                "RingKvCache: capacity {capacity} exceeds max_time {} (a window cannot be \
                 larger than the hard position bound)",
                dims.max_time
            )));
        }
        let per_slot = dims.n_head * dims.d_head;
        let row_width = dims.n_stream * dims.n_codebook * per_slot;
        let store_len = dims
            .n_layer
            .checked_mul(capacity)
            .and_then(|x| x.checked_mul(row_width))
            .ok_or_else(|| {
                VokraError::InvalidArgument("RingKvCache: storage size overflows usize".to_owned())
            })?;
        Ok(Self {
            n_layer: dims.n_layer,
            n_stream: dims.n_stream,
            n_codebook: dims.n_codebook,
            per_slot,
            row_width,
            capacity,
            k: vec![T::ZERO; store_len],
            v: vec![T::ZERO; store_len],
            slot_pos: vec![None; dims.n_layer * capacity],
            highest: None,
            pos: 0,
        })
    }

    /// Appends one position's `k` / `v` row for `(layer, t, s, c)`, evicting
    /// whatever position previously occupied slot `t % capacity` on that layer.
    ///
    /// All `(stream, codebook)` payloads of a given position must be appended
    /// before that position is overwritten; a read after eviction returns
    /// `None`.
    ///
    /// # Errors
    ///
    /// [`VokraError::InvalidArgument`] on an out-of-range axis or a wrong
    /// `k_row` / `v_row` length.
    pub fn append_step(
        &mut self,
        layer: usize,
        t: usize,
        s: usize,
        c: usize,
        k_row: &[T],
        v_row: &[T],
    ) -> Result<()> {
        if layer >= self.n_layer || s >= self.n_stream || c >= self.n_codebook {
            return Err(VokraError::InvalidArgument(format!(
                "RingKvCache::append_step: out-of-range axis (layer={layer}/{}, s={s}/{}, \
                 c={c}/{})",
                self.n_layer, self.n_stream, self.n_codebook
            )));
        }
        if k_row.len() != self.per_slot || v_row.len() != self.per_slot {
            return Err(VokraError::InvalidArgument(format!(
                "RingKvCache::append_step: expected k/v row len {}, got k={} v={}",
                self.per_slot,
                k_row.len(),
                v_row.len()
            )));
        }
        let slot = t % self.capacity;
        let slot_idx = layer * self.capacity + slot;
        let base = slot_idx * self.row_width;
        let sc_offset = (s * self.n_codebook + c) * self.per_slot;
        let lo = base + sc_offset;
        self.k[lo..lo + self.per_slot].copy_from_slice(k_row);
        self.v[lo..lo + self.per_slot].copy_from_slice(v_row);
        self.slot_pos[slot_idx] = Some(t);
        self.highest = Some(self.highest.map_or(t, |h| h.max(t)));
        Ok(())
    }

    /// Reads the K/V row for `(layer, t, s, c)`, or [`None`] if position `t` has
    /// been evicted from the window (or never written). The borrows are stable
    /// until the next [`Self::append_step`] / [`Self::reset`].
    #[must_use]
    pub fn read_step(&self, layer: usize, t: usize, s: usize, c: usize) -> Option<(&[T], &[T])> {
        if layer >= self.n_layer || s >= self.n_stream || c >= self.n_codebook {
            return None;
        }
        let slot = t % self.capacity;
        let slot_idx = layer * self.capacity + slot;
        // The slot must currently hold exactly position `t`; if it holds an
        // older or newer wrap, `t` is evicted (or not yet written) → None.
        if self.slot_pos[slot_idx] != Some(t) {
            return None;
        }
        let base = slot_idx * self.row_width;
        let sc_offset = (s * self.n_codebook + c) * self.per_slot;
        let lo = base + sc_offset;
        Some((
            &self.k[lo..lo + self.per_slot],
            &self.v[lo..lo + self.per_slot],
        ))
    }

    /// Commits `n` newly appended positions (position-clock parity with
    /// [`PagedKvCache::advance`](super::paged::PagedKvCache::advance)). The ring
    /// window is tracked by [`Self::append_step`] itself, so this only advances
    /// the observability clock.
    pub fn advance(&mut self, n: usize) {
        self.pos += n;
    }

    /// Committed-position count (parity with `PagedKvCache::positions`).
    #[inline]
    #[must_use]
    pub const fn positions(&self) -> usize {
        self.pos
    }

    /// Rewinds to empty, retaining the pre-allocated storage (fast barge-in
    /// reset — the arena is reused, no realloc).
    pub fn reset(&mut self) {
        for slot in &mut self.slot_pos {
            *slot = None;
        }
        self.highest = None;
        self.pos = 0;
    }

    /// Number of positions currently resident in the window — bounded by
    /// `capacity` no matter how many positions have streamed through. This is
    /// the ring's analogue of `PagedKvCache::pages_in_use` and the quantity the
    /// bounded-memory test pins.
    #[inline]
    #[must_use]
    pub fn live_len(&self) -> usize {
        match self.highest {
            None => 0,
            Some(h) => (h + 1).min(self.capacity),
        }
    }

    /// The absolute-position half-open range `[lo, hi)` currently resident, i.e.
    /// the keys a streaming query may still read. Empty (`0..0`) before the
    /// first append.
    #[must_use]
    pub fn live_window(&self) -> Range<usize> {
        match self.highest {
            None => 0..0,
            Some(h) => h.saturating_sub(self.capacity - 1)..(h + 1),
        }
    }

    /// The sliding-window length in positions.
    #[inline]
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    /// Total element count of the (fixed) K store — the bounded-memory witness:
    /// it never changes across appends. `V` is the same size.
    #[inline]
    #[must_use]
    pub fn storage_elements(&self) -> usize {
        self.k.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dims(n_layer: usize, n_head: usize, d_head: usize, max_time: usize) -> KvDims {
        KvDims {
            n_layer,
            n_head,
            d_head,
            n_stream: 1,
            n_codebook: 1,
            max_time,
        }
    }

    /// A distinct, position-derived payload so a wrong slot is caught by value.
    fn row(pos: usize, width: usize, tag: f32) -> Vec<f32> {
        (0..width)
            .map(|c| tag + pos as f32 * 100.0 + c as f32)
            .collect()
    }

    #[test]
    fn append_then_read_round_trips_within_capacity() {
        let mut ring = RingKvCache::<f32>::pre_allocate(dims(2, 2, 3, 8), 4).unwrap();
        let width = 2 * 3;
        for t in 0..4 {
            ring.append_step(0, t, 0, 0, &row(t, width, 1.0), &row(t, width, 2.0))
                .unwrap();
            ring.append_step(1, t, 0, 0, &row(t, width, 3.0), &row(t, width, 4.0))
                .unwrap();
        }
        for t in 0..4 {
            let (k, v) = ring.read_step(0, t, 0, 0).expect("resident");
            assert_eq!(k, row(t, width, 1.0).as_slice());
            assert_eq!(v, row(t, width, 2.0).as_slice());
            let (k1, v1) = ring.read_step(1, t, 0, 0).expect("resident layer 1");
            assert_eq!(k1, row(t, width, 3.0).as_slice());
            assert_eq!(v1, row(t, width, 4.0).as_slice());
        }
    }

    #[test]
    fn memory_stays_bounded_and_window_is_correct_after_many_steps() {
        // capacity 4, but stream 20 positions through (>> capacity).
        let capacity = 4;
        let width = 2; // n_head(1) · d_head(2)
        let mut ring = RingKvCache::<f32>::pre_allocate(dims(1, 1, 2, 32), capacity).unwrap();
        let storage_before = ring.storage_elements();
        for t in 0..20 {
            ring.append_step(0, t, 0, 0, &row(t, width, 7.0), &row(t, width, 8.0))
                .unwrap();
            ring.advance(1);
            // Live set never exceeds capacity — the bounded-memory invariant.
            assert!(
                ring.live_len() <= capacity,
                "live_len {} exceeded capacity {capacity} at t={t}",
                ring.live_len()
            );
        }
        // Storage never grew: O(n_layer·capacity·row_width), fixed.
        assert_eq!(ring.storage_elements(), storage_before);
        assert_eq!(ring.live_len(), capacity);
        assert_eq!(
            ring.live_window(),
            16..20,
            "retained window = last `capacity`"
        );

        // The retained window [16, 19] reads back the exact, uncorrupted values
        // written for those positions (ring wraparound integrity).
        for t in 16..20 {
            let (k, v) = ring.read_step(0, t, 0, 0).expect("in-window");
            assert_eq!(
                k,
                row(t, width, 7.0).as_slice(),
                "wraparound corrupted k at {t}"
            );
            assert_eq!(
                v,
                row(t, width, 8.0).as_slice(),
                "wraparound corrupted v at {t}"
            );
        }
        // Everything older than the window is evicted → None (no stale reads).
        for t in 0..16 {
            assert!(
                ring.read_step(0, t, 0, 0).is_none(),
                "evicted position {t} must read None, not a stale slot"
            );
        }
        // A future position was never written → None.
        assert!(ring.read_step(0, 20, 0, 0).is_none());
    }

    #[test]
    fn eviction_boundary_is_exact() {
        // capacity 3: after writing 0,1,2,3 the slot of 0 (0 % 3) is reused by
        // 3, so 0 is evicted and 1,2,3 survive.
        let mut ring = RingKvCache::<f32>::pre_allocate(dims(1, 1, 1, 8), 3).unwrap();
        for t in 0..4 {
            ring.append_step(0, t, 0, 0, &[t as f32 + 0.5], &[t as f32 + 0.25])
                .unwrap();
        }
        assert!(ring.read_step(0, 0, 0, 0).is_none(), "0 evicted by 3");
        for t in 1..4 {
            let (k, _) = ring.read_step(0, t, 0, 0).expect("survivor");
            assert_eq!(k, &[t as f32 + 0.5]);
        }
        assert_eq!(ring.live_window(), 1..4);
    }

    #[test]
    fn reset_clears_the_window_and_reuses_storage() {
        let mut ring = RingKvCache::<f32>::pre_allocate(dims(1, 1, 2, 8), 4).unwrap();
        let width = 2;
        for t in 0..3 {
            ring.append_step(0, t, 0, 0, &row(t, width, 1.0), &row(t, width, 2.0))
                .unwrap();
            ring.advance(1);
        }
        assert!(ring.live_len() > 0);
        let storage = ring.storage_elements();
        ring.reset();
        assert_eq!(ring.live_len(), 0);
        assert_eq!(ring.positions(), 0);
        assert_eq!(ring.live_window(), 0..0);
        assert!(
            ring.read_step(0, 0, 0, 0).is_none(),
            "reset drops all positions"
        );
        assert_eq!(ring.storage_elements(), storage, "reset reuses storage");
    }

    #[test]
    fn layers_and_streams_are_independent() {
        let d = KvDims {
            n_layer: 2,
            n_head: 1,
            d_head: 1,
            n_stream: 2,
            n_codebook: 1,
            max_time: 8,
        };
        let mut ring = RingKvCache::<f32>::pre_allocate(d, 4).unwrap();
        // Distinct value per (layer, stream) at the same position.
        ring.append_step(0, 0, 0, 0, &[10.0], &[11.0]).unwrap();
        ring.append_step(0, 0, 1, 0, &[20.0], &[21.0]).unwrap();
        ring.append_step(1, 0, 0, 0, &[30.0], &[31.0]).unwrap();
        assert_eq!(ring.read_step(0, 0, 0, 0).unwrap().0, &[10.0]);
        assert_eq!(ring.read_step(0, 0, 1, 0).unwrap().0, &[20.0]);
        assert_eq!(ring.read_step(1, 0, 0, 0).unwrap().0, &[30.0]);
    }

    #[test]
    fn reads_before_any_append_return_none() {
        let ring = RingKvCache::<f32>::pre_allocate(dims(1, 1, 1, 4), 2).unwrap();
        assert!(ring.read_step(0, 0, 0, 0).is_none());
        assert_eq!(ring.live_len(), 0);
        assert_eq!(ring.live_window(), 0..0);
    }

    #[test]
    fn construction_and_append_reject_bad_shapes() {
        assert!(
            RingKvCache::<f32>::pre_allocate(dims(1, 1, 1, 4), 0).is_err(),
            "zero capacity"
        );
        assert!(
            RingKvCache::<f32>::pre_allocate(dims(1, 1, 1, 4), 5).is_err(),
            "capacity > max_time"
        );
        assert!(
            RingKvCache::<f32>::pre_allocate(dims(0, 1, 1, 4), 2).is_err(),
            "zero layer"
        );
        let mut ring = RingKvCache::<f32>::pre_allocate(dims(1, 1, 2, 4), 2).unwrap();
        assert!(
            ring.append_step(0, 0, 0, 0, &[1.0], &[1.0, 2.0]).is_err(),
            "short k row"
        );
        assert!(
            ring.append_step(1, 0, 0, 0, &[1.0, 2.0], &[1.0, 2.0])
                .is_err(),
            "layer oob"
        );
    }

    /// The whole point: over a streaming (single new position per step)
    /// session, reading the ring's window is byte-identical to reading the full
    /// [`PagedKvCache`] history over the same window — the masked keys the
    /// paged cache still holds would contribute exactly zero anyway. This is the
    /// equivalence the Moshi wire-in relies on.
    #[test]
    fn streaming_window_matches_paged_history() {
        use super::super::paged::{BlockSize, PagedKvCache};

        let context = 4;
        let max_ctx = 32;
        let d = dims(2, 2, 3, max_ctx);
        let width = 2 * 3;
        let mut ring = RingKvCache::<f32>::pre_allocate(d, context).unwrap();
        let mut paged = PagedKvCache::<f32>::pre_allocate(d, BlockSize::Two).unwrap();

        for t in 0..20 {
            for layer in 0..2 {
                let k = row(t, width, layer as f32 * 10.0 + 1.0);
                let v = row(t, width, layer as f32 * 10.0 + 2.0);
                ring.append_step(layer, t, 0, 0, &k, &v).unwrap();
                paged.append_step(layer, t, 0, 0, &k, &v).unwrap();
            }
            ring.advance(1);
            paged.advance(1);

            // The streaming query at position `t` attends window
            // [t+1-context, t]; every key in it must match between the two.
            let win_lo = (t + 1).saturating_sub(context);
            for layer in 0..2 {
                for j in win_lo..=t {
                    let (rk, rv) = ring.read_step(layer, j, 0, 0).expect("ring in-window");
                    let (pk, pv) = paged.read_step(layer, j, 0, 0).expect("paged history");
                    assert_eq!(rk, pk, "k mismatch layer {layer} pos {j} step {t}");
                    assert_eq!(rv, pv, "v mismatch layer {layer} pos {j} step {t}");
                }
            }
        }
    }
}
