//! Ownable, `Send` key/value cache for autoregressive decoders (FR-EX-02).
//!
//! [`KvCache`] is the promotion of the previously model-internal Whisper
//! self-attention cache (see `vokra-models` `whisper::decoder`) to a first-class
//! runtime type. It bundles the per-layer key / value buffers that grow as
//! tokens are appended, plus the committed position count, behind a narrow
//! surface.
//!
//! # Why it lives here (and why it is `Send`)
//!
//! Promoting the cache to `vokra-core` is the foundation for the M1-08
//! streaming session state: a decode can be paused, moved across threads, and
//! resumed. The type holds nothing but `Vec<f32>` / `usize`, so it is
//! automatically [`Send`] (and [`Sync`]) with no `unsafe`. Keeping the surface
//! minimal — grow (`append`), commit (`advance`), read (`k` / `v` /
//! `positions`), rewind (`reset`) — keeps that guarantee cheap to uphold as
//! more models plug in.
//!
//! # Layout contract
//!
//! Every layer stores its keys and values row-major as `[positions, width]`,
//! where `width` is the model hidden size. All layers grow in lockstep: one
//! decode step appends the same number of rows to each layer and then commits
//! that many positions once with [`advance`](KvCache::advance). The committed
//! [`positions`](KvCache::positions) count is what a causal attention uses as
//! its query offset, so it is advanced *after* a step's layers are processed,
//! never per layer (this also keeps a zero-layer configuration — used by the
//! synthetic decoder tests — advancing its position clock correctly).

/// Per-layer key / value buffers, each row-major `[positions, width]` and grown
/// in lockstep with every other layer.
///
/// `Clone` is derived so [`KvCache`] can be [`Clone`]d as a whole — used by
/// beam search implementations that branch a session's state across candidate
/// hypotheses (M3-10 Voxtral beam search + n-best decode). Cloning does a
/// deep-copy of the `k` / `v` `Vec<f32>` buffers.
#[derive(Clone)]
struct LayerKv {
    /// Key rows, `positions * width` elements.
    k: Vec<f32>,
    /// Value rows, `positions * width` elements.
    v: Vec<f32>,
}

/// A growable key/value cache shared by every layer of an autoregressive
/// decoder.
///
/// Construct with [`with_reserve`](KvCache::with_reserve), then per decode step:
/// [`append`](KvCache::append) each layer's new key/value rows and
/// [`advance`](KvCache::advance) by the number of new positions. Read cached
/// rows with [`k`](KvCache::k) / [`v`](KvCache::v) and the committed length with
/// [`positions`](KvCache::positions). [`reset`](KvCache::reset) rewinds to empty
/// while keeping the reserved capacity.
///
/// ```
/// use vokra_core::KvCache;
///
/// // 2 layers, hidden width 4, room reserved for 16 positions.
/// let mut cache = KvCache::with_reserve(2, 4, 16);
/// assert_eq!(cache.positions(), 0);
/// assert!(cache.capacity_positions() >= 16);
///
/// // One token: append its key/value row (width 4) to each layer, then commit.
/// cache.append(0, &[0.0; 4], &[1.0; 4]);
/// cache.append(1, &[2.0; 4], &[3.0; 4]);
/// cache.advance(1);
/// assert_eq!(cache.positions(), 1);
/// assert_eq!(cache.k(0), &[0.0; 4]);
/// assert_eq!(cache.v(1), &[3.0; 4]);
///
/// // Rewind for a fresh decode of the same audio; capacity is retained.
/// cache.reset();
/// assert_eq!(cache.positions(), 0);
/// assert!(cache.k(0).is_empty());
/// assert!(cache.capacity_positions() >= 16);
/// ```
///
/// # Cloning for beam search
///
/// [`Clone`] is derived so a caller can snapshot the whole cache in one call
/// — used by the Voxtral beam-search + n-best decode (M3-10) that branches
/// a decoding session across candidate hypotheses. Cloning does a deep-copy
/// of every layer's `k` / `v` buffer (`O(n_layers * positions * width)`),
/// so it is intended for beam widths on the order of 1..~16 with per-utterance
/// decode lengths, not for the general hot path.
#[derive(Clone)]
pub struct KvCache {
    /// One `(k, v)` buffer pair per layer.
    layers: Vec<LayerKv>,
    /// Number of committed positions (tokens) across the cache.
    pos: usize,
    /// Hidden width of a single position's key / value row.
    width: usize,
}

impl KvCache {
    /// Creates an empty cache for `n_layers`, each row `width` elements wide,
    /// pre-reserving capacity for `reserve_hint` positions per layer.
    ///
    /// `reserve_hint` is a **capacity hint, not a hard cap**: a decode up to the
    /// hint appends without reallocating, and a longer decode grows the buffers
    /// amortically (`Vec` doubling). Callers seed it to a *typical* decode
    /// length rather than the worst-case window, so a short decode does not
    /// pre-allocate the maximum (M1-04 sub-part 2, variable-length I/O). The
    /// buffers start empty (`len == 0`); only their capacity is reserved.
    #[must_use]
    pub fn with_reserve(n_layers: usize, width: usize, reserve_hint: usize) -> Self {
        let cap = width.saturating_mul(reserve_hint);
        let layers = (0..n_layers)
            .map(|_| LayerKv {
                k: Vec::with_capacity(cap),
                v: Vec::with_capacity(cap),
            })
            .collect();
        Self {
            layers,
            pos: 0,
            width,
        }
    }

    /// Appends one step's key and value rows for `layer`. `k_row` / `v_row` are
    /// row-major `[rows, width]` (`rows` new positions); they extend the layer's
    /// buffers in place. Committing the new positions is a separate
    /// [`advance`](Self::advance) after every layer has been appended.
    ///
    /// # Panics
    ///
    /// Panics if `layer >= n_layers` (an internal decoder invariant). In debug
    /// builds it also checks that `k_row` and `v_row` share a length that is a
    /// whole number of `width`-wide rows.
    pub fn append(&mut self, layer: usize, k_row: &[f32], v_row: &[f32]) {
        debug_assert_eq!(
            k_row.len(),
            v_row.len(),
            "kv cache: key/value row length mismatch"
        );
        debug_assert!(
            self.width == 0 || k_row.len() % self.width == 0,
            "kv cache: row length {} is not a multiple of width {}",
            k_row.len(),
            self.width
        );
        let l = &mut self.layers[layer];
        l.k.extend_from_slice(k_row);
        l.v.extend_from_slice(v_row);
    }

    /// Commits `n_positions` newly appended positions, advancing the position
    /// clock once per decode step (after all layers have been appended). Kept
    /// separate from [`append`](Self::append) so the count is independent of the
    /// layer count — a zero-layer configuration still advances correctly.
    pub fn advance(&mut self, n_positions: usize) {
        self.pos += n_positions;
    }

    /// The cached key rows for `layer`, row-major `[positions, width]`.
    ///
    /// # Panics
    ///
    /// Panics if `layer >= n_layers`.
    #[must_use]
    pub fn k(&self, layer: usize) -> &[f32] {
        &self.layers[layer].k
    }

    /// The cached value rows for `layer`, row-major `[positions, width]`.
    ///
    /// # Panics
    ///
    /// Panics if `layer >= n_layers`.
    #[must_use]
    pub fn v(&self, layer: usize) -> &[f32] {
        &self.layers[layer].v
    }

    /// Number of committed positions (tokens) in the cache.
    #[must_use]
    pub fn positions(&self) -> usize {
        self.pos
    }

    /// Rewinds the cache to empty, keeping the reserved capacity so a fresh
    /// decode of the same length reuses the same allocation.
    pub fn reset(&mut self) {
        for l in &mut self.layers {
            l.k.clear();
            l.v.clear();
        }
        self.pos = 0;
    }

    /// Positions that fit in the currently reserved capacity without a
    /// reallocation (the minimum across layers; `0` if there are no layers or
    /// the width is zero). Useful for a streaming caller to size its reserve.
    #[must_use]
    pub fn capacity_positions(&self) -> usize {
        if self.width == 0 {
            return 0;
        }
        self.layers
            .iter()
            .map(|l| l.k.capacity() / self.width)
            .min()
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Compile-time proof that the cache is thread-transferable (the whole point
    /// of promoting it out of the model): only `Vec<f32>` / `usize` inside.
    fn assert_send<T: Send>() {}

    #[test]
    fn kv_cache_is_send() {
        assert_send::<KvCache>();
    }

    #[test]
    fn with_reserve_starts_empty_with_capacity() {
        let cache = KvCache::with_reserve(3, 4, 8);
        assert_eq!(cache.positions(), 0);
        assert!(cache.k(0).is_empty());
        assert!(cache.v(2).is_empty());
        // Reserved but not filled.
        assert!(cache.capacity_positions() >= 8);
    }

    #[test]
    fn append_and_advance_grow_in_lockstep() {
        let mut cache = KvCache::with_reserve(2, 2, 4);
        // First position: append a width-2 row to each layer, then commit once.
        cache.append(0, &[1.0, 2.0], &[3.0, 4.0]);
        cache.append(1, &[5.0, 6.0], &[7.0, 8.0]);
        cache.advance(1);
        assert_eq!(cache.positions(), 1);
        assert_eq!(cache.k(0), &[1.0, 2.0]);
        assert_eq!(cache.v(0), &[3.0, 4.0]);
        assert_eq!(cache.k(1), &[5.0, 6.0]);

        // Second position on layer 0 only shows growth in that layer's buffer.
        cache.append(0, &[9.0, 10.0], &[11.0, 12.0]);
        assert_eq!(cache.k(0), &[1.0, 2.0, 9.0, 10.0]);
    }

    #[test]
    fn advance_is_independent_of_layer_count() {
        // A zero-layer cache still advances its position clock — the invariant a
        // zero-layer decoder configuration relies on.
        let mut cache = KvCache::with_reserve(0, 2, 4);
        assert_eq!(cache.capacity_positions(), 0);
        cache.advance(3);
        assert_eq!(cache.positions(), 3);
    }

    #[test]
    fn reset_clears_length_but_keeps_capacity() {
        let mut cache = KvCache::with_reserve(1, 2, 16);
        let cap_before = cache.capacity_positions();
        for _ in 0..4 {
            cache.append(0, &[0.0, 0.0], &[0.0, 0.0]);
            cache.advance(1);
        }
        assert_eq!(cache.positions(), 4);
        assert_eq!(cache.k(0).len(), 8);

        cache.reset();
        assert_eq!(cache.positions(), 0);
        assert!(cache.k(0).is_empty());
        // Capacity is retained across a reset.
        assert_eq!(cache.capacity_positions(), cap_before);
    }

    #[test]
    fn multi_row_append_counts_positions_via_advance() {
        // A prefix appended as a single [rows, width] block, committed once.
        let mut cache = KvCache::with_reserve(1, 2, 8);
        cache.append(0, &[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0]);
        cache.advance(2);
        assert_eq!(cache.positions(), 2);
        assert_eq!(cache.k(0).len(), 4);
    }
}
