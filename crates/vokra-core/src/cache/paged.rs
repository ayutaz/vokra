//! Paged KV cache with a `[time, stream, codebook]` 3D logical address
//! (FR-EX-03, M3-03).
//!
//! # Why paged
//!
//! Autoregressive audio decoders (Whisper large-v3, CosyVoice2, Voxtral,
//! piper-plus, and — with M3-06 — Mimi/RVQ codec state) share three
//! requirements that the flat [`KvCache`](super::KvCache) does not model:
//!
//! - **Multi-stream isolation** (FR-SV-06, M3-15 `vokra-server`): several
//!   concurrent decode sessions must live in one cache so a pool can reset
//!   only the state of the stream that just returned.
//! - **Codebook dimension** (M3-06 Mimi = 8 codebooks × 12.5 Hz): kernels
//!   `GEMV` across codebook, so the codebook axis has to stay contiguous.
//! - **Hot-path allocator quiet** (FR-EX-05): a session may never realloc a
//!   `Vec` mid-decode; the M1 cache upheld this only via generous
//!   `Vec::with_capacity` hints, whereas a paged store hands the caller a
//!   bounded arena and an O(1) `free_list` pop with capacity assertions.
//!
//! # Layout contract
//!
//! Logical address is a 4-tuple `(layer, time, stream, codebook)`. `time` is
//! the paged axis; `stream / codebook / head / d_head / k|v` are contiguous
//! inside a page. A single page holds one `layer`'s state for `block_size`
//! consecutive time steps. Page memory layout (row-major, most-significant
//! axis first):
//!
//! ```text
//!   [ block_offset, stream, codebook, head, d_head, k|v ]
//! ```
//!
//! `block_size = 4` (LLM-style, primary for 25–50 Hz decoders like Whisper
//! large-v3 / CosyVoice2 / Voxtral) or `block_size = 2` (audio-native,
//! primary for Mimi 12.5 Hz and streaming). Adopting the LLM default of 16
//! wastes ~75% memory at these frame rates — see
//! `docs/adr/M3-03-paged-kv-cache.md` §3 for the arithmetic.
//!
//! # Zero-dep + safe
//!
//! Everything under this module is safe Rust with no external dependencies
//! (NFR-DS-02, NFR-RL-07). The GPU seam `GpuPagedKvCacheOps` is a **trait
//! declaration only** in the M3-03 land — the Metal / CUDA `paged mode`
//! implementations ship in a follow-up WP co-updating the backend crates.
//!
//! # Example
//!
//! ```
//! use vokra_core::cache::paged::{BlockSize, KvDims, PagedKvCache};
//!
//! let dims = KvDims {
//!     n_layer: 1,
//!     n_head: 2,
//!     d_head: 4,
//!     n_stream: 1,
//!     n_codebook: 1,
//!     max_time: 8,
//! };
//! let mut cache = PagedKvCache::<f32>::pre_allocate(dims, BlockSize::Four)?;
//! let row_len = 2 * 4; // n_head * d_head
//! let k_row = vec![1.0_f32; row_len];
//! let v_row = vec![2.0_f32; row_len];
//! cache.append_step(0, 0, 0, 0, &k_row, &v_row)?;
//! cache.advance(1);
//! assert_eq!(cache.positions(), 1);
//! let (k, v) = cache.read_step(0, 0, 0, 0).expect("row is now committed");
//! assert_eq!(k, &k_row[..]);
//! assert_eq!(v, &v_row[..]);
//! # Ok::<(), vokra_core::VokraError>(())
//! ```

use std::ops::Range;
use std::ptr::NonNull;

use crate::error::{Result, VokraError};

/// Elements storable in a [`PagedKvCache`].
///
/// The M3-03 land only implements the [`f32`] path. The M3-04 quantized
/// variants (`Q4_0` / `Q5_0` / `Q8_0`) live in a **parallel** cache type
/// [`QuantizedPagedKvCache`](super::paged_quant::QuantizedPagedKvCache),
/// not in this trait, so `PagedKvCache<f32>`'s existing callers are
/// unaffected (ADR M3-03 §D3).
///
/// The `quant_kind()` extension point is retained on the trait as
/// documentation of the M3-04 seam — the FP32 impl always returns `None`.
pub trait KvElement: Copy + Send + Sync + 'static {
    /// Additive identity, used to zero pages on session reset.
    const ZERO: Self;

    /// The KV quantization discriminant for this element, if any (M3-04-T05).
    ///
    /// FP32 returns `None`; the M3-04 quantized formats live in
    /// [`QuantizedPagedKvCache`](super::paged_quant::QuantizedPagedKvCache),
    /// and this hook exists to let a future generic caller distinguish
    /// FP32 vs quantized without matching on a concrete type.
    #[inline]
    fn quant_kind() -> Option<crate::kv_quant::QuantKind> {
        None
    }
}

impl KvElement for f32 {
    const ZERO: Self = 0.0;
}

/// Time-axis block size. Only 2 and 4 are supported by design; see
/// `docs/adr/M3-03-paged-kv-cache.md` §D2 for why the LLM-style 16 is
/// deliberately not offered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockSize {
    /// `block_size = 2` — audio-native, for 12.5 Hz Mimi codec state or
    /// fine-grained streaming (1 block = 160 ms at 12.5 Hz).
    Two,
    /// `block_size = 4` — primary for 25–50 Hz Whisper / CosyVoice2 /
    /// Voxtral / piper-plus decoders (1 block = 80–160 ms).
    Four,
}

impl BlockSize {
    /// The block size as a `usize` divisor. Inlined so `t / bs` and
    /// `t % bs` fold into a shift / mask on the two supported values.
    #[inline]
    #[must_use]
    pub const fn divisor(self) -> usize {
        match self {
            Self::Two => 2,
            Self::Four => 4,
        }
    }

    /// `t / block_size`: the physical page index of a logical time step.
    #[inline]
    #[must_use]
    pub const fn page_of(self, t: usize) -> usize {
        t / self.divisor()
    }

    /// `t % block_size`: the row offset inside the page for a logical time
    /// step.
    #[inline]
    #[must_use]
    pub const fn offset_in_page(self, t: usize) -> usize {
        t % self.divisor()
    }
}

/// Shape parameters common to every paged cache session.
///
/// The five axes describe a single-model session; `n_stream > 1` is only
/// meaningful for a multi-session server (M3-15). `max_time` is the *hard*
/// upper bound — the arena is sized to fit it, and [`PagedKvCache::append_step`]
/// returns [`VokraError::KvCacheExhausted`] once every page for that layer is
/// live.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvDims {
    /// Number of transformer decoder layers.
    pub n_layer: usize,
    /// Number of attention heads.
    pub n_head: usize,
    /// Per-head channel width.
    pub d_head: usize,
    /// Number of concurrent decode streams (server multi-session). Set to
    /// `1` for a single-session decoder (Whisper large-v3, piper-plus).
    pub n_stream: usize,
    /// Number of codebooks for RVQ codec state (M3-06 Mimi). Set to `1` for
    /// a plain transformer decoder.
    pub n_codebook: usize,
    /// Hard upper bound on the number of time steps.
    pub max_time: usize,
}

impl KvDims {
    /// One page's row width in element count: `n_stream * n_codebook *
    /// n_head * d_head`. `k` and `v` each carry `block_size` rows of this
    /// width, doubled for `k|v` in [`KvPage::row_len_kv`].
    #[inline]
    #[must_use]
    pub const fn row_width(&self) -> usize {
        self.n_stream * self.n_codebook * self.n_head * self.d_head
    }

    /// Number of pages a single layer needs to cover `max_time`. The `+ bs
    /// - 1` division is the standard ceiling in `usize`.
    #[inline]
    #[must_use]
    pub const fn pages_per_layer(&self, block_size: BlockSize) -> usize {
        self.max_time.div_ceil(block_size.divisor())
    }
}

/// Physical page identifier. Newtype so a caller cannot accidentally arithmetic
/// it against a logical time step.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PageId(pub usize);

/// A resolved slot inside the physical page storage.
///
/// The tuple is what a GPU kernel would upload to its indirect index buffer
/// (see [`GpuPagedKvCacheOps`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvSlot {
    /// The physical page holding this slot.
    pub page_id: PageId,
    /// Row offset inside `page_id`, in the range `0..block_size`.
    pub offset_in_page: usize,
    /// Logical stream index (0..n_stream).
    pub stream: usize,
    /// Logical codebook index (0..n_codebook).
    pub codebook: usize,
}

/// One page of KV data. Sized at construction and never resized after.
///
/// Storage is row-major `[block_offset, stream, codebook, head, d_head]`
/// separately for K and V. The two halves live in one flat `Vec<T>` (K in
/// `0..half`, V in `half..len`) so a page hands its GPU counterpart a single
/// device pointer.
pub(crate) struct KvPage<T: KvElement> {
    data: Vec<T>,
    row_width: usize,
    block_size: usize,
}

impl<T: KvElement> KvPage<T> {
    fn new_zeroed(row_width: usize, block_size: usize) -> Self {
        let len = 2 * block_size * row_width; // K + V
        let mut data = Vec::with_capacity(len);
        data.resize(len, T::ZERO);
        Self {
            data,
            row_width,
            block_size,
        }
    }

    #[inline]
    fn k_row(&self, offset: usize) -> &[T] {
        let base = offset * self.row_width;
        &self.data[base..base + self.row_width]
    }

    #[inline]
    fn v_row(&self, offset: usize) -> &[T] {
        let base = self.block_size * self.row_width + offset * self.row_width;
        &self.data[base..base + self.row_width]
    }

    #[inline]
    fn k_row_mut(&mut self, offset: usize) -> &mut [T] {
        let base = offset * self.row_width;
        &mut self.data[base..base + self.row_width]
    }

    #[inline]
    fn v_row_mut(&mut self, offset: usize) -> &mut [T] {
        let base = self.block_size * self.row_width + offset * self.row_width;
        &mut self.data[base..base + self.row_width]
    }

    fn zero(&mut self) {
        for slot in &mut self.data {
            *slot = T::ZERO;
        }
    }

    fn capacity_bytes(&self) -> usize {
        self.data.capacity() * std::mem::size_of::<T>()
    }
}

/// Session-lifetime page allocator (FR-EX-05).
///
/// Pages are allocated up front by [`PagedKvCache::pre_allocate`] and either
/// live in the free list (available to hand out) or the page table (currently
/// bound to a `(layer, time-block)` logical slot). [`Self::acquire`] pops from
/// the free list in O(1) — no system allocator involvement. [`Self::release`]
/// pushes back, again in O(1); the LIFO order improves cache locality when a
/// short decode segment reuses the pages a previous segment just released.
///
/// `capacity` is fixed at construction; the underlying `Vec`s never grow. The
/// hot path is prevented from reallocating by giving both the arena and the
/// free list `Vec::with_capacity(capacity)` up front (verified by the
/// `capacity_stays_stable_across_hot_path` test).
pub(crate) struct PageAllocator<T: KvElement> {
    /// The page arena. Indexed by `PageId(idx)`. Never grown after
    /// construction.
    arena: Vec<KvPage<T>>,
    /// Available page ids, LIFO. `Vec::with_capacity(capacity)` up front so
    /// hot-path `push` / `pop` never triggers a realloc.
    free_list: Vec<PageId>,
    /// Fixed capacity — sized to `dims.pages_per_layer(bs) * n_layer`.
    capacity: usize,
}

impl<T: KvElement> PageAllocator<T> {
    fn new(capacity: usize, row_width: usize, block_size: usize) -> Self {
        let mut arena = Vec::with_capacity(capacity);
        let mut free_list = Vec::with_capacity(capacity);
        for idx in 0..capacity {
            arena.push(KvPage::new_zeroed(row_width, block_size));
            free_list.push(PageId(idx));
        }
        // LIFO ordering: last pushed = first popped. Doesn't affect correctness
        // but keeps low-numbered pages hot in caches across acquire/release
        // churn (M2-03 CudaDecodeSessionPool pattern).
        free_list.reverse();
        Self {
            arena,
            free_list,
            capacity,
        }
    }

    #[inline]
    fn in_use(&self) -> usize {
        self.capacity - self.free_list.len()
    }

    /// Pops a free page in O(1). Returns [`VokraError::KvCacheExhausted`] on
    /// underflow rather than growing the arena (FR-EX-05).
    fn acquire(&mut self) -> Result<PageId> {
        match self.free_list.pop() {
            Some(id) => Ok(id),
            None => Err(VokraError::KvCacheExhausted {
                capacity: self.capacity,
                in_use: self.capacity, // free_list empty → every page live
            }),
        }
    }

    /// Returns a page to the pool. LIFO so back-to-back
    /// `release → acquire` reuses the same page.
    fn release(&mut self, page: PageId) {
        // `push` on a Vec pre-sized to `capacity` never reallocates as long as
        // `in_use()` was ≥ 1 (i.e. this release matches a prior acquire).
        // Verified by the test `capacity_stays_stable_across_hot_path`.
        self.free_list.push(page);
    }

    /// Bulk-return every acquired page to the pool and zero them, in
    /// preparation for a fresh decode of the same shape. Mirrors
    /// [`KvCache::reset`](super::KvCache::reset).
    fn reset(&mut self) {
        // Zero every page so a subsequent read on a fresh session cannot see
        // stale data from the previous decode (defence in depth against a
        // caller that reads before `append_step` has committed).
        for page in &mut self.arena {
            page.zero();
        }
        // Repopulate the free list in the same LIFO order as construction.
        self.free_list.clear();
        for idx in (0..self.capacity).rev() {
            self.free_list.push(PageId(idx));
        }
    }
}

/// Paged KV cache manager (FR-EX-03).
///
/// See the module docs and `docs/adr/M3-03-paged-kv-cache.md` for the full
/// design contract. Construct with [`Self::pre_allocate`], then per decode
/// step append via [`Self::append_step`] and read via [`Self::read_step`] /
/// [`Self::iter_time_range`]. [`Self::advance`] commits the position clock,
/// [`Self::reset`] rewinds to empty while preserving the arena.
pub struct PagedKvCache<T: KvElement> {
    /// Page arena + free list, sized once at construction.
    allocator: PageAllocator<T>,
    /// Time-axis block size (see [`BlockSize`]).
    block_size: BlockSize,
    /// Shape parameters (see [`KvDims`]).
    dims: KvDims,
    /// Per-layer page table indexed `[layer * pages_per_layer + block]`. `None`
    /// means "no page bound to this time-block yet"; `Some(pid)` names the
    /// live page in the arena. The physical size of this vector is fixed at
    /// construction — no growth on the hot path.
    page_table: Vec<Option<PageId>>,
    /// Number of pages a single layer needs to cover `max_time`.
    pages_per_layer: usize,
    /// Committed positions across the cache (mirrors
    /// [`KvCache::positions`](super::KvCache::positions)).
    pos: usize,
    /// Write stamp per `(page, row-in-page, stream, codebook)` — the generation
    /// of the stream that last wrote that row (cc-37).
    ///
    /// [`Self::release_stream`] retires a stream slot in O(1) by bumping its
    /// entry in `stream_gen`; every row this stream previously stamped then
    /// carries a stale generation and reads as unbound.
    ///
    /// The granularity is the full addressable row, not the page, and both of
    /// the extra axes are load-bearing — a coarser stamp is re-validated by the
    /// next occupant's first write and leaks everything else it covers:
    ///
    /// - per *page* would leak the rest of the time block, since a page holds
    ///   `block_size` consecutive time steps;
    /// - per `(page, row, stream)` would leak the other codebooks at the same
    ///   time step, which is exactly what an RVQ decode varies (M3-03's
    ///   `[time, stream, codebook]`; Mimi / DAC / Moshi all decode
    ///   multi-codebook).
    ///
    /// Sized once in [`Self::pre_allocate`] and never resized (FR-EX-05). The
    /// overhead is one `u64` per `(layer, time, stream, codebook)` against the
    /// `n_head · d_head · 2` FP32 KV values for that same tuple — ~0.08% at
    /// Whisper large-v3's `n_head · d_head = 1280`.
    row_stamp: Vec<u64>,
    /// Current generation of each stream slot, indexed by stream. Starts at 0,
    /// matching the initial `row_stamp` values, so a cache that never calls
    /// [`Self::release_stream`] behaves exactly as it did before cc-37.
    stream_gen: Vec<u64>,
    /// Cached `block_size · n_stream · n_codebook` — the per-page stride into
    /// `row_stamp`.
    stamp_page_stride: usize,
}

impl<T: KvElement> PagedKvCache<T> {
    /// Constructs a fully pre-allocated paged cache for `dims` with the chosen
    /// [`BlockSize`].
    ///
    /// Every page needed to cover `dims.n_layer × dims.max_time` is allocated
    /// eagerly, so subsequent [`Self::append_step`] calls never invoke the
    /// system allocator (FR-EX-05).
    ///
    /// # Errors
    ///
    /// Returns [`VokraError::InvalidArgument`] if any of `dims.n_layer` /
    /// `dims.n_head` / `dims.d_head` / `dims.n_stream` / `dims.n_codebook` /
    /// `dims.max_time` is zero — a zero-sized decoder has no meaningful state
    /// and would only make later reads UB-ish.
    pub fn pre_allocate(dims: KvDims, block_size: BlockSize) -> Result<Self> {
        if dims.n_layer == 0
            || dims.n_head == 0
            || dims.d_head == 0
            || dims.n_stream == 0
            || dims.n_codebook == 0
            || dims.max_time == 0
        {
            return Err(VokraError::InvalidArgument(format!(
                "PagedKvCache::pre_allocate: every axis must be > 0, got {dims:?}"
            )));
        }
        let pages_per_layer = dims.pages_per_layer(block_size);
        let total_pages = pages_per_layer * dims.n_layer;
        let row_width = dims.row_width();
        let allocator = PageAllocator::new(total_pages, row_width, block_size.divisor());
        let page_table = vec![None; total_pages];
        // One stamp per (page, row-in-page, stream, codebook). Allocated once
        // here and never resized, like every other arena in this type
        // (FR-EX-05).
        let stamp_page_stride = block_size.divisor() * dims.n_stream * dims.n_codebook;
        let row_stamp = vec![0u64; total_pages * stamp_page_stride];
        Ok(Self {
            allocator,
            block_size,
            dims,
            page_table,
            pages_per_layer,
            pos: 0,
            row_stamp,
            stream_gen: vec![0u64; dims.n_stream],
            stamp_page_stride,
        })
    }

    /// Index into [`Self::row_stamp`] for one `(page, row-in-page, stream)`.
    #[inline]
    const fn stamp_index(
        &self,
        page: PageId,
        offset_in_page: usize,
        stream: usize,
        codebook: usize,
    ) -> usize {
        page.0 * self.stamp_page_stride
            + (offset_in_page * self.dims.n_stream + stream) * self.dims.n_codebook
            + codebook
    }

    /// Whether the row at `(page, offset_in_page, stream, codebook)` was
    /// written by the stream slot's *current* occupant.
    ///
    /// `false` means the row belongs to a generation retired by
    /// [`Self::release_stream`] and must read as unbound — that is the whole
    /// isolation guarantee of the O(1) release.
    #[inline]
    fn row_is_live(
        &self,
        page: PageId,
        offset_in_page: usize,
        stream: usize,
        codebook: usize,
    ) -> bool {
        self.row_stamp[self.stamp_index(page, offset_in_page, stream, codebook)]
            == self.stream_gen[stream]
    }

    /// Bounds-checked logical → physical resolution (T03).
    ///
    /// Returns [`VokraError::InvalidArgument`] on any out-of-range axis, and
    /// leaves it to the caller to distinguish "unbound" (`Ok` with the returned
    /// [`KvSlot`] pointing at the page table's `None` — surfaced via
    /// [`Self::read_step`] returning [`None`]) from "out of range". Explicitly
    /// avoids `panic!` because the paged cache runs behind a public FFI façade
    /// (NFR-RL-07).
    ///
    /// A row written before a [`Self::release_stream`] on `s` also resolves to
    /// `Ok(None)`: it is physically present but logically retired, and handing
    /// its address out would let a caller read the previous occupant's state
    /// (cc-37).
    pub fn logical_at(&self, layer: usize, t: usize, s: usize, c: usize) -> Result<Option<KvSlot>> {
        self.check_bounds(layer, t, s, c)?;
        let block = self.block_size.page_of(t);
        let table_idx = layer * self.pages_per_layer + block;
        let offset_in_page = self.block_size.offset_in_page(t);
        Ok(self.page_table[table_idx].and_then(|page_id| {
            self.row_is_live(page_id, offset_in_page, s, c)
                .then_some(KvSlot {
                    page_id,
                    offset_in_page,
                    stream: s,
                    codebook: c,
                })
        }))
    }

    /// The physical page index of a logical time step (T04). Convenience
    /// alias for `self.block_size().page_of(t)` — kept as an inherent method
    /// so a reader working purely at the logical layer never has to reach for
    /// [`BlockSize`].
    #[inline]
    #[must_use]
    pub const fn page_of(&self, t: usize) -> usize {
        self.block_size.page_of(t)
    }

    /// Row offset of a logical time step inside its page (T04).
    #[inline]
    #[must_use]
    pub const fn offset_in_page(&self, t: usize) -> usize {
        self.block_size.offset_in_page(t)
    }

    /// Iterator yielding every committed [`KvSlot`] in the half-open time
    /// range `range` for `layer` / `stream` / `codebook = 0` (single-codebook
    /// fast path — the RVQ multi-codebook shape lands with M3-06). Skips
    /// unbound blocks silently.
    ///
    /// Traverses block boundaries transparently — the caller reads
    /// `range.len()` continuous time steps and only pays a page-table lookup
    /// per block.
    ///
    /// # Errors
    ///
    /// Returns [`VokraError::InvalidArgument`] if `range.end > max_time` or
    /// `layer >= n_layer`.
    pub fn iter_time_range(
        &self,
        layer: usize,
        stream: usize,
        codebook: usize,
        range: Range<usize>,
    ) -> Result<TimeRangeIter<'_, T>> {
        if layer >= self.dims.n_layer {
            return Err(VokraError::InvalidArgument(format!(
                "iter_time_range: layer {layer} >= n_layer {}",
                self.dims.n_layer
            )));
        }
        if stream >= self.dims.n_stream {
            return Err(VokraError::InvalidArgument(format!(
                "iter_time_range: stream {stream} >= n_stream {}",
                self.dims.n_stream
            )));
        }
        if codebook >= self.dims.n_codebook {
            return Err(VokraError::InvalidArgument(format!(
                "iter_time_range: codebook {codebook} >= n_codebook {}",
                self.dims.n_codebook
            )));
        }
        if range.end > self.dims.max_time {
            return Err(VokraError::InvalidArgument(format!(
                "iter_time_range: range.end {} > max_time {}",
                range.end, self.dims.max_time
            )));
        }
        Ok(TimeRangeIter {
            cache: self,
            layer,
            stream,
            codebook,
            next: range.start,
            end: range.end,
        })
    }

    /// Appends one time-step's `k` / `v` row to `(layer, t, s, c)` (T09).
    ///
    /// Acquires a fresh page from the free list on the first append to a
    /// block; subsequent appends into the same block reuse the bound page. No
    /// system allocation is invoked — an exhausted arena returns
    /// [`VokraError::KvCacheExhausted`] instead (FR-EX-05).
    ///
    /// # Errors
    ///
    /// - [`VokraError::InvalidArgument`] on out-of-range axes or wrong
    ///   `k_row` / `v_row` length.
    /// - [`VokraError::KvCacheExhausted`] if every page in the arena is
    ///   already live.
    pub fn append_step(
        &mut self,
        layer: usize,
        t: usize,
        s: usize,
        c: usize,
        k_row: &[T],
        v_row: &[T],
    ) -> Result<()> {
        self.check_bounds(layer, t, s, c)?;
        let per_slot = self.dims.n_head * self.dims.d_head;
        if k_row.len() != per_slot || v_row.len() != per_slot {
            return Err(VokraError::InvalidArgument(format!(
                "append_step: expected k/v row len {per_slot}, got k={} v={}",
                k_row.len(),
                v_row.len()
            )));
        }
        let block = self.block_size.page_of(t);
        let table_idx = layer * self.pages_per_layer + block;
        let page_id = match self.page_table[table_idx] {
            Some(pid) => pid,
            None => {
                let pid = self.allocator.acquire()?;
                self.page_table[table_idx] = Some(pid);
                pid
            }
        };
        let offset = self.block_size.offset_in_page(t);
        // Claim this row for the stream slot's current occupant (cc-37). One
        // store on the append hot path; it is what makes `release_stream` O(1).
        let stamp_idx = self.stamp_index(page_id, offset, s, c);
        self.row_stamp[stamp_idx] = self.stream_gen[s];
        let page = &mut self.allocator.arena[page_id.0];
        let sc_offset = (s * self.dims.n_codebook + c) * per_slot;
        page.k_row_mut(offset)[sc_offset..sc_offset + per_slot].copy_from_slice(k_row);
        page.v_row_mut(offset)[sc_offset..sc_offset + per_slot].copy_from_slice(v_row);
        Ok(())
    }

    /// Reads the K/V row previously written by [`Self::append_step`], or
    /// [`None`] if that block has never been written on this `layer` — or if
    /// the row was retired by a [`Self::release_stream`] on `s` (cc-37).
    ///
    /// The returned slices are borrows into the arena and are stable across
    /// subsequent reads until the next [`Self::append_step`] or
    /// [`Self::reset`] on this cache.
    #[must_use]
    pub fn read_step(&self, layer: usize, t: usize, s: usize, c: usize) -> Option<(&[T], &[T])> {
        // Silent early-return on out-of-range: `read_step` is meant to be
        // interior to attention kernels which have already bounds-checked, and
        // callers who care about the distinction should use `logical_at`.
        if layer >= self.dims.n_layer
            || t >= self.dims.max_time
            || s >= self.dims.n_stream
            || c >= self.dims.n_codebook
        {
            return None;
        }
        let block = self.block_size.page_of(t);
        let table_idx = layer * self.pages_per_layer + block;
        let page_id = self.page_table[table_idx]?;
        let offset = self.block_size.offset_in_page(t);
        // Retired-generation rows read as unbound (cc-37) — the physical bytes
        // survive until the page is reused, so the stamp is what enforces
        // isolation between successive occupants of a stream slot.
        if !self.row_is_live(page_id, offset, s, c) {
            return None;
        }
        let page = &self.allocator.arena[page_id.0];
        let per_slot = self.dims.n_head * self.dims.d_head;
        let sc_offset = (s * self.dims.n_codebook + c) * per_slot;
        let k = &page.k_row(offset)[sc_offset..sc_offset + per_slot];
        let v = &page.v_row(offset)[sc_offset..sc_offset + per_slot];
        Some((k, v))
    }

    /// Commits `n_positions` newly appended time steps, advancing the position
    /// clock once per decode step. Matches the semantics of
    /// [`KvCache::advance`](super::KvCache::advance).
    pub fn advance(&mut self, n_positions: usize) {
        self.pos += n_positions;
    }

    /// Number of committed time steps across the cache.
    #[inline]
    #[must_use]
    pub const fn positions(&self) -> usize {
        self.pos
    }

    /// Rewinds the cache to empty while preserving the arena and the free-list
    /// order. A fresh decode of the same shape reuses every buffer.
    ///
    /// The cc-37 stream generations and row stamps are deliberately left as
    /// they are. They cannot leak anything across a reset: the page table is
    /// cleared (so every read short-circuits to [`None`] until a page is
    /// re-bound) and `allocator.reset()` zeroes every page (so a stamp that
    /// happens to still match can only expose zeros). Rewriting them would be
    /// `O(pages · block_size · n_stream)` of pointless work.
    pub fn reset(&mut self) {
        for slot in &mut self.page_table {
            *slot = None;
        }
        self.allocator.reset();
        self.pos = 0;
    }

    /// Releases every page currently bound to `layer`, returning them to the
    /// free list without touching the position clock or other layers'
    /// bindings. Intended for stream-level teardown in the M3-15 server path,
    /// where one finished stream should not require draining the entire
    /// cache. Zeroes the page contents on release so a later reuse cannot see
    /// stale data.
    ///
    /// # Errors
    ///
    /// Returns [`VokraError::InvalidArgument`] if `layer >= n_layer`.
    pub fn release_layer(&mut self, layer: usize) -> Result<()> {
        if layer >= self.dims.n_layer {
            return Err(VokraError::InvalidArgument(format!(
                "release_layer: layer {layer} >= n_layer {}",
                self.dims.n_layer
            )));
        }
        let base = layer * self.pages_per_layer;
        for slot in &mut self.page_table[base..base + self.pages_per_layer] {
            if let Some(pid) = slot.take() {
                // Zero before returning to free list so subsequent acquire()
                // sees a fresh page (defence in depth for the same rationale
                // as PageAllocator::reset).
                self.allocator.arena[pid.0].zero();
                self.allocator.release(pid);
            }
        }
        Ok(())
    }

    /// Retires every row belonging to `stream`, in **O(1)**, so the next
    /// session to take that stream slot cannot observe the previous
    /// occupant's KV state (cc-37, M3-15 T04 follow-up).
    ///
    /// # Why this is not a zero-fill
    ///
    /// A page holds one `(layer, time-block)` for **every** stream, so there is
    /// no page subset that belongs to one stream — the stream axis lives
    /// *inside* each page. Erasing a stream by overwriting its rows therefore
    /// costs `O(n_layer · committed_time · n_head · d_head)`, which is what the
    /// M3-15 server registry did (looping `append_step` with zero rows) and
    /// what this method replaces.
    ///
    /// Instead each stream slot carries a generation counter. Retiring the slot
    /// bumps it, which invalidates every row the previous occupant stamped:
    /// [`Self::read_step`] and [`Self::logical_at`] both return [`None`] for a
    /// stale row, so the bytes — though still physically present until the page
    /// is reused — are unreachable through the public API. Reuse of the
    /// *physical* page still zeroes it ([`Self::release_layer`],
    /// [`Self::reset`]), so the two mechanisms are independent defences.
    ///
    /// # What it does not do
    ///
    /// It does **not** return pages to the free list: pages are shared across
    /// streams by construction, so a page stays live while any stream still
    /// needs it. Page reclamation remains [`Self::release_layer`] /
    /// [`Self::reset`]. It also leaves the position clock untouched — the clock
    /// is cache-global, and the caller (a session registry) owns per-stream
    /// bookkeeping.
    ///
    /// # Generation exhaustion
    ///
    /// The counter is `u64` and increments once per release, so wrap-around
    /// needs 2^64 releases of the same slot — unreachable in practice (at one
    /// release per nanosecond it is ~584 years). A narrower counter would make
    /// wrap-around a silent isolation failure, which is why this is not `u32`.
    ///
    /// # Errors
    ///
    /// Returns [`VokraError::InvalidArgument`] if `stream >= n_stream`.
    pub fn release_stream(&mut self, stream: usize) -> Result<()> {
        if stream >= self.dims.n_stream {
            return Err(VokraError::InvalidArgument(format!(
                "release_stream: stream {stream} >= n_stream {}",
                self.dims.n_stream
            )));
        }
        // The entire release: one increment, independent of n_layer, max_time
        // and the number of committed positions.
        self.stream_gen[stream] += 1;
        Ok(())
    }

    /// The maximum number of time steps this cache can hold before
    /// [`Self::append_step`] returns [`VokraError::KvCacheExhausted`]. Mirrors
    /// [`KvCache::capacity_positions`](super::KvCache::capacity_positions).
    #[inline]
    #[must_use]
    pub const fn capacity_positions(&self) -> usize {
        self.dims.max_time
    }

    /// Number of pages the arena was sized for at construction. Test hook so
    /// the capacity-stability assertion (FR-EX-05) can inspect the underlying
    /// allocator without exposing internal types.
    #[must_use]
    pub fn arena_capacity_pages(&self) -> usize {
        self.allocator.capacity
    }

    /// Number of pages currently checked out (i.e. bound to a `(layer,
    /// time-block)` slot). Complement of [`Self::free_pages`].
    #[must_use]
    pub fn pages_in_use(&self) -> usize {
        self.allocator.in_use()
    }

    /// Number of pages available to hand out.
    #[must_use]
    pub fn free_pages(&self) -> usize {
        self.allocator.free_list.len()
    }

    /// Test hook returning the underlying `Vec`s' capacities for the
    /// hot-path-malloc-free assertion (T14).
    #[must_use]
    pub fn allocator_capacity_snapshot(&self) -> AllocatorSnapshot {
        AllocatorSnapshot {
            arena_capacity: self.allocator.arena.capacity(),
            free_list_capacity: self.allocator.free_list.capacity(),
            page_table_capacity: self.page_table.capacity(),
        }
    }

    /// The chosen block size (accessor).
    #[inline]
    #[must_use]
    pub const fn block_size(&self) -> BlockSize {
        self.block_size
    }

    /// The dimensions this cache was constructed with (accessor).
    #[inline]
    #[must_use]
    pub const fn dims(&self) -> &KvDims {
        &self.dims
    }

    /// Total bytes committed to page storage (used by the T16 footprint
    /// bench). Excludes the page-table + free-list overhead which is O(#pages)
    /// of `Option<usize>` / `usize`.
    #[must_use]
    pub fn page_storage_bytes(&self) -> usize {
        self.allocator
            .arena
            .iter()
            .map(KvPage::capacity_bytes)
            .sum()
    }

    #[inline]
    fn check_bounds(&self, layer: usize, t: usize, s: usize, c: usize) -> Result<()> {
        if layer >= self.dims.n_layer {
            return Err(VokraError::InvalidArgument(format!(
                "layer {layer} >= n_layer {}",
                self.dims.n_layer
            )));
        }
        if t >= self.dims.max_time {
            return Err(VokraError::InvalidArgument(format!(
                "t {t} >= max_time {}",
                self.dims.max_time
            )));
        }
        if s >= self.dims.n_stream {
            return Err(VokraError::InvalidArgument(format!(
                "stream {s} >= n_stream {}",
                self.dims.n_stream
            )));
        }
        if c >= self.dims.n_codebook {
            return Err(VokraError::InvalidArgument(format!(
                "codebook {c} >= n_codebook {}",
                self.dims.n_codebook
            )));
        }
        Ok(())
    }
}

/// Snapshot of the allocator's inner `Vec` capacities. Fields are `usize` so a
/// test can `assert_eq!` two snapshots to prove the hot path did not touch the
/// system allocator (T14 replacement for the deferred global counting
/// allocator).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AllocatorSnapshot {
    /// `Vec::capacity()` of the arena (`Vec<KvPage<T>>`).
    pub arena_capacity: usize,
    /// `Vec::capacity()` of the free list (`Vec<PageId>`).
    pub free_list_capacity: usize,
    /// `Vec::capacity()` of the page table (`Vec<Option<PageId>>`).
    pub page_table_capacity: usize,
}

/// Iterator over [`KvSlot`]s spanning a contiguous time range (T05).
pub struct TimeRangeIter<'a, T: KvElement> {
    cache: &'a PagedKvCache<T>,
    layer: usize,
    stream: usize,
    codebook: usize,
    next: usize,
    end: usize,
}

impl<T: KvElement> Iterator for TimeRangeIter<'_, T> {
    type Item = KvSlot;

    fn next(&mut self) -> Option<Self::Item> {
        while self.next < self.end {
            let t = self.next;
            self.next += 1;
            // logical_at only ever returns Err on out-of-bounds axes; we've
            // already validated `range.end <= max_time` at iterator
            // construction, so any Err here is a bug — fall through to `None`
            // rather than panicking. Unbound blocks are the expected "skip"
            // case (`Ok(None)`).
            match self
                .cache
                .logical_at(self.layer, t, self.stream, self.codebook)
            {
                Ok(Some(slot)) => return Some(slot),
                Ok(None) => continue,
                Err(_) => return None,
            }
        }
        None
    }
}

/// Backend-agnostic seam for a GPU-side paged KV cache (T13).
///
/// This trait is a **shape declaration only** in the M3-03 land — the Metal
/// (T11) and CUDA (T12) implementations ship in a follow-up WP co-updating
/// the backend crates, per `docs/adr/M3-03-paged-kv-cache.md` §2 D5. The
/// trait's presence here lets the M3-15 server code type-check against a
/// backend-neutral surface even before the concrete impls land.
///
/// # Contract
///
/// An implementation of this trait uploads a page-pointer array to device
/// memory (`upload_page_table`), then answers indirect page lookups from a
/// kernel via `lookup_page`. The kernel itself is expected to be co-authored
/// with each backend — the trait does not attempt to abstract over kernel
/// launch shape.
pub trait GpuPagedKvCacheOps {
    /// Uploads a device-side array of page base pointers, one per physical
    /// page in the arena.
    ///
    /// # Safety-adjacent
    ///
    /// The trait is safe Rust; the implementation is free to use FFI to talk
    /// to Metal / CUDA. Callers only pass verified `NonNull<u8>` device
    /// pointers, so the trait signature stays safe on this side.
    fn upload_page_table(&mut self, page_base_ptrs: &[NonNull<u8>]);

    /// Resolves a [`PageId`] to the corresponding device pointer that was
    /// uploaded via [`Self::upload_page_table`].
    ///
    /// # Errors
    ///
    /// Returns [`VokraError::InvalidArgument`] if `page.0 >=
    /// uploaded_len`.
    fn lookup_page(&self, page: PageId) -> Result<NonNull<u8>>;

    /// Appends one time-step's K/V state to the GPU-side paged cache. Kept
    /// abstract to keep the trait small; the concrete backends bind this to
    /// their kernel launch (Metal `MetalKvCache::append_paged`, CUDA
    /// `CudaKvCache::append_paged`) in the follow-up WP.
    fn append_kv_paged(&mut self, layer: usize, t: usize, s: usize, c: usize) -> Result<()>;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Compile-time check that the paged cache is `Send` (parity with the
    /// existing [`KvCache`](super::KvCache) test).
    fn assert_send<T: Send>() {}

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

    #[test]
    fn paged_kv_cache_is_send() {
        assert_send::<PagedKvCache<f32>>();
    }

    #[test]
    fn block_size_page_and_offset_bs4() {
        let bs = BlockSize::Four;
        assert_eq!(bs.divisor(), 4);
        assert_eq!(bs.page_of(0), 0);
        assert_eq!(bs.page_of(3), 0);
        assert_eq!(bs.page_of(4), 1);
        assert_eq!(bs.page_of(15), 3);
        assert_eq!(bs.offset_in_page(0), 0);
        assert_eq!(bs.offset_in_page(3), 3);
        assert_eq!(bs.offset_in_page(4), 0);
        assert_eq!(bs.offset_in_page(15), 3);
    }

    #[test]
    fn block_size_page_and_offset_bs2() {
        let bs = BlockSize::Two;
        assert_eq!(bs.divisor(), 2);
        assert_eq!(bs.page_of(0), 0);
        assert_eq!(bs.page_of(1), 0);
        assert_eq!(bs.page_of(2), 1);
        assert_eq!(bs.page_of(9), 4);
        assert_eq!(bs.offset_in_page(0), 0);
        assert_eq!(bs.offset_in_page(1), 1);
        assert_eq!(bs.offset_in_page(2), 0);
        assert_eq!(bs.offset_in_page(9), 1);
    }

    #[test]
    fn dims_row_width_and_pages_per_layer() {
        // Whisper large-v3 shape (n_text_ctx=448, n_layer=32, n_head=20, d_head=64):
        let d = dims(32, 20, 64, 448);
        assert_eq!(d.row_width(), 20 * 64);
        assert_eq!(d.pages_per_layer(BlockSize::Four), 112);
        assert_eq!(d.pages_per_layer(BlockSize::Two), 224);
        // Ceiling behaviour for a non-multiple max_time.
        let d = dims(1, 1, 1, 5);
        assert_eq!(d.pages_per_layer(BlockSize::Four), 2); // ceil(5/4) = 2
        assert_eq!(d.pages_per_layer(BlockSize::Two), 3); // ceil(5/2) = 3
    }

    #[test]
    fn pre_allocate_rejects_zero_axes() {
        let d = dims(0, 1, 1, 4);
        assert!(matches!(
            PagedKvCache::<f32>::pre_allocate(d, BlockSize::Four),
            Err(VokraError::InvalidArgument(_))
        ));
        let d = dims(1, 1, 1, 0);
        assert!(matches!(
            PagedKvCache::<f32>::pre_allocate(d, BlockSize::Four),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn pre_allocate_sizes_the_arena() {
        // 3 layers * ceil(8/4) = 3 * 2 = 6 pages
        let d = dims(3, 2, 4, 8);
        let cache = PagedKvCache::<f32>::pre_allocate(d, BlockSize::Four).unwrap();
        assert_eq!(cache.arena_capacity_pages(), 6);
        assert_eq!(cache.free_pages(), 6);
        assert_eq!(cache.pages_in_use(), 0);
        assert_eq!(cache.positions(), 0);
        assert_eq!(cache.capacity_positions(), 8);
    }

    #[test]
    fn append_step_binds_page_and_persists_row() {
        // max_time=8 → 2 blocks at bs=4, so we can distinguish "same block,
        // unwritten sibling" (bound-but-zeroed) from "different block, no
        // append yet" (unbound, None).
        let d = dims(1, 2, 3, 8);
        let mut cache = PagedKvCache::<f32>::pre_allocate(d, BlockSize::Four).unwrap();
        let per_slot = 2 * 3;
        let k = vec![1.0_f32; per_slot];
        let v = vec![2.0_f32; per_slot];
        cache.append_step(0, 0, 0, 0, &k, &v).unwrap();
        cache.advance(1);
        assert_eq!(cache.positions(), 1);
        assert_eq!(cache.pages_in_use(), 1);
        let (kr, vr) = cache.read_step(0, 0, 0, 0).unwrap();
        assert_eq!(kr, &k[..]);
        assert_eq!(vr, &v[..]);
        // A sibling step *in the same block* is bound-but-zeroed: writing t=0
        // binds the whole block, so read_step at t=1 returns `Some(zeros)`
        // rather than `None`. This block-level binding matches how a GPU
        // kernel iterates a whole page in one indirect indexing pass.
        let (kr, vr) = cache.read_step(0, 1, 0, 0).expect("block is bound");
        assert_eq!(kr, &[0.0_f32; 6]);
        assert_eq!(vr, &[0.0_f32; 6]);
        // A step in an *unbound* block still returns None (no accidental
        // arena leak across blocks).
        assert!(cache.read_step(0, 4, 0, 0).is_none());
    }

    #[test]
    fn append_step_reuses_page_within_block() {
        let d = dims(1, 1, 2, 8);
        let mut cache = PagedKvCache::<f32>::pre_allocate(d, BlockSize::Four).unwrap();
        let per_slot = 2;
        // Four writes at t=0..3 all fall in block 0 → same page.
        for t in 0..4 {
            let k = vec![t as f32; per_slot];
            let v = vec![-(t as f32); per_slot];
            cache.append_step(0, t, 0, 0, &k, &v).unwrap();
        }
        cache.advance(4);
        assert_eq!(cache.pages_in_use(), 1);
        for t in 0..4 {
            let (kr, vr) = cache.read_step(0, t, 0, 0).unwrap();
            assert_eq!(kr, &vec![t as f32; per_slot][..]);
            assert_eq!(vr, &vec![-(t as f32); per_slot][..]);
        }
    }

    #[test]
    fn append_step_across_block_boundary_binds_new_page() {
        let d = dims(1, 1, 1, 6);
        let mut cache = PagedKvCache::<f32>::pre_allocate(d, BlockSize::Two).unwrap();
        // block_size = 2 → each block holds 2 time steps. Writing t=0,1,2 uses 2 blocks.
        for t in 0..3 {
            let k = [t as f32];
            let v = [(t * 10) as f32];
            cache.append_step(0, t, 0, 0, &k, &v).unwrap();
        }
        assert_eq!(cache.pages_in_use(), 2);
    }

    #[test]
    fn append_step_rejects_bad_row_length() {
        let d = dims(1, 2, 4, 4);
        let mut cache = PagedKvCache::<f32>::pre_allocate(d, BlockSize::Four).unwrap();
        let bad = vec![0.0_f32; 3];
        let good = vec![0.0_f32; 8];
        assert!(matches!(
            cache.append_step(0, 0, 0, 0, &bad, &good),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn logical_at_bounds_check() {
        let d = dims(1, 1, 1, 4);
        let cache = PagedKvCache::<f32>::pre_allocate(d, BlockSize::Four).unwrap();
        // Out-of-range time.
        assert!(matches!(
            cache.logical_at(0, 4, 0, 0),
            Err(VokraError::InvalidArgument(_))
        ));
        // Out-of-range layer.
        assert!(matches!(
            cache.logical_at(1, 0, 0, 0),
            Err(VokraError::InvalidArgument(_))
        ));
        // Unbound block returns Ok(None).
        assert!(matches!(cache.logical_at(0, 0, 0, 0), Ok(None)));
    }

    #[test]
    fn iter_time_range_covers_multi_block_span() {
        let d = dims(1, 1, 1, 8);
        let mut cache = PagedKvCache::<f32>::pre_allocate(d, BlockSize::Two).unwrap();
        // Populate t=0..6 (3 blocks with block_size=2).
        for t in 0..6 {
            cache.append_step(0, t, 0, 0, &[t as f32], &[0.0]).unwrap();
        }
        cache.advance(6);
        // Range fully inside a single block.
        let slots: Vec<_> = cache.iter_time_range(0, 0, 0, 0..2).unwrap().collect();
        assert_eq!(slots.len(), 2);
        assert_eq!(slots[0].page_id, slots[1].page_id);
        assert_eq!(slots[0].offset_in_page, 0);
        assert_eq!(slots[1].offset_in_page, 1);
        // Range that crosses a block boundary.
        let slots: Vec<_> = cache.iter_time_range(0, 0, 0, 1..3).unwrap().collect();
        assert_eq!(slots.len(), 2);
        assert_ne!(slots[0].page_id, slots[1].page_id);
        // Range that spans three consecutive blocks.
        let slots: Vec<_> = cache.iter_time_range(0, 0, 0, 0..6).unwrap().collect();
        assert_eq!(slots.len(), 6);
        // Unique pages = 3.
        let mut pages: Vec<_> = slots.iter().map(|s| s.page_id).collect();
        pages.sort_by_key(|p| p.0);
        pages.dedup();
        assert_eq!(pages.len(), 3);
    }

    #[test]
    fn iter_time_range_skips_unbound_blocks() {
        let d = dims(1, 1, 1, 8);
        let mut cache = PagedKvCache::<f32>::pre_allocate(d, BlockSize::Two).unwrap();
        // Only touch t=0 and t=4 → blocks 0 and 2 bound, block 1 (t=2,3) empty.
        cache.append_step(0, 0, 0, 0, &[1.0], &[1.0]).unwrap();
        cache.append_step(0, 4, 0, 0, &[5.0], &[5.0]).unwrap();
        let slots: Vec<_> = cache.iter_time_range(0, 0, 0, 0..6).unwrap().collect();
        // 2 bound blocks × 1 populated step each (with one implicitly-zeroed
        // sibling in each), plus the block 2 sibling. But logical_at returns
        // Some for any bound block regardless of individual-step writes, so
        // block 0 (offset 0 and 1) + block 2 (offset 0 and 1) = 4 slots. Block 1
        // is unbound and skipped.
        assert_eq!(slots.len(), 4);
    }

    #[test]
    fn iter_time_range_out_of_bounds_fails_explicitly() {
        let d = dims(1, 1, 1, 4);
        let cache = PagedKvCache::<f32>::pre_allocate(d, BlockSize::Four).unwrap();
        assert!(matches!(
            cache.iter_time_range(0, 0, 0, 0..5),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn exhausted_returns_explicit_error_not_realloc() {
        // Arena for exactly 1 page (block_size=4, max_time=4, 1 layer).
        let d = dims(1, 1, 1, 4);
        let mut cache = PagedKvCache::<f32>::pre_allocate(d, BlockSize::Four).unwrap();
        // First append binds the only page.
        cache.append_step(0, 0, 0, 0, &[1.0], &[1.0]).unwrap();
        // Second append lands in the same block, reuses the page — still fine.
        cache.append_step(0, 1, 0, 0, &[1.0], &[1.0]).unwrap();
        // Extend max_time to force a *second* block: build a fresh cache with
        // max_time=8 but drain its arena manually. We simulate exhaustion by
        // directly asking the allocator for more pages than it has.
        let d2 = dims(1, 1, 1, 4); // still 1 page
        let mut cache2 = PagedKvCache::<f32>::pre_allocate(d2, BlockSize::Four).unwrap();
        cache2.append_step(0, 0, 0, 0, &[1.0], &[1.0]).unwrap();
        // Manually clear the page table for block 0 without releasing the page,
        // then try to bind block 0 again. This is a synthetic-yet-realistic
        // simulation of "all pages live, another block needs one".
        cache2.page_table[0] = None;
        let err = cache2.append_step(0, 0, 0, 0, &[1.0], &[1.0]).unwrap_err();
        assert!(matches!(err, VokraError::KvCacheExhausted { .. }));
    }

    #[test]
    fn reset_clears_state_but_keeps_capacity() {
        let d = dims(2, 1, 2, 8);
        let mut cache = PagedKvCache::<f32>::pre_allocate(d, BlockSize::Four).unwrap();
        for t in 0..8 {
            cache
                .append_step(0, t, 0, 0, &[t as f32, 0.0], &[0.0, 0.0])
                .unwrap();
            cache
                .append_step(1, t, 0, 0, &[0.0, 0.0], &[0.0, 0.0])
                .unwrap();
        }
        cache.advance(8);
        let snap_before = cache.allocator_capacity_snapshot();
        assert!(cache.pages_in_use() > 0);

        cache.reset();

        assert_eq!(cache.positions(), 0);
        assert_eq!(cache.pages_in_use(), 0);
        assert_eq!(cache.free_pages(), cache.arena_capacity_pages());
        // No system reallocation happened during reset — capacities identical.
        let snap_after = cache.allocator_capacity_snapshot();
        assert_eq!(snap_before, snap_after);
        // And the previously-populated slot now reads as unbound.
        assert!(cache.read_step(0, 0, 0, 0).is_none());
    }

    /// FR-EX-05: the hot path must not touch the system allocator. We can't
    /// swap the global allocator in a unit test without touching unsafe global
    /// state (deferred to M3-15), so instead we assert that the underlying
    /// `Vec`s' capacities are bit-identical from just-after-construction
    /// through a full decode + reset cycle.
    #[test]
    fn capacity_stays_stable_across_hot_path() {
        let d = dims(4, 2, 4, 32);
        let mut cache = PagedKvCache::<f32>::pre_allocate(d, BlockSize::Four).unwrap();
        let baseline = cache.allocator_capacity_snapshot();
        let per_slot = 8;
        let k = vec![0.5_f32; per_slot];
        let v = vec![-0.5_f32; per_slot];
        for step in 0..3 {
            // Multiple mini-decodes of 32 steps each. Each reset returns every
            // page to the free list; nothing should ever cross the system
            // allocator.
            for t in 0..32 {
                for layer in 0..4 {
                    cache.append_step(layer, t, 0, 0, &k, &v).unwrap();
                }
                cache.advance(1);
            }
            let mid = cache.allocator_capacity_snapshot();
            assert_eq!(baseline, mid, "hot-path realloc detected at step {step}");
            cache.reset();
        }
        let after = cache.allocator_capacity_snapshot();
        assert_eq!(baseline, after);
    }

    #[test]
    fn multi_session_concurrent_streams_isolate() {
        // T15: run 4 streams through one PagedKvCache with n_stream=4, then
        // compare each stream's committed state against the same content run
        // through a *dedicated* single-stream cache. Any cross-stream leakage
        // shows up as a mismatch.
        let n_stream = 4;
        let d = KvDims {
            n_layer: 2,
            n_head: 1,
            d_head: 3,
            n_stream,
            n_codebook: 1,
            max_time: 6,
        };
        let mut shared = PagedKvCache::<f32>::pre_allocate(d, BlockSize::Two).unwrap();

        // A distinct scalar per stream so any bleed is visible.
        let stream_signature: Vec<f32> = (0..n_stream).map(|s| (s + 1) as f32).collect();
        for t in 0..6 {
            for (s, sig) in stream_signature.iter().enumerate() {
                let base = sig * 10.0 + t as f32;
                let k = [base, base + 0.1, base + 0.2];
                let v = [base + 1.0, base + 1.1, base + 1.2];
                for layer in 0..d.n_layer {
                    shared.append_step(layer, t, s, 0, &k, &v).unwrap();
                }
            }
        }

        // Solo runs.
        let solo_dims = KvDims { n_stream: 1, ..d };
        for (s, sig) in stream_signature.iter().enumerate() {
            let mut solo = PagedKvCache::<f32>::pre_allocate(solo_dims, BlockSize::Two).unwrap();
            for t in 0..6 {
                let base = sig * 10.0 + t as f32;
                let k = [base, base + 0.1, base + 0.2];
                let v = [base + 1.0, base + 1.1, base + 1.2];
                for layer in 0..solo_dims.n_layer {
                    solo.append_step(layer, t, 0, 0, &k, &v).unwrap();
                }
            }
            for t in 0..6 {
                for layer in 0..solo_dims.n_layer {
                    let shared_row = shared.read_step(layer, t, s, 0).unwrap();
                    let solo_row = solo.read_step(layer, t, 0, 0).unwrap();
                    assert_eq!(
                        shared_row, solo_row,
                        "stream {s} layer {layer} t {t}: shared cache diverged from solo"
                    );
                }
            }
        }
    }

    /// T16: memory footprint sanity check. block_size=4 must use strictly less
    /// storage than the hypothetical block_size=16 (LLM default) for the same
    /// shape. The exact 75% figure in the ADR is *max waste*, not average;
    /// this test only pins the direction.
    #[test]
    fn block_size_four_is_more_memory_efficient_than_hypothetical_sixteen() {
        // We can't construct a PagedKvCache with block_size=16 (by design), so
        // model the comparison manually against BlockSize::Four for a shape
        // that stresses waste — max_time = 17 (one leftover row into a fresh
        // block for bs=4, and 15 leftover rows for the hypothetical bs=16).
        let n_head = 2;
        let d_head = 4;
        let n_layer = 1;
        let max_time = 17;

        let bs4_pages = (max_time + 3) / 4; // 5
        let bs4_bytes =
            bs4_pages * n_layer * 4 /* block_size */ * n_head * d_head * 2 /* k+v */ * 4; // f32

        let bs16_pages = (max_time + 15) / 16; // 2
        let bs16_bytes = bs16_pages * n_layer * 16 * n_head * d_head * 2 * 4;

        assert!(
            bs4_bytes < bs16_bytes,
            "bs=4 bytes {bs4_bytes} not < bs=16 bytes {bs16_bytes}"
        );

        // And bs=2 is smaller still for the same worst case.
        let bs2_pages = (max_time + 1) / 2; // 9
        let bs2_bytes = bs2_pages * n_layer * 2 * n_head * d_head * 2 * 4;
        assert!(bs2_bytes < bs4_bytes);
    }

    /// T16 support: `page_storage_bytes` matches the ADR's arithmetic for a
    /// canonical Whisper-large-v3 shape at block_size=4.
    #[test]
    fn page_storage_bytes_matches_shape_arithmetic() {
        // Trimmed shape: n_layer=2, n_head=4, d_head=8, max_time=16.
        let d = dims(2, 4, 8, 16);
        let cache = PagedKvCache::<f32>::pre_allocate(d, BlockSize::Four).unwrap();
        // pages = 2 * ceil(16/4) = 8; per-page bytes = 4 * (4*8) * 2 * 4 = 1024.
        let expected = 8 * 4 * (4 * 8) * 2 * std::mem::size_of::<f32>();
        assert_eq!(cache.page_storage_bytes(), expected);
    }

    #[test]
    fn release_layer_returns_pages_to_free_list() {
        // Two layers, each needing 2 pages at bs=4 for max_time=8 = 4 pages total.
        let d = dims(2, 1, 1, 8);
        let mut cache = PagedKvCache::<f32>::pre_allocate(d, BlockSize::Four).unwrap();
        // Bind both blocks of both layers.
        for t in [0usize, 4] {
            for layer in 0..2 {
                cache
                    .append_step(layer, t, 0, 0, &[t as f32], &[0.0])
                    .unwrap();
            }
        }
        assert_eq!(cache.pages_in_use(), 4);
        cache.advance(2);

        // Release only layer 0. Layer 1's bindings must survive.
        cache.release_layer(0).unwrap();
        assert_eq!(cache.pages_in_use(), 2);
        // Layer 0 is now unbound everywhere.
        for t in [0usize, 4] {
            assert!(matches!(cache.logical_at(0, t, 0, 0), Ok(None)));
        }
        // Layer 1 still reads back correctly.
        for t in [0usize, 4] {
            let (kr, vr) = cache.read_step(1, t, 0, 0).unwrap();
            assert_eq!(kr, &[t as f32]);
            assert_eq!(vr, &[0.0]);
        }

        // Reacquiring for layer 0 after release does not exhaust the arena.
        cache.append_step(0, 0, 0, 0, &[99.0], &[0.0]).unwrap();
        assert_eq!(cache.pages_in_use(), 3);

        // Position clock is untouched by release_layer.
        assert_eq!(cache.positions(), 2);
    }

    #[test]
    fn release_layer_rejects_out_of_range() {
        let d = dims(1, 1, 1, 4);
        let mut cache = PagedKvCache::<f32>::pre_allocate(d, BlockSize::Four).unwrap();
        assert!(matches!(
            cache.release_layer(1),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    // ---- cc-37: O(1) `release_stream` -------------------------------------

    /// A released stream's rows are unreadable through every public read path.
    /// This is the property a naive O(1) release — one that merely returned the
    /// slot to a free list without invalidating anything — would violate: the
    /// bytes are still physically in the page, so `read_step` would happily
    /// hand back the previous occupant's K/V.
    #[test]
    fn release_stream_hides_the_previous_occupants_rows() {
        let d = KvDims {
            n_layer: 2,
            n_head: 1,
            d_head: 2,
            n_stream: 2,
            n_codebook: 1,
            max_time: 8,
        };
        let mut cache = PagedKvCache::<f32>::pre_allocate(d, BlockSize::Four).unwrap();

        // Occupant A writes a recognisable signature on stream 0.
        for t in 0..8 {
            for layer in 0..d.n_layer {
                cache
                    .append_step(layer, t, 0, 0, &[7.0, 8.0], &[9.0, 10.0])
                    .unwrap();
            }
        }
        cache.advance(8);
        assert_eq!(
            cache.read_step(0, 0, 0, 0),
            Some((&[7.0, 8.0][..], &[9.0, 10.0][..]))
        );

        cache.release_stream(0).unwrap();

        // Every (layer, t) on stream 0 must now be unreachable — not merely
        // zeroed, but absent. A stale `Some(..)` here is a state leak.
        for t in 0..8 {
            for layer in 0..d.n_layer {
                assert_eq!(
                    cache.read_step(layer, t, 0, 0),
                    None,
                    "layer {layer} t {t}: released stream row is still readable"
                );
                assert_eq!(
                    cache.logical_at(layer, t, 0, 0).unwrap(),
                    None,
                    "layer {layer} t {t}: released stream row still resolves to a slot"
                );
            }
        }
        // `iter_time_range` rides on `logical_at`, so it must skip them too.
        assert_eq!(cache.iter_time_range(0, 0, 0, 0..8).unwrap().count(), 0);
    }

    /// The subtle case a *per-page* generation stamp would miss, and the reason
    /// the stamp is per row.
    ///
    /// A page spans `block_size` consecutive time steps. If the new occupant of
    /// a stream slot writes a single step, a per-page stamp would re-validate
    /// the whole page and expose the previous occupant's rows at the other
    /// offsets in that same page. Per-row stamping keeps them retired.
    #[test]
    fn release_stream_isolation_survives_a_partial_page_rewrite() {
        let d = KvDims {
            n_layer: 1,
            n_head: 1,
            d_head: 2,
            n_stream: 1,
            n_codebook: 1,
            max_time: 4,
        };
        // block_size = 4, so t = 0..4 all live in ONE page.
        let mut cache = PagedKvCache::<f32>::pre_allocate(d, BlockSize::Four).unwrap();
        for t in 0..4 {
            let x = 100.0 + t as f32;
            cache.append_step(0, t, 0, 0, &[x, x], &[x, x]).unwrap();
        }
        cache.advance(4);

        cache.release_stream(0).unwrap();

        // New occupant writes ONLY t = 0, into the very page the old occupant
        // filled.
        cache
            .append_step(0, 0, 0, 0, &[1.0, 1.0], &[2.0, 2.0])
            .unwrap();

        assert_eq!(
            cache.read_step(0, 0, 0, 0),
            Some((&[1.0, 1.0][..], &[2.0, 2.0][..])),
            "the new occupant's own row must be readable"
        );
        for t in 1..4 {
            assert_eq!(
                cache.read_step(0, t, 0, 0),
                None,
                "t {t}: previous occupant's row leaked through a partial page rewrite"
            );
        }
    }

    /// The same partial-rewrite hazard on the **codebook** axis, which is the
    /// axis an RVQ decode actually varies (`[time, stream, codebook]`, M3-03).
    ///
    /// A stamp keyed only by `(page, row, stream)` would be re-validated for
    /// *every* codebook the moment the new occupant wrote any one of them, so
    /// writing codebook 0 at `t` would expose the previous occupant's codebook
    /// 1 at the same `t`. Mimi / DAC / Moshi all decode multi-codebook, so this
    /// is a reachable leak rather than a theoretical one.
    #[test]
    fn release_stream_isolation_is_per_codebook() {
        let d = KvDims {
            n_layer: 1,
            n_head: 1,
            d_head: 2,
            n_stream: 1,
            n_codebook: 2,
            max_time: 4,
        };
        let mut cache = PagedKvCache::<f32>::pre_allocate(d, BlockSize::Four).unwrap();

        // Occupant A fills both codebooks at t = 0.
        cache
            .append_step(0, 0, 0, 0, &[11.0, 11.0], &[12.0, 12.0])
            .unwrap();
        cache
            .append_step(0, 0, 0, 1, &[21.0, 21.0], &[22.0, 22.0])
            .unwrap();
        cache.advance(1);

        cache.release_stream(0).unwrap();

        // Occupant B writes ONLY codebook 0, at the same (layer, t, stream).
        cache
            .append_step(0, 0, 0, 0, &[1.0, 1.0], &[2.0, 2.0])
            .unwrap();

        assert_eq!(
            cache.read_step(0, 0, 0, 0),
            Some((&[1.0, 1.0][..], &[2.0, 2.0][..])),
            "the new occupant's own codebook must be readable"
        );
        assert_eq!(
            cache.read_step(0, 0, 0, 1),
            None,
            "codebook 1 still holds occupant A's row — writing codebook 0 must not \
             re-validate the whole (page, row, stream) tuple"
        );
    }

    /// Releasing one stream leaves every other stream's committed state
    /// bit-intact — the release is scoped to its slot, not to the pages it
    /// happens to share.
    #[test]
    fn release_stream_does_not_disturb_other_streams() {
        let d = KvDims {
            n_layer: 1,
            n_head: 1,
            d_head: 2,
            n_stream: 3,
            n_codebook: 1,
            max_time: 4,
        };
        let mut cache = PagedKvCache::<f32>::pre_allocate(d, BlockSize::Two).unwrap();
        for t in 0..4 {
            for s in 0..3 {
                let x = (s * 100 + t) as f32;
                cache.append_step(0, t, s, 0, &[x, x], &[x, x]).unwrap();
            }
        }
        cache.advance(4);

        cache.release_stream(1).unwrap();

        for t in 0..4 {
            for s in [0usize, 2] {
                let x = (s * 100 + t) as f32;
                assert_eq!(
                    cache.read_step(0, t, s, 0),
                    Some((&[x, x][..], &[x, x][..])),
                    "stream {s} t {t} was disturbed by releasing stream 1"
                );
            }
            assert_eq!(
                cache.read_step(0, t, 1, 0),
                None,
                "stream 1 t {t} not retired"
            );
        }
    }

    /// A released slot is fully reusable: the next occupant writes and reads its
    /// own state normally, and a second release retires *that* state in turn.
    #[test]
    fn released_stream_slot_is_reusable_across_generations() {
        let d = dims(1, 1, 2, 4);
        let mut cache = PagedKvCache::<f32>::pre_allocate(d, BlockSize::Two).unwrap();

        for generation in 1..=4u32 {
            let x = generation as f32;
            for t in 0..4 {
                cache.append_step(0, t, 0, 0, &[x, x], &[x, x]).unwrap();
            }
            for t in 0..4 {
                assert_eq!(
                    cache.read_step(0, t, 0, 0),
                    Some((&[x, x][..], &[x, x][..])),
                    "generation {generation} t {t}: own state not readable"
                );
            }
            cache.release_stream(0).unwrap();
            for t in 0..4 {
                assert_eq!(cache.read_step(0, t, 0, 0), None);
            }
        }
    }

    #[test]
    fn release_stream_rejects_out_of_range() {
        let d = dims(1, 1, 1, 4);
        let mut cache = PagedKvCache::<f32>::pre_allocate(d, BlockSize::Four).unwrap();
        assert!(matches!(
            cache.release_stream(1),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    /// `release_stream` reclaims no pages (they are shared across streams) and
    /// does not move the position clock. Pinning this keeps the contract from
    /// silently acquiring page-reclamation semantics later.
    #[test]
    fn release_stream_leaves_pages_and_clock_alone() {
        let d = dims(2, 1, 1, 8);
        let mut cache = PagedKvCache::<f32>::pre_allocate(d, BlockSize::Four).unwrap();
        for t in 0..8 {
            for layer in 0..2 {
                cache.append_step(layer, t, 0, 0, &[1.0], &[2.0]).unwrap();
            }
        }
        cache.advance(8);
        let pages_before = cache.pages_in_use();
        let pos_before = cache.positions();
        assert!(pages_before > 0, "the fixture must actually bind pages");

        cache.release_stream(0).unwrap();

        assert_eq!(cache.pages_in_use(), pages_before);
        assert_eq!(cache.positions(), pos_before);
    }

    /// The complexity claim, proved deterministically rather than timed:
    /// `release_stream` touches **zero** bytes of page storage and zero stamp
    /// entries, whatever the amount of state it retires.
    ///
    /// This is a stronger and far more stable statement than a wall-clock
    /// ratio. A timing comparison here is worse than useless — the O(1) body is
    /// a single increment, so the baseline loop optimises away entirely and the
    /// "ratio" ends up measuring codegen, not asymptotics. Instead the test
    /// snapshots the whole arena plus the stamp table around the call: if any
    /// future implementation reintroduces per-page (or per-row) work, the
    /// snapshot differs and this fails.
    ///
    /// The cost that was removed is reported concretely: the M3-15 registry
    /// zero-filled a released stream by looping `append_step` with zero rows
    /// over every `(layer, committed t)`, which is `n_layer · committed · 2 ·
    /// n_head · d_head` element writes.
    #[test]
    fn release_stream_does_zero_work_proportional_to_state() {
        let d = KvDims {
            n_layer: 8,
            n_head: 2,
            d_head: 32,
            n_stream: 2,
            n_codebook: 1,
            max_time: 512,
        };
        let mut cache = PagedKvCache::<f32>::pre_allocate(d, BlockSize::Four).unwrap();
        let per_slot = d.n_head * d.d_head;
        let k = vec![1.5f32; per_slot];
        let v = vec![2.5f32; per_slot];
        for t in 0..d.max_time {
            for layer in 0..d.n_layer {
                for s in 0..d.n_stream {
                    cache.append_step(layer, t, s, 0, &k, &v).unwrap();
                }
            }
        }
        cache.advance(d.max_time);

        // Snapshot every physical page and the whole stamp table.
        let pages_before: Vec<Vec<f32>> = cache
            .allocator
            .arena
            .iter()
            .map(|p| p.data.clone())
            .collect();
        let stamps_before = cache.row_stamp.clone();
        let gen_before = cache.stream_gen[0];

        cache.release_stream(0).unwrap();

        for (idx, before) in pages_before.iter().enumerate() {
            assert_eq!(
                &cache.allocator.arena[idx].data, before,
                "release_stream wrote to page {idx}: the release is not O(1)"
            );
        }
        assert_eq!(
            cache.row_stamp, stamps_before,
            "release_stream wrote to the stamp table: the release is not O(1)"
        );
        // The one and only mutation.
        assert_eq!(cache.stream_gen[0], gen_before + 1);
        assert_eq!(cache.stream_gen[1], 0, "the other slot must be untouched");

        // ...and the retirement really happened.
        assert_eq!(cache.read_step(0, 0, 0, 0), None);

        let displaced_writes = d.n_layer * d.max_time * 2 * per_slot;
        eprintln!(
            "release_stream: {displaced_writes} element writes (old zero-fill) → 0 \
             (one u64 increment), for n_layer={} committed={} per_slot={per_slot}",
            d.n_layer, d.max_time
        );
    }

    #[test]
    fn allocator_lifo_order() {
        // Newly-constructed allocator hands out lowest-numbered pages first
        // (arena locality: page 0 was zeroed first and is hottest in cache).
        let d = dims(1, 1, 1, 12); // 3 pages at bs=4
        let mut cache = PagedKvCache::<f32>::pre_allocate(d, BlockSize::Four).unwrap();
        // Force distinct pages by writing three separate blocks.
        for t in [0usize, 4, 8] {
            cache.append_step(0, t, 0, 0, &[t as f32], &[0.0]).unwrap();
        }
        // Grab the bound pages in write order via the page_table for
        // observable-order stability testing.
        let block0 = cache.page_table[0].unwrap().0;
        let block1 = cache.page_table[1].unwrap().0;
        let block2 = cache.page_table[2].unwrap().0;
        assert_eq!(block0, 0);
        assert_eq!(block1, 1);
        assert_eq!(block2, 2);
    }
}
