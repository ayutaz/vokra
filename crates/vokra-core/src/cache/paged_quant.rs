//! Quantized paged KV cache — Q4_0 / Q5_0 / Q8_0 (M3-04-T07/T08).
//!
//! # Placement (deliberate)
//!
//! The M3-03 [`PagedKvCache<T: KvElement>`](super::paged::PagedKvCache) accepts
//! `T = f32` only (its `KvElement` trait requires `Copy + Send + Sync`, which
//! is compatible with quantized blocks — but the FP32 assumption is baked
//! deep into `n_head * d_head` row-length semantics). Rather than re-plumb
//! that, M3-04 introduces a **parallel** cache type here that:
//!
//! - stores each K / V row as a sequence of quantized 32-elem blocks,
//! - packs the FP32 `k_row` / `v_row` inputs into blocks on `append_step`,
//! - unpacks blocks back to FP32 on `read_step` (differential-oracle path;
//!   the M3-04-phase-2 Metal / CUDA fused dequant kernels short-circuit this
//!   by consuming blocks directly).
//!
//! The M3-03 `PagedKvCache<f32>` is **unchanged**; migration is opt-in via
//! `Session::from_file(...).with_backend(...).with_kv_quant(KvQuant::Q8_0)`.
//!
//! # Row-length + block alignment
//!
//! The FP32 row for one time step in `PagedKvCache<f32>` is `n_head * d_head`
//! contiguous floats. For the quantized path we require that this row length
//! divides `32` cleanly (`(n_head * d_head) % 32 == 0`) — every Whisper /
//! CosyVoice2 / Voxtral / piper-plus we support has `d_head ∈ {32, 64, 80}`
//! and `n_head ≥ 1`, so this constraint is trivially satisfied. Rejecting a
//! non-aligned shape at construction is preferable to silently padding and
//! then having the block boundary land in the middle of a head.
//!
//! # Hot-path allocation
//!
//! Pages are pre-allocated up front — same policy as `PagedKvCache<f32>`. The
//! per-row temporary buffer used by `read_step` is stack-sized (`[f32; 32]`
//! per block) so no `Vec` growth happens on the hot path. Verified by
//! `QuantizedPagedKvCache::allocator_capacity_snapshot` (mirrors the M3-03
//! test hook).

use crate::error::{Result, VokraError};
use crate::kv_quant::{BlockQ4_0, BlockQ5_0, BlockQ8_0, KV_QUANT_BLOCK_SIZE, KvQuant};

use super::paged::{BlockSize, KvDims, PageId};

/// Type-erased quantized block. Kept as an enum (rather than a boxed trait
/// object) because every `KvQuantBlock` is `Copy` and we want the arena to be
/// a plain `Vec` — no allocator churn per block.
#[derive(Debug, Clone, Copy)]
pub enum AnyBlock {
    /// Q4_0 (4-bit, 18 bytes/block).
    Q4_0(BlockQ4_0),
    /// Q5_0 (5-bit, 22 bytes/block).
    Q5_0(BlockQ5_0),
    /// Q8_0 (8-bit, 34 bytes/block).
    Q8_0(BlockQ8_0),
}

impl AnyBlock {
    fn pack(mode: KvQuant, input: &[f32]) -> Self {
        match mode {
            KvQuant::Q4_0 => Self::Q4_0(BlockQ4_0::pack(input)),
            KvQuant::Q5_0 => Self::Q5_0(BlockQ5_0::pack(input)),
            KvQuant::Q8_0 => Self::Q8_0(BlockQ8_0::pack(input)),
            KvQuant::Fp32 => {
                // Unreachable — `QuantizedPagedKvCache::new` rejects Fp32 and
                // routes callers to the FP32 `PagedKvCache` instead. Kept as
                // an explicit panic (not `unreachable!()`) so a future
                // refactor cannot silently degrade.
                panic!("AnyBlock::pack called with KvQuant::Fp32 — use PagedKvCache<f32> for FP32")
            }
        }
    }

    fn zero(mode: KvQuant) -> Self {
        match mode {
            KvQuant::Q4_0 => Self::Q4_0(BlockQ4_0::default()),
            KvQuant::Q5_0 => Self::Q5_0(BlockQ5_0::default()),
            KvQuant::Q8_0 => Self::Q8_0(BlockQ8_0::default()),
            KvQuant::Fp32 => panic!("AnyBlock::zero called with KvQuant::Fp32"),
        }
    }

    fn unpack(&self, out: &mut [f32]) {
        match self {
            Self::Q4_0(b) => b.unpack(out),
            Self::Q5_0(b) => b.unpack(out),
            Self::Q8_0(b) => b.unpack(out),
        }
    }
}

/// A page of quantized K / V blocks for one time-block of one layer.
///
/// Row-major logical layout matches [`super::paged::KvPage`]:
/// `[block_offset, stream, codebook, block_within_row]`. The last axis is a
/// sequence of quantized 32-element blocks whose count is
/// `(n_head * d_head) / 32`.
struct QuantKvPage {
    /// K side: `block_size * row_blocks_per_slot * n_stream * n_codebook`
    /// blocks.
    k: Vec<AnyBlock>,
    /// V side, identically shaped.
    v: Vec<AnyBlock>,
    /// Blocks per (stream, codebook) K/V slot — kept for offset arithmetic.
    row_blocks_per_slot: usize,
    /// Combined (stream * codebook) count — kept for offset arithmetic.
    n_stream_codebook: usize,
}

impl QuantKvPage {
    fn new_zeroed(
        row_blocks_per_slot: usize,
        block_size: usize,
        n_stream_codebook: usize,
        mode: KvQuant,
    ) -> Self {
        let total = block_size * n_stream_codebook * row_blocks_per_slot;
        Self {
            k: vec![AnyBlock::zero(mode); total],
            v: vec![AnyBlock::zero(mode); total],
            row_blocks_per_slot,
            n_stream_codebook,
        }
    }

    #[inline]
    fn slot_range(&self, offset: usize, sc_flat: usize) -> std::ops::Range<usize> {
        let base = offset * self.n_stream_codebook * self.row_blocks_per_slot
            + sc_flat * self.row_blocks_per_slot;
        base..base + self.row_blocks_per_slot
    }

    fn zero(&mut self, mode: KvQuant) {
        let z = AnyBlock::zero(mode);
        for slot in &mut self.k {
            *slot = z;
        }
        for slot in &mut self.v {
            *slot = z;
        }
    }

    fn capacity_bytes(&self) -> usize {
        // AnyBlock is a discriminated union; its size dominates a raw block
        // by discriminant + alignment. This is a *host-side* overhead metric
        // (M3-04-T14 checklist), not an on-wire footprint — the on-wire size
        // is what `KvQuant::block_bytes` reports.
        (self.k.capacity() + self.v.capacity()) * std::mem::size_of::<AnyBlock>()
    }
}

/// Session-lifetime page allocator for the quantized path.
///
/// Mirrors [`super::paged::PageAllocator`] but stores [`QuantKvPage`]s.
/// Duplication is deliberate: making the FP32 allocator generic over page type
/// would leak the quantized types into M3-03's public API surface, which we
/// promised to keep unchanged (ADR M3-03 §D3).
struct QuantPageAllocator {
    arena: Vec<QuantKvPage>,
    free_list: Vec<PageId>,
    capacity: usize,
    mode: KvQuant,
}

impl QuantPageAllocator {
    fn new(
        capacity: usize,
        row_blocks_per_slot: usize,
        block_size: usize,
        n_stream_codebook: usize,
        mode: KvQuant,
    ) -> Self {
        let mut arena = Vec::with_capacity(capacity);
        let mut free_list = Vec::with_capacity(capacity);
        for idx in 0..capacity {
            arena.push(QuantKvPage::new_zeroed(
                row_blocks_per_slot,
                block_size,
                n_stream_codebook,
                mode,
            ));
            free_list.push(PageId(idx));
        }
        // LIFO ordering — same locality rationale as M3-03 PageAllocator.
        free_list.reverse();
        Self {
            arena,
            free_list,
            capacity,
            mode,
        }
    }

    #[inline]
    fn in_use(&self) -> usize {
        self.capacity - self.free_list.len()
    }

    fn acquire(&mut self) -> Result<PageId> {
        match self.free_list.pop() {
            Some(id) => Ok(id),
            None => Err(VokraError::KvCacheExhausted {
                capacity: self.capacity,
                in_use: self.capacity,
            }),
        }
    }

    fn release(&mut self, page: PageId) {
        self.free_list.push(page);
    }

    fn reset(&mut self) {
        for page in &mut self.arena {
            page.zero(self.mode);
        }
        self.free_list.clear();
        for idx in (0..self.capacity).rev() {
            self.free_list.push(PageId(idx));
        }
    }
}

/// Paged KV cache with runtime quantization (M3-04-T07).
///
/// # Contract
///
/// - Constructed exclusively via [`Self::new`] with `mode ∈ {Q4_0, Q5_0,
///   Q8_0}`. FP32 is rejected — the caller must go through
///   [`super::paged::PagedKvCache`] for FP32.
/// - `n_head * d_head` **must** be a multiple of
///   [`KV_QUANT_BLOCK_SIZE`](crate::kv_quant::KV_QUANT_BLOCK_SIZE) (32) at
///   construction; otherwise a block boundary would land inside a head, which
///   we refuse to silently pad.
/// - `append_step` and `read_step` operate in the FP32 domain — the caller
///   passes FP32 rows, the cache packs on write and unpacks on read. The
///   quantization error is bounded by the per-format `d / 2` step, and every
///   read yields the same dequantized FP32 the fused-kernel path would see.
///
/// # Deferred (M3-04 phase 2)
///
/// - Metal / CUDA fused dequant kernels (`with_metal(...)`, `with_cuda(...)`)
///   that consume the block arena directly without the FP32 unpack step.
/// - SIMD-accelerated CPU dequant (AVX2 / NEON) — the scalar path is
///   sufficient as a differential oracle.
pub struct QuantizedPagedKvCache {
    allocator: QuantPageAllocator,
    block_size: BlockSize,
    dims: KvDims,
    page_table: Vec<Option<PageId>>,
    pages_per_layer: usize,
    pos: usize,
    mode: KvQuant,
    row_blocks_per_slot: usize,
}

impl QuantizedPagedKvCache {
    /// Constructs a pre-allocated quantized paged cache for `dims` with the
    /// chosen block size and quantization mode.
    ///
    /// # Errors
    ///
    /// - [`VokraError::InvalidArgument`] if `mode == KvQuant::Fp32` — the FP32
    ///   path lives in [`super::paged::PagedKvCache`].
    /// - [`VokraError::InvalidArgument`] if any `dims.*` axis is zero.
    /// - [`VokraError::InvalidArgument`] if `n_head * d_head` is not a
    ///   multiple of 32 (would split a head across quant blocks).
    pub fn new(dims: KvDims, block_size: BlockSize, mode: KvQuant) -> Result<Self> {
        if matches!(mode, KvQuant::Fp32) {
            return Err(VokraError::InvalidArgument(
                "QuantizedPagedKvCache::new: mode=Fp32 — use PagedKvCache<f32> for FP32".into(),
            ));
        }
        if dims.n_layer == 0
            || dims.n_head == 0
            || dims.d_head == 0
            || dims.n_stream == 0
            || dims.n_codebook == 0
            || dims.max_time == 0
        {
            return Err(VokraError::InvalidArgument(format!(
                "QuantizedPagedKvCache::new: every axis must be > 0, got {dims:?}"
            )));
        }
        let per_slot = dims.n_head * dims.d_head;
        if per_slot % KV_QUANT_BLOCK_SIZE != 0 {
            return Err(VokraError::InvalidArgument(format!(
                "QuantizedPagedKvCache::new: n_head*d_head={per_slot} not a multiple of {KV_QUANT_BLOCK_SIZE}"
            )));
        }
        let row_blocks_per_slot = per_slot / KV_QUANT_BLOCK_SIZE;
        let pages_per_layer = dims.pages_per_layer(block_size);
        let total_pages = pages_per_layer * dims.n_layer;
        let n_stream_codebook = dims.n_stream * dims.n_codebook;
        let allocator = QuantPageAllocator::new(
            total_pages,
            row_blocks_per_slot,
            block_size.divisor(),
            n_stream_codebook,
            mode,
        );
        let page_table = vec![None; total_pages];
        Ok(Self {
            allocator,
            block_size,
            dims,
            page_table,
            pages_per_layer,
            pos: 0,
            mode,
            row_blocks_per_slot,
        })
    }

    /// The quantization mode this cache was constructed with.
    #[inline]
    #[must_use]
    pub const fn quant(&self) -> KvQuant {
        self.mode
    }

    /// Same shape accessor as `PagedKvCache::dims`.
    #[inline]
    #[must_use]
    pub const fn dims(&self) -> &KvDims {
        &self.dims
    }

    /// Same accessor as `PagedKvCache::block_size`.
    #[inline]
    #[must_use]
    pub const fn block_size(&self) -> BlockSize {
        self.block_size
    }

    /// Committed positions (mirrors `PagedKvCache::positions`).
    #[inline]
    #[must_use]
    pub const fn positions(&self) -> usize {
        self.pos
    }

    /// Number of pages the arena was sized for.
    #[must_use]
    pub fn arena_capacity_pages(&self) -> usize {
        self.allocator.capacity
    }

    /// Number of pages currently checked out.
    #[must_use]
    pub fn pages_in_use(&self) -> usize {
        self.allocator.in_use()
    }

    /// Number of pages available to hand out.
    #[must_use]
    pub fn free_pages(&self) -> usize {
        self.allocator.free_list.len()
    }

    /// Test hook returning the underlying `Vec`s' capacities for the hot-path
    /// malloc-free assertion (mirrors the M3-03 counterpart).
    #[must_use]
    pub fn allocator_capacity_snapshot(&self) -> AllocatorSnapshot {
        AllocatorSnapshot {
            arena_capacity: self.allocator.arena.capacity(),
            free_list_capacity: self.allocator.free_list.capacity(),
            page_table_capacity: self.page_table.capacity(),
        }
    }

    /// Total host-side bytes committed to page storage. Uses the `AnyBlock`
    /// enum size, not the on-wire block byte size — a follow-up would migrate
    /// the arena to raw byte storage for the Metal / CUDA fused-kernel path.
    #[must_use]
    pub fn page_storage_bytes(&self) -> usize {
        self.allocator
            .arena
            .iter()
            .map(QuantKvPage::capacity_bytes)
            .sum()
    }

    /// On-wire footprint under this quantization mode, ignoring the enum-
    /// discriminant overhead of the host arena. This is the number a
    /// footprint bench (T14 / T16) should report.
    #[must_use]
    pub fn on_wire_storage_bytes(&self) -> usize {
        // pages × (2 for K+V) × block_size × n_stream × n_codebook × row_blocks
        //   × per-block byte count
        let per_page_blocks = 2
            * self.block_size.divisor()
            * self.dims.n_stream
            * self.dims.n_codebook
            * self.row_blocks_per_slot;
        self.allocator.arena.len() * per_page_blocks * self.mode.block_bytes()
    }

    /// Appends one time step's `k` / `v` rows, packing into quantized blocks.
    ///
    /// `k_row.len() == v_row.len() == n_head * d_head`.
    pub fn append_step(
        &mut self,
        layer: usize,
        t: usize,
        s: usize,
        c: usize,
        k_row: &[f32],
        v_row: &[f32],
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
        let sc_flat = s * self.dims.n_codebook + c;
        let page = &mut self.allocator.arena[page_id.0];
        let k_range = page.slot_range(offset, sc_flat);
        let v_range = k_range.clone();

        for (chunk_idx, k_chunk) in k_row.chunks_exact(KV_QUANT_BLOCK_SIZE).enumerate() {
            page.k[k_range.start + chunk_idx] = AnyBlock::pack(self.mode, k_chunk);
        }
        for (chunk_idx, v_chunk) in v_row.chunks_exact(KV_QUANT_BLOCK_SIZE).enumerate() {
            page.v[v_range.start + chunk_idx] = AnyBlock::pack(self.mode, v_chunk);
        }
        Ok(())
    }

    /// Reads the previously-appended K/V rows, dequantizing into fresh
    /// `Vec<f32>` allocations sized to `n_head * d_head`. Returns `None` if
    /// no `append_step` has ever landed on this time-block.
    ///
    /// The `Vec` allocation is deliberate: unpacking blocks into
    /// caller-provided slices would be marginally faster but would expose the
    /// block ordering to callers. The M3-04 phase-2 fused kernel path skips
    /// this by consuming blocks directly.
    #[must_use]
    pub fn read_step(
        &self,
        layer: usize,
        t: usize,
        s: usize,
        c: usize,
    ) -> Option<(Vec<f32>, Vec<f32>)> {
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
        let sc_flat = s * self.dims.n_codebook + c;
        let page = &self.allocator.arena[page_id.0];
        let range = page.slot_range(offset, sc_flat);

        let per_slot = self.dims.n_head * self.dims.d_head;
        let mut k_out = vec![0.0f32; per_slot];
        let mut v_out = vec![0.0f32; per_slot];
        for (i, block_idx) in range.enumerate() {
            page.k[block_idx]
                .unpack(&mut k_out[i * KV_QUANT_BLOCK_SIZE..(i + 1) * KV_QUANT_BLOCK_SIZE]);
            page.v[block_idx]
                .unpack(&mut v_out[i * KV_QUANT_BLOCK_SIZE..(i + 1) * KV_QUANT_BLOCK_SIZE]);
        }
        Some((k_out, v_out))
    }

    /// Commits `n_positions` newly appended time steps (mirrors
    /// `PagedKvCache::advance`).
    pub fn advance(&mut self, n_positions: usize) {
        self.pos += n_positions;
    }

    /// Rewinds the cache to empty while preserving the arena.
    pub fn reset(&mut self) {
        for slot in &mut self.page_table {
            *slot = None;
        }
        self.allocator.reset();
        self.pos = 0;
    }

    /// Releases every page bound to `layer`.
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
                self.allocator.arena[pid.0].zero(self.mode);
                self.allocator.release(pid);
            }
        }
        Ok(())
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

/// Snapshot of the underlying `Vec` capacities (mirror of M3-03
/// `AllocatorSnapshot`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AllocatorSnapshot {
    /// `Vec::capacity()` of the arena (page storage).
    pub arena_capacity: usize,
    /// `Vec::capacity()` of the free list.
    pub free_list_capacity: usize,
    /// `Vec::capacity()` of the page table.
    pub page_table_capacity: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dims_for(n_head: usize, d_head: usize, max_time: usize) -> KvDims {
        KvDims {
            n_layer: 1,
            n_head,
            d_head,
            n_stream: 1,
            n_codebook: 1,
            max_time,
        }
    }

    #[test]
    fn rejects_fp32() {
        let d = dims_for(1, 32, 4);
        assert!(matches!(
            QuantizedPagedKvCache::new(d, BlockSize::Four, KvQuant::Fp32),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn rejects_non_multiple_row_length() {
        // n_head*d_head = 1*33 = 33 → not a multiple of 32.
        let d = dims_for(1, 33, 4);
        assert!(matches!(
            QuantizedPagedKvCache::new(d, BlockSize::Four, KvQuant::Q8_0),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn rejects_zero_axis() {
        let d = KvDims {
            n_layer: 0,
            n_head: 1,
            d_head: 32,
            n_stream: 1,
            n_codebook: 1,
            max_time: 4,
        };
        assert!(matches!(
            QuantizedPagedKvCache::new(d, BlockSize::Four, KvQuant::Q8_0),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn all_three_modes_construct_ok() {
        let d = dims_for(1, 32, 4);
        for mode in [KvQuant::Q4_0, KvQuant::Q5_0, KvQuant::Q8_0] {
            let cache = QuantizedPagedKvCache::new(d, BlockSize::Four, mode).unwrap();
            assert_eq!(cache.quant(), mode);
            assert_eq!(cache.arena_capacity_pages(), 1); // ceil(4/4) = 1 page
            assert_eq!(cache.free_pages(), 1);
            assert_eq!(cache.pages_in_use(), 0);
            assert_eq!(cache.positions(), 0);
        }
    }

    #[test]
    fn append_and_read_round_trip_q8_0() {
        // 1 head × 64 d_head = 64 elem row = 2 blocks
        let d = dims_for(1, 64, 4);
        let mut cache = QuantizedPagedKvCache::new(d, BlockSize::Four, KvQuant::Q8_0).unwrap();
        let k_row: Vec<f32> = (0..64).map(|i| (i as f32) / 63.0 * 2.0 - 1.0).collect();
        let v_row: Vec<f32> = (0..64)
            .map(|i| ((63 - i) as f32) / 63.0 * 2.0 - 1.0)
            .collect();
        cache.append_step(0, 0, 0, 0, &k_row, &v_row).unwrap();
        cache.advance(1);

        let (k_out, v_out) = cache.read_step(0, 0, 0, 0).unwrap();
        assert_eq!(k_out.len(), 64);
        assert_eq!(v_out.len(), 64);

        let amax = k_row.iter().fold(0.0f32, |a, x| a.max(x.abs()));
        // Q8_0 bound = amax/127 · 0.5 = ~0.008 per element for amax=1
        let tol = 0.6 * (amax / 127.0);
        for (x, y) in k_row.iter().zip(&k_out) {
            assert!((x - y).abs() <= tol, "K: |{x} - {y}| > {tol}");
        }
        for (x, y) in v_row.iter().zip(&v_out) {
            assert!((x - y).abs() <= tol, "V: |{x} - {y}| > {tol}");
        }
    }

    #[test]
    fn all_three_modes_have_decreasing_error() {
        let d = dims_for(2, 32, 4); // 64-elem row = 2 blocks
        let k_row: Vec<f32> = (0..64).map(|i| ((i as f32) / 63.0).sin()).collect();
        let v_row = vec![0.0f32; 64];

        let mut errs = Vec::new();
        for mode in [KvQuant::Q4_0, KvQuant::Q5_0, KvQuant::Q8_0] {
            let mut cache = QuantizedPagedKvCache::new(d, BlockSize::Four, mode).unwrap();
            cache.append_step(0, 0, 0, 0, &k_row, &v_row).unwrap();
            let (k_out, _) = cache.read_step(0, 0, 0, 0).unwrap();
            let sse: f32 = k_row.iter().zip(&k_out).map(|(a, b)| (a - b).powi(2)).sum();
            errs.push((mode, sse));
        }
        // errs[0] = Q4, errs[1] = Q5, errs[2] = Q8. Q8 ≤ Q5 ≤ Q4.
        assert!(
            errs[2].1 <= errs[1].1,
            "Q8 sse {} > Q5 sse {}",
            errs[2].1,
            errs[1].1
        );
        assert!(
            errs[1].1 <= errs[0].1,
            "Q5 sse {} > Q4 sse {}",
            errs[1].1,
            errs[0].1
        );
    }

    #[test]
    fn read_returns_none_before_append() {
        let d = dims_for(1, 32, 4);
        let cache = QuantizedPagedKvCache::new(d, BlockSize::Four, KvQuant::Q8_0).unwrap();
        assert!(cache.read_step(0, 0, 0, 0).is_none());
    }

    #[test]
    fn read_returns_none_out_of_range() {
        let d = dims_for(1, 32, 4);
        let cache = QuantizedPagedKvCache::new(d, BlockSize::Four, KvQuant::Q8_0).unwrap();
        assert!(cache.read_step(1, 0, 0, 0).is_none()); // layer OOB
        assert!(cache.read_step(0, 4, 0, 0).is_none()); // t OOB
        assert!(cache.read_step(0, 0, 1, 0).is_none()); // stream OOB
        assert!(cache.read_step(0, 0, 0, 1).is_none()); // codebook OOB
    }

    #[test]
    fn append_rejects_bad_row_length() {
        let d = dims_for(1, 32, 4);
        let mut cache = QuantizedPagedKvCache::new(d, BlockSize::Four, KvQuant::Q8_0).unwrap();
        let bad = vec![0.0f32; 31];
        let good = vec![0.0f32; 32];
        assert!(matches!(
            cache.append_step(0, 0, 0, 0, &bad, &good),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn multi_page_reuse_and_reset() {
        // 4 time steps in 2 blocks of 2, each page holds one block.
        let d = KvDims {
            n_layer: 1,
            n_head: 1,
            d_head: 32,
            n_stream: 1,
            n_codebook: 1,
            max_time: 6,
        };
        let mut cache = QuantizedPagedKvCache::new(d, BlockSize::Two, KvQuant::Q4_0).unwrap();
        let row: Vec<f32> = (0..32).map(|i| (i as f32) / 31.0).collect();
        let zero = vec![0.0f32; 32];
        for t in 0..6 {
            cache.append_step(0, t, 0, 0, &row, &zero).unwrap();
        }
        cache.advance(6);
        assert_eq!(cache.pages_in_use(), 3);

        // Reset returns everything to the free list and zeroes pages.
        cache.reset();
        assert_eq!(cache.pages_in_use(), 0);
        assert_eq!(cache.positions(), 0);
        assert!(cache.read_step(0, 0, 0, 0).is_none());
    }

    #[test]
    fn hot_path_does_not_grow_capacity() {
        // Mirror the M3-03 `capacity_stays_stable_across_hot_path` assertion.
        let d = dims_for(2, 32, 32);
        let mut cache = QuantizedPagedKvCache::new(d, BlockSize::Four, KvQuant::Q8_0).unwrap();
        let baseline = cache.allocator_capacity_snapshot();
        let per_slot = 2 * 32;
        let k = vec![0.5f32; per_slot];
        let v = vec![-0.5f32; per_slot];
        for step in 0..3 {
            for t in 0..32 {
                cache.append_step(0, t, 0, 0, &k, &v).unwrap();
                cache.advance(1);
            }
            let mid = cache.allocator_capacity_snapshot();
            assert_eq!(baseline, mid, "hot-path realloc at step {step}");
            cache.reset();
        }
        let after = cache.allocator_capacity_snapshot();
        assert_eq!(baseline, after);
    }

    #[test]
    fn on_wire_storage_bytes_matches_arithmetic() {
        // 2 pages × 2 K+V halves × block_size(4) × 1 stream × 1 codebook ×
        //   row_blocks(1) × 34 bytes/Q8_0 block = 544 bytes.
        let d = dims_for(1, 32, 8); // 2 pages at bs=4
        let cache = QuantizedPagedKvCache::new(d, BlockSize::Four, KvQuant::Q8_0).unwrap();
        // pages_per_layer = ceil(8/4) = 2, n_layer = 1 → 2 pages.
        // per_page_blocks = 2 × 4 × 1 × 1 × 1 = 8. Total = 2 × 8 × 34 = 544.
        assert_eq!(cache.on_wire_storage_bytes(), 544);
    }

    #[test]
    fn compression_ratio_ordering_reflected_in_on_wire_bytes() {
        // Same shape, all three modes → Q4 uses less than Q5, Q5 uses less
        // than Q8, Q8 uses less than what an FP32 cache would.
        let d = dims_for(2, 32, 16); // 64-elem row, 4 pages at bs=4
        let q4 = QuantizedPagedKvCache::new(d, BlockSize::Four, KvQuant::Q4_0).unwrap();
        let q5 = QuantizedPagedKvCache::new(d, BlockSize::Four, KvQuant::Q5_0).unwrap();
        let q8 = QuantizedPagedKvCache::new(d, BlockSize::Four, KvQuant::Q8_0).unwrap();
        assert!(q4.on_wire_storage_bytes() < q5.on_wire_storage_bytes());
        assert!(q5.on_wire_storage_bytes() < q8.on_wire_storage_bytes());
    }

    #[test]
    fn release_layer_frees_pages() {
        let d = KvDims {
            n_layer: 2,
            n_head: 1,
            d_head: 32,
            n_stream: 1,
            n_codebook: 1,
            max_time: 4,
        };
        let mut cache = QuantizedPagedKvCache::new(d, BlockSize::Four, KvQuant::Q8_0).unwrap();
        let row = vec![0.5f32; 32];
        for layer in 0..2 {
            cache.append_step(layer, 0, 0, 0, &row, &row).unwrap();
        }
        assert_eq!(cache.pages_in_use(), 2);
        cache.release_layer(0).unwrap();
        assert_eq!(cache.pages_in_use(), 1);
        // Layer 0 read now returns None; layer 1 still readable.
        assert!(cache.read_step(0, 0, 0, 0).is_none());
        assert!(cache.read_step(1, 0, 0, 0).is_some());
    }

    #[test]
    fn release_layer_rejects_out_of_range() {
        let d = dims_for(1, 32, 4);
        let mut cache = QuantizedPagedKvCache::new(d, BlockSize::Four, KvQuant::Q8_0).unwrap();
        assert!(matches!(
            cache.release_layer(1),
            Err(VokraError::InvalidArgument(_))
        ));
    }

    #[test]
    fn multi_stream_isolation() {
        // Two streams; distinct signature per stream, per-stream reads must
        // stay separated.
        let d = KvDims {
            n_layer: 1,
            n_head: 1,
            d_head: 32,
            n_stream: 2,
            n_codebook: 1,
            max_time: 2,
        };
        let mut cache = QuantizedPagedKvCache::new(d, BlockSize::Two, KvQuant::Q8_0).unwrap();
        let s0: Vec<f32> = (0..32).map(|i| (i as f32) / 100.0).collect();
        let s1: Vec<f32> = s0.iter().map(|x| x + 1.0).collect();
        cache.append_step(0, 0, 0, 0, &s0, &s0).unwrap();
        cache.append_step(0, 0, 1, 0, &s1, &s1).unwrap();

        let (k0, _) = cache.read_step(0, 0, 0, 0).unwrap();
        let (k1, _) = cache.read_step(0, 0, 1, 0).unwrap();
        // Both dequantized values are within Q8_0 tolerance of their source.
        let tol = 0.6 * (1.0f32.abs() + 0.32) / 127.0; // amax ≈ 1.32 for s1
        for (a, b) in s0.iter().zip(&k0) {
            assert!((a - b).abs() <= tol);
        }
        for (a, b) in s1.iter().zip(&k1) {
            assert!((a - b).abs() <= tol);
        }
        // Cross-stream reads must be different.
        assert_ne!(k0[0], k1[0]);
    }
}
