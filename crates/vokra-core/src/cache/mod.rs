//! Key/value caches for autoregressive decoders (FR-EX-02, FR-EX-03).
//!
//! Two variants live side by side while the M4 / v1.0 GA semver freeze is still
//! ahead of us (README §4 (14)):
//!
//! - [`KvCache`] — the M1-era, growable per-layer `[positions, width]` cache
//!   promoted out of the Whisper decoder (FR-EX-02). This is the type used by
//!   every existing decoder call site (Whisper, piper-plus, Metal / CUDA
//!   sessions). Its public API is **unchanged** by M3-03; migration to the
//!   paged variant only happens when a call site actually needs multi-stream or
//!   codebook state.
//! - [`PagedKvCache`](paged::PagedKvCache) — the M3-03 paged cache with a
//!   `[time, stream, codebook]` 3D logical address, `block_size ∈ {2, 4}`
//!   sized for audio frame rates (12.5–50 Hz), and a session-lifetime page
//!   arena that keeps the hot path free of system allocations (FR-EX-05).
//!   Intended to become the underlying store for the `vokra-server`
//!   multi-session path (FR-SV-06 = M3-15) and for RVQ codec state (M3-06
//!   Mimi).
//!
//! The rationale for the block_size choice, the deferred `#[global_allocator]`
//! CI gate, and the push-out of the GPU page-table indirection to a co-update
//! WP are recorded in `docs/adr/M3-03-paged-kv-cache.md`.

mod kv;
pub mod paged;
pub mod paged_quant;

pub use kv::KvCache;
pub use paged_quant::{
    AllocatorSnapshot as QuantAllocatorSnapshot, AnyBlock, QuantizedPagedKvCache,
};
