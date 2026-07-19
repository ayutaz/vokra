//! Multi-session runtime primitives for `vokra-server` (M3-15).
//!
//! Provides:
//! - [`SessionId`] — collision-free session identifiers issued by an
//!   [`AtomicU64`](std::sync::atomic::AtomicU64) counter. IDs are unique for
//!   the process lifetime; wrap-around is unreachable in practice (`u64` is
//!   ~584 y at 1 GHz issuance).
//! - [`ServerSession`] — the per-request state a scheduler hands out (session
//!   id, paged KV cache stream slot, wall-clock accessors). Cheap to move —
//!   the heavy `PagedKvCache` lives once behind an `Arc<Mutex<_>>` in the
//!   [`SessionRegistry`], and a session only carries its stream index.
//! - [`SessionRegistry`] — the multi-session KV cache mapping layer (T04).
//!   Owns one `PagedKvCache` sized for `n_stream` concurrent streams and
//!   hands out non-overlapping stream slots (0..n_stream) to sessions.
//!   Release is O(1) and does not touch the system allocator on the hot path
//!   (FR-EX-05).
//!
//! # Invariants (M3-15, T01 ADR §5-point semantics)
//!
//! - (a) Concurrent requests are isolated by **stream index** on the shared
//!   [`PagedKvCache`] (`[time, stream, codebook]` 3D address, M3-03).
//!   `SessionRegistry` never hands the same stream slot to two live
//!   sessions.
//! - (b) A single GPU / CPU resource is shared implicitly: sessions carry
//!   only the stream index; the actual per-step compute happens in the
//!   existing per-model engines (Whisper / piper-plus / Kokoro).
//! - (c) HTTP endpoints get a [`SessionGuard`] via [`Scheduler`] (see
//!   `crate::scheduler`) → the guard's [`Drop`] releases the stream slot
//!   back to the registry (RAII, no explicit close needed).
//! - (d) Overload → explicit error surfaced as
//!   [`crate::error::ServerError::ServiceUnavailable`] mapped to HTTP 503;
//!   never a silent low-quality fallback (FR-EX-08).
//! - (e) The scheduler's max concurrent sessions bound is enforced by a
//!   `Semaphore`; the registry's stream slot pool is bounded by
//!   `n_stream`, and they are kept in lock-step so acquisition can never
//!   dangle.
//!
//! # Zero-dep posture
//!
//! Only `std` + `vokra-core` are used here. No new third-party crate is
//! added by this module; the excluded-workspace HTTP stack (tokio / axum /
//! serde) stays untouched.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Instant;

use vokra_core::cache::paged::{BlockSize, KvDims, PagedKvCache};
use vokra_core::{Result as VokraResult, VokraError};

/// Newtype wrapper over a `u64` session id, issued by an atomic counter.
///
/// `SessionId` is `Copy` on purpose — passing it across tokio task
/// boundaries or into logging context is cheap. The counter never wraps in
/// practice: at 1 GHz issuance a `u64` lasts ~584 years, and the registry
/// asserts monotonicity (see the `session_ids_are_monotonic_and_unique`
/// test).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SessionId(pub u64);

impl std::fmt::Display for SessionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "session#{}", self.0)
    }
}

/// Stream slot inside the shared [`PagedKvCache`] (0..n_stream). A newtype
/// so the registry cannot confuse it with a session id or a raw usize.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StreamSlot(pub usize);

/// Per-session state handed out by the [`SessionRegistry`]. Holds the
/// session id, the paged KV cache stream slot, and creation timestamps.
///
/// `ServerSession` itself is `Send + Sync + Copy`-able-in-spirit (id +
/// stream slot are `Copy`, timestamps are `Copy`); the RAII cleanup lives
/// in the sibling [`SessionGuard`] type so `ServerSession` can be freely
/// cloned into log context without disturbing the registry accounting.
#[derive(Debug, Clone, Copy)]
pub struct ServerSession {
    /// Unique identifier for the process lifetime.
    pub id: SessionId,
    /// Stream slot allocated in the shared [`PagedKvCache`]. Two live
    /// sessions never share a stream slot.
    pub stream: StreamSlot,
    /// When the session was allocated.
    pub created_at: Instant,
    /// Last time the session touched an inference path. Updated by the
    /// scheduler layer, not this module. Kept `Copy` (Instant is Copy) so
    /// snapshots can be published without holding the registry lock.
    pub last_active: Instant,
}

impl ServerSession {
    /// Elapsed wall time since this session was created.
    pub fn age(&self) -> std::time::Duration {
        self.created_at.elapsed()
    }
}

// ---------------------------------------------------------------------------
// SessionRegistryConfig — construction knobs
// ---------------------------------------------------------------------------

/// Static configuration for [`SessionRegistry`].
///
/// A single [`SessionRegistry`] is sized once at server startup — every
/// axis (`n_layer`, `n_head`, `d_head`, `n_stream`, `max_time`) is fixed
/// for the process lifetime so the paged arena is pre-allocated exactly
/// (FR-EX-05, M3-03 §D3). At runtime only stream slots are handed out;
/// the arena itself never resizes.
#[derive(Debug, Clone, Copy)]
pub struct SessionRegistryConfig {
    /// Transformer decoder layer count. Zero is rejected (see
    /// [`SessionRegistry::new`]).
    pub n_layer: usize,
    /// Attention head count. Zero is rejected.
    pub n_head: usize,
    /// Per-head channel width. Zero is rejected.
    pub d_head: usize,
    /// Number of concurrent stream slots (== `max_concurrent_sessions`
    /// bound on the paged cache side; the scheduler adds its own bound
    /// on top for connection queueing).
    pub n_stream: usize,
    /// Hard upper bound on the number of time steps per session.
    pub max_time: usize,
    /// Block size for the paged cache. Whisper large-v3 / piper-plus /
    /// Kokoro use [`BlockSize::Four`] (25–50 Hz decoders); Mimi (M3-06)
    /// uses [`BlockSize::Two`] (12.5 Hz).
    pub block_size: BlockSize,
}

impl SessionRegistryConfig {
    /// Minimum-viable config for tests and the T03 default startup path:
    /// a single-layer, 1-head, 4-slot, 32-step cache with block_size=4.
    /// The dimensions match the smallest shape the paged cache accepts
    /// (§D3: every axis > 0).
    pub fn minimum(n_stream: usize) -> Self {
        Self {
            n_layer: 1,
            n_head: 1,
            d_head: 1,
            n_stream,
            max_time: 32,
            block_size: BlockSize::Four,
        }
    }
}

// ---------------------------------------------------------------------------
// SessionRegistry — the paged-KV-cache-per-session mapping (T04)
// ---------------------------------------------------------------------------

/// Internal state held under a `Mutex` inside the registry. The cache and
/// the free list share the same lock because releasing a stream slot must
/// zero its KV rows before returning it — two-lock designs would risk a
/// data race on the "just released, not yet zeroed" window.
struct Inner {
    cache: PagedKvCache<f32>,
    /// Stream slots available for allocation, LIFO. Sized to `n_stream` at
    /// construction; hot path is O(1) pop/push (FR-EX-05).
    free_streams: Vec<StreamSlot>,
    /// Live sessions and the stream slot they currently hold. Sized to
    /// `n_stream` at construction and never grown afterwards; a "session
    /// is live" query is a linear scan bounded by `n_stream` (small — the
    /// M3-15 spec calls out `n_stream = 10` for load tests, so O(n) here
    /// is O(constant) in practice).
    live: Vec<(SessionId, StreamSlot)>,
    /// Whether the registry has entered its shutdown state. Once true,
    /// `try_acquire` fails with [`RegistryError::ShuttingDown`] — the
    /// scheduler layer uses this to drain sessions on graceful shutdown.
    shutting_down: bool,
}

/// The multi-session KV cache mapping layer (T04).
///
/// Owns a single [`PagedKvCache`] sized for `n_stream` concurrent streams
/// and dispenses [`SessionGuard`]s that map each live session to a unique
/// stream slot. Concurrent [`Self::try_acquire`] calls are serialised by
/// an internal `Mutex` so two callers cannot receive the same slot.
///
/// # Thread safety
///
/// `SessionRegistry` is `Send + Sync` (all state is behind `Arc<Mutex<_>>`).
/// The scheduler wraps it in an `Arc<SessionRegistry>` and shares it across
/// tokio worker threads. The lock's critical section is O(1) — a stream
/// slot pop / push and, on release, a bounded per-layer zero.
///
/// # Hot-path allocation
///
/// After construction the registry never allocates: `free_streams` and
/// `live` are pre-sized to `n_stream` with `Vec::with_capacity`, and
/// `PagedKvCache::release_layer` uses the underlying free list without
/// invoking the system allocator (M3-03 §D3).
pub struct SessionRegistry {
    inner: Mutex<Inner>,
    /// Config the registry was built with. Read-only after `new`, so no
    /// lock is needed to consult it.
    config: SessionRegistryConfig,
    /// Session id generator. Monotonic increments under `Ordering::Relaxed`
    /// are fine because uniqueness is guaranteed by
    /// `AtomicU64::fetch_add` — ordering only matters for the debug guard
    /// in `session_ids_are_monotonic_and_unique`.
    next_id: AtomicU64,
}

/// Errors surfaced by [`SessionRegistry::try_acquire`].
///
/// Kept distinct from [`crate::error::ServerError`] so the scheduler can
/// map registry errors to `ServiceUnavailable` while unit tests can match
/// on the registry-side variant without dragging in axum types.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegistryError {
    /// All `n_stream` slots are currently allocated. The caller must
    /// backpressure (queue) or return HTTP 503 (FR-EX-08 — no silent
    /// fallback).
    NoFreeStream {
        /// Number of stream slots the registry was constructed with.
        capacity: usize,
    },
    /// Configuration is inconsistent (a zero axis, incompatible dims).
    InvalidConfig(String),
    /// The registry has been marked shutting-down; new acquires are
    /// refused so in-flight sessions can drain.
    ShuttingDown,
}

impl std::fmt::Display for RegistryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoFreeStream { capacity } => {
                write!(
                    f,
                    "session registry: all {capacity} stream slots are in use"
                )
            }
            Self::InvalidConfig(msg) => write!(f, "session registry: invalid config: {msg}"),
            Self::ShuttingDown => f.write_str("session registry: shutting down"),
        }
    }
}

impl std::error::Error for RegistryError {}

impl SessionRegistry {
    /// Constructs a registry sized for `config.n_stream` concurrent sessions.
    ///
    /// The underlying [`PagedKvCache`] is pre-allocated exactly for
    /// `n_layer × ceil(max_time / block_size) × n_stream` pages — no
    /// system allocation happens on the hot path after this call
    /// (FR-EX-05, M3-03 §D3).
    ///
    /// # Errors
    ///
    /// Returns [`RegistryError::InvalidConfig`] if any axis is zero, or
    /// if `PagedKvCache::pre_allocate` rejects the shape.
    pub fn new(config: SessionRegistryConfig) -> std::result::Result<Arc<Self>, RegistryError> {
        if config.n_stream == 0 {
            return Err(RegistryError::InvalidConfig(
                "n_stream must be > 0 for a multi-session server".into(),
            ));
        }
        let dims = KvDims {
            n_layer: config.n_layer,
            n_head: config.n_head,
            d_head: config.d_head,
            n_stream: config.n_stream,
            n_codebook: 1,
            max_time: config.max_time,
        };
        let cache = PagedKvCache::<f32>::pre_allocate(dims, config.block_size)
            .map_err(|e| RegistryError::InvalidConfig(format!("paged cache rejected: {e}")))?;
        let mut free_streams = Vec::with_capacity(config.n_stream);
        // LIFO order: index 0 popped first (last pushed = first popped)
        // improves cache locality on repeat acquires (see
        // `PageAllocator::new` §M3-03).
        for i in (0..config.n_stream).rev() {
            free_streams.push(StreamSlot(i));
        }
        free_streams.reverse();
        let live = Vec::with_capacity(config.n_stream);
        Ok(Arc::new(Self {
            inner: Mutex::new(Inner {
                cache,
                free_streams,
                live,
                shutting_down: false,
            }),
            config,
            next_id: AtomicU64::new(1),
        }))
    }

    /// Allocates a fresh session and stream slot, or returns
    /// [`RegistryError::NoFreeStream`] if the registry is saturated.
    ///
    /// This is the hot path — the returned [`SessionGuard`] releases the
    /// stream slot on `Drop`, so acquire/release pairs form a matched
    /// stack and never touch the system allocator (FR-EX-05).
    ///
    /// # Errors
    ///
    /// - [`RegistryError::NoFreeStream`] — no stream slot is available.
    /// - [`RegistryError::ShuttingDown`] — the registry is draining and
    ///   refusing new sessions.
    pub fn try_acquire(self: &Arc<Self>) -> std::result::Result<SessionGuard, RegistryError> {
        let mut inner = self.lock();
        if inner.shutting_down {
            return Err(RegistryError::ShuttingDown);
        }
        let slot = match inner.free_streams.pop() {
            Some(s) => s,
            None => {
                return Err(RegistryError::NoFreeStream {
                    capacity: self.config.n_stream,
                });
            }
        };
        // Monotonic id issuance. `Ordering::Relaxed` is safe because
        // uniqueness is a property of `fetch_add`; we do not synchronise
        // any other memory with this counter.
        let id = SessionId(self.next_id.fetch_add(1, Ordering::Relaxed));
        let now = Instant::now();
        let session = ServerSession {
            id,
            stream: slot,
            created_at: now,
            last_active: now,
        };
        inner.live.push((id, slot));
        Ok(SessionGuard {
            session,
            registry: Arc::clone(self),
        })
    }

    /// Marks the registry as draining. New [`Self::try_acquire`] calls
    /// return [`RegistryError::ShuttingDown`]; live sessions are unaffected
    /// and can complete normally.
    pub fn begin_shutdown(&self) {
        self.lock().shutting_down = true;
    }

    /// Number of stream slots currently allocated.
    ///
    /// # Panics
    ///
    /// Panics only if the internal `Mutex` was poisoned by a prior panic
    /// while holding it — see [`Self::lock`].
    pub fn in_use(&self) -> usize {
        let inner = self.lock();
        self.config.n_stream - inner.free_streams.len()
    }

    /// Total stream slot capacity — the max number of concurrent sessions
    /// this registry can host. Fixed at construction.
    pub fn capacity(&self) -> usize {
        self.config.n_stream
    }

    /// Snapshot of live session ids at call time. Order is unspecified
    /// (matches the registry's internal LIFO free-list); repeated calls
    /// may return the same session in different positions.
    ///
    /// # Panics
    ///
    /// Panics only if the internal `Mutex` was poisoned.
    pub fn live_ids(&self) -> Vec<SessionId> {
        let inner = self.lock();
        inner.live.iter().map(|(id, _)| *id).collect()
    }

    /// Config the registry was built with. Read-only.
    pub fn config(&self) -> &SessionRegistryConfig {
        &self.config
    }

    /// KV-cache-side operations for hot inference paths. Callers hand in
    /// the closure `f`, which sees a `&mut PagedKvCache<f32>` under the
    /// registry lock. This funnels every append/read through the same
    /// mutex the acquire/release path uses so a slot cannot be released
    /// mid-append.
    ///
    /// # Panics
    ///
    /// Panics only if the internal `Mutex` was poisoned by a prior panic
    /// while holding it.
    pub fn with_cache<R>(&self, f: impl FnOnce(&mut PagedKvCache<f32>) -> R) -> R {
        let mut inner = self.lock();
        f(&mut inner.cache)
    }

    /// Read-only variant of [`Self::with_cache`] for pure reads.
    ///
    /// # Panics
    ///
    /// Panics only if the internal `Mutex` was poisoned.
    pub fn with_cache_ref<R>(&self, f: impl FnOnce(&PagedKvCache<f32>) -> R) -> R {
        let inner = self.lock();
        f(&inner.cache)
    }

    /// Release the stream slot back to the pool. Called from
    /// [`SessionGuard::drop`] — direct calls exist only for tests where
    /// a session must be forcibly reclaimed without dropping the guard.
    ///
    /// Retires every layer's KV state for `slot` before returning it to
    /// the free list, so a subsequent acquirer cannot observe stale KV
    /// rows even if it reads before appending.
    fn release_internal(&self, id: SessionId, slot: StreamSlot) {
        let mut inner = self.lock();
        // cc-37: `PagedKvCache::release_stream` retires the slot in O(1)
        // by bumping its generation — every row the departing session
        // stamped then reads as unbound. This replaces the M3-15 T04
        // stopgap, which zeroed the slot by looping `append_step` with
        // zero rows over every (layer, committed t): O(n_layer *
        // committed * n_head * d_head) element writes on every release,
        // which does not survive contact with large-v3-sized dims.
        //
        // The only error is an out-of-range stream, which cannot happen
        // here — `slot` came from this registry's own free list, sized to
        // the same `n_stream` the cache was built with. Surfaced as a
        // panic rather than swallowed so a future refactor that breaks
        // that invariant is loud (NFR-RL-07 keeps the runtime up via the
        // HTTP-side CatchPanicLayer).
        inner
            .cache
            .release_stream(slot.0)
            .expect("stream slot is always in range for this registry's cache");
        // Drop from `live` (O(n_stream), n_stream is bounded and small).
        if let Some(pos) = inner.live.iter().position(|(sid, _)| *sid == id) {
            inner.live.swap_remove(pos);
        }
        // Return to the free list. `push` on a Vec sized to `n_stream`
        // never reallocates (see `Self::new`).
        inner.free_streams.push(slot);
    }

    /// Locks the internal `Mutex`, panicking on poisoning. A poisoned
    /// registry is unrecoverable — the paged cache may hold pages in a
    /// half-written state — so we surface the poison as a panic and
    /// rely on the HTTP-side `CatchPanicLayer` (NFR-RL-07) to keep the
    /// runtime up.
    fn lock(&self) -> MutexGuard<'_, Inner> {
        self.inner.lock().expect("session registry mutex poisoned")
    }
}

/// RAII handle to a live session's stream slot.
///
/// On `Drop`, releases the stream slot back to the [`SessionRegistry`]. A
/// caller that panics with an unfinished session gets its slot returned
/// automatically — the pool cannot leak beyond a single `unwind`.
pub struct SessionGuard {
    session: ServerSession,
    registry: Arc<SessionRegistry>,
}

impl SessionGuard {
    /// Snapshot of the wrapped [`ServerSession`] (Copy, safe to log).
    pub fn session(&self) -> ServerSession {
        self.session
    }

    /// The session id (convenience — same as `self.session().id`).
    pub fn id(&self) -> SessionId {
        self.session.id
    }

    /// The stream slot (convenience — same as `self.session().stream`).
    pub fn stream(&self) -> StreamSlot {
        self.session.stream
    }

    /// The registry this guard is bound to. Callers use it to reach
    /// [`SessionRegistry::with_cache`] for per-step KV work under the
    /// same lock the acquire/release path uses.
    pub fn registry(&self) -> &Arc<SessionRegistry> {
        &self.registry
    }

    /// Update the `last_active` field. Purely observational — the
    /// registry does not act on it. Kept behind `&mut self` so callers
    /// serialise their own writes; the registry lock is not taken here
    /// (the guard owns its own `ServerSession` snapshot).
    pub fn touch(&mut self) {
        self.session.last_active = Instant::now();
    }
}

impl Drop for SessionGuard {
    fn drop(&mut self) {
        self.registry
            .release_internal(self.session.id, self.session.stream);
    }
}

// ---------------------------------------------------------------------------
// Bridge to server-side `Result` — kept minimal so the scheduler / handlers
// only need to import [`SessionRegistry`] + [`RegistryError`].
// ---------------------------------------------------------------------------

impl RegistryError {
    /// Map to the wire-facing server error, so scheduler/HTTP handlers
    /// don't have to hand-roll the mapping per call site.
    pub fn to_server_error(&self) -> crate::error::ServerError {
        match self {
            Self::NoFreeStream { capacity } => crate::error::ServerError::ServiceUnavailable {
                detail: format!(
                    "session registry saturated: {capacity} concurrent sessions in flight"
                ),
            },
            Self::ShuttingDown => crate::error::ServerError::ServiceUnavailable {
                detail: "server shutting down".to_string(),
            },
            Self::InvalidConfig(msg) => crate::error::ServerError::InferenceFailed {
                detail: format!("session registry misconfigured: {msg}"),
            },
        }
    }
}

impl From<VokraError> for RegistryError {
    fn from(err: VokraError) -> Self {
        Self::InvalidConfig(format!("paged cache error: {err}"))
    }
}

/// Convenience shim to lift a `Result<T, RegistryError>` into
/// [`vokra_core::Result`] where a caller needs a uniform error type.
/// Not part of the M3-15 spec but callable from the scheduler.
pub fn to_vokra_result<T>(r: std::result::Result<T, RegistryError>) -> VokraResult<T> {
    r.map_err(|e| match e {
        RegistryError::NoFreeStream { capacity } => VokraError::InvalidArgument(format!(
            "session registry exhausted at capacity {capacity}"
        )),
        RegistryError::ShuttingDown => {
            VokraError::BackendUnavailable("server shutting down".into())
        }
        RegistryError::InvalidConfig(msg) => VokraError::InvalidArgument(msg),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn mk_registry(n_stream: usize) -> Arc<SessionRegistry> {
        SessionRegistry::new(SessionRegistryConfig::minimum(n_stream)).expect("registry")
    }

    #[test]
    fn registry_rejects_zero_streams() {
        let cfg = SessionRegistryConfig::minimum(0);
        match SessionRegistry::new(cfg) {
            Err(RegistryError::InvalidConfig(msg)) => {
                assert!(msg.contains("n_stream"), "unexpected msg: {msg}");
            }
            Err(other) => panic!("expected InvalidConfig, got {other:?}"),
            Ok(_) => panic!("zero n_stream must fail"),
        }
    }

    #[test]
    fn session_ids_are_monotonic_and_unique() {
        // Sequential acquires under one thread — ids must strictly
        // increase and never repeat. Guards are dropped at end of scope,
        // returning slots; the id counter does not reset.
        let reg = mk_registry(4);
        let mut ids = Vec::new();
        let g1 = reg.try_acquire().unwrap();
        ids.push(g1.id());
        drop(g1);
        let g2 = reg.try_acquire().unwrap();
        ids.push(g2.id());
        drop(g2);
        let g3 = reg.try_acquire().unwrap();
        ids.push(g3.id());
        // Monotonic
        assert!(ids[0].0 < ids[1].0);
        assert!(ids[1].0 < ids[2].0);
        // Unique
        let set: HashSet<_> = ids.iter().copied().collect();
        assert_eq!(set.len(), ids.len());
    }

    #[test]
    fn capacity_bound_returns_no_free_stream() {
        let reg = mk_registry(2);
        let _g1 = reg.try_acquire().unwrap();
        let _g2 = reg.try_acquire().unwrap();
        assert_eq!(reg.in_use(), 2);
        // `SessionGuard` does not implement `Debug` (it wraps a paged
        // cache handle) so we match on `Result` variants explicitly
        // instead of `unwrap_err()` + `.expect_err`.
        match reg.try_acquire() {
            Err(RegistryError::NoFreeStream { capacity }) => assert_eq!(capacity, 2),
            Err(other) => panic!("expected NoFreeStream, got {other:?}"),
            Ok(_) => panic!("registry must refuse a 3rd session at capacity 2"),
        }
    }

    #[test]
    fn drop_returns_stream_slot() {
        let reg = mk_registry(1);
        {
            let _g = reg.try_acquire().unwrap();
            assert_eq!(reg.in_use(), 1);
        }
        assert_eq!(reg.in_use(), 0);
        // Reacquire must succeed.
        let g2 = reg.try_acquire().unwrap();
        assert_eq!(reg.in_use(), 1);
        drop(g2);
    }

    #[test]
    fn stream_slots_never_alias_two_live_sessions() {
        let reg = mk_registry(3);
        let g1 = reg.try_acquire().unwrap();
        let g2 = reg.try_acquire().unwrap();
        let g3 = reg.try_acquire().unwrap();
        let mut slots = [g1.stream(), g2.stream(), g3.stream()];
        slots.sort_by_key(|s| s.0);
        assert_eq!(slots, [StreamSlot(0), StreamSlot(1), StreamSlot(2)]);
    }

    #[test]
    fn state_release_retires_stream_kv_rows() {
        // T10 (state leak) primitive check: a session writes to its
        // stream, its guard drops, the next session reuses the same
        // stream slot and must NOT observe the previous session's data.
        //
        // Since cc-37 the release is O(1) (a generation bump) rather
        // than a zero-fill, so a retired row reads as `None` instead of
        // `Some(zeros)`. Both satisfy the isolation property; the
        // assertion below is written against the property itself — "the
        // old values are not observable" — so it holds for either
        // mechanism and, critically, cannot pass vacuously: an
        // implementation that skipped invalidation entirely would return
        // `Some([7.0, 8.0])` and fail here.
        let cfg = SessionRegistryConfig {
            n_layer: 1,
            n_head: 1,
            d_head: 2,
            n_stream: 1,
            max_time: 4,
            block_size: BlockSize::Two,
        };
        let reg = SessionRegistry::new(cfg).unwrap();
        {
            let g = reg.try_acquire().unwrap();
            reg.with_cache(|c| {
                c.append_step(0, 0, g.stream().0, 0, &[7.0, 8.0], &[9.0, 10.0])
                    .unwrap();
                c.advance(1);
            });
            // sanity: the row is present under this guard
            reg.with_cache_ref(|c| {
                let (k, v) = c.read_step(0, 0, g.stream().0, 0).unwrap();
                assert_eq!(k, &[7.0, 8.0]);
                assert_eq!(v, &[9.0, 10.0]);
            });
            // guard dropped here → release retires the row
        }
        // Reacquire — same slot, must not see the previous occupant.
        let g2 = reg.try_acquire().unwrap();
        assert_eq!(g2.stream(), StreamSlot(0));
        reg.with_cache_ref(|c| {
            match c.read_step(0, 0, g2.stream().0, 0) {
                // Retired: unreachable through the public read path.
                None => {}
                // Or (pre-cc-37 mechanism) present but zeroed.
                Some((k, v)) => {
                    assert!(k.iter().all(|x| *x == 0.0), "k leaked: {k:?}");
                    assert!(v.iter().all(|x| *x == 0.0), "v leaked: {v:?}");
                }
            }
        });
    }

    /// The same isolation property across the *whole* committed span, with a
    /// partial rewrite by the new occupant — the case a coarser (per-page)
    /// invalidation would miss, since `block_size = 2` puts t=0 and t=1 in one
    /// page. Writing only t=0 must not resurrect the previous session's t=1.
    #[test]
    fn state_release_survives_partial_rewrite_by_next_session() {
        let cfg = SessionRegistryConfig {
            n_layer: 1,
            n_head: 1,
            d_head: 2,
            n_stream: 1,
            max_time: 4,
            block_size: BlockSize::Two,
        };
        let reg = SessionRegistry::new(cfg).unwrap();
        {
            let g = reg.try_acquire().unwrap();
            reg.with_cache(|c| {
                for t in 0..4 {
                    let x = 50.0 + t as f32;
                    c.append_step(0, t, g.stream().0, 0, &[x, x], &[x, x])
                        .unwrap();
                }
                c.advance(4);
            });
        }
        let g2 = reg.try_acquire().unwrap();
        // New occupant writes ONLY t = 0, in the page that also holds t = 1.
        reg.with_cache(|c| {
            c.append_step(0, 0, g2.stream().0, 0, &[1.0, 1.0], &[2.0, 2.0])
                .unwrap();
        });
        reg.with_cache_ref(|c| {
            let (k, v) = c.read_step(0, 0, g2.stream().0, 0).expect("own row");
            assert_eq!(k, &[1.0, 1.0]);
            assert_eq!(v, &[2.0, 2.0]);
            for t in 1..4 {
                let stale = 50.0 + t as f32;
                if let Some((k, v)) = c.read_step(0, t, g2.stream().0, 0) {
                    assert!(
                        !k.contains(&stale) && !v.contains(&stale),
                        "t {t}: previous session's row {stale} leaked through a partial rewrite"
                    );
                }
            }
        });
    }

    #[test]
    fn concurrent_acquire_never_aliases_slots() {
        // Spawn N threads all trying to acquire; keep guards alive until
        // the barrier releases them. Each guard must hold a unique slot.
        use std::sync::Barrier;
        use std::thread;

        let n = 8;
        let reg = mk_registry(n);
        let barrier = Arc::new(Barrier::new(n));

        let handles: Vec<_> = (0..n)
            .map(|_| {
                let reg = Arc::clone(&reg);
                let bar = Arc::clone(&barrier);
                thread::spawn(move || {
                    // All threads hit try_acquire near-simultaneously.
                    let g = reg.try_acquire().expect("all N acquires must succeed");
                    let s = g.stream();
                    // Wait for every thread to hold its guard before
                    // dropping — proves the slots are simultaneously
                    // live and non-aliasing.
                    bar.wait();
                    s
                })
            })
            .collect();

        let mut slots = Vec::new();
        for h in handles {
            slots.push(h.join().unwrap());
        }
        // Every thread must have observed a distinct slot in 0..n.
        let set: HashSet<usize> = slots.iter().map(|s| s.0).collect();
        assert_eq!(set.len(), n);
        for i in 0..n {
            assert!(set.contains(&i), "slot {i} was never handed out");
        }
    }

    #[test]
    fn begin_shutdown_refuses_new_sessions() {
        let reg = mk_registry(2);
        let _g1 = reg.try_acquire().unwrap();
        reg.begin_shutdown();
        match reg.try_acquire() {
            Err(RegistryError::ShuttingDown) => {}
            Err(other) => panic!("expected ShuttingDown, got {other:?}"),
            Ok(_) => panic!("shutdown must refuse new sessions"),
        }
    }

    #[test]
    fn to_server_error_maps_saturation_to_503() {
        let err = RegistryError::NoFreeStream { capacity: 3 };
        let se = err.to_server_error();
        assert!(matches!(
            se,
            crate::error::ServerError::ServiceUnavailable { .. }
        ));
        assert_eq!(se.status(), axum::http::StatusCode::SERVICE_UNAVAILABLE);
    }

    #[test]
    fn to_server_error_maps_shutdown_to_503() {
        let err = RegistryError::ShuttingDown;
        let se = err.to_server_error();
        assert!(matches!(
            se,
            crate::error::ServerError::ServiceUnavailable { .. }
        ));
    }

    #[test]
    fn registry_capacity_matches_config() {
        let reg = mk_registry(7);
        assert_eq!(reg.capacity(), 7);
        assert_eq!(reg.in_use(), 0);
    }

    #[test]
    fn touch_updates_last_active() {
        let reg = mk_registry(1);
        let mut g = reg.try_acquire().unwrap();
        let before = g.session().last_active;
        // Spin a tiny amount so the monotonic clock definitely advances.
        // 1 ms is enough on every platform we target; keeping it small
        // so the test suite stays fast.
        std::thread::sleep(std::time::Duration::from_millis(1));
        g.touch();
        let after = g.session().last_active;
        assert!(after > before);
    }
}
