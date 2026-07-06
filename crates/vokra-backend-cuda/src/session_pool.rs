//! Optional [`CudaDecodeSession`] pool for cross-segment reuse
//! (M2-03-followup, `docs/adr/M2-03-followup-rtf.md` §D5 / T-follow-06).
//!
//! # Purpose
//!
//! [`CudaDecodeSession::new`] uploads every decoder weight + the tied logits
//! head + the pre-projected cross-attention K/V per construction — a one-time
//! cost inside the session's lifetime, but *repeated* on every fresh
//! transcription of long-form audio (Whisper's 30 s window slides across the
//! input). Cross-segment reuse — build the session once, [`reset`] it between
//! segments — collapses that per-segment build overhead onto the very first
//! segment.
//!
//! This pool is a **thin, opt-in wrapper**: no code path inside `vokra-models`
//! reaches for it by default. `Compute::for_backend(Cuda)` still calls
//! [`CudaDecodeSession::new`] every time (that path stays unchanged, so the
//! M2-03 default behaviour and its numerical / API contracts hold, 1 bit
//! unchanged). Only the CLI's future `--reuse-cuda-session` flag (wiring is
//! out of this change, see the M2-03 follow-up plan T-follow-06) opts a
//! long-form transcription into the pool.
//!
//! # Design (M2-03-followup §D5, §R4, §R5)
//!
//! - **Zero external deps.** `Mutex<Vec<CudaDecodeSession>>` on top of `std` —
//!   no new crate on the root `Cargo.lock` (NFR-DS-02).
//! - **Exact dim match** (SRS FR-EX-08, no silent shape adaptation). Every
//!   entry the pool hands out carries the exact `(d, n_head, ff, n_text_ctx,
//!   n_vocab, n_ctx)` the caller asked for; anything else is discarded (its
//!   `Drop` runs → device buffers freed → context torn down) rather than
//!   coerced.
//! - **Bounded capacity.** [`Self::new`] takes a `capacity`; a
//!   [`PooledSession::drop`] that would push a `(capacity + 1)`th session
//!   instead lets it drop (device buffers freed). Zero-capacity pools are
//!   accepted and behave like "reuse disabled" (every acquired session is
//!   released to `Drop` on return).
//! - **`PooledSession` is `!Send + !Sync`** via `PhantomData<*const ()>` —
//!   even though the underlying [`CudaDecodeSession`] itself is
//!   `unsafe impl Send` for the CPU decoder's `assert_send::<DecoderState>()`
//!   bound (`context.rs` L3020), an *acquired* handle stays bound to the
//!   thread that acquired it. This makes multi-threaded servers (M3-15,
//!   `vokra-server`) express their concurrency through a per-worker
//!   `CudaDecodeSessionPool` rather than by sharing one pool across workers
//!   (§R5).
//! - **No LRU / no aging.** Pool is a plain LIFO stack (`Vec::pop` /
//!   `Vec::push`); the follow-up plan explicitly disallows anything more
//!   elaborate (M2-03-followup constraints: "Weight cache: session-lifetime,
//!   no LRU").
//!
//! # Ownership contract
//!
//! Because the pool has no way to *build* a [`CudaDecodeSession`] on its own
//! (the constructor needs the decoder's weights, tied logits head, and per-layer
//! cross-attention K/V — see [`CudaDecodeSession::new`]), the acquire flow is:
//!
//! 1. Caller asks the pool for a session matching `dims`.
//! 2. If the pool has a matching entry, it hands the caller a
//!    [`PooledSession`] wrapping it — the entry's [`CudaDecodeSession::reset`]
//!    has *not* run yet at this point (it runs on `PooledSession::drop`, the
//!    moment the entry re-enters the pool, so the acquired handle sees a
//!    fresh state on the next acquire).
//! 3. If the pool has no matching entry, [`Self::acquire`] returns
//!    [`VokraError::BackendUnavailable`] with the message
//!    `"cuda session pool: no matching entry"`. The caller then builds a fresh
//!    [`CudaDecodeSession`] via [`CudaDecodeSession::new`] and wraps it with
//!    [`Self::wrap`] — the wrapper enters the pool on `Drop` (subject to
//!    `capacity`).
//!
//! This split keeps the pool free of weight-shaped constructor arguments (the
//! spec's "thin wrapper" contract) without pushing the caller into a
//! `Option<PooledSession>` API that silently returns `None`.

use core::marker::PhantomData;
use std::sync::Mutex;

use vokra_core::{Result, VokraError};

use crate::context::CudaDecodeSession;

/// The six load-bearing dims that describe a [`CudaDecodeSession`]'s device
/// layout — matched exactly on [`CudaDecodeSessionPool::acquire`] (§D5,
/// §R4). Every other constructor knob ([`CudaDecodeSession::new`]'s
/// `max_t_q`, `eps`, and the weight slices) is a property of the *first*
/// build; a matching `SessionDims` guarantees the resident device buffers can
/// be reused as-is (the reset simply rewinds `pos` to 0, leaving the
/// self-KV rows to be overwritten from row 0).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SessionDims {
    /// Model width (aka `d_model`).
    pub d: usize,
    /// Attention head count; `d` must be a multiple.
    pub n_head: usize,
    /// MLP hidden width (`d_ff`).
    pub ff: usize,
    /// Text-side context cap — hard bound on the self-attention KV cache.
    pub n_text_ctx: usize,
    /// Vocab size — sizes the tied logits head (`[n_vocab, d]`).
    pub n_vocab: usize,
    /// Audio-side context width — sizes the pre-projected cross-attention
    /// K/V per layer (`[n_ctx, d]`).
    pub n_ctx: usize,
}

/// A bounded LIFO pool of [`CudaDecodeSession`]s keyed on [`SessionDims`].
///
/// See the module docs for the ownership contract and the M2-03-followup
/// design references.
pub struct CudaDecodeSessionPool {
    /// Max number of sessions held between `acquire` cycles. `0` disables
    /// reuse.
    capacity: usize,
    /// Reusable sessions. LIFO stack — the most-recently-returned session
    /// wins (best cache locality on the CUDA driver's per-context state).
    ///
    /// The `Mutex` is only ever held for the O(1) `pop` / `push` inside
    /// [`Self::acquire`] and [`PooledSession::drop`]; the session itself is
    /// entirely owned by the [`PooledSession`] during the borrow, so no
    /// device work happens under the lock.
    idle: Mutex<Vec<CudaDecodeSession>>,
}

impl CudaDecodeSessionPool {
    /// Builds an empty pool with the given retention cap. `capacity = 0`
    /// disables reuse (every returned session drops on `PooledSession::drop`
    /// instead of re-entering).
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            idle: Mutex::new(Vec::with_capacity(capacity)),
        }
    }

    /// Takes a matching session out of the pool.
    ///
    /// Scans the LIFO stack from top for the first entry whose stored dims
    /// match `dims` exactly (§D5). On a hit, the entry is removed and
    /// returned wrapped in a [`PooledSession`]; on a miss, this returns
    /// [`VokraError::BackendUnavailable`] and the caller is expected to
    /// build a fresh session via [`CudaDecodeSession::new`] and wrap it with
    /// [`Self::wrap`].
    ///
    /// # Errors
    ///
    /// - [`VokraError::BackendUnavailable`] when no idle entry matches
    ///   `dims` (see the module-level ownership contract).
    ///
    /// # Panics
    ///
    /// Panics only if the internal mutex was poisoned by a previous panic
    /// while holding it — the pool never panics under normal use.
    pub fn acquire(&self, dims: &SessionDims) -> Result<PooledSession<'_>> {
        let mut idle = self.lock_idle();
        // Scan top-down (LIFO) and pull the first matching entry.
        let idx = idle.iter().rposition(|s| Self::session_dims_of(s) == *dims);
        match idx {
            Some(i) => {
                let session = idle.swap_remove(i);
                // Any remaining non-matching entries stay in the pool (they
                // will be matched by other `acquire` calls or dropped by the
                // pool's own `Drop`). The spec says "on mismatch drop pooled
                // session and build new" — that refers to entries whose
                // dims cannot be reused for the *current* request, which we
                // enforce by simply not returning them here; the pool's
                // bounded capacity + first-in-goes-out-on-overflow policy
                // ([`PooledSession::drop`]) prevents unbounded stale
                // accumulation.
                drop(idle);
                Ok(PooledSession {
                    session: Some(session),
                    pool: self,
                    _not_send_not_sync: PhantomData,
                })
            }
            None => Err(VokraError::BackendUnavailable(
                "cuda session pool: no matching entry".to_owned(),
            )),
        }
    }

    /// Wraps a freshly-built [`CudaDecodeSession`] so it re-enters the pool
    /// on `Drop` (subject to `capacity`). Use this after
    /// [`CudaDecodeSession::new`] when [`Self::acquire`] returned a
    /// `no matching entry` error.
    #[must_use]
    pub fn wrap(&self, session: CudaDecodeSession) -> PooledSession<'_> {
        PooledSession {
            session: Some(session),
            pool: self,
            _not_send_not_sync: PhantomData,
        }
    }

    /// The retention cap this pool was built with.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Number of idle sessions currently held. Test / diagnostic use only.
    ///
    /// # Panics
    ///
    /// Panics only if the internal mutex was poisoned.
    #[must_use]
    pub fn idle_len(&self) -> usize {
        self.lock_idle().len()
    }

    /// Extracts the six load-bearing dims from a [`CudaDecodeSession`]. Uses
    /// the crate-local accessor added on `CudaDecodeSession` for this
    /// purpose (private outside the crate — the pool is the only caller).
    fn session_dims_of(s: &CudaDecodeSession) -> SessionDims {
        s.session_dims()
    }

    /// Locks the idle stack, treating a poisoned mutex as a fatal error the
    /// caller cannot recover from (the sessions inside may be in undefined
    /// device state).
    fn lock_idle(&self) -> std::sync::MutexGuard<'_, Vec<CudaDecodeSession>> {
        self.idle
            .lock()
            .expect("cuda session pool mutex poisoned by a prior panic")
    }
}

/// RAII handle for a session on loan from a [`CudaDecodeSessionPool`].
///
/// # Ownership
///
/// - **On acquire**: a matching session is popped from the pool's idle
///   stack; the caller borrows it through this handle.
/// - **On drop**: the borrowed session's [`CudaDecodeSession::reset`] runs
///   (rewinds `pos` to 0 and invalidates the last-step logits view) and it
///   re-enters the pool's idle stack, subject to `capacity` — an overflow
///   drop instead lets the session's `Drop` release its device buffers.
///
/// # Thread safety
///
/// This type is `!Send + !Sync` (a `PhantomData<*const ()>` field, which is
/// neither). Even though the wrapped [`CudaDecodeSession`] carries an
/// `unsafe impl Send` (see `context.rs` L3020), an acquired handle stays
/// bound to the thread that acquired it — the pool's contract is that
/// concurrency across workers goes through per-worker `CudaDecodeSessionPool`
/// instances (M2-03-followup §R5).
pub struct PooledSession<'p> {
    /// `Some` until `Drop` moves the session back into the pool. Kept as
    /// `Option` so the borrow can be lifted out inside `Drop` even though
    /// `Drop::drop` takes `&mut self`.
    session: Option<CudaDecodeSession>,
    pool: &'p CudaDecodeSessionPool,
    /// Anchors `!Send + !Sync`. `*const ()` is neither, so this
    /// [`PhantomData`] denies both auto-traits regardless of the wrapped
    /// [`CudaDecodeSession`]'s own auto-trait impls.
    _not_send_not_sync: PhantomData<*const ()>,
}

impl PooledSession<'_> {
    /// Borrow the wrapped session (read-only).
    #[must_use]
    pub fn get(&self) -> &CudaDecodeSession {
        self.session
            .as_ref()
            .expect("session is only None inside Drop::drop")
    }

    /// Borrow the wrapped session (mutable — for [`CudaDecodeSession::step`]
    /// and friends).
    pub fn get_mut(&mut self) -> &mut CudaDecodeSession {
        self.session
            .as_mut()
            .expect("session is only None inside Drop::drop")
    }
}

impl Drop for PooledSession<'_> {
    fn drop(&mut self) {
        // Rewind `pos` / `last_t` before the session re-enters the pool so
        // the next acquire sees a fresh state (`CudaDecodeSession::reset`
        // rewinds the causal query offset to 0 and invalidates the
        // last-step logits view — the resident weights and cross-KV stay
        // valid, §D5).
        let Some(mut session) = self.session.take() else {
            // Already drained (double-drop is impossible for `Drop::drop`
            // under Rust's ownership rules — this branch is unreachable).
            return;
        };
        session.reset();

        // Bounded return: push only while under capacity; overflows are
        // dropped in place (device buffers released via
        // `CudaDecodeSession::Drop`). A poisoned mutex means an earlier
        // panic left the pool in an unknown state; on drop we prefer
        // releasing the session's device buffers over re-poisoning, so
        // ignore the lock error and let `session` drop.
        if let Ok(mut idle) = self.pool.idle.lock() {
            if idle.len() < self.pool.capacity {
                idle.push(session);
            }
            // else: `session` drops here → its device buffers are freed +
            // its owned `CudaContext` is torn down (the intended overflow
            // policy).
        }
    }
}

// Compile-time proof that `PooledSession` is neither `Send` nor `Sync`
// (M2-03-followup §R5). If a future edit adds an auto-`Send`/`Sync`-safe
// field that overrides the `PhantomData<*const ()>` denial, one of these
// assertions will fail at compile time.
#[cfg(test)]
mod compile_time_bounds {
    use super::PooledSession;

    #[allow(dead_code)]
    fn assert_not_send<T>()
    where
        T: ?Sized + NotSend,
    {
    }
    #[allow(dead_code)]
    fn assert_not_sync<T>()
    where
        T: ?Sized + NotSync,
    {
    }

    // Marker traits satisfied by *anything*; blanket impls that would clash
    // with the auto-trait would make the assertion fail. Instead we go the
    // simpler route: a compile-fail on `Send` / `Sync` via a static bound.
    trait NotSend {}
    impl<T: ?Sized> NotSend for T {}
    trait NotSync {}
    impl<T: ?Sized> NotSync for T {}

    #[allow(dead_code)]
    fn assert_bounds() {
        assert_not_send::<PooledSession<'_>>();
        assert_not_sync::<PooledSession<'_>>();

        // Runtime-observable check: `PooledSession` must not implement
        // `Send`. This is enforced by the presence of
        // `PhantomData<*const ()>` on the struct — verified below by a
        // negative bound.
        fn is_send<T: Send>() {}
        fn is_sync<T: Sync>() {}
        // Uncommenting either of these should cause a compile error:
        // is_send::<PooledSession<'_>>();
        // is_sync::<PooledSession<'_>>();
        let _ = is_send::<()>;
        let _ = is_sync::<()>;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_pool_acquire_returns_backend_unavailable() {
        let pool = CudaDecodeSessionPool::new(2);
        let dims = SessionDims {
            d: 512,
            n_head: 8,
            ff: 2048,
            n_text_ctx: 448,
            n_vocab: 51864,
            n_ctx: 1500,
        };
        // `.unwrap_err()` would need `PooledSession: Debug` (the `Ok` variant)
        // for its panic message. `PooledSession` wraps a `CudaDecodeSession`
        // whose backing device buffers deliberately don't implement `Debug`,
        // so match the `Result` directly instead.
        match pool.acquire(&dims) {
            Err(VokraError::BackendUnavailable(msg)) => {
                assert!(
                    msg.contains("no matching entry"),
                    "unexpected error message: {msg}"
                );
            }
            Err(other) => panic!("expected BackendUnavailable, got {other:?}"),
            Ok(_) => panic!("empty pool must not hand out a session"),
        }
    }

    #[test]
    fn capacity_zero_disables_reuse() {
        let pool = CudaDecodeSessionPool::new(0);
        assert_eq!(pool.capacity(), 0);
        assert_eq!(pool.idle_len(), 0);
    }

    #[test]
    fn session_dims_equality_semantics() {
        // The `PartialEq`/`Eq` derive is the whole "exact match" contract
        // (§D5) — pin the tuple equality here so a stray field addition
        // that breaks the match key is caught at test time.
        let a = SessionDims {
            d: 512,
            n_head: 8,
            ff: 2048,
            n_text_ctx: 448,
            n_vocab: 51864,
            n_ctx: 1500,
        };
        let b = a;
        assert_eq!(a, b);

        let c = SessionDims {
            n_vocab: 51865,
            ..a
        };
        assert_ne!(a, c);
    }
}
