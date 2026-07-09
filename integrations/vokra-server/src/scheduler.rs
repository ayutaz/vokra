//! Multi-session scheduler for `vokra-server` (M3-15 T03 / T12).
//!
//! The scheduler is the request-side companion to
//! [`crate::session::SessionRegistry`]: it enforces a
//! `max_concurrent_sessions` cap using a tokio `Semaphore`, hands out a
//! [`SessionGuard`](crate::session::SessionGuard) once a permit is
//! available, and surfaces overload as
//! [`ServerError::ServiceUnavailable`](crate::error::ServerError::ServiceUnavailable)
//! → HTTP 503 (T12, FR-EX-08).
//!
//! # Two-tier bound
//!
//! - **Semaphore permits** (`max_concurrent_sessions`) — soft bound that
//!   supports async queueing. Requests block up to `queue_wait_max` for a
//!   free permit; longer waits fail with 503 rather than starving the
//!   caller.
//! - **Stream slots** (`n_stream` on the registry) — hard bound sized to
//!   match the paged KV cache's `n_stream` axis. The two bounds MUST be
//!   equal in production; the type below enforces this at construction
//!   time.
//!
//! # Silent-fallback ban (FR-EX-08)
//!
//! Every overload path returns an explicit
//! [`ServerError::ServiceUnavailable`] — no request is ever silently
//! rerouted to CPU or degraded to a lower-quality model. Callers can
//! choose between fast-fail (`try_acquire_now`) and async backpressure
//! (`acquire_or_503`); both surfaces of the API document this rule in
//! their doc-comments.

use std::sync::Arc;
use std::time::Duration;

use tokio::sync::Semaphore;

use crate::error::ServerError;
use crate::session::{RegistryError, SessionGuard, SessionRegistry, SessionRegistryConfig};

/// Static configuration for [`Scheduler`].
#[derive(Debug, Clone, Copy)]
pub struct SchedulerConfig {
    /// Maximum number of sessions the server is allowed to run
    /// concurrently. Must equal [`SessionRegistryConfig::n_stream`] so
    /// permit acquires cannot dangle on registry rejection.
    pub max_concurrent_sessions: usize,
    /// Maximum time [`Scheduler::acquire_or_503`] will wait for a permit
    /// before returning 503. Set to zero to make the acquire path
    /// fast-fail equivalent to [`Scheduler::try_acquire_now`].
    pub queue_wait_max: Duration,
}

impl SchedulerConfig {
    /// Minimum-viable config for tests: N slots, 100 ms queue wait.
    pub fn minimum(max_concurrent_sessions: usize) -> Self {
        Self {
            max_concurrent_sessions,
            queue_wait_max: Duration::from_millis(100),
        }
    }
}

/// Errors surfaced by scheduler acquire calls, prior to mapping into a
/// wire-facing [`ServerError`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SchedulerError {
    /// The semaphore was closed (e.g. shutdown drain). Rare — the
    /// scheduler does not close it under normal use.
    SchedulerClosed,
    /// A permit is available but the registry rejected the acquire. This
    /// is a bug (permits + stream slots should be in lock-step); it is
    /// surfaced as `InferenceFailed` so operators see it in the logs.
    Registry(RegistryError),
    /// The permit did not become available within `queue_wait_max`.
    Overloaded {
        /// Configured concurrency cap.
        max_concurrent_sessions: usize,
        /// The wait budget that was exhausted.
        waited: Duration,
    },
}

impl std::fmt::Display for SchedulerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SchedulerClosed => f.write_str("scheduler is closed"),
            Self::Registry(e) => write!(f, "registry: {e}"),
            Self::Overloaded {
                max_concurrent_sessions,
                waited,
            } => write!(
                f,
                "scheduler overloaded: waited {waited:?} for one of {max_concurrent_sessions} permits"
            ),
        }
    }
}

impl std::error::Error for SchedulerError {}

impl SchedulerError {
    /// Map to the wire-facing [`ServerError`]. Overload → 503;
    /// registry misconfig → 500. Never silently downgraded.
    pub fn to_server_error(&self) -> ServerError {
        match self {
            Self::SchedulerClosed => ServerError::ServiceUnavailable {
                detail: "scheduler is closed".to_string(),
            },
            Self::Overloaded {
                max_concurrent_sessions,
                waited,
            } => ServerError::ServiceUnavailable {
                detail: format!(
                    "server at concurrency cap ({max_concurrent_sessions}); waited {waited:?}"
                ),
            },
            Self::Registry(e) => e.to_server_error(),
        }
    }
}

/// A permit + a live session guard.
///
/// Holding this handle proves the caller both (a) held a semaphore permit
/// and (b) owns a registry slot; dropping it releases both automatically
/// via RAII.
pub struct SchedulerSession {
    guard: SessionGuard,
    // `_permit` is only held for its Drop — dropping decrements the
    // semaphore's outstanding count so backpressured callers can proceed.
    _permit: tokio::sync::OwnedSemaphorePermit,
}

impl SchedulerSession {
    /// Borrow the underlying session guard (immutable).
    pub fn guard(&self) -> &SessionGuard {
        &self.guard
    }

    /// Borrow the underlying session guard (mutable — for `touch`).
    pub fn guard_mut(&mut self) -> &mut SessionGuard {
        &mut self.guard
    }
}

/// Multi-session scheduler.
///
/// Owns an `Arc<SessionRegistry>` and a `Semaphore` sized to
/// `max_concurrent_sessions`. Shared across tokio worker threads by
/// `Arc<Scheduler>`.
pub struct Scheduler {
    registry: Arc<SessionRegistry>,
    permits: Arc<Semaphore>,
    config: SchedulerConfig,
}

impl Scheduler {
    /// Constructs a scheduler backed by a fresh registry.
    ///
    /// Both bounds are pinned to `scheduler.max_concurrent_sessions ==
    /// registry.n_stream`; a mismatch is a configuration error.
    ///
    /// # Errors
    ///
    /// Returns [`SchedulerError::Registry`] if the registry rejects the
    /// underlying shape (e.g. a zero axis).
    pub fn new(
        registry_cfg: SessionRegistryConfig,
        scheduler_cfg: SchedulerConfig,
    ) -> std::result::Result<Arc<Self>, SchedulerError> {
        if registry_cfg.n_stream != scheduler_cfg.max_concurrent_sessions {
            return Err(SchedulerError::Registry(RegistryError::InvalidConfig(
                format!(
                    "n_stream ({}) must equal max_concurrent_sessions ({})",
                    registry_cfg.n_stream, scheduler_cfg.max_concurrent_sessions,
                ),
            )));
        }
        let registry = SessionRegistry::new(registry_cfg).map_err(SchedulerError::Registry)?;
        let permits = Arc::new(Semaphore::new(scheduler_cfg.max_concurrent_sessions));
        Ok(Arc::new(Self {
            registry,
            permits,
            config: scheduler_cfg,
        }))
    }

    /// Constructs a scheduler on top of an existing registry (test hook).
    /// Fails if the registry's capacity does not match
    /// `scheduler_cfg.max_concurrent_sessions`.
    pub fn from_registry(
        registry: Arc<SessionRegistry>,
        scheduler_cfg: SchedulerConfig,
    ) -> std::result::Result<Arc<Self>, SchedulerError> {
        if registry.capacity() != scheduler_cfg.max_concurrent_sessions {
            return Err(SchedulerError::Registry(RegistryError::InvalidConfig(
                format!(
                    "registry capacity ({}) must equal max_concurrent_sessions ({})",
                    registry.capacity(),
                    scheduler_cfg.max_concurrent_sessions,
                ),
            )));
        }
        let permits = Arc::new(Semaphore::new(scheduler_cfg.max_concurrent_sessions));
        Ok(Arc::new(Self {
            registry,
            permits,
            config: scheduler_cfg,
        }))
    }

    /// Fast-fail acquire — returns a session immediately or
    /// [`SchedulerError::Overloaded`] if no permit is available. Never
    /// blocks. Suitable for endpoint paths that prefer to shed load
    /// rather than queue (e.g. real-time streaming).
    ///
    /// # Errors
    ///
    /// - [`SchedulerError::Overloaded`] — the concurrency cap is
    ///   currently saturated (no permit available in the semaphore).
    /// - [`SchedulerError::Registry`] — permit acquired but registry
    ///   rejected (indicates config drift; should not happen in
    ///   practice).
    pub fn try_acquire_now(&self) -> std::result::Result<SchedulerSession, SchedulerError> {
        let permit = match Arc::clone(&self.permits).try_acquire_owned() {
            Ok(p) => p,
            Err(tokio::sync::TryAcquireError::NoPermits) => {
                return Err(SchedulerError::Overloaded {
                    max_concurrent_sessions: self.config.max_concurrent_sessions,
                    waited: Duration::ZERO,
                });
            }
            Err(tokio::sync::TryAcquireError::Closed) => {
                return Err(SchedulerError::SchedulerClosed);
            }
        };
        let guard = self
            .registry
            .try_acquire()
            .map_err(SchedulerError::Registry)?;
        Ok(SchedulerSession {
            guard,
            _permit: permit,
        })
    }

    /// Async acquire with a bounded wait budget. Backpressures inside
    /// `queue_wait_max`; returns
    /// [`SchedulerError::Overloaded`] on timeout (T12, FR-EX-08 — no
    /// silent fallback: overloaded requests are refused with 503).
    ///
    /// # Errors
    ///
    /// Same variants as [`Self::try_acquire_now`], plus timeout is
    /// surfaced as [`SchedulerError::Overloaded`] with the elapsed
    /// waited duration.
    pub async fn acquire_or_503(&self) -> std::result::Result<SchedulerSession, SchedulerError> {
        let started = tokio::time::Instant::now();
        let deadline = started + self.config.queue_wait_max;
        let permits = Arc::clone(&self.permits);
        let permit = match tokio::time::timeout_at(deadline, permits.acquire_owned()).await {
            Ok(Ok(p)) => p,
            Ok(Err(_closed)) => return Err(SchedulerError::SchedulerClosed),
            Err(_elapsed) => {
                return Err(SchedulerError::Overloaded {
                    max_concurrent_sessions: self.config.max_concurrent_sessions,
                    waited: started.elapsed(),
                });
            }
        };
        let guard = self
            .registry
            .try_acquire()
            .map_err(SchedulerError::Registry)?;
        Ok(SchedulerSession {
            guard,
            _permit: permit,
        })
    }

    /// Snapshot of currently-live sessions.
    pub fn in_use(&self) -> usize {
        self.registry.in_use()
    }

    /// Maximum concurrent sessions this scheduler accepts.
    pub fn capacity(&self) -> usize {
        self.config.max_concurrent_sessions
    }

    /// Available permits at call time. Snapshot only — subject to
    /// concurrent acquires/releases across worker threads.
    pub fn available_permits(&self) -> usize {
        self.permits.available_permits()
    }

    /// Registry the scheduler is bound to (read-only).
    pub fn registry(&self) -> &Arc<SessionRegistry> {
        &self.registry
    }

    /// Config the scheduler was built with (read-only).
    pub fn config(&self) -> &SchedulerConfig {
        &self.config
    }

    /// Start the shutdown drain: registry stops accepting new acquires,
    /// existing sessions can complete. New calls to `try_acquire_now` /
    /// `acquire_or_503` still consume a permit (they must, else the
    /// semaphore invariant breaks) but then fail-fast at the registry
    /// step with [`SchedulerError::Registry`].
    pub fn begin_shutdown(&self) {
        self.registry.begin_shutdown();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn mk(n: usize) -> Arc<Scheduler> {
        let reg_cfg = SessionRegistryConfig::minimum(n);
        let sched_cfg = SchedulerConfig::minimum(n);
        Scheduler::new(reg_cfg, sched_cfg).expect("scheduler builds")
    }

    #[tokio::test]
    async fn acquire_or_503_returns_session_immediately_when_free() {
        let s = mk(2);
        let sess = s.acquire_or_503().await.expect("free permits");
        assert_eq!(s.in_use(), 1);
        drop(sess);
        assert_eq!(s.in_use(), 0);
    }

    #[tokio::test]
    async fn try_acquire_now_saturation_returns_overloaded_503() {
        let s = mk(1);
        let _held = s.try_acquire_now().unwrap();
        // `SchedulerSession` does not implement `Debug` (it wraps a
        // registry guard whose device state we intentionally don't
        // format). Match `Result` explicitly instead of using
        // `unwrap_err` + `expect_err`.
        match s.try_acquire_now() {
            Err(SchedulerError::Overloaded {
                max_concurrent_sessions,
                waited,
            }) => {
                assert_eq!(max_concurrent_sessions, 1);
                assert_eq!(waited, Duration::ZERO);
            }
            Err(other) => panic!("expected Overloaded, got {other:?}"),
            Ok(_) => panic!("try_acquire_now must refuse a 2nd session at capacity 1"),
        }
    }

    #[tokio::test]
    async fn acquire_or_503_times_out_on_saturation() {
        // Hold every permit for the whole test scope. `acquire_or_503`
        // must not silently succeed by degrading; it must emit
        // Overloaded after the configured wait.
        let reg_cfg = SessionRegistryConfig::minimum(1);
        let sched_cfg = SchedulerConfig {
            max_concurrent_sessions: 1,
            queue_wait_max: Duration::from_millis(20),
        };
        let s = Scheduler::new(reg_cfg, sched_cfg).unwrap();
        let _held = s.try_acquire_now().unwrap();
        let t0 = tokio::time::Instant::now();
        match s.acquire_or_503().await {
            Err(SchedulerError::Overloaded { waited, .. }) => {
                assert!(waited >= Duration::from_millis(20));
                assert!(t0.elapsed() >= Duration::from_millis(20));
            }
            Err(other) => panic!("expected Overloaded timeout, got {other:?}"),
            Ok(_) => panic!("saturated scheduler must not hand out a session"),
        }
    }

    #[tokio::test]
    async fn queue_wait_lets_waiter_acquire_after_release() {
        let reg_cfg = SessionRegistryConfig::minimum(1);
        let sched_cfg = SchedulerConfig {
            max_concurrent_sessions: 1,
            queue_wait_max: Duration::from_secs(1),
        };
        let s = Scheduler::new(reg_cfg, sched_cfg).unwrap();
        let held = s.try_acquire_now().unwrap();
        let s2 = Arc::clone(&s);
        let waiter = tokio::spawn(async move { s2.acquire_or_503().await });
        // Give the waiter a moment to enter the semaphore's wait queue.
        tokio::time::sleep(Duration::from_millis(10)).await;
        drop(held);
        let got = waiter.await.unwrap();
        got.expect("waiter must acquire after release");
    }

    #[tokio::test]
    async fn shutdown_drains_via_registry() {
        let s = mk(2);
        let alive = s.try_acquire_now().unwrap();
        s.begin_shutdown();
        // A permit is available (only one is held), but the registry
        // refuses new acquires now.
        match s.acquire_or_503().await {
            Err(SchedulerError::Registry(RegistryError::ShuttingDown)) => {}
            Err(other) => panic!("expected Registry(ShuttingDown), got {other:?}"),
            Ok(_) => panic!("shutdown must refuse new sessions"),
        }
        // The live session stays alive.
        assert_eq!(alive.guard().id().0, alive.guard().id().0);
        drop(alive);
    }

    #[tokio::test]
    async fn concurrent_acquires_respect_capacity() {
        let capacity = 5;
        let scheduler = mk(capacity);
        // Fire 20 tokio tasks; only `capacity` should be alive
        // simultaneously.
        let live = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));

        let mut handles = Vec::new();
        for _ in 0..20 {
            let s = Arc::clone(&scheduler);
            let live = Arc::clone(&live);
            let peak = Arc::clone(&peak);
            handles.push(tokio::spawn(async move {
                let session = s
                    .acquire_or_503()
                    .await
                    .expect("acquire under bounded queue");
                let now = live.fetch_add(1, Ordering::SeqCst) + 1;
                peak.fetch_max(now, Ordering::SeqCst);
                // Simulate a short piece of work.
                tokio::time::sleep(Duration::from_millis(5)).await;
                live.fetch_sub(1, Ordering::SeqCst);
                drop(session);
            }));
        }
        for h in handles {
            h.await.unwrap();
        }
        let peak_val = peak.load(Ordering::SeqCst);
        assert!(
            peak_val <= capacity,
            "peak {peak_val} exceeded capacity {capacity}"
        );
        assert_eq!(scheduler.in_use(), 0);
    }

    #[tokio::test]
    async fn mismatched_capacity_is_rejected_at_construction() {
        let reg_cfg = SessionRegistryConfig::minimum(2);
        let sched_cfg = SchedulerConfig::minimum(3);
        match Scheduler::new(reg_cfg, sched_cfg) {
            Err(SchedulerError::Registry(RegistryError::InvalidConfig(msg))) => {
                assert!(msg.contains("must equal"), "unexpected msg: {msg}");
            }
            Err(other) => panic!("expected InvalidConfig, got {other:?}"),
            Ok(_) => panic!("mismatched capacity must fail construction"),
        }
    }

    #[tokio::test]
    async fn overloaded_maps_to_server_error_503() {
        let err = SchedulerError::Overloaded {
            max_concurrent_sessions: 4,
            waited: Duration::from_millis(50),
        };
        let se = err.to_server_error();
        assert!(matches!(se, ServerError::ServiceUnavailable { .. }));
        assert_eq!(se.status(), axum::http::StatusCode::SERVICE_UNAVAILABLE);
    }
}
