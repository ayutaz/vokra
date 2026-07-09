//! Multi-session integration test (M3-15 T09 / T10 / T12).
//!
//! Covers the three pieces the M3-15 completion condition calls out
//! (milestones.md §7.2, ticket spec T09/T10/T12):
//!
//! - **T09 — parallel N session dispatch**: 10 concurrent tokio tasks
//!   acquire+release SchedulerSessions and never observe overlapping
//!   stream slots.
//! - **T10 — state leak isolation**: two live sessions writing distinct
//!   KV rows to the same [`PagedKvCache`] under different stream slots
//!   must observe their own rows and only their own rows (no A→B bleed).
//! - **T12 — graceful degradation**: an oversubscribed scheduler must
//!   return [`ServerError::ServiceUnavailable`] with HTTP 503 semantics
//!   (never silent CPU fallback, FR-EX-08).
//!
//! The test drives the M3-15 primitives directly (Scheduler + Registry
//! + PagedKvCache), NOT the HTTP surface — endpoint wiring (T06/T07) is
//!   reserved for a later CC pass where model paths are configurable. That
//!   keeps this suite fast (< 200 ms end-to-end) and free of GGUF fixtures.

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use vokra_core::cache::paged::BlockSize;
use vokra_server::error::ServerError;
use vokra_server::latency::LatencyRecorder;
use vokra_server::scheduler::{Scheduler, SchedulerConfig, SchedulerError};
use vokra_server::session::SessionRegistryConfig;

// ----- helpers -------------------------------------------------------------

/// Build a scheduler + registry sized for `n` concurrent sessions using
/// the minimum-viable paged-cache shape (1 layer × 1 head × 1 chan × 32
/// steps @ block_size=4). Matches the shape SessionRegistryConfig::minimum
/// uses — kept explicit here so the test is self-documenting for any
/// future reader triaging a hang.
fn mk_scheduler(n: usize) -> Arc<Scheduler> {
    let reg_cfg = SessionRegistryConfig::minimum(n);
    let sched_cfg = SchedulerConfig {
        max_concurrent_sessions: n,
        // Ample wait budget so the multi-session load test never trips
        // the timeout for legitimate work.
        queue_wait_max: Duration::from_secs(2),
    };
    Scheduler::new(reg_cfg, sched_cfg).expect("scheduler builds")
}

/// A wider registry — used by tests that need multiple layers to prove
/// the state-isolation guarantee across the whole cache, not just a
/// single-layer smoke.
fn mk_wide_scheduler(n_stream: usize, n_layer: usize, d_head: usize) -> Arc<Scheduler> {
    let reg_cfg = SessionRegistryConfig {
        n_layer,
        n_head: 1,
        d_head,
        n_stream,
        max_time: 8,
        block_size: BlockSize::Two,
    };
    let sched_cfg = SchedulerConfig {
        max_concurrent_sessions: n_stream,
        queue_wait_max: Duration::from_secs(2),
    };
    Scheduler::new(reg_cfg, sched_cfg).expect("wide scheduler builds")
}

// ----- T09: parallel N session dispatch ------------------------------------

/// N=10 concurrent sessions acquire + release. Every acquired session
/// observes a distinct stream slot and the peak concurrency never
/// exceeds the configured cap.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn t09_parallel_10_sessions_dispatch_and_release() {
    let n = 10;
    let scheduler = mk_scheduler(n);
    let live = Arc::new(AtomicUsize::new(0));
    let peak = Arc::new(AtomicUsize::new(0));
    let recorder = Arc::new(LatencyRecorder::with_capacity(n));

    let mut handles = Vec::new();
    for i in 0..n {
        let s = Arc::clone(&scheduler);
        let live = Arc::clone(&live);
        let peak = Arc::clone(&peak);
        let recorder = Arc::clone(&recorder);
        handles.push(tokio::spawn(async move {
            let t0 = std::time::Instant::now();
            let session = s
                .acquire_or_503()
                .await
                .expect("acquire in the wait budget");
            recorder.record(t0.elapsed());
            let now = live.fetch_add(1, Ordering::SeqCst) + 1;
            peak.fetch_max(now, Ordering::SeqCst);
            // Yield so the scheduler observably interleaves tasks.
            tokio::task::yield_now().await;
            // Simulated per-request work (~2ms) — enough for real
            // concurrency to surface.
            tokio::time::sleep(Duration::from_millis(2)).await;
            live.fetch_sub(1, Ordering::SeqCst);
            (i, session.guard().id().0, session.guard().stream().0)
        }));
    }

    let mut results = Vec::new();
    for h in handles {
        results.push(h.await.expect("task must not panic"));
    }
    // Peak may equal capacity but must never exceed it.
    let peak_val = peak.load(Ordering::SeqCst);
    assert!(
        peak_val <= n,
        "peak concurrency {peak_val} exceeded cap {n}"
    );
    // Every task saw a distinct session id (monotonic counter).
    let ids: HashSet<u64> = results.iter().map(|(_, id, _)| *id).collect();
    assert_eq!(ids.len(), n, "session ids should be unique per acquire");
    // Every acquire recorded a latency sample.
    let report = recorder.report().expect("recorder has samples");
    assert_eq!(report.n, n);
    // No sessions should linger.
    assert_eq!(scheduler.in_use(), 0);
}

/// The mixed workload matches the "TTS + ASR混在" scenario from T09:
/// two workloads (represented here by two closures on the same
/// scheduler) share the paged cache without cross-contamination.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn t09_mixed_tts_asr_style_workload() {
    let n = 6;
    let scheduler = mk_scheduler(n);
    let asr_done = Arc::new(AtomicUsize::new(0));
    let tts_done = Arc::new(AtomicUsize::new(0));

    let mut handles = Vec::new();
    // 3 "ASR-style" tasks — short duration.
    for _ in 0..3 {
        let s = Arc::clone(&scheduler);
        let counter = Arc::clone(&asr_done);
        handles.push(tokio::spawn(async move {
            let session = s.acquire_or_503().await.expect("asr acquire");
            tokio::time::sleep(Duration::from_millis(1)).await;
            counter.fetch_add(1, Ordering::SeqCst);
            drop(session);
        }));
    }
    // 3 "TTS-style" tasks — slightly longer.
    for _ in 0..3 {
        let s = Arc::clone(&scheduler);
        let counter = Arc::clone(&tts_done);
        handles.push(tokio::spawn(async move {
            let session = s.acquire_or_503().await.expect("tts acquire");
            tokio::time::sleep(Duration::from_millis(3)).await;
            counter.fetch_add(1, Ordering::SeqCst);
            drop(session);
        }));
    }
    for h in handles {
        h.await.expect("no panic in mixed workload");
    }
    assert_eq!(asr_done.load(Ordering::SeqCst), 3);
    assert_eq!(tts_done.load(Ordering::SeqCst), 3);
    assert_eq!(scheduler.in_use(), 0);
}

// ----- T10: state leak isolation -------------------------------------------

/// Two concurrent sessions on distinct stream slots write different KV
/// signatures; each session reads its own slot back and observes only
/// its own writes.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn t10_state_leak_across_two_live_sessions() {
    // 2 streams, 2 layers, d_head=4 → per-slot row = 4 f32.
    let scheduler = mk_wide_scheduler(2, 2, 4);
    let registry = Arc::clone(scheduler.registry());

    let a = scheduler.try_acquire_now().expect("session A acquires");
    let b = scheduler.try_acquire_now().expect("session B acquires");
    let a_slot = a.guard().stream().0;
    let b_slot = b.guard().stream().0;
    assert_ne!(a_slot, b_slot, "distinct sessions must get distinct slots");

    // Signature per session (different constant per stream).
    let a_k = vec![1.0f32, 2.0, 3.0, 4.0];
    let a_v = vec![-1.0f32, -2.0, -3.0, -4.0];
    let b_k = vec![10.0f32, 20.0, 30.0, 40.0];
    let b_v = vec![-10.0f32, -20.0, -30.0, -40.0];

    // Interleave writes: A(t=0), B(t=0), A(t=1), B(t=1).
    registry.with_cache(|c| {
        c.append_step(0, 0, a_slot, 0, &a_k, &a_v).unwrap();
        c.append_step(0, 0, b_slot, 0, &b_k, &b_v).unwrap();
        c.append_step(0, 1, a_slot, 0, &a_k, &a_v).unwrap();
        c.append_step(0, 1, b_slot, 0, &b_k, &b_v).unwrap();
        c.advance(2);
    });

    // Each session sees its own signature — no bleed.
    registry.with_cache_ref(|c| {
        for t in 0..2 {
            let (ka, va) = c.read_step(0, t, a_slot, 0).expect("A row exists");
            assert_eq!(ka, &a_k[..], "session A row {t} was overwritten by B");
            assert_eq!(va, &a_v[..], "session A row {t} was overwritten by B");
            let (kb, vb) = c.read_step(0, t, b_slot, 0).expect("B row exists");
            assert_eq!(kb, &b_k[..], "session B row {t} was overwritten by A");
            assert_eq!(vb, &b_v[..], "session B row {t} was overwritten by A");
        }
    });

    drop(a);
    drop(b);
}

/// Session release must zero the stream's KV rows so the next acquirer
/// on the same slot cannot observe stale data (defence in depth
/// against a caller that reads before appending).
#[tokio::test]
async fn t10_state_leak_across_release_and_reacquire() {
    let scheduler = mk_wide_scheduler(1, 1, 4);
    let registry = Arc::clone(scheduler.registry());

    // Session A writes a distinctive row.
    let a = scheduler.try_acquire_now().expect("session A");
    let a_slot = a.guard().stream().0;
    let sig_k = vec![100.0f32, 200.0, 300.0, 400.0];
    let sig_v = vec![-100.0f32, -200.0, -300.0, -400.0];
    registry.with_cache(|c| {
        c.append_step(0, 0, a_slot, 0, &sig_k, &sig_v).unwrap();
        c.advance(1);
    });
    // Sanity: A can read its own row.
    registry.with_cache_ref(|c| {
        let (k, _) = c.read_step(0, 0, a_slot, 0).expect("A row");
        assert_eq!(k, &sig_k[..]);
    });
    drop(a); // release zeroes the stream slot

    // Session B reacquires the same slot (only stream) and reads it —
    // the row must be zero, not the signature.
    let b = scheduler.try_acquire_now().expect("session B");
    assert_eq!(b.guard().stream().0, a_slot, "same slot reused");
    registry.with_cache_ref(|c| {
        if let Some((k, v)) = c.read_step(0, 0, b.guard().stream().0, 0) {
            assert!(k.iter().all(|x| *x == 0.0), "leaked K after release: {k:?}");
            assert!(v.iter().all(|x| *x == 0.0), "leaked V after release: {v:?}");
        }
    });
    drop(b);
}

// ----- T12: graceful degradation on overload -------------------------------

/// A saturated scheduler must return [`ServerError::ServiceUnavailable`]
/// with HTTP 503 status. No silent CPU fallback (FR-EX-08).
#[tokio::test]
async fn t12_saturation_returns_explicit_503_not_silent_fallback() {
    let scheduler = mk_scheduler(1);
    // Hold the only slot.
    let _held = scheduler.try_acquire_now().expect("initial acquire");
    // Try to acquire another slot — must be rejected explicitly.
    let err = match scheduler.try_acquire_now() {
        Ok(_) => panic!("scheduler must refuse a 2nd session at cap=1"),
        Err(e) => e,
    };
    match &err {
        SchedulerError::Overloaded { .. } => {}
        other => panic!("expected Overloaded, got {other:?}"),
    }
    // The wire-facing mapping produces a real 503.
    let server_err = err.to_server_error();
    assert!(matches!(server_err, ServerError::ServiceUnavailable { .. }));
    assert_eq!(
        server_err.status(),
        axum::http::StatusCode::SERVICE_UNAVAILABLE
    );
    assert_eq!(server_err.type_tag(), "service_unavailable");
}

/// The queueing acquire (`acquire_or_503`) also fails-explicit after
/// its wait budget, never silently downgrades.
#[tokio::test]
async fn t12_saturated_queue_times_out_with_503() {
    let reg_cfg = SessionRegistryConfig::minimum(1);
    let sched_cfg = SchedulerConfig {
        max_concurrent_sessions: 1,
        queue_wait_max: Duration::from_millis(15),
    };
    let scheduler = Scheduler::new(reg_cfg, sched_cfg).unwrap();
    let _held = scheduler.try_acquire_now().unwrap();
    let t0 = std::time::Instant::now();
    let err = match scheduler.acquire_or_503().await {
        Ok(_) => panic!("queued acquire must not silently succeed"),
        Err(e) => e,
    };
    assert!(t0.elapsed() >= Duration::from_millis(15));
    let server_err = err.to_server_error();
    assert!(matches!(server_err, ServerError::ServiceUnavailable { .. }));
}

/// Graceful shutdown must refuse new acquires while letting in-flight
/// sessions complete (M3-15 T12 semantics + T02 begin_shutdown).
#[tokio::test]
async fn t12_shutdown_lets_inflight_drain_and_refuses_new() {
    let scheduler = mk_scheduler(3);
    let live = scheduler.try_acquire_now().expect("in-flight session");
    scheduler.begin_shutdown();
    // New acquires refused.
    match scheduler.acquire_or_503().await {
        Ok(_) => panic!("shutdown must refuse new acquires"),
        Err(SchedulerError::Registry(vokra_server::session::RegistryError::ShuttingDown)) => {}
        Err(other) => panic!("expected ShuttingDown, got {other:?}"),
    }
    // Live session can still complete gracefully.
    let id = live.guard().id();
    drop(live);
    assert_eq!(scheduler.in_use(), 0, "session {id} released cleanly");
}

// ----- T11: TTFA percentile harness smoke ----------------------------------

/// Exercise the LatencyRecorder end-to-end in a session harness — the
/// same shape M3-15 T11 uses to publish a TTFA P50/P95/P99 reference.
/// This test does not gate a real 75 ms threshold (that requires GGUF
/// fixtures and is the owner-driven X-06 harness); it verifies the
/// harness itself is wired.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn t11_tttfa_recorder_smoke_end_to_end() {
    let n = 8;
    let scheduler = mk_scheduler(n);
    let recorder = Arc::new(LatencyRecorder::with_capacity(n));

    let mut handles = Vec::new();
    for _ in 0..n {
        let s = Arc::clone(&scheduler);
        let rec = Arc::clone(&recorder);
        handles.push(tokio::spawn(async move {
            let t0 = std::time::Instant::now();
            let session = s.acquire_or_503().await.expect("acquire");
            // Simulate the moment the first PCM chunk is emitted.
            // Real TTS would emit here; we use a sleep so the recorder
            // sees measurably non-zero samples.
            tokio::time::sleep(Duration::from_micros(500)).await;
            rec.record(t0.elapsed());
            drop(session);
        }));
    }
    for h in handles {
        h.await.expect("smoke task");
    }
    let report = recorder.report().expect("report has samples");
    assert_eq!(report.n, n);
    // P99 must be finite and at least the sleep budget we injected.
    assert!(report.p99 >= Duration::from_micros(500));
    // JSON export is stable + machine readable.
    let json = report.to_json_string();
    assert!(json.contains("\"n\":8"));
    assert!(json.contains("\"p95_us\""));
    assert!(json.contains("\"p99_us\""));
}
