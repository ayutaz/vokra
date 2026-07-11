//! Latency recording + percentile calculation for `vokra-cli bench` and the
//! M3-15 multi-session load harness.
//!
//! # Scope (M3-15 T11)
//!
//! Provides:
//! - [`LatencyRecorder`] — a bounded, thread-safe append-only collector for
//!   `Duration` samples (TTFA, RTF, per-request wall time). Backed by a
//!   `Mutex<Vec<Duration>>`; the vector is pre-allocated to capacity so the
//!   hot path (`record`) does not touch the system allocator once the
//!   recorder is under steady load.
//! - [`LatencyReport`] — the P50 / P95 / P99 summary emitted at the end of
//!   a run. Also carries `mean` and `min` / `max` so the bench harness can
//!   surface the full distribution shape.
//!
//! # Percentile method
//!
//! We use the **nearest-rank** method (also called the C = 0 variant):
//! `p_k = samples_sorted[ceil(k / 100 * n) - 1]`. This matches how most
//! bench tools (wrk, hey, k6) report percentiles and produces a value
//! that is always a real sample (never an interpolation), which makes
//! the M3-15 T11 "75 ms reference reading" reproducible: two runs at the
//! same N will report the exact same P95 as long as the sample
//! multiset is identical.
//!
//! # Zero-dep
//!
//! Uses only `std`. No new third-party crate; the excluded-workspace HTTP
//! stack is untouched.

use std::sync::Mutex;
use std::time::Duration;

/// Bounded latency sample collector.
///
/// Thread-safe (`Send + Sync`) via an internal `Mutex`. Sized at
/// construction; overflow is dropped with an atomic counter so the
/// caller can detect it without disturbing the recorded distribution.
pub struct LatencyRecorder {
    samples: Mutex<Vec<Duration>>,
    capacity: usize,
    dropped: std::sync::atomic::AtomicUsize,
}

impl LatencyRecorder {
    /// Constructs a recorder sized to hold up to `capacity` samples.
    /// Additional samples are dropped and counted in
    /// [`Self::dropped_samples`].
    ///
    /// # Panics
    ///
    /// Panics if `capacity == 0`; a recorder that cannot hold any
    /// sample is never useful, so this is a hard misuse error.
    pub fn with_capacity(capacity: usize) -> Self {
        assert!(capacity > 0, "LatencyRecorder capacity must be > 0");
        Self {
            samples: Mutex::new(Vec::with_capacity(capacity)),
            capacity,
            dropped: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Records a single sample. O(1) — the internal `Vec` never
    /// reallocates on the hot path because it was constructed with
    /// exactly `capacity`.
    ///
    /// Overflow (past `capacity`) is dropped and counted so the caller
    /// can size the recorder deterministically for the bench harness.
    ///
    /// # Panics
    ///
    /// Panics only if the internal mutex was poisoned by a prior panic.
    pub fn record(&self, sample: Duration) {
        let mut v = self.samples.lock().expect("latency recorder poisoned");
        if v.len() < self.capacity {
            v.push(sample);
        } else {
            self.dropped
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// Number of samples dropped due to capacity overflow.
    pub fn dropped_samples(&self) -> usize {
        self.dropped.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Snapshot of recorded samples. Order is insertion order; the
    /// report path sorts internally so the caller does not need to.
    ///
    /// # Panics
    ///
    /// Panics only if the internal mutex was poisoned.
    pub fn snapshot(&self) -> Vec<Duration> {
        self.samples
            .lock()
            .expect("latency recorder poisoned")
            .clone()
    }

    /// Computes the [`LatencyReport`] from the currently-recorded
    /// samples. Returns `None` if no samples have been recorded (an
    /// empty report is not meaningful — the caller should error out).
    pub fn report(&self) -> Option<LatencyReport> {
        let mut v = self
            .samples
            .lock()
            .expect("latency recorder poisoned")
            .clone();
        if v.is_empty() {
            return None;
        }
        v.sort();
        let n = v.len();
        let min = v[0];
        let max = v[n - 1];
        // Mean via integer nanos to avoid float-rounding surprises.
        let total_ns: u128 = v.iter().map(|d| d.as_nanos()).sum();
        let mean_ns = total_ns / (n as u128);
        let mean = Duration::from_nanos(mean_ns as u64);

        Some(LatencyReport {
            n,
            min,
            max,
            mean,
            p50: nearest_rank(&v, 50),
            p95: nearest_rank(&v, 95),
            p99: nearest_rank(&v, 99),
            dropped: self.dropped_samples(),
        })
    }

    /// Configured capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Number of samples currently held (bounded by [`Self::capacity`]).
    pub fn len(&self) -> usize {
        self.samples
            .lock()
            .expect("latency recorder poisoned")
            .len()
    }

    /// `true` iff no samples have been recorded yet.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Nearest-rank percentile. `p` in `1..=100`. Panics on out-of-range `p`;
/// the recorder never calls it out of range.
fn nearest_rank(sorted: &[Duration], p: usize) -> Duration {
    assert!((1..=100).contains(&p), "percentile out of range: {p}");
    let n = sorted.len();
    // Rank formula: ceil(p / 100 * n). For p=100 → n; for p=1 → 1 if n>=1.
    let numerator = p * n;
    let mut rank = numerator / 100;
    if numerator % 100 != 0 {
        rank += 1;
    }
    // Convert 1-based rank to 0-based index; clamp for safety.
    let idx = rank.saturating_sub(1).min(n - 1);
    sorted[idx]
}

/// Immutable snapshot of a distribution's summary statistics.
///
/// All fields are `Duration` so bench tooling can render them in the
/// most-natural unit without further conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LatencyReport {
    /// Number of samples that formed this report.
    pub n: usize,
    /// Minimum recorded sample.
    pub min: Duration,
    /// Maximum recorded sample.
    pub max: Duration,
    /// Arithmetic mean, computed at nanosecond resolution.
    pub mean: Duration,
    /// 50th percentile (median).
    pub p50: Duration,
    /// 95th percentile.
    pub p95: Duration,
    /// 99th percentile.
    pub p99: Duration,
    /// Samples dropped due to capacity overflow.
    pub dropped: usize,
}

impl LatencyReport {
    /// Reference sanity check for the M3-15 TTFA gate:
    /// [`Self::p95`] `<= threshold`.
    ///
    /// Returns `true` iff the 95th-percentile latency is within
    /// `threshold`. The M3-15 T11 CLI harness surfaces this against the
    /// NFR-PF-05 v1.0 value of 75 ms; missed budgets are recorded as
    /// participant data for the quarterly review, not silently masked
    /// (see the ticket spec).
    pub fn within_p95(&self, threshold: Duration) -> bool {
        self.p95 <= threshold
    }

    /// Render as a compact JSON blob (hand-serialised, no `serde` on
    /// this path — this keeps the recorder callable from paths outside
    /// the excluded-workspace serde surface if they ever need it).
    pub fn to_json_string(&self) -> String {
        format!(
            "{{\"n\":{},\"min_us\":{},\"max_us\":{},\"mean_us\":{},\"p50_us\":{},\"p95_us\":{},\"p99_us\":{},\"dropped\":{}}}",
            self.n,
            self.min.as_micros(),
            self.max.as_micros(),
            self.mean.as_micros(),
            self.p50.as_micros(),
            self.p95.as_micros(),
            self.p99.as_micros(),
            self.dropped,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ms(x: u64) -> Duration {
        Duration::from_millis(x)
    }

    #[test]
    fn empty_recorder_returns_none() {
        let r = LatencyRecorder::with_capacity(4);
        assert!(r.report().is_none());
        assert!(r.is_empty());
    }

    #[test]
    fn nearest_rank_matches_known_values() {
        // Standard example: {15,20,35,40,50}, p95 = ceil(0.95*5)=5 → 50.
        let v: Vec<Duration> = [15u64, 20, 35, 40, 50].iter().map(|x| ms(*x)).collect();
        assert_eq!(nearest_rank(&v, 50), ms(35));
        assert_eq!(nearest_rank(&v, 95), ms(50));
        assert_eq!(nearest_rank(&v, 99), ms(50));
        // p=1 must return the smallest sample (rank=1 → idx 0).
        assert_eq!(nearest_rank(&v, 1), ms(15));
        // p=100 must return the largest sample (rank=n → idx n-1).
        assert_eq!(nearest_rank(&v, 100), ms(50));
    }

    #[test]
    fn recorder_computes_report_correctly() {
        let r = LatencyRecorder::with_capacity(5);
        for x in [15u64, 20, 35, 40, 50] {
            r.record(ms(x));
        }
        let rep = r.report().unwrap();
        assert_eq!(rep.n, 5);
        assert_eq!(rep.min, ms(15));
        assert_eq!(rep.max, ms(50));
        // Mean of {15,20,35,40,50} = 32 ms
        assert_eq!(rep.mean, ms(32));
        assert_eq!(rep.p50, ms(35));
        assert_eq!(rep.p95, ms(50));
        assert_eq!(rep.p99, ms(50));
        assert_eq!(rep.dropped, 0);
    }

    #[test]
    fn recorder_drops_overflow_and_counts_it() {
        let r = LatencyRecorder::with_capacity(2);
        r.record(ms(10));
        r.record(ms(20));
        r.record(ms(30)); // dropped
        r.record(ms(40)); // dropped
        let rep = r.report().unwrap();
        assert_eq!(rep.n, 2);
        assert_eq!(rep.min, ms(10));
        assert_eq!(rep.max, ms(20));
        assert_eq!(rep.dropped, 2);
    }

    #[test]
    fn recorder_len_and_snapshot_are_consistent() {
        let r = LatencyRecorder::with_capacity(3);
        r.record(ms(5));
        r.record(ms(15));
        assert_eq!(r.len(), 2);
        assert!(!r.is_empty());
        let snap = r.snapshot();
        assert_eq!(snap, vec![ms(5), ms(15)]);
    }

    #[test]
    fn within_p95_reports_boolean_verdict() {
        let r = LatencyRecorder::with_capacity(10);
        for x in [10u64, 20, 30, 40, 50, 60, 70, 80, 90, 100] {
            r.record(ms(x));
        }
        let rep = r.report().unwrap();
        // p95 with nearest-rank on n=10: rank = ceil(0.95*10) = 10 → 100 ms.
        assert_eq!(rep.p95, ms(100));
        assert!(!rep.within_p95(ms(75)));
        assert!(rep.within_p95(ms(100)));
        assert!(rep.within_p95(ms(200)));
    }

    #[test]
    fn to_json_string_carries_all_fields() {
        let r = LatencyRecorder::with_capacity(3);
        r.record(ms(1));
        r.record(ms(2));
        r.record(ms(3));
        let rep = r.report().unwrap();
        let json = rep.to_json_string();
        assert!(json.contains("\"n\":3"));
        assert!(json.contains("\"p50_us\":2000"));
        assert!(json.contains("\"p95_us\":3000"));
        assert!(json.contains("\"p99_us\":3000"));
    }

    #[test]
    fn recorder_is_send_and_sync() {
        // Compile-time check that Arc<LatencyRecorder> is usable across
        // tokio worker threads (the scheduler needs this).
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<LatencyRecorder>();
    }

    #[test]
    fn ordering_within_report_is_stable_across_runs() {
        // Two runs with identical samples must produce bit-identical
        // reports — the CI parity harness relies on this determinism.
        let inputs = [42u64, 17, 88, 3, 61, 74, 29];
        let r1 = LatencyRecorder::with_capacity(inputs.len());
        let r2 = LatencyRecorder::with_capacity(inputs.len());
        for x in inputs {
            r1.record(ms(x));
        }
        // Same inputs in a different order — sort inside report()
        // must produce the same summary.
        for x in inputs.iter().rev() {
            r2.record(ms(*x));
        }
        assert_eq!(r1.report(), r2.report());
    }

    #[test]
    #[should_panic(expected = "capacity must be > 0")]
    fn zero_capacity_panics() {
        let _ = LatencyRecorder::with_capacity(0);
    }
}
