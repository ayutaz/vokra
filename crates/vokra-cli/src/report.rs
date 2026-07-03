//! Bench statistics + machine-readable report serialization (M1-10a, M1-10b).
//!
//! The pure, model-independent core of the `bench` subcommand: latency summary
//! statistics (mean, jitter, p50/p95/p99), a hand-written `key=value` / JSON
//! serializer (no serde — external deps are forbidden, NFR-DS-02) and the
//! **relative** regression comparison used by the M1-10b perf gate (NFR-PF-13:
//! flag a >5% regression against a committed baseline).
//!
//! The absolute NFR-PF-01 / NFR-PF-02 thresholds (e.g. Whisper base RTF < 1.0,
//! piper-plus RTF < 0.5) are **not** asserted here: they need the real full
//! models and a stable measurement lab, which is out of scope for this WP. Only
//! the relative-regression scaffold is implemented.

/// Summary statistics over a set of per-iteration latency samples (seconds).
///
/// `p50`/`p95`/`p99` use the nearest-rank definition (1-based rank
/// `ceil(q * n)`); `stddev` is the population standard deviation and is used as
/// the jitter metric. All fields carry the same unit as the input samples.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Stats {
    /// Number of samples.
    pub(crate) count: usize,
    /// Smallest sample.
    pub(crate) min: f64,
    /// Largest sample.
    pub(crate) max: f64,
    /// Arithmetic mean.
    pub(crate) mean: f64,
    /// Population standard deviation (jitter).
    pub(crate) stddev: f64,
    /// 50th percentile (median, nearest-rank).
    pub(crate) p50: f64,
    /// 95th percentile (nearest-rank).
    pub(crate) p95: f64,
    /// 99th percentile (nearest-rank).
    pub(crate) p99: f64,
}

/// Nearest-rank percentile of an ascending, non-empty slice.
///
/// `rank = ceil(q * n)` (1-based); a tiny epsilon absorbs float rounding so an
/// exact quantile boundary (e.g. `q = 0.5` with an even `n`) does not spill into
/// the next rank. `q` is clamped to `[0, 1]` by the callers below.
fn percentile(sorted: &[f64], q: f64) -> f64 {
    debug_assert!(!sorted.is_empty(), "percentile of an empty slice");
    let n = sorted.len();
    let rank = (q * n as f64 - 1e-9).ceil();
    let rank = if rank < 1.0 { 1 } else { rank as usize };
    sorted[rank.min(n) - 1]
}

/// Computes [`Stats`] over `samples`, or `None` if `samples` is empty.
pub(crate) fn summarize(samples: &[f64]) -> Option<Stats> {
    if samples.is_empty() {
        return None;
    }
    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    let mean = sorted.iter().sum::<f64>() / n as f64;
    let var = sorted
        .iter()
        .map(|x| {
            let d = x - mean;
            d * d
        })
        .sum::<f64>()
        / n as f64;
    Some(Stats {
        count: n,
        min: sorted[0],
        max: sorted[n - 1],
        mean,
        stddev: var.sqrt(),
        p50: percentile(&sorted, 0.50),
        p95: percentile(&sorted, 0.95),
        p99: percentile(&sorted, 0.99),
    })
}

/// A finished bench measurement for one task, ready to serialize.
///
/// `ttfa_ms` is the time-to-first-audio (TTS) / time-to-first-token (ASR). For
/// the M0/M1 non-streaming synthesis path the first audio chunk *is* the whole
/// clip, so it equals the mean per-iteration latency; the streaming split is a
/// follow-up on the M1-04 / M1-08 streaming APIs.
#[derive(Debug, Clone)]
pub(crate) struct BenchReport {
    /// Task name (`vad` / `asr` / `tts`).
    pub(crate) task: String,
    /// Number of timed iterations.
    pub(crate) iters: usize,
    /// Number of warm-up iterations (untimed).
    pub(crate) warmup: usize,
    /// Duration of the input/output audio in seconds (RTF denominator).
    pub(crate) audio_seconds: f64,
    /// Real-time factor: mean compute time / audio seconds.
    pub(crate) rtf: f64,
    /// Time-to-first-audio / -token in milliseconds (whole-clip for non-streaming).
    pub(crate) ttfa_ms: f64,
    /// Per-iteration latency statistics (seconds).
    pub(crate) latency: Stats,
}

/// Replaces a non-finite value with `0.0` so the serializers never emit `NaN` /
/// `inf` (which are not valid JSON and would break baseline round-tripping).
fn finite(x: f64) -> f64 {
    if x.is_finite() { x } else { 0.0 }
}

impl BenchReport {
    /// Serializes to a single flat `key=value` line (latencies in milliseconds).
    pub(crate) fn to_kv(&self) -> String {
        let l = &self.latency;
        format!(
            "task={} iters={} warmup={} audio_s={:.6} rtf={:.6} ttfa_ms={:.4} \
             p50_ms={:.4} p95_ms={:.4} p99_ms={:.4} mean_ms={:.4} jitter_ms={:.4} \
             min_ms={:.4} max_ms={:.4}",
            self.task,
            self.iters,
            self.warmup,
            finite(self.audio_seconds),
            finite(self.rtf),
            finite(self.ttfa_ms),
            finite(l.p50 * 1e3),
            finite(l.p95 * 1e3),
            finite(l.p99 * 1e3),
            finite(l.mean * 1e3),
            finite(l.stddev * 1e3),
            finite(l.min * 1e3),
            finite(l.max * 1e3),
        )
    }

    /// Serializes to a compact JSON object (latencies in milliseconds).
    ///
    /// The top-level `rtf` field is what [`parse_baseline_rtf`] reads back, so a
    /// committed baseline is just a saved `to_json()` line.
    pub(crate) fn to_json(&self) -> String {
        let l = &self.latency;
        format!(
            "{{\"task\":\"{}\",\"iters\":{},\"warmup\":{},\"audio_seconds\":{:.6},\
             \"rtf\":{:.6},\"ttfa_ms\":{:.4},\"latency_ms\":{{\"p50\":{:.4},\"p95\":{:.4},\
             \"p99\":{:.4},\"mean\":{:.4},\"jitter\":{:.4},\"min\":{:.4},\"max\":{:.4}}}}}",
            self.task,
            self.iters,
            self.warmup,
            finite(self.audio_seconds),
            finite(self.rtf),
            finite(self.ttfa_ms),
            finite(l.p50 * 1e3),
            finite(l.p95 * 1e3),
            finite(l.p99 * 1e3),
            finite(l.mean * 1e3),
            finite(l.stddev * 1e3),
            finite(l.min * 1e3),
            finite(l.max * 1e3),
        )
    }
}

/// Result of comparing a current measurement against a baseline (M1-10b).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct RegressionCheck {
    /// Baseline metric value (lower is better for RTF/latency).
    pub(crate) baseline: f64,
    /// Current metric value.
    pub(crate) current: f64,
    /// `current / baseline` (`+inf` if the baseline is zero).
    pub(crate) ratio: f64,
    /// Relative tolerance (e.g. `0.05` for 5%).
    pub(crate) threshold: f64,
    /// `true` if `current` exceeds `baseline * (1 + threshold)`.
    pub(crate) regressed: bool,
}

/// Relative regression comparison (NFR-PF-13): for a lower-is-better metric,
/// `current` regresses when it exceeds `baseline * (1 + threshold)`.
pub(crate) fn compare(baseline: f64, current: f64, threshold: f64) -> RegressionCheck {
    let ratio = if baseline.abs() > f64::EPSILON {
        current / baseline
    } else {
        f64::INFINITY
    };
    RegressionCheck {
        baseline,
        current,
        ratio,
        threshold,
        regressed: current > baseline * (1.0 + threshold),
    }
}

/// Extracts a numeric JSON value as `f64` (accepts both integer and float forms).
fn json_number(v: &vokra_core::json::JsonValue) -> Option<f64> {
    use vokra_core::json::JsonValue::{Float, Int};
    match v {
        Float(f) => Some(*f),
        Int(i) => Some(*i as f64),
        _ => None,
    }
}

/// Parses a committed baseline JSON blob (a previous [`BenchReport::to_json`]
/// line) and returns its top-level `rtf` value.
pub(crate) fn parse_baseline_rtf(bytes: &[u8]) -> Result<f64, String> {
    let v = vokra_core::json::parse(bytes).map_err(|e| e.to_string())?;
    v.get("rtf")
        .and_then(json_number)
        .ok_or_else(|| "baseline JSON has no numeric `rtf` field".to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-6
    }

    #[test]
    fn summarize_on_one_to_hundred_has_known_percentiles() {
        let samples: Vec<f64> = (1..=100).map(f64::from).collect();
        let s = summarize(&samples).expect("non-empty");
        assert_eq!(s.count, 100);
        assert!(approx(s.min, 1.0));
        assert!(approx(s.max, 100.0));
        assert!(approx(s.mean, 50.5));
        // Nearest-rank on 1..=100.
        assert!(approx(s.p50, 50.0), "p50 = {}", s.p50);
        assert!(approx(s.p95, 95.0), "p95 = {}", s.p95);
        assert!(approx(s.p99, 99.0), "p99 = {}", s.p99);
        // Population stddev of 1..=100 = sqrt((n^2 - 1) / 12).
        let expected = ((100.0f64 * 100.0 - 1.0) / 12.0).sqrt();
        assert!((s.stddev - expected).abs() < 1e-6, "stddev = {}", s.stddev);
    }

    #[test]
    fn summarize_handles_unsorted_input_and_single_element() {
        let s = summarize(&[3.0, 1.0, 2.0]).expect("non-empty");
        assert!(approx(s.min, 1.0) && approx(s.max, 3.0) && approx(s.mean, 2.0));
        assert!(approx(s.p50, 2.0));

        let one = summarize(&[7.5]).expect("non-empty");
        assert!(approx(one.p50, 7.5) && approx(one.p95, 7.5) && approx(one.p99, 7.5));
        assert!(approx(one.stddev, 0.0));
    }

    #[test]
    fn summarize_empty_is_none() {
        assert!(summarize(&[]).is_none());
    }

    #[test]
    fn compare_flags_only_regressions_beyond_threshold() {
        // 4% slower vs a 5% tolerance: OK.
        assert!(!compare(0.100, 0.104, 0.05).regressed);
        // 6% slower: regression.
        let r = compare(0.100, 0.106, 0.05);
        assert!(r.regressed);
        assert!((r.ratio - 1.06).abs() < 1e-9);
        // Improvement (faster): never a regression.
        assert!(!compare(0.100, 0.090, 0.05).regressed);
        // Exactly on the boundary is NOT a regression (strict `>`).
        assert!(!compare(0.100, 0.105, 0.05).regressed);
    }

    #[test]
    fn compare_zero_baseline_is_infinite_ratio_and_regressed() {
        let r = compare(0.0, 0.01, 0.05);
        assert!(r.regressed);
        assert!(r.ratio.is_infinite());
    }

    #[test]
    fn json_report_round_trips_and_baseline_rtf_reads_back() {
        let stats = summarize(&[0.010, 0.020, 0.030]).expect("non-empty");
        let report = BenchReport {
            task: "tts".to_owned(),
            iters: 3,
            warmup: 1,
            audio_seconds: 1.0,
            rtf: 0.020_5,
            ttfa_ms: 20.0,
            latency: stats,
        };
        let json = report.to_json();
        // Parses through the first-party JSON reader (zero-dep).
        let v = vokra_core::json::parse(json.as_bytes()).expect("valid JSON");
        assert_eq!(v.get("task").and_then(|t| t.as_str()), Some("tts"));
        assert_eq!(v.get("iters").and_then(|t| t.as_u64()), Some(3));
        let rtf = parse_baseline_rtf(json.as_bytes()).expect("rtf reads back");
        assert!((rtf - 0.020_5).abs() < 1e-6, "rtf = {rtf}");
    }

    #[test]
    fn kv_report_contains_the_headline_keys() {
        let stats = summarize(&[0.01, 0.02]).expect("non-empty");
        let report = BenchReport {
            task: "vad".to_owned(),
            iters: 2,
            warmup: 0,
            audio_seconds: 2.0,
            rtf: 0.0075,
            ttfa_ms: 15.0,
            latency: stats,
        };
        let kv = report.to_kv();
        for key in [
            "task=vad",
            "rtf=",
            "p50_ms=",
            "p95_ms=",
            "p99_ms=",
            "jitter_ms=",
        ] {
            assert!(kv.contains(key), "missing `{key}` in `{kv}`");
        }
    }

    #[test]
    fn non_finite_values_are_serialized_as_zero() {
        let stats = summarize(&[0.01]).expect("non-empty");
        let report = BenchReport {
            task: "asr".to_owned(),
            iters: 1,
            warmup: 0,
            audio_seconds: 0.0,
            rtf: f64::NAN,
            ttfa_ms: f64::INFINITY,
            latency: stats,
        };
        // A NaN rtf must not poison the JSON: it round-trips as 0.0.
        let rtf = parse_baseline_rtf(report.to_json().as_bytes()).expect("valid JSON");
        assert_eq!(rtf, 0.0);
    }
}
