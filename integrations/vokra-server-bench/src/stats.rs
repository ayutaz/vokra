//! Percentile computation + output formatting.
//!
//! Split out from `bench.rs` so unit tests can exercise it without
//! binding a TCP socket, and so `main.rs` can re-emit the summary in
//! either KV or JSON without re-doing the sort.

use std::io::{self, Write};
use std::time::Duration;

use crate::bench::Timing;
use crate::cli::Args;

/// Aggregated result of one measurement window. All time-typed fields
/// are milliseconds with fractional precision (three decimal places
/// after the `emit_*` layer — computed with f64 so a sub-µs mock
/// server's samples do not round to `0.0`).
#[derive(Debug, Clone, PartialEq)]
pub struct Summary {
    /// TTFA percentiles (ms). See [`crate::bench::Timing::ttfa`].
    pub ttfa_ms: PercentileBundle,
    /// Total round-trip (ms). See [`crate::bench::Timing::total`].
    pub total_ms: PercentileBundle,
    /// Success (2xx) count.
    pub ok_2xx: usize,
    /// FR-SV-06 graceful-degradation 503 count. Kept separate from
    /// `server_error_5xx` so the operator can distinguish "expected
    /// over-capacity" from "server bug".
    pub over_capacity_503: usize,
    /// 4xx (excluding 429; see below) count. Usually indicates a
    /// bench mis-configuration (bad voice tag, unknown model).
    pub client_error_4xx: usize,
    /// 429 (Too Many Requests) — treated as an extra graceful-
    /// degradation bucket alongside 503 so operators using a rate-
    /// limited fronting proxy can distinguish it.
    pub rate_limited_429: usize,
    /// 5xx (excluding 503).
    pub server_error_5xx: usize,
    /// Requests that never reached the server (connect refused, DNS,
    /// TLS handshake failure, timeout). No timing recorded.
    pub transport_errors: usize,
}

/// Median + tail percentiles for one timing series, in milliseconds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PercentileBundle {
    /// 50th percentile.
    pub p50: f64,
    /// 95th percentile.
    pub p95: f64,
    /// 99th percentile.
    pub p99: f64,
    /// Median (identical to p50 for iid samples; kept separate to
    /// match the handover § 4 JSON schema byte-for-byte).
    pub median: f64,
    /// Worst-case sample.
    pub max: f64,
}

impl PercentileBundle {
    /// Empty bundle (all fields `0.0`) used when there are no
    /// successful timings — the JSON schema still emits the object
    /// so downstream parsers do not need to branch on presence.
    pub const EMPTY: Self = Self {
        p50: 0.0,
        p95: 0.0,
        p99: 0.0,
        median: 0.0,
        max: 0.0,
    };
}

/// Nearest-rank (`interpolation='lower'`) percentile picker.
///
/// Numpy's `interpolation='lower'` is the safest choice for small N:
/// it always returns a real observed sample rather than an
/// interpolated value that could sit outside the sample range under
/// FP rounding. The `sorted` slice MUST be pre-sorted in ascending
/// order; callers of [`summarize`] use `Vec::sort_by`.
///
/// `p` is a probability in `[0.0, 1.0]`. Values outside that range
/// are clamped.
///
/// # Panics
///
/// Panics if `sorted.is_empty()` — callers should check first and
/// emit [`PercentileBundle::EMPTY`] in that case.
pub fn percentile_nearest_rank(sorted: &[Duration], p: f64) -> Duration {
    assert!(!sorted.is_empty(), "percentile called on empty slice");
    let p = p.clamp(0.0, 1.0);
    // Nearest-rank: index = ceil(p * N) - 1, clamped to [0, N-1].
    // At p=0 we want index 0, at p=1 we want N-1. For strictly
    // interior probabilities the ceiling matches numpy's 'lower'
    // convention *after* offsetting for zero-based indexing.
    let n = sorted.len();
    let idx = if p <= 0.0 {
        0
    } else {
        let raw = (p * n as f64).ceil() as usize;
        raw.saturating_sub(1).min(n - 1)
    };
    sorted[idx]
}

/// Convert a `Duration` to fractional milliseconds.
///
/// Uses `as_nanos()` (u128) then f64 division so a Duration in the
/// microsecond range (localhost mock server) does not round to
/// `0.0` ms.
pub fn duration_to_ms(d: Duration) -> f64 {
    (d.as_nanos() as f64) / 1_000_000.0
}

/// Compute one [`PercentileBundle`] from a set of timings picked by
/// the `field` closure.
fn bundle_of<F: Fn(&Timing) -> Duration>(timings: &[Timing], field: F) -> PercentileBundle {
    if timings.is_empty() {
        return PercentileBundle::EMPTY;
    }
    let mut xs: Vec<Duration> = timings.iter().map(field).collect();
    xs.sort_unstable();
    let median = percentile_nearest_rank(&xs, 0.5);
    PercentileBundle {
        p50: duration_to_ms(median),
        p95: duration_to_ms(percentile_nearest_rank(&xs, 0.95)),
        p99: duration_to_ms(percentile_nearest_rank(&xs, 0.99)),
        median: duration_to_ms(median),
        max: duration_to_ms(*xs.last().expect("non-empty checked above")),
    }
}

/// Summarise a set of timings into a [`Summary`].
///
/// `transport_errors` is passed in because a transport failure has no
/// meaningful timing (the request never reached the server), so those
/// samples are counted only, not measured.
pub fn summarize(timings: &[Timing], transport_errors: usize) -> Summary {
    // Successful timings (2xx) drive the percentile buckets. Non-2xx
    // *do* have a wire round-trip time, but mixing 503 tails into the
    // p95 would silently make an over-capacity smoke look worse than
    // the healthy path — the operator wants BOTH numbers, so we
    // compute the percentiles over 2xx only and expose the counters.
    let ok: Vec<&Timing> = timings
        .iter()
        .filter(|t| (200..300).contains(&t.status))
        .collect();
    let ok_owned: Vec<Timing> = ok.iter().map(|&t| t.clone()).collect();

    let ttfa = bundle_of(&ok_owned, |t| t.ttfa);
    let total = bundle_of(&ok_owned, |t| t.total);

    let mut ok_2xx = 0;
    let mut over_capacity_503 = 0;
    let mut client_error_4xx = 0;
    let mut rate_limited_429 = 0;
    let mut server_error_5xx = 0;
    for t in timings {
        match t.status {
            200..=299 => ok_2xx += 1,
            429 => rate_limited_429 += 1,
            400..=499 => client_error_4xx += 1,
            503 => over_capacity_503 += 1,
            500..=599 => server_error_5xx += 1,
            _ => {
                // Non-standard status codes get bucketed under 5xx to
                // keep the schema stable — FR-EX-08 no silent drop.
                server_error_5xx += 1;
            }
        }
    }

    Summary {
        ttfa_ms: ttfa,
        total_ms: total,
        ok_2xx,
        over_capacity_503,
        client_error_4xx,
        rate_limited_429,
        server_error_5xx,
        transport_errors,
    }
}

/// Verdict string emitted alongside the artifact. Informational only
/// — the process exits `0` in both PASS and FAIL cases (see
/// `crate::lib.rs` module docs "What the tool decides").
pub fn verdict(summary: &Summary, budget_ms: u64) -> &'static str {
    if summary.ok_2xx == 0 {
        return "NO_SUCCESS";
    }
    if budget_ms == 0 {
        return "NO_BUDGET";
    }
    if summary.ttfa_ms.p50 <= budget_ms as f64 {
        "PASS"
    } else {
        "FAIL"
    }
}

/// Emit the summary in KV (`key=value`) form, one field per line.
///
/// Grep-friendly for the M3-15 results-report template. The key set
/// is the flattened JSON schema (dot-separated).
pub fn emit_kv<W: Write>(w: &mut W, args: &Args, summary: &Summary) -> io::Result<()> {
    let url = args.full_url();
    writeln!(w, "endpoint={url}")?;
    writeln!(w, "utterance={}", args.text)?;
    writeln!(w, "voice={}", args.voice)?;
    writeln!(w, "iterations={}", args.iters)?;
    writeln!(w, "warmup={}", args.warmup)?;
    writeln!(w, "concurrent={}", args.concurrent)?;
    writeln!(w, "budget_ms={}", args.budget_ms)?;
    writeln!(w, "ttfa_ms.p50={:.3}", summary.ttfa_ms.p50)?;
    writeln!(w, "ttfa_ms.p95={:.3}", summary.ttfa_ms.p95)?;
    writeln!(w, "ttfa_ms.p99={:.3}", summary.ttfa_ms.p99)?;
    writeln!(w, "ttfa_ms.median={:.3}", summary.ttfa_ms.median)?;
    writeln!(w, "ttfa_ms.max={:.3}", summary.ttfa_ms.max)?;
    writeln!(w, "total_ms.p50={:.3}", summary.total_ms.p50)?;
    writeln!(w, "total_ms.p95={:.3}", summary.total_ms.p95)?;
    writeln!(w, "total_ms.p99={:.3}", summary.total_ms.p99)?;
    writeln!(w, "total_ms.median={:.3}", summary.total_ms.median)?;
    writeln!(w, "total_ms.max={:.3}", summary.total_ms.max)?;
    writeln!(w, "counters.ok_2xx={}", summary.ok_2xx)?;
    writeln!(
        w,
        "counters.over_capacity_503={}",
        summary.over_capacity_503
    )?;
    writeln!(w, "counters.rate_limited_429={}", summary.rate_limited_429)?;
    writeln!(w, "counters.client_error_4xx={}", summary.client_error_4xx)?;
    writeln!(w, "counters.server_error_5xx={}", summary.server_error_5xx)?;
    writeln!(w, "counters.transport_errors={}", summary.transport_errors)?;
    writeln!(w, "verdict={}", verdict(summary, args.budget_ms))?;
    Ok(())
}

/// Emit the summary as a single-line JSON blob matching
/// `docs/m3-15-server-latency-handover.md` § 4 byte-for-byte.
pub fn emit_json<W: Write>(w: &mut W, args: &Args, summary: &Summary) -> io::Result<()> {
    // Use `serde_json::json!` so `--text` / `--voice` UTF-8 is escaped
    // correctly (a raw `format!` would break on a `"` in the utterance).
    let v = serde_json::json!({
        "endpoint": args.full_url(),
        "utterance": args.text,
        "voice": args.voice,
        "iterations": args.iters,
        "warmup": args.warmup,
        "concurrent": args.concurrent,
        "ttfa_ms": {
            "p50": round3(summary.ttfa_ms.p50),
            "p95": round3(summary.ttfa_ms.p95),
            "p99": round3(summary.ttfa_ms.p99),
            "median": round3(summary.ttfa_ms.median),
            "max": round3(summary.ttfa_ms.max),
        },
        "total_ms": {
            "p50": round3(summary.total_ms.p50),
            "p95": round3(summary.total_ms.p95),
            "p99": round3(summary.total_ms.p99),
            "median": round3(summary.total_ms.median),
            "max": round3(summary.total_ms.max),
        },
        "counters": {
            "ok_2xx": summary.ok_2xx,
            "over_capacity_503": summary.over_capacity_503,
            "rate_limited_429": summary.rate_limited_429,
            "client_error_4xx": summary.client_error_4xx,
            "server_error_5xx": summary.server_error_5xx,
            "transport_errors": summary.transport_errors,
        },
        "budget_ms": args.budget_ms,
        "verdict": verdict(summary, args.budget_ms),
    });
    writeln!(w, "{}", v)
}

fn round3(ms: f64) -> f64 {
    (ms * 1000.0).round() / 1000.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bench::Timing;

    fn t(ttfa_us: u64, total_us: u64, status: u16) -> Timing {
        Timing {
            ttfa: Duration::from_micros(ttfa_us),
            total: Duration::from_micros(total_us),
            status,
        }
    }

    #[test]
    fn percentile_p50_of_odd_series() {
        let mut xs: Vec<Duration> = [10, 20, 30, 40, 50]
            .iter()
            .map(|&us| Duration::from_micros(us))
            .collect();
        xs.sort();
        // Nearest-rank at p=0.5 on N=5 → idx = ceil(0.5*5) - 1 = 2 → 30 µs.
        assert_eq!(percentile_nearest_rank(&xs, 0.5), Duration::from_micros(30));
    }

    #[test]
    fn percentile_p95_on_100() {
        let xs: Vec<Duration> = (1..=100).map(|i| Duration::from_micros(i as u64)).collect();
        // Nearest-rank at p=0.95 on N=100 → idx = ceil(95) - 1 = 94 → 95 µs.
        assert_eq!(
            percentile_nearest_rank(&xs, 0.95),
            Duration::from_micros(95)
        );
    }

    #[test]
    fn percentile_p99_on_100() {
        let xs: Vec<Duration> = (1..=100).map(|i| Duration::from_micros(i as u64)).collect();
        // idx = ceil(99) - 1 = 98 → 99 µs.
        assert_eq!(
            percentile_nearest_rank(&xs, 0.99),
            Duration::from_micros(99)
        );
    }

    #[test]
    fn percentile_p0_and_p100_clamp() {
        let xs: Vec<Duration> = (1..=5).map(|i| Duration::from_micros(i as u64)).collect();
        assert_eq!(percentile_nearest_rank(&xs, 0.0), Duration::from_micros(1));
        assert_eq!(percentile_nearest_rank(&xs, 1.0), Duration::from_micros(5));
        // Out-of-range clamps.
        assert_eq!(percentile_nearest_rank(&xs, -1.0), Duration::from_micros(1));
        assert_eq!(percentile_nearest_rank(&xs, 2.0), Duration::from_micros(5));
    }

    #[test]
    fn duration_to_ms_preserves_sub_ms() {
        assert!((duration_to_ms(Duration::from_micros(500)) - 0.5).abs() < 1e-9);
        assert!((duration_to_ms(Duration::from_millis(42)) - 42.0).abs() < 1e-9);
    }

    #[test]
    fn summarize_computes_only_over_2xx() {
        // Mix of 2xx / 503 / 4xx; percentiles must ignore the non-2xx
        // so the FR-SV-06 over-capacity tail does not contaminate the
        // healthy p95.
        let timings = vec![
            t(1_000, 1_100, 200),
            t(2_000, 2_100, 200),
            t(3_000, 3_100, 200),
            t(999_000, 999_100, 503), // huge tail, must be excluded from p95
            t(500_000, 500_100, 400),
        ];
        let s = summarize(&timings, 2);
        assert_eq!(s.ok_2xx, 3);
        assert_eq!(s.over_capacity_503, 1);
        assert_eq!(s.client_error_4xx, 1);
        assert_eq!(s.rate_limited_429, 0);
        assert_eq!(s.transport_errors, 2);
        // p95 of 3 samples should equal max(200 sample) not the 503.
        assert!(s.ttfa_ms.p95 < 100.0);
        assert!(s.ttfa_ms.p95 > 0.0);
    }

    #[test]
    fn summarize_empty_2xx_yields_zero_bundle() {
        let timings = vec![t(999_000, 999_100, 503)];
        let s = summarize(&timings, 0);
        assert_eq!(s.ttfa_ms, PercentileBundle::EMPTY);
        assert_eq!(s.total_ms, PercentileBundle::EMPTY);
        assert_eq!(s.ok_2xx, 0);
        assert_eq!(s.over_capacity_503, 1);
    }

    #[test]
    fn verdict_pass_and_fail_and_no_success() {
        let empty = Summary {
            ttfa_ms: PercentileBundle::EMPTY,
            total_ms: PercentileBundle::EMPTY,
            ok_2xx: 0,
            over_capacity_503: 0,
            client_error_4xx: 0,
            rate_limited_429: 0,
            server_error_5xx: 0,
            transport_errors: 0,
        };
        assert_eq!(verdict(&empty, 75), "NO_SUCCESS");

        let mut ok = empty.clone();
        ok.ok_2xx = 10;
        ok.ttfa_ms.p50 = 40.0;
        assert_eq!(verdict(&ok, 75), "PASS");

        let mut over = ok.clone();
        over.ttfa_ms.p50 = 120.0;
        assert_eq!(verdict(&over, 75), "FAIL");

        assert_eq!(verdict(&ok, 0), "NO_BUDGET");
    }

    #[test]
    fn emit_json_matches_handover_schema_shape() {
        // Guard-rail: the top-level keys the M3-15 handover § 4 blob
        // documents must be present in this exact spelling.
        let args = Args::defaults();
        let timings = vec![t(50_000, 51_000, 200), t(60_000, 62_000, 200)];
        let s = summarize(&timings, 0);
        let mut buf = Vec::new();
        emit_json(&mut buf, &args, &s).unwrap();
        let out = String::from_utf8(buf).unwrap();
        for key in [
            "\"endpoint\"",
            "\"utterance\"",
            "\"iterations\"",
            "\"warmup\"",
            "\"concurrent\"",
            "\"ttfa_ms\"",
            "\"budget_ms\"",
            "\"p50\"",
            "\"p95\"",
            "\"p99\"",
            "\"median\"",
            "\"max\"",
        ] {
            assert!(out.contains(key), "JSON missing {key}\n{out}");
        }
        // The output must be one line (trailing newline only).
        assert_eq!(out.matches('\n').count(), 1);
    }

    #[test]
    fn emit_json_escapes_unicode_text() {
        // `"` and multibyte should survive round-trip.
        let mut args = Args::defaults();
        args.text = String::from("こんにちは \"quoted\"");
        let s = summarize(&[t(1_000, 1_100, 200)], 0);
        let mut buf = Vec::new();
        emit_json(&mut buf, &args, &s).unwrap();
        let out = String::from_utf8(buf).unwrap();
        // serde_json's escape for `"` inside a string.
        assert!(
            out.contains(r#"\"quoted\""#),
            "no `\\\"quoted\\\"` in {out}"
        );
        // JSON must remain a single line (schema stability).
        assert_eq!(out.matches('\n').count(), 1);
        // Confirm reparse round-trips the text field exactly.
        let parsed: serde_json::Value = serde_json::from_str(out.trim_end()).unwrap();
        assert_eq!(parsed["utterance"], "こんにちは \"quoted\"");
    }

    #[test]
    fn emit_kv_covers_every_field() {
        let args = Args::defaults();
        let s = summarize(&[t(10_000, 12_000, 200), t(20_000, 21_000, 503)], 1);
        let mut buf = Vec::new();
        emit_kv(&mut buf, &args, &s).unwrap();
        let out = String::from_utf8(buf).unwrap();
        for key in [
            "endpoint=",
            "utterance=",
            "voice=",
            "iterations=",
            "warmup=",
            "concurrent=",
            "budget_ms=",
            "ttfa_ms.p50=",
            "ttfa_ms.p95=",
            "ttfa_ms.p99=",
            "ttfa_ms.median=",
            "ttfa_ms.max=",
            "total_ms.p50=",
            "counters.ok_2xx=",
            "counters.over_capacity_503=",
            "counters.rate_limited_429=",
            "counters.client_error_4xx=",
            "counters.server_error_5xx=",
            "counters.transport_errors=",
            "verdict=",
        ] {
            assert!(out.contains(key), "KV missing {key}\n{out}");
        }
    }

    #[test]
    fn round3_stays_within_a_micro() {
        assert!((round3(1.234_5) - 1.235).abs() < 1e-9);
        assert!((round3(1.234_4) - 1.234).abs() < 1e-9);
    }
}
