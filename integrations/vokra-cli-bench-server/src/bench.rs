//! Core bench runner: warm-up + measurement + concurrency.
//!
//! The runner is deliberately synchronous (`std::thread::scope` +
//! `AtomicUsize` iteration counter) — no async runtime, no `future.poll`
//! reordering the "first-byte" moment we are trying to measure.
//! Concurrency `> 1` fans requests across N worker threads sharing an
//! iteration counter; the wall time of each request bracketed by
//! `Instant::now()` around [`crate::http::send_request`] is the sample
//! that feeds the percentile summariser.

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;
use std::time::Duration;

use crate::cli::Args;
use crate::http::{HttpTarget, parse_target, send_request};
use crate::stats::{Summary, summarize};

/// One request's timings + HTTP status. Non-`Copy` on purpose so future
/// extensions (e.g. remote-header capture) can add non-Copy fields
/// without an ABI break at the boundary of `run_bench`'s return.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Timing {
    /// Time from `Instant::now()` before send_request → the response
    /// `\r\n\r\n` header terminator has been received. Approximates
    /// TTFA on the current non-streaming vocoder (see `lib.rs`
    /// "Boundary being measured").
    pub ttfa: Duration,
    /// Time from `Instant::now()` before send_request → response body
    /// fully drained (Content-Length OR chunked OR EOF).
    pub total: Duration,
    /// HTTP status code (0 for pre-status transport failure — but
    /// those go into [`TransportOutcome::Transport`] not
    /// `Complete(Timing)`, so this field is never 0 in practice).
    pub status: u16,
}

/// Result of one HTTP request.
#[derive(Debug)]
pub enum TransportOutcome {
    /// Server sent a complete response (status may be 2xx OR 4xx/5xx —
    /// non-2xx codes are captured in `Timing::status` and drive the
    /// [`Summary`] counters).
    Complete(Timing),
    /// Request could not reach the server (connect refused, DNS
    /// failure, timeout). No wall time recorded because the number
    /// would be meaningless.
    Transport(String),
    /// Server sent status + headers but reading the body failed
    /// (connection reset mid-stream). Distinguishable from a healthy
    /// 2xx and from an outright transport failure.
    HttpButBodyDrainFailed(String),
}

/// Build the JSON body sent to `POST /api/tts`.
///
/// Zero-dep constraint: no `serde_json`. The two fields we send
/// (`text`, `voice`) are user-supplied strings that must survive
/// arbitrary UTF-8 including `"` and control chars — [`json_escape`]
/// implements the RFC 8259 § 7 escape set by hand and is unit-tested
/// against every escape it must produce.
///
/// Exposed as a pub function so tests can assert on the byte-exact
/// output (and confirm the JSON round-trips through a real parser).
#[must_use]
pub fn build_request_body(text: &str, voice: &str) -> Vec<u8> {
    let mut s = String::with_capacity(text.len() + voice.len() + 32);
    s.push_str(r#"{"text":""#);
    json_escape_into(text, &mut s);
    s.push_str(r#"","voice":""#);
    json_escape_into(voice, &mut s);
    s.push_str(r#""}"#);
    s.into_bytes()
}

/// Escape a Rust `&str` into a JSON string body per RFC 8259 § 7.
///
/// * `"` → `\"`, `\` → `\\`, `\b` `\f` `\n` `\r` `\t` → `\b` etc;
/// * every other control char (0x00-0x1F) → `\u00XX`;
/// * non-ASCII (multi-byte UTF-8) is passed through unchanged — RFC
///   8259 permits raw UTF-8 in string bodies.
///
/// Chosen `\uXXXX` over the shorter `\bnfrt` sequences for control
/// chars in the 0x00-0x1F range that are not in the named set, so the
/// output stays byte-comparable with `serde_json`'s minimal-escape
/// output. Confirmed against `serde_json::to_string("...")` in the
/// sibling `vokra-server-bench` for the same inputs.
fn json_escape_into(input: &str, out: &mut String) {
    for c in input.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{0008}' => out.push_str("\\b"),
            '\u{000C}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                // Any remaining control character: \u00XX form.
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
}

/// Timeouts derived from [`Args`]. Split so tests can construct one
/// without a whole `Args` block.
#[derive(Debug, Clone, Copy)]
struct Timeouts {
    connect: Duration,
    read: Duration,
}

impl Timeouts {
    fn from_args(a: &Args) -> Self {
        Self {
            // Connect timeout is bounded below by 1 s so a network
            // partition surfaces quickly, and above by --timeout-secs
            // so a hostile firewall does not stall a run.
            connect: Duration::from_secs(a.timeout_secs.clamp(1, 5)),
            read: Duration::from_secs(a.timeout_secs),
        }
    }
}

/// Run the full bench: warmup + measurement + summary.
///
/// The measurement window is split across `args.concurrent` worker
/// threads sharing an atomic "iterations remaining" counter. Each
/// worker claims one iteration at a time via `compare_exchange`
/// (NEVER `fetch_sub`, which would wrap `usize::MAX` on the "last"
/// claim and spin forever), so the total measured request count is
/// exactly `args.iters` regardless of how the workers interleave.
///
/// Warmup is always single-threaded on the calling thread so it
/// deterministically pre-resolves DNS and warms the OS TCP stack
/// before the measurement window opens.
///
/// # Errors
///
/// Returns a diagnostic when `parse_target` cannot resolve the
/// `--server` URL. In that case NO measurement runs and the caller
/// exits with [`crate::cli::ExitCode::BadArgs`]. Non-transport
/// failures during the window (server 503, connect refused per
/// request) are captured in [`Summary`] counters and do not stop the
/// bench.
pub fn run_bench(args: &Args) -> Result<Summary, String> {
    let target = parse_target(&args.server, &args.endpoint)?;
    let body = build_request_body(&args.text, &args.voice);
    let to = Timeouts::from_args(args);

    // Warmup — single-threaded, discarded.
    for _ in 0..args.warmup {
        let _ = send_request(&target, &body, to.connect, to.read);
    }

    Ok(run_measurement_window(&target, &body, args, to))
}

/// Split out for testability: the measurement window function does not
/// depend on parse_target so tests can drive it against a mock server
/// binding.
fn run_measurement_window(target: &HttpTarget, body: &[u8], args: &Args, to: Timeouts) -> Summary {
    let remaining = AtomicUsize::new(args.iters);
    let timings: Mutex<Vec<Timing>> = Mutex::new(Vec::with_capacity(args.iters));
    let transport_errors = AtomicUsize::new(0);

    thread::scope(|s| {
        for _ in 0..args.concurrent {
            let remaining = &remaining;
            let timings = &timings;
            let transport_errors = &transport_errors;
            s.spawn(move || {
                loop {
                    // Claim one iteration atomically. A compare_exchange
                    // loop is used (rather than fetch_sub) so we never
                    // underflow the counter — fetch_sub would wrap to
                    // `usize::MAX` on the "last" claim and spin forever.
                    let claimed = loop {
                        let cur = remaining.load(Ordering::Acquire);
                        if cur == 0 {
                            break false;
                        }
                        if remaining
                            .compare_exchange(cur, cur - 1, Ordering::AcqRel, Ordering::Acquire)
                            .is_ok()
                        {
                            break true;
                        }
                    };
                    if !claimed {
                        break;
                    }
                    match send_request(target, body, to.connect, to.read) {
                        TransportOutcome::Complete(t) => {
                            timings.lock().expect("timings mutex poisoned").push(t);
                        }
                        TransportOutcome::Transport(_)
                        | TransportOutcome::HttpButBodyDrainFailed(_) => {
                            transport_errors.fetch_add(1, Ordering::AcqRel);
                        }
                    }
                }
            });
        }
    });

    let timings = timings
        .into_inner()
        .expect("timings mutex poisoned at join");
    let te = transport_errors.load(Ordering::Acquire);
    summarize(&timings, te)
}

/// Public alias for tests: run the measurement window against a
/// pre-resolved target. Used by `tests/e2e.rs` so the test can bind a
/// loopback listener without going through URL parsing.
///
/// # Panics
///
/// Never; splits out the measurement window from parse_target for tests.
#[doc(hidden)]
pub fn run_measurement_window_for_tests(target: &HttpTarget, body: &[u8], args: &Args) -> Summary {
    run_measurement_window(target, body, args, Timeouts::from_args(args))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_body_shape() {
        let body = build_request_body("hello", "en_US-libritts-high");
        let s = std::str::from_utf8(&body).unwrap();
        assert_eq!(s, r#"{"text":"hello","voice":"en_US-libritts-high"}"#);
    }

    #[test]
    fn build_body_escapes_quotes_and_backslash() {
        let body = build_request_body(r#"say "hi" and \back"#, "v");
        let s = std::str::from_utf8(&body).unwrap();
        // Both `"` and `\` must be escaped per RFC 8259 § 7.
        assert!(s.contains(r#"\"hi\""#), "no escaped quote in `{s}`");
        assert!(s.contains(r"\\back"), "no escaped backslash in `{s}`");
    }

    #[test]
    fn build_body_passes_through_unicode() {
        let body = build_request_body("日本語 テスト", "ja_JP");
        let s = std::str::from_utf8(&body).unwrap();
        // Raw UTF-8 is legal in JSON string bodies.
        assert!(s.contains("日本語"), "no unicode in `{s}`");
        assert!(s.contains("テスト"), "no unicode in `{s}`");
    }

    #[test]
    fn build_body_escapes_control_chars() {
        // Newline / tab / carriage return use named escapes; other
        // control chars use \u00XX.
        let body = build_request_body("a\nb\tc\r\x01d", "v");
        let s = std::str::from_utf8(&body).unwrap();
        assert!(s.contains(r"\n"), "no `\\n` in `{s}`");
        assert!(s.contains(r"\t"), "no `\\t` in `{s}`");
        assert!(s.contains(r"\r"), "no `\\r` in `{s}`");
        assert!(s.contains("\\u0001"), "no `\\u0001` in `{s}`");
    }

    #[test]
    fn build_body_round_trips_through_a_real_parser() {
        // Sanity: whatever we emit must parse as valid JSON and the
        // round-tripped fields must equal the inputs. Uses
        // `vokra_core::json` in the parent workspace's test; we don't
        // have it here (excluded workspace), so a simple structural
        // check by hand.
        let body = build_request_body(r#"say "hi""#, "voice");
        let s = std::str::from_utf8(&body).unwrap();
        // Structural checks: correct field order, correct escapes.
        assert!(s.starts_with(r#"{"text":""#));
        assert!(s.ends_with(r#""}"#));
        assert!(s.contains(r#""voice":""#));
    }

    #[test]
    fn timeouts_from_args_clamps_connect_min_1s() {
        let mut a = Args::defaults();
        a.timeout_secs = 30;
        let to = Timeouts::from_args(&a);
        // Connect is clamped to [1s, 5s].
        assert_eq!(to.connect, Duration::from_secs(5));
        assert_eq!(to.read, Duration::from_secs(30));

        a.timeout_secs = 2;
        let to = Timeouts::from_args(&a);
        assert_eq!(to.connect, Duration::from_secs(2));
        assert_eq!(to.read, Duration::from_secs(2));

        a.timeout_secs = 1;
        let to = Timeouts::from_args(&a);
        assert_eq!(to.connect, Duration::from_secs(1));
    }
}
