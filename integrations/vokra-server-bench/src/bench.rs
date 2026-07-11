//! Core bench runner: warm-up + measurement + concurrency.
//!
//! The runner is split into small pieces so the pure functions
//! (request-body builder, single-request timing) can be unit-tested
//! without spinning up threads, and so `tests/e2e.rs` can drive the
//! same code path a real operator will drive.

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use crate::cli::Args;
use crate::stats::{Summary, summarize};

/// One request's latency + status code.
///
/// Non-`Copy` (Duration is Copy, u16 is Copy — the struct is Copy in
/// practice — but the trait is not derived so future extension can
/// add non-Copy fields without a version bump).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Timing {
    /// Time from `Instant::now()` before `send_bytes` to `.send_bytes(...)`
    /// returning a `Response` (status + headers received; approximates
    /// TTFA on the current non-streaming vocoder — see `lib.rs` module
    /// docs "Boundary being measured").
    pub ttfa: Duration,
    /// Time from `Instant::now()` before `send_bytes` to the response
    /// body being fully drained.
    pub total: Duration,
    /// HTTP status code (uses 0 for non-transport `body drain`
    /// failures — see [`TransportOutcome::HttpButBodyDrainFailed`]).
    pub status: u16,
}

/// Result of one HTTP request.
#[derive(Debug)]
pub enum TransportOutcome {
    /// The request received a full response (status may be 2xx OR
    /// 4xx/5xx — non-2xx status codes are captured in `Timing::status`
    /// and drive the [`Summary`] counters).
    Complete(Timing),
    /// The request could not reach the server (connect refused, DNS
    /// failure, TLS handshake failure, timeout). No wall time is
    /// recorded because the number would be meaningless.
    Transport(String),
    /// The server sent status + headers but reading the body failed
    /// (connection reset mid-stream). This IS wall-time-measurable up
    /// to the drain failure but is not counted as a healthy 2xx —
    /// bucket it as a transport-adjacent error.
    HttpButBodyDrainFailed(String),
}

/// Build the JSON body sent to `POST /api/tts`.
///
/// Exposed as a pure function so tests can assert on the byte-exact
/// output (and confirm Unicode escaping).
pub fn build_request_body(text: &str, voice: &str) -> Vec<u8> {
    // Piper HTTP schema (`integrations/vokra-server/src/api/piper_http.rs`
    // L96): `text` + `voice` are required, `model` / `length_scale` /
    // `noise_scale` are optional and OMITTED here — the T11
    // per-request override gate in v0.5 rejects anything but the
    // voice's baked default, and the bench measures the default path.
    let v = serde_json::json!({
        "text": text,
        "voice": voice,
    });
    v.to_string().into_bytes()
}

/// Build a shared ureq [`Agent`](ureq::Agent) with the timeouts from
/// [`Args`].
pub fn new_agent(args: &Args) -> ureq::Agent {
    ureq::AgentBuilder::new()
        // Split connect timeout so a firewall drop is caught before
        // the full-round-trip timeout.
        .timeout_connect(Duration::from_secs(5))
        .timeout(Duration::from_secs(args.timeout_secs))
        .build()
}

/// Fire one request and return its outcome.
///
/// Every timing is `Instant::now()`-based; the `Duration` values are
/// deterministic on a warm CPU with `Instant`'s monotonic clock.
pub fn single_request(agent: &ureq::Agent, url: &str, body: &[u8]) -> TransportOutcome {
    let start = Instant::now();
    let resp = agent
        .post(url)
        .set("Content-Type", "application/json")
        .send_bytes(body);
    let ttfa = start.elapsed();

    let (status, mut reader) = match resp {
        Ok(r) => (r.status(), r.into_reader()),
        Err(ureq::Error::Status(code, r)) => (code, r.into_reader()),
        Err(ureq::Error::Transport(t)) => {
            return TransportOutcome::Transport(t.to_string());
        }
    };
    // Drain the body — this is what makes `total` a full round-trip
    // number. Buf cap at 2 MB matches the current single-utterance
    // piper WAV cap (~22 kHz * 2 bytes * a few seconds); anything
    // larger is a server bug we still want to time.
    let mut buf = Vec::with_capacity(64 * 1024);
    match std::io::Read::read_to_end(&mut reader, &mut buf) {
        Ok(_) => {}
        Err(e) => {
            return TransportOutcome::HttpButBodyDrainFailed(e.to_string());
        }
    }
    let total = start.elapsed();
    TransportOutcome::Complete(Timing {
        ttfa,
        total,
        status,
    })
}

/// Run the full bench: warmup + measurement + summary.
///
/// The measurement window is split across `args.concurrent` worker
/// threads sharing an atomic "iterations remaining" counter. Each
/// worker pulls one iteration at a time until the counter reaches
/// zero, so total measured requests == `args.iters` regardless of
/// which worker finishes first (classic ab / wrk pattern).
///
/// Warm-up is always single-threaded on the calling thread so it
/// deterministically primes the shared DNS cache + TLS ticket store
/// before the measurement window opens.
pub fn run_bench(args: &Args) -> Summary {
    let url = args.full_url();
    let body = build_request_body(&args.text, &args.voice);

    // Warmup — single-threaded, discarded.
    let warmup_agent = new_agent(args);
    for _ in 0..args.warmup {
        let _ = single_request(&warmup_agent, &url, &body);
    }

    // Measurement — `concurrent` threads sharing a counter.
    let remaining = AtomicUsize::new(args.iters);
    let timings: Mutex<Vec<Timing>> = Mutex::new(Vec::with_capacity(args.iters));
    let transport_errors = AtomicUsize::new(0);

    thread::scope(|s| {
        for _ in 0..args.concurrent {
            let remaining = &remaining;
            let timings = &timings;
            let transport_errors = &transport_errors;
            let url = &url;
            let body = &body;
            let args = &args;
            s.spawn(move || {
                let agent = new_agent(args);
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
                    match single_request(&agent, url, body) {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_body_shape() {
        let body = build_request_body("hello", "en_US-libritts-high");
        let parsed: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(parsed["text"], "hello");
        assert_eq!(parsed["voice"], "en_US-libritts-high");
        // NO `length_scale`/`noise_scale` in the default body — see
        // rationale in build_request_body doc.
        assert!(parsed.get("length_scale").is_none());
        assert!(parsed.get("noise_scale").is_none());
    }

    #[test]
    fn build_body_escapes_unicode_and_control() {
        let body = build_request_body("日本語 \"quoted\"", "ja_JP-x-y");
        let s = std::str::from_utf8(&body).unwrap();
        // serde_json escapes the `"` inside the string.
        assert!(
            s.contains(r#"\"quoted\""#),
            "body missing escaped quote: {s}"
        );
        // Unicode passes through as UTF-8 bytes.
        assert!(s.contains("日本語"), "body missing unicode: {s}");
    }

    #[test]
    fn new_agent_honours_timeout_arg() {
        // We can't directly probe the Agent's timeouts, but constructing
        // it must succeed with a plausible value (the smoke test for
        // `--timeout-secs`).
        let mut args = Args::defaults();
        args.timeout_secs = 1;
        let _agent = new_agent(&args);
    }
}
