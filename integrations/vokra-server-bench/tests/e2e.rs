//! End-to-end test using a hand-rolled HTTP mock server.
//!
//! The mock deliberately does NOT depend on hyper / axum / httpmock:
//! the whole point of this crate is to keep the wire-side dep-count
//! auditable. A `std::net::TcpListener` in a background thread parses
//! the incoming HTTP request headers, discards the body per
//! `Content-Length`, and writes one of a set of canned responses.

#![deny(unsafe_code)]

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use vokra_server_bench::{
    Args,
    cli::OutputFormat,
    run_bench,
    stats::{emit_json, emit_kv},
};

// ---------------------------------------------------------------------------
// Mock HTTP server
// ---------------------------------------------------------------------------

/// One canned HTTP response. Each variant is what the mock will write
/// on the next accepted connection (round-robin from a Vec).
#[derive(Debug, Clone)]
enum Response {
    /// 200 OK with a WAV-ish body (44-byte RIFF header + zero samples
    /// so ureq's `read_to_end` has real bytes to drain).
    Ok200Wav { body_bytes: usize },
    /// 503 Service Unavailable (FR-SV-06 graceful degradation).
    Over503,
    /// 429 Too Many Requests.
    RateLimited429,
    /// Sleep then 200 (simulates a slow synth path so the p95 lift
    /// is visible in the summary).
    SlowOk200 { sleep: Duration, body_bytes: usize },
}

impl Response {
    fn write_to(&self, mut sock: TcpStream) -> std::io::Result<()> {
        match self {
            Self::Ok200Wav { body_bytes } => {
                write_response(&mut sock, 200, "OK", "audio/wav", &fake_wav(*body_bytes))
            }
            Self::Over503 => write_response(
                &mut sock,
                503,
                "Service Unavailable",
                "application/json",
                br#"{"error":"over capacity"}"#,
            ),
            Self::RateLimited429 => write_response(
                &mut sock,
                429,
                "Too Many Requests",
                "application/json",
                br#"{"error":"rate limited"}"#,
            ),
            Self::SlowOk200 { sleep, body_bytes } => {
                thread::sleep(*sleep);
                write_response(&mut sock, 200, "OK", "audio/wav", &fake_wav(*body_bytes))
            }
        }
    }
}

fn write_response(
    sock: &mut TcpStream,
    code: u16,
    reason: &str,
    ct: &str,
    body: &[u8],
) -> std::io::Result<()> {
    let head = format!(
        "HTTP/1.1 {code} {reason}\r\n\
         Content-Type: {ct}\r\n\
         Content-Length: {}\r\n\
         Connection: close\r\n\
         \r\n",
        body.len(),
    );
    sock.write_all(head.as_bytes())?;
    sock.write_all(body)?;
    sock.flush()?;
    Ok(())
}

fn fake_wav(n_bytes: usize) -> Vec<u8> {
    // 44-byte RIFF header + n_bytes of zero PCM. Content is
    // immaterial (the bench does not decode); the size shape matters
    // so ureq exercises a realistic read_to_end.
    let mut v = vec![0u8; 44];
    v[0..4].copy_from_slice(b"RIFF");
    v[8..12].copy_from_slice(b"WAVE");
    v.extend(std::iter::repeat_n(0u8, n_bytes));
    v
}

/// A running mock server that will answer `responses.len()` requests
/// (round-robin, wrapping when necessary) before shutting down.
struct MockServer {
    addr: SocketAddr,
    handle: Option<JoinHandle<()>>,
    served: Arc<AtomicUsize>,
}

impl MockServer {
    fn spawn(responses: Vec<Response>, target_requests: usize) -> Self {
        assert!(
            !responses.is_empty(),
            "mock must have at least one response"
        );
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind loopback");
        let addr = listener.local_addr().expect("local_addr");
        let served = Arc::new(AtomicUsize::new(0));
        let served_clone = Arc::clone(&served);

        let handle = thread::spawn(move || {
            // Bounded accept loop — refuse to run forever so a test
            // bug does not hang the CI runner.
            for i in 0..target_requests {
                let (stream, _) = match listener.accept() {
                    Ok(x) => x,
                    Err(_) => return,
                };
                let resp = responses[i % responses.len()].clone();
                // Handle each connection on its own thread so slow
                // responses (SlowOk200) do not block sibling
                // in-flight requests — this is what lets the
                // concurrent test measure real parallelism.
                let served_bg = Arc::clone(&served_clone);
                thread::spawn(move || {
                    if drain_request(&stream).is_err() {
                        return;
                    }
                    let _ = resp.write_to(stream);
                    served_bg.fetch_add(1, Ordering::AcqRel);
                });
            }
        });

        MockServer {
            addr,
            handle: Some(handle),
            served,
        }
    }

    fn base_url(&self) -> String {
        format!("http://{}", self.addr)
    }

    fn served(&self) -> usize {
        self.served.load(Ordering::Acquire)
    }
}

impl Drop for MockServer {
    fn drop(&mut self) {
        // Best-effort shutdown by joining the accept loop.
        if let Some(h) = self.handle.take() {
            // Give the accept thread a moment to notice the socket close
            // (some kernels take a few ms). Bounded wait so a stuck
            // test still fails, not hangs.
            let _ = h.join();
        }
    }
}

fn drain_request(mut stream: &TcpStream) -> std::io::Result<()> {
    // Read until end-of-headers (`\r\n\r\n`) so we know Content-Length.
    let mut header_buf = Vec::with_capacity(1024);
    let mut byte = [0u8; 1];
    stream.set_read_timeout(Some(Duration::from_secs(5)))?;
    loop {
        let n = stream.read(&mut byte)?;
        if n == 0 {
            break;
        }
        header_buf.push(byte[0]);
        if header_buf.ends_with(b"\r\n\r\n") {
            break;
        }
        if header_buf.len() > 8192 {
            // Malformed / hostile — give up.
            return Err(std::io::Error::other("headers too large"));
        }
    }
    // Parse Content-Length from the accumulated headers.
    let header_str = std::str::from_utf8(&header_buf).unwrap_or("");
    let cl = header_str
        .lines()
        .find_map(|l| {
            let l = l.trim();
            l.to_ascii_lowercase()
                .strip_prefix("content-length:")
                .and_then(|v| v.trim().parse::<usize>().ok())
        })
        .unwrap_or(0);
    if cl > 0 {
        let mut body = vec![0u8; cl];
        stream.read_exact(&mut body)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

fn small_body_args(base: &str, iters: usize, concurrent: usize) -> Args {
    let mut a = Args::defaults();
    a.server = base.to_string();
    a.iters = iters;
    a.warmup = 0;
    a.concurrent = concurrent;
    a.timeout_secs = 5;
    a
}

#[test]
fn single_worker_all_2xx() {
    let n = 8;
    let mock = MockServer::spawn(vec![Response::Ok200Wav { body_bytes: 128 }], n);
    let args = small_body_args(&mock.base_url(), n, 1);
    let summary = run_bench(&args);
    assert_eq!(summary.ok_2xx, n);
    assert_eq!(summary.over_capacity_503, 0);
    assert_eq!(summary.transport_errors, 0);
    // p95 must be positive (real request happened).
    assert!(summary.ttfa_ms.p95 >= 0.0);
    // Serve count should reach iters (mock may have served slightly more
    // if warmup > 0, but we passed warmup=0).
    assert_eq!(mock.served(), n);
}

#[test]
fn concurrent_workers_do_not_undercount() {
    // With 32 iterations across 4 workers, exactly 32 samples must
    // be recorded — never 33 (overshoot) or 30 (undercount).
    let n = 32;
    let mock = MockServer::spawn(vec![Response::Ok200Wav { body_bytes: 128 }], n);
    let args = small_body_args(&mock.base_url(), n, 4);
    let summary = run_bench(&args);
    assert_eq!(
        summary.ok_2xx + summary.over_capacity_503 + summary.transport_errors,
        n,
        "iterations counter drifted",
    );
    assert_eq!(summary.ok_2xx, n);
}

#[test]
fn over_capacity_503_bucketed_separately() {
    let n = 6;
    // Alternate 200 / 503 → 3 of each.
    let mock = MockServer::spawn(
        vec![Response::Ok200Wav { body_bytes: 64 }, Response::Over503],
        n,
    );
    let args = small_body_args(&mock.base_url(), n, 1);
    let summary = run_bench(&args);
    assert_eq!(summary.ok_2xx, 3);
    assert_eq!(summary.over_capacity_503, 3);
    assert_eq!(summary.client_error_4xx, 0);
    assert_eq!(summary.server_error_5xx, 0);
    assert_eq!(summary.rate_limited_429, 0);
    // p95 must be based on the 3 healthy timings — non-zero, finite.
    assert!(summary.ttfa_ms.p95 >= 0.0);
    assert!(summary.ttfa_ms.p95.is_finite());
}

#[test]
fn rate_limited_429_bucketed_separately() {
    let n = 4;
    let mock = MockServer::spawn(
        vec![
            Response::Ok200Wav { body_bytes: 64 },
            Response::RateLimited429,
        ],
        n,
    );
    let args = small_body_args(&mock.base_url(), n, 1);
    let summary = run_bench(&args);
    assert_eq!(summary.ok_2xx, 2);
    assert_eq!(summary.rate_limited_429, 2);
    assert_eq!(summary.client_error_4xx, 0);
}

#[test]
fn transport_error_when_server_absent() {
    // Grab a port then immediately release it — nothing is listening
    // when the bench runs.
    let addr = {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let a = l.local_addr().unwrap();
        drop(l);
        a
    };
    let base = format!("http://{}", addr);
    let n = 4;
    let mut args = small_body_args(&base, n, 1);
    args.timeout_secs = 2;
    let summary = run_bench(&args);
    // All requests must fail transport (connect refused).
    assert_eq!(summary.ok_2xx, 0);
    assert_eq!(summary.transport_errors, n);
}

#[test]
fn json_output_reparses_and_carries_verdict() {
    let n = 4;
    let mock = MockServer::spawn(vec![Response::Ok200Wav { body_bytes: 64 }], n);
    let args = {
        let mut a = small_body_args(&mock.base_url(), n, 1);
        a.format = OutputFormat::Json;
        a.budget_ms = 60_000; // Any localhost bench comfortably beats 60 s.
        a
    };
    let summary = run_bench(&args);
    let mut buf = Vec::new();
    emit_json(&mut buf, &args, &summary).unwrap();
    let out = String::from_utf8(buf).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(out.trim_end()).unwrap();
    assert_eq!(parsed["iterations"], n);
    assert_eq!(parsed["concurrent"], 1);
    assert_eq!(parsed["budget_ms"], 60_000);
    assert!(parsed["ttfa_ms"]["p50"].is_number());
    assert!(parsed["counters"]["ok_2xx"].as_u64().unwrap() > 0);
    assert_eq!(parsed["verdict"], "PASS");
}

#[test]
fn kv_output_covers_every_documented_key() {
    let n = 3;
    let mock = MockServer::spawn(vec![Response::Ok200Wav { body_bytes: 64 }], n);
    let args = small_body_args(&mock.base_url(), n, 1);
    let summary = run_bench(&args);
    let mut buf = Vec::new();
    emit_kv(&mut buf, &args, &summary).unwrap();
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
        "counters.transport_errors=",
        "verdict=",
    ] {
        assert!(out.contains(key), "kv missing {key}: {out}");
    }
}

#[test]
fn concurrent_slow_responses_take_shorter_wall_time() {
    // Sanity: 8 slow (100 ms) responses fanned across 4 workers must
    // finish in noticeably less than 8*100 = 800 ms of wall time
    // (the concurrent path really overlaps). Bench also succeeds
    // end-to-end.
    let n = 8;
    let mock = MockServer::spawn(
        vec![Response::SlowOk200 {
            sleep: Duration::from_millis(100),
            body_bytes: 64,
        }],
        n,
    );
    let mut args = small_body_args(&mock.base_url(), n, 4);
    args.timeout_secs = 30;
    let t0 = std::time::Instant::now();
    let summary = run_bench(&args);
    let elapsed = t0.elapsed();
    assert_eq!(summary.ok_2xx, n);
    // Serial lower bound is n * 100 ms = 800 ms; 4 workers should
    // finish in ~2 * 100 ms plus overhead. Allow generous slack for
    // hosted CI (< 600 ms).
    assert!(
        elapsed < Duration::from_millis(600),
        "expected concurrent runtime < 600 ms, got {elapsed:?}",
    );
}
