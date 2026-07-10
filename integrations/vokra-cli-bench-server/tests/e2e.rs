//! End-to-end test with a hand-rolled HTTP mock server.
//!
//! The mock deliberately depends on ONLY `std::net::TcpListener` +
//! `std::thread` — the whole point of this crate is to keep the wire-
//! side dep count auditable, so the test double must not sneak in an
//! HTTP crate. The mock parses the incoming request headers, discards
//! the body per `Content-Length`, and writes one of a set of canned
//! responses.

#![deny(unsafe_code)]

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use vokra_cli_bench_server::{
    Args, HttpTarget,
    bench::{build_request_body, run_measurement_window_for_tests},
    cli::OutputFormat,
    http::{parse_target, send_request},
    run_bench,
    stats::{emit_json, emit_kv},
};

// ---------------------------------------------------------------------------
// Mock HTTP server
// ---------------------------------------------------------------------------

/// One canned HTTP response. Each variant is what the mock writes on
/// the next accepted connection (round-robin from a Vec).
#[derive(Debug, Clone)]
enum Response {
    /// 200 OK with a WAV-ish body (44-byte RIFF header + zero samples
    /// so the client's `read_to_end` has real bytes to drain).
    Ok200Wav { body_bytes: usize },
    /// 503 Service Unavailable (FR-SV-06 graceful degradation).
    Over503,
    /// 429 Too Many Requests.
    RateLimited429,
    /// Sleep then 200 (simulates a slow synth path so the p95 lift is
    /// visible in the summary).
    SlowOk200 { sleep: Duration, body_bytes: usize },
    /// 200 with `Transfer-Encoding: chunked` (exercises the chunked
    /// decoder in `http::drain_body`).
    Ok200Chunked { chunks: Vec<Vec<u8>> },
    /// 200 with NEITHER `Content-Length` NOR `Transfer-Encoding` —
    /// body ends at EOF because we close the socket. Exercises the
    /// read-to-EOF branch.
    Ok200EofFramed { body: Vec<u8> },
}

impl Response {
    fn write_to(&self, mut sock: TcpStream) -> std::io::Result<()> {
        match self {
            Self::Ok200Wav { body_bytes } => write_response(
                &mut sock,
                200,
                "OK",
                Some("audio/wav"),
                Some(&fake_wav(*body_bytes)),
                None,
            ),
            Self::Over503 => write_response(
                &mut sock,
                503,
                "Service Unavailable",
                Some("application/json"),
                Some(br#"{"error":"over capacity"}"#),
                None,
            ),
            Self::RateLimited429 => write_response(
                &mut sock,
                429,
                "Too Many Requests",
                Some("application/json"),
                Some(br#"{"error":"rate limited"}"#),
                None,
            ),
            Self::SlowOk200 { sleep, body_bytes } => {
                thread::sleep(*sleep);
                write_response(
                    &mut sock,
                    200,
                    "OK",
                    Some("audio/wav"),
                    Some(&fake_wav(*body_bytes)),
                    None,
                )
            }
            Self::Ok200Chunked { chunks } => {
                let head = "HTTP/1.1 200 OK\r\n\
                            Content-Type: audio/wav\r\n\
                            Transfer-Encoding: chunked\r\n\
                            Connection: close\r\n\
                            \r\n";
                sock.write_all(head.as_bytes())?;
                for c in chunks {
                    let sz = format!("{:x}\r\n", c.len());
                    sock.write_all(sz.as_bytes())?;
                    sock.write_all(c)?;
                    sock.write_all(b"\r\n")?;
                }
                sock.write_all(b"0\r\n\r\n")?;
                sock.flush()?;
                Ok(())
            }
            Self::Ok200EofFramed { body } => write_response(
                &mut sock,
                200,
                "OK",
                Some("audio/wav"),
                Some(body),
                Some("skip_content_length"),
            ),
        }
    }
}

/// Write an HTTP/1.1 response with an optional Content-Length. Setting
/// `skip_cl_hint` = "skip_content_length" omits the CL header and
/// exercises the client's EOF-framed body branch.
fn write_response(
    sock: &mut TcpStream,
    code: u16,
    reason: &str,
    ct: Option<&str>,
    body: Option<&[u8]>,
    skip_cl_hint: Option<&str>,
) -> std::io::Result<()> {
    let mut head = format!("HTTP/1.1 {code} {reason}\r\n");
    if let Some(ct) = ct {
        head.push_str(&format!("Content-Type: {ct}\r\n"));
    }
    let body = body.unwrap_or(&[]);
    if skip_cl_hint != Some("skip_content_length") {
        head.push_str(&format!("Content-Length: {}\r\n", body.len()));
    }
    head.push_str("Connection: close\r\n\r\n");
    sock.write_all(head.as_bytes())?;
    sock.write_all(body)?;
    sock.flush()?;
    Ok(())
}

fn fake_wav(n_bytes: usize) -> Vec<u8> {
    // 44-byte RIFF header + n_bytes of zero PCM. Content is immaterial
    // (the bench does not decode); the size shape matters so the
    // client's read exercises a realistic drain.
    let mut v = vec![0u8; 44];
    v[0..4].copy_from_slice(b"RIFF");
    v[8..12].copy_from_slice(b"WAVE");
    v.extend(std::iter::repeat_n(0u8, n_bytes));
    v
}

/// A running mock server that answers `target_requests` connections
/// (round-robin over `responses`) before shutting down.
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
            for i in 0..target_requests {
                let (stream, _) = match listener.accept() {
                    Ok(x) => x,
                    Err(_) => return,
                };
                let resp = responses[i % responses.len()].clone();
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
        if let Some(h) = self.handle.take() {
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
            return Err(std::io::Error::other("headers too large"));
        }
    }
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
    let summary = run_bench(&args).expect("bench runs");
    assert_eq!(summary.ok_2xx, n);
    assert_eq!(summary.over_capacity_503, 0);
    assert_eq!(summary.transport_errors, 0);
    assert!(summary.ttfa_ms.p95 >= 0.0);
    assert_eq!(mock.served(), n);
}

#[test]
fn concurrent_workers_do_not_undercount() {
    // 32 iters × 4 workers → exactly 32 samples, no overshoot, no undercount.
    let n = 32;
    let mock = MockServer::spawn(vec![Response::Ok200Wav { body_bytes: 128 }], n);
    let args = small_body_args(&mock.base_url(), n, 4);
    let summary = run_bench(&args).expect("bench runs");
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
    let mock = MockServer::spawn(
        vec![Response::Ok200Wav { body_bytes: 64 }, Response::Over503],
        n,
    );
    let args = small_body_args(&mock.base_url(), n, 1);
    let summary = run_bench(&args).expect("bench runs");
    assert_eq!(summary.ok_2xx, 3);
    assert_eq!(summary.over_capacity_503, 3);
    assert_eq!(summary.client_error_4xx, 0);
    assert_eq!(summary.server_error_5xx, 0);
    assert_eq!(summary.rate_limited_429, 0);
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
    let summary = run_bench(&args).expect("bench runs");
    assert_eq!(summary.ok_2xx, 2);
    assert_eq!(summary.rate_limited_429, 2);
    assert_eq!(summary.client_error_4xx, 0);
}

#[test]
fn transport_error_when_server_absent() {
    // Grab a port and immediately release it — nothing listening.
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
    let summary = run_bench(&args).expect("bench runs");
    assert_eq!(summary.ok_2xx, 0);
    assert_eq!(summary.transport_errors, n);
}

#[test]
fn https_scheme_rejected_at_parse() {
    // NFR-DS-02 + FR-EX-08: TLS is not available in pure-std, so
    // `https://` MUST fail at URL-parse time — never silently HTTP.
    let mut args = Args::defaults();
    args.server = "https://127.0.0.1:8080".to_string();
    let err = run_bench(&args).unwrap_err();
    assert!(err.contains("only http:// is supported"), "got: {err}");
    // The redirect target is named so operators don't guess.
    assert!(err.contains("vokra-server-bench"));
}

#[test]
fn json_output_reparses_shape_and_carries_verdict() {
    // We hand-write the JSON so this test asserts a stable schema
    // via string substrings + trailing-newline discipline. There's
    // no serde_json here, but the sibling ureq-based binary was
    // schema-tested against the same substrings byte-for-byte.
    let n = 4;
    let mock = MockServer::spawn(vec![Response::Ok200Wav { body_bytes: 64 }], n);
    let args = {
        let mut a = small_body_args(&mock.base_url(), n, 1);
        a.format = OutputFormat::Json;
        a.budget_ms = 60_000; // localhost bench comfortably beats 60 s
        a
    };
    let summary = run_bench(&args).expect("bench runs");
    let mut buf = Vec::new();
    emit_json(&mut buf, &args, &summary).unwrap();
    let out = String::from_utf8(buf).unwrap();
    // Trailing newline only (single-line JSON blob for pipeline
    // discipline).
    assert_eq!(out.matches('\n').count(), 1);
    // Verdict present and PASS at 60 s budget on loopback.
    assert!(
        out.contains(r#""verdict":"PASS""#),
        "verdict wrong in {out}"
    );
    // Round-tripped iteration count.
    assert!(out.contains(&format!(r#""iterations":{n}"#)));
    // Counters present.
    assert!(out.contains(r#""ok_2xx":4"#));
}

#[test]
fn kv_output_covers_every_documented_key() {
    let n = 3;
    let mock = MockServer::spawn(vec![Response::Ok200Wav { body_bytes: 64 }], n);
    let args = small_body_args(&mock.base_url(), n, 1);
    let summary = run_bench(&args).expect("bench runs");
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
    // 8 slow (100 ms) responses × 4 workers must finish in noticeably
    // less than 8 × 100 = 800 ms. The concurrent path really overlaps.
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
    let summary = run_bench(&args).expect("bench runs");
    let elapsed = t0.elapsed();
    assert_eq!(summary.ok_2xx, n);
    // Serial lower bound is 800 ms; 4 workers should finish in
    // ~200 ms plus overhead. Allow generous CI slack.
    assert!(
        elapsed < Duration::from_millis(600),
        "expected concurrent runtime < 600 ms, got {elapsed:?}",
    );
}

#[test]
fn chunked_response_drains_and_records_ttfa() {
    // Server returns a chunked body — the pure-std client's chunked
    // decoder must run to completion and record a valid Timing.
    let n = 2;
    let chunks = vec![b"HELLO".to_vec(), b"WORLD".to_vec(), b"!".to_vec()];
    let mock = MockServer::spawn(vec![Response::Ok200Chunked { chunks }], n);
    let args = small_body_args(&mock.base_url(), n, 1);
    let summary = run_bench(&args).expect("bench runs");
    assert_eq!(summary.ok_2xx, n);
    assert_eq!(summary.transport_errors, 0);
}

#[test]
fn eof_framed_response_drains_and_records_ttfa() {
    // Server sends body without Content-Length AND without chunked
    // → the client must fall back to read-until-EOF. Both timings
    // must still record OK.
    let n = 2;
    let body = b"body-without-explicit-length".to_vec();
    let mock = MockServer::spawn(vec![Response::Ok200EofFramed { body }], n);
    let args = small_body_args(&mock.base_url(), n, 1);
    let summary = run_bench(&args).expect("bench runs");
    assert_eq!(summary.ok_2xx, n);
    assert_eq!(summary.transport_errors, 0);
}

#[test]
fn single_send_request_records_realistic_ttfa() {
    // Exercise `send_request` directly (bypassing the worker loop)
    // so a regression in the low-level timing calls surfaces here.
    let mock = MockServer::spawn(vec![Response::Ok200Wav { body_bytes: 256 }], 1);
    let target: HttpTarget = parse_target(&mock.base_url(), "/api/tts").unwrap();
    let body = build_request_body("hello", "en_US-libritts-high");
    let outcome = send_request(
        &target,
        &body,
        Duration::from_secs(2),
        Duration::from_secs(5),
    );
    match outcome {
        vokra_cli_bench_server::TransportOutcome::Complete(t) => {
            assert_eq!(t.status, 200);
            assert!(
                t.total >= t.ttfa,
                "total < ttfa: total={:?} ttfa={:?}",
                t.total,
                t.ttfa
            );
            assert!(t.ttfa.as_nanos() > 0);
        }
        other => panic!("expected Complete(2xx), got {other:?}"),
    }
}

#[test]
fn run_measurement_window_for_tests_smoke() {
    // Public test-shim that skips URL parsing so future integration
    // tests can drive the measurement window directly.
    let n = 3;
    let mock = MockServer::spawn(vec![Response::Ok200Wav { body_bytes: 64 }], n);
    let target = parse_target(&mock.base_url(), "/api/tts").unwrap();
    let body = build_request_body("hi", "v");
    let args = small_body_args(&mock.base_url(), n, 1);
    let summary = run_measurement_window_for_tests(&target, &body, &args);
    assert_eq!(summary.ok_2xx, n);
}
