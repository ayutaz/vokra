//! Shared raw-TCP HTTP/1.1 test client for the vokra-server integration
//! suites (cc-09 fix, 2026-07-19 M4-residual audit).
//!
//! # Why this module exists (the vllm_compat loopback flake, root-caused)
//!
//! The previous per-file copies of this helper (`vllm_compat.rs`,
//! `openai_compat.rs`, `piper_http_compat.rs`, `tts_g2p_injection.rs`) wrote
//! the request, then did a single `read_to_end` and treated ANY read error as
//! fatal. Under parallel full-suite load that intermittently failed with
//! `ConnectionReset` (macOS `ECONNRESET`, os error 54) at vllm_compat.rs:329 /
//! :384 — reproduced 3/5 full-suite runs on pristine HEAD `1956ca6`
//! (2026-07-19).
//!
//! Root cause (verified with an instrumented probe, not guessed):
//!
//! 1. The e2e legs POST to routes the discovery-only test server does NOT
//!    mount (`spawn_server_for_test` boots health-only), so axum's fallback
//!    answers `404` WITHOUT draining the request body (the fallback drops the
//!    `Body` unread — normal axum behaviour).
//! 2. hyper writes the response (`Connection: close` framing) and closes the
//!    socket while the client's request-body bytes are still unread in the
//!    server's kernel receive buffer. Per TCP semantics (macOS and Linux
//!    alike), closing a socket with unread received data sends **RST**, not
//!    FIN.
//! 3. The RST discards the client's kernel receive queue and poisons the
//!    socket, so the client's `read_to_end` returns `Err(ConnectionReset)` —
//!    even when the COMPLETE response had already been buffered into the
//!    user-space `Vec`. The probe observed exactly that: on the failing
//!    iteration the full 101-byte `404` response was in the buffer at error
//!    time, and `read_to_end`'s contract ("on error, data read so far remains
//!    in `buf`") means the helper simply threw a valid response away.
//! 4. Load dependence: when the head+body writes coalesce into one segment
//!    (idle machine), hyper's read buffer swallows headers AND body in one
//!    `read()`, nothing is left unread in the kernel, close sends FIN, and the
//!    old helper passed — which is why single-binary runs were 6/6 green while
//!    the parallel full-suite (sibling test binaries + agent compile load
//!    delaying the body segment) reproduced 2/3.
//!
//! This is a **test-harness defect**, not a server bug: the server answered
//! every request correctly (the probe never saw a missing or truncated
//! response — only a discarded complete one). Responding early and closing
//! without draining an unread request body is legal HTTP/1.1; real clients
//! must (and do) tolerate the resulting RST-after-response.
//!
//! # The fix (this module)
//!
//! * **Complete-response detection**: the read loop parses HTTP/1.1 framing
//!   (`Content-Length`, chunked, or close-delimited) as bytes arrive. A read
//!   error AFTER a complete response has been buffered is treated as
//!   end-of-response — the response is returned, not discarded.
//! * **Bounded retry for genuinely-incomplete exchanges**: if the reset lands
//!   BEFORE the response completes (e.g. RST arrives while the response is in
//!   flight, or the request write itself fails on the already-closed socket),
//!   the request is retried on a fresh connection, at most
//!   [`MAX_ATTEMPTS`] times. This is not a sleep-and-hope: the mechanism is
//!   understood, the retry only covers the narrow window where the kernel
//!   discarded response bytes we can never recover, and every test request in
//!   this suite is idempotent (schema probes against a stateless server).
//! * One copy: the four per-file helpers are replaced by this module, so the
//!   failure mode cannot re-diverge per file.

// Each integration-test crate that `mod support;`s this file uses a subset of
// the API; the unused remainder must not trip `-D warnings` per crate.
#![allow(dead_code)]

use std::net::SocketAddr;
use std::time::Duration;

use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// Maximum request attempts. Attempt 2+ only happens when a connection was
/// reset before a complete response was buffered (see module docs) — a
/// narrow race; two retries push the residual failure probability to noise.
const MAX_ATTEMPTS: usize = 3;

/// Hard per-attempt read deadline so a stuck server can never hang a suite.
const READ_TIMEOUT: Duration = Duration::from_secs(5);

/// A parsed HTTP/1.1 response.
#[derive(Debug, Clone)]
pub struct HttpResponse {
    /// Status code from the status line.
    pub status: u16,
    /// Raw header block (status line + headers, without the terminating
    /// blank line).
    pub head: String,
    /// De-framed body bytes (Content-Length-trimmed / chunked-decoded).
    pub body: Vec<u8>,
}

/// POST `body` as `application/json` and return `(status, body_bytes)` —
/// signature-compatible with the old per-file `http_post_json` helpers.
pub async fn http_post_json(
    addr: SocketAddr,
    path: &str,
    body: &[u8],
) -> std::io::Result<(u16, Vec<u8>)> {
    let resp = http_request(addr, "POST", path, Some("application/json"), body).await?;
    Ok((resp.status, resp.body))
}

/// Like [`http_post_json`] but also returns the raw header block —
/// signature-compatible with the old `(status, headers_text, body)` helpers
/// in `piper_http_compat.rs` / `tts_g2p_injection.rs` (they inspect
/// `Content-Type: audio/wav`).
pub async fn http_post_json_with_head(
    addr: SocketAddr,
    path: &str,
    body: &[u8],
) -> std::io::Result<(u16, String, Vec<u8>)> {
    let resp = http_request(addr, "POST", path, Some("application/json"), body).await?;
    Ok((resp.status, resp.head, resp.body))
}

/// [`http_post_json_with_head`] with a caller-chosen per-attempt read
/// deadline — real-weight TTS synthesis (env-gated suites) can legitimately
/// exceed the default 5 s under parallel load.
pub async fn http_post_json_with_head_timeout(
    addr: SocketAddr,
    path: &str,
    body: &[u8],
    read_timeout: Duration,
) -> std::io::Result<(u16, String, Vec<u8>)> {
    let resp = http_request_with_timeout(
        addr,
        "POST",
        path,
        Some("application/json"),
        body,
        read_timeout,
    )
    .await?;
    Ok((resp.status, resp.head, resp.body))
}

/// POST a `multipart/form-data` body (boundary supplied by the caller) and
/// return `(status, body_bytes)` — signature-compatible with the old
/// `http_post_multipart` helpers.
pub async fn http_post_multipart(
    addr: SocketAddr,
    path: &str,
    boundary: &str,
    body: &[u8],
) -> std::io::Result<(u16, Vec<u8>)> {
    let ct = format!("multipart/form-data; boundary={boundary}");
    let resp = http_request(addr, "POST", path, Some(&ct), body).await?;
    Ok((resp.status, resp.body))
}

/// [`http_post_multipart`] with a caller-chosen per-attempt read deadline.
///
/// Same rationale as [`http_post_json_with_head_timeout`], on the ASR side:
/// a real-weight transcription (env-gated suites — cc-40) legitimately takes
/// longer than the default 5 s, especially for the larger Whisper sizes under
/// parallel load. Raising the shared default instead would slow every genuine
/// hang in the hermetic suites down to the same deadline.
pub async fn http_post_multipart_timeout(
    addr: SocketAddr,
    path: &str,
    boundary: &str,
    body: &[u8],
    read_timeout: Duration,
) -> std::io::Result<(u16, Vec<u8>)> {
    let ct = format!("multipart/form-data; boundary={boundary}");
    let resp = http_request_with_timeout(addr, "POST", path, Some(&ct), body, read_timeout).await?;
    Ok((resp.status, resp.body))
}

/// GET `path` and return `(status, body_bytes)`.
pub async fn http_get(addr: SocketAddr, path: &str) -> std::io::Result<(u16, Vec<u8>)> {
    let resp = http_request(addr, "GET", path, None, &[]).await?;
    Ok((resp.status, resp.body))
}

/// One HTTP/1.1 request over a fresh loopback connection, with the cc-09
/// complete-response detection + bounded retry described in the module docs.
pub async fn http_request(
    addr: SocketAddr,
    method: &str,
    path: &str,
    content_type: Option<&str>,
    body: &[u8],
) -> std::io::Result<HttpResponse> {
    http_request_with_timeout(addr, method, path, content_type, body, READ_TIMEOUT).await
}

/// [`http_request`] with a caller-chosen per-attempt read deadline.
pub async fn http_request_with_timeout(
    addr: SocketAddr,
    method: &str,
    path: &str,
    content_type: Option<&str>,
    body: &[u8],
    read_timeout: Duration,
) -> std::io::Result<HttpResponse> {
    let mut last_err: Option<std::io::Error> = None;
    for attempt in 1..=MAX_ATTEMPTS {
        match http_request_once(addr, method, path, content_type, body, read_timeout).await {
            Ok(resp) => return Ok(resp),
            Err(e) if is_retryable(&e) && attempt < MAX_ATTEMPTS => {
                eprintln!(
                    "support::http_request[{method} {path}]: attempt {attempt}/{MAX_ATTEMPTS} \
                     hit {e:?} before a complete response was buffered; retrying on a fresh \
                     connection (cc-09 — see tests/support/mod.rs module docs)",
                );
                last_err = Some(e);
            }
            Err(e) => return Err(e),
        }
    }
    Err(last_err
        .unwrap_or_else(|| std::io::Error::other("http_request: retries exhausted (unreachable)")))
}

/// A reset-family error that may legitimately occur when the server responds
/// early and closes with the request body unread (module docs §fix). Anything
/// else (timeout, malformed response, refused connection) is a real failure
/// and is NOT retried.
fn is_retryable(e: &std::io::Error) -> bool {
    matches!(
        e.kind(),
        std::io::ErrorKind::ConnectionReset
            | std::io::ErrorKind::ConnectionAborted
            | std::io::ErrorKind::BrokenPipe
    )
}

async fn http_request_once(
    addr: SocketAddr,
    method: &str,
    path: &str,
    content_type: Option<&str>,
    body: &[u8],
    read_timeout: Duration,
) -> std::io::Result<HttpResponse> {
    let mut sock = tokio::net::TcpStream::connect(addr).await?;
    let mut head = format!(
        "{method} {path} HTTP/1.1\r\n\
         Host: 127.0.0.1\r\n\
         Connection: close\r\n"
    );
    if let Some(ct) = content_type {
        head.push_str(&format!("Content-Type: {ct}\r\n"));
    }
    if !body.is_empty() || method == "POST" {
        head.push_str(&format!("Content-Length: {}\r\n", body.len()));
    }
    head.push_str("\r\n");

    // The write itself can observe the reset (server already responded to the
    // headers and closed before we finished the body). The response may still
    // be readable from the socket in that case, so a write error does NOT
    // abort the exchange — we fall through to the read and let the
    // complete-response detection decide.
    let write_result: std::io::Result<()> = async {
        sock.write_all(head.as_bytes()).await?;
        sock.write_all(body).await?;
        sock.flush().await
    }
    .await;

    let mut buf: Vec<u8> = Vec::with_capacity(4096);
    let mut chunk = [0u8; 4096];
    let deadline = tokio::time::Instant::now() + read_timeout;
    loop {
        // Return as soon as the buffered bytes form a complete response —
        // do not wait for EOF (the RST may arrive between the response and
        // the FIN that never comes).
        if let Some(resp) = parse_complete_response(&buf)? {
            return Ok(resp);
        }
        let read = tokio::time::timeout_at(deadline, sock.read(&mut chunk)).await;
        match read {
            Ok(Ok(0)) => {
                // Clean EOF. Either the response is complete (close-delimited
                // framing completes exactly here) or it was truncated.
                return finish_at_eof(buf);
            }
            Ok(Ok(n)) => buf.extend_from_slice(&chunk[..n]),
            Ok(Err(e)) if is_retryable(&e) => {
                // The cc-09 case: reset after the server's early close. If the
                // complete response is already buffered, it is the response;
                // close-delimited framing is also complete at this boundary
                // (the reset IS the close). Otherwise propagate for retry.
                if let Some(resp) = parse_complete_response(&buf)? {
                    return Ok(resp);
                }
                if let Some(resp) = parse_close_delimited(&buf)? {
                    return Ok(resp);
                }
                // If the write also failed, surface that context alongside.
                if let Err(we) = write_result {
                    return Err(std::io::Error::new(
                        e.kind(),
                        format!("read reset ({e}) after write error ({we})"),
                    ));
                }
                return Err(e);
            }
            Ok(Err(e)) => return Err(e),
            Err(_) => return Err(std::io::Error::other("http read timeout")),
        }
    }
}

/// Terminal parse at clean EOF: complete framed response, close-delimited
/// response, or an explicit truncation error (never a silent partial).
fn finish_at_eof(buf: Vec<u8>) -> std::io::Result<HttpResponse> {
    if let Some(resp) = parse_complete_response(&buf)? {
        return Ok(resp);
    }
    if let Some(resp) = parse_close_delimited(&buf)? {
        return Ok(resp);
    }
    Err(std::io::Error::other(format!(
        "connection closed mid-response ({} bytes buffered, incomplete framing)",
        buf.len()
    )))
}

/// Attempts to parse `buf` as a COMPLETE response under explicit framing
/// (`Content-Length` or `Transfer-Encoding: chunked`). Returns `Ok(None)`
/// when more bytes are needed (including "headers not terminated yet").
fn parse_complete_response(buf: &[u8]) -> std::io::Result<Option<HttpResponse>> {
    let Some(sep) = find_header_end(buf) else {
        return Ok(None);
    };
    let head_str = std::str::from_utf8(&buf[..sep])
        .map_err(|e| std::io::Error::other(format!("non-utf8 response head: {e}")))?;
    let status = parse_status(head_str)?;
    let raw_body = &buf[sep + 4..];

    if let Some(len) = header_value(head_str, "content-length") {
        let len: usize = len
            .trim()
            .parse()
            .map_err(|_| std::io::Error::other(format!("bad Content-Length: {len:?}")))?;
        if raw_body.len() >= len {
            return Ok(Some(HttpResponse {
                status,
                head: head_str.to_owned(),
                body: raw_body[..len].to_vec(),
            }));
        }
        return Ok(None);
    }

    if header_value(head_str, "transfer-encoding")
        .is_some_and(|v| v.to_ascii_lowercase().contains("chunked"))
    {
        return Ok(decode_chunked(raw_body)?.map(|body| HttpResponse {
            status,
            head: head_str.to_owned(),
            body,
        }));
    }

    // No explicit framing: close-delimited — only complete at EOF/close,
    // which the caller handles via `parse_close_delimited`.
    Ok(None)
}

/// Close-delimited framing: headers terminated, no `Content-Length` and no
/// chunked encoding — whatever has been buffered when the peer closes IS the
/// body (RFC 9112 §6.3 rule 8). Returns `Ok(None)` when the response has
/// explicit framing (the stricter parser owns that case).
fn parse_close_delimited(buf: &[u8]) -> std::io::Result<Option<HttpResponse>> {
    let Some(sep) = find_header_end(buf) else {
        return Ok(None);
    };
    let head_str = std::str::from_utf8(&buf[..sep])
        .map_err(|e| std::io::Error::other(format!("non-utf8 response head: {e}")))?;
    if header_value(head_str, "content-length").is_some()
        || header_value(head_str, "transfer-encoding").is_some()
    {
        return Ok(None);
    }
    let status = parse_status(head_str)?;
    Ok(Some(HttpResponse {
        status,
        head: head_str.to_owned(),
        body: buf[sep + 4..].to_vec(),
    }))
}

fn find_header_end(buf: &[u8]) -> Option<usize> {
    buf.windows(4).position(|w| w == b"\r\n\r\n")
}

fn parse_status(head_str: &str) -> std::io::Result<u16> {
    let first_line = head_str.lines().next().unwrap_or("");
    first_line
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| std::io::Error::other(format!("bad status line: {first_line:?}")))
}

/// Case-insensitive single-header lookup in a raw header block.
fn header_value<'a>(head_str: &'a str, name: &str) -> Option<&'a str> {
    head_str.lines().skip(1).find_map(|line| {
        let (k, v) = line.split_once(':')?;
        k.trim().eq_ignore_ascii_case(name).then_some(v.trim())
    })
}

/// Strict minimal chunked-body decoder. `Ok(None)` = incomplete (need more
/// bytes); `Ok(Some(body))` = terminated by the 0-size chunk. Replaces the
/// old per-file "strip the first chunk-size line" best-effort, which silently
/// corrupted multi-chunk bodies.
fn decode_chunked(mut raw: &[u8]) -> std::io::Result<Option<Vec<u8>>> {
    let mut out = Vec::new();
    loop {
        let Some(line_end) = raw.windows(2).position(|w| w == b"\r\n") else {
            return Ok(None);
        };
        let size_line = std::str::from_utf8(&raw[..line_end])
            .map_err(|e| std::io::Error::other(format!("non-utf8 chunk size: {e}")))?;
        let size_hex = size_line.split(';').next().unwrap_or("").trim();
        let size = usize::from_str_radix(size_hex, 16)
            .map_err(|_| std::io::Error::other(format!("bad chunk size: {size_line:?}")))?;
        let chunk_start = line_end + 2;
        if size == 0 {
            // Terminator chunk. Trailers (if any) end with CRLF; we accept the
            // terminator as completion (our servers do not emit trailers).
            return Ok(Some(out));
        }
        let chunk_end = chunk_start + size;
        if raw.len() < chunk_end + 2 {
            return Ok(None);
        }
        out.extend_from_slice(&raw[chunk_start..chunk_end]);
        if &raw[chunk_end..chunk_end + 2] != b"\r\n" {
            return Err(std::io::Error::other("chunk not CRLF-terminated"));
        }
        raw = &raw[chunk_end + 2..];
    }
}
