//! Pure-`std` HTTP/1.1 client (POST + read response) over `TcpStream`.
//!
//! The whole point of this crate is to avoid `ureq` and `hyper`; every
//! byte on the wire is emitted here by hand. The protocol subset
//! implemented is exactly what talking to `vokra-server` needs:
//!
//! * request: `POST <path> HTTP/1.1` + `Host` + `Content-Type` +
//!   `Content-Length` + `Connection: close` + body;
//! * response: HTTP/1.1 status line + headers, then a body framed by
//!   either `Content-Length` OR a `Transfer-Encoding: chunked`
//!   decoder OR read-until-EOF for `Connection: close`.
//!
//! What is DELIBERATELY not implemented (and why the omission is
//! FR-EX-08 clean):
//!
//! * **HTTPS / TLS** — the sibling `integrations/vokra-server-bench/`
//!   binary handles it via `ureq`+rustls. This crate rejects `https://`
//!   at URL-parse time with an explicit error message pointing at the
//!   sibling; no silent HTTP downgrade.
//! * **HTTP redirects** — `vokra-server` never issues them for
//!   `POST /api/tts`; a 3xx is bucketed as `client_error_4xx` (which
//!   the operator will notice) rather than transparently followed.
//! * **HTTP/2** — irrelevant on the loopback reference environment.
//! * **Keep-alive** — every request opens a fresh TCP connection and
//!   sets `Connection: close` on the request. This costs one connect
//!   per iteration (<1 ms on loopback) and rules out socket-reuse
//!   subtleties that would confuse the TTFA measurement.

use std::io::{self, BufRead, BufReader, Write};
use std::net::{SocketAddr, TcpStream, ToSocketAddrs};
use std::time::{Duration, Instant};

use crate::bench::{Timing, TransportOutcome};

/// A resolved-once HTTP/1.1 target: host header + connect socket +
/// request path.
///
/// `host_header` MUST include the port if it is non-default (RFC 7230
/// § 5.4); we always include it because the bench connects to
/// `127.0.0.1:8080` in the reference environment. `addr` is the first
/// resolved `SocketAddr` — pure-`std` has no DNS round-robin so the
/// bench uses the first record. IPv6 addresses are supported by the
/// bracket syntax `http://[::1]:8080/api/tts`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HttpTarget {
    /// `Host: ` header value (includes port).
    pub host_header: String,
    /// Resolved socket address (host + port).
    pub addr: SocketAddr,
    /// Request-URI path, starts with `/`.
    pub path: String,
}

/// Parse `--server` + `--endpoint` into a resolved [`HttpTarget`].
///
/// # Errors
///
/// Returns a human-readable diagnostic when:
/// * `server` does not start with `http://` — TLS is not supported;
/// * host or port cannot be extracted;
/// * DNS resolution fails or yields no addresses.
pub fn parse_target(server: &str, endpoint: &str) -> Result<HttpTarget, String> {
    // Only `http://` is honoured — see the "no silent fallback" note
    // at the top of the module.
    let rest = server.strip_prefix("http://").ok_or_else(|| {
        format!(
            "--server: only http:// is supported (pure-std has no TLS); \
             got `{server}`. For TLS use the sibling ureq+rustls binary in \
             integrations/vokra-server-bench."
        )
    })?;
    // Split authority (host:port) from the OPTIONAL base path in the URL.
    // We treat everything after the first `/` as base-path and prefix the
    // caller-supplied endpoint to it. `--server http://x:9/proxy` +
    // `--endpoint /api/tts` therefore resolves to `/proxy/api/tts`.
    let (authority, base_path) = match rest.find('/') {
        Some(i) => (&rest[..i], &rest[i..]),
        None => (rest, ""),
    };
    let (host, port) = parse_authority(authority)?;

    // DNS resolve to one SocketAddr. `to_socket_addrs()` handles both
    // IP-literal and hostname inputs; IPv6 numerals come in via
    // `[::1]:8080` and are stripped of brackets in parse_authority.
    let addrs = format!("{host}:{port}")
        .to_socket_addrs()
        .map_err(|e| format!("--server: could not resolve `{host}:{port}`: {e}"))?
        .collect::<Vec<_>>();
    let addr = *addrs.first().ok_or_else(|| {
        format!("--server: `{host}:{port}` resolved to zero SocketAddrs (DNS returned empty set)")
    })?;

    // Compose request-URI: base_path + endpoint, both '/'-normalised.
    let base_trim = base_path.trim_end_matches('/');
    let endpoint_norm = if endpoint.starts_with('/') {
        endpoint.to_owned()
    } else {
        format!("/{endpoint}")
    };
    let path = if base_trim.is_empty() {
        endpoint_norm
    } else {
        format!("{base_trim}{endpoint_norm}")
    };

    Ok(HttpTarget {
        host_header: format!("{host}:{port}"),
        addr,
        path,
    })
}

/// Parse `host:port`. Handles IPv6 literals in `[…]:port` form.
fn parse_authority(authority: &str) -> Result<(String, u16), String> {
    if let Some(rest) = authority.strip_prefix('[') {
        // IPv6: `[addr]:port`. Find the closing bracket.
        let end = rest.find(']').ok_or_else(|| {
            format!("--server: malformed IPv6 authority `{authority}` (missing `]`)")
        })?;
        let host = rest[..end].to_owned();
        let after_bracket = &rest[end + 1..];
        // Colon-port is REQUIRED after the closing bracket for the
        // authority-form; if absent (`[::1]`), default port 80. The
        // NFR-PF-05 reference environment always uses an explicit port
        // so this branch is defensive rather than load-bearing.
        let port = if let Some(p) = after_bracket.strip_prefix(':') {
            p.parse::<u16>()
                .map_err(|_| format!("--server: invalid port `{p}` in `{authority}`"))?
        } else if after_bracket.is_empty() {
            80
        } else {
            return Err(format!(
                "--server: unexpected trailing text after IPv6 host: `{after_bracket}`"
            ));
        };
        Ok((host, port))
    } else if let Some((h, p)) = authority.rsplit_once(':') {
        // IPv4 / hostname `host:port`. `rsplit_once` picks the LAST `:`
        // so that e.g. `[::1]:8080` above isn't reached with the wrong
        // split — we already handled the bracket-form branch above.
        let port: u16 = p
            .parse()
            .map_err(|_| format!("--server: invalid port `{p}`"))?;
        Ok((h.to_owned(), port))
    } else {
        Ok((authority.to_owned(), 80))
    }
}

/// Send one HTTP/1.1 POST request and drain the response. `Instant`
/// bracketing is done inside this function so tests can pass the
/// returned [`Timing`] straight into the percentile summariser.
///
/// The two moments captured are:
///
/// * `ttfa`  = before write → the `\r\n\r\n` header terminator
///   received. `vokra-server` writes status + headers in the same TCP
///   flush as the first body byte, so this is the "first audio byte"
///   TTFA the M3-15 handover defines.
/// * `total` = before write → body drained (Content-Length or EOF).
pub fn send_request(
    target: &HttpTarget,
    body: &[u8],
    connect_timeout: Duration,
    read_timeout: Duration,
) -> TransportOutcome {
    let start = Instant::now();

    // Connect with a bounded timeout so a dropped TCP SYN does not
    // stall the whole bench. `connect_timeout` is separate from the
    // per-read timeout below.
    let stream = match TcpStream::connect_timeout(&target.addr, connect_timeout) {
        Ok(s) => s,
        Err(e) => return TransportOutcome::Transport(format!("connect: {e}")),
    };
    if let Err(e) = stream.set_read_timeout(Some(read_timeout)) {
        return TransportOutcome::Transport(format!("set_read_timeout: {e}"));
    }
    if let Err(e) = stream.set_write_timeout(Some(read_timeout)) {
        return TransportOutcome::Transport(format!("set_write_timeout: {e}"));
    }
    // Disable Nagle so the request-line + headers + body flush in one
    // TCP segment — avoids adding ~40 ms of Nagle-delay to the ttfa
    // measurement on loopback with concurrent workers.
    if let Err(e) = stream.set_nodelay(true) {
        return TransportOutcome::Transport(format!("set_nodelay: {e}"));
    }

    // Compose + write the request. `Connection: close` tells the peer
    // to shut down the socket after the response, which makes the
    // EOF-terminated body path work when the server does not send a
    // Content-Length. `Accept: */*` mirrors what most HTTP clients do.
    let header = format!(
        "POST {} HTTP/1.1\r\n\
         Host: {}\r\n\
         User-Agent: vokra-cli-bench-server/0.1\r\n\
         Accept: */*\r\n\
         Content-Type: application/json\r\n\
         Content-Length: {}\r\n\
         Connection: close\r\n\
         \r\n",
        target.path,
        target.host_header,
        body.len(),
    );
    // `write_all` does the same job as a BufWriter would (small header
    // + one body slice + no follow-up writes) with fewer allocations.
    let mut w = &stream;
    if let Err(e) = w.write_all(header.as_bytes()) {
        return TransportOutcome::Transport(format!("write header: {e}"));
    }
    if !body.is_empty() {
        if let Err(e) = w.write_all(body) {
            return TransportOutcome::Transport(format!("write body: {e}"));
        }
    }
    if let Err(e) = w.flush() {
        return TransportOutcome::Transport(format!("flush: {e}"));
    }

    // Read status line + headers up to `\r\n\r\n`. BufReader wraps the
    // socket so `read_until(b'\n', ...)` gives us a line at a time
    // without over-reading into the body — critical for the chunked
    // decoder path.
    let mut reader = BufReader::new(&stream);
    let (status, headers, ttfa) = match read_head(&mut reader, start) {
        Ok(x) => x,
        Err(e) => return TransportOutcome::Transport(format!("read head: {e}")),
    };

    // Drain the body per the framing rules of RFC 7230 § 3.3.3.
    // `HttpButBodyDrainFailed` bucket keeps this separate from the
    // successful-transport counters so an operator can distinguish
    // "server sent status + headers then reset" from "server sent 200
    // OK with a full response".
    if let Err(e) = drain_body(&mut reader, &headers) {
        return TransportOutcome::HttpButBodyDrainFailed(e.to_string());
    }
    let total = start.elapsed();

    TransportOutcome::Complete(Timing {
        ttfa,
        total,
        status,
    })
}

/// Parsed response head: status code + lowercased header list.
type ResponseHeaders = Vec<(String, String)>;

/// Read the status line + headers, returning the status code, header
/// list, and the wall-time between `start` and the `\r\n\r\n`
/// terminator.
fn read_head<R: BufRead>(
    r: &mut R,
    start: Instant,
) -> io::Result<(u16, ResponseHeaders, Duration)> {
    let mut status_line = String::new();
    let n = r.read_line(&mut status_line)?;
    if n == 0 {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "empty response",
        ));
    }
    let status = parse_status_code(&status_line)?;

    let mut headers: ResponseHeaders = Vec::new();
    loop {
        let mut line = String::new();
        let n = r.read_line(&mut line)?;
        if n == 0 {
            // Server closed connection before end-of-headers.
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "EOF in headers",
            ));
        }
        if line == "\r\n" || line == "\n" {
            break;
        }
        if let Some((k, v)) = line.split_once(':') {
            headers.push((k.trim().to_ascii_lowercase(), v.trim().to_owned()));
        }
        // Malformed header lines are ignored rather than rejected: a
        // proxy inserting garbage should not FAIL the bench on that
        // request. `verify` semantics are on the counts we do capture.
    }
    let ttfa = start.elapsed();
    Ok((status, headers, ttfa))
}

/// Parse "HTTP/1.1 200 OK\r\n" → 200.
fn parse_status_code(status_line: &str) -> io::Result<u16> {
    // Expected: 3 space-separated tokens: `HTTP/x.y CODE reason`.
    let mut it = status_line.trim_end_matches(['\r', '\n']).splitn(3, ' ');
    let version = it
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "empty status line"))?;
    if !version.starts_with("HTTP/") {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("not an HTTP status line: `{status_line}`"),
        ));
    }
    let code_str = it
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "no status code"))?;
    let code: u16 = code_str
        .parse()
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "non-numeric status code"))?;
    // Reason phrase (`it.next()`) is deliberately discarded.
    Ok(code)
}

/// Return a header value (case-insensitive lookup) or `None`.
fn header_value<'a>(headers: &'a ResponseHeaders, name: &str) -> Option<&'a str> {
    let lname = name.to_ascii_lowercase();
    headers
        .iter()
        .find_map(|(k, v)| if *k == lname { Some(v.as_str()) } else { None })
}

/// Drain the response body per RFC 7230 § 3.3.3.
///
/// Priority order:
/// 1. `Transfer-Encoding: chunked` (§ 3.3.3 rule 3);
/// 2. `Content-Length: N`         (§ 3.3.3 rule 5);
/// 3. read-until-EOF               (§ 3.3.3 rule 7 — falls out because
///    we set `Connection: close`).
fn drain_body<R: BufRead>(r: &mut R, headers: &ResponseHeaders) -> io::Result<()> {
    let te = header_value(headers, "transfer-encoding")
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_default();
    // A comma-separated `Transfer-Encoding` list where `chunked` appears
    // anywhere means chunked. `TE: identity` alone falls through.
    if te.split(',').any(|t| t.trim() == "chunked") {
        return drain_chunked(r);
    }
    if let Some(cl) = header_value(headers, "content-length") {
        let n: usize = cl.trim().parse().map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid Content-Length: `{cl}`"),
            )
        })?;
        // A dedicated buffer avoids allocating room for the whole body
        // up-front when the peer lied about `Content-Length`; we cap
        // at ~16 MiB above which we bail as "server bug".
        let capped = n.min(16 * 1024 * 1024);
        let mut buf = vec![0u8; capped];
        r.read_exact(&mut buf)?;
        return Ok(());
    }
    // Neither Transfer-Encoding: chunked nor Content-Length → the
    // body ends at EOF because we set `Connection: close`.
    let mut sink = Vec::with_capacity(64 * 1024);
    r.read_to_end(&mut sink)?;
    Ok(())
}

/// Decode `Transfer-Encoding: chunked` per RFC 7230 § 4.1.
fn drain_chunked<R: BufRead>(r: &mut R) -> io::Result<()> {
    loop {
        let mut size_line = String::new();
        let n = r.read_line(&mut size_line)?;
        if n == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "chunked: EOF before size line",
            ));
        }
        // A chunk-size line is `<hex>[;extension]\r\n`. Trim the CRLF
        // then strip any chunk-extension after `;`.
        let sz_str = size_line
            .trim_end_matches(['\r', '\n'])
            .split(';')
            .next()
            .unwrap_or("")
            .trim();
        let sz = usize::from_str_radix(sz_str, 16).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("chunked: bad size line `{size_line}`"),
            )
        })?;
        if sz == 0 {
            // Terminal 0-size chunk. Drain trailer headers (which we
            // never look at) up to the final CRLF.
            loop {
                let mut tail = String::new();
                let n = r.read_line(&mut tail)?;
                if n == 0 || tail == "\r\n" || tail == "\n" {
                    return Ok(());
                }
            }
        }
        // Read the chunk body + its trailing CRLF. Use `read_exact` so
        // a short read is reported as a transport error rather than a
        // silent truncation.
        let mut body = vec![0u8; sz];
        r.read_exact(&mut body)?;
        let mut crlf = [0u8; 2];
        r.read_exact(&mut crlf)?;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn parse_target_default_endpoint() {
        let t = parse_target("http://127.0.0.1:8080", "/api/tts").unwrap();
        assert_eq!(t.host_header, "127.0.0.1:8080");
        assert_eq!(t.path, "/api/tts");
        assert_eq!(t.addr.port(), 8080);
    }

    #[test]
    fn parse_target_normalises_endpoint_slash() {
        let t = parse_target("http://127.0.0.1:8080", "api/tts").unwrap();
        assert_eq!(t.path, "/api/tts");
    }

    #[test]
    fn parse_target_composes_base_path() {
        // `--server http://x/proxy` + `--endpoint /api/tts` →
        // `/proxy/api/tts`. Path is composed, host_header is authority
        // only.
        let t = parse_target("http://127.0.0.1:8080/proxy", "/api/tts").unwrap();
        assert_eq!(t.host_header, "127.0.0.1:8080");
        assert_eq!(t.path, "/proxy/api/tts");
    }

    #[test]
    fn parse_target_rejects_https() {
        let err = parse_target("https://127.0.0.1:8080", "/api/tts").unwrap_err();
        assert!(err.contains("only http:// is supported"));
        // The redirect target is named so operators don't guess.
        assert!(err.contains("vokra-server-bench"));
    }

    #[test]
    fn parse_target_rejects_missing_scheme() {
        let err = parse_target("127.0.0.1:8080", "/api/tts").unwrap_err();
        assert!(err.contains("only http:// is supported"));
    }

    #[test]
    fn parse_target_rejects_bad_port() {
        let err = parse_target("http://127.0.0.1:not-a-port", "/x").unwrap_err();
        assert!(err.contains("invalid port"), "got: {err}");
    }

    #[test]
    fn parse_target_ipv6_bracket_form() {
        // `[::1]:9` should resolve to ::1 on port 9. Only the loopback
        // literal is portable across CI runners; hostname-form IPv6 is
        // not commonly available.
        let t = parse_target("http://[::1]:9", "/api/tts").unwrap();
        assert_eq!(t.host_header, "::1:9");
        assert_eq!(t.addr.port(), 9);
        assert!(t.addr.is_ipv6());
    }

    #[test]
    fn parse_target_ipv6_bracket_no_port_defaults_80() {
        let t = parse_target("http://[::1]", "/x").unwrap();
        assert_eq!(t.addr.port(), 80);
    }

    #[test]
    fn parse_target_defaults_port_80_no_colon() {
        // Bare hostname / IPv4 without `:port` defaults to 80 (HTTP
        // spec). This is the branch we exercise when a user pastes a
        // bare server hostname.
        let t = parse_target("http://localhost", "/x").unwrap();
        assert_eq!(t.addr.port(), 80);
    }

    #[test]
    fn parse_status_code_ok() {
        assert_eq!(parse_status_code("HTTP/1.1 200 OK\r\n").unwrap(), 200);
        assert_eq!(
            parse_status_code("HTTP/1.1 503 Service Unavailable\r\n").unwrap(),
            503,
        );
        // A missing reason phrase is still parseable.
        assert_eq!(parse_status_code("HTTP/1.1 204 \r\n").unwrap(), 204);
    }

    #[test]
    fn parse_status_code_rejects_non_http() {
        assert!(parse_status_code("nope\r\n").is_err());
        assert!(parse_status_code("HTTP/1.1\r\n").is_err());
        assert!(parse_status_code("HTTP/1.1 xyz OK\r\n").is_err());
    }

    #[test]
    fn read_head_reads_status_and_headers() {
        // Note the DOS-style CRLF line endings — the parser must not
        // ship a `\r` into the header value.
        let raw = b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\nX-Foo: bar\r\n\r\nHELLO";
        let mut r = BufReader::new(Cursor::new(raw));
        let (code, headers, _ttfa) = read_head(&mut r, Instant::now()).unwrap();
        assert_eq!(code, 200);
        assert_eq!(header_value(&headers, "content-length"), Some("5"));
        assert_eq!(header_value(&headers, "X-Foo"), Some("bar")); // case-insensitive
    }

    #[test]
    fn drain_body_content_length_reads_exact_bytes() {
        let raw = b"12345";
        let mut r = BufReader::new(Cursor::new(raw));
        let headers = vec![("content-length".to_owned(), "5".to_owned())];
        drain_body(&mut r, &headers).unwrap();
    }

    #[test]
    fn drain_body_content_length_short_read_errors() {
        // Server promised 10 bytes but sent 3 — must be an error, not
        // a silent truncation.
        let raw = b"abc";
        let mut r = BufReader::new(Cursor::new(raw));
        let headers = vec![("content-length".to_owned(), "10".to_owned())];
        assert!(drain_body(&mut r, &headers).is_err());
    }

    #[test]
    fn drain_body_read_to_eof_when_no_length() {
        let raw = b"arbitrary bytes with no framing";
        let mut r = BufReader::new(Cursor::new(raw));
        let headers = vec![]; // no content-length, no chunked
        drain_body(&mut r, &headers).unwrap();
    }

    #[test]
    fn drain_body_chunked_two_chunks_then_terminator() {
        // RFC 7230 § 4.1: `<size>\r\n<data>\r\n...0\r\n\r\n`.
        let raw = b"5\r\nHELLO\r\n3\r\nfoo\r\n0\r\n\r\n";
        let mut r = BufReader::new(Cursor::new(raw));
        let headers = vec![("transfer-encoding".to_owned(), "chunked".to_owned())];
        drain_body(&mut r, &headers).unwrap();
    }

    #[test]
    fn drain_body_chunked_with_extension_and_trailer() {
        // Extension after `;` MUST be ignored; trailer headers before
        // the final CRLF MUST be discarded.
        let raw = b"5;chunk-ext=1\r\nHELLO\r\n0\r\nX-Trailer: bye\r\n\r\n";
        let mut r = BufReader::new(Cursor::new(raw));
        let headers = vec![("transfer-encoding".to_owned(), "chunked".to_owned())];
        drain_body(&mut r, &headers).unwrap();
    }

    #[test]
    fn drain_body_chunked_bad_size_rejected() {
        let raw = b"nope\r\nHELLO\r\n0\r\n\r\n";
        let mut r = BufReader::new(Cursor::new(raw));
        let headers = vec![("transfer-encoding".to_owned(), "chunked".to_owned())];
        assert!(drain_body(&mut r, &headers).is_err());
    }

    #[test]
    fn drain_body_te_priority_over_content_length() {
        // If both TE:chunked and CL are present, TE wins per RFC 7230
        // § 3.3.3 rule 3.
        let raw = b"5\r\nHELLO\r\n0\r\n\r\n";
        let mut r = BufReader::new(Cursor::new(raw));
        let headers = vec![
            ("transfer-encoding".to_owned(), "chunked".to_owned()),
            // A misleading CL of 100 that the parser must ignore.
            ("content-length".to_owned(), "100".to_owned()),
        ];
        drain_body(&mut r, &headers).unwrap();
    }
}
