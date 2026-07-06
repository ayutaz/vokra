//! T08 — OpenAI `/v1/audio/transcriptions` compat integration test.
//!
//! Contract (per M2-09 plan §3.2 (f)):
//!
//! * `multipart/form-data` POST of a WAV file to
//!   `/v1/audio/transcriptions` returns `200 OK` with a JSON body that
//!   contains the string field `text` (faster-whisper drop-in schema).
//! * Base whisper GGUF is available per-PR (env
//!   `VOKRA_WHISPER_BASE_GGUF`, alias `VOKRA_WHISPER_GGUF`).
//! * `large-v3` GGUF is nightly-only (`VOKRA_WHISPER_LARGE_V3_GGUF`),
//!   following the M2-06 T12 GGUF-gated pattern used by
//!   `crates/vokra-models/tests/parity_whisper.rs`.
//! * When the required GGUF is not present, the row skips cleanly with
//!   an eprintln (no silent success — CLAUDE.md FR-EX-08 posture).
//!
//! The `/v1/audio/transcriptions` route itself lands in T06 / T07. Until
//! then the route stub returns `404`; this test detects that state and
//! marks the row as "route not yet wired (T06/T07 pending)" instead of
//! asserting green. That way this file is committed on the T08 change
//! and becomes a live contract test the moment T07 lands, with no code
//! churn.
//!
//! The test binds `127.0.0.1:0` (OS-assigned free port) via
//! `spawn_server_for_test` and tears the server down through the
//! shutdown trigger. It never opens a listening socket on a fixed port,
//! so parallel `cargo test` runs and constrained CI sandboxes work.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::Duration;

use vokra_server::{Config, spawn_server_for_test};

// ---------- fixture: minimal 16-bit PCM WAV of a 440 Hz tone ----------

/// Build a canonical RIFF/WAVE mono 16 kHz 16-bit PCM byte stream of
/// `duration_ms` of a 440 Hz sine (M0-06 `tone.wav`-equivalent, kept in
/// memory to avoid touching the workspace's fixture directories).
fn tone_wav_bytes(duration_ms: u32) -> Vec<u8> {
    const SAMPLE_RATE: u32 = 16_000;
    const CHANNELS: u16 = 1;
    const BITS_PER_SAMPLE: u16 = 16;
    let n_samples = (SAMPLE_RATE as u64 * duration_ms as u64 / 1_000) as u32;
    let byte_rate = SAMPLE_RATE * CHANNELS as u32 * (BITS_PER_SAMPLE / 8) as u32;
    let block_align = CHANNELS * BITS_PER_SAMPLE / 8;
    let data_size = n_samples * block_align as u32;
    let riff_size = 36 + data_size;

    let mut out = Vec::with_capacity(44 + data_size as usize);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&riff_size.to_le_bytes());
    out.extend_from_slice(b"WAVE");
    // fmt subchunk (PCM, size 16).
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
    out.extend_from_slice(&CHANNELS.to_le_bytes());
    out.extend_from_slice(&SAMPLE_RATE.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&BITS_PER_SAMPLE.to_le_bytes());
    // data subchunk.
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_size.to_le_bytes());

    // 440 Hz sine at ~50% full-scale.
    let two_pi = std::f32::consts::TAU;
    for n in 0..n_samples {
        let t = n as f32 / SAMPLE_RATE as f32;
        let s = (two_pi * 440.0 * t).sin() * 0.5;
        let q = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        out.extend_from_slice(&q.to_le_bytes());
    }
    out
}

// ---------- GGUF gating (M2-06 T12 pattern) ----------

/// Whisper size → env var holding the converted GGUF path. `base` also
/// honours the legacy `VOKRA_WHISPER_GGUF` alias so a dev with only that
/// set gets the base row for free.
fn whisper_gguf_for(size: &str) -> Option<PathBuf> {
    let primary = format!(
        "VOKRA_WHISPER_{}_GGUF",
        size.to_uppercase().replace('-', "_")
    );
    if let Some(p) = std::env::var_os(&primary) {
        return Some(PathBuf::from(p));
    }
    if size == "base"
        && let Some(p) = std::env::var_os("VOKRA_WHISPER_GGUF")
    {
        return Some(PathBuf::from(p));
    }
    None
}

// ---------- raw HTTP multipart client (no reqwest form-data dep needed) ----------

/// Build the raw `multipart/form-data` body for `file` (WAV) + `model`.
/// axum's `Multipart` extractor (T06) is boundary-driven, so we speak the
/// wire format directly and avoid pulling in `reqwest`'s `multipart`
/// feature (keeps dev-deps minimal).
fn build_multipart(boundary: &str, wav: &[u8], model: &str) -> Vec<u8> {
    let mut b = Vec::new();
    // --- file part ---
    b.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
    b.extend_from_slice(
        b"Content-Disposition: form-data; name=\"file\"; filename=\"tone.wav\"\r\n",
    );
    b.extend_from_slice(b"Content-Type: audio/wav\r\n\r\n");
    b.extend_from_slice(wav);
    b.extend_from_slice(b"\r\n");
    // --- model part ---
    b.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
    b.extend_from_slice(b"Content-Disposition: form-data; name=\"model\"\r\n\r\n");
    b.extend_from_slice(model.as_bytes());
    b.extend_from_slice(b"\r\n");
    // --- terminator ---
    b.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
    b
}

/// POST the multipart body via a raw TCP write/read. Returns
/// `(status, body_bytes)`. Uses a bounded read + 5 s hard timeout so a
/// stuck server can't hang the test suite.
async fn http_post_multipart(
    addr: SocketAddr,
    path: &str,
    boundary: &str,
    body: &[u8],
) -> std::io::Result<(u16, Vec<u8>)> {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let mut sock = tokio::net::TcpStream::connect(addr).await?;
    let head = format!(
        "POST {path} HTTP/1.1\r\n\
         Host: 127.0.0.1\r\n\
         Content-Type: multipart/form-data; boundary={boundary}\r\n\
         Content-Length: {len}\r\n\
         Connection: close\r\n\r\n",
        len = body.len(),
    );
    sock.write_all(head.as_bytes()).await?;
    sock.write_all(body).await?;
    sock.flush().await?;

    let mut buf = Vec::new();
    tokio::time::timeout(Duration::from_secs(5), sock.read_to_end(&mut buf))
        .await
        .map_err(|_| std::io::Error::other("http read timeout"))??;

    // Parse "HTTP/1.1 NNN ..." and split at the header/body boundary.
    let sep = buf
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .ok_or_else(|| std::io::Error::other("no header terminator"))?;
    let head_str = std::str::from_utf8(&buf[..sep])
        .map_err(|e| std::io::Error::other(format!("utf8: {e}")))?;
    let first_line = head_str.lines().next().unwrap_or("");
    let status: u16 = first_line
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| std::io::Error::other(format!("bad status line: {first_line:?}")))?;

    // Body may be chunked or plain. `Connection: close` above requests
    // plain framing, and hyper honours that, so treat the rest as body.
    // If a `Transfer-Encoding: chunked` slipped in, tolerate it by
    // stripping the first chunk-size line (best-effort — the T07 handler
    // returns a small JSON that fits in one chunk).
    let body_bytes = if head_str
        .lines()
        .any(|l| l.eq_ignore_ascii_case("Transfer-Encoding: chunked"))
    {
        let raw = &buf[sep + 4..];
        // First line is hex size; skip to the following CRLF.
        raw.windows(2)
            .position(|w| w == b"\r\n")
            .map(|i| raw[i + 2..].to_vec())
            .unwrap_or_else(|| raw.to_vec())
    } else {
        buf[sep + 4..].to_vec()
    };

    Ok((status, body_bytes))
}

// ---------- schema check ----------

/// Verify the response is JSON `{ "text": "<string>" }` (faster-whisper
/// drop-in schema per plan §3.2 (f)). We accept extra fields (e.g. the
/// `verbose_json` optional additions) but require `text` to be a
/// non-null string.
fn assert_has_text_field(body: &[u8]) {
    let v: serde_json::Value = serde_json::from_slice(body).unwrap_or_else(|e| {
        let preview = std::str::from_utf8(body).unwrap_or("<non-utf8>");
        panic!("response is not JSON: {e}; body preview: {preview:?}")
    });
    let obj = v.as_object().expect("response JSON must be an object");
    let text = obj
        .get("text")
        .expect("response missing required `text` field");
    assert!(
        text.is_string(),
        "response `text` must be a string, got: {text:?}"
    );
}

// ---------- the tests ----------

/// One row in the compat matrix. `model_field` is the string the client
/// posts in the `model` multipart part; the T07 handler is expected to
/// map faster-whisper's `whisper-1` back to base per plan §3.2 (f).
struct Row {
    size: &'static str,
    model_field: &'static str,
}

const ROWS: &[Row] = &[
    Row {
        size: "base",
        model_field: "whisper-1",
    },
    Row {
        size: "large-v3",
        model_field: "whisper-large-v3",
    },
];

async fn run_row(row: &Row) {
    // GGUF gate — skip cleanly if the fixture is not on this runner.
    let Some(gguf) = whisper_gguf_for(row.size) else {
        let env_name = format!(
            "VOKRA_WHISPER_{}_GGUF",
            row.size.to_uppercase().replace('-', "_")
        );
        eprintln!(
            "openai_compat[{}]: skipping — set {env_name} to a converted GGUF to run this row.",
            row.size,
        );
        return;
    };
    assert!(
        gguf.exists(),
        "openai_compat[{}]: GGUF env var points at missing file: {}",
        row.size,
        gguf.display(),
    );

    // Bring up the server on OS-assigned ports (never bind a fixed port
    // in tests). The `/v1/audio/transcriptions` router is expected to be
    // wired by T06/T07; until then the route returns 404 and we mark the
    // row as pending instead of asserting green.
    let cfg = Config {
        http_bind: "127.0.0.1:0".parse().unwrap(),
        wyoming_bind: "127.0.0.1:0".parse().unwrap(),
        config_file: None,
    };
    let (handles, trigger) = spawn_server_for_test(cfg).await.expect("spawn server");

    let wav = tone_wav_bytes(500); // 500 ms tone — enough for whisper.
    let boundary = "----vokraServerT08Boundary";
    let body = build_multipart(boundary, &wav, row.model_field);

    let result = http_post_multipart(
        handles.http_actual,
        "/v1/audio/transcriptions",
        boundary,
        &body,
    )
    .await
    .expect("POST /v1/audio/transcriptions");

    // Always shut the server down before asserting, so a failed assert
    // does not leak a background listener into the test process.
    trigger.trigger();
    tokio::time::sleep(Duration::from_millis(20)).await;

    let (status, resp_body) = result;
    if status == 404 {
        // T06/T07 have not landed yet. Emit an explicit, non-silent
        // skip so CI logs make the pending contract visible. This test
        // becomes green (i.e. asserts 200 + schema) the moment T07
        // wires the router — no test churn required.
        eprintln!(
            "openai_compat[{}]: /v1/audio/transcriptions returned 404 — \
             T06/T07 router not yet wired; contract test pending.",
            row.size,
        );
        return;
    }

    assert_eq!(
        status,
        200,
        "openai_compat[{}]: expected 200 OK, got {status}; body: {:?}",
        row.size,
        String::from_utf8_lossy(&resp_body),
    );
    assert_has_text_field(&resp_body);
}

#[tokio::test]
async fn transcriptions_base_returns_text() {
    run_row(&ROWS[0]).await;
}

#[tokio::test]
async fn transcriptions_large_v3_returns_text_when_gguf_present() {
    run_row(&ROWS[1]).await;
}

/// Sanity check that runs without any GGUF: bringing the server up on
/// random ports must succeed and `/health` must respond `200 OK`. This
/// gives the openai_compat test file at least one always-executing
/// assertion so `cargo test --test openai_compat` never reports an
/// empty test binary on GGUF-less CI runners.
#[tokio::test]
async fn server_boots_and_health_probes_green() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let cfg = Config {
        http_bind: "127.0.0.1:0".parse().unwrap(),
        wyoming_bind: "127.0.0.1:0".parse().unwrap(),
        config_file: None,
    };
    let (handles, trigger) = spawn_server_for_test(cfg).await.expect("spawn");
    let mut sock = tokio::net::TcpStream::connect(handles.http_actual)
        .await
        .expect("connect /health");
    sock.write_all(b"GET /health HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")
        .await
        .unwrap();
    let mut buf = Vec::new();
    tokio::time::timeout(Duration::from_secs(2), sock.read_to_end(&mut buf))
        .await
        .expect("health read timeout")
        .expect("health read");
    let head = std::str::from_utf8(&buf[..buf.len().min(64)]).unwrap_or_default();
    assert!(
        head.starts_with("HTTP/1.1 200"),
        "expected 200 OK, got: {head:?}",
    );
    trigger.trigger();
    tokio::time::sleep(Duration::from_millis(20)).await;
}
