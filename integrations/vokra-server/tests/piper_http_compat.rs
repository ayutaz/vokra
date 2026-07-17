//! T13 — piper-plus `/api/tts` HTTP compat integration test.
//!
//! Contract (per M2-09 plan §3.2 (h), row T13):
//!
//! * A JSON `POST /api/tts` with body `{"text": "…", "voice": "…"}` (the
//!   piper-plus HTTP shape confirmed at T11) returns `200 OK` with an
//!   `audio/wav` body whose byte stream is a valid RIFF/WAVE container
//!   (magic `RIFF` … `WAVE`, `fmt `, `data` chunks).
//! * The decoded PCM samples are non-empty AND every sample is finite —
//!   no `NaN`, no `+Inf`, no `-Inf`. This is the CLAUDE.md audio-numerics
//!   invariant: TTS output that emits NaN/Inf silently is a regression
//!   (BigVGAN/Vocos in INT8 fail this — fp16 is required, and even fp16
//!   must produce finite samples end-to-end).
//! * Real weights are gated on `VOKRA_PIPER_GGUF` (voice, same env as
//!   `crates/vokra-models/benches/piper_rtf.rs`) **and**
//!   `VOKRA_WHISPER_GGUF` (whisper-base — the wired server's required
//!   startup minimum is ASR + TTS together, FR-EX-08 no half-wired boot).
//!   When either is absent the row skips cleanly with a non-silent
//!   `eprintln!` (never fake success).
//!
//! T12 landed (campaign-2 P1 #3 fix, 2026-07-17): the route is now merged
//! into `build_http_app` by the production startup path, and this test
//! boots that exact path via `spawn_server_for_test_wired` with
//! `piper_g2p = true`, so the historical "`404` = router pending" skip is
//! GONE — a 404 is now a hard regression. Plain "Hello world" text is
//! synthesized by the real 8-language G2P injected from
//! `integrations/vokra-piper-g2p`.
//!
//! Bind is `127.0.0.1:0` (OS-assigned free port); no fixed port is ever
//! opened. Parallel `cargo test` runs and constrained CI sandboxes both
//! work.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::Duration;

use vokra_server::{Config, spawn_server_for_test, spawn_server_for_test_wired};

// ---------- GGUF gating (mirrors `piper_rtf.rs`) ----------

/// Returns the piper-plus native voice GGUF path when `VOKRA_PIPER_GGUF`
/// is set, else `None` so the row can skip without asserting.
fn piper_gguf() -> Option<PathBuf> {
    std::env::var_os("VOKRA_PIPER_GGUF").map(PathBuf::from)
}

/// Returns the whisper-base GGUF path when `VOKRA_WHISPER_GGUF` is set
/// (the wired server refuses to boot TTS-only — FR-EX-08).
fn whisper_gguf() -> Option<PathBuf> {
    std::env::var_os("VOKRA_WHISPER_GGUF").map(PathBuf::from)
}

// ---------- raw HTTP JSON client ----------

/// POST `body` as `application/json` over a raw TCP connection and
/// return `(status, headers_text, body_bytes)`. A 5 s hard timeout
/// guards against a stuck handler hanging the suite.
///
/// We intentionally do NOT use `reqwest` here so that the response body
/// is delivered to us as raw bytes (no charset re-decoding), which is
/// mandatory when validating a binary `audio/wav` payload byte-for-byte.
async fn http_post_json(
    addr: SocketAddr,
    path: &str,
    body: &[u8],
) -> std::io::Result<(u16, String, Vec<u8>)> {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let mut sock = tokio::net::TcpStream::connect(addr).await?;
    let head = format!(
        "POST {path} HTTP/1.1\r\n\
         Host: 127.0.0.1\r\n\
         Content-Type: application/json\r\n\
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

    let sep = buf
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .ok_or_else(|| std::io::Error::other("no header terminator"))?;
    let head_str = std::str::from_utf8(&buf[..sep])
        .map_err(|e| std::io::Error::other(format!("utf8: {e}")))?
        .to_string();
    let first_line = head_str.lines().next().unwrap_or("");
    let status: u16 = first_line
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| std::io::Error::other(format!("bad status line: {first_line:?}")))?;

    // Body framing. Server sets `Connection: close` (client also asked
    // for it), so hyper serves a plain content-length body — treat the
    // rest verbatim. Tolerate a chunked encoding fallback the same way
    // `openai_compat.rs` does (best-effort strip of the first chunk
    // header) so a future handler variation does not silently corrupt
    // the RIFF header check.
    let body_bytes = if head_str
        .lines()
        .any(|l| l.eq_ignore_ascii_case("Transfer-Encoding: chunked"))
    {
        let raw = &buf[sep + 4..];
        raw.windows(2)
            .position(|w| w == b"\r\n")
            .map(|i| raw[i + 2..].to_vec())
            .unwrap_or_else(|| raw.to_vec())
    } else {
        buf[sep + 4..].to_vec()
    };

    Ok((status, head_str, body_bytes))
}

// ---------- WAV validation ----------

/// Parsed view of the fmt subchunk fields we care about for finite/PCM
/// validation. Only fields consumed by `assert_wav_and_samples_finite`.
#[derive(Debug)]
struct WavFmt {
    audio_format: u16, // 1 = PCM, 3 = IEEE float
    channels: u16,
    sample_rate: u32,
    bits_per_sample: u16,
    data_start: usize,
    data_len: usize,
}

/// Locate a 4-byte subchunk id inside a RIFF payload beginning at `start`.
/// Returns `Some((chunk_start_of_id, size))` where `chunk_start_of_id`
/// is the offset of the id itself and `size` is the little-endian size
/// field that follows. Walks chunks; does not assume `fmt ` precedes
/// `data`, which is spec-correct (arbitrary subchunk ordering is legal
/// in RIFF/WAVE).
fn find_subchunk(buf: &[u8], id: &[u8; 4], start: usize) -> Option<(usize, u32)> {
    let mut i = start;
    while i + 8 <= buf.len() {
        let this_id = &buf[i..i + 4];
        let size_bytes: [u8; 4] = buf[i + 4..i + 8].try_into().ok()?;
        let size = u32::from_le_bytes(size_bytes);
        if this_id == id {
            return Some((i, size));
        }
        // Chunks are word-aligned: pad to even.
        let payload = size as usize;
        let padded = payload + (payload & 1);
        // Guard against overflow / adversarial size fields.
        let next = i.checked_add(8)?.checked_add(padded)?;
        if next <= i {
            return None;
        }
        i = next;
    }
    None
}

fn parse_wav(buf: &[u8]) -> Result<WavFmt, String> {
    if buf.len() < 44 {
        return Err(format!("WAV too small: {} bytes (< 44)", buf.len()));
    }
    if &buf[0..4] != b"RIFF" {
        return Err(format!(
            "expected `RIFF` magic, got: {:?}",
            &buf[..4.min(buf.len())]
        ));
    }
    // buf[4..8] is RIFF size; tolerate any (some encoders write 0xFFFFFFFF for streams).
    if &buf[8..12] != b"WAVE" {
        return Err(format!(
            "expected `WAVE` form, got: {:?}",
            &buf[8..12.min(buf.len())]
        ));
    }

    let (fmt_off, fmt_size) =
        find_subchunk(buf, b"fmt ", 12).ok_or_else(|| "missing `fmt ` subchunk".to_string())?;
    if fmt_size < 16 {
        return Err(format!("`fmt ` subchunk too small: {fmt_size} (< 16)"));
    }
    let fmt_body = fmt_off + 8;
    if fmt_body + 16 > buf.len() {
        return Err("`fmt ` subchunk truncated".to_string());
    }
    let audio_format = u16::from_le_bytes([buf[fmt_body], buf[fmt_body + 1]]);
    let channels = u16::from_le_bytes([buf[fmt_body + 2], buf[fmt_body + 3]]);
    let sample_rate = u32::from_le_bytes([
        buf[fmt_body + 4],
        buf[fmt_body + 5],
        buf[fmt_body + 6],
        buf[fmt_body + 7],
    ]);
    let bits_per_sample = u16::from_le_bytes([buf[fmt_body + 14], buf[fmt_body + 15]]);

    let (data_off, data_size) =
        find_subchunk(buf, b"data", 12).ok_or_else(|| "missing `data` subchunk".to_string())?;
    let data_start = data_off + 8;
    let data_len = (data_size as usize).min(buf.len().saturating_sub(data_start));

    Ok(WavFmt {
        audio_format,
        channels,
        sample_rate,
        bits_per_sample,
        data_start,
        data_len,
    })
}

/// Validate RIFF/WAVE header + walk every PCM sample and assert finite
/// (no NaN, no ±Inf). Covers the two encodings piper-plus native TTS
/// can plausibly emit: 16-bit signed PCM (`audio_format == 1`) and
/// 32-bit IEEE float (`audio_format == 3`). Any other format is a
/// contract violation — we fail loudly rather than silently pass.
fn assert_wav_and_samples_finite(buf: &[u8]) {
    let wav = parse_wav(buf).unwrap_or_else(|e| {
        panic!(
            "invalid WAV: {e}; first 32 bytes: {:x?}",
            &buf[..buf.len().min(32)]
        )
    });
    assert!(
        wav.channels >= 1 && wav.channels <= 2,
        "unexpected channel count: {} (piper-plus emits mono or stereo)",
        wav.channels,
    );
    assert!(
        (8_000..=48_000).contains(&wav.sample_rate),
        "sample rate out of TTS range: {} Hz",
        wav.sample_rate,
    );
    assert!(
        wav.data_len > 0,
        "empty `data` subchunk — TTS returned no samples",
    );

    let data = &buf[wav.data_start..wav.data_start + wav.data_len];
    match (wav.audio_format, wav.bits_per_sample) {
        // PCM signed 16-bit — integer values are trivially finite; still
        // assert non-all-zero so an empty synthesis path fails.
        (1, 16) => {
            assert!(
                data.len() >= 2,
                "PCM16 body too short: {} bytes",
                data.len()
            );
            let mut any_nonzero = false;
            for chunk in data.chunks_exact(2) {
                let s = i16::from_le_bytes([chunk[0], chunk[1]]);
                if s != 0 {
                    any_nonzero = true;
                }
            }
            assert!(
                any_nonzero,
                "PCM16 body is entirely zero — TTS produced silence",
            );
        }
        // IEEE float 32-bit — the spec target for NaN/Inf validation.
        (3, 32) => {
            assert!(
                data.len() >= 4 && data.len() % 4 == 0,
                "IEEE-float body length not a multiple of 4: {}",
                data.len()
            );
            let mut any_nonzero = false;
            for (i, chunk) in data.chunks_exact(4).enumerate() {
                let s = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                assert!(
                    s.is_finite(),
                    "sample #{i} is non-finite: {s} (NaN/Inf leaked through TTS pipeline)",
                );
                if s != 0.0 {
                    any_nonzero = true;
                }
            }
            assert!(
                any_nonzero,
                "IEEE-float body is entirely zero — TTS produced silence",
            );
        }
        (fmt, bits) => {
            panic!(
                "unsupported WAV format code={fmt}, bits_per_sample={bits}; \
                 piper-plus TTS should emit PCM16 (1/16) or IEEE-float (3/32)",
            );
        }
    }
}

// ---------- test body ----------

#[tokio::test]
async fn tts_returns_valid_wav_with_finite_samples() {
    // GGUF gate — skip cleanly if the fixtures are not on this runner.
    let (Some(gguf), Some(whisper)) = (piper_gguf(), whisper_gguf()) else {
        eprintln!(
            "piper_http_compat: skipping — set VOKRA_PIPER_GGUF=<voice.gguf> AND \
             VOKRA_WHISPER_GGUF=<whisper-base.gguf> to exercise the /api/tts \
             contract on the wired production path."
        );
        return;
    };
    assert!(
        gguf.exists(),
        "piper_http_compat: VOKRA_PIPER_GGUF points at missing file: {}",
        gguf.display(),
    );
    assert!(
        whisper.exists(),
        "piper_http_compat: VOKRA_WHISPER_GGUF points at missing file: {}",
        whisper.display(),
    );

    // Full production startup path: eager engine loads + real-G2P injection
    // (`piper_g2p = true` so plain "Hello world" is really phonemized by the
    // 8-language G2P, not rejected by the passthrough).
    let cfg = Config {
        http_bind: "127.0.0.1:0".parse().unwrap(),
        wyoming_bind: "127.0.0.1:0".parse().unwrap(),
        config_file: None,
        whisper_base_gguf: Some(whisper),
        piper_plus_gguf: Some(gguf),
        piper_g2p: Some(true),
        ..Config::default()
    };
    let (handles, trigger) = spawn_server_for_test_wired(cfg)
        .await
        .expect("spawn server");

    // piper-plus HTTP shape (T11 confirms the exact field set). The
    // `text` field is the only required input; `voice` is included
    // because the piper-plus reference server treats voice selection as
    // mandatory once >1 voice is registered.
    let body = br#"{"text":"Hello world","voice":"default"}"#;

    let result = http_post_json(handles.http_actual, "/api/tts", body).await;

    // Always tear the server down before asserting so a failure does
    // not leak the background listener into the process.
    trigger.trigger();
    tokio::time::sleep(Duration::from_millis(20)).await;

    let (status, headers, resp_body) = result.expect("POST /api/tts");

    // T12 is landed: a 404 here is a wiring REGRESSION (the campaign-2
    // server-real leg live-verified exactly this failure), not a pending
    // contract.
    assert_eq!(
        status,
        200,
        "piper_http_compat: expected 200 OK, got {status}; body preview: {:?}",
        String::from_utf8_lossy(&resp_body[..resp_body.len().min(256)]),
    );

    // Content-Type must announce a WAV payload. Accept the two common
    // spellings; a JSON error response would fail this check.
    let ct_is_wav = headers.lines().any(|l| {
        let lower = l.to_ascii_lowercase();
        lower.starts_with("content-type:")
            && (lower.contains("audio/wav") || lower.contains("audio/x-wav"))
    });
    assert!(
        ct_is_wav,
        "piper_http_compat: response is not audio/wav; headers:\n{headers}",
    );

    assert_wav_and_samples_finite(&resp_body);
}

/// Sanity boot assertion so `cargo test --test piper_http_compat` never
/// reports an empty binary on a runner without `VOKRA_PIPER_GGUF`
/// (mirrors the `server_boots_and_health_probes_green` guard in
/// `openai_compat.rs`).
#[tokio::test]
async fn server_boots_and_health_probes_green() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let cfg = Config {
        http_bind: "127.0.0.1:0".parse().unwrap(),
        wyoming_bind: "127.0.0.1:0".parse().unwrap(),
        config_file: None,
        ..Config::default()
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
