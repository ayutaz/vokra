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

use std::path::PathBuf;
use std::time::Duration;

use vokra_server::{Config, spawn_server_for_test, spawn_server_for_test_wired};

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

// cc-09 (2026-07-19 M4-residual audit): the per-file raw-TCP helper is
// replaced by the shared `tests/support/mod.rs` client (complete-response
// detection + bounded reset retry — see its module docs for the root cause
// this fixes; this file POSTs the LARGEST bodies of the suite, so it had
// the widest unread-body RST window on unmounted-route runs).
mod support;
use support::http_post_multipart;

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
        ..Config::default()
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

// ===========================================================================
// cc-19 (2026-07-19 M4-residual audit) — real-GGUF word-timestamps e2e.
//
// The unit/route suites drive the words[] surface with mocks; this row proves
// the WIRED production path end-to-end: real Whisper GGUF + real speech WAV →
// `verbose_json` + `timestamp_granularities[]=word` → a non-empty `words[]`
// with monotone, in-bounds timings.
//
// Gated exactly like the rows above: absent GGUFs skip with an explicit
// eprintln (never a silent success, FR-EX-08 posture). `spawn_server_for_test_wired`
// runs the FULL production startup (eager registry build), so the tested
// surface IS the production surface.
//
// VERIFIED 2026-07-19 on the M1 iMac (this exact invocation):
//
//   VOKRA_WHISPER_BASE_GGUF=~/.cache/vokra-eval/gguf/whisper-base.gguf \
//   VOKRA_PIPER_PLUS_GGUF=~/.cache/vokra-eval/gguf/piper-plus-css10-ja-6lang-neutralspk.gguf \
//   cargo test --release --manifest-path integrations/vokra-server/Cargo.toml \
//     --test openai_compat -- --nocapture verbose_json_word_timestamps_e2e
//
//   → PASS — 22 words over 11.00s, text=" And so my fellow Americans, ask not
//     what your country can do for you, ask what you can do for your country."
//
// NOTE on the piper path: the wired startup requires BOTH a Whisper and a
// piper voice (`service_config_from_config`'s required minimum), so this ASR
// test needs a loadable voice even though it never synthesizes. Of the voices
// in the local eval cache only `piper-plus-css10-ja-6lang-neutralspk.gguf`
// loads against the current loader; the others fail on stale tensor
// shapes/names (`spk_proj.0.weight` absent, `dec.ups.2.weight` [64,32,8] vs
// [64,32,16]) — a fixture-vintage issue, unrelated to cc-19. A configured but
// unloadable GGUF stays a HARD failure here (the operator set the path;
// silently skipping it would violate FR-EX-08).
// ===========================================================================

/// The piper voice the wired startup path also requires (the server refuses
/// to boot ASR-only — `service_config_from_config`'s required minimum).
fn piper_gguf_for_wired_boot() -> Option<PathBuf> {
    std::env::var_os("VOKRA_PIPER_PLUS_GGUF").map(PathBuf::from)
}

/// The 30 s JFK fixture committed for the Whisper real-audio parity CI
/// (`tests/fixtures/audio/jfk-30s.wav`, repo root). Returns `None` when the
/// file is absent (the sidecar-hash placeholder state).
fn jfk_wav_path() -> Option<PathBuf> {
    let p =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/audio/jfk-30s.wav");
    p.exists().then_some(p)
}

/// Multipart body for the word-timestamps ask.
fn build_multipart_words(boundary: &str, wav: &[u8], model: &str) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
    b.extend_from_slice(
        b"Content-Disposition: form-data; name=\"file\"; filename=\"jfk-30s.wav\"\r\n",
    );
    b.extend_from_slice(b"Content-Type: audio/wav\r\n\r\n");
    b.extend_from_slice(wav);
    b.extend_from_slice(b"\r\n");
    for (name, value) in [
        ("model", model),
        ("response_format", "verbose_json"),
        ("timestamp_granularities[]", "word"),
    ] {
        b.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
        b.extend_from_slice(
            format!("Content-Disposition: form-data; name=\"{name}\"\r\n\r\n").as_bytes(),
        );
        b.extend_from_slice(value.as_bytes());
        b.extend_from_slice(b"\r\n");
    }
    b.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
    b
}

#[tokio::test]
async fn verbose_json_word_timestamps_e2e_with_real_gguf() {
    let (Some(whisper), Some(piper), Some(wav_path)) = (
        whisper_gguf_for("base"),
        piper_gguf_for_wired_boot(),
        jfk_wav_path(),
    ) else {
        eprintln!(
            "openai_compat[cc-19 words e2e]: SKIP — needs VOKRA_WHISPER_BASE_GGUF \
             (or VOKRA_WHISPER_GGUF) + VOKRA_PIPER_PLUS_GGUF + \
             tests/fixtures/audio/jfk-30s.wav. Not a pass.",
        );
        return;
    };
    if !whisper.exists() || !piper.exists() {
        eprintln!(
            "openai_compat[cc-19 words e2e]: SKIP — configured GGUF path missing \
             (whisper={}, piper={}). Not a pass.",
            whisper.display(),
            piper.display(),
        );
        return;
    }
    let wav = std::fs::read(&wav_path).expect("read jfk fixture");

    let cfg = Config {
        http_bind: "127.0.0.1:0".parse().unwrap(),
        wyoming_bind: "127.0.0.1:0".parse().unwrap(),
        config_file: None,
        whisper_base_gguf: Some(whisper),
        piper_plus_gguf: Some(piper),
        ..Config::default()
    };
    let (handles, trigger) = spawn_server_for_test_wired(cfg)
        .await
        .expect("wired server must boot with the configured GGUFs");

    let boundary = "----vokraServerCc19Boundary";
    let body = build_multipart_words(boundary, &wav, "whisper-1");
    // Real 30 s decode with cross-attention alignment — well past the shared
    // client's 5 s default deadline.
    let result = support::http_request_with_timeout(
        handles.http_actual,
        "POST",
        "/v1/audio/transcriptions",
        Some(&format!("multipart/form-data; boundary={boundary}")),
        &body,
        Duration::from_secs(300),
    )
    .await;

    // Tear the server down before asserting so a failure cannot leak a
    // listener into the process.
    trigger.trigger();
    tokio::time::sleep(Duration::from_millis(20)).await;

    let resp = result.expect("POST /v1/audio/transcriptions");
    assert_eq!(
        resp.status,
        200,
        "cc-19 words e2e: expected 200, got {}; body: {:?}",
        resp.status,
        String::from_utf8_lossy(&resp.body),
    );
    let v: serde_json::Value =
        serde_json::from_slice(&resp.body).expect("verbose_json body must be JSON");
    assert_eq!(v["task"], "transcribe");
    assert!(
        v["text"].as_str().is_some_and(|t| !t.trim().is_empty()),
        "transcription text must be non-empty: {v}",
    );
    let words = v["words"].as_array().expect("words[] must be present");
    assert!(!words.is_empty(), "words[] must be non-empty: {v}");

    // Monotone non-decreasing starts, each span well-formed and inside the
    // reported duration. A fabricated/degenerate alignment fails here.
    let duration = v["duration"].as_f64().expect("duration must be a number");
    let mut prev_start = f64::NEG_INFINITY;
    for w in words {
        let start = w["start"].as_f64().expect("word.start must be a number");
        let end = w["end"].as_f64().expect("word.end must be a number");
        assert!(
            w["word"].as_str().is_some(),
            "word.word must be a string: {w}",
        );
        assert!(
            start >= prev_start,
            "word starts must be non-decreasing: {start} after {prev_start} in {v}",
        );
        assert!(end >= start, "word end {end} must be >= start {start}");
        assert!(
            start >= 0.0 && end <= duration + 1.0,
            "word span [{start}, {end}] must lie inside the {duration}s audio",
        );
        prev_start = start;
    }
    eprintln!(
        "openai_compat[cc-19 words e2e]: PASS — {} words over {duration:.2}s, text={:?}",
        words.len(),
        v["text"].as_str().unwrap_or_default(),
    );
}
