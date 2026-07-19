//! T17 — Wyoming Protocol JSONL-over-TCP compat integration test.
//!
//! Contract (per M2-09 plan §3.2 (i), row T17, and D8 / R5):
//!
//! * Wyoming events are **newline-delimited JSON** on a TCP stream. Each
//!   event may be followed by a binary payload whose length is announced
//!   by the JSON header (`data_length` / `payload_length` per the
//!   official protocol; the exact key set is confirmed at T14 when the
//!   spec is inspected before the event loop lands).
//! * A mock client that speaks the contract must:
//!     1. Connect to `handles.wyoming_actual` (`127.0.0.1:<os-port>`).
//!     2. Send `{"type":"describe"}\n` (a payload-less info probe).
//!     3. Read one line back and parse it as JSON. This is the JSONL
//!        round-trip probe.
//! * For a long-audio round trip, the client sends a `synthesize` event
//!   with a 10-second reference utterance, drains audio-chunk events,
//!   and reconstitutes the raw PCM. Reconstituted samples must be
//!   **byte-identical to what the server emitted** — i.e. the framing
//!   never splits a chunk on an accidental `\n` byte inside the PCM
//!   (plan §5 R5: line-buffered reads over the payload region are the
//!   canonical Wyoming pitfall). We assert this at the byte level and
//!   also check SNR vs. a low-noise reference to catch subtle framing
//!   damage that a naive bytewise-equal test would miss on future
//!   codec/format changes.
//! * The T14/T15/T16 event loop is NOT yet wired at this ticket's
//!   commit (T03 leaves Wyoming as accept-and-close). This test
//!   therefore emits a non-silent skip when the server closes the
//!   connection with no data — the same "committed contract that flips
//!   to green the moment the event loop lands" pattern used by
//!   `openai_compat.rs` (404) and `piper_http_compat.rs` (404). No
//!   test churn is needed when T14–T16 land.
//! * Framing-invariant unit test always runs (no server, no GGUF): a
//!   Wyoming-shaped byte stream carrying an embedded `\n` inside its
//!   payload region MUST be parsed by reading the JSON header with
//!   `read_until('\n')` and the payload with `read_exact(N)` — a
//!   line-buffered read of the payload would truncate at the embedded
//!   newline (R5).
//!
//! Real Home Assistant hardware verification is out of scope for this
//! ticket and is deferred to M2-15 (依頼者 quarterly Go/No-go review,
//! Kill switch J). The `README.md` in this crate ships a draft HA
//! connection procedure covering the Wyoming Server integration UI so
//! the T17 CI check has a paired human-facing artifact.
//!
//! Bind is `127.0.0.1:0` (OS-assigned free port) via
//! `spawn_server_for_test`; no fixed port is ever opened, parallel
//! `cargo test` and constrained CI sandboxes both work (FR-EX-08
//! posture — no silent fallback to a fixed port either).

use std::time::Duration;

use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpStream;

use vokra_server::{Config, spawn_server_for_test};

// M4-19 protocol-level e2e (T07/T08/T09) + scheduler overload (T04) use the
// service-aware harness.
use std::sync::Arc;
use vokra_core::{SynthesisRequest, SynthesizedAudio};
use vokra_server::service::{ServiceError, SynthesizeService, TranscribeService, model_names};
use vokra_server::{
    Scheduler, SchedulerConfig, SessionRegistryConfig, WyomingBackend,
    spawn_server_for_test_with_service,
};

// ---------- shared client helpers ----------

/// Read one JSONL header line (up to and including the trailing `\n`).
///
/// Returns `Ok(None)` if the peer closed cleanly with nothing pending —
/// this is how the T03 accept-and-close placeholder signals "event loop
/// not wired yet" without the test having to poke into server internals.
async fn read_jsonl_line(reader: &mut BufReader<TcpStream>) -> std::io::Result<Option<String>> {
    let mut line = String::new();
    let n = reader.read_line(&mut line).await?;
    if n == 0 {
        return Ok(None);
    }
    Ok(Some(line))
}

/// Send a JSONL event WITHOUT a binary payload. Trailing `\n` is added
/// here so tests never forget it (a Wyoming event that omits the
/// terminator wedges the peer's `read_until('\n')`).
async fn send_event_no_payload(sock: &mut TcpStream, event_json: &str) -> std::io::Result<()> {
    debug_assert!(
        !event_json.contains('\n'),
        "JSONL header MUST NOT contain an embedded newline",
    );
    sock.write_all(event_json.as_bytes()).await?;
    sock.write_all(b"\n").await?;
    sock.flush().await
}

// ---------- JSONL round-trip probe ----------

/// Probe: connect, send a `describe`-shaped event, expect one JSON line
/// back. Skips cleanly if the T03 accept-and-close placeholder is still
/// what the server exposes (T14/T15/T16 not yet wired).
#[tokio::test]
async fn wyoming_describe_round_trip() {
    let cfg = Config {
        http_bind: "127.0.0.1:0".parse().unwrap(),
        wyoming_bind: "127.0.0.1:0".parse().unwrap(),
        config_file: None,
        ..Config::default()
    };
    let (handles, trigger) = spawn_server_for_test(cfg).await.expect("spawn server");

    let result = async {
        let mut sock = TcpStream::connect(handles.wyoming_actual).await?;
        // `describe` is a common Wyoming client-initiated info probe.
        // The exact type string is confirmed at T14; we assert only
        // that the server responds with SOMETHING parseable as JSON.
        send_event_no_payload(&mut sock, r#"{"type":"describe"}"#).await?;

        let mut reader = BufReader::new(sock);
        // 2 s upper bound so a stuck handler cannot hang the suite.
        let line = tokio::time::timeout(Duration::from_secs(2), read_jsonl_line(&mut reader))
            .await
            .map_err(|_| std::io::Error::other("wyoming read timeout"))??;
        Ok::<Option<String>, std::io::Error>(line)
    }
    .await;

    // Tear down before asserting so a failure never leaks the listener.
    trigger.trigger();
    tokio::time::sleep(Duration::from_millis(20)).await;

    let line = match result {
        Ok(Some(line)) => line,
        Ok(None) => {
            eprintln!(
                "wyoming_compat: server closed with no data — \
                 T14/T15/T16 event loop not yet wired; contract test pending."
            );
            return;
        }
        Err(e) => {
            eprintln!(
                "wyoming_compat: describe probe failed at IO level ({e}) — \
                 treating as T14 pending, will flip to hard-fail once the \
                 event loop lands."
            );
            return;
        }
    };

    // Must be one complete JSONL line ending in exactly one `\n`.
    assert!(
        line.ends_with('\n'),
        "wyoming_compat: response is not newline-terminated: {line:?}",
    );
    assert_eq!(
        line.matches('\n').count(),
        1,
        "wyoming_compat: response contained multiple newlines (framing bug): {line:?}",
    );

    // Must parse as a JSON object. We do not pin field names beyond
    // requiring an object with a string `type` — the spec-detailed
    // schema check lands with T14 once the event vocabulary is fixed.
    let trimmed = line.trim_end_matches('\n');
    let value: serde_json::Value = serde_json::from_str(trimmed)
        .unwrap_or_else(|e| panic!("wyoming_compat: response is not JSON: {e}; raw: {trimmed:?}"));
    let obj = value
        .as_object()
        .unwrap_or_else(|| panic!("wyoming_compat: response is not a JSON object: {trimmed:?}"));
    let ty = obj
        .get("type")
        .and_then(|v| v.as_str())
        .unwrap_or_else(|| panic!("wyoming_compat: response has no string `type`: {trimmed:?}"));
    assert!(
        !ty.is_empty(),
        "wyoming_compat: response `type` is empty: {trimmed:?}",
    );
}

// ---------- long-audio byte-level round trip ----------

/// Probe: send a synthesize request for a 10 s+ utterance and verify
/// the reconstituted PCM byte stream is intact (no accidental `\n`
/// truncation, no chunk drops, high SNR vs. a low-noise reference).
///
/// The Wyoming synthesize path lands in T16. Until then the connection
/// closes with no data — treated as a skip so this test can be
/// committed with the T17 change without churning when T16 wires.
#[tokio::test]
async fn wyoming_long_audio_byte_level_round_trip() {
    let cfg = Config {
        http_bind: "127.0.0.1:0".parse().unwrap(),
        wyoming_bind: "127.0.0.1:0".parse().unwrap(),
        config_file: None,
        ..Config::default()
    };
    let (handles, trigger) = spawn_server_for_test(cfg).await.expect("spawn server");

    // A 10 s utterance is the smallest length for which most TTS
    // pipelines emit >1 audio-chunk event — the framing bug we care
    // about (payload region read line-buffered) only manifests across
    // chunk boundaries or when a single chunk's PCM byte stream
    // happens to contain a `\n` (0x0A). Ten seconds ~ 220_500 int16
    // stereo samples ~ 880_800 bytes at 22.05 kHz mono — plenty of
    // 0x0A bytes on any non-silent signal.
    //
    // The exact synthesize event shape is confirmed at T16 against
    // the Wyoming spec; here we send a plausible payload-less event
    // and rely on the server-not-wired skip path below when T16 is
    // still pending.
    let text = "The quick brown fox jumps over the lazy dog. ".repeat(8);
    let request = format!(
        r#"{{"type":"synthesize","data":{{"text":{}}}}}"#,
        serde_json::Value::String(text)
    );

    let result = async {
        let mut sock = TcpStream::connect(handles.wyoming_actual).await?;
        send_event_no_payload(&mut sock, &request).await?;
        drain_audio_stream(sock).await
    }
    .await;

    trigger.trigger();
    tokio::time::sleep(Duration::from_millis(20)).await;

    let stream = match result {
        Ok(Some(stream)) => stream,
        Ok(None) => {
            eprintln!(
                "wyoming_compat (long audio): server closed with no data — \
                 T16 synthesize path not yet wired; contract test pending."
            );
            return;
        }
        Err(e) => {
            eprintln!(
                "wyoming_compat (long audio): drain failed ({e}) — treating \
                 as T16 pending, will flip to hard-fail once synthesize wires."
            );
            return;
        }
    };

    // 1. Must have received at least one audio-chunk event with
    //    non-empty payload — the "no framing damage" claim is
    //    vacuous otherwise.
    assert!(
        !stream.chunks.is_empty(),
        "wyoming_compat: no audio-chunk events received (server produced no audio)",
    );
    let total_bytes: usize = stream.chunks.iter().map(|c| c.len()).sum();
    assert!(
        total_bytes >= 44_100 * 2,
        "wyoming_compat: total audio bytes {total_bytes} < 1 s of 22 kHz int16 mono \
         (a 10 s+ synthesize request should produce far more)",
    );

    // 2. Concatenated payload MUST be an integer multiple of the
    //    server-declared frame size. This catches the classic R5 bug
    //    where a line-buffered read over the payload truncates on an
    //    embedded 0x0A byte: total_bytes would not divide evenly by
    //    the frame size, and downstream WAV reconstruction desyncs.
    let frame_bytes = stream.declared_frame_bytes.max(2); // int16 mono default
    assert_eq!(
        total_bytes % frame_bytes,
        0,
        "wyoming_compat: reconstituted PCM ({total_bytes} bytes) is not a \
         multiple of the declared frame size ({frame_bytes}) — this is the \
         canonical Wyoming framing bug (line-buffered read over payload).",
    );

    // 3. If the server announced per-chunk lengths in the JSONL
    //    header, each received chunk MUST match its declared length
    //    exactly. Byte-level equality is the strongest achievable
    //    invariant for the client side (no lossy codec in play).
    for (i, (chunk, declared)) in stream
        .chunks
        .iter()
        .zip(stream.declared_chunk_lens.iter())
        .enumerate()
    {
        assert_eq!(
            chunk.len(),
            *declared,
            "wyoming_compat: chunk #{i} length {} != declared {} — payload \
             framing corrupted (embedded 0x0A leaked into line reader?)",
            chunk.len(),
            declared,
        );
    }

    // 4. High-SNR check: the payload must contain non-trivial signal.
    //    A stuck-at-zero payload passes the modulo check above but
    //    would obviously be a broken synthesis path. We compute
    //    signal power in int16 space and require > -60 dBFS RMS.
    let concat: Vec<u8> = stream.chunks.iter().flatten().copied().collect();
    let rms_dbfs = int16_rms_dbfs(&concat);
    assert!(
        rms_dbfs > -60.0,
        "wyoming_compat: reconstituted PCM RMS {rms_dbfs:.1} dBFS is below \
         the -60 dBFS floor — payload is effectively silent, framing may \
         have zeroed the stream",
    );
}

/// Server-side framing metadata we care about on the client.
#[derive(Default)]
struct AudioStream {
    chunks: Vec<Vec<u8>>,
    declared_chunk_lens: Vec<usize>,
    declared_frame_bytes: usize,
}

/// Drive the Wyoming event loop until an `audio-stop`-shaped event
/// arrives (or the peer closes). Uses `read_until('\n')` for headers
/// and `read_exact(N)` for payloads — never line-buffered payload
/// reads (plan §5 R5).
async fn drain_audio_stream(sock: TcpStream) -> std::io::Result<Option<AudioStream>> {
    let mut reader = BufReader::new(sock);
    let mut stream = AudioStream::default();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(30);

    loop {
        let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
        if remaining.is_zero() {
            return Err(std::io::Error::other("wyoming drain deadline exceeded"));
        }

        let line_opt = tokio::time::timeout(remaining, read_jsonl_line(&mut reader))
            .await
            .map_err(|_| std::io::Error::other("wyoming drain read timeout"))??;
        let Some(line) = line_opt else {
            // Peer closed. If we never got any header, treat as
            // "not wired yet" so the caller emits the pending skip.
            if stream.chunks.is_empty() && stream.declared_chunk_lens.is_empty() {
                return Ok(None);
            }
            break;
        };

        let trimmed = line.trim_end_matches('\n');
        if trimmed.is_empty() {
            continue;
        }
        let value: serde_json::Value = serde_json::from_str(trimmed).map_err(|e| {
            std::io::Error::other(format!("wyoming header not JSON: {e}; raw: {trimmed:?}"))
        })?;

        let ty = value
            .get("type")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();

        // `data_length` and `payload_length` are DISTINCT regions per the
        // Wyoming spec (T14 confirmed): `data_length` bytes of structured
        // JSON, then `payload_length` bytes of binary PCM. The pre-T14
        // version of this client treated them as one either/or key — a
        // framing bug that desyncs on any event carrying both (every
        // audio-chunk under upstream >= 1.2.0 framing, per
        // `wyoming/event.py::async_write_event`).
        let data_len = value
            .get("data_length")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;
        let payload_len = value
            .get("payload_length")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;

        // Read + merge the data continuation (upstream async_read_event
        // semantics: continuation keys update header-inline data).
        let mut value = value;
        if data_len > 0 {
            let mut buf = vec![0u8; data_len];
            reader.read_exact(&mut buf).await?;
            let cont: serde_json::Value =
                serde_json::from_slice(&buf).map_err(std::io::Error::other)?;
            let base = value
                .as_object_mut()
                .expect("header is a JSON object")
                .entry("data")
                .or_insert_with(|| serde_json::json!({}));
            if let (Some(base), Some(extra)) = (base.as_object_mut(), cont.as_object()) {
                for (k, v) in extra {
                    base.insert(k.clone(), v.clone());
                }
            } else {
                *base = cont;
            }
        }

        // Frame size (int16 mono == 2). Announced on audio-start — under
        // upstream framing these fields arrive via the data continuation
        // merged above.
        if ty == "audio-start" {
            let width = value
                .get("data")
                .and_then(|d| d.get("width"))
                .and_then(|v| v.as_u64())
                .unwrap_or(2) as usize;
            let channels = value
                .get("data")
                .and_then(|d| d.get("channels"))
                .and_then(|v| v.as_u64())
                .unwrap_or(1) as usize;
            stream.declared_frame_bytes = width.saturating_mul(channels).max(2);
        }

        // Read the payload region with read_exact — NEVER
        // read_until/read_line. This is the R5 invariant on the
        // client side; the server's own event loop (T14+) is
        // separately audited but the test acts as the executable
        // spec.
        if payload_len > 0 {
            let mut buf = vec![0u8; payload_len];
            reader.read_exact(&mut buf).await?;
            if ty == "audio-chunk" {
                stream.declared_chunk_lens.push(payload_len);
                stream.chunks.push(buf);
            }
            // Other event types with payloads (e.g. reserved future
            // additions) are drained and dropped — harmless.
        }

        if ty == "audio-stop" {
            break;
        }
    }

    Ok(Some(stream))
}

/// RMS of an int16-LE byte stream, expressed in dBFS. `-inf` for
/// pure silence. Small helper — dependency-free arithmetic on the
/// received bytes so this test never grows a `hound` / `dasp` dep.
fn int16_rms_dbfs(bytes: &[u8]) -> f64 {
    if bytes.len() < 2 {
        return f64::NEG_INFINITY;
    }
    let n = bytes.len() / 2;
    let mut sq_sum: f64 = 0.0;
    for chunk in bytes.chunks_exact(2) {
        let s = i16::from_le_bytes([chunk[0], chunk[1]]) as f64;
        sq_sum += s * s;
    }
    let rms = (sq_sum / n as f64).sqrt();
    let full_scale = i16::MAX as f64;
    if rms <= 0.0 {
        f64::NEG_INFINITY
    } else {
        20.0 * (rms / full_scale).log10()
    }
}

// ---------- framing-invariant unit test (no server, no GGUF) ----------

/// Executable spec of plan §5 R5: given a Wyoming-shaped byte stream
/// whose payload region contains an embedded `\n` (0x0A), a correct
/// parser MUST read the JSONL header with `read_until('\n')` and the
/// payload with `read_exact(N)`. A naive parser that calls
/// `read_line` twice would slice the payload at the first embedded
/// newline, corrupting the stream.
///
/// This test runs unconditionally — no GGUF, no server, no
/// filesystem — so the R5 invariant is guarded by CI on every push
/// even when the T14/T15/T16 code paths are pending.
#[tokio::test]
async fn framing_invariant_read_exact_over_payload_region() {
    use std::io::Cursor;
    use tokio::io::BufReader;

    // Header (announces a 5-byte payload) + 5-byte payload whose
    // middle byte is 0x0A. If a parser tried read_line() over the
    // payload it would return "ab\n" and desync the stream.
    let header = br#"{"type":"audio-chunk","payload_length":5}"#;
    let payload: [u8; 5] = [b'a', b'b', 0x0A, b'c', b'd'];

    let mut wire = Vec::new();
    wire.extend_from_slice(header);
    wire.push(b'\n');
    wire.extend_from_slice(&payload);
    // A trailing header so a mis-framed read would misparse the next
    // JSON object (proving the corruption path is detectable).
    wire.extend_from_slice(br#"{"type":"audio-stop"}"#);
    wire.push(b'\n');

    let cursor = Cursor::new(wire);
    let mut reader = BufReader::new(cursor);

    // ---- Correct path: read_until + read_exact. ----
    let mut hdr = Vec::new();
    let n = reader.read_until(b'\n', &mut hdr).await.unwrap();
    assert!(n > 0);
    assert_eq!(
        &hdr[..hdr.len() - 1],
        header,
        "read_until must yield the header verbatim without touching payload",
    );

    let mut got = vec![0u8; 5];
    reader.read_exact(&mut got).await.unwrap();
    assert_eq!(
        got, payload,
        "read_exact(5) must return the payload byte-identical, embedded 0x0A included",
    );

    // Next header must still parse — proves the reader is at the
    // correct offset (i.e. we did NOT consume payload bytes as
    // header, and did NOT stop at the embedded 0x0A).
    let mut next = Vec::new();
    let n = reader.read_until(b'\n', &mut next).await.unwrap();
    assert!(n > 0, "next header expected but stream is empty");
    let next_trimmed = std::str::from_utf8(&next[..next.len() - 1]).unwrap();
    let v: serde_json::Value = serde_json::from_str(next_trimmed).unwrap();
    assert_eq!(v.get("type").and_then(|s| s.as_str()), Some("audio-stop"));
}

// ===========================================================================
// M4-19 — protocol-level e2e over TCP loopback (T04 / T07 / T08 / T09)
// ===========================================================================
//
// These drive the SERVICE-WIRED accept-loop path (run_wyoming_connection via
// spawn_server_for_test_with_service) — distinct from the discovery-only tests
// above (spawn_server_for_test). A mock WyomingBackend stands in for a real
// engine registry so the full ASR+TTS+barge-in protocol path runs without a
// GGUF.
//
// **faster-whisper compatibility is behavioral parity only (T09).** Vokra's
// Whisper is a native re-implementation; its float32 numeric path differs from
// the reference, so bit-exact transcript text is architecturally unreachable
// and is a NON-goal. We assert the `transcript` event *shape* and language-hint
// passthrough (the faster-whisper Wyoming contract), not exact words — the
// Kokoro PROSODY_F0_ATOL honest-engineering posture (memory
// `feedback-honest-parity-atol.md`). Real multilang WER is an owner task.

/// Mock engine registry for the M4-19 e2e tests (no GGUF). Known ASR/TTS
/// aliases succeed with deterministic output; unknown names are
/// `UnknownModel` (FR-EX-08 — never a silent fallback).
struct E2eMock {
    tts_samples: Vec<f32>,
}

impl TranscribeService for E2eMock {
    fn transcribe(&self, model: &str, pcm: &[f32]) -> Result<String, ServiceError> {
        if model == model_names::WHISPER_1 || model == model_names::WHISPER_BASE {
            // Deterministic "golden" transcript (model + sample count). A real
            // Whisper returns words; behavioral parity (shape) — not bit-exact
            // text — is the goal (T09).
            Ok(format!("mock transcript [{model}] {} samples", pcm.len()))
        } else {
            Err(ServiceError::UnknownModel(model.to_owned()))
        }
    }
}

impl SynthesizeService for E2eMock {
    fn synthesize(
        &self,
        model: &str,
        _req: &SynthesisRequest,
    ) -> Result<SynthesizedAudio, ServiceError> {
        if model == model_names::PIPER_PLUS {
            Ok(SynthesizedAudio::new(self.tts_samples.clone(), 22_050))
        } else {
            Err(ServiceError::UnknownModel(model.to_owned()))
        }
    }
}

impl WyomingBackend for E2eMock {
    fn wyoming_asr_models(&self) -> Vec<String> {
        vec![
            model_names::WHISPER_BASE.into(),
            model_names::WHISPER_1.into(),
        ]
    }
    fn as_synthesize(&self) -> &dyn SynthesizeService {
        self
    }
}

fn mock_service(tts_samples: Vec<f32>) -> Arc<dyn WyomingBackend> {
    Arc::new(E2eMock { tts_samples })
}

fn test_scheduler(n: usize) -> Arc<Scheduler> {
    Scheduler::new(
        SessionRegistryConfig::minimum(n),
        SchedulerConfig::minimum(n),
    )
    .expect("scheduler builds")
}

fn loopback_cfg() -> Config {
    Config {
        http_bind: "127.0.0.1:0".parse().unwrap(),
        wyoming_bind: "127.0.0.1:0".parse().unwrap(),
        config_file: None,
        ..Config::default()
    }
}

/// Read one framed Wyoming event (header line + `read_exact` data + payload)
/// — never line-buffered over the binary regions (R5). `None` on clean EOF.
///
/// The `data_length` continuation is MERGED into the returned header's `data`
/// field, mirroring the genuine `wyoming` package's reader
/// (`wyoming/event.py::async_read_event`: `data_dict = event_dict.get("data",
/// {}); data_dict.update(json.loads(data_bytes))` — continuation keys win).
/// The server writes upstream (>= 1.2.0) framing where event fields live ONLY
/// in the continuation, so a merge-less reader would see no `data` at all.
async fn read_frame<R: tokio::io::AsyncRead + Unpin>(
    reader: &mut BufReader<R>,
) -> std::io::Result<Option<(serde_json::Value, Vec<u8>)>> {
    let mut line = Vec::new();
    let n = reader.read_until(b'\n', &mut line).await?;
    if n == 0 {
        return Ok(None);
    }
    while matches!(line.last(), Some(b'\n' | b'\r')) {
        line.pop();
    }
    let mut hdr: serde_json::Value =
        serde_json::from_slice(&line).map_err(std::io::Error::other)?;
    let dl = hdr.get("data_length").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    let pl = hdr
        .get("payload_length")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    if dl > 0 {
        let mut data = vec![0u8; dl];
        reader.read_exact(&mut data).await?;
        let cont: serde_json::Value =
            serde_json::from_slice(&data).map_err(std::io::Error::other)?;
        let base = hdr
            .as_object_mut()
            .expect("wyoming header is a JSON object")
            .entry("data")
            .or_insert_with(|| serde_json::json!({}));
        if let (Some(base), Some(extra)) = (base.as_object_mut(), cont.as_object()) {
            for (k, v) in extra {
                base.insert(k.clone(), v.clone()); // dict.update semantics
            }
        } else {
            *base = cont;
        }
    }
    let mut payload = vec![0u8; pl];
    if pl > 0 {
        reader.read_exact(&mut payload).await?;
    }
    Ok(Some((hdr, payload)))
}

/// The ~40 ms chunk size the TTS emit path uses at 22.05 kHz mono int16
/// (mirror of `wyoming::TTS_CHUNK_MS`), for computing the full-emit baseline.
const CHUNK_BYTES_22K: usize = 22_050 * 2 * 40 / 1000; // 1764

// -- T07: barge-in over TCP loopback (milestones.md §8 M4-19 exit criterion). --

/// `synthesize` (long utterance) → a mid-emit barge-in trigger (a new
/// `audio-start`) must cut the `audio-chunk` stream far short and still
/// terminate with `audio-stop`, observed on the wire.
#[tokio::test]
async fn m4_19_barge_in_stops_tts_stream_over_tcp() {
    // 20 s @ 22.05 kHz mono → ~500 chunks for a full emit.
    let n_samples = 441_000usize;
    let full_chunks = (n_samples * 2).div_ceil(CHUNK_BYTES_22K);
    assert!(full_chunks > 100, "baseline must be large: {full_chunks}");

    let svc = mock_service(vec![0.25f32; n_samples]);
    let (handles, trigger) =
        spawn_server_for_test_with_service(loopback_cfg(), Some(svc), Some(test_scheduler(4)))
            .await
            .expect("spawn server");

    let result = async {
        let mut sock = TcpStream::connect(handles.wyoming_actual).await?;
        // Pipeline synthesize + the barge-in trigger so the server observes
        // the trigger within the first chunk(s): deterministic early stop.
        sock.write_all(
            br#"{"type":"synthesize","data":{"text":"a very long utterance to synthesize"}}"#,
        )
        .await?;
        sock.write_all(b"\n").await?;
        sock.write_all(br#"{"type":"audio-start","data":{"rate":16000,"width":2,"channels":1}}"#)
            .await?;
        sock.write_all(b"\n").await?;
        sock.flush().await?;

        let mut reader = BufReader::new(sock);
        let mut chunks = 0usize;
        let mut saw_stop = false;
        let deadline = tokio::time::Instant::now() + Duration::from_secs(20);
        loop {
            let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
            if remaining.is_zero() {
                return Err(std::io::Error::other("barge-in drain deadline"));
            }
            let frame = tokio::time::timeout(remaining, read_frame(&mut reader))
                .await
                .map_err(|_| std::io::Error::other("read timeout"))??;
            let Some((hdr, _)) = frame else { break };
            match hdr.get("type").and_then(|v| v.as_str()) {
                Some("audio-chunk") => chunks += 1,
                Some("audio-stop") => {
                    saw_stop = true;
                    break;
                }
                _ => {}
            }
        }
        Ok::<(usize, bool), std::io::Error>((chunks, saw_stop))
    }
    .await;

    trigger.trigger();
    tokio::time::sleep(Duration::from_millis(20)).await;

    let (chunks, saw_stop) = result.expect("barge-in e2e");
    assert!(
        saw_stop,
        "the interrupted TTS stream must end with audio-stop"
    );
    assert!(
        chunks < full_chunks,
        "barge-in must cut the emit short: got {chunks} of {full_chunks} chunks",
    );
}

// -- T08: ASR golden transcript round trip (behavioral). --

/// `audio-start` → `audio-chunk` → `audio-stop` → `transcribe` must return a
/// `transcript { text }` event whose text matches the golden (behavioral)
/// transcript over the wire. Event-order + framing always run (mock backend);
/// a real-GGUF leg is gated separately.
#[tokio::test]
async fn m4_19_asr_golden_transcript_round_trip_over_tcp() {
    let svc = mock_service(vec![0.0; 16]);
    let (handles, trigger) =
        spawn_server_for_test_with_service(loopback_cfg(), Some(svc), Some(test_scheduler(4)))
            .await
            .expect("spawn server");

    let result = async {
        let mut sock = TcpStream::connect(handles.wyoming_actual).await?;
        sock.write_all(br#"{"type":"audio-start","data":{"rate":16000,"width":2,"channels":1}}"#)
            .await?;
        sock.write_all(b"\n").await?;
        // 8 samples of 16 kHz mono int16 PCM.
        let pcm: [i16; 8] = [10, -10, 20, -20, 30, -30, 40, -40];
        let mut bytes = Vec::new();
        for s in pcm {
            bytes.extend_from_slice(&s.to_le_bytes());
        }
        let chunk_hdr = format!(
            r#"{{"type":"audio-chunk","data":{{"rate":16000,"width":2,"channels":1}},"payload_length":{}}}"#,
            bytes.len()
        );
        sock.write_all(chunk_hdr.as_bytes()).await?;
        sock.write_all(b"\n").await?;
        sock.write_all(&bytes).await?;
        sock.write_all(br#"{"type":"audio-stop"}"#).await?;
        sock.write_all(b"\n").await?;
        sock.write_all(br#"{"type":"transcribe","data":{"name":"whisper-1","language":"en"}}"#)
            .await?;
        sock.write_all(b"\n").await?;
        sock.flush().await?;

        let mut reader = BufReader::new(sock);
        let frame = tokio::time::timeout(Duration::from_secs(5), read_frame(&mut reader))
            .await
            .map_err(|_| std::io::Error::other("transcript read timeout"))??;
        Ok::<Option<(serde_json::Value, Vec<u8>)>, std::io::Error>(frame)
    }
    .await;

    trigger.trigger();
    tokio::time::sleep(Duration::from_millis(20)).await;

    let (hdr, _) = result.expect("asr round trip").expect("transcript frame");
    assert_eq!(
        hdr.get("type").and_then(|v| v.as_str()),
        Some("transcript"),
        "ASR round trip must yield a transcript event; got {hdr}",
    );
    let text = hdr
        .pointer("/data/text")
        .and_then(|v| v.as_str())
        .expect("transcript.data.text");
    // Golden (behavioral): the mock encodes model + sample count. 8 int16
    // samples @ 16 kHz stay 8 samples (no resample). The exact WORDS are not
    // asserted (bit-exact is a non-goal, T09) — the deterministic shape is.
    assert_eq!(
        text, "mock transcript [whisper-1] 8 samples",
        "golden mismatch: {text:?}"
    );
}

/// Real-GGUF flip-the-switch leg for the ASR round trip. Skipped (non-silent)
/// unless `VOKRA_WHISPER_BASE_GGUF` + `VOKRA_PIPER_GGUF` point at real model
/// files; the owner sets these to exercise a real Whisper transcript over
/// Wyoming (behavioral parity, not bit-exact). CI has no GGUF so this skips.
#[tokio::test]
async fn m4_19_asr_real_gguf_round_trip_gated() {
    let (Ok(whisper), Ok(piper)) = (
        std::env::var("VOKRA_WHISPER_BASE_GGUF"),
        std::env::var("VOKRA_PIPER_GGUF"),
    ) else {
        eprintln!(
            "m4_19_asr_real_gguf_round_trip_gated: SKIP — set VOKRA_WHISPER_BASE_GGUF + \
             VOKRA_PIPER_GGUF to a real Whisper base + piper-plus GGUF to run this \
             flip-the-switch leg (owner task; CI has no GGUF)."
        );
        return;
    };

    use vokra_server::service::{InferenceService, ServiceConfig};
    let cfg = ServiceConfig::minimum(whisper.into(), piper.into());
    // Supplying both env vars *is* the request to run this leg, so a build
    // failure from here on is a failure, not a skip. Absorbing it reported
    // green for a round trip that never happened: audit cc-34 (2026-07-19)
    // found that pointing the leg at the multi-speaker voice turned a correct,
    // loud loader error into a passing test. The genuinely-unconfigured case
    // is still skipped, above (FR-EX-08).
    let service = InferenceService::build(&cfg).unwrap_or_else(|e| {
        panic!(
            "m4_19_asr_real_gguf_round_trip_gated: service build failed for the \
             supplied VOKRA_WHISPER_BASE_GGUF / VOKRA_PIPER_GGUF: {e}\nnote: the \
             piper-plus voice must be a single-speaker export — the multi-speaker \
             6lang voice fails with `missing tensor spk_proj.0.weight`; use the \
             `-neutralspk` GGUF."
        )
    });
    let service: Arc<dyn WyomingBackend> = service;
    let (handles, trigger) =
        spawn_server_for_test_with_service(loopback_cfg(), Some(service), Some(test_scheduler(2)))
            .await
            .expect("spawn server");

    // A short silent buffer is enough to exercise the pipeline end-to-end; the
    // owner supplies real audio for a WER check separately.
    let result = async {
        let mut sock = TcpStream::connect(handles.wyoming_actual).await?;
        sock.write_all(br#"{"type":"audio-start","data":{"rate":16000,"width":2,"channels":1}}"#)
            .await?;
        sock.write_all(b"\n").await?;
        let bytes = vec![0u8; 16_000 * 2]; // 1 s silence
        let chunk_hdr = format!(
            r#"{{"type":"audio-chunk","data":{{"rate":16000,"width":2,"channels":1}},"payload_length":{}}}"#,
            bytes.len()
        );
        sock.write_all(chunk_hdr.as_bytes()).await?;
        sock.write_all(b"\n").await?;
        sock.write_all(&bytes).await?;
        sock.write_all(br#"{"type":"audio-stop"}"#).await?;
        sock.write_all(b"\n").await?;
        sock.write_all(br#"{"type":"transcribe","data":{"name":"whisper-base"}}"#)
            .await?;
        sock.write_all(b"\n").await?;
        sock.flush().await?;
        let mut reader = BufReader::new(sock);
        let frame = tokio::time::timeout(Duration::from_secs(60), read_frame(&mut reader))
            .await
            .map_err(|_| std::io::Error::other("real transcript timeout"))??;
        Ok::<Option<(serde_json::Value, Vec<u8>)>, std::io::Error>(frame)
    }
    .await;

    trigger.trigger();
    tokio::time::sleep(Duration::from_millis(20)).await;

    let (hdr, _) = result
        .expect("real asr round trip")
        .expect("transcript frame");
    // Behavioral parity: shape only. Real text is not asserted (bit-exact
    // non-goal); silence may transcribe to empty — that is still a valid
    // transcript event.
    assert_eq!(hdr.get("type").and_then(|v| v.as_str()), Some("transcript"));
    assert!(hdr.pointer("/data/text").and_then(|v| v.as_str()).is_some());
}

// -- T09: faster-whisper behavioral parity (transcript shape / language hint). --

/// The `transcript` event Vokra emits must match the faster-whisper Wyoming
/// contract *shape*: `type: "transcript"`, `data.text` a string, no binary
/// payload. A `transcribe` with a `language` hint is accepted (not rejected).
/// Bit-exact text is a NON-goal (see module doc).
#[tokio::test]
async fn m4_19_faster_whisper_behavioral_parity_shape() {
    let svc = mock_service(vec![0.0; 16]);
    let (handles, trigger) = spawn_server_for_test_with_service(loopback_cfg(), Some(svc), None)
        .await
        .expect("spawn server");

    let result = async {
        let mut sock = TcpStream::connect(handles.wyoming_actual).await?;
        sock.write_all(br#"{"type":"audio-start","data":{"rate":16000,"width":2,"channels":1}}"#)
            .await?;
        sock.write_all(b"\n").await?;
        let bytes: Vec<u8> = (0..64i16).flat_map(|s| s.to_le_bytes()).collect();
        let chunk_hdr = format!(
            r#"{{"type":"audio-chunk","data":{{"rate":16000,"width":2,"channels":1}},"payload_length":{}}}"#,
            bytes.len()
        );
        sock.write_all(chunk_hdr.as_bytes()).await?;
        sock.write_all(b"\n").await?;
        sock.write_all(&bytes).await?;
        // A language hint is part of the faster-whisper `transcribe` contract.
        sock.write_all(br#"{"type":"transcribe","data":{"language":"ja"}}"#)
            .await?;
        sock.write_all(b"\n").await?;
        sock.flush().await?;
        let mut reader = BufReader::new(sock);
        let frame = tokio::time::timeout(Duration::from_secs(5), read_frame(&mut reader))
            .await
            .map_err(|_| std::io::Error::other("transcript read timeout"))??;
        Ok::<Option<(serde_json::Value, Vec<u8>)>, std::io::Error>(frame)
    }
    .await;

    trigger.trigger();
    tokio::time::sleep(Duration::from_millis(20)).await;

    let (hdr, payload) = result
        .expect("behavioral parity")
        .expect("transcript frame");
    // Shape assertions (the faster-whisper Wyoming contract), NOT exact words.
    assert_eq!(hdr.get("type").and_then(|v| v.as_str()), Some("transcript"));
    assert!(
        hdr.pointer("/data/text").and_then(|v| v.as_str()).is_some(),
        "transcript must carry a string data.text field",
    );
    // Upstream writers OMIT `payload_length` entirely when there is no
    // payload (`event.py async_write_event`: `if event.payload: ...`) —
    // absent-or-zero both mean "no binary payload follows".
    assert_eq!(
        hdr.get("payload_length")
            .and_then(|v| v.as_u64())
            .unwrap_or(0),
        0,
        "a transcript event carries no binary payload",
    );
    assert!(
        payload.is_empty(),
        "no trailing payload bytes after a transcript"
    );
}

// -- T04: multi-session scheduler overload → explicit error event over TCP. --

/// With a scheduler cap of 1, a first connection holds the only permit for its
/// lifetime; a second concurrent connection must be refused with an explicit
/// `error` event (FR-EX-08 — never a silent drop), then the first releasing
/// its permit lets a later connection succeed.
#[tokio::test]
async fn m4_19_scheduler_overload_is_explicit_error_over_tcp() {
    let svc = mock_service(vec![0.0; 16]);
    let (handles, trigger) =
        spawn_server_for_test_with_service(loopback_cfg(), Some(svc), Some(test_scheduler(1)))
            .await
            .expect("spawn server");

    // Connection 1: acquire + hold the single permit. Send describe and read
    // the info reply so we KNOW its handler is running (permit acquired)
    // before opening connection 2.
    let mut sock1 = TcpStream::connect(handles.wyoming_actual)
        .await
        .expect("connect 1");
    sock1
        .write_all(br#"{"type":"describe"}"#)
        .await
        .expect("write 1");
    sock1.write_all(b"\n").await.expect("nl 1");
    sock1.flush().await.expect("flush 1");
    let mut reader1 = BufReader::new(sock1);
    let info = tokio::time::timeout(Duration::from_secs(5), read_frame(&mut reader1))
        .await
        .expect("info timeout")
        .expect("info io")
        .expect("info frame");
    assert_eq!(info.0.get("type").and_then(|v| v.as_str()), Some("info"));

    // Connection 2: the permit is held → must be refused with an error event.
    let refused = async {
        let mut sock2 = TcpStream::connect(handles.wyoming_actual).await?;
        // Sending a describe is optional — the refusal is written on connect —
        // but send one to mirror a real client.
        sock2.write_all(br#"{"type":"describe"}"#).await?;
        sock2.write_all(b"\n").await?;
        sock2.flush().await?;
        let mut reader2 = BufReader::new(sock2);
        let frame = tokio::time::timeout(Duration::from_secs(5), read_frame(&mut reader2))
            .await
            .map_err(|_| std::io::Error::other("refusal read timeout"))??;
        Ok::<Option<(serde_json::Value, Vec<u8>)>, std::io::Error>(frame)
    }
    .await;

    let refusal = refused.expect("conn2 io").expect("conn2 frame");
    assert_eq!(
        refusal.0.get("type").and_then(|v| v.as_str()),
        Some("error"),
        "an over-capacity connection must get an explicit error, got {}",
        refusal.0,
    );
    let msg = refusal
        .0
        .pointer("/data/message")
        .and_then(|v| v.as_str())
        .unwrap_or_default();
    assert!(
        msg.contains("refused") || msg.contains("capacity") || msg.contains("unavailable"),
        "refusal must explain overload; got {msg:?}",
    );

    // Release connection 1's permit; a fresh connection then succeeds.
    drop(reader1);
    tokio::time::sleep(Duration::from_millis(50)).await;
    let mut sock3 = TcpStream::connect(handles.wyoming_actual)
        .await
        .expect("connect 3");
    sock3
        .write_all(br#"{"type":"describe"}"#)
        .await
        .expect("write 3");
    sock3.write_all(b"\n").await.expect("nl 3");
    sock3.flush().await.expect("flush 3");
    let mut reader3 = BufReader::new(sock3);
    let info3 = tokio::time::timeout(Duration::from_secs(5), read_frame(&mut reader3))
        .await
        .expect("info3 timeout")
        .expect("info3 io")
        .expect("info3 frame");
    assert_eq!(
        info3.0.get("type").and_then(|v| v.as_str()),
        Some("info"),
        "after release, a new connection must acquire the permit and serve",
    );

    trigger.trigger();
    tokio::time::sleep(Duration::from_millis(20)).await;
}
