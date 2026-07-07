//! M2-15 §8 — Wyoming `describe` → `info` HARD-assertion integration test.
//!
//! # Contract
//!
//! Home Assistant's Wyoming Assist discovery probe sends a single
//! `{"type":"describe"}\n` JSONL header and expects a well-formed
//! `{"type":"info", ...}` JSONL reply on the same TCP connection. Prior to
//! the fix landed at commit `d076b8f` (fix(wyoming): implement
//! discovery-only accept loop + await shutdown) two independent bugs made
//! this fail end-to-end:
//!
//! 1. The T03 `wyoming_accept_loop` was an accept-and-close placeholder.
//!    A client got a bare TCP close after the header write — HA marked the
//!    server as non-responsive.
//! 2. `run_with_config` returned from `block_on(...)` right after
//!    `spawn_server`. The listeners lived on the runtime, so `block_on`
//!    unwinding dropped the runtime and every spawned task died mid-flight
//!    before serving any application-layer bytes.
//!
//! Both were fixed in `d076b8f`; the ASR-empty describe path is now
//! actually wired via `run_describe_only_connection` (see
//! `src/api/wyoming.rs`). This test exists so a regression in either the
//! accept-loop wiring OR the runtime lifetime is caught at the unit-test
//! level, without a Docker Home Assistant smoke.
//!
//! `wyoming_compat::wyoming_describe_round_trip` covers the same wire
//! flow with a soft-skip pattern ("will flip to hard-fail once the event
//! loop lands"), left intact for historical compatibility. This file is
//! the hard-fail companion — a broken accept loop or a runtime-drop
//! regression makes this test fail loudly (timeout / EOF), not skip.
//!
//! # Bind
//!
//! Uses `spawn_server_for_test` with `127.0.0.1:0` (OS-assigned port) so
//! parallel `cargo test` runs never collide on a fixed port. FR-EX-08
//! posture: no silent fallback to a fixed port.

use std::time::Duration;

use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpStream;

use vokra_server::{Config, spawn_server_for_test};

/// The `describe` -> `info` round trip MUST return a well-formed
/// `type: "info"` JSONL reply within a bounded time. A broken accept loop
/// (regressed to accept-and-close) OR a dropped runtime (regressed
/// `run_with_config`) both surface here as a `read_line` timeout or a
/// zero-byte EOF, and this test hard-fails in either case.
///
/// Bounds `read_line` at 5 s so a wedged handler cannot hang CI.
#[tokio::test]
async fn wyoming_info_reply_is_returned_for_describe() {
    let cfg = Config {
        http_bind: "127.0.0.1:0".parse().unwrap(),
        wyoming_bind: "127.0.0.1:0".parse().unwrap(),
        config_file: None,
    };
    let (handles, trigger) = spawn_server_for_test(cfg).await.expect("spawn server");

    // Wrap the client work in an async block so we can always trigger the
    // shutdown afterwards — even on assertion panic, tokio drops the
    // TcpStream and the shutdown trigger drains the listener.
    let result: Result<String, String> = async {
        let mut sock = TcpStream::connect(handles.wyoming_actual)
            .await
            .map_err(|e| format!("connect to wyoming port failed: {e}"))?;

        // Wyoming JSONL: one header line terminated by \n, no payload.
        sock.write_all(br#"{"type":"describe"}"#)
            .await
            .map_err(|e| format!("write describe header failed: {e}"))?;
        sock.write_all(b"\n")
            .await
            .map_err(|e| format!("write header newline failed: {e}"))?;
        sock.flush()
            .await
            .map_err(|e| format!("flush describe request failed: {e}"))?;

        // Read exactly one JSONL header line with a hard 5 s deadline.
        // Prior to the accept-loop fix this returned Ok(0) (EOF); prior
        // to the runtime-lifetime fix this timed out because the runtime
        // had already been dropped. After both fixes this returns a
        // populated `info` line.
        let mut reader = BufReader::new(sock);
        let mut line = String::new();
        let n = tokio::time::timeout(Duration::from_secs(5), reader.read_line(&mut line))
            .await
            .map_err(|_| {
                "read_line timed out after 5 s — accept loop or runtime lifetime regression?"
                    .to_string()
            })?
            .map_err(|e| format!("read_line I/O error: {e}"))?;
        if n == 0 {
            return Err(
                "server closed with zero bytes — accept loop regressed to accept-and-close?".into(),
            );
        }
        Ok(line)
    }
    .await;

    // Always tear down first so a failing assert never leaks the listener
    // to a parallel test run.
    trigger.trigger();
    // Give the shutdown watch a beat to propagate through the listeners.
    tokio::time::sleep(Duration::from_millis(20)).await;

    // Now assert. The message deliberately points at the two regressed
    // paths so a future maintainer sees the fix boundary in the failure.
    let line = result.unwrap_or_else(|e| panic!("wyoming info reply not returned: {e}"));

    // 1. Wire framing invariants — exactly one JSONL terminator, no
    //    embedded newlines. A framing bug in `write_response` would trip
    //    this before we get to the semantic assertions.
    assert!(
        line.ends_with('\n'),
        "info reply is not newline-terminated: {line:?}",
    );
    assert_eq!(
        line.matches('\n').count(),
        1,
        "info reply contained multiple newlines (framing bug): {line:?}",
    );

    // 2. Semantic assertions — the reply MUST be `{"type":"info",...}`.
    let trimmed = line.trim_end_matches('\n');
    let value: serde_json::Value = serde_json::from_str(trimmed)
        .unwrap_or_else(|e| panic!("info reply is not JSON ({e}): raw {trimmed:?}"));
    let obj = value
        .as_object()
        .unwrap_or_else(|| panic!("info reply is not a JSON object: {trimmed:?}"));

    let ty = obj
        .get("type")
        .and_then(|v| v.as_str())
        .unwrap_or_else(|| panic!("info reply has no string `type` field: {trimmed:?}"));
    assert_eq!(
        ty, "info",
        "expected `type` == \"info\" for a `describe` probe (got {ty:?}); full reply: {trimmed:?}"
    );

    // 3. Structural assertion — the info body advertises an `asr` list.
    //    In the discovery-only startup path this list is EMPTY (no
    //    `InferenceService` wired yet), but the field itself MUST be
    //    present and an array. This is the shape HA parses against.
    let data = obj
        .get("data")
        .unwrap_or_else(|| panic!("info reply has no `data` field: {trimmed:?}"));
    let data_obj = data
        .as_object()
        .unwrap_or_else(|| panic!("info reply `data` is not a JSON object: {trimmed:?}"));
    let asr = data_obj
        .get("asr")
        .unwrap_or_else(|| panic!("info reply `data.asr` is missing: {trimmed:?}"));
    let asr_arr = asr
        .as_array()
        .unwrap_or_else(|| panic!("info reply `data.asr` is not an array: {trimmed:?}"));
    // Discovery-only path: asr is empty. If the accept loop ever changes
    // to the full `run_asr_connection` path, this assertion will need to
    // become an inclusion check for the wired model names — kept as a
    // pinned exact assertion so that transition is a deliberate edit
    // rather than a silent drift.
    assert!(
        asr_arr.is_empty(),
        "info reply `data.asr` should be empty in the discovery-only startup path (no \
         `InferenceService` wired yet); got {asr_arr:?}. If T04 has wired the ASR \
         registry, update this assertion to check the expected model names.",
    );

    // 4. Data-length / payload-length are 0 in the JSONL header — a
    //    non-zero value here would indicate a mis-serialised header
    //    frame (client would try to read binary payload bytes that
    //    weren't emitted).
    let data_length = obj
        .get("data_length")
        .and_then(|v| v.as_u64())
        .unwrap_or_else(|| panic!("info reply has no `data_length`: {trimmed:?}"));
    let payload_length = obj
        .get("payload_length")
        .and_then(|v| v.as_u64())
        .unwrap_or_else(|| panic!("info reply has no `payload_length`: {trimmed:?}"));
    assert_eq!(
        data_length, 0,
        "info reply should not announce a separate data blob for a describe probe",
    );
    assert_eq!(
        payload_length, 0,
        "info reply must not announce a binary payload (describe is header-only)",
    );
}

/// After the accept loop has served a `describe` on one connection, a
/// SEPARATE fresh TCP connection MUST also succeed. This catches an
/// accept-loop bug where the first successful describe leaves the loop
/// in a state where it never accepts again (e.g. a `break` in the wrong
/// arm, or a per-connection task that stalls the parent).
///
/// This is a stronger guarantee than the single-connection test above
/// because a regression that only serves the first connection would
/// pass the first test but fail this one.
#[tokio::test]
async fn wyoming_accept_loop_serves_multiple_connections() {
    let cfg = Config {
        http_bind: "127.0.0.1:0".parse().unwrap(),
        wyoming_bind: "127.0.0.1:0".parse().unwrap(),
        config_file: None,
    };
    let (handles, trigger) = spawn_server_for_test(cfg).await.expect("spawn server");

    // Helper: open a fresh connection, send describe, expect one info line.
    async fn describe_once(addr: std::net::SocketAddr) -> Result<String, String> {
        let mut sock = TcpStream::connect(addr)
            .await
            .map_err(|e| format!("connect failed: {e}"))?;
        sock.write_all(br#"{"type":"describe"}"#)
            .await
            .map_err(|e| format!("write header failed: {e}"))?;
        sock.write_all(b"\n")
            .await
            .map_err(|e| format!("write newline failed: {e}"))?;
        sock.flush()
            .await
            .map_err(|e| format!("flush failed: {e}"))?;
        let mut reader = BufReader::new(sock);
        let mut line = String::new();
        let n = tokio::time::timeout(Duration::from_secs(5), reader.read_line(&mut line))
            .await
            .map_err(|_| "read_line timed out after 5 s".to_string())?
            .map_err(|e| format!("read_line failed: {e}"))?;
        if n == 0 {
            return Err("server closed with zero bytes on subsequent connection".into());
        }
        Ok(line)
    }

    let first = describe_once(handles.wyoming_actual).await;
    let second = describe_once(handles.wyoming_actual).await;
    let third = describe_once(handles.wyoming_actual).await;

    trigger.trigger();
    tokio::time::sleep(Duration::from_millis(20)).await;

    for (i, res) in [&first, &second, &third].iter().enumerate() {
        let line = res
            .as_ref()
            .unwrap_or_else(|e| panic!("describe #{i} failed: {e}"));
        let trimmed = line.trim_end_matches('\n');
        let value: serde_json::Value = serde_json::from_str(trimmed)
            .unwrap_or_else(|e| panic!("describe #{i} reply not JSON ({e}): {trimmed:?}"));
        assert_eq!(
            value.get("type").and_then(|v| v.as_str()),
            Some("info"),
            "describe #{i} did not return info; raw: {trimmed:?}",
        );
    }
}

/// A `describe` followed by an unknown event on the SAME connection
/// exercises the loop-back-to-header path in `run_describe_only_connection`.
/// The unknown event MUST produce an explicit `error` reply (FR-EX-08:
/// no silent drop) and the accept loop MUST stay alive for a subsequent
/// event — otherwise a satellite that mis-speaks the protocol wedges the
/// connection.
#[tokio::test]
async fn wyoming_unknown_event_after_describe_returns_error_not_silence() {
    let cfg = Config {
        http_bind: "127.0.0.1:0".parse().unwrap(),
        wyoming_bind: "127.0.0.1:0".parse().unwrap(),
        config_file: None,
    };
    let (handles, trigger) = spawn_server_for_test(cfg).await.expect("spawn server");

    let result: Result<(String, String), String> = async {
        let mut sock = TcpStream::connect(handles.wyoming_actual)
            .await
            .map_err(|e| format!("connect failed: {e}"))?;

        // First: describe -> info.
        sock.write_all(br#"{"type":"describe"}"#)
            .await
            .map_err(|e| e.to_string())?;
        sock.write_all(b"\n").await.map_err(|e| e.to_string())?;
        sock.flush().await.map_err(|e| e.to_string())?;

        let (read_half, mut write_half) = sock.into_split();
        let mut reader = BufReader::new(read_half);
        let mut info_line = String::new();
        let n = tokio::time::timeout(Duration::from_secs(5), reader.read_line(&mut info_line))
            .await
            .map_err(|_| "info read timeout".to_string())?
            .map_err(|e| e.to_string())?;
        if n == 0 {
            return Err("EOF on info read".into());
        }

        // Second: unknown event on the same connection -> error.
        write_half
            .write_all(br#"{"type":"unknown-satellite-event"}"#)
            .await
            .map_err(|e| e.to_string())?;
        write_half
            .write_all(b"\n")
            .await
            .map_err(|e| e.to_string())?;
        write_half.flush().await.map_err(|e| e.to_string())?;

        let mut err_line = String::new();
        let m = tokio::time::timeout(Duration::from_secs(5), reader.read_line(&mut err_line))
            .await
            .map_err(|_| "error read timeout".to_string())?
            .map_err(|e| e.to_string())?;
        if m == 0 {
            return Err("EOF on error read — loop did not survive unknown event".into());
        }
        Ok((info_line, err_line))
    }
    .await;

    trigger.trigger();
    tokio::time::sleep(Duration::from_millis(20)).await;

    let (info_line, err_line) = result.unwrap_or_else(|e| panic!("multi-event flow failed: {e}"));

    let info_val: serde_json::Value = serde_json::from_str(info_line.trim_end_matches('\n'))
        .expect("info line must parse as JSON");
    assert_eq!(
        info_val.get("type").and_then(|v| v.as_str()),
        Some("info"),
        "first reply must be info; got: {info_line:?}",
    );

    let err_val: serde_json::Value = serde_json::from_str(err_line.trim_end_matches('\n'))
        .expect("error line must parse as JSON");
    assert_eq!(
        err_val.get("type").and_then(|v| v.as_str()),
        Some("error"),
        "second reply must be error (unknown event, discovery-only mode); got: {err_line:?}",
    );
    // The error message should mention discovery-only mode so operators
    // can diagnose the "why doesn't ASR work?" question at the wire level.
    let msg = err_val
        .pointer("/data/message")
        .and_then(|v| v.as_str())
        .unwrap_or_default();
    assert!(
        msg.contains("discovery-only") || msg.contains("model registry"),
        "error message should identify the discovery-only startup state; got: {msg:?}",
    );
}
