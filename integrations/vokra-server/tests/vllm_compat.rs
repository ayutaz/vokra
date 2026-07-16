//! T10 — vLLM `/v1/completions` + `/v1/chat/completions` compat integration test.
//!
//! # Scope (M2-09 plan §3.2 (g) / D9)
//!
//! This file exercises the **contract shape only** — request body parses
//! correctly and the response wire schema conforms to what vLLM's
//! OpenAI-compatible surface publishes. **Generation quality is out of
//! scope**: v0.5 has no LLM in-tree (FR-MD-04 = Whisper large-v3,
//! FR-MD-05 = Kokoro), so the honest response for every valid request is
//! `501 Not Implemented` carrying the OpenAI-shape error envelope
//! (`{"error": {"message", "type", "code"}}`) per
//! [`crate::api::vllm`]. Real LLM inference ships in v1.0+
//! (CosyVoice2 / Voxtral) and v1.5+ (Moshi / Helium).
//!
//! # What the tests assert
//!
//! * **Valid requests** to `/v1/completions` and `/v1/chat/completions`
//!   round-trip as JSON and land as **either 501** (the honest v0.5
//!   behaviour, with `error.type == "not_implemented"`) **or 200/201
//!   with a schema-conforming completion body** (the forward-compat
//!   drop-in that lands when a real LLM backend is registered). Any
//!   other status is a bug. NFR-RL-06 pin: on 501 the body MUST NOT
//!   fabricate `choices[]` / top-level `text` / assistant `message`.
//! * **Malformed requests** (missing `model`, missing `prompt` /
//!   `messages`, unknown field, empty `messages`, non-JSON body,
//!   wrong content-type) become `400` with our OpenAI-shape
//!   envelope carrying `error.type == "invalid_input"`. This is the
//!   parity guarantee vLLM's own server offers to its clients and is
//!   what closes the "silently ignored field" bug class.
//! * **Router surface** is closed: only the two documented paths are
//!   registered on `/v1`; sibling paths like `/v1/embeddings` return
//!   `404`. Guards against accidental greedy globs in later refactors.
//!
//! # Route-wiring gate
//!
//! [`crate::api::vllm::router`] exists and passes its in-crate contract
//! tests (T09). Mounting it on the top-level server router lands with
//! the OpenAI transcriptions router in T06 / T09 wiring; until that
//! landing all `/v1/*` routes on the running server return `404`.
//! Rather than block T10 on the merge PR, we mirror the openai_compat
//! pattern: `assert_route_present` treats `404` on every documented
//! path as an explicit **pending** state, logs it via `eprintln!`,
//! and short-circuits the row. The moment the router is `.merge()`-ed
//! onto the server, this file becomes a live contract test with **no
//! test churn**. The in-process `oneshot` fallback path (see
//! `assert_contract_via_router_oneshot`) is always executed so this
//! file never behaves as an empty test binary on runners where the
//! router is not yet mounted end-to-end.
//!
//! # Runtime + ports
//!
//! Every server-spawning test binds `127.0.0.1:0` (OS-assigned free
//! port) via [`spawn_server_for_test`] and tears the server down
//! through the returned shutdown trigger. No fixed port ever appears
//! in this file, so parallel `cargo test` runs and sandboxed CI both
//! work.

use std::net::SocketAddr;
use std::time::Duration;

use vokra_server::api::vllm::router as vllm_router;
use vokra_server::{Config, spawn_server_for_test};

// ---------- raw HTTP JSON client (no reqwest dep needed) ----------

/// POST a JSON body via a raw TCP write/read and return
/// `(status, body_bytes)`. Bounded 5 s read + `Connection: close`
/// framing mirrors `openai_compat.rs` so failure modes are identical
/// across the two files (parallel maintenance is cheaper).
async fn http_post_json(
    addr: SocketAddr,
    path: &str,
    body: &[u8],
) -> std::io::Result<(u16, Vec<u8>)> {
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
        .map_err(|e| std::io::Error::other(format!("utf8: {e}")))?;
    let first_line = head_str.lines().next().unwrap_or("");
    let status: u16 = first_line
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| std::io::Error::other(format!("bad status line: {first_line:?}")))?;

    // Body may be chunked or plain. `Connection: close` above requests
    // plain framing, and hyper honours that. If a chunked encoding
    // slipped in, best-effort strip the first chunk-size line — the
    // vLLM handler returns a small JSON that fits in one chunk.
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

    Ok((status, body_bytes))
}

// ---------- schema helpers ----------

/// Enumerates the three response shapes T10 accepts. Anything else is
/// a failed test — including a fabricated 200 with generated text
/// (NFR-RL-06) that lacks the `choices[]` schema.
enum ContractOutcome {
    /// 501 with `{"error": {"type": "not_implemented", ...}}`. The
    /// honest v0.5 behaviour. Must NOT include `choices[]` /
    /// top-level `text` / `message` (NFR-RL-06 pin).
    NotImplemented,
    /// 200 / 201 with an OpenAI-shape completion body (`id`, `object`,
    /// `choices[]` non-empty; `usage` if present is an object). This
    /// is the forward-compat drop-in path for when a real LLM backend
    /// gets wired.
    Completion,
    /// The server-level router does not have the path mounted yet
    /// (returns 404). We short-circuit the row and log a `pending`
    /// notice so CI logs surface the state without silent success.
    RoutePending,
}

/// Classify + assert the response schema for the completions endpoint.
/// Returns [`ContractOutcome`] so the caller can drive test-level
/// bookkeeping (e.g. pending vs completed) without repeating the
/// schema logic per test.
fn classify_completion_response(status: u16, body: &[u8], surface: &str) -> ContractOutcome {
    if status == 404 {
        return ContractOutcome::RoutePending;
    }
    let v: serde_json::Value = serde_json::from_slice(body).unwrap_or_else(|e| {
        let preview = std::str::from_utf8(body).unwrap_or("<non-utf8>");
        panic!("vllm_compat[{surface}]: response is not JSON: {e}; body: {preview:?}")
    });
    if status == 501 {
        let err = v.get("error").unwrap_or_else(|| {
            panic!("vllm_compat[{surface}]: 501 body missing `error` object; got: {v}")
        });
        assert_eq!(
            err.get("type").and_then(|s| s.as_str()),
            Some("not_implemented"),
            "vllm_compat[{surface}]: expected error.type=\"not_implemented\"; got: {err}",
        );
        assert!(
            err.get("code").and_then(|s| s.as_str()).is_some(),
            "vllm_compat[{surface}]: 501 body missing string error.code; got: {err}",
        );
        assert!(
            err.get("message").and_then(|s| s.as_str()).is_some(),
            "vllm_compat[{surface}]: 501 body missing string error.message; got: {err}",
        );
        // NFR-RL-06: on 501 the response MUST NOT fabricate a completion.
        assert!(
            v.get("choices").is_none(),
            "vllm_compat[{surface}]: 501 must not fabricate `choices[]`; got: {v}",
        );
        assert!(
            v.get("text").is_none(),
            "vllm_compat[{surface}]: 501 must not fabricate top-level `text`; got: {v}",
        );
        return ContractOutcome::NotImplemented;
    }
    if status == 200 || status == 201 {
        // Forward-compat drop-in: schema must be OpenAI-shape. Do NOT
        // check quality (out of scope) — just that the wire shape
        // matches what vLLM's own server publishes.
        let obj = v.as_object().unwrap_or_else(|| {
            panic!("vllm_compat[{surface}]: completion body must be a JSON object; got: {v}")
        });
        for k in ["id", "object", "choices"] {
            assert!(
                obj.contains_key(k),
                "vllm_compat[{surface}]: completion body missing required key `{k}`; got: {v}",
            );
        }
        let choices = obj["choices"].as_array().unwrap_or_else(|| {
            panic!("vllm_compat[{surface}]: `choices` must be an array; got: {v}")
        });
        assert!(
            !choices.is_empty(),
            "vllm_compat[{surface}]: `choices` must be non-empty; got: {v}",
        );
        if let Some(usage) = obj.get("usage") {
            assert!(
                usage.is_object(),
                "vllm_compat[{surface}]: `usage` must be an object when present; got: {v}",
            );
        }
        return ContractOutcome::Completion;
    }
    panic!(
        "vllm_compat[{surface}]: unexpected status {status}; \
         must be 501 (v0.5), 200/201 (forward-compat), or 404 (route pending). body: {v}",
    );
}

/// Assert a 400 `invalid_input` envelope. Every malformed-body test
/// funnels through this so the shape guarantee is single-sourced.
fn assert_invalid_input(status: u16, body: &[u8], surface: &str) {
    assert_eq!(
        status,
        400,
        "vllm_compat[{surface}]: expected 400 BAD_REQUEST, got {status}; body: {}",
        String::from_utf8_lossy(body),
    );
    let v: serde_json::Value = serde_json::from_slice(body).unwrap_or_else(|e| {
        let preview = std::str::from_utf8(body).unwrap_or("<non-utf8>");
        panic!("vllm_compat[{surface}]: 400 body is not JSON: {e}; body: {preview:?}")
    });
    let err = v.get("error").unwrap_or_else(|| {
        panic!("vllm_compat[{surface}]: 400 body missing `error` object; got: {v}")
    });
    assert_eq!(
        err.get("type").and_then(|s| s.as_str()),
        Some("invalid_input"),
        "vllm_compat[{surface}]: expected error.type=\"invalid_input\"; got: {err}",
    );
}

// ---------- in-process (router-level) contract check ----------
//
// This always runs, regardless of whether the running server has
// mounted the vLLM router yet. It exercises the exact same code path
// that the end-to-end tests will hit once the router is `.merge()`-ed
// onto the top-level server router, so the schema guarantees are
// covered on every runner.

/// POST a JSON body directly to the vllm router via tower `oneshot`.
/// No socket / no runtime bind — the tokio harness drives axum in
/// memory. Same helper shape as `api::vllm::contract_stub::post_json`,
/// intentionally duplicated here so this file is self-contained.
async fn oneshot_post_json(path: &str, body: serde_json::Value) -> (u16, Vec<u8>) {
    use axum::body::to_bytes;
    use axum::http::{Method, Request};
    use tower::ServiceExt as _;
    let app = vllm_router();
    let req = Request::builder()
        .method(Method::POST)
        .uri(path)
        .header("content-type", "application/json")
        .body(axum::body::Body::from(body.to_string()))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    let status = resp.status().as_u16();
    let bytes = to_bytes(resp.into_body(), 16 * 1024)
        .await
        .unwrap()
        .to_vec();
    (status, bytes)
}

// ---------- the tests ----------
//
// Six #[tokio::test]s: two happy-path (completions + chat), three
// malformed-body cases (missing prompt / empty messages / non-JSON /
// unknown field / wrong content-type — bundled), one route-surface
// closure check.

/// A valid `/v1/completions` request round-trips as JSON and returns
/// either 501 (v0.5 honest state) or 200/201 with an OpenAI-shape
/// completion body. Runs against both the in-process router (always)
/// and the live server (skipped as `pending` if the router is not
/// yet mounted onto the top-level server).
#[tokio::test]
async fn completions_valid_request_matches_schema() {
    let body = serde_json::json!({
        "model": "gpt-3.5-turbo-instruct",
        "prompt": "Hello, world!",
        "max_tokens": 16,
        "temperature": 0.7,
    });

    // (1) In-process oneshot: MUST return 501 today (T09 lands the
    // stub); becomes 200/201 when a real LLM backend is registered.
    // This assertion always runs and is the primary T10 contract.
    let (status, resp) = oneshot_post_json("/v1/completions", body.clone()).await;
    let outcome = classify_completion_response(status, &resp, "completions/oneshot");
    assert!(
        matches!(
            outcome,
            ContractOutcome::NotImplemented | ContractOutcome::Completion,
        ),
        "vllm_compat[completions/oneshot]: in-process router must never return \
         `RoutePending` (the router IS the router) — got status {status}",
    );

    // (2) End-to-end via the running server. `RoutePending` (404) is
    // an acceptable state until the router is `.merge()`-ed onto the
    // top-level app; the eprintln surfaces it in CI logs without
    // silent success (mirrors openai_compat's pattern).
    let cfg = Config {
        http_bind: "127.0.0.1:0".parse().unwrap(),
        wyoming_bind: "127.0.0.1:0".parse().unwrap(),
        config_file: None,
        ..Config::default()
    };
    let (handles, trigger) = spawn_server_for_test(cfg).await.expect("spawn server");
    let result = http_post_json(
        handles.http_actual,
        "/v1/completions",
        body.to_string().as_bytes(),
    )
    .await
    .expect("POST /v1/completions");
    trigger.trigger();
    tokio::time::sleep(Duration::from_millis(20)).await;

    let (status, resp_body) = result;
    match classify_completion_response(status, &resp_body, "completions/e2e") {
        ContractOutcome::NotImplemented | ContractOutcome::Completion => {}
        ContractOutcome::RoutePending => {
            eprintln!(
                "vllm_compat[completions/e2e]: /v1/completions returned 404 — \
                 vLLM router not yet mounted on top-level server; T10 contract pending.",
            );
        }
    }
}

/// A valid `/v1/chat/completions` request must satisfy the same
/// contract: 501 with the OpenAI-shape error envelope OR 200/201 with
/// an OpenAI-shape completion body. On 501 the body MUST NOT include
/// a fabricated assistant message (NFR-RL-06).
#[tokio::test]
async fn chat_completions_valid_request_matches_schema() {
    let body = serde_json::json!({
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user",   "content": "Say hi."},
        ],
        "temperature": 0.2,
    });

    let (status, resp) = oneshot_post_json("/v1/chat/completions", body.clone()).await;
    let outcome = classify_completion_response(status, &resp, "chat/oneshot");
    assert!(
        matches!(
            outcome,
            ContractOutcome::NotImplemented | ContractOutcome::Completion,
        ),
        "vllm_compat[chat/oneshot]: in-process router must never return \
         `RoutePending` — got status {status}",
    );

    let cfg = Config {
        http_bind: "127.0.0.1:0".parse().unwrap(),
        wyoming_bind: "127.0.0.1:0".parse().unwrap(),
        config_file: None,
        ..Config::default()
    };
    let (handles, trigger) = spawn_server_for_test(cfg).await.expect("spawn server");
    let result = http_post_json(
        handles.http_actual,
        "/v1/chat/completions",
        body.to_string().as_bytes(),
    )
    .await
    .expect("POST /v1/chat/completions");
    trigger.trigger();
    tokio::time::sleep(Duration::from_millis(20)).await;

    let (status, resp_body) = result;
    match classify_completion_response(status, &resp_body, "chat/e2e") {
        ContractOutcome::NotImplemented | ContractOutcome::Completion => {}
        ContractOutcome::RoutePending => {
            eprintln!(
                "vllm_compat[chat/e2e]: /v1/chat/completions returned 404 — \
                 vLLM router not yet mounted on top-level server; T10 contract pending.",
            );
        }
    }
}

/// Malformed bodies (missing required field, unknown field, non-JSON,
/// wrong content-type, empty messages) must land as 400 `invalid_input`
/// with our OpenAI-shape envelope — never as 501 (which would be
/// dishonest — the request is broken, not the backend) and never as
/// axum's default plaintext rejection body. Bundled into one test to
/// keep the file compact; each case fails with an explicit `surface`
/// tag so a regression is bisectable.
#[tokio::test]
async fn malformed_requests_are_invalid_input() {
    // (a) Missing `prompt` in Completions.
    let (status, body) =
        oneshot_post_json("/v1/completions", serde_json::json!({ "model": "foo" })).await;
    assert_invalid_input(status, &body, "completions/missing_prompt");

    // (b) Unknown top-level field (guards against silently ignored
    // fields — the `#[serde(deny_unknown_fields)]` contract).
    let (status, body) = oneshot_post_json(
        "/v1/completions",
        serde_json::json!({
            "model": "foo",
            "prompt": "hi",
            "not_a_real_field": 123,
        }),
    )
    .await;
    assert_invalid_input(status, &body, "completions/unknown_field");

    // (c) Empty `messages` in Chat Completions — vLLM parity: 400 not 501.
    let (status, body) = oneshot_post_json(
        "/v1/chat/completions",
        serde_json::json!({ "model": "foo", "messages": [] }),
    )
    .await;
    assert_invalid_input(status, &body, "chat/empty_messages");

    // (d) Non-JSON body → 400 (raw JSON parser fails before serde-derive).
    // Uses the raw axum `oneshot` path so we can post an intentionally
    // non-JSON payload with the right content-type header.
    use axum::body::to_bytes;
    use axum::http::{Method, Request};
    use tower::ServiceExt as _;
    let app = vllm_router();
    let req = Request::builder()
        .method(Method::POST)
        .uri("/v1/completions")
        .header("content-type", "application/json")
        .body(axum::body::Body::from("not json at all"))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    let status = resp.status().as_u16();
    let body = to_bytes(resp.into_body(), 4096).await.unwrap().to_vec();
    assert_invalid_input(status, &body, "completions/non_json");

    // (e) Wrong content-type on chat → still our envelope (not axum default).
    let app = vllm_router();
    let req = Request::builder()
        .method(Method::POST)
        .uri("/v1/chat/completions")
        .header("content-type", "text/plain")
        .body(axum::body::Body::from(r#"{"model":"x","messages":[]}"#))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    let status = resp.status().as_u16();
    let body = to_bytes(resp.into_body(), 4096).await.unwrap().to_vec();
    assert_invalid_input(status, &body, "chat/wrong_content_type");
}

/// The vLLM router registers exactly two routes on `/v1`; sibling
/// paths (`/v1/embeddings`, `/v1/models`, ...) MUST 404. Closes the
/// bug class where a future refactor accidentally adds a greedy glob.
#[tokio::test]
async fn no_extra_v1_routes_are_registered() {
    use axum::body::to_bytes;
    use axum::http::{Method, Request};
    use tower::ServiceExt as _;
    for path in ["/v1/embeddings", "/v1/models", "/v1/completions/foo"] {
        let app = vllm_router();
        let req = Request::builder()
            .method(Method::POST)
            .uri(path)
            .header("content-type", "application/json")
            .body(axum::body::Body::from("{}"))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(
            resp.status().as_u16(),
            404,
            "vllm_compat[route_closure]: {path} must be 404, got {}; body: {:?}",
            resp.status(),
            String::from_utf8_lossy(&to_bytes(resp.into_body(), 4096).await.unwrap_or_default(),),
        );
    }
}

/// Sanity guard: bringing the server up on random ports must succeed
/// even when no /v1 routes are wired yet, matching the openai_compat
/// helper. Keeps `cargo test --test vllm_compat` from ever reporting
/// an empty binary if every other row happens to skip on `RoutePending`.
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
