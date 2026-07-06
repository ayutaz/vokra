//! Server-wide error type, HTTP mapping, panic isolation, and structured logs.
//!
//! # Scope (T05, plan §3.2.e / D7)
//!
//! * `enum ServerError` — the single failure surface every HTTP handler and
//!   every Wyoming TCP task funnels through. Six variants (plan §3.2.e):
//!   `ModelNotFound`, `UnsupportedOp`, `InvalidInput`, `InferenceFailed`,
//!   `NotImplemented`, `InternalPanic`.
//! * `impl IntoResponse for ServerError` — maps to the OpenAI-compatible
//!   `{"error": {"message", "type", "code"}}` JSON envelope so faster-whisper
//!   / vLLM / piper-plus clients read the same shape they read from the
//!   upstream services.
//! * [`catch_panic_layer`] — a `tower_http::catch_panic::CatchPanicLayer`
//!   preconfigured to convert a handler panic into a 500 JSON error
//!   (`InternalPanic`). NFR-RL-07: a single panicked request must not tear
//!   down the runtime.
//! * [`spawn_isolated_wyoming_task`] — spawns a Wyoming TCP task inside
//!   `std::panic::catch_unwind` so a panicked JSONL parser only closes ONE
//!   TCP session, never the whole listener. Same NFR-RL-07 boundary as HTTP
//!   handlers, but async-task-shaped.
//! * [`log_request`] — the shared structured-log helper emitted per HTTP
//!   request with the fields plan §3.2.e lists: `method`, `path`, `status`,
//!   `latency_ms`, `model`, `error_type`. Routed through `tracing`.
//!
//! # Invariants
//!
//! * **FR-EX-08 — no silent fallback.** `UnsupportedOp` from the engine
//!   layer (Metal / CUDA op holes) maps to HTTP `501` with
//!   `type: "unsupported_op"`. We never rewrite the request to CPU behind
//!   the caller's back.
//! * **NFR-RL-07 — API boundary safety.** A panic anywhere inside a handler
//!   or a Wyoming task becomes `InternalPanic` → `500` with a fixed body;
//!   the runtime keeps serving other requests.
//! * **NFR-RL-06 — no fabricated output.** `NotImplemented` is the honest
//!   answer when a feature is not wired (LLM completions at v0.5, Kokoro
//!   synth in M2-07 skeleton). We never return a plausible-looking placeholder
//!   audio buffer or generation text.
//! * Serialization uses `serde_json::to_string` on a hand-rolled envelope
//!   struct. No `strtod`-shaped code is in this path (NFR-RL-01 defence).

use axum::Json;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::Serialize;
use std::any::Any;
use std::sync::Arc;
use std::time::Duration;
use tower_http::catch_panic::CatchPanicLayer;

/// Every failure the HTTP + Wyoming surfaces can return.
///
/// This is a small, closed enum by design — the six variants each correspond
/// to a distinct HTTP status and a distinct client-actionable failure mode.
/// The engine layer (`WhisperAsr`, `PiperPlusTts`, `KokoroTts`, ...) already
/// returns rich error types; the service layer (T04) is responsible for
/// mapping those to one of these variants without losing the human-readable
/// message.
#[derive(Debug, Clone)]
pub enum ServerError {
    /// The `model=` field (or Wyoming `describe` name) did not match anything
    /// in the registry. Explicit — never a silent alias (FR-EX-08).
    /// Maps to HTTP 404.
    ModelNotFound {
        /// The user-supplied model name we could not resolve.
        model: String,
    },
    /// The selected backend does not implement an op the model needs
    /// (typical: Metal / CUDA kernel not landed yet for a specific op).
    /// **We never silently rewrite to CPU.** Maps to HTTP 501.
    UnsupportedOp {
        /// Human-readable description from the engine layer.
        detail: String,
    },
    /// The request payload was malformed — bad multipart, missing field,
    /// unparseable JSON, PCM sample-rate out of range, etc. Maps to HTTP 400.
    InvalidInput {
        /// Which field / part of the request was rejected.
        detail: String,
    },
    /// The engine returned a non-panic error mid-inference (e.g. GGUF I/O
    /// glitch, tokenizer shape mismatch, streaming buffer overflow).
    /// Maps to HTTP 500.
    InferenceFailed {
        /// Passthrough of the engine-layer message.
        detail: String,
    },
    /// The endpoint / feature is intentionally not wired at v0.5. Examples:
    /// real LLM generation on `/v1/completions` (FR-SV-03 contract-first),
    /// Kokoro TTS synth (M2-07 skeleton, `TtsEngine::synthesize` returns
    /// `NotImplemented`). NFR-RL-06: we tell the caller honestly, never
    /// fabricate a response. Maps to HTTP 501.
    NotImplemented {
        /// What is not implemented and where it's scheduled (for the log).
        detail: String,
    },
    /// A panic was caught by the tower `CatchPanic` layer or by a Wyoming
    /// task's `catch_unwind` guard. NFR-RL-07 API-boundary safety: the
    /// runtime stays up, this one request fails. Maps to HTTP 500.
    InternalPanic {
        /// Best-effort panic message extracted from `panic_info.payload()`.
        message: String,
    },
}

impl std::fmt::Display for ServerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ModelNotFound { model } => write!(f, "model not found: {model}"),
            Self::UnsupportedOp { detail } => write!(f, "unsupported op: {detail}"),
            Self::InvalidInput { detail } => write!(f, "invalid input: {detail}"),
            Self::InferenceFailed { detail } => write!(f, "inference failed: {detail}"),
            Self::NotImplemented { detail } => write!(f, "not implemented: {detail}"),
            Self::InternalPanic { message } => write!(f, "internal panic: {message}"),
        }
    }
}

impl std::error::Error for ServerError {}

impl ServerError {
    /// HTTP status code for this error. Kept as a method so `log_request` can
    /// stamp the outgoing status into the tracing event without materializing
    /// the full `Response`.
    pub fn status(&self) -> StatusCode {
        match self {
            Self::ModelNotFound { .. } => StatusCode::NOT_FOUND,
            Self::InvalidInput { .. } => StatusCode::BAD_REQUEST,
            Self::UnsupportedOp { .. } | Self::NotImplemented { .. } => StatusCode::NOT_IMPLEMENTED,
            Self::InferenceFailed { .. } | Self::InternalPanic { .. } => {
                StatusCode::INTERNAL_SERVER_ERROR
            }
        }
    }

    /// The `type` string emitted in the OpenAI-shaped envelope. These strings
    /// are stable API — clients (faster-whisper / vLLM tooling) key off them.
    pub fn type_tag(&self) -> &'static str {
        match self {
            Self::ModelNotFound { .. } => "model_not_found",
            Self::UnsupportedOp { .. } => "unsupported_op",
            Self::InvalidInput { .. } => "invalid_input",
            Self::InferenceFailed { .. } => "inference_failed",
            Self::NotImplemented { .. } => "not_implemented",
            Self::InternalPanic { .. } => "internal_panic",
        }
    }

    /// The `code` field. OpenAI uses `code` for a machine-readable short id;
    /// we mirror `type_tag` for now (they're already distinct and stable).
    /// Kept as a separate method so if the two ever need to diverge (e.g.
    /// finer-grained codes under `inference_failed`), only this changes.
    pub fn code(&self) -> &'static str {
        self.type_tag()
    }

    /// The human-readable message emitted in the envelope.
    pub fn message(&self) -> String {
        // Reuses Display so log_request and IntoResponse stay consistent.
        self.to_string()
    }
}

/// Wire-format envelope: `{"error": {"message", "type", "code"}}`. Matches
/// what the OpenAI API returns so clients (faster-whisper, openai-python,
/// vLLM CLI) can decode failures with their existing error path.
#[derive(Serialize)]
struct ErrorEnvelope<'a> {
    error: ErrorBody<'a>,
}

#[derive(Serialize)]
struct ErrorBody<'a> {
    message: String,
    #[serde(rename = "type")]
    type_tag: &'a str,
    code: &'a str,
}

impl IntoResponse for ServerError {
    fn into_response(self) -> Response {
        let status = self.status();
        let body = ErrorEnvelope {
            error: ErrorBody {
                message: self.message(),
                type_tag: self.type_tag(),
                code: self.code(),
            },
        };
        // `Json` sets `content-type: application/json`. `serde_json` cannot
        // fail for a `Serialize` derive on `String`/`&str` fields, so a
        // panic here would itself be a bug — the CatchPanic layer covers
        // that, keeping the request boundary safe.
        (status, Json(body)).into_response()
    }
}

/// The `ResponseForPanic` function pointer the catch-panic layer stores.
/// Pulled out as a `type` alias to keep the return signature of
/// [`catch_panic_layer`] readable (clippy::type_complexity).
pub type PanicResponder = fn(Box<dyn Any + Send + 'static>) -> Response<axum::body::Body>;

/// Build the `tower_http::CatchPanicLayer` that converts any panic inside a
/// handler into an `InternalPanic` → 500 JSON. The response body is the same
/// OpenAI-shaped envelope so clients see a uniform failure schema.
///
/// Attach in `server.rs` with `Router::new().layer(catch_panic_layer())`.
/// NFR-RL-07: this is the single, uniform HTTP-side panic boundary; without
/// it, a `.unwrap()` inside any handler would abort the tokio worker.
pub fn catch_panic_layer() -> CatchPanicLayer<PanicResponder> {
    // Cast the fn item to a plain fn pointer so `CatchPanicLayer`'s
    // `ResponseForPanic` bound is satisfied by the pointer type (fn item
    // types are unnameable and would otherwise force us into a `Box<dyn>`
    // and lose `Clone` on the layer).
    let handler: PanicResponder = panic_payload_to_response;
    CatchPanicLayer::custom(handler)
}

/// Convert a `catch_unwind` payload (`Box<dyn Any + Send>`) into an
/// `InternalPanic` and render as a full HTTP response. Shared by the tower
/// layer AND [`spawn_isolated_wyoming_task`] so both surfaces speak the same
/// wire shape.
fn panic_payload_to_response(payload: Box<dyn Any + Send + 'static>) -> Response<axum::body::Body> {
    let err = ServerError::InternalPanic {
        message: extract_panic_message(&payload),
    };
    err.into_response()
}

/// Best-effort extraction of a printable message from a panic payload.
/// Rust panics carry their payload as `Box<dyn Any + Send>`; the most common
/// concrete types are `&'static str` and `String`. Anything else falls back
/// to a fixed sentinel so we never expose an uninitialized read to callers.
pub(crate) fn extract_panic_message(payload: &(dyn Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&'static str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "panic with non-string payload".to_string()
    }
}

/// Wyoming-side sibling of [`catch_panic_layer`]. Spawns a tokio task and
/// runs the passed future under `AssertUnwindSafe` + `catch_unwind`. A panic
/// closes that one TCP session (the caller drops its stream on return) and
/// logs an `internal_panic` event; other Wyoming sessions and both HTTP
/// listeners keep running.
///
/// Design note (plan D8, R5): the actual JSONL event-loop lands in T14+.
/// Landing this helper at T05 means the loop can be spawned inside the
/// isolation from day one instead of retrofitting it later.
pub fn spawn_isolated_wyoming_task<F>(
    peer: std::net::SocketAddr,
    fut: F,
) -> tokio::task::JoinHandle<()>
where
    F: std::future::Future<Output = ()> + Send + 'static,
{
    tokio::spawn(async move {
        // `AssertUnwindSafe` is required because most futures don't
        // implement `UnwindSafe`; we accept the "state may be inconsistent
        // after panic" trade-off because the task's owned state
        // (`TcpStream`, per-session buffers) is dropped on task exit anyway.
        use futures_util_stub::FutureExt as _;
        let result = std::panic::AssertUnwindSafe(fut).catch_unwind().await;
        if let Err(payload) = result {
            let message = extract_panic_message(&*payload);
            tracing::error!(
                target: "vokra_server",
                surface = "wyoming",
                peer = %peer,
                error_type = ServerError::InternalPanic { message: message.clone() }.type_tag(),
                message = %message,
                "wyoming task panicked; session terminated, listener still up",
            );
        }
    })
}

/// Local shim: `futures::FutureExt::catch_unwind` without adding the
/// `futures` crate to our lockfile. We only need the one adapter, so this
/// keeps the dep footprint of the excluded workspace tighter.
mod futures_util_stub {
    use std::future::Future;
    use std::panic::{AssertUnwindSafe, UnwindSafe};
    use std::pin::Pin;
    use std::task::{Context, Poll};

    /// A future that runs the inner future inside `catch_unwind`. A panic
    /// during the inner future's poll becomes `Err(payload)`.
    pub struct CatchUnwind<F> {
        inner: F,
    }

    impl<F> Future for CatchUnwind<F>
    where
        F: Future + UnwindSafe,
    {
        type Output = std::thread::Result<F::Output>;

        fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
            // SAFETY: standard structural projection; we never move `inner`
            // out of the pin and we never expose a `&mut` past this poll.
            let this = unsafe { self.get_unchecked_mut() };
            let inner = unsafe { Pin::new_unchecked(&mut this.inner) };
            match std::panic::catch_unwind(AssertUnwindSafe(|| inner.poll(cx))) {
                Ok(Poll::Ready(v)) => Poll::Ready(Ok(v)),
                Ok(Poll::Pending) => Poll::Pending,
                Err(payload) => Poll::Ready(Err(payload)),
            }
        }
    }

    /// Trait extension so callers write `fut.catch_unwind().await` like they
    /// would with `futures::FutureExt`.
    pub trait FutureExt: Future + Sized {
        fn catch_unwind(self) -> CatchUnwind<Self>
        where
            Self: UnwindSafe,
        {
            CatchUnwind { inner: self }
        }
    }

    impl<F: Future> FutureExt for F {}
}

/// Structured per-request log emitted once per HTTP request. Fields match
/// the plan §3.2.e list: `method`, `path`, `status`, `latency_ms`, `model`,
/// `error_type`. Emitted at `INFO` for 2xx/3xx and `WARN` for 4xx/5xx so
/// filtering by level surfaces failures without hiding successes.
///
/// The caller (an axum middleware in T06+) records the request start time,
/// runs the inner service, and calls this helper with the terminal status.
pub fn log_request(fields: RequestLogFields<'_>) {
    let status_u16 = fields.status.as_u16();
    let error_type = fields.error.as_deref().unwrap_or("");
    let model = fields.model.unwrap_or("");
    let latency_ms = fields.latency.as_secs_f64() * 1000.0;
    if status_u16 >= 400 {
        tracing::warn!(
            target: "vokra_server",
            surface = "http",
            method = fields.method,
            path = fields.path,
            status = status_u16,
            latency_ms = latency_ms,
            model = model,
            error_type = error_type,
            "request failed",
        );
    } else {
        tracing::info!(
            target: "vokra_server",
            surface = "http",
            method = fields.method,
            path = fields.path,
            status = status_u16,
            latency_ms = latency_ms,
            model = model,
            error_type = error_type,
            "request",
        );
    }
}

/// Fields captured for the per-request tracing event. Borrowed so the
/// middleware (T06+) can pass in the request-scoped `&str` values without
/// cloning.
#[derive(Debug, Clone)]
pub struct RequestLogFields<'a> {
    /// HTTP method (`GET`, `POST`, ...).
    pub method: &'a str,
    /// Matched or original path (e.g. `/v1/audio/transcriptions`).
    pub path: &'a str,
    /// Terminal HTTP status.
    pub status: StatusCode,
    /// Wall-clock latency from request receipt to response emission.
    pub latency: Duration,
    /// The `model=` field the caller asked for, if any (empty on `/health`).
    pub model: Option<&'a str>,
    /// If the request produced a `ServerError`, its `type_tag`. Empty on
    /// success. Kept as `String` so the caller can pass either a stable
    /// `&'static str` or an owned `type_tag().to_string()` freely.
    pub error: Option<String>,
}

/// Convenience: wrap an already-built `Result` at a handler boundary so the
/// per-request log fires exactly once, then convert to the outgoing
/// `Response`. Used by T06/T09/T11/T15+ handlers.
pub fn finish_request<T: IntoResponse>(
    method: &str,
    path: &str,
    model: Option<&str>,
    start: std::time::Instant,
    result: Result<T, ServerError>,
) -> Response {
    match result {
        Ok(ok) => {
            let resp = ok.into_response();
            log_request(RequestLogFields {
                method,
                path,
                status: resp.status(),
                latency: start.elapsed(),
                model,
                error: None,
            });
            resp
        }
        Err(err) => {
            let error_tag = err.type_tag().to_string();
            let resp = err.into_response();
            log_request(RequestLogFields {
                method,
                path,
                status: resp.status(),
                latency: start.elapsed(),
                model,
                error: Some(error_tag),
            });
            resp
        }
    }
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Error mapping ------------------------------------------------------

    /// Every variant's status + type_tag are exhaustively pinned. A future
    /// refactor that changes any of these strings will break clients keying
    /// on the OpenAI-compat schema, so we lock them here.
    #[test]
    fn error_variant_mapping_is_pinned() {
        let cases: &[(ServerError, StatusCode, &str)] = &[
            (
                ServerError::ModelNotFound {
                    model: "nope".into(),
                },
                StatusCode::NOT_FOUND,
                "model_not_found",
            ),
            (
                ServerError::UnsupportedOp {
                    detail: "metal:softmax".into(),
                },
                StatusCode::NOT_IMPLEMENTED,
                "unsupported_op",
            ),
            (
                ServerError::InvalidInput {
                    detail: "missing file".into(),
                },
                StatusCode::BAD_REQUEST,
                "invalid_input",
            ),
            (
                ServerError::InferenceFailed {
                    detail: "gguf read".into(),
                },
                StatusCode::INTERNAL_SERVER_ERROR,
                "inference_failed",
            ),
            (
                ServerError::NotImplemented {
                    detail: "v0.5 llm".into(),
                },
                StatusCode::NOT_IMPLEMENTED,
                "not_implemented",
            ),
            (
                ServerError::InternalPanic {
                    message: "boom".into(),
                },
                StatusCode::INTERNAL_SERVER_ERROR,
                "internal_panic",
            ),
        ];
        for (err, want_status, want_tag) in cases {
            assert_eq!(err.status(), *want_status, "status for {err:?}");
            assert_eq!(err.type_tag(), *want_tag, "tag for {err:?}");
            assert_eq!(err.code(), *want_tag, "code for {err:?}");
            assert!(err.message().contains(match err {
                ServerError::ModelNotFound { model } => model.as_str(),
                ServerError::UnsupportedOp { detail } => detail.as_str(),
                ServerError::InvalidInput { detail } => detail.as_str(),
                ServerError::InferenceFailed { detail } => detail.as_str(),
                ServerError::NotImplemented { detail } => detail.as_str(),
                ServerError::InternalPanic { message } => message.as_str(),
            }));
        }
    }

    /// Renders the wire envelope for one representative variant and confirms
    /// the JSON matches the OpenAI-shape contract. Full 4-endpoint schema
    /// tests land in T08/T10/T13/T17; this covers the shared serializer.
    #[tokio::test]
    async fn error_response_body_is_openai_shape() {
        use axum::body::to_bytes;
        let err = ServerError::UnsupportedOp {
            detail: "metal:layer_norm not implemented".into(),
        };
        let resp = err.into_response();
        assert_eq!(resp.status(), StatusCode::NOT_IMPLEMENTED);
        let body = to_bytes(resp.into_body(), 4096).await.unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let obj = parsed.get("error").expect("has 'error' key");
        assert_eq!(obj["type"], "unsupported_op");
        assert_eq!(obj["code"], "unsupported_op");
        assert!(
            obj["message"]
                .as_str()
                .unwrap()
                .contains("metal:layer_norm not implemented")
        );
    }

    // ---- Panic isolation ----------------------------------------------------
    //
    // These two tests live in a separate `mod panic_isolation` (sibling of
    // `mod tests`) so the T05 spec filter `cargo test error::panic_isolation`
    // resolves them by the exact module-path prefix rather than relying on
    // cargo's substring match. See file docs.
}

#[cfg(test)]
mod panic_isolation {
    use super::*;

    /// The tower CatchPanic layer must convert a panic inside a handler into
    /// a 500 with the OpenAI-shape body. The router itself must keep serving
    /// subsequent requests: a panic on one request is not a panic on the
    /// runtime (NFR-RL-07).
    ///
    /// This is the acceptance test T05 spec calls out
    /// (`cargo test error::panic_isolation`).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn returns_500_json_and_keeps_router_alive() {
        use axum::Router;
        use axum::body::to_bytes;
        use axum::http::{Method, Request};
        use axum::routing::get;
        use tower::ServiceExt as _; // for `oneshot`

        // Router with two routes: `/boom` panics; `/ok` succeeds. Both are
        // covered by the CatchPanic layer so a panic on `/boom` MUST NOT
        // affect a subsequent `/ok`.
        // Type-annotate the panicking handler's return type as `&'static str`
        // so axum's `Handler` bound sees a concrete `IntoResponse` type. The
        // handler diverges before returning, but the compiler still needs an
        // `IntoResponse` type for the return position (`panic!(...)` alone
        // resolves to `!`, which does not implement `IntoResponse`).
        async fn boom_handler() -> &'static str {
            panic!("intentional test panic in handler");
        }
        let app: Router = Router::new()
            .route("/boom", get(boom_handler))
            .route("/ok", get(|| async { "hello" }))
            .layer(catch_panic_layer());

        // 1. Panicking request → 500 + OpenAI-shape body.
        let boom_req = Request::builder()
            .method(Method::GET)
            .uri("/boom")
            .body(axum::body::Body::empty())
            .unwrap();
        let boom_resp = app.clone().oneshot(boom_req).await.unwrap();
        assert_eq!(
            boom_resp.status(),
            StatusCode::INTERNAL_SERVER_ERROR,
            "panicked handler must produce 500, not runtime crash",
        );
        let boom_body = to_bytes(boom_resp.into_body(), 4096).await.unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&boom_body).unwrap();
        let obj = parsed.get("error").expect("body has 'error' key");
        assert_eq!(obj["type"], "internal_panic");
        assert_eq!(obj["code"], "internal_panic");
        let msg = obj["message"].as_str().unwrap();
        assert!(
            msg.contains("internal panic"),
            "message should identify the failure surface, got: {msg}",
        );

        // 2. A follow-up request on the SAME router must still succeed.
        // This is what makes it "isolation" and not just "500 on panic":
        // one bad handler cannot poison the router or the runtime.
        let ok_req = Request::builder()
            .method(Method::GET)
            .uri("/ok")
            .body(axum::body::Body::empty())
            .unwrap();
        let ok_resp = app.oneshot(ok_req).await.unwrap();
        assert_eq!(
            ok_resp.status(),
            StatusCode::OK,
            "router must keep serving after another handler panicked",
        );
    }

    /// `spawn_isolated_wyoming_task` must swallow a panic in the passed
    /// future — the returned `JoinHandle` completes cleanly rather than
    /// reporting a JoinError. NFR-RL-07: one Wyoming session's panic must
    /// not abort peer sessions or the accept loop.
    #[tokio::test]
    async fn wyoming_task_survives_panic() {
        let peer: std::net::SocketAddr = "127.0.0.1:0".parse().unwrap();
        let handle = spawn_isolated_wyoming_task(peer, async {
            panic!("boom in a wyoming session");
        });
        // Must complete WITHOUT a JoinError. If catch_unwind was missing,
        // the spawned task would abort and this would be Err(JoinError).
        tokio::time::timeout(Duration::from_secs(1), handle)
            .await
            .expect("wyoming task did not complete promptly")
            .expect("wyoming task should not surface a JoinError after panic");
    }
}

#[cfg(test)]
mod remaining_tests {
    use super::*;

    // ---- extract_panic_message ---------------------------------------------

    #[test]
    fn error_extract_panic_message_handles_common_payload_shapes() {
        // &'static str payload (from `panic!("literal")`).
        let s: Box<dyn Any + Send> = Box::new("literal panic");
        assert_eq!(extract_panic_message(&*s), "literal panic");
        // String payload (from `panic!("{}", var)`).
        let s: Box<dyn Any + Send> = Box::new(String::from("owned panic"));
        assert_eq!(extract_panic_message(&*s), "owned panic");
        // Other payload → sentinel, never uninitialized read.
        let s: Box<dyn Any + Send> = Box::new(42u32);
        assert_eq!(extract_panic_message(&*s), "panic with non-string payload");
    }

    // ---- finish_request ------------------------------------------------------

    /// `finish_request` renders both success and failure branches through
    /// the same log-then-respond path. We only check the response side
    /// here (the tracing side is validated end-to-end in the compat tests
    /// T08/T10/T13/T17).
    #[tokio::test]
    async fn error_finish_request_renders_success_and_failure() {
        // Success branch.
        let start = std::time::Instant::now();
        let resp = finish_request::<&'static str>(
            "GET",
            "/v1/audio/transcriptions",
            Some("whisper-1"),
            start,
            Ok("hello"),
        );
        assert_eq!(resp.status(), StatusCode::OK);

        // Failure branch.
        let start = std::time::Instant::now();
        let resp = finish_request::<&'static str>(
            "POST",
            "/v1/audio/transcriptions",
            Some("whisper-1"),
            start,
            Err(ServerError::InvalidInput {
                detail: "missing 'file' field".into(),
            }),
        );
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }
}

// -----------------------------------------------------------------------------
// Dead-code and unused-import silencing while T06/T09/T11/T15+ are pending.
// `Arc` is imported for the future service-integration surface (T04 wires
// engines via `Arc<Engine>` and passes them through `ServerError` in a few
// error paths). Suppress `unused_imports` locally rather than dropping the
// import — flipping it back on in T06 would require a re-edit here.
// -----------------------------------------------------------------------------
#[allow(dead_code)]
#[doc(hidden)]
pub(crate) fn _future_service_arc_marker(_: Arc<()>) {}
