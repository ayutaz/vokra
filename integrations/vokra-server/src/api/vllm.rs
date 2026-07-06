//! vLLM `/v1/completions` + `/v1/chat/completions` router (T09 / T10).
//!
//! # Scope (FR-SV-03, plan §3.2.g / D9)
//!
//! v0.5 has NO LLM in-tree: FR-MD-04 covers Whisper large-v3 (ASR) and
//! FR-MD-05 covers Kokoro TTS. Real completion/chat generation ships in
//! v1.0+ (CosyVoice2 / Voxtral) and v1.5+ (Moshi / Helium). This module
//! is contract-first: it accepts the OpenAI-compatible request schema
//! that vLLM speaks, validates the input shape, and returns
//! `501 Not Implemented` with the standard OpenAI error envelope
//! (`{"error": {"message", "type", "code"}}`) via
//! [`crate::error::ServerError::NotImplemented`].
//!
//! # Cross-cutting invariants (see `api/mod.rs`)
//!
//! * **NFR-RL-06 — no fabricated output.** We NEVER emit a plausible-
//!   looking `choices[].text` / `choices[].message.content` in v0.5.
//!   The response is a JSON error, honestly signalling "not implemented".
//! * **FR-EX-08 — no silent fallback.** An LLM model name that vLLM would
//!   normally route to a different backend is not silently mapped to
//!   another modality. If the caller asks for a completion, they get
//!   501, not a transcription or synthesis.
//! * **NFR-RL-07 — API boundary safety.** All parsing failures become
//!   400 `InvalidInput`; runtime panics are caught upstream by the
//!   tower `CatchPanic` layer (see `crate::error::catch_panic_layer`).
//! * **NFR-RL-01 — LC_NUMERIC.** All numeric fields (`temperature`,
//!   `top_p`, `max_tokens`, ...) are parsed by `serde_json`, which does
//!   not use `strtod`. The startup path pins `LC_NUMERIC=C` as a defence
//!   (see `crate::enforce_c_numeric_locale`).
//!
//! # T09 acceptance test
//!
//! The `contract_stub` test module exercises the two routes end-to-end:
//! valid request → 501 + OpenAI-shape `{"error": ...}`; malformed body
//! → 400 `invalid_input`. T10 layers the full schema-conformance test on
//! top when the vLLM contract test suite lands.
//!
//! When real LLM inference lands (v1.0+), the router below is the drop-in
//! point: swap `not_implemented_response` for a call into a new
//! `LlmService` trait added to `crate::service`.

use crate::error::{ServerError, finish_request};
use axum::Json;
use axum::Router;
use axum::extract::rejection::JsonRejection;
use axum::response::Response;
use axum::routing::post;
use serde::{Deserialize, Serialize};

// -----------------------------------------------------------------------------
// Request DTOs (OpenAI + vLLM shape)
// -----------------------------------------------------------------------------

/// `POST /v1/completions` request body. Subset of the OpenAI Completions API
/// large enough for vLLM clients (openai-python, vLLM CLI, LangChain, ...) to
/// pass their request through us and see a well-formed 501.
///
/// Fields are `Option<_>` where OpenAI/vLLM permits omission. `model` and
/// `prompt` are required; the rest are informational for a v0.5 501 response.
///
/// Kept in a hand-rolled struct rather than `serde_json::Value` so the parser
/// enforces the schema shape at request time (an object with a `model` and
/// `prompt` string/array, not arbitrary JSON). This matches how vLLM's own
/// server validates the request and gives us free 400s on obvious garbage.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompletionsRequest {
    /// Model name the caller asked for. In v0.5 every value returns 501 —
    /// no LLM is loaded. We still record it so [`ServerError::NotImplemented`]
    /// carries it into the log fields (`model=` column).
    pub model: String,
    /// Prompt as a single string OR array of strings (OpenAI supports both).
    pub prompt: PromptField,
    /// Optional generation controls. Not consulted in v0.5 but accepted so
    /// clients pass their normal payload through without a schema error.
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub n: Option<u32>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub stop: Option<StopField>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub logit_bias: Option<serde_json::Map<String, serde_json::Value>>,
    #[serde(default)]
    pub user: Option<String>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub suffix: Option<String>,
    #[serde(default)]
    pub echo: Option<bool>,
    #[serde(default)]
    pub best_of: Option<u32>,
    #[serde(default)]
    pub logprobs: Option<u32>,
}

/// `prompt` may be a string or an array of strings (OpenAI Completions API).
#[derive(Debug, Deserialize, Serialize)]
#[serde(untagged)]
pub enum PromptField {
    Single(String),
    Batch(Vec<String>),
}

/// `stop` may be a string or an array of strings.
#[derive(Debug, Deserialize, Serialize)]
#[serde(untagged)]
pub enum StopField {
    Single(String),
    Multi(Vec<String>),
}

/// `POST /v1/chat/completions` request body. Same shape treatment as
/// [`CompletionsRequest`]: `model` + `messages` required, generation controls
/// optional and informational.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChatCompletionsRequest {
    pub model: String,
    /// Non-empty conversation. We enforce non-empty in `validate_chat` so a
    /// zero-message request is a 400 `invalid_input`, not a 501.
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub n: Option<u32>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub stop: Option<StopField>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    #[serde(default)]
    pub logit_bias: Option<serde_json::Map<String, serde_json::Value>>,
    #[serde(default)]
    pub user: Option<String>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub tools: Option<serde_json::Value>,
    #[serde(default)]
    pub tool_choice: Option<serde_json::Value>,
    #[serde(default)]
    pub response_format: Option<serde_json::Value>,
    #[serde(default)]
    pub logprobs: Option<bool>,
    #[serde(default)]
    pub top_logprobs: Option<u32>,
}

/// One chat message. `role` is one of `system` / `user` / `assistant` / `tool`
/// per the OpenAI spec; we accept any string and defer validation to the
/// (future) LLM service so vLLM's extension roles (e.g. `developer`) still
/// parse. Content can be a plain string or an array of content parts
/// (multimodal); v0.5 accepts both shapes as opaque JSON.
#[derive(Debug, Deserialize, Serialize)]
pub struct ChatMessage {
    pub role: String,
    #[serde(default)]
    pub content: Option<serde_json::Value>,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub tool_calls: Option<serde_json::Value>,
    #[serde(default)]
    pub tool_call_id: Option<String>,
}

// -----------------------------------------------------------------------------
// Router
// -----------------------------------------------------------------------------

/// Build the vLLM-compatible router. Mount under the top-level app in
/// `server.rs` with `.merge(api::vllm::router())` when T09 lands there.
///
/// Kept as a bare `Router` (no state generic) because v0.5 does not touch
/// the inference registry — every request is a fast schema check followed
/// by 501. When LLM inference lands (v1.0+), this becomes
/// `Router::<Arc<InferenceService>>::new()...` and picks up state exactly
/// like the OpenAI transcriptions router (T06).
pub fn router() -> Router {
    Router::new()
        .route("/v1/completions", post(handle_completions))
        .route("/v1/chat/completions", post(handle_chat_completions))
}

/// `POST /v1/completions` handler.
///
/// v0.5 behaviour: parse → validate → 501 `NotImplemented`. Any malformed
/// body (bad JSON, missing `model`/`prompt`, unknown field due to
/// `deny_unknown_fields`) becomes 400 `InvalidInput` via [`extract_model`].
async fn handle_completions(body: Result<Json<CompletionsRequest>, JsonRejection>) -> Response {
    let start = std::time::Instant::now();
    let (model, result) = match body {
        Ok(Json(req)) => (Some(req.model.clone()), Err(not_implemented("completions"))),
        Err(err) => (None, Err(invalid_input_from_rejection(err))),
    };
    finish_request::<Json<()>>("POST", "/v1/completions", model.as_deref(), start, result)
}

/// `POST /v1/chat/completions` handler. Same shape as completions plus a
/// non-empty `messages` check.
async fn handle_chat_completions(
    body: Result<Json<ChatCompletionsRequest>, JsonRejection>,
) -> Response {
    let start = std::time::Instant::now();
    let (model, result) = match body {
        Ok(Json(req)) => {
            if req.messages.is_empty() {
                (
                    Some(req.model),
                    Err(ServerError::InvalidInput {
                        detail: "`messages` must be a non-empty array".into(),
                    }),
                )
            } else {
                (Some(req.model), Err(not_implemented("chat.completions")))
            }
        }
        Err(err) => (None, Err(invalid_input_from_rejection(err))),
    };
    finish_request::<Json<()>>(
        "POST",
        "/v1/chat/completions",
        model.as_deref(),
        start,
        result,
    )
}

/// The single, honest v0.5 response: 501 + OpenAI-shape error envelope.
/// Message text mirrors the plan §3.2.g wording so clients (and grep-ing
/// operators) see a consistent signal across both endpoints.
fn not_implemented(surface: &str) -> ServerError {
    ServerError::NotImplemented {
        detail: format!("LLM inference not available in v0.5 ({surface}); scheduled for v1.0+",),
    }
}

/// Convert an axum `JsonRejection` into our 400 `InvalidInput`. This is the
/// path that catches unknown fields (via `deny_unknown_fields`), wrong types,
/// missing required fields, and unparseable JSON — vLLM clients see one
/// stable OpenAI-shape error instead of axum's default rejection body.
fn invalid_input_from_rejection(err: JsonRejection) -> ServerError {
    ServerError::InvalidInput {
        detail: err.to_string(),
    }
}

// -----------------------------------------------------------------------------
// Tests (T09 contract stub)
// -----------------------------------------------------------------------------
//
// Invoked by `cargo test vllm::contract_stub` (see M2-09 T09 spec).
// T10 will layer the vLLM compat matrix on top; the base contract lives here.

#[cfg(test)]
mod contract_stub {
    use super::*;
    use axum::body::to_bytes;
    use axum::http::{Method, Request, StatusCode};
    use tower::ServiceExt as _;

    /// Helper: POST a JSON body to `path` on our router and return the
    /// terminal (status, body-json) pair. Uses `oneshot` so no bind is needed
    /// — the test suite runs in-process and never touches a real socket.
    async fn post_json(path: &str, body: serde_json::Value) -> (StatusCode, serde_json::Value) {
        let app = router();
        let req = Request::builder()
            .method(Method::POST)
            .uri(path)
            .header("content-type", "application/json")
            .body(axum::body::Body::from(body.to_string()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        let status = resp.status();
        let bytes = to_bytes(resp.into_body(), 16 * 1024).await.unwrap();
        let json: serde_json::Value = if bytes.is_empty() {
            serde_json::Value::Null
        } else {
            serde_json::from_slice(&bytes).expect("response body must be JSON")
        };
        (status, json)
    }

    /// A valid Completions request (any model name) must return 501 with the
    /// OpenAI-shape error envelope. NFR-RL-06: no fabricated `choices[]`.
    #[tokio::test]
    async fn completions_returns_501_not_implemented_with_openai_envelope() {
        let (status, body) = post_json(
            "/v1/completions",
            serde_json::json!({
                "model": "gpt-3.5-turbo-instruct",
                "prompt": "Hello, world!",
                "max_tokens": 16,
                "temperature": 0.7,
            }),
        )
        .await;
        assert_eq!(status, StatusCode::NOT_IMPLEMENTED);
        let err = body.get("error").expect("body has 'error' key");
        assert_eq!(err["type"], "not_implemented");
        assert_eq!(err["code"], "not_implemented");
        let msg = err["message"].as_str().unwrap();
        assert!(
            msg.contains("LLM inference not available in v0.5"),
            "message must honestly say v0.5 lacks LLM, got: {msg}",
        );
        // NFR-RL-06 pin: response must NOT contain a synthesized generation.
        assert!(
            body.get("choices").is_none(),
            "must not fabricate choices[] at v0.5",
        );
        assert!(
            body.get("text").is_none(),
            "must not fabricate top-level text at v0.5",
        );
    }

    /// Prompt as an array (OpenAI batch form) must also parse and 501.
    #[tokio::test]
    async fn completions_accepts_prompt_array_form() {
        let (status, body) = post_json(
            "/v1/completions",
            serde_json::json!({
                "model": "any-llm",
                "prompt": ["hello", "world"],
            }),
        )
        .await;
        assert_eq!(status, StatusCode::NOT_IMPLEMENTED);
        assert_eq!(body["error"]["type"], "not_implemented");
    }

    /// A valid Chat Completions request must also return 501 with the same
    /// envelope shape (only `detail` differs to name the surface).
    #[tokio::test]
    async fn chat_completions_returns_501_not_implemented_with_openai_envelope() {
        let (status, body) = post_json(
            "/v1/chat/completions",
            serde_json::json!({
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Say hi."},
                ],
                "temperature": 0.2,
            }),
        )
        .await;
        assert_eq!(status, StatusCode::NOT_IMPLEMENTED);
        let err = body.get("error").expect("body has 'error' key");
        assert_eq!(err["type"], "not_implemented");
        assert_eq!(err["code"], "not_implemented");
        assert!(
            err["message"]
                .as_str()
                .unwrap()
                .contains("LLM inference not available in v0.5"),
        );
        // NFR-RL-06 pin: response must NOT contain a synthesized assistant message.
        assert!(body.get("choices").is_none());
    }

    /// Chat message content may be multimodal (array of parts). We accept the
    /// shape as opaque JSON and still 501 without a parse error.
    #[tokio::test]
    async fn chat_completions_accepts_multimodal_content() {
        let (status, _body) = post_json(
            "/v1/chat/completions",
            serde_json::json!({
                "model": "gpt-4o",
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe this image"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
                    ],
                }],
            }),
        )
        .await;
        assert_eq!(status, StatusCode::NOT_IMPLEMENTED);
    }

    /// Missing required field (`prompt`) → 400 `invalid_input`, not 501.
    /// vLLM's own server behaves the same way, so this preserves parity.
    #[tokio::test]
    async fn completions_missing_prompt_is_invalid_input() {
        let (status, body) =
            post_json("/v1/completions", serde_json::json!({ "model": "foo" })).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["error"]["type"], "invalid_input");
    }

    /// Empty `messages` array in chat completions → 400 `invalid_input`.
    #[tokio::test]
    async fn chat_completions_empty_messages_is_invalid_input() {
        let (status, body) = post_json(
            "/v1/chat/completions",
            serde_json::json!({ "model": "foo", "messages": [] }),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["error"]["type"], "invalid_input");
        assert!(
            body["error"]["message"]
                .as_str()
                .unwrap()
                .contains("`messages` must be a non-empty array"),
        );
    }

    /// Unknown top-level field is rejected as invalid_input (via
    /// `#[serde(deny_unknown_fields)]`). This closes a common vLLM-client
    /// bug class where a mis-spelled field is silently ignored server-side.
    #[tokio::test]
    async fn completions_unknown_field_is_invalid_input() {
        let (status, body) = post_json(
            "/v1/completions",
            serde_json::json!({
                "model": "foo",
                "prompt": "hi",
                "not_a_real_field": 123,
            }),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["error"]["type"], "invalid_input");
    }

    /// Non-JSON body → 400 `invalid_input`. Confirms we hand back our own
    /// envelope even when the raw JSON parser fails before serde-derive runs.
    #[tokio::test]
    async fn completions_non_json_body_is_invalid_input() {
        let app = router();
        let req = Request::builder()
            .method(Method::POST)
            .uri("/v1/completions")
            .header("content-type", "application/json")
            .body(axum::body::Body::from("not json at all"))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let bytes = to_bytes(resp.into_body(), 4096).await.unwrap();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body["error"]["type"], "invalid_input");
    }

    /// Chat completions with wrong content-type still returns our envelope
    /// (not axum's default plaintext rejection).
    #[tokio::test]
    async fn chat_completions_wrong_content_type_is_invalid_input() {
        let app = router();
        let req = Request::builder()
            .method(Method::POST)
            .uri("/v1/chat/completions")
            .header("content-type", "text/plain")
            .body(axum::body::Body::from(r#"{"model":"x","messages":[]}"#))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let bytes = to_bytes(resp.into_body(), 4096).await.unwrap();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body["error"]["type"], "invalid_input");
    }

    /// Only the two documented routes exist; a sibling path is 404 by axum
    /// default. Guards against accidental "greedy" route glob later.
    #[tokio::test]
    async fn no_extra_routes_are_registered() {
        let app = router();
        let req = Request::builder()
            .method(Method::POST)
            .uri("/v1/embeddings")
            .header("content-type", "application/json")
            .body(axum::body::Body::from("{}"))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }
}
