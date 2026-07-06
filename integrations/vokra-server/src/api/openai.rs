//! OpenAI `/v1/audio/transcriptions` HTTP surface + response schema (T06 + T07).
//!
//! # What this cut lands
//!
//! * **T06 (this change)** — HTTP routing + `multipart/form-data`
//!   extractor for `POST /v1/audio/transcriptions`. Parses the OpenAI
//!   documented fields (`file`, `model`, `language`, `response_format`,
//!   `temperature`, `prompt`), decodes the uploaded audio to `f32` PCM
//!   mono at 16 kHz, and dispatches through
//!   [`transcribe_to_response`] (the T07 dispatch layer landed below).
//!   The route is attached via [`attach_routes`] so T09/T11 can hang
//!   their handlers on the same axum router without churn.
//! * **T07 (landed earlier)** — the pure, in-process schema layer that
//!   maps a resolved `(model, response_format, pcm)` triple through
//!   [`TranscribeService`](crate::service::TranscribeService) into a
//!   faster-whisper drop-in response body.
//!
//! # faster-whisper drop-in schema
//!
//! * Default (`response_format` unset OR `"json"`) → JSON body
//!   `{"text": "<transcribed>"}`. This is the shape the OpenAI Python
//!   SDK, the `faster-whisper` `WhisperModel.transcribe()` FastAPI
//!   servers, and every documented client (Home Assistant Whisper
//!   integration, LangChain OpenAI Whisper wrapper, etc.) already
//!   parse.
//! * `"text"` → same string in a plain-text body (documented for T06 —
//!   this file returns the string; T06 sets `text/plain` content-type).
//! * `"verbose_json"`, `"srt"`, `"vtt"` → **501 Not Implemented** with
//!   a stable, documented note. Word-level timestamps are deferred to
//!   v1.0+ (see the top-of-crate scope doc in `main.rs`: audio-dialect
//!   `beam_search` already carries a word-level-timestamps attribute
//!   spec, but the server surface is intentionally not wired until
//!   v1.0). Returning 501 (not silent stubs, not 500) is the FR-EX-08
//!   "no silent fallback" rule applied to a schema hole.
//!
//! # Wiring
//!
//! [`transcribe_to_response`] is the single entry point the T06 axum
//! handler will call once it has decoded the multipart body. It:
//!
//! 1. Parses `response_format` (case-insensitive; unknown → 400 via
//!    [`OpenAiTranscribeError::UnsupportedResponseFormat`]).
//! 2. If the format is one of the not-yet-wired variants, returns
//!    [`OpenAiTranscribeError::VerboseJsonNotImplemented`] etc. — the
//!    HTTP mapper (T05) will render 501 with the exact note string.
//! 3. Otherwise dispatches to the injected
//!    [`TranscribeService`](crate::service::TranscribeService), which
//!    connects to `WhisperAsr` via the T04 [`InferenceService`](crate::service::InferenceService).
//! 4. Wraps the resulting text in [`TranscriptionResponse`], ready to
//!    be serialized with `serde_json::to_vec` for the response body.
//!
//! No silent CPU fallback (FR-EX-08): `ServiceError::Inference(inner)`
//! propagates as [`OpenAiTranscribeError::Inference`] so the HTTP
//! mapper can pick 4xx vs 5xx off the inner [`VokraError`].

// `OpenAiTranscribeError::Service(ServiceError)` is intentionally rich
// (mirrors `service.rs`, which carries the same allow at the module
// level); we propagate it end-to-end so the T05 HTTP mapper can pick
// 4xx vs 5xx off the inner variant. Boxing here would erase that
// pattern-match affordance for callers.
#![allow(clippy::result_large_err)]

use serde::{Deserialize, Serialize};

use crate::service::{ServiceError, TranscribeService};

/// `response_format` field parsed from the OpenAI multipart body.
///
/// Kept as a plain enum (not `serde` tagged) because we accept the raw
/// string from the multipart field, run case-insensitive matching, and
/// only *then* fold it into this enum. This mirrors what the OpenAI
/// SDK sends (`"json"`, `"text"`, `"verbose_json"`, `"srt"`, `"vtt"`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResponseFormat {
    /// `{"text": "..."}` JSON body. Default when `response_format` is
    /// absent.
    Json,
    /// Plain-text body containing just the transcription. Content-type
    /// `text/plain; charset=utf-8` (T06 sets the header).
    Text,
    /// Verbose JSON with segment / word-level timestamps. **Not
    /// implemented in v0.5**; word-level timestamps are deferred to
    /// v1.0+ (audio-dialect `beam_search` word-timestamps attribute
    /// exists but the server surface is intentionally not wired).
    VerboseJson,
    /// SRT subtitle body. **Not implemented in v0.5** — depends on
    /// per-segment timestamps, same deferral as `VerboseJson`.
    Srt,
    /// WebVTT subtitle body. **Not implemented in v0.5** — same
    /// deferral as `Srt`.
    Vtt,
}

impl ResponseFormat {
    /// Parses the raw multipart string. Case-insensitive because
    /// clients in the wild (Home Assistant configs, curl examples,
    /// SDK versions) mix casing. `None` maps to [`Self::Json`] (OpenAI
    /// default). Unknown strings return `Err(raw)` so the caller can
    /// surface an [`OpenAiTranscribeError::UnsupportedResponseFormat`]
    /// with the offending value.
    ///
    /// # Errors
    ///
    /// Returns the raw (un-normalised) string when it is neither empty
    /// nor one of the five OpenAI-documented formats.
    pub fn parse(raw: Option<&str>) -> Result<Self, String> {
        match raw {
            None => Ok(Self::Json),
            Some(s) => {
                let t = s.trim();
                if t.is_empty() {
                    return Ok(Self::Json);
                }
                // ASCII lowercase compare — the five documented formats
                // are all ASCII so this is loss-less and locale-safe
                // (NFR-RL-01 also pins `LC_ALL=C` at startup).
                let low = t.to_ascii_lowercase();
                match low.as_str() {
                    "json" => Ok(Self::Json),
                    "text" => Ok(Self::Text),
                    "verbose_json" => Ok(Self::VerboseJson),
                    "srt" => Ok(Self::Srt),
                    "vtt" => Ok(Self::Vtt),
                    _ => Err(s.to_owned()),
                }
            }
        }
    }

    /// Returns `true` when this format needs word-level or per-segment
    /// timestamps, which are the M2-09 v0.5 deferral scope.
    pub fn requires_timestamps(self) -> bool {
        matches!(self, Self::VerboseJson | Self::Srt | Self::Vtt)
    }
}

/// Default `{"text": "..."}` response body (`response_format="json"`
/// and the unset default). This is the exact shape faster-whisper
/// FastAPI servers, the OpenAI Python SDK, Home Assistant's Whisper
/// integration, and every documented drop-in client parse. Do **not**
/// add fields to this struct without bumping the compat matrix in the
/// T23 docs — clients pattern-match `{"text": ...}` and additional
/// top-level keys are a compat hazard.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionResponse {
    /// The transcribed text. Never `null`; empty audio yields `""`.
    pub text: String,
}

impl TranscriptionResponse {
    /// Constructs the default JSON body.
    pub fn new(text: impl Into<String>) -> Self {
        Self { text: text.into() }
    }
}

/// The high-level outcome of a `/v1/audio/transcriptions` request as
/// far as this schema layer is concerned. The T06 handler wraps this
/// into an axum `Response` (JSON body, plain-text body, or the T05
/// error mapper for the not-implemented / error paths).
#[derive(Debug, Clone)]
pub enum TranscriptionOutcome {
    /// JSON body: `{"text": "..."}`. Content-type
    /// `application/json`.
    Json(TranscriptionResponse),
    /// Plain-text body: the transcription itself. Content-type
    /// `text/plain; charset=utf-8`.
    Text(String),
}

/// Errors this schema layer surfaces to the T05 HTTP error mapper.
///
/// Kept distinct from [`ServiceError`] so the mapper can pick status
/// codes off the schema-level reason (400 for bad
/// `response_format`, 501 for the deferred variants) without
/// re-parsing an inner text bag (FR-EX-08 spirit: preserve the
/// failure kind end-to-end).
#[derive(Debug)]
pub enum OpenAiTranscribeError {
    /// The client sent a `response_format` we do not recognise.
    /// HTTP mapper renders 400.
    UnsupportedResponseFormat(String),
    /// The client asked for `response_format="verbose_json"`. Not
    /// implemented in v0.5 — word-level timestamps are deferred to
    /// v1.0+ (see the top-of-file docs). HTTP mapper renders 501 with
    /// [`Self::V0_5_TIMESTAMP_DEFERRAL_NOTE`].
    VerboseJsonNotImplemented,
    /// The client asked for `response_format="srt"`. Same deferral
    /// as [`Self::VerboseJsonNotImplemented`]; distinct variant so
    /// the mapper can log the exact client ask.
    SrtNotImplemented,
    /// The client asked for `response_format="vtt"`. Same deferral as
    /// [`Self::VerboseJsonNotImplemented`].
    VttNotImplemented,
    /// The T04 service layer refused the request (unknown model,
    /// Kokoro-style unavailability, inference failure). Preserved
    /// verbatim so the T05 mapper can pick 4xx vs 5xx off the inner
    /// variant.
    Service(ServiceError),
}

impl OpenAiTranscribeError {
    /// Stable, doc-referenced note the T05 HTTP mapper embeds in the
    /// 501 response body for the three deferred formats. Kept as a
    /// `const &'static str` so tests can assert on it exactly and the
    /// T23 docs can reference it byte-for-byte.
    pub const V0_5_TIMESTAMP_DEFERRAL_NOTE: &'static str = "response_format requires per-segment or word-level timestamps; \
         deferred to v1.0+ (see docs/deliverables.md for the timestamp \
         surface plan). Use response_format=\"json\" or \"text\" in v0.5.";
}

impl std::fmt::Display for OpenAiTranscribeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedResponseFormat(s) => {
                write!(f, "unsupported response_format: `{s}`")
            }
            Self::VerboseJsonNotImplemented => {
                write!(
                    f,
                    "response_format=verbose_json not implemented: {}",
                    Self::V0_5_TIMESTAMP_DEFERRAL_NOTE
                )
            }
            Self::SrtNotImplemented => {
                write!(
                    f,
                    "response_format=srt not implemented: {}",
                    Self::V0_5_TIMESTAMP_DEFERRAL_NOTE
                )
            }
            Self::VttNotImplemented => {
                write!(
                    f,
                    "response_format=vtt not implemented: {}",
                    Self::V0_5_TIMESTAMP_DEFERRAL_NOTE
                )
            }
            Self::Service(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for OpenAiTranscribeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Service(e) => Some(e),
            _ => None,
        }
    }
}

/// Dispatch the request through [`TranscribeService`] and shape the
/// response according to `response_format`.
///
/// This is the pure function the T06 axum handler calls once it has
/// extracted the multipart fields. Splitting it out (vs inlining in
/// the handler) lets the `response_schema` test module drive the full
/// schema surface without a live tokio runtime or multipart body.
///
/// * `service` — injected trait object (real `InferenceService` in
///   production, mocks in T08 tests). Connects the OpenAI surface to
///   the T04 `InferenceService` which owns the `WhisperAsr` engines.
/// * `model` — raw `model` field from the multipart body. Forwarded
///   verbatim to `TranscribeService`; unknown → [`OpenAiTranscribeError::Service`]
///   wrapping [`ServiceError::UnknownModel`].
/// * `response_format_raw` — raw multipart field; `None` = default
///   (JSON).
/// * `pcm` — mono `f32` PCM at 16 kHz, produced by the T06 audio
///   decoder.
///
/// # Errors
///
/// See [`OpenAiTranscribeError`] variants.
pub fn transcribe_to_response(
    service: &dyn TranscribeService,
    model: &str,
    response_format_raw: Option<&str>,
    pcm: &[f32],
) -> Result<TranscriptionOutcome, OpenAiTranscribeError> {
    let fmt = ResponseFormat::parse(response_format_raw)
        .map_err(OpenAiTranscribeError::UnsupportedResponseFormat)?;

    // Reject deferred formats up-front — do NOT run inference we then
    // can't shape (wastes engine time; FR-EX-08 no silent fallback).
    match fmt {
        ResponseFormat::VerboseJson => {
            return Err(OpenAiTranscribeError::VerboseJsonNotImplemented);
        }
        ResponseFormat::Srt => return Err(OpenAiTranscribeError::SrtNotImplemented),
        ResponseFormat::Vtt => return Err(OpenAiTranscribeError::VttNotImplemented),
        ResponseFormat::Json | ResponseFormat::Text => {}
    }

    let text = service
        .transcribe(model, pcm)
        .map_err(OpenAiTranscribeError::Service)?;

    Ok(match fmt {
        ResponseFormat::Json => TranscriptionOutcome::Json(TranscriptionResponse::new(text)),
        ResponseFormat::Text => TranscriptionOutcome::Text(text),
        // The three deferred variants are handled above.
        ResponseFormat::VerboseJson | ResponseFormat::Srt | ResponseFormat::Vtt => unreachable!(),
    })
}

// ===========================================================================
// T06 — HTTP surface: axum routing + `multipart/form-data` extractor.
//
// Lands the actual `POST /v1/audio/transcriptions` endpoint on top of the
// T07 schema layer above. Kept in this file so the schema types and the
// HTTP surface stay together (T05 will absorb `HttpTranscriptionError`
// into the crate-wide `ServerError` without moving anything).
// ===========================================================================

use axum::Router;
use axum::extract::{DefaultBodyLimit, Multipart, State};
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use std::sync::Arc;

/// Upper bound on the raw multipart body for the transcription route.
/// Mirrors OpenAI's documented 25 MiB cap on `/v1/audio/transcriptions`
/// and doubles as a DoS guard on the file field. Enforced per-route via
/// [`DefaultBodyLimit::max`], so an oversized body is rejected by the
/// extractor before any handler code runs.
pub const MAX_BODY_BYTES: usize = 25 * 1024 * 1024;

/// Target sample rate for the Whisper front-end. All decoded PCM is
/// bounced to `f32` mono at this rate before dispatch. Any WAV header
/// carrying a different rate is rejected up-front (see
/// [`decode_pcm_wav`]); resampling belongs in the front-end
/// (`vokra-ops::resample`), not in the HTTP handler.
///
/// Matches `vokra-core`'s `frontend_spec` for Whisper base / large-v3
/// (`sample_rate = 16000`, `n_mels = 80` at v0.5).
pub const TARGET_SAMPLE_RATE_HZ: u32 = 16_000;

/// Default `model` when the client omits the multipart field.
/// `whisper-1` is the OpenAI catalogue alias for Whisper base;
/// [`InferenceService::resolve_asr`](crate::service::InferenceService::resolve_asr)
/// already maps it to the always-present base engine.
const DEFAULT_MODEL: &str = crate::service::model_names::WHISPER_1;

/// Attaches the OpenAI `/v1/audio/transcriptions` route (and its 25 MiB
/// body cap) to `router`. Callers own the router / State composition;
/// this function stays layer-only so T09 / T11 can hang their routes on
/// the same instance without churn.
///
/// The `S` state bound requires the caller's app state to expose an
/// `Arc<dyn TranscribeService>` via `FromRef` — the same trait object
/// shape the T04 [`InferenceService`](crate::service::InferenceService)
/// implements. This keeps the handler free of any concrete engine type,
/// which is what lets the T06 tests inject a fake service without
/// linking a real Whisper GGUF.
pub fn attach_routes<S>(router: Router<S>) -> Router<S>
where
    S: Clone + Send + Sync + 'static,
    Arc<dyn TranscribeService>: axum::extract::FromRef<S>,
{
    router.route(
        "/v1/audio/transcriptions",
        post(transcriptions_handler).layer(DefaultBodyLimit::max(MAX_BODY_BYTES)),
    )
}

/// `POST /v1/audio/transcriptions` — the OpenAI Audio API handler.
///
/// Parses the `multipart/form-data` body into a [`TranscriptionRequest`],
/// decodes the uploaded WAV to `f32` PCM mono at
/// [`TARGET_SAMPLE_RATE_HZ`], and dispatches to
/// [`transcribe_to_response`] (T07). The T07 layer resolves the model
/// name mapping (`whisper-1` → base, `whisper-large-v3` → large-v3),
/// runs the engine, and produces the faster-whisper drop-in envelope.
///
/// Failures propagate through [`HttpTranscriptionError`] into the
/// OpenAI-shaped `{"error": {"message", "type", "code"}}` JSON envelope.
/// No silent CPU fallback (FR-EX-08): every failure kind is preserved
/// end-to-end.
pub async fn transcriptions_handler(
    State(service): State<Arc<dyn TranscribeService>>,
    multipart: Multipart,
) -> Result<Response, HttpTranscriptionError> {
    let req = TranscriptionRequest::from_multipart(multipart).await?;
    let TranscriptionRequest {
        audio_bytes,
        model,
        response_format,
        // T06 accepts and preserves these fields on the request DTO.
        // Forwarding them into the Whisper decoder options is a
        // T07-followup responsibility; ignoring them here is intentional
        // (documented) rather than a mistake — the T07 dispatch
        // signature only accepts `(model, response_format, pcm)` at v0.5.
        language: _,
        temperature: _,
        prompt: _,
    } = req;

    let pcm = decode_pcm_wav(&audio_bytes)?;
    let outcome =
        transcribe_to_response(service.as_ref(), &model, response_format.as_deref(), &pcm)
            .map_err(HttpTranscriptionError::from_schema)?;

    Ok(match outcome {
        TranscriptionOutcome::Json(body) => axum::Json(body).into_response(),
        TranscriptionOutcome::Text(text) => (
            StatusCode::OK,
            [(header::CONTENT_TYPE, "text/plain; charset=utf-8")],
            text,
        )
            .into_response(),
    })
}

/// Parsed multipart form. Mirrors the OpenAI request shape.
///
/// Made `pub` so T07 / T08 followups can extend it (e.g. add
/// `word_timestamps: bool`) without editing the handler signature.
#[derive(Debug)]
pub struct TranscriptionRequest {
    /// Raw bytes of the uploaded audio file. The `Content-Type` from
    /// the multipart part is not trusted; the WAV RIFF header is the
    /// source of truth (see [`decode_pcm_wav`]).
    pub audio_bytes: Vec<u8>,
    /// Requested model name (post-default). Passed verbatim to the T07
    /// dispatch layer; the service registry does the alias mapping.
    pub model: String,
    /// Optional ISO-639 language hint (documented pass-through at T06;
    /// T07-followup forwards it to Whisper's decoder options).
    pub language: Option<String>,
    /// Raw `response_format` string as sent by the client. `None` when
    /// the field was absent. T07's [`ResponseFormat::parse`] does the
    /// case-insensitive matching (and rejects unknowns with a 400).
    pub response_format: Option<String>,
    /// Optional sampling temperature in `[0.0, 1.0]`. Parsed with
    /// `str::parse::<f32>` which is locale-independent (NFR-RL-01).
    pub temperature: Option<f32>,
    /// Optional decoding prompt. Size-bounded by the outer 25 MiB body
    /// cap ([`MAX_BODY_BYTES`]).
    pub prompt: Option<String>,
}

impl TranscriptionRequest {
    /// Drains the multipart stream, populating each field it recognises
    /// and rejecting the request the moment an OpenAI invariant is
    /// broken (missing `file`, unknown field, malformed number).
    ///
    /// Unknown fields fail closed rather than being silently discarded
    /// so a typo like `respose_format` surfaces as
    /// `400 invalid_multipart` instead of silently defaulting. This is
    /// the FR-EX-08 "no silent fallback" rule applied to the request
    /// surface.
    pub async fn from_multipart(mut multipart: Multipart) -> Result<Self, HttpTranscriptionError> {
        let mut audio_bytes: Option<Vec<u8>> = None;
        let mut model: Option<String> = None;
        let mut language: Option<String> = None;
        let mut response_format: Option<String> = None;
        let mut temperature: Option<f32> = None;
        let mut prompt: Option<String> = None;

        while let Some(field) = multipart
            .next_field()
            .await
            .map_err(|e| HttpTranscriptionError::bad_multipart(format!("{e}")))?
        {
            // `name()` borrows the field; take a fresh String before we
            // consume the field with `bytes()` / `text()`.
            let Some(name) = field.name().map(|s| s.to_owned()) else {
                return Err(HttpTranscriptionError::bad_multipart(
                    "multipart field without a name".to_owned(),
                ));
            };

            match name.as_str() {
                "file" => {
                    if audio_bytes.is_some() {
                        return Err(HttpTranscriptionError::bad_multipart(
                            "duplicate `file` field".to_owned(),
                        ));
                    }
                    let bytes = field.bytes().await.map_err(|e| {
                        HttpTranscriptionError::bad_multipart(format!("read `file`: {e}"))
                    })?;
                    audio_bytes = Some(bytes.to_vec());
                }
                "model" => model = Some(read_text_field(field, "model").await?),
                "language" => language = Some(read_text_field(field, "language").await?),
                "response_format" => {
                    response_format = Some(read_text_field(field, "response_format").await?);
                }
                "temperature" => {
                    let s = read_text_field(field, "temperature").await?;
                    // `str::parse::<f32>` is locale-independent (see the
                    // NFR-RL-01 note at the top of the module).
                    let value = s.parse::<f32>().map_err(|_| {
                        HttpTranscriptionError::bad_multipart(format!(
                            "`temperature` is not a finite number: {s:?}"
                        ))
                    })?;
                    if !value.is_finite() {
                        return Err(HttpTranscriptionError::bad_multipart(format!(
                            "`temperature` must be finite, got {value}"
                        )));
                    }
                    // OpenAI documents `[0.0, 1.0]` for this field.
                    if !(0.0..=1.0).contains(&value) {
                        return Err(HttpTranscriptionError::bad_multipart(format!(
                            "`temperature` must be in [0.0, 1.0], got {value}"
                        )));
                    }
                    temperature = Some(value);
                }
                "prompt" => prompt = Some(read_text_field(field, "prompt").await?),
                other => {
                    return Err(HttpTranscriptionError::bad_multipart(format!(
                        "unknown multipart field `{other}`"
                    )));
                }
            }
        }

        let Some(audio_bytes) = audio_bytes else {
            return Err(HttpTranscriptionError::bad_multipart(
                "missing required `file` field".to_owned(),
            ));
        };
        if audio_bytes.is_empty() {
            return Err(HttpTranscriptionError::bad_multipart(
                "`file` field is empty".to_owned(),
            ));
        }

        Ok(Self {
            audio_bytes,
            model: model.unwrap_or_else(|| DEFAULT_MODEL.to_owned()),
            language,
            response_format,
            temperature,
            prompt,
        })
    }
}

/// Reads a text-only multipart field into an owned `String`.
async fn read_text_field(
    field: axum::extract::multipart::Field<'_>,
    field_name: &'static str,
) -> Result<String, HttpTranscriptionError> {
    field
        .text()
        .await
        .map_err(|e| HttpTranscriptionError::bad_multipart(format!("read `{field_name}`: {e}")))
}

// ---------------------------------------------------------------------------
// Audio decode — WAV RIFF only at T06.
// ---------------------------------------------------------------------------

/// Decodes a PCM-WAV byte slice to `f32` mono at
/// [`TARGET_SAMPLE_RATE_HZ`].
///
/// Supported subset (everything else is rejected with `400
/// unsupported_audio_format`):
///
/// * RIFF / WAVE container, `fmt ` + `data` chunks (order-agnostic).
/// * PCM integer (`format 1`) at 16 bit or 32 bit, or IEEE-float
///   (`format 3`) at 32 bit.
/// * Mono or stereo (stereo is averaged to mono).
/// * Sample rate exactly [`TARGET_SAMPLE_RATE_HZ`] — resampling belongs
///   in the front-end, not the HTTP handler.
///
/// Deliberately narrow: T06 is about routing + dispatch, and a full WAV
/// decoder is a wide fuzz target. Callers that need mp3 / flac / opus
/// or non-16 kHz should preprocess on the client until a proper
/// decoder path lands (out of scope for M2-09 unless the compat matrix
/// explicitly asks for it).
pub fn decode_pcm_wav(bytes: &[u8]) -> Result<Vec<f32>, HttpTranscriptionError> {
    // Minimum size: RIFF(4) + size(4) + WAVE(4) + fmt (4) + size(4) +
    // fmt body(16) + data(4) + size(4) = 44 bytes.
    if bytes.len() < 44 {
        return Err(HttpTranscriptionError::unsupported_audio(format!(
            "audio too short to be a WAV file ({} bytes)",
            bytes.len()
        )));
    }
    if &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(HttpTranscriptionError::unsupported_audio(
            "expected RIFF/WAVE header (only PCM WAV is supported at v0.5)".to_owned(),
        ));
    }

    // Walk the chunk list. We do not trust the RIFF size (some encoders
    // leave it as `-1`); every read is bound-checked against
    // `bytes.len()`.
    let mut cursor = 12usize;
    let mut fmt_body: Option<&[u8]> = None;
    let mut data: Option<&[u8]> = None;
    while cursor + 8 <= bytes.len() {
        let id = &bytes[cursor..cursor + 4];
        let size = u32::from_le_bytes(bytes[cursor + 4..cursor + 8].try_into().unwrap()) as usize;
        let body_start = cursor + 8;
        let body_end = body_start.checked_add(size).ok_or_else(|| {
            HttpTranscriptionError::unsupported_audio("WAV chunk size overflow".to_owned())
        })?;
        if body_end > bytes.len() {
            return Err(HttpTranscriptionError::unsupported_audio(format!(
                "WAV chunk `{}` runs past end of file",
                std::str::from_utf8(id).unwrap_or("????")
            )));
        }
        match id {
            b"fmt " => fmt_body = Some(&bytes[body_start..body_end]),
            b"data" => data = Some(&bytes[body_start..body_end]),
            _ => {} // ignore LIST / bext / etc.
        }
        // Chunk bodies of odd size have a pad byte for word alignment.
        cursor = body_end + (size & 1);
    }

    let fmt = fmt_body.ok_or_else(|| {
        HttpTranscriptionError::unsupported_audio("WAV file is missing the `fmt ` chunk".to_owned())
    })?;
    let data = data.ok_or_else(|| {
        HttpTranscriptionError::unsupported_audio("WAV file is missing the `data` chunk".to_owned())
    })?;
    if fmt.len() < 16 {
        return Err(HttpTranscriptionError::unsupported_audio(
            "WAV `fmt ` chunk truncated (need at least 16 bytes)".to_owned(),
        ));
    }

    let format_tag = u16::from_le_bytes(fmt[0..2].try_into().unwrap());
    let channels = u16::from_le_bytes(fmt[2..4].try_into().unwrap());
    let sample_rate = u32::from_le_bytes(fmt[4..8].try_into().unwrap());
    let bits_per_sample = u16::from_le_bytes(fmt[14..16].try_into().unwrap());

    if sample_rate != TARGET_SAMPLE_RATE_HZ {
        return Err(HttpTranscriptionError::unsupported_audio(format!(
            "sample rate {sample_rate} Hz not supported; expected {TARGET_SAMPLE_RATE_HZ} Hz \
             (resample on the client at v0.5)"
        )));
    }
    if !(1..=2).contains(&channels) {
        return Err(HttpTranscriptionError::unsupported_audio(format!(
            "channel count {channels} not supported (mono or stereo only)"
        )));
    }

    // Convert each interleaved frame to f32; average to mono.
    let bytes_per_sample = match (format_tag, bits_per_sample) {
        (1, 16) | (1, 32) | (3, 32) => (bits_per_sample / 8) as usize,
        (tag, bps) => {
            return Err(HttpTranscriptionError::unsupported_audio(format!(
                "unsupported WAV encoding: format_tag={tag}, bits_per_sample={bps}"
            )));
        }
    };
    let frame_bytes = bytes_per_sample * channels as usize;
    if frame_bytes == 0 || data.len() % frame_bytes != 0 {
        return Err(HttpTranscriptionError::unsupported_audio(
            "WAV `data` chunk length is not a whole number of frames".to_owned(),
        ));
    }

    let num_frames = data.len() / frame_bytes;
    let mut pcm = Vec::with_capacity(num_frames);
    for frame_idx in 0..num_frames {
        let base = frame_idx * frame_bytes;
        let mut acc = 0.0f32;
        for ch in 0..channels as usize {
            let s = base + ch * bytes_per_sample;
            let sample = match (format_tag, bits_per_sample) {
                (1, 16) => {
                    let v = i16::from_le_bytes(data[s..s + 2].try_into().unwrap());
                    f32::from(v) / f32::from(i16::MAX)
                }
                (1, 32) => {
                    let v = i32::from_le_bytes(data[s..s + 4].try_into().unwrap());
                    (v as f32) / (i32::MAX as f32)
                }
                (3, 32) => f32::from_le_bytes(data[s..s + 4].try_into().unwrap()),
                _ => unreachable!("guarded above"),
            };
            acc += sample;
        }
        pcm.push(acc / channels as f32);
    }
    Ok(pcm)
}

// ---------------------------------------------------------------------------
// HTTP error envelope — OpenAI-shaped `{"error": {...}}`.
// ---------------------------------------------------------------------------

/// HTTP-layer error for the OpenAI transcriptions route.
///
/// Distinct from [`OpenAiTranscribeError`] (the T07 schema-layer error)
/// so request-parsing failures (multipart, WAV decode) do not have to
/// synthesise a fake `ServiceError`. Both types render into the
/// OpenAI-documented `{"error": {"message", "type", "code"}}` envelope
/// via [`IntoResponse`]. When the T05 crate-wide `ServerError` lands,
/// this type will be absorbed by an `impl From<HttpTranscriptionError>
/// for ServerError` shim; the wire shape stays stable.
#[derive(Debug)]
pub struct HttpTranscriptionError {
    status: StatusCode,
    error_type: &'static str,
    message: String,
    /// Optional stable machine code. Distinct from `error_type` so
    /// clients can branch on `code` (e.g. `"model_not_found"`) without
    /// parsing English.
    code: Option<&'static str>,
}

impl HttpTranscriptionError {
    fn bad_multipart(message: String) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            error_type: "invalid_request_error",
            message,
            code: Some("invalid_multipart"),
        }
    }

    fn unsupported_audio(message: String) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            error_type: "invalid_request_error",
            message,
            code: Some("unsupported_audio_format"),
        }
    }

    fn model_not_found(model: &str) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            error_type: "invalid_request_error",
            message: format!("model `{model}` is not registered"),
            code: Some("model_not_found"),
        }
    }

    fn synthesize_unavailable(message: String) -> Self {
        Self {
            status: StatusCode::NOT_IMPLEMENTED,
            error_type: "invalid_request_error",
            message,
            code: Some("not_implemented"),
        }
    }

    fn inference_failed(message: String) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            error_type: "server_error",
            message,
            code: Some("inference_failed"),
        }
    }

    fn invalid_config(message: String) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            error_type: "server_error",
            message,
            code: Some("invalid_config"),
        }
    }

    fn not_implemented(message: String) -> Self {
        Self {
            status: StatusCode::NOT_IMPLEMENTED,
            error_type: "invalid_request_error",
            message,
            code: Some("not_implemented"),
        }
    }

    /// Maps a T07 [`OpenAiTranscribeError`] to its HTTP form.
    ///
    /// Preserves the failure kind so a Metal / CUDA `UnsupportedOp`
    /// from the engine layer is NEVER silently rewritten to a 200
    /// (FR-EX-08). The T05 error mapper will subsume this once it
    /// lands; T06 keeps the mapping local so the tickets stay
    /// independent.
    pub fn from_schema(err: OpenAiTranscribeError) -> Self {
        match err {
            OpenAiTranscribeError::UnsupportedResponseFormat(s) => Self::bad_multipart(format!(
                "unsupported `response_format`: `{s}` (accepted: json, text, verbose_json, srt, vtt)"
            )),
            OpenAiTranscribeError::VerboseJsonNotImplemented
            | OpenAiTranscribeError::SrtNotImplemented
            | OpenAiTranscribeError::VttNotImplemented => Self::not_implemented(err.to_string()),
            OpenAiTranscribeError::Service(inner) => Self::from_service(inner),
        }
    }

    /// Maps a [`ServiceError`] to its HTTP form. Kept `pub` so sibling
    /// API modules can share the mapping when T05 fully consolidates
    /// the error surface.
    pub fn from_service(err: ServiceError) -> Self {
        match err {
            ServiceError::UnknownModel(m) => Self::model_not_found(&m),
            ServiceError::SynthesizeUnavailable { model, reason } => {
                Self::synthesize_unavailable(format!(
                    "model `{model}` is registered but its synthesize path is unavailable: {reason}"
                ))
            }
            ServiceError::ModelLoadFailed { slot, path, source } => Self::inference_failed(
                format!("model `{slot}` at {path:?} failed to load: {source}"),
            ),
            ServiceError::InvalidConfig(msg) => Self::invalid_config(msg),
            ServiceError::Inference(inner) => Self::inference_failed(inner.to_string()),
        }
    }
}

impl std::fmt::Display for HttpTranscriptionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} ({})", self.message, self.status)
    }
}

impl std::error::Error for HttpTranscriptionError {}

impl IntoResponse for HttpTranscriptionError {
    fn into_response(self) -> Response {
        // OpenAI envelope: `{"error": {"message", "type", "code"?}}`.
        // Explicit `serde_json::json!` (not derive-Serialize) so the
        // exact wire shape is visible in one place and easy to align
        // with T05 when it consolidates the crate-wide error surface.
        let body = serde_json::json!({
            "error": {
                "message": self.message,
                "type": self.error_type,
                "code": self.code,
            }
        });
        (self.status, axum::Json(body)).into_response()
    }
}

// ---------------------------------------------------------------------------
// Tests — cargo test --test _ or `cargo test openai::response_schema`
// ---------------------------------------------------------------------------

#[cfg(test)]
mod response_schema {
    //! T07 acceptance tests: response_format parsing, DTO shape, and
    //! `transcribe_to_response` dispatch through a mocked
    //! `TranscribeService`.
    //!
    //! These tests exercise the schema surface **without** a live
    //! Whisper engine — the mock proves the wiring shape, and T08
    //! integration tests exercise the real `InferenceService` path
    //! end-to-end.

    use super::*;
    use crate::service::{ServiceError, model_names};
    use vokra_core::VokraError;

    // -------- Mock TranscribeService --------

    /// Deterministic mock: echoes the model name so tests can assert
    /// the dispatch table without loading a Whisper GGUF.
    struct EchoTranscribe;

    impl TranscribeService for EchoTranscribe {
        fn transcribe(&self, model: &str, pcm: &[f32]) -> Result<String, ServiceError> {
            match model {
                model_names::WHISPER_1 | model_names::WHISPER_BASE => {
                    Ok(format!("[{model}] {} samples", pcm.len()))
                }
                model_names::WHISPER_LARGE_V3 => Ok(format!("[large-v3] {} samples", pcm.len())),
                other => Err(ServiceError::UnknownModel(other.to_owned())),
            }
        }
    }

    /// Mock that surfaces an engine-side unsupported-op — used to
    /// verify FR-EX-08: the schema layer does NOT retry, fall back to
    /// CPU, or convert to a silent success.
    struct UnsupportedTranscribe;

    impl TranscribeService for UnsupportedTranscribe {
        fn transcribe(&self, _model: &str, _pcm: &[f32]) -> Result<String, ServiceError> {
            Err(ServiceError::Inference(VokraError::UnsupportedOp(
                "stft on Metal (M2-01 hole)".into(),
            )))
        }
    }

    // -------- ResponseFormat::parse --------

    #[test]
    fn response_format_default_when_absent() {
        assert_eq!(ResponseFormat::parse(None).unwrap(), ResponseFormat::Json);
    }

    #[test]
    fn response_format_default_when_empty_or_whitespace() {
        assert_eq!(
            ResponseFormat::parse(Some("")).unwrap(),
            ResponseFormat::Json
        );
        assert_eq!(
            ResponseFormat::parse(Some("   ")).unwrap(),
            ResponseFormat::Json
        );
    }

    #[test]
    fn response_format_case_insensitive() {
        for s in ["json", "JSON", "Json", "jSoN"] {
            assert_eq!(
                ResponseFormat::parse(Some(s)).unwrap(),
                ResponseFormat::Json
            );
        }
        for s in ["text", "TEXT", "Text"] {
            assert_eq!(
                ResponseFormat::parse(Some(s)).unwrap(),
                ResponseFormat::Text
            );
        }
        for s in ["verbose_json", "VERBOSE_JSON", "Verbose_Json"] {
            assert_eq!(
                ResponseFormat::parse(Some(s)).unwrap(),
                ResponseFormat::VerboseJson
            );
        }
        assert_eq!(
            ResponseFormat::parse(Some("srt")).unwrap(),
            ResponseFormat::Srt
        );
        assert_eq!(
            ResponseFormat::parse(Some("vtt")).unwrap(),
            ResponseFormat::Vtt
        );
    }

    #[test]
    fn response_format_unknown_returns_raw() {
        let err = ResponseFormat::parse(Some("yaml")).unwrap_err();
        assert_eq!(err, "yaml");
        // Preserves original casing so error messages can echo it.
        let err = ResponseFormat::parse(Some("PROTOBUF")).unwrap_err();
        assert_eq!(err, "PROTOBUF");
    }

    #[test]
    fn requires_timestamps_flags_deferred_variants() {
        assert!(!ResponseFormat::Json.requires_timestamps());
        assert!(!ResponseFormat::Text.requires_timestamps());
        assert!(ResponseFormat::VerboseJson.requires_timestamps());
        assert!(ResponseFormat::Srt.requires_timestamps());
        assert!(ResponseFormat::Vtt.requires_timestamps());
    }

    // -------- TranscriptionResponse serde shape --------

    #[test]
    fn default_json_body_is_text_only() {
        // faster-whisper drop-in: {"text": "..."} and nothing else.
        // Adding fields here is a compat break, so we assert on the
        // exact serialised shape.
        let body = TranscriptionResponse::new("hello world");
        let json = serde_json::to_string(&body).unwrap();
        assert_eq!(json, r#"{"text":"hello world"}"#);
    }

    #[test]
    fn default_json_body_roundtrips() {
        let body = TranscriptionResponse::new("こんにちは");
        let json = serde_json::to_string(&body).unwrap();
        let parsed: TranscriptionResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.text, "こんにちは");
    }

    #[test]
    fn empty_transcription_serialises_as_empty_string_not_null() {
        // Empty audio yields "", never null — client SDKs pattern-
        // match `.get("text")` and choke on null.
        let body = TranscriptionResponse::new("");
        let json = serde_json::to_string(&body).unwrap();
        assert_eq!(json, r#"{"text":""}"#);
    }

    // -------- transcribe_to_response dispatch --------

    #[test]
    fn json_default_wraps_service_output() {
        let pcm = vec![0.0f32; 16_000]; // 1s at 16 kHz
        let outcome = transcribe_to_response(&EchoTranscribe, model_names::WHISPER_1, None, &pcm)
            .expect("must succeed");
        match outcome {
            TranscriptionOutcome::Json(r) => {
                assert_eq!(r.text, "[whisper-1] 16000 samples");
                // Also check the on-wire shape.
                let json = serde_json::to_string(&r).unwrap();
                assert_eq!(json, r#"{"text":"[whisper-1] 16000 samples"}"#);
            }
            other => panic!("expected Json outcome, got {other:?}"),
        }
    }

    #[test]
    fn text_format_returns_plain_string() {
        let pcm = vec![0.0f32; 32_000];
        let outcome = transcribe_to_response(
            &EchoTranscribe,
            model_names::WHISPER_BASE,
            Some("text"),
            &pcm,
        )
        .expect("must succeed");
        match outcome {
            TranscriptionOutcome::Text(s) => {
                assert_eq!(s, "[whisper-base] 32000 samples");
            }
            other => panic!("expected Text outcome, got {other:?}"),
        }
    }

    #[test]
    fn json_alias_matches_default() {
        let pcm = vec![0.0f32; 4];
        let a =
            transcribe_to_response(&EchoTranscribe, model_names::WHISPER_1, None, &pcm).unwrap();
        let b = transcribe_to_response(&EchoTranscribe, model_names::WHISPER_1, Some("json"), &pcm)
            .unwrap();
        // Both wrap in JSON with the same text.
        match (a, b) {
            (TranscriptionOutcome::Json(a), TranscriptionOutcome::Json(b)) => {
                assert_eq!(a.text, b.text);
            }
            _ => panic!("both must be Json"),
        }
    }

    #[test]
    fn verbose_json_returns_501_with_documented_note() {
        let pcm = vec![0.0f32; 4];
        let err = transcribe_to_response(
            &EchoTranscribe,
            model_names::WHISPER_1,
            Some("verbose_json"),
            &pcm,
        )
        .expect_err("must reject");
        match err {
            OpenAiTranscribeError::VerboseJsonNotImplemented => {
                // The stable note text must be present in Display.
                let msg = err.to_string();
                assert!(msg.contains("verbose_json"));
                assert!(msg.contains("deferred to v1.0+"));
                // And exposes the exact byte-stable constant for T23
                // docs to reference and T22 CI to lock.
                assert!(
                    OpenAiTranscribeError::V0_5_TIMESTAMP_DEFERRAL_NOTE
                        .contains("deferred to v1.0+")
                );
                assert!(
                    OpenAiTranscribeError::V0_5_TIMESTAMP_DEFERRAL_NOTE
                        .contains("response_format=\"json\" or \"text\"")
                );
            }
            other => panic!("expected VerboseJsonNotImplemented, got {other}"),
        }
    }

    #[test]
    fn srt_and_vtt_return_501_with_documented_note() {
        let pcm = vec![0.0f32; 4];
        for (fmt_str, want) in [("srt", "SrtNotImplemented"), ("vtt", "VttNotImplemented")] {
            let err = transcribe_to_response(
                &EchoTranscribe,
                model_names::WHISPER_1,
                Some(fmt_str),
                &pcm,
            )
            .expect_err("must reject");
            let msg = err.to_string();
            assert!(msg.contains(fmt_str), "want {fmt_str} in {msg}, tag={want}");
            assert!(msg.contains("deferred to v1.0+"));
        }
    }

    #[test]
    fn unknown_response_format_is_400_with_raw_value() {
        let pcm = vec![0.0f32; 4];
        let err =
            transcribe_to_response(&EchoTranscribe, model_names::WHISPER_1, Some("yaml"), &pcm)
                .expect_err("must reject");
        match err {
            OpenAiTranscribeError::UnsupportedResponseFormat(s) => assert_eq!(s, "yaml"),
            other => panic!("expected UnsupportedResponseFormat, got {other}"),
        }
    }

    #[test]
    fn unknown_model_propagates_as_service_error_not_silent_success() {
        // FR-EX-08: unknown model must NOT be silently rerouted to
        // base; the schema layer forwards the ServiceError verbatim so
        // the HTTP mapper can render 404.
        let pcm = vec![0.0f32; 4];
        let err = transcribe_to_response(&EchoTranscribe, "gpt-4-audio", None, &pcm)
            .expect_err("must reject");
        match err {
            OpenAiTranscribeError::Service(ServiceError::UnknownModel(m)) => {
                assert_eq!(m, "gpt-4-audio");
            }
            other => panic!("expected Service(UnknownModel), got {other}"),
        }
    }

    #[test]
    fn engine_unsupported_op_propagates_not_silent_cpu_fallback() {
        // FR-EX-08 spirit: an engine-side UnsupportedOp on the caller's
        // requested backend must surface up, never a silent CPU
        // fallback that pretends the request succeeded on the wrong
        // path.
        let pcm = vec![0.0f32; 4];
        let err =
            transcribe_to_response(&UnsupportedTranscribe, model_names::WHISPER_1, None, &pcm)
                .expect_err("must reject");
        match err {
            OpenAiTranscribeError::Service(ServiceError::Inference(VokraError::UnsupportedOp(
                msg,
            ))) => {
                assert!(msg.contains("stft on Metal"));
            }
            other => panic!("expected Inference(UnsupportedOp), got {other}"),
        }
        // And the Error::source chain reaches the ServiceError so the
        // T05 log layer can walk the chain.
        let err =
            transcribe_to_response(&UnsupportedTranscribe, model_names::WHISPER_1, None, &pcm)
                .expect_err("must reject");
        assert!(std::error::Error::source(&err).is_some());
    }

    #[test]
    fn deferred_formats_do_not_invoke_inference() {
        // Sanity: the deferred variants short-circuit BEFORE hitting
        // the service — we must not waste an engine call producing
        // text we can't shape (also matters for logging: no
        // half-transcribed audio in the trace on 501).
        struct PanicOnCall;
        impl TranscribeService for PanicOnCall {
            fn transcribe(&self, _model: &str, _pcm: &[f32]) -> Result<String, ServiceError> {
                panic!("must not be called for deferred response_format");
            }
        }
        let pcm = vec![0.0f32; 4];
        for fmt in ["verbose_json", "srt", "vtt"] {
            let _ = transcribe_to_response(&PanicOnCall, model_names::WHISPER_1, Some(fmt), &pcm)
                .expect_err("must reject before engine call");
        }
    }

    #[test]
    fn transcription_outcome_carries_faster_whisper_compat_bytes() {
        // The T06 axum handler will serialize `Json(TranscriptionResponse)`
        // straight into the body. Assert the byte-exact shape here so
        // T08 doesn't have to reproduce the JSON string.
        let pcm = vec![0.0f32; 8_000];
        let outcome =
            transcribe_to_response(&EchoTranscribe, model_names::WHISPER_1, Some("json"), &pcm)
                .unwrap();
        let TranscriptionOutcome::Json(body) = outcome else {
            panic!("expected Json");
        };
        let bytes = serde_json::to_vec(&body).unwrap();
        let expected = br#"{"text":"[whisper-1] 8000 samples"}"#;
        assert_eq!(bytes, expected);
    }
}

#[cfg(test)]
mod route_transcriptions {
    //! T06 acceptance tests: HTTP routing + `multipart/form-data`
    //! extraction + dispatch through a mocked
    //! [`TranscribeService`](crate::service::TranscribeService).
    //!
    //! Runs the full axum stack via `tower::ServiceExt::oneshot` — no
    //! TCP bind — so tests are hermetic (no `127.0.0.1:0` handshake,
    //! no port allocation).
    //!
    //! T08 will re-run these paths end-to-end with a real per-PR
    //! Whisper base GGUF; T06 stays independent by injecting a fake
    //! service, which is why the model-name mapping
    //! (`whisper-1` → base, `whisper-large-v3` → large-v3) is verified
    //! in the T04 `service::registry` tests, not here.

    use super::*;
    use crate::service::{ServiceError, TranscribeService, model_names};
    use axum::body::Body;
    use axum::extract::FromRef;
    use axum::http::{Request, StatusCode};
    use std::sync::{Arc, Mutex};
    use tower::ServiceExt; // brings `oneshot` into scope

    // -----------------------------------------------------------------
    // Fake TranscribeService — records dispatches and returns canned
    // text. Same shape T08 mocks will use.
    // -----------------------------------------------------------------

    #[derive(Default)]
    struct FakeTranscribe {
        calls: Mutex<Vec<(String, usize)>>,
        known: Vec<&'static str>,
        response: String,
    }

    impl FakeTranscribe {
        fn new(response: &str, known: Vec<&'static str>) -> Arc<Self> {
            Arc::new(Self {
                calls: Mutex::new(Vec::new()),
                known,
                response: response.to_owned(),
            })
        }

        fn calls(&self) -> Vec<(String, usize)> {
            self.calls.lock().unwrap().clone()
        }
    }

    impl TranscribeService for FakeTranscribe {
        fn transcribe(&self, model: &str, pcm: &[f32]) -> Result<String, ServiceError> {
            self.calls
                .lock()
                .unwrap()
                .push((model.to_owned(), pcm.len()));
            if !self.known.contains(&model) {
                return Err(ServiceError::UnknownModel(model.to_owned()));
            }
            Ok(self.response.clone())
        }
    }

    /// Shared app state exposing the trait object as
    /// `Arc<dyn TranscribeService>` — the shape [`attach_routes`]
    /// requires via `FromRef`. Mirrors what the real server holds
    /// (`Arc<InferenceService>` behind the same trait).
    #[derive(Clone)]
    struct AppState {
        transcribe: Arc<dyn TranscribeService>,
    }

    impl FromRef<AppState> for Arc<dyn TranscribeService> {
        fn from_ref(app: &AppState) -> Self {
            app.transcribe.clone()
        }
    }

    fn build_app(svc: Arc<dyn TranscribeService>) -> Router {
        let state = AppState { transcribe: svc };
        attach_routes(Router::new()).with_state(state)
    }

    // -----------------------------------------------------------------
    // Minimal WAV writer: emits a valid 16-bit PCM mono @ 16 kHz file
    // so the handler's `decode_pcm_wav` step has real bytes to chew on.
    // -----------------------------------------------------------------

    fn make_wav(num_samples: usize) -> Vec<u8> {
        let sample_rate: u32 = TARGET_SAMPLE_RATE_HZ;
        let channels: u16 = 1;
        let bits_per_sample: u16 = 16;
        let byte_rate: u32 = sample_rate * u32::from(channels) * u32::from(bits_per_sample) / 8;
        let block_align: u16 = channels * bits_per_sample / 8;
        let data_bytes: u32 = (num_samples as u32) * u32::from(block_align);
        let mut buf = Vec::with_capacity(44 + data_bytes as usize);
        buf.extend_from_slice(b"RIFF");
        buf.extend_from_slice(&(36 + data_bytes).to_le_bytes());
        buf.extend_from_slice(b"WAVE");
        buf.extend_from_slice(b"fmt ");
        buf.extend_from_slice(&16u32.to_le_bytes());
        buf.extend_from_slice(&1u16.to_le_bytes()); // PCM
        buf.extend_from_slice(&channels.to_le_bytes());
        buf.extend_from_slice(&sample_rate.to_le_bytes());
        buf.extend_from_slice(&byte_rate.to_le_bytes());
        buf.extend_from_slice(&block_align.to_le_bytes());
        buf.extend_from_slice(&bits_per_sample.to_le_bytes());
        buf.extend_from_slice(b"data");
        buf.extend_from_slice(&data_bytes.to_le_bytes());
        for i in 0..num_samples {
            // Deterministic sawtooth so `decode_pcm_wav` has real work.
            let s = ((i as i32 % 32) - 16) as i16 * 512;
            buf.extend_from_slice(&s.to_le_bytes());
        }
        buf
    }

    // -----------------------------------------------------------------
    // Manual multipart body builder — kept in-crate so the test does
    // not add a fresh dependency for one string, and so the test tree
    // itself can be audited for RFC-7578 compliance.
    // -----------------------------------------------------------------

    fn build_multipart(parts: Vec<(&str, &[u8], Option<&str>)>, boundary: &str) -> Vec<u8> {
        let mut body = Vec::new();
        for (name, bytes, filename) in parts {
            body.extend_from_slice(b"--");
            body.extend_from_slice(boundary.as_bytes());
            body.extend_from_slice(b"\r\n");
            body.extend_from_slice(b"Content-Disposition: form-data; name=\"");
            body.extend_from_slice(name.as_bytes());
            body.extend_from_slice(b"\"");
            if let Some(fname) = filename {
                body.extend_from_slice(b"; filename=\"");
                body.extend_from_slice(fname.as_bytes());
                body.extend_from_slice(b"\"");
                body.extend_from_slice(b"\r\nContent-Type: audio/wav");
            }
            body.extend_from_slice(b"\r\n\r\n");
            body.extend_from_slice(bytes);
            body.extend_from_slice(b"\r\n");
        }
        body.extend_from_slice(b"--");
        body.extend_from_slice(boundary.as_bytes());
        body.extend_from_slice(b"--\r\n");
        body
    }

    async fn read_body(response: Response) -> Vec<u8> {
        let body = response.into_body();
        axum::body::to_bytes(body, usize::MAX)
            .await
            .unwrap()
            .to_vec()
    }

    fn post_request(boundary: &str, body: Vec<u8>) -> Request<Body> {
        Request::builder()
            .method("POST")
            .uri("/v1/audio/transcriptions")
            .header(
                "content-type",
                format!("multipart/form-data; boundary={boundary}"),
            )
            .body(Body::from(body))
            .unwrap()
    }

    // -----------------------------------------------------------------
    // The T06 acceptance test suite.
    // -----------------------------------------------------------------

    /// End-to-end happy path: multipart WAV + `model=whisper-1` reaches
    /// the injected service and returns the OpenAI JSON envelope. The
    /// registry maps `whisper-1` → base in production; here the fake
    /// receives the alias verbatim (dispatch-shape test).
    #[tokio::test]
    async fn route_transcriptions_dispatches_whisper1_to_service() {
        let fake = FakeTranscribe::new(
            "hello world",
            vec![model_names::WHISPER_1, model_names::WHISPER_BASE],
        );
        let app = build_app(fake.clone() as Arc<dyn TranscribeService>);

        let wav = make_wav(3200); // 200 ms @ 16 kHz.
        let boundary = "----vokra-test-boundary";
        let body = build_multipart(
            vec![
                ("file", &wav, Some("hello.wav")),
                ("model", b"whisper-1", None),
                ("response_format", b"json", None),
            ],
            boundary,
        );

        let response = app.oneshot(post_request(boundary, body)).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let ct = response
            .headers()
            .get("content-type")
            .and_then(|h| h.to_str().ok())
            .unwrap_or_default()
            .to_owned();
        let bytes = read_body(response).await;
        assert!(
            ct.starts_with("application/json"),
            "expected JSON content-type, got {ct:?}"
        );
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["text"], "hello world");

        // Handler forwarded the alias verbatim and passed a non-empty
        // decoded PCM buffer.
        let calls = fake.calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].0, "whisper-1");
        assert_eq!(calls[0].1, 3200);
    }

    /// `whisper-large-v3` is forwarded verbatim so the registry can do
    /// the (base vs large-v3) alias resolution; T06 only proves the
    /// alias reaches the service.
    #[tokio::test]
    async fn route_transcriptions_dispatches_large_v3_to_service() {
        let fake = FakeTranscribe::new("large text", vec![model_names::WHISPER_LARGE_V3]);
        let app = build_app(fake.clone() as Arc<dyn TranscribeService>);

        let wav = make_wav(800);
        let boundary = "----vokra-test-boundary";
        let body = build_multipart(
            vec![
                ("file", &wav, Some("x.wav")),
                ("model", b"whisper-large-v3", None),
            ],
            boundary,
        );

        let response = app.oneshot(post_request(boundary, body)).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let calls = fake.calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].0, "whisper-large-v3");
    }

    /// Absent `model` defaults to `whisper-1` (OpenAI's catalogue
    /// alias), matching the client SDKs' behaviour.
    #[tokio::test]
    async fn route_transcriptions_defaults_model_to_whisper_1() {
        let fake = FakeTranscribe::new(
            "ok",
            vec![model_names::WHISPER_1, model_names::WHISPER_BASE],
        );
        let app = build_app(fake.clone() as Arc<dyn TranscribeService>);

        let wav = make_wav(160);
        let boundary = "----vokra-test-boundary";
        let body = build_multipart(vec![("file", &wav, Some("hi.wav"))], boundary);

        let response = app.oneshot(post_request(boundary, body)).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(fake.calls()[0].0, "whisper-1");
    }

    /// `response_format=text` returns `text/plain` with the raw
    /// transcription — the OpenAI-documented shape for the plain-text
    /// variant.
    #[tokio::test]
    async fn route_transcriptions_response_format_text() {
        let fake = FakeTranscribe::new(
            "こんにちは",
            vec![model_names::WHISPER_1, model_names::WHISPER_BASE],
        );
        let app = build_app(fake as Arc<dyn TranscribeService>);

        let wav = make_wav(1600);
        let boundary = "----vokra-test-boundary";
        let body = build_multipart(
            vec![
                ("file", &wav, Some("hi.wav")),
                ("response_format", b"text", None),
            ],
            boundary,
        );

        let response = app.oneshot(post_request(boundary, body)).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let ct = response
            .headers()
            .get("content-type")
            .and_then(|h| h.to_str().ok())
            .unwrap_or_default()
            .to_owned();
        assert!(
            ct.starts_with("text/plain"),
            "expected text/plain, got {ct:?}"
        );
        let bytes = read_body(response).await;
        assert_eq!(std::str::from_utf8(&bytes).unwrap(), "こんにちは");
    }

    /// Unknown model in the registry produces `404 model_not_found` —
    /// never a silent fallback to base (FR-EX-08 on the request surface).
    #[tokio::test]
    async fn route_transcriptions_unknown_model_is_404() {
        let fake = FakeTranscribe::new(
            "irrelevant",
            vec![model_names::WHISPER_1, model_names::WHISPER_BASE],
        );
        let app = build_app(fake as Arc<dyn TranscribeService>);

        let wav = make_wav(800);
        let boundary = "----vokra-test-boundary";
        let body = build_multipart(
            vec![
                ("file", &wav, Some("x.wav")),
                // whisper-large-v3 is not in the fake's known list.
                ("model", b"whisper-large-v3", None),
            ],
            boundary,
        );

        let response = app.oneshot(post_request(boundary, body)).await.unwrap();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
        let bytes = read_body(response).await;
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["error"]["code"], "model_not_found");
        assert_eq!(json["error"]["type"], "invalid_request_error");
    }

    /// Missing `file` field is `400 invalid_multipart`. Fake with an
    /// empty `known` list proves the handler never dispatches.
    #[tokio::test]
    async fn route_transcriptions_missing_file_is_400() {
        let fake = FakeTranscribe::new("unused", vec![]);
        let app = build_app(fake.clone() as Arc<dyn TranscribeService>);

        let boundary = "----vokra-test-boundary";
        let body = build_multipart(vec![("model", b"whisper-1", None)], boundary);

        let response = app.oneshot(post_request(boundary, body)).await.unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let bytes = read_body(response).await;
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["error"]["code"], "invalid_multipart");
        assert!(fake.calls().is_empty(), "handler must not dispatch");
    }

    /// Non-WAV bytes decode to `400 unsupported_audio_format` — never
    /// a silent success. Closes the "every byte string maps to an
    /// empty PCM buffer" failure mode.
    #[tokio::test]
    async fn route_transcriptions_non_wav_is_400() {
        let fake = FakeTranscribe::new(
            "unused",
            vec![model_names::WHISPER_1, model_names::WHISPER_BASE],
        );
        let app = build_app(fake.clone() as Arc<dyn TranscribeService>);

        let boundary = "----vokra-test-boundary";
        let body = build_multipart(
            vec![("file", b"not a wav file at all", Some("x.mp3"))],
            boundary,
        );

        let response = app.oneshot(post_request(boundary, body)).await.unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let bytes = read_body(response).await;
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["error"]["code"], "unsupported_audio_format");
        assert!(fake.calls().is_empty(), "handler must not dispatch");
    }

    /// `temperature` outside `[0.0, 1.0]` is `400 invalid_multipart`.
    /// Also proves `.` parses regardless of locale (NFR-RL-01).
    #[tokio::test]
    async fn route_transcriptions_bad_temperature_is_400() {
        let fake = FakeTranscribe::new(
            "unused",
            vec![model_names::WHISPER_1, model_names::WHISPER_BASE],
        );
        let app = build_app(fake as Arc<dyn TranscribeService>);

        let wav = make_wav(800);
        let boundary = "----vokra-test-boundary";
        let body = build_multipart(
            vec![("file", &wav, Some("x.wav")), ("temperature", b"2.5", None)],
            boundary,
        );

        let response = app.oneshot(post_request(boundary, body)).await.unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    /// Unknown multipart field is rejected (fails closed). Prevents a
    /// typo like `respose_format` from silently defaulting.
    #[tokio::test]
    async fn route_transcriptions_unknown_field_is_400() {
        let fake = FakeTranscribe::new(
            "unused",
            vec![model_names::WHISPER_1, model_names::WHISPER_BASE],
        );
        let app = build_app(fake as Arc<dyn TranscribeService>);

        let wav = make_wav(800);
        let boundary = "----vokra-test-boundary";
        let body = build_multipart(
            vec![
                ("file", &wav, Some("x.wav")),
                // Deliberate typo — must NOT silently become the default.
                ("respose_format", b"json", None),
            ],
            boundary,
        );

        let response = app.oneshot(post_request(boundary, body)).await.unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    /// All documented OpenAI fields are accepted (T06 accepts + passes
    /// through `language` / `temperature` / `prompt` — forwarding into
    /// the Whisper decoder options is a T07-followup).
    #[tokio::test]
    async fn route_transcriptions_accepts_all_documented_fields() {
        let fake = FakeTranscribe::new(
            "全部",
            vec![model_names::WHISPER_1, model_names::WHISPER_BASE],
        );
        let app = build_app(fake.clone() as Arc<dyn TranscribeService>);

        let wav = make_wav(1600);
        let boundary = "----vokra-test-boundary";
        let body = build_multipart(
            vec![
                ("file", &wav, Some("x.wav")),
                ("model", b"whisper-1", None),
                ("language", b"ja", None),
                ("response_format", b"json", None),
                ("temperature", b"0.3", None),
                ("prompt", b"technical speech", None),
            ],
            boundary,
        );

        let response = app.oneshot(post_request(boundary, body)).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = read_body(response).await;
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["text"], "全部");
    }

    /// `verbose_json` short-circuits at the T07 schema layer with 501
    /// and never invokes the fake service.
    #[tokio::test]
    async fn route_transcriptions_verbose_json_is_501() {
        let fake = FakeTranscribe::new(
            "unused",
            vec![model_names::WHISPER_1, model_names::WHISPER_BASE],
        );
        let app = build_app(fake.clone() as Arc<dyn TranscribeService>);

        let wav = make_wav(800);
        let boundary = "----vokra-test-boundary";
        let body = build_multipart(
            vec![
                ("file", &wav, Some("x.wav")),
                ("response_format", b"verbose_json", None),
            ],
            boundary,
        );

        let response = app.oneshot(post_request(boundary, body)).await.unwrap();
        assert_eq!(response.status(), StatusCode::NOT_IMPLEMENTED);
        let bytes = read_body(response).await;
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["error"]["code"], "not_implemented");
        // The T07 stable deferral note is preserved end-to-end.
        assert!(
            json["error"]["message"]
                .as_str()
                .unwrap_or_default()
                .contains("deferred to v1.0+"),
            "message: {}",
            json["error"]["message"]
        );
        assert!(fake.calls().is_empty(), "handler must not dispatch");
    }

    /// Unsupported `response_format` (e.g. `yaml`) is `400
    /// invalid_multipart` — the T07 schema layer's
    /// `UnsupportedResponseFormat` mapped by
    /// [`HttpTranscriptionError::from_schema`].
    #[tokio::test]
    async fn route_transcriptions_unknown_response_format_is_400() {
        let fake = FakeTranscribe::new(
            "unused",
            vec![model_names::WHISPER_1, model_names::WHISPER_BASE],
        );
        let app = build_app(fake.clone() as Arc<dyn TranscribeService>);

        let wav = make_wav(800);
        let boundary = "----vokra-test-boundary";
        let body = build_multipart(
            vec![
                ("file", &wav, Some("x.wav")),
                ("response_format", b"yaml", None),
            ],
            boundary,
        );

        let response = app.oneshot(post_request(boundary, body)).await.unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let bytes = read_body(response).await;
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["error"]["code"], "invalid_multipart");
        assert!(fake.calls().is_empty());
    }

    /// Direct WAV decoder branches the routing layer cannot easily
    /// reach.
    #[test]
    fn route_transcriptions_wav_wrong_sample_rate_rejected() {
        let mut wav = make_wav(160);
        // Overwrite the sample-rate field at offset 24..28 with 44_100.
        wav[24..28].copy_from_slice(&44_100u32.to_le_bytes());
        let err = decode_pcm_wav(&wav).unwrap_err();
        assert_eq!(err.status, StatusCode::BAD_REQUEST);
        assert_eq!(err.code, Some("unsupported_audio_format"));
    }

    #[test]
    fn route_transcriptions_wav_stereo_averaged_to_mono() {
        // Minimal stereo PCM16 WAV: 4 frames of `[1000, -1000]` → mono 0.
        let sample_rate: u32 = TARGET_SAMPLE_RATE_HZ;
        let channels: u16 = 2;
        let bits: u16 = 16;
        let block_align: u16 = channels * bits / 8;
        let byte_rate: u32 = sample_rate * u32::from(block_align);
        let num_samples = 4usize;
        let data_bytes: u32 = num_samples as u32 * u32::from(block_align);
        let mut buf = Vec::new();
        buf.extend_from_slice(b"RIFF");
        buf.extend_from_slice(&(36 + data_bytes).to_le_bytes());
        buf.extend_from_slice(b"WAVE");
        buf.extend_from_slice(b"fmt ");
        buf.extend_from_slice(&16u32.to_le_bytes());
        buf.extend_from_slice(&1u16.to_le_bytes());
        buf.extend_from_slice(&channels.to_le_bytes());
        buf.extend_from_slice(&sample_rate.to_le_bytes());
        buf.extend_from_slice(&byte_rate.to_le_bytes());
        buf.extend_from_slice(&block_align.to_le_bytes());
        buf.extend_from_slice(&bits.to_le_bytes());
        buf.extend_from_slice(b"data");
        buf.extend_from_slice(&data_bytes.to_le_bytes());
        for _ in 0..num_samples {
            buf.extend_from_slice(&1000i16.to_le_bytes());
            buf.extend_from_slice(&(-1000i16).to_le_bytes());
        }
        let pcm = decode_pcm_wav(&buf).unwrap();
        assert_eq!(pcm.len(), num_samples);
        for v in pcm {
            assert!(v.abs() < 1e-6, "stereo average should be ~0, got {v}");
        }
    }
}
