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
//! * `"verbose_json"` + `timestamp_granularities[]=word` → **200** with
//!   the OpenAI verbose envelope carrying a `words[]` array (cc-19,
//!   2026-07-19 M4-residual audit: the historical 501's deferral premise
//!   — "core support absent" — was satisfied by `da13bfd` word-timing
//!   accuracy + `eb41648` exact alignment-heads emission, so the server
//!   surface is now wired through
//!   [`TranscribeService::transcribe_beam`]). Only fields the runtime
//!   actually computed are emitted (`task` / `duration` / `text` /
//!   `words`) — no fabricated `language` / `segments` keys.
//! * `"verbose_json"` without the word granularity (OpenAI defaults to
//!   segment), `timestamp_granularities[]=segment`, `"srt"`, `"vtt"` →
//!   **501 Not Implemented** with a stable, documented note:
//!   segment-level timestamps remain unimplemented. Returning 501 (not
//!   silent stubs, not 500) is the FR-EX-08 "no silent fallback" rule
//!   applied to a schema hole.
//! * `beam_size` (Vokra extension, bounded at [`MAX_BEAM_SIZE`]) routes
//!   the decode through beam search; unset/0/1 keeps the legacy greedy
//!   path byte-for-byte.
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

use crate::service::{ServiceError, TranscribeBeamRequest, TranscribeService};
use vokra_core::VokraError;

/// Upper bound on the `beam_size` multipart field (cc-19). Beam search cost
/// scales linearly in width per step (plus n-best detokenization); 8 covers
/// every documented Whisper/faster-whisper preset (`beam_size=5` is the
/// upstream default) while keeping a hostile client from requesting an
/// arbitrarily wide decode. Larger values are an explicit 400.
pub const MAX_BEAM_SIZE: usize = 8;

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
    /// Verbose JSON. **Word-level** timestamps are implemented (cc-19):
    /// `timestamp_granularities[]=word` returns the OpenAI verbose
    /// envelope with a `words[]` array. **Segment-level** timestamps
    /// (the OpenAI default when no granularity is sent, and the
    /// explicit `segment` granularity) remain a 501.
    VerboseJson,
    /// SRT subtitle body. **Not implemented** — depends on per-segment
    /// timestamps, which remain deferred.
    Srt,
    /// WebVTT subtitle body. **Not implemented** — same deferral as
    /// `Srt`.
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

/// One word-level timestamp entry of the verbose_json `words[]` array —
/// the OpenAI-documented shape `{"word", "start", "end"}` (seconds as
/// floats). Mapped 1:1 from
/// [`crate::service::WordTimestamp`] (M4-20 cross-attention alignment).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WordEntry {
    /// Detokenized word text (spaces as the tokenizer produced them).
    pub word: String,
    /// Word start time in seconds.
    pub start: f64,
    /// Word end time in seconds.
    pub end: f64,
}

/// `response_format=verbose_json` + `timestamp_granularities[]=word`
/// response body (cc-19). Deliberately carries ONLY fields the runtime
/// actually computed:
///
/// * `task` — constant `"transcribe"` (this route never translates);
/// * `duration` — real decoded-audio length in seconds
///   (`pcm.len() / 16 kHz`), not a fabricated estimate;
/// * `text` — the top-1 transcription;
/// * `words` — the aligned word spans.
///
/// The OpenAI reference response also carries `language` (detected) and
/// `segments[]`; emitting a guessed language or an empty-but-present
/// `segments` array would fabricate data the runtime did not produce
/// (NFR-RL-06), so those keys are ABSENT — clients that need them get an
/// explicit 501 through the granularity gate instead of silent nulls.
#[derive(Debug, Clone, Serialize)]
pub struct VerboseTranscriptionResponse {
    /// Always `"transcribe"`.
    pub task: &'static str,
    /// Decoded audio duration in seconds (real, from the PCM length).
    pub duration: f64,
    /// Top-1 transcription (identical to the `json` format's `text`).
    pub text: String,
    /// Word-level timestamps for the top-1 hypothesis.
    pub words: Vec<WordEntry>,
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
    /// Verbose JSON body with word-level timestamps (cc-19).
    /// Content-type `application/json`.
    VerboseJson(VerboseTranscriptionResponse),
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
    /// The client asked for `response_format="verbose_json"` WITHOUT the
    /// `word` timestamp granularity. OpenAI's default granularity is
    /// `segment`, and segment-level timestamps are not implemented —
    /// silently returning a words-only body for a segments request would
    /// fabricate the shape (NFR-RL-06). HTTP mapper renders 501 with
    /// [`Self::SEGMENT_TIMESTAMP_DEFERRAL_NOTE`], which names the working
    /// word-level path (cc-19).
    VerboseJsonNotImplemented,
    /// The client asked for `response_format="srt"`. Segment-timestamp
    /// deferral, same note; distinct variant so the mapper can log the
    /// exact client ask.
    SrtNotImplemented,
    /// The client asked for `response_format="vtt"`. Same deferral as
    /// [`Self::SrtNotImplemented`].
    VttNotImplemented,
    /// The client sent `timestamp_granularities[]=segment` explicitly
    /// (alone or alongside `word`). Segment timestamps are not
    /// implemented; answering with a words-only body would silently drop
    /// the request (FR-EX-08). HTTP mapper renders 501.
    SegmentTimestampsNotImplemented,
    /// The client sent a `timestamp_granularities[]` value that is
    /// neither `word` nor `segment`. HTTP mapper renders 400.
    UnsupportedTimestampGranularity(String),
    /// `timestamp_granularities[]` was sent with a `response_format`
    /// other than `verbose_json`. OpenAI documents the field as
    /// verbose_json-only; accepting-and-ignoring it would be a silent
    /// no-op (FR-EX-08). HTTP mapper renders 400. Carries the format the
    /// client actually asked for.
    GranularityRequiresVerboseJson(String),
    /// `beam_size` exceeded [`MAX_BEAM_SIZE`]. HTTP mapper renders 400.
    BeamSizeTooLarge {
        /// The width the client asked for.
        requested: usize,
    },
    /// The T04 service layer refused the request (unknown model,
    /// Kokoro-style unavailability, inference failure). Preserved
    /// verbatim so the T05 mapper can pick 4xx vs 5xx off the inner
    /// variant.
    Service(ServiceError),
}

impl OpenAiTranscribeError {
    /// Stable, doc-referenced note the T05 HTTP mapper embeds in the 501
    /// response body for the segment-timestamp deferrals. Kept as a
    /// `const &'static str` so tests can assert on it exactly.
    ///
    /// History: this replaced `V0_5_TIMESTAMP_DEFERRAL_NOTE` when cc-19
    /// (2026-07-19 M4-residual audit) landed the word-level path — the old
    /// note's "word-level timestamps are deferred" claim became false.
    pub const SEGMENT_TIMESTAMP_DEFERRAL_NOTE: &'static str = "segment-level timestamps (verbose_json segments[] / srt / vtt) are \
         not implemented. Word-level timestamps ARE available: send \
         response_format=\"verbose_json\" with timestamp_granularities[]=word \
         (cc-19 unlock, 2026-07-19). Use response_format=\"json\" or \"text\" \
         for plain transcription.";
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
                    "response_format=verbose_json without \
                     timestamp_granularities[]=word defaults to segment \
                     timestamps, which are not implemented: {}",
                    Self::SEGMENT_TIMESTAMP_DEFERRAL_NOTE
                )
            }
            Self::SrtNotImplemented => {
                write!(
                    f,
                    "response_format=srt not implemented: {}",
                    Self::SEGMENT_TIMESTAMP_DEFERRAL_NOTE
                )
            }
            Self::VttNotImplemented => {
                write!(
                    f,
                    "response_format=vtt not implemented: {}",
                    Self::SEGMENT_TIMESTAMP_DEFERRAL_NOTE
                )
            }
            Self::SegmentTimestampsNotImplemented => {
                write!(
                    f,
                    "timestamp_granularities[]=segment not implemented: {}",
                    Self::SEGMENT_TIMESTAMP_DEFERRAL_NOTE
                )
            }
            Self::UnsupportedTimestampGranularity(s) => {
                write!(
                    f,
                    "unsupported timestamp_granularities value: `{s}` \
                     (accepted: word, segment)"
                )
            }
            Self::GranularityRequiresVerboseJson(fmt) => {
                write!(
                    f,
                    "timestamp_granularities[] requires \
                     response_format=\"verbose_json\" (got `{fmt}`); refusing \
                     to silently ignore the field (FR-EX-08)"
                )
            }
            Self::BeamSizeTooLarge { requested } => {
                write!(
                    f,
                    "beam_size {requested} exceeds the maximum {MAX_BEAM_SIZE} \
                     (bounded decode budget; use 0/1 for greedy or 2..={MAX_BEAM_SIZE} \
                     for beam search)"
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
    // Legacy entry point (no granularities, no beam) — kept so the T07/T08
    // callers and tests stay valid; delegates to the cc-19 full dispatch.
    transcribe_request_to_response(service, model, response_format_raw, &[], None, pcm)
}

/// Full request dispatch (cc-19, 2026-07-19 M4-residual audit): adds the
/// `timestamp_granularities[]` and `beam_size` surfaces on top of the T07
/// flow. Pure so the schema tests can drive every branch without a runtime.
///
/// * `granularities_raw` — raw `timestamp_granularities[]` values from the
///   multipart body (repeatable field). `word` / `segment` accepted
///   (case-insensitive); anything else is a 400. Only meaningful under
///   `verbose_json` — with any other format the field is an explicit 400
///   rather than a silent no-op (FR-EX-08).
/// * `beam_size` — decode width. `None`/`0`/`1` keeps the legacy greedy
///   path byte-for-byte; `2..=`[`MAX_BEAM_SIZE`] routes through
///   [`TranscribeService::transcribe_beam`]; larger is a 400.
///
/// # Errors
///
/// See [`OpenAiTranscribeError`] variants.
pub fn transcribe_request_to_response(
    service: &dyn TranscribeService,
    model: &str,
    response_format_raw: Option<&str>,
    granularities_raw: &[String],
    beam_size: Option<usize>,
    pcm: &[f32],
) -> Result<TranscriptionOutcome, OpenAiTranscribeError> {
    let fmt = ResponseFormat::parse(response_format_raw)
        .map_err(OpenAiTranscribeError::UnsupportedResponseFormat)?;

    // Parse granularities strictly (unknown value → 400) BEFORE any
    // inference — same fail-early posture as the format parse.
    let mut want_word = false;
    let mut want_segment = false;
    for g in granularities_raw {
        match g.trim().to_ascii_lowercase().as_str() {
            "word" => want_word = true,
            "segment" => want_segment = true,
            _ => {
                return Err(OpenAiTranscribeError::UnsupportedTimestampGranularity(
                    g.clone(),
                ));
            }
        }
    }

    // OpenAI documents `timestamp_granularities[]` as verbose_json-only.
    // Accepting it under json/text and ignoring it would be a silent no-op
    // (FR-EX-08) → explicit 400 naming the mismatch.
    if (want_word || want_segment) && fmt != ResponseFormat::VerboseJson {
        let got = response_format_raw.unwrap_or("json").to_owned();
        return Err(OpenAiTranscribeError::GranularityRequiresVerboseJson(got));
    }

    // Bounded beam width (cc-19): reject before inference.
    if let Some(b) = beam_size
        && b > MAX_BEAM_SIZE
    {
        return Err(OpenAiTranscribeError::BeamSizeTooLarge { requested: b });
    }

    match fmt {
        // Subtitle formats need per-segment timestamps — still deferred.
        ResponseFormat::Srt => return Err(OpenAiTranscribeError::SrtNotImplemented),
        ResponseFormat::Vtt => return Err(OpenAiTranscribeError::VttNotImplemented),
        ResponseFormat::VerboseJson => {
            // Explicit `segment` granularity (alone or with `word`) —
            // answering words-only for a segments ask would silently drop
            // half the request (FR-EX-08).
            if want_segment {
                return Err(OpenAiTranscribeError::SegmentTimestampsNotImplemented);
            }
            // No granularity: OpenAI's verbose_json default is `segment`.
            if !want_word {
                return Err(OpenAiTranscribeError::VerboseJsonNotImplemented);
            }
            // Word path (cc-19 unlock): route through transcribe_beam with
            // the M4-20 word_timestamps flag. A model without alignment
            // heads raises an explicit engine error (mapped to 501 by the
            // HTTP layer) — timings are never fabricated.
            let req = TranscribeBeamRequest {
                beam_size,
                word_timestamps: Some(true),
                ..TranscribeBeamRequest::default()
            };
            let resp = service
                .transcribe_beam(model, pcm, &req)
                .map_err(OpenAiTranscribeError::Service)?;
            let words = resp
                .words
                .into_iter()
                .map(|w| WordEntry {
                    word: w.word,
                    start: w.start,
                    end: w.end,
                })
                .collect();
            return Ok(TranscriptionOutcome::VerboseJson(
                VerboseTranscriptionResponse {
                    task: "transcribe",
                    // Real decoded length — the handler decoded this exact
                    // buffer at TARGET_SAMPLE_RATE_HZ.
                    duration: pcm.len() as f64 / f64::from(TARGET_SAMPLE_RATE_HZ),
                    text: resp.text,
                    words,
                },
            ));
        }
        ResponseFormat::Json | ResponseFormat::Text => {}
    }

    // json / text. Greedy stays on the legacy `transcribe` call so the
    // default request is byte-for-byte unchanged; a real beam ask routes
    // through `transcribe_beam` (its `text` is the ranked top-1).
    let text = match beam_size {
        Some(b) if b > 1 => {
            let req = TranscribeBeamRequest {
                beam_size: Some(b),
                ..TranscribeBeamRequest::default()
            };
            service
                .transcribe_beam(model, pcm, &req)
                .map_err(OpenAiTranscribeError::Service)?
                .text
        }
        _ => service
            .transcribe(model, pcm)
            .map_err(OpenAiTranscribeError::Service)?,
    };

    Ok(match fmt {
        ResponseFormat::Json => TranscriptionOutcome::Json(TranscriptionResponse::new(text)),
        ResponseFormat::Text => TranscriptionOutcome::Text(text),
        // Handled above.
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
        timestamp_granularities,
        beam_size,
        // T06 accepts and preserves these fields on the request DTO.
        // Forwarding them into the Whisper decoder options is a
        // T07-followup responsibility; ignoring them here is intentional
        // (documented) rather than a mistake.
        language: _,
        temperature: _,
        prompt: _,
    } = req;

    let pcm = decode_pcm_wav(&audio_bytes)?;

    // cc-18 (2026-07-19 M4-residual audit): model inference is CPU-bound
    // and previously ran ON the async worker, stalling every other task on
    // that worker for the whole decode. Move it to the blocking pool —
    // same migration wyoming.rs's `transcribe` arm already did (the copy
    // pattern). The service Arc is Send + Sync; the owned request pieces
    // move into the closure. A JoinError (panicked blocking task) is an
    // explicit 500, never a hang.
    let svc = Arc::clone(&service);
    let join = tokio::task::spawn_blocking(move || {
        transcribe_request_to_response(
            svc.as_ref(),
            &model,
            response_format.as_deref(),
            &timestamp_granularities,
            beam_size,
            &pcm,
        )
    })
    .await;
    let outcome = match join {
        Ok(res) => res.map_err(HttpTranscriptionError::from_schema)?,
        Err(join_err) => {
            return Err(HttpTranscriptionError::inference_failed(format!(
                "transcribe task failed: {join_err}"
            )));
        }
    };

    Ok(match outcome {
        TranscriptionOutcome::Json(body) => axum::Json(body).into_response(),
        TranscriptionOutcome::Text(text) => (
            StatusCode::OK,
            [(header::CONTENT_TYPE, "text/plain; charset=utf-8")],
            text,
        )
            .into_response(),
        TranscriptionOutcome::VerboseJson(body) => axum::Json(body).into_response(),
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
    /// `timestamp_granularities[]` values, in arrival order (cc-19). The
    /// OpenAI SDK sends the bracketed field name once per value; the bare
    /// name is accepted too. Validated (`word` / `segment`) at dispatch.
    pub timestamp_granularities: Vec<String>,
    /// Optional beam width (Vokra extension, cc-19). `None`/`0`/`1` =
    /// greedy (legacy path, byte-compatible); `2..=`[`MAX_BEAM_SIZE`] =
    /// beam search; larger → 400 at dispatch.
    pub beam_size: Option<usize>,
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
        let mut timestamp_granularities: Vec<String> = Vec::new();
        let mut beam_size: Option<usize> = None;

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
                // cc-19: the OpenAI SDK sends the bracketed array field name
                // once per value; some clients send the bare name. Both
                // accepted; values validated (`word` / `segment`) at dispatch.
                "timestamp_granularities[]" | "timestamp_granularities" => {
                    timestamp_granularities
                        .push(read_text_field(field, "timestamp_granularities").await?);
                }
                // cc-19: bounded beam width (Vokra extension).
                "beam_size" => {
                    let s = read_text_field(field, "beam_size").await?;
                    let value = s.trim().parse::<usize>().map_err(|_| {
                        HttpTranscriptionError::bad_multipart(format!(
                            "`beam_size` is not a non-negative integer: {s:?}"
                        ))
                    })?;
                    beam_size = Some(value);
                }
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
            timestamp_granularities,
            beam_size,
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

    /// cc-19: engine-declared capability hole (FR-EX-08 explicit error) —
    /// 501 with the distinct `unsupported_op` code so clients can branch on
    /// "this build/model cannot do that" vs the schema-level
    /// `not_implemented` deferrals.
    fn unsupported_op(message: String) -> Self {
        Self {
            status: StatusCode::NOT_IMPLEMENTED,
            error_type: "invalid_request_error",
            message,
            code: Some("unsupported_op"),
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
            | OpenAiTranscribeError::VttNotImplemented
            | OpenAiTranscribeError::SegmentTimestampsNotImplemented => {
                Self::not_implemented(err.to_string())
            }
            // cc-19 request-shape rejections — client-fixable, 400.
            OpenAiTranscribeError::UnsupportedTimestampGranularity(_)
            | OpenAiTranscribeError::GranularityRequiresVerboseJson(_)
            | OpenAiTranscribeError::BeamSizeTooLarge { .. } => {
                Self::bad_multipart(err.to_string())
            }
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
            // cc-19: an engine-declared capability hole (e.g. word
            // timestamps on a model whose GGUF carries no alignment heads,
            // or beam search on an engine that only implements greedy) is a
            // 501 carrying the engine's precise message — never a generic
            // 500 and never fabricated output. Matches piper_http's
            // `server_error_from_service` table for the same variant.
            ServiceError::Inference(VokraError::UnsupportedOp(detail)) => {
                Self::unsupported_op(detail)
            }
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

    /// INTENTIONALLY UPDATED for cc-19 (2026-07-19 M4-residual audit):
    /// bare `verbose_json` (no word granularity = OpenAI's segment
    /// default) is still 501, but the note now names the WORKING
    /// word-level path instead of claiming a blanket v1.0+ deferral.
    #[test]
    fn verbose_json_without_word_granularity_returns_501_with_documented_note() {
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
                assert!(msg.contains("segment"));
                // And exposes the exact byte-stable constant, which must
                // point clients at the working word-level path.
                assert!(
                    OpenAiTranscribeError::SEGMENT_TIMESTAMP_DEFERRAL_NOTE
                        .contains("timestamp_granularities[]=word")
                );
                assert!(
                    OpenAiTranscribeError::SEGMENT_TIMESTAMP_DEFERRAL_NOTE
                        .contains("response_format=\"json\" or \"text\"")
                );
            }
            other => panic!("expected VerboseJsonNotImplemented, got {other}"),
        }
    }

    /// INTENTIONALLY UPDATED for cc-19: srt/vtt stay 501 (they need
    /// per-segment timestamps), note text refreshed.
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
            assert!(msg.contains("segment-level timestamps"));
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

    // ===================================================================
    // cc-19 (2026-07-19 M4-residual audit): word timestamps + beam_size.
    // ===================================================================

    use crate::service::{TranscribeBeamResponse, WordTimestamp};
    use std::sync::Mutex;

    /// Beam-capable mock: records every [`TranscribeBeamRequest`] it sees
    /// and returns two canned word timings. `transcribe` (greedy) is
    /// recorded separately so tests can prove WHICH path dispatched.
    #[derive(Default)]
    struct BeamRecorder {
        beam_calls: Mutex<Vec<TranscribeBeamRequest>>,
        greedy_calls: Mutex<usize>,
    }

    impl TranscribeService for BeamRecorder {
        fn transcribe(&self, _model: &str, _pcm: &[f32]) -> Result<String, ServiceError> {
            *self.greedy_calls.lock().unwrap() += 1;
            Ok("greedy text".to_owned())
        }

        fn transcribe_beam(
            &self,
            _model: &str,
            _pcm: &[f32],
            req: &TranscribeBeamRequest,
        ) -> Result<TranscribeBeamResponse, ServiceError> {
            self.beam_calls.lock().unwrap().push(req.clone());
            let words = if req.word_timestamps == Some(true) {
                vec![
                    WordTimestamp {
                        word: " and".to_owned(),
                        start: 0.0,
                        end: 0.42,
                    },
                    WordTimestamp {
                        word: " so".to_owned(),
                        start: 0.42,
                        end: 0.9,
                    },
                ]
            } else {
                Vec::new()
            };
            Ok(TranscribeBeamResponse {
                text: "beam text".to_owned(),
                alternatives: Vec::new(),
                words,
            })
        }
    }

    #[test]
    fn verbose_json_word_granularity_returns_words() {
        let svc = BeamRecorder::default();
        let pcm = vec![0.0f32; 32_000]; // 2 s @ 16 kHz.
        let outcome = transcribe_request_to_response(
            &svc,
            model_names::WHISPER_1,
            Some("verbose_json"),
            &["word".to_owned()],
            None,
            &pcm,
        )
        .expect("word granularity must succeed");
        let TranscriptionOutcome::VerboseJson(body) = outcome else {
            panic!("expected VerboseJson outcome");
        };
        assert_eq!(body.task, "transcribe");
        assert_eq!(body.text, "beam text");
        // Real duration from the decoded PCM length — 32 000 / 16 000 Hz.
        assert!((body.duration - 2.0).abs() < 1e-9);
        assert_eq!(body.words.len(), 2);
        assert_eq!(body.words[0].word, " and");
        assert!((body.words[0].start - 0.0).abs() < 1e-9);
        assert!((body.words[1].end - 0.9).abs() < 1e-9);
        // The word ask reached the service as the M4-20 flag.
        let calls = svc.beam_calls.lock().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].word_timestamps, Some(true));
        assert_eq!(calls[0].beam_size, None);
        assert_eq!(*svc.greedy_calls.lock().unwrap(), 0);
    }

    #[test]
    fn verbose_json_word_with_beam_size_forwards_width() {
        let svc = BeamRecorder::default();
        let pcm = vec![0.0f32; 1_600];
        let outcome = transcribe_request_to_response(
            &svc,
            model_names::WHISPER_1,
            Some("verbose_json"),
            &["word".to_owned()],
            Some(5),
            &pcm,
        )
        .expect("must succeed");
        assert!(matches!(outcome, TranscriptionOutcome::VerboseJson(_)));
        let calls = svc.beam_calls.lock().unwrap();
        assert_eq!(calls[0].beam_size, Some(5));
        assert_eq!(calls[0].word_timestamps, Some(true));
    }

    #[test]
    fn granularity_values_are_case_insensitive_and_repeatable() {
        let svc = BeamRecorder::default();
        let pcm = vec![0.0f32; 4];
        let outcome = transcribe_request_to_response(
            &svc,
            model_names::WHISPER_1,
            Some("verbose_json"),
            &["Word".to_owned(), "WORD".to_owned()],
            None,
            &pcm,
        )
        .expect("case-insensitive word must succeed");
        assert!(matches!(outcome, TranscriptionOutcome::VerboseJson(_)));
    }

    #[test]
    fn unknown_granularity_is_400_before_engine_call() {
        struct PanicOnCall;
        impl TranscribeService for PanicOnCall {
            fn transcribe(&self, _m: &str, _p: &[f32]) -> Result<String, ServiceError> {
                panic!("must not dispatch");
            }
        }
        let pcm = vec![0.0f32; 4];
        let err = transcribe_request_to_response(
            &PanicOnCall,
            model_names::WHISPER_1,
            Some("verbose_json"),
            &["paragraph".to_owned()],
            None,
            &pcm,
        )
        .expect_err("must reject");
        match err {
            OpenAiTranscribeError::UnsupportedTimestampGranularity(s) => {
                assert_eq!(s, "paragraph");
            }
            other => panic!("expected UnsupportedTimestampGranularity, got {other}"),
        }
    }

    #[test]
    fn segment_granularity_is_501_even_alongside_word() {
        let svc = BeamRecorder::default();
        let pcm = vec![0.0f32; 4];
        // Alone.
        let err = transcribe_request_to_response(
            &svc,
            model_names::WHISPER_1,
            Some("verbose_json"),
            &["segment".to_owned()],
            None,
            &pcm,
        )
        .expect_err("segment must reject");
        assert!(matches!(
            err,
            OpenAiTranscribeError::SegmentTimestampsNotImplemented
        ));
        // Alongside word: answering words-only would silently drop the
        // segments half of the ask (FR-EX-08).
        let err = transcribe_request_to_response(
            &svc,
            model_names::WHISPER_1,
            Some("verbose_json"),
            &["word".to_owned(), "segment".to_owned()],
            None,
            &pcm,
        )
        .expect_err("word+segment must reject");
        assert!(matches!(
            err,
            OpenAiTranscribeError::SegmentTimestampsNotImplemented
        ));
        assert!(svc.beam_calls.lock().unwrap().is_empty());
    }

    #[test]
    fn granularity_under_json_format_is_400_not_silent_ignore() {
        let svc = BeamRecorder::default();
        let pcm = vec![0.0f32; 4];
        for fmt in [None, Some("json"), Some("text")] {
            let err = transcribe_request_to_response(
                &svc,
                model_names::WHISPER_1,
                fmt,
                &["word".to_owned()],
                None,
                &pcm,
            )
            .expect_err("granularity without verbose_json must reject");
            assert!(matches!(
                err,
                OpenAiTranscribeError::GranularityRequiresVerboseJson(_)
            ));
        }
        assert!(svc.beam_calls.lock().unwrap().is_empty());
        assert_eq!(*svc.greedy_calls.lock().unwrap(), 0);
    }

    #[test]
    fn beam_size_above_max_is_400_before_engine_call() {
        struct PanicOnCall;
        impl TranscribeService for PanicOnCall {
            fn transcribe(&self, _m: &str, _p: &[f32]) -> Result<String, ServiceError> {
                panic!("must not dispatch");
            }
        }
        let pcm = vec![0.0f32; 4];
        let err = transcribe_request_to_response(
            &PanicOnCall,
            model_names::WHISPER_1,
            None,
            &[],
            Some(MAX_BEAM_SIZE + 1),
            &pcm,
        )
        .expect_err("oversized beam must reject");
        match err {
            OpenAiTranscribeError::BeamSizeTooLarge { requested } => {
                assert_eq!(requested, MAX_BEAM_SIZE + 1);
            }
            other => panic!("expected BeamSizeTooLarge, got {other}"),
        }
    }

    #[test]
    fn beam_size_routes_json_through_beam_path() {
        let svc = BeamRecorder::default();
        let pcm = vec![0.0f32; 4];
        let outcome =
            transcribe_request_to_response(&svc, model_names::WHISPER_1, None, &[], Some(4), &pcm)
                .expect("beam json must succeed");
        let TranscriptionOutcome::Json(body) = outcome else {
            panic!("expected Json outcome");
        };
        assert_eq!(body.text, "beam text");
        let calls = svc.beam_calls.lock().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].beam_size, Some(4));
        // No word ask on the plain-json beam path.
        assert_eq!(calls[0].word_timestamps, None);
        assert_eq!(*svc.greedy_calls.lock().unwrap(), 0);
    }

    #[test]
    fn greedy_default_stays_on_legacy_transcribe_path() {
        // beam_size unset / 0 / 1 must stay byte-for-byte on the legacy
        // greedy call — a service whose transcribe_beam panics proves the
        // beam path is never touched.
        struct GreedyOnly;
        impl TranscribeService for GreedyOnly {
            fn transcribe(&self, _m: &str, _p: &[f32]) -> Result<String, ServiceError> {
                Ok("greedy".to_owned())
            }
            fn transcribe_beam(
                &self,
                _m: &str,
                _p: &[f32],
                _req: &TranscribeBeamRequest,
            ) -> Result<TranscribeBeamResponse, ServiceError> {
                panic!("greedy request must not route through transcribe_beam");
            }
        }
        let pcm = vec![0.0f32; 4];
        for beam in [None, Some(0), Some(1)] {
            let outcome = transcribe_request_to_response(
                &GreedyOnly,
                model_names::WHISPER_1,
                None,
                &[],
                beam,
                &pcm,
            )
            .expect("greedy must succeed");
            let TranscriptionOutcome::Json(body) = outcome else {
                panic!("expected Json outcome");
            };
            assert_eq!(body.text, "greedy");
        }
    }

    /// The stock (non-overridden) `transcribe_beam` default must fail
    /// CLOSED on a word ask — never fold to greedy with a fabricated empty
    /// `words` list (the trait-default hardening that cc-19 added).
    #[test]
    fn default_impl_word_ask_is_explicit_unsupported_not_empty_words() {
        // EchoTranscribe does NOT override transcribe_beam.
        let pcm = vec![0.0f32; 4];
        let err = transcribe_request_to_response(
            &EchoTranscribe,
            model_names::WHISPER_1,
            Some("verbose_json"),
            &["word".to_owned()],
            None,
            &pcm,
        )
        .expect_err("word ask on default impl must reject");
        match err {
            OpenAiTranscribeError::Service(ServiceError::Inference(VokraError::UnsupportedOp(
                msg,
            ))) => {
                assert!(msg.contains("word_timestamps"), "{msg}");
            }
            other => panic!("expected Service(Inference(UnsupportedOp)), got {other}"),
        }
    }

    #[test]
    fn verbose_json_serialises_openai_word_shape() {
        // Byte-shape pin: keys and casing of the verbose envelope.
        let body = VerboseTranscriptionResponse {
            task: "transcribe",
            duration: 1.5,
            text: "hi there".to_owned(),
            words: vec![WordEntry {
                word: " hi".to_owned(),
                start: 0.1,
                end: 0.4,
            }],
        };
        let json = serde_json::to_value(&body).unwrap();
        assert_eq!(json["task"], "transcribe");
        assert_eq!(json["duration"], 1.5);
        assert_eq!(json["text"], "hi there");
        assert_eq!(json["words"][0]["word"], " hi");
        assert_eq!(json["words"][0]["start"], 0.1);
        assert_eq!(json["words"][0]["end"], 0.4);
        // NFR-RL-06: no fabricated language / segments keys.
        assert!(json.get("language").is_none());
        assert!(json.get("segments").is_none());
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
        // INTENTIONALLY UPDATED for cc-19: the stable deferral note now
        // names the working word-level path instead of "deferred to v1.0+".
        assert!(
            json["error"]["message"]
                .as_str()
                .unwrap_or_default()
                .contains("timestamp_granularities[]=word"),
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

    // -----------------------------------------------------------------
    // cc-19 (2026-07-19 M4-residual audit): word timestamps + beam_size
    // over the real multipart route.
    // -----------------------------------------------------------------

    use crate::service::{TranscribeBeamRequest, TranscribeBeamResponse, WordTimestamp};
    use vokra_core::VokraError;

    /// Beam-capable fake with recorded requests; returns canned word
    /// timings when asked.
    #[derive(Default)]
    struct FakeBeam {
        beam_calls: Mutex<Vec<TranscribeBeamRequest>>,
    }

    impl TranscribeService for FakeBeam {
        fn transcribe(&self, _model: &str, _pcm: &[f32]) -> Result<String, ServiceError> {
            Ok("greedy".to_owned())
        }
        fn transcribe_beam(
            &self,
            _model: &str,
            _pcm: &[f32],
            req: &TranscribeBeamRequest,
        ) -> Result<TranscribeBeamResponse, ServiceError> {
            self.beam_calls.lock().unwrap().push(req.clone());
            Ok(TranscribeBeamResponse {
                text: "and so my fellow".to_owned(),
                alternatives: Vec::new(),
                words: vec![
                    WordTimestamp {
                        word: " and".to_owned(),
                        start: 0.0,
                        end: 0.32,
                    },
                    WordTimestamp {
                        word: " so".to_owned(),
                        start: 0.32,
                        end: 0.58,
                    },
                ],
            })
        }
    }

    /// Fake mirroring a model whose GGUF carries no alignment heads: the
    /// engine raises the explicit FR-EX-08 `UnsupportedOp` from
    /// `vokra-core::decode::beam_search` instead of fabricating timings.
    struct HeadlessBeam;
    impl TranscribeService for HeadlessBeam {
        fn transcribe(&self, _model: &str, _pcm: &[f32]) -> Result<String, ServiceError> {
            Ok("greedy".to_owned())
        }
        fn transcribe_beam(
            &self,
            _model: &str,
            _pcm: &[f32],
            _req: &TranscribeBeamRequest,
        ) -> Result<TranscribeBeamResponse, ServiceError> {
            Err(ServiceError::Inference(VokraError::UnsupportedOp(
                "word_timestamps requested but the model supplies no alignment \
                 (cross-attention) — FR-EX-08"
                    .to_owned(),
            )))
        }
    }

    /// `verbose_json` + `timestamp_granularities[]=word` over multipart →
    /// 200 with the OpenAI words[] shape (the cc-19 501 unlock).
    #[tokio::test]
    async fn route_verbose_json_word_returns_words_array() {
        let fake = Arc::new(FakeBeam::default());
        let app = build_app(fake.clone() as Arc<dyn TranscribeService>);

        let wav = make_wav(16_000); // 1 s @ 16 kHz.
        let boundary = "----vokra-test-boundary";
        let body = build_multipart(
            vec![
                ("file", &wav, Some("jfk.wav")),
                ("model", b"whisper-1", None),
                ("response_format", b"verbose_json", None),
                ("timestamp_granularities[]", b"word", None),
                ("beam_size", b"5", None),
            ],
            boundary,
        );

        let response = app.oneshot(post_request(boundary, body)).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = read_body(response).await;
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["task"], "transcribe");
        assert_eq!(json["text"], "and so my fellow");
        // duration = 16 000 samples / 16 000 Hz.
        assert!((json["duration"].as_f64().unwrap() - 1.0).abs() < 1e-9);
        let words = json["words"].as_array().expect("words[] present");
        assert_eq!(words.len(), 2);
        assert_eq!(words[0]["word"], " and");
        assert!(words[0]["start"].as_f64().is_some());
        assert!(words[0]["end"].as_f64().is_some());
        // NFR-RL-06: nothing fabricated.
        assert!(json.get("language").is_none());
        assert!(json.get("segments").is_none());
        // The ask reached the service: word flag + bounded beam width.
        let calls = fake.beam_calls.lock().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].word_timestamps, Some(true));
        assert_eq!(calls[0].beam_size, Some(5));
    }

    /// A model without alignment heads → explicit 501 `unsupported_op`
    /// carrying the engine's precise message — never fabricated timings,
    /// never a generic 500 (cc-19).
    #[tokio::test]
    async fn route_word_ask_on_headless_model_is_501_unsupported_op() {
        let app = build_app(Arc::new(HeadlessBeam) as Arc<dyn TranscribeService>);

        let wav = make_wav(800);
        let boundary = "----vokra-test-boundary";
        let body = build_multipart(
            vec![
                ("file", &wav, Some("x.wav")),
                ("response_format", b"verbose_json", None),
                ("timestamp_granularities[]", b"word", None),
            ],
            boundary,
        );

        let response = app.oneshot(post_request(boundary, body)).await.unwrap();
        assert_eq!(response.status(), StatusCode::NOT_IMPLEMENTED);
        let bytes = read_body(response).await;
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["error"]["code"], "unsupported_op");
        assert!(
            json["error"]["message"]
                .as_str()
                .unwrap()
                .contains("no alignment"),
            "must carry the engine's precise message, got {json}"
        );
    }

    /// `beam_size` above the bound is 400 before dispatch.
    #[tokio::test]
    async fn route_beam_size_above_max_is_400() {
        let fake = Arc::new(FakeBeam::default());
        let app = build_app(fake.clone() as Arc<dyn TranscribeService>);

        let wav = make_wav(800);
        let boundary = "----vokra-test-boundary";
        let body = build_multipart(
            vec![("file", &wav, Some("x.wav")), ("beam_size", b"9", None)],
            boundary,
        );

        let response = app.oneshot(post_request(boundary, body)).await.unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let bytes = read_body(response).await;
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(json["error"]["code"], "invalid_multipart");
        assert!(
            json["error"]["message"]
                .as_str()
                .unwrap()
                .contains("beam_size 9 exceeds the maximum 8"),
            "got {json}"
        );
        assert!(fake.beam_calls.lock().unwrap().is_empty());
    }

    /// Non-integer `beam_size` is a 400 at multipart parse.
    #[tokio::test]
    async fn route_beam_size_non_integer_is_400() {
        let fake = Arc::new(FakeBeam::default());
        let app = build_app(fake as Arc<dyn TranscribeService>);

        let wav = make_wav(800);
        let boundary = "----vokra-test-boundary";
        let body = build_multipart(
            vec![("file", &wav, Some("x.wav")), ("beam_size", b"wide", None)],
            boundary,
        );

        let response = app.oneshot(post_request(boundary, body)).await.unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    /// `timestamp_granularities[]` with a non-verbose format is an explicit
    /// 400 (never a silent ignore) over the wire too.
    #[tokio::test]
    async fn route_granularity_without_verbose_json_is_400() {
        let fake = Arc::new(FakeBeam::default());
        let app = build_app(fake.clone() as Arc<dyn TranscribeService>);

        let wav = make_wav(800);
        let boundary = "----vokra-test-boundary";
        let body = build_multipart(
            vec![
                ("file", &wav, Some("x.wav")),
                ("response_format", b"json", None),
                ("timestamp_granularities[]", b"word", None),
            ],
            boundary,
        );

        let response = app.oneshot(post_request(boundary, body)).await.unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let bytes = read_body(response).await;
        let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert!(
            json["error"]["message"]
                .as_str()
                .unwrap()
                .contains("verbose_json"),
            "got {json}"
        );
        assert!(fake.beam_calls.lock().unwrap().is_empty());
    }
}
