//! piper-plus `POST /api/tts` request/response schema (T11).
//!
//! # What this cut lands (T11)
//!
//! * Request DTO ([`TtsRequest`]) matching the piper-plus HTTP API
//!   contract confirmed at implementation time (T11 spec):
//!   `{"text", "voice", "model"?, "length_scale"?, "noise_scale"?}`.
//! * Response outcome ([`TtsOutcome`]) — a raw WAV byte body
//!   (PCM 16-bit LE mono at the voice's native sample rate). This is the
//!   MIME `audio/wav` payload piper-plus HTTP clients read directly
//!   (`response.content` in requests / `await resp.read()` in aiohttp),
//!   without a JSON envelope.
//! * [`dispatch_tts`] — the pure, in-process schema layer the T12 axum
//!   handler will call after decoding the JSON body. Splits routing
//!   (model / voice resolution + parameter policy) from the axum
//!   surface so schema tests do not need a live tokio runtime.
//! * [`PiperTtsError`] — schema-level failure envelope preserving the
//!   T04 [`ServiceError`] verbatim so the T05 HTTP mapper can pick
//!   4xx vs 5xx off the inner variant. FR-EX-08 spirit: preserve the
//!   failure kind end-to-end, never silently reroute.
//!
//! T12 wires this into an axum route; T13 stands up a compat integration
//! test that POSTs against a live server.
//!
//! # piper-plus HTTP API contract (as confirmed 2026-07-06)
//!
//! The Rhasspy piper HTTP server (and its faster-whisper-style drop-in
//! clones the ecosystem has settled on) accepts a JSON POST body with
//! the fields above and streams back a raw WAV. `voice` is required
//! (piper voices are per-file); `model` is an optional Vokra extension
//! that selects the underlying engine (defaults to
//! [`model_names::PIPER_PLUS`](crate::service::model_names::PIPER_PLUS)).
//!
//! `length_scale` and `noise_scale` are the classical piper knobs.
//! Vokra's native MB-iSTFT-VITS2 engine currently sources both from the
//! voice's own defaults baked into the GGUF (`vokra.piper.length_scale`
//! / `vokra.piper.noise_scale` — see `crates/vokra-models/src/piper_plus
//! /config.rs`); per-request overrides are **not wired** in v0.5. Rather
//! than silently ignore the override (FR-EX-08 / NFR-RL-06 — no
//! fabricated output), a request that sets either knob to a value that
//! is not (near-)equal to the voice's default returns
//! [`PiperTtsError::PerRequestOverrideNotImplemented`] → 501. This
//! mirrors the OpenAI `verbose_json` deferral at T07: the surface accepts
//! the field, tells the caller honestly it is not wired, and points at
//! the deferral milestone.
//!
//! # Boundaries
//!
//! * **G2P is NOT eSpeak-NG** — that is GPL-3.0 and never linked into
//!   Vokra. The service layer (T04) drives synthesis through
//!   `vokra_piper_plus::Phonemizer`, backed by the excluded workspace
//!   `integrations/vokra-piper-g2p` in production. Schema tests use
//!   the `PassthroughPhonemizer` default so they run without a G2P dep.
//! * **Watermark firing is disabled at v0.5** (2026-07-04 owner
//!   decision). This surface accepts audio-only responses and never
//!   embeds an inaudible marker; the `WatermarkConfig` forward-compat
//!   hook lives on `InferenceService`, not here.
//! * **Voice resolution is by string tag** for the T11 schema. The T12
//!   handler will thread the `voice` field into the `InferenceService`
//!   registry; T11 forwards it verbatim through the `TtsRequest`
//!   struct so the schema surface is round-trip-testable.

#![allow(clippy::result_large_err)] // PiperTtsError intentionally embeds ServiceError verbatim (FR-EX-08 kind preservation).

use serde::{Deserialize, Serialize};

use crate::service::{ServiceError, SynthesizeService, model_names};
use vokra_core::SynthesisRequest;

/// Sample width in bytes for the 16-bit LE PCM WAV bodies this surface
/// emits. Named so the WAV header builder and the test that inspects
/// its bytes read from the same constant.
const WAV_BITS_PER_SAMPLE: u16 = 16;

/// piper-plus voice channel count. Vokra's native MB-iSTFT-VITS2 emits
/// mono audio at the voice's own sample rate.
const WAV_CHANNELS: u16 = 1;

/// Absolute tolerance around the voice's default `length_scale` /
/// `noise_scale` values inside which a per-request override is accepted
/// as a no-op (the caller sent the default explicitly). Outside this
/// band, the override is a real request and — because the runtime does
/// not wire per-request overrides in v0.5 — must be rejected.
///
/// 1e-4 is well under any perceptual difference for these knobs (piper
/// defaults are on the order of 0.667 / 1.1) and larger than any
/// round-trip precision loss from client JSON serializers.
const OVERRIDE_TOLERANCE: f32 = 1e-4;

/// `POST /api/tts` JSON request body.
///
/// Round-trippable via serde so the T13 compat test can build the
/// same struct clients will send. Field naming matches piper-plus's
/// HTTP contract byte-for-byte.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct TtsRequest {
    /// Text to synthesize. Passed to the phonemizer verbatim; the
    /// `InferenceService` handles G2P via `vokra_piper_plus::Phonemizer`
    /// (never eSpeak-NG).
    pub text: String,
    /// Voice tag (piper voices are per-file: `en_US-lessac-medium`,
    /// `ja_JP-lessac-medium`, ...). Required by piper-plus. In v0.5 the
    /// registry advertises a single voice per model (the GGUF loaded at
    /// startup); mismatched tags are surfaced as
    /// [`PiperTtsError::VoiceNotAvailable`] rather than silently rerouted.
    pub voice: String,
    /// Optional model selector — Vokra extension. Defaults to
    /// [`model_names::PIPER_PLUS`]. Unknown model → 404 via the T04
    /// service layer.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Optional per-request `length_scale` override. Not wired in v0.5:
    /// a value that differs from the voice's default by more than
    /// [`OVERRIDE_TOLERANCE`] returns
    /// [`PiperTtsError::PerRequestOverrideNotImplemented`]. `None` uses
    /// the voice's default and is always accepted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub length_scale: Option<f32>,
    /// Optional per-request `noise_scale` override. Same v0.5 deferral
    /// as `length_scale`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub noise_scale: Option<f32>,
    /// Optional language tag (`"ja"`, `"en"`, ... — the voice GGUF's
    /// `vokra.piper.language_codes` inventory). cc-18 (2026-07-19
    /// M4-residual audit): `SynthesisRequest.language` existed in the piper
    /// synthesis layer since M0 but no server surface could set it. `None`
    /// keeps the engine's language detection (the G2P's detected dominant
    /// language / voice default). A language the loaded voice does NOT
    /// support is an explicit 400 from the service layer (FR-EX-08 — the
    /// engine would otherwise silently fall back to the detected language).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
}

/// High-level outcome of a `/api/tts` request as far as this schema
/// layer is concerned. The T12 handler wraps this into an axum
/// `Response` (content-type `audio/wav`) or through the T05 error
/// mapper for the failure paths.
#[derive(Debug, Clone)]
pub enum TtsOutcome {
    /// Raw WAV bytes (RIFF/WAVE, PCM 16-bit LE, mono). Content-type
    /// `audio/wav`.
    Wav(Vec<u8>),
}

impl TtsOutcome {
    /// Returns the raw WAV bytes for the `Wav` variant.
    pub fn wav_bytes(&self) -> &[u8] {
        match self {
            Self::Wav(b) => b,
        }
    }
}

/// Errors this schema layer surfaces to the T05 HTTP error mapper.
///
/// Kept distinct from [`ServiceError`] (a) so the mapper can pick
/// status codes off the schema-level reason without re-parsing an
/// inner text bag, and (b) to isolate the "per-request override
/// deferred" 501 case which does NOT correspond to any engine-level
/// failure (FR-EX-08 spirit: preserve the failure kind end-to-end).
#[derive(Debug)]
pub enum PiperTtsError {
    /// Request body missing / malformed field. HTTP mapper renders 400.
    InvalidRequest(String),
    /// The `voice` tag did not resolve to a loaded voice. Distinct
    /// from `ServiceError::UnknownModel` because in piper-plus's API
    /// the `voice` is the primary selector; the mapper renders 404.
    VoiceNotAvailable(String),
    /// The client set a non-default `length_scale` or `noise_scale`.
    /// The native runtime does not wire per-request overrides in
    /// v0.5. HTTP mapper renders 501 with [`Self::V0_5_OVERRIDE_DEFERRAL_NOTE`].
    /// (FR-EX-08: no silent fallback; NFR-RL-06: no fabricated audio.)
    PerRequestOverrideNotImplemented {
        /// Which knob was overridden (`"length_scale"` / `"noise_scale"`).
        knob: &'static str,
        /// The value the caller sent (echoed in the error message for
        /// debuggability).
        requested: f32,
        /// The voice's baked default (from `vokra.piper.*` metadata).
        default: f32,
    },
    /// The T04 service layer refused the request (unknown model,
    /// Kokoro-style unavailability, inference failure). Preserved
    /// verbatim so the T05 mapper can pick 4xx vs 5xx off the inner
    /// variant.
    Service(ServiceError),
}

impl PiperTtsError {
    /// Stable, doc-referenced note the T05 HTTP mapper embeds in the
    /// 501 response body for per-request overrides. Kept as
    /// `const &'static str` so tests can assert on it exactly and the
    /// T23 docs can reference it byte-for-byte.
    pub const V0_5_OVERRIDE_DEFERRAL_NOTE: &'static str = "per-request length_scale/noise_scale overrides are not wired in v0.5; \
         the voice's baked defaults (vokra.piper.length_scale / vokra.piper.noise_scale) \
         are used. Omit the field to accept the default, or bake a variant voice.";
}

impl std::fmt::Display for PiperTtsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidRequest(s) => write!(f, "invalid /api/tts request: {s}"),
            Self::VoiceNotAvailable(v) => write!(f, "voice not available: `{v}`"),
            Self::PerRequestOverrideNotImplemented {
                knob,
                requested,
                default,
            } => write!(
                f,
                "{knob}={requested} not implemented (voice default {default}): {}",
                Self::V0_5_OVERRIDE_DEFERRAL_NOTE
            ),
            Self::Service(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for PiperTtsError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Service(e) => Some(e),
            _ => None,
        }
    }
}

/// Voice-defaults hook the T11 schema layer consults to decide whether
/// a per-request `length_scale` / `noise_scale` is a no-op (accepted) or
/// a real override (rejected as not-implemented, v0.5).
///
/// The T04 `InferenceService` exposes these via the loaded piper
/// [`PiperConfig`](vokra_models::piper_plus::PiperConfig); tests inject
/// a simple double.
pub trait VoiceDefaults: Send + Sync {
    /// Returns `Some((default_length_scale, default_noise_scale))` iff
    /// the named voice is loaded. `None` = voice not available (schema
    /// layer maps to [`PiperTtsError::VoiceNotAvailable`]).
    fn defaults_for(&self, voice: &str) -> Option<(f32, f32)>;
}

/// Dispatch the request through [`SynthesizeService`] and shape the
/// response as a WAV byte body.
///
/// * `service` — injected trait object (real `InferenceService` in
///   production, mocks in T13 tests). Owns the piper-plus TTS engine
///   via T04.
/// * `voices` — injected voice-defaults hook, used only to compare
///   any per-request override against the voice's baked default.
/// * `req` — the parsed JSON body.
///
/// # Errors
///
/// See [`PiperTtsError`] variants.
#[allow(clippy::result_large_err)] // PiperTtsError is intentionally rich; boxing here would
// ripple through every T11 caller and T05 mapper without
// a measurable benefit — this is a single-shot handler,
// not a hot-path Result being propagated across many `?`s.
pub fn dispatch_tts(
    service: &dyn SynthesizeService,
    voices: &dyn VoiceDefaults,
    req: &TtsRequest,
) -> Result<TtsOutcome, PiperTtsError> {
    // 1) Basic input validation — do NOT run inference for empty text
    //    (FR-EX-08: no fabricated audio; also matches piper-plus HTTP
    //    behaviour which 400s on empty text).
    if req.text.is_empty() {
        return Err(PiperTtsError::InvalidRequest(
            "`text` must not be empty".into(),
        ));
    }
    if req.voice.is_empty() {
        return Err(PiperTtsError::InvalidRequest(
            "`voice` must not be empty".into(),
        ));
    }
    // An explicitly-empty language is a malformed request, not "use the
    // default" — omit the field for default behaviour (FR-EX-08: no silent
    // reinterpretation of a present-but-empty value).
    if req.language.as_deref() == Some("") {
        return Err(PiperTtsError::InvalidRequest(
            "`language` must not be empty when present (omit the field to use \
             the voice's language detection)"
                .into(),
        ));
    }

    // 2) Voice must resolve — piper's primary selector. If the caller
    //    also asked for per-request knob overrides we STILL need the
    //    voice defaults to compare against, so this precedes the
    //    override check.
    let (default_length, default_noise) = voices
        .defaults_for(&req.voice)
        .ok_or_else(|| PiperTtsError::VoiceNotAvailable(req.voice.clone()))?;

    // 3) Per-request length_scale / noise_scale overrides are not wired
    //    in v0.5. Accept the field only when it is (near-)equal to the
    //    voice's default; anything else is 501 with the documented
    //    deferral note (FR-EX-08, NFR-RL-06).
    if let Some(ls) = req.length_scale {
        if !nearly_equal(ls, default_length) {
            return Err(PiperTtsError::PerRequestOverrideNotImplemented {
                knob: "length_scale",
                requested: ls,
                default: default_length,
            });
        }
    }
    if let Some(ns) = req.noise_scale {
        if !nearly_equal(ns, default_noise) {
            return Err(PiperTtsError::PerRequestOverrideNotImplemented {
                knob: "noise_scale",
                requested: ns,
                default: default_noise,
            });
        }
    }

    // 4) Resolve model. Defaults to piper-plus; the service layer maps
    //    an unknown model to `ServiceError::UnknownModel` → 404.
    let model: &str = req.model.as_deref().unwrap_or(model_names::PIPER_PLUS);

    // 5) Build the vokra-core `SynthesisRequest`. `length_scale` /
    //    `noise_scale` are consumed at voice defaults (step 3 above)
    //    so we do NOT plumb them through — the runtime uses the voice's
    //    baked values. `language` (cc-18) IS plumbed: the service layer
    //    validates it against the voice's `vokra.piper.language_codes`
    //    and rejects unsupported codes with an explicit 400 (FR-EX-08).
    let mut syn_req = SynthesisRequest::new(&req.text);
    if let Some(lang) = &req.language {
        syn_req = syn_req.with_language(lang);
    }

    let audio = service
        .synthesize(model, &syn_req)
        .map_err(PiperTtsError::Service)?;

    // 6) Package as WAV. piper-plus HTTP returns audio/wav directly
    //    (no JSON envelope), so the T12 handler will emit these bytes
    //    with content-type `audio/wav`.
    let bytes = encode_wav_pcm16(&audio.samples, audio.sample_rate);
    Ok(TtsOutcome::Wav(bytes))
}

/// Returns `true` iff `a` and `b` are within [`OVERRIDE_TOLERANCE`] of
/// each other. Neither value may be NaN; NaN inputs are treated as "not
/// equal" so an all-NaN request cannot silently pass as the default.
fn nearly_equal(a: f32, b: f32) -> bool {
    if a.is_nan() || b.is_nan() {
        return false;
    }
    (a - b).abs() <= OVERRIDE_TOLERANCE
}

/// Encode `samples` (mono, in `[-1.0, 1.0]`) as a RIFF/WAVE PCM 16-bit
/// LE byte body. Header layout: standard 44-byte prelude (RIFF/WAVE +
/// `fmt ` + `data`). Sample scaling saturates on out-of-range values
/// (never wraps, so a spike in the model output cannot produce a
/// mis-signed sample — NFR-RL-06 spirit).
///
/// Kept inline (not pulled from `vokra-cli`, which has its own WAV
/// writer) so the excluded workspace does not need a cross-crate WAV
/// helper — one file, one policy, one hand-audited byte layout.
fn encode_wav_pcm16(samples: &[f32], sample_rate: u32) -> Vec<u8> {
    let bytes_per_sample = (WAV_BITS_PER_SAMPLE / 8) as u32;
    let byte_rate = sample_rate * WAV_CHANNELS as u32 * bytes_per_sample;
    let block_align = WAV_CHANNELS * (WAV_BITS_PER_SAMPLE / 8);
    let data_size = (samples.len() as u32) * bytes_per_sample;
    let riff_size = 36 + data_size; // 4 ("WAVE") + 24 (fmt chunk) + 8 (data hdr) + data_size

    let mut buf = Vec::with_capacity(44 + data_size as usize);
    // RIFF header.
    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&riff_size.to_le_bytes());
    buf.extend_from_slice(b"WAVE");
    // fmt subchunk.
    buf.extend_from_slice(b"fmt ");
    buf.extend_from_slice(&16u32.to_le_bytes()); // subchunk size
    buf.extend_from_slice(&1u16.to_le_bytes()); // PCM format tag
    buf.extend_from_slice(&WAV_CHANNELS.to_le_bytes());
    buf.extend_from_slice(&sample_rate.to_le_bytes());
    buf.extend_from_slice(&byte_rate.to_le_bytes());
    buf.extend_from_slice(&block_align.to_le_bytes());
    buf.extend_from_slice(&WAV_BITS_PER_SAMPLE.to_le_bytes());
    // data subchunk.
    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&data_size.to_le_bytes());
    // Samples: saturating f32 → i16 in [-1.0, 1.0].
    for &s in samples {
        let clamped = s.clamp(-1.0, 1.0);
        let i = (clamped * i16::MAX as f32).round() as i32;
        let i = i.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
        buf.extend_from_slice(&i.to_le_bytes());
    }
    buf
}

// ===========================================================================
// T12 — HTTP surface: axum route for `POST /api/tts` (campaign-2 P1 #3
// follow-through; the 2026-07-17 server-real leg live-verified this route
// 404'd because `build_http_app` never merged it).
//
// Mirrors the T06 (openai.rs) / T09 (vllm.rs) conventions: JSON extractor
// rejection → 400 `invalid_input`, every failure through the crate-wide
// `ServerError` envelope via `finish_request` (per-request log + OpenAI-shaped
// error JSON), success = raw `audio/wav` bytes exactly as piper-plus HTTP
// clients expect (no JSON envelope).
// ===========================================================================

use axum::Router;
use axum::extract::{State, rejection::JsonRejection};
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use std::sync::Arc;

use crate::error::{ServerError, finish_request};
use crate::service::{AUDIO_WAV_CONTENT_TYPE, InferenceService};
use vokra_core::VokraError;

/// Route path — named so `server.rs` wiring tests and the T13 compat test
/// key on the same literal.
pub const TTS_ROUTE: &str = "/api/tts";

/// The generic voice tag the single-voice v0.5 registry answers in addition
/// to the model alias (`piper-plus`): piper HTTP clients that do not know
/// the per-file voice name send `"default"` (the T13 compat test does).
pub const DEFAULT_VOICE: &str = "default";

/// State for the `/api/tts` router: the synthesize dispatch plus the
/// voice-defaults hook, both as trait objects so tests can drive the route
/// with mocks and production passes two views of the same
/// [`InferenceService`] `Arc`.
#[derive(Clone)]
pub struct TtsHttpState {
    /// Synthesis dispatch (`InferenceService` in production).
    pub synth: Arc<dyn SynthesizeService>,
    /// Voice-defaults hook for the per-request override policy
    /// (`InferenceService` in production).
    pub voices: Arc<dyn VoiceDefaults>,
}

impl TtsHttpState {
    /// Production state: both views onto the same registry.
    pub fn from_service(service: Arc<InferenceService>) -> Self {
        Self {
            synth: service.clone(),
            voices: service,
        }
    }
}

/// [`VoiceDefaults`] for the production registry. v0.5 loads exactly one
/// piper voice (startup-required), so the resolvable tags are the generic
/// [`DEFAULT_VOICE`] and the model alias `piper-plus`; anything else is
/// [`PiperTtsError::VoiceNotAvailable`] → 404 (never a silent reroute to
/// the loaded voice, FR-EX-08). Lives here rather than `service.rs` so the
/// T04 layer stays free of HTTP-surface traits.
impl VoiceDefaults for InferenceService {
    fn defaults_for(&self, voice: &str) -> Option<(f32, f32)> {
        if voice == DEFAULT_VOICE || voice == model_names::PIPER_PLUS {
            let cfg = self.piper_voice().config();
            Some((cfg.length_scale, cfg.noise_scale))
        } else {
            None
        }
    }
}

/// Build the `/api/tts` router bound to `state`. Returns a plain
/// `Router<()>` (state already applied) so `build_http_app` merges it
/// exactly like the vLLM router.
pub fn router(state: TtsHttpState) -> Router {
    Router::new()
        .route(TTS_ROUTE, post(tts_handler))
        .with_state(state)
}

/// `POST /api/tts` — the piper-plus HTTP TTS handler.
///
/// * Malformed / non-JSON body → 400 `invalid_input` (same
///   `JsonRejection` funnel as the vLLM contract routes — axum's default
///   rejection body never leaks).
/// * Schema/dispatch failures map through [`server_error_from_tts`]
///   (status table pinned in the `route_tts_http` tests below).
/// * Success → `200 OK` with `Content-Type: audio/wav` raw bytes.
///
/// Synthesis runs on `tokio::task::spawn_blocking` (cc-18, 2026-07-19
/// M4-residual audit — the follow-up this doc used to carry). The engines
/// are CPU-bound; running them on the async worker previously stalled every
/// other task on that worker for the whole synthesis. This mirrors the
/// wyoming.rs `transcribe` migration (the copy pattern): a `JoinError`
/// (panicked/cancelled blocking task) is an explicit 500, never a hang.
async fn tts_handler(
    State(state): State<TtsHttpState>,
    body: Result<axum::Json<TtsRequest>, JsonRejection>,
) -> Response {
    let start = std::time::Instant::now();
    let (model, result) = match body {
        Ok(axum::Json(req)) => {
            let model_tag = req
                .model
                .clone()
                .unwrap_or_else(|| model_names::PIPER_PLUS.to_owned());
            // Move the owned request + Arc views onto the blocking pool;
            // both trait objects are Send + Sync so the closure is Send.
            let synth = Arc::clone(&state.synth);
            let voices = Arc::clone(&state.voices);
            let join = tokio::task::spawn_blocking(move || {
                dispatch_tts(synth.as_ref(), voices.as_ref(), &req)
            })
            .await;
            let outcome = match join {
                Ok(res) => res.map(wav_response).map_err(server_error_from_tts),
                Err(join_err) => Err(ServerError::InferenceFailed {
                    detail: format!("tts task failed: {join_err}"),
                }),
            };
            (Some(model_tag), outcome)
        }
        Err(err) => (
            None,
            Err(ServerError::InvalidInput {
                detail: err.to_string(),
            }),
        ),
    };
    finish_request("POST", TTS_ROUTE, model.as_deref(), start, result)
}

/// Success body: raw WAV bytes, `Content-Type: audio/wav` — piper-plus
/// clients read the body directly (`response.content`), no JSON envelope.
fn wav_response(outcome: TtsOutcome) -> Response {
    let TtsOutcome::Wav(bytes) = outcome;
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, AUDIO_WAV_CONTENT_TYPE)],
        bytes,
    )
        .into_response()
}

/// Map the T11 schema error onto the crate-wide [`ServerError`] envelope.
///
/// Status table (pinned by `route_tts_http::error_status_mapping_is_pinned`):
///
/// * `InvalidRequest` → 400 `invalid_input`
/// * `VoiceNotAvailable` → 404 `model_not_found` (the voice is piper's
///   primary selector; rendered as ``voice `x``` in the message)
/// * `PerRequestOverrideNotImplemented` → 501 `not_implemented` (carries
///   the documented v0.5 deferral note)
/// * `Service(UnknownModel)` → 404, `Service(SynthesizeUnavailable)` → 501
/// * `Service(Inference(UnsupportedOp))` → 501 (FR-EX-08 — a backend hole
///   is surfaced, never silently rewritten to CPU)
/// * `Service(Inference(InvalidArgument))` → 400 — **this is the
///   plain-text-without-G2P path**: the `PassthroughPhonemizer` raises an
///   explicit `InvalidArgument` for non-phoneme-id text, and the caller
///   sees an honest 400 naming the fix, never silent garbage audio
/// * `Service(Inference(NotImplemented))` → 501; everything else → 500
fn server_error_from_tts(err: PiperTtsError) -> ServerError {
    match err {
        PiperTtsError::InvalidRequest(msg) => ServerError::InvalidInput {
            detail: format!("/api/tts: {msg}"),
        },
        PiperTtsError::VoiceNotAvailable(v) => ServerError::ModelNotFound {
            model: format!("voice `{v}`"),
        },
        e @ PiperTtsError::PerRequestOverrideNotImplemented { .. } => ServerError::NotImplemented {
            detail: e.to_string(),
        },
        PiperTtsError::Service(inner) => server_error_from_service(inner),
    }
}

/// [`ServiceError`] → [`ServerError`] leg of [`server_error_from_tts`].
/// Split out so the status table is testable per service variant.
fn server_error_from_service(err: ServiceError) -> ServerError {
    match err {
        ServiceError::UnknownModel(m) => ServerError::ModelNotFound { model: m },
        e @ ServiceError::SynthesizeUnavailable { .. } => ServerError::NotImplemented {
            detail: e.to_string(),
        },
        ServiceError::Inference(VokraError::UnsupportedOp(detail)) => {
            ServerError::UnsupportedOp { detail }
        }
        ServiceError::Inference(VokraError::InvalidArgument(detail)) => {
            ServerError::InvalidInput { detail }
        }
        ServiceError::Inference(VokraError::NotImplemented(what)) => ServerError::NotImplemented {
            detail: what.to_owned(),
        },
        e @ (ServiceError::Inference(_)
        | ServiceError::ModelLoadFailed { .. }
        | ServiceError::InvalidConfig(_)) => ServerError::InferenceFailed {
            detail: e.to_string(),
        },
    }
}

// ---------------------------------------------------------------------------
// Tests — `cargo test piper_http::route_tts`
// ---------------------------------------------------------------------------

#[cfg(test)]
mod route_tts {
    //! T11 acceptance tests: request/response schema, dispatch through
    //! a mocked [`SynthesizeService`], and the per-request override
    //! deferral policy. These tests exercise the schema surface
    //! **without** loading a piper-plus voice GGUF; T13 covers the
    //! end-to-end compat path with a real engine.

    use super::*;
    use crate::service::model_names;
    use vokra_core::{SynthesisRequest, SynthesizedAudio, VokraError};

    // -------- Mock SynthesizeService --------

    /// Emits a deterministic square-wave PCM so tests can key on the
    /// resulting WAV byte layout without a real engine.
    struct FakeSynth {
        sample_rate: u32,
    }

    impl SynthesizeService for FakeSynth {
        fn synthesize(
            &self,
            model: &str,
            _request: &SynthesisRequest,
        ) -> Result<SynthesizedAudio, ServiceError> {
            match model {
                model_names::PIPER_PLUS => {
                    // 4 samples — enough to exercise the header and byte
                    // ordering without bloating test output.
                    Ok(SynthesizedAudio::new(
                        vec![0.0, 0.5, -0.5, 1.0],
                        self.sample_rate,
                    ))
                }
                other => Err(ServiceError::UnknownModel(other.to_owned())),
            }
        }
    }

    /// Surfaces an engine-side unsupported-op — used to verify FR-EX-08:
    /// the schema layer forwards the error rather than silently
    /// falling back to CPU.
    struct UnsupportedSynth;

    impl SynthesizeService for UnsupportedSynth {
        fn synthesize(
            &self,
            _model: &str,
            _request: &SynthesisRequest,
        ) -> Result<SynthesizedAudio, ServiceError> {
            Err(ServiceError::Inference(VokraError::UnsupportedOp(
                "flow on Metal (M2-01 hole)".into(),
            )))
        }
    }

    // -------- Mock VoiceDefaults --------

    /// Single-voice defaults double: matches the shape the T04
    /// `InferenceService` exposes.
    struct FakeVoices {
        voice: &'static str,
        length: f32,
        noise: f32,
    }

    impl VoiceDefaults for FakeVoices {
        fn defaults_for(&self, voice: &str) -> Option<(f32, f32)> {
            if voice == self.voice {
                Some((self.length, self.noise))
            } else {
                None
            }
        }
    }

    fn default_voices() -> FakeVoices {
        // Matches the piper-plus voice defaults baked into the M0-07
        // reference voice (`vokra.piper.length_scale` = 1.1,
        // `vokra.piper.noise_scale` = 0.667; see
        // crates/vokra-models/src/piper_plus/config.rs).
        FakeVoices {
            voice: "en_US-lessac-medium",
            length: 1.1,
            noise: 0.667,
        }
    }

    // -------- Request round-trip --------

    #[test]
    fn tts_request_deserialises_piper_plus_contract() {
        // The exact JSON body piper-plus clients send.
        let raw = r#"{"text":"hello","voice":"en_US-lessac-medium"}"#;
        let req: TtsRequest = serde_json::from_str(raw).unwrap();
        assert_eq!(req.text, "hello");
        assert_eq!(req.voice, "en_US-lessac-medium");
        assert!(req.model.is_none());
        assert!(req.length_scale.is_none());
        assert!(req.noise_scale.is_none());
    }

    #[test]
    fn tts_request_accepts_all_optional_fields() {
        let raw = r#"{"text":"hi","voice":"ja_JP-lessac-medium","model":"piper-plus","length_scale":1.1,"noise_scale":0.667}"#;
        let req: TtsRequest = serde_json::from_str(raw).unwrap();
        assert_eq!(req.model.as_deref(), Some("piper-plus"));
        assert_eq!(req.length_scale, Some(1.1));
        assert_eq!(req.noise_scale, Some(0.667));
    }

    #[test]
    fn tts_request_round_trip_via_serde() {
        // Serializer skips None fields so the wire body stays clean.
        let req = TtsRequest {
            text: "hello".into(),
            voice: "en_US-lessac-medium".into(),
            model: None,
            length_scale: None,
            noise_scale: None,
            language: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        assert_eq!(json, r#"{"text":"hello","voice":"en_US-lessac-medium"}"#);
        let back: TtsRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(back, req);
    }

    // -------- Empty text / voice rejected --------

    #[test]
    fn empty_text_is_400_before_engine_call() {
        struct PanicOnCall;
        impl SynthesizeService for PanicOnCall {
            fn synthesize(
                &self,
                _model: &str,
                _request: &SynthesisRequest,
            ) -> Result<SynthesizedAudio, ServiceError> {
                panic!("must not be called for empty text");
            }
        }
        let req = TtsRequest {
            text: String::new(),
            voice: "en_US-lessac-medium".into(),
            model: None,
            length_scale: None,
            noise_scale: None,
            language: None,
        };
        let voices = default_voices();
        let err = dispatch_tts(&PanicOnCall, &voices, &req).expect_err("must reject empty text");
        match err {
            PiperTtsError::InvalidRequest(msg) => assert!(msg.contains("text")),
            other => panic!("expected InvalidRequest, got {other}"),
        }
    }

    #[test]
    fn empty_voice_is_400_before_engine_call() {
        let req = TtsRequest {
            text: "hello".into(),
            voice: String::new(),
            model: None,
            length_scale: None,
            noise_scale: None,
            language: None,
        };
        let voices = default_voices();
        let err = dispatch_tts(
            &FakeSynth {
                sample_rate: 22_050,
            },
            &voices,
            &req,
        )
        .expect_err("must reject empty voice");
        match err {
            PiperTtsError::InvalidRequest(msg) => assert!(msg.contains("voice")),
            other => panic!("expected InvalidRequest, got {other}"),
        }
    }

    // -------- Voice resolution --------

    #[test]
    fn unknown_voice_is_404_not_silent_reroute() {
        let req = TtsRequest {
            text: "hello".into(),
            voice: "de_DE-thorsten-medium".into(),
            model: None,
            length_scale: None,
            noise_scale: None,
            language: None,
        };
        let voices = default_voices();
        let err = dispatch_tts(
            &FakeSynth {
                sample_rate: 22_050,
            },
            &voices,
            &req,
        )
        .expect_err("must reject unknown voice");
        match err {
            PiperTtsError::VoiceNotAvailable(v) => assert_eq!(v, "de_DE-thorsten-medium"),
            other => panic!("expected VoiceNotAvailable, got {other}"),
        }
    }

    // -------- Per-request knob overrides --------

    #[test]
    fn none_length_and_noise_uses_voice_defaults_and_succeeds() {
        let req = TtsRequest {
            text: "hello".into(),
            voice: "en_US-lessac-medium".into(),
            model: None,
            length_scale: None,
            noise_scale: None,
            language: None,
        };
        let voices = default_voices();
        let outcome = dispatch_tts(
            &FakeSynth {
                sample_rate: 22_050,
            },
            &voices,
            &req,
        )
        .expect("defaults must succeed");
        let bytes = outcome.wav_bytes();
        // Header sanity: RIFF...WAVE...fmt ...data.
        assert_eq!(&bytes[0..4], b"RIFF");
        assert_eq!(&bytes[8..12], b"WAVE");
        assert_eq!(&bytes[12..16], b"fmt ");
        assert_eq!(&bytes[36..40], b"data");
    }

    #[test]
    fn explicit_default_length_scale_is_accepted() {
        // Sending the voice's own default (or a value inside the
        // tolerance) is a valid no-op — the caller just said what the
        // engine was going to do anyway.
        let req = TtsRequest {
            text: "hello".into(),
            voice: "en_US-lessac-medium".into(),
            model: None,
            length_scale: Some(1.1),
            noise_scale: None,
            language: None,
        };
        let voices = default_voices();
        dispatch_tts(
            &FakeSynth {
                sample_rate: 22_050,
            },
            &voices,
            &req,
        )
        .expect("explicit default must be accepted");
    }

    #[test]
    fn non_default_length_scale_is_501_with_documented_note() {
        let req = TtsRequest {
            text: "hello".into(),
            voice: "en_US-lessac-medium".into(),
            model: None,
            length_scale: Some(0.8),
            noise_scale: None,
            language: None,
        };
        let voices = default_voices();
        let err = dispatch_tts(
            &FakeSynth {
                sample_rate: 22_050,
            },
            &voices,
            &req,
        )
        .expect_err("non-default length_scale must reject");
        match err {
            PiperTtsError::PerRequestOverrideNotImplemented {
                knob,
                requested,
                default,
            } => {
                assert_eq!(knob, "length_scale");
                assert!((requested - 0.8).abs() < 1e-6);
                assert!((default - 1.1).abs() < 1e-6);
            }
            other => panic!("expected PerRequestOverrideNotImplemented, got {other}"),
        }
        // The stable deferral note is documented and referenced.
        assert!(PiperTtsError::V0_5_OVERRIDE_DEFERRAL_NOTE.contains("not wired in v0.5"));
    }

    #[test]
    fn non_default_noise_scale_is_501() {
        let req = TtsRequest {
            text: "hello".into(),
            voice: "en_US-lessac-medium".into(),
            model: None,
            length_scale: None,
            noise_scale: Some(1.0),
            language: None,
        };
        let voices = default_voices();
        let err = dispatch_tts(
            &FakeSynth {
                sample_rate: 22_050,
            },
            &voices,
            &req,
        )
        .expect_err("non-default noise_scale must reject");
        assert!(matches!(
            err,
            PiperTtsError::PerRequestOverrideNotImplemented {
                knob: "noise_scale",
                ..
            }
        ));
    }

    #[test]
    fn nan_override_is_treated_as_non_default() {
        // A NaN override must NOT silently pass as "close enough" —
        // FR-EX-08 spirit: an all-NaN request is honestly rejected.
        let req = TtsRequest {
            text: "hello".into(),
            voice: "en_US-lessac-medium".into(),
            model: None,
            length_scale: Some(f32::NAN),
            noise_scale: None,
            language: None,
        };
        let voices = default_voices();
        let err = dispatch_tts(
            &FakeSynth {
                sample_rate: 22_050,
            },
            &voices,
            &req,
        )
        .expect_err("NaN must reject");
        assert!(matches!(
            err,
            PiperTtsError::PerRequestOverrideNotImplemented { .. }
        ));
    }

    // -------- Language plumbing (cc-18) --------

    /// `language` must reach `SynthesisRequest.language` verbatim — the
    /// field existed in the synthesis layer since M0 with no server surface
    /// able to set it (cc-18, 2026-07-19 M4-residual audit).
    #[test]
    fn language_is_plumbed_into_synthesis_request() {
        use std::sync::Mutex;
        struct CaptureSynth {
            seen: Mutex<Vec<Option<String>>>,
        }
        impl SynthesizeService for CaptureSynth {
            fn synthesize(
                &self,
                _model: &str,
                request: &SynthesisRequest,
            ) -> Result<SynthesizedAudio, ServiceError> {
                self.seen.lock().unwrap().push(request.language.clone());
                Ok(SynthesizedAudio::new(vec![0.0], 22_050))
            }
        }
        let capture = CaptureSynth {
            seen: Mutex::new(Vec::new()),
        };
        let voices = default_voices();

        // With a language → forwarded verbatim.
        let req = TtsRequest {
            text: "hello".into(),
            voice: "en_US-lessac-medium".into(),
            model: None,
            length_scale: None,
            noise_scale: None,
            language: Some("ja".into()),
        };
        dispatch_tts(&capture, &voices, &req).expect("must dispatch");
        // Without → None (voice default / detection preserved).
        let req = TtsRequest {
            language: None,
            ..req
        };
        dispatch_tts(&capture, &voices, &req).expect("must dispatch");

        let seen = capture.seen.lock().unwrap().clone();
        assert_eq!(seen, vec![Some("ja".to_owned()), None]);
    }

    /// Present-but-empty `language` is 400 before any engine call — never a
    /// silent "treat as default" (FR-EX-08).
    #[test]
    fn empty_language_is_400_before_engine_call() {
        struct PanicOnCall;
        impl SynthesizeService for PanicOnCall {
            fn synthesize(
                &self,
                _model: &str,
                _request: &SynthesisRequest,
            ) -> Result<SynthesizedAudio, ServiceError> {
                panic!("must not be called for empty language");
            }
        }
        let req = TtsRequest {
            text: "hello".into(),
            voice: "en_US-lessac-medium".into(),
            model: None,
            length_scale: None,
            noise_scale: None,
            language: Some(String::new()),
        };
        let voices = default_voices();
        let err =
            dispatch_tts(&PanicOnCall, &voices, &req).expect_err("must reject empty language");
        match err {
            PiperTtsError::InvalidRequest(msg) => assert!(msg.contains("language")),
            other => panic!("expected InvalidRequest, got {other}"),
        }
    }

    // -------- Model routing --------

    #[test]
    fn model_defaults_to_piper_plus_when_absent() {
        let req = TtsRequest {
            text: "hello".into(),
            voice: "en_US-lessac-medium".into(),
            model: None,
            length_scale: None,
            noise_scale: None,
            language: None,
        };
        let voices = default_voices();
        // FakeSynth only recognises `piper-plus`; a wrong default would
        // 404 here.
        dispatch_tts(
            &FakeSynth {
                sample_rate: 22_050,
            },
            &voices,
            &req,
        )
        .expect("default model must route to piper-plus");
    }

    #[test]
    fn unknown_model_propagates_as_service_error_not_silent_success() {
        let req = TtsRequest {
            text: "hello".into(),
            voice: "en_US-lessac-medium".into(),
            model: Some("elevenlabs".into()),
            length_scale: None,
            noise_scale: None,
            language: None,
        };
        let voices = default_voices();
        let err = dispatch_tts(
            &FakeSynth {
                sample_rate: 22_050,
            },
            &voices,
            &req,
        )
        .expect_err("unknown model must reject");
        match err {
            PiperTtsError::Service(ServiceError::UnknownModel(m)) => assert_eq!(m, "elevenlabs"),
            other => panic!("expected Service(UnknownModel), got {other}"),
        }
    }

    #[test]
    fn engine_unsupported_op_propagates_not_silent_cpu_fallback() {
        // FR-EX-08 spirit: an engine-side UnsupportedOp on the caller's
        // requested backend must surface up, never a silent CPU
        // fallback that pretends the request succeeded on the wrong
        // path.
        let req = TtsRequest {
            text: "hello".into(),
            voice: "en_US-lessac-medium".into(),
            model: None,
            length_scale: None,
            noise_scale: None,
            language: None,
        };
        let voices = default_voices();
        let err =
            dispatch_tts(&UnsupportedSynth, &voices, &req).expect_err("unsupported op must reject");
        match err {
            PiperTtsError::Service(ServiceError::Inference(VokraError::UnsupportedOp(msg))) => {
                assert!(msg.contains("flow on Metal"));
            }
            other => panic!("expected Inference(UnsupportedOp), got {other}"),
        }
        // Error::source reaches into the ServiceError so the T05 log
        // layer can walk the chain.
        let err = dispatch_tts(&UnsupportedSynth, &voices, &req).unwrap_err();
        assert!(std::error::Error::source(&err).is_some());
    }

    // -------- WAV byte layout --------

    #[test]
    fn wav_body_has_valid_pcm16_header_and_correct_data_size() {
        let req = TtsRequest {
            text: "hello".into(),
            voice: "en_US-lessac-medium".into(),
            model: None,
            length_scale: None,
            noise_scale: None,
            language: None,
        };
        let voices = default_voices();
        let outcome = dispatch_tts(
            &FakeSynth {
                sample_rate: 22_050,
            },
            &voices,
            &req,
        )
        .unwrap();
        let bytes = outcome.wav_bytes();
        // Standard 44-byte header + 4 samples * 2 bytes = 52 bytes total.
        assert_eq!(bytes.len(), 44 + 4 * 2);
        // Sample rate (offset 24, 4 bytes LE) = 22050.
        let sr = u32::from_le_bytes([bytes[24], bytes[25], bytes[26], bytes[27]]);
        assert_eq!(sr, 22_050);
        // bits per sample (offset 34, 2 bytes LE) = 16.
        let bps = u16::from_le_bytes([bytes[34], bytes[35]]);
        assert_eq!(bps, 16);
        // Channels (offset 22, 2 bytes LE) = 1.
        let ch = u16::from_le_bytes([bytes[22], bytes[23]]);
        assert_eq!(ch, 1);
        // Data size (offset 40, 4 bytes LE) = 4 samples * 2 bytes.
        let data_size = u32::from_le_bytes([bytes[40], bytes[41], bytes[42], bytes[43]]);
        assert_eq!(data_size, 8);
        // First sample = 0.0 → 0i16 → bytes [0, 0].
        assert_eq!(&bytes[44..46], &[0x00, 0x00]);
    }

    #[test]
    fn wav_encoding_saturates_out_of_range_samples() {
        // A sample above +1.0 must clamp to i16::MAX, not wrap to a
        // negative value. NFR-RL-06 spirit: no fabricated / mis-signed
        // audio.
        let bytes = encode_wav_pcm16(&[2.0, -2.0], 22_050);
        let s0 = i16::from_le_bytes([bytes[44], bytes[45]]);
        let s1 = i16::from_le_bytes([bytes[46], bytes[47]]);
        assert_eq!(s0, i16::MAX);
        assert_eq!(s1, i16::MIN + 1); // -1.0 * 32767 = -32767 (not MIN)
    }

    // -------- Error mapping stability --------

    #[test]
    fn error_variants_display_and_source_are_wired() {
        // The T05 HTTP mapper walks Display for the user text and
        // source() for the log chain.
        let e = PiperTtsError::InvalidRequest("bad json".into());
        assert!(e.to_string().contains("bad json"));
        assert!(std::error::Error::source(&e).is_none());

        let e = PiperTtsError::VoiceNotAvailable("fr_FR".into());
        assert!(e.to_string().contains("fr_FR"));

        let e = PiperTtsError::PerRequestOverrideNotImplemented {
            knob: "length_scale",
            requested: 0.8,
            default: 1.1,
        };
        assert!(e.to_string().contains("length_scale"));
        assert!(e.to_string().contains("not wired in v0.5"));

        let e = PiperTtsError::Service(ServiceError::UnknownModel("gpt".into()));
        assert!(e.to_string().contains("gpt"));
        assert!(std::error::Error::source(&e).is_some());
    }
}

// ---------------------------------------------------------------------------
// T12 tests — `cargo test piper_http::route_tts_http`
// ---------------------------------------------------------------------------

#[cfg(test)]
mod route_tts_http {
    //! Route-level acceptance: the axum surface (JSON extractor, status
    //! mapping, `audio/wav` success body) driven via `tower::ServiceExt`
    //! oneshot — no live listener, no GGUF, mocks only. The live wired-server
    //! path is covered by `tests/piper_http_compat.rs` +
    //! `tests/tts_g2p_injection.rs` (env-gated on real voices).

    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt as _;
    use vokra_core::{SynthesisRequest, SynthesizedAudio, VokraError};

    /// Mock synth: 4 deterministic samples for `piper-plus`, else the same
    /// errors the production registry raises.
    struct MockSynth;
    impl SynthesizeService for MockSynth {
        fn synthesize(
            &self,
            model: &str,
            _request: &SynthesisRequest,
        ) -> Result<SynthesizedAudio, ServiceError> {
            match model {
                model_names::PIPER_PLUS => {
                    Ok(SynthesizedAudio::new(vec![0.0, 0.5, -0.5, 1.0], 22_050))
                }
                other => Err(ServiceError::UnknownModel(other.to_owned())),
            }
        }
    }

    /// Mock synth that raises the exact error the production
    /// `PassthroughPhonemizer` raises for plain text when no real G2P is
    /// configured — the FR-EX-08 explicit-error path this route must
    /// surface as 400, never as fabricated audio.
    struct PassthroughStyleSynth;
    impl SynthesizeService for PassthroughStyleSynth {
        fn synthesize(
            &self,
            _model: &str,
            request: &SynthesisRequest,
        ) -> Result<SynthesizedAudio, ServiceError> {
            Err(ServiceError::Inference(VokraError::InvalidArgument(
                format!(
                    "PassthroughPhonemizer: `{}` is not a phoneme id (use `[[symbol]]` for symbols)",
                    request.text.split_whitespace().next().unwrap_or(""),
                ),
            )))
        }
    }

    /// Single-voice defaults mock accepting the same tags the production
    /// `impl VoiceDefaults for InferenceService` accepts.
    struct MockVoices;
    impl VoiceDefaults for MockVoices {
        fn defaults_for(&self, voice: &str) -> Option<(f32, f32)> {
            (voice == DEFAULT_VOICE || voice == model_names::PIPER_PLUS).then_some((1.1, 0.667))
        }
    }

    fn app(synth: Arc<dyn SynthesizeService>) -> Router {
        router(TtsHttpState {
            synth,
            voices: Arc::new(MockVoices),
        })
    }

    async fn post_tts(app: Router, body: &str) -> (StatusCode, String, Vec<u8>) {
        let req = Request::builder()
            .method("POST")
            .uri(TTS_ROUTE)
            .header("content-type", "application/json")
            .body(Body::from(body.to_owned()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        let status = resp.status();
        let ct = resp
            .headers()
            .get(header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or_default()
            .to_owned();
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap()
            .to_vec();
        (status, ct, bytes)
    }

    #[tokio::test]
    async fn tts_route_returns_wav_from_mock_service() {
        let (status, ct, body) = post_tts(
            app(Arc::new(MockSynth)),
            r#"{"text":"1 2 3","voice":"default"}"#,
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(ct, AUDIO_WAV_CONTENT_TYPE);
        assert_eq!(&body[0..4], b"RIFF", "body must be a RIFF/WAVE container");
        assert_eq!(&body[8..12], b"WAVE");
        // 44-byte header + 4 samples * 2 bytes.
        assert_eq!(body.len(), 44 + 8);
    }

    #[tokio::test]
    async fn tts_route_malformed_json_is_400_invalid_input() {
        let (status, _ct, body) = post_tts(app(Arc::new(MockSynth)), "{not json").await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        let v: serde_json::Value = serde_json::from_slice(&body).expect("error envelope is JSON");
        assert_eq!(v["error"]["type"], "invalid_input");
    }

    #[tokio::test]
    async fn tts_route_unknown_voice_is_404_model_not_found() {
        let (status, _ct, body) = post_tts(
            app(Arc::new(MockSynth)),
            r#"{"text":"1 2","voice":"de_DE-thorsten-medium"}"#,
        )
        .await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["error"]["type"], "model_not_found");
        assert!(
            v["error"]["message"]
                .as_str()
                .unwrap()
                .contains("de_DE-thorsten-medium")
        );
    }

    /// FR-EX-08 anchor: plain text on a server whose registry still holds
    /// the default `PassthroughPhonemizer` (no `--piper-g2p`) must be an
    /// explicit 400 that names the phonemizer — never a 200 with garbage
    /// audio, never a 500 masking a client-fixable input.
    #[tokio::test]
    async fn tts_route_plain_text_without_g2p_is_explicit_400() {
        let (status, _ct, body) = post_tts(
            app(Arc::new(PassthroughStyleSynth)),
            r#"{"text":"Hello world","voice":"default"}"#,
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["error"]["type"], "invalid_input");
        assert!(
            v["error"]["message"]
                .as_str()
                .unwrap()
                .contains("PassthroughPhonemizer"),
            "the explicit error must name the phonemizer so the operator \
             knows to pass --piper-g2p; got {v}"
        );
    }

    /// cc-18: a `language` the loaded voice does not support surfaces as an
    /// explicit 400 naming the supported set (the `InferenceService` piper
    /// arm raises `VokraError::InvalidArgument` BEFORE the engine's silent
    /// detected-language fallback can fire; this pins the HTTP mapping).
    #[tokio::test]
    async fn tts_route_unsupported_language_is_explicit_400() {
        struct RejectLangSynth;
        impl SynthesizeService for RejectLangSynth {
            fn synthesize(
                &self,
                _model: &str,
                request: &SynthesisRequest,
            ) -> Result<SynthesizedAudio, ServiceError> {
                // Mirrors the InferenceService::synthesize language gate
                // (service.rs) byte-for-byte in spirit: unsupported code →
                // InvalidArgument listing the supported inventory.
                match request.language.as_deref() {
                    Some(lang) if lang != "ja" && lang != "en" => Err(ServiceError::Inference(
                        VokraError::InvalidArgument(format!(
                            "synthesize: language `{lang}` is not supported by the loaded \
                                 piper-plus voice (supported: [ja, en])"
                        )),
                    )),
                    _ => Ok(SynthesizedAudio::new(vec![0.0], 22_050)),
                }
            }
        }
        let (status, _ct, body) = post_tts(
            app(Arc::new(RejectLangSynth)),
            r#"{"text":"1 2","voice":"default","language":"fr"}"#,
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["error"]["type"], "invalid_input");
        assert!(
            v["error"]["message"]
                .as_str()
                .unwrap()
                .contains("language `fr` is not supported"),
            "message must name the rejected language, got {v}"
        );
    }

    /// The full PiperTtsError → ServerError status table, pinned.
    #[test]
    fn error_status_mapping_is_pinned() {
        use PiperTtsError as E;
        let cases: Vec<(E, StatusCode, &str)> = vec![
            (
                E::InvalidRequest("`text` must not be empty".into()),
                StatusCode::BAD_REQUEST,
                "invalid_input",
            ),
            (
                E::VoiceNotAvailable("fr_FR".into()),
                StatusCode::NOT_FOUND,
                "model_not_found",
            ),
            (
                E::PerRequestOverrideNotImplemented {
                    knob: "length_scale",
                    requested: 0.8,
                    default: 1.1,
                },
                StatusCode::NOT_IMPLEMENTED,
                "not_implemented",
            ),
            (
                E::Service(ServiceError::UnknownModel("elevenlabs".into())),
                StatusCode::NOT_FOUND,
                "model_not_found",
            ),
            (
                E::Service(ServiceError::SynthesizeUnavailable {
                    model: "kokoro".into(),
                    reason: "M2-07 deferred",
                }),
                StatusCode::NOT_IMPLEMENTED,
                "not_implemented",
            ),
            (
                E::Service(ServiceError::Inference(
                    vokra_core::VokraError::UnsupportedOp("flow on Metal".into()),
                )),
                StatusCode::NOT_IMPLEMENTED,
                "unsupported_op",
            ),
            (
                E::Service(ServiceError::Inference(
                    vokra_core::VokraError::InvalidArgument("PassthroughPhonemizer: ...".into()),
                )),
                StatusCode::BAD_REQUEST,
                "invalid_input",
            ),
            (
                E::Service(ServiceError::Inference(
                    vokra_core::VokraError::NotImplemented("kokoro G2P bridge"),
                )),
                StatusCode::NOT_IMPLEMENTED,
                "not_implemented",
            ),
            (
                E::Service(ServiceError::Inference(vokra_core::VokraError::ModelLoad(
                    "gguf glitch".into(),
                ))),
                StatusCode::INTERNAL_SERVER_ERROR,
                "inference_failed",
            ),
            (
                E::Service(ServiceError::InvalidConfig("bad pair".into())),
                StatusCode::INTERNAL_SERVER_ERROR,
                "inference_failed",
            ),
        ];
        for (err, want_status, want_tag) in cases {
            let desc = err.to_string();
            let mapped = server_error_from_tts(err);
            assert_eq!(mapped.status(), want_status, "status for {desc:?}");
            assert_eq!(mapped.type_tag(), want_tag, "type tag for {desc:?}");
        }
    }

    /// The override-deferral 501 must carry the documented note so clients
    /// learn the omission fix without reading source.
    #[tokio::test]
    async fn tts_route_override_501_carries_deferral_note() {
        let (status, _ct, body) = post_tts(
            app(Arc::new(MockSynth)),
            r#"{"text":"1 2","voice":"default","length_scale":0.5}"#,
        )
        .await;
        assert_eq!(status, StatusCode::NOT_IMPLEMENTED);
        let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(
            v["error"]["message"]
                .as_str()
                .unwrap()
                .contains("not wired in v0.5")
        );
    }
}
