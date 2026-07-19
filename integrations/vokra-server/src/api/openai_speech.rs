//! OpenAI `POST /v1/audio/speech` (TTS) — cc-38, 2026-07-19 M4-residual audit.
//!
//! # Why this exists
//!
//! `integrations/vokra-server/README.md`'s compatibility matrix has declared
//! this endpoint since M2-09 with the horizon "out of scope (v1.0+)". That
//! horizon has passed: the old v1.0 is today's M3/v0.9, the TTS service layer
//! ships (M2-09-T04/T11), the real 8-language G2P is injectable (campaign-2
//! `581758a`), and voice GGUFs exist. The declared surface simply had no
//! route. This module is that route — a README promise being kept, not a new
//! requirement (there is no FR mandating it).
//!
//! # Why a separate module
//!
//! `openai.rs` (transcriptions) is bound to `Arc<dyn TranscribeService>`;
//! this surface needs the TTS state ([`TtsHttpState`] = `SynthesizeService` +
//! [`VoiceDefaults`]). Rather than widen either state, the file follows the
//! `vllm.rs` precedent, where the stateless contract router and the
//! state-bound `models_router` coexist as separate constructors. One surface
//! per file also keeps `openai.rs` (2.6k lines) from growing further.
//!
//! # Deliberate deviations from OpenAI, and why each is honest
//!
//! Vokra is a *compatible* server, not a clone; every gap below is an
//! explicit status code rather than a plausible-looking response.
//!
//! * **`response_format` defaults to `wav`, not OpenAI's `mp3`.** Vokra has
//!   no audio encoder and will not grow one: mp3/opus/aac/flac would each
//!   drag a third-party codec into the dependency graph (and LAME/FFmpeg
//!   bring licence terms this project rejects outright). An explicitly
//!   requested compressed format is [`SpeechError::ResponseFormatUnsupported`]
//!   → **501** naming the supported set. Transcoding is not a TODO here; it
//!   is out of scope by design.
//! * **`pcm` is also 501.** OpenAI's `pcm` is *headerless 24 kHz* mono s16le.
//!   Our voices synthesize at their own native rate, and emitting those
//!   samples raw under the `pcm` label would be undetectably wrong — there is
//!   no header for the client to notice the rate from. Refusing is the only
//!   honest answer short of a real resampler.
//! * **`speed` must be absent or `1.0`.** Per-request speed maps onto piper's
//!   `length_scale`, which the native MB-iSTFT-VITS2 runtime does not wire
//!   per request (see [`PiperTtsError::PerRequestOverrideNotImplemented`] —
//!   the same v0.5 deferral `/api/tts` already reports). Accepting `speed:
//!   2.0` and returning normal-speed audio would be a fabrication, so it is
//!   **501**. Values outside OpenAI's documented `0.25..=4.0` are **400**
//!   (a malformed request, not an unimplemented feature).
//! * **Voices are Vokra's tags, not OpenAI's.** `alloy` / `echo` / `fable` /
//!   `onyx` / `nova` / `shimmer` name six distinct voices this server does
//!   not have. Folding them onto the single loaded voice is precisely the
//!   silent substitution FR-EX-08 exists to prevent, so an unknown voice is
//!   **404** — with the accepted tags listed (see
//!   [`VoiceDefaults::available_voices`]).
//! * **Unknown JSON fields are 400** (`deny_unknown_fields`, the `vllm.rs`
//!   convention). This matters most for `instructions` (gpt-4o-mini-tts voice
//!   steering): accepting it and ignoring it would silently drop the caller's
//!   stylistic intent.
//!
//! # What is NOT a deviation
//!
//! Plain text with no G2P injected keeps the existing explicit **400**: the
//! `PassthroughPhonemizer` raises `InvalidArgument` for non-phoneme-id text
//! and the shared service→HTTP mapper renders it, exactly as on `/api/tts`.
//! Boot with `--piper-g2p` to synthesize natural language.

#![allow(clippy::result_large_err)] // Mirrors PiperTtsError: the error kind is preserved verbatim for the status mapper (FR-EX-08).

use axum::Router;
use axum::extract::{State, rejection::JsonRejection};
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::api::piper_http::{
    PiperTtsError, TtsHttpState, TtsOutcome, TtsRequest, VoiceDefaults, dispatch_tts,
    server_error_from_service,
};
use crate::error::{ServerError, finish_request};
use crate::service::{AUDIO_WAV_CONTENT_TYPE, SynthesizeService};

/// Route path — named so `server.rs` wiring tests and the compat suite key on
/// the same literal.
pub const SPEECH_ROUTE: &str = "/v1/audio/speech";

/// Response formats this server can actually produce. One entry, because the
/// runtime has exactly one encoder (the hand-audited RIFF/WAVE writer in
/// `piper_http::encode_wav_pcm16`) and adding a lossy codec would mean adding
/// a third-party dependency.
pub const SUPPORTED_RESPONSE_FORMATS: &[&str] = &["wav"];

/// Formats OpenAI documents that this server cannot produce. Listed
/// explicitly so they map to **501 "not implemented"** (a real feature we do
/// not have) rather than **400 "malformed"** (a value that is not part of the
/// API at all) — the distinction tells a client whether to fix its request or
/// stop asking.
pub const UNSUPPORTED_RESPONSE_FORMATS: &[&str] = &["mp3", "opus", "aac", "flac", "pcm"];

/// OpenAI's documented `speed` range. Outside this band the request is
/// malformed per the API contract, independent of what Vokra implements.
const SPEED_MIN: f32 = 0.25;
const SPEED_MAX: f32 = 4.0;

/// The only `speed` this runtime can honour (no per-request `length_scale`).
const SPEED_UNSCALED: f32 = 1.0;

/// Tolerance for "the caller sent the default explicitly". Matches the
/// `OVERRIDE_TOLERANCE` rationale in `piper_http`: far below any perceptual
/// difference, far above client JSON round-trip error.
const SPEED_TOLERANCE: f32 = 1e-4;

/// `POST /v1/audio/speech` request body (OpenAI Audio API shape).
///
/// `deny_unknown_fields` — see the module docs: silently dropping a field
/// like `instructions` would discard caller intent.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SpeechRequest {
    /// TTS model id. `tts-1` (OpenAI stock alias), `piper-plus`, or `kokoro`.
    /// Unknown ⇒ 404 from the service layer.
    pub model: String,
    /// Text to synthesize. OpenAI calls this `input` (the transcription
    /// surface calls the analogous field `text`); empty ⇒ 400.
    pub input: String,
    /// Voice tag. **Vokra's** tags (`default`, `piper-plus`), not OpenAI's
    /// stock names — see the module docs. Unknown ⇒ 404 listing the
    /// accepted set.
    pub voice: String,
    /// Output container. Omitted ⇒ `wav`. Note this differs from OpenAI's
    /// `mp3` default: Vokra ships no audio encoder, so `wav` is the only
    /// format it can produce (module docs).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_format: Option<String>,
    /// Playback-rate multiplier, `0.25..=4.0`. Only `1.0` (or omitted) is
    /// honoured; anything else is 501 rather than silently-normal-speed audio.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speed: Option<f32>,
}

/// Errors this surface raises before (or instead of) reaching the engine.
#[derive(Debug)]
pub enum SpeechError {
    /// Malformed field. → 400.
    InvalidRequest(String),
    /// A documented OpenAI format Vokra cannot encode. → 501.
    ResponseFormatUnsupported {
        /// What the caller asked for.
        requested: String,
    },
    /// A `speed` the runtime cannot apply. → 501.
    SpeedNotImplemented {
        /// What the caller asked for.
        requested: f32,
    },
    /// Voice tag did not resolve. → 404, listing [`Self::available`].
    VoiceNotAvailable {
        /// What the caller asked for.
        requested: String,
        /// What this server actually loaded.
        available: Vec<String>,
    },
    /// Anything the shared TTS dispatch raised — preserved verbatim so the
    /// status mapper picks 4xx/5xx off the inner kind (FR-EX-08).
    Tts(PiperTtsError),
}

impl SpeechError {
    /// Stable note embedded in the 501 body for an unsupported
    /// `response_format`. `const` so tests can assert it byte-for-byte.
    pub const FORMAT_DEFERRAL_NOTE: &'static str = "vokra-server synthesizes PCM and serialises it as RIFF/WAVE; it links no audio \
         encoder (mp3/opus/aac/flac would each add a third-party codec dependency) and no \
         resampler for OpenAI's headerless 24 kHz `pcm`. Request response_format=\"wav\", or \
         transcode client-side.";

    /// Stable note embedded in the 501 body for a non-unit `speed`.
    pub const SPEED_DEFERRAL_NOTE: &'static str = "per-request speed maps onto piper's length_scale, which the native MB-iSTFT-VITS2 \
         runtime does not wire per request (same v0.5 deferral as /api/tts). Omit `speed` (or \
         send 1.0) to use the voice's baked default, or bake a variant voice.";
}

impl std::fmt::Display for SpeechError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidRequest(s) => write!(f, "invalid {SPEECH_ROUTE} request: {s}"),
            Self::ResponseFormatUnsupported { requested } => write!(
                f,
                "response_format={requested:?} is not supported (supported: [{}]): {}",
                SUPPORTED_RESPONSE_FORMATS.join(", "),
                Self::FORMAT_DEFERRAL_NOTE
            ),
            Self::SpeedNotImplemented { requested } => write!(
                f,
                "speed={requested} is not implemented (only {SPEED_UNSCALED} is honoured): {}",
                Self::SPEED_DEFERRAL_NOTE
            ),
            Self::VoiceNotAvailable {
                requested,
                available,
            } => write!(
                f,
                "voice `{requested}` is not available on this server (loaded voices: [{}]). \
                 OpenAI's stock voice names (alloy, echo, fable, onyx, nova, shimmer) are not \
                 aliased onto the loaded voice — that would be a silent substitution",
                available.join(", "),
            ),
            Self::Tts(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for SpeechError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Tts(e) => Some(e),
            _ => None,
        }
    }
}

/// Validate the OpenAI-specific knobs, then delegate to the shared
/// [`dispatch_tts`] so both TTS routes run identical synthesis, voice
/// resolution, and WAV encoding.
///
/// Field-name errors are raised HERE rather than inside `dispatch_tts` so the
/// caller sees `input` (OpenAI's name), never `/api/tts`'s `text`.
///
/// # Errors
///
/// See [`SpeechError`].
pub fn dispatch_speech(
    service: &dyn SynthesizeService,
    voices: &dyn VoiceDefaults,
    req: &SpeechRequest,
) -> Result<TtsOutcome, SpeechError> {
    // 1) Field validation, using OpenAI's field names.
    if req.input.is_empty() {
        return Err(SpeechError::InvalidRequest(
            "`input` must not be empty".into(),
        ));
    }
    if req.voice.is_empty() {
        return Err(SpeechError::InvalidRequest(
            "`voice` must not be empty".into(),
        ));
    }
    if req.model.is_empty() {
        return Err(SpeechError::InvalidRequest(
            "`model` must not be empty".into(),
        ));
    }

    // 2) response_format. Omitted = wav (module docs: a documented deviation
    //    from OpenAI's mp3 default, discoverable from the Content-Type).
    if let Some(fmt) = req.response_format.as_deref() {
        let lower = fmt.to_ascii_lowercase();
        if !SUPPORTED_RESPONSE_FORMATS.contains(&lower.as_str()) {
            if UNSUPPORTED_RESPONSE_FORMATS.contains(&lower.as_str()) {
                // A real OpenAI format we cannot encode → 501.
                return Err(SpeechError::ResponseFormatUnsupported {
                    requested: fmt.to_owned(),
                });
            }
            // Not part of the OpenAI enum at all → malformed request → 400.
            return Err(SpeechError::InvalidRequest(format!(
                "`response_format` must be one of the OpenAI values \
                 [{}, {}]; got {fmt:?}",
                SUPPORTED_RESPONSE_FORMATS.join(", "),
                UNSUPPORTED_RESPONSE_FORMATS.join(", "),
            )));
        }
    }

    // 3) speed. Out of OpenAI's documented range = malformed (400); in range
    //    but not 1.0 = a feature we do not have (501). NaN fails the range
    //    check first: `RangeInclusive::contains` is false for NaN, so it is
    //    reported as malformed rather than as an unimplemented speed.
    if let Some(speed) = req.speed {
        if !(SPEED_MIN..=SPEED_MAX).contains(&speed) {
            return Err(SpeechError::InvalidRequest(format!(
                "`speed` must be in {SPEED_MIN}..={SPEED_MAX}; got {speed}"
            )));
        }
        if (speed - SPEED_UNSCALED).abs() > SPEED_TOLERANCE {
            return Err(SpeechError::SpeedNotImplemented { requested: speed });
        }
    }

    // 4) Voice resolution, with an actionable 404 list. `dispatch_tts` would
    //    also reject an unknown voice, but only as a bare
    //    `VoiceNotAvailable(tag)` — resolving here lets the response tell the
    //    caller what this server DOES have.
    if voices.defaults_for(&req.voice).is_none() {
        return Err(SpeechError::VoiceNotAvailable {
            requested: req.voice.clone(),
            available: voices.available_voices(),
        });
    }

    // 5) Delegate. `length_scale` / `noise_scale` stay `None` (step 3 already
    //    rejected any non-unit speed), and `language` is not part of the
    //    OpenAI speech schema.
    let tts_req = TtsRequest {
        text: req.input.clone(),
        voice: req.voice.clone(),
        model: Some(req.model.clone()),
        length_scale: None,
        noise_scale: None,
        language: None,
    };
    dispatch_tts(service, voices, &tts_req).map_err(SpeechError::Tts)
}

/// Build the `/v1/audio/speech` router bound to `state`.
///
/// Mounted by `build_http_app` only when a TTS registry exists — a
/// health-only boot therefore 404s this path, an honest absence rather than
/// an endpoint that cannot synthesize (the same rule `/v1/models` follows).
pub fn speech_router(state: TtsHttpState) -> Router {
    Router::new()
        .route(SPEECH_ROUTE, post(speech_handler))
        .with_state(state)
}

/// `POST /v1/audio/speech` handler.
///
/// Synthesis runs on `tokio::task::spawn_blocking` for the same reason as
/// `/api/tts` (cc-18): the engines are CPU-bound and would otherwise stall
/// every other task on the async worker for the whole synthesis.
async fn speech_handler(
    State(state): State<TtsHttpState>,
    body: Result<axum::Json<SpeechRequest>, JsonRejection>,
) -> Response {
    let start = std::time::Instant::now();
    let (model, result) = match body {
        Ok(axum::Json(req)) => {
            let model_tag = req.model.clone();
            let synth = Arc::clone(&state.synth);
            let voices = Arc::clone(&state.voices);
            let join = tokio::task::spawn_blocking(move || {
                dispatch_speech(synth.as_ref(), voices.as_ref(), &req)
            })
            .await;
            let outcome = match join {
                Ok(res) => res.map(wav_response).map_err(server_error_from_speech),
                Err(join_err) => Err(ServerError::InferenceFailed {
                    detail: format!("speech task failed: {join_err}"),
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
    finish_request("POST", SPEECH_ROUTE, model.as_deref(), start, result)
}

/// Success body: raw WAV bytes with `Content-Type: audio/wav`. OpenAI's SDK
/// streams the body to a file (`response.stream_to_file(...)`), so the
/// Content-Type is how a caller sees which container it actually received.
fn wav_response(outcome: TtsOutcome) -> Response {
    let TtsOutcome::Wav(bytes) = outcome;
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, AUDIO_WAV_CONTENT_TYPE)],
        bytes,
    )
        .into_response()
}

/// Map [`SpeechError`] onto the crate-wide [`ServerError`] envelope.
///
/// Status table (pinned by `route_speech::error_status_mapping_is_pinned`):
///
/// * `InvalidRequest` → 400 `invalid_input`
/// * `ResponseFormatUnsupported` → 501 `not_implemented`
/// * `SpeedNotImplemented` → 501 `not_implemented`
/// * `VoiceNotAvailable` → 404 `model_not_found`
/// * `Tts(_)` → the shared `/api/tts` table (unknown model 404, Kokoro
///   `SynthesizeUnavailable` 501, `UnsupportedOp` 501, `InvalidArgument` 400
///   — which is the plain-text-without-G2P path — everything else 500)
fn server_error_from_speech(err: SpeechError) -> ServerError {
    match err {
        SpeechError::InvalidRequest(msg) => ServerError::InvalidInput {
            detail: format!("{SPEECH_ROUTE}: {msg}"),
        },
        e @ (SpeechError::ResponseFormatUnsupported { .. }
        | SpeechError::SpeedNotImplemented { .. }) => ServerError::NotImplemented {
            detail: e.to_string(),
        },
        e @ SpeechError::VoiceNotAvailable { .. } => ServerError::ModelNotFound {
            model: e.to_string(),
        },
        SpeechError::Tts(PiperTtsError::Service(inner)) => server_error_from_service(inner),
        SpeechError::Tts(PiperTtsError::VoiceNotAvailable(v)) => ServerError::ModelNotFound {
            model: format!("voice `{v}`"),
        },
        SpeechError::Tts(PiperTtsError::InvalidRequest(msg)) => ServerError::InvalidInput {
            detail: format!("{SPEECH_ROUTE}: {msg}"),
        },
        SpeechError::Tts(e @ PiperTtsError::PerRequestOverrideNotImplemented { .. }) => {
            ServerError::NotImplemented {
                detail: e.to_string(),
            }
        }
    }
}

/// The OpenAI stock voice names, used only to explain in tests (and in the
/// README) which vocabulary this server deliberately does NOT alias.
#[cfg(test)]
const OPENAI_STOCK_VOICES: &[&str] = &["alloy", "echo", "fable", "onyx", "nova", "shimmer"];

// ---------------------------------------------------------------------------
// Tests — `cargo test openai_speech::route_speech`
// ---------------------------------------------------------------------------

#[cfg(test)]
mod route_speech {
    //! cc-38 acceptance tests. They drive the schema + dispatch layer with
    //! mocks (no voice GGUF), plus the axum route through `oneshot`. The
    //! real-weight e2e (a wired server actually returning audio) is the
    //! env-gated `tests/real_gguf_slots.rs` leg — cc-40.

    use super::*;
    use crate::service::{ServiceError, model_names};
    use axum::body::to_bytes;
    use axum::http::{Method, Request};
    use tower::ServiceExt as _;
    use vokra_core::{SynthesisRequest, SynthesizedAudio, VokraError};

    const SAMPLE_RATE: u32 = 22_050;

    /// Emits a short deterministic buffer so tests can key on WAV framing
    /// without a real engine.
    struct FakeSynth;
    impl SynthesizeService for FakeSynth {
        fn synthesize(
            &self,
            model: &str,
            request: &SynthesisRequest,
        ) -> Result<SynthesizedAudio, ServiceError> {
            match model {
                model_names::PIPER_PLUS | model_names::TTS_1 => Ok(SynthesizedAudio::new(
                    vec![0.0, 0.5, -0.5, 1.0],
                    SAMPLE_RATE,
                )),
                model_names::KOKORO => Err(ServiceError::SynthesizeUnavailable {
                    model: model.to_owned(),
                    reason: "kokoro TtsEngine::synthesize needs a G2P bridge (M2-07 deferred)",
                }),
                // Stands in for the PassthroughPhonemizer's plain-text
                // rejection so the 400 contract is testable without a G2P.
                "no-g2p" => Err(ServiceError::Inference(VokraError::InvalidArgument(
                    format!("phoneme id parse failed for {:?}", request.text),
                ))),
                other => Err(ServiceError::UnknownModel(other.to_owned())),
            }
        }
    }

    struct FakeVoices;
    impl VoiceDefaults for FakeVoices {
        fn defaults_for(&self, voice: &str) -> Option<(f32, f32)> {
            (voice == "default" || voice == model_names::PIPER_PLUS).then_some((1.0, 0.667))
        }
        fn available_voices(&self) -> Vec<String> {
            vec!["default".to_owned(), model_names::PIPER_PLUS.to_owned()]
        }
    }

    fn req(input: &str, voice: &str) -> SpeechRequest {
        SpeechRequest {
            model: model_names::TTS_1.to_owned(),
            input: input.to_owned(),
            voice: voice.to_owned(),
            response_format: None,
            speed: None,
        }
    }

    fn dispatch(r: &SpeechRequest) -> Result<TtsOutcome, SpeechError> {
        dispatch_speech(&FakeSynth, &FakeVoices, r)
    }

    // ---- happy path ----

    #[test]
    fn minimal_request_returns_wav_bytes() {
        let out = dispatch(&req("hello", "default")).expect("synthesis must succeed");
        let bytes = out.wav_bytes();
        assert_eq!(&bytes[0..4], b"RIFF", "must be a RIFF container");
        assert_eq!(&bytes[8..12], b"WAVE");
        // 4 mono i16 samples + the 44-byte header.
        assert_eq!(bytes.len(), 44 + 4 * 2);
        let sr = u32::from_le_bytes([bytes[24], bytes[25], bytes[26], bytes[27]]);
        assert_eq!(sr, SAMPLE_RATE, "voice's native rate, not a resampled one");
    }

    /// `response_format` omitted or explicitly `"wav"` (any case) both work;
    /// omitted defaulting to wav is the documented OpenAI deviation.
    #[test]
    fn wav_format_is_accepted_omitted_or_explicit() {
        assert!(dispatch(&req("hi", "default")).is_ok());
        for fmt in ["wav", "WAV", "Wav"] {
            let r = SpeechRequest {
                response_format: Some(fmt.to_owned()),
                ..req("hi", "default")
            };
            assert!(dispatch(&r).is_ok(), "response_format={fmt:?} must be OK");
        }
    }

    /// `tts-1` is the OpenAI stock alias; `piper-plus` is the native name.
    /// Both must reach the same engine.
    #[test]
    fn tts_1_alias_and_native_model_name_both_synthesize() {
        for model in [model_names::TTS_1, model_names::PIPER_PLUS] {
            let r = SpeechRequest {
                model: model.to_owned(),
                ..req("hi", "default")
            };
            assert!(dispatch(&r).is_ok(), "model={model} must synthesize");
        }
    }

    // ---- explicit-error contracts ----

    /// RED-LINE: a compressed format is 501 naming the supported set — never
    /// a silent wav body under an mp3 label, never a transcode.
    #[test]
    fn compressed_formats_are_501_naming_the_supported_set() {
        for fmt in UNSUPPORTED_RESPONSE_FORMATS {
            let r = SpeechRequest {
                response_format: Some((*fmt).to_owned()),
                ..req("hi", "default")
            };
            let err = dispatch(&r).expect_err("{fmt} must be rejected");
            assert!(
                matches!(err, SpeechError::ResponseFormatUnsupported { .. }),
                "{fmt}: expected ResponseFormatUnsupported, got {err:?}"
            );
            let msg = err.to_string();
            assert!(
                msg.contains("wav"),
                "{fmt}: the 501 must name the supported set; got {msg}"
            );
            assert_eq!(
                server_error_from_speech(err).status().as_u16(),
                501,
                "{fmt} must map to 501"
            );
        }
    }

    /// A value outside OpenAI's enum entirely is a malformed request (400),
    /// distinct from a real-but-unimplemented format (501).
    #[test]
    fn unknown_format_is_400_not_501() {
        let r = SpeechRequest {
            response_format: Some("ogg".to_owned()),
            ..req("hi", "default")
        };
        let err = dispatch(&r).expect_err("ogg must be rejected");
        assert!(
            matches!(err, SpeechError::InvalidRequest(_)),
            "expected InvalidRequest, got {err:?}"
        );
        assert_eq!(server_error_from_speech(err).status().as_u16(), 400);
    }

    /// speed==1.0 (or omitted) passes; in-range non-unit speed is 501;
    /// out-of-range (and NaN) is 400.
    #[test]
    fn speed_policy_splits_400_and_501() {
        assert!(dispatch(&req("hi", "default")).is_ok(), "omitted speed");
        for ok in [1.0, 1.0 + SPEED_TOLERANCE / 2.0] {
            let r = SpeechRequest {
                speed: Some(ok),
                ..req("hi", "default")
            };
            assert!(dispatch(&r).is_ok(), "speed={ok} must be accepted");
        }
        for unimplemented in [0.25, 0.5, 1.5, 2.0, 4.0] {
            let r = SpeechRequest {
                speed: Some(unimplemented),
                ..req("hi", "default")
            };
            let err = dispatch(&r).expect_err("non-unit speed must be rejected");
            assert!(
                matches!(err, SpeechError::SpeedNotImplemented { .. }),
                "speed={unimplemented}: expected SpeedNotImplemented, got {err:?}"
            );
            assert_eq!(server_error_from_speech(err).status().as_u16(), 501);
        }
        for malformed in [0.0, 0.1, 4.5, 100.0, f32::NAN, f32::INFINITY] {
            let r = SpeechRequest {
                speed: Some(malformed),
                ..req("hi", "default")
            };
            let err = dispatch(&r).expect_err("out-of-range speed must be rejected");
            assert!(
                matches!(err, SpeechError::InvalidRequest(_)),
                "speed={malformed}: expected InvalidRequest, got {err:?}"
            );
            assert_eq!(server_error_from_speech(err).status().as_u16(), 400);
        }
    }

    /// RED-LINE: OpenAI's stock voices are NOT aliased onto the loaded voice.
    /// Each is a 404 that names what the server actually has.
    #[test]
    fn openai_stock_voices_are_404_never_substituted() {
        for voice in OPENAI_STOCK_VOICES {
            let err = dispatch(&req("hi", voice)).expect_err("{voice} must not resolve");
            match &err {
                SpeechError::VoiceNotAvailable {
                    requested,
                    available,
                } => {
                    assert_eq!(requested, voice);
                    assert!(
                        available.contains(&"default".to_owned()),
                        "the 404 must list the loaded voices; got {available:?}"
                    );
                }
                other => panic!("{voice}: expected VoiceNotAvailable, got {other:?}"),
            }
            assert_eq!(server_error_from_speech(err).status().as_u16(), 404);
        }
    }

    #[test]
    fn empty_fields_are_400_with_openai_field_names() {
        // `input`, not `/api/tts`'s `text`.
        let err = dispatch(&req("", "default")).expect_err("empty input");
        assert!(err.to_string().contains("`input`"), "got {err}");
        assert_eq!(server_error_from_speech(err).status().as_u16(), 400);

        let err = dispatch(&req("hi", "")).expect_err("empty voice");
        assert!(err.to_string().contains("`voice`"), "got {err}");

        let r = SpeechRequest {
            model: String::new(),
            ..req("hi", "default")
        };
        let err = dispatch(&r).expect_err("empty model");
        assert!(err.to_string().contains("`model`"), "got {err}");
    }

    /// Kokoro is advertised but its synthesize is deferred → 501, the same
    /// contract `/api/tts` has (cc-40 verifies this against the real GGUF).
    #[test]
    fn kokoro_is_501_not_a_fabricated_wav() {
        let r = SpeechRequest {
            model: model_names::KOKORO.to_owned(),
            ..req("hi", "default")
        };
        let err = dispatch(&r).expect_err("kokoro synthesis is unavailable");
        assert_eq!(server_error_from_speech(err).status().as_u16(), 501);
    }

    #[test]
    fn unknown_model_is_404() {
        let r = SpeechRequest {
            model: "tts-1-hd".to_owned(),
            ..req("hi", "default")
        };
        let err = dispatch(&r).expect_err("tts-1-hd is deliberately not aliased");
        assert_eq!(
            server_error_from_speech(err).status().as_u16(),
            404,
            "tts-1-hd names a quality tier this server does not have"
        );
    }

    /// The plain-text-without-G2P contract: the phonemizer's
    /// `InvalidArgument` must surface as an honest 400 naming the problem,
    /// not as garbage audio (FR-EX-08).
    #[test]
    fn plain_text_without_g2p_stays_an_explicit_400() {
        let r = SpeechRequest {
            model: "no-g2p".to_owned(),
            ..req("plain english sentence", "default")
        };
        let err = dispatch(&r).expect_err("passthrough phonemizer must reject plain text");
        let mapped = server_error_from_speech(err);
        assert_eq!(mapped.status().as_u16(), 400);
        assert!(
            mapped.to_string().contains("phoneme"),
            "the 400 must explain the phoneme parse failure; got {mapped}"
        );
    }

    // ---- axum route ----

    fn app() -> Router {
        speech_router(TtsHttpState {
            synth: Arc::new(FakeSynth),
            voices: Arc::new(FakeVoices),
        })
    }

    async fn post(body: &str) -> (u16, Vec<u8>, String) {
        let request = Request::builder()
            .method(Method::POST)
            .uri(SPEECH_ROUTE)
            .header("content-type", "application/json")
            .body(axum::body::Body::from(body.to_owned()))
            .unwrap();
        let resp = app().oneshot(request).await.unwrap();
        let status = resp.status().as_u16();
        let ct = resp
            .headers()
            .get(header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or_default()
            .to_owned();
        let body = to_bytes(resp.into_body(), 1 << 20).await.unwrap().to_vec();
        (status, body, ct)
    }

    #[tokio::test]
    async fn route_returns_audio_wav_on_success() {
        let (status, body, ct) =
            post(r#"{"model":"tts-1","input":"hello","voice":"default"}"#).await;
        assert_eq!(status, 200, "body: {}", String::from_utf8_lossy(&body));
        assert_eq!(ct, AUDIO_WAV_CONTENT_TYPE);
        assert_eq!(&body[0..4], b"RIFF");
    }

    #[tokio::test]
    async fn route_status_codes_match_the_dispatch_table() {
        for (body, want) in [
            (r#"{"model":"tts-1","input":"hi","voice":"alloy"}"#, 404),
            (
                r#"{"model":"tts-1","input":"hi","voice":"default","response_format":"mp3"}"#,
                501,
            ),
            (
                r#"{"model":"tts-1","input":"hi","voice":"default","speed":2.0}"#,
                501,
            ),
            (r#"{"model":"tts-1","input":"","voice":"default"}"#, 400),
            // Unknown field (e.g. gpt-4o-mini-tts `instructions`) → 400
            // rather than silently dropping the caller's intent.
            (
                r#"{"model":"tts-1","input":"hi","voice":"default","instructions":"cheerful"}"#,
                400,
            ),
            // Malformed JSON → 400 through the same JsonRejection funnel the
            // vLLM routes use (axum's default body never leaks).
            ("not json", 400),
        ] {
            let (status, resp, _ct) = post(body).await;
            assert_eq!(
                status,
                want,
                "request {body} must be {want}; got {status}, body: {}",
                String::from_utf8_lossy(&resp)
            );
        }
    }

    /// Every error body is the OpenAI-shaped envelope, never a bare string —
    /// SDK clients parse `error.type`.
    #[tokio::test]
    async fn error_bodies_are_openai_shaped() {
        let (status, body, _ct) = post(r#"{"model":"tts-1","input":"hi","voice":"nope"}"#).await;
        assert_eq!(status, 404);
        let v: serde_json::Value = serde_json::from_slice(&body).expect("error body must be JSON");
        assert!(v.get("error").is_some(), "got {v}");
        assert_eq!(v["error"]["type"], "model_not_found");
    }

    /// The status table is the contract; pin it so a refactor of
    /// `server_error_from_speech` cannot quietly re-map a variant.
    #[test]
    fn error_status_mapping_is_pinned() {
        assert_eq!(
            server_error_from_speech(SpeechError::InvalidRequest("x".into()))
                .status()
                .as_u16(),
            400
        );
        assert_eq!(
            server_error_from_speech(SpeechError::ResponseFormatUnsupported {
                requested: "mp3".into()
            })
            .status()
            .as_u16(),
            501
        );
        assert_eq!(
            server_error_from_speech(SpeechError::SpeedNotImplemented { requested: 2.0 })
                .status()
                .as_u16(),
            501
        );
        assert_eq!(
            server_error_from_speech(SpeechError::VoiceNotAvailable {
                requested: "alloy".into(),
                available: vec!["default".into()],
            })
            .status()
            .as_u16(),
            404
        );
        assert_eq!(
            server_error_from_speech(SpeechError::Tts(PiperTtsError::Service(
                ServiceError::UnknownModel("zzz".into())
            )))
            .status()
            .as_u16(),
            404
        );
        assert_eq!(
            server_error_from_speech(SpeechError::Tts(PiperTtsError::Service(
                ServiceError::Inference(VokraError::UnsupportedOp("no metal kernel".into()))
            )))
            .status()
            .as_u16(),
            501,
            "a backend hole must surface as 501, never be rewritten to CPU"
        );
    }

    /// Request DTO round-trips so a client can build the same struct.
    #[test]
    fn request_dto_round_trips() {
        let r = SpeechRequest {
            model: "tts-1".into(),
            input: "hello".into(),
            voice: "default".into(),
            response_format: Some("wav".into()),
            speed: Some(1.0),
        };
        let json = serde_json::to_string(&r).unwrap();
        assert_eq!(serde_json::from_str::<SpeechRequest>(&json).unwrap(), r);
        // Optional fields are omitted when unset (no `"speed":null` noise).
        let minimal = SpeechRequest {
            response_format: None,
            speed: None,
            ..r
        };
        let json = serde_json::to_string(&minimal).unwrap();
        assert!(!json.contains("speed"), "got {json}");
        assert!(!json.contains("response_format"), "got {json}");
    }
}
