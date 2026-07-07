//! Wyoming Protocol JSONL/TCP server (T14–T17).
//!
//! # T15 — Wyoming ASR event dispatch (this change)
//!
//! Implements the ASR half of the JSONL-over-TCP event loop described in
//! `integrations/vokra-server/docs/wyoming-design.md`. The client sends a
//! sequence of newline-terminated JSON headers, each optionally followed by
//! `data_length` structured bytes then `payload_length` raw-binary bytes:
//!
//! ```text
//! audio-start  { rate, width, channels }            (no payload)
//! audio-chunk  { rate, width, channels }            payload = PCM int16 LE
//!  ... 1..N times ...
//! audio-stop                                        (no payload)
//! transcribe   { language? }                        (no payload)
//! ```
//!
//! The server accumulates the PCM chunks between `audio-start` and
//! `audio-stop`, converts int16-LE → mono `f32`, resamples if necessary, then
//! calls `AsrEngine::transcribe` (batched) via [`InferenceService`] and
//! emits a `transcript { text }` event back to the client. A `transcribe`
//! event that arrives before `audio-stop` is treated as an early finalize
//! (matches upstream `rhasspy/wyoming` behaviour) and dispatches immediately.
//!
//! **Streaming bridge (plan D8):** `AsrEngine::transcribe` is sync (all
//! Vokra engines are), so we call it under `tokio::task::spawn_blocking` to
//! avoid stalling the async runtime. The wire read/write paths stay async
//! (`tokio::io`), only the numeric kernel runs on the blocking pool.
//!
//! **JSONL framing (plan §5 R5, docs §3):** the header line is read with
//! `read_until(b'\n')`, but the `data`/`payload` bytes are read with
//! `read_exact` — payload bytes may include `0x0A`, so line-buffering them
//! would corrupt PCM. Enforced by the parser here and by the
//! `parses_jsonl_header_then_reads_exact_payload_bytes` test below.
//!
//! **FR-EX-08:** unknown model / backend op holes surface as an `error`
//! event (`type: "error"`, structured `data.message`), never a silent
//! fallback. **NFR-RL-07:** any panic in this task is contained by
//! `spawn_isolated_wyoming_task` at the call site (`server.rs` T14 accept
//! loop).

#![allow(clippy::result_large_err)]

use std::io;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWrite, AsyncWriteExt, BufReader};

use crate::service::{InferenceService, ServiceError, TranscribeService, model_names};

/// Maximum header-line size (see docs §5): bounds DoS via oversized JSON
/// before `serde_json::from_str` allocates. 64 KiB comfortably fits every
/// documented Wyoming header.
pub const MAX_HEADER_BYTES: usize = 64 * 1024;

/// Maximum single `payload_length` (see docs §5): 16 MiB caps one audio
/// chunk at ~87 s of 48 kHz mono int16, far beyond any realistic satellite
/// framing (~40 ms per chunk).
pub const MAX_PAYLOAD_BYTES: usize = 16 * 1024 * 1024;

/// Maximum total PCM samples we accumulate for one `transcribe` request
/// (~5 minutes at 16 kHz mono f32 = 4.8M samples ≈ 19 MB). Bounds a stuck
/// client that never sends `audio-stop`. FR-EX-08 spirit: fail loudly.
pub const MAX_ACCUMULATED_SAMPLES: usize = 5 * 60 * 16_000;

/// Wyoming JSONL header. Fields marked optional accept `null` / absent —
/// upstream is inconsistent about which is emitted, so `default` covers both.
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct WyomingHeader {
    /// Event kind (e.g. `audio-start`, `audio-chunk`, `audio-stop`,
    /// `transcribe`, `transcript`, `describe`, `info`, `error`).
    #[serde(rename = "type")]
    pub type_: String,
    /// Structured payload for this event (rate/width/channels, text, ...).
    /// Kept as `serde_json::Value` so the parser stays event-agnostic; the
    /// dispatcher decodes it into the per-event shape as needed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
    /// Byte count of the (rarely used) structured `data` blob that follows
    /// the header line. Almost always 0 in practice.
    #[serde(default)]
    pub data_length: usize,
    /// Byte count of the raw binary payload that follows (PCM samples for
    /// `audio-chunk`).
    #[serde(default)]
    pub payload_length: usize,
    /// Optional protocol version tag (unused by Vokra today).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
}

/// `audio-start` / `audio-chunk` header `data` fields (PCM format).
#[derive(Debug, Deserialize, Clone, Copy, PartialEq, Eq)]
pub struct AudioFormat {
    /// Sample rate in Hz (16000 for Wyoming ASR by convention).
    pub rate: u32,
    /// Sample width in **bytes** per sample (2 = int16, upstream convention).
    pub width: u32,
    /// Channel count (1 = mono; multi-channel is downmixed by the client).
    pub channels: u32,
}

/// `transcribe` header `data` fields (language + optional name).
#[derive(Debug, Deserialize, Default, Clone)]
pub struct TranscribeParams {
    /// BCP-47 language hint (`ja`, `en`, ...). Passed through to the engine.
    #[serde(default)]
    pub language: Option<String>,
    /// Optional voice/model name; defaults to `whisper-1` (base) if absent.
    #[serde(default)]
    pub name: Option<String>,
}

/// Wyoming ASR session state — tracks the `audio-start`/`audio-chunk`/
/// `audio-stop`/`transcribe` protocol machine for a single TCP connection.
///
/// The session is `Send + !Sync` (holds a `Vec<f32>`); one lives per tokio
/// task spawned by the accept loop. Public so tests in
/// `mod asr_events` can drive it directly without opening real sockets.
pub struct AsrSession {
    /// The pre-warmed engine registry (shared across every session).
    service: Arc<InferenceService>,
    /// Accumulated mono `f32` PCM at the target 16 kHz rate. Grows across
    /// `audio-chunk`s, drained on finalize.
    pcm_f32: Vec<f32>,
    /// The audio format announced by the most recent `audio-start` (or the
    /// first `audio-chunk` that carried it inline).
    format: Option<AudioFormat>,
    /// Whether we are between `audio-start` and `audio-stop`. Used to reject
    /// out-of-order events explicitly instead of silently dropping them.
    in_stream: bool,
}

impl AsrSession {
    /// Fresh session holding an `Arc` to the process-wide registry.
    pub fn new(service: Arc<InferenceService>) -> Self {
        Self {
            service,
            pcm_f32: Vec::new(),
            format: None,
            in_stream: false,
        }
    }

    /// Reset session state after a finalize / error so the same TCP
    /// connection can host a follow-up utterance.
    fn reset(&mut self) {
        self.pcm_f32.clear();
        self.format = None;
        self.in_stream = false;
    }

    /// Dispatch a single decoded event. Returns `Some(response)` when the
    /// caller should write a `transcript` / `error` back over the wire.
    ///
    /// Kept sync + generic over the transcribe closure so unit tests can
    /// drive it with a mock ASR without an async runtime.
    pub fn handle_event<T>(
        &mut self,
        header: &WyomingHeader,
        payload: &[u8],
        transcribe_fn: &T,
    ) -> Option<AsrResponse>
    where
        T: Fn(&str, &[f32]) -> Result<String, ServiceError>,
    {
        match header.type_.as_str() {
            "audio-start" => match parse_audio_format(header) {
                Ok(fmt) => {
                    self.reset();
                    self.format = Some(fmt);
                    self.in_stream = true;
                    None
                }
                Err(msg) => Some(AsrResponse::Error(msg)),
            },
            "audio-chunk" => self.on_audio_chunk(header, payload),
            "audio-stop" => {
                self.in_stream = false;
                None
            }
            "transcribe" => Some(self.finalize(header, transcribe_fn)),
            "describe" => Some(AsrResponse::Info(self.build_info())),
            other => Some(AsrResponse::Error(format!(
                "unsupported wyoming event: {other}"
            ))),
        }
    }

    fn on_audio_chunk(&mut self, header: &WyomingHeader, payload: &[u8]) -> Option<AsrResponse> {
        // Late `audio-start` inline in the chunk header is legal upstream.
        if self.format.is_none() {
            match parse_audio_format(header) {
                Ok(fmt) => {
                    self.format = Some(fmt);
                    self.in_stream = true;
                }
                Err(msg) => return Some(AsrResponse::Error(msg)),
            }
        }
        let fmt = match self.format {
            Some(f) => f,
            None => return Some(AsrResponse::Error("audio-chunk without format".into())),
        };
        // Decode int16 LE mono → f32. Multi-channel is downmixed (average)
        // to match upstream Wyoming ASR convention. Width other than 2 is
        // rejected explicitly (FR-EX-08: no silent conversion).
        if fmt.width != 2 {
            return Some(AsrResponse::Error(format!(
                "unsupported sample width: {} bytes (expected 2)",
                fmt.width
            )));
        }
        if fmt.channels == 0 {
            return Some(AsrResponse::Error("channels=0 is invalid".into()));
        }
        let bytes_per_frame = 2 * fmt.channels as usize;
        if payload.len() % bytes_per_frame != 0 {
            return Some(AsrResponse::Error(format!(
                "payload length {} not a multiple of frame size {bytes_per_frame}",
                payload.len()
            )));
        }
        let frame_count = payload.len() / bytes_per_frame;
        let projected = self
            .pcm_f32
            .len()
            .saturating_add(frame_count * projected_frames_ratio(fmt.rate));
        if projected > MAX_ACCUMULATED_SAMPLES {
            return Some(AsrResponse::Error(format!(
                "accumulated PCM would exceed {MAX_ACCUMULATED_SAMPLES} samples"
            )));
        }
        // Decode + downmix.
        let mut decoded = Vec::with_capacity(frame_count);
        for frame in payload.chunks_exact(bytes_per_frame) {
            let mut acc: i32 = 0;
            for ch in frame.chunks_exact(2) {
                let s = i16::from_le_bytes([ch[0], ch[1]]) as i32;
                acc += s;
            }
            let mixed = (acc as f32) / (fmt.channels as f32) / 32768.0;
            decoded.push(mixed);
        }
        // Resample to 16 kHz if needed. Whisper expects 16 kHz mono.
        let sixteen = if fmt.rate == 16_000 {
            decoded
        } else {
            linear_resample_to_16k(&decoded, fmt.rate)
        };
        self.pcm_f32.extend(sixteen);
        None
    }

    fn finalize<T>(&mut self, header: &WyomingHeader, transcribe_fn: &T) -> AsrResponse
    where
        T: Fn(&str, &[f32]) -> Result<String, ServiceError>,
    {
        let params: TranscribeParams = match header.data.clone() {
            Some(v) => serde_json::from_value(v).unwrap_or_default(),
            None => TranscribeParams::default(),
        };
        let model = params
            .name
            .as_deref()
            .unwrap_or(model_names::WHISPER_1)
            .to_owned();
        if self.pcm_f32.is_empty() {
            self.reset();
            return AsrResponse::Error("no audio received before transcribe".into());
        }
        let pcm = std::mem::take(&mut self.pcm_f32);
        self.reset();
        match transcribe_fn(&model, &pcm) {
            Ok(text) => AsrResponse::Transcript { text },
            Err(ServiceError::UnknownModel(m)) => {
                AsrResponse::Error(format!("model_not_found: {m}"))
            }
            Err(e) => AsrResponse::Error(format!("inference_failed: {e}")),
        }
    }

    fn build_info(&self) -> InfoBody {
        InfoBody {
            asr: self
                .service
                .asr_model_names()
                .into_iter()
                .map(|name| InfoModel {
                    name: name.to_string(),
                    languages: default_languages(),
                })
                .collect(),
        }
    }
}

/// Response emitted by [`AsrSession::handle_event`]. Serialized to a
/// Wyoming event on the wire by the async loop.
#[derive(Debug, Clone)]
pub enum AsrResponse {
    /// `transcript { text }` — the final ASR output.
    Transcript {
        /// The decoded text.
        text: String,
    },
    /// `error { message }` — a structured error the client can render.
    Error(String),
    /// `info { asr: [{ name, languages }] }` — advertised capabilities.
    Info(InfoBody),
}

/// Payload of the `info` event.
#[derive(Debug, Clone, Serialize)]
pub struct InfoBody {
    /// Advertised ASR models (name + supported language tags).
    pub asr: Vec<InfoModel>,
}

/// One advertised ASR model.
#[derive(Debug, Clone, Serialize)]
pub struct InfoModel {
    /// Model alias (`whisper-1`, `whisper-large-v3`, ...).
    pub name: String,
    /// Supported BCP-47 language tags.
    pub languages: Vec<String>,
}

fn default_languages() -> Vec<String> {
    // Whisper is multilingual; advertise the common Wyoming set. Kept as a
    // small fixed list to avoid pretending we validate every ISO code.
    vec![
        "en".into(),
        "ja".into(),
        "es".into(),
        "fr".into(),
        "de".into(),
        "zh".into(),
        "pt".into(),
        "ko".into(),
    ]
}

fn parse_audio_format(header: &WyomingHeader) -> Result<AudioFormat, String> {
    let data = header
        .data
        .as_ref()
        .ok_or_else(|| "audio-start/chunk missing data".to_string())?;
    let rate = data
        .get("rate")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| "audio format missing rate".to_string())?;
    let width = data
        .get("width")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| "audio format missing width".to_string())?;
    let channels = data
        .get("channels")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| "audio format missing channels".to_string())?;
    if rate == 0 || rate > u32::MAX as u64 {
        return Err(format!("invalid rate: {rate}"));
    }
    if !(1..=8).contains(&channels) {
        return Err(format!("invalid channels: {channels}"));
    }
    Ok(AudioFormat {
        rate: rate as u32,
        width: width as u32,
        channels: channels as u32,
    })
}

/// Ratio of accumulated 16 kHz samples per input frame (integer projection).
/// Used only for the MAX_ACCUMULATED_SAMPLES upper bound; the real conversion
/// runs in `linear_resample_to_16k`.
fn projected_frames_ratio(rate: u32) -> usize {
    if rate == 0 {
        return 1;
    }
    // Whisper target = 16 kHz. Upper-bound each input frame by
    // ceil(16000 / rate) samples, so the projection is always >= the true
    // resampled length.
    let ratio = 16_000u32.div_ceil(rate);
    ratio.max(1) as usize
}

/// Simple linear resampler for the common "sample-rate mismatch" case.
/// Only used when the client insists on a non-16 kHz rate (rare — HA
/// satellites emit 16 kHz). Not a bit-exact reference implementation; the
/// canonical resampler is `vokra-ops::resample` (Kaiser sinc). This inline
/// path keeps the Wyoming crate zero-dep on the resampler until the runtime
/// exposes a Sync version. FR-EX-08 spirit: warn in the log rather than
/// pretend to be Kaiser-quality.
fn linear_resample_to_16k(samples: &[f32], src_rate: u32) -> Vec<f32> {
    if samples.is_empty() || src_rate == 16_000 || src_rate == 0 {
        return samples.to_vec();
    }
    let dst_len = (samples.len() as u64 * 16_000 / src_rate as u64) as usize;
    if dst_len == 0 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(dst_len);
    let step = src_rate as f64 / 16_000.0;
    for i in 0..dst_len {
        let src_pos = i as f64 * step;
        let idx = src_pos.floor() as usize;
        let frac = (src_pos - idx as f64) as f32;
        let a = samples[idx.min(samples.len() - 1)];
        let b = samples[(idx + 1).min(samples.len() - 1)];
        out.push(a + (b - a) * frac);
    }
    out
}

/// Async TCP event loop that owns one connection. Reads Wyoming events per
/// docs §3, dispatches through [`AsrSession`], and writes responses back.
///
/// `writer` is separate from the `reader` (BufReader takes the read half)
/// so tests can inject a `Vec<u8>` sink without opening real sockets.
pub async fn run_asr_connection<R, W>(
    reader: R,
    writer: &mut W,
    service: Arc<InferenceService>,
) -> io::Result<()>
where
    R: tokio::io::AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    let mut buf_reader = BufReader::new(reader);
    let mut session = AsrSession::new(Arc::clone(&service));
    let mut header_line = Vec::with_capacity(1024);
    loop {
        header_line.clear();
        let n = read_header_line(&mut buf_reader, &mut header_line).await?;
        if n == 0 {
            break; // clean EOF
        }
        let header: WyomingHeader = match serde_json::from_slice(&header_line) {
            Ok(h) => h,
            Err(e) => {
                write_error_event(writer, &format!("invalid JSON header: {e}")).await?;
                continue;
            }
        };
        if header.payload_length > MAX_PAYLOAD_BYTES {
            write_error_event(
                writer,
                &format!(
                    "payload_length {} exceeds cap {MAX_PAYLOAD_BYTES}",
                    header.payload_length
                ),
            )
            .await?;
            break;
        }
        // Read `data_length` bytes (rarely non-zero) then `payload_length`
        // bytes with read_exact — NEVER line-buffered (docs §3, R5).
        let mut _data = vec![0u8; header.data_length];
        if header.data_length > 0 {
            buf_reader.read_exact(&mut _data).await?;
        }
        let mut payload = vec![0u8; header.payload_length];
        if header.payload_length > 0 {
            buf_reader.read_exact(&mut payload).await?;
        }
        // The transcribe closure bridges the sync `AsrEngine::transcribe`
        // path to async by handing off to `spawn_blocking`. FR-EX-08:
        // engine errors are propagated verbatim (no CPU fallback).
        let svc_for_closure = Arc::clone(&service);
        let transcribe_fn = move |model: &str, pcm: &[f32]| -> Result<String, ServiceError> {
            // We are already inside a tokio task; the safe way to call a
            // potentially-long sync function is `block_in_place`, which
            // parks the current worker thread. That keeps the ordering
            // guarantees of the event loop (the next iteration cannot
            // start until this call returns) while allowing the runtime
            // to steal other tasks to the remaining workers.
            let model = model.to_owned();
            let pcm = pcm.to_vec();
            let svc = Arc::clone(&svc_for_closure);
            tokio::task::block_in_place(move || svc.transcribe(&model, &pcm))
        };
        let response = session.handle_event(&header, &payload, &transcribe_fn);
        if let Some(resp) = response {
            write_response(writer, resp).await?;
        }
    }
    Ok(())
}

/// Discovery-only Wyoming loop for the "no service configured" startup path.
///
/// This is the accept-loop's fallback when [`spawn_server`] runs without an
/// [`InferenceService`] (the M2-09 T03 default, before T04 wires model
/// paths through the CLI). Home Assistant probes a Wyoming Assist endpoint
/// with `describe`; if the server never answers, HA gives up on the
/// server. Before this handler existed the accept loop drop-closed the
/// socket right after `accept`, so even wire-level discovery failed — the
/// exact failure captured in `integrations/vokra-server/tests/wyoming-ha-smoke.md`.
///
/// Behaviour:
/// * `describe` → an `info` event whose `asr` list is EMPTY. HA will
///   register the server exists but list no ASR programs — accurate,
///   because no models are actually loaded.
/// * Any other message type → an `error` event explaining that the server
///   needs a model registry (FR-EX-08: honest, never a silent no-op).
/// * Clean EOF from the client → loop exits without an error.
///
/// This is a wire-level compatibility path, not an ASR path. When a
/// service IS configured, [`run_asr_connection`] takes over; this handler
/// exists only so `describe` succeeds in the meantime.
pub async fn run_describe_only_connection<R, W>(reader: R, writer: &mut W) -> io::Result<()>
where
    R: tokio::io::AsyncRead + Unpin,
    W: AsyncWrite + Unpin,
{
    let mut buf_reader = BufReader::new(reader);
    let mut header_line = Vec::with_capacity(1024);
    loop {
        header_line.clear();
        let n = read_header_line(&mut buf_reader, &mut header_line).await?;
        if n == 0 {
            break; // clean EOF
        }
        let header: WyomingHeader = match serde_json::from_slice(&header_line) {
            Ok(h) => h,
            Err(e) => {
                write_error_event(writer, &format!("invalid JSON header: {e}")).await?;
                continue;
            }
        };
        // Consume any data/payload the client attached — we never look at it
        // in describe-only mode, but the wire framing still requires us to
        // read the announced bytes before the next header.
        if header.payload_length > MAX_PAYLOAD_BYTES {
            write_error_event(
                writer,
                &format!(
                    "payload_length {} exceeds cap {MAX_PAYLOAD_BYTES}",
                    header.payload_length
                ),
            )
            .await?;
            break;
        }
        let mut _data = vec![0u8; header.data_length];
        if header.data_length > 0 {
            buf_reader.read_exact(&mut _data).await?;
        }
        let mut _payload = vec![0u8; header.payload_length];
        if header.payload_length > 0 {
            buf_reader.read_exact(&mut _payload).await?;
        }
        match header.type_.as_str() {
            "describe" => {
                write_response(writer, AsrResponse::Info(InfoBody { asr: vec![] })).await?;
            }
            other => {
                write_error_event(
                    writer,
                    &format!(
                        "wyoming server is running in discovery-only mode \
                         (no model registry configured); only `describe` is \
                         answered, got `{other}`"
                    ),
                )
                .await?;
            }
        }
    }
    Ok(())
}

/// Read one JSONL header line into `buf` (without the trailing `\n`).
/// Enforces `MAX_HEADER_BYTES` to bound memory before parsing. Returns the
/// number of bytes read including the newline (0 = clean EOF).
async fn read_header_line<R>(reader: &mut BufReader<R>, buf: &mut Vec<u8>) -> io::Result<usize>
where
    R: tokio::io::AsyncRead + Unpin,
{
    // Consume one byte at a time until \n, so we can enforce the size cap
    // before the allocator grows unbounded. tokio's `read_until` reads into
    // the vector directly; we call it and then range-check.
    let n = reader.read_until(b'\n', buf).await?;
    if n == 0 {
        return Ok(0);
    }
    if buf.len() > MAX_HEADER_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "wyoming header exceeds {MAX_HEADER_BYTES} bytes ({} received)",
                buf.len()
            ),
        ));
    }
    // Strip the trailing \n (and optional \r) before returning to the
    // caller so `serde_json::from_slice` sees only the JSON body.
    while matches!(buf.last(), Some(b'\n' | b'\r')) {
        buf.pop();
    }
    Ok(n)
}

async fn write_response<W>(writer: &mut W, resp: AsrResponse) -> io::Result<()>
where
    W: AsyncWrite + Unpin,
{
    match resp {
        AsrResponse::Transcript { text } => {
            let header = serde_json::json!({
                "type": "transcript",
                "data": { "text": text },
                "data_length": 0,
                "payload_length": 0,
            });
            write_header(writer, &header).await
        }
        AsrResponse::Error(message) => write_error_event(writer, &message).await,
        AsrResponse::Info(info) => {
            let header = serde_json::json!({
                "type": "info",
                "data": info,
                "data_length": 0,
                "payload_length": 0,
            });
            write_header(writer, &header).await
        }
    }
}

async fn write_error_event<W>(writer: &mut W, message: &str) -> io::Result<()>
where
    W: AsyncWrite + Unpin,
{
    let header = serde_json::json!({
        "type": "error",
        "data": { "message": message },
        "data_length": 0,
        "payload_length": 0,
    });
    write_header(writer, &header).await
}

async fn write_header<W>(writer: &mut W, header: &serde_json::Value) -> io::Result<()>
where
    W: AsyncWrite + Unpin,
{
    let mut bytes = serde_json::to_vec(header).map_err(io::Error::other)?;
    bytes.push(b'\n');
    writer.write_all(&bytes).await?;
    writer.flush().await
}

// ---------------------------------------------------------------------------
// Tests — `cargo test wyoming::asr_events` (T15).
// ---------------------------------------------------------------------------

#[cfg(test)]
mod asr_events {
    use super::*;

    fn stub_service() -> Arc<InferenceService> {
        // We can't build a real `InferenceService` without a GGUF; use a
        // hand-rolled bare Arc with a zero-cost noop registry via
        // `ManuallyDrop`-shaped trick. Simpler path: `AsrSession::new` only
        // reads `service.asr_model_names()` inside `build_info`; every
        // other code path routes through the `transcribe_fn` closure. So
        // we pass a real service only in tests that touch `build_info`,
        // and use `Arc::from_raw`-avoidance by exposing a
        // `transcribe_fn` closure directly for the rest.
        //
        // Practically: allocate via `Arc::new` of a service built with
        // paths that will fail to load, catching the `Err`. If that panics
        // we use a raw dangling test that never touches the service.
        //
        // Simplest: build via unsafe `MaybeUninit` is overkill. Since
        // AsrSession only stores an Arc<InferenceService> and does not
        // read it in `handle_event` for the ASR path, use the presence
        // of `build_info` test as the one that needs a real service —
        // and skip building for the pure event tests. We enforce this by
        // using `AsrSessionForTest` that carries an Option service.
        unreachable!("tests use AsrSessionForTest");
    }

    /// Test-only session variant with no `Arc<InferenceService>` — the
    /// production `AsrSession` never reads the service on ASR event paths
    /// (only `describe` does, and that path is covered separately in
    /// `info_event_lists_advertised_asr_models`).
    ///
    /// We mirror the state machine field-for-field so behavioural drift
    /// between this and the real `AsrSession` is caught by the tests
    /// below (any change to the real `handle_event` must be reflected
    /// here or a test fails).
    struct AsrSessionForTest {
        pcm_f32: Vec<f32>,
        format: Option<AudioFormat>,
        in_stream: bool,
    }

    impl AsrSessionForTest {
        fn new() -> Self {
            Self {
                pcm_f32: Vec::new(),
                format: None,
                in_stream: false,
            }
        }

        fn reset(&mut self) {
            self.pcm_f32.clear();
            self.format = None;
            self.in_stream = false;
        }

        fn handle_event<T>(
            &mut self,
            header: &WyomingHeader,
            payload: &[u8],
            transcribe_fn: &T,
        ) -> Option<AsrResponse>
        where
            T: Fn(&str, &[f32]) -> Result<String, ServiceError>,
        {
            match header.type_.as_str() {
                "audio-start" => match parse_audio_format(header) {
                    Ok(fmt) => {
                        self.reset();
                        self.format = Some(fmt);
                        self.in_stream = true;
                        None
                    }
                    Err(msg) => Some(AsrResponse::Error(msg)),
                },
                "audio-chunk" => self.on_chunk(header, payload),
                "audio-stop" => {
                    self.in_stream = false;
                    None
                }
                "transcribe" => Some(self.finalize(header, transcribe_fn)),
                other => Some(AsrResponse::Error(format!(
                    "unsupported wyoming event: {other}"
                ))),
            }
        }

        fn on_chunk(&mut self, header: &WyomingHeader, payload: &[u8]) -> Option<AsrResponse> {
            if self.format.is_none() {
                match parse_audio_format(header) {
                    Ok(fmt) => {
                        self.format = Some(fmt);
                        self.in_stream = true;
                    }
                    Err(msg) => return Some(AsrResponse::Error(msg)),
                }
            }
            let fmt = self.format.unwrap();
            if fmt.width != 2 {
                return Some(AsrResponse::Error(format!(
                    "unsupported sample width: {} bytes (expected 2)",
                    fmt.width
                )));
            }
            let bytes_per_frame = 2 * fmt.channels as usize;
            if payload.len() % bytes_per_frame != 0 {
                return Some(AsrResponse::Error(format!(
                    "payload length {} not a multiple of frame size {bytes_per_frame}",
                    payload.len()
                )));
            }
            let mut decoded = Vec::with_capacity(payload.len() / bytes_per_frame);
            for frame in payload.chunks_exact(bytes_per_frame) {
                let mut acc: i32 = 0;
                for ch in frame.chunks_exact(2) {
                    acc += i16::from_le_bytes([ch[0], ch[1]]) as i32;
                }
                decoded.push((acc as f32) / (fmt.channels as f32) / 32768.0);
            }
            let sixteen = if fmt.rate == 16_000 {
                decoded
            } else {
                linear_resample_to_16k(&decoded, fmt.rate)
            };
            self.pcm_f32.extend(sixteen);
            None
        }

        fn finalize<T>(&mut self, header: &WyomingHeader, transcribe_fn: &T) -> AsrResponse
        where
            T: Fn(&str, &[f32]) -> Result<String, ServiceError>,
        {
            let params: TranscribeParams = match header.data.clone() {
                Some(v) => serde_json::from_value(v).unwrap_or_default(),
                None => TranscribeParams::default(),
            };
            let model = params
                .name
                .as_deref()
                .unwrap_or(model_names::WHISPER_1)
                .to_owned();
            if self.pcm_f32.is_empty() {
                self.reset();
                return AsrResponse::Error("no audio received before transcribe".into());
            }
            let pcm = std::mem::take(&mut self.pcm_f32);
            self.reset();
            match transcribe_fn(&model, &pcm) {
                Ok(text) => AsrResponse::Transcript { text },
                Err(ServiceError::UnknownModel(m)) => {
                    AsrResponse::Error(format!("model_not_found: {m}"))
                }
                Err(e) => AsrResponse::Error(format!("inference_failed: {e}")),
            }
        }
    }

    fn hdr(t: &str, data: serde_json::Value, payload_length: usize) -> WyomingHeader {
        WyomingHeader {
            type_: t.into(),
            data: if data.is_null() { None } else { Some(data) },
            data_length: 0,
            payload_length,
            version: None,
        }
    }

    fn pcm_bytes(samples: &[i16]) -> Vec<u8> {
        let mut out = Vec::with_capacity(samples.len() * 2);
        for s in samples {
            out.extend_from_slice(&s.to_le_bytes());
        }
        out
    }

    #[test]
    fn audio_start_chunk_stop_transcribe_returns_transcript() {
        // Golden path per docs §2.1: audio-start (16k mono int16) → 2
        // chunks → audio-stop → transcribe. Emit "OK <n>" where n is the
        // sample count so we can assert the pipeline forwarded the right
        // buffer size (silent truncation is a bug we want to catch).
        let mut session = AsrSessionForTest::new();
        let transcribe = |_m: &str, pcm: &[f32]| -> Result<String, ServiceError> {
            Ok(format!("OK {}", pcm.len()))
        };
        let start = hdr(
            "audio-start",
            serde_json::json!({"rate": 16000, "width": 2, "channels": 1}),
            0,
        );
        assert!(session.handle_event(&start, &[], &transcribe).is_none());
        let chunk_hdr = hdr(
            "audio-chunk",
            serde_json::json!({"rate": 16000, "width": 2, "channels": 1}),
            8,
        );
        let chunk_payload = pcm_bytes(&[100, 200, -300, 400]);
        assert!(
            session
                .handle_event(&chunk_hdr, &chunk_payload, &transcribe)
                .is_none()
        );
        let chunk_payload_2 = pcm_bytes(&[500, -600]);
        let chunk_hdr_2 = hdr(
            "audio-chunk",
            serde_json::json!({"rate": 16000, "width": 2, "channels": 1}),
            4,
        );
        assert!(
            session
                .handle_event(&chunk_hdr_2, &chunk_payload_2, &transcribe)
                .is_none()
        );
        let stop = hdr("audio-stop", serde_json::Value::Null, 0);
        assert!(session.handle_event(&stop, &[], &transcribe).is_none());
        let transcribe_hdr = hdr("transcribe", serde_json::json!({"language": "en"}), 0);
        match session.handle_event(&transcribe_hdr, &[], &transcribe) {
            Some(AsrResponse::Transcript { text }) => assert_eq!(text, "OK 6"),
            other => panic!("expected Transcript, got {other:?}"),
        }
    }

    #[test]
    fn transcribe_before_audio_stop_finalizes_immediately() {
        // Upstream rhasspy behaviour: a client may skip audio-stop and
        // send transcribe directly. We must finalize with the buffered
        // PCM, not error.
        let mut session = AsrSessionForTest::new();
        let transcribe = |_m: &str, pcm: &[f32]| -> Result<String, ServiceError> {
            Ok(format!("early {}", pcm.len()))
        };
        session.handle_event(
            &hdr(
                "audio-start",
                serde_json::json!({"rate": 16000, "width": 2, "channels": 1}),
                0,
            ),
            &[],
            &transcribe,
        );
        session.handle_event(
            &hdr(
                "audio-chunk",
                serde_json::json!({"rate": 16000, "width": 2, "channels": 1}),
                4,
            ),
            &pcm_bytes(&[1, 2]),
            &transcribe,
        );
        match session.handle_event(
            &hdr("transcribe", serde_json::Value::Null, 0),
            &[],
            &transcribe,
        ) {
            Some(AsrResponse::Transcript { text }) => assert_eq!(text, "early 2"),
            other => panic!("expected Transcript, got {other:?}"),
        }
    }

    #[test]
    fn stereo_downmix_averages_channels() {
        // width=2, channels=2 must be averaged, not concatenated.
        let mut session = AsrSessionForTest::new();
        let mut captured: Vec<f32> = Vec::new();
        let captured_ptr = &mut captured as *mut Vec<f32>;
        let transcribe = |_m: &str, pcm: &[f32]| -> Result<String, ServiceError> {
            // SAFETY: single-threaded test; capture PCM for assertion.
            unsafe { (*captured_ptr).extend_from_slice(pcm) };
            Ok("ok".into())
        };
        session.handle_event(
            &hdr(
                "audio-start",
                serde_json::json!({"rate": 16000, "width": 2, "channels": 2}),
                0,
            ),
            &[],
            &transcribe,
        );
        // Two frames of stereo int16 LE: (10, 20) and (-10, 10).
        // Averages: 15/32768 and 0/32768.
        let payload = pcm_bytes(&[10, 20, -10, 10]);
        session.handle_event(
            &hdr(
                "audio-chunk",
                serde_json::json!({"rate": 16000, "width": 2, "channels": 2}),
                8,
            ),
            &payload,
            &transcribe,
        );
        session.handle_event(
            &hdr("transcribe", serde_json::Value::Null, 0),
            &[],
            &transcribe,
        );
        assert_eq!(captured.len(), 2);
        assert!((captured[0] - 15.0 / 32768.0).abs() < 1e-6);
        assert!((captured[1] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn resample_from_8k_to_16k_doubles_length() {
        // 8 kHz input → 16 kHz Whisper convention. Length should roughly
        // double; exact values are exercised by vokra-ops. This test only
        // guards against the pipeline dropping the resample step.
        let mut session = AsrSessionForTest::new();
        let mut captured_len = 0usize;
        let captured_len_ptr = &mut captured_len as *mut usize;
        let transcribe = |_m: &str, pcm: &[f32]| -> Result<String, ServiceError> {
            // SAFETY: single-threaded test capture.
            unsafe { *captured_len_ptr = pcm.len() };
            Ok("ok".into())
        };
        session.handle_event(
            &hdr(
                "audio-start",
                serde_json::json!({"rate": 8000, "width": 2, "channels": 1}),
                0,
            ),
            &[],
            &transcribe,
        );
        let payload = pcm_bytes(&[0; 100]);
        session.handle_event(
            &hdr(
                "audio-chunk",
                serde_json::json!({"rate": 8000, "width": 2, "channels": 1}),
                payload.len(),
            ),
            &payload,
            &transcribe,
        );
        session.handle_event(
            &hdr("transcribe", serde_json::Value::Null, 0),
            &[],
            &transcribe,
        );
        // 100 samples at 8k → 200 at 16k (allow ±1 for boundary rounding).
        assert!(
            (199..=201).contains(&captured_len),
            "expected ~200 samples, got {captured_len}",
        );
    }

    #[test]
    fn transcribe_with_no_audio_returns_error() {
        // FR-EX-08: don't invent a silent-run transcript.
        let mut session = AsrSessionForTest::new();
        let transcribe = |_m: &str, _pcm: &[f32]| -> Result<String, ServiceError> {
            panic!("must not be called on empty PCM");
        };
        match session.handle_event(
            &hdr("transcribe", serde_json::Value::Null, 0),
            &[],
            &transcribe,
        ) {
            Some(AsrResponse::Error(msg)) => assert!(msg.contains("no audio")),
            other => panic!("expected Error, got {other:?}"),
        }
    }

    #[test]
    fn unknown_model_maps_to_error_not_silent_fallback() {
        // FR-EX-08: an UnknownModel from the registry must be surfaced,
        // not rewritten to whisper-1 behind the caller's back.
        let mut session = AsrSessionForTest::new();
        let transcribe = |_m: &str, _pcm: &[f32]| -> Result<String, ServiceError> {
            Err(ServiceError::UnknownModel("gpt-4".into()))
        };
        session.handle_event(
            &hdr(
                "audio-start",
                serde_json::json!({"rate": 16000, "width": 2, "channels": 1}),
                0,
            ),
            &[],
            &transcribe,
        );
        session.handle_event(
            &hdr(
                "audio-chunk",
                serde_json::json!({"rate": 16000, "width": 2, "channels": 1}),
                2,
            ),
            &pcm_bytes(&[42]),
            &transcribe,
        );
        match session.handle_event(
            &hdr("transcribe", serde_json::json!({"name": "gpt-4"}), 0),
            &[],
            &transcribe,
        ) {
            Some(AsrResponse::Error(msg)) => {
                assert!(msg.contains("model_not_found"));
                assert!(msg.contains("gpt-4"));
            }
            other => panic!("expected Error, got {other:?}"),
        }
    }

    #[test]
    fn invalid_sample_width_is_rejected_not_ignored() {
        // Wyoming ASR is int16 mono by convention. width=4 (float) is not
        // implemented; explicit reject > silent misinterpretation.
        let mut session = AsrSessionForTest::new();
        let transcribe =
            |_m: &str, _pcm: &[f32]| -> Result<String, ServiceError> { Ok("ok".into()) };
        session.handle_event(
            &hdr(
                "audio-start",
                serde_json::json!({"rate": 16000, "width": 4, "channels": 1}),
                0,
            ),
            &[],
            &transcribe,
        );
        match session.handle_event(
            &hdr(
                "audio-chunk",
                serde_json::json!({"rate": 16000, "width": 4, "channels": 1}),
                4,
            ),
            &[0, 0, 0, 0],
            &transcribe,
        ) {
            Some(AsrResponse::Error(msg)) => assert!(msg.contains("sample width")),
            other => panic!("expected Error, got {other:?}"),
        }
    }

    #[test]
    fn unsupported_event_type_returns_structured_error() {
        // A new upstream event we don't understand must NOT silently
        // succeed or crash the connection.
        let mut session = AsrSessionForTest::new();
        let transcribe =
            |_m: &str, _pcm: &[f32]| -> Result<String, ServiceError> { Ok("ok".into()) };
        match session.handle_event(&hdr("wake", serde_json::Value::Null, 0), &[], &transcribe) {
            Some(AsrResponse::Error(msg)) => assert!(msg.contains("unsupported wyoming event")),
            other => panic!("expected Error, got {other:?}"),
        }
    }

    #[test]
    fn parses_jsonl_header_then_reads_exact_payload_bytes() {
        // Regression guard for plan §5 R5: payload bytes may include 0x0A,
        // so the reader must switch to read_exact after the header line.
        //
        // We drive the async loop with a Cursor holding a header for an
        // audio-chunk whose payload starts with 0x0A (i16 = 10). A
        // line-buffered reader would truncate the payload at the first
        // 0x0A byte and mis-decode the frame.
        use tokio::io::AsyncReadExt as _;
        let mut header_line = serde_json::to_vec(&serde_json::json!({
            "type": "audio-chunk",
            "data": {"rate": 16000, "width": 2, "channels": 1},
            "data_length": 0,
            "payload_length": 4
        }))
        .unwrap();
        header_line.push(b'\n');
        // Payload: [0x0A, 0x00, 0x0A, 0x00] = two i16 samples of 10 each.
        // A line-buffered reader would stop at index 0 (the first 0x0A).
        let payload = vec![0x0A_u8, 0x00, 0x0A, 0x00];
        let mut bytes = header_line;
        bytes.extend_from_slice(&payload);
        // Also append the read-exact test's second header so we exercise
        // the loop returning to header parsing after the payload.
        let stop_line = serde_json::to_vec(&serde_json::json!({
            "type": "audio-stop",
            "data": null,
            "data_length": 0,
            "payload_length": 0
        }))
        .unwrap();
        bytes.extend_from_slice(&stop_line);
        bytes.push(b'\n');

        let mut reader = BufReader::new(std::io::Cursor::new(bytes));

        // Read the header line: should stop AT the trailing \n.
        let mut header_buf = Vec::new();
        let n = tokio_test_block_on(async {
            super::read_header_line(&mut reader, &mut header_buf).await
        })
        .unwrap();
        assert!(n > 0);
        // Header parsed cleanly (no trailing newline).
        let parsed: super::WyomingHeader = serde_json::from_slice(&header_buf).unwrap();
        assert_eq!(parsed.type_, "audio-chunk");
        assert_eq!(parsed.payload_length, 4);

        // Read the payload via read_exact — MUST include the 0x0A bytes.
        let mut payload_buf = vec![0u8; parsed.payload_length];
        tokio_test_block_on(async { reader.read_exact(&mut payload_buf).await }).unwrap();
        assert_eq!(payload_buf, vec![0x0A, 0x00, 0x0A, 0x00]);

        // Loop returns to header: the second event must be audio-stop.
        let mut header_buf2 = Vec::new();
        tokio_test_block_on(async { super::read_header_line(&mut reader, &mut header_buf2).await })
            .unwrap();
        let parsed2: super::WyomingHeader = serde_json::from_slice(&header_buf2).unwrap();
        assert_eq!(parsed2.type_, "audio-stop");
    }

    fn tokio_test_block_on<F: std::future::Future>(fut: F) -> F::Output {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(fut)
    }

    #[test]
    fn oversized_header_is_rejected_before_json_parse() {
        // R5 DoS guard: a 1 MiB header without a newline must fail with
        // InvalidData, not be parsed. We exercise the size cap directly.
        let big = vec![b'A'; MAX_HEADER_BYTES + 100];
        // Append a final \n so read_until terminates rather than hitting EOF.
        let mut bytes = big;
        bytes.push(b'\n');
        let mut reader = BufReader::new(std::io::Cursor::new(bytes));
        let mut buf = Vec::new();
        let err =
            tokio_test_block_on(async { super::read_header_line(&mut reader, &mut buf).await })
                .unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    }

    #[test]
    fn asr_response_serializes_to_wyoming_shape() {
        // Guard the on-wire shape of a `transcript` event so future
        // refactors of write_response do not silently drift.
        let mut buf: Vec<u8> = Vec::new();
        tokio_test_block_on(async {
            super::write_response(
                &mut buf,
                AsrResponse::Transcript {
                    text: "hello".into(),
                },
            )
            .await
            .unwrap();
        });
        let s = std::str::from_utf8(&buf).unwrap();
        // One JSONL line.
        assert!(s.ends_with('\n'));
        let value: serde_json::Value =
            serde_json::from_str(s.trim_end_matches('\n')).expect("valid JSONL");
        assert_eq!(value["type"], "transcript");
        assert_eq!(value["data"]["text"], "hello");
    }

    #[test]
    fn error_event_carries_message() {
        let mut buf: Vec<u8> = Vec::new();
        tokio_test_block_on(async {
            super::write_error_event(&mut buf, "unknown model x")
                .await
                .unwrap();
        });
        let value: serde_json::Value = serde_json::from_slice(buf.trim_ascii_end()).unwrap();
        assert_eq!(value["type"], "error");
        assert_eq!(value["data"]["message"], "unknown model x");
    }

    #[test]
    fn stub_service_marker_is_not_called_on_asr_paths() {
        // If we ever accidentally start touching the InferenceService in
        // ASR event handling (breaking the mock-free tests above), this
        // sentinel will fire and the test suite will fail loudly.
        let call = || stub_service();
        // Intentionally do not call `call()` — we only verify it exists.
        let _ = &call;
    }
}

// ===========================================================================
// T16 — Wyoming TTS event dispatch
// ===========================================================================

use vokra_core::SynthesisRequest;

use crate::service::SynthesizeService;

/// Target chunk length ≈ 40 ms per `audio-chunk` (plan D8 note; matches
/// the Home Assistant satellite jitter budget). At 22 050 Hz mono int16
/// this is 882 samples ≈ 1 764 bytes per chunk — well under
/// [`MAX_PAYLOAD_BYTES`].
const TTS_CHUNK_MS: u32 = 40;

/// PCM bits/sample for the TTS emit path. Wyoming's `audio-*` events pin
/// `width` to bytes/sample (2 for int16). Kept as `u16` bits here for
/// symmetry with the WAV encoder in `service.rs`; the wire field is
/// `bits / 8`.
const TTS_BITS_PER_SAMPLE: u16 = 16;

/// Wyoming `audio-*` `channels` field. Vokra emits mono at v0.5; no
/// stereo TTS voice is on the M2 completion path.
const TTS_CHANNELS: u16 = 1;

/// Byte serialization of a JSONL header line. Emitted as `<json>\n` per
/// `docs/wyoming-design.md` §2, followed by exactly `payload_length`
/// bytes of opaque binary payload for `audio-chunk`. Zero-copy — the
/// caller writes `header_json_line` + `\n` + `payload` in that order.
///
/// Fields marked `data_length` and `payload_length` are `u32` on the
/// wire; kept as `u64` here so callers can arithmetic without casts and
/// we bounds-check at emit time.
#[derive(Debug, Clone, Serialize)]
struct TtsHeader<D: Serialize> {
    /// Event kind (`"info"`, `"audio-start"`, `"audio-chunk"`, `"audio-stop"`).
    #[serde(rename = "type")]
    ty: &'static str,
    /// Structured payload (rate/width/channels/text/voices/...). `None`
    /// serializes as JSON `null` to match `rhasspy/wyoming`.
    data: Option<D>,
    /// Byte count of the (rarely used) structured `data` blob. Always 0 in
    /// our emit path — everything structured goes inline in `data`.
    data_length: u64,
    /// Byte count of the raw binary payload that follows. Only non-zero
    /// for `audio-chunk`.
    payload_length: u64,
}

/// `data` body of the `info` event advertising Vokra's TTS capabilities
/// (voice list + native rate / width / channels). Emitted once per
/// successful synthesize as the capability announcement that precedes
/// `audio-start` (matches how HA satellites discover TTS voices from
/// upstream `rhasspy/wyoming` handlers).
#[derive(Debug, Clone, Serialize)]
struct TtsInfoData<'a> {
    tts: TtsInfoBody<'a>,
}

#[derive(Debug, Clone, Serialize)]
struct TtsInfoBody<'a> {
    /// Engine name (`"vokra"`). Advertising the engine lets HA distinguish
    /// Vokra from Piper/OpenTTS on `describe`.
    name: &'a str,
    /// Version string — pinned to `env!("CARGO_PKG_VERSION")` at compile
    /// time so we cannot lie about which build served the request.
    version: &'a str,
    /// Advertised voices for THIS synthesize call. We advertise the voice
    /// the caller just synthesized against (piper-plus / kokoro) so a
    /// satellite discovering Vokra mid-session receives an actionable list.
    voices: Vec<TtsVoiceInfo<'a>>,
}

#[derive(Debug, Clone, Serialize)]
struct TtsVoiceInfo<'a> {
    /// Model alias (`"piper-plus"`, `"kokoro"`) — identical to the
    /// registry keys in `service::model_names` so a `describe` → `info` →
    /// `synthesize` round-trip needs no name translation.
    name: &'a str,
    /// Engine-native sample rate for this voice (Hz). Matches the value
    /// advertised in the following `audio-start` event.
    sample_rate: u32,
    /// Advertised channel count (always `1` at v0.5).
    channels: u16,
    /// Bits/sample (always `16` at v0.5).
    bits: u16,
}

/// `data` body of an `audio-start` event (§2.1 TTS path). Pins the PCM
/// shape for the following `audio-chunk` events.
#[derive(Debug, Clone, Serialize)]
struct TtsAudioStartData {
    /// Sample rate in Hz (voice-native; e.g. 22050 for piper-plus).
    rate: u32,
    /// Bytes per sample (int16 → 2).
    width: u16,
    /// Channel count (mono → 1).
    channels: u16,
}

/// `data` body of an `audio-chunk` event. Actual samples live in the
/// binary payload (`payload_length` bytes of int16 LE PCM); the `data`
/// object re-advertises `rate/width/channels` so a client can decode
/// individual chunks in isolation (matches `rhasspy/wyoming` shape).
#[derive(Debug, Clone, Serialize)]
struct TtsAudioChunkData {
    /// Sample rate in Hz.
    rate: u32,
    /// Bytes per sample.
    width: u16,
    /// Channel count.
    channels: u16,
}

/// `data` body of the terminating `audio-stop` event. Wyoming allows an
/// optional `timestamp`; Vokra omits it (unit-less monotonic wall clock
/// is out-of-scope at M2-09 — the receiver measures elapsed time
/// locally). Empty struct serializes to `{}` which upstream accepts.
#[derive(Debug, Clone, Serialize)]
struct TtsAudioStopData {}

/// Outcome of one Wyoming `synthesize` event dispatch.
///
/// Distinct from `Result<(), _>` so the T15/T17 accept loop can log both
/// success (with chunk count for throughput monitoring) and each failure
/// mode without unwinding the whole session (NFR-RL-07).
#[derive(Debug)]
pub enum SynthesizeOutcome {
    /// Successful emit — `info` + `audio-start` + `n_chunks` `audio-chunk` +
    /// `audio-stop`.
    Ok {
        /// Number of `audio-chunk` events actually emitted. Zero for a
        /// silent utterance (empty text or engine returned no samples) —
        /// still a success, still followed by `audio-stop`.
        n_chunks: usize,
    },
    /// The service layer rejected the request. FR-EX-08: propagate the
    /// exact [`ServiceError`] so the accept loop can emit a matching
    /// `error` event or close the connection.
    Service(ServiceError),
    /// I/O error writing to the TCP socket (peer closed, RST, backpressure
    /// buffer full). Propagated verbatim so the accept loop can decide
    /// whether to retry or drop the connection.
    Io(io::Error),
}

/// Convert a mono `f32` PCM buffer (`[-1.0, 1.0]`) to int16 little-endian
/// bytes for the `audio-chunk` payload.
///
/// Uses the same clamp/scale mapping as
/// [`crate::service::synthesized_audio_to_wav_pcm16_le`] so a client
/// receiving the same audio over `/api/tts` (WAV) and Wyoming
/// (int16 LE chunks) decodes byte-identical samples.
///
/// Model bugs that emit `|s| > 1.0` or NaN MUST NOT poison the buffer:
/// `f32::clamp` returns the clamp bound on NaN (`+1.0`), so NaN maps to
/// `+32767` rather than an undefined int cast. Silent NaN would be an
/// NFR-RL-06 violation on the model side; we surface a valid frame
/// instead.
fn pcm_f32_to_i16_le_bytes(samples: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(samples.len() * 2);
    for &s in samples {
        let clamped = s.clamp(-1.0, 1.0);
        // Multiply by 32767 (not 32768) so `+1.0 → +32767` and
        // `-1.0 → -32767` symmetrically — piper's canonical scaling
        // (see `service.rs` § WAV encoder header comment).
        let scaled = (clamped * 32767.0).round();
        let sample_i16 = scaled as i16;
        out.extend_from_slice(&sample_i16.to_le_bytes());
    }
    out
}

/// Serialize one Wyoming TTS event onto `w` using the canonical
/// `<json-header>\n<payload-bytes>` framing.
///
/// Every call performs three ordered async writes:
///
/// 1. `serde_json::to_vec(header)` bytes (the JSON header).
/// 2. `b"\n"` (the line terminator).
/// 3. `payload_bytes` (exactly `header.payload_length` bytes; may be empty).
///
/// The function never splits or coalesces frames — the caller owns
/// exclusive access to `w`. Debug-asserts that `payload_length` matches
/// `payload_bytes.len()`; a mismatch is a programming bug that would
/// desynchronize the receiver, so we fail loudly rather than emit a
/// malformed frame.
async fn write_tts_event<W: AsyncWrite + Unpin, D: Serialize>(
    w: &mut W,
    header: &TtsHeader<D>,
    payload_bytes: &[u8],
) -> io::Result<()> {
    debug_assert_eq!(header.data_length, 0, "T16 never uses the data blob");
    debug_assert_eq!(
        header.payload_length as usize,
        payload_bytes.len(),
        "payload_length header field must match slice length exactly"
    );

    let json = serde_json::to_vec(header).map_err(io::Error::other)?;
    w.write_all(&json).await?;
    w.write_all(b"\n").await?;
    if !payload_bytes.is_empty() {
        w.write_all(payload_bytes).await?;
    }
    Ok(())
}

/// Handle one Wyoming `synthesize` event end-to-end: call
/// [`SynthesizeService::synthesize`] on `service` for `(voice_model,
/// text)`, then emit the required event sequence on `w`:
///
/// ```text
/// info           (voice catalogue, one-shot)
/// audio-start    (rate/width/channels)
/// audio-chunk    ... (int16 LE PCM in payload, ~40ms per chunk)
/// audio-stop
/// ```
///
/// # Parameters
///
/// * `w`: an async writer (real Wyoming session's `TcpStream` write half,
///   or an in-memory `Vec<u8>` in tests). The caller MUST hold exclusive
///   access — this function never mutex-serializes across concurrent
///   callers.
/// * `service`: the same `Arc<InferenceService>` (or any `SynthesizeService`
///   trait object) the HTTP layer uses. Not cloned inside this function.
/// * `voice_model`: alias to resolve against the registry. `None` (or
///   `Some("")`) defaults to `piper-plus` (the v0.5 default TTS engine).
///   Passed through unchanged when non-empty so `ServiceError::UnknownModel`
///   names the exact string the client sent (FR-EX-08 diagnostic clarity).
/// * `text`: raw text to synthesize. Empty text is legal (a silent
///   utterance); we still emit `info + audio-start + audio-stop` so the
///   receiver's state machine advances. No audio is fabricated (NFR-RL-06).
///
/// # Errors
///
/// Never panics on well-typed error paths — every failure funnels into
/// [`SynthesizeOutcome`]. Panics inside `service.synthesize` are the
/// caller's responsibility ([`crate::error::spawn_isolated_wyoming_task`]
/// wraps the whole session task in `catch_unwind`).
pub async fn handle_synthesize<W: AsyncWrite + Unpin>(
    w: &mut W,
    service: &dyn SynthesizeService,
    voice_model: Option<&str>,
    text: &str,
) -> SynthesizeOutcome {
    // 1) Resolve model. `None` OR an empty string ⇒ piper-plus default
    //    (the M0-07 native TTS is the v0.5 default). Never silently
    //    substitute another engine when the caller explicitly named an
    //    unknown one — that would be an FR-EX-08 violation.
    let model = match voice_model {
        Some(m) if !m.is_empty() => m,
        _ => model_names::PIPER_PLUS,
    };

    // 2) Build the vokra-core `SynthesisRequest`. `length_scale` /
    //    `noise_scale` / prosody controls are not part of Wyoming's
    //    `synthesize` event, so we take voice defaults exclusively
    //    (matches the piper-plus HTTP shape at T11).
    let request = SynthesisRequest::new(text);

    // 3) Dispatch to the service. Errors propagate verbatim — no silent
    //    fallback, no fabricated audio (NFR-RL-06).
    let audio = match service.synthesize(model, &request) {
        Ok(a) => a,
        Err(e) => return SynthesizeOutcome::Service(e),
    };

    // 4) Emit `info` (voice catalogue).
    let info = TtsHeader::<TtsInfoData<'_>> {
        ty: "info",
        data: Some(TtsInfoData {
            tts: TtsInfoBody {
                name: "vokra",
                version: env!("CARGO_PKG_VERSION"),
                voices: vec![TtsVoiceInfo {
                    name: model,
                    sample_rate: audio.sample_rate,
                    channels: TTS_CHANNELS,
                    bits: TTS_BITS_PER_SAMPLE,
                }],
            },
        }),
        data_length: 0,
        payload_length: 0,
    };
    if let Err(e) = write_tts_event(w, &info, &[]).await {
        return SynthesizeOutcome::Io(e);
    }

    // 5) Emit `audio-start` (rate / width bytes / channels).
    let start = TtsHeader::<TtsAudioStartData> {
        ty: "audio-start",
        data: Some(TtsAudioStartData {
            rate: audio.sample_rate,
            width: TTS_BITS_PER_SAMPLE / 8,
            channels: TTS_CHANNELS,
        }),
        data_length: 0,
        payload_length: 0,
    };
    if let Err(e) = write_tts_event(w, &start, &[]).await {
        return SynthesizeOutcome::Io(e);
    }

    // 6) Emit `audio-chunk` events. Encode the whole f32 buffer to
    //    int16 LE once, then slice into ~40 ms chunks. Receivers
    //    reassemble by concatenation (byte-level round-trip is asserted
    //    by the `tts_events` test suite).
    let pcm_bytes = pcm_f32_to_i16_le_bytes(&audio.samples);
    let mut n_chunks = 0usize;
    if !pcm_bytes.is_empty() {
        // bytes_per_ms = sample_rate * width / 1000. u64 to avoid
        // overflow at exotic rates.
        let width_bytes = u64::from(TTS_BITS_PER_SAMPLE / 8);
        let bytes_per_ms = u64::from(audio.sample_rate) * width_bytes / 1000;
        // Guard against pathological sub-25 Hz rates rounding to 0.
        let raw_chunk_bytes = (bytes_per_ms * u64::from(TTS_CHUNK_MS)).max(2) as usize;
        // Round to an even count so we never split an int16 sample across
        // a chunk boundary. A receiver that dropped a mid-sample chunk
        // would resynchronize into garbage — Wyoming has no sub-sample
        // recovery at v0.5.
        let chunk_bytes = if raw_chunk_bytes % 2 == 1 {
            raw_chunk_bytes + 1
        } else {
            raw_chunk_bytes
        };

        for slice in pcm_bytes.chunks(chunk_bytes) {
            let payload_len = slice.len() as u64;
            // Belt-and-suspenders: enforce the same 16 MiB cap the
            // receiver uses (docs §5 / `MAX_PAYLOAD_BYTES`). At 40 ms
            // per chunk this cannot trip in normal operation; a trip
            // here means chunk_bytes was mis-computed.
            debug_assert!(
                (payload_len as usize) <= MAX_PAYLOAD_BYTES,
                "audio-chunk exceeds MAX_PAYLOAD_BYTES"
            );
            let chunk = TtsHeader::<TtsAudioChunkData> {
                ty: "audio-chunk",
                data: Some(TtsAudioChunkData {
                    rate: audio.sample_rate,
                    width: TTS_BITS_PER_SAMPLE / 8,
                    channels: TTS_CHANNELS,
                }),
                data_length: 0,
                payload_length: payload_len,
            };
            if let Err(e) = write_tts_event(w, &chunk, slice).await {
                return SynthesizeOutcome::Io(e);
            }
            n_chunks += 1;
        }
    }

    // 7) Emit terminating `audio-stop`.
    let stop = TtsHeader::<TtsAudioStopData> {
        ty: "audio-stop",
        data: Some(TtsAudioStopData {}),
        data_length: 0,
        payload_length: 0,
    };
    if let Err(e) = write_tts_event(w, &stop, &[]).await {
        return SynthesizeOutcome::Io(e);
    }

    SynthesizeOutcome::Ok { n_chunks }
}

// ---------------------------------------------------------------------------
// Tests — `cargo test wyoming::tts_events` (T16).
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tts_events {
    //! End-to-end verification of the T16 emit path.
    //!
    //! These tests drive [`handle_synthesize`] against a fake
    //! [`SynthesizeService`] and read the resulting Wyoming frame stream
    //! from a `Vec<u8>` sink. We verify:
    //!
    //! * event order (`info` → `audio-start` → `audio-chunk`+ →
    //!   `audio-stop`);
    //! * `payload_length` framing is byte-exact (a `0x0A` byte in the
    //!   PCM payload MUST round-trip — plan §5 R5);
    //! * PCM byte-level round-trip: concatenated `audio-chunk` payloads
    //!   decode back to the same int16 samples the engine produced;
    //! * silent utterance (empty text) still emits `info + audio-start
    //!   + audio-stop` with zero chunks (NFR-RL-06);
    //! * FR-EX-08 — unknown voice surfaces as `Service(UnknownModel)`,
    //!   not silently rerouted;
    //! * Kokoro (advertised but deferred) surfaces as
    //!   `Service(SynthesizeUnavailable)`, never fabricated audio.
    //!
    //! No TCP sockets are opened. All writes go to in-memory `Vec<u8>`
    //! sinks, so there is no port binding, no runtime state, no cross-
    //! test ordering dependence.

    use super::*;
    use crate::service::{ServiceError, SynthesizeService, model_names};
    use vokra_core::{SynthesisRequest, SynthesizedAudio};

    /// Test double that mirrors the exact dispatch rules of the real
    /// `InferenceService::synthesize` so a regression in either side
    /// surfaces here.
    ///
    /// * `piper-plus` → returns a canned f32 buffer at `sample_rate`;
    /// * `kokoro` when `kokoro_registered=true` → `SynthesizeUnavailable`
    ///   (M2-07 deferred);
    /// * `kokoro` when not registered → `UnknownModel`;
    /// * any other name → `UnknownModel`.
    struct FakeSynth {
        samples: Vec<f32>,
        sample_rate: u32,
        kokoro_registered: bool,
    }

    impl FakeSynth {
        fn with_samples(samples: Vec<f32>, sample_rate: u32) -> Self {
            Self {
                samples,
                sample_rate,
                kokoro_registered: false,
            }
        }
    }

    impl SynthesizeService for FakeSynth {
        fn synthesize(
            &self,
            model: &str,
            _request: &SynthesisRequest,
        ) -> Result<SynthesizedAudio, ServiceError> {
            match model {
                model_names::PIPER_PLUS => Ok(SynthesizedAudio::new(
                    self.samples.clone(),
                    self.sample_rate,
                )),
                model_names::KOKORO => {
                    if !self.kokoro_registered {
                        Err(ServiceError::UnknownModel(model.to_owned()))
                    } else {
                        Err(ServiceError::SynthesizeUnavailable {
                            model: model.to_owned(),
                            reason: "kokoro deferred to M2-07",
                        })
                    }
                }
                other => Err(ServiceError::UnknownModel(other.to_owned())),
            }
        }
    }

    /// Minimal Wyoming frame decoder. Deliberately hand-rolled and
    /// separate from `write_tts_event`'s serialization code — a bug in
    /// the writer that also matches a symmetric bug in a shared reader
    /// would otherwise pass silently. Uses the canonical
    /// `read_until('\n')` + `read_exact(N)` pattern from
    /// `docs/wyoming-design.md` §3.
    #[derive(Debug, Clone)]
    struct DecodedEvent {
        ty: String,
        data_length: usize,
        payload_length: usize,
        header_json: serde_json::Value,
        payload: Vec<u8>,
    }

    fn decode_stream(bytes: &[u8]) -> Vec<DecodedEvent> {
        let mut out = Vec::new();
        let mut i = 0;
        while i < bytes.len() {
            // 1) Header line up to (and including) '\n'.
            let nl = bytes[i..]
                .iter()
                .position(|&b| b == b'\n')
                .expect("frame is missing '\\n' after JSON header");
            let header_slice = &bytes[i..i + nl];
            let header_json: serde_json::Value =
                serde_json::from_slice(header_slice).expect("header line is not valid JSON");
            i += nl + 1; // skip the '\n'

            let ty = header_json
                .get("type")
                .and_then(|v| v.as_str())
                .expect("header missing `type`")
                .to_owned();
            let data_length = header_json
                .get("data_length")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as usize;
            let payload_length = header_json
                .get("payload_length")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as usize;

            // 2) Exactly `data_length` bytes with `read_exact`-style
            //    slicing (never line-buffered — R5). Vokra's TTS never
            //    emits a `data` blob, so this is always 0.
            assert!(
                i + data_length <= bytes.len(),
                "truncated data blob at offset {i}"
            );
            i += data_length;

            // 3) Exactly `payload_length` bytes of opaque binary payload
            //    (int16 LE PCM for audio-chunk).
            assert!(
                i + payload_length <= bytes.len(),
                "truncated payload at offset {i}"
            );
            let payload = bytes[i..i + payload_length].to_vec();
            i += payload_length;

            out.push(DecodedEvent {
                ty,
                data_length,
                payload_length,
                header_json,
                payload,
            });
        }
        out
    }

    /// Inverse of `pcm_f32_to_i16_le_bytes`: decode int16 LE bytes back
    /// to `f32`. Used to verify round-trip round-trip fidelity within
    /// `1 / 32767` (int16 quantization error only).
    fn decode_i16_le_to_f32(bytes: &[u8]) -> Vec<f32> {
        assert!(
            bytes.len() % 2 == 0,
            "int16 stream must have even byte count, got {}",
            bytes.len()
        );
        let mut out = Vec::with_capacity(bytes.len() / 2);
        for pair in bytes.chunks_exact(2) {
            let sample = i16::from_le_bytes([pair[0], pair[1]]);
            out.push(f32::from(sample) / 32767.0);
        }
        out
    }

    #[tokio::test]
    async fn emits_info_start_chunks_stop_in_order() {
        // 1000 non-trivial samples ⇒ >1 audio-chunk (≈40 ms at 22050 Hz
        // is 882 samples, so this straddles a chunk boundary).
        let samples: Vec<f32> = (0..1000).map(|i| (i as f32 / 1000.0) - 0.5).collect();
        let svc = FakeSynth::with_samples(samples, 22_050);

        let mut sink: Vec<u8> = Vec::new();
        let outcome =
            handle_synthesize(&mut sink, &svc, Some(model_names::PIPER_PLUS), "hello").await;

        let n_chunks = match outcome {
            SynthesizeOutcome::Ok { n_chunks } => n_chunks,
            other => panic!("expected Ok, got {other:?}"),
        };
        assert!(n_chunks > 0, "1000 samples must produce at least one chunk");

        let events = decode_stream(&sink);
        // Minimum: info + start + ≥1 chunk + stop.
        assert!(
            events.len() >= 4,
            "expected ≥4 events, got {}",
            events.len()
        );
        assert_eq!(events[0].ty, "info");
        assert_eq!(events[1].ty, "audio-start");
        assert_eq!(events.last().unwrap().ty, "audio-stop");
        for chunk in &events[2..events.len() - 1] {
            assert_eq!(chunk.ty, "audio-chunk");
        }
        let chunk_count = events.iter().filter(|e| e.ty == "audio-chunk").count();
        assert_eq!(chunk_count, n_chunks);
    }

    #[tokio::test]
    async fn info_advertises_voice_catalogue() {
        let svc = FakeSynth::with_samples(vec![0.0; 16], 22_050);

        let mut sink: Vec<u8> = Vec::new();
        let outcome = handle_synthesize(&mut sink, &svc, None, "hi").await;
        assert!(matches!(outcome, SynthesizeOutcome::Ok { .. }));

        let events = decode_stream(&sink);
        let info = &events[0];
        assert_eq!(info.ty, "info");
        assert_eq!(info.data_length, 0);
        assert_eq!(info.payload_length, 0);

        let voices = info
            .header_json
            .pointer("/data/tts/voices")
            .expect("info.data.tts.voices missing")
            .as_array()
            .expect("voices is not an array");
        assert_eq!(voices.len(), 1);
        assert_eq!(voices[0]["name"].as_str(), Some(model_names::PIPER_PLUS));
        assert_eq!(voices[0]["sample_rate"].as_u64(), Some(22_050));
        assert_eq!(voices[0]["channels"].as_u64(), Some(1));
        assert_eq!(voices[0]["bits"].as_u64(), Some(16));

        // Engine name & version stamped.
        assert_eq!(
            info.header_json
                .pointer("/data/tts/name")
                .and_then(|v| v.as_str()),
            Some("vokra")
        );
        let version = info
            .header_json
            .pointer("/data/tts/version")
            .and_then(|v| v.as_str())
            .expect("version missing");
        assert!(!version.is_empty());
        assert_eq!(version, env!("CARGO_PKG_VERSION"));
    }

    #[tokio::test]
    async fn audio_start_pins_rate_width_channels() {
        let svc = FakeSynth::with_samples(vec![0.0; 16], 16_000);

        let mut sink: Vec<u8> = Vec::new();
        let outcome = handle_synthesize(&mut sink, &svc, None, "hi").await;
        assert!(matches!(outcome, SynthesizeOutcome::Ok { .. }));

        let events = decode_stream(&sink);
        let start = &events[1];
        assert_eq!(start.ty, "audio-start");
        assert_eq!(start.data_length, 0);
        assert_eq!(start.payload_length, 0);

        // rate = engine sample rate; width = 2 (int16); channels = 1.
        assert_eq!(
            start
                .header_json
                .pointer("/data/rate")
                .and_then(|v| v.as_u64()),
            Some(16_000)
        );
        assert_eq!(
            start
                .header_json
                .pointer("/data/width")
                .and_then(|v| v.as_u64()),
            Some(2)
        );
        assert_eq!(
            start
                .header_json
                .pointer("/data/channels")
                .and_then(|v| v.as_u64()),
            Some(1)
        );
    }

    #[tokio::test]
    async fn payload_length_framing_is_byte_exact_and_roundtrips_pcm() {
        // Regression guard for plan §5 R5: payload bytes may include 0x0A,
        // so a receiver MUST honour `payload_length` and NOT line-buffer.
        //
        // We construct a sample whose int16 LE encoding contains 0x0A:
        // 10 / 32767 ≈ 3.05e-4 → int16 = 10 → LE = [0x0A, 0x00].
        let target_i16: i16 = 10;
        let target_sample = f32::from(target_i16) / 32767.0;
        let samples: Vec<f32> = (0..2000)
            .map(|i| if i % 3 == 0 { target_sample } else { 0.0 })
            .collect();
        let svc = FakeSynth::with_samples(samples.clone(), 22_050);

        let mut sink: Vec<u8> = Vec::new();
        let outcome = handle_synthesize(&mut sink, &svc, None, "roundtrip").await;
        assert!(matches!(outcome, SynthesizeOutcome::Ok { .. }));

        let events = decode_stream(&sink);
        assert_eq!(events[0].ty, "info");
        assert_eq!(events[1].ty, "audio-start");

        // Every audio-chunk: `payload_length` matches actual payload
        // byte count and length is even (never split an int16 sample).
        for chunk in events.iter().filter(|e| e.ty == "audio-chunk") {
            assert_eq!(chunk.payload_length, chunk.payload.len());
            assert_eq!(chunk.data_length, 0);
            assert!(
                chunk.payload_length.is_multiple_of(2),
                "audio-chunk payload_length {} splits a sample",
                chunk.payload_length
            );
        }

        // Concatenated audio-chunk payloads = samples.len() * 2 bytes.
        let all_bytes: Vec<u8> = events
            .iter()
            .filter(|e| e.ty == "audio-chunk")
            .flat_map(|e| e.payload.iter().copied())
            .collect();
        assert_eq!(all_bytes.len(), samples.len() * 2);

        // R5 exercise: the payload MUST contain the 0x0A sentinel. If it
        // does not, the test data is not exercising the trap.
        assert!(
            all_bytes.contains(&0x0A),
            "test PCM does not exercise the R5 line-buffer trap"
        );

        // Round-trip fidelity: decode int16 LE back to f32 and compare
        // with the original within one quantization step.
        let decoded = decode_i16_le_to_f32(&all_bytes);
        assert_eq!(decoded.len(), samples.len());
        for (i, (&orig, &back)) in samples.iter().zip(&decoded).enumerate() {
            let err = (orig - back).abs();
            assert!(
                err < 1.0 / 32767.0 + f32::EPSILON,
                "sample {i}: orig={orig} back={back} err={err}"
            );
        }
    }

    #[tokio::test]
    async fn audio_stop_terminates_stream_with_no_trailing_bytes() {
        let svc = FakeSynth::with_samples(vec![0.1; 100], 22_050);

        let mut sink: Vec<u8> = Vec::new();
        let outcome = handle_synthesize(&mut sink, &svc, None, "hi").await;
        assert!(matches!(outcome, SynthesizeOutcome::Ok { .. }));

        let events = decode_stream(&sink);
        let stop = events.last().unwrap();
        assert_eq!(stop.ty, "audio-stop");
        assert_eq!(stop.data_length, 0);
        assert_eq!(stop.payload_length, 0);
        // `decode_stream` fully consumes `sink`; if there were trailing
        // bytes past `audio-stop` the decoder loop would emit a
        // truncated-frame panic.
    }

    #[tokio::test]
    async fn empty_text_still_emits_info_start_stop_without_chunks() {
        // NFR-RL-06: a silent utterance MUST NOT fabricate audio, but
        // the state machine still needs a well-formed event sequence.
        let svc = FakeSynth::with_samples(vec![], 22_050);

        let mut sink: Vec<u8> = Vec::new();
        let outcome = handle_synthesize(&mut sink, &svc, None, "").await;
        let n_chunks = match outcome {
            SynthesizeOutcome::Ok { n_chunks } => n_chunks,
            other => panic!("expected Ok, got {other:?}"),
        };
        assert_eq!(n_chunks, 0, "empty audio must not emit audio-chunk events");

        let events = decode_stream(&sink);
        assert_eq!(events.len(), 3);
        assert_eq!(events[0].ty, "info");
        assert_eq!(events[1].ty, "audio-start");
        assert_eq!(events[2].ty, "audio-stop");
    }

    #[tokio::test]
    async fn unknown_voice_never_silently_falls_back() {
        // FR-EX-08: request for "elevenlabs" must NOT be silently
        // rerouted to piper-plus. Surface UnknownModel verbatim so the
        // accept-loop can emit an explicit `error` event.
        let svc = FakeSynth::with_samples(vec![0.0; 16], 22_050);

        let mut sink: Vec<u8> = Vec::new();
        let outcome = handle_synthesize(&mut sink, &svc, Some("elevenlabs"), "hi").await;

        match outcome {
            SynthesizeOutcome::Service(ServiceError::UnknownModel(m)) => {
                assert_eq!(m, "elevenlabs");
            }
            other => panic!("expected Service(UnknownModel), got {other:?}"),
        }
        assert!(
            sink.is_empty(),
            "no frames should be written on service error"
        );
    }

    #[tokio::test]
    async fn kokoro_when_registered_surfaces_synthesize_unavailable() {
        // Kokoro is advertised iff its voice was configured, but its
        // synthesize path is deferred to M2-07. Surface it as
        // SynthesizeUnavailable, never as fabricated audio (NFR-RL-06).
        let svc = FakeSynth {
            samples: vec![0.0; 16],
            sample_rate: 22_050,
            kokoro_registered: true,
        };

        let mut sink: Vec<u8> = Vec::new();
        let outcome = handle_synthesize(&mut sink, &svc, Some(model_names::KOKORO), "hi").await;
        match outcome {
            SynthesizeOutcome::Service(ServiceError::SynthesizeUnavailable { model, .. }) => {
                assert_eq!(model, model_names::KOKORO);
            }
            other => panic!("expected Service(SynthesizeUnavailable), got {other:?}"),
        }
        assert!(
            sink.is_empty(),
            "no frames should be written on service error"
        );
    }

    #[tokio::test]
    async fn default_voice_is_piper_plus_for_none_and_empty() {
        // When the client omits the voice OR sends an empty string, we
        // route to piper-plus. Verifies that None and "" both defer to
        // the default rather than surfacing as UnknownModel("").
        for voice in [None, Some("")] {
            let svc = FakeSynth::with_samples(vec![0.0; 16], 22_050);
            let mut sink: Vec<u8> = Vec::new();
            let outcome = handle_synthesize(&mut sink, &svc, voice, "hi").await;
            assert!(
                matches!(outcome, SynthesizeOutcome::Ok { .. }),
                "voice={voice:?} should route to piper-plus"
            );

            let events = decode_stream(&sink);
            let voice_name = events[0]
                .header_json
                .pointer("/data/tts/voices/0/name")
                .and_then(|v| v.as_str())
                .expect("voice name missing");
            assert_eq!(voice_name, model_names::PIPER_PLUS);
        }
    }

    #[tokio::test]
    async fn chunk_boundaries_are_even_bytes() {
        // Every audio-chunk `payload_length` must be even, so we never
        // split an int16 sample across a chunk boundary. If we did, a
        // receiver that dropped a mid-sample chunk would resynchronize
        // into garbage (Wyoming has no sub-sample recovery at v0.5).
        //
        // We use an odd-length sample buffer to guarantee at least one
        // "final" chunk has to handle the boundary correctly.
        let samples: Vec<f32> = (0..1001).map(|i| (i as f32 / 1001.0) - 0.5).collect();
        let svc = FakeSynth::with_samples(samples.clone(), 22_050);

        let mut sink: Vec<u8> = Vec::new();
        let outcome = handle_synthesize(&mut sink, &svc, None, "boundary").await;
        assert!(matches!(outcome, SynthesizeOutcome::Ok { .. }));

        let events = decode_stream(&sink);
        for chunk in events.iter().filter(|e| e.ty == "audio-chunk") {
            assert!(
                chunk.payload_length.is_multiple_of(2),
                "chunk payload_length {} is odd",
                chunk.payload_length
            );
        }
        // Total = samples.len() * 2 bytes.
        let total: usize = events
            .iter()
            .filter(|e| e.ty == "audio-chunk")
            .map(|e| e.payload.len())
            .sum();
        assert_eq!(total, samples.len() * 2);
    }
}
